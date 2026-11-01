"""Human-in-the-loop review gate between a generated brief and the client.

Briefs wait in a file-backed queue (one JSON file per brief) until a reviewer
approves, rejects or requests changes. Approval publishes the brief
(``pipeline.publish_brief``): client deliverables, archive copy and RAG index.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from marketbrief.config import Settings
from marketbrief.schemas import Brief, BriefStatus, ReviewDecision

logger = logging.getLogger(__name__)

Decision = Literal["approved", "rejected", "changes_requested"]

# Statuses from which a reviewer may still decide.
DECIDABLE: frozenset[str] = frozenset({"needs_review", "changes_requested"})

_BRIEF_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]{0,127}$")


def _check_id(brief_id: str) -> str:
    """Reject ids that could escape the queue directory."""
    if not _BRIEF_ID.match(brief_id):
        raise KeyError(f"invalid brief id: {brief_id!r}")
    return brief_id


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


class ReviewQueue:
    """File-backed queue of briefs awaiting (or past) human review."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.dir = settings.path(settings.paths.review_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, brief_id: str) -> Path:
        return self.dir / f"{_check_id(brief_id)}.json"

    def submit(self, brief: Brief) -> Path:
        """Queue ``brief`` for review (status ``needs_review``); return its file."""
        queued = brief.model_copy(update={"status": "needs_review", "review": None})
        path = self.path_for(queued.id)
        _write_atomic(path, queued.model_dump_json(indent=2))
        logger.info("queued brief %s for review at %s", queued.id, path)
        return path

    def load(self, brief_id: str) -> Brief:
        """Load a queued brief; raises ``KeyError`` when it does not exist."""
        path = self.path_for(brief_id)
        if not path.exists():
            raise KeyError(f"no brief {brief_id!r} in review queue {self.dir}")
        return Brief.model_validate_json(path.read_text(encoding="utf-8"))

    def decide(self, brief_id: str, decision: Decision, reviewer: str, notes: str = "") -> Brief:
        """Record a reviewer decision; approval also publishes the brief.

        Only briefs in ``needs_review`` or ``changes_requested`` can be decided;
        anything else (e.g. an already-approved brief) raises ``ValueError``.
        """
        if decision not in ("approved", "rejected", "changes_requested"):
            raise ValueError(f"unknown review decision {decision!r}")
        if not reviewer.strip():
            raise ValueError("reviewer name is required")
        brief = self.load(brief_id)
        if brief.status not in DECIDABLE:
            raise ValueError(
                f"brief {brief_id} is already {brief.status}; only briefs awaiting review can be decided"
            )
        review = ReviewDecision(decision=decision, reviewer=reviewer.strip(), notes=notes)
        status: BriefStatus = decision
        decided = brief.model_copy(update={"status": status, "review": review})
        if decision == "approved":
            # Imported here: pipeline imports this module.
            from marketbrief.pipeline import publish_brief

            publish_brief(decided, self.settings)
        # Persist only after a successful publish, so a failed publish leaves
        # the brief decidable.
        _write_atomic(self.path_for(brief_id), decided.model_dump_json(indent=2))
        logger.info("brief %s %s by %s", brief_id, decision, reviewer)
        return decided

    def list(self, status: BriefStatus | None = None) -> list[Brief]:
        """All queued briefs (optionally filtered by status), newest first."""
        briefs: list[Brief] = []
        for path in sorted(self.dir.glob("*.json")):
            try:
                brief = Brief.model_validate_json(path.read_text(encoding="utf-8"))
            except ValueError as exc:
                logger.warning("skipping unreadable review file %s: %s", path, exc)
                continue
            if status is None or brief.status == status:
                briefs.append(brief)
        briefs.sort(key=lambda b: b.created_at, reverse=True)
        return briefs


# --------------------------------------------------------------------------- #
# Interactive terminal review
# --------------------------------------------------------------------------- #

_CHOICES: dict[str, Decision | None] = {
    "a": "approved",
    "r": "rejected",
    "c": "changes_requested",
    "s": None,
}


def interactive_review(
    brief: Brief,
    *,
    reviewer: str,
    ask: Callable[[str], str] | None = None,
    console: object | None = None,
) -> ReviewDecision | None:
    """Show ``brief`` in the terminal and ask for a decision.

    Returns ``None`` when the reviewer skips (the brief stays queued). ``ask``
    and ``console`` are injectable for tests; defaults use Rich.
    """
    from rich.console import Console
    from rich.prompt import Prompt

    from marketbrief.render import number_sources

    con: Console = console if isinstance(console, Console) else Console()
    con.rule(f"[bold]Review: {brief.title}")
    con.print(f"[dim]{brief.id} · {brief.topic} · {len(brief.sources)} sources[/dim]\n")
    numbers = {e.ref.article_id: e.number for e in number_sources(brief)}

    def show(claims_title: str, claims: list) -> None:
        if not claims:
            return
        con.print(f"[bold cyan]{claims_title}[/bold cyan]")
        for claim in claims:
            refs = "".join(f"[{numbers[c]}]" for c in claim.citations if c in numbers)
            flag = {"weak": " [yellow](weak)[/yellow]", "unsupported": " [red](unsupported)[/red]"}.get(
                claim.verification, ""
            )
            con.print(f"  • {claim.text} [dim]{refs}[/dim]{flag}", highlight=False)
        con.print()

    show("Executive summary", brief.executive_summary)
    for section in brief.sections:
        show(section.title, section.claims)
    show("Implications", brief.implications)

    if brief.validation:
        v = brief.validation
        colour = "green" if v.passed else "red"
        con.print(
            f"[{colour}]Citation check {'passed' if v.passed else 'FAILED'}[/{colour}]: "
            f"{v.supported} supported, {v.weak} weak, {v.unsupported} unsupported of {v.claims_checked}"
        )
        for issue in v.issues[:10]:
            con.print(f"  [{'red' if issue.severity == 'error' else 'yellow'}]{issue.severity}[/] "
                      f"{issue.claim_id} {issue.kind}: {issue.detail}", highlight=False)
        con.print()

    prompt = "Decision: [a]pprove, [r]eject, request [c]hanges, [s]kip (leave queued)"
    if ask is None:
        answer = Prompt.ask(prompt, choices=list(_CHOICES), default="s", console=con)
    else:
        answer = ask(prompt).strip().lower()[:1] or "s"
        if answer not in _CHOICES:
            raise ValueError(f"unknown review choice {answer!r}")
    decision = _CHOICES[answer]
    if decision is None:
        return None
    notes = Prompt.ask("Notes (optional)", default="", console=con) if ask is None else ask("Notes (optional)")
    return ReviewDecision(decision=decision, reviewer=reviewer, notes=notes)
