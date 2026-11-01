"""Render a :class:`Brief` into client deliverables: Markdown, HTML and JSON.

Both text formats share one view model: sources are numbered in order of first
citation, and every claim carries footnote markers that point at the numbered
source list. Templates live in ``marketbrief/templates``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from jinja2 import Environment, PackageLoader, StrictUndefined, select_autoescape

from marketbrief.schemas import Brief, Claim, SourceRef

STATUS_LABELS: dict[str, str] = {
    "draft": "Draft",
    "needs_review": "Awaiting review",
    "approved": "Approved",
    "rejected": "Rejected",
    "changes_requested": "Changes requested",
}

RELATION_LABELS: dict[str, str] = {
    "continues": "Continues",
    "updates": "Updates",
    "contradicts": "Contradicts",
}


@dataclass(frozen=True)
class SourceEntry:
    """A numbered entry in the brief's source list."""

    number: int
    anchor: str
    ref: SourceRef


def number_sources(brief: Brief) -> list[SourceEntry]:
    """Number sources in order of first citation; uncited sources go last."""
    by_id = {s.article_id: s for s in brief.sources}
    ordered: list[str] = []
    seen: set[str] = set()
    for claim in brief.all_claims():
        for article_id in claim.citations:
            if article_id in by_id and article_id not in seen:
                seen.add(article_id)
                ordered.append(article_id)
    ordered.extend(s.article_id for s in brief.sources if s.article_id not in seen)
    return [SourceEntry(number=i, anchor=f"src-{i}", ref=by_id[aid]) for i, aid in enumerate(ordered, 1)]


def _format_date(value: datetime | None, fmt: str = "%d %b %Y") -> str:
    return value.strftime(fmt) if value else ""


_RELATION_PREFIX = re.compile(r"^\s*(continues|updates|contradicts)\s*[:\-–—]\s*", re.IGNORECASE)


def _strip_relation(note: str) -> str:
    """Drop a leading "Updates:"-style prefix; templates show the relation as a label."""
    stripped = _RELATION_PREFIX.sub("", note, count=1)
    return stripped[:1].upper() + stripped[1:] if stripped else note


def _markdown_text(text: str) -> str:
    """Keep literal angle brackets in source text from being read as HTML."""
    return text.replace("<", "\\<")


def _format_period(brief: Brief) -> str:
    start, end = brief.period_start, brief.period_end
    if start and end:
        if start.date() == end.date():
            return _format_date(start)
        return f"{_format_date(start)} – {_format_date(end)}"
    return _format_date(start or end)


@lru_cache(maxsize=1)
def _environment() -> Environment:
    env = Environment(
        loader=PackageLoader("marketbrief", "templates"),
        # Escape only the HTML template; Markdown is rendered verbatim.
        autoescape=select_autoescape(enabled_extensions=("html.j2",), default_for_string=False),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    env.filters["date"] = _format_date
    env.filters["strip_relation"] = _strip_relation
    env.filters["md"] = _markdown_text
    return env


def _context(brief: Brief) -> dict[str, Any]:
    sources = number_sources(brief)
    number_of = {entry.ref.article_id: entry for entry in sources}

    def footnotes(claim: Claim) -> list[SourceEntry]:
        """Source entries cited by ``claim``, de-duplicated, in citation order."""
        out: list[SourceEntry] = []
        for article_id in dict.fromkeys(claim.citations):
            if article_id in number_of:
                out.append(number_of[article_id])
        return out

    validation = brief.validation
    return {
        "brief": brief,
        "sources": sources,
        "footnotes": footnotes,
        "status_label": STATUS_LABELS.get(brief.status, brief.status),
        "is_approved": brief.status == "approved",
        "period": _format_period(brief),
        "relation_labels": RELATION_LABELS,
        "validation": validation,
        "claim_count": len(brief.all_claims()),
    }


def render_markdown(brief: Brief) -> str:
    """Brief as Markdown with ``[^n]`` footnotes resolving to the source list."""
    return _environment().get_template("brief.md.j2").render(**_context(brief))


def render_html(brief: Brief) -> str:
    """Self-contained, print-friendly HTML (inline CSS, no external assets)."""
    return _environment().get_template("brief.html.j2").render(**_context(brief))


def render_json(brief: Brief) -> str:
    """Machine-readable brief (the full validated model)."""
    return brief.model_dump_json(indent=2) + "\n"


def write_deliverables(brief: Brief, out_dir: Path) -> dict[str, Path]:
    """Write ``<brief_id>.md/.html/.json`` into ``out_dir``; return paths by format."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    renderers = {"md": render_markdown, "html": render_html, "json": render_json}
    paths: dict[str, Path] = {}
    for ext, render in renderers.items():
        path = out_dir / f"{brief.id}.{ext}"
        path.write_text(render(brief), encoding="utf-8")
        paths[ext] = path
    return paths
