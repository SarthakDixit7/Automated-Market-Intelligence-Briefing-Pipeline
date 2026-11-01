"""Command-line interface: ``marketbrief`` (or ``python -m marketbrief``).

Exit codes: 0 success, 1 failure (pipeline error, failed eval, missing brief),
2 usage error (bad option or argument).
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, time, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

if TYPE_CHECKING:
    from marketbrief.config import Settings


def _tolerate_narrow_consoles() -> None:
    """Never crash on characters a legacy console code page can't encode.

    Brief titles and article text can contain any Unicode. Pipes and files
    get UTF-8; an interactive legacy (e.g. cp1252) console keeps its code page
    and shows unencodable characters as '?' instead of raising.
    """
    for stream in (sys.stdout, sys.stderr):
        encoding = (getattr(stream, "encoding", None) or "").lower().replace("-", "")
        if encoding == "utf8" or not hasattr(stream, "reconfigure"):
            continue
        try:
            if stream.isatty():
                stream.reconfigure(errors="replace")
            else:
                stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # detached or closed stream
            pass


_tolerate_narrow_consoles()

console = Console()
err_console = Console(stderr=True)

app = typer.Typer(
    name="marketbrief",
    help=(
        "Automated market-intelligence briefings: ingest news, de-duplicate, extract typed records, "
        "write a cited brief, validate it, and route it through human review before delivery.\n\n"
        "Runs fully offline with the deterministic [bold]offline[/bold] provider; set ANTHROPIC_API_KEY "
        "to use Claude. Try [bold]marketbrief demo[/bold] first."
    ),
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)
review_app = typer.Typer(help="Human review gate: list, inspect and decide queued briefs.", no_args_is_help=True)
index_app = typer.Typer(help="Retrieval index over approved prior briefs.", no_args_is_help=True)
cache_app = typer.Typer(help="LLM prompt cache (exact + semantic hits).", no_args_is_help=True)
app.add_typer(review_app, name="review")
app.add_typer(index_app, name="index")
app.add_typer(cache_app, name="cache")

PROVIDERS = ("auto", "anthropic", "offline")
STATUSES = ("draft", "needs_review", "approved", "rejected", "changes_requested")

# --------------------------------------------------------------------------- #
# Shared option definitions
# --------------------------------------------------------------------------- #

ConfigOpt = typer.Option(
    None,
    "--config",
    "-c",
    help="Path to settings.yaml (default: config/settings.yaml under the project root).",
    exists=True,
    dir_okay=False,
)
ProviderOpt = typer.Option(
    None,
    "--provider",
    "-p",
    help="LLM provider: auto (Claude if ANTHROPIC_API_KEY is set, else offline), anthropic, or offline.",
    case_sensitive=False,
)
SinceOpt = typer.Option(None, "--since", help="Only articles published on/after this ISO date, e.g. 2026-10-01.")
UntilOpt = typer.Option(None, "--until", help="Only articles published on/before this ISO date (inclusive).")
SourcesOpt = typer.Option(
    None, "--sources", help="Comma-separated source ids to ingest (see `marketbrief sources`). Default: all enabled."
)


@app.callback()
def main(
    ctx: typer.Context,
    config: Optional[Path] = ConfigOpt,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show INFO-level logs from every stage."),
) -> None:
    """Global options apply to every command."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = config
    _setup_logging(logging.INFO if verbose else logging.WARNING)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _setup_logging(level: int) -> None:
    from rich.logging import RichHandler

    root = logging.getLogger()
    if not any(isinstance(h, RichHandler) for h in root.handlers):
        handler = RichHandler(console=err_console, show_path=False, rich_tracebacks=False)
        handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
        root.addHandler(handler)
    root.setLevel(level)


def _settings(ctx: typer.Context, config: Path | None = None, provider: str | None = None) -> Settings:
    """Load settings from the command's --config, else the global one."""
    from marketbrief.config import load_settings

    path = config or (ctx.obj or {}).get("config")
    overrides: dict = {}
    if provider:
        provider = provider.lower()
        if provider not in PROVIDERS:
            raise typer.BadParameter(f"must be one of {', '.join(PROVIDERS)}", param_hint="--provider")
        overrides["llm"] = {"provider": provider}
    return load_settings(path, overrides=overrides or None)


def _parse_date(value: str | None, *, end_of_day: bool, option: str) -> datetime | None:
    """Parse an ISO date/datetime; bare dates in --until cover the whole day."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise typer.BadParameter(f"{value!r} is not an ISO date (YYYY-MM-DD)", param_hint=option) from None
    if end_of_day and len(value) <= 10:
        parsed = datetime.combine(parsed.date(), time.max)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _split(value: str | None) -> list[str] | None:
    items = [v.strip() for v in (value or "").split(",") if v.strip()]
    return items or None


@contextmanager
def _errors() -> Iterator[None]:
    """Turn expected failures into a red message and exit code 1."""
    from marketbrief.llm.base import LLMError

    try:
        yield
    except typer.Exit:
        raise
    except KeyError as exc:  # KeyError str() wraps the message in quotes
        err_console.print(f"[red]Error:[/red] {escape(str(exc.args[0] if exc.args else exc))}")
        raise typer.Exit(1) from None
    except (LLMError, ValueError, OSError, RuntimeError) as exc:
        err_console.print(f"[red]Error:[/red] {escape(str(exc))}")
        logging.getLogger(__name__).info("command failed", exc_info=True)
        raise typer.Exit(1) from None


def _progress(message: str) -> None:
    console.print(f"[dim]•[/dim] {escape(message)}", highlight=False)


def _status_style(status: str) -> str:
    return {
        "approved": "green",
        "needs_review": "yellow",
        "changes_requested": "magenta",
        "rejected": "red",
    }.get(status, "white")


def _print_paths(title: str, paths: dict[str, Path]) -> None:
    table = Table(title=title, show_header=False, box=None, title_justify="left", title_style="bold")
    table.add_column(style="cyan", no_wrap=True)
    table.add_column(overflow="fold")
    for key, path in paths.items():
        table.add_row(key, str(path))
    console.print(table)


def _print_run(run) -> None:  # run: PipelineRun
    r, brief = run.report, run.brief
    status = f"[{_status_style(brief.status)}]{brief.status}[/]"
    table = Table(title="Run summary", show_header=False, title_justify="left", title_style="bold")
    table.add_column(style="cyan", no_wrap=True)
    table.add_column(overflow="fold")
    rows = [
        ("Run", r.run_id),
        ("Brief", brief.id),
        ("Title", escape(brief.title)),
        ("Status", status),
        ("Provider", f"{r.provider} ({r.model})"),
        ("Articles", f"{r.articles_fetched} fetched -> {r.articles_after_dedup} after dedup "
                     f"({r.duplicate_clusters} duplicate clusters)"),
        ("Records", str(r.records_extracted)),
        ("Claims", f"{len(brief.all_claims())} citing {len(brief.sources)} sources"),
    ]
    if brief.validation:
        v = brief.validation
        ok = "[green]passed[/]" if v.passed else "[red]issues found[/]"
        rows.append(("Citations", f"{ok}: {v.supported} supported, {v.weak} weak, {v.unsupported} unsupported"
                                  f"{' (fact-checked)' if v.fact_checked else ''}"))
    rows.append(("LLM", f"{r.llm_calls} calls, {r.input_tokens:,} in / {r.output_tokens:,} out tokens, "
                        f"prompt cache {r.prompt_cache_hits} hits / {r.prompt_cache_misses} misses"))
    rows.append(("Time", " · ".join(f"{s.stage} {s.seconds:.2f}s" for s in r.stages)))
    for key, value in rows:
        table.add_row(key, value)
    console.print(table)
    for warning in r.warnings:
        console.print(f"[yellow]warning:[/yellow] {escape(warning)}")
    _print_paths("Deliverables", run.deliverables)
    if brief.status in ("needs_review", "changes_requested"):
        console.print("\nNext steps:")
        console.print(f"  [bold]marketbrief review show {brief.id}[/bold]", soft_wrap=True)
        console.print(f"  [bold]marketbrief review approve {brief.id} --reviewer <name>[/bold]", soft_wrap=True)


# --------------------------------------------------------------------------- #
# Pipeline commands
# --------------------------------------------------------------------------- #


@app.command()
def run(
    ctx: typer.Context,
    topic: Optional[str] = typer.Option(None, "--topic", "-t", help="Brief topic (default: pipeline.default_topic)."),
    since: Optional[str] = SinceOpt,
    until: Optional[str] = UntilOpt,
    provider: Optional[str] = ProviderOpt,
    sources: Optional[str] = SourcesOpt,
    auto_approve: Optional[bool] = typer.Option(
        None,
        "--auto-approve/--interactive",
        help="--auto-approve publishes immediately as --reviewer; --interactive asks for a decision in the "
        "terminal. Default: queue the brief for `marketbrief review`.",
    ),
    reviewer: str = typer.Option("auto", "--reviewer", help="Reviewer name recorded on the decision."),
    config: Optional[Path] = ConfigOpt,
) -> None:
    """Run the full pipeline and produce a cited brief (queued for review by default)."""
    since_dt = _parse_date(since, end_of_day=False, option="--since")
    until_dt = _parse_date(until, end_of_day=True, option="--until")
    mode = "queue" if auto_approve is None else ("auto_approve" if auto_approve else "interactive")
    with _errors():
        from marketbrief.pipeline import run_pipeline

        settings = _settings(ctx, config, provider)
        console.print(f"[bold]marketbrief[/bold] · provider [cyan]{settings.resolved_provider()}[/cyan] · "
                      f"review [cyan]{mode}[/cyan]")
        result = run_pipeline(
            settings,
            topic=topic,
            since=since_dt,
            until=until_dt,
            source_ids=_split(sources),
            review_mode=mode,
            reviewer=reviewer,
            progress=_progress,
        )
        _print_run(result)


@app.command()
def ingest(
    ctx: typer.Context,
    since: Optional[str] = SinceOpt,
    until: Optional[str] = UntilOpt,
    sources: Optional[str] = SourcesOpt,
    config: Optional[Path] = ConfigOpt,
) -> None:
    """Fetch and normalise sources into structured JSON (no LLM calls)."""
    since_dt = _parse_date(since, end_of_day=False, option="--since")
    until_dt = _parse_date(until, end_of_day=True, option="--until")
    with _errors():
        from marketbrief.ingest import ingest as run_ingest
        from marketbrief.pipeline import new_run_id

        settings = _settings(ctx, config)
        result = run_ingest(settings, source_ids=_split(sources), since=since_dt, until=until_dt, run_id=new_run_id())
        table = Table(title="Ingested", title_justify="left", title_style="bold")
        table.add_column("Source")
        table.add_column("Articles", justify="right")
        for source_id, count in result.per_source.items():
            table.add_row(source_id, str(count))
        console.print(table)
        console.print(f"{len(result.articles)} articles -> [cyan]{result.raw_path}[/cyan]")
        for error in result.errors:
            console.print(f"[yellow]warning:[/yellow] {escape(error)}")


@app.command()
def sources(ctx: typer.Context, config: Optional[Path] = ConfigOpt) -> None:
    """List configured news sources (config/sources.yaml)."""
    with _errors():
        from marketbrief.ingest.sources import load_sources

        settings = _settings(ctx, config)
        table = Table(title="Sources", title_justify="left", title_style="bold")
        for col in ("id", "name", "kind", "location", "category", "enabled"):
            table.add_column(col, overflow="fold")
        for src in load_sources(settings.path(settings.paths.sources_file)):
            enabled = "[green]yes[/]" if src.enabled else "[dim]no[/]"
            table.add_row(src.id, escape(src.name), src.kind, src.url or src.path or "", src.category, enabled)
        console.print(table)


@app.command()
def demo(ctx: typer.Context, config: Optional[Path] = ConfigOpt) -> None:
    """Offline end-to-end demo: run the pipeline, auto-approve as 'demo', then evaluate."""
    with _errors():
        from marketbrief.eval.runner import run_eval
        from marketbrief.llm.factory import build_llm
        from marketbrief.pipeline import run_pipeline

        settings = _settings(ctx, config, "offline")
        llm = build_llm(settings)
        console.print("[bold]marketbrief demo[/bold] · offline provider · synthetic fixture feed\n")
        result = run_pipeline(settings, llm=llm, review_mode="auto_approve", reviewer="demo", progress=_progress)
        _print_run(result)
        console.print()
        evaluation = run_eval(settings, brief_path=result.deliverables["archive"], llm=llm)
        _print_eval(evaluation)
    if not evaluation.passed:
        raise typer.Exit(1)


# --------------------------------------------------------------------------- #
# Review commands
# --------------------------------------------------------------------------- #


@review_app.command("list")
def review_list(
    ctx: typer.Context,
    status: Optional[str] = typer.Option(None, "--status", "-s", help=f"Filter by status: {', '.join(STATUSES)}."),
    config: Optional[Path] = ConfigOpt,
) -> None:
    """List briefs in the review queue, newest first."""
    if status and status not in STATUSES:
        raise typer.BadParameter(f"must be one of {', '.join(STATUSES)}", param_hint="--status")
    with _errors():
        from marketbrief.review import ReviewQueue

        briefs = ReviewQueue(_settings(ctx, config)).list(status)  # type: ignore[arg-type]
        if not briefs:
            console.print("Review queue is empty." if not status else f"No briefs with status {status}.")
            return
        table = Table(title="Review queue", title_justify="left", title_style="bold")
        # Ids are copied into other commands, so fixed columns keep their full
        # width and only the title is shortened to fit the terminal.
        fixed = {
            "brief id": max(len(b.id) for b in briefs),
            "status": max(len(b.status) for b in briefs),
            "created": 10,
            "claims": 6,
            "check": 6,
        }
        for name, width in fixed.items():
            table.add_column(name, no_wrap=True, min_width=width, justify="right" if name == "claims" else "left")
        table.add_column("title", no_wrap=True)
        borders_and_padding = 3 * (len(fixed) + 1) + 1
        title_budget = max(8, console.width - sum(fixed.values()) - borders_and_padding)
        for b in briefs:
            check = "-" if b.validation is None else ("[green]ok[/]" if b.validation.passed else "[red]issues[/]")
            title = b.title if len(b.title) <= title_budget else b.title[: title_budget - 1] + "…"
            table.add_row(b.id, f"[{_status_style(b.status)}]{b.status}[/]", f"{b.created_at:%Y-%m-%d}",
                          str(len(b.all_claims())), check, escape(title))
        console.print(table)


@review_app.command("show")
def review_show(
    ctx: typer.Context,
    brief_id: str = typer.Argument(..., help="Brief id, e.g. b_20261030_1a2b3c."),
    as_json: bool = typer.Option(False, "--json", help="Print the raw brief JSON instead of the rendered brief."),
    config: Optional[Path] = ConfigOpt,
) -> None:
    """Show a queued brief with its citation-validation results."""
    with _errors():
        from rich.markdown import Markdown

        from marketbrief.render import render_json, render_markdown
        from marketbrief.review import ReviewQueue

        brief = ReviewQueue(_settings(ctx, config)).load(brief_id)
        if as_json:
            console.print_json(render_json(brief))
            return
        console.print(Markdown(render_markdown(brief)))
        if brief.validation and brief.validation.issues:
            console.rule("Validation issues")
            for issue in brief.validation.issues:
                colour = "red" if issue.severity == "error" else "yellow"
                console.print(f"[{colour}]{issue.severity}[/] {issue.claim_id} {issue.kind}: {escape(issue.detail)}")


def _decide(ctx: typer.Context, brief_id: str, decision: str, reviewer: str, notes: str, config: Path | None) -> None:
    with _errors():
        from marketbrief.pipeline import published_paths
        from marketbrief.review import ReviewQueue

        settings = _settings(ctx, config)
        brief = ReviewQueue(settings).decide(brief_id, decision, reviewer, notes)  # type: ignore[arg-type]
        console.print(f"Brief [bold]{brief.id}[/bold] is now [{_status_style(brief.status)}]{brief.status}[/].")
        if brief.status == "approved":
            _print_paths("Published", published_paths(brief, settings))


ReviewerOpt = typer.Option(..., "--reviewer", "-r", help="Your name, recorded on the review decision.")
NotesOpt = typer.Option("", "--notes", "-n", help="Reviewer notes stored with the decision.")
BriefIdArg = typer.Argument(..., help="Brief id from `marketbrief review list`.")


@review_app.command("approve")
def review_approve(
    ctx: typer.Context,
    brief_id: str = BriefIdArg,
    reviewer: str = ReviewerOpt,
    notes: str = NotesOpt,
    config: Optional[Path] = ConfigOpt,
) -> None:
    """Approve a brief and publish it to output/client/ (also indexes it for future continuity)."""
    _decide(ctx, brief_id, "approved", reviewer, notes, config)


@review_app.command("reject")
def review_reject(
    ctx: typer.Context,
    brief_id: str = BriefIdArg,
    reviewer: str = ReviewerOpt,
    notes: str = NotesOpt,
    config: Optional[Path] = ConfigOpt,
) -> None:
    """Reject a brief; it is never delivered."""
    _decide(ctx, brief_id, "rejected", reviewer, notes, config)


@review_app.command("request-changes")
def review_request_changes(
    ctx: typer.Context,
    brief_id: str = BriefIdArg,
    reviewer: str = ReviewerOpt,
    notes: str = NotesOpt,
    config: Optional[Path] = ConfigOpt,
) -> None:
    """Send a brief back for changes; it stays in the queue and can be decided again."""
    _decide(ctx, brief_id, "changes_requested", reviewer, notes, config)


# --------------------------------------------------------------------------- #
# Analyst agent
# --------------------------------------------------------------------------- #


def _print_agent(result) -> None:  # result: AnalystResult
    state = result.state
    console.print(f"[dim]trace:[/dim] {' -> '.join(result.trace)}")
    if state.chart_path:
        console.print(f"[dim]chart:[/dim] [cyan]{state.chart_path}[/cyan]")
    if state.answer:
        console.rule("Answer")
        console.print(escape(state.answer))
        if state.citations:
            console.print(f"[dim]sources: {', '.join(state.citations)}[/dim]")
    if result.status == "interrupted":
        console.print("\n[yellow]Held for human review[/yellow] (sensitive topic). Resume with:")
        console.print(
            f"  [bold]marketbrief agent-resume {result.thread_id} --approve[/bold]  (or --reject)", soft_wrap=True
        )
    elif state.approved is not None:
        verdict = "[green]approved[/]" if state.approved else "[red]rejected[/]"
        console.print(f"\nReviewer decision: {verdict}{' — ' + escape(state.reviewer_notes) if state.reviewer_notes else ''}")


@app.command()
def agent(
    ctx: typer.Context,
    question: str = typer.Argument(..., help='Question for the analyst agent, e.g. "Who raised money this month?"'),
    chart: bool = typer.Option(False, "--chart", help="Also draw a chart (implied when the question mentions a chart, plot or graph)."),
    auto_approve: Optional[bool] = typer.Option(
        None,
        "--auto-approve/--no-auto-approve",
        help="Decide the human-review checkpoint for sensitive answers immediately. "
        "Default: pause and print a thread id for `agent-resume`.",
    ),
    max_steps: int = typer.Option(8, "--max-steps", min=2, help="Maximum planner steps."),
    provider: Optional[str] = ProviderOpt,
    config: Optional[Path] = ConfigOpt,
) -> None:
    """Ask the analyst agent (search -> summarise -> chart -> answer, with a HITL checkpoint)."""
    with _errors():
        from marketbrief.agent.analyst import run_analyst

        settings = _settings(ctx, config, provider)
        result = run_analyst(
            question,
            settings,
            auto_approve=auto_approve,
            max_steps=max_steps,
            wants_chart=True if chart else None,  # None: inferred from the question
        )
        _print_agent(result)


@app.command("agent-resume")
def agent_resume(
    ctx: typer.Context,
    thread_id: str = typer.Argument(..., help="Thread id printed when the agent paused for review."),
    approve: bool = typer.Option(..., "--approve/--reject", help="Approve or reject the held answer."),
    notes: str = typer.Option("", "--notes", "-n", help="Reviewer notes."),
    provider: Optional[str] = ProviderOpt,
    config: Optional[Path] = ConfigOpt,
) -> None:
    """Resume an analyst-agent run paused at its human-review checkpoint."""
    with _errors():
        from marketbrief.agent.analyst import resume_analyst

        settings = _settings(ctx, config, provider)
        result = resume_analyst(thread_id, settings, approved=approve, notes=notes)
        _print_agent(result)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


def _print_eval(result) -> None:  # result: EvalResult
    from marketbrief.eval.judge import judge_mean

    table = Table(title="Evaluation", show_header=False, title_justify="left", title_style="bold")
    table.add_column(style="cyan", no_wrap=True)
    table.add_column(overflow="fold")
    recall = result.recall
    table.add_row("Prompt version", result.prompt_version)
    table.add_row("Extraction recall", f"{recall.extraction_recall:.0%} of {recall.total} golden events")
    table.add_row("Brief recall", f"{recall.brief_recall:.0%}")
    if recall.must_catch_missed:
        table.add_row("Must-catch missed", ", ".join(recall.must_catch_missed))
    if result.judge is not None:
        j = result.judge
        table.add_row("Judge", f"factuality {j.factuality}/5 · coverage {j.coverage}/5 · consistency "
                               f"{j.consistency}/5 (mean {judge_mean(j):.2f})")
    if result.citation is not None:
        c = result.citation
        table.add_row("Citations", f"{'passed' if c.passed else 'failed'}: {c.supported}/{c.claims_checked} supported")
    table.add_row("Result", "[green]PASSED[/]" if result.passed else "[red]FAILED[/]")
    if result.report_path:
        table.add_row("Report", str(result.report_path))
    console.print(table)
    for failure in result.failures:
        console.print(f"[red]x[/red] {escape(failure)}")


@app.command("eval")
def eval_cmd(
    ctx: typer.Context,
    brief: Optional[Path] = typer.Option(
        None, "--brief", help="Evaluate this brief JSON. Default: run the pipeline (queue mode) and evaluate its draft.",
        exists=True, dir_okay=False,
    ),
    no_judge: bool = typer.Option(False, "--no-judge", help="Skip the LLM-as-a-judge pass (recall + citations only)."),
    fail_under_recall: Optional[float] = typer.Option(
        None, "--fail-under-recall", min=0.0, max=1.0,
        help="Also fail (exit 1) when extraction or brief recall is below this fraction, e.g. 0.9.",
    ),
    provider: Optional[str] = ProviderOpt,
    config: Optional[Path] = ConfigOpt,
) -> None:
    """Score a brief: golden-event recall, citation checks and LLM judge. Exits 1 when it fails."""
    with _errors():
        from marketbrief.eval.runner import run_eval

        settings = _settings(ctx, config, provider)
        result = run_eval(settings, brief_path=brief, use_judge=not no_judge)
        _print_eval(result)
    failed = not result.passed
    if fail_under_recall is not None:
        low = min(result.recall.extraction_recall, result.recall.brief_recall)
        if low < fail_under_recall:
            console.print(f"[red]x[/red] recall {low:.0%} is below --fail-under-recall {fail_under_recall:.0%}")
            failed = True
    if failed:
        raise typer.Exit(1)


# --------------------------------------------------------------------------- #
# Index + cache maintenance
# --------------------------------------------------------------------------- #


@index_app.command("rebuild")
def index_rebuild(ctx: typer.Context, config: Optional[Path] = ConfigOpt) -> None:
    """Rebuild the prior-brief index from data/briefs and the configured extra brief dirs."""
    with _errors():
        from marketbrief.retrieval import build_index

        settings = _settings(ctx, config)
        dirs = [settings.path(settings.paths.briefs_dir)] + [settings.path(d) for d in settings.retrieval.extra_brief_dirs]
        index = build_index(settings)
        briefs = index.rebuild([d for d in dirs if d.exists()])
        console.print(f"Indexed {len(index)} chunks from {briefs} approved briefs "
                      f"-> [cyan]{settings.path(settings.retrieval.index_dir)}[/cyan]")


@cache_app.command("stats")
def cache_stats(ctx: typer.Context, config: Optional[Path] = ConfigOpt) -> None:
    """Show prompt-cache size and hit rates from recent runs."""
    with _errors():
        settings = _settings(ctx, config)
        path = settings.path(settings.cache.path)
        table = Table(title="Prompt cache", show_header=False, title_justify="left", title_style="bold")
        table.add_column(style="cyan", no_wrap=True)
        table.add_column(overflow="fold")
        table.add_row("Enabled", "yes" if settings.cache.enabled else "no")
        table.add_row("Path", str(path))
        if path.exists():
            from marketbrief.llm.cache import PromptCache

            table.add_row("Entries", str(len(PromptCache(path))))
            table.add_row("Size", f"{path.stat().st_size / 1024:.1f} KiB")
        else:
            table.add_row("Entries", "0 (no cache file yet)")
        table.add_row("Semantic tasks", ", ".join(settings.cache.semantic_tasks) or "none")
        hits, misses, runs = _recent_cache_counts(settings)
        if runs:
            rate = hits / (hits + misses) if hits + misses else 0.0
            table.add_row(f"Last {runs} runs", f"{hits} hits / {misses} misses ({rate:.0%} hit rate)")
        console.print(table)


def _recent_cache_counts(settings: Settings, limit: int = 10) -> tuple[int, int, int]:
    from marketbrief.schemas import RunReport

    runs_dir = settings.path(settings.paths.runs_dir)
    reports = sorted(runs_dir.glob("r_*.json"), reverse=True)[:limit] if runs_dir.exists() else []
    hits = misses = 0
    for path in reports:
        try:
            report = RunReport.model_validate_json(path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        hits += report.prompt_cache_hits
        misses += report.prompt_cache_misses
    return hits, misses, len(reports)


@cache_app.command("clear")
def cache_clear(
    ctx: typer.Context,
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask for confirmation."),
    config: Optional[Path] = ConfigOpt,
) -> None:
    """Delete every cached LLM response (the next run is billed in full)."""
    with _errors():
        settings = _settings(ctx, config)
        path = settings.path(settings.cache.path)
        if not path.exists():
            console.print("Cache is already empty.")
            return
        if not yes and not typer.confirm(f"Clear the prompt cache at {path}?"):
            raise typer.Exit(1)
        from marketbrief.llm.cache import PromptCache

        cache = PromptCache(path)
        count = len(cache)
        cache.clear()
        console.print(f"Cleared {count} cached responses.")


if __name__ == "__main__":  # pragma: no cover
    app()
