"""End-to-end briefing pipeline.

ingest → dedup → extract → map → retrieve prior context → reduce → edit →
validate citations (+ fact-check) → repair → continuity notes → assemble →
render drafts → human review gate → publish.

Every stage is timed into a :class:`RunReport`, which is saved to
``data/runs/<run_id>.json`` together with LLM usage and prompt-cache counters.
"""

from __future__ import annotations

import logging
import secrets
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from marketbrief.config import Settings
from marketbrief.llm.base import LLMClient
from marketbrief.render import write_deliverables
from marketbrief.review import ReviewQueue, interactive_review
from marketbrief.schemas import (
    Article,
    Brief,
    BriefDraft,
    CitationReport,
    ContinuityNote,
    RunReport,
    SourceRef,
    StageTiming,
)

logger = logging.getLogger(__name__)

ReviewMode = Literal["queue", "auto_approve", "interactive"]
ProgressFn = Callable[[str], None]


class PipelineError(RuntimeError):
    """The pipeline cannot produce a brief (e.g. nothing was ingested)."""


@dataclass
class PipelineRun:
    """Outcome of one pipeline run.

    ``deliverables`` points at the client copies when the brief was approved
    and published during the run, otherwise at the drafts awaiting review.
    """

    run_id: str
    brief: Brief
    report: RunReport
    deliverables: dict[str, Path]
    review_path: Path | None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def new_run_id(now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    return f"r_{now:%Y%m%d%H%M%S}_{secrets.token_hex(2)}"


def new_brief_id(day: datetime | None = None) -> str:
    day = day or datetime.now(timezone.utc)
    return f"b_{day:%Y%m%d}_{secrets.token_hex(3)}"


def _article_time(article: Article) -> datetime:
    return article.published_at or article.fetched_at


class _UsageMeter:
    """Reads LLM usage counters before/after a run so reports show the delta.

    Uses the ``UsageTracker`` returned by ``build_llm``; any other client is
    wrapped in one so calls are still counted.
    """

    def __init__(self, llm: LLMClient) -> None:
        if not callable(getattr(llm, "snapshot", None)):
            from marketbrief.llm.factory import UsageTracker

            llm = UsageTracker(llm)
        self.llm = llm
        self._start = self._read()

    def _read(self) -> dict[str, int]:
        snap: dict[str, Any] = self.llm.snapshot()  # type: ignore[attr-defined]
        usage: dict[str, Any] = snap.get("usage") or {}
        return {
            "calls": int(snap.get("calls", 0)),
            "input_tokens": int(usage.get("input_tokens", 0)),
            "output_tokens": int(usage.get("output_tokens", 0)),
            "cache_read_input_tokens": int(usage.get("cache_read_input_tokens", 0)),
            "cache_hits": int(snap.get("cache_hits", 0)),
            "cache_misses": int(snap.get("cache_misses", 0)),
        }

    def apply(self, report: RunReport) -> None:
        now = self._read()
        delta = {k: now[k] - self._start.get(k, 0) for k in now}
        report.llm_calls = delta["calls"]
        report.input_tokens = delta["input_tokens"]
        report.output_tokens = delta["output_tokens"]
        report.cache_read_input_tokens = delta["cache_read_input_tokens"]
        report.prompt_cache_hits = delta["cache_hits"]
        report.prompt_cache_misses = delta["cache_misses"]


class _StageClock:
    """Times named stages into the run report and emits progress messages."""

    def __init__(self, report: RunReport, progress: ProgressFn | None) -> None:
        self.report = report
        self.progress = progress

    def say(self, message: str) -> None:
        logger.info(message)
        if self.progress:
            self.progress(message)

    @contextmanager
    def stage(self, name: str, items_in: int = 0) -> Iterator[StageTiming]:
        timing = StageTiming(stage=name, seconds=0.0, items_in=items_in)
        start = perf_counter()
        try:
            yield timing
        finally:
            timing.seconds = round(perf_counter() - start, 4)
            self.report.stages.append(timing)


def _source_refs(brief_claims_citations: list[list[str]], articles: dict[str, Article]) -> list[SourceRef]:
    """One SourceRef per cited article, in order of first citation."""
    refs: list[SourceRef] = []
    seen: set[str] = set()
    for citations in brief_claims_citations:
        for article_id in citations:
            if article_id in seen or article_id not in articles:
                continue
            seen.add(article_id)
            a = articles[article_id]
            refs.append(
                SourceRef(
                    article_id=a.id,
                    title=a.title,
                    url=a.url,
                    source_name=a.source_name,
                    published_at=a.published_at,
                    corroborating_sources=list(a.corroborating_sources),
                )
            )
    return refs


def assemble_brief(
    draft: BriefDraft,
    *,
    topic: str,
    articles: dict[str, Article],
    validation: CitationReport | None,
    continuity: list[ContinuityNote],
    run_id: str,
    prompt_version: str,
    status: Literal["draft", "needs_review"] = "needs_review",
) -> Brief:
    """Turn the validated draft into a :class:`Brief` with sources and metadata.

    ``articles`` are the articles the brief was written from; the period spans
    their dates.
    """
    claims = list(draft.executive_summary) + [c for s in draft.sections for c in s.claims] + list(draft.implications)
    times = sorted(_article_time(a) for a in articles.values())
    period_start, period_end = (times[0], times[-1]) if times else (None, None)
    return Brief(
        id=new_brief_id(period_end),
        topic=topic,
        title=draft.title,
        period_start=period_start,
        period_end=period_end,
        executive_summary=draft.executive_summary,
        sections=[s for s in draft.sections if s.claims],
        implications=draft.implications,
        watchlist=draft.watchlist,
        continuity=continuity,
        continuity_text=list(draft.continuity),
        sources=_source_refs([c.citations for c in claims], articles),
        status=status,
        validation=validation,
        prompt_version=prompt_version,
        run_id=run_id,
    )


def publish_brief(brief: Brief, settings: Settings) -> dict[str, Path]:
    """Ship an approved brief: client deliverables, archive JSON, RAG index.

    Returns the deliverable paths (``md``, ``html``, ``json``) plus ``archive``
    (the brief JSON under ``paths.briefs_dir``). Safe to repeat.
    """
    if brief.status != "approved":
        raise ValueError(f"only approved briefs can be published (brief {brief.id} is {brief.status})")
    from marketbrief.retrieval import build_index

    client_dir = settings.path(settings.paths.output_dir) / "client" / brief.id
    paths = write_deliverables(brief, client_dir)

    archive_dir = settings.path(settings.paths.briefs_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive = archive_dir / f"{brief.id}.json"
    archive.write_text(brief.model_dump_json(indent=2), encoding="utf-8")

    # Approved briefs become retrieval context for future cycles.
    build_index(settings).add_brief(brief)
    logger.info("published brief %s to %s", brief.id, client_dir)
    return {**paths, "archive": archive}


def _save_report(report: RunReport, settings: Settings) -> Path:
    runs_dir = settings.path(settings.paths.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{report.run_id}.json"
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #


def run_pipeline(
    settings: Settings,
    *,
    topic: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    source_ids: list[str] | None = None,
    llm: LLMClient | None = None,
    review_mode: ReviewMode = "queue",
    reviewer: str = "auto",
    progress: ProgressFn | None = None,
) -> PipelineRun:
    """Produce one market-intelligence brief and route it through review.

    ``review_mode``: ``queue`` leaves the brief in the review queue;
    ``auto_approve`` approves it as ``reviewer`` and publishes; ``interactive``
    asks in the terminal. Raises :class:`PipelineError` when nothing usable
    was ingested or extracted.
    """
    if review_mode not in ("queue", "auto_approve", "interactive"):
        raise ValueError(f"unknown review_mode {review_mode!r}")
    if llm is None:
        from marketbrief.llm.factory import build_llm

        llm = build_llm(settings)
    meter = _UsageMeter(llm)
    llm = meter.llm

    run_id = new_run_id()
    topic = topic or settings.pipeline.default_topic
    report = RunReport(
        run_id=run_id,
        topic=topic,
        provider=getattr(llm, "provider", "unknown"),
        model=getattr(llm, "model", "unknown"),
        started_at=datetime.now(timezone.utc),
    )
    clock = _StageClock(report, progress)
    try:
        return _run(settings, report, clock, llm, meter, topic, since, until, source_ids, review_mode, reviewer)
    except Exception as exc:
        report.warnings.append(f"run failed: {type(exc).__name__}: {exc}")
        report.finished_at = datetime.now(timezone.utc)
        meter.apply(report)
        _save_report(report, settings)
        raise


def _run(
    settings: Settings,
    report: RunReport,
    clock: _StageClock,
    llm: LLMClient,
    meter: _UsageMeter,
    topic: str,
    since: datetime | None,
    until: datetime | None,
    source_ids: list[str] | None,
    review_mode: ReviewMode,
    reviewer: str,
) -> PipelineRun:
    # Stage modules are resolved at call time: keeps `import marketbrief.pipeline`
    # light (review/CLI import it) and lets tests swap individual stages.
    from marketbrief.citations import repair_brief, validate_citations
    from marketbrief.dedup import dedupe
    from marketbrief.embeddings import get_embedder
    from marketbrief.extract import extract_records
    from marketbrief.ingest import ingest
    from marketbrief.prompts.common import prompt_version
    from marketbrief.retrieval import build_index, continuity_context, continuity_notes
    from marketbrief.summarize import edit_brief, map_summaries, reduce_brief

    run_id = report.run_id

    # 1. Ingest -------------------------------------------------------------
    with clock.stage("ingest") as t:
        ingested = ingest(settings, source_ids=source_ids, since=since, until=until, run_id=run_id)
        t.items_out = len(ingested.articles)
    report.articles_fetched = len(ingested.articles)
    report.warnings.extend(f"ingest: {err}" for err in ingested.errors)
    if not ingested.articles:
        raise PipelineError(
            "no articles ingested — check the enabled sources (`marketbrief sources`) and the --since/--until window"
        )
    clock.say(f"Ingested {len(ingested.articles)} articles from {len(ingested.per_source)} source(s)")

    # 2. De-duplicate -------------------------------------------------------
    with clock.stage("dedup", len(ingested.articles)) as t:
        embedder = get_embedder(settings)
        deduped = dedupe(
            ingested.articles,
            embedder,
            threshold=settings.dedup.threshold,
            lead_chars=settings.dedup.lead_chars,
        )
        t.items_out = len(deduped.kept)
    report.articles_after_dedup = len(deduped.kept)
    report.duplicate_clusters = len(deduped.clusters)
    clock.say(f"De-duplicated to {len(deduped.kept)} stories ({deduped.removed} duplicates folded)")

    # 3. Extract ------------------------------------------------------------
    with clock.stage("extract", len(deduped.kept)) as t:
        records = extract_records(deduped.kept, llm, settings)
        t.items_out = len(records)
    report.records_extracted = len(records)
    if not records:
        raise PipelineError("extraction produced no records — see the log for per-article failures")
    record_ids = {r.article_id for r in records}
    used = [a for a in deduped.kept if a.id in record_ids]
    skipped = len(deduped.kept) - len(used)
    if skipped:
        report.warnings.append(f"extract: {skipped} article(s) skipped after failed extraction")
    clock.say(f"Extracted {len(records)} typed records")

    # 4. Map ----------------------------------------------------------------
    with clock.stage("map", len(used)) as t:
        summaries = map_summaries(used, records, llm, settings)
        t.items_out = len(summaries)
    clock.say(f"Summarised {len(summaries)} articles")

    # 5. Retrieve prior-brief context (optional enrichment) -----------------
    snippets = []
    with clock.stage("retrieve") as t:
        try:
            index = build_index(settings)
            # Only briefs that ended before this cycle's latest article count as
            # "prior"; an earlier run over the same window must not feed back in.
            cycle_end = max(_article_time(a) for a in used)
            snippets = continuity_context(
                index,
                topic,
                records,
                k=settings.retrieval.top_k,
                min_score=settings.retrieval.min_score,
                exclude_brief_ids=index.briefs_dated_from(cycle_end),
            )
        except Exception as exc:  # continuity is enrichment; never block a brief on it
            logger.warning("retrieval failed, continuing without prior context: %s", exc, exc_info=True)
            report.warnings.append(f"retrieval: skipped prior-brief context ({type(exc).__name__}: {exc})")
        t.items_out = len(snippets)
    clock.say(f"Retrieved {len(snippets)} snippets from prior briefs")

    # 6. Reduce + 7. Edit ---------------------------------------------------
    with clock.stage("reduce", len(summaries)) as t:
        draft = reduce_brief(topic, summaries, records, llm, settings, prior_context=snippets)
        t.items_out = len(_draft_claims(draft))
    article_ids = [a.id for a in used]
    with clock.stage("edit", len(_draft_claims(draft))) as t:
        draft = edit_brief(draft, article_ids, llm, settings)
        t.items_out = len(_draft_claims(draft))
    clock.say(f"Drafted brief with {len(_draft_claims(draft))} claims")

    # 8. Validate citations (+ LLM fact-check) and 9. repair ----------------
    # Duplicates stay resolvable so corroborating texts count as support.
    by_id: dict[str, Article] = {a.id: a for a in ingested.articles}
    by_id.update({a.id: a for a in deduped.kept})
    with clock.stage("validate", len(_draft_claims(draft))) as t:
        draft, citation_report = validate_citations(
            draft,
            by_id,
            min_support=settings.pipeline.min_support,
            llm=llm if settings.pipeline.fact_check else None,
            settings=settings,
        )
        t.items_out = citation_report.supported
    with clock.stage("repair", len(_draft_claims(draft))) as t:
        draft = repair_brief(draft, citation_report)
        t.items_out = len(_draft_claims(draft))
    report.citation_passed = citation_report.passed
    errors = [i for i in citation_report.issues if i.severity == "error"]
    if errors:
        report.warnings.append(
            f"citations: {len(errors)} validation error(s); unsupported claims were removed and the brief "
            "is queued for review with the issues attached"
        )
    clock.say(
        f"Validated citations: {citation_report.supported} supported, {citation_report.weak} weak, "
        f"{citation_report.unsupported} unsupported"
    )

    # 10. Continuity notes --------------------------------------------------
    with clock.stage("continuity", len(snippets)) as t:
        notes = continuity_notes(draft, snippets) if snippets else []
        t.items_out = len(notes)

    # 11. Assemble + render drafts -----------------------------------------
    with clock.stage("assemble") as t:
        brief = assemble_brief(
            draft,
            topic=topic,
            articles={a.id: a for a in used},
            validation=citation_report,
            continuity=notes,
            run_id=run_id,
            prompt_version=prompt_version(),
            status="needs_review" if settings.pipeline.require_review else "draft",
        )
        drafts = write_deliverables(brief, settings.path(settings.paths.output_dir) / "drafts" / brief.id)
        t.items_out = len(brief.sources)
    report.brief_id = brief.id
    clock.say(f"Rendered draft deliverables to {drafts['html'].parent}")

    # 12. Review gate -------------------------------------------------------
    queue = ReviewQueue(settings)
    review_path = queue.submit(brief)
    brief = queue.load(brief.id)
    deliverables = drafts
    mode = review_mode
    notes_text = ""
    if mode == "queue" and not settings.pipeline.require_review:
        mode, notes_text = "auto_approve", "Published without review (pipeline.require_review is false)."
    with clock.stage("review") as t:
        decision = None
        if mode == "auto_approve":
            decision = ("approved", reviewer, notes_text or "Auto-approved by pipeline run.")
        elif mode == "interactive":
            answer = interactive_review(brief, reviewer=reviewer)
            if answer is not None:
                decision = (answer.decision, answer.reviewer, answer.notes)
        if decision is not None:
            brief = queue.decide(brief.id, decision[0], decision[1], decision[2])
            if brief.status == "approved":
                deliverables = published_paths(brief, settings)
        t.items_out = int(brief.status == "approved")
    clock.say(f"Brief {brief.id} is {brief.status.replace('_', ' ')}")

    report.brief_status = brief.status
    report.finished_at = datetime.now(timezone.utc)
    meter.apply(report)
    _save_report(report, settings)
    return PipelineRun(run_id=run_id, brief=brief, report=report, deliverables=deliverables, review_path=review_path)


def _draft_claims(draft: BriefDraft) -> list:
    return list(draft.executive_summary) + [c for s in draft.sections for c in s.claims] + list(draft.implications)


def published_paths(brief: Brief, settings: Settings) -> dict[str, Path]:
    """Paths written by :func:`publish_brief` for ``brief``."""
    client_dir = settings.path(settings.paths.output_dir) / "client" / brief.id
    paths = {ext: client_dir / f"{brief.id}.{ext}" for ext in ("md", "html", "json")}
    paths["archive"] = settings.path(settings.paths.briefs_dir) / f"{brief.id}.json"
    return paths
