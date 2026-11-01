"""Eval runner: golden recall + citation checks + LLM judge, versioned by prompt.

Every report is stamped with :func:`prompt_version` (a hash of all registered
prompt templates) and appended to ``evals_dir/history.jsonl``, so a prompt
change shows up as a new version whose scores can be compared with the last.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Sequence

from marketbrief.eval.golden import load_golden, recall_report
from marketbrief.eval.judge import judge_brief, judge_mean
from marketbrief.llm.base import LLMClient, LLMError
from marketbrief.schemas import (
    Article,
    ArticleRecord,
    Brief,
    BriefDraft,
    CitationReport,
    JudgeScore,
    RecallReport,
)

if TYPE_CHECKING:
    from marketbrief.config import Settings

log = logging.getLogger(__name__)


@dataclass
class EvalResult:
    recall: RecallReport
    judge: JudgeScore | None
    citation: CitationReport | None
    prompt_version: str
    passed: bool
    failures: list[str] = field(default_factory=list)
    report_path: Path | None = None


def current_prompt_version() -> str:
    """Hash of every registered prompt; ``"unknown"`` if prompts can't load."""
    try:
        from marketbrief.prompts.common import prompt_version
    except ImportError:  # pragma: no cover - partial install
        return "unknown"
    return prompt_version()


def _as_map(articles: Sequence[Article] | Mapping[str, Article]) -> dict[str, Article]:
    if isinstance(articles, Mapping):
        return dict(articles)
    return {a.id: a for a in articles}


def _cited_ids(brief: Brief | BriefDraft) -> set[str]:
    claims = [*brief.executive_summary, *brief.implications]
    for section in brief.sections:
        claims.extend(section.claims)
    return {cid for c in claims for cid in c.citations}


def evaluate_brief(
    brief: Brief | BriefDraft,
    articles: Sequence[Article] | Mapping[str, Article],
    records: Sequence[ArticleRecord],
    settings: "Settings",
    *,
    llm: LLMClient | None = None,
    golden_path: Path | str | None = None,
    use_judge: bool = True,
    judge_articles: Sequence[Article] | Mapping[str, Article] | None = None,
) -> EvalResult:
    """Score one brief. Thresholds come from ``settings.eval``.

    ``articles`` back the citation check (include syndicated duplicates);
    ``judge_articles`` (default: ``articles``) are what the judge reads.
    """
    from marketbrief.citations import validate_citations

    thresholds = settings.eval
    articles_by_id = _as_map(articles)
    golden = load_golden(golden_path or settings.path(settings.paths.golden_file))
    failures: list[str] = []

    recall = recall_report(golden, records, brief)
    if recall.extraction_recall < thresholds.min_extraction_recall:
        failures.append(
            f"extraction recall {recall.extraction_recall:.2f} < {thresholds.min_extraction_recall:.2f}"
        )
    if recall.brief_recall < thresholds.min_brief_recall:
        failures.append(f"brief recall {recall.brief_recall:.2f} < {thresholds.min_brief_recall:.2f}")
    if recall.must_catch_missed:
        failures.append(f"missed must-catch events: {', '.join(recall.must_catch_missed)}")

    _, citation = validate_citations(brief, articles_by_id, min_support=settings.pipeline.min_support)
    if not citation.passed:
        errors = sum(1 for i in citation.issues if i.severity == "error")
        failures.append(f"citation validation failed: {errors} error(s)")

    judge: JudgeScore | None = None
    if use_judge:
        if llm is None:
            from marketbrief.llm.factory import build_llm

            llm = build_llm(settings, model=settings.llm.judge_model)
        judged = _as_map(judge_articles) if judge_articles is not None else articles_by_id
        try:
            judge = judge_brief(brief, judged, llm, settings, golden=golden)
        except LLMError as exc:
            log.warning("judge failed: %s", exc)
            failures.append(f"judge failed: {type(exc).__name__}: {exc}")
        else:
            mean = judge_mean(judge)
            if mean < thresholds.min_judge_score:
                failures.append(f"judge mean {mean:.2f} < {thresholds.min_judge_score:.2f}")

    result = EvalResult(
        recall=recall,
        judge=judge,
        citation=citation,
        prompt_version=current_prompt_version(),
        passed=not failures,
        failures=failures,
    )
    log.info("eval %s (%s)", "passed" if result.passed else "FAILED", "; ".join(failures) or "all checks ok")
    return result


# --------------------------------------------------------------------------- #
# End-to-end runner
# --------------------------------------------------------------------------- #


def load_brief(path: Path | str) -> Brief:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"brief not found: {path}")
    return Brief.model_validate_json(path.read_text(encoding="utf-8"))


def eval_corpus(settings: "Settings", brief: Brief) -> tuple[list[Article], list[Article]]:
    """Articles for evaluating ``brief``: ``(all_in_period, deduped)``.

    Reads the article store for the brief's period (plus anything it cites),
    ingesting the configured sources when the store has nothing.
    """
    from marketbrief.dedup import dedupe
    from marketbrief.embeddings import get_embedder
    from marketbrief.ingest import ingest
    from marketbrief.ingest.store import ArticleStore

    store = ArticleStore(settings.path(settings.paths.store_dir) / "articles.jsonl")
    if len(store) == 0:
        log.info("article store empty; ingesting sources for the eval")
        ingest(settings, since=brief.period_start, until=brief.period_end, run_id="eval")
    in_period = {a.id: a for a in store.between(brief.period_start, brief.period_end)}
    for article_id in _cited_ids(brief) - set(in_period):
        cited = store.get(article_id)
        if cited is not None:
            in_period[article_id] = cited
    everything = list(in_period.values())
    kept = dedupe(
        everything,
        get_embedder(settings),
        threshold=settings.dedup.threshold,
        lead_chars=settings.dedup.lead_chars,
    ).kept
    return everything, kept


def save_eval_report(result: EvalResult, brief: Brief, settings: "Settings") -> Path:
    """Write the JSON report and append a summary line to ``history.jsonl``."""
    evals_dir = settings.path(settings.paths.evals_dir)
    evals_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    judge_avg = judge_mean(result.judge) if result.judge else None

    report = {
        "brief_id": brief.id,
        "topic": brief.topic,
        "prompt_version": result.prompt_version,
        "brief_prompt_version": brief.prompt_version,
        "evaluated_at": now,
        "passed": result.passed,
        "failures": result.failures,
        "recall": result.recall.model_dump(mode="json"),
        "judge": result.judge.model_dump(mode="json") if result.judge else None,
        "judge_mean": judge_avg,
        "citation": result.citation.model_dump(mode="json") if result.citation else None,
        "thresholds": settings.eval.model_dump(mode="json"),
    }
    path = evals_dir / f"eval_{result.prompt_version}_{brief.id}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    history_line = {
        "evaluated_at": now,
        "brief_id": brief.id,
        "prompt_version": result.prompt_version,
        "extraction_recall": result.recall.extraction_recall,
        "brief_recall": result.recall.brief_recall,
        "must_catch_missed": result.recall.must_catch_missed,
        "judge_mean": judge_avg,
        "citation_passed": result.citation.passed if result.citation else None,
        "passed": result.passed,
    }
    with (evals_dir / "history.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(history_line, ensure_ascii=False) + "\n")
    return path


def load_history(settings: "Settings") -> list[dict]:
    """All past eval summary lines, oldest first (malformed lines skipped)."""
    path = settings.path(settings.paths.evals_dir) / "history.jsonl"
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            log.warning("skipping malformed eval history line: %.80s", line)
    return out


def run_eval(
    settings: "Settings",
    *,
    brief_path: Path | None = None,
    llm: LLMClient | None = None,
    use_judge: bool = True,
) -> EvalResult:
    """Evaluate a saved brief, or run the pipeline (queue mode) and evaluate its draft.

    Records are re-extracted for the brief's articles; with the prompt cache
    enabled these are cache hits, so the eval adds no extraction cost.
    """
    from marketbrief.extract import extract_records

    if brief_path is not None:
        brief = load_brief(brief_path)
    else:
        from marketbrief.pipeline import run_pipeline

        brief = run_pipeline(settings, review_mode="queue", llm=llm).brief

    extract_llm = llm
    if extract_llm is None:
        from marketbrief.llm.factory import build_llm

        extract_llm = build_llm(settings)
    all_articles, kept = eval_corpus(settings, brief)
    if not kept:
        log.warning("no source articles found for brief %s", brief.id)
    records = extract_records(kept, extract_llm, settings)

    result = evaluate_brief(
        brief,
        all_articles,
        records,
        settings,
        # None -> evaluate_brief builds a client on settings.llm.judge_model.
        llm=llm,
        use_judge=use_judge,
        judge_articles=kept,
    )
    result.report_path = save_eval_report(result, brief, settings)
    return result
