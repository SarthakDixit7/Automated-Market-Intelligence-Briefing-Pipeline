"""Pipeline orchestration.

Two layers of tests:

* stubbed: every stage function is replaced so orchestration logic (ordering,
  timing, brief assembly, review routing, failure handling, usage metering)
  is tested in isolation and fast;
* end-to-end: the real stages on the synthetic fixture feed with the offline
  provider.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import marketbrief.citations as citations_mod
import marketbrief.dedup as dedup_mod
import marketbrief.embeddings as embeddings_mod
import marketbrief.extract as extract_mod
import marketbrief.ingest as ingest_mod
import marketbrief.pipeline as pipeline
import marketbrief.prompts.common as prompts_common
import marketbrief.retrieval as retrieval_mod
import marketbrief.summarize as summarize_mod
from marketbrief.llm.base import LLMRequest, LLMResult, Usage
from marketbrief.pipeline import PipelineError, publish_brief, run_pipeline
from marketbrief.schemas import (
    Article,
    ArticleSummary,
    BriefDraft,
    BriefSection,
    CitationIssue,
    CitationReport,
    Claim,
    ContinuityNote,
    ReviewDecision,
    RetrievedSnippet,
    RunReport,
)

EXPECTED_STAGES = [
    "ingest", "dedup", "extract", "map", "retrieve", "reduce", "edit",
    "validate", "repair", "continuity", "assemble", "review",
]


# --------------------------------------------------------------------------- #
# Stubbed stages
# --------------------------------------------------------------------------- #


class CountingLLM:
    """Minimal LLM client: every call costs 10 input / 5 output tokens."""

    provider = "stub"
    model = "stub-model"

    def complete(self, request: LLMRequest) -> LLMResult:
        return LLMResult(text="ok", usage=Usage(input_tokens=10, output_tokens=5), provider="stub", model="stub-model")


def _article(article_id: str, day: int, source: str, title: str, **extra) -> Article:
    return Article(
        id=article_id,
        source_id="stub",
        source_name=source,
        url=f"https://example.com/{article_id}",
        title=title,
        published_at=datetime(2026, 10, day, tzinfo=timezone.utc),
        text=f"{title}. Body text for {article_id}.",
        content_hash=article_id,
        **extra,
    )


@pytest.fixture
def stubs(monkeypatch):
    """Replace every stage; returns a namespace to inspect calls and tweak behaviour."""
    a1 = _article("a_000000000001", 6, "The Silicon Wire", "Aurora Silicon raises $410 million")
    a2 = _article("a_000000000002", 9, "Datacenter Dispatch", "Northwind Compute cuts prices 22%")
    a3 = _article("a_000000000003", 7, "Cloud Ledger", "Aurora Silicon closes $410M Series D")
    canonical = a1.model_copy(update={"duplicate_ids": [a3.id], "corroborating_sources": ["Cloud Ledger"]})
    ns = SimpleNamespace(
        articles=[a1, a2, a3],
        calls=[],
        validate_kwargs={},
        report=CitationReport(passed=True, claims_checked=3, supported=3, weak=0, unsupported=0),
        retrieval_error=None,
    )

    def fake_ingest(settings, *, source_ids=None, since=None, until=None, run_id="manual"):
        ns.calls.append("ingest")
        return SimpleNamespace(articles=list(ns.articles), per_source={"stub": len(ns.articles)},
                               errors=[], raw_path=None)

    def fake_dedupe(articles, embedder, *, threshold, lead_chars):
        ns.calls.append("dedup")
        kept = [canonical if a.id == a1.id else a for a in articles if a.id != a3.id]
        return SimpleNamespace(kept=kept, clusters=[object()] if len(kept) < len(articles) else [],
                               removed=len(articles) - len(kept))

    def fake_extract(articles, llm, settings):
        ns.calls.append("extract")
        llm.complete(LLMRequest(task="extract", system="s", prompt="p"))
        return [SimpleNamespace(article_id=a.id) for a in articles]

    def fake_map(articles, records, llm, settings):
        ns.calls.append("map")
        return [ArticleSummary(article_id=a.id, summary=a.title, key_points=[], why_it_matters="",
                               companies=[], event_types=[]) for a in articles]

    def fake_build_index(settings):
        ns.calls.append("build_index")
        if ns.retrieval_error:
            raise ns.retrieval_error
        return SimpleNamespace(
            add_brief=lambda brief: ns.calls.append(f"index:{brief.id}") or 1,
            briefs_dated_from=lambda cutoff: ns.calls.append("index:dated_from") or ["b_same_period"],
        )

    def fake_context(index, topic, records, *, k, min_score, exclude_brief_ids=()):
        ns.excluded = list(exclude_brief_ids)
        return [RetrievedSnippet(brief_id="b_20260930_prior1", brief_title="Prior", claim_id="c1",
                                 text="Northwind pricing pressure expected.", score=0.7)]

    def fake_reduce(topic, summaries, records, llm, settings, prior_context):
        ns.calls.append("reduce")
        ns.prior_context = prior_context
        return BriefDraft(
            title=f"{topic}: weekly brief",
            executive_summary=[Claim(id="c1", text="Northwind cut prices 22%.", citations=[a2.id])],
            sections=[
                BriefSection(title="Funding & M&A", claims=[
                    Claim(id="c2", text="Aurora Silicon raised $410 million.", citations=[a1.id])]),
                BriefSection(title="Empty", claims=[]),
            ],
            implications=[Claim(id="c3", text="Price war.", citations=[a2.id, a1.id], kind="analysis")],
            watchlist=["Northwind Compute"],
            continuity=["Continues: pricing pressure."],
        )

    def fake_edit(draft, article_ids, llm, settings):
        ns.calls.append("edit")
        ns.edit_ids = list(article_ids)
        return draft

    def fake_validate(draft, articles, *, min_support, llm=None, settings=None):
        ns.calls.append("validate")
        ns.validate_kwargs = {"articles": articles, "min_support": min_support, "llm": llm}
        return draft, ns.report

    def fake_repair(draft, report):
        ns.calls.append("repair")
        return draft

    def fake_notes(draft, snippets):
        return [ContinuityNote(prior_brief_id="b_20260930_prior1", prior_claim=snippets[0].text,
                               relation="continues", note="Continues: pricing pressure.")]

    monkeypatch.setattr(ingest_mod, "ingest", fake_ingest)
    monkeypatch.setattr(dedup_mod, "dedupe", fake_dedupe)
    monkeypatch.setattr(embeddings_mod, "get_embedder", lambda settings: None)
    monkeypatch.setattr(extract_mod, "extract_records", fake_extract)
    monkeypatch.setattr(summarize_mod, "map_summaries", fake_map)
    monkeypatch.setattr(summarize_mod, "reduce_brief", fake_reduce)
    monkeypatch.setattr(summarize_mod, "edit_brief", fake_edit)
    monkeypatch.setattr(retrieval_mod, "build_index", fake_build_index)
    monkeypatch.setattr(retrieval_mod, "continuity_context", fake_context)
    monkeypatch.setattr(retrieval_mod, "continuity_notes", fake_notes)
    monkeypatch.setattr(citations_mod, "validate_citations", fake_validate)
    monkeypatch.setattr(citations_mod, "repair_brief", fake_repair)
    monkeypatch.setattr(prompts_common, "prompt_version", lambda: "testpv000000")
    return ns


def test_stages_run_in_order_and_brief_is_assembled(settings, stubs):
    messages: list[str] = []
    run = run_pipeline(settings, topic="AI infra", llm=CountingLLM(), progress=messages.append)

    assert [s.stage for s in run.report.stages] == EXPECTED_STAGES
    assert all(s.seconds >= 0 for s in run.report.stages)
    assert re.fullmatch(r"r_\d{14}_[0-9a-f]{4}", run.run_id)
    brief = run.brief
    # Brief id carries the period end date (the latest article).
    assert re.fullmatch(r"b_20261009_[0-9a-f]{6}", brief.id)
    assert brief.topic == "AI infra"
    assert brief.status == "needs_review"
    assert brief.prompt_version == "testpv000000"
    assert brief.run_id == run.run_id
    assert brief.period_start.day == 6 and brief.period_end.day == 9
    # Sources: every cited article once, in order of first citation, with corroboration.
    assert [s.article_id for s in brief.sources] == ["a_000000000002", "a_000000000001"]
    assert brief.sources[1].corroborating_sources == ["Cloud Ledger"]
    assert [s.title for s in brief.sections] == ["Funding & M&A"], "empty sections are dropped"
    assert brief.continuity[0].relation == "continues"
    assert brief.continuity_text == ["Continues: pricing pressure."]
    assert brief.validation == stubs.report

    # Duplicates stay resolvable for citation support; edit sees only kept ids.
    assert "a_000000000003" in stubs.validate_kwargs["articles"]
    assert stubs.validate_kwargs["articles"]["a_000000000001"].duplicate_ids == ["a_000000000003"]
    assert stubs.edit_ids == ["a_000000000001", "a_000000000002"]
    assert stubs.validate_kwargs["llm"] is not None, "fact_check is on in settings"
    assert len(stubs.prior_context) == 1
    # Briefs dated at/after this cycle's end (e.g. a re-run of the same window) are not "prior".
    assert stubs.excluded == ["b_same_period"]
    assert not any(w.startswith("retrieval:") for w in run.report.warnings)

    # Review gate + deliverables
    assert run.review_path is not None and run.review_path.exists()
    assert set(run.deliverables) == {"md", "html", "json"}
    assert all(p.exists() and "drafts" in p.parts for p in run.deliverables.values())

    # Run report: counts, usage delta, saved to disk.
    r = run.report
    assert (r.articles_fetched, r.articles_after_dedup, r.duplicate_clusters, r.records_extracted) == (3, 2, 1, 2)
    assert (r.llm_calls, r.input_tokens, r.output_tokens) == (1, 10, 5)
    assert (r.provider, r.model) == ("stub", "stub-model")
    assert r.brief_id == brief.id and r.brief_status == "needs_review" and r.citation_passed is True
    saved = RunReport.model_validate_json(
        (settings.path(settings.paths.runs_dir) / f"{run.run_id}.json").read_text(encoding="utf-8")
    )
    assert saved == r
    assert any("Ingested 3 articles" in m for m in messages)
    assert any("needs review" in m for m in messages)


def test_fact_check_can_be_disabled(settings, stubs):
    settings.pipeline.fact_check = False
    run_pipeline(settings, llm=CountingLLM())
    assert stubs.validate_kwargs["llm"] is None


def test_retrieval_failure_degrades_gracefully(settings, stubs):
    stubs.retrieval_error = RuntimeError("index corrupt")
    run = run_pipeline(settings, llm=CountingLLM())
    assert run.brief.status == "needs_review"
    assert run.brief.continuity == []
    assert any("retrieval" in w and "index corrupt" in w for w in run.report.warnings)


def test_citation_errors_are_queued_with_warning(settings, stubs):
    stubs.report = CitationReport(
        passed=False, claims_checked=3, supported=2, weak=0, unsupported=1,
        issues=[CitationIssue(claim_id="c2", severity="error", kind="number_mismatch", detail="410 not found")],
    )
    run = run_pipeline(settings, llm=CountingLLM())
    assert run.brief.status == "needs_review"
    assert run.report.citation_passed is False
    assert any("citations: 1 validation error" in w for w in run.report.warnings)


def test_no_articles_raises_and_still_saves_report(settings, stubs):
    stubs.articles = []
    with pytest.raises(PipelineError, match="no articles ingested"):
        run_pipeline(settings, llm=CountingLLM())
    reports = list(settings.path(settings.paths.runs_dir).glob("r_*.json"))
    assert len(reports) == 1
    report = RunReport.model_validate_json(reports[0].read_text(encoding="utf-8"))
    assert report.finished_at is not None
    assert any("run failed: PipelineError" in w for w in report.warnings)


def test_auto_approve_publishes(settings, stubs):
    run = run_pipeline(settings, llm=CountingLLM(), review_mode="auto_approve", reviewer="ci-bot")
    brief = run.brief
    assert brief.status == "approved"
    assert brief.review.reviewer == "ci-bot"
    assert set(run.deliverables) == {"md", "html", "json", "archive"}
    assert all(p.exists() for p in run.deliverables.values())
    assert run.deliverables["html"].parent == settings.path(settings.paths.output_dir) / "client" / brief.id
    assert "Not for client distribution" not in run.deliverables["html"].read_text(encoding="utf-8")
    assert f"index:{brief.id}" in stubs.calls
    assert run.report.brief_status == "approved"


def test_interactive_mode_records_terminal_decision(settings, stubs, monkeypatch):
    seen = []

    def fake_review(brief, *, reviewer):
        seen.append(brief.status)
        return ReviewDecision(decision="changes_requested", reviewer=reviewer, notes="cite more")

    monkeypatch.setattr(pipeline, "interactive_review", fake_review)
    run = run_pipeline(settings, llm=CountingLLM(), review_mode="interactive", reviewer="Jane")
    assert seen == ["needs_review"]
    assert run.brief.status == "changes_requested"
    assert run.brief.review.notes == "cite more"
    assert all("drafts" in p.parts for p in run.deliverables.values())


def test_review_not_required_publishes_directly(settings, stubs):
    settings.pipeline.require_review = False
    run = run_pipeline(settings, llm=CountingLLM())
    assert run.brief.status == "approved"
    assert "require_review" in run.brief.review.notes


def test_unknown_review_mode_rejected(settings):
    with pytest.raises(ValueError):
        run_pipeline(settings, llm=CountingLLM(), review_mode="yolo")  # type: ignore[arg-type]


def test_publish_brief_requires_approval(settings, sample_brief):
    with pytest.raises(ValueError, match="only approved"):
        publish_brief(sample_brief, settings)


# --------------------------------------------------------------------------- #
# End-to-end (real stages, offline provider, synthetic fixture feed)
# --------------------------------------------------------------------------- #


def test_end_to_end_offline_queue(settings, fixture_articles):
    run = run_pipeline(settings)
    brief, report = run.brief, run.report

    assert [s.stage for s in report.stages] == EXPECTED_STAGES
    assert report.provider == "offline"
    assert report.articles_fetched == len(fixture_articles) >= 20
    assert report.articles_after_dedup < report.articles_fetched, "syndicated duplicates are folded"
    assert report.duplicate_clusters >= 3
    assert report.llm_calls > 0

    assert brief.status == "needs_review"
    claims = brief.all_claims()
    assert len(claims) >= 8
    source_ids = {s.article_id for s in brief.sources}
    cited = {cid for c in claims for cid in c.citations}
    assert cited <= source_ids and source_ids <= cited
    assert any(s.corroborating_sources for s in brief.sources)
    assert brief.validation is not None and brief.validation.claims_checked > 0
    assert brief.prompt_version

    # Prior briefs (fixtures) seed continuity.
    assert brief.continuity or brief.continuity_text

    html = run.deliverables["html"].read_text(encoding="utf-8")
    assert "Not for client distribution" in html and brief.title in html
    queued = json.loads(run.review_path.read_text(encoding="utf-8"))
    assert queued["status"] == "needs_review"


def test_end_to_end_publish_index_and_cache(settings, fixture_articles):
    from marketbrief.llm.factory import build_llm
    from marketbrief.retrieval import build_index

    first = run_pipeline(settings, review_mode="auto_approve", reviewer="demo")
    assert first.brief.status == "approved"
    assert first.deliverables["archive"].exists()
    assert first.brief.id in build_index(settings).brief_ids()

    # Same inputs again: the prompt cache serves the repeated per-article calls.
    second = run_pipeline(settings, llm=build_llm(settings))
    assert second.report.prompt_cache_hits > 0
    assert second.brief.id != first.brief.id
