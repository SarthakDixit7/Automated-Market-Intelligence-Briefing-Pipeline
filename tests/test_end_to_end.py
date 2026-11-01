"""End-to-end quality checks: the full offline pipeline on the synthetic fixtures.

The pipeline runs once per module in an isolated tmp tree (fresh prompt cache,
index, review queue and outputs), then the brief is approved through the real
review queue and the run is repeated. The tests assert the properties the
project claims, not just that the code runs:

* syndicated rewrites are folded, distinct stories about one company are not;
* golden-set extraction / brief recall clear the configured thresholds;
* every claim passes citation validation;
* continuity notes link back to the prior (September) briefs;
* deliverables are footnoted and only reach the client folder after approval;
* an identical second run is served entirely from the prompt cache;
* the analyst agent answers with citations and a chart, and pauses sensitive
  answers for human review.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pytest

from marketbrief.config import Settings, load_settings
from marketbrief.dedup import DedupResult, dedupe
from marketbrief.embeddings import get_embedder
from marketbrief.eval.runner import EvalResult, run_eval
from marketbrief.ingest import ingest
from marketbrief.llm import build_llm
from marketbrief.pipeline import PipelineRun, run_pipeline
from marketbrief.retrieval import build_index
from marketbrief.review import ReviewQueue
from marketbrief.schemas import Article

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _e2e_settings(root: Path) -> Settings:
    """Offline settings reading the real fixtures and writing only under ``root``."""

    def real(rel: str) -> str:
        return str(PROJECT_ROOT / rel)

    def tmp(rel: str) -> str:
        return str(root / rel)

    return load_settings(
        root=PROJECT_ROOT,
        overrides={
            "llm": {"provider": "offline"},
            "cache": {"enabled": True, "path": tmp("data/cache/llm_cache.sqlite")},
            "retrieval": {"index_dir": tmp("data/index"), "extra_brief_dirs": [real("data/fixtures/prior_briefs")]},
            "paths": {
                "sources_file": real("config/sources.yaml"),
                "entities_file": real("data/fixtures/entities.yaml"),
                "golden_file": real("data/golden/golden_set.yaml"),
                "raw_dir": tmp("data/raw"),
                "store_dir": tmp("data/store"),
                "runs_dir": tmp("data/runs"),
                "review_dir": tmp("data/review"),
                "briefs_dir": tmp("data/briefs"),
                "output_dir": tmp("output"),
                "evals_dir": tmp("data/evals"),
                "agent_dir": tmp("data/agent"),
            },
        },
    )


@dataclass
class E2E:
    settings: Settings
    articles: list[Article]
    dedup: DedupResult
    first: PipelineRun
    published_before_approval: dict[str, bool]
    approved_paths: dict[str, Path]
    second: PipelineRun
    evaluation: EvalResult


@pytest.fixture(scope="module")
def e2e(tmp_path_factory: pytest.TempPathFactory) -> E2E:
    settings = _e2e_settings(tmp_path_factory.mktemp("e2e"))
    articles = ingest(settings, run_id="e2e_probe").articles
    dedup = dedupe(articles, get_embedder(settings), threshold=settings.dedup.threshold,
                   lead_chars=settings.dedup.lead_chars)

    # Run 1: cold cache, queued for review.
    first = run_pipeline(settings, llm=build_llm(settings), review_mode="queue")
    brief_id = first.brief.id
    published_before_approval = {
        "client_dir": (settings.path(settings.paths.output_dir) / "client" / brief_id).exists(),
        "archive": (settings.path(settings.paths.briefs_dir) / f"{brief_id}.json").exists(),
        "indexed": brief_id in build_index(settings).brief_ids(),
    }

    # Human approval publishes it (deliverables + archive + retrieval index).
    ReviewQueue(settings).decide(brief_id, "approved", reviewer="e2e", notes="ok")
    approved_paths = {
        "html": settings.path(settings.paths.output_dir) / "client" / brief_id / f"{brief_id}.html",
        "md": settings.path(settings.paths.output_dir) / "client" / brief_id / f"{brief_id}.md",
        "archive": settings.path(settings.paths.briefs_dir) / f"{brief_id}.json",
    }

    # Run 2: identical inputs with a fresh client; the prompt cache should answer everything.
    second = run_pipeline(settings, llm=build_llm(settings), review_mode="queue")

    evaluation = run_eval(settings, brief_path=approved_paths["archive"], use_judge=True)
    return E2E(settings, articles, dedup, first, published_before_approval, approved_paths, second, evaluation)


def _cluster_of(result: DedupResult, title_fragment: str, articles: list[Article]) -> frozenset[str]:
    """Ids folded together with the (single) article whose title contains ``title_fragment``."""
    matches = [a for a in articles if title_fragment.lower() in a.title.lower()]
    assert len(matches) == 1, f"expected one article titled like {title_fragment!r}, got {[a.title for a in matches]}"
    article_id = matches[0].id
    for cluster in result.clusters:
        if article_id in cluster.member_ids:
            return frozenset(cluster.member_ids)
    return frozenset({article_id})


# --------------------------------------------------------------------------- dedup


@pytest.mark.parametrize(
    ("story", "canonical_title", "rewrite_title"),
    [
        ("G1 Aurora Series D", "Aurora Silicon raises $410 million Series D", "AI chip designer Aurora Silicon raises"),
        ("G2 Meridian-Vireo deal", "Meridian Cloud to acquire Vireo", "Meridian Cloud agrees to acquire Vireo"),
        ("G3 Northwind price cut", "Northwind Compute cuts H-class", "Northwind Compute slashes H-class"),
    ],
)
def test_syndicated_rewrites_are_folded(e2e: E2E, story: str, canonical_title: str, rewrite_title: str) -> None:
    a = _cluster_of(e2e.dedup, canonical_title, e2e.articles)
    b = _cluster_of(e2e.dedup, rewrite_title, e2e.articles)
    assert len(a) >= 2 and a == b, f"{story}: syndicated copies were not clustered"


def test_distinct_stories_about_one_company_stay_separate(e2e: E2E) -> None:
    meridian = [
        _cluster_of(e2e.dedup, "Meridian Cloud to acquire Vireo", e2e.articles),
        _cluster_of(e2e.dedup, "egress", e2e.articles),
        _cluster_of(e2e.dedup, "Frankfurt", e2e.articles),
    ]
    assert len(set(meridian)) == 3, "acquisition, EU inquiry and sovereign region must stay distinct"
    northwind = [
        _cluster_of(e2e.dedup, "Northwind Compute cuts H-class", e2e.articles),
        _cluster_of(e2e.dedup, "Northwind Compute Q3", e2e.articles),
    ]
    assert northwind[0] != northwind[1]


def test_kept_canonicals_record_corroborating_sources(e2e: E2E) -> None:
    folded = [a for a in e2e.dedup.kept if a.duplicate_ids]
    assert len(folded) == len(e2e.dedup.clusters) >= 3
    assert sum(bool(a.corroborating_sources) for a in folded) >= 3, "other outlets are kept as corroboration"
    assert e2e.first.report.articles_after_dedup == len(e2e.dedup.kept) < e2e.first.report.articles_fetched


# --------------------------------------------------------------------------- quality


def test_golden_recall_meets_thresholds(e2e: E2E) -> None:
    recall = e2e.evaluation.recall
    assert recall.total == 12
    assert recall.extraction_recall >= max(0.9, e2e.settings.eval.min_extraction_recall)
    assert recall.brief_recall >= max(0.75, e2e.settings.eval.min_brief_recall)
    assert recall.must_catch_missed == []
    assert e2e.evaluation.passed, e2e.evaluation.failures


def test_judge_scores_clear_the_bar(e2e: E2E) -> None:
    judge = e2e.evaluation.judge
    assert judge is not None
    assert min(judge.factuality, judge.coverage, judge.consistency) >= e2e.settings.eval.min_judge_score
    assert judge.rationale.strip()


def test_citation_validation_passes_with_no_errors(e2e: E2E) -> None:
    brief = e2e.first.brief
    report = brief.validation
    assert report is not None and report.passed and report.fact_checked
    assert [i for i in report.issues if i.severity == "error"] == []
    assert report.unsupported == 0 and report.supported == report.claims_checked
    assert e2e.first.report.citation_passed is True

    source_ids = {s.article_id for s in brief.sources}
    for claim in brief.all_claims():
        assert claim.citations, f"claim {claim.id} has no citation"
        assert set(claim.citations) <= source_ids
        assert claim.verification == "supported"
    assert len({c.id for c in brief.all_claims()}) == len(brief.all_claims()), "claim ids are unique"


def test_continuity_links_to_prior_briefs(e2e: E2E) -> None:
    notes = e2e.first.brief.continuity
    assert notes, "prior briefs should seed continuity notes"
    assert all(n.prior_brief_id.startswith("b_202609") for n in notes)
    # A rumour from September is now a confirmed acquisition.
    assert any(
        n.relation == "updates" and "Vireo" in n.prior_claim and "talks" in n.prior_claim for n in notes
    ), [(n.relation, n.prior_claim) for n in notes]
    # The September GPU price war carries into this cycle's Northwind price cut.
    assert any("Northwind" in n.prior_claim for n in notes)


def test_rerun_does_not_treat_same_period_brief_as_prior(e2e: E2E) -> None:
    first_id = e2e.first.brief.id
    assert all(n.prior_brief_id != first_id for n in e2e.second.brief.continuity)


# --------------------------------------------------------------------------- review + deliverables


def test_published_only_after_approval(e2e: E2E) -> None:
    assert e2e.first.brief.status == "needs_review"
    assert e2e.published_before_approval == {"client_dir": False, "archive": False, "indexed": False}
    assert set(e2e.first.deliverables) == {"md", "html", "json"}
    assert all(p.exists() and "drafts" in p.parts for p in e2e.first.deliverables.values())

    assert all(p.exists() for p in e2e.approved_paths.values())
    queue = ReviewQueue(e2e.settings)
    approved = queue.load(e2e.first.brief.id)
    assert approved.status == "approved" and approved.review is not None and approved.review.reviewer == "e2e"
    assert e2e.first.brief.id in build_index(e2e.settings).brief_ids()


def test_html_deliverable_has_footnoted_sources(e2e: E2E) -> None:
    draft_html = e2e.first.deliverables["html"].read_text(encoding="utf-8")
    client_html = e2e.approved_paths["html"].read_text(encoding="utf-8")
    assert "Not for client distribution" in draft_html
    assert "Not for client distribution" not in client_html

    brief = e2e.first.brief
    links = set(re.findall(r'href="#(src-\d+)"', client_html))
    anchors = set(re.findall(r'id="(src-\d+)"', client_html))
    assert links and links <= anchors
    assert len(anchors) == len(brief.sources)
    assert "<script" not in client_html.lower(), "client HTML is static and self-contained"
    assert "Also reported by" in client_html, "corroborating outlets are listed with their source"

    markdown = e2e.approved_paths["md"].read_text(encoding="utf-8")
    assert re.search(r"\[\^1\]:", markdown), "markdown carries footnote definitions"


# --------------------------------------------------------------------------- cache


def test_identical_rerun_is_served_from_prompt_cache(e2e: E2E) -> None:
    cold, warm = e2e.first.report, e2e.second.report
    assert cold.llm_calls > 0 and cold.prompt_cache_hits == 0
    assert warm.llm_calls == cold.llm_calls
    assert warm.prompt_cache_hits == warm.llm_calls, f"{warm.prompt_cache_misses} prompt-cache misses on rerun"
    assert warm.input_tokens == 0 and warm.output_tokens == 0, "cache hits are never re-billed"
    # Same inputs, same cached outputs -> same brief content.
    assert [c.text for c in e2e.second.brief.all_claims()] == [c.text for c in e2e.first.brief.all_claims()]


# --------------------------------------------------------------------------- agent


def test_agent_answers_with_citations_and_chart(e2e: E2E) -> None:
    from marketbrief.agent import run_analyst

    result = run_analyst(
        "How is GPU cloud pricing changing and who is affected? Include a chart.", e2e.settings, max_steps=8
    )
    state = result.state
    assert result.status == "completed"
    assert state.answer and state.citations
    assert all(f"[{cid}]" in state.answer for cid in state.citations)
    known = {a.id for a in e2e.articles}
    assert set(state.citations) <= known
    assert "Northwind Compute" in state.answer
    assert state.chart_path and Path(state.chart_path).suffix == ".png" and Path(state.chart_path).stat().st_size > 0


def test_agent_pauses_sensitive_answers_for_review(e2e: E2E) -> None:
    from marketbrief.agent import resume_analyst, run_analyst

    paused = run_analyst("What is the Kestrel Networks lawsuit about?", e2e.settings, max_steps=8)
    assert paused.status == "interrupted"
    # Halted after drafting, before the human_review node runs.
    assert paused.trace[-1] == "draft_answer" and "human_review" not in paused.trace
    assert paused.state.sensitive and paused.state.approved is None
    assert "Kestrel Networks" in paused.state.answer and "patent" in paused.state.answer

    resumed = resume_analyst(paused.thread_id, e2e.settings, approved=True, notes="checked")
    assert resumed.status == "completed"
    assert resumed.state.approved is True and resumed.state.reviewer_notes == "checked"
    assert resumed.trace[-1] == "human_review"
