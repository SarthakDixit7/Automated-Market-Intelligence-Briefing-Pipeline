"""Shared pytest fixtures.

``settings`` points every writable path into ``tmp_path`` while reading the
real project inputs (sources, fixture articles, entities, golden set, prior
briefs), and forces the deterministic offline provider. Fixture names
``settings``, ``offline_llm`` and ``fixture_articles`` are relied on by every
test module; keep them stable.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from marketbrief.config import Settings, load_settings
from marketbrief.schemas import (
    Article,
    ArticleRecord,
    Brief,
    BriefSection,
    CitationIssue,
    CitationReport,
    Claim,
    CompanyMention,
    ContinuityNote,
    EventType,
    ExtractedEvent,
    Sentiment,
    SourceRef,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Offline settings with all outputs isolated under ``tmp_path``."""
    # Environment overrides from the developer's shell must not leak into tests.
    for var in ("MB_PROVIDER", "MB_MODEL", "MB_JUDGE_MODEL", "MB_CACHE_ENABLED", "MB_EMBEDDINGS", "MB_MAX_WORKERS"):
        monkeypatch.delenv(var, raising=False)

    def real(rel: str) -> str:
        return str(PROJECT_ROOT / rel)

    def tmp(rel: str) -> str:
        return str(tmp_path / rel)

    overrides = {
        "llm": {"provider": "offline"},
        "cache": {"path": tmp("data/cache/llm_cache.sqlite")},
        "retrieval": {
            "index_dir": tmp("data/index"),
            "extra_brief_dirs": [real("data/fixtures/prior_briefs")],
        },
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
    }
    return load_settings(root=PROJECT_ROOT, overrides=overrides)


@pytest.fixture
def offline_llm(settings: Settings):
    """Deterministic offline LLM client using the fixture entity gazetteer."""
    from marketbrief.llm.offline import OfflineLLMClient

    return OfflineLLMClient(entities_path=settings.path(settings.paths.entities_file))


@pytest.fixture
def fixture_articles(settings: Settings) -> list[Article]:
    """The synthetic demo feed, ingested through the real ingest stage."""
    from marketbrief.ingest import ingest

    return ingest(settings, run_id="test").articles


@pytest.fixture
def sample_record() -> ArticleRecord:
    """A hand-written extraction record (independent of the offline provider)."""
    quote = "Aurora Silicon said on Tuesday it has raised $410 million in a Series D round led by Halcyon Partners."
    return ArticleRecord(
        article_id="a_sample000001",
        companies=[CompanyMention(name="Aurora Silicon", role="subject")],
        events=[
            ExtractedEvent(
                event_type=EventType.FUNDING,
                headline="Aurora Silicon raises $410M Series D",
                companies=["Aurora Silicon"],
                amount="$410M",
                event_date="2026-10-06",
                evidence_quote=quote,
            )
        ],
        categories=["ai chips"],
        sentiment=Sentiment.POSITIVE,
        confidence=0.9,
        key_facts=[quote],
    )


def _oct(day: int) -> datetime:
    return datetime(2026, 10, day, 9, 0, tzinfo=timezone.utc)


@pytest.fixture
def sample_brief() -> Brief:
    """A small, fully populated brief: weak claim, continuity, corroboration."""
    sources = [
        SourceRef(
            article_id="a_sample000001",
            title="Aurora Silicon raises $410 million Series D",
            url="https://example.com/aurora-series-d",
            source_name="The Silicon Wire",
            published_at=_oct(6),
            corroborating_sources=["Cloud Ledger"],
        ),
        SourceRef(
            article_id="a_sample000002",
            title="Meridian Cloud to acquire Vireo Analytics for $1.2 billion",
            url="https://example.com/meridian-vireo",
            source_name="Market Pulse Daily",
            published_at=_oct(8),
        ),
        SourceRef(
            article_id="a_sample000003",
            title="Northwind Compute cuts GPU-hour prices <22%> & more",
            url=None,
            source_name="Datacenter Dispatch",
            published_at=_oct(9),
        ),
    ]
    exec_claims = [
        Claim(id="c1", text="Meridian Cloud agreed to acquire Vireo Analytics for $1.2 billion.",
              citations=["a_sample000002"], verification="supported", support_score=0.8),
        Claim(id="c2", text="Aurora Silicon raised $410 million in a Series D round.",
              citations=["a_sample000001"], verification="supported", support_score=0.9),
    ]
    sections = [
        BriefSection(title="Funding & M&A", claims=[
            Claim(id="c3", text="Aurora Silicon's Series D was led by Halcyon Partners.",
                  citations=["a_sample000001"], verification="supported", support_score=0.7),
        ]),
        BriefSection(title="Pricing & competition", claims=[
            Claim(id="c4", text="Northwind Compute cut H-class GPU-hour pricing by 22% <effective immediately>.",
                  citations=["a_sample000003"], verification="weak", support_score=0.3),
        ]),
    ]
    implications = [
        Claim(id="c5", text="Consolidation and price cuts point to margin pressure across GPU clouds.",
              citations=["a_sample000002", "a_sample000003"], kind="analysis", verification="supported"),
    ]
    validation = CitationReport(
        passed=True, claims_checked=5, supported=4, weak=1, unsupported=0, fact_checked=True,
        issues=[CitationIssue(claim_id="c4", severity="warning", kind="low_support", detail="support 0.30 < 0.35")],
    )
    return Brief(
        id="b_20261030_abc123",
        topic="Enterprise AI infrastructure",
        title="Enterprise AI infrastructure: consolidation and a GPU price war",
        created_at=datetime(2026, 10, 31, 12, 0, tzinfo=timezone.utc),
        period_start=_oct(6),
        period_end=_oct(9),
        executive_summary=exec_claims,
        sections=sections,
        implications=implications,
        watchlist=["Meridian Cloud", "Northwind Compute"],
        continuity=[
            ContinuityNote(prior_brief_id="b_20260930_prior1",
                           prior_claim="Meridian Cloud was reported to be in talks to buy Vireo Analytics.",
                           relation="updates", note="Updates: the reported Meridian-Vireo talks became a $1.2B deal."),
        ],
        continuity_text=["Updates: the reported Meridian-Vireo talks became a $1.2B deal."],
        sources=sources,
        status="needs_review",
        validation=validation,
        prompt_version="0123456789ab",
        run_id="r_20261031120000_beef",
    )
