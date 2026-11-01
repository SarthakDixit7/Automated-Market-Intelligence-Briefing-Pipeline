"""RAG over prior briefs: persistent index, retrieval for a cycle, continuity labelling."""

from __future__ import annotations

import re
import zlib
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from marketbrief.retrieval import BriefIndex, build_index, continuity_context, continuity_notes
from marketbrief.schemas import (
    ArticleRecord,
    Brief,
    BriefDraft,
    BriefSection,
    Claim,
    CompanyMention,
    EventType,
    ExtractedEvent,
    RetrievedSnippet,
    Sentiment,
    SourceRef,
)


class BagOfWordsEmbedder:
    """Tiny deterministic embedder for tests (crc32 feature hashing over words)."""

    name = "test-bow"

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.calls = 0

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        self.calls += 1
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                if len(word) > 2:
                    out[row, zlib.crc32(word.encode()) % self.dim] += 1.0
            norm = np.linalg.norm(out[row])
            if norm:
                out[row] /= norm
        return out


def make_brief(brief_id: str, claims: list[str], *, status: str = "approved", watchlist: list[str] | None = None) -> Brief:
    claim_objs = [Claim(id=f"c{i}", text=t, citations=["a_prior0000001"]) for i, t in enumerate(claims, start=1)]
    return Brief(
        id=brief_id,
        topic="Enterprise AI infrastructure",
        title=f"Brief {brief_id}",
        created_at=datetime(2026, 9, 30, tzinfo=timezone.utc),
        period_start=datetime(2026, 9, 1, tzinfo=timezone.utc),
        period_end=datetime(2026, 9, 30, tzinfo=timezone.utc),
        executive_summary=claim_objs[:1],
        sections=[BriefSection(title="Funding & M&A", claims=claim_objs[1:])],
        implications=[],
        watchlist=watchlist or [],
        continuity_text=["Continues: GPU pricing pressure noted in August."],
        sources=[SourceRef(article_id="a_prior0000001", title="Prior", source_name="Cloud Ledger")],
        status=status,  # type: ignore[arg-type]
    )


PRIOR = make_brief(
    "b_20260930_aaaaaa",
    [
        "Meridian Cloud was reported to be in talks to acquire Vireo Analytics.",
        "Northwind Compute faces GPU pricing pressure and a price war is expected.",
        "Aurora Silicon is in fundraising talks with growth investors.",
        "Northwind Compute raised GPU-hour prices for H-class instances.",
    ],
    watchlist=["Meridian Cloud — EU scrutiny of cloud egress fees"],
)


@pytest.fixture
def embedder() -> BagOfWordsEmbedder:
    return BagOfWordsEmbedder()


@pytest.fixture
def index(tmp_path: Path, embedder) -> BriefIndex:
    idx = BriefIndex(tmp_path / "index", embedder)
    idx.add_brief(PRIOR)
    return idx


# --------------------------------------------------------------------------- #
# BriefIndex
# --------------------------------------------------------------------------- #


def test_chunks_cover_claims_watchlist_and_continuity():
    chunks = BriefIndex.chunks_for(PRIOR)
    texts = [c["text"] for c in chunks]
    assert texts[0] == "Executive summary: Meridian Cloud was reported to be in talks to acquire Vireo Analytics."
    assert "Funding & M&A: Aurora Silicon is in fundraising talks with growth investors." in texts
    assert any(t.startswith("Watchlist: Meridian Cloud") for t in texts)
    assert any(t.startswith("Continuity: ") for t in texts)
    assert len({c["claim_id"] for c in chunks}) == len(chunks)


def test_add_and_search(index):
    hits = index.search("Meridian Cloud acquire Vireo Analytics", k=3, min_score=0.1)
    assert hits and hits[0].claim_id == "c1"
    assert hits[0].brief_id == PRIOR.id and hits[0].brief_title == PRIOR.title
    assert hits[0].brief_date == PRIOR.period_end
    assert all(h.score >= 0.1 for h in hits)
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_index_persists_and_reloads(tmp_path, index, embedder):
    assert (tmp_path / "index" / "chunks.jsonl").exists() and (tmp_path / "index" / "vectors.npy").exists()
    reloaded = BriefIndex(tmp_path / "index", embedder)
    assert len(reloaded) == len(index)
    assert reloaded.brief_ids() == [PRIOR.id]
    assert reloaded.search("Aurora Silicon fundraising", k=1)[0].claim_id == "c3"


def test_reload_with_different_dimension_starts_empty(tmp_path, index):
    assert len(BriefIndex(tmp_path / "index", BagOfWordsEmbedder(dim=64))) == 0


def test_add_brief_is_idempotent_and_remove_works(index):
    before = len(index)
    assert index.add_brief(PRIOR) == before
    assert len(index) == before
    index.add_brief(make_brief("b_20260915_bbbbbb", ["Kestrel Networks expanded hiring in Austin."]))
    assert index.brief_ids() == [PRIOR.id, "b_20260915_bbbbbb"]
    index.remove_brief(PRIOR.id)
    assert index.brief_ids() == ["b_20260915_bbbbbb"]
    index.remove_brief("b_missing")  # no-op


def test_search_excludes_briefs_filters_and_dedupes(index):
    # Same claim text in a second brief must not produce a duplicate hit.
    index.add_brief(make_brief("b_20260915_bbbbbb", ["Meridian Cloud was reported to be in talks to acquire Vireo Analytics."]))
    hits = index.search("Meridian Cloud talks acquire Vireo Analytics", k=5, min_score=0.2)
    texts = [h.text.split(": ", 1)[1] for h in hits]
    assert len(texts) == len(set(texts))
    excluded = index.search("Meridian Cloud talks acquire Vireo Analytics", k=5, exclude_brief_ids=[PRIOR.id])
    assert excluded and all(h.brief_id == "b_20260915_bbbbbb" for h in excluded)
    assert index.search("completely unrelated zebra migration", k=5, min_score=0.3) == []
    assert index.search("   ", k=5) == []


def test_rebuild_indexes_only_approved_briefs(tmp_path, embedder):
    d1, d2 = tmp_path / "briefs", tmp_path / "fixtures"
    d1.mkdir()
    d2.mkdir()
    (d1 / "approved.json").write_text(PRIOR.model_dump_json(), encoding="utf-8")
    (d1 / "draft.json").write_text(make_brief("b_20260901_cccccc", ["Draft claim."], status="needs_review").model_dump_json(), encoding="utf-8")
    (d2 / "broken.json").write_text("{not json", encoding="utf-8")
    (d2 / "other.json").write_text(make_brief("b_20260815_dddddd", ["Older approved claim."]).model_dump_json(), encoding="utf-8")
    idx = BriefIndex(tmp_path / "idx", embedder)
    idx.add_brief(make_brief("b_stale", ["stale"]))
    assert idx.rebuild([d1, d2, tmp_path / "missing"]) == 2
    assert sorted(idx.brief_ids()) == ["b_20260815_dddddd", PRIOR.id]


def test_build_index_seeds_from_configured_dirs(settings, tmp_path):
    briefs_dir = tmp_path / "published"
    briefs_dir.mkdir()
    (briefs_dir / f"{PRIOR.id}.json").write_text(PRIOR.model_dump_json(), encoding="utf-8")
    settings.paths.briefs_dir = str(briefs_dir)
    settings.retrieval.extra_brief_dirs = []
    settings.retrieval.index_dir = str(tmp_path / "built_index")
    idx = build_index(settings)
    assert idx.brief_ids() == [PRIOR.id]
    assert idx.search("Meridian Cloud Vireo Analytics acquisition talks", k=1)[0].brief_id == PRIOR.id


# --------------------------------------------------------------------------- #
# Continuity
# --------------------------------------------------------------------------- #


def record(article_id: str, company: str, event_type: EventType, headline: str) -> ArticleRecord:
    return ArticleRecord(
        article_id=article_id,
        companies=[CompanyMention(name=company, role="subject")],
        events=[ExtractedEvent(event_type=event_type, headline=headline, companies=[company], evidence_quote=headline)],
        categories=["test"],
        sentiment=Sentiment.NEUTRAL,
        confidence=0.9,
        key_facts=[],
    )


def test_continuity_context_queries_topic_and_companies(index):
    records = [
        record("a1", "Meridian Cloud", EventType.M_AND_A, "Meridian Cloud agrees to acquire Vireo Analytics for $1.2B"),
        record("a2", "Aurora Silicon", EventType.FUNDING, "Aurora Silicon raises $410M Series D"),
    ]
    snippets = continuity_context(index, "Enterprise AI infrastructure", records, k=3, min_score=0.15)
    keys = [(s.brief_id, s.claim_id) for s in snippets]
    assert len(keys) == len(set(keys))
    assert len(snippets) <= 6
    assert ("b_20260930_aaaaaa", "c1") in keys and ("b_20260930_aaaaaa", "c3") in keys
    assert [s.score for s in snippets] == sorted((s.score for s in snippets), reverse=True)



def test_briefs_dated_from_and_exclusion_keep_reruns_out_of_prior_context(index):
    # A brief over the current cycle (period ending 30 Oct) must not count as "prior".
    current = make_brief("b_20261030_bbbbbb", ["Meridian Cloud agreed to acquire Vireo Analytics for $1.2 billion."])
    current = current.model_copy(update={"period_end": datetime(2026, 10, 30, 12, tzinfo=timezone.utc)})
    index.add_brief(current)
    cycle_end = datetime(2026, 10, 30, 12, tzinfo=timezone.utc)
    assert index.briefs_dated_from(cycle_end) == ["b_20261030_bbbbbb"]
    assert index.briefs_dated_from(datetime(2026, 9, 1)) == ["b_20260930_aaaaaa", "b_20261030_bbbbbb"]  # naive = UTC

    records = [record("a1", "Meridian Cloud", EventType.M_AND_A, "Meridian Cloud agrees to acquire Vireo Analytics")]
    every = continuity_context(index, "Enterprise AI infrastructure", records, k=4, min_score=0.1)
    assert "b_20261030_bbbbbb" in {s.brief_id for s in every}
    prior_only = continuity_context(
        index, "Enterprise AI infrastructure", records, k=4, min_score=0.1,
        exclude_brief_ids=index.briefs_dated_from(cycle_end),
    )
    assert prior_only and {s.brief_id for s in prior_only} == {"b_20260930_aaaaaa"}

def snippet(claim_id: str, text: str, score: float = 0.5) -> RetrievedSnippet:
    return RetrievedSnippet(brief_id=PRIOR.id, brief_title=PRIOR.title, brief_date=PRIOR.period_end, claim_id=claim_id, text=text, score=score)


def make_draft(*texts: str) -> BriefDraft:
    claims = [Claim(id=f"c{i}", text=t, citations=["a_new000000001"]) for i, t in enumerate(texts, start=1)]
    return BriefDraft(title="New", executive_summary=claims, sections=[], implications=[], watchlist=[])


def test_continuity_notes_label_updates_contradictions_and_continuations():
    draft = make_draft(
        "Meridian Cloud agreed to acquire Vireo Analytics for $1.2 billion.",
        "Northwind Compute cut H-class GPU-hour prices by 22%.",
        "Helix Data Systems deepened its partnership with LumenForge AI.",
    )
    snippets = [
        snippet("c1", "Funding & M&A: Meridian Cloud was reported to be in talks to acquire Vireo Analytics."),
        snippet("c4", "Pricing & competition: Northwind Compute raised GPU-hour prices for H-class instances."),
        snippet("c9", "Products & partnerships: Helix Data Systems announced a partnership with LumenForge AI."),
        snippet("c7", "Regulatory & legal: Quarry Storage faces a pricing inquiry in the UK."),
    ]
    notes = continuity_notes(draft, snippets)
    relations = {n.prior_claim: n.relation for n in notes}
    assert relations == {
        "Meridian Cloud was reported to be in talks to acquire Vireo Analytics.": "updates",
        "Northwind Compute raised GPU-hour prices for H-class instances.": "contradicts",
        "Helix Data Systems announced a partnership with LumenForge AI.": "continues",
    }
    update = next(n for n in notes if n.relation == "updates")
    assert update.prior_brief_id == PRIOR.id
    assert update.note.startswith(f"Updates {PRIOR.title} ({PRIOR.id}/c1)")
    assert "[c1]" in update.note


def test_expected_price_war_materialising_is_an_update():
    draft = make_draft("Northwind Compute cut H-class GPU-hour pricing by 22%.")
    notes = continuity_notes(draft, [snippet("c2", "Funding & M&A: Northwind Compute faces GPU pricing pressure and a price war is expected.")])
    assert [n.relation for n in notes] == ["updates"]


def test_continuity_notes_handle_empty_inputs():
    assert continuity_notes(make_draft(), [snippet("c1", "x")]) == []
    assert continuity_notes(make_draft("Something."), []) == []


def test_bare_watchlist_lines_need_a_matching_event():
    draft = make_draft("Meridian Cloud agreed to acquire Vireo Analytics for $1.2 billion.")
    bare = RetrievedSnippet(brief_id=PRIOR.id, brief_title=PRIOR.title, claim_id="watch1", text="Watchlist: Vireo Analytics", score=0.4)
    assert continuity_notes(draft, [bare]) == []
    with_event = bare.model_copy(update={"claim_id": "watch2", "text": "Watchlist: Vireo Analytics — acquisition talks with Meridian Cloud"})
    assert [n.relation for n in continuity_notes(draft, [with_event])] == ["updates"]
