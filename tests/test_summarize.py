"""Map-reduce summarisation: map, single and hierarchical reduce, editor pass."""

from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import datetime, timezone

import pytest
from pydantic import BaseModel

from marketbrief.citations import all_claims
from marketbrief.llm.base import LLMError, LLMRequest, LLMResult, Usage
from marketbrief.schemas import (
    Article,
    ArticleRecord,
    ArticleSummary,
    BriefDraft,
    BriefSection,
    Claim,
    CompanyMention,
    EventType,
    RetrievedSnippet,
    Sentiment,
)
from marketbrief.summarize import draft_errors, edit_brief, map_summaries, reduce_brief, renumber_claims


class FakeLLM:
    """LLMClient double: ``respond(request)`` returns a model, or an exception to raise."""

    provider = "fake"
    model = "fake-1"

    def __init__(self, respond: Callable[[LLMRequest], BaseModel | Exception]) -> None:
        self.respond = respond
        self.requests: list[LLMRequest] = []
        self._lock = threading.Lock()

    def complete(self, request: LLMRequest) -> LLMResult:
        with self._lock:
            self.requests.append(request)
        out = self.respond(request)
        if isinstance(out, Exception):
            raise out
        return LLMResult(text=out.model_dump_json(), parsed=out, usage=Usage(input_tokens=10), provider="fake", model="fake-1")


def aid(i: int) -> str:
    return f"a_{i:012d}"


def make_article(i: int) -> Article:
    return Article(
        id=aid(i),
        source_id="t",
        source_name="InfraWeekly",
        title=f"Story {i}",
        published_at=datetime(2026, 10, 10, tzinfo=timezone.utc),
        text=f"Company {i} did thing {i}. It matters.",
        content_hash=str(i),
    )


def make_record(i: int) -> ArticleRecord:
    return ArticleRecord(
        article_id=aid(i),
        companies=[CompanyMention(name=f"Company {i}", role="subject")],
        events=[],
        categories=["test"],
        sentiment=Sentiment.NEUTRAL,
        confidence=0.8,
        key_facts=[],
    )


def make_summary(i: int) -> ArticleSummary:
    return ArticleSummary(
        article_id=aid(i),
        summary=f"Company {i} did thing {i}.",
        key_points=[f"thing {i}"],
        why_it_matters="It may matter.",
        companies=[f"Company {i}"],
        event_types=[EventType.OTHER],
    )


def draft_for(article_ids: list[str], *, title: str = "Draft", start: int = 1) -> BriefDraft:
    """A valid draft with one fact claim per article and one analysis claim."""
    claims = [Claim(id=f"c{start + n}", text=f"Fact about {a}.", citations=[a]) for n, a in enumerate(article_ids)]
    return BriefDraft(
        title=title,
        executive_summary=claims[:1],
        sections=[BriefSection(title="Other developments", claims=claims[1:])] if len(claims) > 1 else [],
        implications=[
            Claim(id=f"c{start + len(claims)}", text="Analysis.", citations=article_ids[:1], kind="analysis")
        ],
        watchlist=["Company 1"],
    )


SNIPPET = RetrievedSnippet(brief_id="b_20260930_aaaaaa", brief_title="Prior", claim_id="c1", text="Prior: x", score=0.5)

# --------------------------------------------------------------------------- #
# Map
# --------------------------------------------------------------------------- #


def test_map_summaries_skips_articles_without_records_and_keeps_order(settings):
    articles = [make_article(i) for i in range(1, 6)]
    records = [make_record(i) for i in (5, 1, 3)]  # record order differs from article order
    llm = FakeLLM(lambda req: make_summary(int(req.context["article"]["id"][2:])))
    summaries = map_summaries(articles, records, llm, settings)
    assert [s.article_id for s in summaries] == [aid(1), aid(3), aid(5)]
    req = llm.requests[0]
    assert req.task == "map_summary" and req.response_model is ArticleSummary
    assert set(req.context) == {"article", "record"}
    assert req.effort == settings.llm.effort["map_summary"]
    assert "<extracted_record" in req.prompt


def test_map_summary_with_wrong_id_is_reasked_then_skipped(settings):
    articles = [make_article(1), make_article(2)]
    records = [make_record(1), make_record(2)]

    def respond(req):
        if req.context["article"]["id"] == aid(2):
            return make_summary(99)  # always wrong
        return make_summary(1)

    llm = FakeLLM(respond)
    summaries = map_summaries(articles, records, llm, settings)
    assert [s.article_id for s in summaries] == [aid(1)]
    assert sum(r.context["article"]["id"] == aid(2) for r in llm.requests) == 1 + settings.llm.max_validation_retries


# --------------------------------------------------------------------------- #
# Reduce
# --------------------------------------------------------------------------- #


def test_single_batch_reduce_passes_contract_context(settings):
    summaries = [make_summary(i) for i in range(1, 4)]
    records = [make_record(i) for i in range(1, 4)]
    llm = FakeLLM(lambda req: draft_for([s["article_id"] for s in req.context["summaries"]]))
    draft = reduce_brief("AI infra", summaries, records, llm, settings, prior_context=[SNIPPET])
    assert draft_errors(draft, {aid(i) for i in range(1, 4)}) == []
    assert len(llm.requests) == 1
    req = llm.requests[0]
    assert req.task == "reduce_brief" and req.response_model is BriefDraft
    assert set(req.context) == {"topic", "summaries", "records", "prior_context"}
    assert req.context["prior_context"][0]["brief_id"] == "b_20260930_aaaaaa"
    assert 'brief_id="b_20260930_aaaaaa"' in req.prompt


def test_reduce_reasks_on_unknown_citation_and_missing_citation(settings):
    summaries = [make_summary(1), make_summary(2)]

    def respond(req):
        if "<validation_feedback>" in req.prompt:
            return draft_for([aid(1), aid(2)])
        bad = draft_for([aid(1), aid(2)])
        bad.executive_summary[0].citations = ["a_invented0000"]
        bad.sections[0].claims[0].citations = []
        return bad

    llm = FakeLLM(respond)
    draft = reduce_brief("T", summaries, [], llm, settings, prior_context=[])
    assert len(llm.requests) == 2
    feedback = llm.requests[1].prompt
    assert "a_invented0000" in feedback and "no citation" in feedback
    assert draft_errors(draft, {aid(1), aid(2)}) == []


def test_reduce_rejects_duplicate_claim_ids(settings):
    def respond(req):
        d = draft_for([aid(1), aid(2)])
        if "<validation_feedback>" not in req.prompt:
            d.sections[0].claims[0].id = "c1"
        return d

    llm = FakeLLM(respond)
    reduce_brief("T", [make_summary(1), make_summary(2)], [], llm, settings, prior_context=[])
    assert "unique" in llm.requests[1].prompt


def test_hierarchical_reduce_batches_then_merges(settings):
    settings.pipeline.reduce_batch_size = 2
    summaries = [make_summary(i) for i in range(1, 6)]
    records = [make_record(i) for i in range(1, 6)]

    def respond(req):
        if "partials" in req.context:
            # Merge: concatenate partial claims, keeping their (colliding) ids.
            partials = [BriefDraft.model_validate(p) for p in req.context["partials"]]
            claims = [c for p in partials for c in all_claims(p) if c.kind == "fact"]
            return BriefDraft(
                title="Merged",
                executive_summary=claims[:2],
                sections=[BriefSection(title="All", claims=claims[2:])],
                implications=[c for p in partials for c in p.implications],
                watchlist=[],
                continuity=[f"Continues [{s['brief_id']}/{s['claim_id']}]: x" for s in req.context["prior_context"]],
            )
        return draft_for([s["article_id"] for s in req.context["summaries"]])

    llm = FakeLLM(respond)
    draft = reduce_brief("T", summaries, records, llm, settings, prior_context=[SNIPPET])

    batch_calls = [r for r in llm.requests if "summaries" in r.context]
    merge_calls = [r for r in llm.requests if "partials" in r.context]
    assert [len(r.context["summaries"]) for r in batch_calls] == [2, 2, 1]
    assert all(r.context["prior_context"] == [] for r in batch_calls)  # continuity decided at the merge
    assert len(merge_calls) == 2  # 3 partials > batch size 2 -> one intermediate level, then the final merge
    assert merge_calls[-1].context["prior_context"][0]["brief_id"] == SNIPPET.brief_id
    assert set(merge_calls[-1].context) == {"topic", "partials", "prior_context"}
    assert "<partial_briefs>" in merge_calls[-1].prompt

    ids = [c.id for c in all_claims(draft)]
    assert ids == [f"c{i}" for i in range(1, len(ids) + 1)]  # renumbered, unique
    cited = {cid for c in all_claims(draft) for cid in c.citations}
    assert cited == {aid(i) for i in range(1, 6)}
    assert draft.continuity == ["Continues [b_20260930_aaaaaa/c1]: x"]


def test_reduce_requires_summaries(settings):
    with pytest.raises(ValueError):
        reduce_brief("T", [], [], FakeLLM(lambda r: draft_for([aid(1)])), settings, prior_context=[])


def test_renumber_claims_is_reading_order_and_non_mutating():
    d = draft_for([aid(1), aid(2), aid(3)], start=7)
    out = renumber_claims(d)
    assert [c.id for c in all_claims(out)] == ["c1", "c2", "c3", "c4"]
    assert all_claims(d)[0].id == "c7"


# --------------------------------------------------------------------------- #
# Edit
# --------------------------------------------------------------------------- #


def test_edit_brief_returns_edited_draft(settings):
    original = draft_for([aid(1), aid(2)])
    edited = original.model_copy(update={"title": "Tighter title"})
    llm = FakeLLM(lambda req: edited)
    out = edit_brief(original, [aid(1), aid(2)], llm, settings)
    assert out.title == "Tighter title"
    req = llm.requests[0]
    assert req.task == "edit" and set(req.context) == {"draft"}
    assert req.effort == settings.llm.effort["edit"]


def test_edit_brief_rejects_new_citations_and_keeps_original(settings):
    original = draft_for([aid(1)])

    def respond(req):
        bad = original.model_copy(deep=True)
        bad.executive_summary[0].citations.append("a_notallowed00")
        return bad

    llm = FakeLLM(respond)
    out = edit_brief(original, [aid(1)], llm, settings)
    assert out == original
    assert len(llm.requests) == 1 + settings.llm.max_validation_retries
    assert "a_notallowed00" in llm.requests[1].prompt


def test_edit_brief_keeps_original_when_all_claims_dropped(settings):
    original = draft_for([aid(1)])
    empty = BriefDraft(title="Empty", executive_summary=[], sections=[], implications=[], watchlist=[])
    assert edit_brief(original, [aid(1)], FakeLLM(lambda r: empty), settings) == original


def test_edit_brief_survives_provider_errors(settings):
    original = draft_for([aid(1)])
    assert edit_brief(original, [aid(1)], FakeLLM(lambda r: LLMError("boom")), settings) == original


def test_offline_map_reduce_round_trip(settings):
    try:
        from marketbrief.llm.offline import OfflineLLMClient
    except ImportError:
        pytest.skip("offline provider not available")
    from marketbrief.extract import extract_records

    llm = OfflineLLMClient()
    articles = [
        Article(
            id=aid(1),
            source_id="t",
            source_name="Cloud Ledger",
            title="Northwind Compute cuts GPU pricing",
            published_at=datetime(2026, 10, 9, tzinfo=timezone.utc),
            text="Northwind Compute Inc said it will cut GPU-hour pricing by 22% from November. Rivals are expected to respond.",
            content_hash="1",
        ),
        Article(
            id=aid(2),
            source_id="t",
            source_name="InfraWeekly",
            title="Kestrel Networks cuts jobs",
            published_at=datetime(2026, 10, 21, tzinfo=timezone.utc),
            text="Kestrel Networks Inc said it cuts 9% of its workforce as orders slow. The layoffs affect 210 staff.",
            content_hash="2",
        ),
    ]
    records = extract_records(articles, llm, settings)
    summaries = map_summaries(articles, records, llm, settings)
    assert len(summaries) == len(records) >= 1
    draft = reduce_brief("AI infra", summaries, records, llm, settings, prior_context=[])
    assert draft_errors(draft, {a.id for a in articles}) == []
    edited = edit_brief(draft, [a.id for a in articles], llm, settings)
    assert all_claims(edited)
