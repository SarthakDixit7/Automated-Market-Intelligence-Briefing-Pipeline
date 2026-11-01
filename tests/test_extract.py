"""Typed extraction: validators, re-ask loop, parallelism and failure isolation."""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone

import pytest
from pydantic import BaseModel

from marketbrief.extract import extract_records, quote_in_text, validate_record
from marketbrief.llm.base import LLMError, LLMRequest, LLMResult, Usage
from marketbrief.schemas import Article, ArticleRecord, CompanyMention, EventType, ExtractedEvent, Sentiment

TEXT = (
    "Northwind Compute said on Friday it will cut H-class GPU-hour pricing by 22%, effective immediately.\n\n"
    "“We want every start-up to afford serious compute,” chief executive Lena Ortiz said."
)


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
        return LLMResult(
            text=out.model_dump_json(), parsed=out, usage=Usage(input_tokens=10, output_tokens=5), provider="fake", model="fake-1"
        )


def make_article(article_id: str = "a_000000000001", text: str = TEXT) -> Article:
    return Article(
        id=article_id,
        source_id="test",
        source_name="Datacenter Dispatch",
        title="Northwind Compute cuts GPU pricing",
        published_at=datetime(2026, 10, 9, tzinfo=timezone.utc),
        text=text,
        content_hash=article_id,
    )


def make_record(article_id: str = "a_000000000001", quote: str | None = None, **updates) -> ArticleRecord:
    record = ArticleRecord(
        article_id=article_id,
        companies=[CompanyMention(name="Northwind Compute", role="subject")],
        events=[
            ExtractedEvent(
                event_type=EventType.PRICING,
                headline="Northwind Compute cuts H-class GPU-hour pricing by 22%",
                companies=["Northwind Compute"],
                amount="22%",
                evidence_quote=quote
                or "Northwind Compute said on Friday it will cut H-class GPU-hour pricing by 22%, effective immediately.",
            )
        ],
        categories=["gpu cloud"],
        sentiment=Sentiment.MIXED,
        confidence=0.9,
        key_facts=["Northwind Compute cut H-class GPU-hour pricing by 22%."],
    )
    return record.model_copy(update=updates)


# --------------------------------------------------------------------------- #
# Validators
# --------------------------------------------------------------------------- #


def test_valid_record_has_no_errors():
    assert validate_record(make_record(), make_article()) == []


def test_wrong_article_id_is_reported_unless_disabled():
    record = make_record("a_wrong")
    errors = validate_record(record, make_article())
    assert any("article_id" in e for e in errors)
    assert validate_record(record, make_article(), check_article_id=False) == []


def test_paraphrased_quote_is_rejected():
    record = make_record(quote="Northwind Compute is slashing prices by 22 percent.")
    errors = validate_record(record, make_article())
    assert len(errors) == 1 and "verbatim" in errors[0]


def test_quote_matching_tolerates_case_quotes_whitespace_and_ellipsis():
    article = make_article()
    assert quote_in_text('"we want every start-up   to afford serious compute," chief executive lena ortiz said', article.text)
    assert quote_in_text("Northwind Compute said on Friday ... by 22%", article.text)
    assert not quote_in_text("by 22% ... Northwind Compute said", article.text)  # parts must be in order
    assert not quote_in_text("   ", article.text)


def test_quote_from_title_only_is_rejected():
    record = make_record(quote="Northwind Compute cuts GPU pricing")
    assert validate_record(record, make_article())


def test_event_without_company_is_rejected_except_other():
    record = make_record()
    record.events[0].companies = []
    assert any("at least one company" in e for e in validate_record(record, make_article()))
    record.events[0].event_type = EventType.OTHER
    assert validate_record(record, make_article()) == []


def test_key_fact_cap_and_lowercase_categories():
    record = make_record(key_facts=[f"fact {i}" for i in range(6)], categories=["GPU Cloud"])
    errors = validate_record(record, make_article())
    assert any("key_facts" in e for e in errors)
    assert any("lowercase" in e for e in errors)


# --------------------------------------------------------------------------- #
# extract_records
# --------------------------------------------------------------------------- #


@pytest.fixture
def fast_settings(settings):
    settings.pipeline.max_workers = 4
    settings.llm.max_validation_retries = 2
    return settings


def test_extract_records_builds_the_contract_request(fast_settings):
    llm = FakeLLM(lambda req: make_record(req.context["article"]["id"]))
    records = extract_records([make_article()], llm, fast_settings)
    assert [r.article_id for r in records] == ["a_000000000001"]
    req = llm.requests[0]
    assert req.task == "extract"
    assert req.response_model is ArticleRecord
    assert req.effort == fast_settings.llm.effort["extract"]
    assert req.max_tokens == fast_settings.llm.max_tokens
    assert req.temperature == fast_settings.llm.temperature
    assert req.context["article"]["id"] == "a_000000000001"
    assert req.context["article"]["text"] == TEXT
    assert TEXT.splitlines()[0] in req.prompt  # the live model sees the article in the prompt


def test_extract_records_preserves_order_under_parallelism(fast_settings):
    articles = [make_article(f"a_{i:012d}") for i in range(12)]

    def respond(req: LLMRequest) -> ArticleRecord:
        time.sleep(random.uniform(0, 0.01))
        return make_record(req.context["article"]["id"])

    records = extract_records(articles, FakeLLM(respond), fast_settings)
    assert [r.article_id for r in records] == [a.id for a in articles]


def test_wrong_article_id_is_fixed_silently(fast_settings):
    llm = FakeLLM(lambda req: make_record("a_hallucinated"))
    records = extract_records([make_article()], llm, fast_settings)
    assert records[0].article_id == "a_000000000001"
    assert len(llm.requests) == 1  # no re-ask needed


def test_invalid_quote_triggers_reask_with_feedback(fast_settings):
    def respond(req: LLMRequest) -> ArticleRecord:
        if "<validation_feedback>" not in req.prompt:
            return make_record(quote="Northwind slashes prices.")
        return make_record()

    llm = FakeLLM(respond)
    records = extract_records([make_article()], llm, fast_settings)
    assert len(records) == 1
    assert len(llm.requests) == 2
    assert "evidence_quote is not a verbatim sentence" in llm.requests[1].prompt


def test_failures_are_isolated_and_skipped(fast_settings, caplog):
    articles = [make_article(f"a_{i:012d}") for i in range(4)]

    def respond(req: LLMRequest):
        article_id = req.context["article"]["id"]
        if article_id == articles[1].id:
            return LLMError("provider exploded")
        if article_id == articles[2].id:
            return make_record(article_id, quote="never valid")  # fails every re-ask
        return make_record(article_id)

    llm = FakeLLM(respond)
    with caplog.at_level("WARNING"):
        records = extract_records(articles, llm, fast_settings)
    assert [r.article_id for r in records] == [articles[0].id, articles[3].id]
    assert sum(r.context["article"]["id"] == articles[2].id for r in llm.requests) == 3  # 1 + 2 re-asks
    assert "extraction failed" in caplog.text


def test_empty_input_makes_no_calls(fast_settings):
    llm = FakeLLM(lambda req: make_record())
    assert extract_records([], llm, fast_settings) == []
    assert llm.requests == []


def test_offline_provider_output_passes_validation(settings):
    try:
        from marketbrief.llm.offline import OfflineLLMClient
    except ImportError:
        pytest.skip("offline provider not available")
    entities = settings.path(settings.paths.entities_file)
    llm = OfflineLLMClient(entities_path=entities if entities.exists() else None)
    article = make_article(
        text=(
            "Aurora Silicon said on Tuesday it has raised $410 million in a Series D round led by Halcyon Partners.\n\n"
            "The chip designer will use the money to expand its inference-accelerator roadmap."
        )
    )
    records = extract_records([article], llm, settings)
    assert len(records) == 1
    assert records[0].article_id == article.id
    assert validate_record(records[0], article) == []
