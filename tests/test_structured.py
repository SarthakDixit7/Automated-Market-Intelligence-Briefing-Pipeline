"""Instructor-style re-ask loop (structured_call)."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from pydantic import BaseModel, Field, ValidationError

from marketbrief.llm import LLMRequest, LLMResult, StructuredOutputError, Usage, structured_call
from marketbrief.llm.base import RefusalError


class Answer(BaseModel):
    value: int = Field(ge=0)
    label: str


class ScriptedClient:
    """Plays back a list of outcomes: an Answer/dict, None, or an exception."""

    provider = "scripted"
    model = "scripted-1"

    def __init__(self, outcomes: list) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[LLMRequest] = []
        self._lock = threading.Lock()

    def complete(self, request: LLMRequest) -> LLMResult:
        with self._lock:
            self.requests.append(request)
            outcome = self.outcomes[min(len(self.requests) - 1, len(self.outcomes) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        return LLMResult(text="{}", parsed=outcome, usage=Usage(input_tokens=100, output_tokens=10))


def make_request() -> LLMRequest:
    return LLMRequest(task="extract", system="sys", prompt="Extract the answer.", response_model=Answer)


def validation_error() -> ValidationError:
    try:
        Answer.model_validate({"value": -1})
    except ValidationError as exc:
        return exc
    raise AssertionError("unreachable")


def test_first_attempt_passes():
    client = ScriptedClient([Answer(value=1, label="ok")])
    result = structured_call(client, make_request())
    assert result.parsed == Answer(value=1, label="ok")
    assert len(client.requests) == 1
    assert result.usage == Usage(input_tokens=100, output_tokens=10)


def test_reasks_after_pydantic_validation_error_with_feedback():
    client = ScriptedClient([validation_error(), Answer(value=2, label="fixed")])
    result = structured_call(client, make_request())
    assert result.parsed.value == 2
    retry_prompt = client.requests[1].prompt
    assert retry_prompt.startswith("Extract the answer.")
    assert "<validation_feedback>" in retry_prompt
    assert "- value: Input should be greater than or equal to 0" in retry_prompt
    assert "- label: Field required" in retry_prompt


def test_reasks_when_parsed_is_missing():
    client = ScriptedClient([None, Answer(value=3, label="x")])
    result = structured_call(client, make_request())
    assert result.parsed.value == 3
    assert "did not contain an object" in client.requests[1].prompt


def test_semantic_validators_drive_reask_and_usage_accumulates():
    def must_be_even(obj: BaseModel) -> list[str]:
        return [] if obj.value % 2 == 0 else [f"value {obj.value} must be even"]

    def label_lowercase(obj: BaseModel) -> list[str]:
        return [] if obj.label.islower() else ["label must be lowercase"]

    client = ScriptedClient([Answer(value=1, label="A"), Answer(value=4, label="B"), Answer(value=4, label="b")])
    result = structured_call(client, make_request(), validators=[must_be_even, label_lowercase], max_retries=2)
    assert result.parsed == Answer(value=4, label="b")
    assert result.usage == Usage(input_tokens=300, output_tokens=30)
    second, third = client.requests[1].prompt, client.requests[2].prompt
    assert "- value 1 must be even\n- label must be lowercase" in second
    # feedback is rebuilt from the original prompt, not stacked
    assert third.count("<validation_feedback>") == 1
    assert "must be even" not in third and "label must be lowercase" in third


def test_raises_after_exhausting_retries():
    client = ScriptedClient([Answer(value=1, label="x")])
    with pytest.raises(StructuredOutputError, match="after 3 attempt"):
        structured_call(client, make_request(), validators=[lambda o: ["always wrong"]], max_retries=2)
    assert len(client.requests) == 3


def test_zero_retries_means_single_attempt():
    client = ScriptedClient([None])
    with pytest.raises(StructuredOutputError):
        structured_call(client, make_request(), max_retries=0)
    assert len(client.requests) == 1


def test_dict_output_is_coerced_into_response_model():
    client = ScriptedClient([{"value": 5, "label": "from-dict"}])
    result = structured_call(client, make_request())
    assert isinstance(result.parsed, Answer) and result.parsed.value == 5


def test_invalid_dict_output_triggers_reask():
    client = ScriptedClient([{"value": "nope"}, {"value": 1, "label": "ok"}])
    result = structured_call(client, make_request())
    assert result.parsed.value == 1
    assert "value:" in client.requests[1].prompt


def test_provider_errors_propagate_without_retry():
    client = ScriptedClient([RefusalError("declined")])
    with pytest.raises(RefusalError):
        structured_call(client, make_request())
    assert len(client.requests) == 1


def test_free_text_requests_skip_schema_checks():
    client = ScriptedClient([None])
    request = LLMRequest(task="note", system="s", prompt="p")
    assert structured_call(client, request).parsed is None


def test_thread_safe_under_concurrency():
    def validator(obj: BaseModel) -> list[str]:
        return [] if obj.value >= 0 else ["negative"]

    client = ScriptedClient([Answer(value=7, label="x")])
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda _: structured_call(client, make_request(), validators=[validator]), range(64)))
    assert all(r.parsed.value == 7 for r in results)
    assert len(client.requests) == 64


def test_works_end_to_end_with_offline_provider(tmp_path):
    from marketbrief.llm import OfflineLLMClient
    from marketbrief.schemas import ArticleRecord

    article = {
        "id": "a_1",
        "title": "Orbital Labs raises $20 million Series A",
        "text": "Orbital Labs raised $20 million in a Series A round on Monday. The startup builds AI security tools.",
    }
    request = LLMRequest(
        task="extract", system="s", prompt="p", response_model=ArticleRecord, context={"article": article}
    )
    client = OfflineLLMClient(entities_path=tmp_path / "none.yaml")

    def quotes_are_verbatim(record: BaseModel) -> list[str]:
        return [f"quote not in article: {e.evidence_quote}" for e in record.events if e.evidence_quote not in article["text"]]

    result = structured_call(client, request, validators=[quotes_are_verbatim])
    assert result.parsed.events[0].amount == "$20M"

    with pytest.raises(StructuredOutputError):
        structured_call(client, request, validators=[lambda r: ["impossible requirement"]], max_retries=1)
