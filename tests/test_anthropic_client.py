"""AnthropicLLMClient request shaping and response handling (no network)."""

from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ValidationError

from marketbrief.llm import AnthropicLLMClient, LLMError, LLMRequest, StructuredOutputError, structured_call
from marketbrief.llm.anthropic_client import FALLBACK_BETA
from marketbrief.llm.base import RefusalError, TruncatedOutputError

anthropic = pytest.importorskip("anthropic")

MODEL = "claude-opus-5-5"


class Verdict(BaseModel):
    supported: bool
    reason: str


def text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def response(
    *,
    text: str = "",
    parsed=None,
    stop_reason: str = "end_turn",
    stop_details=None,
    usage=None,
    content=None,
    model: str = MODEL,
) -> SimpleNamespace:
    return SimpleNamespace(
        content=content if content is not None else ([text_block(text)] if text else []),
        parsed_output=parsed,
        stop_reason=stop_reason,
        stop_details=stop_details,
        usage=usage
        or SimpleNamespace(input_tokens=120, output_tokens=30, cache_read_input_tokens=900, cache_creation_input_tokens=None),
        model=model,
    )


class FakeMessages:
    def __init__(self, responses: list) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, dict]] = []

    def _next(self, method: str, kwargs: dict):
        self.calls.append((method, kwargs))
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(item, Exception):
            raise item
        return item

    def parse(self, **kwargs):
        return self._next("parse", kwargs)

    def create(self, **kwargs):
        return self._next("create", kwargs)


def fake_client(*responses) -> tuple[AnthropicLLMClient, FakeMessages]:
    messages = FakeMessages(list(responses))
    sdk = SimpleNamespace(beta=SimpleNamespace(messages=messages))
    return AnthropicLLMClient(MODEL, client=sdk), messages


def structured_request(**overrides) -> LLMRequest:
    data = dict(
        task="fact_check",
        system="You are a meticulous fact-checker.",
        prompt="<claims>...</claims>",
        response_model=Verdict,
        effort="medium",
        temperature=0.0,
    )
    data.update(overrides)
    return LLMRequest(**data)


# --------------------------------------------------------------------------- #
# Request shape
# --------------------------------------------------------------------------- #


def test_structured_call_uses_parse_with_cached_system_and_fallbacks():
    verdict = Verdict(supported=True, reason="matches")
    client, messages = fake_client(response(text=verdict.model_dump_json(), parsed=verdict))
    result = client.complete(structured_request())

    method, kwargs = messages.calls[0]
    assert method == "parse"
    assert kwargs["model"] == MODEL
    assert kwargs["max_tokens"] == 16000
    assert kwargs["output_format"] is Verdict
    assert kwargs["output_config"] == {"effort": "medium"}
    assert kwargs["system"] == [
        {"type": "text", "text": "You are a meticulous fact-checker.", "cache_control": {"type": "ephemeral"}}
    ]
    assert kwargs["messages"] == [{"role": "user", "content": "<claims>...</claims>"}]
    assert kwargs["betas"] == [FALLBACK_BETA] and kwargs["fallbacks"] == "default"
    for forbidden in ("temperature", "top_p", "top_k", "thinking"):
        assert forbidden not in kwargs

    assert result.parsed == verdict
    assert result.provider == "anthropic" and result.model == MODEL
    assert result.stop_reason == "end_turn"
    assert result.usage.model_dump() == {
        "input_tokens": 120,
        "output_tokens": 30,
        "cache_read_input_tokens": 900,
        "cache_creation_input_tokens": 0,
    }


def test_text_request_uses_create_and_joins_text_blocks():
    blocks = [SimpleNamespace(type="thinking", thinking=""), text_block("Hello "), text_block("world")]
    client, messages = fake_client(response(content=blocks))
    result = client.complete(LLMRequest(task="note", system="s", prompt="p", effort="low", max_tokens=500))
    method, kwargs = messages.calls[0]
    assert method == "create" and "output_format" not in kwargs
    assert kwargs["max_tokens"] == 500 and kwargs["output_config"] == {"effort": "low"}
    assert result.text == "Hello world" and result.parsed is None


def test_fallbacks_can_be_disabled_and_empty_system_is_omitted():
    messages = FakeMessages([response(text="ok")])
    client = AnthropicLLMClient(MODEL, use_server_fallbacks=False, client=SimpleNamespace(beta=SimpleNamespace(messages=messages)))
    client.complete(LLMRequest(task="note", system="  ", prompt="p"))
    kwargs = messages.calls[0][1]
    assert "betas" not in kwargs and "fallbacks" not in kwargs
    assert "system" not in kwargs
    assert kwargs["output_config"] == {"effort": "medium"}  # explicit default


def test_invalid_effort_is_rejected_before_calling_the_api():
    client, messages = fake_client(response(text="ok"))
    with pytest.raises(LLMError, match="invalid effort"):
        client.complete(structured_request(effort="extreme"))
    assert messages.calls == []


def test_params_bind_to_the_installed_sdk_signatures():
    """Every kwarg we send must exist on the real SDK methods."""
    sdk = anthropic.Anthropic(api_key="test-key")
    params = AnthropicLLMClient(MODEL).build_params(structured_request())
    inspect.signature(sdk.beta.messages.parse).bind(output_format=Verdict, **params)
    inspect.signature(sdk.beta.messages.create).bind(**params)
    assert "temperature" not in inspect.signature(sdk.beta.messages.parse).parameters


# --------------------------------------------------------------------------- #
# Stop reasons and errors
# --------------------------------------------------------------------------- #


def test_refusal_raises_with_category():
    details = SimpleNamespace(type="refusal", category="cyber", explanation="Policy decline.")
    client, _ = fake_client(response(stop_reason="refusal", stop_details=details))
    with pytest.raises(RefusalError, match=r"category: cyber.*Policy decline"):
        client.complete(structured_request())


def test_refusal_without_details():
    client, _ = fake_client(response(stop_reason="refusal", stop_details=None))
    with pytest.raises(RefusalError, match="declined task 'fact_check'"):
        client.complete(structured_request())


def test_max_tokens_raises_truncated():
    client, _ = fake_client(response(text='{"supported": tr', stop_reason="max_tokens"))
    with pytest.raises(TruncatedOutputError):
        client.complete(structured_request())


def test_validation_error_from_parse_propagates_and_structured_call_reasks():
    try:
        Verdict.model_validate({"supported": "maybe"})
    except ValidationError as exc:
        error = exc
    good = Verdict(supported=False, reason="number mismatch")
    client, messages = fake_client(error, response(text=good.model_dump_json(), parsed=good))

    with pytest.raises(ValidationError):
        AnthropicLLMClient(MODEL, client=SimpleNamespace(beta=SimpleNamespace(messages=FakeMessages([error])))).complete(
            structured_request()
        )

    result = structured_call(client, structured_request())
    assert result.parsed == good
    assert "<validation_feedback>" in messages.calls[1][1]["messages"][0]["content"]


def test_unparsed_text_is_validated_locally():
    client, _ = fake_client(response(text='{"supported": true}', parsed=None))
    with pytest.raises(ValidationError):
        client.complete(structured_request())


def test_missing_structured_output_surfaces_as_structured_error():
    client, _ = fake_client(response(content=[], parsed=None))
    assert client.complete(structured_request()).parsed is None
    with pytest.raises(StructuredOutputError):
        structured_call(client, structured_request(), max_retries=1)


def test_api_errors_are_wrapped_in_llm_error():
    import httpx2

    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    bad_request = anthropic.BadRequestError(
        "temperature is not supported", response=httpx2.Response(400, request=req), body=None
    )
    client, _ = fake_client(bad_request)
    with pytest.raises(LLMError, match="400"):
        client.complete(structured_request())

    client, _ = fake_client(anthropic.APIConnectionError(request=req))
    with pytest.raises(LLMError, match="could not reach"):
        client.complete(structured_request())


def test_lazy_sdk_client_construction(monkeypatch):
    created = {}

    class Recorder:
        def __init__(self, **kwargs):
            created.update(kwargs)
            self.beta = SimpleNamespace(messages=FakeMessages([response(text="ok")]))

    monkeypatch.setattr(anthropic, "Anthropic", Recorder)
    client = AnthropicLLMClient(MODEL, timeout_s=42.0)
    assert client._client is None
    client.complete(LLMRequest(task="note", system="s", prompt="p"))
    assert created == {"timeout": 42.0, "max_retries": 3}


# --------------------------------------------------------------------------- #
# Real SDK, mocked HTTP transport: checks the actual wire format
# --------------------------------------------------------------------------- #


def test_wire_request_through_real_sdk():
    httpx2 = pytest.importorskip("httpx2")
    seen: dict = {}
    verdict = Verdict(supported=True, reason="ok")

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen["headers"] = dict(request.headers)
        seen["body"] = json.loads(request.content)
        return httpx2.Response(
            200,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": MODEL,
                "content": [{"type": "text", "text": verdict.model_dump_json()}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 50, "output_tokens": 9, "cache_read_input_tokens": 1200, "cache_creation_input_tokens": 0},
            },
        )

    sdk = anthropic.Anthropic(api_key="test-key", http_client=httpx2.Client(transport=httpx2.MockTransport(handler)), max_retries=0)
    result = AnthropicLLMClient(MODEL, client=sdk).complete(structured_request(effort="high"))

    body = seen["body"]
    assert result.parsed == verdict
    assert result.usage.cache_read_input_tokens == 1200
    assert body["output_config"]["effort"] == "high"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert body["fallbacks"] == "default"
    assert FALLBACK_BETA in seen["headers"].get("anthropic-beta", "")
    for forbidden in ("temperature", "top_p", "top_k", "thinking", "output_format"):
        assert forbidden not in body
