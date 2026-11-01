"""Provider-neutral LLM interface used by every stage.

A stage builds an :class:`LLMRequest` (task tag, role system prompt, rendered
user prompt, optional Pydantic output schema) and hands it to any
:class:`LLMClient`. Providers: ``AnthropicLLMClient`` (Claude),
``OfflineLLMClient`` (deterministic heuristics, for tests and demos) and the
``CachedLLMClient`` wrapper (prompt cache).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

# Task tags. The offline provider and the prompt cache dispatch on these.
TASKS = (
    "extract",
    "map_summary",
    "reduce_brief",
    "edit",
    "fact_check",
    "judge",
    "agent_plan",
)


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens + other.cache_read_input_tokens,
            cache_creation_input_tokens=self.cache_creation_input_tokens + other.cache_creation_input_tokens,
        )


class LLMRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    task: str
    system: str
    prompt: str
    response_model: type[BaseModel] | None = None
    effort: str | None = None
    max_tokens: int | None = None
    temperature: float | None = 0.0
    # Structured inputs mirroring what was rendered into ``prompt``. Only the
    # offline provider reads this; it is NOT part of the cache key.
    context: dict[str, Any] = Field(default_factory=dict)

    def schema_name(self) -> str:
        return self.response_model.__name__ if self.response_model else "text"

    def schema_fingerprint(self) -> str:
        if not self.response_model:
            return "text"
        raw = json.dumps(self.response_model.model_json_schema(), sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def cache_key(self, model: str) -> str:
        payload = json.dumps(
            {
                "model": model,
                "task": self.task,
                "system": self.system,
                "prompt": self.prompt,
                "schema": self.schema_fingerprint(),
                "effort": self.effort,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def with_feedback(self, feedback: str) -> "LLMRequest":
        """Copy of this request with validation feedback appended (re-ask)."""
        return self.model_copy(
            update={
                "prompt": self.prompt
                + "\n\n<validation_feedback>\nYour previous output failed validation:\n"
                + feedback
                + "\nReturn a corrected output that fixes every problem listed.\n</validation_feedback>"
            }
        )


class LLMResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    text: str = ""
    parsed: Any = None  # instance of request.response_model when one was given
    usage: Usage = Field(default_factory=Usage)
    model: str = ""
    provider: str = ""
    cached: bool = False
    cache_kind: str | None = None  # "exact" | "semantic" when cached
    stop_reason: str | None = None
    latency_s: float = 0.0


class LLMError(RuntimeError):
    pass


class RefusalError(LLMError):
    """Model declined (stop_reason == 'refusal') after any fallbacks."""


class TruncatedOutputError(LLMError):
    """Output hit max_tokens before completing."""


class StructuredOutputError(LLMError):
    """Output still failed validation after all re-ask attempts."""


@runtime_checkable
class LLMClient(Protocol):
    provider: str
    model: str

    def complete(self, request: LLMRequest) -> LLMResult: ...
