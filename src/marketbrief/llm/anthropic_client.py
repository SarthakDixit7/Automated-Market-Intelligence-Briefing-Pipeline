"""Claude provider built on the official ``anthropic`` SDK.

Request shape (per call):

* ``system`` is sent as one text block with ``cache_control: ephemeral`` so the
  stable role prompt (with its few-shot exemplars) is cached server-side and
  re-read at a fraction of the input price on every later call.
* Structured tasks use ``client.beta.messages.parse(output_format=Model)``:
  the SDK converts the Pydantic model into ``output_config.format`` (constrained
  decoding) and validates the reply into ``response.parsed_output``.
* Depth is controlled only through ``output_config.effort``. Sampling params
  (temperature/top_p/top_k) and explicit thinking configs are never sent:
  current Claude models reject them, and thinking is adaptive by default.
* ``fallbacks="default"`` (server-side refusal fallback beta) lets the API
  re-run a declined request on a fallback model inside the same call.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Any

from marketbrief.llm.base import (
    LLMError,
    LLMRequest,
    LLMResult,
    RefusalError,
    TruncatedOutputError,
    Usage,
)

if TYPE_CHECKING:  # imported lazily at runtime so offline runs never load the SDK
    import anthropic

logger = logging.getLogger(__name__)

EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
# Claude Opus 5.5 defaults to "medium"; we always send effort explicitly.
DEFAULT_EFFORT = "medium"
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicLLMClient:
    """:class:`LLMClient` backed by Claude via ``client.beta.messages``."""

    provider = "anthropic"

    def __init__(
        self,
        model: str,
        *,
        max_tokens: int = 16000,
        use_server_fallbacks: bool = True,
        timeout_s: float = 600.0,
        client: "anthropic.Anthropic | None" = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.use_server_fallbacks = use_server_fallbacks
        self.timeout_s = timeout_s
        self._client = client
        self._client_lock = threading.Lock()

    # ------------------------------------------------------------------ SDK

    @property
    def client(self) -> "anthropic.Anthropic":
        """The SDK client, created on first use (thread-safe)."""
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    import anthropic

                    # 429/5xx/connection errors are retried by the SDK itself.
                    self._client = anthropic.Anthropic(timeout=self.timeout_s, max_retries=3)
        return self._client

    def build_params(self, request: LLMRequest) -> dict[str, Any]:
        """Keyword arguments for ``beta.messages.parse/create`` (without ``output_format``)."""
        effort = request.effort or DEFAULT_EFFORT
        if effort not in EFFORT_LEVELS:
            raise LLMError(f"invalid effort {effort!r}; expected one of {', '.join(EFFORT_LEVELS)}")
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": request.max_tokens or self.max_tokens,
            "messages": [{"role": "user", "content": request.prompt}],
            "output_config": {"effort": effort},
        }
        if request.system.strip():  # the API rejects empty text blocks
            params["system"] = [{"type": "text", "text": request.system, "cache_control": {"type": "ephemeral"}}]
        if self.use_server_fallbacks:
            params["betas"] = [FALLBACK_BETA]
            params["fallbacks"] = "default"
        # request.temperature is deliberately ignored: Claude rejects sampling params.
        return params

    # ------------------------------------------------------------------ API

    def complete(self, request: LLMRequest) -> LLMResult:
        """Run one request. Pydantic ``ValidationError`` from parsing propagates so
        ``structured_call`` can re-ask with feedback."""
        import anthropic

        params = self.build_params(request)
        started = time.perf_counter()
        try:
            if request.response_model is not None:
                response = self.client.beta.messages.parse(output_format=request.response_model, **params)
            else:
                response = self.client.beta.messages.create(**params)
        except anthropic.APIStatusError as exc:
            raise LLMError(
                f"Claude API error {exc.status_code} on task {request.task!r}: {exc.message}"
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"could not reach the Claude API for task {request.task!r}: {exc}") from exc
        latency = time.perf_counter() - started

        self._check_stop_reason(request, response)
        text = "".join(
            getattr(block, "text", "") for block in (response.content or []) if getattr(block, "type", None) == "text"
        )
        self._log_fallbacks(request, response)

        parsed = None
        if request.response_model is not None:
            parsed = getattr(response, "parsed_output", None)
            if parsed is None and text.strip():
                # Raises ValidationError with precise feedback for the re-ask loop.
                parsed = request.response_model.model_validate_json(text)

        return LLMResult(
            text=text,
            parsed=parsed,
            usage=_usage(getattr(response, "usage", None)),
            model=getattr(response, "model", None) or self.model,
            provider=self.provider,
            stop_reason=getattr(response, "stop_reason", None),
            latency_s=latency,
        )

    # -------------------------------------------------------------- helpers

    @staticmethod
    def _check_stop_reason(request: LLMRequest, response: Any) -> None:
        stop = getattr(response, "stop_reason", None)
        if stop == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details is not None else None
            explanation = getattr(details, "explanation", None) if details is not None else None
            msg = f"Claude declined task {request.task!r}"
            if category:
                msg += f" (category: {category})"
            if explanation:
                msg += f": {explanation}"
            raise RefusalError(msg)
        if stop == "max_tokens":
            raise TruncatedOutputError(
                f"task {request.task!r} hit max_tokens before finishing; raise llm.max_tokens or shorten the input"
            )

    @staticmethod
    def _log_fallbacks(request: LLMRequest, response: Any) -> None:
        for block in response.content or []:
            if getattr(block, "type", None) == "fallback":
                src = getattr(getattr(block, "from_", None), "model", "?")
                dst = getattr(getattr(block, "to", None), "model", "?")
                logger.info("task %s: %s declined, served by fallback model %s", request.task, src, dst)


def _usage(raw: Any) -> Usage:
    """Map SDK usage (fields may be missing or None) onto :class:`Usage`."""
    if raw is None:
        return Usage()

    def get(name: str) -> int:
        return int(getattr(raw, name, 0) or 0)

    return Usage(
        input_tokens=get("input_tokens"),
        output_tokens=get("output_tokens"),
        cache_read_input_tokens=get("cache_read_input_tokens"),
        cache_creation_input_tokens=get("cache_creation_input_tokens"),
    )
