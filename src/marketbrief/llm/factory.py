"""Build the configured LLM client stack.

    UsageTracker  (counts calls, tokens, prompt-cache hits)
      -> CachedLLMClient  (exact + semantic prompt cache, optional)
        -> AnthropicLLMClient | OfflineLLMClient
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from marketbrief.config import Settings
from marketbrief.llm.base import LLMClient, LLMRequest, LLMResult, Usage
from marketbrief.llm.offline import DEFAULT_OFFLINE_MODEL

logger = logging.getLogger(__name__)


class UsageTracker:
    """Thread-safe :class:`LLMClient` wrapper that meters every call."""

    def __init__(self, inner: LLMClient) -> None:
        self.inner = inner
        self._lock = threading.Lock()
        self.calls = 0
        self.errors = 0
        self.usage = Usage()
        self.cache_hits = 0
        self.cache_misses = 0
        self.by_task: dict[str, int] = {}

    @property
    def provider(self) -> str:
        return self.inner.provider

    @property
    def model(self) -> str:
        return self.inner.model

    def complete(self, request: LLMRequest) -> LLMResult:
        try:
            result = self.inner.complete(request)
        except Exception:
            with self._lock:
                self.calls += 1
                self.errors += 1
                self.by_task[request.task] = self.by_task.get(request.task, 0) + 1
            raise
        with self._lock:
            self.calls += 1
            self.usage = self.usage + result.usage
            if result.cached:
                self.cache_hits += 1
            else:
                self.cache_misses += 1
            self.by_task[request.task] = self.by_task.get(request.task, 0) + 1
        return result

    def snapshot(self) -> dict[str, Any]:
        """Counters as plain data. Flat keys mirror ``RunReport`` field names."""
        with self._lock:
            usage = self.usage.model_dump()
            return {
                "provider": self.provider,
                "model": self.model,
                "calls": self.calls,
                "llm_calls": self.calls,
                "errors": self.errors,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "prompt_cache_hits": self.cache_hits,
                "prompt_cache_misses": self.cache_misses,
                "usage": usage,
                **usage,
                "by_task": dict(self.by_task),
            }

    def reset(self) -> None:
        with self._lock:
            self.calls = self.errors = self.cache_hits = self.cache_misses = 0
            self.usage = Usage()
            self.by_task = {}


def build_llm(
    settings: Settings,
    *,
    provider: str | None = None,
    model: str | None = None,
    use_cache: bool | None = None,
    track: bool = True,
) -> LLMClient:
    """Create the client for ``provider`` (default: ``settings.resolved_provider()``).

    ``model`` overrides ``settings.llm.model`` for Claude; the offline provider
    keeps its own model name unless an ``offline-*`` name is given.
    """
    name = (provider or settings.resolved_provider()).lower()
    if name == "auto":
        name = settings.resolved_provider()

    client: LLMClient
    if name == "anthropic":
        from marketbrief.llm.anthropic_client import AnthropicLLMClient

        client = AnthropicLLMClient(
            model or settings.llm.model,
            max_tokens=settings.llm.max_tokens,
            use_server_fallbacks=settings.llm.use_server_fallbacks,
            timeout_s=settings.llm.timeout_s,
        )
    elif name == "offline":
        from marketbrief.llm.offline import OfflineLLMClient

        client = OfflineLLMClient(
            entities_path=settings.path(settings.paths.entities_file),
            temperature=settings.llm.temperature,
            model=model if model and model.startswith("offline") else DEFAULT_OFFLINE_MODEL,
        )
    else:
        raise ValueError(f"unknown LLM provider {name!r}; expected 'anthropic' or 'offline'")

    cached = settings.cache.enabled if use_cache is None else use_cache
    if cached:
        # Imported here: the cache module imports llm.base, and embeddings pull in numpy.
        from marketbrief.embeddings import get_embedder
        from marketbrief.llm.cache import CachedLLMClient, PromptCache

        cache = PromptCache(
            settings.path(settings.cache.path),
            embedder=get_embedder(settings),
            semantic_tasks=settings.cache.semantic_tasks,
            semantic_threshold=settings.cache.semantic_threshold,
        )
        client = CachedLLMClient(client, cache)

    logger.debug("built LLM client provider=%s model=%s cache=%s", client.provider, client.model, cached)
    return UsageTracker(client) if track else client
