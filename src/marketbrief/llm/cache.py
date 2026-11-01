"""GPTCache-style prompt cache: exact-hash hits plus embedding (semantic) hits.

Every LLM call is keyed by ``LLMRequest.cache_key(model)`` (model, task,
system prompt, user prompt, output schema, effort). Repeated stories and
unchanged prompts are answered from SQLite at zero token cost. For tasks
listed in ``semantic_tasks`` a near-identical prompt (cosine >= threshold)
also hits, but only when model, system prompt, output schema and effort are
identical. Re-asks (prompts carrying validation feedback) never hit
semantically.

This is a client-side cache; it complements Anthropic's server-side prompt
caching (``cache_control`` on the stable system prompt).
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from pydantic import ValidationError

from marketbrief.embeddings import Embedder
from marketbrief.llm.base import LLMClient, LLMRequest, LLMResult, Usage

log = logging.getLogger(__name__)

# Re-asks carry validation feedback and must get a fresh answer: serving the
# near-identical cached (rejected) output again would loop until retries run out.
_FEEDBACK_MARKER = "<validation_feedback>"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_cache (
    key          TEXT PRIMARY KEY,
    task         TEXT NOT NULL,
    model        TEXT NOT NULL,
    system_hash  TEXT NOT NULL,
    schema_fp    TEXT NOT NULL,
    effort       TEXT NOT NULL DEFAULT '',
    prompt       TEXT NOT NULL,
    embedding    BLOB,
    result_json  TEXT NOT NULL,
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_llm_cache_semantic
    ON llm_cache (task, model, system_hash, schema_fp, effort);
"""


@dataclass
class CacheStats:
    hits_exact: int = 0
    hits_semantic: int = 0
    misses: int = 0
    saved_input_tokens: int = 0
    saved_output_tokens: int = 0

    @property
    def hits(self) -> int:
        return self.hits_exact + self.hits_semantic

    def as_dict(self) -> dict[str, int]:
        return {**asdict(self), "hits": self.hits}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class PromptCache:
    """SQLite-backed prompt cache, safe to share between threads."""

    def __init__(
        self,
        path: str | Path,
        *,
        embedder: Embedder | None = None,
        semantic_tasks: Sequence[str] = (),
        semantic_threshold: float = 0.97,
    ) -> None:
        self.path = Path(path)
        self.embedder = embedder
        self.semantic_tasks = frozenset(semantic_tasks)
        self.semantic_threshold = float(semantic_threshold)
        self.stats = CacheStats()
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        with self._lock, self._conn:
            self._migrate()
            self._conn.executescript(_SCHEMA)

    def _migrate(self) -> None:
        """Add columns introduced after a cache file was first created."""
        cols = {row[1] for row in self._conn.execute("PRAGMA table_info(llm_cache)")}
        if cols and "effort" not in cols:
            self._conn.execute("ALTER TABLE llm_cache ADD COLUMN effort TEXT NOT NULL DEFAULT ''")
            self._conn.execute("DROP INDEX IF EXISTS idx_llm_cache_semantic")

    # -- lookup ------------------------------------------------------------ #

    def get(self, request: LLMRequest, model: str) -> LLMResult | None:
        """Cached result for ``request`` (exact, then semantic), or ``None``."""
        key = request.cache_key(model)
        with self._lock:
            row = self._conn.execute("SELECT result_json FROM llm_cache WHERE key = ?", (key,)).fetchone()
        if row is not None:
            result = self._rebuild(request, row[0], "exact")
            if result is not None:
                return result
            self._delete(key)

        if self._semantic_enabled(request):
            match = self._semantic_lookup(request, model)
            if match is not None:
                match_key, result_json, score = match
                result = self._rebuild(request, result_json, "semantic")
                if result is not None:
                    log.debug("semantic cache hit (%.4f) for task %s", score, request.task)
                    return result
                self._delete(match_key)

        with self._lock:
            self.stats.misses += 1
        return None

    def _semantic_enabled(self, request: LLMRequest) -> bool:
        return (
            self.embedder is not None
            and request.task in self.semantic_tasks
            and _FEEDBACK_MARKER not in request.prompt
        )

    def _semantic_lookup(self, request: LLMRequest, model: str) -> tuple[str, str, float] | None:
        assert self.embedder is not None
        with self._lock:
            rows = self._conn.execute(
                "SELECT key, embedding, result_json FROM llm_cache "
                "WHERE task = ? AND model = ? AND system_hash = ? AND schema_fp = ? AND effort = ? "
                "AND embedding IS NOT NULL",
                (request.task, model, _sha256(request.system), request.schema_fingerprint(), request.effort or ""),
            ).fetchall()
        dim = self.embedder.dim
        rows = [r for r in rows if len(r[1]) == dim * 4]  # skip rows from another embedder
        if not rows:
            return None
        query = self.embedder.embed([request.prompt])[0]
        if not np.any(query):
            return None
        matrix = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
        scores = matrix @ query
        best = int(np.argmax(scores))
        score = float(scores[best])
        if score < self.semantic_threshold:
            return None
        return rows[best][0], rows[best][2], score

    def _rebuild(self, request: LLMRequest, result_json: str, kind: str) -> LLMResult | None:
        """LLMResult from a stored row, with zero usage; None if unusable."""
        try:
            stored: dict[str, Any] = json.loads(result_json)
            parsed = None
            if request.response_model is not None:
                if stored.get("parsed") is None:
                    return None
                parsed = request.response_model.model_validate_json(stored["parsed"])
        except (ValueError, ValidationError) as exc:  # corrupt or stale schema
            log.warning("discarding unusable cache entry: %s", exc)
            return None

        original = Usage.model_validate(stored.get("usage") or {})
        with self._lock:
            if kind == "exact":
                self.stats.hits_exact += 1
            else:
                self.stats.hits_semantic += 1
            self.stats.saved_input_tokens += original.input_tokens
            self.stats.saved_output_tokens += original.output_tokens
        return LLMResult(
            text=stored.get("text", ""),
            parsed=parsed,
            usage=Usage(),
            model=stored.get("model", ""),
            provider=stored.get("provider", ""),
            cached=True,
            cache_kind=kind,
            stop_reason=stored.get("stop_reason"),
            latency_s=0.0,
        )

    # -- storage ----------------------------------------------------------- #

    def put(self, request: LLMRequest, model: str, result: LLMResult) -> bool:
        """Store ``result``; returns False when it is not cacheable."""
        if request.response_model is not None and result.parsed is None:
            return False
        parsed_json = result.parsed.model_dump_json() if hasattr(result.parsed, "model_dump_json") else None
        payload = json.dumps(
            {
                "text": result.text,
                "parsed": parsed_json,
                "usage": result.usage.model_dump(),
                "model": result.model,
                "provider": result.provider,
                "stop_reason": result.stop_reason,
            }
        )
        embedding: bytes | None = None
        if self.embedder is not None and request.task in self.semantic_tasks:
            vec = self.embedder.embed([request.prompt])[0]
            embedding = np.asarray(vec, dtype=np.float32).tobytes()
        row = (
            request.cache_key(model),
            request.task,
            model,
            _sha256(request.system),
            request.schema_fingerprint(),
            request.effort or "",
            request.prompt,
            embedding,
            payload,
            datetime.now(timezone.utc).isoformat(),
        )
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO llm_cache (key, task, model, system_hash, schema_fp, effort, prompt, "
                "embedding, result_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                row,
            )
        return True

    def _delete(self, key: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM llm_cache WHERE key = ?", (key,))

    def clear(self) -> int:
        """Delete every entry; returns how many were removed."""
        with self._lock, self._conn:
            removed = self._conn.execute("DELETE FROM llm_cache").rowcount
        return int(removed)

    def __len__(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0])

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> PromptCache:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class CachedLLMClient(LLMClient):
    """Wraps any :class:`LLMClient`; serves repeated requests from the cache."""

    def __init__(self, inner: LLMClient, cache: PromptCache) -> None:
        self.inner = inner
        self.cache = cache

    @property
    def provider(self) -> str:
        return self.inner.provider

    @property
    def model(self) -> str:
        return self.inner.model

    def complete(self, request: LLMRequest) -> LLMResult:
        try:
            hit = self.cache.get(request, self.model)
        except sqlite3.Error as exc:  # a broken cache degrades to a pass-through
            log.warning("prompt cache read failed: %s", exc)
            hit = None
        if hit is not None:
            return hit
        result = self.inner.complete(request)
        try:
            self.cache.put(request, self.model, result)
        except sqlite3.Error as exc:  # a cache failure must never fail the call
            log.warning("prompt cache write failed: %s", exc)
        return result
