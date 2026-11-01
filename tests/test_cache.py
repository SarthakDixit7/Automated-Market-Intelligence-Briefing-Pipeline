"""Prompt cache: exact + semantic hits at zero cost, persistence, safety rules."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from marketbrief.embeddings import HashingEmbedder
from marketbrief.llm.base import LLMClient, LLMRequest, LLMResult, Usage
from marketbrief.llm.cache import CachedLLMClient, CacheStats, PromptCache
from marketbrief.schemas import ArticleSummary, EventType

SYSTEM = "You are a senior market analyst. Summarise the article neutrally."

ARTICLE = (
    "<article id=\"a_123456789abc\" source=\"Cloud Ledger\" published=\"2026-10-08\">"
    "<title>Meridian Cloud to acquire Vireo Analytics for $1.2 billion</title><text>"
    "Meridian Cloud has agreed to acquire Vireo Analytics for $1.2 billion in cash, the hyperscale cloud "
    "provider said on Thursday, adding a fast-growing observability business to its portfolio of developer "
    "tools. Vireo, founded in 2019, sells software that tracks the performance, cost and reliability of "
    "applications and AI workloads running across multiple clouds. It says it has about 2,300 paying "
    "customers. Meridian said Vireo would continue to support other clouds after the deal closes, which it "
    "expects in the first half of 2027. Vireo's roughly 900 employees will join Meridian's developer "
    "platform division.</text></article>"
)
# Same prompt with trivial whitespace/punctuation edits -> semantically identical.
ARTICLE_NEAR = ARTICLE.replace("on Thursday,", "on Thursday ;").replace("  ", " ") + " "
OTHER_ARTICLE = (
    "<article id=\"a_999999999999\"><title>Kestrel Networks to cut 9% of workforce</title><text>"
    "Kestrel Networks will cut 9% of its workforce after hyperscale customers delayed switch orders."
    "</text></article>"
)


class FakeLLM(LLMClient):
    """Counts calls; returns an ArticleSummary (or text) with non-zero usage."""

    provider = "fake"
    model = "fake-model-1"

    def __init__(self) -> None:
        self.calls = 0
        self._lock = threading.Lock()

    def complete(self, request: LLMRequest) -> LLMResult:
        with self._lock:
            self.calls += 1
            n = self.calls
        if request.response_model is None:
            return LLMResult(text=f"text answer {n}", usage=Usage(input_tokens=100, output_tokens=10),
                             model=self.model, provider=self.provider)
        parsed = ArticleSummary(
            article_id="a_123456789abc",
            summary=f"Meridian Cloud agreed to buy Vireo Analytics (call {n}).",
            key_points=["$1.2 billion cash deal"],
            why_it_matters="Consolidation in observability.",
            companies=["Meridian Cloud", "Vireo Analytics"],
            event_types=[EventType.M_AND_A],
        )
        return LLMResult(text=parsed.model_dump_json(), parsed=parsed,
                         usage=Usage(input_tokens=1200, output_tokens=150),
                         model=self.model, provider=self.provider, stop_reason="end_turn")


def req(prompt: str = ARTICLE, *, task: str = "map_summary", system: str = SYSTEM,
        model: type | None = ArticleSummary, effort: str | None = "low") -> LLMRequest:
    return LLMRequest(task=task, system=system, prompt=prompt, response_model=model, effort=effort,
                      context={"ignored": "not part of the key"})


@pytest.fixture
def cache(tmp_path: Path):
    c = PromptCache(tmp_path / "cache" / "llm.sqlite", embedder=HashingEmbedder(1024),
                    semantic_tasks=["map_summary"], semantic_threshold=0.97)
    yield c
    c.close()


def test_near_prompt_is_above_semantic_threshold() -> None:
    emb = HashingEmbedder(1024)
    near, other = emb.embed([ARTICLE_NEAR, OTHER_ARTICLE]) @ emb.embed([ARTICLE])[0]
    assert ARTICLE_NEAR != ARTICLE
    assert near >= 0.97 > other


def test_exact_hit_returns_parsed_model_with_zero_usage(cache: PromptCache) -> None:
    llm = FakeLLM()
    client = CachedLLMClient(llm, cache)
    first = client.complete(req())
    assert not first.cached and first.usage.input_tokens == 1200

    second = client.complete(req())
    assert llm.calls == 1
    assert second.cached and second.cache_kind == "exact"
    assert second.usage == Usage()
    assert isinstance(second.parsed, ArticleSummary)
    assert second.parsed == first.parsed
    assert second.provider == "fake" and second.model == "fake-model-1"
    assert cache.stats == CacheStats(hits_exact=1, misses=1, saved_input_tokens=1200, saved_output_tokens=150)


def test_semantic_hit_for_near_identical_prompt(cache: PromptCache) -> None:
    llm = FakeLLM()
    client = CachedLLMClient(llm, cache)
    original = client.complete(req(ARTICLE))
    near = client.complete(req(ARTICLE_NEAR))
    assert llm.calls == 1
    assert near.cached and near.cache_kind == "semantic"
    assert near.usage == Usage()
    assert near.parsed == original.parsed
    assert cache.stats.hits_semantic == 1

    # A genuinely different article is a miss.
    client.complete(req(OTHER_ARTICLE))
    assert llm.calls == 2


@pytest.mark.parametrize(
    "variant",
    [
        {"task": "extract"},                          # task not in semantic_tasks
        {"system": SYSTEM + " Be terse."},            # different system prompt
        {"model": None},                              # different output schema
        {"effort": "high"},                           # different effort
    ],
)
def test_semantic_hits_require_same_task_system_schema_effort(cache: PromptCache, variant: dict) -> None:
    llm = FakeLLM()
    client = CachedLLMClient(llm, cache)
    client.complete(req(ARTICLE))
    client.complete(req(ARTICLE_NEAR, **variant))
    assert llm.calls == 2
    assert cache.stats.hits_semantic == 0


def test_lookup_depends_on_model_and_effort(cache: PromptCache) -> None:
    llm = FakeLLM()
    cache.put(req(), "model-a", llm.complete(req()))
    assert cache.get(req(), "model-a") is not None
    assert cache.get(req(), "model-b") is None
    assert cache.get(req(effort="high"), "model-a") is None


def test_reask_with_feedback_bypasses_semantic_match(cache: PromptCache) -> None:
    llm = FakeLLM()
    client = CachedLLMClient(llm, cache)
    first = req()
    client.complete(first)
    reask = first.with_feedback("- article_id does not match")
    result = client.complete(reask)
    assert not result.cached and llm.calls == 2


def test_cache_survives_reopen(tmp_path: Path) -> None:
    path = tmp_path / "persist.sqlite"
    emb = HashingEmbedder(1024)
    with PromptCache(path, embedder=emb, semantic_tasks=["map_summary"]) as c1:
        CachedLLMClient(FakeLLM(), c1).complete(req())
        CachedLLMClient(FakeLLM(), c1).complete(req(task="extract"))
        assert len(c1) == 2

    with PromptCache(path, embedder=emb, semantic_tasks=["map_summary"]) as c2:
        llm = FakeLLM()
        client = CachedLLMClient(llm, c2)
        exact = client.complete(req())
        semantic = client.complete(req(ARTICLE_NEAR))
        assert llm.calls == 0
        assert exact.cache_kind == "exact" and semantic.cache_kind == "semantic"
        assert isinstance(semantic.parsed, ArticleSummary)


def test_old_cache_file_without_effort_column_is_migrated(tmp_path: Path) -> None:
    import sqlite3

    path = tmp_path / "old.sqlite"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE llm_cache (key TEXT PRIMARY KEY, task TEXT NOT NULL, model TEXT NOT NULL, "
        "system_hash TEXT NOT NULL, schema_fp TEXT NOT NULL, prompt TEXT NOT NULL, embedding BLOB, "
        "result_json TEXT NOT NULL, created_at TEXT NOT NULL)"
    )
    conn.execute("CREATE INDEX idx_llm_cache_semantic ON llm_cache (task, model, system_hash, schema_fp)")
    conn.commit()
    conn.close()
    with PromptCache(path, embedder=HashingEmbedder(1024), semantic_tasks=["map_summary"]) as cache:
        llm = FakeLLM()
        client = CachedLLMClient(llm, cache)
        client.complete(req())
        assert client.complete(req(ARTICLE_NEAR)).cache_kind == "semantic"
        assert llm.calls == 1


def test_text_results_are_cached(cache: PromptCache) -> None:
    llm = FakeLLM()
    client = CachedLLMClient(llm, cache)
    first = client.complete(req(task="judge", model=None))
    second = client.complete(req(task="judge", model=None))
    assert llm.calls == 1 and second.cached and second.text == first.text and second.parsed is None


def test_structured_result_without_parsed_is_not_cached(cache: PromptCache) -> None:
    broken = LLMResult(text="{}", parsed=None, usage=Usage(input_tokens=5))
    assert cache.put(req(), "m", broken) is False
    assert len(cache) == 0
    assert cache.get(req(), "m") is None
    assert cache.stats.misses == 1


def test_stale_schema_entry_is_discarded(cache: PromptCache) -> None:
    """A row whose stored JSON no longer validates is dropped and treated as a miss."""
    request = req(task="extract")
    cache.put(request, "m", FakeLLM().complete(request))
    with cache._conn:  # simulate a row written by an older schema version
        cache._conn.execute(
            "UPDATE llm_cache SET result_json = ?",
            ('{"text": "x", "parsed": "{\\"article_id\\": 1}", "usage": {}}',),
        )
    assert len(cache) == 1
    assert cache.get(req(task="extract"), "m") is None
    assert len(cache) == 0


def test_clear_and_len(cache: PromptCache) -> None:
    client = CachedLLMClient(FakeLLM(), cache)
    client.complete(req())
    client.complete(req(OTHER_ARTICLE))
    assert len(cache) == 2
    assert cache.clear() == 2
    assert len(cache) == 0


def test_cached_client_mirrors_inner_and_is_thread_safe(cache: PromptCache) -> None:
    llm = FakeLLM()
    client = CachedLLMClient(llm, cache)
    assert isinstance(client, LLMClient)
    assert client.provider == "fake" and client.model == "fake-model-1"

    client.complete(req())  # warm
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: client.complete(req()), range(32)))
    assert all(r.cached for r in results)
    assert llm.calls == 1
    assert cache.stats.hits_exact == 32
