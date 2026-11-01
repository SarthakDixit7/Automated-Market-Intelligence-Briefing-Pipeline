"""Settings: config/settings.yaml, overridden by MB_* environment variables.

Relative paths are resolved against the project root (the directory holding
config/settings.yaml, or the current working directory if there is none).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

DEFAULT_SETTINGS_FILE = Path("config") / "settings.yaml"


class LLMSettings(BaseModel):
    # auto -> "anthropic" when ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN /
    # ANTHROPIC_PROFILE is set, otherwise the deterministic "offline" provider.
    provider: Literal["auto", "anthropic", "offline"] = "auto"
    model: str = "claude-opus-5-5"
    judge_model: str = "claude-opus-5-5"
    # Per-task effort (Claude's thinking-depth control). Extraction runs low so
    # repeated runs stay consistent; synthesis and judging run higher.
    effort: dict[str, str] = Field(
        default_factory=lambda: {
            "extract": "low",
            "map_summary": "low",
            "reduce_brief": "high",
            "edit": "medium",
            "fact_check": "medium",
            "judge": "high",
            "agent_plan": "medium",
        }
    )
    max_tokens: int = 16000
    # Server-side refusal fallback (fallbacks="default").
    use_server_fallbacks: bool = True
    # Instructor-style re-ask attempts when output fails validation.
    max_validation_retries: int = 2
    # Honoured only by providers that accept sampling params (the offline
    # provider). Current Claude models reject temperature, so determinism
    # there comes from low effort + schema-constrained output + few-shots.
    temperature: float = 0.0
    timeout_s: float = 600.0


class CacheSettings(BaseModel):
    enabled: bool = True
    path: str = "data/cache/llm_cache.sqlite"
    # Semantic (embedding) hits are only allowed for these tasks, and only when
    # system prompt + output schema match exactly.
    semantic_tasks: list[str] = Field(default_factory=lambda: ["map_summary"])
    semantic_threshold: float = 0.97


class EmbeddingSettings(BaseModel):
    backend: Literal["hashing", "sentence-transformers"] = "hashing"
    dim: int = 1024
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"


class DedupSettings(BaseModel):
    threshold: float = 0.80
    # Characters of body text (after the title) used for the dedup embedding.
    lead_chars: int = 600


class RetrievalSettings(BaseModel):
    index_dir: str = "data/index"
    top_k: int = 6
    min_score: float = 0.25
    extra_brief_dirs: list[str] = Field(default_factory=lambda: ["data/fixtures/prior_briefs"])


class PathsSettings(BaseModel):
    sources_file: str = "config/sources.yaml"
    entities_file: str = "data/fixtures/entities.yaml"
    raw_dir: str = "data/raw"
    store_dir: str = "data/store"
    runs_dir: str = "data/runs"
    review_dir: str = "data/review"
    briefs_dir: str = "data/briefs"
    output_dir: str = "output"
    golden_file: str = "data/golden/golden_set.yaml"
    evals_dir: str = "data/evals"
    agent_dir: str = "data/agent"


class PipelineSettings(BaseModel):
    default_topic: str = "Enterprise AI infrastructure"
    max_articles: int = 80
    max_workers: int = 4
    # Map summaries per reduce call; above this the reduce step is hierarchical.
    reduce_batch_size: int = 20
    min_support: float = 0.35
    fact_check: bool = True
    require_review: bool = True


class EvalSettings(BaseModel):
    min_extraction_recall: float = 0.9
    min_brief_recall: float = 0.75
    min_judge_score: float = 3.5


class Settings(BaseModel):
    llm: LLMSettings = Field(default_factory=LLMSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    embeddings: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    dedup: DedupSettings = Field(default_factory=DedupSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    paths: PathsSettings = Field(default_factory=PathsSettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
    eval: EvalSettings = Field(default_factory=EvalSettings)
    root: Path = Field(default_factory=Path.cwd, exclude=True)

    def path(self, rel: str | Path) -> Path:
        """Resolve a configured path against the project root."""
        p = Path(rel)
        return p if p.is_absolute() else self.root / p

    def resolved_provider(self) -> Literal["anthropic", "offline"]:
        if self.llm.provider != "auto":
            return self.llm.provider
        for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
            if os.environ.get(var):
                return "anthropic"
        return "offline"


_ENV_OVERRIDES: dict[str, tuple[str, str]] = {
    "MB_PROVIDER": ("llm", "provider"),
    "MB_MODEL": ("llm", "model"),
    "MB_JUDGE_MODEL": ("llm", "judge_model"),
    "MB_CACHE_ENABLED": ("cache", "enabled"),
    "MB_EMBEDDINGS": ("embeddings", "backend"),
    "MB_MAX_WORKERS": ("pipeline", "max_workers"),
}


def _deep_merge(base: dict, extra: dict) -> dict:
    out = dict(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def find_root(start: Path | None = None) -> Path:
    here = (start or Path.cwd()).resolve()
    for p in (here, *here.parents):
        if (p / DEFAULT_SETTINGS_FILE).exists():
            return p
    return here


def load_settings(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    root: Path | None = None,
) -> Settings:
    """Load settings from YAML, then MB_* env vars, then explicit overrides.

    ``overrides`` is a nested dict, e.g. ``{"llm": {"provider": "offline"}}``.
    """
    root = root or (Path(path).resolve().parent.parent if path else find_root())
    cfg_file = Path(path) if path else root / DEFAULT_SETTINGS_FILE
    data: dict[str, Any] = {}
    if cfg_file.exists():
        data = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) or {}
    for env, (section, key) in _ENV_OVERRIDES.items():
        if env in os.environ:
            data.setdefault(section, {})[key] = os.environ[env]
    if overrides:
        data = _deep_merge(data, overrides)
    settings = Settings.model_validate(data)
    settings.root = root
    return settings
