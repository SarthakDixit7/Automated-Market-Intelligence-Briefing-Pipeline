"""Ingestion stage: configured sources -> normalised, windowed, stored articles."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import httpx

from marketbrief.config import Settings
from marketbrief.ingest.normalize import (
    article_id,
    as_utc,
    canonicalize_url,
    clean_text,
    content_hash,
    make_article,
)
from marketbrief.ingest.sources import _new_http_client, fetch_source, load_sources
from marketbrief.ingest.store import ArticleStore, JsonlStore, article_time
from marketbrief.schemas import Article, SourceConfig

log = logging.getLogger(__name__)

__all__ = [
    "IngestResult",
    "ingest",
    "select_sources",
    "load_sources",
    "fetch_source",
    "make_article",
    "canonicalize_url",
    "clean_text",
    "article_id",
    "content_hash",
    "ArticleStore",
    "JsonlStore",
]


@dataclass
class IngestResult:
    articles: list[Article]
    raw_path: Path
    per_source: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def select_sources(
    sources: list[SourceConfig], source_ids: list[str] | None, errors: list[str] | None = None
) -> list[SourceConfig]:
    """Enabled sources, or exactly the named ones (enabled or not) when given."""
    if not source_ids:
        return [s for s in sources if s.enabled]
    by_id = {s.id: s for s in sources}
    chosen: list[SourceConfig] = []
    for sid in source_ids:
        if sid in by_id:
            chosen.append(by_id[sid])
        else:
            msg = f"unknown source id {sid!r}"
            log.warning(msg)
            if errors is not None:
                errors.append(msg)
    return chosen


def ingest(
    settings: Settings,
    *,
    source_ids: list[str] | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    run_id: str = "manual",
) -> IngestResult:
    """Fetch, normalise and window articles from the configured sources.

    Writes the run's structured JSON snapshot to ``<raw_dir>/<run_id>.json``,
    upserts the articles into the article store, and returns them oldest
    first. When more than ``pipeline.max_articles`` match, the most recent
    ones are kept.
    """
    errors: list[str] = []
    sources = select_sources(load_sources(settings.path(settings.paths.sources_file)), source_ids, errors)
    since_utc, until_utc = as_utc(since), as_utc(until)

    per_source: dict[str, int] = {}
    by_id: dict[str, Article] = {}
    needs_http = any(s.kind == "rss" for s in sources)
    http: httpx.Client | None = _new_http_client() if needs_http else None
    try:
        for src in sources:
            raw_items = fetch_source(src, settings, http=http, since=since_utc, errors=errors)
            count = 0
            for raw in raw_items:
                try:
                    art = make_article(raw, src)
                except ValueError as exc:
                    errors.append(f"{src.id}: could not normalise {raw.title!r}: {exc}")
                    continue
                ts = article_time(art)
                if (since_utc and ts < since_utc) or (until_utc and ts > until_utc):
                    continue
                if art.id in by_id:  # same canonical URL seen from another source
                    continue
                by_id[art.id] = art
                count += 1
            per_source[src.id] = count
            log.info("ingested %d articles from %s", count, src.id)
    finally:
        if http is not None:
            http.close()

    articles = sorted(by_id.values(), key=article_time)
    cap = settings.pipeline.max_articles
    if cap and len(articles) > cap:
        articles = articles[-cap:]

    store = ArticleStore(settings.path(settings.paths.store_dir) / "articles.jsonl")
    # Keep the first-seen fetch time for articles that are already stored.
    articles = [
        a.model_copy(update={"fetched_at": prev.fetched_at}) if (prev := store.get(a.id)) else a
        for a in articles
    ]

    raw_dir = settings.path(settings.paths.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{run_id}.json"
    raw_path.write_text(
        json.dumps([a.model_dump(mode="json") for a in articles], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    store.upsert(articles)
    return IngestResult(articles=articles, raw_path=raw_path, per_source=per_source, errors=errors)
