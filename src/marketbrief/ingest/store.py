"""Tiny JSONL document store keyed by an id field.

Good enough for thousands of articles: the file is loaded once, kept in
memory, and rewritten atomically (temp file + replace) on every upsert.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Generic, Iterable, TypeVar

from pydantic import BaseModel, ValidationError

from marketbrief.ingest.normalize import as_utc
from marketbrief.schemas import Article

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class JsonlStore(Generic[T]):
    """Pydantic models persisted one-per-line, unique by ``key``."""

    def __init__(self, path: str | Path, model: type[T], key: str = "id") -> None:
        self.path = Path(path)
        self.model = model
        self.key = key
        self._lock = threading.Lock()
        self._items: dict[str, T] | None = None

    # -- reading ----------------------------------------------------------- #

    def _load(self) -> dict[str, T]:
        if self._items is not None:
            return self._items
        items: dict[str, T] = {}
        if self.path.exists():
            with self.path.open(encoding="utf-8") as fh:
                for lineno, line in enumerate(fh, start=1):
                    if not line.strip():
                        continue
                    try:
                        item = self.model.model_validate_json(line)
                    except ValidationError as exc:
                        log.warning("%s:%d: skipping invalid record: %s", self.path.name, lineno, exc)
                        continue
                    items[str(getattr(item, self.key))] = item
        self._items = items
        return items

    def all(self) -> list[T]:
        with self._lock:
            return list(self._load().values())

    def get(self, key: str) -> T | None:
        with self._lock:
            return self._load().get(key)

    def __len__(self) -> int:
        with self._lock:
            return len(self._load())

    def __contains__(self, key: object) -> bool:
        with self._lock:
            return key in self._load()

    # -- writing ----------------------------------------------------------- #

    def upsert(self, items: Iterable[T]) -> int:
        """Insert or replace items by key; returns how many keys were new."""
        with self._lock:
            current = self._load()
            new = 0
            for item in items:
                k = str(getattr(item, self.key))
                if k not in current:
                    new += 1
                current[k] = item
            self._write(current.values())
            return new

    def _write(self, items: Iterable[T]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        with tmp.open("w", encoding="utf-8", newline="\n") as fh:
            for item in items:
                fh.write(item.model_dump_json() + "\n")
        os.replace(tmp, self.path)


class ArticleStore(JsonlStore[Article]):
    """All articles ever ingested, keyed by article id."""

    def __init__(self, path: str | Path) -> None:
        super().__init__(path, Article, key="id")

    def between(self, start: datetime | None, end: datetime | None) -> list[Article]:
        """Articles with ``start <= published_at <= end`` (fetched_at when
        there is no publish date), oldest first. ``None`` bounds are open."""
        lo, hi = as_utc(start), as_utc(end)
        out: list[tuple[datetime, Article]] = []
        for art in self.all():
            ts = article_time(art)
            if (lo is None or ts >= lo) and (hi is None or ts <= hi):
                out.append((ts, art))
        out.sort(key=lambda pair: pair[0])
        return [art for _, art in out]


def article_time(article: Article) -> datetime:
    """Timestamp used for ordering/windowing: published_at, else fetched_at (UTC)."""
    return as_utc(article.published_at or article.fetched_at)  # type: ignore[return-value]
