"""Source adapters: RSS/Atom feeds, JSON directories and JSONL files -> RawItem.

Every adapter is defensive: a broken feed, a missing file or a malformed record
logs a warning and is skipped, so one bad source never kills a pipeline run.
"""

from __future__ import annotations

import calendar
import json
import logging
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable

import httpx
import yaml

from marketbrief.config import Settings
from marketbrief.ingest.normalize import as_utc, clean_text
from marketbrief.schemas import RawItem, SourceConfig

log = logging.getLogger(__name__)

USER_AGENT = "marketbrief/0.1"
HTTP_TIMEOUT_S = 20.0


def load_sources(path: str | Path) -> list[SourceConfig]:
    """Read ``sources:`` from a YAML file (missing file -> empty list)."""
    path = Path(path)
    if not path.exists():
        log.warning("sources file not found: %s", path)
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = data.get("sources", []) if isinstance(data, dict) else data
    return [SourceConfig.model_validate(entry) for entry in entries or []]


def fetch_source(
    src: SourceConfig,
    settings: Settings,
    *,
    http: httpx.Client | None = None,
    since: datetime | None = None,
    errors: list[str] | None = None,
) -> list[RawItem]:
    """Fetch raw items from one source.

    Items published before ``since`` are dropped (items without a date are
    kept); at most ``src.max_items`` are returned. Failures are logged and,
    when ``errors`` is given, appended to it; they never raise.
    """
    try:
        if src.kind == "json_dir":
            items = _fetch_json_dir(src, settings, errors)
        elif src.kind == "jsonl":
            items = _fetch_jsonl(src, settings, errors)
        elif src.kind == "rss":
            items = _fetch_rss(src, http, errors)
        else:  # pragma: no cover - SourceConfig.kind is a Literal
            raise ValueError(f"unsupported source kind {src.kind!r}")
    except Exception as exc:  # noqa: BLE001 - a source must never crash the run
        _report(errors, f"{src.id}: fetch failed: {exc}")
        return []

    since_utc = as_utc(since)
    if since_utc is not None:
        items = [it for it in items if it.published_at is None or as_utc(it.published_at) >= since_utc]
    return items[: max(0, src.max_items)]


# --------------------------------------------------------------------------- #
# Local files
# --------------------------------------------------------------------------- #


def _fetch_json_dir(src: SourceConfig, settings: Settings, errors: list[str] | None) -> list[RawItem]:
    if not src.path:
        raise ValueError("json_dir source needs a 'path'")
    folder = settings.path(src.path)
    if not folder.is_dir():
        raise FileNotFoundError(f"directory not found: {folder}")
    items: list[RawItem] = []
    for file in sorted(folder.glob("*.json")):
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            _report(errors, f"{src.id}: skipping {file.name}: {exc}")
            continue
        records = data if isinstance(data, list) else [data]
        items.extend(_records_to_items(src, records, f"{file.name}", errors))
    return items


def _fetch_jsonl(src: SourceConfig, settings: Settings, errors: list[str] | None) -> list[RawItem]:
    if not src.path:
        raise ValueError("jsonl source needs a 'path'")
    file = settings.path(src.path)
    records: list[Any] = []
    with file.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                _report(errors, f"{src.id}: skipping {file.name}:{lineno}: {exc}")
    return _records_to_items(src, records, file.name, errors)


def _records_to_items(
    src: SourceConfig, records: Iterable[Any], origin: str, errors: list[str] | None
) -> list[RawItem]:
    items: list[RawItem] = []
    for i, rec in enumerate(records):
        if not isinstance(rec, dict) or not rec.get("title"):
            _report(errors, f"{src.id}: skipping record {i} in {origin}: not an object with a title")
            continue
        try:
            items.append(
                RawItem(
                    source_id=src.id,
                    source_name=str(rec.get("source") or src.name),
                    title=str(rec["title"]),
                    url=rec.get("url") or None,
                    published_at=_parse_iso(rec.get("published_at")),
                    text=str(rec.get("text") or ""),
                    author=rec.get("author") or None,
                    raw=rec,
                )
            )
        except ValueError as exc:
            _report(errors, f"{src.id}: skipping record {i} in {origin}: {exc}")
    return items


def _parse_iso(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return as_utc(value)
    text = str(value).strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    return as_utc(datetime.fromisoformat(text))


# --------------------------------------------------------------------------- #
# RSS / Atom
# --------------------------------------------------------------------------- #


def _fetch_rss(src: SourceConfig, http: httpx.Client | None, errors: list[str] | None) -> list[RawItem]:
    import feedparser  # imported lazily: only needed for live feeds

    if not src.url:
        raise ValueError("rss source needs a 'url'")
    own_client = http is None
    client = http or _new_http_client()
    try:
        try:
            resp = client.get(src.url)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            _report(errors, f"{src.id}: could not fetch feed {src.url}: {exc}")
            return []
        feed = feedparser.parse(resp.content)
        if feed.get("bozo") and not feed.get("entries"):
            _report(errors, f"{src.id}: unparseable feed: {feed.get('bozo_exception')}")
            return []

        items: list[RawItem] = []
        for entry in feed.get("entries", [])[: max(0, src.max_items)]:
            title = entry.get("title")
            if not title:
                continue
            link = entry.get("link")
            text = _entry_text(entry)
            if src.fetch_full_text and link:
                text = _fetch_full_text(client, link, errors) or text
            items.append(
                RawItem(
                    source_id=src.id,
                    source_name=src.name,
                    title=title,
                    url=link,
                    published_at=_struct_to_utc(entry.get("published_parsed") or entry.get("updated_parsed")),
                    text=text,
                    author=entry.get("author"),
                    raw={"id": entry.get("id"), "tags": [t.get("term") for t in entry.get("tags", [])]},
                )
            )
        return items
    finally:
        if own_client:
            client.close()


def _new_http_client() -> httpx.Client:
    return httpx.Client(
        timeout=HTTP_TIMEOUT_S,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    )


def _entry_text(entry: Any) -> str:
    content = entry.get("content") or []
    if content and content[0].get("value"):
        return content[0]["value"]
    return entry.get("summary", "") or ""


def _struct_to_utc(value: Any) -> datetime | None:
    if not value:
        return None
    return datetime.fromtimestamp(calendar.timegm(value), tz=timezone.utc)


class _ParagraphExtractor(HTMLParser):
    """Collects the inner text of every ``<p>`` element."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.paragraphs: list[str] = []
        self._depth = 0
        self._buf: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "p":
            if self._depth == 0:
                self._buf = []
            self._depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "p" and self._depth:
            self._depth -= 1
            if self._depth == 0:
                self.paragraphs.append("".join(self._buf))

    def handle_data(self, data: str) -> None:
        if self._depth:
            self._buf.append(data)


def _fetch_full_text(client: httpx.Client, url: str, errors: list[str] | None) -> str:
    """Download ``url`` and return its ``<p>`` text, or "" on failure."""
    try:
        resp = client.get(url)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        _report(errors, f"full-text fetch failed for {url}: {exc}")
        return ""
    parser = _ParagraphExtractor()
    parser.feed(resp.text)
    parser.close()
    paragraphs = (clean_text(p) for p in parser.paragraphs)
    return "\n\n".join(p for p in paragraphs if p)


def _report(errors: list[str] | None, message: str) -> None:
    log.warning(message)
    if errors is not None:
        errors.append(message)
