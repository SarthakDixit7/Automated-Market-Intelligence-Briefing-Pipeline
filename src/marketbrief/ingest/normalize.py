"""Normalisation: raw source items -> clean, stably identified :class:`Article`.

Article ids are derived from the canonical URL (or source + title when there
is no URL), so re-ingesting the same story always yields the same id.
"""

from __future__ import annotations

import hashlib
import html
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from marketbrief.schemas import Article, RawItem, SourceConfig

# Query parameters that only track the click and never change the content.
_TRACKING_PARAMS = {"fbclid", "gclid", "ref", "ref_src", "mc_cid", "mc_eid", "igshid"}

# Tags whose boundaries become paragraph breaks.
_BLOCK_TAGS = {
    "p", "div", "br", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6",
    "blockquote", "section", "article", "header", "footer", "tr", "table", "pre", "hr",
}
# Tags whose content is never readable text.
_SKIP_TAGS = {"script", "style", "noscript", "template", "svg", "head"}

_PARA_MARK = "\u0000PARA\u0000"
_PARA_SPLIT = re.compile(r"(?:\u0000PARA\u0000|\n[ \t\r\f\v]*\n)")
_WS = re.compile(r"\s+")


def canonicalize_url(url: str | None) -> str:
    """Canonical form of ``url`` for identity checks ("" for empty input).

    Lowercases scheme and host, drops the fragment, tracking parameters
    (``utm_*``, ``fbclid``, ``gclid``, ``ref`` ...), default ports and the
    trailing slash. Remaining query parameters are sorted.
    """
    if not url or not url.strip():
        return ""
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower()
    port = parts.port if parts.port and (scheme, parts.port) not in {("http", 80), ("https", 443)} else None
    netloc = f"{host}:{port}" if port else host
    if parts.username:
        netloc = f"{parts.username}@{netloc}"
    query = sorted(
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not k.lower().startswith("utm_") and k.lower() not in _TRACKING_PARAMS
    )
    path = parts.path.rstrip("/")
    return urlunsplit((scheme, netloc, path, urlencode(query), ""))


class _TextExtractor(HTMLParser):
    """Collects visible text, marking block-level boundaries."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag in _BLOCK_TAGS:
            self.chunks.append(_PARA_MARK)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _BLOCK_TAGS:
            self.chunks.append(_PARA_MARK)

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in _BLOCK_TAGS:
            self.chunks.append(_PARA_MARK)

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self.chunks.append(data)


def clean_text(html_or_text: str | None) -> str:
    """Plain text from HTML or text: tags stripped, entities unescaped,
    whitespace collapsed, paragraphs separated by a blank line ("\\n\\n")."""
    if not html_or_text:
        return ""
    parser = _TextExtractor()
    parser.feed(html_or_text)
    parser.close()
    # Unescape again for double-escaped feeds ("&amp;amp;").
    joined = html.unescape("".join(parser.chunks))
    paragraphs = (_WS.sub(" ", p).strip() for p in _PARA_SPLIT.split(joined))
    return "\n\n".join(p for p in paragraphs if p)


def article_id(source_id: str, url: str | None, title: str) -> str:
    """Stable id: ``"a_" + sha1(canonical url or "source|title")[:12]``."""
    basis = canonicalize_url(url) or f"{source_id}|{(title or '').strip().lower()}"
    return "a_" + hashlib.sha1(basis.encode("utf-8")).hexdigest()[:12]


def content_hash(text: str) -> str:
    """sha1 of the lowercase alphanumeric characters only.

    Formatting, punctuation and whitespace differences do not change the hash,
    so verbatim syndicated copies collapse to the same value.
    """
    normalised = "".join(ch for ch in (text or "").lower() if ch.isalnum())
    return hashlib.sha1(normalised.encode("utf-8")).hexdigest()


def as_utc(value: datetime | None) -> datetime | None:
    """Timezone-aware UTC datetime (naive values are assumed to be UTC)."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def make_article(raw: RawItem, source: SourceConfig) -> Article:
    """Normalise one :class:`RawItem` into an :class:`Article`."""
    title = _WS.sub(" ", clean_text(raw.title)).strip() or "(untitled)"
    text = clean_text(raw.text)
    url = canonicalize_url(raw.url) or None
    return Article(
        id=article_id(raw.source_id, raw.url, title),
        source_id=raw.source_id,
        source_name=raw.source_name or source.name,
        url=url,
        title=title,
        published_at=as_utc(raw.published_at),
        text=text,
        content_hash=content_hash(text or title),
        category=source.category,
    )
