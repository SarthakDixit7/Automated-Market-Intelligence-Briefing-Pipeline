"""Typed extraction: Article -> ArticleRecord, Instructor-style.

Each article goes through ``structured_call`` with a schema-constrained output
(``ArticleRecord``) plus semantic validators; validator errors are fed back to
the model as a re-ask. Extraction runs in parallel and degrades gracefully: an
article that still fails after the re-asks is logged and skipped.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Callable, Sequence
from typing import TypeVar
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel

from marketbrief.config import Settings
from marketbrief.llm.base import LLMClient, LLMRequest
from marketbrief.llm.structured import structured_call
from marketbrief.prompts.analyst import extraction_prompt
from marketbrief.schemas import Article, ArticleRecord, EventType

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")

MAX_KEY_FACTS = 5

_QUOTE_CHARS = str.maketrans(
    {
        "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
        "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"',
        "–": "-", "—": "-", "−": "-", " ": " ",
    }
)  # fmt: skip
_ELLIPSIS_RE = re.compile(r"\s*(?:\.\.\.|…)\s*")


def normalize_for_match(text: str) -> str:
    """Case-, quote-, dash- and whitespace-insensitive form used for quote matching."""
    text = unicodedata.normalize("NFKC", text).translate(_QUOTE_CHARS).lower()
    return " ".join(text.split())


def quote_in_text(quote: str, text: str) -> bool:
    """True if ``quote`` appears in ``text`` (normalised; ellipsis-separated parts in order)."""
    haystack = normalize_for_match(text)
    parts = [normalize_for_match(p).strip(" \"'.,;:") for p in _ELLIPSIS_RE.split(quote)]
    parts = [p for p in parts if p]
    if not parts:
        return False
    pos = 0
    for part in parts:
        found = haystack.find(part, pos)
        if found < 0:
            return False
        pos = found + len(part)
    return True


def validate_record(record: ArticleRecord, article: Article, *, check_article_id: bool = True) -> list[str]:
    """Semantic checks a schema can't express. Returns human-readable errors (empty = valid)."""
    errors: list[str] = []
    if check_article_id and record.article_id != article.id:
        errors.append(f"article_id must be '{article.id}', got '{record.article_id}'")
    for i, event in enumerate(record.events, start=1):
        label = f"event {i} ({event.event_type.value})"
        if not event.evidence_quote.strip():
            errors.append(f"{label}: evidence_quote is empty")
        elif not quote_in_text(event.evidence_quote, article.text):
            errors.append(
                f"{label}: evidence_quote is not a verbatim sentence from the article text: "
                f"{event.evidence_quote[:120]!r}"
            )
        if event.event_type != EventType.OTHER and not [c for c in event.companies if c.strip()]:
            errors.append(f"{label}: list at least one company involved")
    if len(record.key_facts) > MAX_KEY_FACTS:
        errors.append(f"key_facts has {len(record.key_facts)} items; keep at most {MAX_KEY_FACTS}")
    bad_categories = [c for c in record.categories if c != c.lower()]
    if bad_categories:
        errors.append(f"categories must be lowercase: {bad_categories}")
    return errors


def build_extract_request(article: Article, settings: Settings) -> LLMRequest:
    """The LLM request for one article's extraction pass."""
    pair = extraction_prompt(article)
    return LLMRequest(
        task="extract",
        system=pair.system,
        prompt=pair.prompt,
        response_model=ArticleRecord,
        effort=settings.llm.effort.get("extract"),
        max_tokens=settings.llm.max_tokens,
        temperature=settings.llm.temperature,
        context={"article": article.model_dump(mode="json")},
    )


def extract_record(article: Article, llm: LLMClient, settings: Settings) -> ArticleRecord:
    """Extract one record; raises if the output still fails validation after re-asks."""

    def validator(model: BaseModel) -> list[str]:
        if not isinstance(model, ArticleRecord):
            return [f"expected ArticleRecord, got {type(model).__name__}"]
        # The id is fixed up below rather than re-asked: it is trivially recoverable.
        return validate_record(model, article, check_article_id=False)

    result = structured_call(
        llm,
        build_extract_request(article, settings),
        validators=[validator],
        max_retries=settings.llm.max_validation_retries,
    )
    record: ArticleRecord = result.parsed
    if record.article_id != article.id:
        logger.debug("fixing article_id %r -> %r", record.article_id, article.id)
        record = record.model_copy(update={"article_id": article.id})
    return record


def extract_records(articles: Sequence[Article], llm: LLMClient, settings: Settings) -> list[ArticleRecord]:
    """Extract records for many articles in parallel, preserving input order.

    Failures are isolated: an article that cannot be extracted is logged and
    left out, so one bad article never stops the run.
    """
    if not articles:
        return []

    def run(article: Article) -> ArticleRecord | None:
        try:
            return extract_record(article, llm, settings)
        except Exception as exc:  # isolate per-article failures (validation, refusal, transport)
            logger.warning("extraction failed for %s (%s): %s", article.id, article.title[:60], exc)
            return None

    results = parallel_map(run, articles, settings.pipeline.max_workers, name="extract")
    records = [r for r in results if r is not None]
    logger.info("extracted %d/%d records", len(records), len(articles))
    return records


def parallel_map(fn: Callable[[T], R], items: Sequence[T], max_workers: int, *, name: str = "worker") -> list[R]:
    """``map`` over a thread pool, results in input order (sequential when 1 worker)."""
    workers = max(1, min(max_workers, len(items)))
    if workers == 1:
        return [fn(item) for item in items]
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=name) as pool:
        return list(pool.map(fn, items))


__all__ = [
    "extract_records",
    "extract_record",
    "validate_record",
    "build_extract_request",
    "quote_in_text",
    "normalize_for_match",
    "parallel_map",
]
