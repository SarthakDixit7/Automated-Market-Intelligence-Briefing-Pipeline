"""Tools the analyst agent can call: search, summarise and chart."""

from __future__ import annotations

import logging
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence

import numpy as np

from marketbrief.eval.golden import EVENT_TYPE_SYNONYMS
from marketbrief.schemas import Article, ArticleRecord, ArticleSummary

if TYPE_CHECKING:
    from marketbrief.config import Settings
    from marketbrief.embeddings import Embedder
    from marketbrief.llm.base import LLMClient

log = logging.getLogger(__name__)

ChartDimension = Literal["company", "event_type", "category", "sentiment"]
CHART_DIMENSIONS: tuple[str, ...] = ("company", "event_type", "category", "sentiment")
_LABELS = {"m_and_a": "M&A", "product_launch": "product launch"}

# Characters of body text embedded per article for search.
SEARCH_TEXT_CHARS = 2000
# Score boosts on top of cosine similarity.
COMPANY_BOOST = 0.15
EVENT_BOOST = 0.10
MAX_COMPANY_BOOSTS = 2


# --------------------------------------------------------------------------- #
# search
# --------------------------------------------------------------------------- #


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9][a-z0-9\-]*", text.lower()))


def _mentions(text_lower: str, words: set[str], name: str) -> bool:
    """Company named in text: full name, or its distinctive first word."""
    name_l = name.lower().strip()
    if not name_l:
        return False
    if name_l in text_lower:
        return True
    first = name_l.split()[0]
    return len(first) >= 4 and first in words


def search_articles(
    query: str,
    articles: Sequence[Article],
    records: Sequence[ArticleRecord],
    embedder: "Embedder",
    *,
    k: int = 6,
    min_relative_score: float = 0.35,
) -> list[dict[str, Any]]:
    """Rank articles for ``query``: cosine similarity plus company/event boosts.

    Returns at most ``k`` hits as plain dicts
    ``{"article_id", "title", "source", "published_at", "score"}``. Hits
    scoring below ``min_relative_score`` x the best score are dropped, so a
    narrow question doesn't drag in unrelated filler.
    """
    if not articles or not query.strip() or k <= 0:
        return []

    docs = [f"{a.title}\n{a.text[:SEARCH_TEXT_CHARS]}" for a in articles]
    vectors = np.asarray(embedder.embed([query, *docs]), dtype=np.float32)
    query_vec, doc_vecs = vectors[0], vectors[1:]
    similarity = doc_vecs @ query_vec

    records_by_id = {r.article_id: r for r in records}
    q_lower = query.lower()
    q_words = _words(query)
    wanted_events = {
        event_type
        for event_type, synonyms in EVENT_TYPE_SYNONYMS.items()
        if any((s in q_words) if " " not in s else (s in q_lower) for s in synonyms)
    }

    hits: list[dict[str, Any]] = []
    for article, base in zip(articles, similarity):
        score = float(base)
        record = records_by_id.get(article.id)
        if record is not None:
            companies = {c.name for c in record.companies} | {n for ev in record.events for n in ev.companies}
            company_hits = sum(_mentions(q_lower, q_words, name) for name in sorted(companies))
            score += COMPANY_BOOST * min(company_hits, MAX_COMPANY_BOOSTS)
            if wanted_events & {ev.event_type for ev in record.events}:
                score += EVENT_BOOST
        if score <= 0:
            continue
        hits.append(
            {
                "article_id": article.id,
                "title": article.title,
                "source": article.source_name,
                "published_at": article.published_at.isoformat() if article.published_at else None,
                "score": round(score, 4),
            }
        )

    # Highest score first; ties -> newest first, then id for determinism.
    hits.sort(key=lambda h: (-h["score"], -_timestamp(h["published_at"]), h["article_id"]))
    if hits:
        floor = hits[0]["score"] * min_relative_score
        hits = [h for h in hits if h["score"] >= floor]
    return hits[:k]


def _timestamp(iso: str | None) -> float:
    if not iso:
        return 0.0
    try:
        return datetime.fromisoformat(iso).timestamp()
    except ValueError:
        return 0.0


# --------------------------------------------------------------------------- #
# summarise
# --------------------------------------------------------------------------- #


def summarise_articles(
    article_ids: Sequence[str],
    articles_by_id: Mapping[str, Article],
    records_by_id: Mapping[str, ArticleRecord],
    llm: "LLMClient",
    settings: "Settings",
) -> list[ArticleSummary]:
    """Map-summarise the given articles (reuses the pipeline's map step).

    Unknown ids and articles without an extracted record are skipped.
    """
    from marketbrief.summarize import map_summaries  # lazy: avoids an import cycle

    ids = list(dict.fromkeys(article_ids))  # dedupe, keep order
    usable = [i for i in ids if i in articles_by_id and i in records_by_id]
    skipped = sorted(set(ids) - set(usable))
    if skipped:
        log.warning("summarise: skipping ids without article/record: %s", skipped)
    if not usable:
        return []
    return map_summaries(
        [articles_by_id[i] for i in usable],
        [records_by_id[i] for i in usable],
        llm,
        settings,
    )


# --------------------------------------------------------------------------- #
# chart
# --------------------------------------------------------------------------- #


def count_dimension(records: Sequence[ArticleRecord], dimension: str) -> Counter[str]:
    """Counts behind :func:`chart_event_counts`.

    company: articles mentioning each company; event_type: extracted events;
    category: articles per category; sentiment: articles per sentiment.
    """
    if dimension not in CHART_DIMENSIONS:
        raise ValueError(f"unknown chart dimension {dimension!r}; expected one of {CHART_DIMENSIONS}")
    counts: Counter[str] = Counter()
    for r in records:
        if dimension == "company":
            names = {c.name for c in r.companies} | {n for ev in r.events for n in ev.companies}
            counts.update(names)
        elif dimension == "event_type":
            counts.update(ev.event_type.value for ev in r.events)
        elif dimension == "category":
            counts.update(set(r.categories))
        else:
            counts[r.sentiment.value] += 1
    return counts


def chart_event_counts(
    records: Sequence[ArticleRecord],
    dimension: str,
    out_path: Path,
    *,
    title: str | None = None,
    top_n: int = 15,
) -> Path:
    """Horizontal bar chart of counts by ``dimension``, saved as PNG."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    counts = count_dimension(records, dimension)
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]
    labels = [_LABELS.get(label, label.replace("_", " ")) for label, _ in reversed(items)]
    values = [value for _, value in reversed(items)]

    # Figure + Agg canvas instead of pyplot: always headless, no global figure
    # registry to leak into, safe off the main thread, and it leaves the
    # caller's matplotlib backend untouched.
    fig = Figure(figsize=(8, max(2.5, 0.45 * len(items) + 1.2)), dpi=120)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)
    if items:
        bars = ax.barh(labels, values, color="#2f6db5")
        ax.bar_label(bars, padding=3, fontsize=9)
        ax.set_xlim(0, max(values) * 1.15 + 0.5)
        ax.xaxis.get_major_locator().set_params(integer=True)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, color="#666666")
        ax.set_yticks([])
    unit = "events" if dimension == "event_type" else "articles"
    ax.set_xlabel(f"Number of {unit}")
    ax.set_title(title or f"Count by {dimension.replace('_', ' ')}", loc="left", fontsize=12)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="png")
    log.info("chart saved: %s", out_path)
    return out_path
