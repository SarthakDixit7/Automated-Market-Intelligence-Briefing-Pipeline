"""Source de-duplication: exact content hash + embedding near-duplicates.

Syndicated copies of one story are folded into a single canonical article
that remembers the other members (``duplicate_ids``) and the outlets that
also ran it (``corroborating_sources``), so briefs cite one article while
still showing corroboration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Sequence

import numpy as np

from marketbrief.embeddings import Embedder, cosine_matrix
from marketbrief.ingest.normalize import content_hash
from marketbrief.schemas import Article, DuplicateCluster

log = logging.getLogger(__name__)

_FAR_FUTURE = datetime.max.replace(tzinfo=timezone.utc)


@dataclass
class DedupResult:
    kept: list[Article]
    clusters: list[DuplicateCluster] = field(default_factory=list)
    removed: int = 0


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Keep the lower index as root so groups are ordered by first member.
            self.parent[max(ra, rb)] = min(ra, rb)


def dedup_text(article: Article, lead_chars: int = 600) -> str:
    """Text embedded for near-duplicate detection: title + lead of the body."""
    return f"{article.title}\n{article.text[:lead_chars]}"


def _canonical_index(members: list[int], articles: Sequence[Article]) -> int:
    """Longest text wins; ties go to the earliest published, then input order."""

    def rank(i: int) -> tuple[int, datetime, int]:
        a = articles[i]
        published = a.published_at
        if published is not None and published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        return (-len(a.text), published or _FAR_FUTURE, i)

    return min(members, key=rank)


def dedupe(
    articles: Sequence[Article],
    embedder: Embedder,
    *,
    threshold: float = 0.80,
    lead_chars: int = 600,
) -> DedupResult:
    """Collapse exact and near-duplicate articles.

    1. Articles with the same :func:`content_hash` are grouped.
    2. ``title + lead`` of every article is embedded and pairs with cosine
       ``>= threshold`` are joined (union-find, i.e. single linkage).

    ``kept`` holds one canonical copy per group, in the input order of the
    canonicals, annotated with ``duplicate_ids`` and ``corroborating_sources``.
    """
    n = len(articles)
    if n == 0:
        return DedupResult(kept=[], clusters=[], removed=0)

    uf = _UnionFind(n)
    hashes = [a.content_hash or content_hash(a.text or a.title) for a in articles]
    first_with_hash: dict[str, int] = {}
    for i, h in enumerate(hashes):
        if h in first_with_hash:
            uf.union(first_with_hash[h], i)
        else:
            first_with_hash[h] = i

    vectors = embedder.embed([dedup_text(a, lead_chars) for a in articles])
    sims = cosine_matrix(vectors, vectors) if n > 1 else np.ones((1, 1), dtype=np.float32)
    rows, cols = np.nonzero(np.triu(sims >= threshold, k=1))
    for i, j in zip(rows.tolist(), cols.tolist()):
        uf.union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(uf.find(i), []).append(i)

    canonicals: list[int] = []
    replacements: dict[int, Article] = {}
    clusters: list[DuplicateCluster] = []
    for members in groups.values():
        canon = _canonical_index(members, articles)
        canonicals.append(canon)
        if len(members) == 1:
            continue
        others = [m for m in members if m != canon]
        canon_article = articles[canon]
        duplicate_ids = _unique([*canon_article.duplicate_ids, *(articles[m].id for m in others)])
        corroborating = _unique(
            [
                *canon_article.corroborating_sources,
                *(articles[m].source_name for m in others),
            ],
            exclude=canon_article.source_name,
        )
        replacements[canon] = canon_article.model_copy(
            update={"duplicate_ids": duplicate_ids, "corroborating_sources": corroborating}
        )
        exact = len({hashes[m] for m in members}) == 1
        pair_sims = [float(sims[a, b]) for a in members for b in members if a < b]
        clusters.append(
            DuplicateCluster(
                canonical_id=canon_article.id,
                member_ids=[articles[m].id for m in members],
                max_similarity=1.0 if exact else round(min(1.0, max(pair_sims)), 4),
                method="exact" if exact else "embedding",
            )
        )
        log.debug("dedup cluster %s <- %s", canon_article.id, [articles[m].id for m in others])

    canonicals.sort()
    kept = [replacements.get(i, articles[i]) for i in canonicals]
    return DedupResult(kept=kept, clusters=clusters, removed=n - len(kept))


def _unique(values: Sequence[str], exclude: str | None = None) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        if v and v != exclude and v not in seen:
            seen.add(v)
            out.append(v)
    return out
