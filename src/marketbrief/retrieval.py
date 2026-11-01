"""RAG over prior briefs, so each new brief stays consistent with what was said before.

* :class:`BriefIndex` — a small persistent vector index of approved briefs, one
  chunk per claim (plus watchlist and continuity lines);
* :func:`continuity_context` — retrieves prior-brief snippets relevant to this
  cycle's topic and companies (fed to the reduce prompt);
* :func:`continuity_notes` — labels how this cycle relates to each snippet:
  ``continues``, ``updates`` or ``contradicts``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from collections import Counter
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from marketbrief.citations import all_claims, content_tokens, numbers_in
from marketbrief.config import Settings
from marketbrief.embeddings import Embedder
from marketbrief.schemas import (
    ArticleRecord,
    Brief,
    BriefDraft,
    Claim,
    ContinuityNote,
    RetrievedSnippet,
)

logger = logging.getLogger(__name__)

CHUNKS_FILE = "chunks.jsonl"
VECTORS_FILE = "vectors.npy"
# Hits whose vectors are at least this similar to a better hit are near-duplicates.
NEAR_DUPLICATE = 0.95
# claim_id prefixes of chunks that come from watchlist / continuity lines, not claims.
_LINE_CHUNKS = ("watch", "cont")


def _norm_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _as_aware(value: datetime) -> datetime:
    """Timezone-aware datetime (naive values are taken as UTC) so dates compare safely."""
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (x / norms).astype(np.float32)


class BriefIndex:
    """Persistent embedding index over prior briefs (``chunks.jsonl`` + ``vectors.npy``)."""

    def __init__(self, index_dir: Path, embedder: Embedder) -> None:
        self.index_dir = Path(index_dir)
        self.embedder = embedder
        self._lock = threading.RLock()
        self._chunks: list[dict] = []
        self._vectors = np.zeros((0, embedder.dim), dtype=np.float32)
        self._load()

    # ----------------------------------------------------------------- storage

    @property
    def _chunks_path(self) -> Path:
        return self.index_dir / CHUNKS_FILE

    @property
    def _vectors_path(self) -> Path:
        return self.index_dir / VECTORS_FILE

    def _load(self) -> None:
        if not self._chunks_path.exists() or not self._vectors_path.exists():
            return
        try:
            with self._chunks_path.open(encoding="utf-8") as fh:
                chunks = [json.loads(line) for line in fh if line.strip()]
            vectors = np.load(self._vectors_path)
        except (OSError, ValueError) as exc:
            logger.warning("brief index at %s is unreadable, starting empty: %s", self.index_dir, exc)
            return
        if vectors.ndim != 2 or len(chunks) != vectors.shape[0] or vectors.shape[1] != self.embedder.dim:
            logger.warning(
                "brief index at %s does not match the embedder (%s chunks, vectors %s, dim %d); starting empty",
                self.index_dir,
                len(chunks),
                vectors.shape,
                self.embedder.dim,
            )
            return
        self._chunks, self._vectors = chunks, vectors.astype(np.float32)

    def _save(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        tmp_chunks = self._chunks_path.with_suffix(".jsonl.tmp")
        tmp_vectors = self.index_dir / "vectors.tmp.npy"
        with tmp_chunks.open("w", encoding="utf-8") as fh:
            for chunk in self._chunks:
                fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        np.save(tmp_vectors, self._vectors)
        os.replace(tmp_chunks, self._chunks_path)
        os.replace(tmp_vectors, self._vectors_path)

    # ------------------------------------------------------------------ writes

    @staticmethod
    def chunks_for(brief: Brief) -> list[dict]:
        """Index chunks for one brief: claims, watchlist items and continuity lines."""
        date = (brief.period_end or brief.created_at).isoformat()
        base = {"brief_id": brief.id, "brief_title": brief.title, "brief_date": date}
        chunks: list[dict] = []

        def add(claim_id: str, text: str, kind: str) -> None:
            if text.strip():
                chunks.append({**base, "claim_id": claim_id, "text": text.strip(), "kind": kind})

        for claim in brief.executive_summary:
            add(claim.id, f"Executive summary: {claim.text}", claim.kind)
        for section in brief.sections:
            for claim in section.claims:
                add(claim.id, f"{section.title}: {claim.text}", claim.kind)
        for claim in brief.implications:
            add(claim.id, f"Implications: {claim.text}", claim.kind)
        for i, item in enumerate(brief.watchlist, start=1):
            add(f"watch{i}", f"Watchlist: {item}", "watchlist")
        lines = list(brief.continuity_text) or [n.note for n in brief.continuity]
        for i, line in enumerate(lines, start=1):
            add(f"cont{i}", f"Continuity: {line}", "continuity")
        return chunks

    def add_brief(self, brief: Brief) -> int:
        """Index a brief (replacing any earlier version of it). Returns chunks added."""
        chunks = self.chunks_for(brief)
        vectors = _l2_normalize(self.embedder.embed([c["text"] for c in chunks])) if chunks else None
        with self._lock:
            self._drop(brief.id)
            if vectors is not None:
                self._chunks.extend(chunks)
                self._vectors = np.vstack([self._vectors, vectors])
            self._save()
        logger.debug("indexed brief %s (%d chunks)", brief.id, len(chunks))
        return len(chunks)

    def remove_brief(self, brief_id: str) -> None:
        """Remove every chunk of ``brief_id`` (no-op when absent)."""
        with self._lock:
            if self._drop(brief_id):
                self._save()

    def _drop(self, brief_id: str) -> bool:
        keep = [i for i, c in enumerate(self._chunks) if c["brief_id"] != brief_id]
        if len(keep) == len(self._chunks):
            return False
        self._chunks = [self._chunks[i] for i in keep]
        self._vectors = self._vectors[keep] if keep else np.zeros((0, self.embedder.dim), dtype=np.float32)
        return True

    def clear(self) -> None:
        with self._lock:
            self._chunks = []
            self._vectors = np.zeros((0, self.embedder.dim), dtype=np.float32)
            self._save()

    def rebuild(self, dirs: Sequence[Path]) -> int:
        """Re-index from scratch: every approved ``*.json`` Brief in ``dirs``. Returns briefs indexed."""
        briefs: dict[str, Brief] = {}
        for directory in dirs:
            directory = Path(directory)
            if not directory.is_dir():
                logger.debug("brief dir %s does not exist; skipped", directory)
                continue
            for path in sorted(directory.glob("*.json")):
                try:
                    brief = Brief.model_validate_json(path.read_text(encoding="utf-8"))
                except (OSError, ValidationError, ValueError) as exc:
                    logger.warning("skipping %s: not a valid brief (%s)", path, str(exc).splitlines()[0])
                    continue
                if brief.status == "approved":
                    briefs[brief.id] = brief  # later dirs win on duplicate ids
        with self._lock:
            self._chunks = []
            self._vectors = np.zeros((0, self.embedder.dim), dtype=np.float32)
            for brief in briefs.values():
                chunks = self.chunks_for(brief)
                if chunks:
                    self._chunks.extend(chunks)
                    self._vectors = np.vstack([self._vectors, _l2_normalize(self.embedder.embed([c["text"] for c in chunks]))])
            self._save()
        logger.info("rebuilt brief index: %d briefs, %d chunks", len(briefs), len(self._chunks))
        return len(briefs)

    # ------------------------------------------------------------------- reads

    def brief_ids(self) -> list[str]:
        with self._lock:
            return list(dict.fromkeys(c["brief_id"] for c in self._chunks))

    def __len__(self) -> int:
        return len(self._chunks)

    def briefs_dated_from(self, cutoff: datetime) -> list[str]:
        """Ids of indexed briefs whose date (period end) is at or after ``cutoff``.

        Used to keep re-runs of the same period (or later briefs) out of the
        "prior" context, so a brief never cites continuity with itself.
        """
        cutoff = _as_aware(cutoff)
        with self._lock:
            dated = {c["brief_id"]: c.get("brief_date") for c in self._chunks}
        return [
            brief_id
            for brief_id, date in dated.items()
            if date and _as_aware(datetime.fromisoformat(date)) >= cutoff
        ]

    def search(
        self,
        query: str,
        *,
        k: int = 6,
        min_score: float = 0.0,
        exclude_brief_ids: Iterable[str] = (),
    ) -> list[RetrievedSnippet]:
        """Top-``k`` chunks by cosine similarity, near-identical texts collapsed."""
        if not query.strip() or k <= 0:
            return []
        excluded = set(exclude_brief_ids)
        q = _l2_normalize(self.embedder.embed([query]))[0]
        with self._lock:
            if not self._chunks:
                return []
            chunks, vectors = list(self._chunks), self._vectors
        scores = vectors @ q
        hits: list[RetrievedSnippet] = []
        chosen: list[int] = []
        seen_texts: set[str] = set()
        for idx in np.argsort(-scores, kind="stable"):
            score = float(scores[idx])
            if score < min_score:
                break
            chunk = chunks[idx]
            if chunk["brief_id"] in excluded:
                continue
            key = _norm_text(chunk["text"].split(": ", 1)[-1])
            if key in seen_texts or any(float(vectors[idx] @ vectors[j]) >= NEAR_DUPLICATE for j in chosen):
                continue
            seen_texts.add(key)
            chosen.append(int(idx))
            hits.append(
                RetrievedSnippet(
                    brief_id=chunk["brief_id"],
                    brief_title=chunk["brief_title"],
                    brief_date=datetime.fromisoformat(chunk["brief_date"]) if chunk.get("brief_date") else None,
                    claim_id=chunk["claim_id"],
                    text=chunk["text"],
                    score=round(score, 4),
                )
            )
            if len(hits) >= k:
                break
        return hits


def build_index(settings: Settings) -> BriefIndex:
    """Open the configured index; seed it from published + fixture briefs when empty."""
    from marketbrief.embeddings import get_embedder

    index = BriefIndex(settings.path(settings.retrieval.index_dir), get_embedder(settings))
    if len(index) == 0:
        dirs = [settings.path(settings.paths.briefs_dir), *(settings.path(d) for d in settings.retrieval.extra_brief_dirs)]
        index.rebuild(dirs)
    return index


# --------------------------------------------------------------------------- #
# Continuity
# --------------------------------------------------------------------------- #


def _top_companies(records: Sequence[ArticleRecord], n: int = 5) -> list[tuple[str, list[str]]]:
    """Most-mentioned companies with the headlines of the events they appear in."""
    counts: Counter[str] = Counter()
    headlines: dict[str, list[str]] = {}
    for record in records:
        names = {m.name for m in record.companies} | {c for e in record.events for c in e.companies}
        counts.update(names)
        for event in record.events:
            for name in event.companies:
                headlines.setdefault(name, []).append(event.headline)
    return [(name, headlines.get(name, [])) for name, _ in counts.most_common(n)]


def continuity_context(
    index: BriefIndex,
    topic: str,
    records: Sequence[ArticleRecord],
    *,
    k: int,
    min_score: float,
    exclude_brief_ids: Iterable[str] = (),
) -> list[RetrievedSnippet]:
    """Prior-brief snippets relevant to this cycle: by topic and by top companies' events.

    ``exclude_brief_ids`` keeps briefs that are not genuinely prior (e.g. an
    earlier run over the same period) out of the context.
    """
    excluded = tuple(exclude_brief_ids)
    queries = [topic]
    for company, heads in _top_companies(records):
        queries.append(f"{company}: " + "; ".join(dict.fromkeys(heads[:3])) if heads else company)
    best: dict[tuple[str, str], RetrievedSnippet] = {}
    for query in queries:
        for hit in index.search(query, k=k, min_score=min_score, exclude_brief_ids=excluded):
            key = (hit.brief_id, hit.claim_id)
            if key not in best or hit.score > best[key].score:
                best[key] = hit
    ranked = sorted(best.values(), key=lambda s: (-s.score, s.brief_id, s.claim_id))
    return ranked[: 2 * k]


# Keyword families used to decide whether two statements are about the same kind of event.
EVENT_KEYWORDS: dict[str, frozenset[str]] = {
    "funding": frozenset({"raise", "raises", "raised", "raising", "fundraising", "funding", "series", "round", "investment", "investor", "valuation"}),
    "m_and_a": frozenset({"acquire", "acquires", "acquired", "acquiring", "acquisition", "buy", "buys", "buying", "merger", "merge", "takeover", "deal"}),
    "pricing": frozenset({"price", "prices", "pricing", "priced", "discount", "discounting", "discounts", "gpu-hour", "rate", "rates"}),
    "regulatory": frozenset({"regulator", "regulatory", "inquiry", "investigation", "probe", "antitrust", "commission", "scrutiny", "egress", "fine"}),
    "leadership": frozenset({"ceo", "chief", "appoint", "appointed", "appoints", "executive", "resign", "resigned"}),
    "layoffs": frozenset({"layoff", "layoffs", "workforce", "jobs", "headcount", "redundancies"}),
    "earnings": frozenset({"revenue", "earnings", "quarter", "profit", "guidance", "sales"}),
    "expansion": frozenset({"region", "expand", "expansion", "opens", "opened", "centre", "center", "office", "facility"}),
    "litigation": frozenset({"lawsuit", "sues", "sued", "patent", "infringement", "court", "litigation"}),
    "partnership": frozenset({"partner", "partners", "partnership", "alliance", "integrate", "integration"}),
    "product_launch": frozenset({"launch", "launches", "launched", "unveil", "unveils", "release", "released", "model"}),
}

# Words marking an earlier statement as unconfirmed; a later concrete report "updates" it.
SPECULATIVE = frozenset({
    "talks", "reported", "reportedly", "rumour", "rumor", "rumours", "rumors", "expected", "expects",
    "considering", "weighing", "exploring", "plans", "planned", "could", "may", "might", "likely", "potential", "if",
})  # fmt: skip

# Opposing direction words: one side in the prior claim and the other in the new one signals a contradiction.
OPPOSITES: tuple[tuple[frozenset[str], frozenset[str]], ...] = (
    (frozenset({"raise", "raises", "raised", "increase", "increases", "increased", "hike", "hikes", "higher"}),
     frozenset({"cut", "cuts", "lower", "lowers", "lowered", "decrease", "decreases", "decreased", "reduce", "reduces", "reduced"})),
    (frozenset({"up", "rise", "rises", "rose", "grew", "growth", "gain", "gains"}),
     frozenset({"down", "fall", "falls", "fell", "declined", "decline", "drop", "drops", "dropped"})),
    (frozenset({"agreed", "agrees", "completed", "closes", "closed", "signed", "approved", "approves"}),
     frozenset({"abandoned", "abandons", "collapsed", "terminated", "blocked", "blocks", "rejected", "rejects", "denied", "denies", "scrapped"})),
    (frozenset({"hire", "hires", "hiring", "expands", "expanding"}),
     frozenset({"layoff", "layoffs", "shrinks", "shrinking"})),
)  # fmt: skip

NEGATIONS = frozenset({"not", "never", "denied", "denies", "denial"})

# Capitalised words that are not company names (sentence starters, months, section labels...).
_NON_NAMES = frozenset(
    """
    The A An In On At For And But Its It This That These Those With From After Before Over Under Of To By As
    Executive Summary Implications Watchlist Continuity Funding Products Pricing Regulatory People Financial
    Other Developments Results Operations Competition Legal Partnerships Continues Updates Contradicts
    January February March April May June July August September October November December
    Monday Tuesday Wednesday Thursday Friday Saturday Sunday Q1 Q2 Q3 Q4 CEO CFO CTO GPU AI EU US UK
    """.split()
)


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z][a-z0-9\-]*", text.lower()))


def _names(text: str) -> set[str]:
    """Capitalised tokens likely to be (parts of) company names."""
    return {w.lower() for w in re.findall(r"\b[A-Z][A-Za-z0-9\-]+", text) if w not in _NON_NAMES}


def _event_families(words: set[str]) -> set[str]:
    return {family for family, keys in EVENT_KEYWORDS.items() if words & keys}


def _opposed(a: set[str], b: set[str]) -> bool:
    for left, right in OPPOSITES:
        a_left, a_right, b_left, b_right = bool(a & left), bool(a & right), bool(b & left), bool(b & right)
        # Only a clean flip counts: one side in each text, not both sides in either.
        if (a_left and not a_right and b_right and not b_left) or (a_right and not a_left and b_left and not b_right):
            return True
    return bool(a & NEGATIONS) != bool(b & NEGATIONS)


def _overlap(a: str, b: str) -> float:
    """Overlap coefficient of content tokens (|A∩B| / min(|A|,|B|))."""
    ta, tb = content_tokens(a), content_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def _strip_label(text: str) -> str:
    """Drop the "Section title: " prefix that index chunks carry."""
    head, sep, rest = text.partition(": ")
    return rest if sep and len(head) <= 40 else text


def continuity_notes(
    draft: BriefDraft | Brief,
    snippets: Sequence[RetrievedSnippet],
    *,
    min_overlap: float = 0.3,
) -> list[ContinuityNote]:
    """Relate each prior-brief snippet to the best-matching claim in this draft.

    * ``contradicts`` — same company and event family, opposite direction or negation;
    * ``updates`` — same company and event family, and the prior claim was
      speculative ("talks", "expected"...) or the figures changed;
    * ``continues`` — otherwise related (shared company + event family, or
      token overlap >= ``min_overlap``).
    Unrelated snippets are skipped.
    """
    claims: list[Claim] = [c for c in all_claims(draft) if c.text.strip()]
    if not claims:
        return []
    notes: list[ContinuityNote] = []
    seen: set[tuple[str, str]] = set()
    for snippet in snippets:
        prior = _strip_label(snippet.text)
        if (snippet.brief_id, _norm_text(prior)) in seen:
            continue
        prior_words, prior_names = _words(prior), _names(prior)
        prior_families = _event_families(prior_words)

        def match_score(claim: Claim) -> float:
            shared_names = prior_names & _names(claim.text)
            shared_families = prior_families & _event_families(_words(claim.text))
            return _overlap(prior, claim.text) + (0.5 if shared_names else 0.0) + (0.25 if shared_families else 0.0)

        claim = max(claims, key=match_score)  # first claim wins ties (reading order)
        claim_words = _words(claim.text)
        same_company = bool(prior_names & _names(claim.text))
        same_event = bool(prior_families & _event_families(claim_words))
        overlap = _overlap(prior, claim.text)

        if same_company and same_event:
            if _opposed(prior_words, claim_words):
                relation = "contradicts"
            elif prior_words & SPECULATIVE or (numbers_in(claim.text) - numbers_in(prior)):
                relation = "updates"
            else:
                relation = "continues"
        elif same_company and overlap >= min_overlap and not snippet.claim_id.startswith(_LINE_CHUNKS):
            # Bare watchlist/continuity lines ("Watchlist: Vireo Analytics") only count
            # when the event matches too; a shared name alone is noise.
            relation = "continues"
        else:
            continue

        seen.add((snippet.brief_id, _norm_text(prior)))
        verb = {"continues": "Continues", "updates": "Updates", "contradicts": "Contradicts"}[relation]
        notes.append(
            ContinuityNote(
                prior_brief_id=snippet.brief_id,
                prior_claim=prior,
                relation=relation,  # type: ignore[arg-type]
                note=f"{verb} {snippet.brief_title} ({snippet.brief_id}/{snippet.claim_id}): "
                f"earlier \"{prior}\"; now \"{claim.text}\" [{claim.id}]",
            )
        )
    return notes


__all__ = [
    "BriefIndex",
    "build_index",
    "continuity_context",
    "continuity_notes",
]
