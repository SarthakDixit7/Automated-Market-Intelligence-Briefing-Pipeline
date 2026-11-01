"""Golden set of must-catch market events and recall scoring.

Recall is checked at two levels:

* extraction - did any :class:`ArticleRecord` contain a matching event?
* brief - does any fact claim in the brief state it?
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable, Sequence

import yaml

from marketbrief.schemas import (
    ArticleRecord,
    Brief,
    BriefDraft,
    Claim,
    EventType,
    ExtractedEvent,
    GoldenEvent,
    GoldenMatch,
    RecallReport,
)

log = logging.getLogger(__name__)

# Words that signal an event type in free text (claims, questions). Shared
# with the agent's search tool so both read text the same way.
EVENT_TYPE_SYNONYMS: dict[EventType, tuple[str, ...]] = {
    EventType.FUNDING: ("raise", "raises", "raised", "raising", "funding", "money", "series", "round", "investment", "investors"),
    EventType.M_AND_A: ("acquire", "acquires", "acquired", "acquiring", "acquisition", "merger", "buy", "buyout", "takeover", "deal"),
    EventType.PRODUCT_LAUNCH: ("launch", "launches", "unveil", "unveils", "release", "releases", "debut"),
    EventType.PARTNERSHIP: ("partner", "partners", "partnership", "alliance", "collaboration", "embed"),
    EventType.EARNINGS: ("revenue", "earnings", "quarter", "q1", "q2", "q3", "q4", "profit", "results"),
    EventType.PRICING: ("price", "prices", "pricing", "cut", "cuts", "discount", "gpu-hour"),
    EventType.REGULATORY: ("regulator", "regulatory", "inquiry", "investigation", "antitrust", "commission", "probe"),
    EventType.LEADERSHIP: ("ceo", "appoints", "appointed", "chief", "executive", "names", "hires"),
    EventType.LAYOFFS: ("layoff", "layoffs", "job cuts", "workforce", "redundancies", "headcount"),
    EventType.EXPANSION: ("opens", "expands", "expansion", "region", "office", "centre", "center", "new site"),
    EventType.LITIGATION: ("lawsuit", "sues", "sued", "suing", "patent", "infringement", "litigation", "court"),
    EventType.OTHER: (),
}


def load_golden(path: str | Path) -> list[GoldenEvent]:
    """Load ``events:`` from a golden-set YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"golden set not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw_events = data.get("events") if isinstance(data, dict) else None
    if not isinstance(raw_events, list):
        raise ValueError(f"{path}: expected a top-level 'events:' list")
    events = [GoldenEvent.model_validate(item) for item in raw_events]
    ids = [e.id for e in events]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ValueError(f"{path}: duplicate golden ids {dupes}")
    return events


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _company_match(golden_companies: Iterable[str], event_companies: Iterable[str]) -> bool:
    """Case-insensitive, alias-tolerant: either name contains the other."""
    wanted = [_norm(c) for c in golden_companies if c.strip()]
    for got in (_norm(c) for c in event_companies if c.strip()):
        if any(w in got or got in w for w in wanted):
            return True
    return False


def _contains_term(text: str, term: str) -> bool:
    """Whole-word containment for words, plain substring for numbers/phrases with symbols."""
    term = _norm(term)
    if not term:
        return False
    if re.fullmatch(r"[a-z][a-z0-9 \-]*", term):
        return re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text) is not None
    return term in text


def _company_in_text(company: str, text: str) -> bool:
    """Full name, or its distinctive first word (``"Meridian"`` for ``"Meridian Cloud"``)."""
    name = _norm(company)
    if _contains_term(text, name):
        return True
    first = name.split(" ")[0]
    return len(first) >= 4 and first != name and _contains_term(text, first)


def event_matches(golden: GoldenEvent, record_event: ExtractedEvent) -> bool:
    """Same event type, a shared company, and (if any keywords) a keyword hit."""
    if record_event.event_type != golden.event_type:
        return False
    if not _company_match(golden.companies, record_event.companies):
        return False
    if not golden.keywords:
        return True
    haystack = _norm(" ".join([record_event.headline, record_event.evidence_quote, record_event.amount or ""]))
    return any(_contains_term(haystack, kw) for kw in golden.keywords)


def claim_matches(golden: GoldenEvent, claim_text: str) -> bool:
    """The claim names a golden company and a keyword or an event-type synonym."""
    text = _norm(claim_text)
    if not any(_company_in_text(c, text) for c in golden.companies):
        return False
    terms = list(golden.keywords) + list(EVENT_TYPE_SYNONYMS.get(golden.event_type, ()))
    return any(_contains_term(text, t) for t in terms)


# --------------------------------------------------------------------------- #
# Recall
# --------------------------------------------------------------------------- #


def brief_fact_claims(brief: Brief | BriefDraft) -> list[Claim]:
    """Fact claims from executive summary, sections and implications (deduped by id)."""
    claims: list[Claim] = list(brief.executive_summary)
    for section in brief.sections:
        claims.extend(section.claims)
    claims.extend(brief.implications)
    seen: set[str] = set()
    out: list[Claim] = []
    for c in claims:
        if c.kind == "fact" and c.id not in seen:
            seen.add(c.id)
            out.append(c)
    return out


def recall_report(
    golden: Sequence[GoldenEvent],
    records: Sequence[ArticleRecord],
    brief: Brief | BriefDraft | None,
) -> RecallReport:
    """Score how many golden events were extracted and made it into the brief.

    A must-catch event is *missed* if extraction did not catch it, or if a
    brief was given and no fact claim states it.
    """
    claims = brief_fact_claims(brief) if brief is not None else []
    matches: list[GoldenMatch] = []
    for g in golden:
        article_ids = sorted({r.article_id for r in records if any(event_matches(g, ev) for ev in r.events)})
        claim_ids = [c.id for c in claims if claim_matches(g, c.text)]
        matches.append(
            GoldenMatch(
                golden_id=g.id,
                caught_in_extraction=bool(article_ids),
                caught_in_brief=bool(claim_ids),
                matched_article_ids=article_ids,
                matched_claim_ids=claim_ids,
            )
        )

    total = len(golden)
    extracted = sum(m.caught_in_extraction for m in matches)
    in_brief = sum(m.caught_in_brief for m in matches)
    missed = [
        g.id
        for g, m in zip(golden, matches)
        if g.must_catch and (not m.caught_in_extraction or (brief is not None and not m.caught_in_brief))
    ]
    report = RecallReport(
        total=total,
        extraction_recall=round(extracted / total, 4) if total else 1.0,
        brief_recall=(round(in_brief / total, 4) if total else 1.0) if brief is not None else 0.0,
        must_catch_missed=missed,
        matches=matches,
    )
    log.info(
        "golden recall: extraction %.2f, brief %.2f, missed %s",
        report.extraction_recall,
        report.brief_recall,
        missed or "none",
    )
    return report
