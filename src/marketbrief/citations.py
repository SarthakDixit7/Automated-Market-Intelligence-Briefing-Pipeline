"""Citation validation: every claim must be traceable to the articles it cites.

Checks, per claim:

* citations present (fact claims) and pointing at known article ids;
* lexical support — the share of the claim's content words found in the best
  cited source (source text includes folded near-duplicate articles);
* number consistency — every figure in the claim appears in a cited source
  ("$1.2 billion" and "$1.2B" are the same figure);
* optionally an LLM fact-checker pass that can downgrade lexically-supported
  claims it disagrees with.

``validate_citations`` returns an annotated copy of the brief plus a
:class:`CitationReport`; ``repair_brief`` drops what failed.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable, Mapping
from typing import TypeVar

from pydantic import BaseModel

from marketbrief.config import Settings
from marketbrief.llm.base import LLMClient, LLMRequest
from marketbrief.llm.structured import structured_call
from marketbrief.prompts.fact_checker import fact_check_prompt
from marketbrief.schemas import (
    Article,
    Brief,
    BriefDraft,
    CitationIssue,
    CitationReport,
    Claim,
    FactCheckReport,
)

logger = logging.getLogger(__name__)

B = TypeVar("B", Brief, BriefDraft)

FACT_CHECK_BATCH = 25

STOPWORDS = frozenset(
    """
    a about above after again against all also although among an and any are around as at be because been
    before being below between both but by can could did does doing down during each either else even ever
    every few for from further had has have having he her here hers him his how however i if in into is it
    its itself just last least less like made make many may more most much must my near new next no nor not
    now of off on once one only or other others our out over own per said same says she should since so
    some such than that the their theirs them then there these they this those though through thus to too
    under until up upon very via was we were what when where whether which while who whom whose why will
    with within without would year years yet you your company companies according week weeks month months
    reported report reports announced announces announce statement today yesterday including include
    includes
    """.split()
)

_WORD_RE = re.compile(r"[a-z0-9]+(?:['’][a-z]+)?")

# A number, optionally preceded by a currency symbol and followed by a scale or
# percent marker. Lookbehind keeps identifiers like "H100", "Q3" and "LF-3" out.
_NUMBER_RE = re.compile(
    r"""
    (?<![\w.\-/])
    [$€£]?\s?
    (?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)
    (?:
        \s?(?P<pct>%|percent\b|per\s?cent\b)
      | \s?(?P<scale>trillion|billion|million|thousand|tn|bn|mn|[kmbt])\b
    )?
    (?![\w])
    """,
    re.IGNORECASE | re.VERBOSE,
)
_SCALES = {
    "thousand": 1e3, "k": 1e3,
    "million": 1e6, "mn": 1e6, "m": 1e6,
    "billion": 1e9, "bn": 1e9, "b": 1e9,
    "trillion": 1e12, "tn": 1e12, "t": 1e12,
}  # fmt: skip


# --------------------------------------------------------------------------- #
# Lexical helpers
# --------------------------------------------------------------------------- #


def _stem(word: str) -> str:
    """Very light plural/possessive stemming (enough to match 'prices' to 'price')."""
    word = re.sub(r"['’]s$", "", word)
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("es") and word[-3] in "sxz":
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def content_tokens(text: str) -> set[str]:
    """Lowercased, stemmed content words (length > 3, stopwords removed)."""
    out: set[str] = set()
    for raw in _WORD_RE.findall(text.lower()):
        if len(raw) <= 3 or raw in STOPWORDS:
            continue
        word = _stem(raw)
        if word not in STOPWORDS:
            out.add(word)
    return out


def support_score(claim_text: str, source_text: str) -> float:
    """Fraction of the claim's content tokens that appear in the source (0..1)."""
    claim = content_tokens(claim_text)
    if not claim:
        return 0.0
    return len(claim & content_tokens(source_text)) / len(claim)


def _format_number(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text or "0"


def numbers_in(text: str) -> set[str]:
    """Normalised figures in ``text``.

    * money/scaled amounts -> value in the largest unit: "$1.2 billion", "$1.2B",
      "1,200 million" -> "1.2b"; "$410M" -> "410m";
    * percentages -> "22%" (any size);
    * other plain numbers >= 10 -> "300", "2026" (years are kept, consistently on
      both sides of a comparison). Small bare numbers are ignored as noise.
    """
    found: set[str] = set()
    for m in _NUMBER_RE.finditer(text):
        value = float(m.group("num").replace(",", ""))
        if m.group("pct"):
            found.add(_format_number(value) + "%")
            continue
        scale = m.group("scale")
        if scale:
            value *= _SCALES[scale.lower()]
        if value >= 1e12:
            found.add(_format_number(value / 1e12) + "t")
        elif value >= 1e9:
            found.add(_format_number(value / 1e9) + "b")
        elif value >= 1e6:
            found.add(_format_number(value / 1e6) + "m")
        elif value >= 10:
            found.add(_format_number(value))
    return found


# --------------------------------------------------------------------------- #
# Brief helpers
# --------------------------------------------------------------------------- #


def all_claims(brief: Brief | BriefDraft) -> list[Claim]:
    """Every claim in reading order: executive summary, sections, implications."""
    out = list(brief.executive_summary)
    for section in brief.sections:
        out.extend(section.claims)
    out.extend(brief.implications)
    return out


def article_source_text(article: Article, articles: Mapping[str, Article]) -> str:
    """Title + text of an article plus the texts of its folded near-duplicates."""
    parts = [article.title, article.text]
    for dup_id in article.duplicate_ids:
        dup = articles.get(dup_id)
        if dup is not None:
            parts.extend([dup.title, dup.text])
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def validate_citations(
    brief: B,
    articles: Mapping[str, Article],
    *,
    min_support: float,
    llm: LLMClient | None = None,
    settings: Settings | None = None,
) -> tuple[B, CitationReport]:
    """Annotate a copy of ``brief`` with per-claim verification and return a report.

    Unknown citation ids are recorded as errors and removed from the copy, so
    downstream stages never see dangling ids. ``passed`` is True when there
    are no error-severity issues.
    """
    checked = brief.model_copy(deep=True)
    issues: list[CitationIssue] = []
    source_cache: dict[str, str] = {}

    def source_text(article_id: str) -> str:
        if article_id not in source_cache:
            source_cache[article_id] = article_source_text(articles[article_id], articles)
        return source_cache[article_id]

    for claim in all_claims(checked):
        unknown = [cid for cid in claim.citations if cid not in articles]
        for cid in unknown:
            issues.append(_issue(claim, "error", "unknown_source", f"cites unknown article id '{cid}'"))
        claim.citations = [cid for cid in dict.fromkeys(claim.citations) if cid in articles]

        if not claim.citations:
            claim.support_score = 0.0
            if claim.kind == "fact":
                claim.verification = "unsupported"
                if not unknown:
                    issues.append(_issue(claim, "error", "missing_citation", "fact claim has no citation"))
            else:
                claim.verification = "unsupported" if unknown else "weak"
                if not unknown:
                    issues.append(_issue(claim, "warning", "missing_citation", "analysis claim has no citation"))
            continue

        texts = [source_text(cid) for cid in claim.citations]
        claim.support_score = round(max(support_score(claim.text, t) for t in texts), 3)
        missing_numbers = sorted(numbers_in(claim.text) - set().union(*(numbers_in(t) for t in texts)))

        if claim.kind == "analysis":
            claim.verification = "supported"
            if missing_numbers:
                claim.verification = "weak"
                issues.append(
                    _issue(claim, "warning", "number_mismatch", f"figures not in cited sources: {', '.join(missing_numbers)}")
                )
            continue

        if missing_numbers:
            claim.verification = "unsupported"
            issues.append(
                _issue(claim, "error", "number_mismatch", f"figures not in cited sources: {', '.join(missing_numbers)}")
            )
        elif claim.support_score >= min_support:
            claim.verification = "supported"
        elif claim.support_score >= min_support / 2:
            claim.verification = "weak"
            issues.append(
                _issue(claim, "warning", "low_support", f"support {claim.support_score:.2f} < {min_support:.2f}")
            )
        else:
            claim.verification = "unsupported"
            issues.append(
                _issue(claim, "error", "low_support", f"support {claim.support_score:.2f} < {min_support / 2:.2f}")
            )

    fact_checked = False
    if llm is not None:
        fact_checked = _llm_fact_check(checked, articles, source_text, llm, settings or Settings(), issues)

    claims = all_claims(checked)
    report = CitationReport(
        passed=not any(i.severity == "error" for i in issues),
        claims_checked=len(claims),
        supported=sum(c.verification == "supported" for c in claims),
        weak=sum(c.verification == "weak" for c in claims),
        unsupported=sum(c.verification == "unsupported" for c in claims),
        issues=issues,
        fact_checked=fact_checked,
    )
    logger.info(
        "citation check: %d claims, %d supported, %d weak, %d unsupported, %d issues",
        report.claims_checked,
        report.supported,
        report.weak,
        report.unsupported,
        len(issues),
    )
    return checked, report


def _issue(claim: Claim, severity: str, kind: str, detail: str) -> CitationIssue:
    return CitationIssue(claim_id=claim.id, severity=severity, kind=kind, detail=detail)  # type: ignore[arg-type]


def _llm_fact_check(
    brief: Brief | BriefDraft,
    articles: Mapping[str, Article],
    source_text: Callable[[str], str],
    llm: LLMClient,
    settings: Settings,
    issues: list[CitationIssue],
) -> bool:
    """Run the fact-checker over lexically supported claims; downgrade disagreements.

    Returns True when every batch was checked. A failing batch is logged and
    skipped; the lexical verdicts stand for its claims.
    """
    candidates = [c for c in all_claims(brief) if c.verification == "supported"]
    all_ok = True
    for start in range(0, len(candidates), FACT_CHECK_BATCH):
        batch = candidates[start : start + FACT_CHECK_BATCH]
        sources = {cid: source_text(cid) for c in batch for cid in c.citations if cid in articles}
        pair = fact_check_prompt(batch, sources)
        request = LLMRequest(
            task="fact_check",
            system=pair.system,
            prompt=pair.prompt,
            response_model=FactCheckReport,
            effort=settings.llm.effort.get("fact_check"),
            max_tokens=settings.llm.max_tokens,
            temperature=settings.llm.temperature,
            context={
                "claims": [c.model_dump(mode="json") for c in batch],
                "sources": sources,
            },
        )
        batch_ids = {c.id for c in batch}
        try:
            result = structured_call(
                llm,
                request,
                validators=[_verdict_validator(batch_ids)],
                max_retries=settings.llm.max_validation_retries,
            )
        except Exception as exc:  # optional pass: any provider failure keeps the lexical verdicts
            logger.warning("fact-check batch %d failed, keeping lexical verdicts: %s", start // FACT_CHECK_BATCH, exc)
            all_ok = False
            continue
        report: FactCheckReport = result.parsed
        by_id = {c.id: c for c in batch}
        for verdict in report.verdicts:
            claim = by_id.get(verdict.claim_id)
            if claim is not None and not verdict.supported and claim.verification == "supported":
                claim.verification = "weak"
                issues.append(_issue(claim, "warning", "fact_check_failed", verdict.reason.strip() or "fact-checker disagreed"))
    return all_ok


def _verdict_validator(batch_ids: set[str]) -> Callable[[BaseModel], list[str]]:
    def check(report: BaseModel) -> list[str]:
        if not isinstance(report, FactCheckReport):
            return [f"expected FactCheckReport, got {type(report).__name__}"]
        errors = [f"verdict for unknown claim id '{v.claim_id}'" for v in report.verdicts if v.claim_id not in batch_ids]
        seen = {v.claim_id for v in report.verdicts}
        missing = sorted(batch_ids - seen)
        if missing:
            errors.append(f"missing verdicts for claim ids: {', '.join(missing)}")
        return errors

    return check


# --------------------------------------------------------------------------- #
# Repair
# --------------------------------------------------------------------------- #


def repair_brief(brief: B, report: CitationReport) -> B:
    """Copy of ``brief`` without unsupported claims (and sections left empty).

    Weak claims are kept (renderers flag them). Claims are judged by the
    ``verification`` that :func:`validate_citations` set; claims the report
    lists with error-severity issues are dropped too, in case the brief was
    edited after validation.
    """
    error_ids = {i.claim_id for i in report.issues if i.severity == "error"}

    def keep(claim: Claim) -> bool:
        if claim.verification == "unsupported":
            return False
        if claim.verification == "unverified" and claim.id in error_ids:
            return False
        return True

    repaired = brief.model_copy(deep=True)
    repaired.executive_summary = [c for c in repaired.executive_summary if keep(c)]
    repaired.implications = [c for c in repaired.implications if keep(c)]
    sections = []
    for section in repaired.sections:
        section.claims = [c for c in section.claims if keep(c)]
        if section.claims:
            sections.append(section)
    repaired.sections = sections
    dropped = len(all_claims(brief)) - len(all_claims(repaired))
    if dropped:
        logger.info("repair dropped %d unsupported claim(s)", dropped)
    return repaired


def citation_ids(brief: Brief | BriefDraft) -> list[str]:
    """Distinct cited article ids in reading order."""
    return list(dict.fromkeys(cid for c in all_claims(brief) for cid in c.citations))


def unknown_citations(brief: Brief | BriefDraft, known: Iterable[str]) -> list[str]:
    """Cited ids that are not in ``known``."""
    allowed = set(known)
    return [cid for cid in citation_ids(brief) if cid not in allowed]


__all__ = [
    "support_score",
    "numbers_in",
    "content_tokens",
    "validate_citations",
    "repair_brief",
    "all_claims",
    "citation_ids",
    "unknown_citations",
    "article_source_text",
]

