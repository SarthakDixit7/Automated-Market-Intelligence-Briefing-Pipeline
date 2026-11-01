"""Citation validation: lexical support, number consistency, fact-check pass, repair."""

from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import datetime, timezone

import pytest
from pydantic import BaseModel

from marketbrief.citations import (
    FACT_CHECK_BATCH,
    all_claims,
    numbers_in,
    repair_brief,
    support_score,
    validate_citations,
)
from marketbrief.llm.base import LLMError, LLMRequest, LLMResult, Usage
from marketbrief.schemas import (
    Article,
    BriefDraft,
    BriefSection,
    Claim,
    FactCheckReport,
    FactCheckVerdict,
)

MIN_SUPPORT = 0.35


class FakeLLM:
    provider = "fake"
    model = "fake-1"

    def __init__(self, respond: Callable[[LLMRequest], BaseModel | Exception]) -> None:
        self.respond = respond
        self.requests: list[LLMRequest] = []
        self._lock = threading.Lock()

    def complete(self, request: LLMRequest) -> LLMResult:
        with self._lock:
            self.requests.append(request)
        out = self.respond(request)
        if isinstance(out, Exception):
            raise out
        return LLMResult(text=out.model_dump_json(), parsed=out, usage=Usage(), provider="fake", model="fake-1")


def article(article_id: str, title: str, text: str, duplicate_ids: list[str] | None = None) -> Article:
    return Article(
        id=article_id,
        source_id="t",
        source_name="Cloud Ledger",
        title=title,
        published_at=datetime(2026, 10, 8, tzinfo=timezone.utc),
        text=text,
        content_hash=article_id,
        duplicate_ids=duplicate_ids or [],
    )


ARTICLES = {
    "a_m": article(
        "a_m",
        "Meridian Cloud to acquire Vireo Analytics",
        "Meridian Cloud agreed to acquire Vireo Analytics for $1.2 billion in cash. "
        "The observability company has about 900 employees.",
        duplicate_ids=["a_m2"],
    ),
    "a_m2": article(
        "a_m2",
        "Meridian buys Vireo",
        "The deal is expected to close in the first quarter, pending regulatory approval in Brussels.",
    ),
    "a_n": article(
        "a_n",
        "Northwind Compute cuts GPU pricing",
        "Northwind Compute cut H-class GPU-hour prices by 22 percent on Thursday.",
    ),
}


def draft(*claims: Claim, sections: list[BriefSection] | None = None, implications: list[Claim] | None = None) -> BriefDraft:
    return BriefDraft(
        title="Test",
        executive_summary=list(claims),
        sections=sections or [],
        implications=implications or [],
        watchlist=[],
    )


def by_id(brief) -> dict[str, Claim]:
    return {c.id: c for c in all_claims(brief)}


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #


def test_support_score_uses_content_words_and_light_stemming():
    source = "Northwind Compute cut H-class GPU-hour prices by 22 percent."
    assert support_score("Northwind Compute cut prices", source) == 1.0
    assert support_score("Northwind Compute price", source) == 1.0  # plural stemming
    assert support_score("The and of with", source) == 0.0  # no content words
    half = support_score("Northwind Compute acquired Vireo Analytics", source)
    assert 0.3 < half < 0.6


@pytest.mark.parametrize(
    "text, expected",
    [
        ("$1.2 billion", {"1.2b"}),
        ("$1.2B", {"1.2b"}),
        ("1,200 million dollars", {"1.2b"}),
        ("$410M Series D", {"410m"}),
        ("410 million", {"410m"}),
        ("22%", {"22%"}),
        ("22 percent", {"22%"}),
        ("a 9% cut", {"9%"}),
        ("300 engineers", {"300"}),
        ("in 2026", {"2026"}),
        ("Q3 revenue on H100 and LF-3", set()),
        ("3 deals and 7 regions", set()),
        ("$312 million, up 48%", {"312m", "48%"}),
        ("$1,500,000", {"1.5m"}),
    ],
)
def test_numbers_in_normalises_figures(text, expected):
    assert numbers_in(text) == expected


# --------------------------------------------------------------------------- #
# validate_citations
# --------------------------------------------------------------------------- #


def test_supported_weak_and_unsupported_fact_claims():
    brief = draft(
        Claim(id="c1", text="Meridian Cloud agreed to acquire Vireo Analytics for $1.2B.", citations=["a_m"]),
        Claim(id="c2", text="Northwind Compute cut prices amid fierce competition from aggressive hyperscalers globally.", citations=["a_n"]),
        Claim(id="c3", text="Vireo founders celebrated outstanding quarterly profitability with bonuses overseas.", citations=["a_m"]),
    )
    checked, report = validate_citations(brief, ARTICLES, min_support=MIN_SUPPORT)
    claims = by_id(checked)
    assert claims["c1"].verification == "supported" and claims["c1"].support_score >= 0.9
    assert claims["c2"].verification == "weak"
    assert claims["c3"].verification == "unsupported"
    assert report.supported == 1 and report.weak == 1 and report.unsupported == 1
    assert not report.passed
    kinds = {(i.claim_id, i.kind, i.severity) for i in report.issues}
    assert ("c2", "low_support", "warning") in kinds and ("c3", "low_support", "error") in kinds
    # The input is untouched.
    assert by_id(brief)["c1"].verification == "unverified"


def test_duplicate_article_text_counts_as_support():
    brief = draft(Claim(id="c1", text="The deal needs regulatory approval in Brussels.", citations=["a_m"]))
    checked, report = validate_citations(brief, ARTICLES, min_support=MIN_SUPPORT)
    assert by_id(checked)["c1"].verification == "supported"
    assert report.passed


def test_missing_and_unknown_citations_are_errors_and_unknown_ids_are_removed():
    brief = draft(
        Claim(id="c1", text="Meridian Cloud will acquire Vireo Analytics.", citations=[]),
        Claim(id="c2", text="Meridian Cloud will acquire Vireo Analytics.", citations=["a_ghost"]),
        Claim(id="c3", text="Meridian Cloud will acquire Vireo Analytics.", citations=["a_ghost", "a_m"]),
    )
    checked, report = validate_citations(brief, ARTICLES, min_support=MIN_SUPPORT)
    claims = by_id(checked)
    assert claims["c1"].verification == "unsupported"
    assert claims["c2"].verification == "unsupported" and claims["c2"].citations == []
    assert claims["c3"].verification == "supported" and claims["c3"].citations == ["a_m"]
    issues = {(i.claim_id, i.kind) for i in report.issues if i.severity == "error"}
    assert issues == {("c1", "missing_citation"), ("c2", "unknown_source"), ("c3", "unknown_source")}
    assert not report.passed


def test_number_mismatch_is_an_error_even_with_high_lexical_support():
    brief = draft(Claim(id="c1", text="Northwind Compute cut H-class GPU-hour prices by 25 percent.", citations=["a_n"]))
    checked, report = validate_citations(brief, ARTICLES, min_support=MIN_SUPPORT)
    assert by_id(checked)["c1"].verification == "unsupported"
    assert [i.kind for i in report.issues] == ["number_mismatch"]
    assert "25%" in report.issues[0].detail


def test_equivalent_number_formats_match():
    brief = draft(Claim(id="c1", text="Northwind Compute cut GPU-hour prices by 22%.", citations=["a_n"]))
    _, report = validate_citations(brief, ARTICLES, min_support=MIN_SUPPORT)
    assert report.passed and report.supported == 1


def test_analysis_claims_need_only_valid_citations():
    brief = draft(
        Claim(id="c1", text="Meridian Cloud agreed to acquire Vireo Analytics.", citations=["a_m"]),
        implications=[
            Claim(id="c2", text="Consolidation may squeeze independent tooling vendors.", citations=["a_m"], kind="analysis"),
            Claim(id="c3", text="Expect 40% more deals next year.", citations=["a_m"], kind="analysis"),
            Claim(id="c4", text="Consolidation is coming.", citations=[], kind="analysis"),
        ],
    )
    checked, report = validate_citations(brief, ARTICLES, min_support=MIN_SUPPORT)
    claims = by_id(checked)
    assert claims["c2"].verification == "supported"
    assert claims["c3"].verification == "weak"  # invented figure in analysis -> warning
    assert claims["c4"].verification == "weak"
    assert report.passed  # warnings only


def test_fact_check_downgrades_lexically_supported_claims(settings):
    brief = draft(
        Claim(id="c1", text="Meridian Cloud agreed to acquire Vireo Analytics.", citations=["a_m"]),
        Claim(id="c2", text="Northwind Compute cut GPU-hour prices by 22%.", citations=["a_n"]),
        Claim(id="c3", text="Vireo founders celebrated outstanding quarterly profitability with bonuses overseas.", citations=["a_m"]),
    )

    def respond(req):
        ids = [c["id"] for c in req.context["claims"]]
        return FactCheckReport(
            verdicts=[FactCheckVerdict(claim_id=i, supported=(i != "c2"), reason="direction wrong" if i == "c2" else "ok") for i in ids]
        )

    llm = FakeLLM(respond)
    checked, report = validate_citations(brief, ARTICLES, min_support=MIN_SUPPORT, llm=llm, settings=settings)
    claims = by_id(checked)
    assert report.fact_checked
    assert claims["c1"].verification == "supported"
    assert claims["c2"].verification == "weak"
    assert ("c2", "fact_check_failed", "warning") in {(i.claim_id, i.kind, i.severity) for i in report.issues}
    # Only lexically supported claims are sent to the fact-checker, with their cited sources.
    req = llm.requests[0]
    assert req.task == "fact_check" and req.response_model is FactCheckReport
    assert [c["id"] for c in req.context["claims"]] == ["c1", "c2"]
    assert set(req.context["sources"]) == {"a_m", "a_n"}
    assert "Brussels" in req.context["sources"]["a_m"]  # duplicate text travels with its canonical
    assert '<claim id="c2"' in req.prompt


def test_fact_check_is_batched(settings):
    claims = [Claim(id=f"c{i}", text="Meridian Cloud agreed to acquire Vireo Analytics.", citations=["a_m"]) for i in range(1, 31)]
    llm = FakeLLM(
        lambda req: FactCheckReport(
            verdicts=[FactCheckVerdict(claim_id=c["id"], supported=True, reason="ok") for c in req.context["claims"]]
        )
    )
    _, report = validate_citations(draft(*claims), ARTICLES, min_support=MIN_SUPPORT, llm=llm, settings=settings)
    assert [len(r.context["claims"]) for r in llm.requests] == [FACT_CHECK_BATCH, 30 - FACT_CHECK_BATCH]
    assert report.fact_checked and report.supported == 30


def test_fact_check_failure_keeps_lexical_verdicts(settings):
    brief = draft(Claim(id="c1", text="Meridian Cloud agreed to acquire Vireo Analytics.", citations=["a_m"]))
    checked, report = validate_citations(
        brief, ARTICLES, min_support=MIN_SUPPORT, llm=FakeLLM(lambda r: LLMError("down")), settings=settings
    )
    assert not report.fact_checked
    assert by_id(checked)["c1"].verification == "supported"
    assert report.passed


# --------------------------------------------------------------------------- #
# repair_brief
# --------------------------------------------------------------------------- #


def test_repair_drops_unsupported_claims_and_empty_sections():
    brief = draft(
        Claim(id="c1", text="Meridian Cloud agreed to acquire Vireo Analytics for $1.2 billion.", citations=["a_m"]),
        sections=[
            BriefSection(
                title="Pricing",
                claims=[Claim(id="c2", text="Northwind Compute cut prices by 30%.", citations=["a_n"])],
            ),
            BriefSection(
                title="Weak",
                claims=[Claim(id="c3", text="Northwind Compute cut prices amid fierce competition from aggressive hyperscalers globally.", citations=["a_n"])],
            ),
        ],
    )
    checked, report = validate_citations(brief, ARTICLES, min_support=MIN_SUPPORT)
    repaired = repair_brief(checked, report)
    assert [c.id for c in all_claims(repaired)] == ["c1", "c3"]
    assert [s.title for s in repaired.sections] == ["Weak"]
    assert len(all_claims(checked)) == 3  # input untouched


def test_offline_fact_checker_integration(settings):
    try:
        from marketbrief.llm.offline import OfflineLLMClient
    except ImportError:
        pytest.skip("offline provider not available")
    brief = draft(Claim(id="c1", text="Meridian Cloud agreed to acquire Vireo Analytics for $1.2 billion.", citations=["a_m"]))
    _, report = validate_citations(brief, ARTICLES, min_support=MIN_SUPPORT, llm=OfflineLLMClient(), settings=settings)
    assert report.fact_checked and report.passed and report.supported == 1
