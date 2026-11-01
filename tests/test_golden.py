"""Golden set loading, event/claim matching and recall scoring."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from marketbrief.eval.golden import (
    EVENT_TYPE_SYNONYMS,
    brief_fact_claims,
    claim_matches,
    event_matches,
    load_golden,
    recall_report,
)
from marketbrief.schemas import (
    ArticleRecord,
    BriefDraft,
    BriefSection,
    Claim,
    CompanyMention,
    EventType,
    ExtractedEvent,
    GoldenEvent,
    Sentiment,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_FILE = PROJECT_ROOT / "data" / "golden" / "golden_set.yaml"
FIXTURE_ARTICLES = PROJECT_ROOT / "data" / "fixtures" / "articles"
ENTITIES_FILE = PROJECT_ROOT / "data" / "fixtures" / "entities.yaml"


def event(event_type: EventType, companies: list[str], headline: str, quote: str = "", amount=None) -> ExtractedEvent:
    return ExtractedEvent(
        event_type=event_type,
        headline=headline,
        companies=companies,
        amount=amount,
        evidence_quote=quote or headline,
    )


def record(article_id: str, *events: ExtractedEvent) -> ArticleRecord:
    names = sorted({c for ev in events for c in ev.companies})
    return ArticleRecord(
        article_id=article_id,
        companies=[CompanyMention(name=n) for n in names],
        events=list(events),
        categories=[],
        sentiment=Sentiment.NEUTRAL,
        confidence=0.8,
        key_facts=[],
    )


def claim(cid: str, text: str, kind: str = "fact") -> Claim:
    return Claim(id=cid, text=text, citations=["a_x"], kind=kind)


G1 = GoldenEvent(
    id="G1",
    description="Aurora Silicon raises $410 million Series D",
    event_type=EventType.FUNDING,
    companies=["Aurora Silicon"],
    keywords=["Series D", "410"],
)
G2 = GoldenEvent(
    id="G2",
    description="Meridian Cloud to acquire Vireo Analytics",
    event_type=EventType.M_AND_A,
    companies=["Meridian Cloud", "Vireo Analytics"],
    keywords=["1.2 billion", "acquire"],
)
G_OPTIONAL = GoldenEvent(
    id="G9",
    description="Meridian Cloud opens Frankfurt region",
    event_type=EventType.EXPANSION,
    companies=["Meridian Cloud"],
    keywords=["Frankfurt"],
    must_catch=False,
)


# --------------------------------------------------------------------------- #
# The shipped golden set
# --------------------------------------------------------------------------- #


def test_shipped_golden_set_matches_the_fixture_storyline():
    events = load_golden(GOLDEN_FILE)
    assert [e.id for e in events] == [f"G{i}" for i in range(1, 13)]
    by_id = {e.id: e for e in events}
    expected_types = {
        "G1": EventType.FUNDING,
        "G2": EventType.M_AND_A,
        "G3": EventType.PRICING,
        "G4": EventType.PRODUCT_LAUNCH,
        "G5": EventType.PARTNERSHIP,
        "G6": EventType.REGULATORY,
        "G7": EventType.LEADERSHIP,
        "G8": EventType.LAYOFFS,
        "G9": EventType.EARNINGS,
        "G10": EventType.EXPANSION,
        "G11": EventType.LITIGATION,
        "G12": EventType.EXPANSION,
    }
    assert {k: v.event_type for k, v in by_id.items()} == expected_types
    assert {e.id for e in events if not e.must_catch} == {"G10", "G12"}
    assert all(e.companies and e.keywords for e in events)
    assert "Aurora Silicon" in by_id["G1"].companies and "410" in by_id["G1"].keywords
    assert set(by_id["G2"].companies) == {"Meridian Cloud", "Vireo Analytics"}
    assert set(by_id["G11"].companies) == {"Kestrel Networks", "Helix Data Systems"}


def test_golden_companies_are_in_the_entity_gazetteer():
    if not ENTITIES_FILE.exists():
        pytest.skip("entities.yaml not generated yet")
    data = yaml.safe_load(ENTITIES_FILE.read_text(encoding="utf-8"))
    canonical = {c["name"] for c in data["companies"]}
    for g in load_golden(GOLDEN_FILE):
        assert set(g.companies) <= canonical, g.id


def test_load_golden_rejects_bad_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_golden(tmp_path / "missing.yaml")
    no_list = tmp_path / "no_list.yaml"
    no_list.write_text("events: nope\n", encoding="utf-8")
    with pytest.raises(ValueError, match="events"):
        load_golden(no_list)
    dupes = tmp_path / "dupes.yaml"
    dupes.write_text(
        "events:\n"
        "  - {id: X, description: a, event_type: funding, companies: [A]}\n"
        "  - {id: X, description: b, event_type: funding, companies: [B]}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_golden(dupes)


# --------------------------------------------------------------------------- #
# event_matches / claim_matches
# --------------------------------------------------------------------------- #


def test_event_matches_requires_type_company_and_keyword():
    good = event(EventType.FUNDING, ["Aurora Silicon"], "Aurora Silicon raises $410M Series D")
    assert event_matches(G1, good)
    wrong_type = event(EventType.EXPANSION, ["Aurora Silicon"], "Aurora Silicon raises $410M Series D")
    assert not event_matches(G1, wrong_type)
    wrong_company = event(EventType.FUNDING, ["Northwind Compute"], "Northwind raises $410M Series D")
    assert not event_matches(G1, wrong_company)
    no_keyword = event(EventType.FUNDING, ["Aurora Silicon"], "Aurora Silicon closes a funding round")
    assert not event_matches(G1, no_keyword)


def test_event_matches_is_alias_tolerant_and_reads_evidence():
    aliased = event(
        EventType.M_AND_A,
        ["meridian"],  # alias, lower case
        "Meridian buys analytics firm",
        quote="Meridian said it would acquire Vireo for $1.2 billion in cash.",
    )
    assert event_matches(G2, aliased)
    longer_name = event(EventType.M_AND_A, ["Vireo Analytics Inc."], "Deal agreed", quote="It will acquire the firm.")
    assert event_matches(G2, longer_name)


def test_event_matches_without_keywords_needs_only_type_and_company():
    g = GoldenEvent(id="X", description="", event_type=EventType.LAYOFFS, companies=["Kestrel Networks"])
    assert event_matches(g, event(EventType.LAYOFFS, ["Kestrel Networks"], "Kestrel trims staff"))


def test_keyword_matching_is_word_bounded_for_words():
    g = GoldenEvent(id="X", description="", event_type=EventType.EARNINGS, companies=["Northwind Compute"], keywords=["Q3"])
    assert event_matches(g, event(EventType.EARNINGS, ["Northwind Compute"], "Northwind Q3 revenue up 48%"))
    assert not event_matches(g, event(EventType.EARNINGS, ["Northwind Compute"], "Northwind reports FQ34 revenue"))


def test_claim_matches_company_plus_keyword_or_synonym():
    assert claim_matches(G1, "Aurora Silicon raised $410 million in a Series D round.")
    # First-word alias ("Aurora") plus an event-type synonym ("funding").
    assert claim_matches(G1, "Aurora closed new funding from growth investors.")
    assert not claim_matches(G1, "Northwind raised $410 million.")  # wrong company
    assert not claim_matches(G1, "Aurora Silicon opened a design centre.")  # no keyword/synonym


def test_every_event_type_has_synonyms_except_other():
    for et in EventType:
        assert bool(EVENT_TYPE_SYNONYMS[et]) == (et is not EventType.OTHER)


# --------------------------------------------------------------------------- #
# recall_report
# --------------------------------------------------------------------------- #


def make_draft(*claims: Claim) -> BriefDraft:
    return BriefDraft(
        title="t",
        executive_summary=[claims[0]] if claims else [],
        sections=[BriefSection(title="All", claims=list(claims))],
        implications=[claim("i1", "Aurora funding signals a Series D boom.", kind="analysis")],
        watchlist=[],
    )


def test_recall_report_scores_extraction_and_brief():
    records = [
        record("a_1", event(EventType.FUNDING, ["Aurora Silicon"], "Aurora Silicon raises $410M Series D")),
        record("a_2", event(EventType.FUNDING, ["Aurora Silicon"], "Aurora's $410M Series D, a second report")),
        record("a_3", event(EventType.EXPANSION, ["Meridian Cloud"], "Meridian opens Frankfurt region")),
    ]
    draft = make_draft(claim("c1", "Aurora Silicon raised $410 million in a Series D round."))
    report = recall_report([G1, G2, G_OPTIONAL], records, draft)

    assert report.total == 3
    assert report.extraction_recall == pytest.approx(2 / 3, abs=1e-3)
    assert report.brief_recall == pytest.approx(1 / 3, abs=1e-3)
    # G2 missed everywhere; optional G9 missing from the brief is not "must catch".
    assert report.must_catch_missed == ["G2"]
    m1 = report.matches[0]
    assert m1.caught_in_extraction and m1.caught_in_brief
    assert m1.matched_article_ids == ["a_1", "a_2"]
    assert m1.matched_claim_ids == ["c1"]  # exec-summary duplicate counted once
    assert report.matches[2].caught_in_extraction and not report.matches[2].caught_in_brief


def test_analysis_claims_do_not_count_for_brief_recall():
    draft = BriefDraft(
        title="t",
        executive_summary=[],
        sections=[],
        implications=[claim("i1", "Aurora Silicon's $410 million Series D may spark rivals.", kind="analysis")],
        watchlist=[],
    )
    assert brief_fact_claims(draft) == []
    report = recall_report([G1], [], draft)
    assert report.brief_recall == 0.0
    assert report.must_catch_missed == ["G1"]


def test_recall_without_brief_only_judges_extraction():
    records = [record("a_1", event(EventType.FUNDING, ["Aurora Silicon"], "Aurora Silicon raises $410M Series D"))]
    report = recall_report([G1], records, None)
    assert report.extraction_recall == 1.0
    assert report.brief_recall == 0.0
    assert report.must_catch_missed == []


def test_recall_with_empty_golden_set_is_vacuously_complete():
    report = recall_report([], [], make_draft())
    assert report.total == 0 and report.extraction_recall == 1.0 and report.brief_recall == 1.0


# --------------------------------------------------------------------------- #
# End to end against the synthetic fixture feed (offline extractor)
# --------------------------------------------------------------------------- #


HAVE_FIXTURES = bool(list(FIXTURE_ARTICLES.glob("*.json"))) and ENTITIES_FILE.exists()


@pytest.mark.skipif(not HAVE_FIXTURES, reason="fixture articles not generated yet")
def test_offline_extraction_catches_the_golden_events(settings, offline_llm, fixture_articles):
    from marketbrief.extract import extract_records

    records = extract_records(fixture_articles, offline_llm, settings)
    report = recall_report(load_golden(GOLDEN_FILE), records, None)
    missed = [m.golden_id for m in report.matches if not m.caught_in_extraction]
    assert report.extraction_recall >= settings.eval.min_extraction_recall, f"missed {missed}"
    assert report.must_catch_missed == []
