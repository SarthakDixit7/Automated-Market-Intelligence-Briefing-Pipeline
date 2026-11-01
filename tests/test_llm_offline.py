"""Offline provider, client factory and usage tracker."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import yaml

from marketbrief.llm import (
    AnthropicLLMClient,
    LLMError,
    LLMRequest,
    LLMResult,
    OfflineLLMClient,
    Usage,
    UsageTracker,
    build_llm,
)
from marketbrief.llm.offline import (
    Gazetteer,
    check_claim,
    detect_event_types,
    first_amount,
    split_sentences,
)
from marketbrief.schemas import (
    AgentDecision,
    ArticleRecord,
    ArticleSummary,
    BriefDraft,
    Claim,
    EventType,
    FactCheckReport,
    GoldenEvent,
    JudgeScore,
    RetrievedSnippet,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = PROJECT_ROOT / "data" / "fixtures" / "articles"
ENTITIES_FILE = PROJECT_ROOT / "data" / "fixtures" / "entities.yaml"
GOLDEN_FILE = PROJECT_ROOT / "data" / "golden" / "golden_set.yaml"

GAZETTEER = {
    "companies": [
        {"name": "Meridian Cloud", "aliases": ["Meridian"], "category": "hyperscale cloud"},
        {"name": "Vireo Analytics", "aliases": ["Vireo"], "category": "observability"},
        {"name": "Aurora Silicon", "aliases": ["Aurora"], "category": "AI chips"},
        {"name": "Northwind Compute", "aliases": ["Northwind"], "category": "GPU cloud"},
        {"name": "Kestrel Networks", "aliases": ["Kestrel"], "category": "datacenter networking"},
        {"name": "Helix Data Systems", "aliases": ["Helix"], "category": "data platform"},
    ]
}

ARTICLES = {
    "acq": {
        "id": "a_acq",
        "title": "Meridian Cloud agrees to acquire Vireo Analytics for $1.2 billion",
        "published_at": "2026-10-08T09:00:00Z",
        "text": (
            "Meridian Cloud said on Wednesday it has agreed to acquire Vireo Analytics for $1.2 billion in cash.\n\n"
            "The acquisition is expected to close in the first quarter and is subject to regulatory approval. "
            "Vireo's 340 employees will join Meridian's observability unit.\n\n"
            "“This accelerates our roadmap,” said Meridian CEO Priya Raman. "
            "Rival Northwind Compute declined to comment."
        ),
    },
    "pricing": {
        "id": "a_price",
        "title": "Northwind Compute cuts H-class GPU-hour pricing by 22%",
        "published_at": "2026-10-09T09:00:00Z",
        "text": (
            "Northwind Compute cut on-demand pricing for its H-class instances by 22% to $2.10 per GPU-hour on Thursday. "
            "Analysts said the move could trigger a price war among GPU clouds.\n\n"
            "The company said the cut reflects better utilisation across its 14 regions."
        ),
    },
    "funding": {
        "id": "a_fund",
        "title": "Aurora Silicon raises $410 million Series D",
        "published_at": "2026-10-06T09:00:00Z",
        "text": (
            "Aurora Silicon has raised $410 million in a Series D round led by Halcyon Ventures. "
            "The company will use the funds to expand production.\n\n"
            "Aurora opens a design centre in Bengaluru with 300 engineers next year."
        ),
    },
    "inquiry": {
        "id": "a_inq",
        "title": "European Commission opens inquiry into Meridian Cloud egress fees",
        "published_at": "2026-10-16T09:00:00Z",
        "text": (
            "The European Commission opened a formal inquiry into Meridian Cloud's data egress fees on Friday. "
            "Regulators will examine whether the fees lock customers in.\n\n"
            "Meridian said it would cooperate fully."
        ),
    },
    "region": {
        "id": "a_region",
        "title": "Meridian Cloud launches sovereign cloud region in Frankfurt",
        "published_at": "2026-10-29T09:00:00Z",
        "text": (
            "Meridian Cloud launched a sovereign cloud region in Frankfurt on Thursday, its third in Europe. "
            "The region is operated by a German subsidiary."
        ),
    },
    "lawsuit": {
        "id": "a_suit",
        "title": "Kestrel Networks sues Helix Data Systems over patent infringement",
        "published_at": "2026-10-28T09:00:00Z",
        "text": (
            "Kestrel Networks sued Helix Data Systems on Tuesday, alleging infringement of four patents. "
            "Kestrel is seeking $90 million in damages."
        ),
    },
}


@pytest.fixture
def entities_path(tmp_path: Path) -> Path:
    path = tmp_path / "entities.yaml"
    path.write_text(yaml.safe_dump(GAZETTEER), encoding="utf-8")
    return path


@pytest.fixture
def client(entities_path: Path) -> OfflineLLMClient:
    return OfflineLLMClient(entities_path=entities_path)


def req(task: str, model, **context) -> LLMRequest:
    return LLMRequest(task=task, system="role prompt", prompt=f"prompt for {task}", response_model=model, context=context)


def extract(client: OfflineLLMClient, article: dict) -> ArticleRecord:
    return client.complete(req("extract", ArticleRecord, article=article)).parsed


def summarise(client: OfflineLLMClient, article: dict, record: ArticleRecord) -> ArticleSummary:
    return client.complete(
        req("map_summary", ArticleSummary, article=article, record=record.model_dump(mode="json"))
    ).parsed


# --------------------------------------------------------------------------- #
# Text heuristics
# --------------------------------------------------------------------------- #


def test_split_sentences_are_verbatim_and_respect_abbreviations():
    text = "Helix Data Systems Inc. said revenue rose 12%. It also hired Dr. Mehta.\n\n“We are pleased,” she said. Shares rose $1.50 to $31."
    sentences = split_sentences(text)
    assert sentences == [
        "Helix Data Systems Inc. said revenue rose 12%.",
        "It also hired Dr. Mehta.",
        "“We are pleased,” she said.",
        "Shares rose $1.50 to $31.",
    ]
    assert all(s in text for s in sentences)


@pytest.mark.parametrize(
    ("sentence", "expected"),
    [
        ("Aurora Silicon raised $410 million in a Series D round.", EventType.FUNDING),
        ("Meridian Cloud agreed to acquire Vireo Analytics.", EventType.M_AND_A),
        ("LumenForge AI launched its LF-3 model on Monday.", EventType.PRODUCT_LAUNCH),
        ("Helix Data Systems will partner with LumenForge AI.", EventType.PARTNERSHIP),
        ("Northwind Compute reported Q3 revenue of $312 million.", EventType.EARNINGS),
        ("Northwind cut prices to $2.10 per GPU-hour.", EventType.PRICING),
        ("The European Commission opened an inquiry into Meridian.", EventType.REGULATORY),
        ("Bastion Labs appoints Dana Okafor as CEO.", EventType.LEADERSHIP),
        ("Kestrel Networks cuts 9% of its workforce.", EventType.LAYOFFS),
        ("Aurora opens a design centre in Bengaluru.", EventType.EXPANSION),
        ("Kestrel Networks sues Helix over patent infringement.", EventType.LITIGATION),
    ],
)
def test_event_lexicon(sentence: str, expected: EventType):
    assert expected in detect_event_types(sentence)


def test_lexicon_disambiguation():
    region = detect_event_types("Meridian Cloud launches a sovereign cloud region in Frankfurt.")
    assert EventType.EXPANSION in region and EventType.PRODUCT_LAUNCH not in region
    inquiry = detect_event_types("The Commission opens an inquiry into egress fees.")
    assert EventType.EXPANSION not in inquiry and EventType.REGULATORY in inquiry
    assert EventType.REGULATORY not in detect_event_types("The deal is subject to regulatory approval.")
    assert detect_event_types("“We are excited,” said CEO Jane Doe.") == []


@pytest.mark.parametrize(
    ("sentence", "amount"),
    [
        ("It raised $410 million.", "$410M"),
        ("A deal worth $1.2 billion.", "$1.2B"),
        ("Prices fell 22% to $2.10.", "22%"),
        ("Revenue of $312M, up 48 percent.", "$312M"),
        ("No figures here.", None),
    ],
)
def test_first_amount(sentence: str, amount: str | None):
    assert first_amount(sentence) == amount


def test_gazetteer_aliases_and_regex_fallback(entities_path: Path):
    gaz = Gazetteer.load(entities_path)
    text = "Meridian's deal for Vireo was welcomed by Orbital Labs and The Quanta Systems team."
    assert gaz.companies_in(text) == ["Meridian Cloud", "Vireo Analytics", "Orbital Labs", "Quanta Systems"]
    assert gaz.category("Aurora Silicon") == "ai chips"


def test_missing_gazetteer_is_tolerated(tmp_path: Path):
    client = OfflineLLMClient(entities_path=tmp_path / "missing.yaml")
    record = extract(client, ARTICLES["acq"])
    names = [c.name for c in record.companies]
    assert names[:2] == ["Meridian Cloud", "Vireo Analytics"]  # found by the suffix regex


# --------------------------------------------------------------------------- #
# extract
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("key", "event_type", "amount", "companies"),
    [
        ("acq", EventType.M_AND_A, "$1.2B", {"Meridian Cloud", "Vireo Analytics"}),
        ("pricing", EventType.PRICING, "22%", {"Northwind Compute"}),
        ("funding", EventType.FUNDING, "$410M", {"Aurora Silicon"}),
        ("inquiry", EventType.REGULATORY, None, {"Meridian Cloud"}),
        ("region", EventType.EXPANSION, None, {"Meridian Cloud"}),
        ("lawsuit", EventType.LITIGATION, None, {"Kestrel Networks", "Helix Data Systems"}),
    ],
)
def test_extract_primary_event(client, key, event_type, amount, companies):
    article = ARTICLES[key]
    record = extract(client, article)
    assert record.article_id == article["id"]
    primary = record.events[0]
    assert primary.event_type == event_type
    assert companies <= set(primary.companies)
    assert primary.amount == amount
    assert primary.evidence_quote in article["text"]  # verbatim evidence
    assert len(primary.headline) <= 160


def test_extract_record_fields(client):
    record = extract(client, ARTICLES["acq"])
    roles = {c.name: c.role for c in record.companies}
    assert roles["Meridian Cloud"] == "subject"
    assert roles["Vireo Analytics"] == "counterparty"
    assert roles["Northwind Compute"] == "competitor"
    assert record.categories[0] == "hyperscale cloud"
    assert all(c == c.lower() for c in record.categories)
    assert 0.55 <= record.confidence <= 0.95
    assert 1 <= len(record.key_facts) <= 5
    assert all(f in ARTICLES["acq"]["text"] for f in record.key_facts)
    # "subject to regulatory approval" must not become a regulatory event
    assert EventType.REGULATORY not in {e.event_type for e in record.events}
    # every event names at least one company and quotes the article
    for event in record.events:
        assert event.companies
        assert event.evidence_quote in ARTICLES["acq"]["text"]


def test_extract_prefers_evidence_naming_a_company(client):
    record = extract(client, ARTICLES["funding"])
    expansion = next(e for e in record.events if e.event_type == EventType.EXPANSION)
    assert "Bengaluru" in expansion.evidence_quote


def test_sentiment(client):
    assert extract(client, ARTICLES["lawsuit"]).sentiment.value == "negative"
    assert extract(client, ARTICLES["funding"]).sentiment.value == "positive"


def test_extract_is_deterministic(entities_path):
    a = OfflineLLMClient(entities_path=entities_path).complete(req("extract", ArticleRecord, article=ARTICLES["acq"]))
    b = OfflineLLMClient(entities_path=entities_path).complete(req("extract", ArticleRecord, article=ARTICLES["acq"]))
    assert a.text == b.text
    assert a.parsed == b.parsed


def test_result_metadata(client):
    request = req("extract", ArticleRecord, article=ARTICLES["acq"])
    result = client.complete(request)
    assert result.provider == "offline" and result.model == "offline-heuristic-v1"
    assert result.usage.input_tokens == len(request.prompt) // 4
    assert result.usage.output_tokens == len(result.text) // 4
    assert ArticleRecord.model_validate_json(result.text) == result.parsed
    assert result.stop_reason == "end_turn" and not result.cached


def test_response_model_subclass_is_honoured(client):
    class StrictRecord(ArticleRecord):
        pass

    result = client.complete(req("extract", StrictRecord, article=ARTICLES["acq"]))
    assert isinstance(result.parsed, StrictRecord)


# --------------------------------------------------------------------------- #
# map / reduce / edit
# --------------------------------------------------------------------------- #


def test_map_summary(client):
    article = ARTICLES["acq"]
    record = extract(client, article)
    summary = summarise(client, article, record)
    assert summary.article_id == article["id"]
    assert summary.summary.startswith("Meridian Cloud said on Wednesday")
    assert summary.key_points == record.key_facts[:4]
    assert summary.event_types[0] == EventType.M_AND_A
    assert "Meridian Cloud" in summary.why_it_matters
    assert summary.companies[0] == "Meridian Cloud"


def _cycle(client):
    keys = ["pricing", "acq", "funding", "inquiry", "region", "lawsuit"]
    records = [extract(client, ARTICLES[k]) for k in keys]
    summaries = [summarise(client, ARTICLES[k], r) for k, r in zip(keys, records)]
    return records, summaries


def _all_claims(draft: BriefDraft) -> list[Claim]:
    return [*draft.executive_summary, *(c for s in draft.sections for c in s.claims), *draft.implications]


def test_reduce_brief(client):
    records, summaries = _cycle(client)
    prior = [
        RetrievedSnippet(
            brief_id="b_20260930_prior1",
            brief_title="September brief",
            claim_id="c2",
            text="Funding & M&A: Meridian Cloud was reported to be in talks to buy Vireo Analytics.",
            score=0.7,
        ).model_dump(mode="json"),
        RetrievedSnippet(
            brief_id="b_20260930_prior1", brief_title="September brief", claim_id="c9",
            text="Unrelated: a company nobody covers this cycle shipped a router.", score=0.4,
        ).model_dump(mode="json"),
    ]
    draft = client.complete(
        req(
            "reduce_brief",
            BriefDraft,
            topic="Enterprise AI infrastructure",
            summaries=[s.model_dump(mode="json") for s in summaries],
            records=[r.model_dump(mode="json") for r in records],
            prior_context=prior,
        )
    ).parsed

    claims = _all_claims(draft)
    ids = [c.id for c in claims]
    assert ids == [f"c{i}" for i in range(1, len(ids) + 1)]
    known = {s.article_id for s in summaries}
    assert all(c.citations and set(c.citations) <= known for c in claims)

    titles = [s.title for s in draft.sections]
    assert titles[0] == "Funding & M&A"
    assert {"Pricing & competition", "Regulatory & legal", "People & operations"} <= set(titles)
    # one fact claim per article summary, text = the summary's lead sentence
    section_claims = [c for s in draft.sections for c in s.claims]
    assert sorted(c.citations[0] for c in section_claims) == sorted(known)
    assert all(c.kind == "fact" for c in section_claims)

    assert 3 <= len(draft.executive_summary) <= 5
    assert draft.executive_summary[0].citations == ["a_acq"]  # M&A ranks first
    assert draft.implications and all(c.kind == "analysis" for c in draft.implications)
    assert draft.watchlist[0].startswith("Meridian Cloud")
    assert len(draft.continuity) == 1 and draft.continuity[0].startswith("Updates:")
    assert "b_20260930_prior1" in draft.continuity[0]


def test_reduce_merges_partials(client):
    records, summaries = _cycle(client)

    def partial(idx: slice) -> dict:
        return client.complete(
            req(
                "reduce_brief",
                BriefDraft,
                topic="T",
                summaries=[s.model_dump(mode="json") for s in summaries[idx]],
                records=[r.model_dump(mode="json") for r in records[idx]],
                prior_context=[],
            )
        ).parsed.model_dump(mode="json")

    merged = client.complete(
        req("reduce_brief", BriefDraft, topic="T", partials=[partial(slice(0, 3)), partial(slice(3, 6))], prior_context=[])
    ).parsed
    claims = _all_claims(merged)
    assert len({c.id for c in claims}) == len(claims)
    cited = {c.citations[0] for s in merged.sections for c in s.claims}
    assert cited == {s.article_id for s in summaries}
    assert len(merged.executive_summary) <= 5
    assert [s.title for s in merged.sections][0] == "Funding & M&A"
    assert merged.watchlist[0].startswith("Meridian Cloud: 3 articles")


def test_edit_dedupes_trims_and_caps(client):
    claim = lambda i, text: Claim(id=f"c{i}", text=text, citations=["a1"])  # noqa: E731
    draft = BriefDraft(
        title="  Weekly   brief ",
        executive_summary=[claim(i, f"Headline {i}") for i in range(1, 8)],
        sections=[
            {"title": "A", "claims": [claim(10, "  Same   claim. "), claim(11, "Same claim.")]},
            {"title": "B", "claims": [claim(12, "same claim.")]},
        ],
        implications=[],
        watchlist=["X", "X", " "],
    )
    edited = client.complete(req("edit", BriefDraft, draft=draft.model_dump(mode="json"))).parsed
    assert edited.title == "Weekly brief"
    assert len(edited.executive_summary) == 5
    assert [s.title for s in edited.sections] == ["A"]  # B emptied by cross-section dedupe
    assert edited.sections[0].claims == [Claim(id="c10", text="Same claim.", citations=["a1"])]
    assert edited.watchlist == ["X"]


# --------------------------------------------------------------------------- #
# fact_check / judge
# --------------------------------------------------------------------------- #

SOURCES = {"a1": ARTICLES["acq"]["title"] + "\n" + ARTICLES["acq"]["text"]}


@pytest.mark.parametrize(
    ("text", "kind", "citations", "supported", "reason"),
    [
        ("Meridian Cloud agreed to acquire Vireo Analytics for $1.2 billion in cash.", "fact", ["a1"], True, "numbers match"),
        ("Meridian Cloud agreed to acquire Vireo Analytics for $1.5 billion.", "fact", ["a1"], False, "1.5"),
        ("Bananas are an excellent source of potassium for athletes.", "fact", ["a1"], False, "claim terms"),
        ("Consolidation favours scale players.", "analysis", ["a1"], True, "analysis"),
        ("Meridian Cloud agreed to acquire Vireo Analytics.", "fact", ["zz"], False, "unknown source"),
        ("Meridian Cloud agreed to acquire Vireo Analytics.", "fact", [], False, "no sources"),
    ],
)
def test_check_claim(text, kind, citations, supported, reason):
    verdict = check_claim(Claim(id="c1", text=text, citations=citations, kind=kind), SOURCES)
    assert verdict.supported is supported
    assert reason in verdict.reason


def test_fact_check_task(client):
    claims = [
        Claim(id="c1", text="Meridian Cloud agreed to acquire Vireo Analytics for $1.2 billion.", citations=["a1"]),
        Claim(id="c2", text="Vireo has 999 employees.", citations=["a1"]),
    ]
    report = client.complete(
        req("fact_check", FactCheckReport, claims=[c.model_dump() for c in claims], sources=SOURCES)
    ).parsed
    assert [(v.claim_id, v.supported) for v in report.verdicts] == [("c1", True), ("c2", False)]


def test_judge_scores_good_and_bad_briefs(client):
    records, summaries = _cycle(client)
    draft = client.complete(
        req(
            "reduce_brief", BriefDraft, topic="T",
            summaries=[s.model_dump(mode="json") for s in summaries],
            records=[r.model_dump(mode="json") for r in records], prior_context=[],
        )
    ).parsed
    sources = {a["id"]: a["title"] + "\n" + a["text"] for a in ARTICLES.values()}
    golden = [
        GoldenEvent(id="G2", description="Meridian buys Vireo", event_type="m_and_a",
                    companies=["Meridian Cloud", "Vireo Analytics"], keywords=["1.2 billion"]).model_dump(mode="json"),
        GoldenEvent(id="GX", description="Bastion Labs names a CEO", event_type="leadership",
                    companies=["Bastion Labs"], keywords=["CEO"]).model_dump(mode="json"),
    ]
    good = client.complete(req("judge", JudgeScore, brief=draft.model_dump(mode="json"), sources=sources)).parsed
    assert (good.factuality, good.coverage, good.consistency) == (5, 5, 5)
    assert good.rationale.strip()

    with_golden = client.complete(
        req("judge", JudgeScore, brief=draft.model_dump(mode="json"), sources=sources, golden=golden)
    ).parsed
    assert with_golden.missing_topics == ["Bastion Labs names a CEO"]
    assert with_golden.coverage < 5

    bad = draft.model_copy(deep=True)
    bad.sections[0].claims[0].text = "Meridian Cloud paid $9.9 billion for a bakery chain in Lisbon."
    bad.sections[0].claims.append(bad.sections[0].claims[0].model_copy(update={"id": "c99"}))
    poor = client.complete(req("judge", JudgeScore, brief=bad.model_dump(mode="json"), sources=sources)).parsed
    assert poor.factuality < 5 and poor.consistency < 5
    assert bad.sections[0].claims[0].id in poor.unsupported_claim_ids


# --------------------------------------------------------------------------- #
# agent_plan
# --------------------------------------------------------------------------- #


def _plan(client, question: str, **state) -> AgentDecision:
    base = {"steps": 0, "searched": False, "search_results": [], "summaries": [], "chart_path": None,
            "max_steps": 8, "wants_chart": False}
    base.update(state)
    return client.complete(req("agent_plan", AgentDecision, question=question, state=base)).parsed


def test_agent_plan_policy(client):
    q = "How is GPU pricing changing? Chart the sentiment."
    assert _plan(client, q).action == "search"
    assert _plan(client, q).query == q

    results = [{"article_id": f"a{i}", "title": f"t{i}", "score": 1 - i / 10} for i in range(6)]
    d = _plan(client, q, steps=1, searched=True, search_results=results)
    assert d.action == "summarise" and d.article_ids == ["a0", "a1", "a2", "a3"]

    record = extract(client, ARTICLES["pricing"])
    summary = summarise(client, ARTICLES["pricing"], record).model_dump(mode="json")
    d = _plan(client, q, steps=2, searched=True, search_results=results, summaries=[summary], wants_chart=True)
    assert d.action == "chart" and d.chart_dimension == "sentiment"

    d = _plan(client, q, steps=3, searched=True, search_results=results, summaries=[summary],
              wants_chart=True, chart_path="chart.png")
    assert d.action == "finish"
    assert d.citations == ["a_price"] and "[a_price]" in d.answer


@pytest.mark.parametrize(
    ("question", "dimension"),
    [("Which categories grew?", "category"), ("Plot events by type", "event_type"), ("Who is most active?", "company")],
)
def test_agent_chart_dimension(client, question, dimension):
    d = _plan(client, question, searched=True, summaries=[{"article_id": "a1", "summary": "S.", "key_points": [],
              "why_it_matters": "W", "companies": [], "event_types": []}], wants_chart=True)
    assert d.chart_dimension == dimension


def test_agent_plan_stops_at_step_budget_and_without_results(client):
    assert _plan(client, "q", steps=7, max_steps=8).action == "finish"
    d = _plan(client, "q", steps=1, searched=True, search_results=[])
    assert d.action == "finish" and d.citations == [] and d.answer


# --------------------------------------------------------------------------- #
# errors / text tasks
# --------------------------------------------------------------------------- #


def test_errors(client):
    with pytest.raises(LLMError, match="no handler"):
        client.complete(req("translate", ArticleRecord, article=ARTICLES["acq"]))
    with pytest.raises(LLMError, match="needs a response_model"):
        client.complete(LLMRequest(task="extract", system="s", prompt="p"))
    with pytest.raises(LLMError, match="context"):
        client.complete(req("extract", ArticleRecord))


def test_free_text_task_is_deterministic(client):
    request = LLMRequest(task="note", system="s", prompt="<doc>Northwind cut prices. Rivals may follow. More.</doc>")
    first, second = client.complete(request), client.complete(request)
    assert first.text == second.text == "[offline:note] Northwind cut prices. Rivals may follow."
    assert first.parsed is None


# --------------------------------------------------------------------------- #
# Golden-set recall on the real fixture articles (skipped until they exist)
# --------------------------------------------------------------------------- #


def _load_fixture_articles() -> list[dict]:
    articles = []
    for path in sorted(FIXTURE_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for i, item in enumerate(data if isinstance(data, list) else [data]):
            articles.append({**item, "id": item.get("id") or f"{path.stem}_{i}"})
    return articles


def _event_matches(golden: GoldenEvent, event) -> bool:
    if event.event_type != golden.event_type:
        return False
    names = [c.lower() for c in event.companies]
    if not any(g.lower() in n or n in g.lower() for g in golden.companies for n in names):
        return False
    haystack = f"{event.headline} {event.evidence_quote}".lower()
    return not golden.keywords or any(k.lower() in haystack for k in golden.keywords)


@pytest.mark.skipif(
    not (list(FIXTURE_DIR.glob("*.json")) and ENTITIES_FILE.exists() and GOLDEN_FILE.exists()),
    reason="fixture articles / entities / golden set not written yet",
)
def test_golden_extraction_recall_on_fixtures():
    client = OfflineLLMClient(entities_path=ENTITIES_FILE)
    articles = _load_fixture_articles()
    records = [extract(client, a) for a in articles]
    for article, record in zip(articles, records):
        for event in record.events:
            assert event.evidence_quote in article["text"]
    golden = [GoldenEvent.model_validate(g) for g in yaml.safe_load(GOLDEN_FILE.read_text(encoding="utf-8"))["events"]]
    caught = {g.id for g in golden if any(_event_matches(g, e) for r in records for e in r.events)}
    missed = [g.id for g in golden if g.id not in caught]
    assert not [g.id for g in golden if g.must_catch and g.id not in caught], f"missed must-catch events: {missed}"
    assert len(caught) / len(golden) >= 0.9, f"missed: {missed}"


# --------------------------------------------------------------------------- #
# Factory + usage tracker
# --------------------------------------------------------------------------- #


def test_build_llm_offline_without_cache(settings):
    llm = build_llm(settings, use_cache=False)
    assert isinstance(llm, UsageTracker)
    assert isinstance(llm.inner, OfflineLLMClient)
    assert llm.provider == "offline" and llm.model == "offline-heuristic-v1"
    bare = build_llm(settings, use_cache=False, track=False, model="claude-opus-5-5")
    assert isinstance(bare, OfflineLLMClient) and bare.model == "offline-heuristic-v1"


def test_build_llm_anthropic_is_lazy(settings):
    llm = build_llm(settings, provider="anthropic", model="claude-opus-5-5", use_cache=False, track=False)
    assert isinstance(llm, AnthropicLLMClient)
    assert llm.model == "claude-opus-5-5" and llm._client is None  # no SDK client until first call
    assert llm.max_tokens == settings.llm.max_tokens


def test_build_llm_unknown_provider(settings):
    with pytest.raises(ValueError, match="unknown LLM provider"):
        build_llm(settings, provider="mystery")


def test_build_llm_with_prompt_cache(settings):
    pytest.importorskip("marketbrief.llm.cache")
    pytest.importorskip("marketbrief.embeddings")
    from marketbrief.llm.cache import CachedLLMClient

    llm = build_llm(settings, use_cache=True)
    assert isinstance(llm.inner, CachedLLMClient)
    request = req("extract", ArticleRecord, article=ARTICLES["acq"])
    first, second = llm.complete(request), llm.complete(request)
    assert not first.cached and second.cached
    assert second.parsed == first.parsed
    snap = llm.snapshot()
    assert (snap["calls"], snap["cache_hits"], snap["cache_misses"]) == (2, 1, 1)


class _FakeInner:
    provider = "fake"
    model = "fake-1"

    def __init__(self) -> None:
        self.n = 0
        self._lock = threading.Lock()

    def complete(self, request: LLMRequest) -> LLMResult:
        if request.task == "boom":
            raise LLMError("boom")
        with self._lock:
            self.n += 1
            cached = self.n % 2 == 0
        return LLMResult(text="x", usage=Usage(input_tokens=10, output_tokens=2, cache_read_input_tokens=1), cached=cached)


def test_usage_tracker_counts_concurrently():
    tracker = UsageTracker(_FakeInner())
    request = LLMRequest(task="extract", system="s", prompt="p")
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda _: tracker.complete(request), range(200)))
    with pytest.raises(LLMError):
        tracker.complete(LLMRequest(task="boom", system="s", prompt="p"))
    snap = tracker.snapshot()
    assert snap["calls"] == snap["llm_calls"] == 201
    assert snap["errors"] == 1
    assert snap["cache_hits"] + snap["cache_misses"] == 200 and snap["cache_hits"] == 100
    assert snap["input_tokens"] == snap["usage"]["input_tokens"] == 2000
    assert snap["output_tokens"] == 400 and snap["cache_read_input_tokens"] == 200
    assert snap["by_task"] == {"extract": 200, "boom": 1}
    assert (tracker.provider, tracker.model) == ("fake", "fake-1")
    tracker.reset()
    assert tracker.snapshot()["calls"] == 0
