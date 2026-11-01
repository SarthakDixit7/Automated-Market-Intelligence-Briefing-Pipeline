"""LLM-as-a-judge prompt + scoring, and the prompt-versioned eval runner."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from marketbrief.eval.judge import judge_brief, judge_mean
from marketbrief.eval.runner import EvalResult, evaluate_brief, load_history, run_eval
from marketbrief.llm.base import LLMError, LLMRequest, LLMResult, StructuredOutputError
from marketbrief.prompts.common import PROMPT_REGISTRY, prompt_version
from marketbrief.prompts.judge import JUDGE_SYSTEM, judge_prompt
from marketbrief.schemas import (
    Article,
    ArticleRecord,
    Brief,
    CompanyMention,
    EventType,
    ExtractedEvent,
    GoldenEvent,
    JudgeScore,
    Sentiment,
)

GOLDEN_YAML = """\
events:
  - id: G1
    description: Aurora Silicon raises $410 million Series D
    event_type: funding
    companies: [Aurora Silicon]
    keywords: ["Series D", "410"]
  - id: G2
    description: Meridian Cloud agrees to acquire Vireo Analytics for $1.2 billion
    event_type: m_and_a
    companies: [Meridian Cloud, Vireo Analytics]
    keywords: ["1.2 billion", "acquire"]
  - id: G3
    description: Northwind Compute cuts GPU-hour pricing by 22%
    event_type: pricing
    companies: [Northwind Compute]
    keywords: ["22%"]
    must_catch: false
"""


class JudgeLLM:
    """Returns scripted JudgeScore objects (or raises) for the ``judge`` task."""

    provider = "scripted"
    model = "judge-v1"

    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResult:
        self.requests.append(request)
        assert request.task == "judge"
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return LLMResult(text=out.model_dump_json() if out else "", parsed=out)


def score(f=5, c=4, k=4, unsupported=(), rationale="Claims match their sources.") -> JudgeScore:
    return JudgeScore(
        factuality=f, coverage=c, consistency=k, unsupported_claim_ids=list(unsupported), rationale=rationale
    )


def _article(aid: str, day: int, title: str, text: str) -> Article:
    return Article(
        id=aid,
        source_id="demo_wire",
        source_name="The Silicon Wire",
        title=title,
        published_at=datetime(2026, 10, day, 9, tzinfo=timezone.utc),
        text=text,
        content_hash=hashlib.sha1(text.encode()).hexdigest(),
    )


@pytest.fixture
def articles() -> dict[str, Article]:
    """Sources backing every claim in conftest's ``sample_brief``."""
    items = [
        _article(
            "a_sample000001", 6, "Aurora Silicon raises $410 million Series D",
            "Aurora Silicon said on Tuesday it has raised $410 million in a Series D round led by Halcyon "
            "Partners. Aurora Silicon's Series D was led by Halcyon Partners and values the chip designer "
            "at $3.1 billion.",
        ),
        _article(
            "a_sample000002", 8, "Meridian Cloud to acquire Vireo Analytics for $1.2 billion",
            "Meridian Cloud agreed to acquire Vireo Analytics for $1.2 billion in cash, the companies said. "
            "The deal is expected to close in the first quarter.",
        ),
        _article(
            "a_sample000003", 9, "Northwind Compute cuts GPU-hour prices by 22%",
            "Northwind Compute cut H-class GPU-hour pricing by 22%, effective immediately, as competition "
            "among GPU clouds intensifies.",
        ),
    ]
    return {a.id: a for a in items}


def _record(aid: str, event_type: EventType, companies: list[str], quote: str) -> ArticleRecord:
    return ArticleRecord(
        article_id=aid,
        companies=[CompanyMention(name=c) for c in companies],
        events=[ExtractedEvent(event_type=event_type, headline=quote, companies=companies, evidence_quote=quote)],
        categories=[],
        sentiment=Sentiment.NEUTRAL,
        confidence=0.9,
        key_facts=[],
    )


@pytest.fixture
def records() -> list[ArticleRecord]:
    return [
        _record("a_sample000001", EventType.FUNDING, ["Aurora Silicon"], "Aurora Silicon raised $410 million in a Series D."),
        _record("a_sample000002", EventType.M_AND_A, ["Meridian Cloud", "Vireo Analytics"],
                "Meridian Cloud agreed to acquire Vireo Analytics for $1.2 billion."),
        _record("a_sample000003", EventType.PRICING, ["Northwind Compute"], "Northwind cut GPU-hour pricing by 22%."),
    ]


@pytest.fixture
def golden_path(tmp_path) -> Path:
    path = tmp_path / "golden.yaml"
    path.write_text(GOLDEN_YAML, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #


def test_judge_prompt_lists_claims_sources_and_golden(sample_brief: Brief, articles):
    golden = [GoldenEvent(id="G1", description="Aurora raises", event_type=EventType.FUNDING, companies=["Aurora Silicon"])]
    sources = {aid: a.text for aid, a in articles.items()}
    pair = judge_prompt(sample_brief, sources, golden)
    assert pair.system == JUDGE_SYSTEM
    for cid in ("c1", "c2", "c3", "c4", "c5"):
        assert f'<claim id="{cid}"' in pair.prompt
    assert 'kind="analysis"' in pair.prompt  # c5
    assert 'cites="a_sample000002,a_sample000003"' in pair.prompt
    assert '<source id="a_sample000001">' in pair.prompt
    assert '<event id="G1" type="funding" must_cover="true">' in pair.prompt
    assert "&lt;effective immediately&gt;" in pair.prompt  # claim text is escaped
    assert "Updates: the reported Meridian-Vireo talks" in pair.prompt  # continuity shown
    assert pair.prompt.rstrip().endswith("Score this brief.")


def test_judge_prompt_truncates_long_sources(sample_brief: Brief):
    pair = judge_prompt(sample_brief, {"a_long": "word " * 5000})
    assert "[...]" in pair.prompt
    assert "<must_cover_events>" not in pair.prompt


def test_judge_system_prompt_is_stable_and_registered():
    assert PROMPT_REGISTRY["judge.system"] == JUDGE_SYSTEM
    assert "judge.template" in PROMPT_REGISTRY
    for anchor in ("factuality", "coverage", "consistency", "5 ", "1 "):
        assert anchor in JUDGE_SYSTEM
    assert len(prompt_version()) == 12


# --------------------------------------------------------------------------- #
# judge_brief
# --------------------------------------------------------------------------- #


def test_judge_brief_builds_request_and_filters_unknown_claim_ids(settings, sample_brief, articles):
    llm = JudgeLLM(score(unsupported=["c4", "c99"]))
    golden = [GoldenEvent(id="G1", description="x", event_type=EventType.FUNDING, companies=["Aurora Silicon"])]
    result = judge_brief(sample_brief, articles, llm, settings, golden=golden)

    assert result.unsupported_claim_ids == ["c4"]
    req = llm.requests[0]
    assert req.response_model is JudgeScore
    assert req.effort == settings.llm.effort["judge"]
    assert req.system == JUDGE_SYSTEM
    assert set(req.context) == {"brief", "sources", "golden"}
    assert req.context["sources"]["a_sample000002"].startswith("Meridian Cloud to acquire")
    assert req.context["golden"][0]["id"] == "G1"
    assert req.context["brief"]["id"] == sample_brief.id


def test_judge_brief_re_asks_when_output_missing(settings, sample_brief, articles):
    llm = JudgeLLM(None, score())
    assert judge_brief(sample_brief, articles, llm, settings).factuality == 5
    assert len(llm.requests) == 2
    assert "<validation_feedback>" in llm.requests[1].prompt


def test_judge_brief_gives_up_after_retries(settings, sample_brief, articles):
    llm = JudgeLLM(None, None, None)
    with pytest.raises(StructuredOutputError):
        judge_brief(sample_brief, articles, llm, settings)


def test_judge_mean():
    assert judge_mean(score(5, 4, 3)) == pytest.approx(4.0)
    assert judge_mean(score(5, 5, 4)) == pytest.approx(4.6667, abs=1e-4)


def test_judge_score_requires_rationale():
    with pytest.raises(ValueError):
        score(rationale="   ")


def test_offline_judge_scores_sample_brief(settings, offline_llm, sample_brief, articles):
    result = judge_brief(sample_brief, articles, offline_llm, settings)
    assert 1 <= result.factuality <= 5 and result.rationale.strip()


# --------------------------------------------------------------------------- #
# evaluate_brief
# --------------------------------------------------------------------------- #


def test_evaluate_brief_passes_good_brief(settings, sample_brief, articles, records, golden_path):
    result = evaluate_brief(sample_brief, articles, records, settings, llm=JudgeLLM(score()), golden_path=golden_path)
    assert isinstance(result, EvalResult)
    assert result.failures == []
    assert result.passed
    assert result.recall.extraction_recall == 1.0
    assert result.recall.brief_recall == 1.0
    assert result.citation is not None and result.citation.passed
    assert result.judge is not None and judge_mean(result.judge) >= settings.eval.min_judge_score
    assert result.prompt_version == prompt_version()
    assert result.report_path is None


def test_evaluate_brief_reports_every_failure(settings, sample_brief, articles, golden_path):
    llm = JudgeLLM(score(2, 2, 3, rationale="Unsupported claims."))
    result = evaluate_brief(sample_brief, list(articles.values()), [], settings, llm=llm, golden_path=golden_path)
    assert not result.passed
    text = "\n".join(result.failures)
    assert "extraction recall 0.00" in text
    assert "missed must-catch events: G1, G2" in text  # G3 is optional
    assert "judge mean 2.33" in text
    assert "brief recall" not in text  # the brief itself still states all three events


def test_evaluate_brief_flags_citation_errors(settings, sample_brief, articles, records, golden_path):
    bad = sample_brief.model_copy(deep=True)
    bad.sections[0].claims[0].citations = ["a_missing00000"]
    result = evaluate_brief(bad, articles, records, settings, golden_path=golden_path, use_judge=False)
    assert result.judge is None
    assert result.citation is not None and not result.citation.passed
    assert any(f.startswith("citation validation failed") for f in result.failures)


def test_evaluate_brief_survives_judge_errors(settings, sample_brief, articles, records, golden_path):
    llm = JudgeLLM(LLMError("overloaded"))
    result = evaluate_brief(sample_brief, articles, records, settings, llm=llm, golden_path=golden_path)
    assert result.judge is None
    assert result.failures == ["judge failed: LLMError: overloaded"]


# --------------------------------------------------------------------------- #
# run_eval
# --------------------------------------------------------------------------- #


class ExtractAndJudgeLLM:
    """Offline extraction + a scripted judge, the way run_eval shares one client."""

    provider = "mixed"
    model = "mixed"

    def __init__(self, offline, judge_score: JudgeScore):
        self.offline = offline
        self.judge_score = judge_score
        self.tasks: list[str] = []

    def complete(self, request: LLMRequest) -> LLMResult:
        self.tasks.append(request.task)
        if request.task == "judge":
            return LLMResult(text=self.judge_score.model_dump_json(), parsed=self.judge_score)
        return self.offline.complete(request)


def test_run_eval_on_saved_brief_writes_report_and_history(settings, offline_llm, sample_brief, articles, golden_path):
    from marketbrief.ingest.store import ArticleStore

    settings.paths.golden_file = str(golden_path)
    ArticleStore(settings.path(settings.paths.store_dir) / "articles.jsonl").upsert(list(articles.values()))
    brief_path = settings.path(settings.paths.briefs_dir) / f"{sample_brief.id}.json"
    brief_path.parent.mkdir(parents=True, exist_ok=True)
    brief_path.write_text(sample_brief.model_dump_json(indent=2), encoding="utf-8")

    llm = ExtractAndJudgeLLM(offline_llm, score())
    first = run_eval(settings, brief_path=brief_path, llm=llm)
    assert "extract" in llm.tasks and llm.tasks[-1] == "judge"
    assert first.report_path is not None and first.report_path.exists()
    assert first.report_path.name == f"eval_{first.prompt_version}_{sample_brief.id}.json"
    report = json.loads(first.report_path.read_text(encoding="utf-8"))
    assert report["brief_id"] == sample_brief.id
    assert report["passed"] == first.passed
    assert report["judge_mean"] == pytest.approx(judge_mean(score()))
    assert report["recall"]["total"] == 3

    run_eval(settings, brief_path=brief_path, llm=ExtractAndJudgeLLM(offline_llm, score()), use_judge=False)
    history = load_history(settings)
    assert len(history) == 2
    assert history[0]["prompt_version"] == first.prompt_version
    assert history[0]["judge_mean"] is not None and history[1]["judge_mean"] is None
    assert {"extraction_recall", "brief_recall", "passed", "brief_id"} <= set(history[0])


def test_run_eval_missing_brief_raises(settings, tmp_path):
    with pytest.raises(FileNotFoundError):
        run_eval(settings, brief_path=tmp_path / "nope.json", use_judge=False)
