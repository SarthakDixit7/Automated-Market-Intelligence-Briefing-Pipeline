"""Analyst agent: tools, planner loop, HITL checkpoint and resume.

The planner LLM is a scripted fake that replays a list of AgentDecision
objects, so these tests pin down the agent's control flow exactly.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pytest

from marketbrief.agent import tools
from marketbrief.agent.analyst import (
    AnalystState,
    clean_citations,
    resume_analyst,
    run_analyst,
)
from marketbrief.agent.graph import GraphError
from marketbrief.llm.base import LLMError, LLMRequest, LLMResult, Usage
from marketbrief.prompts.agent import AGENT_SYSTEM, plan_prompt
from marketbrief.prompts.common import PROMPT_REGISTRY
from marketbrief.schemas import (
    AgentDecision,
    Article,
    ArticleRecord,
    ArticleSummary,
    CompanyMention,
    EventType,
    ExtractedEvent,
    Sentiment,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
A_AURORA, A_MERIDIAN, A_INQUIRY, A_NORTHWIND = (f"a_00000000000{i}" for i in range(1, 5))


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #


class WordEmbedder:
    """Tiny deterministic bag-of-words embedder (keeps tests independent of embeddings.py)."""

    name = "test-words"
    dim = 256

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        rows = []
        for text in texts:
            vec = np.zeros(self.dim, dtype=np.float32)
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                if len(word) > 2:
                    vec[int(hashlib.sha1(word.encode()).hexdigest(), 16) % self.dim] += 1.0
            norm = np.linalg.norm(vec)
            rows.append(vec / norm if norm else vec)
        return np.vstack(rows) if rows else np.zeros((0, self.dim), dtype=np.float32)


class ScriptedLLM:
    """Replays AgentDecisions for ``agent_plan``; templated ``map_summary`` output."""

    provider = "scripted"
    model = "scripted-v1"

    def __init__(self, decisions: Iterable[AgentDecision]):
        self._decisions = iter(decisions)
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResult:
        self.requests.append(request)
        if request.task == "agent_plan":
            try:
                decision = next(self._decisions)
            except StopIteration:
                raise AssertionError("planner called more often than scripted") from None
            return LLMResult(text=decision.model_dump_json(), parsed=decision, usage=Usage(input_tokens=10))
        if request.task == "map_summary":
            article, record = request.context["article"], request.context["record"]
            summary = ArticleSummary(
                article_id=article["id"],
                summary=f"{article['title']}. Reported by {article['source_name']}.",
                key_points=record["key_facts"][:4],
                why_it_matters="It shifts the competitive picture.",
                companies=[c["name"] for c in record["companies"]],
                event_types=[e["event_type"] for e in record["events"]],
            )
            return LLMResult(text=summary.model_dump_json(), parsed=summary)
        raise LLMError(f"unexpected task {request.task}")

    def plan_requests(self) -> list[LLMRequest]:
        return [r for r in self.requests if r.task == "agent_plan"]


class FailingLLM:
    provider = "failing"
    model = "failing"

    def complete(self, request: LLMRequest) -> LLMResult:
        raise LLMError("service unavailable")


def search(query: str) -> AgentDecision:
    return AgentDecision(thought="look", action="search", query=query)


def summarise(*ids: str) -> AgentDecision:
    return AgentDecision(thought="read", action="summarise", article_ids=list(ids))


def finish(answer: str, *citations: str) -> AgentDecision:
    return AgentDecision(thought="done", action="finish", answer=answer, citations=list(citations))


# --------------------------------------------------------------------------- #
# Corpus
# --------------------------------------------------------------------------- #


def _article(aid: str, day: int, title: str, text: str, source: str) -> Article:
    return Article(
        id=aid,
        source_id="test",
        source_name=source,
        title=title,
        published_at=datetime(2026, 10, day, 9, tzinfo=timezone.utc),
        text=text,
        content_hash=hashlib.sha1(text.encode()).hexdigest(),
    )


def _record(aid: str, event_type: EventType, companies: list[str], quote: str, category: str, sentiment: Sentiment) -> ArticleRecord:
    return ArticleRecord(
        article_id=aid,
        companies=[CompanyMention(name=c, role="subject" if i == 0 else "counterparty") for i, c in enumerate(companies)],
        events=[
            ExtractedEvent(event_type=event_type, headline=quote[:150], companies=companies, evidence_quote=quote)
        ],
        categories=[category],
        sentiment=sentiment,
        confidence=0.9,
        key_facts=[quote],
    )


@pytest.fixture
def corpus() -> tuple[list[Article], list[ArticleRecord]]:
    aurora = "Aurora Silicon raised $410 million in a Series D round to scale production of its inference chips."
    meridian = "Meridian Cloud agreed to acquire Vireo Analytics for $1.2 billion in cash."
    inquiry = "The European Commission opened an inquiry into Meridian Cloud data egress fees."
    northwind = "Northwind Compute cut H-class GPU-hour pricing by 22% across all regions."
    articles = [
        _article(A_AURORA, 6, "Aurora Silicon raises $410 million Series D", aurora, "The Silicon Wire"),
        _article(A_MERIDIAN, 8, "Meridian Cloud to acquire Vireo Analytics", meridian, "Cloud Ledger"),
        _article(A_INQUIRY, 16, "EU opens inquiry into Meridian Cloud egress fees", inquiry, "Market Pulse Daily"),
        _article(A_NORTHWIND, 9, "Northwind Compute cuts GPU prices by 22%", northwind, "Datacenter Dispatch"),
    ]
    records = [
        _record(A_AURORA, EventType.FUNDING, ["Aurora Silicon"], aurora, "ai chips", Sentiment.POSITIVE),
        _record(A_MERIDIAN, EventType.M_AND_A, ["Meridian Cloud", "Vireo Analytics"], meridian, "hyperscale cloud", Sentiment.POSITIVE),
        _record(A_INQUIRY, EventType.REGULATORY, ["Meridian Cloud"], inquiry, "hyperscale cloud", Sentiment.NEGATIVE),
        _record(A_NORTHWIND, EventType.PRICING, ["Northwind Compute"], northwind, "gpu cloud", Sentiment.MIXED),
    ]
    return articles, records


def run(question, settings, corpus, llm, **kwargs):
    articles, records = corpus
    return run_analyst(question, settings, llm=llm, articles=articles, records=records, embedder=WordEmbedder(), **kwargs)


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #


def test_search_ranks_by_similarity_with_company_and_event_boosts(corpus):
    articles, records = corpus
    hits = tools.search_articles("Who raised funding? Aurora Silicon", articles, records, WordEmbedder(), k=3)
    assert hits[0]["article_id"] == A_AURORA
    assert set(hits[0]) == {"article_id", "title", "source", "published_at", "score"}
    assert hits[0]["published_at"].startswith("2026-10-06")
    assert [h["score"] for h in hits] == sorted((h["score"] for h in hits), reverse=True)
    assert len(hits) <= 3


def test_search_company_boost_separates_same_company_stories(corpus):
    articles, records = corpus
    hits = tools.search_articles("Meridian acquisition", articles, records, WordEmbedder(), k=4)
    assert hits[0]["article_id"] == A_MERIDIAN  # company + m_and_a event boost
    assert A_INQUIRY in [h["article_id"] for h in hits[:2]]  # company boost only


def test_search_edge_cases(corpus):
    articles, records = corpus
    assert tools.search_articles("   ", articles, records, WordEmbedder()) == []
    assert tools.search_articles("anything", [], records, WordEmbedder()) == []
    assert tools.search_articles("Aurora", articles, records, WordEmbedder(), k=0) == []


def test_count_dimension(corpus):
    _, records = corpus
    assert tools.count_dimension(records, "company")["Meridian Cloud"] == 2
    assert tools.count_dimension(records, "event_type") == {
        "funding": 1, "m_and_a": 1, "regulatory": 1, "pricing": 1,
    }
    assert tools.count_dimension(records, "category")["hyperscale cloud"] == 2
    assert tools.count_dimension(records, "sentiment")["positive"] == 2
    with pytest.raises(ValueError, match="unknown chart dimension"):
        tools.count_dimension(records, "colour")


@pytest.mark.parametrize("dimension", tools.CHART_DIMENSIONS)
def test_chart_event_counts_writes_png(tmp_path, corpus, dimension):
    _, records = corpus
    out = tools.chart_event_counts(records, dimension, tmp_path / "charts" / f"{dimension}.png", title="Test")
    assert out.exists() and out.read_bytes()[:8] == PNG_MAGIC


def test_chart_with_no_records_still_renders(tmp_path):
    out = tools.chart_event_counts([], "company", tmp_path / "empty.png", title="Nothing")
    assert out.read_bytes()[:8] == PNG_MAGIC


def test_summarise_articles_skips_unknown_ids(settings, corpus):
    articles, records = corpus
    llm = ScriptedLLM([])
    out = tools.summarise_articles(
        [A_AURORA, "a_unknown00000", A_AURORA],
        {a.id: a for a in articles},
        {r.article_id: r for r in records},
        llm,
        settings,
    )
    assert [s.article_id for s in out] == [A_AURORA]
    assert tools.summarise_articles([], {}, {}, llm, settings) == []


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #


def test_plan_prompt_renders_state_and_is_registered():
    state = AnalystState(
        question="Who raised money?",
        steps=2,
        searched=True,
        search_results=[{"article_id": A_AURORA, "title": 'Aurora "raises"', "source": "Wire", "score": 0.81}],
        summaries=[
            ArticleSummary(
                article_id=A_AURORA, summary="Aurora raised $410M.", key_points=["Series D"],
                why_it_matters="Capital for chips.", companies=["Aurora Silicon"], event_types=[EventType.FUNDING],
            )
        ],
    )
    pair = plan_prompt(state)
    assert pair.system == AGENT_SYSTEM
    assert "<question>Who raised money?</question>" in pair.prompt
    assert f'id="{A_AURORA}"' in pair.prompt and 'score="0.810"' in pair.prompt
    assert "&quot;raises&quot;" in pair.prompt  # attribute-safe escaping
    assert "Aurora raised $410M." in pair.prompt
    assert PROMPT_REGISTRY["agent.system"] == AGENT_SYSTEM
    assert "agent.plan" in PROMPT_REGISTRY


# --------------------------------------------------------------------------- #
# Agent runs
# --------------------------------------------------------------------------- #


def test_non_sensitive_question_completes_without_review(settings, corpus):
    llm = ScriptedLLM([
        search("Aurora Silicon funding"),
        summarise(A_AURORA),
        finish(f"Aurora Silicon raised $410 million in a Series D [{A_AURORA}].", A_AURORA),
    ])
    result = run("Who raised funding this month?", settings, corpus, llm, thread_id="t-funding")

    assert result.status == "completed"
    assert result.thread_id == "t-funding"
    assert result.trace == ["plan", "search", "plan", "summarise", "plan", "draft_answer"]
    state = result.state
    assert state.answer.startswith("Aurora Silicon raised $410 million")
    assert state.citations == [A_AURORA]
    assert not state.sensitive and state.delivered and state.approved is None
    assert state.steps == 3
    assert [s.article_id for s in state.summaries] == [A_AURORA]
    assert state.search_results[0]["article_id"] == A_AURORA

    # The planner gets the structured context the offline provider reads.
    plans = llm.plan_requests()
    assert len(plans) == 3
    ctx = plans[1].context
    assert ctx["question"] == "Who raised funding this month?"
    assert set(ctx["state"]) == {"steps", "max_steps", "searched", "search_results", "summaries", "chart_path", "wants_chart"}
    assert ctx["state"]["searched"] is True and ctx["state"]["steps"] == 1
    assert set(ctx["state"]["search_results"][0]) == {"article_id", "title", "score"}
    assert plans[0].response_model is AgentDecision
    assert plans[0].effort == settings.llm.effort["agent_plan"]
    assert plans[0].system == AGENT_SYSTEM


def test_sensitive_answer_pauses_for_review_then_resumes(settings, corpus):
    llm = ScriptedLLM([
        search("Meridian Cloud inquiry"),
        summarise(A_INQUIRY),
        finish(f"The European Commission opened an inquiry into Meridian Cloud [{A_INQUIRY}].", A_INQUIRY),
    ])
    paused = run("Is Meridian under regulatory scrutiny?", settings, corpus, llm, thread_id="t-sens")

    assert paused.status == "interrupted"
    assert paused.trace[-1] == "draft_answer"
    assert paused.state.sensitive and not paused.state.delivered
    assert any("regulatory" in r for r in paused.state.sensitive_reasons)
    assert any("inquiry" in r for r in paused.state.sensitive_reasons)
    ckpt = settings.path(settings.paths.agent_dir) / "t-sens.json"
    data = json.loads(ckpt.read_text(encoding="utf-8"))
    assert data["next_node"] == "human_review" and data["status"] == "interrupted"

    # A later process resumes with only the thread id (no corpus, no LLM).
    done = resume_analyst("t-sens", settings, approved=True, notes="checked against source")
    assert done.status == "completed"
    assert done.trace[-1] == "human_review"
    assert done.state.approved is True and done.state.delivered
    assert done.state.reviewer_notes == "checked against source"
    assert done.state.answer == paused.state.answer
    assert "human_review: approved (checked against source)" in done.state.log

    with pytest.raises(GraphError, match="already completed"):
        resume_analyst("t-sens", settings, approved=False)


def test_rejected_review_withholds_delivery(settings, corpus):
    llm = ScriptedLLM([
        search("Meridian"), summarise(A_INQUIRY),
        finish(f"Meridian faces an EU inquiry [{A_INQUIRY}].", A_INQUIRY),
    ])
    run("Meridian news?", settings, corpus, llm, thread_id="t-rej")
    done = resume_analyst("t-rej", settings, approved=False, notes="too speculative")
    assert done.state.approved is False and not done.state.delivered


def test_auto_approve_resumes_immediately(settings, corpus):
    llm = ScriptedLLM([
        search("Meridian"), summarise(A_INQUIRY),
        finish(f"Meridian faces an EU inquiry [{A_INQUIRY}].", A_INQUIRY),
    ])
    result = run("Meridian news?", settings, corpus, llm, auto_approve=False)
    assert result.status == "completed"
    assert result.thread_id.startswith("t_")
    assert result.state.approved is False and result.state.reviewer_notes == "auto-rejected"
    assert result.trace[-2:] == ["draft_answer", "human_review"]


def test_resume_unknown_thread_raises(settings):
    with pytest.raises(GraphError, match="no paused analyst thread"):
        resume_analyst("does-not-exist", settings, approved=True)


def test_unknown_citations_are_stripped(settings, corpus):
    llm = ScriptedLLM([
        search("Aurora"), summarise(A_AURORA),
        finish(f"Aurora raised $410M [{A_AURORA}] and rivals followed [a_deadbeef0000].", A_AURORA, "a_deadbeef0000"),
    ])
    state = run("Aurora?", settings, corpus, llm).state
    assert state.citations == [A_AURORA]
    assert "a_deadbeef0000" not in state.answer
    assert state.answer.endswith("rivals followed.")


def test_clean_citations_keeps_non_citation_brackets():
    known = {A_AURORA, A_MERIDIAN}
    text, cites = clean_citations(
        f"Deal [sic] closed [{A_MERIDIAN}, a_ffffffffffff]. Funding too [{A_AURORA}].", [A_AURORA], known
    )
    assert text == f"Deal [sic] closed [{A_MERIDIAN}]. Funding too [{A_AURORA}]."
    assert cites == [A_MERIDIAN, A_AURORA]


def test_chart_requested_in_question_is_drawn(settings, corpus):
    llm = ScriptedLLM([
        search("Meridian"),
        summarise(A_MERIDIAN, A_INQUIRY),
        AgentDecision(thought="plot", action="chart", chart_dimension="event_type"),
        finish(f"Meridian agreed to buy Vireo [{A_MERIDIAN}].", A_MERIDIAN),
    ])
    result = run("Chart Meridian's events this month", settings, corpus, llm, thread_id="t-chart")
    assert result.state.wants_chart is True
    assert result.trace.count("chart") == 1
    chart = Path(result.state.chart_path)
    assert chart.name == "t-chart_event_type.png"
    assert chart.read_bytes()[:8] == PNG_MAGIC
    assert chart.parent == settings.path(settings.paths.agent_dir) / "charts"


def test_wants_chart_flag_overrides_question(settings, corpus):
    llm = ScriptedLLM([finish("Nothing to add.")])
    assert run("Chart it", settings, corpus, llm, wants_chart=False).state.wants_chart is False


def test_invalid_decision_is_re_asked_with_feedback(settings, corpus):
    llm = ScriptedLLM([
        search("Aurora"),
        summarise("a_notarealid00"),  # rejected by the validator -> re-ask
        summarise(A_AURORA),
        finish(f"Aurora raised money [{A_AURORA}].", A_AURORA),
    ])
    result = run("Aurora?", settings, corpus, llm)
    assert result.status == "completed"
    assert result.trace == ["plan", "search", "plan", "summarise", "plan", "draft_answer"]
    plans = llm.plan_requests()
    assert len(plans) == 4
    assert "<validation_feedback>" in plans[2].prompt and A_AURORA in plans[2].prompt


def test_step_budget_forces_a_final_answer(settings, corpus):
    llm = ScriptedLLM(itertools.repeat(search("Aurora")))
    result = run("Aurora?", settings, corpus, llm, max_steps=3)
    assert result.status == "completed"
    assert result.trace == ["plan", "search"] * 3 + ["plan", "draft_answer"]
    assert len(llm.plan_requests()) == 3  # the forced finish needs no LLM call
    assert result.state.answer and "could not find" in result.state.answer


def test_planner_failure_falls_back_to_available_evidence(settings, corpus):
    result = run("Anything?", settings, corpus, FailingLLM())
    assert result.status == "completed"
    assert result.trace == ["plan", "draft_answer"]
    assert "could not find" in result.state.answer
    assert result.state.citations == []


def test_run_analyst_rejects_empty_question(settings, corpus):
    with pytest.raises(ValueError):
        run("  ", settings, corpus, ScriptedLLM([]))


# --------------------------------------------------------------------------- #
# Offline provider end to end (real tools, offline planner)
# --------------------------------------------------------------------------- #


def test_offline_agent_end_to_end(settings, corpus):
    from marketbrief.llm.offline import OfflineLLMClient

    llm = OfflineLLMClient(entities_path=settings.path(settings.paths.entities_file))
    result = run("Chart funding and acquisition activity for Aurora Silicon and Meridian Cloud", settings, corpus, llm,
                 auto_approve=True)
    assert result.status == "completed"
    assert result.trace[:4] == ["plan", "search", "plan", "summarise"]
    assert "chart" in result.trace and result.state.chart_path
    assert result.state.answer and result.state.citations
    assert set(result.state.citations) <= {a.id for a in corpus[0]}
