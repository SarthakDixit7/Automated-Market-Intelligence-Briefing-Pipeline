"""Analyst agent: plan -> (search | summarise | chart)* -> answer -> human review.

The planner is an LLM call (task ``agent_plan``) that returns an
:class:`AgentDecision`; tool nodes run deterministic code and loop back to the
planner. Answers that touch sensitive events (regulatory, litigation,
leadership, layoffs, allegations) stop at a ``human_review`` checkpoint and
ship only after a reviewer resumes the thread with a decision.

Dependencies that must not be serialised into checkpoints (articles, records,
embedder, LLM client) are bound to the nodes through closures, see
:func:`build_analyst_graph`.
"""

from __future__ import annotations

import hashlib
import logging
import re
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from marketbrief.agent import tools
from marketbrief.agent.graph import END, CompiledGraph, GraphError, JsonCheckpointer, StateGraph
from marketbrief.llm.base import LLMClient, LLMError, LLMRequest
from marketbrief.llm.structured import structured_call
from marketbrief.prompts.agent import plan_prompt
from marketbrief.schemas import AgentDecision, Article, ArticleRecord, ArticleSummary, EventType

if TYPE_CHECKING:
    from marketbrief.config import Settings
    from marketbrief.embeddings import Embedder

log = logging.getLogger(__name__)

HUMAN_REVIEW = "human_review"
SENSITIVE_EVENT_TYPES = frozenset(
    {EventType.REGULATORY, EventType.LITIGATION, EventType.LEADERSHIP, EventType.LAYOFFS}
)
_SENSITIVE_WORDS = re.compile(
    r"\b(lawsuits?|su(?:es|ed|ing)|inquir(?:y|ies)|investigations?|layoffs?|laid off|allegations?|alleged(?:ly)?)\b",
    re.IGNORECASE,
)
_CHART_WORDS = re.compile(r"\b(chart|plot|graph|visuali[sz]e|visual)\b", re.IGNORECASE)
_BRACKET_IDS = re.compile(r"\[([^\[\]]+)\]")
_ARTICLE_ID = re.compile(r"^a_[0-9a-f]{6,}$")
MAX_SUMMARISE = 4


class AnalystState(BaseModel):
    """Everything the agent knows; checkpointed as JSON between steps."""

    question: str
    steps: int = 0
    max_steps: int = 8
    searched: bool = False
    search_results: list[dict[str, Any]] = Field(default_factory=list)
    summaries: list[ArticleSummary] = Field(default_factory=list)
    chart_path: str | None = None
    wants_chart: bool = False
    answer: str | None = None
    citations: list[str] = Field(default_factory=list)
    sensitive: bool = False
    sensitive_reasons: list[str] = Field(default_factory=list)
    approved: bool | None = None
    reviewer_notes: str = ""
    delivered: bool = False
    last_decision: AgentDecision | None = None
    log: list[str] = Field(default_factory=list)


@dataclass
class AnalystDeps:
    """Non-serialisable dependencies the graph nodes close over."""

    settings: "Settings"
    llm: LLMClient | None
    articles: list[Article]
    records: list[ArticleRecord]
    embedder: "Embedder | None"
    thread_id: str = "default"
    articles_by_id: dict[str, Article] = field(init=False)
    records_by_id: dict[str, ArticleRecord] = field(init=False)

    def __post_init__(self) -> None:
        self.articles_by_id = {a.id: a for a in self.articles}
        self.records_by_id = {r.article_id: r for r in self.records}

    @property
    def chart_dir(self) -> Path:
        return self.settings.path(self.settings.paths.agent_dir) / "charts"


@dataclass
class AnalystResult:
    state: AnalystState
    status: Literal["completed", "interrupted"]
    thread_id: str
    trace: list[str]


# --------------------------------------------------------------------------- #
# Planner helpers
# --------------------------------------------------------------------------- #


def plan_context(state: AnalystState) -> dict[str, Any]:
    """Structured mirror of the planner prompt (read by the offline provider)."""
    return {
        "question": state.question,
        "state": {
            "steps": state.steps,
            "max_steps": state.max_steps,
            "searched": state.searched,
            "search_results": [
                {"article_id": r.get("article_id"), "title": r.get("title"), "score": r.get("score")}
                for r in state.search_results
            ],
            "summaries": [s.model_dump(mode="json") for s in state.summaries],
            "chart_path": state.chart_path,
            "wants_chart": state.wants_chart,
        },
    }


def _decision_validator(state: AnalystState, known_ids: set[str]):
    """Re-ask feedback for decisions the tools could not execute."""
    result_ids = [str(r.get("article_id")) for r in state.search_results]

    def validate(decision: BaseModel) -> list[str]:
        assert isinstance(decision, AgentDecision)
        errors: list[str] = []
        if decision.action == "summarise":
            usable = [i for i in decision.article_ids if i in known_ids]
            if not usable:
                errors.append(
                    "summarise needs article_ids chosen from the search results "
                    f"({', '.join(result_ids) or 'none - search first or finish'})."
                )
        elif decision.action == "chart" and decision.chart_dimension is None:
            errors.append("chart needs chart_dimension: one of company, event_type, category, sentiment.")
        elif decision.action == "finish" and not (decision.answer or "").strip():
            errors.append("finish needs a non-empty answer.")
        return errors

    return validate


def fallback_answer(state: AnalystState, reason: str) -> AgentDecision:
    """Deterministic finish used when the step budget is spent or planning fails."""
    if not state.summaries:
        return AgentDecision(
            thought=reason,
            action="finish",
            answer="I could not find enough coverage in the current article set to answer this question.",
        )
    parts = []
    for s in state.summaries[:MAX_SUMMARISE]:
        lead = re.split(r"(?<=[.!?])\s+", s.summary.strip(), maxsplit=1)[0]
        parts.append(f"{lead} [{s.article_id}]")
    return AgentDecision(
        thought=reason,
        action="finish",
        answer=" ".join(parts),
        citations=[s.article_id for s in state.summaries[:MAX_SUMMARISE]],
    )


def clean_citations(answer: str, cited: list[str], known_ids: set[str]) -> tuple[str, list[str]]:
    """Keep only known article ids, in order of first appearance.

    Bracketed article-id references to unknown ids are removed from the text.
    """
    found: list[str] = []

    def keep_or_strip(match: re.Match[str]) -> str:
        ids = [part.strip() for part in re.split(r"[,;]", match.group(1))]
        if not all(_ARTICLE_ID.match(i) or i in known_ids for i in ids):
            return match.group(0)  # not a citation bracket - leave untouched
        valid = [i for i in ids if i in known_ids]
        found.extend(valid)
        return f"[{', '.join(valid)}]" if valid else ""

    text = _BRACKET_IDS.sub(keep_or_strip, answer)
    text = re.sub(r"\s+([.,;:])", r"\1", re.sub(r"[ \t]{2,}", " ", text)).strip()
    ordered = list(dict.fromkeys(found + [c for c in cited if c in known_ids]))
    return text, ordered


def sensitivity(answer: str, citations: list[str], deps: AnalystDeps, summaries: list[ArticleSummary]) -> list[str]:
    """Reasons an answer needs human review (empty list = not sensitive)."""
    reasons: list[str] = []
    event_types: set[EventType] = set()
    for article_id in citations:
        record = deps.records_by_id.get(article_id)
        if record is not None:
            event_types.update(ev.event_type for ev in record.events)
    for s in summaries:
        if s.article_id in citations:
            event_types.update(s.event_types)
    for et in sorted(event_types & SENSITIVE_EVENT_TYPES, key=lambda e: e.value):
        reasons.append(f"cites {et.value} event")
    words = sorted({m.group(0).lower() for m in _SENSITIVE_WORDS.finditer(answer)})
    if words:
        reasons.append("mentions " + ", ".join(words))
    return reasons


# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #


def build_analyst_graph(deps: AnalystDeps) -> StateGraph[AnalystState]:
    """Wire the agent's nodes and edges, binding ``deps`` via closures."""
    settings = deps.settings

    def plan(state: AnalystState) -> dict[str, Any]:
        if state.steps >= state.max_steps:
            decision = fallback_answer(state, "Step budget exhausted.")
        else:
            decision = _plan_with_llm(state)
        entry = f"plan[{state.steps + 1}]: {decision.action}"
        if decision.thought:
            entry += f" - {decision.thought}"
        return {"steps": state.steps + 1, "last_decision": decision, "log": [*state.log, entry]}

    def _plan_with_llm(state: AnalystState) -> AgentDecision:
        if deps.llm is None:
            raise GraphError("the planner needs an LLM client")
        pair = plan_prompt(state)
        known = {str(r.get("article_id")) for r in state.search_results} | set(deps.articles_by_id)
        request = LLMRequest(
            task="agent_plan",
            system=pair.system,
            prompt=pair.prompt,
            response_model=AgentDecision,
            effort=settings.llm.effort.get("agent_plan"),
            max_tokens=settings.llm.max_tokens,
            temperature=settings.llm.temperature,
            context=plan_context(state),
        )
        try:
            result = structured_call(
                deps.llm,
                request,
                validators=[_decision_validator(state, known)],
                max_retries=settings.llm.max_validation_retries,
            )
        except LLMError as exc:
            log.warning("planner failed, finishing with available evidence: %s", exc)
            return fallback_answer(state, f"Planner failed ({type(exc).__name__}); answering with what I have.")
        return result.parsed

    def route_plan(state: AnalystState) -> str:
        assert state.last_decision is not None
        return state.last_decision.action

    def search(state: AnalystState) -> dict[str, Any]:
        decision = state.last_decision
        query = (decision.query if decision and decision.query else state.question).strip()
        if deps.embedder is None:
            raise GraphError("the search tool needs an embedder")
        hits = tools.search_articles(query, deps.articles, deps.records, deps.embedder, k=6)
        entry = f"search({query!r}): {len(hits)} hits"
        return {"searched": True, "search_results": hits, "log": [*state.log, entry]}

    def summarise(state: AnalystState) -> dict[str, Any]:
        decision = state.last_decision
        requested = list(decision.article_ids) if decision else []
        if not requested:
            requested = [str(r["article_id"]) for r in state.search_results]
        done = {s.article_id for s in state.summaries}
        todo = [i for i in dict.fromkeys(requested) if i in deps.articles_by_id and i not in done][:MAX_SUMMARISE]
        new = tools.summarise_articles(todo, deps.articles_by_id, deps.records_by_id, deps.llm, settings) if todo else []
        entry = f"summarise({', '.join(todo) or '-'}): {len(new)} summaries"
        return {"summaries": [*state.summaries, *new], "log": [*state.log, entry]}

    def chart(state: AnalystState) -> dict[str, Any]:
        decision = state.last_decision
        dimension = (decision.chart_dimension if decision else None) or "company"
        ids = [str(r["article_id"]) for r in state.search_results]
        records = [deps.records_by_id[i] for i in ids if i in deps.records_by_id] or list(deps.records)
        safe_thread = re.sub(r"[^A-Za-z0-9_.-]+", "_", deps.thread_id)
        out = deps.chart_dir / f"{safe_thread}_{dimension}.png"
        title = f"{dimension.replace('_', ' ').title()} counts - {len(records)} articles"
        path = tools.chart_event_counts(records, dimension, out, title=title)
        return {"chart_path": str(path), "log": [*state.log, f"chart({dimension}): {path.name}"]}

    def draft_answer(state: AnalystState) -> dict[str, Any]:
        decision = state.last_decision
        if decision is None or decision.action != "finish" or not (decision.answer or "").strip():
            decision = fallback_answer(state, "No usable final answer from the planner.")
        known = set(deps.articles_by_id) | {s.article_id for s in state.summaries}
        answer, citations = clean_citations(decision.answer or "", list(decision.citations), known)
        reasons = sensitivity(answer, citations, deps, state.summaries)
        entry = f"draft_answer: {len(citations)} citations" + (f"; sensitive ({'; '.join(reasons)})" if reasons else "")
        return {
            "answer": answer,
            "citations": citations,
            "sensitive": bool(reasons),
            "sensitive_reasons": reasons,
            "delivered": not reasons,
            "log": [*state.log, entry],
        }

    def route_draft(state: AnalystState) -> str:
        return HUMAN_REVIEW if state.sensitive else END

    def human_review(state: AnalystState) -> dict[str, Any]:
        if state.approved is None:
            entry = "human_review: resumed without a decision - answer withheld"
            return {"approved": False, "delivered": False, "log": [*state.log, entry]}
        verdict = "approved" if state.approved else "rejected"
        entry = f"human_review: {verdict}" + (f" ({state.reviewer_notes})" if state.reviewer_notes else "")
        return {"delivered": state.approved, "log": [*state.log, entry]}

    graph = StateGraph(AnalystState)
    graph.add_node("plan", plan)
    graph.add_node("search", search)
    graph.add_node("summarise", summarise)
    graph.add_node("chart", chart)
    graph.add_node("draft_answer", draft_answer)
    graph.add_node(HUMAN_REVIEW, human_review)
    graph.set_entry_point("plan")
    graph.add_conditional_edges(
        "plan",
        route_plan,
        {"search": "search", "summarise": "summarise", "chart": "chart", "finish": "draft_answer"},
    )
    for tool_node in ("search", "summarise", "chart"):
        graph.add_edge(tool_node, "plan")
    graph.add_conditional_edges("draft_answer", route_draft, {HUMAN_REVIEW: HUMAN_REVIEW, END: END})
    graph.add_edge(HUMAN_REVIEW, END)
    return graph


def compile_analyst(deps: AnalystDeps) -> CompiledGraph[AnalystState]:
    """Compile with a JSON checkpointer and the HITL interrupt before review."""
    checkpointer = JsonCheckpointer(deps.settings.path(deps.settings.paths.agent_dir))
    return build_analyst_graph(deps).compile(checkpointer=checkpointer, interrupt_before=[HUMAN_REVIEW])


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #


def new_thread_id(question: str) -> str:
    now = datetime.now(timezone.utc)
    digest = hashlib.sha1(question.encode("utf-8")).hexdigest()[:4]
    return f"t_{now:%Y%m%d%H%M%S}_{digest}{secrets.token_hex(2)}"


def load_corpus(
    settings: "Settings", llm: LLMClient, embedder: "Embedder"
) -> tuple[list[Article], list[ArticleRecord]]:
    """Articles from the store (ingesting if it is empty), deduped, with records."""
    from marketbrief.dedup import dedupe
    from marketbrief.extract import extract_records
    from marketbrief.ingest import ingest
    from marketbrief.ingest.store import ArticleStore, article_time

    store = ArticleStore(settings.path(settings.paths.store_dir) / "articles.jsonl")
    articles = store.all()
    if not articles:
        log.info("article store is empty; ingesting configured sources")
        articles = ingest(settings, run_id="agent").articles
    articles = sorted(articles, key=article_time)[-settings.pipeline.max_articles :]
    articles = dedupe(
        articles, embedder, threshold=settings.dedup.threshold, lead_chars=settings.dedup.lead_chars
    ).kept
    # Extraction goes through the prompt cache, so repeat questions are free.
    return articles, extract_records(articles, llm, settings)


def run_analyst(
    question: str,
    settings: "Settings",
    *,
    llm: LLMClient | None = None,
    articles: list[Article] | None = None,
    records: list[ArticleRecord] | None = None,
    thread_id: str | None = None,
    auto_approve: bool | None = None,
    max_steps: int = 8,
    wants_chart: bool | None = None,
    embedder: "Embedder | None" = None,
) -> AnalystResult:
    """Answer ``question`` with the analyst agent.

    Sensitive answers pause at the human-review checkpoint (status
    ``"interrupted"``) unless ``auto_approve`` is given, in which case the
    checkpoint is resumed immediately with that decision. ``wants_chart``
    defaults to whether the question asks for a chart/plot/graph.
    """
    if not question.strip():
        raise ValueError("question must not be empty")
    if llm is None:
        from marketbrief.llm.factory import build_llm

        llm = build_llm(settings)
    if embedder is None:
        from marketbrief.embeddings import get_embedder

        embedder = get_embedder(settings)
    if articles is None:
        articles, loaded_records = load_corpus(settings, llm, embedder)
        records = records if records is not None else loaded_records
    elif records is None:
        from marketbrief.extract import extract_records

        records = extract_records(articles, llm, settings)

    thread_id = thread_id or new_thread_id(question)
    deps = AnalystDeps(settings, llm, list(articles), list(records), embedder, thread_id=thread_id)
    graph = compile_analyst(deps)
    initial = AnalystState(
        question=question.strip(),
        max_steps=max_steps,
        wants_chart=bool(_CHART_WORDS.search(question)) if wants_chart is None else wants_chart,
    )
    run = graph.invoke(initial, thread_id=thread_id, max_steps=_graph_step_limit(max_steps))
    if run.status == "interrupted" and auto_approve is not None:
        notes = "auto-approved" if auto_approve else "auto-rejected"
        run = graph.resume(thread_id, updates={"approved": auto_approve, "reviewer_notes": notes})
    log.info("analyst thread %s %s after %s", thread_id, run.status, " -> ".join(run.trace))
    return AnalystResult(state=run.state, status=run.status, thread_id=thread_id, trace=run.trace)


def resume_analyst(
    thread_id: str,
    settings: "Settings",
    *,
    approved: bool,
    notes: str = "",
    llm: LLMClient | None = None,
) -> AnalystResult:
    """Record a reviewer decision on a paused thread and finish it."""
    checkpoint = JsonCheckpointer(settings.path(settings.paths.agent_dir)).load(thread_id)
    if checkpoint is None:
        raise GraphError(f"no paused analyst thread {thread_id!r}")
    if checkpoint.next_node == HUMAN_REVIEW:
        # Only the review node remains: no corpus, embedder or LLM needed.
        deps = AnalystDeps(settings, llm, [], [], None, thread_id=thread_id)
    else:
        # A thread that stopped mid-run (e.g. a crash) needs the full toolset.
        if llm is None:
            from marketbrief.llm.factory import build_llm

            llm = build_llm(settings)
        from marketbrief.embeddings import get_embedder

        embedder = get_embedder(settings)
        articles, records = load_corpus(settings, llm, embedder)
        deps = AnalystDeps(settings, llm, articles, records, embedder, thread_id=thread_id)

    graph = compile_analyst(deps)
    max_steps = int(checkpoint.state.get("max_steps", 8))
    run = graph.resume(
        thread_id,
        updates={"approved": approved, "reviewer_notes": notes},
        max_steps=_graph_step_limit(max_steps),
    )
    return AnalystResult(state=run.state, status=run.status, thread_id=thread_id, trace=run.trace)


def _graph_step_limit(max_plan_steps: int) -> int:
    # Each planner step runs at most two nodes (plan + tool); plus answer/review.
    return 2 * max(1, max_plan_steps) + 4
