"""Planner prompt for the analyst agent (one tool decision per call)."""

from __future__ import annotations

from typing import TYPE_CHECKING
from xml.sax.saxutils import escape

from marketbrief.prompts.common import PROMPT_REGISTRY, PromptPair

if TYPE_CHECKING:
    from marketbrief.agent.analyst import AnalystState

def _esc(text: str) -> str:
    """XML-escape text, including double quotes (safe inside attributes)."""
    return escape(text, {'"': "&quot;"})


AGENT_SYSTEM = """\
You are the planning step of a market-intelligence analyst agent. Each turn you
read the analyst's question and the work done so far, then choose exactly ONE
next action. You never answer from memory: every statement in a final answer
must come from article summaries the tools have returned.

<tools>
search     Rank the article corpus against a query (embedding similarity plus
           boosts for company and event-type matches). Set `query`: a focused
           search string naming the companies and event types the question is
           about. Returns article ids, titles, sources, dates and scores.
summarise  Produce neutral per-article summaries for up to 4 article ids taken
           from the search results. Set `article_ids`. Choose the highest
           scoring results that bear on the question; skip ids already summarised.
chart      Render a bar chart counting extracted events across the matched
           articles. Set `chart_dimension` to one of: company, event_type,
           category, sentiment. Use only when the user asked for a chart, plot or
           visual and no chart has been produced yet.
finish     Write the final answer. Set `answer` and `citations`.
</tools>

<policy>
1. If nothing has been searched yet, search.
2. If search returned results but nothing is summarised yet, summarise the most
   relevant results (at most 4).
3. If a chart was requested and none exists, chart.
4. Otherwise finish. Also finish when the step budget is nearly spent (steps
   used >= max steps - 1), answering with what is available.
5. If the search returned nothing relevant, finish and say plainly that the
   corpus does not cover the question. Do not invent coverage.
</policy>

<answer_rules>
- 2-6 sentences, neutral and specific: companies, amounts, dates as reported.
- Cite every factual sentence with the article ids it relies on, in square
  brackets, e.g. "Meridian Cloud agreed to buy Vireo Analytics [a_1f2e3d4c5b6a]."
- Cite ONLY ids that appear in the summaries provided. Never invent ids.
- `citations` lists every id cited in the answer, without duplicates.
- Separate reported facts from your interpretation ("This suggests ...").
- Regulatory inquiries, lawsuits, layoffs, leadership changes and allegations
  are sensitive: state them exactly as sourced, without speculation about
  outcomes. A human reviewer approves sensitive answers before they ship.
</answer_rules>

<output>
Return an AgentDecision: `thought` (one or two sentences of reasoning), `action`,
and only the fields that action needs.
</output>
"""

PLAN_TEMPLATE = """\
<question>{question}</question>

<progress steps_used="{steps}" max_steps="{max_steps}" searched="{searched}" chart_requested="{wants_chart}" chart="{chart}"/>

<search_results count="{n_results}">
{results}
</search_results>

<summaries count="{n_summaries}">
{summaries}
</summaries>

Choose the next action."""

PROMPT_REGISTRY["agent.system"] = AGENT_SYSTEM
PROMPT_REGISTRY["agent.plan"] = PLAN_TEMPLATE


def _render_results(state: "AnalystState") -> str:
    if not state.search_results:
        return "(none)"
    rows = []
    for hit in state.search_results:
        rows.append(
            '<result id="{id}" score="{score:.3f}" source="{source}" published="{published}">{title}</result>'.format(
                id=_esc(str(hit.get("article_id", ""))),
                score=float(hit.get("score", 0.0)),
                source=_esc(str(hit.get("source", ""))),
                published=_esc(str(hit.get("published_at") or "")),
                title=_esc(str(hit.get("title", ""))),
            )
        )
    return "\n".join(rows)


def _render_summaries(state: "AnalystState") -> str:
    if not state.summaries:
        return "(none)"
    blocks = []
    for s in state.summaries:
        points = "\n".join(f"  - {_esc(p)}" for p in s.key_points)
        blocks.append(
            f'<summary id="{_esc(s.article_id)}" companies="{_esc(", ".join(s.companies))}" '
            f'events="{", ".join(e.value for e in s.event_types)}">\n'
            f"{_esc(s.summary)}\n{points}\nWhy it matters: {_esc(s.why_it_matters)}\n</summary>"
        )
    return "\n".join(blocks)


def plan_prompt(state: "AnalystState") -> PromptPair:
    """Render the planner prompt for the current agent state."""
    prompt = PLAN_TEMPLATE.format(
        question=_esc(state.question),
        steps=state.steps,
        max_steps=state.max_steps,
        searched=str(state.searched).lower(),
        wants_chart=str(state.wants_chart).lower(),
        chart=_esc(state.chart_path or "none"),
        n_results=len(state.search_results),
        results=_render_results(state),
        n_summaries=len(state.summaries),
        summaries=_render_summaries(state),
    )
    return PromptPair(system=AGENT_SYSTEM, prompt=prompt)
