"""LLM-as-a-judge prompt: scores a brief against its sources on a 1-5 rubric."""

from __future__ import annotations

from typing import Mapping, Sequence
from xml.sax.saxutils import escape

from marketbrief.prompts.common import PROMPT_REGISTRY, PromptPair
from marketbrief.schemas import Brief, BriefDraft, Claim, GoldenEvent

def _esc(text: str) -> str:
    """XML-escape text, including double quotes (safe inside attributes)."""
    return escape(text, {'"': "&quot;"})


# Per-source character budget inside the judge prompt.
SOURCE_CHAR_LIMIT = 3000

JUDGE_SYSTEM = """\
You are a strict evaluator of market-intelligence briefs. You receive a brief
(claims with ids and the article ids each one cites), the full text of the
source articles, and optionally a list of events that a complete brief must
cover. Score the brief on three axes using the anchored rubric below. Judge
only against the sources provided - not against your own knowledge.

<rubric>
factuality - are the claims supported by the articles they cite?
  5  Every fact claim is fully supported by its cited sources; numbers, names
     and dates match exactly.
  4  All claims supported; at most one minor imprecision (rounding, a
     paraphrase that slightly overstates) that does not mislead.
  3  Mostly supported, but one claim is unsupported by its citations or has a
     wrong number/name.
  2  Several claims are unsupported, mis-cited or carry wrong figures.
  1  The brief is largely unsupported or contradicts its sources.

coverage - does the brief cover the important events in the sources?
  5  Every material event (funding, M&A, pricing, regulatory, legal,
     leadership, layoffs, earnings, major launches/expansions) is covered,
     including every listed must-cover event.
  4  One secondary event missing; all must-cover events present.
  3  One must-cover event or two secondary events missing.
  2  Several material events missing.
  1  The brief misses most of what happened.

consistency - is the brief coherent with itself and with prior context?
  5  No internal contradictions or duplicated claims; executive summary
     matches the sections; continuity notes agree with the claims.
  4  One redundant or slightly inconsistent claim.
  3  A clear contradiction or several duplicates.
  2  Multiple contradictions between sections or with continuity notes.
  1  Incoherent.
</rubric>

<procedure>
1. For each fact claim, read every cited source and decide whether the claim
   is supported. List the ids of unsupported fact claims in
   `unsupported_claim_ids`. Analysis claims (kind="analysis") are judged on
   whether they reason reasonably from their cited sources, not on literal
   support.
2. List events from the sources (and the must-cover list) that the brief omits
   in `missing_topics`, as short phrases.
3. Score each axis with the rubric. When between two anchors, choose the lower.
4. `rationale`: 2-5 sentences explaining the scores, naming the specific claim
   ids or events that cost points.
</procedure>

Return a JudgeScore. Use only claim ids that appear in the brief.
"""

JUDGE_TEMPLATE = """\
<brief title="{title}">
{claims}
</brief>
{continuity}
<sources count="{n_sources}">
{sources}
</sources>
{golden}
Score this brief."""

PROMPT_REGISTRY["judge.system"] = JUDGE_SYSTEM
PROMPT_REGISTRY["judge.template"] = JUDGE_TEMPLATE


def _claim_line(claim: Claim, where: str) -> str:
    cites = ",".join(claim.citations) or "none"
    return (
        f'<claim id="{_esc(claim.id)}" kind="{claim.kind}" section="{_esc(where)}" cites="{_esc(cites)}">'
        f"{_esc(claim.text)}</claim>"
    )


def render_brief_claims(brief: Brief | BriefDraft) -> str:
    """Every claim in the brief, tagged with id, kind, section and citations."""
    lines = [_claim_line(c, "Executive summary") for c in brief.executive_summary]
    for section in brief.sections:
        lines.extend(_claim_line(c, section.title) for c in section.claims)
    lines.extend(_claim_line(c, "Implications") for c in brief.implications)
    if brief.watchlist:
        lines.append(f"<watchlist>{_esc('; '.join(brief.watchlist))}</watchlist>")
    return "\n".join(lines)


def _continuity_lines(brief: Brief | BriefDraft) -> list[str]:
    if isinstance(brief, Brief):
        return [n.note for n in brief.continuity] or list(brief.continuity_text)
    return list(brief.continuity)


def judge_prompt(
    brief: Brief | BriefDraft,
    sources: Mapping[str, str],
    golden: Sequence[GoldenEvent] | None = None,
) -> PromptPair:
    """Render the judge prompt. ``sources`` maps article id -> article text."""
    source_blocks = []
    for article_id in sorted(sources):
        text = sources[article_id]
        if len(text) > SOURCE_CHAR_LIMIT:
            text = text[:SOURCE_CHAR_LIMIT].rstrip() + " [...]"
        source_blocks.append(f'<source id="{_esc(article_id)}">\n{_esc(text)}\n</source>')

    continuity = _continuity_lines(brief)
    continuity_block = (
        "<continuity>\n" + "\n".join(f"- {_esc(line)}" for line in continuity) + "\n</continuity>\n"
        if continuity
        else ""
    )
    golden_block = ""
    if golden:
        items = "\n".join(
            f'<event id="{_esc(g.id)}" type="{g.event_type.value}" must_cover="{str(g.must_catch).lower()}">'
            f"{_esc(g.description)}</event>"
            for g in golden
        )
        golden_block = f"<must_cover_events>\n{items}\n</must_cover_events>\n"

    prompt = JUDGE_TEMPLATE.format(
        title=_esc(brief.title),
        claims=render_brief_claims(brief),
        continuity=continuity_block,
        n_sources=len(source_blocks),
        sources="\n".join(source_blocks) or "(none)",
        golden=golden_block,
    )
    return PromptPair(system=JUDGE_SYSTEM, prompt=prompt)
