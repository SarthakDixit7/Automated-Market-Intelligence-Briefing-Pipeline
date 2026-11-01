"""LLM-as-a-judge: factuality / coverage / consistency scores for a brief."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Mapping, Sequence

from marketbrief.llm.base import LLMClient, LLMRequest
from marketbrief.llm.structured import structured_call
from marketbrief.prompts.judge import judge_prompt
from marketbrief.schemas import Article, Brief, BriefDraft, GoldenEvent, JudgeScore

if TYPE_CHECKING:
    from marketbrief.config import Settings

log = logging.getLogger(__name__)


def _claim_ids(brief: Brief | BriefDraft) -> set[str]:
    ids = {c.id for c in brief.executive_summary} | {c.id for c in brief.implications}
    for section in brief.sections:
        ids.update(c.id for c in section.claims)
    return ids


def source_texts(articles: Mapping[str, Article]) -> dict[str, str]:
    """``{article_id: title + body}`` - what the judge reads as sources."""
    return {aid: f"{a.title}\n\n{a.text}" for aid, a in articles.items()}


def judge_brief(
    brief: Brief | BriefDraft,
    articles: Mapping[str, Article],
    llm: LLMClient,
    settings: "Settings",
    golden: Sequence[GoldenEvent] | None = None,
) -> JudgeScore:
    """Score ``brief`` against ``articles`` (and optional must-cover events)."""
    sources = source_texts(articles)
    golden = list(golden or [])
    pair = judge_prompt(brief, sources, golden or None)
    request = LLMRequest(
        task="judge",
        system=pair.system,
        prompt=pair.prompt,
        response_model=JudgeScore,
        effort=settings.llm.effort.get("judge"),
        max_tokens=settings.llm.max_tokens,
        temperature=settings.llm.temperature,
        context={
            "brief": brief.model_dump(mode="json"),
            "sources": sources,
            "golden": [g.model_dump(mode="json") for g in golden],
        },
    )
    result = structured_call(llm, request, max_retries=settings.llm.max_validation_retries)
    score: JudgeScore = result.parsed

    # The judge may only flag claims that exist; drop anything else.
    known = _claim_ids(brief)
    unknown = [cid for cid in score.unsupported_claim_ids if cid not in known]
    if unknown:
        log.warning("judge flagged unknown claim ids, ignoring: %s", unknown)
        score = score.model_copy(
            update={"unsupported_claim_ids": [cid for cid in score.unsupported_claim_ids if cid in known]}
        )
    log.info(
        "judge: factuality %d, coverage %d, consistency %d (mean %.2f)",
        score.factuality,
        score.coverage,
        score.consistency,
        judge_mean(score),
    )
    return score


def judge_mean(score: JudgeScore) -> float:
    """Unweighted mean of the three rubric scores."""
    return round((score.factuality + score.coverage + score.consistency) / 3.0, 4)
