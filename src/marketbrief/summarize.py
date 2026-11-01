"""Map-reduce summarisation: per-article summaries -> one brief draft -> editor pass.

* map: one ``ArticleSummary`` per article (parallel, cheap, cacheable);
* reduce: one ``BriefDraft`` from all summaries; when there are more than
  ``pipeline.reduce_batch_size`` summaries the reduce is hierarchical — each
  batch is reduced to a partial draft and the partials are merged;
* edit: an editor pass that may tighten wording but never adds citations.

Every LLM output passes through ``structured_call`` with validators that turn
citation mistakes into re-ask feedback.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection, Sequence

from pydantic import BaseModel

from marketbrief.citations import all_claims
from marketbrief.config import Settings
from marketbrief.extract import parallel_map
from marketbrief.llm.base import LLMClient, LLMRequest
from marketbrief.llm.structured import structured_call
from marketbrief.prompts.analyst import map_prompt, merge_prompt, reduce_prompt
from marketbrief.prompts.common import PromptPair
from marketbrief.prompts.editor import edit_prompt
from marketbrief.schemas import (
    Article,
    ArticleRecord,
    ArticleSummary,
    BriefDraft,
    RetrievedSnippet,
)

logger = logging.getLogger(__name__)

Validator = Callable[[BaseModel], list[str]]


def _request(task: str, pair: PromptPair, response_model: type[BaseModel], settings: Settings, context: dict) -> LLMRequest:
    return LLMRequest(
        task=task,
        system=pair.system,
        prompt=pair.prompt,
        response_model=response_model,
        effort=settings.llm.effort.get(task),
        max_tokens=settings.llm.max_tokens,
        temperature=settings.llm.temperature,
        context=context,
    )


def _call(llm: LLMClient, request: LLMRequest, validators: Sequence[Validator], settings: Settings):
    return structured_call(
        llm, request, validators=validators, max_retries=settings.llm.max_validation_retries
    ).parsed


# --------------------------------------------------------------------------- #
# Map
# --------------------------------------------------------------------------- #


def summarize_article(article: Article, record: ArticleRecord, llm: LLMClient, settings: Settings) -> ArticleSummary:
    """Map step for one article."""

    def validator(model: BaseModel) -> list[str]:
        if not isinstance(model, ArticleSummary):
            return [f"expected ArticleSummary, got {type(model).__name__}"]
        errors = []
        if model.article_id != article.id:
            errors.append(f"article_id must be '{article.id}', got '{model.article_id}'")
        if not model.summary.strip():
            errors.append("summary must not be empty")
        return errors

    request = _request(
        "map_summary",
        map_prompt(article, record),
        ArticleSummary,
        settings,
        {"article": article.model_dump(mode="json"), "record": record.model_dump(mode="json")},
    )
    return _call(llm, request, [validator], settings)


def map_summaries(
    articles: Sequence[Article],
    records: Sequence[ArticleRecord],
    llm: LLMClient,
    settings: Settings,
) -> list[ArticleSummary]:
    """Summarise every article that has a record, in parallel, preserving article order.

    Articles without a record (extraction failed) and articles whose summary
    fails validation are logged and skipped.
    """
    by_id = {r.article_id: r for r in records}
    pairs = [(a, by_id[a.id]) for a in articles if a.id in by_id]
    skipped = len(articles) - len(pairs)
    if skipped:
        logger.info("map: %d article(s) have no record and are not summarised", skipped)

    def run(pair: tuple[Article, ArticleRecord]) -> ArticleSummary | None:
        article, record = pair
        try:
            return summarize_article(article, record, llm, settings)
        except Exception as exc:  # isolate per-article failures
            logger.warning("map summary failed for %s: %s", article.id, exc)
            return None

    results = parallel_map(run, pairs, settings.pipeline.max_workers, name="map")
    return [s for s in results if s is not None]


# --------------------------------------------------------------------------- #
# Reduce
# --------------------------------------------------------------------------- #


def draft_errors(draft: BriefDraft, allowed_ids: Collection[str]) -> list[str]:
    """Citation/structure problems in a draft, phrased as re-ask feedback."""
    allowed = set(allowed_ids)
    errors: list[str] = []
    claims = all_claims(draft)
    if not claims:
        errors.append("the draft contains no claims")
    seen: set[str] = set()
    duplicates: set[str] = set()
    for claim in claims:
        if claim.id in seen:
            duplicates.add(claim.id)
        seen.add(claim.id)
        if claim.kind == "fact" and not claim.citations:
            errors.append(f"claim {claim.id} is a fact claim with no citation; cite the article(s) that state it")
        unknown = [cid for cid in claim.citations if cid not in allowed]
        if unknown:
            errors.append(
                f"claim {claim.id} cites ids that were not provided: {', '.join(unknown)}; "
                "use only ids from <allowed_article_ids>"
            )
    if duplicates:
        errors.append(f"claim ids must be unique across the draft; duplicated: {', '.join(sorted(duplicates))}")
    return errors


def _draft_validator(allowed_ids: Collection[str]) -> Validator:
    def check(model: BaseModel) -> list[str]:
        if not isinstance(model, BriefDraft):
            return [f"expected BriefDraft, got {type(model).__name__}"]
        return draft_errors(model, allowed_ids)

    return check


def renumber_claims(draft: BriefDraft) -> BriefDraft:
    """Copy of ``draft`` with claim ids c1..cN in reading order."""
    out = draft.model_copy(deep=True)
    for i, claim in enumerate(all_claims(out), start=1):
        claim.id = f"c{i}"
    return out


def _reduce_batch(
    topic: str,
    summaries: Sequence[ArticleSummary],
    records: Sequence[ArticleRecord],
    llm: LLMClient,
    settings: Settings,
    prior_context: Sequence[RetrievedSnippet],
) -> BriefDraft:
    ids = {s.article_id for s in summaries}
    batch_records = [r for r in records if r.article_id in ids]
    request = _request(
        "reduce_brief",
        reduce_prompt(topic, summaries, batch_records, prior_context),
        BriefDraft,
        settings,
        {
            "topic": topic,
            "summaries": [s.model_dump(mode="json") for s in summaries],
            "records": [r.model_dump(mode="json") for r in batch_records],
            "prior_context": [p.model_dump(mode="json") for p in prior_context],
        },
    )
    return _call(llm, request, [_draft_validator(ids)], settings)


def _merge(
    topic: str,
    partials: Sequence[BriefDraft],
    llm: LLMClient,
    settings: Settings,
    prior_context: Sequence[RetrievedSnippet],
) -> BriefDraft:
    allowed = {cid for p in partials for c in all_claims(p) for cid in c.citations}
    request = _request(
        "reduce_brief",
        merge_prompt(topic, partials, prior_context),
        BriefDraft,
        settings,
        {
            "topic": topic,
            "partials": [p.model_dump(mode="json") for p in partials],
            "prior_context": [p.model_dump(mode="json") for p in prior_context],
        },
    )
    # Merge outputs are renumbered afterwards, so duplicate ids are not worth a re-ask.
    def validator(model: BaseModel) -> list[str]:
        return [e for e in _draft_validator(allowed)(model) if not e.startswith("claim ids must be unique")]

    return renumber_claims(_call(llm, request, [validator], settings))


def reduce_brief(
    topic: str,
    summaries: Sequence[ArticleSummary],
    records: Sequence[ArticleRecord],
    llm: LLMClient,
    settings: Settings,
    prior_context: list[RetrievedSnippet],
) -> BriefDraft:
    """Reduce map summaries into one draft (hierarchically when there are many)."""
    if not summaries:
        raise ValueError("reduce_brief needs at least one article summary")
    batch_size = max(2, settings.pipeline.reduce_batch_size)
    if len(summaries) <= batch_size:
        return _reduce_batch(topic, summaries, records, llm, settings, prior_context)

    batches = [summaries[i : i + batch_size] for i in range(0, len(summaries), batch_size)]
    logger.info("reduce: %d summaries -> %d batches (hierarchical)", len(summaries), len(batches))
    # Prior context is applied once, at the merge, where continuity is decided.
    partials = parallel_map(
        lambda batch: _reduce_batch(topic, batch, records, llm, settings, []),
        batches,
        settings.pipeline.max_workers,
        name="reduce",
    )
    # Merge level by level until one call can take every partial.
    while len(partials) > batch_size:
        groups = [partials[i : i + batch_size] for i in range(0, len(partials), batch_size)]
        partials = [group[0] if len(group) == 1 else _merge(topic, group, llm, settings, []) for group in groups]
    return _merge(topic, partials, llm, settings, prior_context)


# --------------------------------------------------------------------------- #
# Edit
# --------------------------------------------------------------------------- #


def edit_brief(draft: BriefDraft, article_ids: Collection[str], llm: LLMClient, settings: Settings) -> BriefDraft:
    """Editor pass. Falls back to the unedited draft when the edit is unusable.

    The edit is rejected (re-asked, then discarded) if it cites anything
    outside ``article_ids``; it is discarded if it drops every claim.
    """
    allowed = set(article_ids)

    def validator(model: BaseModel) -> list[str]:
        if not isinstance(model, BriefDraft):
            return [f"expected BriefDraft, got {type(model).__name__}"]
        errors = []
        for claim in all_claims(model):
            unknown = [cid for cid in claim.citations if cid not in allowed]
            if unknown:
                errors.append(f"claim {claim.id} introduces citations not in the allowed list: {', '.join(unknown)}")
        return errors

    request = _request("edit", edit_prompt(draft, sorted(allowed)), BriefDraft, settings, {"draft": draft.model_dump(mode="json")})
    try:
        edited: BriefDraft = _call(llm, request, [validator], settings)
    except Exception as exc:  # the editor is polish; never lose the draft over it
        logger.warning("editor pass failed, keeping the unedited draft: %s", exc)
        return draft
    if not all_claims(edited):
        logger.warning("editor pass dropped every claim; keeping the unedited draft")
        return draft
    ids = [c.id for c in all_claims(edited)]
    if len(ids) != len(set(ids)):
        edited = renumber_claims(edited)
    return edited


__all__ = [
    "map_summaries",
    "summarize_article",
    "reduce_brief",
    "edit_brief",
    "renumber_claims",
    "draft_errors",
]
