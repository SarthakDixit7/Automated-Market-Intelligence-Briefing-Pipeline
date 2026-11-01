"""Instructor-style structured calls: validate, give feedback, re-ask.

``structured_call`` wraps any :class:`LLMClient`. A response fails when the
provider raises a Pydantic ``ValidationError`` (malformed / schema-invalid
output), when no parsed object comes back, or when a caller-supplied semantic
validator reports problems (e.g. "evidence_quote is not in the article"). The
failure reasons are sent back to the model as ``<validation_feedback>`` and the
request is retried, up to ``max_retries`` extra attempts.

The function keeps no shared state, so it is safe to call from many threads.
"""

from __future__ import annotations

import logging
from typing import Callable, Sequence

from pydantic import BaseModel, ValidationError

from marketbrief.llm.base import LLMClient, LLMRequest, LLMResult, StructuredOutputError, Usage

logger = logging.getLogger(__name__)

Validator = Callable[[BaseModel], list[str]]

# Feedback is capped so a pathological output can't balloon the re-ask prompt.
_MAX_FEEDBACK_ITEMS = 12


def structured_call(
    client: LLMClient,
    request: LLMRequest,
    *,
    validators: Sequence[Validator] = (),
    max_retries: int = 2,
) -> LLMResult:
    """Call ``client`` until the output parses and passes every validator.

    Returns the first passing result with ``usage`` summed over all attempts.
    Raises :class:`StructuredOutputError` after ``1 + max_retries`` failures.
    Other provider errors (refusals, truncation, API errors) propagate as-is.
    """
    attempts = 1 + max(0, max_retries)
    total = Usage()
    current = request
    errors: list[str] = []

    for attempt in range(1, attempts + 1):
        try:
            result = client.complete(current)
        except ValidationError as exc:
            errors = _pydantic_errors(exc)
        else:
            total = total + result.usage
            errors, parsed = _check(request, result, validators)
            if not errors:
                if attempt > 1:
                    logger.debug("task %s passed validation on attempt %d/%d", request.task, attempt, attempts)
                return result.model_copy(update={"parsed": parsed, "usage": total})

        logger.debug(
            "task %s attempt %d/%d failed validation: %s", request.task, attempt, attempts, "; ".join(errors)
        )
        if attempt < attempts:
            # Re-ask from the original prompt so feedback doesn't pile up across attempts.
            current = request.with_feedback(_format_feedback(errors))

    raise StructuredOutputError(
        f"{request.task}: output failed validation after {attempts} attempt(s): " + "; ".join(errors[:5])
    )


def _check(
    request: LLMRequest, result: LLMResult, validators: Sequence[Validator]
) -> tuple[list[str], BaseModel | None]:
    """Validation errors for one result (empty list = pass), plus the coerced object."""
    if request.response_model is None:
        return [], result.parsed  # free-text request: nothing to validate structurally
    parsed = result.parsed
    if parsed is None:
        return ["the response did not contain an object matching the required schema"], None
    if not isinstance(parsed, request.response_model):
        try:
            data = parsed.model_dump() if isinstance(parsed, BaseModel) else parsed
            parsed = request.response_model.model_validate(data)
        except ValidationError as exc:
            return _pydantic_errors(exc), None
    errors: list[str] = []
    for validate in validators:
        errors.extend(str(e) for e in validate(parsed) or [])
    return errors, parsed


def _pydantic_errors(exc: ValidationError) -> list[str]:
    out = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err.get("loc", ())) or "<root>"
        out.append(f"{loc}: {err.get('msg', 'invalid value')}")
    return out or [str(exc)]


def _format_feedback(errors: list[str]) -> str:
    shown = errors[:_MAX_FEEDBACK_ITEMS]
    lines = [f"- {e}" for e in shown]
    if len(errors) > len(shown):
        lines.append(f"- ...and {len(errors) - len(shown)} more problem(s) of the same kind")
    return "\n".join(lines)
