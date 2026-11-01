"""Role-based prompt templates (analyst, editor, fact-checker, judge, agent planner).

Importing this package registers the analysis-layer prompts; the judge and
agent prompts register when their modules are imported (``prompt_version``
imports them for you).
"""

from __future__ import annotations

from marketbrief.prompts.common import (
    PROMPT_REGISTRY,
    PromptPair,
    prompt_version,
    register_prompt,
    render_article,
)
from marketbrief.prompts.analyst import extraction_prompt, map_prompt, merge_prompt, reduce_prompt
from marketbrief.prompts.editor import edit_prompt
from marketbrief.prompts.fact_checker import fact_check_prompt

__all__ = [
    "PROMPT_REGISTRY",
    "PromptPair",
    "prompt_version",
    "register_prompt",
    "render_article",
    "extraction_prompt",
    "map_prompt",
    "reduce_prompt",
    "merge_prompt",
    "edit_prompt",
    "fact_check_prompt",
]
