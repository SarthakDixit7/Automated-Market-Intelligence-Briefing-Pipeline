"""Shared prompt plumbing: the prompt pair type, XML renderers and the registry.

Every system prompt and user-prompt template is registered in
:data:`PROMPT_REGISTRY` when its module is imported. :func:`prompt_version`
hashes the registry, so any wording change produces a new version string that
is stamped on briefs and eval reports.

Rendering conventions (shared by every role):

* Per-run data lives only in the user prompt, wrapped in XML tags. System
  prompts stay byte-identical across runs so they can be prompt-cached.
* Long documents come first, instructions last.
* Article text is passed through verbatim (no XML escaping) so the model can
  quote it exactly; only closing tags that could break our framing are
  neutralised.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel

from marketbrief.schemas import (
    Article,
    ArticleRecord,
    ArticleSummary,
    BriefDraft,
    Claim,
    RetrievedSnippet,
)

logger = logging.getLogger(__name__)

# name -> template text. Filled by each prompt module at import time.
PROMPT_REGISTRY: dict[str, str] = {}

# Modules whose import registers prompts. judge/agent belong to the eval/agent
# layer and may be absent in a partial install.
_PROMPT_MODULES = (
    "marketbrief.prompts.analyst",
    "marketbrief.prompts.editor",
    "marketbrief.prompts.fact_checker",
    "marketbrief.prompts.judge",
    "marketbrief.prompts.agent",
)

# Tags used to frame untrusted text. A closing tag inside article text is
# defused so a hostile page can't end the <article> block early.
_FRAMING_TAGS = (
    "article",
    "text",
    "title",
    "source",
    "sources",
    "claim",
    "claims",
    "summary",
    "article_summaries",
    "prior_brief_context",
    "snippet",
    "instructions",
    "draft",
    "partial_brief",
    "partial_briefs",
    "extracted_record",
    "validation_feedback",
)
_CLOSING_TAG_RE = re.compile(r"</\s*(" + "|".join(_FRAMING_TAGS) + r")\s*>", re.IGNORECASE)


@dataclass(frozen=True)
class PromptPair:
    """A role system prompt (stable, cacheable) plus a rendered user prompt."""

    system: str
    prompt: str


def register_prompt(name: str, text: str) -> str:
    """Add a template to the registry and return it unchanged."""
    PROMPT_REGISTRY[name] = text
    return text


def prompt_version() -> str:
    """Short fingerprint of every registered prompt (12 hex chars)."""
    for module in _PROMPT_MODULES:
        try:
            importlib.import_module(module)
        except ImportError:
            logger.debug("prompt module %s not available; excluded from prompt_version", module)
    digest = hashlib.sha256()
    for name, text in sorted(PROMPT_REGISTRY.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(text.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Low-level helpers
# --------------------------------------------------------------------------- #


def sanitize(text: str) -> str:
    """Defuse closing framing tags inside untrusted text (keeps it readable)."""
    return _CLOSING_TAG_RE.sub(lambda m: "</ " + m.group(1) + ">", text)


def attr(value: object) -> str:
    """Render a value safely inside a double-quoted XML attribute."""
    return (
        str(value)
        .replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", " ")
    )


def iso_date(value: datetime | None) -> str:
    return value.date().isoformat() if value else "unknown"


def to_json(model: BaseModel | Sequence[BaseModel], *, exclude: set[str] | None = None) -> str:
    """Compact, stable JSON for a model (or list of models) embedded in a prompt."""
    if isinstance(model, BaseModel):
        data = model.model_dump(mode="json", exclude=exclude)
    else:
        data = [m.model_dump(mode="json", exclude=exclude) for m in model]
    return json.dumps(data, ensure_ascii=False, indent=2)


def id_list(ids: Iterable[str]) -> str:
    """Comma-separated, de-duplicated, order-preserving id list."""
    return ", ".join(dict.fromkeys(ids))


# --------------------------------------------------------------------------- #
# Renderers for per-run data
# --------------------------------------------------------------------------- #


def render_article(article: Article, max_chars: int = 6000) -> str:
    """Render one article as an XML block the model can quote from verbatim."""
    text = article.text.strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "\n[... truncated ...]"
    extra = ""
    if article.corroborating_sources:
        extra = f' corroborated_by="{attr(", ".join(article.corroborating_sources))}"'
    return (
        f'<article id="{attr(article.id)}" source="{attr(article.source_name)}" '
        f'published="{iso_date(article.published_at)}"{extra}>\n'
        f"<title>{sanitize(article.title.strip())}</title>\n"
        f"<text>\n{sanitize(text)}\n</text>\n"
        f"</article>"
    )


def render_record(record: ArticleRecord) -> str:
    """Render an extracted record as JSON inside an <extracted_record> tag."""
    return f'<extracted_record article_id="{attr(record.article_id)}">\n{to_json(record)}\n</extracted_record>'


def render_summary(summary: ArticleSummary, record: ArticleRecord | None = None) -> str:
    """Render a map summary (plus its record's events, when known) for the reduce step."""
    lines = [
        f'<summary article_id="{attr(summary.article_id)}">',
        f"<text>{sanitize(summary.summary.strip())}</text>",
        "<key_points>",
        *(f"- {sanitize(p.strip())}" for p in summary.key_points),
        "</key_points>",
        f"<why_it_matters>{sanitize(summary.why_it_matters.strip())}</why_it_matters>",
        f"<companies>{sanitize(', '.join(summary.companies))}</companies>",
        f"<event_types>{', '.join(e.value for e in summary.event_types)}</event_types>",
    ]
    if record is not None:
        lines.append(f"<sentiment>{record.sentiment.value}</sentiment>")
        if record.events:
            lines.append("<events>")
            for ev in record.events:
                amount = f' amount="{attr(ev.amount)}"' if ev.amount else ""
                date = f' date="{attr(ev.event_date)}"' if ev.event_date else ""
                lines.append(
                    f'<event type="{ev.event_type.value}" companies="{attr(", ".join(ev.companies))}"{amount}{date}>'
                    f"{sanitize(ev.headline.strip())}</event>"
                )
            lines.append("</events>")
    lines.append("</summary>")
    return "\n".join(lines)


def render_snippets(snippets: Sequence[RetrievedSnippet]) -> str:
    """Render prior-brief retrieval hits, keeping brief and claim ids visible."""
    if not snippets:
        return "<prior_brief_context>\n(no relevant prior briefs found)\n</prior_brief_context>"
    lines = ["<prior_brief_context>"]
    for s in snippets:
        lines.append(
            f'<snippet brief_id="{attr(s.brief_id)}" claim_id="{attr(s.claim_id)}" '
            f'brief_title="{attr(s.brief_title)}" brief_date="{iso_date(s.brief_date)}" '
            f'relevance="{s.score:.2f}">{sanitize(s.text.strip())}</snippet>'
        )
    lines.append("</prior_brief_context>")
    return "\n".join(lines)


def render_claims(claims: Sequence[Claim]) -> str:
    """Render claims for checking; verification fields are deliberately omitted."""
    lines = ["<claims>"]
    for c in claims:
        lines.append(
            f'<claim id="{attr(c.id)}" kind="{c.kind}" citations="{attr(", ".join(c.citations))}">'
            f"{sanitize(c.text.strip())}</claim>"
        )
    lines.append("</claims>")
    return "\n".join(lines)


_VALIDATOR_FIELDS = {"verification", "support_score"}


def render_draft(draft: BriefDraft) -> str:
    """Render a draft as JSON without the validator-owned claim fields."""
    claim_fields = {"__all__": _VALIDATOR_FIELDS}
    data = draft.model_dump(
        mode="json",
        exclude={
            "executive_summary": claim_fields,
            "implications": claim_fields,
            "sections": {"__all__": {"claims": claim_fields}},
        },
    )
    return json.dumps(data, ensure_ascii=False, indent=2)
