"""Shared data contracts for every pipeline stage.

Everything that moves between stages is a Pydantic model, so each hand-off is
typed, validated and serialisable to JSON. Models whose name ends in ``Draft``,
``Record``, ``Summary``, ``Report`` or ``Decision`` are also used as LLM output
schemas, so they avoid types that structured outputs can't express (no
datetimes, no free-form dicts).
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #


class SourceConfig(BaseModel):
    """One configured source in config/sources.yaml."""

    id: str
    name: str
    kind: Literal["rss", "json_dir", "jsonl"]
    url: str | None = None  # rss
    path: str | None = None  # json_dir / jsonl (relative to project root)
    category: str = "general"
    enabled: bool = True
    fetch_full_text: bool = False  # rss only: also download the linked page
    max_items: int = 50


class RawItem(BaseModel):
    """An item exactly as a source produced it, before normalisation."""

    source_id: str
    source_name: str
    title: str
    url: str | None = None
    published_at: datetime | None = None
    text: str = ""
    author: str | None = None
    raw: dict = Field(default_factory=dict)


class Article(BaseModel):
    """A normalised news item. ``id`` is stable across runs."""

    id: str  # "a_" + 12 hex chars
    source_id: str
    source_name: str
    url: str | None = None
    title: str
    published_at: datetime | None = None
    fetched_at: datetime = Field(default_factory=utcnow)
    text: str
    content_hash: str
    category: str = "general"
    # Filled by dedup: ids of near-duplicate articles folded into this one.
    duplicate_ids: list[str] = Field(default_factory=list)
    # Source names of those duplicates (corroboration shown in briefs).
    corroborating_sources: list[str] = Field(default_factory=list)


class DuplicateCluster(BaseModel):
    canonical_id: str
    member_ids: list[str]  # includes canonical_id
    max_similarity: float
    method: Literal["exact", "embedding"]


# --------------------------------------------------------------------------- #
# Structured extraction (LLM output schemas)
# --------------------------------------------------------------------------- #


class EventType(str, Enum):
    FUNDING = "funding"
    M_AND_A = "m_and_a"
    PRODUCT_LAUNCH = "product_launch"
    PARTNERSHIP = "partnership"
    EARNINGS = "earnings"
    PRICING = "pricing"
    REGULATORY = "regulatory"
    LEADERSHIP = "leadership"
    LAYOFFS = "layoffs"
    EXPANSION = "expansion"
    LITIGATION = "litigation"
    OTHER = "other"


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    MIXED = "mixed"


class CompanyMention(BaseModel):
    name: str = Field(description="Canonical company name as written in the article")
    role: Literal["subject", "counterparty", "competitor", "mentioned"] = "mentioned"


class ExtractedEvent(BaseModel):
    event_type: EventType
    headline: str = Field(description="One-line description of the event, <= 160 chars")
    companies: list[str] = Field(description="Companies involved, canonical names")
    amount: str | None = Field(default=None, description="Monetary amount or % if stated, e.g. '$410M' or '22%'")
    event_date: str | None = Field(default=None, description="ISO date if stated in the article")
    evidence_quote: str = Field(description="Verbatim sentence from the article that supports the event")


class ArticleRecord(BaseModel):
    """Typed record extracted from one article (Instructor-style extraction)."""

    article_id: str
    companies: list[CompanyMention]
    events: list[ExtractedEvent]
    categories: list[str] = Field(description="Market categories, lowercase, e.g. 'gpu cloud'")
    sentiment: Sentiment
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence the extraction is complete and correct")
    key_facts: list[str] = Field(description="Up to 5 short factual statements from the article")


class ArticleSummary(BaseModel):
    """Map-step output: one per article."""

    article_id: str
    summary: str = Field(description="2-4 sentence neutral summary")
    key_points: list[str]
    why_it_matters: str = Field(description="One sentence on the strategic implication")
    companies: list[str]
    event_types: list[EventType]


# --------------------------------------------------------------------------- #
# Briefs
# --------------------------------------------------------------------------- #


class Claim(BaseModel):
    id: str = Field(description="Short id unique within the brief, e.g. 'c1'")
    text: str
    citations: list[str] = Field(description="Article ids that support this claim")
    kind: Literal["fact", "analysis"] = "fact"
    # Set by validation, not by the LLM.
    verification: Literal["unverified", "supported", "weak", "unsupported"] = "unverified"
    support_score: float | None = None


class BriefSection(BaseModel):
    title: str
    claims: list[Claim]


class BriefDraft(BaseModel):
    """Reduce-step output (and editor-pass output)."""

    title: str
    executive_summary: list[Claim] = Field(description="3-6 headline claims")
    sections: list[BriefSection]
    implications: list[Claim] = Field(description="Strategic implications; kind='analysis'")
    watchlist: list[str] = Field(description="Companies or signals to watch next cycle")
    continuity: list[str] = Field(
        default_factory=list,
        description="How this cycle relates to prior briefs (continues / updates / contradicts)",
    )


class SourceRef(BaseModel):
    article_id: str
    title: str
    url: str | None = None
    source_name: str
    published_at: datetime | None = None
    corroborating_sources: list[str] = Field(default_factory=list)


class CitationIssue(BaseModel):
    claim_id: str
    severity: Literal["error", "warning"]
    kind: Literal[
        "missing_citation",
        "unknown_source",
        "low_support",
        "number_mismatch",
        "fact_check_failed",
    ]
    detail: str


class CitationReport(BaseModel):
    passed: bool
    claims_checked: int
    supported: int
    weak: int
    unsupported: int
    issues: list[CitationIssue] = Field(default_factory=list)
    fact_checked: bool = False


class FactCheckVerdict(BaseModel):
    claim_id: str
    supported: bool
    reason: str


class FactCheckReport(BaseModel):
    """Fact-checker role output."""

    verdicts: list[FactCheckVerdict]


class RetrievedSnippet(BaseModel):
    brief_id: str
    brief_title: str
    brief_date: datetime | None = None
    claim_id: str
    text: str
    score: float


class ContinuityNote(BaseModel):
    prior_brief_id: str
    prior_claim: str
    relation: Literal["continues", "updates", "contradicts"]
    note: str


BriefStatus = Literal["draft", "needs_review", "approved", "rejected", "changes_requested"]


class ReviewDecision(BaseModel):
    decision: Literal["approved", "rejected", "changes_requested"]
    reviewer: str
    notes: str = ""
    decided_at: datetime = Field(default_factory=utcnow)


class Brief(BaseModel):
    id: str  # "b_" + yyyymmdd + "_" + 6 hex
    topic: str
    title: str
    created_at: datetime = Field(default_factory=utcnow)
    period_start: datetime | None = None
    period_end: datetime | None = None
    executive_summary: list[Claim]
    sections: list[BriefSection]
    implications: list[Claim]
    watchlist: list[str] = Field(default_factory=list)
    continuity: list[ContinuityNote] = Field(default_factory=list)
    continuity_text: list[str] = Field(default_factory=list)
    sources: list[SourceRef]
    status: BriefStatus = "draft"
    validation: CitationReport | None = None
    review: ReviewDecision | None = None
    prompt_version: str = ""
    run_id: str = ""

    def all_claims(self) -> list[Claim]:
        out = list(self.executive_summary)
        for s in self.sections:
            out.extend(s.claims)
        out.extend(self.implications)
        return out


# --------------------------------------------------------------------------- #
# Analyst agent
# --------------------------------------------------------------------------- #


class AgentDecision(BaseModel):
    """Planner-node output: which tool the analyst agent runs next."""

    thought: str = Field(description="Brief reasoning for the next step")
    action: Literal["search", "summarise", "chart", "finish"]
    query: str | None = Field(default=None, description="search: the query text")
    article_ids: list[str] = Field(default_factory=list, description="summarise: articles to summarise")
    chart_dimension: Literal["company", "event_type", "category", "sentiment"] | None = Field(
        default=None, description="chart: what to count"
    )
    answer: str | None = Field(default=None, description="finish: the final answer, citing article ids in [brackets]")
    citations: list[str] = Field(default_factory=list, description="finish: article ids cited by the answer")


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


class GoldenEvent(BaseModel):
    id: str
    description: str
    event_type: EventType
    companies: list[str]
    keywords: list[str] = Field(default_factory=list)
    must_catch: bool = True


class GoldenMatch(BaseModel):
    golden_id: str
    caught_in_extraction: bool
    caught_in_brief: bool
    matched_article_ids: list[str] = Field(default_factory=list)
    matched_claim_ids: list[str] = Field(default_factory=list)


class RecallReport(BaseModel):
    total: int
    extraction_recall: float
    brief_recall: float
    must_catch_missed: list[str]
    matches: list[GoldenMatch]


class JudgeScore(BaseModel):
    """LLM-as-a-judge output."""

    factuality: int = Field(ge=1, le=5, description="1-5: claims are supported by cited sources")
    coverage: int = Field(ge=1, le=5, description="1-5: important events in the sources are covered")
    consistency: int = Field(ge=1, le=5, description="1-5: internally consistent and consistent with prior briefs")
    unsupported_claim_ids: list[str] = Field(default_factory=list)
    missing_topics: list[str] = Field(default_factory=list)
    rationale: str

    @field_validator("rationale")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("rationale must not be empty")
        return v


# --------------------------------------------------------------------------- #
# Run bookkeeping
# --------------------------------------------------------------------------- #


class StageTiming(BaseModel):
    stage: str
    seconds: float
    items_in: int = 0
    items_out: int = 0


class RunReport(BaseModel):
    run_id: str
    topic: str
    provider: str
    model: str
    started_at: datetime
    finished_at: datetime | None = None
    stages: list[StageTiming] = Field(default_factory=list)
    articles_fetched: int = 0
    articles_after_dedup: int = 0
    duplicate_clusters: int = 0
    records_extracted: int = 0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    prompt_cache_hits: int = 0
    prompt_cache_misses: int = 0
    brief_id: str | None = None
    brief_status: BriefStatus | None = None
    citation_passed: bool | None = None
    warnings: list[str] = Field(default_factory=list)
