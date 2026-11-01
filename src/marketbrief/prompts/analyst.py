"""Analyst role prompts: extraction, map (per-article summary) and reduce/merge.

Each system prompt is a fixed string (cacheable) that carries the role, the
event taxonomy, the output-field rules and few-shot exemplars. The exemplars
are built from real schema objects, so they are guaranteed to validate and
tests can check that every exemplar evidence quote is verbatim.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone

from marketbrief.prompts.common import (
    PromptPair,
    id_list,
    register_prompt,
    render_article,
    render_draft,
    render_record,
    render_snippets,
    render_summary,
    to_json,
)
from marketbrief.schemas import (
    Article,
    ArticleRecord,
    ArticleSummary,
    BriefDraft,
    BriefSection,
    Claim,
    CompanyMention,
    EventType,
    ExtractedEvent,
    RetrievedSnippet,
    Sentiment,
)

# --------------------------------------------------------------------------- #
# Shared policy text (reused across analyst variants and by the editor)
# --------------------------------------------------------------------------- #

EVENT_TAXONOMY = """\
<event_taxonomy>
Classify each market event into exactly one of these event_type values:
- funding: a company raises equity or debt (seed/Series rounds, strategic investment received, credit facility). Revenue is NOT funding.
- m_and_a: an acquisition, merger, agreed takeover, majority-stake purchase or divestiture. Reported talks count only when the talks themselves are the news; say "in talks" or "reportedly" in the headline.
- product_launch: launch, general availability, release or unveiling of a product, model, chip, service or major version.
- partnership: a commercial alliance, integration, co-development, reseller or supply agreement between named companies.
- earnings: reported financial results or guidance for a period (revenue, profit, margins, bookings).
- pricing: a price change, new pricing model, discount or price war affecting a company's offering.
- regulatory: an action by a regulator or government aimed at named companies: inquiries, investigations, fines, approvals, rulings, new binding rules.
- leadership: appointment or departure of a C-level executive, president or board chair.
- layoffs: job cuts or restructurings that reduce headcount.
- expansion: a new cloud region, data centre, office, design centre or factory, entry into a new market, or a site-linked hiring drive.
- litigation: lawsuits, patent-infringement claims, court rulings and settlements between parties.
- other: a material market event that fits none of the above. Use sparingly.
Tie-breaks: a new cloud region is expansion, not product_launch; a regulator suing a company is regulatory; a deal that combines an investment with a commercial agreement yields two events (funding and partnership) only when the article describes both.
Plans, intentions, forecasts and background mentions of past events are not events of this article.
</event_taxonomy>"""

STYLE_RULES = """\
<style_rules>
- Neutral, third-person newswire register. No hype ("game-changing", "massive"), no investment advice.
- State only what the sources state. Do not speculate beyond them and do not use outside knowledge to add facts.
- Keep attributions the source makes ("according to the company", "people familiar with the matter said").
- Use canonical company names (the full name at first mention in the article, e.g. "Meridian Cloud", not "Meridian").
- Copy numbers exactly as the source gives them; compact currency forms are fine ($410 million -> $410M, $1.2 billion -> $1.2B).
- Use absolute dates (e.g. "on 8 October"), never "yesterday" or "this week".
</style_rules>"""

CITATION_RULES = """\
<citation_rules>
1. Every claim with kind="fact" cites at least one article id from <allowed_article_ids> — the article(s) that state it.
2. Never cite an id that is not in <allowed_article_ids>. Never invent ids. Prior-brief ids (b_...) are never citations.
3. kind="analysis" claims cite the articles whose facts they reason from.
4. One claim = one verifiable statement. Every number, name and date in a fact claim must appear in a cited article.
5. When several articles report the same development, cite all of them on one claim instead of repeating the claim.
6. When sources conflict, say so in the claim and cite each side.
</citation_rules>"""

PRIOR_CONTEXT_RULES = """\
<prior_brief_rules>
<prior_brief_context> holds snippets from earlier, already-approved briefs, retrieved because they look related.
- They are background for continuity, not sources for this cycle: never cite them and never restate them as new facts.
- For each snippet that clearly relates to a development in this cycle, add one line to "continuity" that starts with exactly one of:
  "Continues [<brief_id>/<claim_id>]: " — this cycle extends the same storyline without changing it;
  "Updates [<brief_id>/<claim_id>]: " — this cycle confirms, completes or changes the numbers of something reported earlier (e.g. talks became a signed deal; an expected price war materialised);
  "Contradicts [<brief_id>/<claim_id>]: " — this cycle's sources conflict with the earlier claim (opposite direction, deal called off, denial). Always flag these; never silently overwrite the earlier view.
- Ignore snippets that do not relate to anything in this cycle. If none relate, return "continuity": [].
</prior_brief_rules>"""

# --------------------------------------------------------------------------- #
# Few-shot exemplars (fictional companies, distinct from any real feed)
# --------------------------------------------------------------------------- #

_EX_DATE = datetime(2026, 3, 10, tzinfo=timezone.utc)


def _exemplar_article(article_id: str, source: str, title: str, text: str) -> Article:
    return Article(
        id=article_id,
        source_id="exemplar",
        source_name=source,
        title=title,
        published_at=_EX_DATE,
        fetched_at=_EX_DATE,
        text=text,
        content_hash="exemplar",
    )


EXEMPLAR_ARTICLES: tuple[Article, ...] = (
    _exemplar_article(
        "a_ex0000000001",
        "Harbour Business Review",
        "Tidewater Robotics raises $85 million Series C",
        "Tidewater Robotics, the warehouse-automation start-up, said on Tuesday it has raised "
        "$85 million in a Series C round led by Granite Peak Ventures. The round values the "
        "company at roughly $600 million, according to a person familiar with the matter.\n\n"
        "Chief executive Mara Lindqvist said the money would fund a second assembly line in "
        'Rotterdam. "Demand from grocery distributors has doubled since spring," she said.',
    ),
    _exemplar_article(
        "a_ex0000000002",
        "Storage Weekly",
        "Quarry Storage cuts 12% of staff as UK watchdog probes pricing",
        "Quarry Storage will cut about 12% of its workforce, roughly 140 jobs, the cloud-storage "
        "provider said in a filing on Monday. The cuts follow a weaker-than-expected quarter in "
        "which revenue fell 6% to $210 million.\n\n"
        "Separately, the UK Competition and Markets Authority has opened an inquiry into Quarry "
        "Storage's bundled pricing. Rival Fernbank Cloud, which lowered its archive-tier prices "
        "last month, declined to comment.",
    ),
    _exemplar_article(
        "a_ex0000000003",
        "Storage Weekly",
        "Opinion: the storage market needs fewer logos",
        "Every quarter brings another storage start-up promising cheaper archives. Quarry Storage "
        "and Fernbank Cloud were once those start-ups. Consolidation, in this columnist's view, "
        "is overdue.",
    ),
)

EXEMPLAR_RECORDS: tuple[ArticleRecord, ...] = (
    ArticleRecord(
        article_id="a_ex0000000001",
        companies=[
            CompanyMention(name="Tidewater Robotics", role="subject"),
            CompanyMention(name="Granite Peak Ventures", role="counterparty"),
        ],
        events=[
            ExtractedEvent(
                event_type=EventType.FUNDING,
                headline="Tidewater Robotics raises $85M Series C led by Granite Peak Ventures",
                companies=["Tidewater Robotics", "Granite Peak Ventures"],
                amount="$85M",
                event_date=None,
                evidence_quote=(
                    "Tidewater Robotics, the warehouse-automation start-up, said on Tuesday it has "
                    "raised $85 million in a Series C round led by Granite Peak Ventures."
                ),
            )
        ],
        categories=["warehouse automation", "robotics"],
        sentiment=Sentiment.POSITIVE,
        confidence=0.92,
        key_facts=[
            "Tidewater Robotics raised $85 million in a Series C round.",
            "Granite Peak Ventures led the round.",
            "The round values Tidewater Robotics at roughly $600 million, per a person familiar with the matter.",
            "The funds will pay for a second assembly line in Rotterdam.",
        ],
    ),
    ArticleRecord(
        article_id="a_ex0000000002",
        companies=[
            CompanyMention(name="Quarry Storage", role="subject"),
            CompanyMention(name="Fernbank Cloud", role="competitor"),
        ],
        events=[
            ExtractedEvent(
                event_type=EventType.LAYOFFS,
                headline="Quarry Storage cuts about 12% of its workforce, roughly 140 jobs",
                companies=["Quarry Storage"],
                amount="12%",
                event_date=None,
                evidence_quote=(
                    "Quarry Storage will cut about 12% of its workforce, roughly 140 jobs, the "
                    "cloud-storage provider said in a filing on Monday."
                ),
            ),
            ExtractedEvent(
                event_type=EventType.EARNINGS,
                headline="Quarry Storage quarterly revenue falls 6% to $210M",
                companies=["Quarry Storage"],
                amount="$210M",
                event_date=None,
                evidence_quote=(
                    "The cuts follow a weaker-than-expected quarter in which revenue fell 6% to $210 million."
                ),
            ),
            ExtractedEvent(
                event_type=EventType.REGULATORY,
                headline="UK Competition and Markets Authority opens inquiry into Quarry Storage bundled pricing",
                companies=["Quarry Storage"],
                amount=None,
                event_date=None,
                evidence_quote=(
                    "Separately, the UK Competition and Markets Authority has opened an inquiry into "
                    "Quarry Storage's bundled pricing."
                ),
            ),
        ],
        categories=["cloud storage"],
        sentiment=Sentiment.NEGATIVE,
        confidence=0.88,
        key_facts=[
            "Quarry Storage will cut about 12% of its workforce, roughly 140 jobs.",
            "Quarry Storage revenue fell 6% to $210 million in a weaker-than-expected quarter.",
            "The UK Competition and Markets Authority opened an inquiry into Quarry Storage's bundled pricing.",
            "Fernbank Cloud declined to comment.",
        ],
    ),
    ArticleRecord(
        article_id="a_ex0000000003",
        companies=[
            CompanyMention(name="Quarry Storage", role="mentioned"),
            CompanyMention(name="Fernbank Cloud", role="mentioned"),
        ],
        events=[],
        categories=["cloud storage"],
        sentiment=Sentiment.NEUTRAL,
        confidence=0.8,
        key_facts=["The columnist argues consolidation in the storage market is overdue."],
    ),
)

EXEMPLAR_SUMMARY = ArticleSummary(
    article_id="a_ex0000000001",
    summary=(
        "Tidewater Robotics raised $85 million in a Series C round led by Granite Peak Ventures. "
        "A person familiar with the matter put the valuation at roughly $600 million. Chief executive "
        "Mara Lindqvist said the funds will pay for a second assembly line in Rotterdam."
    ),
    key_points=[
        "$85M Series C led by Granite Peak Ventures",
        "Valuation of roughly $600M, per a person familiar with the matter",
        "Funds earmarked for a second assembly line in Rotterdam",
    ],
    why_it_matters=(
        "Fresh capital for added manufacturing capacity suggests investors expect warehouse-automation "
        "demand to keep growing."
    ),
    companies=["Tidewater Robotics", "Granite Peak Ventures"],
    event_types=[EventType.FUNDING],
)

EXEMPLAR_PRIOR_SNIPPET = RetrievedSnippet(
    brief_id="b_20260215_ex0001",
    brief_title="Logistics tech brief — February",
    brief_date=datetime(2026, 2, 15, tzinfo=timezone.utc),
    claim_id="c4",
    text="Funding & M&A: Tidewater Robotics was reported to be seeking new funding.",
    score=0.71,
)

EXEMPLAR_DRAFT = BriefDraft(
    title="Logistics tech: Tidewater raises $85M as Quarry Storage cuts jobs under regulatory scrutiny",
    executive_summary=[
        Claim(
            id="c1",
            text="Tidewater Robotics raised $85M in a Series C round led by Granite Peak Ventures.",
            citations=["a_ex0000000001"],
        ),
        Claim(
            id="c2",
            text="Quarry Storage is cutting about 12% of its workforce after quarterly revenue fell 6% to $210M.",
            citations=["a_ex0000000002"],
        ),
        Claim(
            id="c3",
            text="The UK Competition and Markets Authority opened an inquiry into Quarry Storage's bundled pricing.",
            citations=["a_ex0000000002"],
        ),
    ],
    sections=[
        BriefSection(
            title="Funding & M&A",
            claims=[
                Claim(
                    id="c4",
                    text="Tidewater Robotics' $85M Series C will fund a second assembly line in Rotterdam.",
                    citations=["a_ex0000000001"],
                )
            ],
        ),
        BriefSection(
            title="People & operations",
            claims=[
                Claim(
                    id="c5",
                    text="Quarry Storage will cut roughly 140 jobs, about 12% of its workforce.",
                    citations=["a_ex0000000002"],
                )
            ],
        ),
        BriefSection(
            title="Regulatory & legal",
            claims=[
                Claim(
                    id="c6",
                    text="The UK Competition and Markets Authority is examining Quarry Storage's bundled pricing.",
                    citations=["a_ex0000000002"],
                )
            ],
        ),
    ],
    implications=[
        Claim(
            id="c7",
            text=(
                "Quarry Storage faces cost cuts and a pricing inquiry at the same time, which may limit "
                "its room to respond to rivals' price moves."
            ),
            citations=["a_ex0000000002"],
            kind="analysis",
        )
    ],
    watchlist=[
        "Quarry Storage — outcome of the CMA pricing inquiry",
        "Tidewater Robotics — Rotterdam assembly line",
    ],
    continuity=[
        "Updates [b_20260215_ex0001/c4]: the funding Tidewater Robotics was reported to be seeking "
        "closed as an $85M Series C."
    ],
)

# --------------------------------------------------------------------------- #
# System prompts
# --------------------------------------------------------------------------- #

_ROLE = (
    "You are a senior market-intelligence analyst at a research firm that writes briefings on "
    "enterprise technology markets for corporate strategy teams. Your readers act on what you "
    "write, so accuracy and traceability matter more than flair."
)


def _extraction_examples() -> str:
    blocks = []
    for i, (article, record) in enumerate(zip(EXEMPLAR_ARTICLES, EXEMPLAR_RECORDS), start=1):
        blocks.append(
            f'<example index="{i}">\n<input>\n{render_article(article)}\n</input>\n'
            f"<output>\n{to_json(record)}\n</output>\n</example>"
        )
    notes = (
        "Notes on the examples: in example 1 the Rotterdam assembly line is a plan, so it is a key "
        "fact, not an expansion event. In example 2 Fernbank Cloud's earlier price cut is background, "
        "so it is not extracted, and the regulator is not listed as a company. Example 3 is an opinion "
        "column with no reported event, so events is empty and confidence still reflects how sure you "
        "are that nothing was missed."
    )
    return "<examples>\n" + "\n".join(blocks) + "\n</examples>\n" + notes


EXTRACTION_SYSTEM = register_prompt(
    "analyst.extraction.system",
    f"""{_ROLE}

Your current job is structured extraction: turn one news article into a typed ArticleRecord that downstream systems trust without re-reading the article. Be literal and conservative; extract what the article says, nothing more.

{EVENT_TAXONOMY}

<output_fields>
Return a single ArticleRecord object:
- article_id: copy the id attribute of the <article> tag exactly.
- companies: every business named in the article, each once, with its canonical name (full name at first mention) and role: "subject" (the company the story is mainly about; usually one), "counterparty" (other party to an event: target, acquirer, investor, partner, litigant), "competitor" (a rival mentioned for contrast) or "mentioned". Do not list regulators, government bodies, universities or people.
- events: one entry per distinct event the article reports (see taxonomy); [] when there is none. For each event:
  - headline: one factual line, at most 160 characters, naming the company and including the amount if any.
  - companies: canonical names of the companies involved (at least one; regulators are not companies).
  - amount: the headline figure as stated, in compact form ("$410M", "$1.2B", "22%"); null if none.
  - event_date: ISO date (YYYY-MM-DD) only when the article states a specific date for the event; otherwise null. Never use the publication date as a guess.
  - evidence_quote: one complete sentence copied character for character from the article <text> (not the title) that supports the event. Do not paraphrase, shorten, join sentences or use ellipses. Prefer the sentence that contains the trigger word, the company and the amount.
- categories: 1-4 lowercase market-category labels for the article (e.g. "gpu cloud", "ai chips", "data platform", "hyperscale cloud", "foundation models", "ai security", "observability", "datacenter networking").
- sentiment: the tone of the news for the subject company: "positive" (growth, wins, funding), "negative" (cuts, losses, probes, lawsuits against it), "mixed" (clear positives and negatives) or "neutral" (factual or no clear direction).
- confidence: 0.90-1.00 when every event is explicit; 0.70-0.89 when classification needed some judgement; below 0.70 when the article is ambiguous, thin or truncated.
- key_facts: at most 5 short, standalone factual statements from the article, each containing a number, a name or a date. No opinions.
</output_fields>

{STYLE_RULES}

{_extraction_examples()}""",
)

MAP_SYSTEM = register_prompt(
    "analyst.map.system",
    f"""{_ROLE}

Your current job is the map step of a map-reduce briefing: write a short, neutral ArticleSummary of one article. You receive the article and the record already extracted from it; your summary must agree with that record and with the article.

{EVENT_TAXONOMY}

<output_fields>
Return a single ArticleSummary object:
- article_id: copy the id attribute of the <article> tag exactly.
- summary: 2-4 sentences. Lead with the main development (who did what, the amount, when), then the most decision-relevant detail. Every fact must be stated in the article.
- key_points: 2-5 terse bullet-style facts (numbers, names, dates), no duplicates of each other.
- why_it_matters: one sentence on the strategic implication for the market or for competitors. Hedge it ("suggests", "may"), and do not introduce facts or numbers that are not in the article.
- companies: canonical names of the companies that matter to the story (start from the record's companies).
- event_types: the distinct event types of the record's events, most important first; ["other"] if the record has no events.
</output_fields>

{STYLE_RULES}

<example>
<input>
{render_article(EXEMPLAR_ARTICLES[0])}
{render_record(EXEMPLAR_RECORDS[0])}
</input>
<output>
{to_json(EXEMPLAR_SUMMARY)}
</output>
</example>""",
)


def _reduce_example_input() -> str:
    by_id = {r.article_id: r for r in EXEMPLAR_RECORDS}
    second = ArticleSummary(
        article_id="a_ex0000000002",
        summary=(
            "Quarry Storage will cut about 12% of its workforce, roughly 140 jobs, after quarterly revenue "
            "fell 6% to $210 million. The UK Competition and Markets Authority opened an inquiry into its "
            "bundled pricing."
        ),
        key_points=["About 12% of staff, roughly 140 jobs", "Revenue down 6% to $210M", "CMA pricing inquiry"],
        why_it_matters="Cost cuts during a pricing inquiry may constrain how Quarry Storage competes on price.",
        companies=["Quarry Storage"],
        event_types=[EventType.LAYOFFS, EventType.EARNINGS, EventType.REGULATORY],
    )
    return "\n".join(
        [
            "<topic>Logistics technology</topic>",
            '<article_summaries count="2">',
            render_summary(EXEMPLAR_SUMMARY, by_id["a_ex0000000001"]),
            render_summary(second, by_id["a_ex0000000002"]),
            "</article_summaries>",
            render_snippets([EXEMPLAR_PRIOR_SNIPPET]),
            "<allowed_article_ids>a_ex0000000001, a_ex0000000002</allowed_article_ids>",
        ]
    )


REDUCE_SYSTEM = register_prompt(
    "analyst.reduce.system",
    f"""{_ROLE}

Your current job is the reduce step of a map-reduce briefing: combine per-article summaries into one client-ready BriefDraft for the given topic and period. You may also be asked to merge several partial drafts (built from batches of articles) into one; the same rules apply.

{EVENT_TAXONOMY}

{CITATION_RULES}

{PRIOR_CONTEXT_RULES}

<output_fields>
Return a single BriefDraft object:
- title: specific and informative, at most 90 characters, naming the topic and the one or two biggest developments. No dates.
- executive_summary: 3-6 kind="fact" claims, the most consequential developments first (typically M&A, funding, regulatory action and pricing moves outrank launches and appointments). Each claim stands alone and repeats the key number.
- sections: group the remaining detail by theme. Use these titles where they fit, in this order, and omit empty ones: "Funding & M&A", "Products & partnerships", "Pricing & competition", "Financial results", "Regulatory & legal", "People & operations", "Other developments". Every article that reports an event must be represented by at least one section claim; low-signal pieces (no events) may be left out. One development per claim; merge articles that report the same development into one claim citing all of them.
- implications: 2-5 kind="analysis" claims on what the developments mean for the market, competitors or buyers. Hedge them ("may", "suggests", "is likely to"); do not introduce facts or numbers that are not in the cited articles.
- watchlist: 3-8 short items, each a company or signal to watch next cycle with a few words on why (e.g. "Quarry Storage — outcome of the CMA pricing inquiry").
- continuity: as defined in the prior-brief rules; [] when no prior snippet relates.
- Claim fields: id is "c1", "c2", ... unique across the whole draft (executive summary, sections and implications together); citations follow the citation rules; kind is "fact" or "analysis"; leave verification as "unverified" and support_score as null — validation fills them later.
</output_fields>

<merge_mode>
When the input contains <partial_briefs> instead of <article_summaries>: produce one BriefDraft covering every partial. Merge claims that describe the same development (union their citations), keep every citation drawn from the partials' own citations, re-rank the executive summary across all partials, rebuild sections by theme, rewrite implications for the combined picture and re-number claim ids so they are unique.
</merge_mode>

{STYLE_RULES}

<example>
<input>
{_reduce_example_input()}
</input>
<output>
{render_draft(EXEMPLAR_DRAFT)}
</output>
</example>""",
)

# --------------------------------------------------------------------------- #
# User-prompt templates (per-run data only)
# --------------------------------------------------------------------------- #

EXTRACTION_TEMPLATE = register_prompt(
    "analyst.extraction.user",
    """Extract an ArticleRecord from this article.

{article}

<instructions>
- Set article_id to exactly "{article_id}".
- Each evidence_quote must be one full sentence copied verbatim from the <text> block above.
- Extract only events this article itself reports; return "events": [] if there are none.
- Follow the taxonomy, field rules and confidence calibration in your instructions.
</instructions>""",
)

MAP_TEMPLATE = register_prompt(
    "analyst.map.user",
    """Write the ArticleSummary for this article. Its extracted record is included so your summary agrees with it.

{article}

{record}

<instructions>
- Set article_id to exactly "{article_id}".
- Use only facts stated in the article; the record is a guide, the article is the authority.
- Keep why_it_matters to one hedged sentence without new facts or numbers.
</instructions>""",
)

REDUCE_TEMPLATE = register_prompt(
    "analyst.reduce.user",
    """Write this cycle's BriefDraft.

<topic>{topic}</topic>

<article_summaries count="{count}">
{summaries}
</article_summaries>

{prior_context}

<allowed_article_ids>{article_ids}</allowed_article_ids>

<instructions>
- Cite only ids from <allowed_article_ids>; every fact claim needs at least one.
- Cover every article that reports an event; merge articles about the same development into one claim with all their ids.
- Use <prior_brief_context> only for "continuity" lines (Continues / Updates / Contradicts with [brief_id/claim_id]); never cite it.
- Claim ids must be unique across the whole draft: c1, c2, c3, ...
</instructions>""",
)

MERGE_TEMPLATE = register_prompt(
    "analyst.merge.user",
    """Merge these {count} partial brief drafts, each built from a different batch of articles, into one BriefDraft.

<topic>{topic}</topic>

<partial_briefs>
{partials}
</partial_briefs>

{prior_context}

<allowed_article_ids>{article_ids}</allowed_article_ids>

<instructions>
- Keep only citations that appear in the partials; every fact claim needs at least one.
- Merge claims about the same development across partials (union their citations) and re-rank the executive summary for the whole set.
- Rebuild "continuity" from <prior_brief_context> for the combined picture; never cite prior briefs.
- Re-number claim ids so they are unique across the whole draft: c1, c2, c3, ...
</instructions>""",
)

# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def extraction_prompt(article: Article) -> PromptPair:
    """Prompt for the extraction pass (task ``extract``)."""
    return PromptPair(
        system=EXTRACTION_SYSTEM,
        prompt=EXTRACTION_TEMPLATE.format(article=render_article(article), article_id=article.id),
    )


def map_prompt(article: Article, record: ArticleRecord) -> PromptPair:
    """Prompt for the per-article map summary (task ``map_summary``)."""
    return PromptPair(
        system=MAP_SYSTEM,
        prompt=MAP_TEMPLATE.format(
            article=render_article(article),
            record=render_record(record),
            article_id=article.id,
        ),
    )


def reduce_prompt(
    topic: str,
    summaries: Sequence[ArticleSummary],
    records: Sequence[ArticleRecord],
    prior_context: Sequence[RetrievedSnippet],
) -> PromptPair:
    """Prompt for the reduce step over one batch of map summaries (task ``reduce_brief``)."""
    by_id = {r.article_id: r for r in records}
    rendered = "\n".join(render_summary(s, by_id.get(s.article_id)) for s in summaries)
    return PromptPair(
        system=REDUCE_SYSTEM,
        prompt=REDUCE_TEMPLATE.format(
            topic=topic,
            count=len(summaries),
            summaries=rendered,
            prior_context=render_snippets(prior_context),
            article_ids=id_list(s.article_id for s in summaries),
        ),
    )


def merge_prompt(
    topic: str,
    partials: Sequence[BriefDraft],
    prior_context: Sequence[RetrievedSnippet],
) -> PromptPair:
    """Prompt for the hierarchical merge of partial drafts (task ``reduce_brief``)."""
    blocks = [
        f'<partial_brief index="{i}">\n{render_draft(p)}\n</partial_brief>' for i, p in enumerate(partials, start=1)
    ]
    cited = (cid for p in partials for c in _draft_claims(p) for cid in c.citations)
    return PromptPair(
        system=REDUCE_SYSTEM,
        prompt=MERGE_TEMPLATE.format(
            count=len(partials),
            topic=topic,
            partials="\n".join(blocks),
            prior_context=render_snippets(prior_context),
            article_ids=id_list(cited),
        ),
    )


def _draft_claims(draft: BriefDraft) -> list[Claim]:
    out = list(draft.executive_summary)
    for section in draft.sections:
        out.extend(section.claims)
    out.extend(draft.implications)
    return out


__all__ = [
    "EXTRACTION_SYSTEM",
    "MAP_SYSTEM",
    "REDUCE_SYSTEM",
    "EXEMPLAR_ARTICLES",
    "EXEMPLAR_RECORDS",
    "EXEMPLAR_SUMMARY",
    "EXEMPLAR_DRAFT",
    "extraction_prompt",
    "map_prompt",
    "reduce_prompt",
    "merge_prompt",
]
