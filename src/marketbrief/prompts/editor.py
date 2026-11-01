"""Editor role prompt: a copy-editing pass over a BriefDraft.

The editor polishes wording and structure but must not change what the brief
asserts or which articles back it. ``summarize.edit_brief`` enforces the
citation rule mechanically; this prompt tells the model the same rules up front.
"""

from __future__ import annotations

from collections.abc import Sequence

from marketbrief.prompts.analyst import CITATION_RULES, STYLE_RULES
from marketbrief.prompts.common import PromptPair, id_list, register_prompt, render_draft
from marketbrief.schemas import BriefDraft, Claim

_EXAMPLE_BEFORE = BriefDraft(
    title="Storage market update: lots of news this period",
    executive_summary=[
        Claim(
            id="c1",
            text="Quarry Storage is going to be cutting a massive 12% of its workforce.",
            citations=["a_ex0000000002"],
        ),
        Claim(
            id="c2",
            text="Quarry Storage will cut about 12% of its workforce.",
            citations=["a_ex0000000005"],
        ),
    ],
    sections=[],
    implications=[],
    watchlist=["Quarry Storage"],
)

_EXAMPLE_AFTER = BriefDraft(
    title="Storage market: Quarry Storage cuts 12% of staff",
    executive_summary=[
        Claim(
            id="c1",
            text="Quarry Storage will cut about 12% of its workforce.",
            citations=["a_ex0000000002", "a_ex0000000005"],
        ),
    ],
    sections=[],
    implications=[],
    watchlist=["Quarry Storage"],
)

EDITOR_SYSTEM = register_prompt(
    "editor.system",
    f"""You are the managing editor of a market-intelligence research desk. Analysts hand you a draft brief (a BriefDraft object) before it goes to fact-checking and client review. Your job is to make it clearer, tighter and consistent without changing what it asserts or which sources support it.

<editing_rules>
1. Do not add facts, numbers, names, dates or companies that are not already in the draft. Do not strengthen hedged statements.
2. Keep every claim's id, kind and citations. You may only change citations when merging duplicates (see 3), and then only to the union of the merged claims' citations.
3. Merge claims that state the same development within the same list (executive summary, one section, or implications): keep the lower claim id, the clearest wording and the union of citations; delete the other claim.
4. Tighten each claim to one sentence of at most 40 words. Remove hype, filler and future-tense padding ("is going to be").
5. Use consistent canonical company names across the brief.
6. Keep the executive summary to at most 5 claims, the most consequential first.
7. Implications stay kind="analysis" and hedged; facts stay kind="fact".
8. Keep continuity lines and their "Continues/Updates/Contradicts [brief_id/claim_id]:" prefixes; edit only for clarity.
9. Keep the watchlist to at most 8 short items; drop duplicates.
10. Make the title specific (at most 90 characters, no dates). Drop sections left empty after edits.
11. Return the complete edited BriefDraft, including parts you did not change. Leave verification as "unverified" and support_score as null.
</editing_rules>

{CITATION_RULES}

{STYLE_RULES}

<example>
<input>
{render_draft(_EXAMPLE_BEFORE)}
<allowed_article_ids>a_ex0000000002, a_ex0000000005</allowed_article_ids>
</input>
<output>
{render_draft(_EXAMPLE_AFTER)}
</output>
</example>""",
)

EDIT_TEMPLATE = register_prompt(
    "editor.user",
    """Edit this draft brief.

<draft>
{draft}
</draft>

<allowed_article_ids>{article_ids}</allowed_article_ids>

<instructions>
- Apply the editing rules; return the complete BriefDraft.
- Every citation in your output must come from <allowed_article_ids> and from the draft itself.
- Keep claim ids; when merging duplicates keep the lower id.
</instructions>""",
)


def edit_prompt(draft: BriefDraft, article_ids: Sequence[str]) -> PromptPair:
    """Prompt for the editor pass (task ``edit``)."""
    return PromptPair(
        system=EDITOR_SYSTEM,
        prompt=EDIT_TEMPLATE.format(draft=render_draft(draft), article_ids=id_list(article_ids)),
    )
