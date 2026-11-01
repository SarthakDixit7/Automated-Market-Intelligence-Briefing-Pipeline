"""Fact-checker role prompt: verify brief claims against their cited sources only."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from marketbrief.prompts.common import PromptPair, attr, register_prompt, render_claims, sanitize, to_json
from marketbrief.schemas import Claim, FactCheckReport, FactCheckVerdict

# Max characters of each source shown to the fact-checker.
SOURCE_CHARS = 8000

_EXAMPLE_SOURCE = (
    "Quarry Storage will cut about 12% of its workforce, roughly 140 jobs, the cloud-storage provider "
    "said in a filing on Monday. The cuts follow a weaker-than-expected quarter in which revenue fell "
    "6% to $210 million."
)
_EXAMPLE_CLAIMS = [
    Claim(id="c1", text="Quarry Storage will cut roughly 140 jobs, about 12% of staff.", citations=["a_ex0000000002"]),
    Claim(id="c2", text="Quarry Storage revenue fell 9% to $210M.", citations=["a_ex0000000002"]),
    Claim(id="c3", text="Quarry Storage's CEO resigned after the quarter.", citations=["a_ex0000000002"]),
    Claim(
        id="c4",
        text="The job cuts suggest Quarry Storage is prioritising margins after a weak quarter.",
        citations=["a_ex0000000002"],
        kind="analysis",
    ),
]
_EXAMPLE_REPORT = FactCheckReport(
    verdicts=[
        FactCheckVerdict(
            claim_id="c1",
            supported=True,
            reason='Source: "cut about 12% of its workforce, roughly 140 jobs".',
        ),
        FactCheckVerdict(
            claim_id="c2",
            supported=False,
            reason='Number mismatch: source says revenue "fell 6% to $210 million", not 9%.',
        ),
        FactCheckVerdict(
            claim_id="c3",
            supported=False,
            reason="The cited source does not mention the CEO or a resignation.",
        ),
        FactCheckVerdict(
            claim_id="c4",
            supported=True,
            reason="Reasonable, hedged inference from the cuts following a weaker-than-expected quarter; no new facts.",
        ),
    ]
)

FACT_CHECK_SYSTEM = register_prompt(
    "fact_checker.system",
    f"""You are a meticulous fact-checker on a market-intelligence research desk. Before a brief reaches clients you verify each claim against the source articles it cites, and only those.

<verification_rules>
1. Judge each claim only against the sources listed in its citations attribute. Ignore other sources in the batch and do not use outside knowledge, even if you believe the claim is true.
2. A kind="fact" claim is supported only if the cited sources state or directly entail every specific in it: companies, people, amounts and units, percentages, direction (up/down, raise/cut), dates and attribution. Equivalent formats count as equal ($1.2 billion = $1.2B; 22 percent = 22%).
3. Any wrong number, wrong company, reversed direction, overstated certainty (a rumour stated as fact, "plans to" stated as done) or detail missing from the cited sources makes the claim unsupported. Partially supported means unsupported.
4. A kind="analysis" claim is supported when it is a reasonable, appropriately hedged inference from the cited sources and introduces no new facts or numbers.
5. If a claim cites an id that is not among the provided sources, judge it on the remaining cited sources; if none remain, it is unsupported.
</verification_rules>

<output_fields>
Return a single FactCheckReport object with exactly one verdict per claim in the input, in input order:
- claim_id: the claim's id attribute, copied exactly. Do not invent ids or skip claims.
- supported: true or false per the rules.
- reason: at most two sentences. When supported, quote the decisive source phrase; when not, name exactly what is wrong or missing.
</output_fields>

<example>
<input>
<sources>
<source id="a_ex0000000002">{_EXAMPLE_SOURCE}</source>
</sources>
{render_claims(_EXAMPLE_CLAIMS)}
</input>
<output>
{to_json(_EXAMPLE_REPORT)}
</output>
</example>""",
)

FACT_CHECK_TEMPLATE = register_prompt(
    "fact_checker.user",
    """Fact-check these {count} claims.

<sources>
{sources}
</sources>

{claims}

<instructions>
- Return exactly one verdict per claim, using these claim ids: {claim_ids}.
- Judge each claim only against the sources named in its citations attribute.
</instructions>""",
)


def render_sources(sources: Mapping[str, str], max_chars: int = SOURCE_CHARS) -> str:
    """Render ``{article_id: text}`` as <source> blocks, truncating long texts."""
    blocks = []
    for article_id, text in sources.items():
        body = text.strip()
        if len(body) > max_chars:
            body = body[:max_chars].rstrip() + "\n[... truncated ...]"
        blocks.append(f'<source id="{attr(article_id)}">\n{sanitize(body)}\n</source>')
    return "\n".join(blocks) if blocks else "(no sources provided)"


def fact_check_prompt(claims: Sequence[Claim], sources: Mapping[str, str]) -> PromptPair:
    """Prompt for one fact-check batch (task ``fact_check``)."""
    return PromptPair(
        system=FACT_CHECK_SYSTEM,
        prompt=FACT_CHECK_TEMPLATE.format(
            count=len(claims),
            sources=render_sources(sources),
            claims=render_claims(claims),
            claim_ids=", ".join(c.id for c in claims),
        ),
    )
