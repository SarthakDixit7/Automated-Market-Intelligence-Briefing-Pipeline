"""Deterministic, offline stand-in for an LLM.

``OfflineLLMClient`` lets the whole pipeline (and every test) run without an
API key. It never parses the rendered prompt: each stage passes the structured
inputs it rendered into the prompt via ``request.context``, and the offline
provider builds a schema-valid instance of ``request.response_model`` from
them with transparent heuristics:

* ``extract``      - gazetteer + regex company detection, keyword event lexicon,
                     verbatim evidence sentences, amounts, sentiment, key facts.
* ``map_summary``  - extractive summary + templated "why it matters".
* ``reduce_brief`` - claims grouped into themed sections, executive summary by
                     event importance, templated implications, watchlist and
                     continuity notes against prior briefs (also merges partial
                     drafts for the hierarchical reduce).
* ``edit``         - editor pass: dedupe, trim, cap the executive summary.
* ``fact_check``   - lexical support + number consistency per claim.
* ``judge``        - factuality / coverage / consistency scores (1-5).
* ``agent_plan``   - a fixed search -> summarise -> chart -> finish policy.

Same input, same output: there is no randomness anywhere in this module.
"""

from __future__ import annotations

import logging
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import yaml
from pydantic import BaseModel

from marketbrief.llm.base import LLMError, LLMRequest, LLMResult, Usage
from marketbrief.schemas import (
    AgentDecision,
    ArticleRecord,
    ArticleSummary,
    BriefDraft,
    BriefSection,
    Claim,
    CompanyMention,
    EventType,
    ExtractedEvent,
    FactCheckReport,
    FactCheckVerdict,
    JudgeScore,
    Sentiment,
)

logger = logging.getLogger(__name__)

DEFAULT_OFFLINE_MODEL = "offline-heuristic-v1"


# =========================================================================== #
# Text utilities
# =========================================================================== #

STOPWORDS = frozenset(
    """
    a about above after again against all also am an and any are as at be because been before
    being below between both but by can could did do does doing down during each few for from
    further had has have having he her here hers him his how i if in into is it its itself just
    more most my no nor not now of off on once only or other our ours out over own same she should
    so some such than that the their theirs them then there these they this those through to too
    under until up very was we were what when where which while who whom why will with would you
    your yours said says say according per new including include includes added told year years
    week weeks month months company companies firm will would which while into over than
    """.split()
)

_ABBREVIATIONS = frozenset(
    {"inc", "corp", "ltd", "co", "mr", "ms", "mrs", "dr", "st", "jr", "sr", "vs", "no", "u.s", "e.g", "i.e", "approx", "est", "dept"}
)
_SENTENCE_BOUNDARY = re.compile(r"[.!?][\"'”’)\]]*\s+")
_SENTENCE_STARTERS = "\"'“‘("
_WORD = re.compile(r"[a-z0-9]+(?:[.'][a-z0-9]+)*")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_QUOTE_CHARS = "\"“”‘’'"


def split_sentences(text: str) -> list[str]:
    """Split text into sentences that are verbatim substrings of ``text``."""
    sentences: list[str] = []
    for para in re.split(r"\n+", text or ""):
        if not para.strip():
            continue
        start = 0
        for m in _SENTENCE_BOUNDARY.finditer(para):
            nxt = para[m.end() : m.end() + 1]
            if not nxt or not (nxt.isupper() or nxt.isdigit() or nxt in _SENTENCE_STARTERS):
                continue
            prev = re.search(r"(\S+)$", para[start : m.start() + 1])
            token = prev.group(1).strip(_QUOTE_CHARS + "(").rstrip(".").lower() if prev else ""
            if token in _ABBREVIATIONS or re.fullmatch(r"[a-z]", token):
                continue  # "Inc." / "J." are not sentence ends
            chunk = para[start : m.end()].strip()
            if chunk:
                sentences.append(chunk)
            start = m.end()
        tail = para[start:].strip()
        if tail:
            sentences.append(tail)
    return sentences


def content_words(text: str) -> list[str]:
    """Lowercase content tokens (len > 3, stopwords removed, light plural stemming)."""
    out = []
    for tok in _WORD.findall((text or "").lower()):
        if len(tok) <= 3 or tok in STOPWORDS or tok.isdigit():
            continue
        if tok.endswith("ies") and len(tok) > 5:
            tok = tok[:-3] + "y"
        elif tok.endswith("s") and not tok.endswith("ss") and len(tok) > 4:
            tok = tok[:-1]
        out.append(tok)
    return out


def numbers_in(text: str) -> set[str]:
    """Numbers in ``text``, commas stripped ("1,200" -> "1200")."""
    return {n.replace(",", "").rstrip(".") for n in _NUMBER.findall(text or "")}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def _join_names(names: Sequence[str]) -> str:
    names = list(names)
    if not names:
        return "the companies involved"
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def _truncate(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    cut = text[: limit - 1].rsplit(" ", 1)[0]
    return cut.rstrip(",;:") + "…"


def _is_quote(sentence: str) -> bool:
    return sentence[:1] in _QUOTE_CHARS


def _unique(items: Iterable[Any]) -> list[Any]:
    seen: set[Any] = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


# =========================================================================== #
# Company detection
# =========================================================================== #

_COMPANY_SUFFIX = r"(?:Inc\.?|Labs|AI|Systems|Networks|Cloud|Compute|Silicon|Analytics)"
_FALLBACK_COMPANY = re.compile(rf"\b(?:[A-Z][A-Za-z0-9&\-]*\s+){{1,3}}{_COMPANY_SUFFIX}(?![A-Za-z])")
# Capitalised words that start a sentence or describe a market rather than name a firm.
_NOT_A_NAME = frozenset(
    """
    The A An This That These Those Its Their Our Both Rival Rivals Former Shares Analysts At In On For
    With And But Or As After Before While When Under Over According Meanwhile However Earlier Last Next
    January February March April May June July August September October November December Monday
    Tuesday Wednesday Thursday Friday Saturday Sunday Enterprise Generative Agentic Sovereign Public
    Private Hybrid Open Edge Responsible Applied Conversational Physical Industrial European American
    Global Multi Big Major New
    """.split()
)


@dataclass(frozen=True)
class CompanyEntry:
    name: str
    aliases: tuple[str, ...] = ()
    category: str | None = None


@dataclass
class Gazetteer:
    """Known companies (canonical name + aliases + market category)."""

    entries: list[CompanyEntry] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._canonical: dict[str, str] = {}
        self._category: dict[str, str] = {}
        for e in self.entries:
            if e.category:
                self._category[e.name] = e.category.lower()
            for surface in (e.name, *e.aliases):
                if surface:
                    self._canonical[surface] = e.name
        surfaces = sorted(self._canonical, key=len, reverse=True)
        self._pattern = (
            re.compile(r"(?<![\w])(" + "|".join(re.escape(s) for s in surfaces) + r")(?![\w])")
            if surfaces
            else None
        )

    @classmethod
    def load(cls, path: str | Path | None) -> "Gazetteer":
        """Load ``companies:`` from a YAML file; a missing/invalid file yields an empty gazetteer."""
        if not path:
            return cls()
        p = Path(path)
        if not p.exists():
            logger.info("entities gazetteer not found at %s; using regex company detection only", p)
            return cls()
        try:
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            logger.warning("could not read entities gazetteer %s: %s", p, exc)
            return cls()
        entries = []
        for item in data.get("companies") or []:
            if isinstance(item, str):
                entries.append(CompanyEntry(name=item))
            elif isinstance(item, dict) and item.get("name"):
                aliases = tuple(str(a) for a in (item.get("aliases") or []) if a)
                entries.append(CompanyEntry(str(item["name"]), aliases, item.get("category")))
        return cls(entries)

    def category(self, name: str) -> str | None:
        return self._category.get(name)

    def is_known(self, surface: str) -> bool:
        return any(s in surface for s in self._canonical)

    def find(self, text: str) -> list[tuple[int, str]]:
        """(position, canonical name) for every company mention in ``text``."""
        hits: list[tuple[int, str]] = []
        if self._pattern:
            hits.extend((m.start(), self._canonical[m.group(1)]) for m in self._pattern.finditer(text))
        for m in _FALLBACK_COMPANY.finditer(text):
            words = m.group(0).split()
            offset = 0
            while len(words) > 1 and words[0] in _NOT_A_NAME:
                offset += len(words[0]) + 1
                words.pop(0)
            if len(words) < 2 or all(w in _NOT_A_NAME for w in words[:-1]):
                continue
            name = " ".join(words).rstrip(".")
            if self.is_known(name):
                continue
            hits.append((m.start() + offset, name))
        hits.sort()
        return hits

    def companies_in(self, text: str) -> list[str]:
        """Unique company names in order of first mention."""
        return _unique(name for _, name in self.find(text))


# =========================================================================== #
# Event lexicon
# =========================================================================== #

_I = re.IGNORECASE
_MONEY = re.compile(
    r"(?:US)?([$€£])\s?(\d+(?:[.,]\d+)*)(?:\s?(billion|million|thousand|bn|mn|b|m|k)\b)?", _I
)
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s?(?:%|percent\b|per cent\b)", _I)
_MONTH_DATE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+(\d{1,2})(?:,\s*(\d{4}))?\b"
)
_MONTHS = {
    m: i + 1
    for i, m in enumerate(
        "January February March April May June July August September October November December".split()
    )
}


@dataclass(frozen=True)
class _Trigger:
    pattern: re.Pattern[str]
    requires: re.Pattern[str] | None = None
    excludes: re.Pattern[str] | None = None

    def matches(self, sentence: str) -> bool:
        if not self.pattern.search(sentence):
            return False
        if self.requires is not None and not self.requires.search(sentence):
            return False
        return not (self.excludes is not None and self.excludes.search(sentence))


def _t(pattern: str, requires: str | None = None, excludes: str | None = None) -> _Trigger:
    return _Trigger(
        re.compile(pattern, _I),
        re.compile(requires, _I) if requires else None,
        re.compile(excludes, _I) if excludes else None,
    )


_HAS_FIGURE = r"[$€£]\s?\d|\d\s?(?:%|percent)"
_CUT_WORDS = r"\b(?:cut|cuts|cutting|reduc\w*|eliminat\w*|trim\w*|shed\w*|slash\w*)\b"
# "launches a sovereign cloud region": the launched thing is a facility, so the
# event is an expansion rather than a product launch.
_EXPANSION_CONTEXT = re.compile(
    r"\b(?:launch\w*|unveil\w*|introduc\w*|debut\w*|roll\w* out)\s+(?:[\w-]+\s+){0,4}?"
    r"(?:region|data ?cent(?:er|re)|offices?|campus|facility|design cent(?:er|re))\b",
    _I,
)

EVENT_TRIGGERS: dict[EventType, tuple[_Trigger, ...]] = {
    EventType.FUNDING: (
        _t(r"\bseries [a-h]\b|\bfunding\b|\bfundrais\w*|\bround led by\b|\bventure (?:capital|round)\b"),
        _t(
            r"\brais(?:e|es|ed|ing)\b",
            requires=r"[$€£]\s?\d",
            excludes=r"\brais(?:e|es|ed|ing) (?:its |their )?(?:prices?|concerns?|questions?|rates?|fees?)\b",
        ),
    ),
    EventType.M_AND_A: (
        _t(r"\bacquir(?:e|es|ed|ing)\b|\bacquisitions?\b|\bmergers?\b|\btakeover\b|\bbuy-?out\b|\b(?:agrees?|agreed|deal|talks|plans?) to buy\b"),
    ),
    EventType.PRODUCT_LAUNCH: (
        _t(
            r"\blaunch(?:es|ed|ing)?\b|\bunveil(?:s|ed|ing)?\b|\breleas(?:e|es|ed|ing)\b|\bintroduc(?:es|ed|ing)\b"
            r"|\bgeneral availability\b|\brolls? out\b|\bdebut(?:s|ed)?\b",
            excludes=r"\bpress release\b",
        ),
    ),
    EventType.PARTNERSHIP: (
        _t(r"\bpartner(?:s|ed|ing|ship|ships)?\b|\balliance\b|\bteam(?:s|ed)? up\b|\bcollaborat\w*\b|\bjoint venture\b"),
    ),
    EventType.EARNINGS: (
        _t(r"\brevenue\b|\bearnings\b|\bnet income\b|\bprofit\b|\boperating margin\b|\bquarterly results\b|\bQ[1-4]\b"),
        _t(r"\bquarter\b", requires=_HAS_FIGURE),
    ),
    EventType.PRICING: (
        _t(r"\bprice cuts?\b|\bpricing\b|\bper GPU-?hours?\b|\bGPU-?hour\b|\bprice war\b"),
        _t(r"\bprices?\b", requires=r"\b(?:cut|cuts|lower\w*|reduc\w*|slash\w*|rais\w*|increas\w*|hike\w*)\b"),
    ),
    EventType.REGULATORY: (
        _t(
            r"\bregulators?\b|\bregulatory\b|\binquiry\b|\binvestigation\b|\bantitrust\b|\bprobe\b"
            r"|\bcommission\b|\bcompetition authorit\w+\b|\bwatchdog\b",
            excludes=r"\bsubject to (?:customary )?(?:regulatory|shareholder)|\bregulatory approvals?\b",
        ),
    ),
    EventType.LEADERSHIP: (
        _t(
            r"\bappoint(?:s|ed|ing|ment)?\b|\bnam(?:es|ed)\b.{0,60}\b(?:chief|CEO|CFO|CTO|COO|president)\b"
            r"|\bsteps? down\b|\bresign(?:s|ed|ation)?\b|\bnew (?:chief executive|CEO)\b"
            r"|\bhir(?:es|ed)\b.{0,60}\bas (?:its )?(?:new )?(?:chief|CEO|CFO|CTO|president)\b"
        ),
    ),
    EventType.LAYOFFS: (
        _t(r"\blayoffs?\b|\blay(?:s|ing)? off\b|\blaid off\b|\bjob cuts?\b|\bcuts? (?:\d[\d,]* )?jobs\b|\bredundanc\w+\b"),
        _t(r"\bworkforce\b|\bheadcount\b|\bstaff\b|\bemployees\b|\broles\b", requires=_CUT_WORDS),
    ),
    EventType.EXPANSION: (
        _t(
            r"\bopens\b|\bopened\b|\bopening\b|\bexpan(?:ds?|ded|ding|sion)\b|\bnew (?:cloud )?region\b"
            r"|\bcloud region\b|\bdesign cent(?:re|er)\b|\bnew offices?\b",
            excludes=r"\bopen(?:s|ed|ing)? (?:an? |the |formal |its |in-depth )*(?:inquiry|investigation|probe|case|door)\b"
            r"|\bopening (?:keynote|remarks|session|day|panel)\b",
        ),
    ),
    EventType.LITIGATION: (
        _t(r"\blawsuits?\b|\bsu(?:es|ed|ing)\b|\bpatents?\b|\binfring\w+\b|\bcourt\b|\blitigation\b"),
    ),
}

# Ordering used for the executive summary, implications and tie-breaks.
EVENT_IMPORTANCE: tuple[EventType, ...] = (
    EventType.M_AND_A,
    EventType.FUNDING,
    EventType.REGULATORY,
    EventType.PRICING,
    EventType.EARNINGS,
    EventType.LITIGATION,
    EventType.PRODUCT_LAUNCH,
    EventType.PARTNERSHIP,
    EventType.LEADERSHIP,
    EventType.LAYOFFS,
    EventType.EXPANSION,
    EventType.OTHER,
)
_RANK = {et: i for i, et in enumerate(EVENT_IMPORTANCE)}

EVENT_CATEGORY = {
    EventType.FUNDING: "venture funding",
    EventType.M_AND_A: "m&a",
    EventType.PRODUCT_LAUNCH: "product launches",
    EventType.PARTNERSHIP: "partnerships",
    EventType.EARNINGS: "earnings",
    EventType.PRICING: "pricing",
    EventType.REGULATORY: "regulation",
    EventType.LEADERSHIP: "leadership",
    EventType.LAYOFFS: "workforce",
    EventType.EXPANSION: "expansion",
    EventType.LITIGATION: "litigation",
}

# Words that count as naming an event type in free text (judge coverage check).
EVENT_SYNONYMS: dict[EventType, tuple[str, ...]] = {
    EventType.FUNDING: ("raise", "raises", "raised", "funding", "series", "round", "investors"),
    EventType.M_AND_A: ("acquire", "acquires", "acquisition", "merger", "deal", "buy", "takeover"),
    EventType.PRODUCT_LAUNCH: ("launch", "launches", "launched", "unveils", "release", "model"),
    EventType.PARTNERSHIP: ("partner", "partners", "partnership", "alliance", "embed"),
    EventType.EARNINGS: ("revenue", "earnings", "quarter", "results", "profit"),
    EventType.PRICING: ("price", "prices", "pricing", "cut", "cuts"),
    EventType.REGULATORY: ("inquiry", "regulator", "regulatory", "commission", "antitrust", "probe"),
    EventType.LEADERSHIP: ("ceo", "chief", "appoints", "appointed", "names"),
    EventType.LAYOFFS: ("layoffs", "workforce", "cuts", "jobs", "headcount"),
    EventType.EXPANSION: ("opens", "expands", "expansion", "region", "centre", "center", "office"),
    EventType.LITIGATION: ("sues", "lawsuit", "patent", "infringement", "court"),
    EventType.OTHER: (),
}

_POSITIVE = frozenset(
    "growth grow grows grew surge surges surged record gains gain strong stronger raises raised launch launches "
    "expands expansion partnership wins win milestone accelerate accelerates beat beats boost boosts upbeat "
    "momentum demand profitable opportunity improved improves rises rose".split()
)
_NEGATIVE = frozenset(
    "cuts cut layoffs layoff decline declines declined loss losses inquiry probe lawsuit sues sued infringement "
    "slump slumps weak weaker warning warns concern concerns risk risks pressure scrutiny fines fined penalty "
    "drop drops dropped falls fell shortfall delay delays delayed criticism investigation dispute".split()
)
_EVENT_SENTIMENT = {
    EventType.FUNDING: 2,
    EventType.M_AND_A: 1,
    EventType.PRODUCT_LAUNCH: 2,
    EventType.PARTNERSHIP: 2,
    EventType.EXPANSION: 2,
    EventType.EARNINGS: 0,
    EventType.PRICING: 0,
    EventType.LEADERSHIP: 0,
    EventType.REGULATORY: -2,
    EventType.LAYOFFS: -2,
    EventType.LITIGATION: -2,
}


def detect_event_types(sentence: str) -> list[EventType]:
    """Event types whose trigger fires in ``sentence`` (lexicon order)."""
    found = [et for et, triggers in EVENT_TRIGGERS.items() if any(t.matches(sentence) for t in triggers)]
    # "launches a cloud region / opens a data centre" is an expansion, not a product.
    if EventType.PRODUCT_LAUNCH in found and _EXPANSION_CONTEXT.search(sentence):
        found.remove(EventType.PRODUCT_LAUNCH)
        if EventType.EXPANSION not in found:
            found.append(EventType.EXPANSION)
    return found


def first_amount(sentence: str) -> str | None:
    """First money amount or percentage in ``sentence``, normalised ('$410M', '22%')."""
    candidates: list[tuple[int, str]] = []
    m = _MONEY.search(sentence)
    if m:
        symbol, num, unit = m.group(1), m.group(2).replace(",", ""), (m.group(3) or "").lower()
        suffix = {"billion": "B", "bn": "B", "b": "B", "million": "M", "mn": "M", "m": "M", "thousand": "K", "k": "K"}.get(unit, "")
        candidates.append((m.start(), f"{symbol}{num}{suffix}"))
    p = _PERCENT.search(sentence)
    if p:
        candidates.append((p.start(), f"{p.group(1)}%"))
    return min(candidates)[1] if candidates else None


def _event_date(sentence: str, year: int | None) -> str | None:
    m = _MONTH_DATE.search(sentence)
    if not m:
        return None
    y = int(m.group(3)) if m.group(3) else year
    if not y:
        return None
    return f"{y:04d}-{_MONTHS[m.group(1)]:02d}-{int(m.group(2)):02d}"


# =========================================================================== #
# Templates
# =========================================================================== #

_WHY_IT_MATTERS = {
    EventType.FUNDING: "Fresh capital gives {subject} room to scale capacity and hiring, and signals continued investor appetite for {category}.",
    EventType.M_AND_A: "The deal would consolidate {companies} and could reshape competitive positioning in {category}.",
    EventType.PRODUCT_LAUNCH: "The launch widens {subject}'s offering and raises the competitive bar for rivals in {category}.",
    EventType.PARTNERSHIP: "The tie-up between {companies} deepens integration across the stack and may lock in enterprise customers.",
    EventType.EARNINGS: "The results are a read-through on enterprise demand for {category} and on {subject}'s pricing power.",
    EventType.PRICING: "Pricing moves by {subject} pressure rivals' margins and could accelerate a price war in {category}.",
    EventType.REGULATORY: "Regulatory scrutiny of {subject} could constrain its commercial practices and set a precedent for the wider {category} market.",
    EventType.LEADERSHIP: "The leadership change at {subject} may shift its strategy and execution priorities.",
    EventType.LAYOFFS: "Job cuts at {subject} point to cost pressure and possibly softer demand in {category}.",
    EventType.EXPANSION: "The expansion extends {subject}'s footprint and capacity, intensifying regional competition in {category}.",
    EventType.LITIGATION: "The dispute between {companies} adds legal risk and could affect product roadmaps or licensing costs.",
    EventType.OTHER: "Useful context on {subject} and the {category} market; no discrete market event was reported.",
}

_IMPLICATIONS = {
    EventType.M_AND_A: "Consolidation is accelerating: deal activity involving {companies} suggests platform players are buying capabilities rather than building them.",
    EventType.FUNDING: "Capital keeps flowing to {companies}, extending the runway to compete on capacity, talent and price.",
    EventType.REGULATORY: "Regulatory exposure is rising for {companies}; the outcome may set precedents on fees and market conduct across the sector.",
    EventType.PRICING: "Price competition is intensifying around {companies}; expect rivals to answer with cuts or bundled differentiation, squeezing margins.",
    EventType.EARNINGS: "Reported results from {companies} show how strongly enterprise demand is converting into revenue this cycle.",
    EventType.LITIGATION: "Legal disputes involving {companies} introduce execution and licensing risk that buyers should factor into vendor assessments.",
    EventType.PRODUCT_LAUNCH: "New offerings from {companies} broaden buyer choice and raise the competitive bar for incumbents.",
    EventType.PARTNERSHIP: "Partnerships among {companies} point to ecosystem consolidation, with integrated stacks competing against standalone tools.",
    EventType.LEADERSHIP: "Leadership changes at {companies} could shift strategy; watch for early signals in product and go-to-market priorities.",
    EventType.LAYOFFS: "Workforce reductions at {companies} signal cost discipline and possibly softer demand in their segments.",
    EventType.EXPANSION: "Footprint expansion by {companies} shows continued investment in capacity and in regional, including sovereign, demand.",
}

SECTION_GROUPS: tuple[tuple[str, tuple[EventType, ...]], ...] = (
    ("Funding & M&A", (EventType.M_AND_A, EventType.FUNDING)),
    ("Pricing & competition", (EventType.PRICING,)),
    ("Earnings & financials", (EventType.EARNINGS,)),
    ("Products & partnerships", (EventType.PRODUCT_LAUNCH, EventType.PARTNERSHIP)),
    ("Regulatory & legal", (EventType.REGULATORY, EventType.LITIGATION)),
    ("People & operations", (EventType.LEADERSHIP, EventType.LAYOFFS, EventType.EXPANSION)),
    ("Other developments", (EventType.OTHER,)),
)
_SECTION_OF = {et: title for title, types in SECTION_GROUPS for et in types}

_SPECULATIVE = re.compile(
    r"\b(?:talks|reported(?:ly)?|expected|expects|could|may|might|plans?|planning|considering|exploring|"
    r"weighing|rumou?r\w*|potential|likely|pending|preparing)\b",
    _I,
)


# =========================================================================== #
# The client
# =========================================================================== #


class OfflineLLMClient:
    """Deterministic heuristic provider implementing :class:`LLMClient`."""

    provider = "offline"

    def __init__(
        self,
        *,
        entities_path: str | Path | None = None,
        temperature: float = 0.0,
        model: str = DEFAULT_OFFLINE_MODEL,
    ) -> None:
        self.model = model
        # Accepted for interface parity; output is deterministic at any temperature.
        self.temperature = temperature
        self.gazetteer = Gazetteer.load(entities_path)
        self._handlers: dict[str, Callable[[dict[str, Any]], BaseModel]] = {
            "extract": self._extract,
            "map_summary": self._map_summary,
            "reduce_brief": self._reduce_brief,
            "edit": self._edit,
            "fact_check": self._fact_check,
            "judge": self._judge,
            "agent_plan": self._agent_plan,
        }

    # ------------------------------------------------------------------ API

    def complete(self, request: LLMRequest) -> LLMResult:
        started = time.perf_counter()
        if request.response_model is None:
            if request.task in self._handlers:
                raise LLMError(f"offline task {request.task!r} is structured and needs a response_model")
            text = self._free_text(request)
            parsed = None
        else:
            handler = self._handlers.get(request.task)
            if handler is None:
                raise LLMError(f"offline provider has no handler for task {request.task!r}")
            built = handler(request.context or {})
            parsed = (
                built
                if isinstance(built, request.response_model)
                else request.response_model.model_validate(built.model_dump(mode="json"))
            )
            text = parsed.model_dump_json()
        return LLMResult(
            text=text,
            parsed=parsed,
            usage=Usage(input_tokens=len(request.prompt) // 4, output_tokens=len(text) // 4),
            model=self.model,
            provider=self.provider,
            stop_reason="end_turn",
            latency_s=time.perf_counter() - started,
        )

    # ------------------------------------------------------------ free text

    @staticmethod
    def _free_text(request: LLMRequest) -> str:
        plain = re.sub(r"<[^>]+>", " ", request.prompt)
        sentences = split_sentences(re.sub(r"[ \t]+", " ", plain))
        body = " ".join(sentences[:2]) if sentences else "No input provided."
        return _truncate(f"[offline:{request.task}] {body}", 400)

    # --------------------------------------------------------------- extract

    def _extract(self, ctx: dict[str, Any]) -> ArticleRecord:
        article = _require(ctx, "article", "extract")
        return self.extract_record(article)

    def extract_record(self, article: dict[str, Any]) -> ArticleRecord:
        """Heuristic ``ArticleRecord`` for one article dump (public for reuse in tests/tools)."""
        article_id = str(article.get("id") or "")
        title = str(article.get("title") or "")
        text = str(article.get("text") or "")
        year = _year_of(article.get("published_at"))
        sentences = split_sentences(text)

        title_companies = self.gazetteer.companies_in(title)
        text_companies = self.gazetteer.companies_in(text)
        all_companies = _unique([*title_companies, *text_companies])
        subject = title_companies[0] if title_companies else (text_companies[0] if text_companies else None)

        # Event types: those named in the headline lead, then body order.
        title_types = detect_event_types(title)
        # Evidence: the first sentence with the trigger, preferring one that also
        # names a company (so the quote ties the event to who did it).
        first_hit: dict[EventType, str] = {}
        named_hit: dict[EventType, str] = {}
        for sentence in sentences:
            types = detect_event_types(sentence)
            names_company = bool(types) and bool(self.gazetteer.find(sentence))
            for et in types:
                first_hit.setdefault(et, sentence)
                if names_company:
                    named_hit.setdefault(et, sentence)
        body_hits = {et: named_hit.get(et, first) for et, first in first_hit.items()}
        ordered_types = _unique([*title_types, *first_hit])

        events: list[ExtractedEvent] = []
        lede = sentences[0] if sentences else ""
        for et in ordered_types:
            evidence = body_hits.get(et) or lede
            if not evidence:
                continue
            companies = self.gazetteer.companies_in(evidence)
            if et in title_types:
                companies = _unique([*title_companies, *companies])
            if not companies and subject:
                companies = [subject]
            if not companies:
                continue
            headline_src = title if et in title_types and title else evidence
            events.append(
                ExtractedEvent(
                    event_type=et,
                    headline=_truncate(headline_src, 160),
                    companies=companies,
                    amount=first_amount(evidence) or (first_amount(title) if et in title_types else None),
                    event_date=_event_date(evidence, year),
                    evidence_quote=evidence,
                )
            )

        mentions = self._company_mentions(all_companies, subject, events, text)
        categories = self._categories(all_companies, subject, events)
        sentiment = self._sentiment(title + " " + text, [e.event_type for e in events])
        key_facts = self._key_facts(sentences, all_companies)
        signals = sum(
            [
                subject is not None,
                bool(events),
                len(events) >= 2,
                any(e.amount for e in events),
                bool(title_types),
                any(self.gazetteer.category(c) for c in all_companies),
                len(key_facts) >= 3,
            ]
        )
        confidence = round(min(0.95, 0.55 + 0.06 * signals), 2)
        return ArticleRecord(
            article_id=article_id,
            companies=mentions,
            events=events,
            categories=categories,
            sentiment=sentiment,
            confidence=confidence,
            key_facts=key_facts,
        )

    def _company_mentions(
        self, companies: list[str], subject: str | None, events: list[ExtractedEvent], text: str
    ) -> list[CompanyMention]:
        counterparties = {c for e in events[:1] for c in e.companies if c != subject}
        competitor_ctx = {
            c
            for s in split_sentences(text)
            if re.search(r"\b(?:rival|rivals|competitor|competitors|compet\w+)\b", s, _I)
            for c in self.gazetteer.companies_in(s)
        }
        mentions = []
        for name in companies:
            if name == subject:
                role = "subject"
            elif name in counterparties:
                role = "counterparty"
            elif name in competitor_ctx:
                role = "competitor"
            else:
                role = "mentioned"
            mentions.append(CompanyMention(name=name, role=role))
        return mentions

    def _categories(self, companies: list[str], subject: str | None, events: list[ExtractedEvent]) -> list[str]:
        ordered = [subject] if subject else []
        ordered += [c for c in companies if c != subject]
        cats = [self.gazetteer.category(c) for c in ordered]
        cats += [EVENT_CATEGORY.get(e.event_type) for e in events]
        return _unique(c.lower() for c in cats if c)[:6]

    @staticmethod
    def _sentiment(text: str, event_types: list[EventType]) -> Sentiment:
        words = _WORD.findall(text.lower())
        pos = sum(w in _POSITIVE for w in words)
        neg = sum(w in _NEGATIVE for w in words)
        for et in event_types[:2]:
            weight = _EVENT_SENTIMENT.get(et, 0)
            pos += max(weight, 0)
            neg += max(-weight, 0)
        if pos == 0 and neg == 0:
            return Sentiment.NEUTRAL
        if pos and neg and min(pos, neg) / max(pos, neg) >= 0.6:
            return Sentiment.MIXED
        if pos > neg:
            return Sentiment.POSITIVE
        if neg > pos:
            return Sentiment.NEGATIVE
        return Sentiment.NEUTRAL

    def _key_facts(self, sentences: list[str], companies: list[str]) -> list[str]:
        scored = []
        for pos, s in enumerate(sentences):
            if _is_quote(s) or len(s) < 25:
                continue
            score = int(bool(re.search(r"\d", s))) + int(any(c in s for c in companies) or bool(self.gazetteer.find(s)))
            if score:
                scored.append((-score, pos, s))
        best = sorted(scored)[:5]
        return [s for _, _, s in sorted(best, key=lambda t: t[1])]

    # ----------------------------------------------------------- map_summary

    def _map_summary(self, ctx: dict[str, Any]) -> ArticleSummary:
        article = _require(ctx, "article", "map_summary")
        record = (
            ArticleRecord.model_validate(ctx["record"]) if ctx.get("record") else self.extract_record(article)
        )
        sentences = split_sentences(str(article.get("text") or ""))
        chosen: list[str] = []
        for s in sentences:
            chosen.append(s)
            if len(chosen) >= 3 or (len(chosen) >= 2 and sum(map(len, chosen)) >= 320):
                break
        summary = " ".join(chosen) or str(article.get("title") or "No text available.")
        event_types = _unique(e.event_type for e in record.events)
        companies = [c.name for c in record.companies]
        key_points = list(record.key_facts[:4]) or [s for s in sentences[len(chosen) : len(chosen) + 3]]
        return ArticleSummary(
            article_id=str(article.get("id") or record.article_id),
            summary=summary,
            key_points=key_points,
            why_it_matters=self._why_it_matters(record, event_types),
            companies=companies,
            event_types=event_types,
        )

    @staticmethod
    def _why_it_matters(record: ArticleRecord, event_types: list[EventType]) -> str:
        primary = event_types[0] if event_types else EventType.OTHER
        subject = next((c.name for c in record.companies if c.role == "subject"), None)
        names = record.events[0].companies if record.events else [c.name for c in record.companies]
        subject = subject or (names[0] if names else "the company")
        category = record.categories[0] if record.categories else "the sector"
        return _WHY_IT_MATTERS[primary].format(
            subject=subject, companies=_join_names(names[:3] or [subject]), category=category
        )

    # ---------------------------------------------------------- reduce_brief

    def _reduce_brief(self, ctx: dict[str, Any]) -> BriefDraft:
        topic = str(ctx.get("topic") or "Market brief")
        prior = list(ctx.get("prior_context") or [])
        if ctx.get("partials"):
            return self._merge_partials(topic, ctx["partials"], prior)
        summaries = [ArticleSummary.model_validate(s) for s in ctx.get("summaries") or []]
        if not summaries:
            raise LLMError("offline reduce_brief needs context['summaries'] or context['partials']")
        records = {
            r.article_id: r for r in (ArticleRecord.model_validate(x) for x in ctx.get("records") or [])
        }

        def primary(s: ArticleSummary) -> EventType:
            return s.event_types[0] if s.event_types else EventType.OTHER

        order = {s.article_id: i for i, s in enumerate(summaries)}
        sections: list[BriefSection] = []
        for title, types in SECTION_GROUPS:
            members = [s for s in summaries if primary(s) in types]
            members.sort(key=lambda s: (types.index(primary(s)), order[s.article_id]))
            claims = [
                Claim(id="", text=self._claim_text(s, records.get(s.article_id)), citations=[s.article_id], kind="fact")
                for s in members
            ]
            if claims:
                sections.append(BriefSection(title=title, claims=claims))

        ranked = sorted(summaries, key=lambda s: (_RANK[primary(s)], order[s.article_id]))
        exec_claims: list[Claim] = []
        seen_events: set[tuple[EventType, frozenset[str]]] = set()
        for s in ranked:
            if len(exec_claims) >= 5 or (primary(s) == EventType.OTHER and len(exec_claims) >= 3):
                break
            # One headline per real-world event: an unclustered follow-up or analysis
            # piece about the same deal must not take a second executive slot.
            event_key = (primary(s), frozenset(self._event_companies(s, records.get(s.article_id))))
            if primary(s) != EventType.OTHER and event_key in seen_events:
                continue
            seen_events.add(event_key)
            exec_claims.append(
                Claim(id="", text=self._headline_for(s, records.get(s.article_id)), citations=[s.article_id], kind="fact")
            )

        implications = self._implications(summaries, records, primary)
        mention_counts = Counter(c for s in summaries for c in _unique(s.companies))
        types_by_company: dict[str, list[EventType]] = {}
        for s in summaries:
            for c in s.companies:
                types_by_company.setdefault(c, []).extend(s.event_types)
        watchlist = [
            self._watch_line(c, n, types_by_company.get(c, [])) for c, n in mention_counts.most_common(5)
        ]
        cycle_claims = [c for sec in sections for c in sec.claims]
        draft = BriefDraft(
            title=self._title(topic, mention_counts, len(summaries)),
            executive_summary=exec_claims,
            sections=sections,
            implications=implications,
            watchlist=watchlist,
            continuity=self._continuity(prior, cycle_claims),
        )
        return _renumber(draft)

    def _claim_text(self, summary: ArticleSummary, record: ArticleRecord | None) -> str:
        """The summary's lead sentence, unless it names no company (e.g. an opinion
        piece's scene-setting opener) and the record has firmer evidence."""
        lead = _lead_sentence(summary.summary)
        if self.gazetteer.find(lead) or not (record and record.events):
            return lead
        return record.events[0].evidence_quote

    @staticmethod
    def _event_companies(summary: ArticleSummary, record: ArticleRecord | None) -> list[str]:
        if record and record.events:
            primary = summary.event_types[0] if summary.event_types else record.events[0].event_type
            event = next((e for e in record.events if e.event_type == primary), record.events[0])
            return event.companies
        return summary.companies[:1]

    @staticmethod
    def _headline_for(summary: ArticleSummary, record: ArticleRecord | None) -> str:
        if record and record.events:
            primary = summary.event_types[0] if summary.event_types else record.events[0].event_type
            event = next((e for e in record.events if e.event_type == primary), record.events[0])
            return event.headline
        return _lead_sentence(summary.summary)

    @staticmethod
    def _implications(
        summaries: list[ArticleSummary],
        records: dict[str, ArticleRecord],
        primary: Callable[[ArticleSummary], EventType],
    ) -> list[Claim]:
        by_type: dict[EventType, list[ArticleSummary]] = {}
        for s in summaries:
            by_type.setdefault(primary(s), []).append(s)

        def involved(s: ArticleSummary, et: EventType) -> list[str]:
            # Companies party to the event itself, not every company the article mentions.
            record = records.get(s.article_id)
            event = next((e for e in record.events if e.event_type == et), None) if record else None
            return event.companies if event else s.companies[:2]

        claims = []
        for et in EVENT_IMPORTANCE:
            if et not in by_type or et not in _IMPLICATIONS:
                continue
            group = by_type[et]
            companies = Counter(c for s in group for c in _unique(involved(s, et)))
            names = [c for c, _ in companies.most_common(3)]
            claims.append(
                Claim(
                    id="",
                    text=_IMPLICATIONS[et].format(companies=_join_names(names)),
                    citations=_unique(s.article_id for s in group)[:4],
                    kind="analysis",
                )
            )
            if len(claims) >= 5:
                break
        return claims

    @staticmethod
    def _watch_line(company: str, count: int, types: list[EventType]) -> str:
        labels = [EVENT_CATEGORY.get(t, "") for t in _unique(types) if t != EventType.OTHER]
        labels = [label for label in labels if label]
        detail = f" ({', '.join(labels[:3])})" if labels else ""
        noun = "article" if count == 1 else "articles"
        return f"{company}: {count} {noun} this cycle{detail}"

    @staticmethod
    def _title(topic: str, mention_counts: Counter, n_articles: int) -> str:
        n_companies = len(mention_counts)
        noun = "development" if n_articles == 1 else "developments"
        return f"{topic} briefing: {n_articles} {noun} across {n_companies} companies"

    def _continuity(self, prior: list[Any], claims: list[Claim]) -> list[str]:
        """Link prior-brief snippets to this cycle's claims about the same companies."""
        notes: list[str] = []
        seen: set[str] = set()
        linked: set[tuple[str, str]] = set()
        claim_companies = [(c, set(self.gazetteer.companies_in(c.text))) for c in claims]
        for raw in prior:
            snippet = raw if isinstance(raw, dict) else raw.model_dump(mode="json")
            text = str(snippet.get("text") or "").strip()
            prior_text = re.sub(r"^[^:]{1,60}:\s+", "", text)  # drop the "Section title: " prefix
            # Bare watchlist entries ("Vireo Analytics") carry no claim to continue.
            if len(prior_text.split()) < 6 or _norm(text) in seen:
                continue
            snippet_companies = set(self.gazetteer.companies_in(text))
            candidates = [(c, cos) for c, cos in claim_companies if cos & snippet_companies]
            if not candidates:
                continue
            snippet_words = set(content_words(text))
            snippet_types = set(detect_event_types(text))

            def relevance(pair: tuple[Claim, set[str]]) -> tuple[int, int]:
                claim = pair[0]
                same_event = bool(snippet_types & set(detect_event_types(claim.text)))
                overlap = len(snippet_words & set(content_words(claim.text)))
                return (overlap + (5 if same_event else 0), -claims.index(claim))

            best, _ = max(candidates, key=relevance)
            seen.add(_norm(text))
            prior_id = str(snippet.get("brief_id") or "a prior brief")
            if (prior_id, best.text) in linked:
                continue  # one note per prior brief and cycle claim
            linked.add((prior_id, best.text))
            relation = "Updates" if _SPECULATIVE.search(prior_text) else "Continues"
            verb = "reported" if relation == "Updates" else "noted"
            notes.append(f'{relation}: prior brief {prior_id} {verb} "{prior_text.rstrip()}" This cycle: {best.text}')
            if len(notes) >= 6:
                break
        return notes

    def _merge_partials(self, topic: str, partials: list[Any], prior: list[Any]) -> BriefDraft:
        drafts = [BriefDraft.model_validate(p) for p in partials]
        section_claims: dict[str, list[Claim]] = {}
        for d in drafts:
            for sec in d.sections:
                section_claims.setdefault(sec.title, []).extend(sec.claims)
        known_order = [t for t, _ in SECTION_GROUPS]
        titles = sorted(section_claims, key=lambda t: (known_order.index(t) if t in known_order else len(known_order), t))
        sections = [BriefSection(title=t, claims=_dedupe_claims(section_claims[t])) for t in titles]

        # Round-robin over partial executive summaries keeps each batch's top item.
        exec_claims: list[Claim] = []
        for i in range(max((len(d.executive_summary) for d in drafts), default=0)):
            for d in drafts:
                if i < len(d.executive_summary):
                    exec_claims.append(d.executive_summary[i])
        exec_claims = _dedupe_claims(exec_claims)[:5]
        implications = _dedupe_claims([c for d in drafts for c in d.implications])[:6]

        watch_counts: Counter[str] = Counter()
        watch_line: dict[str, str] = {}
        for d in drafts:
            for line in d.watchlist:
                company = line.split(":", 1)[0].strip()
                m = re.search(r":\s*(\d+)", line)
                watch_counts[company] += int(m.group(1)) if m else 1
                watch_line.setdefault(company, line)
        watchlist = [
            re.sub(r":\s*\d+ articles?", f": {n} {'article' if n == 1 else 'articles'}", watch_line[c], count=1)
            for c, n in watch_counts.most_common(5)
        ]
        all_claims = [c for s in sections for c in s.claims]
        continuity = _unique([line for d in drafts for line in d.continuity])
        if not continuity and prior:
            continuity = self._continuity(prior, all_claims)
        n_articles = len({cid for c in all_claims for cid in c.citations})
        draft = BriefDraft(
            title=self._title(topic, watch_counts, n_articles) if watch_counts else drafts[0].title,
            executive_summary=exec_claims,
            sections=sections,
            implications=implications,
            watchlist=watchlist,
            continuity=continuity[:6],
        )
        return _renumber(draft)

    # ------------------------------------------------------------------ edit

    def _edit(self, ctx: dict[str, Any]) -> BriefDraft:
        draft = BriefDraft.model_validate(_require(ctx, "draft", "edit"))

        def tidy(claims: list[Claim]) -> list[Claim]:
            return [c.model_copy(update={"text": re.sub(r"\s+", " ", c.text).strip()}) for c in claims]

        seen: set[str] = set()
        sections = []
        for sec in draft.sections:
            kept = []
            for c in tidy(sec.claims):
                key = _norm(c.text)
                if key and key not in seen:
                    seen.add(key)
                    kept.append(c)
            if kept:
                sections.append(BriefSection(title=sec.title.strip(), claims=kept))
        return BriefDraft(
            title=re.sub(r"\s+", " ", draft.title).strip(),
            executive_summary=_dedupe_claims(tidy(draft.executive_summary))[:5],
            sections=sections,
            implications=_dedupe_claims(tidy(draft.implications)),
            watchlist=_unique(w.strip() for w in draft.watchlist if w.strip()),
            continuity=_unique(c.strip() for c in draft.continuity if c.strip()),
        )

    # ------------------------------------------------------------ fact_check

    def _fact_check(self, ctx: dict[str, Any]) -> FactCheckReport:
        claims = [Claim.model_validate(c) for c in ctx.get("claims") or []]
        sources = {str(k): str(v) for k, v in (ctx.get("sources") or {}).items()}
        return FactCheckReport(verdicts=[check_claim(c, sources) for c in claims])

    # ----------------------------------------------------------------- judge

    def _judge(self, ctx: dict[str, Any]) -> JudgeScore:
        brief = _require(ctx, "brief", "judge")
        sources = {str(k): str(v) for k, v in (ctx.get("sources") or {}).items()}
        golden = list(ctx.get("golden") or [])
        exec_claims = [Claim.model_validate(c) for c in brief.get("executive_summary") or []]
        body_claims = [Claim.model_validate(c) for s in brief.get("sections") or [] for c in s.get("claims") or []]
        analysis = [Claim.model_validate(c) for c in brief.get("implications") or []]
        claims = exec_claims + body_claims + analysis

        # Factuality: share of fact claims the lexical checker supports.
        verdicts = {c.id: check_claim(c, sources) for c in claims}
        facts = [c for c in claims if c.kind == "fact"]
        unsupported = _unique(c.id for c in claims if not verdicts[c.id].supported)
        ratio = (sum(verdicts[c.id].supported for c in facts) / len(facts)) if facts else 0.0
        factuality = _scale(ratio, (0.95, 0.85, 0.7, 0.5))

        # Coverage: golden events reflected in claims, else share of sources cited.
        texts = [c.text for c in exec_claims + body_claims]
        missing: list[str] = []
        if golden:
            hits = 0
            for g in golden:
                if any(_claim_matches_golden(g, t) for t in texts):
                    hits += 1
                else:
                    missing.append(str(g.get("description") or g.get("id")))
            cov_ratio = hits / len(golden)
            cov_detail = f"{hits}/{len(golden)} golden events covered"
        else:
            cited = {cid for c in claims for cid in c.citations}
            uncited = [sid for sid in sources if sid not in cited]
            missing = [f"source {sid} not cited" for sid in uncited]
            cov_ratio = (1 - len(uncited) / len(sources)) if sources else 0.0
            cov_detail = f"{len(sources) - len(uncited)}/{len(sources)} sources cited"
        coverage = _scale(cov_ratio, (0.9, 0.75, 0.6, 0.4))

        # Consistency: duplicates, unknown citations, id collisions, contradictions.
        problems = []
        dup = len(body_claims) - len({_norm(c.text) for c in body_claims})
        if dup:
            problems.append(f"{dup} duplicate claim(s)")
        if len({c.id for c in claims}) != len(claims):
            problems.append("claim ids are not unique")
        unknown = sorted({cid for c in claims for cid in c.citations if sources and cid not in sources})
        if unknown:
            problems.append(f"citations to unknown sources {unknown}")
        contradictions = _contradictions(body_claims)
        if contradictions:
            problems.append(f"{contradictions} possible contradiction(s)")
        flagged = [
            item
            for item in [*(brief.get("continuity") or []), *(brief.get("continuity_text") or [])]
            if (isinstance(item, str) and item.lower().startswith("contradicts"))
            or (isinstance(item, dict) and item.get("relation") == "contradicts")
        ]
        if flagged:
            problems.append(f"{len(flagged)} contradiction(s) with prior briefs flagged")
        consistency = max(1, 5 - len(problems))

        rationale = (
            f"Factuality {factuality}/5: {sum(verdicts[c.id].supported for c in facts)}/{len(facts)} fact claims "
            f"lexically supported by their cited sources. Coverage {coverage}/5: {cov_detail}. "
            f"Consistency {consistency}/5: " + ("; ".join(problems) if problems else "no duplicates or contradictions found") + "."
        )
        return JudgeScore(
            factuality=factuality,
            coverage=coverage,
            consistency=consistency,
            unsupported_claim_ids=unsupported,
            missing_topics=missing[:10],
            rationale=rationale,
        )

    # ------------------------------------------------------------ agent_plan

    def _agent_plan(self, ctx: dict[str, Any]) -> AgentDecision:
        question = str(ctx.get("question") or "").strip()
        state = dict(ctx.get("state") or {})
        steps = int(state.get("steps") or 0)
        max_steps = int(state.get("max_steps") or 8)
        results = list(state.get("search_results") or [])
        summaries = [ArticleSummary.model_validate(s) for s in state.get("summaries") or []]

        if steps >= max_steps - 1:
            return self._finish(summaries, "Step budget nearly exhausted; answering with what I have.")
        if not state.get("searched"):
            return AgentDecision(thought="Find articles relevant to the question first.", action="search", query=question)
        if not summaries and results:
            ids = _unique(str(r.get("article_id")) for r in results if r.get("article_id"))[:4]
            return AgentDecision(
                thought=f"Summarise the {len(ids)} most relevant articles before answering.",
                action="summarise",
                article_ids=ids,
            )
        if state.get("wants_chart") and not state.get("chart_path"):
            q = question.lower()
            dimension = (
                "sentiment" if "sentiment" in q else "category" if "categor" in q else "event_type" if "event" in q else "company"
            )
            return AgentDecision(thought=f"The user asked for a chart; plot counts by {dimension}.", action="chart", chart_dimension=dimension)
        return self._finish(summaries, "I have enough evidence to answer.")

    def _answer_sentence(self, summary: ArticleSummary) -> str:
        """First summary sentence that names a known company (skips scene-setting
        openers such as an analysis piece's anecdote); falls back to the lead."""
        for sentence in split_sentences(summary.summary):
            if self.gazetteer.find(sentence):
                return sentence
        return _lead_sentence(summary.summary)

    def _finish(self, summaries: list[ArticleSummary], thought: str) -> AgentDecision:
        if not summaries:
            return AgentDecision(
                thought=thought,
                action="finish",
                answer="I could not find coverage of this question in the current article set.",
                citations=[],
            )
        parts = [f"{self._answer_sentence(s)} [{s.article_id}]" for s in summaries[:4]]
        lead = "Based on 1 relevant article: " if len(parts) == 1 else f"Based on {len(parts)} relevant articles: "
        return AgentDecision(
            thought=thought,
            action="finish",
            answer=lead + " ".join(parts),
            citations=[s.article_id for s in summaries[:4]],
        )


# =========================================================================== #
# Helpers shared by the handlers
# =========================================================================== #


def check_claim(claim: Claim, sources: dict[str, str]) -> FactCheckVerdict:
    """Lexical fact check: >=40% content-word overlap with a cited source and every number present."""
    cited = [cid for cid in claim.citations if cid in sources]
    missing = [cid for cid in claim.citations if cid not in sources]
    if not claim.citations:
        return FactCheckVerdict(claim_id=claim.id, supported=False, reason="claim cites no sources")
    if missing:
        return FactCheckVerdict(claim_id=claim.id, supported=False, reason=f"cites unknown source(s): {', '.join(missing)}")
    if claim.kind == "analysis":
        return FactCheckVerdict(claim_id=claim.id, supported=True, reason="analysis claim; citations exist")
    words = set(content_words(claim.text))
    best = 0.0
    for cid in cited:
        src_words = set(content_words(sources[cid]))
        if words:
            best = max(best, len(words & src_words) / len(words))
    source_numbers = set().union(*(numbers_in(sources[cid]) for cid in cited))
    missing_numbers = sorted(numbers_in(claim.text) - source_numbers)
    if missing_numbers:
        return FactCheckVerdict(
            claim_id=claim.id, supported=False, reason=f"number(s) {', '.join(missing_numbers)} not found in cited sources"
        )
    if best < 0.4:
        return FactCheckVerdict(claim_id=claim.id, supported=False, reason=f"only {best:.0%} of claim terms appear in cited sources")
    return FactCheckVerdict(claim_id=claim.id, supported=True, reason=f"{best:.0%} of claim terms found in cited source; numbers match")


def _claim_matches_golden(golden: dict[str, Any], text: str) -> bool:
    low = text.lower()
    companies = [str(c).lower() for c in golden.get("companies") or []]
    if not any(c in low for c in companies):
        return False
    keywords = [str(k).lower() for k in golden.get("keywords") or []]
    try:
        synonyms = EVENT_SYNONYMS.get(EventType(golden.get("event_type")), ())
    except ValueError:
        synonyms = ()
    words = set(_WORD.findall(low))
    return any(k in low for k in keywords) or any(s in words for s in synonyms)


_OPPOSITES = (
    ({"rises", "rose", "increase", "increases", "increased", "grew", "grows", "up"}, {"falls", "fell", "decrease", "decreases", "decreased", "shrank", "shrinks", "down"}),
    ({"approved", "approves", "cleared"}, {"blocked", "blocks", "rejected", "rejects"}),
    ({"completed", "completes", "closed"}, {"abandoned", "terminated", "collapsed"}),
)


def _contradictions(claims: list[Claim]) -> int:
    """Count claim pairs that cite different sources about the same subject with opposite verbs."""
    found = 0
    for i, a in enumerate(claims):
        wa = set(_WORD.findall(a.text.lower()))
        for b in claims[i + 1 :]:
            wb = set(_WORD.findall(b.text.lower()))
            shared = set(content_words(a.text)) & set(content_words(b.text))
            if len(shared) < 3:
                continue
            if any((wa & x and wb & y) or (wa & y and wb & x) for x, y in _OPPOSITES):
                found += 1
    return found


def _scale(ratio: float, cutoffs: tuple[float, float, float, float]) -> int:
    for score, cutoff in zip((5, 4, 3, 2), cutoffs):
        if ratio >= cutoff:
            return score
    return 1


def _dedupe_claims(claims: list[Claim]) -> list[Claim]:
    seen: set[str] = set()
    out = []
    for c in claims:
        key = _norm(c.text)
        if key and key not in seen:
            seen.add(key)
            out.append(c)
    return out


def _renumber(draft: BriefDraft) -> BriefDraft:
    """Assign ids c1..cN in reading order (executive summary, sections, implications)."""
    counter = 0

    def renumber(claims: list[Claim]) -> list[Claim]:
        nonlocal counter
        out = []
        for c in claims:
            counter += 1
            out.append(c.model_copy(update={"id": f"c{counter}"}))
        return out

    return draft.model_copy(
        update={
            "executive_summary": renumber(draft.executive_summary),
            "sections": [s.model_copy(update={"claims": renumber(s.claims)}) for s in draft.sections],
            "implications": renumber(draft.implications),
        }
    )


def _lead_sentence(text: str) -> str:
    sentences = split_sentences(text)
    return sentences[0] if sentences else text.strip()


def _year_of(value: Any) -> int | None:
    m = re.match(r"(\d{4})-", str(value or ""))
    return int(m.group(1)) if m else None


def _require(ctx: dict[str, Any], key: str, task: str) -> Any:
    value = ctx.get(key)
    if value is None:
        raise LLMError(f"offline task {task!r} needs context[{key!r}]")
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value
