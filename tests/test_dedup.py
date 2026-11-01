"""De-duplication: exact hashes, embedding near-duplicates, canonical selection."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from marketbrief.config import load_settings
from marketbrief.dedup import DedupResult, dedup_text, dedupe
from marketbrief.embeddings import HashingEmbedder, cosine_matrix
from marketbrief.ingest.normalize import content_hash, make_article
from marketbrief.ingest.sources import fetch_source, load_sources
from marketbrief.schemas import Article

ROOT = Path(__file__).resolve().parents[1]
UTC = timezone.utc
THRESHOLD = load_settings(root=ROOT).dedup.threshold


@pytest.fixture(scope="module")
def embedder() -> HashingEmbedder:
    return HashingEmbedder(1024)


@pytest.fixture(scope="module")
def fixture_articles_by_slug() -> dict[str, Article]:
    settings = load_settings(root=ROOT)
    src = next(s for s in load_sources(settings.path(settings.paths.sources_file)) if s.id == "demo_wire")
    articles = [make_article(raw, src) for raw in fetch_source(src, settings)]
    return {a.url.rsplit("/", 1)[1]: a for a in articles}


@pytest.fixture(scope="module")
def fixture_result(fixture_articles_by_slug: dict[str, Article], embedder: HashingEmbedder) -> DedupResult:
    return dedupe(list(fixture_articles_by_slug.values()), embedder, threshold=THRESHOLD)


def _cluster_of(result: DedupResult, article_id: str) -> set[str]:
    for c in result.clusters:
        if article_id in c.member_ids:
            return set(c.member_ids)
    return {article_id}


SYNDICATED_PAIRS = [
    ("aurora-silicon-series-d", "aurora-silicon-funding-syndicated"),
    ("meridian-to-acquire-vireo", "meridian-vireo-deal-syndicated"),
    ("northwind-gpu-price-cut", "northwind-pricing-syndicated"),
    ("lumenforge-lf3-launch", "lumenforge-lf3-wire-copy"),
]

# Different stories about the same company (or the same deal) must stay apart.
DISTINCT_SAME_COMPANY = [
    "meridian-to-acquire-vireo",
    "eu-inquiry-meridian-egress",
    "meridian-frankfurt-sovereign-region",
    "aurora-silicon-series-d",
    "aurora-silicon-bengaluru-design-centre",
    "aurora-northwind-a9-supply-deal",
    "northwind-gpu-price-cut",
    "northwind-q3-results",
    "opinion-gpu-rental-price-war",
    "kestrel-800g-fabric-switch",
    "kestrel-networks-workforce-reduction",
    "kestrel-sues-helix-patents",
    "vireo-llm-trace",
]


@pytest.mark.parametrize(("a", "b"), SYNDICATED_PAIRS)
def test_syndicated_near_duplicates_cluster(
    fixture_result: DedupResult, fixture_articles_by_slug: dict[str, Article], a: str, b: str
) -> None:
    ida, idb = fixture_articles_by_slug[a].id, fixture_articles_by_slug[b].id
    assert {ida, idb} <= _cluster_of(fixture_result, ida)


def test_distinct_stories_about_same_company_do_not_cluster(
    fixture_result: DedupResult, fixture_articles_by_slug: dict[str, Article]
) -> None:
    ids = [fixture_articles_by_slug[s].id for s in DISTINCT_SAME_COMPANY]
    for i, x in enumerate(ids):
        for y in ids[i + 1:]:
            assert y not in _cluster_of(fixture_result, x), (x, y)


def test_threshold_has_margin_on_fixtures(
    fixture_articles_by_slug: dict[str, Article], embedder: HashingEmbedder
) -> None:
    """The configured threshold sits well inside the gap between syndicated
    copies and distinct stories, so small text edits will not flip results."""
    arts = fixture_articles_by_slug
    vec = {slug: embedder.embed([dedup_text(a)])[0] for slug, a in arts.items()}
    dup_sims = [float(cosine_matrix(vec[a], vec[b])[0, 0]) for a, b in SYNDICATED_PAIRS]
    distinct_sims = [
        float(cosine_matrix(vec[x], vec[y])[0, 0])
        for i, x in enumerate(DISTINCT_SAME_COMPANY)
        for y in DISTINCT_SAME_COMPANY[i + 1:]
    ]
    analysis = float(cosine_matrix(vec["meridian-to-acquire-vireo"], vec["meridian-vireo-analysis"])[0, 0])
    assert min(dup_sims) >= THRESHOLD + 0.05
    assert max(distinct_sims + [analysis]) <= THRESHOLD - 0.3


def test_fixture_dedup_summary(fixture_result: DedupResult, fixture_articles_by_slug: dict[str, Article]) -> None:
    arts = fixture_articles_by_slug
    assert len(fixture_result.clusters) == len(SYNDICATED_PAIRS)
    assert fixture_result.removed == len(SYNDICATED_PAIRS)
    assert len(fixture_result.kept) == len(arts) - len(SYNDICATED_PAIRS)
    methods = {c.canonical_id: c.method for c in fixture_result.clusters}
    lf3 = arts["lumenforge-lf3-launch"]
    assert methods[lf3.id] == "exact"  # verbatim wire copy
    assert methods[arts["aurora-silicon-series-d"].id] == "embedding"

    kept = {a.id: a for a in fixture_result.kept}
    # The longer original wins over the shorter syndicated rewrite and lists it.
    aurora = kept[arts["aurora-silicon-series-d"].id]
    assert aurora.duplicate_ids == [arts["aurora-silicon-funding-syndicated"].id]
    assert aurora.corroborating_sources == ["Market Pulse Daily"]
    # Exact copy with equal length: earliest publication is canonical.
    assert kept[lf3.id].corroborating_sources == ["InfraWeekly"]
    assert arts["lumenforge-lf3-wire-copy"].id not in kept


def _make(i: int, text: str, *, source: str = "Outlet A", title: str | None = None, day: int = 1) -> Article:
    return Article(
        id=f"a_{i:012d}",
        source_id="src",
        source_name=source,
        title=title or f"Story {i}",
        published_at=datetime(2026, 10, 1, tzinfo=UTC) + timedelta(days=day),
        text=text,
        content_hash=content_hash(text),
    )


BODY = (
    "Northwind Compute is cutting on-demand pricing for its H-class GPU instances by 22%, the GPU cloud "
    "provider said on Friday. The reduction applies in all regions and to cluster reservations."
)


def test_exact_duplicates_collapse_despite_formatting(embedder: HashingEmbedder) -> None:
    a = _make(1, BODY, source="Outlet A", title="Original headline", day=2)
    # Same length, different case and punctuation -> same content hash.
    b = _make(2, BODY.upper().replace(",", ";"), source="Outlet B", title="Completely different words", day=1)
    c = _make(3, "Unrelated story about datacenter power and grid connection queues in Europe.", day=3)
    result = dedupe([a, b, c], embedder, threshold=0.99)
    assert result.removed == 1
    assert [x.id for x in result.kept] == [b.id, c.id]  # equal length -> earliest published wins
    assert result.clusters[0].method == "exact"
    assert result.clusters[0].max_similarity == 1.0
    assert result.kept[0].duplicate_ids == [a.id]
    assert result.kept[0].corroborating_sources == ["Outlet A"]


def test_canonical_is_longest_and_kept_order_follows_input(embedder: HashingEmbedder) -> None:
    other = _make(1, "Kestrel Networks will cut 9% of its workforce after customers delayed switch orders.")
    short = _make(2, BODY, source="Wire", title="Northwind cuts GPU pricing 22%")
    longer = _make(3, BODY + " Contracts signed in the past 90 days will be repriced.", source="Desk",
                   title="Northwind cuts GPU pricing 22%")
    result = dedupe([other, short, longer], embedder, threshold=0.8)
    assert [a.id for a in result.kept] == [other.id, longer.id]
    canon = result.kept[1]
    assert canon.duplicate_ids == [short.id] and canon.corroborating_sources == ["Wire"]
    assert result.clusters[0].method == "embedding"
    assert result.clusters[0].member_ids == [short.id, longer.id]
    # Inputs are not mutated.
    assert longer.duplicate_ids == []


def test_same_outlet_duplicates_do_not_count_as_corroboration(embedder: HashingEmbedder) -> None:
    a = _make(1, BODY, source="Outlet A")
    b = _make(2, BODY, source="Outlet A", day=3)
    result = dedupe([a, b], embedder)
    assert result.kept[0].duplicate_ids == [b.id]
    assert result.kept[0].corroborating_sources == []


def test_dedupe_edge_cases(embedder: HashingEmbedder) -> None:
    assert dedupe([], embedder) == DedupResult(kept=[], clusters=[], removed=0)
    single = _make(1, BODY)
    result = dedupe([single], embedder)
    assert result.kept == [single] and result.clusters == [] and result.removed == 0
