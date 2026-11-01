"""Ingestion: URL/text normalisation, source adapters, the JSONL store and ingest()."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest
import yaml

from marketbrief.config import Settings, load_settings
from marketbrief.ingest import IngestResult, ingest, select_sources
from marketbrief.ingest.normalize import (
    article_id,
    canonicalize_url,
    clean_text,
    content_hash,
    make_article,
)
from marketbrief.ingest.sources import fetch_source, load_sources
from marketbrief.ingest.store import ArticleStore, JsonlStore
from marketbrief.schemas import Article, RawItem, SourceConfig

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "data" / "fixtures" / "articles"
UTC = timezone.utc


@pytest.fixture
def data_settings(tmp_path: Path) -> Settings:
    """Real sources/fixtures, every writable path inside tmp_path."""
    return load_settings(
        root=ROOT,
        overrides={
            "llm": {"provider": "offline"},
            "paths": {
                "raw_dir": str(tmp_path / "raw"),
                "store_dir": str(tmp_path / "store"),
            },
        },
    )


def demo_source(settings: Settings) -> SourceConfig:
    return next(s for s in load_sources(settings.path(settings.paths.sources_file)) if s.id == "demo_wire")


# --------------------------------------------------------------------------- #
# normalize.py
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("HTTPS://Example.COM/News/Story/", "https://example.com/News/Story"),
        ("https://example.com/a?utm_source=rss&utm_medium=x&id=7#comments", "https://example.com/a?id=7"),
        ("https://example.com/a?fbclid=abc&gclid=def&ref=hn", "https://example.com/a"),
        ("https://example.com:443/a?b=2&a=1", "https://example.com/a?a=1&b=2"),
        ("http://example.com:8080/a", "http://example.com:8080/a"),
        ("", ""),
        (None, ""),
    ],
)
def test_canonicalize_url(url: str | None, expected: str) -> None:
    assert canonicalize_url(url) == expected


def test_clean_text_strips_html_and_keeps_paragraphs() -> None:
    html = (
        "<div><h1>Title</h1><p>First &amp; <b>bold</b>\n   paragraph.</p>"
        "<script>var x = 1;</script><style>p{}</style>"
        "<p>Second&nbsp;one &#8212; with   spaces.</p><br/>Tail</div>"
    )
    assert clean_text(html) == "Title\n\nFirst & bold paragraph.\n\nSecond one — with spaces.\n\nTail"


def test_clean_text_plain_text_paragraphs_preserved() -> None:
    text = "  One line\nwrapped.  \n\n\n  Two   here. \n \n Three."
    assert clean_text(text) == "One line wrapped.\n\nTwo here.\n\nThree."
    assert clean_text("") == "" and clean_text(None) == ""


def test_article_id_is_stable_and_ignores_tracking_params() -> None:
    a = article_id("src", "https://Example.com/x/?utm_source=feed", "Title")
    b = article_id("other_src", "https://example.com/x", "Different title")
    assert a == b
    assert re.fullmatch(r"a_[0-9a-f]{12}", a)
    # Without a URL the id falls back to source + case-insensitive title.
    assert article_id("s", None, "Big News") == article_id("s", "", "big news")
    assert article_id("s", None, "Big News") != article_id("t", None, "Big News")


def test_content_hash_ignores_case_whitespace_and_punctuation() -> None:
    assert content_hash("Hello, World!\n\nAgain.") == content_hash("hello world again")
    assert content_hash("hello world") != content_hash("hello there world")


def test_make_article_normalises_fields() -> None:
    src = SourceConfig(id="s1", name="Feed", kind="rss", category="ai")
    raw = RawItem(
        source_id="s1",
        source_name="Outlet",
        title="  <b>Big</b>   news ",
        url="https://EXAMPLE.com/story/?utm_campaign=z",
        published_at=datetime(2026, 10, 5, 12, 0),  # naive -> UTC
        text="<p>Para one.</p><p>Para two.</p>",
    )
    art = make_article(raw, src)
    assert art.title == "Big news"
    assert art.url == "https://example.com/story"
    assert art.text == "Para one.\n\nPara two."
    assert art.published_at == datetime(2026, 10, 5, 12, 0, tzinfo=UTC)
    assert art.source_name == "Outlet" and art.category == "ai"
    assert art.content_hash == content_hash(art.text)
    assert art.id == article_id("s1", raw.url, "Big news")


# --------------------------------------------------------------------------- #
# sources.py
# --------------------------------------------------------------------------- #


def test_load_sources_reads_config(data_settings: Settings) -> None:
    sources = load_sources(data_settings.path(data_settings.paths.sources_file))
    assert any(s.id == "demo_wire" and s.kind == "json_dir" and s.enabled for s in sources)
    assert load_sources(ROOT / "does-not-exist.yaml") == []


def test_fetch_json_dir_reads_fixtures(data_settings: Settings) -> None:
    items = fetch_source(demo_source(data_settings), data_settings)
    assert len(items) == len(list(FIXTURE_DIR.glob("*.json")))
    assert all(it.raw.get("synthetic") is True for it in items)
    # The per-record "source" key overrides the configured source name.
    assert {it.source_name for it in items} == {
        "Datacenter Dispatch", "Cloud Ledger", "The Silicon Wire", "Market Pulse Daily", "InfraWeekly",
    }
    assert all(it.published_at and it.published_at.tzinfo is not None for it in items)


def test_fetch_json_dir_since_and_max_items(data_settings: Settings) -> None:
    src = demo_source(data_settings)
    since = datetime(2026, 10, 20, tzinfo=UTC)
    recent = fetch_source(src, data_settings, since=since)
    assert recent and all(it.published_at >= since for it in recent)
    capped = fetch_source(src.model_copy(update={"max_items": 3}), data_settings)
    assert len(capped) == 3


def test_fetch_json_dir_handles_lists_bad_files_and_missing_dir(tmp_path: Path) -> None:
    folder = tmp_path / "feed"
    folder.mkdir()
    (folder / "a.json").write_text(
        json.dumps([
            {"title": "One", "url": "https://x.example/1", "published_at": "2026-10-01T00:00:00Z", "text": "t1"},
            {"title": "Two", "published_at": "2026-10-02", "text": "t2", "source": "Override"},
            {"no_title": True},
        ]),
        encoding="utf-8",
    )
    (folder / "broken.json").write_text("{not json", encoding="utf-8")
    settings = load_settings(root=tmp_path)
    src = SourceConfig(id="f", name="Folder feed", kind="json_dir", path="feed")
    errors: list[str] = []
    items = fetch_source(src, settings, errors=errors)
    assert [it.title for it in items] == ["One", "Two"]
    assert items[0].source_name == "Folder feed" and items[1].source_name == "Override"
    assert items[1].published_at == datetime(2026, 10, 2, tzinfo=UTC)
    assert len(errors) == 2  # broken file + record without a title

    missing = SourceConfig(id="m", name="Missing", kind="json_dir", path="nope")
    errors.clear()
    assert fetch_source(missing, settings, errors=errors) == []
    assert errors and "not found" in errors[0]


def test_fetch_jsonl_skips_bad_lines(tmp_path: Path) -> None:
    file = tmp_path / "items.jsonl"
    file.write_text(
        '{"title": "A", "text": "alpha"}\n\nnot-json\n{"title": "B", "text": "beta"}\n', encoding="utf-8"
    )
    settings = load_settings(root=tmp_path)
    src = SourceConfig(id="j", name="Lines", kind="jsonl", path="items.jsonl")
    errors: list[str] = []
    items = fetch_source(src, settings, errors=errors)
    assert [it.title for it in items] == ["A", "B"]
    assert len(errors) == 1


RSS_FEED = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>Example feed</title>
<item>
  <title>Kestrel Networks ships a switch</title>
  <link>https://news.example/kestrel?utm_source=rss</link>
  <pubDate>Tue, 06 Oct 2026 10:00:00 GMT</pubDate>
  <description>&lt;p&gt;Short &lt;b&gt;summary&lt;/b&gt;.&lt;/p&gt;</description>
</item>
<item>
  <title>Older story</title>
  <link>https://news.example/old</link>
  <pubDate>Mon, 01 Sep 2026 10:00:00 GMT</pubDate>
  <description>Old.</description>
</item>
</channel></rss>"""

ARTICLE_PAGE = b"""<html><body><nav>Menu</nav>
<p>Full paragraph one.</p><div>ad</div><p>Full <a href="#">paragraph</a> two.</p></body></html>"""


def _mock_http(routes: dict[str, bytes]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["user-agent"].startswith("marketbrief/")
        url = str(request.url).split("?")[0]
        if url in routes:
            return httpx.Response(200, content=routes[url])
        return httpx.Response(404)

    return httpx.Client(transport=httpx.MockTransport(handler), headers={"User-Agent": "marketbrief/0.1"})


def test_fetch_rss_parses_feed_and_filters_since() -> None:
    http = _mock_http({"https://feed.example/rss": RSS_FEED})
    src = SourceConfig(id="r", name="Example", kind="rss", url="https://feed.example/rss")
    items = fetch_source(src, Settings(), http=http)
    assert [it.title for it in items] == ["Kestrel Networks ships a switch", "Older story"]
    assert items[0].published_at == datetime(2026, 10, 6, 10, tzinfo=UTC)
    assert "summary" in items[0].text

    recent = fetch_source(src, Settings(), http=http, since=datetime(2026, 10, 1, tzinfo=UTC))
    assert [it.title for it in recent] == ["Kestrel Networks ships a switch"]


def test_fetch_rss_full_text() -> None:
    http = _mock_http({
        "https://feed.example/rss": RSS_FEED,
        "https://news.example/kestrel": ARTICLE_PAGE,
    })
    src = SourceConfig(id="r", name="Example", kind="rss", url="https://feed.example/rss", fetch_full_text=True)
    items = fetch_source(src, Settings(), http=http)
    assert items[0].text == "Full paragraph one.\n\nFull paragraph two."
    # Page fetch failed (404) -> falls back to the feed summary.
    assert items[1].text == "Old."


def test_fetch_rss_network_error_returns_empty() -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    http = httpx.Client(transport=httpx.MockTransport(boom))
    src = SourceConfig(id="r", name="Example", kind="rss", url="https://feed.example/rss")
    errors: list[str] = []
    assert fetch_source(src, Settings(), http=http, errors=errors) == []
    assert errors and "r:" in errors[0]


# --------------------------------------------------------------------------- #
# store.py
# --------------------------------------------------------------------------- #


def _article(i: int, day: int | None, **kw: object) -> Article:
    return Article(
        id=f"a_{i:012x}",
        source_id="s",
        source_name="S",
        title=f"T{i}",
        published_at=datetime(2026, 10, day, tzinfo=UTC) if day else None,
        fetched_at=datetime(2026, 10, 31, tzinfo=UTC),
        text=f"text {i}",
        content_hash=content_hash(f"text {i}"),
        **kw,
    )


def test_jsonl_store_upsert_replace_and_reopen(tmp_path: Path) -> None:
    path = tmp_path / "s" / "articles.jsonl"
    store = JsonlStore(path, Article)
    assert len(store) == 0 and store.get("x") is None
    assert store.upsert([_article(1, 1), _article(2, 2)]) == 2
    assert store.upsert([_article(2, 2, category="updated"), _article(3, 3)]) == 1
    assert len(store) == 3
    assert store.get("a_000000000002").category == "updated"

    reopened = JsonlStore(path, Article)
    assert [a.id for a in reopened.all()] == [a.id for a in store.all()]
    assert reopened.get("a_000000000002").category == "updated"
    assert "a_000000000003" in reopened


def test_jsonl_store_skips_corrupt_lines(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    path.write_text(_article(1, 1).model_dump_json() + "\n{broken\n", encoding="utf-8")
    assert len(JsonlStore(path, Article)) == 1


def test_article_store_between(tmp_path: Path) -> None:
    store = ArticleStore(tmp_path / "articles.jsonl")
    store.upsert([_article(3, 20), _article(1, 5), _article(2, 10), _article(4, None)])
    got = store.between(datetime(2026, 10, 5, tzinfo=UTC), datetime(2026, 10, 20, tzinfo=UTC))
    assert [a.id for a in got] == ["a_000000000001", "a_000000000002", "a_000000000003"]
    # Undated article falls back to fetched_at (Oct 31); bounds are inclusive and optional.
    assert [a.id for a in store.between(datetime(2026, 10, 21), None)] == ["a_000000000004"]
    assert len(store.between(None, None)) == 4


# --------------------------------------------------------------------------- #
# ingest()
# --------------------------------------------------------------------------- #


def test_ingest_fixture_feed(data_settings: Settings) -> None:
    result = ingest(data_settings, run_id="r_test")
    assert isinstance(result, IngestResult)
    n_files = len(list(FIXTURE_DIR.glob("*.json")))
    assert len(result.articles) == n_files
    assert result.per_source == {"demo_wire": n_files}
    assert result.errors == []

    times = [a.published_at for a in result.articles]
    assert times == sorted(times)
    assert len({a.id for a in result.articles}) == n_files

    raw = json.loads(result.raw_path.read_text(encoding="utf-8"))
    assert result.raw_path.name == "r_test.json"
    assert [r["id"] for r in raw] == [a.id for a in result.articles]

    store = ArticleStore(data_settings.path(data_settings.paths.store_dir) / "articles.jsonl")
    assert len(store) == n_files

    # Re-ingesting is idempotent: same ids, nothing new in the store, fetch time kept.
    again = ingest(data_settings, run_id="r_test2")
    assert [a.id for a in again.articles] == [a.id for a in result.articles]
    assert [a.fetched_at for a in again.articles] == [a.fetched_at for a in result.articles]
    assert len(ArticleStore(store.path)) == n_files


def test_ingest_window_and_cap(data_settings: Settings) -> None:
    window = ingest(
        data_settings,
        since=datetime(2026, 10, 9, tzinfo=UTC),
        until=datetime(2026, 10, 16, 23, 59, tzinfo=UTC),
    )
    assert window.articles
    assert all(datetime(2026, 10, 9, tzinfo=UTC) <= a.published_at <= datetime(2026, 10, 17, tzinfo=UTC)
               for a in window.articles)

    capped_settings = data_settings.model_copy(deep=True)
    capped_settings.pipeline.max_articles = 5
    capped = ingest(capped_settings)
    assert len(capped.articles) == 5
    # The most recent articles are kept, oldest first.
    assert capped.articles[-1].published_at.date().isoformat() == "2026-10-30"


def test_ingest_source_selection(data_settings: Settings) -> None:
    sources = load_sources(data_settings.path(data_settings.paths.sources_file))
    assert [s.id for s in select_sources(sources, None)] == [s.id for s in sources if s.enabled]
    errors: list[str] = []
    picked = select_sources(sources, ["techcrunch", "nope"], errors)
    assert [s.id for s in picked] == ["techcrunch"]  # named sources run even when disabled
    assert errors == ["unknown source id 'nope'"]

    result = ingest(data_settings, source_ids=["nope"])
    assert result.articles == [] and result.errors


# --------------------------------------------------------------------------- #
# Fixture dataset quality
# --------------------------------------------------------------------------- #

OUTLETS = {"Datacenter Dispatch", "Cloud Ledger", "The Silicon Wire", "Market Pulse Daily", "InfraWeekly"}

# (file slug, subject company, trigger keyword, amount or None) for G1..G12.
GOLDEN_TRIGGERS = [
    ("aurora-silicon-series-d", "Aurora Silicon", "Series D", "$410 million"),
    ("aurora-silicon-funding-syndicated", "Aurora Silicon", "funding", "$410 million"),
    ("meridian-to-acquire-vireo", "Vireo Analytics", "acquire", "$1.2 billion"),
    ("meridian-vireo-deal-syndicated", "Meridian Cloud", "acquire", "$1.2 billion"),
    ("meridian-vireo-analysis", "Vireo Analytics", "acquire", "$1.2 billion"),
    ("northwind-gpu-price-cut", "Northwind Compute", "pricing", "22%"),
    ("northwind-pricing-syndicated", "Northwind Compute", "pricing", "22%"),
    ("lumenforge-lf3-launch", "LumenForge AI", "launch", None),
    ("helix-lumenforge-partnership", "LumenForge AI", "partnership", None),
    ("eu-inquiry-meridian-egress", "Meridian Cloud", "inquiry", "9%"),
    ("bastion-labs-appoints-okafor-ceo", "Bastion Labs", "CEO", None),
    ("kestrel-networks-workforce-reduction", "Kestrel Networks", "workforce", "9%"),
    ("northwind-q3-results", "Northwind Compute", "revenue", "$312 million"),
    ("aurora-silicon-bengaluru-design-centre", "Aurora Silicon", "opens", None),
    ("kestrel-sues-helix-patents", "Helix Data Systems", "patent", "$85 million"),
    ("meridian-frankfurt-sovereign-region", "Meridian Cloud", "opens", None),
]


def _fixture_docs() -> dict[str, dict]:
    return {p.stem.split("_", 1)[1]: json.loads(p.read_text(encoding="utf-8")) for p in FIXTURE_DIR.glob("*.json")}


def test_fixture_articles_are_synthetic_and_well_formed() -> None:
    docs = _fixture_docs()
    assert 20 <= len(docs) <= 28
    for path in FIXTURE_DIR.glob("*.json"):
        doc = json.loads(path.read_text(encoding="utf-8"))
        assert doc["synthetic"] is True
        assert doc["source"] in OUTLETS
        assert path.name.startswith(doc["published_at"][:10] + "_")
        assert "2026-10-05" <= doc["published_at"][:10] <= "2026-10-30"
        assert 120 <= len(doc["text"].split()) <= 260, path.name
        assert doc["text"].count("\n\n") >= 3, f"{path.name} should have several paragraphs"
        assert ".example/" in doc["url"]  # reserved, clearly fictional domains


@pytest.mark.parametrize(("slug", "company", "trigger", "amount"), GOLDEN_TRIGGERS)
def test_golden_event_signals_share_a_sentence(slug: str, company: str, trigger: str, amount: str | None) -> None:
    """Heuristic extraction takes the first sentence holding the trigger; the
    company and amount must be in that same sentence."""
    text = _fixture_docs()[slug]["text"]
    sentences = re.split(r"(?<=[.!?])\s+", text.replace("\n\n", " "))
    first = next(s for s in sentences if trigger in s)
    assert company in first, first
    if amount:
        assert amount in first, first


def test_entities_gazetteer_covers_fixture_companies() -> None:
    data = yaml.safe_load((ROOT / "data" / "fixtures" / "entities.yaml").read_text(encoding="utf-8"))
    names = {c["name"] for c in data["companies"]}
    assert names == {
        "Northwind Compute", "Helix Data Systems", "Aurora Silicon", "Meridian Cloud",
        "LumenForge AI", "Bastion Labs", "Vireo Analytics", "Kestrel Networks",
    }
    assert all(c["category"] == c["category"].lower() and c["aliases"] for c in data["companies"])


def test_prior_briefs_are_valid_approved_briefs() -> None:
    from marketbrief.schemas import Brief

    paths = sorted((ROOT / "data" / "fixtures" / "prior_briefs").glob("*.json"))
    assert any(p.name == "b_20260930_prior1.json" for p in paths)
    for p in paths:
        brief = Brief.model_validate_json(p.read_text(encoding="utf-8"))
        assert brief.status == "approved" and brief.topic == "Enterprise AI infrastructure"
        cited = {c for claim in brief.all_claims() for c in claim.citations}
        assert cited and cited <= {s.article_id for s in brief.sources}
        ids = [c.id for c in brief.all_claims()]
        assert len(ids) == len(set(ids))
    prior1 = Brief.model_validate_json((paths[-1]).read_text(encoding="utf-8"))
    text = " ".join(c.text for c in prior1.all_claims()).lower()
    for needle in ("price war", "talks to buy vireo", "aurora silicon", "egress"):
        assert needle in text
