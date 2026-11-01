"""Deliverable rendering: Markdown, HTML and JSON."""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser

from marketbrief.render import number_sources, render_html, render_json, render_markdown, write_deliverables
from marketbrief.schemas import Brief, ReviewDecision


class _Collector(HTMLParser):
    """Collects element ids and in-page href targets."""

    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.anchors: list[str] = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "id" in a:
            self.ids.add(a["id"])
        if tag == "a" and (a.get("href") or "").startswith("#"):
            self.anchors.append(a["href"][1:])


def _approved(brief: Brief) -> Brief:
    return brief.model_copy(
        update={"status": "approved", "review": ReviewDecision(decision="approved", reviewer="Jane Analyst")}
    )


def test_sources_numbered_by_first_citation(sample_brief):
    entries = number_sources(sample_brief)
    # c1 cites the Meridian article first, so it is source [1].
    assert [e.ref.article_id for e in entries] == ["a_sample000002", "a_sample000001", "a_sample000003"]
    assert [e.number for e in entries] == [1, 2, 3]
    assert entries[0].anchor == "src-1"


def test_html_footnotes_link_to_source_anchors(sample_brief):
    html = render_html(sample_brief)
    parser = _Collector()
    parser.feed(html)
    assert parser.anchors, "claims should carry footnote links"
    assert set(parser.anchors) <= parser.ids, "every footnote must resolve to an anchor"
    assert {"src-1", "src-2", "src-3"} <= parser.ids
    assert '<a href="#src-1"' in html


def test_html_is_self_contained_and_print_friendly(sample_brief):
    html = render_html(sample_brief)
    assert html.lstrip().startswith("<!DOCTYPE html>")
    assert "<style>" in html and "@media print" in html
    assert not re.search(r'<link[^>]+stylesheet|<script\b', html), "no external assets or scripts"
    for heading in ("Executive summary", "Strategic implications", "Watchlist", "Continuity with prior briefs", "Sources"):
        assert heading in html
    assert sample_brief.title in html
    assert "Also reported by Cloud Ledger" in html


def test_html_marks_weak_claims_and_escapes_text(sample_brief):
    html = render_html(sample_brief)
    assert "(weakly supported)" in html
    assert "&lt;effective immediately&gt;" in html
    assert "<effective immediately>" not in html
    assert "&lt;22%&gt; &amp; more" in html


def test_draft_watermark_only_until_approved(sample_brief):
    draft_html = render_html(sample_brief)
    assert 'class="watermark"' in draft_html
    assert "Not for client distribution" in draft_html

    approved_html = render_html(_approved(sample_brief))
    assert 'class="watermark"' not in approved_html
    assert "Not for client distribution" not in approved_html
    assert "Jane Analyst" in approved_html
    assert "badge-approved" in approved_html


def test_markdown_footnotes_and_sections(sample_brief):
    md = render_markdown(sample_brief)
    assert md.startswith(f"# {sample_brief.title}")
    assert "> **DRAFT" in md
    # c1 cites source 1; implications cite sources 1 and 3.
    assert "Meridian Cloud agreed to acquire Vireo Analytics for $1.2 billion.[^1]" in md
    assert r"by 22% \<effective immediately>" in md
    assert "[^1][^3]" in md
    assert "*(weakly supported)*" in md
    assert "[^1]: **Meridian Cloud to acquire Vireo Analytics for $1.2 billion.** Market Pulse Daily, 08 Oct 2026." in md
    assert "Also reported by: Cloud Ledger." in md
    assert "**Updates** (`b_20260930_prior1`): The reported Meridian-Vireo talks" in md
    # Every marker used in the text has a definition.
    used = set(re.findall(r"\[\^(\d+)\](?!:)", md))
    defined = set(re.findall(r"^\[\^(\d+)\]:", md, flags=re.M))
    assert used and used <= defined
    assert "\n\n\n\n" not in md, "templates should not leave large blank gaps"


def test_markdown_drops_draft_banner_when_approved(sample_brief):
    md = render_markdown(_approved(sample_brief))
    assert "DRAFT" not in md
    assert "Approved by Jane Analyst" in md


def test_continuity_text_used_when_no_structured_notes(sample_brief):
    brief = sample_brief.model_copy(update={"continuity": [], "continuity_text": ["Continues: GPU price war."]})
    assert "Continues: GPU price war." in render_markdown(brief)
    assert "Continues: GPU price war." in render_html(brief)


def test_render_json_round_trips(sample_brief):
    data = json.loads(render_json(sample_brief))
    assert data["id"] == sample_brief.id
    assert Brief.model_validate(data) == sample_brief


def test_write_deliverables(tmp_path, sample_brief):
    paths = write_deliverables(sample_brief, tmp_path / "out")
    assert set(paths) == {"md", "html", "json"}
    for ext, path in paths.items():
        assert path.name == f"{sample_brief.id}.{ext}"
        assert path.read_text(encoding="utf-8").strip()
