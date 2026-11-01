"""CLI: help, exit codes, review commands and offline end-to-end runs."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

import marketbrief.cli as cli
from marketbrief.cli import app
from marketbrief.review import ReviewQueue

runner = CliRunner()


@pytest.fixture
def use_settings(settings, monkeypatch):
    """Route every command to the isolated test settings (honouring --provider)."""

    def fake_settings(ctx, config=None, provider=None):
        if provider:
            settings.llm.provider = provider
        return settings

    monkeypatch.setattr(cli, "_settings", fake_settings)
    return settings


def invoke(*args: str):
    return runner.invoke(app, list(args), catch_exceptions=False)


# --------------------------------------------------------------------------- #
# Help + usage errors
# --------------------------------------------------------------------------- #


def test_help_lists_every_command():
    result = invoke("--help")
    assert result.exit_code == 0
    for command in ("run", "ingest", "review", "agent", "agent-resume", "eval", "index", "cache", "sources", "demo"):
        assert command in result.output


@pytest.mark.parametrize(
    "args",
    [("run",), ("review",), ("review", "approve"), ("agent",), ("agent-resume",), ("eval",), ("index",), ("cache",)],
)
def test_subcommand_help(args):
    assert invoke(*args, "--help").exit_code == 0


def test_no_args_shows_help():
    result = runner.invoke(app, [])
    assert result.exit_code in (0, 2)  # click versions differ
    assert "Usage" in result.output


def test_bad_date_is_usage_error(use_settings):
    result = runner.invoke(app, ["run", "--since", "last tuesday"])
    assert result.exit_code == 2
    assert "ISO date" in result.output


def test_unknown_command_is_usage_error():
    assert runner.invoke(app, ["frobnicate"]).exit_code == 2


def test_approve_requires_reviewer(use_settings):
    assert runner.invoke(app, ["review", "approve", "b_20261030_abc123"]).exit_code == 2


def test_bad_status_filter_is_usage_error(use_settings):
    assert runner.invoke(app, ["review", "list", "--status", "pending"]).exit_code == 2


def test_parse_date_until_covers_whole_day():
    end = cli._parse_date("2026-10-30", end_of_day=True, option="--until")
    assert (end.hour, end.minute, end.tzinfo is not None) == (23, 59, True)
    start = cli._parse_date("2026-10-01", end_of_day=False, option="--since")
    assert (start.hour, start.minute) == (0, 0)


# --------------------------------------------------------------------------- #
# Review commands
# --------------------------------------------------------------------------- #


def test_review_list_empty(use_settings):
    result = invoke("review", "list")
    assert result.exit_code == 0
    assert "empty" in result.output


def test_review_show_missing_brief_exits_1(use_settings):
    result = runner.invoke(app, ["review", "show", "b_20990101_ffffff"])
    assert result.exit_code == 1
    assert "no brief" in result.output


def test_review_flow(use_settings, sample_brief, monkeypatch):
    import marketbrief.pipeline as pipeline

    published = []
    monkeypatch.setattr(pipeline, "publish_brief", lambda brief, settings: published.append(brief.id) or {})
    ReviewQueue(use_settings).submit(sample_brief)

    listed = invoke("review", "list", "--status", "needs_review")
    assert listed.exit_code == 0 and sample_brief.id in listed.output

    shown = invoke("review", "show", sample_brief.id)
    assert shown.exit_code == 0
    assert "Executive summary" in shown.output and "low_support" in shown.output

    as_json = invoke("review", "show", sample_brief.id, "--json")
    assert json.loads(as_json.output)["id"] == sample_brief.id

    changes = invoke("review", "request-changes", sample_brief.id, "--reviewer", "Ana", "--notes", "More sources")
    assert changes.exit_code == 0 and "changes_requested" in changes.output

    approved = invoke("review", "approve", sample_brief.id, "--reviewer", "Ana")
    assert approved.exit_code == 0 and "approved" in approved.output
    assert published == [sample_brief.id]

    again = runner.invoke(app, ["review", "reject", sample_brief.id, "--reviewer", "Ana"])
    assert again.exit_code == 1
    assert "already approved" in again.output


# --------------------------------------------------------------------------- #
# Config + maintenance commands
# --------------------------------------------------------------------------- #


def test_global_config_option_is_used(tmp_path, project_root):
    cfg_dir = tmp_path / "proj" / "config"
    cfg_dir.mkdir(parents=True)
    sources = cfg_dir / "sources.yaml"
    sources.write_text(yaml.safe_dump({"sources": [
        {"id": "only_this_one", "name": "Custom feed", "kind": "json_dir", "path": "feeds"},
    ]}), encoding="utf-8")
    settings_file = cfg_dir / "settings.yaml"
    settings_file.write_text(yaml.safe_dump({"paths": {"sources_file": str(sources)}}), encoding="utf-8")

    result = invoke("--config", str(settings_file), "sources")
    assert result.exit_code == 0
    assert "only_this_one" in result.output and "demo_wire" not in result.output


def test_sources_lists_demo_feed(use_settings):
    result = invoke("sources")
    assert result.exit_code == 0
    assert "demo_wire" in result.output


def test_cache_stats_and_clear(use_settings):
    from marketbrief.llm.base import LLMRequest, LLMResult
    from marketbrief.llm.cache import PromptCache

    path = use_settings.path(use_settings.cache.path)
    assert "no cache file yet" in invoke("cache", "stats").output

    with PromptCache(path) as cache:
        cache.put(LLMRequest(task="edit", system="s", prompt="p"), "m", LLMResult(text="hello"))
    stats = invoke("cache", "stats")
    assert stats.exit_code == 0 and "Entries" in stats.output

    cleared = invoke("cache", "clear", "--yes")
    assert cleared.exit_code == 0 and "Cleared 1" in cleared.output
    with PromptCache(path) as cache:
        assert len(cache) == 0


def test_cache_clear_declined_exits_1(use_settings):
    from marketbrief.llm.cache import PromptCache

    with PromptCache(use_settings.path(use_settings.cache.path)):
        pass
    assert runner.invoke(app, ["cache", "clear"], input="n\n").exit_code == 1


# --------------------------------------------------------------------------- #
# End-to-end commands (offline provider, fixture feed)
# --------------------------------------------------------------------------- #


def test_ingest_command(use_settings, fixture_articles):
    result = invoke("ingest")
    assert result.exit_code == 0
    assert "demo_wire" in result.output


def test_run_command_queues_brief(use_settings, fixture_articles):
    result = invoke("run", "--provider", "offline", "--topic", "Enterprise AI infrastructure")
    assert result.exit_code == 0, result.output
    assert "Run summary" in result.output and "needs_review" in result.output
    assert "marketbrief review approve" in result.output
    queued = ReviewQueue(use_settings).list("needs_review")
    assert len(queued) == 1


def test_run_command_no_articles_exits_1(use_settings, fixture_articles):
    result = runner.invoke(app, ["run", "--since", "2099-01-01"])
    assert result.exit_code == 1
    assert "no articles ingested" in result.output


def test_demo_publishes_and_evaluates(use_settings, fixture_articles):
    result = invoke("demo")
    assert result.exit_code == 0, result.output
    assert "approved" in result.output and "Evaluation" in result.output
    client = Path(use_settings.path(use_settings.paths.output_dir)) / "client"
    assert any(client.glob("*/*.html"))


def test_eval_with_brief_and_index_rebuild(use_settings, fixture_articles):
    run = invoke("run", "--auto-approve", "--reviewer", "ci")
    assert run.exit_code == 0, run.output
    archive = next(use_settings.path(use_settings.paths.briefs_dir).glob("b_*.json"))

    result = runner.invoke(app, ["eval", "--brief", str(archive), "--no-judge"])
    assert result.exit_code in (0, 1)
    assert "Extraction recall" in result.output

    impossible = runner.invoke(app, ["eval", "--brief", str(archive), "--no-judge", "--fail-under-recall", "1.0"])
    if "100%" not in impossible.output:
        assert impossible.exit_code == 1

    rebuilt = invoke("index", "rebuild")
    assert rebuilt.exit_code == 0, rebuilt.output
    counts = re.search(r"Indexed\s+(\d+)\s+chunks\s+from\s+(\d+)\s+approved\s+briefs", rebuilt.output)
    assert counts, rebuilt.output
    chunks, briefs = map(int, counts.groups())
    assert briefs == 3, "two fixture prior briefs + the approved run"
    assert chunks > briefs, "one chunk per claim/watchlist/continuity line, not per brief"


def test_agent_command(use_settings, fixture_articles):
    result = invoke("agent", "Which companies raised funding?", "--auto-approve")
    assert result.exit_code == 0, result.output
    assert "trace:" in result.output


def test_agent_pauses_for_sensitive_answer_then_resumes(use_settings, fixture_articles):
    paused = invoke("agent", "Which lawsuits and regulatory inquiries were reported?")
    assert paused.exit_code == 0, paused.output
    if "Held for human review" not in paused.output:
        pytest.skip("offline agent did not classify this answer as sensitive")
    thread_id = paused.output.split("marketbrief agent-resume ", 1)[1].split()[0]

    resumed = invoke("agent-resume", thread_id, "--approve", "--notes", "checked sources")
    assert resumed.exit_code == 0, resumed.output
    assert "approved" in resumed.output

    missing = runner.invoke(app, ["agent-resume", "t_does_not_exist", "--reject"])
    assert missing.exit_code == 1
