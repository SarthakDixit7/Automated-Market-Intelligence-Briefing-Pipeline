"""Human review gate: queue persistence, decisions, publishing and the terminal prompt."""

from __future__ import annotations

import io
from datetime import timedelta

import pytest
from rich.console import Console

import marketbrief.pipeline as pipeline
from marketbrief.review import ReviewQueue, interactive_review


@pytest.fixture
def queue(settings):
    return ReviewQueue(settings)


@pytest.fixture
def published(monkeypatch):
    """Record publish calls instead of writing client deliverables."""
    calls = []
    monkeypatch.setattr(pipeline, "publish_brief", lambda brief, settings: calls.append(brief) or {})
    return calls


def test_submit_and_load_round_trip(queue, sample_brief):
    draft = sample_brief.model_copy(update={"status": "draft"})
    path = queue.submit(draft)
    assert path == queue.dir / f"{sample_brief.id}.json"
    assert path.exists()
    loaded = queue.load(sample_brief.id)
    assert loaded.status == "needs_review"
    assert loaded.review is None
    assert loaded.all_claims() == sample_brief.all_claims()


def test_load_missing_or_invalid_id_raises_key_error(queue):
    with pytest.raises(KeyError):
        queue.load("b_20260101_000000")
    with pytest.raises(KeyError):
        queue.load("../../etc/passwd")


def test_list_filters_by_status_newest_first(queue, sample_brief, published):
    older = sample_brief.model_copy(update={"id": "b_20261001_aaaaaa",
                                            "created_at": sample_brief.created_at - timedelta(days=3)})
    queue.submit(older)
    queue.submit(sample_brief)
    queue.decide(older.id, "rejected", "Ana")

    assert [b.id for b in queue.list()] == [sample_brief.id, older.id]
    assert [b.id for b in queue.list("needs_review")] == [sample_brief.id]
    assert [b.id for b in queue.list("rejected")] == [older.id]
    assert queue.list("approved") == []


def test_list_skips_corrupt_files(queue, sample_brief):
    queue.submit(sample_brief)
    (queue.dir / "b_broken.json").write_text("{not json", encoding="utf-8")
    assert [b.id for b in queue.list()] == [sample_brief.id]


def test_approve_publishes_and_locks_the_brief(queue, sample_brief, published):
    queue.submit(sample_brief)
    approved = queue.decide(sample_brief.id, "approved", "Jane Analyst", "Looks good")

    assert approved.status == "approved"
    assert approved.review.reviewer == "Jane Analyst"
    assert approved.review.notes == "Looks good"
    assert [b.id for b in published] == [sample_brief.id]
    assert published[0].status == "approved"
    assert queue.load(sample_brief.id).status == "approved"

    with pytest.raises(ValueError, match="already approved"):
        queue.decide(sample_brief.id, "rejected", "Someone else")


def test_reject_is_final(queue, sample_brief, published):
    queue.submit(sample_brief)
    rejected = queue.decide(sample_brief.id, "rejected", "Ana", "Off-topic")
    assert rejected.status == "rejected"
    assert queue.load(sample_brief.id).review.notes == "Off-topic"
    assert published == []
    with pytest.raises(ValueError):
        queue.decide(sample_brief.id, "approved", "Ana")


def test_changes_requested_can_be_decided_again(queue, sample_brief, published):
    queue.submit(sample_brief)
    assert queue.decide(sample_brief.id, "changes_requested", "Ana", "Tighten c4").status == "changes_requested"
    assert queue.decide(sample_brief.id, "approved", "Ana").status == "approved"
    assert len(published) == 1


def test_failed_publish_leaves_brief_decidable(queue, sample_brief, monkeypatch):
    def boom(brief, settings):
        raise OSError("disk full")

    monkeypatch.setattr(pipeline, "publish_brief", boom)
    queue.submit(sample_brief)
    with pytest.raises(OSError):
        queue.decide(sample_brief.id, "approved", "Ana")
    assert queue.load(sample_brief.id).status == "needs_review"


def test_decide_validates_input(queue, sample_brief):
    queue.submit(sample_brief)
    with pytest.raises(ValueError, match="unknown review decision"):
        queue.decide(sample_brief.id, "maybe", "Ana")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="reviewer"):
        queue.decide(sample_brief.id, "approved", "  ")
    with pytest.raises(KeyError):
        queue.decide("b_missing_000000", "approved", "Ana")


def _quiet_console() -> Console:
    return Console(file=io.StringIO(), width=100)


@pytest.mark.parametrize(
    ("answer", "expected"),
    [("a", "approved"), ("r", "rejected"), ("c", "changes_requested"), ("approve", "approved")],
)
def test_interactive_review_decisions(sample_brief, answer, expected):
    answers = iter([answer, "my notes"])
    decision = interactive_review(sample_brief, reviewer="Jane", ask=lambda _: next(answers), console=_quiet_console())
    assert decision is not None
    assert decision.decision == expected
    assert decision.reviewer == "Jane"
    assert decision.notes == "my notes"


def test_interactive_review_skip_and_shows_brief(sample_brief):
    console = _quiet_console()
    assert interactive_review(sample_brief, reviewer="Jane", ask=lambda _: "s", console=console) is None
    shown = console.file.getvalue()
    assert sample_brief.title in shown
    assert "Citation check passed" in shown
    assert "(weak)" in shown


def test_interactive_review_rejects_unknown_choice(sample_brief):
    with pytest.raises(ValueError):
        interactive_review(sample_brief, reviewer="Jane", ask=lambda _: "x", console=_quiet_console())
