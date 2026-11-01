"""Unit tests for the LangGraph-style state machine (no LLMs involved)."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, Field

from marketbrief.agent.graph import (
    END,
    Checkpoint,
    GraphError,
    JsonCheckpointer,
    MemoryCheckpointer,
    StateGraph,
)


class Counter(BaseModel):
    n: int = 0
    visited: list[str] = Field(default_factory=list)
    approved: bool | None = None
    note: str = ""


def _visit(name: str, inc: int = 0):
    def node(state: Counter) -> dict:
        return {"n": state.n + inc, "visited": [*state.visited, name]}

    return node


def linear_graph() -> StateGraph[Counter]:
    g = StateGraph(Counter)
    g.add_node("a", _visit("a", 1))
    g.add_node("b", _visit("b", 10))
    g.set_entry_point("a")
    g.add_edge("a", "b")
    g.add_edge("b", END)
    return g


def loop_graph() -> StateGraph[Counter]:
    """inc -> (n < 3 ? inc : review) -> END; review is a natural HITL point."""
    g = StateGraph(Counter)
    g.add_node("inc", _visit("inc", 1))
    g.add_node("review", lambda s: {"note": f"reviewed approved={s.approved}"})
    g.set_entry_point("inc")
    g.add_conditional_edges("inc", lambda s: "again" if s.n < 3 else "done", {"again": "inc", "done": "review"})
    g.add_edge("review", END)
    return g


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #


def test_linear_run_merges_partial_updates_and_traces():
    run = linear_graph().compile().invoke(Counter())
    assert run.status == "completed"
    assert run.next_node is None
    assert run.trace == ["a", "b"]
    assert run.state.n == 11
    assert run.state.visited == ["a", "b"]


def test_invoke_accepts_dict_and_does_not_mutate_caller_state():
    initial = Counter(n=5)
    run = linear_graph().compile().invoke(initial)
    assert run.state.n == 16
    assert initial.n == 5 and initial.visited == []
    assert linear_graph().compile().invoke({"n": 1}).state.n == 12


def test_conditional_loop_runs_until_router_exits():
    run = loop_graph().compile().invoke(Counter())
    assert run.trace == ["inc", "inc", "inc", "review"]
    assert run.state.n == 3


def test_router_without_mapping_returns_node_names_or_end():
    g = StateGraph(Counter)
    g.add_node("start", _visit("start"))
    g.add_node("odd", _visit("odd"))
    g.set_entry_point("start")
    g.add_conditional_edges("start", lambda s: "odd" if s.n % 2 else END)
    g.add_edge("odd", END)
    compiled = g.compile()
    assert compiled.invoke(Counter(n=1)).trace == ["start", "odd"]
    assert compiled.invoke(Counter(n=2)).trace == ["start"]


def test_node_may_return_full_state_or_none():
    g = StateGraph(Counter)
    g.add_node("full", lambda s: Counter(n=s.n + 100))
    g.add_node("noop", lambda s: None)
    g.set_entry_point("full")
    g.add_edge("full", "noop")
    g.add_edge("noop", END)
    run = g.compile().invoke(Counter(n=1))
    assert run.state.n == 101
    assert run.trace == ["full", "noop"]


def test_unknown_update_key_and_invalid_values_raise():
    g = StateGraph(Counter)
    g.add_node("bad", lambda s: {"nope": 1})
    g.set_entry_point("bad")
    g.add_edge("bad", END)
    with pytest.raises(GraphError, match="unknown state keys"):
        g.compile().invoke(Counter())

    g2 = StateGraph(Counter)
    g2.add_node("bad", lambda s: {"n": "not a number"})
    g2.set_entry_point("bad")
    g2.add_edge("bad", END)
    with pytest.raises(GraphError, match="invalid state"):
        g2.compile().invoke(Counter())


def test_node_returning_wrong_type_raises():
    g = StateGraph(Counter)
    g.add_node("bad", lambda s: 42)
    g.set_entry_point("bad")
    g.add_edge("bad", END)
    with pytest.raises(GraphError, match="must return"):
        g.compile().invoke(Counter())


def test_step_limit_raises():
    g = StateGraph(Counter)
    g.add_node("spin", _visit("spin", 1))
    g.set_entry_point("spin")
    g.add_edge("spin", "spin")
    with pytest.raises(GraphError, match="step limit"):
        g.compile().invoke(Counter(), max_steps=5)


def test_router_returning_unmapped_key_raises():
    g = StateGraph(Counter)
    g.add_node("a", _visit("a"))
    g.set_entry_point("a")
    g.add_conditional_edges("a", lambda s: "elsewhere", {"done": END})
    with pytest.raises(GraphError, match="unmapped key"):
        g.compile().invoke(Counter())

    g2 = StateGraph(Counter)
    g2.add_node("a", _visit("a"))
    g2.set_entry_point("a")
    g2.add_conditional_edges("a", lambda s: "ghost")
    with pytest.raises(GraphError, match="unknown node"):
        g2.compile().invoke(Counter())


# --------------------------------------------------------------------------- #
# Definition errors
# --------------------------------------------------------------------------- #


def test_missing_entry_point_raises():
    g = StateGraph(Counter)
    g.add_node("a", _visit("a"))
    g.add_edge("a", END)
    with pytest.raises(GraphError, match="entry point"):
        g.compile()


def test_entry_point_must_be_a_node():
    g = linear_graph()
    g.set_entry_point("zzz")
    with pytest.raises(GraphError, match="entry point"):
        g.compile()


def test_edge_to_unknown_node_raises_at_compile():
    g = StateGraph(Counter)
    g.add_node("a", _visit("a"))
    g.set_entry_point("a")
    g.add_edge("a", "missing")
    with pytest.raises(GraphError, match="unknown node 'missing'"):
        g.compile()


def test_conditional_mapping_to_unknown_node_raises_at_compile():
    g = StateGraph(Counter)
    g.add_node("a", _visit("a"))
    g.set_entry_point("a")
    g.add_conditional_edges("a", lambda s: "x", {"x": "missing"})
    with pytest.raises(GraphError, match="unknown node"):
        g.compile()


def test_dead_end_node_raises_at_compile():
    g = StateGraph(Counter)
    g.add_node("a", _visit("a"))
    g.add_node("b", _visit("b"))
    g.set_entry_point("a")
    g.add_edge("a", "b")
    with pytest.raises(GraphError, match="no outgoing edge"):
        g.compile()


def test_duplicate_nodes_and_double_routing_rejected():
    g = StateGraph(Counter)
    g.add_node("a", _visit("a"))
    with pytest.raises(GraphError):
        g.add_node("a", _visit("a"))
    with pytest.raises(GraphError):
        g.add_node(END, _visit("x"))
    g.add_edge("a", END)
    with pytest.raises(GraphError, match="already has an outgoing edge"):
        g.add_conditional_edges("a", lambda s: END)


def test_interrupt_before_unknown_node_raises():
    with pytest.raises(GraphError, match="unknown node"):
        linear_graph().compile(checkpointer=MemoryCheckpointer(), interrupt_before=["ghost"])


def test_interrupt_requires_checkpointer():
    compiled = linear_graph().compile(interrupt_before=["b"])
    with pytest.raises(GraphError, match="checkpointer"):
        compiled.invoke(Counter())


def test_state_type_must_be_pydantic():
    with pytest.raises(GraphError):
        StateGraph(dict)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Checkpoints, interrupts and resume
# --------------------------------------------------------------------------- #


@pytest.fixture(params=["memory", "json"])
def checkpointer(request, tmp_path):
    return MemoryCheckpointer() if request.param == "memory" else JsonCheckpointer(tmp_path / "ckpt")


def test_interrupt_before_saves_checkpoint_and_stops(checkpointer):
    compiled = loop_graph().compile(checkpointer=checkpointer, interrupt_before=["review"])
    run = compiled.invoke(Counter(), thread_id="t1")
    assert run.status == "interrupted"
    assert run.next_node == "review"
    assert run.trace == ["inc", "inc", "inc"]
    assert run.state.note == ""  # review has not run

    saved = checkpointer.load("t1")
    assert saved is not None
    assert saved.status == "interrupted"
    assert saved.next_node == "review"
    assert saved.state["n"] == 3


def test_resume_applies_updates_and_does_not_reinterrupt(checkpointer):
    compiled = loop_graph().compile(checkpointer=checkpointer, interrupt_before=["review"])
    compiled.invoke(Counter(), thread_id="t1")

    run = compiled.resume("t1", updates={"approved": True})
    assert run.status == "completed"
    assert run.trace == ["inc", "inc", "inc", "review"]
    assert run.state.approved is True
    assert run.state.note == "reviewed approved=True"
    assert checkpointer.load("t1").status == "completed"


def test_resume_completed_thread_raises(checkpointer):
    compiled = loop_graph().compile(checkpointer=checkpointer, interrupt_before=["review"])
    compiled.invoke(Counter(), thread_id="t1")
    compiled.resume("t1", updates={"approved": False})
    with pytest.raises(GraphError, match="already completed"):
        compiled.resume("t1")


def test_resume_unknown_thread_raises(checkpointer):
    compiled = loop_graph().compile(checkpointer=checkpointer, interrupt_before=["review"])
    with pytest.raises(GraphError, match="no checkpoint"):
        compiled.resume("nope")


def test_resume_with_invalid_update_raises(checkpointer):
    compiled = loop_graph().compile(checkpointer=checkpointer, interrupt_before=["review"])
    compiled.invoke(Counter(), thread_id="t1")
    with pytest.raises(GraphError, match="unknown state keys"):
        compiled.resume("t1", updates={"bogus": 1})


def test_threads_are_isolated(checkpointer):
    compiled = loop_graph().compile(checkpointer=checkpointer, interrupt_before=["review"])
    compiled.invoke(Counter(), thread_id="t1")
    compiled.invoke(Counter(n=10), thread_id="t2")  # skips the loop body after one inc
    assert checkpointer.load("t1").state["n"] == 3
    assert checkpointer.load("t2").state["n"] == 11
    assert compiled.resume("t2", updates={"approved": False}).state.approved is False
    assert checkpointer.load("t1").status == "interrupted"


def test_checkpoint_written_after_every_node(checkpointer):
    seen: list[tuple[str | None, list[str]]] = []

    class Spy:
        def save(self, cp: Checkpoint) -> None:
            seen.append((cp.next_node, list(cp.trace)))
            checkpointer.save(cp)

        def load(self, thread_id):
            return checkpointer.load(thread_id)

        def delete(self, thread_id):
            checkpointer.delete(thread_id)

    linear_graph().compile(checkpointer=Spy()).invoke(Counter(), thread_id="t")
    assert seen == [("b", ["a"]), (None, ["a", "b"])]
    assert checkpointer.load("t").status == "completed"


def test_resume_after_crash_continues_from_last_checkpoint(checkpointer):
    calls = {"b": 0}

    def flaky_b(state: Counter) -> dict:
        calls["b"] += 1
        if calls["b"] == 1:
            raise RuntimeError("boom")
        return {"n": state.n + 10, "visited": [*state.visited, "b"]}

    g = StateGraph(Counter)
    g.add_node("a", _visit("a", 1))
    g.add_node("b", flaky_b)
    g.set_entry_point("a")
    g.add_edge("a", "b")
    g.add_edge("b", END)
    compiled = g.compile(checkpointer=checkpointer)
    with pytest.raises(RuntimeError):
        compiled.invoke(Counter(), thread_id="crash")
    assert checkpointer.load("crash").next_node == "b"

    run = compiled.resume("crash")
    assert run.status == "completed"
    assert run.state.n == 11  # node a did not run twice
    assert run.trace == ["a", "b"]


def test_json_checkpoint_survives_new_graph_instance(tmp_path):
    ckpt_dir = tmp_path / "agent"
    first = loop_graph().compile(checkpointer=JsonCheckpointer(ckpt_dir), interrupt_before=["review"])
    first.invoke(Counter(), thread_id="run/42")  # unsafe chars are sanitised in the file name

    files = list(ckpt_dir.glob("*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text(encoding="utf-8"))
    assert data["thread_id"] == "run/42"
    assert data["next_node"] == "review"
    assert data["state"]["n"] == 3

    # A fresh process: new checkpointer + newly compiled graph.
    second = loop_graph().compile(checkpointer=JsonCheckpointer(ckpt_dir), interrupt_before=["review"])
    assert second.get_state("run/42").n == 3
    run = second.resume("run/42", updates={"approved": True, "note": "ignored-overwritten"})
    assert run.state.note == "reviewed approved=True"


def test_corrupt_json_checkpoint_raises(tmp_path):
    cp = JsonCheckpointer(tmp_path)
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(GraphError, match="corrupt"):
        cp.load("bad")


def test_memory_checkpointer_isolated_from_later_mutation():
    cp = MemoryCheckpointer()
    state = {"n": 1, "visited": ["a"]}
    cp.save(Checkpoint(thread_id="t", state=state, next_node="b"))
    state["visited"].append("mutated")
    assert cp.load("t").state["visited"] == ["a"]
    assert cp.thread_ids() == ["t"]
    cp.delete("t")
    assert cp.load("t") is None


def test_draw_mermaid_lists_nodes_edges_and_interrupts():
    compiled = loop_graph().compile(checkpointer=MemoryCheckpointer(), interrupt_before=["review"])
    chart = compiled.draw_mermaid()
    assert chart.startswith("flowchart TD")
    assert "inc -. again .-> inc" in chart
    assert "inc -. done .-> review" in chart
    assert "review --> __end__" in chart
    assert "review{{review}}" in chart  # interrupt nodes drawn as hexagons
