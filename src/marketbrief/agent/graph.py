"""A small LangGraph-style state machine with checkpoints and interrupts.

The engine runs a typed Pydantic state through named nodes:

* ``add_node(name, fn)`` - ``fn(state)`` returns a new state, a partial-update
  ``dict`` merged into the state (and re-validated), or ``None`` for no change.
* ``add_edge(src, dst)`` - unconditional transition (``dst`` may be ``END``).
* ``add_conditional_edges(src, router, mapping)`` - ``router(state)`` returns a
  key; ``mapping`` turns it into a node name (without a mapping the key *is*
  the node name).
* ``compile(checkpointer=..., interrupt_before=[...])`` - a runnable graph.
  Before entering a node listed in ``interrupt_before`` the run is
  checkpointed and returned with ``status="interrupted"``; ``resume()`` loads
  the checkpoint, applies human-supplied updates and continues from that node
  without interrupting at it a second time.

A checkpoint is written after every node, so a crashed run can also be
resumed from the last completed step.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Generic, Literal, Mapping, Protocol, Sequence, TypeVar

from pydantic import BaseModel, ValidationError

log = logging.getLogger(__name__)

END = "__end__"

S = TypeVar("S", bound=BaseModel)

NodeFn = Callable[[Any], Any]
Router = Callable[[Any], str]
RunStatus = Literal["completed", "interrupted"]


class GraphError(RuntimeError):
    """Invalid graph definition or a run that cannot proceed."""


# --------------------------------------------------------------------------- #
# Run results and checkpoints
# --------------------------------------------------------------------------- #


@dataclass
class GraphRun(Generic[S]):
    """Outcome of ``invoke``/``resume``."""

    state: S
    status: RunStatus
    next_node: str | None
    trace: list[str]


@dataclass
class Checkpoint:
    """Serialisable snapshot of a thread: state plus where to continue."""

    thread_id: str
    state: dict[str, Any]
    next_node: str | None
    trace: list[str] = field(default_factory=list)
    status: Literal["running", "interrupted", "completed"] = "running"
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_json(self) -> dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "state": self.state,
            "next_node": self.next_node,
            "trace": list(self.trace),
            "status": self.status,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "Checkpoint":
        try:
            return cls(
                thread_id=str(data["thread_id"]),
                state=dict(data["state"]),
                next_node=data.get("next_node"),
                trace=list(data.get("trace", [])),
                status=data.get("status", "running"),
                updated_at=data.get("updated_at", ""),
            )
        except (KeyError, TypeError) as exc:
            raise GraphError(f"malformed checkpoint: {exc}") from exc


class Checkpointer(Protocol):
    def save(self, checkpoint: Checkpoint) -> None: ...

    def load(self, thread_id: str) -> Checkpoint | None: ...

    def delete(self, thread_id: str) -> None: ...


class MemoryCheckpointer:
    """Keeps checkpoints in a dict (tests, single-process use)."""

    def __init__(self) -> None:
        self._data: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def save(self, checkpoint: Checkpoint) -> None:
        # Store a JSON round-trip so later mutation of the live state can't leak in.
        with self._lock:
            self._data[checkpoint.thread_id] = json.loads(json.dumps(checkpoint.to_json()))

    def load(self, thread_id: str) -> Checkpoint | None:
        with self._lock:
            raw = self._data.get(thread_id)
        return Checkpoint.from_json(raw) if raw is not None else None

    def delete(self, thread_id: str) -> None:
        with self._lock:
            self._data.pop(thread_id, None)

    def thread_ids(self) -> list[str]:
        with self._lock:
            return sorted(self._data)


_SAFE_ID = re.compile(r"[^A-Za-z0-9_.-]+")


class JsonCheckpointer:
    """One JSON file per thread in ``dir`` (survives process restarts)."""

    def __init__(self, dir: Path | str) -> None:  # noqa: A002 - contract name
        self.dir = Path(dir)
        self._lock = threading.Lock()

    def _file(self, thread_id: str) -> Path:
        safe = _SAFE_ID.sub("_", thread_id).strip("._") or "default"
        return self.dir / f"{safe}.json"

    def save(self, checkpoint: Checkpoint) -> None:
        path = self._file(checkpoint.thread_id)
        payload = json.dumps(checkpoint.to_json(), indent=2, ensure_ascii=False)
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(payload, encoding="utf-8")
            tmp.replace(path)  # atomic on the same filesystem

    def load(self, thread_id: str) -> Checkpoint | None:
        path = self._file(thread_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise GraphError(f"corrupt checkpoint file {path}: {exc}") from exc
        return Checkpoint.from_json(data)

    def delete(self, thread_id: str) -> None:
        self._file(thread_id).unlink(missing_ok=True)

    def thread_ids(self) -> list[str]:
        if not self.dir.exists():
            return []
        return sorted(p.stem for p in self.dir.glob("*.json"))


# --------------------------------------------------------------------------- #
# Graph definition
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _Conditional:
    router: Router
    mapping: dict[str, str] | None


class StateGraph(Generic[S]):
    """Builder for a state machine over a Pydantic ``state_type``."""

    def __init__(self, state_type: type[S]) -> None:
        if not (isinstance(state_type, type) and issubclass(state_type, BaseModel)):
            raise GraphError("state_type must be a pydantic BaseModel subclass")
        self.state_type = state_type
        self.nodes: dict[str, NodeFn] = {}
        self.edges: dict[str, str] = {}
        self.conditional: dict[str, _Conditional] = {}
        self.entry_point: str | None = None

    def add_node(self, name: str, fn: NodeFn) -> "StateGraph[S]":
        if not name or name == END:
            raise GraphError(f"invalid node name {name!r}")
        if name in self.nodes:
            raise GraphError(f"node {name!r} already exists")
        if not callable(fn):
            raise GraphError(f"node {name!r} must be callable")
        self.nodes[name] = fn
        return self

    def add_edge(self, src: str, dst: str) -> "StateGraph[S]":
        self._check_unrouted(src)
        self.edges[src] = dst
        return self

    def add_conditional_edges(
        self, src: str, router: Router, mapping: dict[str, str] | None = None
    ) -> "StateGraph[S]":
        self._check_unrouted(src)
        if not callable(router):
            raise GraphError(f"router for {src!r} must be callable")
        self.conditional[src] = _Conditional(router, dict(mapping) if mapping is not None else None)
        return self

    def set_entry_point(self, name: str) -> "StateGraph[S]":
        self.entry_point = name
        return self

    def _check_unrouted(self, src: str) -> None:
        if src == END:
            raise GraphError("END cannot have outgoing edges")
        if src in self.edges or src in self.conditional:
            raise GraphError(f"node {src!r} already has an outgoing edge")

    def validate(self, interrupt_before: Sequence[str] = ()) -> None:
        """Raise :class:`GraphError` if the graph is not runnable."""
        if self.entry_point is None:
            raise GraphError("no entry point set (call set_entry_point)")
        if self.entry_point not in self.nodes:
            raise GraphError(f"entry point {self.entry_point!r} is not a node")
        for src, dst in self.edges.items():
            self._require_node(src, "edge source")
            if dst != END:
                self._require_node(dst, f"edge target from {src!r}")
        for src, cond in self.conditional.items():
            self._require_node(src, "conditional edge source")
            for dst in (cond.mapping or {}).values():
                if dst != END:
                    self._require_node(dst, f"conditional target from {src!r}")
        for name in self.nodes:
            if name not in self.edges and name not in self.conditional:
                raise GraphError(f"node {name!r} has no outgoing edge (route it to END explicitly)")
        for name in interrupt_before:
            self._require_node(name, "interrupt_before entry")

    def _require_node(self, name: str, what: str) -> None:
        if name not in self.nodes:
            raise GraphError(f"unknown node {name!r} ({what})")

    def compile(
        self,
        *,
        checkpointer: Checkpointer | None = None,
        interrupt_before: Sequence[str] = (),
    ) -> "CompiledGraph[S]":
        self.validate(interrupt_before)
        return CompiledGraph(self, checkpointer=checkpointer, interrupt_before=interrupt_before)


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #


class CompiledGraph(Generic[S]):
    """A validated, runnable graph. Holds a frozen copy of the definition."""

    def __init__(
        self,
        graph: StateGraph[S],
        *,
        checkpointer: Checkpointer | None,
        interrupt_before: Sequence[str],
    ) -> None:
        self.state_type: type[S] = graph.state_type
        self.nodes = dict(graph.nodes)
        self.edges = dict(graph.edges)
        self.conditional = dict(graph.conditional)
        self.entry_point: str = graph.entry_point  # type: ignore[assignment]  # validated
        self.checkpointer = checkpointer
        self.interrupt_before = frozenset(interrupt_before)

    # -- public API --------------------------------------------------------- #

    def invoke(
        self, state: S | Mapping[str, Any], *, thread_id: str = "default", max_steps: int = 50
    ) -> GraphRun[S]:
        """Run from the entry point until END or an interrupt."""
        current = self._coerce_state(state)
        return self._run(current, self.entry_point, [], thread_id, max_steps, skip_interrupt=None)

    def resume(
        self, thread_id: str, *, updates: Mapping[str, Any] | None = None, max_steps: int = 50
    ) -> GraphRun[S]:
        """Continue a checkpointed thread, optionally applying state updates first."""
        checkpoint = self.get_checkpoint(thread_id)
        if checkpoint is None:
            raise GraphError(f"no checkpoint for thread {thread_id!r}")
        if checkpoint.status == "completed" or checkpoint.next_node in (None, END):
            raise GraphError(f"thread {thread_id!r} has already completed")
        if checkpoint.next_node not in self.nodes:
            raise GraphError(f"checkpoint points at unknown node {checkpoint.next_node!r}")
        state = self._coerce_state(checkpoint.state)
        if updates:
            state = self._merge(state, dict(updates), source="resume updates")
        return self._run(
            state,
            checkpoint.next_node,
            list(checkpoint.trace),
            thread_id,
            max_steps,
            skip_interrupt=checkpoint.next_node,
        )

    def get_checkpoint(self, thread_id: str) -> Checkpoint | None:
        if self.checkpointer is None:
            raise GraphError("graph was compiled without a checkpointer")
        return self.checkpointer.load(thread_id)

    def get_state(self, thread_id: str) -> S | None:
        """The last checkpointed state of a thread, if any."""
        checkpoint = self.get_checkpoint(thread_id)
        return self._coerce_state(checkpoint.state) if checkpoint else None

    def draw_mermaid(self) -> str:
        """Mermaid flowchart of the graph (for docs and debugging)."""
        lines = ["flowchart TD", f"    __start__([start]) --> {self.entry_point}"]
        for name in self.nodes:
            shape = f"{name}{{{{{name}}}}}" if name in self.interrupt_before else f"{name}[{name}]"
            lines.append(f"    {shape}")
        for src, dst in self.edges.items():
            lines.append(f"    {src} --> {self._mermaid_id(dst)}")
        for src, cond in self.conditional.items():
            if cond.mapping:
                for key, dst in cond.mapping.items():
                    lines.append(f"    {src} -. {key} .-> {self._mermaid_id(dst)}")
            else:
                lines.append(f"    {src} -. router .-> ?")
        lines.append("    __end__([end])")
        return "\n".join(lines)

    # -- internals ---------------------------------------------------------- #

    @staticmethod
    def _mermaid_id(name: str) -> str:
        return "__end__" if name == END else name

    def _run(
        self,
        state: S,
        node: str,
        trace: list[str],
        thread_id: str,
        max_steps: int,
        *,
        skip_interrupt: str | None,
    ) -> GraphRun[S]:
        steps = 0
        while node != END:
            if node in self.interrupt_before and node != skip_interrupt:
                if self.checkpointer is None:
                    raise GraphError(f"interrupt before {node!r} requires a checkpointer")
                self._save(thread_id, state, node, trace, "interrupted")
                log.info("thread %s interrupted before %s", thread_id, node)
                return GraphRun(state=state, status="interrupted", next_node=node, trace=trace)
            skip_interrupt = None
            if steps >= max_steps:
                raise GraphError(f"step limit of {max_steps} reached (last node {trace[-1] if trace else node!r})")

            log.debug("thread %s -> %s", thread_id, node)
            result = self.nodes[node](state)
            state = self._merge(state, result, source=f"node {node!r}")
            trace.append(node)
            steps += 1
            node = self._next_node(node, state)
            if self.checkpointer is not None:
                self._save(thread_id, state, node, trace, "completed" if node == END else "running")

        return GraphRun(state=state, status="completed", next_node=None, trace=trace)

    def _next_node(self, current: str, state: S) -> str:
        if current in self.edges:
            return self.edges[current]
        cond = self.conditional[current]
        key = cond.router(state)
        if cond.mapping is not None:
            if key not in cond.mapping:
                raise GraphError(f"router for {current!r} returned unmapped key {key!r}")
            return cond.mapping[key]
        if key != END and key not in self.nodes:
            raise GraphError(f"router for {current!r} returned unknown node {key!r}")
        return key

    def _merge(self, state: S, result: Any, *, source: str) -> S:
        if result is None:
            return state
        if isinstance(result, self.state_type):
            return result
        if isinstance(result, Mapping):
            unknown = set(result) - set(self.state_type.model_fields)
            if unknown:
                raise GraphError(f"{source} returned unknown state keys: {sorted(unknown)}")
            merged = {name: getattr(state, name) for name in self.state_type.model_fields}
            merged.update(result)
            try:
                return self.state_type.model_validate(merged)
            except ValidationError as exc:
                raise GraphError(f"{source} produced an invalid state: {exc}") from exc
        raise GraphError(
            f"{source} must return {self.state_type.__name__}, a dict update or None, "
            f"not {type(result).__name__}"
        )

    def _coerce_state(self, state: S | Mapping[str, Any]) -> S:
        if isinstance(state, self.state_type):
            return state.model_copy(deep=True)
        if isinstance(state, BaseModel):
            state = state.model_dump()
        try:
            return self.state_type.model_validate(state)
        except ValidationError as exc:
            raise GraphError(f"invalid initial state: {exc}") from exc

    def _save(self, thread_id: str, state: S, next_node: str, trace: list[str], status: str) -> None:
        assert self.checkpointer is not None
        self.checkpointer.save(
            Checkpoint(
                thread_id=thread_id,
                state=state.model_dump(mode="json"),
                next_node=None if next_node == END else next_node,
                trace=list(trace),
                status=status,  # type: ignore[arg-type]
            )
        )
