"""LangGraph-style analyst agent: a small state-machine engine plus tools."""

from marketbrief.agent.graph import (
    END,
    Checkpoint,
    CompiledGraph,
    GraphError,
    GraphRun,
    JsonCheckpointer,
    MemoryCheckpointer,
    StateGraph,
)

__all__ = [
    "END",
    "Checkpoint",
    "CompiledGraph",
    "GraphError",
    "GraphRun",
    "JsonCheckpointer",
    "MemoryCheckpointer",
    "StateGraph",
    "AnalystResult",
    "AnalystState",
    "run_analyst",
    "resume_analyst",
]


def __getattr__(name: str):
    # The analyst pulls in the LLM/summarise stack; load it only when asked for.
    if name in {"AnalystResult", "AnalystState", "run_analyst", "resume_analyst"}:
        from marketbrief.agent import analyst

        return getattr(analyst, name)
    raise AttributeError(f"module 'marketbrief.agent' has no attribute {name!r}")
