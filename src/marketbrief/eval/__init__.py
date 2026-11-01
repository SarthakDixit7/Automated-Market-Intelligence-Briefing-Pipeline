"""Evals: golden-set recall, LLM-as-a-judge and a prompt-versioned runner."""

from marketbrief.eval.golden import claim_matches, event_matches, load_golden, recall_report

__all__ = [
    "claim_matches",
    "event_matches",
    "load_golden",
    "recall_report",
    "EvalResult",
    "evaluate_brief",
    "run_eval",
    "judge_brief",
    "judge_mean",
]


def __getattr__(name: str):
    # Judge/runner import the LLM stack; load them only when asked for.
    if name in {"EvalResult", "evaluate_brief", "run_eval"}:
        from marketbrief.eval import runner

        return getattr(runner, name)
    if name in {"judge_brief", "judge_mean"}:
        from marketbrief.eval import judge

        return getattr(judge, name)
    raise AttributeError(f"module 'marketbrief.eval' has no attribute {name!r}")
