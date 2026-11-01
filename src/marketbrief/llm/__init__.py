"""LLM layer: provider-neutral requests, Claude + offline providers, re-ask loop."""

from marketbrief.llm.anthropic_client import AnthropicLLMClient
from marketbrief.llm.base import (
    LLMClient,
    LLMError,
    LLMRequest,
    LLMResult,
    RefusalError,
    StructuredOutputError,
    TruncatedOutputError,
    Usage,
)
from marketbrief.llm.factory import UsageTracker, build_llm
from marketbrief.llm.offline import OfflineLLMClient
from marketbrief.llm.structured import structured_call

__all__ = [
    "AnthropicLLMClient",
    "LLMClient",
    "LLMError",
    "LLMRequest",
    "LLMResult",
    "OfflineLLMClient",
    "RefusalError",
    "StructuredOutputError",
    "TruncatedOutputError",
    "Usage",
    "UsageTracker",
    "build_llm",
    "structured_call",
]
