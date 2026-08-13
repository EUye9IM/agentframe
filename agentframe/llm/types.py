from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class LLMRequest:
    """Request body entering LLMClient / before_llm hook."""

    model: str
    messages: list[BaseMessage]
    tools: list[dict] | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class LLMResponse:
    """Response body produced by LLMClient / after_llm hook."""

    message: AIMessage
    usage: Usage | None = None
    reasoning: str = ""
    finish_reason: str | None = None
    model: str | None = None
    raw: dict | None = None


@dataclass
class LLMStreamEvent:
    """Streaming event yielded by LLMClient.stream()."""

    type: str  # "reasoning" | "content" | "done"
    content: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    usage: Usage | None = None
