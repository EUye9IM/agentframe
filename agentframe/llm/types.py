from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol

from langchain_core.messages import AIMessage, BaseMessage


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class LLMRequest:
    """Request body entering LLMClient / before_llm hook.

    The model is owned by the client (endpoint), not the request; a request only
    carries the payload that can vary per call.
    """

    messages: list[BaseMessage]
    tools: list[dict[str, Any]] | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """Response body produced by LLMClient / after_llm hook."""

    message: AIMessage
    usage: Usage | None = None
    reasoning: str = ""
    finish_reason: str | None = None
    model: str | None = None
    raw: dict[str, Any] | None = None


@dataclass
class LLMStreamEvent:
    """Streaming event yielded by LLMClient.stream()."""

    type: str  # "reasoning" | "content" | "done"
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: Usage | None = None


class LLMClientProtocol(Protocol):
    """Structural interface for LLM clients.

    A client is an endpoint bound to a `model`; `_act_llm` reads `model` back for
    the response and only consumes `stream`, so a fake client (e.g. tests'
    `ScriptedLLMClient`) needs just these two.
    """

    model: str

    def stream(self, request: LLMRequest) -> Iterator[LLMStreamEvent]: ...
