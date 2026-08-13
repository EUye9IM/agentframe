from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, cast, override

import pytest
from langchain_core.messages import BaseMessage, ToolCall
from langgraph.errors import NodeError
from langgraph.types import Command

from agentframe import BaseAgent
from agentframe.core.phases import Phase
from agentframe.core.state import AgentState
from agentframe.llm.types import LLMRequest, LLMResponse, LLMStreamEvent, Usage


def content(text: str) -> LLMStreamEvent:
    return LLMStreamEvent(type="content", content=text)


def reasoning(text: str) -> LLMStreamEvent:
    return LLMStreamEvent(type="reasoning", content=text)


def done(
    tool_calls: list[dict[str, Any]] | None = None,
    usage: Usage | None = None,
    finish_reason: str | None = None,
) -> LLMStreamEvent:
    return LLMStreamEvent(type="done", tool_calls=tool_calls or [], usage=usage, finish_reason=finish_reason)


class ScriptedLLMClient:
    def __init__(
        self,
        scripts: list[list[LLMStreamEvent]],
        *,
        model: str = "test-model",
        raise_at: int | None = None,
        exc: BaseException | None = None,
    ) -> None:
        self.model: str = model
        self.scripts: list[list[LLMStreamEvent]] = list(scripts)
        self.requests: list[LLMRequest] = []
        self.raise_at: int | None = raise_at
        self.exc: BaseException | None = exc

    def stream(self, request: LLMRequest) -> Iterator[LLMStreamEvent]:
        self.requests.append(request)
        if self.raise_at is not None and len(self.requests) == self.raise_at:
            assert self.exc is not None
            raise self.exc
        for event in self.scripts.pop(0):
            yield event


class RecordingAgent(BaseAgent):
    def __init__(
        self,
        *,
        scripts: list[list[LLMStreamEvent]],
        hooks: dict[str, Callable[..., Any]] | None = None,
        model: str = "test-model",
        raise_at: int | None = None,
        exc: BaseException | None = None,
        system_prompt: str | None = None,
        compile_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            llm_client=ScriptedLLMClient(scripts, model=model, raise_at=raise_at, exc=exc),
            system_prompt=system_prompt,
            compile_kwargs=compile_kwargs,
        )
        self.log: list[str] = []
        if hooks:
            for name, fn in hooks.items():
                setattr(self, name, fn)

    @property
    def requests(self) -> list[LLMRequest]:
        return cast(ScriptedLLMClient, self._llm_client).requests

    @override
    def before_trace(self, input_text: str, session_id: str | None) -> str:
        self.log.append("before_trace")
        return super().before_trace(input_text, session_id)

    @override
    def after_trace(self, data: AgentState, session_id: str | None) -> str:
        self.log.append("after_trace")
        return super().after_trace(data, session_id)

    @override
    def before_turn(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        self.log.append("before_turn")
        return super().before_turn(messages)

    @override
    def after_turn(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        self.log.append("after_turn")
        return super().after_turn(messages)

    @override
    def before_llm(self, request: LLMRequest) -> LLMRequest:
        self.log.append("before_llm")
        return super().before_llm(request)

    @override
    def on_llm_reasoning(self, text: str) -> None:
        self.log.append("on_llm_reasoning")
        super().on_llm_reasoning(text)

    @override
    def on_reasoning_end(self, reasoning: str) -> None:
        self.log.append("on_reasoning_end")
        super().on_reasoning_end(reasoning)

    @override
    def on_llm_content(self, text: str) -> None:
        self.log.append("on_llm_content")
        super().on_llm_content(text)

    @override
    def on_content_end(self, content: str) -> None:
        self.log.append("on_content_end")
        super().on_content_end(content)

    @override
    def after_llm(self, response: LLMResponse) -> list[BaseMessage]:
        self.log.append("after_llm")
        return super().after_llm(response)

    @override
    def before_tool_call(self, tool_calls: list[ToolCall]) -> list[ToolCall]:
        self.log.append("before_tool_call")
        return super().before_tool_call(tool_calls)

    @override
    def after_tool_result(self, name: str, result: str, tool_call_id: str) -> list[BaseMessage]:
        self.log.append(f"after_tool_result:{name}")
        return super().after_tool_result(name, result, tool_call_id)

    @override
    def handle_next(self, from_node: Phase, default: Phase) -> Phase:
        self.log.append("handle_next")
        return super().handle_next(from_node, default)

    @override
    def handle_error(self, error: NodeError, node: str) -> Command[Phase]:
        self.log.append("handle_error")
        return super().handle_error(error, node)


@pytest.fixture
def make_agent() -> Callable[..., RecordingAgent]:
    def _make(
        scripts: list[list[LLMStreamEvent]],
        *,
        hooks: dict[str, Callable[..., Any]] | None = None,
        raise_at: int | None = None,
        exc: BaseException | None = None,
        system_prompt: str | None = None,
        compile_kwargs: dict[str, Any] | None = None,
    ) -> RecordingAgent:
        return RecordingAgent(
            scripts=scripts,
            hooks=hooks,
            raise_at=raise_at,
            exc=exc,
            system_prompt=system_prompt,
            compile_kwargs=compile_kwargs,
        )

    return _make
