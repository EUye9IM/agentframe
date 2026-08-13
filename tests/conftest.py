from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from agentframe import BaseAgent
from agentframe.llm.types import LLMRequest, LLMStreamEvent, Usage


def content(text: str) -> LLMStreamEvent:
    return LLMStreamEvent(type="content", content=text)


def reasoning(text: str) -> LLMStreamEvent:
    return LLMStreamEvent(type="reasoning", content=text)


def done(tool_calls: list[dict[str, Any]] | None = None, usage: Usage | None = None) -> LLMStreamEvent:
    return LLMStreamEvent(type="done", tool_calls=tool_calls or [], usage=usage)


class ScriptedLLMClient:
    def __init__(
        self,
        scripts: list[list[LLMStreamEvent]],
        *,
        model: str = "test-model",
        raise_at: int | None = None,
        exc: BaseException | None = None,
    ) -> None:
        self.model = model
        self.scripts = list(scripts)
        self.requests: list[LLMRequest] = []
        self.raise_at = raise_at
        self.exc = exc

    def stream(self, request: LLMRequest) -> Iterator[LLMStreamEvent]:
        self.requests.append(request)
        if self.raise_at is not None and len(self.requests) == self.raise_at:
            assert self.exc is not None
            raise self.exc
        for event in self.scripts.pop(0):
            yield event

    def invoke(self, request: LLMRequest) -> Any:
        raise NotImplementedError


class RecordingAgent(BaseAgent):
    def __init__(
        self,
        *,
        scripts,
        hooks=None,
        model="test-model",
        raise_at: int | None = None,
        exc: BaseException | None = None,
        **kw,
    ) -> None:
        super().__init__(
            llm_client=ScriptedLLMClient(
                scripts, model=model, raise_at=raise_at, exc=exc
            ),
            **kw,
        )
        self.log: list[str] = []
        if hooks:
            for name, fn in hooks.items():
                setattr(self, name, fn)

    def before_trace(self, input_text, session):
        self.log.append("before_trace")
        return super().before_trace(input_text, session)

    def after_trace(self, data, session):
        self.log.append("after_trace")
        return super().after_trace(data, session)

    def before_turn(self, data):
        self.log.append("before_turn")
        return super().before_turn(data)

    def after_turn(self, data):
        self.log.append("after_turn")
        return super().after_turn(data)

    def before_llm(self, request):
        self.log.append("before_llm")
        return super().before_llm(request)

    def on_llm_reasoning(self, text):
        self.log.append("on_llm_reasoning")
        super().on_llm_reasoning(text)

    def on_reasoning_end(self, reasoning):
        self.log.append("on_reasoning_end")
        super().on_reasoning_end(reasoning)

    def on_llm_content(self, text):
        self.log.append("on_llm_content")
        super().on_llm_content(text)

    def on_content_end(self, content):
        self.log.append("on_content_end")
        super().on_content_end(content)

    def after_llm(self, response):
        self.log.append("after_llm")
        return super().after_llm(response)

    def before_tool_call(self, tool_calls):
        self.log.append("before_tool_call")
        return super().before_tool_call(tool_calls)

    def after_tool_result(self, name, result):
        self.log.append(f"after_tool_result:{name}")
        return super().after_tool_result(name, result)

    def handle_next(self, from_node, default):
        self.log.append("handle_next")
        return super().handle_next(from_node, default)

    def handle_error(self, error, node):
        self.log.append("handle_error")
        return super().handle_error(error, node)

    def on_state_changed(self, messages):
        self.log.append("on_state_changed")
        super().on_state_changed(messages)


@pytest.fixture
def make_agent():
    def _make(scripts, *, hooks=None, raise_at=None, exc=None, **kw) -> RecordingAgent:
        agent = RecordingAgent(
            scripts=scripts, hooks=hooks, raise_at=raise_at, exc=exc, **kw
        )
        return agent

    return _make
