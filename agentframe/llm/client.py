from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

import openai
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)


def _convert_messages(messages: list[BaseMessage]) -> list[dict]:
    openai_messages = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            openai_messages.append({"role": "system", "content": msg.content})
        elif isinstance(msg, HumanMessage):
            openai_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            d: dict = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                d["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["args"]),
                        },
                    }
                    for tc in msg.tool_calls
                ]
            openai_messages.append(d)
        elif isinstance(msg, ToolMessage):
            openai_messages.append({
                "role": "tool",
                "content": msg.content,
                "tool_call_id": msg.tool_call_id,
            })
    return openai_messages


def _build_kwargs(
    model: str,
    openai_messages: list[dict],
    tools: list[dict] | None = None,
    **extra: Any,
) -> dict:
    kwargs: dict = {"model": model, "messages": openai_messages}
    kwargs.update(extra)
    if tools:
        kwargs["tools"] = tools
    return kwargs


def _finish_tool_calls(tool_call_acc: dict[int, dict]) -> list[dict]:
    tool_calls = []
    for tc in tool_call_acc.values():
        tool_calls.append({
            "name": tc["name"],
            "args": json.loads(tc["args"]) if tc["args"] else {},
            "id": tc["id"],
            "type": "tool_call",
        })
    return tool_calls


def _parse_tool_calls_from_message(msg) -> list[dict]:
    tool_calls = []
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        for tc in msg.tool_calls:
            tool_calls.append({
                "name": tc.function.name,
                "args": json.loads(tc.function.arguments),
                "id": tc.id,
                "type": "tool_call",
            })
    return tool_calls


def _parse_completion_response(response) -> dict:
    msg = response.choices[0].message
    tool_calls = _parse_tool_calls_from_message(msg)
    ai_msg = AIMessage(content=msg.content or "", tool_calls=tool_calls or [])
    usage = {}
    if hasattr(response, "usage") and response.usage:
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
    return {"message": ai_msg, "usage": usage}


def _process_stream_chunk(chunk, tool_call_acc: dict[int, dict]) -> list[dict]:
    events: list[dict] = []
    delta = chunk.choices[0].delta

    if hasattr(delta, "reasoning_content") and delta.reasoning_content:
        events.append({"type": "reasoning", "content": delta.reasoning_content})
    if delta.content:
        events.append({"type": "content", "content": delta.content})
    if hasattr(delta, "tool_calls") and delta.tool_calls:
        for tc_delta in delta.tool_calls:
            idx = tc_delta.index
            if idx not in tool_call_acc:
                tool_call_acc[idx] = {"id": "", "name": "", "args": ""}
            if tc_delta.id:
                tool_call_acc[idx]["id"] = tc_delta.id
            if tc_delta.function:
                if tc_delta.function.name:
                    tool_call_acc[idx]["name"] = tc_delta.function.name
                if tc_delta.function.arguments:
                    tool_call_acc[idx]["args"] += tc_delta.function.arguments
    return events


def _extract_stream_usage(response) -> dict:
    if hasattr(response, "usage") and response.usage:
        return {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
    return {}


class LLMClient:
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.kwargs = kwargs
        self._client: openai.OpenAI | None = None
        self._aclient: openai.AsyncOpenAI | None = None

    def _get_client(self) -> openai.OpenAI:
        if self._client is None:
            self._client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def _get_aclient(self) -> openai.AsyncOpenAI:
        if self._aclient is None:
            self._aclient = openai.AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._aclient

    def invoke(self, messages: list[BaseMessage], tools: list[dict] | None = None) -> dict:
        openai_messages = _convert_messages(messages)
        kwargs = _build_kwargs(self.model, openai_messages, tools, **self.kwargs)
        response = self._get_client().chat.completions.create(**kwargs)
        return _parse_completion_response(response)

    async def ainvoke(self, messages: list[BaseMessage], tools: list[dict] | None = None) -> dict:
        openai_messages = _convert_messages(messages)
        kwargs = _build_kwargs(self.model, openai_messages, tools, **self.kwargs)
        response = await self._get_aclient().chat.completions.create(**kwargs)
        return _parse_completion_response(response)

    def stream(self, messages: list[BaseMessage], tools: list[dict] | None = None) -> Iterator[dict]:
        openai_messages = _convert_messages(messages)
        kwargs = _build_kwargs(self.model, openai_messages, tools, **self.kwargs)
        kwargs["stream"] = True
        response = self._get_client().chat.completions.create(**kwargs)

        tool_call_acc: dict[int, dict] = {}
        for chunk in response:
            yield from _process_stream_chunk(chunk, tool_call_acc)

        yield {"type": "done", "tool_calls": _finish_tool_calls(tool_call_acc),
               "usage": _extract_stream_usage(response)}

    async def astream(self, messages: list[BaseMessage], tools: list[dict] | None = None) -> AsyncIterator[dict]:
        openai_messages = _convert_messages(messages)
        kwargs = _build_kwargs(self.model, openai_messages, tools, **self.kwargs)
        kwargs["stream"] = True
        response = await self._get_aclient().chat.completions.create(**kwargs)

        tool_call_acc: dict[int, dict] = {}
        async for chunk in response:
            for event in _process_stream_chunk(chunk, tool_call_acc):
                yield event

        yield {"type": "done", "tool_calls": _finish_tool_calls(tool_call_acc),
               "usage": _extract_stream_usage(response)}
