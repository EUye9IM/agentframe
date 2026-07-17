from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

from litellm import completion, acompletion
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
    api_key: str | None = None,
    **extra: Any,
) -> dict:
    kwargs: dict = {"model": model, "messages": openai_messages}
    kwargs.update(extra)
    if tools:
        kwargs["tools"] = tools
    if api_key:
        kwargs["api_key"] = api_key
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


class LLMClient:
    def __init__(self, model: str, api_key: str | None = None, **kwargs: Any):
        self.model = model
        self.api_key = api_key
        self.kwargs = kwargs

    def invoke(self, messages: list[BaseMessage], tools: list[dict] | None = None) -> dict:
        openai_messages = _convert_messages(messages)
        kwargs = _build_kwargs(self.model, openai_messages, tools, self.api_key, **self.kwargs)

        response = completion(**kwargs)

        choice = response.choices[0]
        msg = choice.message

        tool_calls = []
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append({
                    "name": tc.function.name,
                    "args": json.loads(tc.function.arguments),
                    "id": tc.id,
                    "type": "tool_call",
                })

        ai_msg = AIMessage(content=msg.content or "", tool_calls=tool_calls or [])

        usage = {}
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return {"message": ai_msg, "usage": usage}

    def stream(self, messages: list[BaseMessage], tools: list[dict] | None = None) -> Iterator[dict]:
        openai_messages = _convert_messages(messages)
        kwargs = _build_kwargs(self.model, openai_messages, tools, self.api_key, **self.kwargs)
        kwargs["stream"] = True

        response = completion(**kwargs)

        tool_call_acc: dict[int, dict] = {}
        for chunk in response:
            delta = chunk.choices[0].delta

            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                yield {"type": "reasoning", "content": delta.reasoning_content}

            if delta.content:
                yield {"type": "content", "content": delta.content}

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

        tool_calls = _finish_tool_calls(tool_call_acc)

        usage = {}
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        yield {"type": "done", "tool_calls": tool_calls, "usage": usage}

    async def astream(self, messages: list[BaseMessage], tools: list[dict] | None = None) -> AsyncIterator[dict]:
        openai_messages = _convert_messages(messages)
        kwargs = _build_kwargs(self.model, openai_messages, tools, self.api_key, **self.kwargs)
        kwargs["stream"] = True

        response = await acompletion(**kwargs)

        tool_call_acc: dict[int, dict] = {}
        async for chunk in response:
            delta = chunk.choices[0].delta

            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                yield {"type": "reasoning", "content": delta.reasoning_content}

            if delta.content:
                yield {"type": "content", "content": delta.content}

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

        tool_calls = _finish_tool_calls(tool_call_acc)

        usage = {}
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        yield {"type": "done", "tool_calls": tool_calls, "usage": usage}
