from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from agentframe import function_tool


def make_response(
    content: str = "mock response",
    tool_calls: list[dict] | None = None,
    total_tokens: int = 30,
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
) -> dict:
    return {
        "message": AIMessage(content=content, tool_calls=tool_calls or []),
        "usage": {
            "total_tokens": total_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    }


def make_tool_call(name: str, args: dict[str, Any] | None = None, id: str = "call_1") -> list[dict]:
    return [{"name": name, "args": args or {}, "id": id, "type": "tool_call"}]


@function_tool
def get_weather(city: str) -> str:
    return f"{city}: sunny, 22°C"


@function_tool
def add(a: int, b: int) -> int:
    return a + b
