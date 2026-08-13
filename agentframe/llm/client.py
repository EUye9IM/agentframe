from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.messages.utils import convert_to_openai_messages

from .types import LLMRequest, LLMResponse, LLMStreamEvent, Usage

_DONE = "[DONE]"


def _build_payload(request: LLMRequest, *, stream: bool) -> dict:
    payload: dict[str, Any] = {
        "model": request.model,
        "messages": convert_to_openai_messages(request.messages),
    }
    if request.tools:
        payload["tools"] = request.tools
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    if request.max_tokens is not None:
        payload["max_tokens"] = request.max_tokens
    payload.update(request.extra)
    payload["stream"] = stream
    return payload


def _finish_tool_calls(acc: dict[int, dict]) -> list[dict]:
    result = []
    for idx in sorted(acc):
        tc = acc[idx]
        result.append(
            {
                "id": tc["id"],
                "name": tc["name"],
                "arguments": json.loads(tc["arguments"]) if tc["arguments"] else {},
            }
        )
    return result


def _parse_chunk(data: dict, tool_call_acc: dict[int, dict]) -> list[LLMStreamEvent]:
    events: list[LLMStreamEvent] = []
    choices = data.get("choices") or []
    if not choices:
        return events
    delta = choices[0].get("delta") or {}
    if delta.get("reasoning_content"):
        events.append(LLMStreamEvent(type="reasoning", content=delta["reasoning_content"]))
    if delta.get("content"):
        events.append(LLMStreamEvent(type="content", content=delta["content"]))
    if delta.get("tool_calls"):
        for tc_delta in delta["tool_calls"]:
            idx = tc_delta.get("index", 0)
            acc = tool_call_acc.setdefault(idx, {"id": "", "name": "", "arguments": ""})
            if tc_delta.get("id"):
                acc["id"] = tc_delta["id"]
            fn = tc_delta.get("function") or {}
            if fn.get("name"):
                acc["name"] = fn["name"]
            if fn.get("arguments"):
                acc["arguments"] += fn["arguments"]
    return events


def _parse_completion_response(data: dict) -> LLMResponse:
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    tool_calls = []
    for tc in msg.get("tool_calls") or []:
        tool_calls.append(
            {
                "id": tc.get("id", ""),
                "name": tc["function"]["name"],
                "arguments": json.loads(tc["function"].get("arguments") or "{}"),
            }
        )
    ai = AIMessage(content=msg.get("content") or "", tool_calls=tool_calls or [])
    usage_raw = data.get("usage") or {}
    usage = None
    if usage_raw:
        usage = Usage(
            prompt_tokens=usage_raw.get("prompt_tokens", 0),
            completion_tokens=usage_raw.get("completion_tokens", 0),
            total_tokens=usage_raw.get("total_tokens", 0),
        )
    return LLMResponse(
        message=ai,
        usage=usage,
        finish_reason=choice.get("finish_reason"),
        model=data.get("model"),
        raw=data,
    )


class LLMClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 120.0,
        **defaults: Any,
    ) -> None:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._http = httpx.Client(base_url=base_url, headers=headers, timeout=timeout)
        self._defaults: dict[str, Any] = defaults

    def _post(self, request: LLMRequest, *, stream: bool) -> dict:
        body = _build_payload(request, stream=stream)
        for k, v in self._defaults.items():
            body.setdefault(k, v)
        return body

    def invoke(self, request: LLMRequest) -> LLMResponse:
        resp = self._http.post("/chat/completions", json=self._post(request, stream=False))
        resp.raise_for_status()
        return _parse_completion_response(resp.json())

    def stream(self, request: LLMRequest) -> Iterator[LLMStreamEvent]:
        body = self._post(request, stream=True)
        tool_call_acc: dict[int, dict] = {}
        with self._http.stream("POST", "/chat/completions", json=body) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == _DONE:
                    break
                yield from _parse_chunk(json.loads(data), tool_call_acc)
        yield LLMStreamEvent(type="done", tool_calls=_finish_tool_calls(tool_call_acc))
