from __future__ import annotations

import json
from collections.abc import Iterator
from types import TracebackType
from typing import Any, cast

import httpx
from langchain_core.messages import AIMessage
from langchain_core.messages.utils import convert_to_openai_messages

from .types import LLMRequest, LLMResponse, LLMStreamEvent, Usage

_DONE = "[DONE]"


def _build_payload(request: LLMRequest, *, stream: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
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


def _parse_usage(data: dict[str, Any]) -> Usage | None:
    usage_raw = cast(dict[str, Any], data.get("usage") or {})
    if not usage_raw:
        return None
    return Usage(
        prompt_tokens=cast(int, usage_raw.get("prompt_tokens", 0)),
        completion_tokens=cast(int, usage_raw.get("completion_tokens", 0)),
        total_tokens=cast(int, usage_raw.get("total_tokens", 0)),
    )


def _safe_parse_arguments(raw: str) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        return cast(dict[str, Any], json.loads(raw))
    except ValueError:
        return {}


def _finish_tool_calls(acc: dict[int, dict[str, str]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for idx in sorted(acc):
        tc = acc[idx]
        result.append(
            {
                "id": tc["id"],
                "name": tc["name"],
                "arguments": _safe_parse_arguments(tc["arguments"]),
            }
        )
    return result


def _parse_chunk(data: dict[str, Any], tool_call_acc: dict[int, dict[str, str]]) -> list[LLMStreamEvent]:
    events: list[LLMStreamEvent] = []
    choices = cast(list[dict[str, Any]], data.get("choices") or [])
    if not choices:
        return events
    delta = cast(dict[str, Any], choices[0].get("delta") or {})
    for key in ("reasoning_content", "reasoning", "reasoning_text"):
        if delta.get(key):
            events.append(LLMStreamEvent(type="reasoning", content=cast(str, delta[key])))
            break
    if delta.get("content"):
        events.append(LLMStreamEvent(type="content", content=cast(str, delta["content"])))
    if delta.get("tool_calls"):
        for tc_delta in cast(list[dict[str, Any]], delta["tool_calls"]):
            acc = tool_call_acc.setdefault(
                cast(int, tc_delta.get("index", 0)), {"id": "", "name": "", "arguments": ""}
            )
            if tc_delta.get("id"):
                acc["id"] = cast(str, tc_delta["id"])
            fn = cast(dict[str, Any], tc_delta.get("function") or {})
            if fn.get("name"):
                acc["name"] = cast(str, fn["name"])
            if fn.get("arguments"):
                acc["arguments"] += cast(str, fn["arguments"])
    return events


def _parse_completion_response(data: dict[str, Any]) -> LLMResponse:
    choice = cast(dict[str, Any], (data.get("choices") or [{}])[0])
    msg = cast(dict[str, Any], choice.get("message") or {})
    tool_calls: list[dict[str, Any]] = []
    for tc in cast(list[dict[str, Any]], msg.get("tool_calls") or []):
        fn = cast(dict[str, Any], tc["function"])
        tool_calls.append(
            {
                "id": tc.get("id", ""),
                "name": fn["name"],
                "args": _safe_parse_arguments(cast(str, fn.get("arguments") or "{}")),
                "type": "tool_call",
            }
        )
    ai = AIMessage(content=cast(str, msg.get("content") or ""), tool_calls=tool_calls or [])
    usage = _parse_usage(data)
    return LLMResponse(
        message=ai,
        usage=usage,
        finish_reason=cast(str | None, choice.get("finish_reason")),
        model=cast(str | None, data.get("model")),
        raw=data,
    )


class LLMClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
        thinking: bool = True,
        **defaults: object,
    ) -> None:
        self.model: str = model
        self.thinking: bool = thinking
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if transport is not None:
            self._http: httpx.Client = httpx.Client(
                base_url=base_url, headers=headers, timeout=timeout, transport=transport
            )
        else:
            self._http = httpx.Client(base_url=base_url, headers=headers, timeout=timeout)
        self._defaults: dict[str, object] = defaults

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> LLMClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _post(self, request: LLMRequest, *, stream: bool) -> dict[str, Any]:
        body = _build_payload(request, stream=stream)
        for k, v in self._defaults.items():
            body.setdefault(k, v)
        body["model"] = self.model
        if not self.thinking:
            body["thinking"] = {"enabled": False}
        return body

    def invoke(self, request: LLMRequest) -> LLMResponse:
        resp = self._http.post("/chat/completions", json=self._post(request, stream=False))
        resp.raise_for_status()
        return _parse_completion_response(cast(dict[str, Any], resp.json()))

    def stream(self, request: LLMRequest) -> Iterator[LLMStreamEvent]:
        body = self._post(request, stream=True)
        tool_call_acc: dict[int, dict[str, str]] = {}
        usage: Usage | None = None
        finish_reason: str | None = None
        with self._http.stream("POST", "/chat/completions", json=body) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == _DONE:
                    break
                chunk = cast(dict[str, Any], json.loads(data))
                usage = _parse_usage(chunk) or usage
                for choice in cast(list[dict[str, Any]], chunk.get("choices") or []):
                    fr = choice.get("finish_reason")
                    if fr:
                        finish_reason = cast(str, fr)
                yield from _parse_chunk(chunk, tool_call_acc)
        yield LLMStreamEvent(
            type="done",
            tool_calls=_finish_tool_calls(tool_call_acc),
            usage=usage,
            finish_reason=finish_reason,
        )
