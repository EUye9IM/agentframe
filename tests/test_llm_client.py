from __future__ import annotations

import json
from typing import Any

import httpx
from langchain_core.messages import HumanMessage

from agentframe import LLMClient
from agentframe.llm.types import LLMRequest


def _mock_transport(lines: list[str]) -> httpx.MockTransport:
    body = "".join(lines).encode()

    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body, request=request)

    return httpx.MockTransport(handle)


def _make_client(lines: list[str]) -> LLMClient:
    return LLMClient(base_url="http://test", model="m", transport=_mock_transport(lines))


def _sse_data(chunk: dict[str, Any]) -> str:
    return f"data: {json.dumps(chunk)}\n"


def _request() -> LLMRequest:
    return LLMRequest(messages=[HumanMessage(content="q")])


class TestInvoke:
    def test_parses_content_usage_finish(self):
        body = {
            "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "model": "m",
        }
        client = _make_client([json.dumps(body)])
        resp = client.invoke(_request())
        assert resp.message.content == "hi"
        assert resp.usage is not None
        assert (resp.usage.prompt_tokens, resp.usage.completion_tokens, resp.usage.total_tokens) == (10, 5, 15)
        assert resp.finish_reason == "stop"
        assert resp.model == "m"

    def test_parses_tool_calls(self):
        body = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {"name": "echo", "arguments": '{"x": 1}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
        client = _make_client([json.dumps(body)])
        resp = client.invoke(_request())
        assert resp.message.tool_calls[0]["name"] == "echo"
        assert resp.message.tool_calls[0]["args"] == {"x": 1}
        assert resp.finish_reason == "tool_calls"

    def test_truncated_tool_arguments_parse_safely(self):
        body = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {"name": "echo", "arguments": '{"x":'},
                            }
                        ],
                    }
                }
            ]
        }
        client = _make_client([json.dumps(body)])
        resp = client.invoke(_request())
        assert resp.message.tool_calls[0]["args"] == {}

class TestStream:
    def test_content_events_and_usage_from_final_chunk(self):
        client = _make_client(
            [
                _sse_data({"choices": [{"delta": {"content": "hel"}}]}),
                _sse_data({"choices": [{"delta": {"content": "lo"}}]}),
                _sse_data({"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}}),
                "data: [DONE]\n",
            ]
        )
        events = list(client.stream(_request()))
        assert [(e.type, e.content) for e in events if e.type != "done"] == [
            ("content", "hel"),
            ("content", "lo"),
        ]
        done = events[-1]
        assert done.type == "done"
        assert done.usage is not None
        assert (done.usage.prompt_tokens, done.usage.completion_tokens, done.usage.total_tokens) == (3, 4, 7)

    def test_finish_reason_captured_from_content_chunk(self):
        client = _make_client(
            [
                _sse_data({"choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}]}),
                "data: [DONE]\n",
            ]
        )
        events = list(client.stream(_request()))
        assert events[-1].finish_reason == "stop"

    def test_reasoning_accepts_alternative_keys(self):
        for key in ("reasoning_content", "reasoning", "reasoning_text"):
            client = _make_client(
                [
                    _sse_data({"choices": [{"delta": {key: "think"}}]}),
                    _sse_data({"choices": [{"delta": {"content": "answer"}}]}),
                    "data: [DONE]\n",
                ]
            )
            events = list(client.stream(_request()))
            types = [(e.type, e.content) for e in events]
            assert ("reasoning", "think") in types
            assert ("content", "answer") in types

    def test_aggregates_tool_call_deltas(self):
        client = _make_client(
            [
                _sse_data(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {"index": 0, "id": "c1", "function": {"name": "echo", "arguments": ""}}
                                    ]
                                }
                            }
                        ]
                    }
                ),
                _sse_data(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {"index": 0, "function": {"arguments": '{"x": '}}
                                    ]
                                }
                            }
                        ]
                    }
                ),
                _sse_data(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {"index": 0, "function": {"arguments": "1}"}}
                                    ]
                                }
                            }
                        ]
                    }
                ),
                "data: [DONE]\n",
            ]
        )
        events = list(client.stream(_request()))
        done = events[-1]
        assert done.tool_calls == [{"id": "c1", "name": "echo", "arguments": {"x": 1}}]

    def test_truncated_stream_tool_arguments_parse_safely(self):
        client = _make_client(
            [
                _sse_data(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {"index": 0, "id": "c1", "function": {"name": "echo", "arguments": '{"x":'}}
                                    ]
                                }
                            }
                        ]
                    }
                ),
                "data: [DONE]\n",
            ]
        )
        events = list(client.stream(_request()))
        assert events[-1].tool_calls == [{"id": "c1", "name": "echo", "arguments": {}}]


class TestLifecycle:
    def test_close_is_idempotent(self):
        client = _make_client(["data: [DONE]\n"])
        client.close()
        client.close()

    def test_context_manager_closes(self):
        with _make_client(["data: [DONE]\n"]) as client:
            assert client is not None
