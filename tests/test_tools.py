from __future__ import annotations

from typing import Any, cast

from langchain_core.messages import ToolMessage

from agentframe import Agent
from agentframe.middlewares import tools

from .conftest import ScriptedLLMClient, content, done


def get_weather(city: str, days: int = 3) -> str:
    """查询城市天气。

    Args:
        city: 城市名
        days: 天数
    """
    return f"{city} 晴天"


def add(a: int, b: int) -> int:
    """两数相加。"""
    return a + b


def no_args() -> str:
    return "x"


class TestToolsMiddleware:
    def test_injects_tool_schema(self):
        client = ScriptedLLMClient([[content("hi"), done()]])
        agent = Agent(llm_client=client, middlewares=[tools([get_weather])()])
        agent.invoke("北京天气")
        req = client.requests[0]
        assert req.tools is not None
        assert len(req.tools) == 1
        spec = req.tools[0]
        assert spec["type"] == "function"
        fn = spec["function"]
        assert fn["name"] == "get_weather"
        assert fn["description"] == "查询城市天气。"
        params = fn["parameters"]
        assert params["type"] == "object"
        assert set(params["required"]) == {"city"}
        assert set(params["properties"]) == {"city", "days"}

    def test_registers_and_dispatches_tool(self):
        client = ScriptedLLMClient(
            [
                [
                    content("t"),
                    done(tool_calls=[{"id": "c1", "name": "get_weather", "arguments": {"city": "北京"}}]),
                ],
                [content("final"), done()],
            ]
        )
        agent = Agent(llm_client=client, middlewares=[tools([get_weather])()])
        result = agent.invoke("北京天气")
        assert result == "final"
        # 工具结果以 ToolMessage 进历史,第二轮请求带上它
        req2 = client.requests[1]
        assert any(
            isinstance(m, ToolMessage) and "北京 晴天" in str(m.content) for m in req2.messages
        )

    def test_register_adds_tool(self):
        client = ScriptedLLMClient([[content("hi"), done()]])
        agent = Agent(llm_client=client, middlewares=[tools([get_weather])()])
        cast(Any, agent).register(add)
        agent.invoke("hi")
        names = [t["function"]["name"] for t in client.requests[0].tools or []]
        assert names == ["get_weather", "add"]

    def test_unregister_removes_tool(self):
        client = ScriptedLLMClient([[content("hi"), done()], [content("hi2"), done()]])
        agent = Agent(llm_client=client, middlewares=[tools([get_weather, add])()])
        agent.invoke("hi")
        assert "add" in cast(Any, agent)._tools
        cast(Any, agent).unregister("add")
        agent.invoke("hi2")
        names = [t["function"]["name"] for t in client.requests[1].tools or []]
        assert names == ["get_weather"]
        assert "add" not in cast(Any, agent)._tools

    def test_no_docstring_no_args(self):
        client = ScriptedLLMClient([[content("hi"), done()]])
        agent = Agent(llm_client=client, middlewares=[tools([no_args])()])
        agent.invoke("hi")
        fn = (client.requests[0].tools or [])[0]["function"]
        assert fn["name"] == "no_args"
        assert fn["description"] == ""
        assert fn["parameters"]["type"] == "object"
        assert fn["parameters"]["properties"] == {}
