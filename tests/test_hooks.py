from __future__ import annotations

from typing import override

from langchain_core.messages import AIMessage, ToolMessage

from agentframe import Agent

from .conftest import ScriptedLLMClient, content, done, reasoning


class TestHookChain:
    def test_single_llm_hook_order(self, make_agent):
        agent = make_agent([[reasoning("think1"), content("hi"), done()]])
        agent.invoke("hello")
        order = list(agent.log)
        llm_span = order[order.index("before_llm") : order.index("after_llm") + 1]
        assert llm_span == [
            "before_llm",
            "on_llm_reasoning",
            "on_llm_content",
            "on_reasoning_end",
            "on_content_end",
            "after_llm",
        ]

    def test_middleware_switches_client_model(self):
        from agentframe import Middleware

        default = ScriptedLLMClient([[content("hi"), done()]], model="default-model")
        other = ScriptedLLMClient([[content("hi"), done()]], model="changed-model")

        def make_switcher(other_client):
            class Switcher(Middleware):
                @override
                def before_llm(self, request):
                    request = super().before_llm(request)
                    setattr(self, "llm_client", other_client)
                    return request

            return Switcher

        agent = Agent(llm_client=default, middlewares=[make_switcher(other)()])
        agent.invoke("hello")
        assert agent.llm_client.model == "changed-model"
        assert len(other.requests) == 1
        assert len(default.requests) == 0

    def test_after_llm_prepends_reasoning(self, make_agent):
        seen = {}

        def hook(response):
            return [AIMessage(content=f"[thinking] {response.reasoning}"), response.message]

        def after_turn(messages):
            seen["messages"] = list(messages)
            return messages

        agent = make_agent(
            [[reasoning("secret"), content("answer"), done()]],
            hooks={"after_llm": hook, "after_turn": after_turn},
        )
        result = agent.invoke("q")
        assert result == "answer"
        contents = [m.content for m in seen["messages"]]
        assert "[thinking] secret" in contents
        assert "answer" in contents
        assert contents[-1] == "answer"

    def test_on_end_events_get_full_text(self, make_agent):
        seen = {}

        def on_reasoning_end(text):
            seen["reasoning"] = text

        def on_content_end(text):
            seen["content"] = text

        agent = make_agent(
            [[reasoning("r1"), reasoning("r2"), content("a"), content("b"), done()]],
            hooks={"on_reasoning_end": on_reasoning_end, "on_content_end": on_content_end},
        )
        agent.invoke("q")
        assert seen["reasoning"] == "r1r2"
        assert seen["content"] == "ab"

    def test_after_tool_result_receives_tool_call_id(self, make_agent):
        seen = {}

        def echo_a() -> str:
            return "a"

        def echo_b() -> str:
            return "b"

        def after_tool_result(name, result, tool_call_id):
            seen.setdefault("calls", []).append((name, result, tool_call_id))
            return [ToolMessage(content=result, tool_call_id=tool_call_id)]

        agent = make_agent(
            [
                [
                    content("t"),
                    done(
                        tool_calls=[
                            {"id": "c1", "name": "echo_a", "arguments": {}},
                            {"id": "c2", "name": "echo_b", "arguments": {}},
                        ]
                    ),
                ],
                [content("final"), done()],
            ],
            hooks={"after_tool_result": after_tool_result},
        )
        agent.register_tool(echo_a)
        agent.register_tool(echo_b)
        agent.invoke("go")
        assert seen["calls"] == [("echo_a", "a", "c1"), ("echo_b", "b", "c2")]

    def test_after_turn_return_value_writes_back(self, make_agent):
        from langchain_core.messages import AIMessage

        def after_turn(messages):
            for m in messages:
                if isinstance(m, AIMessage):
                    m.content = "censored"
            return messages

        agent = make_agent(
            [[content("secret"), done()]],
            hooks={"after_turn": after_turn},
        )
        result = agent.invoke("go")
        assert result == "censored"

    def test_after_llm_receives_streamed_usage_and_finish_reason(self, make_agent):
        from agentframe.llm.types import Usage

        seen = {}

        def after_llm(response):
            seen["usage"] = response.usage
            seen["finish_reason"] = response.finish_reason
            return [response.message]

        agent = make_agent(
            [
                [
                    content("hi"),
                    done(
                        usage=Usage(prompt_tokens=3, completion_tokens=4, total_tokens=7),
                        finish_reason="stop",
                    ),
                ]
            ],
            hooks={"after_llm": after_llm},
        )
        agent.invoke("go")
        assert seen["usage"] == Usage(prompt_tokens=3, completion_tokens=4, total_tokens=7)
        assert seen["finish_reason"] == "stop"


class TestDynamicInheritance:
    def test_mro_order_follows_list_order(self):
        from agentframe import Middleware
        from agentframe.llm.types import LLMRequest

        order: list[str] = []

        def make_middleware(tag: str):
            class M(Middleware):
                @override
                def before_llm(self, request):
                    request = super().before_llm(request)
                    order.append(tag)
                    return request

            return M

        m1, m2 = make_middleware("m1"), make_middleware("m2")
        agent = Agent(llm_client=ScriptedLLMClient([]), middlewares=[m1(), m2()])
        agent.before_llm(LLMRequest(messages=[]))
        assert order == ["m1", "m2"]

    def test_duplicate_middleware_classes_no_conflict(self):
        from agentframe import Middleware
        from agentframe.llm.types import LLMRequest

        def tools(fn):
            class ToolsMiddleware(Middleware):
                @override
                def before_llm(self, request):
                    request = super().before_llm(request)
                    request.tools = (request.tools or []) + [fn]
                    return request

            return ToolsMiddleware

        agent = Agent(llm_client=ScriptedLLMClient([]), middlewares=[tools("t1")(), tools("t2")()])
        req = agent.before_llm(LLMRequest(messages=[]))
        assert req.tools == ["t1", "t2"]
