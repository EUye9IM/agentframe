from __future__ import annotations

from typing import override

from langchain_core.messages import AIMessage

from agentframe import Agent

from .conftest import ScriptedLLMClient, content, done, reasoning


class TestHookChain:
    def test_single_llm_hook_order(self, make_agent):
        agent = make_agent([[reasoning("think1"), content("hi"), done()]])
        agent.invoke("hello")
        order = [e for e in agent.log if e != "on_state_changed"]
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

        def on_state_changed(messages):
            seen["messages"] = list(messages)

        agent = make_agent(
            [[reasoning("secret"), content("answer"), done()]],
            hooks={"after_llm": hook, "on_state_changed": on_state_changed},
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

    def test_on_state_changed_fires_once_per_llm(self, make_agent):
        agent = make_agent([[content("hi"), done()]])
        agent.invoke("hello")
        assert agent.log.count("on_state_changed") == 1


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
