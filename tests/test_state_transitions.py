from __future__ import annotations

from typing import Any

from langchain_core.messages import ToolMessage

from .conftest import content, done


def _bash_tool_call(tool_call_id: str = "call_1") -> list[dict[str, Any]]:
    return [
        {
            "id": tool_call_id,
            "name": "echo_hi",
            "arguments": {},
            "type": "tool_call",
        }
    ]


def _echo_hi() -> str:
    return "hi from tool"


class TestStateTransitions:
    def test_direct_to_end_no_tools(self, make_agent):
        agent = make_agent([[content("hi"), done()]])
        result = agent.invoke("hello")
        assert result == "hi"
        log = [e for e in agent.log if e != "on_state_changed"]
        assert log == [
            "before_trace",
            "before_turn",
            "before_llm",
            "on_llm_content",
            "on_reasoning_end",
            "on_content_end",
            "after_llm",
            "after_turn",
            "handle_next",
            "after_trace",
        ]
        assert agent.log.count("before_turn") == 1
        assert agent.log.count("after_turn") == 1

    def test_single_tool_loop(self, make_agent):
        agent = make_agent(
            [
                [content("tool"), done(tool_calls=_bash_tool_call())],
                [content("final"), done()],
            ]
        )
        agent.register_tool(_echo_hi)
        result = agent.invoke("do it")
        assert result == "final"
        assert agent.log.count("before_llm") == 2
        assert agent.log.count("before_tool_call") == 1
        assert agent.log.count("after_tool_result:echo_hi") == 1
        assert "before_tool_call" in agent.log
        second_request = agent.requests[1]
        assert any(isinstance(m, ToolMessage) for m in second_request.messages)

    def test_multiple_tool_rounds(self, make_agent):
        agent = make_agent(
            [
                [content("t1"), done(tool_calls=_bash_tool_call("c1"))],
                [content("t2"), done(tool_calls=_bash_tool_call("c2"))],
                [content("final"), done()],
            ]
        )
        agent.register_tool(_echo_hi)
        result = agent.invoke("go")
        assert result == "final"
        assert agent.log.count("before_tool_call") == 2
        assert agent.log.count("after_turn") == 3

    def test_tool_result_enters_history(self, make_agent):
        agent = make_agent(
            [
                [content("tool"), done(tool_calls=_bash_tool_call())],
                [content("done"), done()],
            ]
        )
        agent.register_tool(_echo_hi)
        agent.invoke("go")
        assert agent.log.count("after_tool_result:echo_hi") == 1
        assert agent.log.count("on_state_changed") >= 3
