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
        log = list(agent.log)
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

    def test_after_turn_sees_full_turn_messages(self, make_agent):
        seen = {}

        def after_turn(messages):
            seen.setdefault("turns", []).append([type(m).__name__ for m in messages])
            return messages

        agent = make_agent(
            [
                [content("tool"), done(tool_calls=_bash_tool_call("c1"))],
                [content("final"), done()],
            ],
            hooks={"after_turn": after_turn},
        )
        agent.register_tool(_echo_hi)
        agent.invoke("go")
        # 有工具回合收到 [AIMessage, ToolMessage]，收尾无工具回合只收到 AIMessage
        assert ["AIMessage", "ToolMessage"] in seen["turns"]
        assert ["AIMessage"] in seen["turns"]

    def test_after_turn_can_edit_tool_message(self, make_agent):
        from langchain_core.messages import ToolMessage

        seen = {}

        def after_turn(messages):
            out: list[Any] = []
            for m in messages:
                if isinstance(m, ToolMessage):
                    m = ToolMessage(content="redacted", tool_call_id=m.tool_call_id)
                out.append(m)
            return out

        def before_llm(request):
            seen["saw_redacted"] = any(
                isinstance(m, ToolMessage) and m.content == "redacted" for m in request.messages
            )
            return request

        agent = make_agent(
            [
                [content("tool"), done(tool_calls=_bash_tool_call("c1"))],
                [content("final"), done()],
            ],
            hooks={"after_turn": after_turn, "before_llm": before_llm},
        )
        agent.register_tool(_echo_hi)
        agent.invoke("go")
        assert seen["saw_redacted"] is True
