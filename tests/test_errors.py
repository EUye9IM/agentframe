from __future__ import annotations

from agentframe import Phase, StreamStop

from .conftest import content, done, reasoning


class TestErrors:
    def test_exception_defaults_to_end(self, make_agent):
        agent = make_agent(
            [[content("hi"), done()]],
            raise_at=1,
            exc=RuntimeError("boom"),
        )
        result = agent.invoke("go")
        assert "handle_error" in agent.log
        assert result is not None

    def test_handle_error_override_retries_once(self, make_agent):
        from langgraph.types import Command

        attempts = {"n": 0}

        def hook(error, node):
            attempts["n"] += 1
            if attempts["n"] == 1:
                return Command(goto=Phase.LLM)
            return Command(goto=Phase.END)

        agent = make_agent(
            [[content("hi"), done()]],
            raise_at=1,
            exc=RuntimeError("boom"),
            hooks={"handle_error": hook},
        )
        result = agent.invoke("go")
        assert result == "hi"
        assert attempts["n"] == 1

    def test_streamstop_with_override_keeps_partial(self, make_agent):
        from langchain_core.messages import AIMessage
        from langgraph.types import Command

        handled = {"n": 0}

        def on_content(text):
            if text == "halt":
                raise StreamStop(goto=Phase.END, message="interrupted")

        def handle_error(error, node):
            if isinstance(error.error, StreamStop):
                handled["n"] += 1
                assert error.error.partial == "partial-halt"  # incl. the halting chunk
                return Command(
                    update={"messages": [AIMessage(content=error.error.partial)]},
                    goto=error.error.goto,
                )
            return Command(goto=Phase.END)

        agent = make_agent(
            [[content("partial-"), content("halt"), content("rest"), done()]],
            hooks={"on_llm_content": on_content, "handle_error": handle_error},
        )
        result = agent.invoke("go")
        assert handled["n"] == 1
        assert result == "partial-halt"
        assert len(agent.requests) == 1

    def test_streamstop_no_override_defaults_end(self, make_agent):
        def on_content(text):
            raise StreamStop(goto=Phase.END)

        agent = make_agent(
            [[content("x"), done()]],
            hooks={"on_llm_content": on_content},
        )
        result = agent.invoke("go")
        assert result is not None
        assert agent.log.count("handle_error") == 1

    def test_keyboardinterrupt_converts_to_streamstop(self, make_agent):
        from langgraph.types import Command

        caught = {}

        def handle_error(error, node):
            caught["is_stream_stop"] = isinstance(error.error, StreamStop)
            return Command(goto=Phase.END)

        agent = make_agent(
            [[content("hi"), done()]],
            raise_at=1,
            exc=KeyboardInterrupt(),
            hooks={"handle_error": handle_error},
        )
        result = agent.invoke("go")
        assert caught["is_stream_stop"] is True
        assert result is not None

    def test_first_turn_failure_does_not_echo_user_input(self, make_agent):
        agent = make_agent(
            [[content("hi"), done()]],
            raise_at=1,
            exc=RuntimeError("boom"),
        )
        result = agent.invoke("用户输入")
        assert result == ""

    def test_streamstop_keeps_partial_reasoning(self, make_agent):
        from langgraph.types import Command

        handled = {}

        def on_reasoning(text):
            if text == "halt":
                raise StreamStop(goto=Phase.END, message="interrupted")

        def handle_error(error, node):
            if isinstance(error.error, StreamStop):
                handled["reasoning"] = error.error.partial_reasoning
            return Command(goto=Phase.END)

        agent = make_agent(
            [[reasoning("think-"), reasoning("halt"), content("rest"), done()]],
            hooks={"on_llm_reasoning": on_reasoning, "handle_error": handle_error},
        )
        agent.invoke("go")
        assert handled["reasoning"] == "think-halt"

    def test_tool_error_then_retry_keeps_correct_tool_call_id(self, make_agent):
        from langchain_core.messages import ToolMessage
        from langgraph.types import Command

        seen = {}

        def ok_tool() -> str:
            return "ok"

        def boom_tool() -> str:
            raise RuntimeError("boom")

        def after_tool_result(name, result, tool_call_id):
            seen.setdefault("ids", []).append((name, tool_call_id))
            return [ToolMessage(content=result, tool_call_id=tool_call_id)]

        def handle_error(error, node):
            return Command(goto=Phase.LLM)

        agent = make_agent(
            [
                [
                    content("t1"),
                    done(
                        tool_calls=[
                            {"id": "c1", "name": "ok_tool", "arguments": {}},
                            {"id": "c2", "name": "boom_tool", "arguments": {}},
                        ]
                    ),
                ],
                [
                    content("t2"),
                    done(tool_calls=[{"id": "c3", "name": "ok_tool", "arguments": {}}]),
                ],
                [content("final"), done()],
            ],
            hooks={"after_tool_result": after_tool_result, "handle_error": handle_error},
        )
        agent.register_tool(ok_tool)
        agent.register_tool(boom_tool)
        result = agent.invoke("go")
        # 第二个工具在 dispatch 阶段抛异常(不走 after_tool_result)→ handle_error
        # 改道 LLM 重试；每次 after_tool_result 都拿到当次工具的正确 id，无残留
        assert result == "final"
        assert seen["ids"] == [("ok_tool", "c1"), ("ok_tool", "c3")]
