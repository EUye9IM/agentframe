from __future__ import annotations

from unittest.mock import patch

from langgraph.checkpoint.memory import MemorySaver

from agentframe import Agent
from tests.conftest import make_response


class TestPersistence:

    def test_same_session_remembers_history(self):
        """Same session_id preserves context across invocations."""
        agent = Agent(model="gpt-4o", checkpointer=MemorySaver())

        with patch.object(agent.llm_client, "invoke", side_effect=[
            make_response("Hi, I am Bot"),
            make_response("You said: hello"),
        ]) as mock:
            agent.invoke("hello", session_id="s1")
            agent.invoke("What did I say?", session_id="s1")

            second_msgs = mock.call_args_list[1].args[0]
            contents = [m.content for m in second_msgs]
            assert any("hello" in c for c in contents)

    def test_different_sessions_isolated(self):
        """Different session_ids don't share history."""
        agent = Agent(model="gpt-4o", checkpointer=MemorySaver())

        with patch.object(agent.llm_client, "invoke", side_effect=[
            make_response("Hello A"),
            make_response("Hello B"),
        ]) as mock:
            agent.invoke("msg_a", session_id="s1")
            agent.invoke("msg_b", session_id="s2")

            first_msgs = mock.call_args_list[0].args[0]
            second_msgs = mock.call_args_list[1].args[0]

            assert any("msg_a" in m.content for m in first_msgs)
            assert any("msg_b" in m.content for m in second_msgs)

    def test_invoke_without_session_no_checkpointer(self):
        """Without checkpointer, each invoke is independent."""
        agent = Agent(model="gpt-4o")

        with patch.object(agent.llm_client, "invoke", side_effect=[
            make_response("First"),
            make_response("Second"),
        ]) as mock:
            agent.invoke("msg1")
            agent.invoke("msg2")

            first_msgs = mock.call_args_list[0].args[0]
            second_msgs = mock.call_args_list[1].args[0]

            assert all("msg1" not in m.content for m in second_msgs)

    def test_no_checkpointer_with_session_id(self):
        """If no checkpointer, session_id is ignored."""
        agent = Agent(model="gpt-4o")

        with patch.object(agent.llm_client, "invoke", return_value=make_response("ok")):
            result = agent.invoke("hello", session_id="s1")
            assert result == "ok"

    def test_conversation_chain_preserved(self):
        """All turns in a conversation are preserved in order."""
        agent = Agent(model="gpt-4o", checkpointer=MemorySaver())

        with patch.object(agent.llm_client, "invoke", side_effect=[
            make_response("Hi there"),
            make_response("You said: first"),
            make_response("You said: second"),
        ]):
            agent.invoke("first message", session_id="chain")
            agent.invoke("second message", session_id="chain")

            state = agent._graph.get_state({"configurable": {"thread_id": "chain"}})
            hist = [m.content for m in state.values["messages"] if hasattr(m, "content")]
            assert "first message" in hist
            assert "second message" in hist
            assert hist.count("first message") == 1

    def test_system_prompt_not_duplicated(self):
        """System prompt with fixed id is not duplicated across calls."""
        agent = Agent(model="gpt-4o", system_prompt="You are Bot", checkpointer=MemorySaver())

        with patch.object(agent.llm_client, "invoke", side_effect=[
            make_response("Hi"),
            make_response("Bye"),
        ]):
            agent.invoke("hello", session_id="sys")
            state = agent._graph.get_state({"configurable": {"thread_id": "sys"}})
            sys_count = sum(1 for m in state.values["messages"]
                          if getattr(m, "id", None) == "system"
                             and "You are Bot" in getattr(m, "content", ""))
            assert sys_count == 1


class TestAsyncPersistence:

    async def test_async_same_session_remembers_history(self):
        """Async path: same session_id preserves context across invocations."""
        agent = Agent(model="gpt-4o", checkpointer=MemorySaver())

        with patch.object(agent.llm_client, "astream", self._make_astream(
            [{"type": "done", "tool_calls": [], "usage": {"total_tokens": 10}}],
            [{"type": "done", "tool_calls": [], "usage": {"total_tokens": 10}}],
        )):
            await agent.ainvoke("hello", session_id="s1")
            second_call = await agent.ainvoke("What did I say?", session_id="s1")

        assert second_call == ""

    async def test_async_conversation_chain_full(self):
        """Async path: all three turns visible in final state."""
        agent = Agent(model="gpt-4o", checkpointer=MemorySaver())

        async def mock_astream(messages, tools=None):
            if "hello" in messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1]):
                yield {"type": "content", "content": "Hello!"}
            else:
                yield {"type": "content", "content": "Full history seen"}
            yield {"type": "done", "tool_calls": [], "usage": {"total_tokens": 10}}

        with patch.object(agent.llm_client, "astream", mock_astream):
            await agent.ainvoke("hello", session_id="chain")
            await agent.ainvoke("second", session_id="chain")
            result = await agent.ainvoke("third", session_id="chain")

        assert result == "Full history seen"

    async def test_async_system_prompt_not_duplicated(self):
        """Async path: system prompt with id='system' appears once."""
        agent = Agent(model="gpt-4o", system_prompt="You are a helpful assistant.",
                      checkpointer=MemorySaver())

        async def mock_astream(messages, tools=None):
            yield {"type": "done", "tool_calls": [], "usage": {"total_tokens": 10}}

        with patch.object(agent.llm_client, "astream", mock_astream):
            await agent.ainvoke("hi", session_id="sys")
            await agent.ainvoke("bye", session_id="sys")

        state = agent._graph.get_state({"configurable": {"thread_id": "sys"}})
        sys_msgs = [m for m in state.values["messages"]
                    if getattr(m, "id", None) == "system"
                       and "helpful assistant" in getattr(m, "content", "")]
        assert len(sys_msgs) == 1

    async def test_async_different_sessions_isolated(self):
        """Async path: different session_ids don't share history."""
        agent = Agent(model="gpt-4o", checkpointer=MemorySaver())

        calls = []

        async def mock_astream(messages, tools=None):
            calls.append([m.content for m in messages if hasattr(m, "content")])
            yield {"type": "done", "tool_calls": [], "usage": {"total_tokens": 10}}

        with patch.object(agent.llm_client, "astream", mock_astream):
            await agent.ainvoke("session_a_msg", session_id="a")
            await agent.ainvoke("session_b_msg", session_id="b")

        msg_in_s1 = any("session_a_msg" in c for c in calls[0])
        msg_in_s2 = any("session_b_msg" in c for c in calls[1])
        assert msg_in_s1
        assert msg_in_s2
        assert len(calls) == 2

    @staticmethod
    def _make_astream(*event_lists):
        calls = iter(event_lists)
        async def mock(messages, tools=None):
            for event in next(calls):
                yield event
        return mock
