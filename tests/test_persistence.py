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

            # Second call should include messages from the first call
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

            # Each call has only its own input
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

            # Second call should NOT include first call's input
            assert all("msg1" not in m.content for m in second_msgs)

    def test_no_checkpointer_with_session_id(self):
        """If no checkpointer, session_id is ignored."""
        agent = Agent(model="gpt-4o")

        with patch.object(agent.llm_client, "invoke", return_value=make_response("ok")):
            result = agent.invoke("hello", session_id="s1")
            assert result == "ok"
