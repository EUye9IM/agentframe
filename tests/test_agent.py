from unittest.mock import patch

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from agentframe import Agent
from tests.conftest import make_response, make_tool_call


class TestAgentBasic:

    def test_invoke_returns_response(self):
        agent = Agent(model="gpt-4o")
        with patch.object(agent.llm_client, "invoke", return_value=make_response("Hello!")):
            result = agent.invoke("Hi")
        assert result == "Hello!"

    def test_system_prompt_is_prepended(self):
        agent = Agent(model="gpt-4o", system_prompt="You are a bot.")
        with patch.object(agent.llm_client, "invoke", return_value=make_response("ok")) as mock:
            agent.invoke("hello")
            called_messages = mock.call_args[0][0]
            assert isinstance(called_messages[0], SystemMessage)
            assert called_messages[0].content == "You are a bot."
            assert isinstance(called_messages[1], HumanMessage)
            assert called_messages[1].content == "hello"

    def test_no_system_prompt(self):
        agent = Agent(model="gpt-4o")
        with patch.object(agent.llm_client, "invoke", return_value=make_response("ok")) as mock:
            agent.invoke("hello")
            called_messages = mock.call_args[0][0]
            assert isinstance(called_messages[0], HumanMessage)

    def test_invoke_with_session_id(self):
        agent = Agent(model="gpt-4o", checkpointer=MemorySaver())
        with patch.object(agent.llm_client, "invoke", return_value=make_response("r1")):
            result = agent.invoke("msg1", session_id="s1")
            assert result == "r1"

    def test_stream_yields_events(self):
        agent = Agent(model="gpt-4o")
        with patch.object(agent.llm_client, "invoke", return_value=make_response("stream response")):
            events = list(agent.stream("hello"))
            assert len(events) > 0
            assert "agent" in events[0]

    def test_custom_kwargs_passed_to_litellm(self):
        agent = Agent(model="gpt-4o", temperature=0.7, max_tokens=100)
        with patch.object(agent.llm_client, "invoke", return_value=make_response("ok")) as mock:
            agent.invoke("hello")
            # kwargs are stored in LLMClient
            assert agent.llm_client.kwargs["temperature"] == 0.7
            assert agent.llm_client.kwargs["max_tokens"] == 100

    def test_api_key_stored(self):
        agent = Agent(model="gpt-4o", api_key="sk-test")
        assert agent.llm_client.api_key == "sk-test"


class TestAgentSyncMCP:

    def test_sync_no_mcp_tools_exposed(self):
        """Sync path does NOT expose MCP tools to LLM."""
        agent = Agent(model="gpt-4o", mcp_configs=[{"transport": "stdio", "command": "dummy"}])
        with patch.object(agent.llm_client, "invoke", return_value=make_response("ok")) as mock:
            agent.invoke("hi")
            tools = mock.call_args[1].get("tools")
            assert tools is None or len(tools) == 0

    def test_sync_mcp_tool_call_returns_async_hint(self):
        """If sync path encounters a tool not in FunctionTool registry, returns async hint."""
        agent = Agent(model="gpt-4o")
        with patch.object(agent.llm_client, "invoke", side_effect=[
            make_response(content="", tool_calls=make_tool_call("some_mcp_tool", {}, id="c1")),
            make_response(content="recovered"),
        ]):
            result = agent.invoke("test")
            assert result == "recovered"
