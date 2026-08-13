from __future__ import annotations

from unittest.mock import Mock, patch

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from agentframe import Agent
from tests.conftest import (
    add,
    make_response,
    make_tool_call,
    make_mock_mcp_tool,
    make_mock_mcp_client,
)


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

    def test_sync_exposes_mcp_tools_to_llm(self):
        """Sync path exposes MCP tools when MCP clients are injected."""
        agent = Agent(model="gpt-4o")
        mcp_client = make_mock_mcp_client(tools=[make_mock_mcp_tool("mcp_search")])
        agent._mcp_clients = [mcp_client]

        with patch.object(agent.llm_client, "invoke", return_value=make_response("ok")) as mock:
            agent.invoke("hi")
            tools = mock.call_args[1].get("tools")
            names = {t["function"]["name"] for t in tools}
            assert "mcp_search" in names

    def test_sync_mcp_tool_call_executed(self):
        """Sync path executes MCP tools via call_tool_sync."""
        agent = Agent(model="gpt-4o")
        mcp_client = make_mock_mcp_client(tools=[make_mock_mcp_tool("mcp_search")])
        mcp_client.call_tool_sync = Mock(return_value="sync result")
        agent._mcp_clients = [mcp_client]

        with patch.object(agent.llm_client, "invoke", side_effect=[
            make_response(content="", tool_calls=make_tool_call("mcp_search", {"q": "x"}, id="c1")),
            make_response(content="sync result"),
        ]):
            result = agent.invoke("test")

        assert result == "sync result"
        mcp_client.call_tool_sync.assert_called_once_with("mcp_search", {"q": "x"})

    def test_sync_mcp_and_registry_tools_coexist(self):
        """Both registry and MCP tools work in sync path."""
        agent = Agent(model="gpt-4o", tools=[add])
        mcp_client = make_mock_mcp_client(tools=[make_mock_mcp_tool("mcp_search")])
        mcp_client.call_tool_sync = Mock(return_value="mcp result")
        agent._mcp_clients = [mcp_client]

        with patch.object(agent.llm_client, "invoke", side_effect=[
            make_response(content="", tool_calls=[
                {"name": "add", "args": {"a": 1, "b": 2}, "id": "c1", "type": "tool_call"},
                {"name": "mcp_search", "args": {"q": "x"}, "id": "c2", "type": "tool_call"},
            ]),
            make_response(content="3 and mcp result"),
        ]):
            result = agent.invoke("test")

        assert result == "3 and mcp result"

    def test_sync_mcp_tool_unknown_returns_error(self):
        """Sync MCP tool call for tool not found in any client returns error message."""
        agent = Agent(model="gpt-4o")
        mcp_client = make_mock_mcp_client(tools=[])
        mcp_client.call_tool_sync = Mock(side_effect=Exception("not found"))
        agent._mcp_clients = [mcp_client]

        with patch.object(agent.llm_client, "invoke", side_effect=[
            make_response(content="", tool_calls=make_tool_call("nonexistent", {}, id="c1")),
            make_response(content="recovered"),
        ]):
            result = agent.invoke("test")

        assert result == "recovered"

    def test_sync_no_mcp_clients_no_mcp_tools(self):
        """Without mcp_configs/_mcp_clients, no MCP tools in sync tools list."""
        agent = Agent(model="gpt-4o")
        with patch.object(agent.llm_client, "invoke", return_value=make_response("ok")) as mock:
            agent.invoke("hi")
            tools = mock.call_args[1].get("tools")
            assert tools is None
