from unittest.mock import patch, ANY

from langchain_core.messages import AIMessage, ToolMessage

from agentframe import Agent, function_tool
from tests.conftest import make_response, make_tool_call, get_weather, add


class TestToolExecution:

    def test_tool_call_loop(self):
        """LLM returns tool_call → tools execute → LLM returns final text."""
        agent = Agent(model="gpt-4o", tools=[get_weather])

        with patch.object(agent.llm_client, "invoke", side_effect=[
            make_response(content="", tool_calls=make_tool_call("get_weather", {"city": "Beijing"}, id="c1")),
            make_response(content="Beijing is warm"),
        ]):
            result = agent.invoke("weather in Beijing?")
            assert result == "Beijing is warm"

    def test_tool_result_in_next_llm_call(self):
        """Tool result shows up in messages passed to second LLM call."""
        agent = Agent(model="gpt-4o", tools=[get_weather])

        with patch.object(agent.llm_client, "invoke", side_effect=[
            make_response(content="", tool_calls=make_tool_call("get_weather", {"city": "Beijing"}, id="c1")),
            make_response(content="Beijing is warm"),
        ]) as mock:
            agent.invoke("weather?")
            # Second call should have ToolMessage with tool result
            second_call_msgs = mock.call_args_list[1].args[0]
            assert any(isinstance(m, ToolMessage) for m in second_call_msgs)
            assert any("sunny" in str(m.content) for m in second_call_msgs if isinstance(m, ToolMessage))

    def test_multiple_tool_calls(self):
        """Multiple tool_calls in one response are all executed."""
        agent = Agent(model="gpt-4o", tools=[add])

        with patch.object(agent.llm_client, "invoke", side_effect=[
            make_response(content="", tool_calls=[
                {"name": "add", "args": {"a": 1, "b": 2}, "id": "c1", "type": "tool_call"},
                {"name": "add", "args": {"a": 3, "b": 4}, "id": "c2", "type": "tool_call"},
            ]),
            make_response(content="3 and 7"),
        ]):
            result = agent.invoke("compute")
            assert result == "3 and 7"

    def test_tool_error_returns_error_message(self):
        @function_tool
        def broken(x: int) -> str:
            raise ValueError("broken!")

        agent = Agent(model="gpt-4o", tools=[broken])
        with patch.object(agent.llm_client, "invoke", side_effect=[
            make_response(content="", tool_calls=make_tool_call("broken", {"x": 1}, id="c1")),
            make_response(content="got error"),
        ]):
            result = agent.invoke("test")
            assert result == "got error"

    def test_no_tools_passed_to_llm_when_empty(self):
        """When no tools registered, tools=None passed to LLM."""
        agent = Agent(model="gpt-4o")
        with patch.object(agent.llm_client, "invoke", return_value=make_response("ok")) as mock:
            agent.invoke("hello")
            assert mock.call_args[1]["tools"] is None

    def test_tools_passed_to_llm_when_registered(self):
        """When tools registered, tools list passed to LLM."""
        agent = Agent(model="gpt-4o", tools=[get_weather])
        with patch.object(agent.llm_client, "invoke", return_value=make_response("ok")) as mock:
            agent.invoke("hello")
            assert mock.call_args[1]["tools"] is not None
            assert len(mock.call_args[1]["tools"]) == 1
