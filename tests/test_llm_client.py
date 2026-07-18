import json
from unittest.mock import MagicMock, patch

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)

from agentframe.llm.client import LLMClient


def _make_mock_response(content="ok", tool_calls=None):
    choice = MagicMock()
    choice.message.content = content
    choice.message.tool_calls = tool_calls
    response = MagicMock()
    response.choices = [choice]
    response.usage = None
    return response


class TestLLMClientMessageConversion:

    def _capture_kwargs(self, messages, tools=None):
        client = LLMClient(model="gpt-4o")
        with patch.object(client, "_get_client") as mock_get:
            mock_openai = MagicMock()
            mock_openai.chat.completions.create.return_value = _make_mock_response()
            mock_get.return_value = mock_openai
            try:
                client.invoke(messages, tools=tools)
            except Exception:
                pass
            return mock_openai.chat.completions.create.call_args[1]

    def test_system_message(self):
        kwargs = self._capture_kwargs([SystemMessage(content="You are a bot")])
        assert kwargs["messages"][0] == {"role": "system", "content": "You are a bot"}

    def test_human_message(self):
        kwargs = self._capture_kwargs([HumanMessage(content="hello")])
        assert kwargs["messages"][0] == {"role": "user", "content": "hello"}

    def test_ai_message(self):
        kwargs = self._capture_kwargs([AIMessage(content="hi")])
        assert kwargs["messages"][0] == {"role": "assistant", "content": "hi"}

    def test_ai_message_with_tool_calls(self):
        ai = AIMessage(
            content="",
            tool_calls=[{"name": "get_weather", "args": {"city": "Beijing"}, "id": "c1", "type": "tool_call"}],
        )
        kwargs = self._capture_kwargs([ai])
        msg = kwargs["messages"][0]
        assert msg["role"] == "assistant"
        assert len(msg["tool_calls"]) == 1
        tc = msg["tool_calls"][0]
        assert tc["id"] == "c1"
        assert tc["function"]["name"] == "get_weather"
        assert json.loads(tc["function"]["arguments"]) == {"city": "Beijing"}

    def test_tool_message(self):
        tm = ToolMessage(content="result", tool_call_id="c1")
        kwargs = self._capture_kwargs([tm])
        assert kwargs["messages"][0] == {"role": "tool", "content": "result", "tool_call_id": "c1"}

    def test_tools_passed(self):
        kwargs = self._capture_kwargs(
            [HumanMessage(content="hi")],
            tools=[{"type": "function", "function": {"name": "test"}}],
        )
        assert kwargs["tools"] == [{"type": "function", "function": {"name": "test"}}]

    def test_no_tools_not_passed(self):
        kwargs = self._capture_kwargs([HumanMessage(content="hi")])
        assert "tools" not in kwargs

    def test_ai_message_with_tool_calls_in_history(self):
        inner = AIMessage(
            content="Let me check",
            tool_calls=[{"name": "search", "args": {"q": "weather"}, "id": "c2", "type": "tool_call"}],
        )
        kwargs = self._capture_kwargs([HumanMessage(content="hi"), inner])
        msg = kwargs["messages"][1]
        assert msg["role"] == "assistant"
        assert msg["tool_calls"][0]["function"]["name"] == "search"

    def test_ai_message_no_content(self):
        kwargs = self._capture_kwargs([AIMessage(content="")])
        assert kwargs["messages"][0]["content"] == ""

    def test_model_passed(self):
        client = LLMClient(model="gpt-4o-mini")
        with patch.object(client, "_get_client") as mock_get:
            mock_openai = MagicMock()
            mock_openai.chat.completions.create.return_value = _make_mock_response()
            mock_get.return_value = mock_openai
            try:
                client.invoke([HumanMessage(content="hi")])
            except Exception:
                pass
            assert mock_openai.chat.completions.create.call_args[1]["model"] == "gpt-4o-mini"

    def test_api_key_stored_on_client(self):
        client = LLMClient(model="gpt-4o", api_key="sk-test")
        assert client.api_key == "sk-test"

    def test_base_url_stored_on_client(self):
        client = LLMClient(model="gpt-4o", base_url="https://api.example.com/v1")
        assert client.base_url == "https://api.example.com/v1"
