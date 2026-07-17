from __future__ import annotations

from unittest.mock import patch

from langchain_core.messages import ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from agentframe import Agent
from tests.conftest import make_response, make_tool_call, get_weather, add


def _make_astream(*event_lists):
    """Return a mock async function that yields each event_list per call.

    Usage: _make_astream(
        [{"type": "done", "tool_calls": [...], "usage": {...}}],
        [{"type": "content", "content": "final"}, {"type": "done", ...}],
    )
    """
    calls = iter(event_lists)

    async def mock(messages, tools=None):
        for event in next(calls):
            yield event

    return mock


class TestAgentAinvoke:

    async def test_ainvoke_returns_response(self):
        agent = Agent(model="gpt-4o")
        mock = _make_astream([
            {"type": "content", "content": "Hello async world"},
            {"type": "done", "tool_calls": [], "usage": {"total_tokens": 15}},
        ])
        with patch.object(agent.llm_client, "astream", mock):
            result = await agent.ainvoke("Hi")
        assert result == "Hello async world"

    async def test_ainvoke_streams_multiple_content_chunks(self):
        agent = Agent(model="gpt-4o")
        mock = _make_astream([
            {"type": "content", "content": "Part1"},
            {"type": "content", "content": "Part2"},
            {"type": "content", "content": "Part3"},
            {"type": "done", "tool_calls": [], "usage": {"total_tokens": 20}},
        ])
        with patch.object(agent.llm_client, "astream", mock):
            result = await agent.ainvoke("Hi")
        assert result == "Part1Part2Part3"

    async def test_ainvoke_with_session_id(self):
        agent = Agent(model="gpt-4o", checkpointer=MemorySaver())
        mock = _make_astream([
            {"type": "content", "content": "r1"},
            {"type": "done", "tool_calls": [], "usage": {"total_tokens": 10}},
        ])
        with patch.object(agent.llm_client, "astream", mock):
            result = await agent.ainvoke("msg1", session_id="s1")
        assert result == "r1"


class TestAgentAsyncToolCalling:

    async def test_async_tool_call_loop(self):
        """ainvoke → _acall_agent → tool_calls in done event → _acall_tools → _acall_agent → final."""
        agent = Agent(model="gpt-4o", tools=[get_weather])

        mock = _make_astream(
            # first _acall_agent: returns tool_call
            [{"type": "done", "tool_calls": make_tool_call("get_weather", {"city": "Beijing"}, id="c1"),
              "usage": {"total_tokens": 30}}],
            # second _acall_agent: returns final text
            [{"type": "content", "content": "Beijing is sunny and 22°C"},
             {"type": "done", "tool_calls": [], "usage": {"total_tokens": 20}}],
        )
        with patch.object(agent.llm_client, "astream", mock):
            result = await agent.ainvoke("weather in Beijing?")
        assert result == "Beijing is sunny and 22°C"

    async def test_async_tool_result_in_next_llm_call(self):
        """Tool result appears in messages sent to second _acall_agent."""
        agent = Agent(model="gpt-4o", tools=[get_weather])

        calls = []

        async def mock(messages, tools=None):
            calls.append(dict(messages=messages, tools=tools))
            if len(calls) == 1:
                yield {"type": "done", "tool_calls": make_tool_call("get_weather", {"city": "Beijing"}, id="c1"),
                       "usage": {"total_tokens": 30}}
            else:
                yield {"type": "content", "content": "done"}
                yield {"type": "done", "tool_calls": [], "usage": {"total_tokens": 20}}

        with patch.object(agent.llm_client, "astream", mock):
            await agent.ainvoke("weather?")

        assert len(calls) == 2
        second_msgs = calls[1]["messages"]
        tool_msgs = [m for m in second_msgs if isinstance(m, ToolMessage)]
        assert len(tool_msgs) >= 1
        assert "Beijing: sunny, 22°C" in tool_msgs[0].content

    async def test_async_multiple_tool_calls(self):
        """Multiple tool_calls in one async response are all executed."""
        agent = Agent(model="gpt-4o", tools=[add])

        mock = _make_astream(
            [{"type": "done", "tool_calls": [
                {"name": "add", "args": {"a": 1, "b": 2}, "id": "c1", "type": "tool_call"},
                {"name": "add", "args": {"a": 3, "b": 4}, "id": "c2", "type": "tool_call"},
            ], "usage": {"total_tokens": 30}}],
            [{"type": "content", "content": "3 and 7"},
             {"type": "done", "tool_calls": [], "usage": {"total_tokens": 15}}],
        )
        with patch.object(agent.llm_client, "astream", mock):
            result = await agent.ainvoke("compute")
        assert result == "3 and 7"

    async def test_async_tool_error_reported(self):
        from agentframe import function_tool

        @function_tool
        def broken(x: int) -> str:
            raise ValueError("broken!")

        agent = Agent(model="gpt-4o", tools=[broken])
        mock = _make_astream(
            [{"type": "done", "tool_calls": make_tool_call("broken", {"x": 1}, id="c1"),
              "usage": {"total_tokens": 30}}],
            [{"type": "content", "content": "got error"},
             {"type": "done", "tool_calls": [], "usage": {"total_tokens": 10}}],
        )
        with patch.object(agent.llm_client, "astream", mock):
            result = await agent.ainvoke("test")
        assert result == "got error"


class TestAgentAsyncHooks:

    async def test_reasoning_hook_called(self):
        agent = Agent(model="gpt-4o")
        reasoning_chunks: list[str] = []

        agent.on_llm_reasoning = lambda text: reasoning_chunks.append(text)

        mock = _make_astream([
            {"type": "reasoning", "content": "Let me"},
            {"type": "reasoning", "content": " think..."},
            {"type": "content", "content": "Answer"},
            {"type": "done", "tool_calls": [], "usage": {"total_tokens": 15}},
        ])
        with patch.object(agent.llm_client, "astream", mock):
            await agent.ainvoke("Hi")

        assert reasoning_chunks == ["Let me", " think..."]

    async def test_content_hook_called(self):
        agent = Agent(model="gpt-4o")
        content_chunks: list[str] = []

        agent.on_llm_content = lambda text: content_chunks.append(text)

        mock = _make_astream([
            {"type": "content", "content": "Hello"},
            {"type": "content", "content": " world"},
            {"type": "done", "tool_calls": [], "usage": {"total_tokens": 10}},
        ])
        with patch.object(agent.llm_client, "astream", mock):
            await agent.ainvoke("Hi")

        assert content_chunks == ["Hello", " world"]

    async def test_reasoning_and_content_hooks_together(self):
        agent = Agent(model="gpt-4o")
        events: list[tuple[str, str]] = []

        agent.on_llm_reasoning = lambda text: events.append(("reasoning", text))
        agent.on_llm_content = lambda text: events.append(("content", text))

        mock = _make_astream([
            {"type": "reasoning", "content": "thinking"},
            {"type": "content", "content": "A"},
            {"type": "content", "content": "B"},
            {"type": "done", "tool_calls": [], "usage": {"total_tokens": 12}},
        ])
        with patch.object(agent.llm_client, "astream", mock):
            await agent.ainvoke("Hi")

        assert events == [
            ("reasoning", "thinking"),
            ("content", "A"),
            ("content", "B"),
        ]

    async def test_tool_call_hook_called(self):
        agent = Agent(model="gpt-4o", tools=[get_weather])
        approved_list: list[list[dict]] = []

        agent.on_tool_call = lambda tcs: (approved_list.append(tcs), tcs)[1]

        mock = _make_astream(
            [{"type": "done", "tool_calls": make_tool_call("get_weather", {"city": "Beijing"}, id="c1"),
              "usage": {"total_tokens": 30}}],
            [{"type": "content", "content": "done"},
             {"type": "done", "tool_calls": [], "usage": {"total_tokens": 10}}],
        )
        with patch.object(agent.llm_client, "astream", mock):
            await agent.ainvoke("weather?")

        assert len(approved_list) == 1
        assert approved_list[0][0]["name"] == "get_weather"

    async def test_tool_result_hook_called(self):
        agent = Agent(model="gpt-4o", tools=[get_weather])
        results: list[tuple[str, str]] = []

        agent.on_tool_result = lambda name, result: results.append((name, result))

        mock = _make_astream(
            [{"type": "done", "tool_calls": make_tool_call("get_weather", {"city": "Beijing"}, id="c1"),
              "usage": {"total_tokens": 30}}],
            [{"type": "content", "content": "done"},
             {"type": "done", "tool_calls": [], "usage": {"total_tokens": 10}}],
        )
        with patch.object(agent.llm_client, "astream", mock):
            await agent.ainvoke("weather?")

        assert len(results) == 1
        assert results[0][0] == "get_weather"
        assert "Beijing: sunny, 22°C" in results[0][1]


class TestAgentAsyncCompression:

    async def test_async_compress_triggers(self):
        """Compressor is called asynchronously when threshold is exceeded."""
        agent = Agent(model="gpt-4o", compress_threshold=-1)

        compressor_called = False

        async def fake_acompress(messages):
            nonlocal compressor_called
            compressor_called = True
            return messages[-3:]

        agent.compressor.acompress = fake_acompress

        # Simulate state with tokens over threshold
        mock = _make_astream([
            {"type": "content", "content": "ok"},
            {"type": "done", "tool_calls": [], "usage": {"total_tokens": 10}},
        ])
        with patch.object(agent.llm_client, "astream", mock):
            await agent.ainvoke("Hi")

        assert compressor_called
