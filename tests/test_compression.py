from unittest.mock import patch

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from agentframe import Agent
from tests.conftest import make_response


class TestCompression:

    def test_compress_when_over_threshold(self):
        """When total_tokens > threshold, compressor is called."""
        agent = Agent(model="gpt-4o", compress_threshold=100)
        # Simulate high token count
        state = {"messages": [HumanMessage(content="hello")], "total_tokens": 200}

        with patch.object(agent.compressor, "compress", return_value=[SystemMessage(content="summary")]) as mock_compress:
            with patch.object(agent.llm_client, "invoke", return_value=make_response("ok")):
                agent._call_agent(state)
                mock_compress.assert_called_once()

    def test_no_compress_when_below_threshold(self):
        """When total_tokens <= threshold, compressor is not called."""
        agent = Agent(model="gpt-4o", compress_threshold=1000)
        state = {"messages": [HumanMessage(content="hello")], "total_tokens": 50}

        with patch.object(agent.compressor, "compress") as mock_compress:
            with patch.object(agent.llm_client, "invoke", return_value=make_response("ok")):
                agent._call_agent(state)
                mock_compress.assert_not_called()

    def test_no_threshold_no_compression(self):
        """When compress_threshold is None, compressor is not created."""
        agent = Agent(model="gpt-4o")
        assert agent.compressor is None

    def test_compress_resets_total_tokens(self):
        """After compression, total_tokens is reset to 0."""
        agent = Agent(model="gpt-4o", compress_threshold=100)

        with patch.object(agent.compressor, "compress", return_value=[SystemMessage(content="s")]):
            with patch.object(agent.llm_client, "invoke", return_value=make_response("ok", total_tokens=20)):
                result = agent._call_agent({
                    "messages": [HumanMessage(content="hello")],
                    "total_tokens": 200,
                })
                # Total tokens should be just the new response token count
                assert result["total_tokens"] == 20

    def test_compressor_summarize_called(self):
        """The compressor actually uses the LLM to summarize."""
        from agentframe.compression.summarizer import Compressor

        compressor = Compressor(
            llm_invoke_fn=lambda msgs, tools=None: make_response("summarized content", total_tokens=50),
            threshold=100,
        )

        messages = [HumanMessage(content="a")] * 20
        result = compressor.compress(messages)

        assert len(result) < len(messages)
        assert any(isinstance(m, SystemMessage) for m in result)

    def test_compressor_short_history_unchanged(self):
        """If history is short enough, compressor returns as-is."""
        from agentframe.compression.summarizer import Compressor

        compressor = Compressor(
            llm_invoke_fn=lambda msgs, tools=None: make_response("s"),
            threshold=100,
            keep_last=5,
        )

        messages = [HumanMessage(content=f"msg{i}") for i in range(3)]
        result = compressor.compress(messages)
        assert len(result) == 3
