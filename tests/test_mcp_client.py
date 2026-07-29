from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentframe.tools.mcp_client import MCPClient, MCPTool, MCPPrompt


class TestMCPTool:
    def test_to_openai_tool(self):
        session = MagicMock()
        tool = MCPTool(
            name="test_tool",
            description="A test tool",
            input_schema={"type": "object", "properties": {"x": {"type": "string"}}},
            session=session,
        )
        result = tool.to_openai_tool()
        assert result["function"]["name"] == "test_tool"
        assert result["function"]["description"] == "A test tool"
        assert result["function"]["parameters"]["properties"]["x"]["type"] == "string"

    @pytest.mark.asyncio
    async def test_call_returns_text(self):
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_item = MagicMock()
        mock_item.text = "result text"
        mock_result.content = [mock_item]
        mock_session.call_tool = AsyncMock(return_value=mock_result)

        tool = MCPTool(name="t", description="", input_schema={}, session=mock_session)
        result = await tool.call({"x": "y"})
        assert result == "result text"
        mock_session.call_tool.assert_called_once_with("t", {"x": "y"})

    @pytest.mark.asyncio
    async def test_call_handles_non_text_items(self):
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_item = MagicMock()
        del mock_item.text
        mock_result.content = [mock_item, {"type": "text", "text": "hello"}]
        mock_session.call_tool = AsyncMock(return_value=mock_result)

        tool = MCPTool(name="t", description="", input_schema={}, session=mock_session)
        result = await tool.call({})
        assert "hello" in result


class TestMCPPrompt:
    @pytest.mark.asyncio
    async def test_render_returns_formatted_text(self):
        mock_session = AsyncMock()
        mock_result = MagicMock()
        msg1 = MagicMock()
        msg1.role = "user"
        msg1.content.text = "Hello"
        msg2 = MagicMock()
        msg2.role = "assistant"
        msg2.content.text = "World"
        mock_result.messages = [msg1, msg2]
        mock_session.get_prompt = AsyncMock(return_value=mock_result)

        prompt = MCPPrompt(name="greeting", description="A greeting", arguments=[], session=mock_session)
        result = await prompt.render()
        assert "[user]" in result
        assert "Hello" in result
        assert "[assistant]" in result
        assert "World" in result

    def test_to_openai_tool(self):
        prompt = MCPPrompt(
            name="greeting",
            description="A greeting prompt",
            arguments=[{"name": "name", "description": "The name", "required": True}],
            session=MagicMock(),
        )
        result = prompt.to_openai_tool()
        assert result["function"]["name"] == "__prompt_greeting"
        assert "name" in result["function"]["parameters"]["required"]


class TestMCPClientConnect:

    @pytest.mark.asyncio
    async def test_connect_stdio(self):
        with (
            patch("mcp.client.stdio.stdio_client") as mock_stdio,
            patch("mcp.ClientSession") as mock_session_cls,
        ):
            mock_read, mock_write = MagicMock(), MagicMock()
            mock_stdio.return_value.__aenter__ = AsyncMock(return_value=(mock_read, mock_write))

            mock_session = AsyncMock()
            mock_session.initialize = AsyncMock()

            tools_result = MagicMock()
            tool = MagicMock()
            tool.name = "my_tool"
            tool.description = "My tool"
            tool.inputSchema = {"type": "object"}
            tools_result.tools = [tool]
            mock_session.list_tools = AsyncMock(return_value=tools_result)

            prompts_result = MagicMock()
            prompts_result.prompts = []
            mock_session.list_prompts = AsyncMock(return_value=prompts_result)

            mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=mock_session)

            client = MCPClient({"transport": "stdio", "command": "dummy", "args": []})
            await client.connect()

            assert "my_tool" in client._tools
            assert client._tools["my_tool"].name == "my_tool"
            mock_session.list_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_sse(self):
        with (
            patch("mcp.client.sse.sse_client") as mock_sse,
            patch("mcp.ClientSession") as mock_session_cls,
        ):
            mock_read, mock_write = MagicMock(), MagicMock()
            mock_sse.return_value.__aenter__ = AsyncMock(return_value=(mock_read, mock_write))

            mock_session = AsyncMock()
            mock_session.initialize = AsyncMock()
            tools_result = MagicMock()
            tools_result.tools = []
            mock_session.list_tools = AsyncMock(return_value=tools_result)
            prompts_result = MagicMock()
            prompts_result.prompts = []
            mock_session.list_prompts = AsyncMock(return_value=prompts_result)
            mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=mock_session)

            client = MCPClient({"transport": "sse", "url": "http://localhost:8000"})
            await client.connect()

            mock_session.list_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_unknown_transport(self):
        client = MCPClient({"transport": "unknown"})
        with pytest.raises(ValueError, match="Unsupported MCP transport"):
            await client.connect()

    @pytest.mark.asyncio
    async def test_prompts_discovered_on_connect(self):
        with (
            patch("mcp.client.stdio.stdio_client") as mock_stdio,
            patch("mcp.ClientSession") as mock_session_cls,
        ):
            mock_read, mock_write = MagicMock(), MagicMock()
            mock_stdio.return_value.__aenter__ = AsyncMock(return_value=(mock_read, mock_write))

            mock_session = AsyncMock()
            mock_session.initialize = AsyncMock()
            tools_result = MagicMock()
            tools_result.tools = []
            mock_session.list_tools = AsyncMock(return_value=tools_result)

            prompts_result = MagicMock()
            p = MagicMock()
            p.name = "greeting"
            p.description = "A greeting"
            p.arguments = [{"name": "name", "required": True}]
            prompts_result.prompts = [p]
            mock_session.list_prompts = AsyncMock(return_value=prompts_result)

            mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=mock_session)

            client = MCPClient({"transport": "stdio", "command": "dummy"})
            await client.connect()

            assert "greeting" in client.prompts
            assert client.prompts["greeting"].name == "greeting"

    @pytest.mark.asyncio
    async def test_prompts_discovery_fails_gracefully(self):
        with (
            patch("mcp.client.stdio.stdio_client") as mock_stdio,
            patch("mcp.ClientSession") as mock_session_cls,
        ):
            mock_read, mock_write = MagicMock(), MagicMock()
            mock_stdio.return_value.__aenter__ = AsyncMock(return_value=(mock_read, mock_write))

            mock_session = AsyncMock()
            mock_session.initialize = AsyncMock()
            tools_result = MagicMock()
            tools_result.tools = []
            mock_session.list_tools = AsyncMock(return_value=tools_result)
            mock_session.list_prompts = AsyncMock(side_effect=Exception("not supported"))

            mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=mock_session)

            client = MCPClient({"transport": "stdio", "command": "dummy"})
            await client.connect()

            assert client.prompts == {}


class TestMCPClientPrompts:

    @pytest.mark.asyncio
    async def test_get_prompt_returns_rendered_text(self):
        with (
            patch("mcp.client.stdio.stdio_client") as mock_stdio,
            patch("mcp.ClientSession") as mock_session_cls,
        ):
            mock_read, mock_write = MagicMock(), MagicMock()
            mock_stdio.return_value.__aenter__ = AsyncMock(return_value=(mock_read, mock_write))

            mock_session = AsyncMock()
            mock_session.initialize = AsyncMock()
            tools_result = MagicMock()
            tools_result.tools = []
            mock_session.list_tools = AsyncMock(return_value=tools_result)

            prompts_result = MagicMock()
            p = MagicMock()
            p.name = "greeting"
            p.description = ""
            p.arguments = []
            prompts_result.prompts = [p]
            mock_session.list_prompts = AsyncMock(return_value=prompts_result)

            get_result = MagicMock()
            msg = MagicMock()
            msg.role = "assistant"
            msg.content.text = "Hello!"
            get_result.messages = [msg]
            mock_session.get_prompt = AsyncMock(return_value=get_result)

            mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=mock_session)

            client = MCPClient({"transport": "stdio", "command": "dummy"})
            await client.connect()

            text = await client.get_prompt("greeting")
            assert "Hello!" in text

    @pytest.mark.asyncio
    async def test_get_prompt_unknown_name_raises(self):
        client = MCPClient({"transport": "stdio", "command": "dummy"})
        client._prompts = {}
        with pytest.raises(KeyError, match="not found"):
            await client.get_prompt("nonexistent")


class TestMCPClientTools:

    @pytest.mark.asyncio
    async def test_call_tool_returns_result(self):
        with (
            patch("mcp.client.stdio.stdio_client") as mock_stdio,
            patch("mcp.ClientSession") as mock_session_cls,
        ):
            mock_read, mock_write = MagicMock(), MagicMock()
            mock_stdio.return_value.__aenter__ = AsyncMock(return_value=(mock_read, mock_write))

            mock_session = AsyncMock()
            mock_session.initialize = AsyncMock()

            tools_result = MagicMock()
            tool = MagicMock()
            tool.name = "my_tool"
            tool.description = ""
            tool.inputSchema = {"type": "object"}
            tools_result.tools = [tool]
            mock_session.list_tools = AsyncMock(return_value=tools_result)

            prompts_result = MagicMock()
            prompts_result.prompts = []
            mock_session.list_prompts = AsyncMock(return_value=prompts_result)

            call_result = MagicMock()
            item = MagicMock()
            item.text = "done"
            call_result.content = [item]
            mock_session.call_tool = AsyncMock(return_value=call_result)

            mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=mock_session)

            client = MCPClient({"transport": "stdio", "command": "dummy"})
            await client.connect()

            result = await client.call_tool("my_tool", {"x": 1})
            assert result == "done"

    @pytest.mark.asyncio
    async def test_call_tool_unknown_returns_error(self):
        client = MCPClient({"transport": "stdio", "command": "dummy"})
        client._tools = {}
        result = await client.call_tool("nonexistent", {})
        assert "Error" in result

    def test_get_openai_tools(self):
        session = MagicMock()
        client = MCPClient({"transport": "stdio", "command": "dummy"})
        client._tools = {
            "t1": MCPTool(name="t1", description="Tool 1", input_schema={}, session=session),
            "t2": MCPTool(name="t2", description="Tool 2", input_schema={}, session=session),
        }
        result = client.get_openai_tools()
        assert len(result) == 2
        names = {t["function"]["name"] for t in result}
        assert names == {"t1", "t2"}

    def test_openai_tools_empty_when_no_tools(self):
        client = MCPClient({"transport": "stdio", "command": "dummy"})
        client._tools = {}
        assert client.get_openai_tools() == []
