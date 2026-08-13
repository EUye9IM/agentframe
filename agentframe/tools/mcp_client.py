from __future__ import annotations

import asyncio
import threading
from contextlib import AsyncExitStack
from typing import Any


class MCPTool:
    def __init__(self, name: str, description: str, input_schema: dict, session: Any) -> None:
        self.name: str = name
        self.description: str = description
        self.input_schema: dict = input_schema
        self._session: Any = session

    def to_openai_tool(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }

    async def call(self, arguments: dict[str, Any]) -> str:
        result = await self._session.call_tool(self.name, arguments)
        text_parts = []
        for item in result.content:
            if hasattr(item, "text"):
                text_parts.append(item.text)
        return "\n".join(text_parts) if text_parts else str(result.content)


class MCPPrompt:
    def __init__(self, name: str, description: str, arguments: list[dict], session: Any) -> None:
        self.name: str = name
        self.description: str = description
        self.arguments: list[dict] = arguments
        self._session: Any = session

    async def render(self, args: dict[str, Any] | None = None) -> str:
        result = await self._session.get_prompt(self.name, args or {})
        parts: list[str] = []
        for msg in result.messages:
            parts.append(f"[{msg.role}]\n{msg.content.text}")
        return "\n\n".join(parts)

    def to_openai_tool(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": f"__prompt_{self.name}",
                "description": f"MCP Prompt: {self.description}",
                "parameters": {
                    "type": "object",
                    "properties": {
                        a["name"]: {"type": "string", "description": a.get("description", "")}
                        for a in self.arguments
                    },
                    "required": [a["name"] for a in self.arguments if a.get("required", False)],
                },
            },
        }


class MCPClient:
    def __init__(self, config: dict) -> None:
        self.config: dict = config
        self._session: Any = None
        self._exit_stack: Any = None
        self._tools: dict[str, MCPTool] = {}
        self._read: Any = None
        self._write: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Sync bridge: run the async MCP SDK on a persistent background loop
    # ------------------------------------------------------------------

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
            self._loop_thread = threading.Thread(
                target=self._loop.run_forever,
                name=f"mcp-{self.config.get('command', self.config.get('url', 'client'))}",
                daemon=True,
            )
            self._loop_thread.start()
        return self._loop

    def _run_sync(self, coro: Any) -> Any:
        loop = self._ensure_loop()
        return asyncio.run_coroutine_threadsafe(coro, loop).result()

    def connect_sync(self) -> None:
        self._run_sync(self.connect())

    def call_tool_sync(self, name: str, arguments: dict[str, Any]) -> str:
        if name not in self._tools:
            return f"Error: MCP tool '{name}' not found"
        return self._run_sync(self._tools[name].call(arguments))

    def get_prompt_sync(self, name: str, args: dict[str, Any] | None = None) -> str:
        if name not in self.prompts:
            raise KeyError(f"MCP prompt '{name}' not found")
        return self._run_sync(self.prompts[name].render(args))

    def close_sync(self) -> None:
        if self._loop is None or self._loop.is_closed():
            return
        try:
            self._run_sync(self.close())
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._loop_thread:
                self._loop_thread.join(timeout=5)
            self._loop = None
            self._loop_thread = None

    async def connect(self) -> None:
        transport = self.config.get("transport", "stdio")
        from mcp import ClientSession

        if transport == "stdio":
            from mcp.client.stdio import stdio_client, StdioServerParameters

            command = self.config["command"]
            args = self.config.get("args", [])
            server_params = StdioServerParameters(command=command, args=args)

            self._exit_stack = AsyncExitStack()
            transport_ctx = stdio_client(server_params)
            self._read, self._write = await self._exit_stack.enter_async_context(transport_ctx)

        elif transport == "sse":
            from mcp.client.sse import sse_client

            url = self.config["url"]

            self._exit_stack = AsyncExitStack()
            transport_ctx = sse_client(url)
            self._read, self._write = await self._exit_stack.enter_async_context(transport_ctx)
        else:
            raise ValueError(f"Unsupported MCP transport: {transport}")

        session_ctx = ClientSession(self._read, self._write)
        self._session = await self._exit_stack.enter_async_context(session_ctx)
        await self._session.initialize()

        tools_result = await self._session.list_tools()
        for tool in tools_result.tools:
            mcp_tool = MCPTool(
                name=tool.name,
                description=tool.description or "",
                input_schema=tool.inputSchema,
                session=self._session,
            )
            self._tools[tool.name] = mcp_tool

        try:
            prompts_result = await self._session.list_prompts()
            self._prompts: dict[str, MCPPrompt] = {
                p.name: MCPPrompt(
                    name=p.name,
                    description=p.description or "",
                    arguments=p.arguments or [],
                    session=self._session,
                )
                for p in prompts_result.prompts
            }
        except Exception:
            self._prompts = {}

    async def close(self) -> None:
        if self._exit_stack:
            await self._exit_stack.aclose()

    @property
    def prompts(self) -> dict[str, MCPPrompt]:
        return getattr(self, "_prompts", {})

    def get_openai_tools(self) -> list[dict]:
        return [tool.to_openai_tool() for tool in self._tools.values()]

    async def get_prompt(self, name: str, args: dict[str, Any] | None = None) -> str:
        if name not in self.prompts:
            raise KeyError(f"MCP prompt '{name}' not found")
        return await self.prompts[name].render(args)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        if name not in self._tools:
            return f"Error: MCP tool '{name}' not found"
        return await self._tools[name].call(arguments)


class MCPManager:
    def __init__(self) -> None:
        self.clients: list[MCPClient] = []

    def add_server(self, config: dict) -> None:
        client = MCPClient(config)
        self.clients.append(client)

    async def connect_all(self) -> None:
        for client in self.clients:
            await client.connect()

    async def close_all(self) -> None:
        for client in self.clients:
            await client.close()

    def get_all_tools(self) -> list[dict]:
        tools: list[dict] = []
        for client in self.clients:
            tools.extend(client.get_openai_tools())
        return tools
