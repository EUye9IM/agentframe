from __future__ import annotations

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


class MCPClient:
    def __init__(self, config: dict) -> None:
        self.config: dict = config
        self._session: Any = None
        self._exit_stack: Any = None
        self._tools: dict[str, MCPTool] = {}
        self._read: Any = None
        self._write: Any = None

    async def connect(self) -> None:
        transport = self.config.get("transport", "stdio")
        try:
            from mcp import ClientSession
        except ImportError:
            raise ImportError("MCP support requires: pip install mcp")

        import anyio

        if transport == "stdio":
            from mcp.client.stdio import stdio_client, StdioServerParameters

            command = self.config["command"]
            args = self.config.get("args", [])
            server_params = StdioServerParameters(command=command, args=args)

            self._exit_stack = anyio.AsyncExitStack()
            transport_ctx = stdio_client(server_params)
            self._read, self._write = await self._exit_stack.enter_async_context(transport_ctx)

        elif transport == "sse":
            from mcp.client.sse import sse_client

            url = self.config["url"]

            self._exit_stack = anyio.AsyncExitStack()
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

    async def close(self) -> None:
        if self._exit_stack:
            await self._exit_stack.aclose()

    def get_openai_tools(self) -> list[dict]:
        return [tool.to_openai_tool() for tool in self._tools.values()]

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
