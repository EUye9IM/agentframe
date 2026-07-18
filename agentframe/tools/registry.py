from __future__ import annotations

from typing import Any, Callable

from .function_tool import FunctionTool, function_tool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, FunctionTool | dict] = {}

    def register(self, tool: FunctionTool | Callable | dict) -> None:
        if isinstance(tool, FunctionTool):
            self._tools[tool.name] = tool
        elif callable(tool):
            ft = function_tool(tool)
            self._tools[ft.name] = ft
        elif isinstance(tool, dict):
            name = tool.get("function", {}).get("name")
            if not name:
                raise ValueError("dict tool must have 'function.name' field")
            self._tools[name] = tool
        else:
            raise TypeError(f"Unsupported tool type: {type(tool)}")

    @property
    def tools(self) -> dict[str, FunctionTool | dict]:
        return self._tools

    def call(self, name: str, args: dict[str, Any]) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: tool '{name}' not found"
        if isinstance(tool, FunctionTool):
            return tool.call(**args)
        if isinstance(tool, dict):
            return f"MCP tool '{name}' called with {args}"
        return f"Error: unknown tool type for '{name}'"

    def get_openai_tools(self) -> list[dict]:
        result: list[dict] = []
        for tool in self._tools.values():
            if isinstance(tool, FunctionTool):
                result.append(tool.openai_tool)
            elif isinstance(tool, dict):
                result.append(tool)
        return result
