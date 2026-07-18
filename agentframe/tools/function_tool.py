from __future__ import annotations

import inspect
from typing import Any, Callable, overload, get_type_hints

_type_map: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
    type(None): "null",
}


class FunctionTool:
    def __init__(
        self,
        func: Callable,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        self.func: Callable = func
        self.name: str = name or func.__name__
        self.description: str = description or (func.__doc__ or "").strip()
        self.openai_tool: dict = self._build_schema()

    def _build_schema(self) -> dict:
        sig = inspect.signature(self.func)
        hints = get_type_hints(self.func)

        properties: dict[str, dict] = {}
        required: list[str] = []

        for name, param in sig.parameters.items():
            if name in ("self", "cls"):
                continue
            pytype = hints.get(name, str)
            properties[name] = {
                "type": _type_map.get(pytype, "string"),
                "description": "",
            }
            if param.default is inspect.Parameter.empty:
                required.append(name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def call(self, **kwargs: Any) -> str:
        try:
            result = self.func(**kwargs)
            return str(result)
        except Exception as e:
            return f"Error: {e}"


@overload
def function_tool(
    func: Callable,
    *,
    name: str | None = None,
    description: str | None = None,
) -> FunctionTool: ...

@overload
def function_tool(
    func: None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Callable[[Callable], FunctionTool]: ...

def function_tool(
    func: Callable | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> FunctionTool | Callable[[Callable], FunctionTool]:
    if func is None:
        return lambda f: FunctionTool(f, name=name, description=description)
    return FunctionTool(func, name=name, description=description)
