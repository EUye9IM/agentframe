from __future__ import annotations

from collections.abc import Callable
from typing import Any, override

import langchain_core.tools  # pyright: ignore[reportUnusedImport]  确保子模块加载,规避 langchain-core 1.5.x convert 的懒加载 import 缺陷
from langchain_core.utils.function_calling import convert_to_openai_tool

from ..core.hooks import Middleware
from ..llm.types import LLMRequest


def tools(functions: list[Callable[..., Any]] | None = None) -> type[Middleware]:
    """工具中间件工厂：把函数暴露为 OpenAI function calling 工具。

    `before_llm`：把工具反射（docstring→description、类型注解→parameters）成 schema
    注入 `request.tools`，并兜底注册进 `self._tools` 供分发。

    动态增删：实例方法 `register(fn)` / `unregister(name)` 即时同步 `spec` 与
    `self._tools`，下一次 `invoke` 生效。
    """

    spec: dict[str, Callable[..., Any]] = {fn.__name__: fn for fn in (functions or [])}

    class ToolsMiddleware(Middleware):
        # 运行时 self 即 Agent 实例;`_tools` 来自 BaseAgent 构造,类级声明仅为静态可见性
        _tools: dict[str, Callable[..., Any]] = {}

        def register(self, fn: Callable[..., Any]) -> None:
            spec[fn.__name__] = fn
            self._tools[fn.__name__] = fn

        def unregister(self, name: str) -> None:
            spec.pop(name, None)
            self._tools.pop(name, None)

        @override
        def before_llm(self, request: LLMRequest) -> LLMRequest:
            request = super().before_llm(request)
            for name, fn in spec.items():
                self._tools[name] = fn
            request.tools = (request.tools or []) + [convert_to_openai_tool(fn) for fn in spec.values()]
            return request

    return ToolsMiddleware
