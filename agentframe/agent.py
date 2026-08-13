from __future__ import annotations

from typing import Any

from .core.base import BaseAgent
from .core.hooks import Middleware
from .llm.types import LLMClientProtocol


class Agent(BaseAgent):
    """Public agent class.

    - Inherit to override hooks.
    - Pass `middlewares=[...]` to stack middleware classes via dynamic class
      inheritance (MRO order = list order = execution order).
    """

    def __init__(
        self,
        *,
        llm_client: LLMClientProtocol,
        system_prompt: str | None = None,
        middlewares: list[Middleware] | None = None,
        compile_kwargs: dict[str, Any] | None = None,
    ) -> None:
        mws = list(middlewares or [])
        if mws:
            # middlewares 接收中间件类的实例；列表第一个 = 执行顺序第一个（最外层）。
            # 动态 __class__ 替换是刻意为之，绕过静态 MRO 检查。
            Concrete = type(  # pyright: ignore[reportGeneralTypeIssues]
                "_ConcreteAgent", (*[type(m) for m in mws][::-1], type(self)), {}
            )
            setattr(self, "__class__", Concrete)
        super().__init__(
            llm_client=llm_client,
            system_prompt=system_prompt,
            compile_kwargs=compile_kwargs,
        )
