from __future__ import annotations

from typing import Any, cast

from .core.base import BaseAgent
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
        middlewares: list[Any] | None = None,
        checkpointer: Any | None = None,
    ) -> None:
        mws = list(middlewares or [])
        if mws:
            # middlewares 接收中间件类的实例；列表第一个 = 执行顺序第一个（最外层）
            Concrete = type(
                "_ConcreteAgent", (*[type(m) for m in mws][::-1], type(self)), {}
            )
            self.__class__ = cast(Any, Concrete)
        super().__init__(
            llm_client=llm_client,
            system_prompt=system_prompt,
            checkpointer=checkpointer,
        )
