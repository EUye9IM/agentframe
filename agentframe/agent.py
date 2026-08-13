from __future__ import annotations

from typing import Any

from .core.base import BaseAgent


class Agent(BaseAgent):
    """Public agent class.

    - Inherit to override hooks.
    - Pass `middlewares=[...]` to stack middleware classes via dynamic class
      inheritance (MRO order = list order = execution order).
    """

    def __init__(
        self,
        model: str,
        *,
        system_prompt: str | None = None,
        middlewares: list[Any] | None = None,
        checkpointer: Any | None = None,
        llm_client: Any | None = None,
        **kwargs: Any,
    ) -> None:
        mws = list(middlewares or [])
        if mws:
            # middlewares 接收中间件类的实例；列表第一个 = 执行顺序第一个（最外层）
            Concrete = type(
                "_ConcreteAgent", (*[type(m) for m in mws][::-1], type(self)), {}
            )
            self.__class__ = Concrete
        super().__init__(
            model,
            system_prompt=system_prompt,
            checkpointer=checkpointer,
            llm_client=llm_client,
            **kwargs,
        )
