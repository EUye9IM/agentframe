from __future__ import annotations

import logging
import time
from typing import override

from langchain_core.messages import BaseMessage, ToolCall
from langgraph.errors import NodeError
from langgraph.types import Command

from ..core.hooks import Middleware
from ..core.phases import Phase
from ..core.state import AgentState
from ..llm.types import LLMResponse


def log(logger: logging.Logger) -> type[Middleware]:
    """日志中间件工厂：用标准 `logging.Logger` 记录关键事件。

    事件：trace start/end、turn start/end、llm、tool、error。
    配置（logger）由闭包捕获，规避中间件实例 `__init__` 状态丢失。
    计时/turn 计数用类属性默认值 + 运行时覆盖；`session_id` 读 `self.session_id`。
    """

    class LoggingMiddleware(Middleware):
        session_id: str = "default"
        _trace_t0: float = 0.0
        _turn_t0: float = 0.0
        _turn_count: int = 0

        @override
        def before_trace(self, messages: list[BaseMessage], session_id: str) -> list[BaseMessage]:
            self._trace_t0 = time.monotonic()
            logger.info("trace start: session=%s ctx_msgs=%d", session_id, len(messages))
            return super().before_trace(messages, session_id)

        @override
        def after_trace(self, data: AgentState, session_id: str) -> str:
            dt = time.monotonic() - self._trace_t0
            logger.info("trace end: session=%s elapsed=%.3fs", session_id, dt)
            return super().after_trace(data, session_id)

        @override
        def before_turn(self, messages: list[BaseMessage]) -> list[BaseMessage]:
            self._turn_t0 = time.monotonic()
            self._turn_count += 1
            logger.info(
                "turn start: session=%s turn=%d ctx_msgs=%d",
                self.session_id,
                self._turn_count,
                len(messages),
            )
            return super().before_turn(messages)

        @override
        def after_turn(self, messages: list[BaseMessage]) -> list[BaseMessage]:
            dt = time.monotonic() - self._turn_t0
            logger.info(
                "turn end: session=%s turn=%d elapsed=%.3fs added=%d",
                self.session_id,
                self._turn_count,
                dt,
                len(messages),
            )
            return super().after_turn(messages)

        @override
        def after_llm(self, response: LLMResponse) -> list[BaseMessage]:
            usage = response.usage
            logger.info(
                "llm: session=%s model=%s prompt=%d completion=%d total=%d finish=%s",
                self.session_id,
                response.model,
                usage.prompt_tokens if usage else 0,
                usage.completion_tokens if usage else 0,
                usage.total_tokens if usage else 0,
                response.finish_reason,
            )
            return super().after_llm(response)

        @override
        def before_tool_call(self, tool_calls: list[ToolCall]) -> list[ToolCall]:
            calls = " | ".join(f"{tc['name']}({tc['id']})" for tc in tool_calls)
            logger.info("tool start: session=%s calls=%s", self.session_id, calls)
            return super().before_tool_call(tool_calls)

        @override
        def after_tool_result(self, name: str, result: str, tool_call_id: str) -> list[BaseMessage]:
            logger.info(
                "tool end: session=%s name=%s id=%s result_len=%d",
                self.session_id,
                name,
                tool_call_id,
                len(result),
            )
            return super().after_tool_result(name, result, tool_call_id)

        @override
        def handle_error(self, error: NodeError, node: str) -> Command[Phase]:
            exc = error.error
            logger.error(
                "error: session=%s node=%s type=%s: %s",
                self.session_id,
                node,
                type(exc).__name__,
                exc,
            )
            return super().handle_error(error, node)

    return LoggingMiddleware
