from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar, override

from langchain_core.messages import BaseMessage, SystemMessage

from ..core.hooks import Middleware
from ..llm.types import LLMClientProtocol, LLMRequest


def _estimate_chars(messages: list[BaseMessage]) -> int:
    """无 token 计数器的粗估：内容字符数（≈ 1 token / 3~4 字符）。"""
    return sum(len(str(m.content)) for m in messages)


def compress(
    summarizer: Callable[[list[BaseMessage]], str] | None = None,
    *,
    threshold_chars: int = 16000,
    keep_recent: int = 0,
    summary_prefix: str = "[conversation summary]",
) -> type[Middleware]:
    """压缩中间件工厂：在 `before_trace` 估算历史大小，超阈值时把旧消息压成一条摘要。

    - `summarizer`：`(old_messages) -> summary_str`，缺省用 `self.llm_client` 发起一次摘要请求。
    - `threshold_chars`：历史（不含 system 消息）字符数阈值，超过才压缩。
    - `keep_recent`：压缩后保留的历史尾部消息条数（0 = 全部丢弃只留摘要）。
    - system 角色消息不参与压缩，摘要会**合并进 system 消息**（保留其 id），
      保证压缩后仍是一条 system 在首位、不产生 AI→System 的非法序列；
      无 system 时摘要自成一条。重写结果由框架写回 checkpointer，
      后续回合从摘要续接（旧消息不再进上下文）。
    """

    class CompressionMiddleware(Middleware):
        # 运行时 self 即 Agent 实例；`llm_client` 来自 BaseAgent 属性,类级注解仅为静态可见性
        llm_client: ClassVar[LLMClientProtocol]

        @override
        def before_trace(self, messages: list[BaseMessage], session_id: str) -> list[BaseMessage]:
            messages = super().before_trace(messages, session_id)
            if not messages:
                return messages
            target = messages[-1]
            old = messages[:-1]
            sys_msgs = [m for m in old if isinstance(m, SystemMessage)]
            rest = [m for m in old if not isinstance(m, SystemMessage)]
            if not rest or _estimate_chars(rest) < threshold_chars:
                return messages
            summary = (summarizer or self._summarize)(rest)
            tail = rest[-keep_recent:] if keep_recent > 0 else []
            summary_msg = SystemMessage(content=f"{summary_prefix}\n{summary}")
            if sys_msgs:
                sys_first = next((m for m in sys_msgs if m.id == "system"), sys_msgs[0])
                merged = SystemMessage(
                    content=f"{sys_first.content}\n\n{summary_msg.content}", id=sys_first.id
                )
                others = [m for m in sys_msgs if m is not sys_first]
                return [merged, *others, *tail, target]
            return [summary_msg, *tail, target]

        def _summarize(self, messages: list[BaseMessage]) -> str:
            request = LLMRequest(
                messages=[
                    SystemMessage(
                        content="Summarize the conversation above concisely, "
                        + "keeping key decisions, facts, and open questions."
                    ),
                    *messages,
                ]
            )
            full = ""
            for event in self.llm_client.stream(request):
                if event.type == "content":
                    full += event.content
            return full

    return CompressionMiddleware
