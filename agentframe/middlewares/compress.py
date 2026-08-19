from __future__ import annotations

from typing import Any, override

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph.message import RemoveMessage

from ..core.hooks import Middleware
from ..llm.types import LLMRequest


def compress(*, keep_recent: int = 4) -> type[Middleware]:
    """上下文压缩中间件：保留系统提示词和最近 `keep_recent` 条消息，
    更早的消息由 LLM 摘要成一条，写入 state（写回）。

    `before_turn`：超阈值时调 LLM 摘要旧消息，存待删 id + 摘要文本到 pending。
    `after_turn`：通过 `RemoveMessage` 删除被截断的消息，写入摘要，写回 state。
    """

    class CompressMiddleware(Middleware):
        _pending: Any = None

        @override
        def before_turn(self, messages: list[BaseMessage]) -> list[BaseMessage]:
            system = [m for m in messages if m.id == "system"]
            summary = [m for m in messages if m.id == "summary"]
            real = [m for m in messages if m.id not in ("system", "summary")]
            if len(real) <= keep_recent:
                self._pending = None
                return [*system, *summary, *real]
            old = real[:-keep_recent]
            recent = real[-keep_recent:]
            old_summary = str(summary[0].content) if summary else None
            new_summary_text = self._summarize(old_summary, old)
            to_delete = {m.id for m in old}
            if summary:
                to_delete.add("summary")
            self._pending = (to_delete, new_summary_text)
            return [*system, SystemMessage(content=new_summary_text, id="summary"), *recent]

        def _summarize(self, old_summary: str | None, old_messages: list[BaseMessage]) -> str:
            lines: list[str] = []
            if old_summary:
                lines.append(f"[之前的摘要]\n{old_summary}")
            for m in old_messages:
                lines.append(f"{m.__class__.__name__}: {m.content}")
            text = "\n".join(lines)
            prompt = f"请总结以下对话历史,保留关键信息(目标、已完成的步骤、重要结论):\n\n{text}"
            full = ""
            for event in getattr(self, "llm_client").stream(
                LLMRequest(messages=[HumanMessage(content=prompt)])
            ):
                if event.type == "content":
                    full += event.content
            return full

        @override
        def after_turn(self, messages: list[BaseMessage]) -> list[BaseMessage]:
            pending = self._pending
            self._pending = None
            if pending is None:
                return messages
            to_delete, summary_text = pending
            return [
                *(RemoveMessage(id=mid) for mid in to_delete),
                SystemMessage(content=summary_text, id="summary"),
                *messages,
            ]

    return CompressMiddleware