from __future__ import annotations

from typing import Callable, Awaitable

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage


class Compressor:
    def __init__(
        self,
        llm_invoke_fn: Callable[..., dict],
        threshold: int = 100_000,
        keep_last: int = 5,
        summary_model: str | None = None,
        llm_ainvoke_fn: Callable[..., Awaitable[dict]] | None = None,
    ) -> None:
        self.llm_invoke_fn: Callable[..., dict] = llm_invoke_fn
        self.llm_ainvoke_fn: Callable[..., Awaitable[dict]] | None = llm_ainvoke_fn
        self.threshold: int = threshold
        self.keep_last: int = keep_last
        self.summary_model: str | None = summary_model

    def compress(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        if len(messages) <= self.keep_last + 1:
            return messages

        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        other_msgs = [m for m in messages if not isinstance(m, SystemMessage)]

        if len(other_msgs) <= self.keep_last:
            return messages

        to_summarize = other_msgs[: -self.keep_last]
        to_keep = other_msgs[-self.keep_last :]

        summary = self._summarize(to_summarize)

        return system_msgs + [summary] + to_keep

    async def acompress(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        if len(messages) <= self.keep_last + 1:
            return messages

        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        other_msgs = [m for m in messages if not isinstance(m, SystemMessage)]

        if len(other_msgs) <= self.keep_last:
            return messages

        to_summarize = other_msgs[: -self.keep_last]
        to_keep = other_msgs[-self.keep_last :]

        summary = await self._asummarize(to_summarize)

        return system_msgs + [summary] + to_keep

    def _build_summary_prompt(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        return [
            SystemMessage(
                content=(
                    "You are a conversation summarizer. Summarize the key information "
                    "from the following conversation history. Focus on facts, decisions, "
                    "user preferences, and any context needed for future responses. "
                    "Be concise but thorough."
                )
            ),
        ] + messages + [
            HumanMessage(content="Please provide a concise summary of the above conversation.")
        ]

    def _summarize(self, messages: list[BaseMessage]) -> SystemMessage:
        summary_prompt = self._build_summary_prompt(messages)
        kwargs = {"model": self.summary_model} if self.summary_model else {}
        response = self.llm_invoke_fn(summary_prompt, tools=None, **kwargs)
        summary_text = response["message"].content
        return SystemMessage(content=f"[Conversation Summary]\n{summary_text}")

    async def _asummarize(self, messages: list[BaseMessage]) -> SystemMessage:
        if self.llm_ainvoke_fn is None:
            raise RuntimeError("Compressor has no async invoke function; use compress() instead")
        summary_prompt = self._build_summary_prompt(messages)
        kwargs = {"model": self.summary_model} if self.summary_model else {}
        response = await self.llm_ainvoke_fn(summary_prompt, tools=None, **kwargs)
        summary_text = response["message"].content
        return SystemMessage(content=f"[Conversation Summary]\n{summary_text}")
