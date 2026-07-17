from __future__ import annotations

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage


class Compressor:
    def __init__(
        self,
        llm_invoke_fn,
        threshold: int = 100_000,
        keep_last: int = 5,
        summary_model: str | None = None,
    ):
        self.llm_invoke_fn = llm_invoke_fn
        self.threshold = threshold
        self.keep_last = keep_last
        self.summary_model = summary_model

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

    def _summarize(self, messages: list[BaseMessage]) -> SystemMessage:
        summary_prompt = [
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

        kwargs = {"model": self.summary_model} if self.summary_model else {}
        response = self.llm_invoke_fn(summary_prompt, tools=None, **kwargs)

        summary_text = response["message"].content
        return SystemMessage(content=f"[Conversation Summary]\n{summary_text}")
