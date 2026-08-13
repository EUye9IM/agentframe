from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, ToolCall, ToolMessage
from langgraph.types import Command
from langgraph.errors import NodeError

from .phases import Phase
from .state import AgentState
from ..llm.types import LLMRequest, LLMResponse


class Middleware:
    """Base class for middleware hook implementations.

    Middleware classes are stacked onto `BaseAgent` via multiple inheritance in
    `Agent.__init__`. Each hook here is a no-op default; concrete middlewares
    override a subset and call `super().hook(...)` to continue the chain.
    """

    def before_trace(self, input_text: str, session: str | None) -> str:
        """回合入口（仅 invoke 有文本输入，stream/invoke_messages 不触发）。"""
        return input_text

    def after_trace(self, data: AgentState, session: str | None) -> str:
        """回合出口（invoke / invoke_messages 收尾）。取最后一条 AI 消息；
        首轮失败时 history 里只有 HumanMessage，回显用户输入会误导调用方，
        故此时返回空串。"""
        for m in reversed(data["messages"]):
            if isinstance(m, AIMessage):
                return str(m.content)
        return ""

    def before_turn(self, data: AgentState) -> AgentState:
        return data

    def after_turn(self, data: AgentState) -> AgentState:
        return data

    def before_llm(self, request: LLMRequest) -> LLMRequest:
        return request

    def on_llm_reasoning(self, text: str) -> None: ...

    def on_reasoning_end(self, reasoning: str) -> None: ...

    def on_llm_content(self, text: str) -> None: ...

    def on_content_end(self, content: str) -> None: ...

    def after_llm(self, response: LLMResponse) -> list[BaseMessage]:
        return [response.message]

    def before_tool_call(self, tool_calls: list[ToolCall]) -> list[ToolCall]:
        return tool_calls

    def after_tool_result(self, name: str, result: str) -> list[BaseMessage]:
        call_id = getattr(self, "_last_tool_call_id", "")
        return [ToolMessage(content=result, tool_call_id=call_id)]

    def handle_next(self, from_node: Phase, default: Phase) -> Phase:
        return default

    def handle_error(self, error: NodeError, node: str) -> Command[Phase]:
        return Command(goto=Phase.END)

    def on_state_changed(self, messages: list[BaseMessage]) -> None: ...
