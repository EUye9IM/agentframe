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

    def before_trace(self, messages: list[BaseMessage], session_id: str) -> list[BaseMessage]:
        """回合入口（仅 invoke 触发）。入参 = 恢复的历史 + [system?, 新的 HumanMessage]；
        返回值 = 本回合起始上下文，框架会将重写结果持久化回 checkpointer
        （如把过长历史压缩成摘要、注入长期记忆）。

        契约：返回列表的末位必须是入参携带的 HumanMessage（用户本轮输入），
        框架据此断言；中间件可在其前方任意增删/重写其他消息。"""
        return messages

    def after_trace(self, data: AgentState, session_id: str) -> str:
        """回合出口（invoke 收尾）。取最后一条 AI 消息；
        首轮失败时 history 里只有 HumanMessage，回显用户输入会误导调用方，
        故此时返回空串。"""
        for m in reversed(data["messages"]):
            if isinstance(m, AIMessage):
                return str(m.content)
        return ""

    def before_turn(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        """每次 LLM 调用前触发。入参 = 当前完整消息列表；
        返回值用于构造本轮请求，仅影响本轮 LLM 调用，不写回 state。"""
        return messages

    def after_turn(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        """一次 LLM 调用（含其触发的工具执行）结束后触发。
        入参 = 本 turn 将写入历史的新消息：有工具时含 [本 turn 的 AIMessage,
        ToolMessage...]，无工具时仅 AIMessage。返回修改后的列表并真正写回 state。"""
        return messages

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

    def after_tool_result(self, name: str, result: str, tool_call_id: str) -> list[BaseMessage]:
        """工具结果 → 消息历史。默认包成一条 ToolMessage。
        `tool_call_id` 为对应 ToolCall 的 id，须回填到返回的 ToolMessage。"""
        return [ToolMessage(content=result, tool_call_id=tool_call_id)]

    def handle_next(self, from_node: Phase, default: Phase) -> Phase:
        return default

    def handle_error(self, error: NodeError, node: str) -> Command[Phase]:
        """错误处理。`error` 为 LangGraph NodeError，原始异常在 `error.error`；
        `node` 为出错节点名。默认中止（goto END）；中间件覆写时通常
        `isinstance(error.error, ...)` 认领具体异常并改道。"""
        return Command(goto=Phase.END)
