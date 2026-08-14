from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import NodeError
from langgraph.graph import END, StateGraph
from langgraph.types import Command

from .hooks import Middleware
from .phases import Phase
from .state import AgentState
from ..llm.types import LLMClientProtocol, LLMRequest, LLMResponse, Usage


class _AgentGraph(Protocol):
    """Minimal compiled-graph surface used by the engine (LangGraph's own
    signatures are heavily overloaded/loosely typed)."""

    def invoke(
        self, input: AgentState, config: RunnableConfig | None = None
    ) -> AgentState: ...

    def update_state(self, config: RunnableConfig, values: dict[str, Any]) -> RunnableConfig: ...


class StreamStop(Exception):
    """Raised by a streaming hook to interrupt generation.

    `_act_llm` attaches the accumulated partial content to `partial` and re-raises;
    the interrupter's `handle_error` override owns recovery.
    """

    def __init__(self, message: str = "", goto: Phase = Phase.END) -> None:
        super().__init__(message)
        self.message: str = message
        self.goto: Phase = goto
        self.partial: str = ""
        self.partial_reasoning: str = ""


class BaseAgent(Middleware):
    def __init__(
        self,
        *,
        llm_client: LLMClientProtocol,
        system_prompt: str | None = None,
        checkpointer: BaseCheckpointSaver[Any] | None = None,
        session_id: str = "default",
    ) -> None:
        self.system_prompt: str | None = system_prompt
        self.checkpointer: BaseCheckpointSaver[Any] = (
            checkpointer if checkpointer is not None else InMemorySaver()
        )
        self.session_id: str = session_id
        self._llm_client: LLMClientProtocol = llm_client
        self._graph: _AgentGraph | None = None
        self._tools: dict[str, Callable[..., Any]] = {}

    # ------------------------------------------------------------------
    # LLM client access (swappable by middleware for model routing)
    # ------------------------------------------------------------------

    @property
    def llm_client(self) -> LLMClientProtocol:
        return self._llm_client

    @llm_client.setter
    def llm_client(self, client: LLMClientProtocol) -> None:
        self._llm_client = client

    # ------------------------------------------------------------------
    # Hooks (from Middleware) — subclasses/middlewares override
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _act_llm(self, state: AgentState) -> dict[str, Any]:
        data = self.before_turn(list(state["messages"]))
        request = self.before_llm(self._build_request(data))
        full = ""
        reasoning = ""
        tool_calls: list[dict[str, Any]] = []
        usage: Usage | None = None
        finish_reason: str | None = None
        try:
            for event in self.llm_client.stream(request):
                if event.type == "reasoning":
                    reasoning += event.content
                    self.on_llm_reasoning(event.content)
                elif event.type == "content":
                    full += event.content
                    self.on_llm_content(event.content)
                elif event.type == "done":
                    tool_calls = event.tool_calls
                    usage = event.usage
                    finish_reason = event.finish_reason
        except StreamStop as stop:
            stop.partial = full
            stop.partial_reasoning = reasoning
            raise
        except KeyboardInterrupt:
            raise StreamStop()
        self.on_reasoning_end(reasoning)
        self.on_content_end(full)
        response = LLMResponse(
            message=AIMessage(content=full, tool_calls=self._to_langchain_tool_calls(tool_calls)),
            reasoning=reasoning,
            model=self.llm_client.model,
            usage=usage,
            finish_reason=finish_reason,
        )
        messages = self.after_llm(response)
        if not tool_calls:
            messages = self.after_turn(messages)
        return {"messages": messages}

    @staticmethod
    def _to_langchain_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for tc in tool_calls:
            out.append(
                {
                    "name": cast(str, tc.get("name", "")),
                    "args": cast(dict[str, Any], tc.get("arguments", {})),
                    "id": cast(str, tc.get("id", "")),
                    "type": "tool_call",
                }
            )
        return out

    def _act_tools(self, state: AgentState) -> dict[str, Any]:
        messages = list(state["messages"])
        last = messages[-1]
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return {"messages": messages}
        approved = self.before_tool_call(last.tool_calls)
        out: list[BaseMessage] = []
        for tc in approved:
            result_str = str(self._dispatch_tool(tc["name"], tc["args"]))
            out.extend(self.after_tool_result(tc["name"], result_str, tc["id"] or ""))
        return {"messages": self.after_turn([last, *out])}

    def _dispatch_tool(self, name: str, arguments: dict[str, Any]) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: tool '{name}' not found"
        return str(cast(object, tool(**arguments)))

    def register_tool(self, fn: Callable[..., Any], *, name: str | None = None) -> None:
        self._tools[name or fn.__name__] = fn

    def _build_request(self, messages: list[BaseMessage]) -> LLMRequest:
        return LLMRequest(messages=messages)

    def _route_after_llm(self, state: AgentState) -> str:
        has_tools = False
        for m in reversed(state["messages"]):
            if isinstance(m, AIMessage):
                has_tools = bool(m.tool_calls)
                break
        default = Phase.TOOLS if has_tools else Phase.END
        return self.handle_next(Phase.LLM, default)

    def _default_error_handler(self, state: AgentState, error: NodeError) -> Command[Phase]:
        return self.handle_error(error, error.node)

    def _build_graph(self) -> None:
        workflow = StateGraph(AgentState)
        # langgraph 通过依赖注入传入 (state, error)，但其 StateNode 类型存根只声明单参数
        error_handler = cast(Runnable[Any, Any], self._default_error_handler)
        workflow.add_node(Phase.LLM, self._act_llm, error_handler=error_handler)  # pyright: ignore[reportUnknownMemberType]
        workflow.add_node(Phase.TOOLS, self._act_tools, error_handler=error_handler)  # pyright: ignore[reportUnknownMemberType]
        workflow.add_conditional_edges(
            Phase.LLM,
            self._route_after_llm,
            {Phase.TOOLS: Phase.TOOLS, Phase.END: END},
        )
        workflow.add_edge(Phase.TOOLS, Phase.LLM)
        workflow.set_entry_point(Phase.LLM)
        # checkpointer 是编译期唯一参数;session_id → thread_id 恢复历史
        graph = workflow.compile(checkpointer=self.checkpointer)  # pyright: ignore[reportUnknownMemberType]
        self._graph = cast(_AgentGraph, cast(object, graph))

    def _ensure_graph(self) -> None:
        if self._graph is None:
            self._build_graph()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _build_input(self, input_text: str) -> list[BaseMessage]:
        """构造本回合新增消息：`[system?, human]`。system 固定 `id="system"`，
        由 langgraph 的 add_messages 在重入时原位去重。"""
        messages: list[BaseMessage] = []
        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt, id="system"))
        messages.append(HumanMessage(content=input_text))
        return messages

    def _load_history(self, session_id: str) -> list[BaseMessage]:
        """从 checkpointer 读回该 session（=thread_id）已持久化的消息历史；无则 `[]`。"""
        config: RunnableConfig = {"configurable": {"thread_id": session_id}}
        tup = self.checkpointer.get_tuple(config)
        if tup is None:
            return []
        return cast(list[BaseMessage], tup.checkpoint["channel_values"].get("messages", []))

    def _seed(self, session_id: str, start: list[BaseMessage]) -> RunnableConfig:
        """把 `before_trace` 重写后的起始上下文 `start` 播种进 checkpoint：
        清掉现有历史、写入 `start[:-1]`，随后以 `[start[-1]]`（human）作为图输入，
        使图从重写后的会话起跑并持久化。`start[-1]` 由调用方断言为 HumanMessage。"""
        assert self._graph is not None
        config: RunnableConfig = {"configurable": {"thread_id": session_id}}
        history = self._load_history(session_id)
        writes: list[BaseMessage] = [
            *[RemoveMessage(id=m.id) for m in history if m.id],
            *start[:-1],
        ]
        return self._graph.update_state(config, {"messages": writes})

    @staticmethod
    def _hoist_system(start: list[BaseMessage]) -> list[BaseMessage]:
        """把框架的 system 消息（`id="system"`）提升到列表首位，
        保证任意 `before_trace` 重写下 system prompt 都位于会话开头。"""
        idx = next(
            (i for i, m in enumerate(start) if isinstance(m, SystemMessage) and m.id == "system"),
            None,
        )
        if idx is None or idx == 0:
            return start
        return [start[idx], *start[:idx], *start[idx + 1 :]]

    def invoke(self, input_text: str) -> str:
        """同步执行完整回合，包在 before_trace / after_trace 生命周期内。
        `session_id` 为构造参数，本调用映射为 LangGraph thread_id，
        同 session 经 checkpointer 沿用之前上下文恢复。

        `before_trace` 收到恢复的历史 + 本轮新消息，可整体重写会话
        （压缩/注入记忆）；system 消息被提升到首位；重写结果经 `_seed`
        写回 checkpointer，本回合图从重写后的会话起跑，下回合同样从
        重写后的会话续接。钩子未改写时直接走图原生恢复路径（跳过播种）。"""
        self._ensure_graph()
        history = self._load_history(self.session_id)
        existing_ids = {m.id for m in history if m.id}
        fresh = [m for m in self._build_input(input_text) if m.id not in existing_ids]
        start = self.before_trace([*history, *fresh], self.session_id)
        if not start or not isinstance(start[-1], HumanMessage):
            raise ValueError("before_trace 返回的会话末位必须是 HumanMessage（用户本轮输入）")
        start = self._hoist_system(start)
        assert self._graph is not None
        if start == [*history, *fresh]:
            config: RunnableConfig = {"configurable": {"thread_id": self.session_id}}
            state = self._graph.invoke({"messages": fresh}, config=config)
        else:
            config = self._seed(self.session_id, start)
            state = self._graph.invoke({"messages": [start[-1]]}, config=config)
        return self.after_trace(state, self.session_id)
