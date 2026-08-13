from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.errors import NodeError
from langgraph.graph import END, StateGraph
from langgraph.types import Command

from .hooks import Middleware
from .phases import Phase
from .state import AgentState
from ..llm.client import LLMClient
from ..llm.types import LLMRequest, LLMResponse


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


class BaseAgent(Middleware):
    def __init__(
        self,
        model: str,
        *,
        system_prompt: str | None = None,
        checkpointer: Any | None = None,
        llm_client: LLMClient | None = None,
        **kwargs: Any,
    ) -> None:
        self.model: str = model
        self.system_prompt: str | None = system_prompt
        self.checkpointer: Any | None = checkpointer
        self._llm_client = llm_client
        self._kwargs = kwargs
        self._graph: Any = None
        self._tools: dict[str, Any] = {}
        self._last_tool_call_id: str = ""

    # ------------------------------------------------------------------
    # LLM client access (overridable for model switching via middleware)
    # ------------------------------------------------------------------

    @property
    def llm_client(self) -> LLMClient:
        if self._llm_client is None:
            self._llm_client = LLMClient(base_url="", **self._kwargs)
        return self._llm_client

    @llm_client.setter
    def llm_client(self, client: LLMClient) -> None:
        self._llm_client = client

    # ------------------------------------------------------------------
    # Hooks (from Middleware) — subclasses/middlewares override
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _act_llm(self, state: AgentState) -> dict:
        data = dict(state)
        data = self.before_turn(data)
        request = self.before_llm(self._build_request(data))
        full = ""
        reasoning = ""
        tool_calls: list[dict] = []
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
        except StreamStop as stop:
            stop.partial = full
            raise
        except KeyboardInterrupt:
            raise StreamStop()
        self.on_reasoning_end(reasoning)
        self.on_content_end(full)
        response = LLMResponse(
            message=AIMessage(content=full, tool_calls=self._to_langchain_tool_calls(tool_calls)),
            reasoning=reasoning,
            model=request.model,
        )
        messages = self.after_llm(response)
        if not tool_calls:
            self.after_turn(dict(state))
        self._notify_state_changed(messages)
        return {"messages": messages}

    @staticmethod
    def _to_langchain_tool_calls(tool_calls: list[dict]) -> list[dict]:
        out = []
        for tc in tool_calls:
            out.append(
                {
                    "name": tc.get("name", ""),
                    "args": tc.get("arguments", {}),
                    "id": tc.get("id", ""),
                    "type": "tool_call",
                }
            )
        return out

    def _act_tools(self, state: AgentState) -> dict:
        messages = list(state["messages"])
        last = messages[-1]
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return {"messages": messages}
        approved = self.before_tool_call(last.tool_calls)
        out: list[BaseMessage] = []
        for tc in approved:
            self._last_tool_call_id = tc["id"]
            result_str = str(self._dispatch_tool(tc["name"], tc.get("args", tc.get("arguments", {}))))
            out.extend(self.after_tool_result(tc["name"], result_str))
        self._last_tool_call_id = ""
        self.after_turn(dict(state))
        self._notify_state_changed(out)
        return {"messages": out}

    def _dispatch_tool(self, name: str, arguments: dict) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: tool '{name}' not found"
        return str(tool(**arguments))

    def register_tool(self, fn: Any, *, name: str | None = None) -> None:
        self._tools[name or fn.__name__] = fn

    def _notify_state_changed(self, messages: list[BaseMessage]) -> None:
        self.on_state_changed(messages)

    def _build_request(self, data: AgentState) -> LLMRequest:
        return LLMRequest(
            model=self.model,
            messages=list(data["messages"]),
        )

    def _route_after_llm(self, state: AgentState) -> str:
        has_tools = False
        for m in reversed(state["messages"]):
            if isinstance(m, AIMessage):
                has_tools = bool(m.tool_calls)
                break
        default = Phase.TOOLS if has_tools else Phase.END
        return self.handle_next(Phase.LLM, default)

    def _default_error_handler(self, state: AgentState, error: NodeError) -> Command:
        return self.handle_error(error, error.node)

    def _build_graph(self) -> None:
        workflow = StateGraph(AgentState)
        workflow.add_node(Phase.LLM, self._act_llm, error_handler=self._default_error_handler)
        workflow.add_node(Phase.TOOLS, self._act_tools, error_handler=self._default_error_handler)
        workflow.add_conditional_edges(
            Phase.LLM,
            self._route_after_llm,
            {Phase.TOOLS: Phase.TOOLS, Phase.END: END},
        )
        workflow.add_edge(Phase.TOOLS, Phase.LLM)
        workflow.set_entry_point(Phase.LLM)
        self._graph = workflow.compile(checkpointer=self.checkpointer)

    def _ensure_graph(self) -> None:
        if self._graph is None:
            self._build_graph()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _build_input(self, input_text: str) -> dict:
        messages: list[BaseMessage] = []
        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt, id="system"))
        messages.append(HumanMessage(content=input_text))
        return {"messages": messages}

    def invoke(self, input_text: str, *, session_id: str | None = None) -> str:
        input_text = self.before_trace(input_text, session_id)
        self._ensure_graph()
        config = self._make_config(session_id)
        state = self._graph.invoke(self._build_input(input_text), config=config)
        return self.after_trace(state, session_id)

    def stream(self, input_text: str, *, session_id: str | None = None) -> Iterator[dict]:
        self._ensure_graph()
        config = self._make_config(session_id)
        yield from self._graph.stream(self._build_input(input_text), config=config)

    def invoke_messages(self, messages: list[BaseMessage], *, session_id: str | None = None) -> str:
        self._ensure_graph()
        config = self._make_config(session_id)
        state = self._graph.invoke({"messages": messages}, config=config)
        return state["messages"][-1].content

    @staticmethod
    def _make_config(session_id: str | None) -> dict | None:
        if session_id is None:
            return None
        return {"configurable": {"thread_id": session_id}}
