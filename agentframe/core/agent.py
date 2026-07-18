from __future__ import annotations

from typing import Any, Callable, Iterator, AsyncIterator

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage

from .state import AgentState
from ..llm.client import LLMClient
from ..tools.registry import ToolRegistry
from ..tools.function_tool import FunctionTool
from ..compression.summarizer import Compressor


class Agent:
    def __init__(
        self,
        model: str,
        *,
        system_prompt: str | None = None,
        tools: list[FunctionTool | Callable | dict] | None = None,
        mcp_configs: list[dict] | None = None,
        compress_threshold: int | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.model: str = model
        self.system_prompt: str | None = system_prompt
        self.checkpointer: BaseCheckpointSaver | None = checkpointer

        self.llm_client: LLMClient = LLMClient(model, api_key=api_key, base_url=base_url, **kwargs)

        self.tool_registry: ToolRegistry = ToolRegistry()
        if tools:
            for t in tools:
                self.tool_registry.register(t)

        self.mcp_configs: list[dict] = mcp_configs or []
        self._mcp_clients: list | None = None

        self.compressor: Compressor | None = None
        if compress_threshold is not None:
            self.compressor = Compressor(
                llm_invoke_fn=self.llm_client.invoke,
                llm_ainvoke_fn=self.llm_client.ainvoke,
                threshold=compress_threshold,
            )

        self._graph: Any = None

    # =============================================
    # Hooks – override in subclass for CLI etc.
    # =============================================

    def on_llm_reasoning(self, text: str) -> None:
        """Called for each reasoning chunk during streaming."""

    def on_llm_content(self, text: str) -> None:
        """Called for each content chunk during streaming."""

    def on_tool_call(self, tool_calls: list[dict]) -> list[dict]:
        """Called before tool execution. Return approved subset. Default: approve all."""
        return tool_calls

    def on_tool_result(self, name: str, result: str) -> None:
        """Called after each tool execution."""

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph_impl(
        self,
        agent_node: Callable[[AgentState], dict],
        tools_node: Callable[[AgentState], dict],
        should_continue_fn: Callable[[AgentState], str],
    ) -> None:
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", agent_node)  # type: ignore[arg-type]
        workflow.add_node("tools", tools_node)  # type: ignore[arg-type]
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            should_continue_fn,
            {"tools": "tools", END: END},
        )
        workflow.add_edge("tools", "agent")
        self._graph = workflow.compile(checkpointer=self.checkpointer)

    def _build_graph(self) -> None:
        self._build_graph_impl(self._call_agent, self._call_tools, self._should_continue)

    async def _abuild_graph(self) -> None:
        self._build_graph_impl(self._acall_agent, self._acall_tools, self._ashould_continue)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _prepare_agent_state(self, state: AgentState) -> tuple[list[BaseMessage], int, list[dict] | None]:
        messages = list(state["messages"])
        total_tokens = state.get("total_tokens", 0)
        if self.compressor and total_tokens > self.compressor.threshold:
            messages = self.compressor.compress(messages)
            total_tokens = 0
        tools = self.tool_registry.get_openai_tools() or None
        return messages, total_tokens, tools

    def _process_tool_calls(
        self, messages: list, last_message: AIMessage
    ) -> tuple[list[dict], list]:
        approved = self.on_tool_call(last_message.tool_calls)  # type: ignore[arg-type]
        approved_ids = {tc["id"] for tc in approved}
        for tc in last_message.tool_calls:
            if tc["id"] not in approved_ids:
                messages.append(
                    ToolMessage(content="(tool call rejected by user)", tool_call_id=tc["id"])
                )
        return approved, messages

    # ------------------------------------------------------------------
    # Sync graph nodes
    # ------------------------------------------------------------------

    def _call_agent(self, state: AgentState) -> dict:
        messages, total_tokens, tools = self._prepare_agent_state(state)
        response = self.llm_client.invoke(messages, tools=tools)
        total_tokens += response["usage"].get("total_tokens", 0)
        messages.append(response["message"])
        return {"messages": messages, "total_tokens": total_tokens}

    def _call_tools(self, state: AgentState) -> dict:
        messages = list(state["messages"])
        last_message = messages[-1]

        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return {"messages": messages}

        approved, messages = self._process_tool_calls(messages, last_message)
        for tc in approved:
            result = self.tool_registry.call(tc["name"], tc["args"])
            self.on_tool_result(tc["name"], str(result))
            messages.append(
                ToolMessage(content=str(result), tool_call_id=tc["id"])
            )

        return {"messages": messages}

    def _should_continue(self, state: AgentState) -> str:
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        return END

    # ------------------------------------------------------------------
    # Async graph nodes  (use streaming + hooks)
    # ------------------------------------------------------------------

    async def _acall_agent(self, state: AgentState) -> dict:
        messages, total_tokens, tools = await self._prepare_agent_state_async(state)

        full_content = ""
        tool_calls: list[dict] = []
        async for event in self.llm_client.astream(messages, tools=tools):
            if event["type"] == "reasoning":
                self.on_llm_reasoning(event["content"])
            elif event["type"] == "content":
                self.on_llm_content(event["content"])
                full_content += event["content"]
            elif event["type"] == "done":
                tool_calls = event["tool_calls"]
                total_tokens += event["usage"].get("total_tokens", 0)

        ai_msg = AIMessage(content=full_content, tool_calls=tool_calls or [])
        messages.append(ai_msg)

        return {"messages": messages, "total_tokens": total_tokens}

    async def _prepare_agent_state_async(self, state: AgentState) -> tuple[list[BaseMessage], int, list[dict] | None]:
        messages = list(state["messages"])
        total_tokens = state.get("total_tokens", 0)
        if self.compressor and total_tokens > self.compressor.threshold:
            messages = await self.compressor.acompress(messages)
            total_tokens = 0
        tools = self.tool_registry.get_openai_tools() or None
        return messages, total_tokens, tools

    async def _acall_tools(self, state: AgentState) -> dict:
        messages = list(state["messages"])
        last_message = messages[-1]

        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return {"messages": messages}

        approved, messages = self._process_tool_calls(messages, last_message)
        for tc in approved:
            tool = self.tool_registry.tools.get(tc["name"])
            if isinstance(tool, dict):
                result = await self._call_mcp_tool(tc["name"], tc["args"])
            else:
                result = self.tool_registry.call(tc["name"], tc["args"])
            self.on_tool_result(tc["name"], str(result))
            messages.append(
                ToolMessage(content=str(result), tool_call_id=tc["id"])
            )

        return {"messages": messages}

    async def _ensure_mcp_connected(self) -> None:
        if self._mcp_clients is not None:
            return
        from ..tools.mcp_client import MCPClient

        self._mcp_clients = []
        for config in self.mcp_configs:
            client = MCPClient(config)
            await client.connect()
            self._mcp_clients.append(client)

    async def _call_mcp_tool(self, name: str, args: dict) -> str:
        await self._ensure_mcp_connected()
        assert self._mcp_clients is not None
        last_error: Exception | None = None
        for client in self._mcp_clients:
            try:
                return await client.call_tool(name, args)
            except Exception as e:
                last_error = e
        if last_error:
            return f"Error: MCP tool '{name}' failed: {last_error}"
        return f"Error: MCP tool '{name}' not found"

    async def aclose_mcp(self) -> None:
        if self._mcp_clients is not None:
            for client in self._mcp_clients:
                await client.close()
            self._mcp_clients = None

    async def _ashould_continue(self, state: AgentState) -> str:
        return self._should_continue(state)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _ensure_graph(self) -> None:
        if self._graph is None:
            self._build_graph()

    async def _aensure_graph(self) -> None:
        if self._graph is None:
            await self._abuild_graph()

    def _build_input(self, input_text: str) -> dict:
        messages: list[BaseMessage] = []
        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt, id="system"))
        messages.append(HumanMessage(content=input_text))
        return {"messages": messages}

    def invoke(
        self,
        input_text: str,
        *,
        session_id: str | None = None,
    ) -> str:
        self._ensure_graph()
        config = self._make_config(session_id)
        state = self._graph.invoke(self._build_input(input_text), config=config)
        return state["messages"][-1].content

    async def ainvoke(
        self,
        input_text: str,
        *,
        session_id: str | None = None,
    ) -> str:
        await self._aensure_graph()
        config = self._make_config(session_id)
        state = await self._graph.ainvoke(
            self._build_input(input_text), config=config
        )
        return state["messages"][-1].content

    def stream(
        self,
        input_text: str,
        *,
        session_id: str | None = None,
    ) -> Iterator[dict]:
        self._ensure_graph()
        config = self._make_config(session_id)
        for event in self._graph.stream(
            self._build_input(input_text), config=config
        ):
            yield event

    async def astream(
        self,
        input_text: str,
        *,
        session_id: str | None = None,
    ) -> AsyncIterator[dict]:
        await self._aensure_graph()
        config = self._make_config(session_id)
        async for event in self._graph.astream(
            self._build_input(input_text), config=config
        ):
            yield event

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_config(session_id: str | None) -> dict | None:
        if session_id is None:
            return None
        return {"configurable": {"thread_id": session_id}}
