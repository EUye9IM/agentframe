from __future__ import annotations

import json
from typing import Any, Callable

from langgraph.checkpoint import BaseCheckpointSaver
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
        **kwargs: Any,
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.checkpointer = checkpointer

        self.llm_client = LLMClient(model, api_key=api_key, **kwargs)

        self.tool_registry = ToolRegistry()
        if tools:
            for t in tools:
                self.tool_registry.register(t)

        self.mcp_configs = mcp_configs or []

        self.compressor: Compressor | None = None
        if compress_threshold is not None:
            self.compressor = Compressor(
                llm_invoke_fn=self.llm_client.invoke,
                threshold=compress_threshold,
            )

        self._graph = None

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self) -> None:
        workflow = StateGraph(AgentState)

        workflow.add_node("agent", self._call_agent)
        workflow.add_node("tools", self._call_tools)

        workflow.set_entry_point("agent")

        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {"tools": "tools", END: END},
        )
        workflow.add_edge("tools", "agent")

        self._graph = workflow.compile(checkpointer=self.checkpointer)

    async def _abuild_graph(self) -> None:
        workflow = StateGraph(AgentState)

        workflow.add_node("agent", self._acall_agent)
        workflow.add_node("tools", self._acall_tools)

        workflow.set_entry_point("agent")

        workflow.add_conditional_edges(
            "agent",
            self._ashould_continue,
            {"tools": "tools", END: END},
        )
        workflow.add_edge("tools", "agent")

        self._graph = workflow.compile(checkpointer=self.checkpointer)

    # ------------------------------------------------------------------
    # Sync graph nodes
    # ------------------------------------------------------------------

    def _call_agent(self, state: AgentState) -> dict:
        messages = list(state["messages"])
        total_tokens = state.get("total_tokens", 0)

        if self.compressor and total_tokens > self.compressor.threshold:
            messages = self.compressor.compress(messages)
            total_tokens = 0

        tools = self.tool_registry.get_openai_tools() or None
        response = self.llm_client.invoke(messages, tools=tools)

        total_tokens += response["usage"].get("total_tokens", 0)
        messages.append(response["message"])

        return {"messages": messages, "total_tokens": total_tokens}

    def _call_tools(self, state: AgentState) -> dict:
        messages = list(state["messages"])
        last_message = messages[-1]

        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return {"messages": messages}

        for tc in last_message.tool_calls:
            result = self.tool_registry.call(tc["name"], tc["args"])
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
    # Async graph nodes  (used when MCP tools are configured)
    # ------------------------------------------------------------------

    async def _acall_agent(self, state: AgentState) -> dict:
        messages = list(state["messages"])
        total_tokens = state.get("total_tokens", 0)

        if self.compressor and total_tokens > self.compressor.threshold:
            messages = self.compressor.compress(messages)
            total_tokens = 0

        tools = self.tool_registry.get_openai_tools() or None
        response = self.llm_client.invoke(messages, tools=tools)

        total_tokens += response["usage"].get("total_tokens", 0)
        messages.append(response["message"])

        return {"messages": messages, "total_tokens": total_tokens}

    async def _acall_tools(self, state: AgentState) -> dict:
        messages = list(state["messages"])
        last_message = messages[-1]

        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return {"messages": messages}

        for tc in last_message.tool_calls:
            tool = self.tool_registry.tools.get(tc["name"])
            if isinstance(tool, dict):
                result = await self._call_mcp_tool(tc["name"], tc["args"])
            else:
                result = self.tool_registry.call(tc["name"], tc["args"])
            messages.append(
                ToolMessage(content=str(result), tool_call_id=tc["id"])
            )

        return {"messages": messages}

    async def _call_mcp_tool(self, name: str, args: dict) -> str:
        from ..tools.mcp_client import MCPClient

        for config in self.mcp_configs:
            client = MCPClient(config)
            await client.connect()
            try:
                result = await client.call_tool(name, args)
                return result
            finally:
                await client.close()
        return f"Error: MCP tool '{name}' not found"

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
            messages.append(SystemMessage(content=self.system_prompt))
        messages.append(HumanMessage(content=input_text))
        return {"messages": messages, "total_tokens": 0}

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
    ):
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
    ):
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
