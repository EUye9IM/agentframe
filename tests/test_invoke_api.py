from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver

from .conftest import content, done


class TestInvokeApi:
    def test_invoke_builds_system_and_human(self, make_agent):
        agent = make_agent([[content("hi"), done()]], system_prompt="sys")
        agent.invoke("hello")
        req = agent.requests[0]
        assert isinstance(req.messages[0], SystemMessage)
        assert req.messages[0].content == "sys"
        assert isinstance(req.messages[1], HumanMessage)

    def test_invoke_no_system_prompt(self, make_agent):
        agent = make_agent([[content("hi"), done()]])
        agent.invoke("hello")
        req = agent.requests[0]
        assert isinstance(req.messages[0], HumanMessage)

    def test_history_via_checkpointer_and_thread_id(self, make_agent):
        agent = make_agent(
            [[content("first"), done()], [content("second"), done()]],
            system_prompt="sys",
            compile_kwargs={"checkpointer": InMemorySaver()},
        )
        r1 = agent.invoke("q1", config={"configurable": {"thread_id": "s1"}})
        assert r1 == "first"
        r2 = agent.invoke("q2", config={"configurable": {"thread_id": "s1"}})
        assert r2 == "second"
        # 历史 = checkpointer 恢复的 state；第二轮请求带上前轮 AIMessage
        req2 = agent.requests[1]
        assert any(isinstance(m, AIMessage) for m in req2.messages)
        assert any(m.content == "first" for m in req2.messages)
        # system 消息按 id 去重，第二轮不重复
        assert sum(isinstance(m, SystemMessage) for m in req2.messages) == 1

    def test_history_is_per_thread(self, make_agent):
        agent = make_agent(
            [[content("first"), done()], [content("second"), done()]],
            compile_kwargs={"checkpointer": InMemorySaver()},
        )
        agent.invoke("q1", config={"configurable": {"thread_id": "s1"}})
        agent.invoke("q2", config={"configurable": {"thread_id": "s2"}})
        # 不同 thread_id 互不可见，第二轮不带 s1 的历史
        req2 = agent.requests[1]
        assert not any(m.content == "first" for m in req2.messages)


class TestGraphStructure:
    def test_graph_has_llm_and_tools_nodes(self, make_agent):
        agent = make_agent([[content("hi"), done()]])
        agent.invoke("go")
        graph = agent._graph
        node_names = {str(n) for n in graph.nodes.keys()}
        assert "LLM" in node_names
        assert "TOOLS" in node_names
