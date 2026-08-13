from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
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

    def test_history_same_session_accumulates(self, make_agent):
        agent = make_agent(
            [[content("first"), done()], [content("second"), done()]],
            system_prompt="sys",
        )
        r1 = agent.invoke("q1")
        assert r1 == "first"
        r2 = agent.invoke("q2")
        assert r2 == "second"
        # 同 session 沿用之前上下文:第二轮请求带上首轮 AIMessage
        req2 = agent.requests[1]
        assert any(m.content == "first" for m in req2.messages)
        # system 按 id 去重,第二轮不重复
        assert sum(isinstance(m, SystemMessage) for m in req2.messages) == 1

    def test_history_is_per_session(self, make_agent):
        saver = InMemorySaver()
        a1 = make_agent(
            [[content("first"), done()], [content("second"), done()]],
            checkpointer=saver,
            session_id="s1",
        )
        a2 = make_agent(
            [[content("first"), done()], [content("second"), done()]],
            checkpointer=saver,
            session_id="s2",
        )
        a1.invoke("q1")
        a2.invoke("q1")
        # 共享 checkpointer 下,不同 session 互不可见
        req = a2.requests[0]
        assert not any(m.content == "first" for m in req.messages)

    def test_shared_checkpointer_and_session(self, make_agent):
        saver = InMemorySaver()
        a1 = make_agent([[content("first"), done()]], checkpointer=saver, session_id="s1")
        a2 = make_agent(
            [[content("second"), done()], [content("third"), done()]],
            checkpointer=saver,
            session_id="s1",
        )
        a1.invoke("q1")
        r2 = a2.invoke("q2")
        assert r2 == "second"
        # 同一 saver + 同 session:a2 的首轮请求带着 a1 写入的历史
        assert any(m.content == "first" for m in a2.requests[0].messages)

    def test_custom_checkpointer_used(self, make_agent):
        saver = InMemorySaver()
        agent = make_agent([[content("hi"), done()]], checkpointer=saver)
        agent.invoke("go")
        assert agent.checkpointer is saver
        tup = saver.get_tuple({"configurable": {"thread_id": "default"}})
        assert tup is not None


class TestGraphStructure:
    def test_graph_has_llm_and_tools_nodes(self, make_agent):
        agent = make_agent([[content("hi"), done()]])
        agent.invoke("go")
        graph = agent._graph
        node_names = {str(n) for n in graph.nodes.keys()}
        assert "LLM" in node_names
        assert "TOOLS" in node_names
