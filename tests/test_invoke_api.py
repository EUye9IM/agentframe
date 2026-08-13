from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver

from .conftest import content, done


class TestInvokeApi:
    def test_invoke_builds_system_and_human(self, make_agent):
        agent = make_agent([[content("hi"), done()]], system_prompt="sys")
        agent.invoke("hello")
        req = agent.llm_client.requests[0]
        assert isinstance(req.messages[0], SystemMessage)
        assert req.messages[0].content == "sys"
        assert isinstance(req.messages[1], HumanMessage)

    def test_invoke_no_system_prompt(self, make_agent):
        agent = make_agent([[content("hi"), done()]])
        agent.invoke("hello")
        req = agent.llm_client.requests[0]
        assert isinstance(req.messages[0], HumanMessage)

    def test_invoke_messages_uses_given_messages(self, make_agent):
        from langchain_core.messages import HumanMessage

        agent = make_agent([[content("hi"), done()]], system_prompt="sys")
        agent.invoke_messages([HumanMessage(content="custom")])
        req = agent.llm_client.requests[0]
        assert len(req.messages) == 1
        assert req.messages[0].content == "custom"

    def test_session_persistence_via_compile_kwargs_checkpointer(self, make_agent):
        agent = make_agent(
            [[content("first"), done()], [content("second"), done()]],
            compile_kwargs={"checkpointer": InMemorySaver()},
        )
        r1 = agent.invoke("q1", config={"configurable": {"thread_id": "s1"}})
        assert r1 == "first"
        r2 = agent.invoke("q2", config={"configurable": {"thread_id": "s1"}})
        assert r2 == "second"
        # second request should include first turn's AIMessage history
        req2 = agent.llm_client.requests[1]
        assert any(isinstance(m, AIMessage) for m in req2.messages)
        assert any(m.content == "first" for m in req2.messages)


class TestGraphStructure:
    def test_graph_has_llm_and_tools_nodes(self, make_agent):
        agent = make_agent([[content("hi"), done()]])
        agent.invoke("go")
        graph = agent._graph
        node_names = {str(n) for n in graph.nodes.keys()}
        assert "LLM" in node_names
        assert "TOOLS" in node_names
