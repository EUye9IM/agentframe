from __future__ import annotations

from agentframe import Phase

from .conftest import content, done


class TestFlowControl:
    def test_default_routing_with_tools(self, make_agent):
        agent = make_agent(
            [
                [content("tool"), done(tool_calls=[{"id": "c1", "name": "echo_hi", "arguments": {}}])],
                [content("final"), done()],
            ]
        )
        agent.register_tool(lambda: "ok")
        agent.invoke("go")
        assert agent.log.count("before_tool_call") == 1

    def test_default_routing_no_tools(self, make_agent):
        agent = make_agent([[content("hi"), done()]])
        agent.invoke("go")
        assert agent.log.count("before_tool_call") == 0

    def test_handle_next_override_forces_end(self, make_agent):
        def hook(from_node, default):
            return Phase.END

        agent = make_agent(
            [[content("tool"), done(tool_calls=[{"id": "c1", "name": "echo_hi", "arguments": {}}])]],
            hooks={"handle_next": hook},
        )
        agent.register_tool(lambda: "ok")
        result = agent.invoke("go")
        assert result == "tool"
        assert agent.log.count("before_tool_call") == 0

    def test_handle_next_receives_arguments(self, make_agent):
        seen = {}

        def hook(from_node, default):
            seen["from"] = from_node
            seen["default"] = default
            return default

        agent = make_agent([[content("hi"), done()]], hooks={"handle_next": hook})
        agent.invoke("go")
        assert seen["from"] == Phase.LLM
        assert seen["default"] == Phase.END
