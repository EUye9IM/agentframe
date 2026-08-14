from __future__ import annotations

import logging

from agentframe import Agent
from agentframe.middlewares import log

from .conftest import ScriptedLLMClient, content, done


def _test_logger() -> logging.Logger:
    return logging.getLogger("agentframe.test.logging")


class TestLoggingMiddleware:
    def test_logs_trace_turn_llm(self, caplog):
        logger = _test_logger()
        with caplog.at_level(logging.INFO, logger="agentframe.test.logging"):
            agent = Agent(
                llm_client=ScriptedLLMClient([[content("hi"), done()]]),
                middlewares=[log(logger)()],
                session_id="s1",
            )
            agent.invoke("hello")
        messages = caplog.messages
        assert any("trace start: session=s1 ctx_msgs=1" in m for m in messages)
        assert any("trace end: session=s1" in m for m in messages)
        assert any(m.startswith("turn start: session=s1") for m in messages)
        assert any(m.startswith("turn end: session=s1") for m in messages)
        assert any(m.startswith("llm: session=s1") for m in messages)

    def test_logs_tool_event(self, caplog):
        logger = _test_logger()
        with caplog.at_level(logging.INFO, logger="agentframe.test.logging"):
            agent = Agent(
                llm_client=ScriptedLLMClient(
                    [
                        [content("t"), done(tool_calls=[{"id": "c1", "name": "echo_hi", "arguments": {}}])],
                        [content("final"), done()],
                    ]
                ),
                middlewares=[log(logger)()],
            )
            agent.register_tool(lambda: "ok")
            agent.invoke("go")
        assert any(
            m.startswith("tool start: session=default calls=echo_hi(c1)") for m in caplog.messages
        )
        assert any(
            m.startswith("tool end: session=default name=echo_hi id=c1") for m in caplog.messages
        )

    def test_logs_error_event(self, caplog):
        logger = _test_logger()
        with caplog.at_level(logging.INFO, logger="agentframe.test.logging"):
            agent = Agent(
                llm_client=ScriptedLLMClient(
                    [[content("hi"), done()]], raise_at=1, exc=RuntimeError("boom")
                ),
                middlewares=[log(logger)()],
            )
            agent.invoke("go")
        assert any(
            m.startswith("error: session=default node=") and "type=RuntimeError: boom" in m
            for m in caplog.messages
        )
