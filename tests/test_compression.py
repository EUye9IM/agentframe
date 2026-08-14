from __future__ import annotations

from typing import Any, cast

import pytest
from langchain_core.messages import SystemMessage

from agentframe import Agent
from agentframe.middlewares import compress

from .conftest import ScriptedLLMClient, content, done

LONG = "x" * 200


def _agent(scripts: list[list[Any]], **kw: Any) -> Agent:
    return Agent(llm_client=ScriptedLLMClient(scripts), **kw)


def _requests(agent: Agent) -> list[Any]:
    return cast(ScriptedLLMClient, agent._llm_client).requests  # pyright: ignore[reportPrivateUsage]


class TestCompression:
    def test_compresses_oversized_history(self):
        agent = _agent(
            [[content(LONG), done()], [content("second"), done()]],
            middlewares=[compress(lambda msgs: "COMPRESSED_SUMMARY", threshold_chars=100)()],
            system_prompt="sys",
            session_id="s1",
        )
        agent.invoke("q1")  # 长 AI 回复进入历史
        agent.invoke("q2")  # 历史超阈值 → 压缩
        req2 = _requests(agent)[1]
        # 摘要进入第二轮 LLM 上下文，旧长消息移出
        assert any("COMPRESSED_SUMMARY" in str(m.content) for m in req2.messages)
        assert not any(LONG in str(m.content) for m in req2.messages)

    def test_compression_persists_to_checkpoint(self):
        agent = _agent(
            [[content(LONG), done()], [content("second"), done()], [content("third"), done()]],
            middlewares=[compress(lambda msgs: "SUMMARY", threshold_chars=100)()],
            system_prompt="sys",
            session_id="s1",
        )
        agent.invoke("q1")
        agent.invoke("q2")  # 触发压缩并持久化
        agent.invoke("q3")  # 下一轮从压缩后会话续接，不再重新压缩
        req3 = _requests(agent)[2]
        assert any("SUMMARY" in str(m.content) for m in req3.messages)
        assert not any(LONG in str(m.content) for m in req3.messages)

    def test_no_compression_below_threshold(self):
        agent = _agent(
            [[content("first"), done()], [content("second"), done()]],
            middlewares=[compress(lambda msgs: "SUMMARY", threshold_chars=1000)()],
            system_prompt="sys",
            session_id="s1",
        )
        agent.invoke("q1")
        agent.invoke("q2")
        req2 = _requests(agent)[1]
        assert any(m.content == "first" for m in req2.messages)
        assert not any("SUMMARY" in str(m.content) for m in req2.messages)

    def test_default_summarizer_uses_llm_client(self):
        # 默认 summarizer：压缩时用 self.llm_client 发起一次摘要请求
        agent = _agent(
            [
                [content(LONG), done()],        # invoke1 主请求
                [content("SUMMARY_X"), done()],  # invoke2 摘要请求
                [content("second"), done()],     # invoke2 主请求
            ],
            middlewares=[compress(threshold_chars=100)()],
            system_prompt="sys",
            session_id="s1",
        )
        agent.invoke("q1")
        agent.invoke("q2")
        summary_req = _requests(agent)[1]
        main_req2 = _requests(agent)[2]
        # 摘要请求携带旧历史；主请求上下文以摘要开头
        assert any(LONG in str(m.content) for m in summary_req.messages)
        assert any("SUMMARY_X" in str(m.content) for m in main_req2.messages)
        assert not any(LONG in str(m.content) for m in main_req2.messages)


class TestCompressionContract:
    def test_before_trace_must_keep_human_last(self, make_agent):
        def drop_human(messages: list[Any], session_id: str) -> list[Any]:
            return messages[:-1]

        agent = make_agent(
            [[content("hi"), done()]],
            hooks={"before_trace": drop_human},
        )
        with pytest.raises(ValueError, match="HumanMessage"):
            agent.invoke("hello")


class TestCompressionExtras:
    def test_keep_recent_retains_tail(self):
        agent = _agent(
            [[content(LONG), done()], [content("second"), done()]],
            middlewares=[compress(lambda msgs: "SUMMARY", threshold_chars=100, keep_recent=2)()],
            system_prompt="sys",
            session_id="s1",
        )
        agent.invoke("q1")  # 历史: [sys, q1, LONG]
        agent.invoke("q2")  # 压缩,保留最近 2 条旧消息
        req2 = _requests(agent)[1]
        msgs = req2.messages
        # system 与摘要合并为首位一条,最近 2 条旧消息在其后,human 末位
        assert isinstance(msgs[0], SystemMessage)
        assert msgs[0].content.startswith("sys")
        assert "SUMMARY" in msgs[0].content
        assert sum(isinstance(m, SystemMessage) for m in msgs) == 1
        assert any(m.content == "q1" for m in msgs)
        assert any(LONG in str(m.content) for m in msgs)
        assert str(msgs[-1].content) == "q2"

    def test_summary_merged_into_system_before_kept_tail(self):
        # 摘要合并进 system 置于最前,保留的 AI 跟在其后 —— 不会出现 AI→System 非法序列
        agent = _agent(
            [[content(LONG), done()], [content("second"), done()]],
            middlewares=[compress(lambda msgs: "SUMMARY", threshold_chars=100, keep_recent=1)()],
            system_prompt="sys",
            session_id="s1",
        )
        agent.invoke("q1")
        agent.invoke("q2")
        req2 = _requests(agent)[1]
        msgs = req2.messages
        assert isinstance(msgs[0], SystemMessage)
        assert "SUMMARY" in msgs[0].content
        assert any(LONG in str(m.content) for m in msgs)
        assert str(msgs[-1].content) == "q2"

    def test_no_compression_when_only_system_prompt(self):
        # 首轮只有 system 提示词（长度 ≥ 阈值）也不触发压缩
        agent = _agent(
            [[content("hi"), done()]],
            middlewares=[compress(lambda msgs: "SUMMARY", threshold_chars=10)()],
            system_prompt="sys" * 20,
            session_id="s1",
        )
        agent.invoke("hello")
        req1 = _requests(agent)[0]
        msgs = req1.messages
        assert len(_requests(agent)) == 1  # 未发起摘要请求
        assert len(msgs) == 2 and isinstance(msgs[0], SystemMessage)
        assert msgs[0].content == "sys" * 20
        assert not any("SUMMARY" in str(m.content) for m in msgs)

    def test_first_call_can_inject_memory(self, make_agent):
        # 无历史首轮注入记忆,并持久化到 checkpoint(第二轮历史里仍在)
        calls = {"n": 0}

        def inject(messages: list[Any], session_id: str) -> list[Any]:
            calls["n"] += 1
            if calls["n"] == 1:
                return [SystemMessage(content="MEMORY_FACT", id="mem"), *messages]
            return messages

        agent = make_agent(
            [[content("first"), done()], [content("second"), done()]],
            hooks={"before_trace": inject},
            system_prompt="sys",
            session_id="s1",
        )
        agent.invoke("q1")
        req1 = agent.requests[0]
        assert any("MEMORY_FACT" in str(m.content) for m in req1.messages)
        agent.invoke("q2")  # 不再注入;记忆来自持久化的 checkpoint
        req2 = agent.requests[1]
        assert any("MEMORY_FACT" in str(m.content) for m in req2.messages)

    def test_system_prompt_merged_with_summary_and_stays_first(self):
        agent = _agent(
            [[content(LONG), done()], [content("second"), done()], [content("third"), done()]],
            middlewares=[compress(lambda msgs: "SUMMARY", threshold_chars=100)()],
            system_prompt="sys",
            session_id="s1",
        )
        agent.invoke("q1")  # [sys, q1, LONG]
        agent.invoke("q2")  # 压缩: system 与摘要合并为一条,保持在首位
        agent.invoke("q3")  # 后续轮已低于阈值不再压缩,system 恒在首位
        req3 = _requests(agent)[2]
        msgs = req3.messages
        assert isinstance(msgs[0], SystemMessage)
        assert msgs[0].content.startswith("sys")
        assert "SUMMARY" in msgs[0].content
        assert sum(isinstance(m, SystemMessage) for m in msgs) == 1

    def test_system_hoisted_to_front(self, make_agent):
        def inject_memory(messages: list[Any], session_id: str) -> list[Any]:
            return [SystemMessage(content="MEM", id="mem"), *messages]

        agent = make_agent(
            [[content("hi"), done()]],
            hooks={"before_trace": inject_memory},
            system_prompt="sys",
        )
        agent.invoke("hello")
        req = agent.requests[0]
        # 记忆被前置到 system 之前,框架把 system 提升回首位
        assert isinstance(req.messages[0], SystemMessage)
        assert req.messages[0].content == "sys"
        assert any(m.id == "mem" for m in req.messages)
