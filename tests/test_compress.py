from __future__ import annotations

from agentframe import Agent
from agentframe.middlewares import compress

from .conftest import ScriptedLLMClient, content, done


class TestCompress:
    def test_noop_under_threshold(self):
        """未超阈值，不触发摘要，request 不变。"""
        client = ScriptedLLMClient([[content("a1"), done()]])
        agent = Agent(
            llm_client=client,
            middlewares=[compress(keep_recent=10)()],
            session_id="s",
        )
        agent.invoke("q1")
        assert len(client.requests) == 1
        assert client.requests[0].messages[0].content == "q1"

    def test_summarizes_and_writes_back(self):
        """超阈值时调 LLM 摘要，request = [system, summary, recent]；写回后旧消息被删。"""
        client = ScriptedLLMClient([
            [content("a1"), done()],       # invoke1 主
            [content("S1"), done()],        # invoke2 摘要
            [content("a2"), done()],        # invoke2 主
        ])
        agent = Agent(
            llm_client=client,
            middlewares=[compress(keep_recent=2)()],
            session_id="s",
            system_prompt="sys",
        )
        agent.invoke("q1")  # [system, h1, a1], real=1 ≤2
        agent.invoke("q2")  # [system, h1, a1, h2], real=3>2
        assert len(client.requests) == 3
        main = client.requests[2]
        # 请求结构: [system, summary, a1, h2]
        assert main.messages[0].content == "sys"
        assert main.messages[1].id == "summary"
        assert "S1" in str(main.messages[1].content)
        assert main.messages[2].content == "a1"
        # 写回: checkpointer 里 h1 被删，summary 存在
        tup = agent.checkpointer.get_tuple({"configurable": {"thread_id": "s"}})
        assert tup is not None
        contents = [m.content for m in tup.checkpoint["channel_values"]["messages"]]
        assert "q1" not in contents  # h1 被删
        assert "S1" in str(contents)  # summary 存在
        assert "a1" in contents  # 保留

    def test_summary_carries_forward(self):
        """压缩后 summary 参与下一轮，不再重复摘要旧消息。"""
        client = ScriptedLLMClient([
            [content("a1"), done()],       # invoke1 主
            [content("S1"), done()],        # invoke2 摘要
            [content("a2"), done()],        # invoke2 主
            [content("a3"), done()],        # invoke3 主
        ])
        agent = Agent(
            llm_client=client,
            middlewares=[compress(keep_recent=3)()],
            session_id="s",
            system_prompt="sys",
        )
        agent.invoke("q1")
        agent.invoke("q2")  # 触发压缩
        agent.invoke("q3")  # 已压缩，不再触发
        req3 = client.requests[3]
        # 请求应包含 summary
        assert any(m.id == "summary" for m in req3.messages)
        # 且删掉的 h1 不在请求中
        assert not any("q1" in str(m.content) for m in req3.messages if m.id != "summary")