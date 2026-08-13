from __future__ import annotations

from agentframe import Agent
from agentframe.multiagent import Chatroom, Member
from agentframe.multiagent.events import (
    RoundStarted,
    MemberSpeaking,
    MemberOpinion,
    MemberVote,
    ConsensusStatus,
    SummaryDraft,
    MemberApproval as MemberApprovalEvent,
    FinalSummary,
)


def make_member_agent(responses: list[str]) -> Agent:
    """Create an Agent whose astream returns the scripted responses in order."""
    agent = Agent(model="gpt-4o")
    it = iter(responses)

    async def mock(messages, tools=None):
        yield {"type": "content", "content": next(it)}
        yield {"type": "done", "tool_calls": [], "usage": {"total_tokens": 1}}

    mock.__name__ = "mock_astream"  # type: ignore[attr-defined]
    agent.llm_client.astream = mock
    return agent


def make_member(name: str, responses: list[str], persona: str = "") -> Member:
    return Member(name=name, agent=make_member_agent(responses), persona=persona)


def make_approve_vote(reason: str = "ok") -> str:
    return f"VOTE:APPROVE\nREASON: {reason}"


class TestChatroomFullFlow:

    async def test_full_flow_consensus_and_summary(self):
        """3 members speak, all approve round 1, summary drafted and approved."""
        m1 = make_member("M1", ["I propose 315", make_approve_vote(), "APPROVED"])
        m2 = make_member("M2", ["315 works for me", make_approve_vote(), "APPROVED"])
        m3 = make_member("M3", ["I agree with 315", make_approve_vote(), "APPROVED"])
        summarizer = make_member("Summarizer", ["Summary: 315"], persona="You are the summarizer.")

        chatroom = Chatroom([m1, m2, m3], summarizer=summarizer)
        result = await chatroom.discuss("Find a number everyone accepts")

        assert len(result.turns) == 3
        assert result.votes[0].reached is True
        assert "315" in result.summary
        assert result.all_approved is True

    async def test_member_pass_is_skipped(self):
        """A member replying PASS is skipped from the transcript."""
        m1 = make_member("M1", ["PASS", make_approve_vote(), "APPROVED"])
        m2 = make_member("M2", ["I propose 630", make_approve_vote(), "APPROVED"])
        m3 = make_member("M3", ["630 good", make_approve_vote(), "APPROVED"])
        summarizer = make_member("Summarizer", ["Summary: 630"])

        chatroom = Chatroom([m1, m2, m3], summarizer=summarizer)
        result = await chatroom.discuss("topic")

        speakers = [t.speaker for t in result.turns]
        assert speakers == ["M2", "M3"]
        assert "630" in result.summary

    async def test_consensus_after_multiple_rounds(self):
        """Round 1 has disagreement; round 2 converges and is approved."""
        m1 = make_member("M1", [
            "I propose 5", "VOTE:DISAGREE\nREASON: only 5",
            "315 is divisible by 5", make_approve_vote(), "APPROVED",
        ])
        m2 = make_member("M2", [
            "I propose 7", "VOTE:DISAGREE\nREASON: only 7",
            "315 is divisible by 7", make_approve_vote(), "APPROVED",
        ])
        m3 = make_member("M3", [
            "I propose 9", "VOTE:DISAGREE\nREASON: only 9",
            "315 is divisible by 9", make_approve_vote(), "APPROVED",
        ])
        summarizer = make_member("Summarizer", ["Summary: 315"])

        chatroom = Chatroom([m1, m2, m3], summarizer=summarizer)
        result = await chatroom.discuss("topic")

        assert len(result.turns) == 6
        assert len(result.votes) == 2
        assert result.votes[0].reached is False
        assert result.votes[1].reached is True
        assert result.all_approved is True

    async def test_runs_to_max_rounds_without_consensus(self):
        """No consensus ever: runs all max_rounds, then still produces a summary."""
        disagree = "VOTE:DISAGREE\nREASON: nope"
        responses = []
        for _ in range(3):
            responses += ["no good", disagree]
        responses.append("APPROVED")

        m1 = make_member("M1", list(responses))
        m2 = make_member("M2", list(responses))
        m3 = make_member("M3", list(responses))
        summarizer = make_member("Summarizer", ["Summary: fallback"])

        chatroom = Chatroom([m1, m2, m3], summarizer=summarizer, max_rounds=3)
        result = await chatroom.discuss("topic")

        assert len(result.votes) == 3
        assert all(not rv.reached for rv in result.votes)
        assert result.summary != ""
        assert result.all_approved is True


class TestChatroomSummaryLoop:

    async def test_summary_revision_loop(self):
        """First draft rejected with feedback; revised draft approved by all."""
        m1 = make_member("M1", ["ok", make_approve_vote(), "APPROVED", "APPROVED"])
        m2 = make_member("M2", ["ok", make_approve_vote(), "Not divisible by 7", "APPROVED"])
        m3 = make_member("M3", ["ok", make_approve_vote(), "Not divisible by 9", "APPROVED"])
        summarizer = make_member("Summarizer", ["Summary: 5", "Summary: 315"])

        chatroom = Chatroom([m1, m2, m3], summarizer=summarizer)
        result = await chatroom.discuss("topic")

        assert "315" in result.summary
        assert result.all_approved is True
        # Two revision iterations happened
        assert sum(1 for a in result.approvals if a.approved) == 3

    async def test_summary_max_iters_exceeded(self):
        """Approval never unanimous: caps at max_summary_iters, all_approved False."""
        m1 = make_member("M1", ["ok", make_approve_vote(), "APPROVED", "APPROVED"])
        m2 = make_member("M2", ["ok", make_approve_vote(), "Not divisible by 7", "Not divisible by 7"])
        m3 = make_member("M3", ["ok", make_approve_vote(), "Not divisible by 9", "Not divisible by 9"])
        summarizer = make_member("Summarizer", ["draft1", "draft2", "draft3"])

        chatroom = Chatroom([m1, m2, m3], summarizer=summarizer, max_summary_iters=2)
        result = await chatroom.discuss("topic")

        assert result.summary == "draft3"
        assert result.all_approved is False
        assert sum(1 for a in result.approvals if a.approved) == 1


class TestChatroomEvents:

    async def test_event_sequence(self):
        """Events are emitted in the expected order."""
        m1 = make_member("M1", ["a", make_approve_vote(), "APPROVED"])
        m2 = make_member("M2", ["b", make_approve_vote(), "APPROVED"])
        summarizer = make_member("Summarizer", ["sum"])

        chatroom = Chatroom([m1, m2], summarizer=summarizer)
        events = [e async for e in chatroom.stream_discussion("topic")]

        types = [type(e) for e in events]
        assert types[0] == RoundStarted
        assert types[-1] == FinalSummary

        idx = {t: types.index(t) for t in (RoundStarted, MemberSpeaking, MemberOpinion, MemberVote, ConsensusStatus, SummaryDraft, MemberApprovalEvent)}
        assert idx[RoundStarted] < idx[MemberSpeaking]
        assert idx[MemberSpeaking] < idx[MemberOpinion]
        assert idx[MemberOpinion] < idx[MemberVote]
        assert idx[MemberVote] < idx[ConsensusStatus]
        assert idx[ConsensusStatus] < idx[SummaryDraft]
        assert idx[SummaryDraft] < idx[MemberApprovalEvent]
        assert idx[MemberApprovalEvent] < types.index(FinalSummary)


class TestVoteParsing:

    async def test_unknown_vote_falls_back_to_abstain(self):
        """An unparseable vote is treated as ABSTAIN, not a crash."""
        m1 = make_member("M1", ["a", "hmm not sure", "APPROVED"])
        m2 = make_member("M2", ["b", "VOTE:APPROVE\nREASON: ok", "APPROVED"])
        summarizer = make_member("Summarizer", ["sum"])

        chatroom = Chatroom([m1, m2], summarizer=summarizer)
        result = await chatroom.discuss("topic")

        votes = [v for rv in result.votes for v in rv.votes]
        m1_vote = next(v for v in votes if v.name == "M1")
        assert m1_vote.vote == "ABSTAIN"


class TestSecretMode:

    async def test_secret_vote_and_review_prompts_forbid_revealing(self):
        """Secret mode instructs members not to reveal their rule in votes/reviews."""
        m1 = make_member("M1", ["a", "VOTE:APPROVE\nREASON: ok", "APPROVED"])
        m2 = make_member("M2", ["b", "VOTE:APPROVE\nREASON: ok", "APPROVED"])
        summarizer = make_member("Summarizer", ["sum"])

        chatroom = Chatroom([m1, m2], summarizer=summarizer, secret=True)
        vote_msgs = chatroom._vote_msgs(m1, "topic", [])
        review_msgs = chatroom._review_msgs(m1, "draft")

        vote_text = "\n".join(str(m.content) for m in vote_msgs)
        review_text = "\n".join(str(m.content) for m in review_msgs)
        assert "does NOT reveal your secret rule" in vote_text
        assert "WITHOUT revealing your secret rule" in review_text

    async def test_secret_mode_full_flow(self):
        """Secret mode still runs the full discussion/vote/summary/approval cycle."""
        m1 = make_member("M1", ["I propose 10", make_approve_vote(), "APPROVED"])
        m2 = make_member("M2", ["I propose 9", make_approve_vote(), "APPROVED"])
        summarizer = make_member("Summarizer", ["315"])

        chatroom = Chatroom([m1, m2], summarizer=summarizer, secret=True)
        result = await chatroom.discuss("topic")

        assert len(result.turns) == 2
        assert result.votes[0].reached is True
        assert result.summary == "315"
        assert result.all_approved is True
