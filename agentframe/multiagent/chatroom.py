from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from .member import Member, Turn, RoundVotes, MemberApproval, ChatroomResult
from .events import (
    RoundStarted,
    MemberSpeaking,
    MemberOpinion,
    MemberVote,
    ConsensusStatus,
    SummaryDraft,
    MemberApproval as MemberApprovalEvent,
    FinalSummary,
)

PASS_MARKERS = {
    "pass", "pass.", "pass)", "（pass）", "(pass)",
    "no comment", "nothing to add", "no opinion", "agree with the above",
    "无补充", "无意见", "过", "跳过", "同意", "认同", "没有意见",
}

_DISCUSSION_RULES = (
    "You are in a group discussion. Everyone is trying to reach a mutually "
    "acceptable answer. Be concise. If you have nothing to add, reply exactly: PASS"
)


def _is_pass(text: str) -> bool:
    cleaned = re.sub(r"[.!。！\s]+$", "", text.strip().lower())
    return cleaned in PASS_MARKERS or (cleaned.startswith("pass") and len(cleaned) <= 10)


def _parse_vote(text: str) -> tuple[str, str]:
    content = text.strip()
    match = re.search(r"VOTE\s*:\s*(APPROVE|ABSTAIN|DISAGREE)", content, re.IGNORECASE)
    if match:
        vote = match.group(1).upper()
        reason = re.sub(r"VOTE\s*:\s*[A-Z]+\s*[:：]?\s*", "", content, flags=re.IGNORECASE).strip()
        return vote, reason or "(no reason)"
    lowered = content.lower()
    if any(k in lowered for k in ("approve", "agreed", "agree ", "认可", "同意", "yes")):
        return "APPROVE", content[:200]
    if any(k in lowered for k in ("disagree", "反对", "不认可", "no ", "reject")):
        return "DISAGREE", content[:200]
    return "ABSTAIN", content[:200]


def _parse_review(text: str) -> tuple[bool, str]:
    cleaned = text.strip()
    lowered = cleaned.lower()
    if lowered.startswith("approved") or "approved" in lowered:
        return True, cleaned
    if _is_pass(text):
        return True, cleaned
    return False, cleaned


class Chatroom:
    def __init__(
        self,
        members: list[Member],
        *,
        summarizer: Member,
        max_rounds: int = 5,
        max_summary_iters: int = 5,
        secret: bool = False,
    ) -> None:
        if not members:
            raise ValueError("Chatroom requires at least one member")
        self.members: list[Member] = members
        self.summarizer: Member = summarizer
        self.max_rounds: int = max_rounds
        self.max_summary_iters: int = max_summary_iters
        self.secret: bool = secret
        self._last_result: ChatroomResult | None = None

    # ------------------------------------------------------------------
    # Message construction
    # ------------------------------------------------------------------

    def _base_msgs(self, member: Member, topic: str, transcript: list[Turn]) -> list[BaseMessage]:
        messages: list[BaseMessage] = []
        persona = member.persona or (member.agent.system_prompt or "")
        if persona:
            messages.append(SystemMessage(content=persona, id="persona"))
        messages.append(SystemMessage(content=_DISCUSSION_RULES, id="rules"))
        messages.append(HumanMessage(content=f"Topic: {topic}"))
        for turn in transcript:
            messages.append(HumanMessage(content=f"{turn.speaker}: {turn.content}"))
        return messages

    def _speak_msgs(self, member: Member, topic: str, transcript: list[Turn]) -> list[BaseMessage]:
        return self._base_msgs(member, topic, transcript) + [
            HumanMessage(content=f"{member.name}, it's your turn to speak:")
        ]

    def _vote_msgs(self, member: Member, topic: str, transcript: list[Turn]) -> list[BaseMessage]:
        if self.secret:
            format_line = (
                "VOTE:APPROVE if you accept, ABSTAIN if neutral, DISAGREE if the proposed "
                "answer violates your secret rule (no need to explain why).\n"
                "REASON: a brief reason that does NOT reveal your secret rule"
            )
        else:
            format_line = "VOTE:APPROVE|ABSTAIN|DISAGREE\nREASON: your reasoning"
        return self._base_msgs(member, topic, transcript) + [
            HumanMessage(
                content=(
                    f"{member.name}, do you agree the group has reached consensus on the topic?\n"
                    "Reply in this exact format:\n"
                    f"{format_line}"
                )
            )
        ]

    def _summary_msgs(self, topic: str, transcript: list[Turn]) -> list[BaseMessage]:
        messages: list[BaseMessage] = []
        if self.summarizer.persona:
            messages.append(SystemMessage(content=self.summarizer.persona, id="persona"))
        messages.append(SystemMessage(content="You are the summarizer.", id="rules"))
        messages.append(HumanMessage(content=f"Topic: {topic}"))
        for turn in transcript:
            messages.append(HumanMessage(content=f"{turn.speaker}: {turn.content}"))
        messages.append(HumanMessage(
            content="Produce a final summary that states one concrete answer acceptable to everyone."
        ))
        return messages

    def _revise_msgs(self, topic: str, transcript: list[Turn], draft: str, reviews: list[MemberApproval]) -> list[BaseMessage]:
        messages: list[BaseMessage] = []
        if self.summarizer.persona:
            messages.append(SystemMessage(content=self.summarizer.persona, id="persona"))
        messages.append(SystemMessage(content="You are the summarizer.", id="rules"))
        messages.append(HumanMessage(content=f"Topic: {topic}"))
        for turn in transcript:
            messages.append(HumanMessage(content=f"{turn.speaker}: {turn.content}"))
        messages.append(HumanMessage(content=f"Previous draft:\n{draft}"))
        feedback = "\n".join(
            f"{r.name}: {r.feedback or '(approved)'}" for r in reviews if not r.approved
        )
        messages.append(HumanMessage(
            content=(
                "Members raised these objections to your draft. Revise the summary "
                "to address them and state one concrete answer everyone accepts.\n"
                f"{feedback}"
            )
        ))
        return messages

    def _review_msgs(self, member: Member, draft: str) -> list[BaseMessage]:
        messages: list[BaseMessage] = []
        if member.persona:
            messages.append(SystemMessage(content=member.persona, id="persona"))
        messages.append(SystemMessage(content=_DISCUSSION_RULES, id="rules"))
        if self.secret:
            instruction = (
                "reply exactly 'APPROVED' to accept, 'PASS' if you have no objection "
                "(counts as acceptance), or state what must change WITHOUT revealing "
                "your secret rule. You may give indirect hints, e.g. list numbers you would accept."
            )
        else:
            instruction = (
                "reply exactly 'APPROVED' to accept, 'PASS' if you have no objection "
                "(counts as acceptance), or state exactly what must change."
            )
        messages.append(HumanMessage(
            content=(
                f"Final summary proposal:\n{draft}\n\n"
                f"{member.name}, {instruction}"
            )
        ))
        return messages

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    async def stream_discussion(self, topic: str) -> AsyncIterator[Any]:
        transcript: list[Turn] = []
        result_votes: list[RoundVotes] = []

        for round_no in range(1, self.max_rounds + 1):
            yield RoundStarted(round_no=round_no)
            spoke_count = 0
            for member in self.members:
                yield MemberSpeaking(name=member.name, round_no=round_no)
                text = await member.agent.ainvoke_messages(self._speak_msgs(member, topic, transcript))
                if _is_pass(text):
                    continue
                spoke_count += 1
                transcript.append(Turn(speaker=member.name, content=text.strip()))
                yield MemberOpinion(name=member.name, content=text.strip())

            if spoke_count == 0:
                break

            votes = await asyncio.gather(
                *(self._collect_vote(member, topic, transcript) for member in self.members),
                return_exceptions=True,
            )
            round_votes = RoundVotes(round_no=round_no)
            for member, vote in zip(self.members, votes):
                if isinstance(vote, Exception):
                    round_votes.votes.append(MemberVote(name=member.name, vote="ABSTAIN", reason=f"error: {vote}"))
                else:
                    round_votes.votes.append(vote)
                yield MemberVote(name=round_votes.votes[-1].name, vote=round_votes.votes[-1].vote, reason=round_votes.votes[-1].reason)
            result_votes.append(round_votes)
            yield ConsensusStatus(round_no=round_no, reached=round_votes.reached)
            if round_votes.reached:
                break

        draft = await self.summarizer.agent.ainvoke_messages(self._summary_msgs(topic, transcript))
        approvals: list[MemberApproval] = []
        for iteration in range(self.max_summary_iters):
            yield SummaryDraft(draft=draft, iteration=iteration)
            reviews = await asyncio.gather(
                *(self._review(member, draft) for member in self.members),
                return_exceptions=True,
            )
            approvals = []
            for member, rev in zip(self.members, reviews):
                if isinstance(rev, Exception):
                    approvals.append(MemberApproval(name=member.name, approved=False, feedback=f"error: {rev}"))
                else:
                    approvals.append(rev)
                yield MemberApprovalEvent(name=approvals[-1].name, approved=approvals[-1].approved, feedback=approvals[-1].feedback)
            if all(a.approved for a in approvals):
                break
            draft = await self.summarizer.agent.ainvoke_messages(
                self._revise_msgs(topic, transcript, draft, approvals)
            )

        yield FinalSummary(content=draft)

        self._last_result = ChatroomResult(
            topic=topic,
            turns=transcript,
            votes=result_votes,
            summary=draft,
            approvals=approvals,
        )

    async def discuss(self, topic: str) -> ChatroomResult:
        async for _ in self.stream_discussion(topic):
            pass
        assert self._last_result is not None
        return self._last_result

    # ------------------------------------------------------------------
    # Single-member async helpers
    # ------------------------------------------------------------------

    async def _collect_vote(self, member: Member, topic: str, transcript: list[Turn]) -> MemberVote:
        text = await member.agent.ainvoke_messages(self._vote_msgs(member, topic, transcript))
        vote, reason = _parse_vote(text)
        return MemberVote(name=member.name, vote=vote, reason=reason)

    async def _review(self, member: Member, draft: str) -> MemberApproval:
        text = await member.agent.ainvoke_messages(self._review_msgs(member, draft))
        approved, feedback = _parse_review(text)
        return MemberApproval(name=member.name, approved=approved, feedback=feedback)
