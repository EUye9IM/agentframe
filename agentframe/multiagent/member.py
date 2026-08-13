from __future__ import annotations

from dataclasses import dataclass, field

from agentframe import Agent

from .events import MemberVote, MemberApproval


@dataclass
class Member:
    name: str
    agent: Agent
    persona: str = ""


@dataclass
class Turn:
    speaker: str
    content: str


@dataclass
class RoundVotes:
    round_no: int
    votes: list[MemberVote] = field(default_factory=list)

    @property
    def reached(self) -> bool:
        if not self.votes:
            return False
        approved = [v for v in self.votes if v.vote == "APPROVE"]
        disallowed = [v for v in self.votes if v.vote == "DISAGREE"]
        return len(approved) >= 1 and not disallowed


@dataclass
class ChatroomResult:
    topic: str
    turns: list[Turn] = field(default_factory=list)
    votes: list[RoundVotes] = field(default_factory=list)
    summary: str = ""
    approvals: list[MemberApproval] = field(default_factory=list)

    @property
    def all_approved(self) -> bool:
        return len(self.approvals) > 0 and all(a.approved for a in self.approvals)
