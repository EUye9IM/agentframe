from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RoundStarted:
    round_no: int


@dataclass
class MemberSpeaking:
    name: str
    round_no: int


@dataclass
class MemberOpinion:
    name: str
    content: str


@dataclass
class MemberVote:
    name: str
    vote: str  # APPROVE | ABSTAIN | DISAGREE
    reason: str


@dataclass
class ConsensusStatus:
    round_no: int
    reached: bool


@dataclass
class SummaryDraft:
    draft: str
    iteration: int


@dataclass
class MemberApproval:
    name: str
    approved: bool
    feedback: str


@dataclass
class FinalSummary:
    content: str
