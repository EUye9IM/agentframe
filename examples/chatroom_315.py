"""Multi-agent chatroom trial: 3 gatekeeper agents must agree on a number.

M5 requires divisibility by 5, M7 by 7, M9 by 9. Any mutually acceptable
integer must be divisible by lcm(5,7,9) = 315. The script asserts this
strongly, along with unanimous approval of the final summary.

With `--secret`, the gatekeepers must NOT reveal their rule (they can only
accept/reject and propose their own candidates). This probes whether the
flow can still converge when members hide their constraints.

Run:
    DEEPSEEK_API_KEY=sk-xxx .venv/bin/python examples/chatroom_315.py
    DEEPSEEK_API_KEY=sk-xxx .venv/bin/python examples/chatroom_315.py --secret
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re

from agentframe import Agent
from agentframe.multiagent import Chatroom, Member

BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

MAX_ROUNDS = 25
MAX_SUMMARY_ITERS = 25


GATEKEEPER = (
    "你是守门员 {name}，只认可能被 {n} 整除的整数。\n"
    "规则：\n"
    "1. 你绝对不能接受任何不能被 {n} 整除的数字。\n"
    "2. 你必须总是提出或支持一个具体的整数，并验证它是否能被 {n} 整除。\n"
    "3. 如果别人的提议不能被 {n} 整除，你必须明确反对并纠正，同时提出一个满足所有人约束的候选数。"
)

GATEKEEPER_SECRET = (
    "你是守门员 {name}。你的秘密验收规则是：只接受能被 {n} 整除的整数。"
    "这条规则只有你自己知道，绝不能向任何人直接说出或暗示。\n"
    "规则：\n"
    "1. 绝不要直接说出或暗示规则本身，例如不能说“能被 {n} 整除”“是 {n} 的倍数”“我的规则是…”等。\n"
    "2. 你必须严格验证每个候选数字：对每个新提出的数，实际计算它是否满足你的规则（如做除法验算），"
    "满足就“接受”，不满足必须“不接受”，即使其他人接受了也不能动摇。\n"
    "3. 你列出的接受数字只是举例，不代表只有这些数可接受——任何满足你规则的新数字都要接受。\n"
    "4. 你可以间接交流：通过列举你接受的数字（例如“我接受 {n}、{n2}、{n3} 这类数”）或拒绝的数字来传递信息，这不算透露规则。\n"
    "5. 你提出的候选数字必须是你私下验证过能满足自己规则的。\n"
    "6. 目标是找到一个所有成员都接受的整数——你的坚持与间接提示是达成正确共识的关键。"
)

SUMMARIZER = "你是记录员，负责汇总出一个能被 5、7、9 同时整除的整数（即 315 的倍数），供所有守门员批准。"
SUMMARIZER_SECRET = (
    "你是记录员。守门员们不能直接说出各自的验收规则，但会通过接受/拒绝具体数字间接传递信息。\n"
    "你的任务：\n"
    "1. 观察每位守门员接受和拒绝的数字，找出能同时被所有守门员接受的数（它们的交集）。\n"
    "2. 汇总出一个具体的整数，作为最终答案供守门员批准。"
)


def make_agent() -> Agent:
    return Agent(model=MODEL, base_url=BASE_URL, api_key=API_KEY or None)


def build_chatroom(secret: bool) -> Chatroom:
    if secret:
        m5 = Member(name="M5", agent=make_agent(), persona=GATEKEEPER_SECRET.format(name="M5", n=5, n2=10, n3=15))
        m7 = Member(name="M7", agent=make_agent(), persona=GATEKEEPER_SECRET.format(name="M7", n=7, n2=14, n3=21))
        m9 = Member(name="M9", agent=make_agent(), persona=GATEKEEPER_SECRET.format(name="M9", n=9, n2=18, n3=27))
    else:
        m5 = Member(name="M5", agent=make_agent(), persona=GATEKEEPER.format(name="M5", n=5))
        m7 = Member(name="M7", agent=make_agent(), persona=GATEKEEPER.format(name="M7", n=7))
        m9 = Member(name="M9", agent=make_agent(), persona=GATEKEEPER.format(name="M9", n=9))
    summarizer = Member(
        name="Summarizer",
        agent=make_agent(),
        persona=SUMMARIZER_SECRET if secret else SUMMARIZER,
    )
    return Chatroom(
        [m5, m7, m9],
        summarizer=summarizer,
        max_rounds=MAX_ROUNDS,
        max_summary_iters=MAX_SUMMARY_ITERS,
        secret=secret,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-agent consensus demo (5/7/9 -> 315)")
    parser.add_argument("--secret", action="store_true", help="agents must not reveal their rules")
    args = parser.parse_args()

    if not API_KEY:
        raise SystemExit("DEEPSEEK_API_KEY is not set")

    chatroom = build_chatroom(secret=args.secret)
    result = asyncio.run(chatroom.discuss("提出一个大家都能接受的正整数。"))

    print("\n===== 讨论记录 =====")
    for i, turn in enumerate(result.turns, 1):
        print(f"[{i}] {turn.speaker}: {turn.content[:300]}")

    print("\n===== 每轮投票 =====")
    for rv in result.votes:
        votes = ", ".join(f"{v.name}={v.vote}" for v in rv.votes)
        print(f"round {rv.round_no}: {votes} -> reached={rv.reached}")

    print("\n===== 汇总 =====")
    print(result.summary)

    print("\n===== 全员批准 =====")
    for a in result.approvals:
        print(f"{a.name}: {'APPROVED' if a.approved else 'REJECTED'} - {a.feedback[:200]}")

    # --- 强断言 ---
    tail = result.summary.split("最终答案")[-1] if "最终答案" in result.summary else result.summary
    numbers = [int(n) for n in re.findall(r"\d+", tail)]
    assert result.all_approved, f"FAIL: 未达成全员批准 {result.approvals}"
    assert any(n % 315 == 0 for n in numbers), f"FAIL: 最终答案无 315 的倍数: {tail}"
    print(f"\nPASS: 全员批准 + 汇总含 315 的倍数 {[n for n in numbers if n % 315 == 0]}")


if __name__ == "__main__":
    main()
