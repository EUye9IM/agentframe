from __future__ import annotations

import asyncio
import json

from agentframe import Agent
from agentframe.cli.config import load_config


class ChatAgent(Agent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # silence the inherited invoke/stream — we only chat
        self._chat_history: list = []
        self._reasoning_printed = False

    # ---- hooks ----

    def on_llm_reasoning(self, text: str) -> None:
        if not self._reasoning_printed:
            print("  ── reasoning ──")
            self._reasoning_printed = True
        print(f"  {text}", end="", flush=True)

    def on_llm_content(self, text: str) -> None:
        print(text, end="", flush=True)

    def on_tool_call(self, tool_calls: list[dict]) -> list[dict]:
        if not tool_calls:
            return tool_calls
        print()
        print("  ── tool calls ──")
        for i, tc in enumerate(tool_calls):
            args_str = json.dumps(tc["args"], ensure_ascii=False)
            print(f"  [{i + 1}] {tc['name']}({args_str})")
        print("  ────────────────")
        while True:
            choice = input("  Execute? [y/n/i]: ").strip().lower()
            if choice == "y":
                return tool_calls
            if choice == "n":
                return []
            if choice == "i":
                approved = []
                for i, tc in enumerate(tool_calls):
                    args_str = json.dumps(tc["args"], ensure_ascii=False)
                    c = input(f"  [{i + 1}] {tc['name']}({args_str})? [y/n]: ").strip().lower()
                    if c == "y":
                        approved.append(tc)
                return approved

    def on_tool_result(self, name: str, result: str) -> None:
        print(f"  [{name}] {result[:300]}")

    # ---- chat loop ----

    async def chat(self, user_input: str, session_id: str | None = None) -> str:
        self._reasoning_printed = False
        return await self.ainvoke(user_input, session_id=session_id)

    @classmethod
    def from_config(cls, cfg: dict) -> ChatAgent:
        return cls(
            model=cfg["model"],
            system_prompt=cfg["system_prompt"],
            api_key=cfg["api_key"],
            compress_threshold=cfg["compress_threshold"],
        )


async def main_loop(agent: ChatAgent, session_id: str | None = None) -> None:
    print(f"  model: {agent.model}")
    print(f"  type /quit to exit\n")

    while True:
        try:
            user_input = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        text = user_input.strip()
        if not text:
            continue
        if text in ("/quit", "/exit"):
            break

        print()
        try:
            await agent.chat(text, session_id=session_id)
        except Exception as e:
            print(f"\n  [error] {e}")
        print("\n")


def main() -> None:
    cfg = load_config()
    agent = ChatAgent.from_config(cfg)
    asyncio.run(main_loop(agent))
