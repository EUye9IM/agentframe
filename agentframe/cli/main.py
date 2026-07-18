from __future__ import annotations

import asyncio
import json
import re
import sys
import termios
import tty


# ---------------------------------------------------------------------------
# Custom UTF‑8 aware input (Chinese backspace fix)
# ---------------------------------------------------------------------------

def _clean_reasoning(text: str) -> str:
    text = re.sub(r"  +", " ", text)
    text = re.sub(r"(?<=[\u4e00-\u9fff]) (?=[\u4e00-\u9fff])", "", text)
    text = re.sub(r"(?<=[\u4e00-\u9fff]) ", "", text)
    text = re.sub(r" (?=[\u4e00-\u9fff])", "", text)
    return text.strip()


def _utf8_input(prompt: str = "") -> str:
    """Read a line from stdin with proper UTF‑8 backspace handling."""
    if not sys.stdin.isatty():
        return input(prompt)

    if prompt:
        sys.stdout.write(prompt)
        sys.stdout.flush()

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    chars: list[bytes] = []
    try:
        tty.setraw(fd)
        while True:
            ch = sys.stdin.buffer.read(1)
            if ch in (b"\r", b"\n"):
                sys.stdout.write("\r\n")
                break
            if ch == b"\x03":  # Ctrl+C
                raise KeyboardInterrupt
            if ch == b"\x04":  # Ctrl+D
                raise EOFError
            if ch in (b"\x7f", b"\x08"):  # Backspace
                if chars:
                    chars.pop()
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
            else:
                chars.append(ch)
                sys.stdout.buffer.write(ch)
                sys.stdout.flush()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

    return b"".join(chars).decode("utf-8", errors="replace").strip()


# ---------------------------------------------------------------------------
# ChatAgent
# ---------------------------------------------------------------------------

class ChatAgent:

    def __init__(self, model: str, *, base_url: str = "", system_prompt: str = "", api_key: str = "", compress_threshold: int = 100000):
        self._model = model
        self._base_url = base_url
        self._system_prompt = system_prompt
        self._api_key = api_key
        self._compress_threshold = compress_threshold
        self._agent = None
        self._reasoning_buf: list[str] = []

    @property
    def model(self) -> str:
        return self._model

    def _ensure_agent(self):
        if self._agent is not None:
            return
        from agentframe import Agent as _Agent
        from agentframe.tools.builtin.bash import run_bash

        self._agent = _Agent(
            model=self._model,
            system_prompt=self._system_prompt or None,
            api_key=self._api_key or None,
            base_url=self._base_url or None,
            compress_threshold=self._compress_threshold,
            tools=[run_bash],
        )
        self._agent.on_llm_reasoning = self._on_reasoning
        self._agent.on_llm_content = self._on_content
        self._agent.on_tool_call = self._on_tool_call
        self._agent.on_tool_result = self._on_tool_result

    # ---- hooks ----

    def _on_reasoning(self, text: str) -> None:
        self._reasoning_buf.append(text)

    def _on_content(self, text: str) -> None:
        if self._reasoning_buf:
            self._flush_reasoning(text)
        print(text, end="", flush=True)

    def _flush_reasoning(self, first_content: str) -> None:
        raw = "".join(self._reasoning_buf)
        self._reasoning_buf = []
        cleaned = _clean_reasoning(raw)
        if not cleaned:
            return
        if cleaned and cleaned in first_content.strip():
            return
        print()
        print(f"\033[2m{cleaned[:600]}\033[0m")
        print()

    def _on_tool_call(self, tool_calls: list[dict]) -> list[dict]:
        if not tool_calls:
            return tool_calls
        print()
        print("  ── tool calls ──")
        for i, tc in enumerate(tool_calls):
            args_str = json.dumps(tc["args"], ensure_ascii=False)
            print(f"  [{i + 1}] {tc['name']}({args_str})")
        print("  ────────────────")
        while True:
            choice = _utf8_input("  Execute? [y/n/i]: ").lower()
            if choice == "y":
                return tool_calls
            if choice == "n":
                return []
            if choice == "i":
                approved = []
                for i, tc in enumerate(tool_calls):
                    args_str = json.dumps(tc["args"], ensure_ascii=False)
                    c = _utf8_input(f"  [{i + 1}] {tc['name']}({args_str})? [y/n]: ").lower()
                    if c == "y":
                        approved.append(tc)
                return approved

    def _on_tool_result(self, name: str, result: str) -> None:
        print(f"  [{name}] {result[:300]}")

    # ---- chat ----

    async def chat(self, user_input: str, session_id: str | None = None) -> str:
        self._ensure_agent()
        self._reasoning_buf = []
        return await self._agent.ainvoke(user_input, session_id=session_id)

    @classmethod
    def from_config(cls, cfg: dict) -> ChatAgent:
        return cls(
            model=cfg["model"],
            base_url=cfg.get("base_url", ""),
            system_prompt=cfg["system_prompt"],
            api_key=cfg["api_key"],
            compress_threshold=cfg["compress_threshold"],
        )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def main_loop(agent: ChatAgent) -> None:
    print()
    print(f"  model: {agent.model}")
    print(f"  /quit to exit")
    print()

    while True:
        try:
            user_input = _utf8_input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input in ("/quit", "/exit"):
            break

        try:
            result = await agent.chat(user_input)
        except Exception as e:
            print(f"\n  [error] {e}")
            continue

        if result:
            print()


def main() -> None:
    from agentframe.cli.config import load_config

    cfg = load_config()
    agent = ChatAgent.from_config(cfg)
    asyncio.run(main_loop(agent))
