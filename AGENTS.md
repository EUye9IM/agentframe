# AGENTS.md

> **Keep this file in sync with the code.** If you change architecture, hooks, commands, test infra, dependencies, or implement something listed under "Repo-state gaps", update this file in the same change. Stale AGENTS.md misleads every future session.

AgentFrame v0.2: a **pure-sync**, hook-driven agent framework on LangGraph. Master branch is the active v0.2 rewrite; `dev` holds the frozen v0.1 monolith (tagged v0.1.0) — do not port dev code into master; reference it only for behavior ideas.

## Architecture (read `docs/ARCHITECTURE.md` first)

- `BaseAgent` (`agentframe/core/base.py`) = thin engine + hook protocol. No features built in; everything is a middleware.
- **LangGraph StateGraph is the state machine**: nodes `Phase.LLM` / `Phase.TOOLS`, conditional edge to `Phase.END`. No other nodes. `Phase` (`core/phases.py`) is a closed `StrEnum`; StrEnum members are `str` so they pass LangGraph's `isinstance(x, str)` checks directly (no `.value`).
- `Agent` (`agentframe/agent.py`) = public class. `middlewares=[...]` stacks middleware **instances** via dynamic class inheritance (`type(m)`), MRO order = list order = execution order. Factories (e.g. `tools(...)`) return classes, so call them: `middlewares=[tools([run_bash])()]`. Duplicate factory calls are safe (fresh closure class each time).
- **State is deliberately thin**: `AgentState = {messages}` only. No token counters, no error fields. Errors flow via LangGraph `NodeError` → `error_handler` → `handle_error(error, node) -> Command(goto=...)` (never through state/messages).
- Hooks (`core/hooks.py`, 15 sync hooks): `before_*`/`after_*` = transforms (return modified data), `on_*` = events (no return), `handle_next`/`handle_error` = flow. `after_llm` and `after_tool_result` return `list[BaseMessage]` — they own converting response→history, subclasses call `super()` then reorder.
- Streaming hooks (`on_llm_reasoning`/`on_llm_content`) are **pure events** (no bool). Interruption = raise `StreamStop`; `_act_llm` attaches `partial` and re-raises; the interrupter's own middleware `handle_error` override claims it. Base `handle_error` is uniform (`Command(goto=END)`), never type-switches.

## LLM layer

- `LLMClient` (`agentframe/llm/client.py`) is **raw httpx**, no openai SDK: struct-in/struct-out (`LLMRequest` → `LLMResponse`; `stream` yields `LLMStreamEvent`). SSE parsed manually.
- **Gotcha**: tool_calls format differs — OpenAI uses `arguments`, langchain `AIMessage` needs `args`. `BaseAgent._to_langchain_tool_calls` handles it; `_act_tools` reads `tc["args"]` (falls back to `arguments`). If you see "Unsupported message type" / duplicate-base / KeyError in the graph, it's usually a tool_calls format or middleware instance-vs-class bug.

## Commands

- Install: `uv sync` (or `.venv/bin/pip install -e ".[dev]"`). Python 3.12 (`.python-version`).
- Run all tests: `.venv/bin/python -m pytest tests/ -q` (currently 25, all pass).
- Run one test file: `.venv/bin/python -m pytest tests/test_errors.py -q`.
- **Do not use `basedpyright`**: its node binary fails to load in this environment (libstdc++ relocation errors). Use `.venv/bin/python -m compileall -q agentframe/ tests/` for syntax checks instead.
- Code style: **annotate all types** on public signatures (explicit function/parameter/return annotations), `from __future__ import annotations`, no comments unless they explain non-obvious design rationale.

## Tests (`tests/`, see `docs/TESTING.md`)

- No real network/LLM. `tests/conftest.py` provides `ScriptedLLMClient` (pre-scripted `LLMStreamEvent` sequences per call, `raise_at`/`exc` to inject failures), `RecordingAgent` (logs every hook call then `super()`s), and `make_agent` fixture. Helpers `content()`/`reasoning()`/`done()` build events.
- Hook overrides in tests are passed as `hooks={"before_llm": fn}` — plain functions bound to the instance (no `self`), NOT via `RecordingAgent` subclasses. The `hooks` dict overrides replace `RecordingAgent`'s logging for that hook.
- `_act_llm` accumulates content **before** calling `on_llm_content`, so a `StreamStop` raised in the hook includes the triggering chunk in `partial`.
- LLMClient's `stream` is always used by `_act_llm`; a fake client only needs `stream` implemented.

## Repo-state gaps (implementation in progress)

- `pyproject.toml` is **stale**: lists `openai` (unused — code uses `httpx`, add it to deps) and `afcli = agentframe.cli.main:main` (no `cli/` module yet). `mcp` dep declared but no MCP code yet.
- `middlewares/`, `multiagent/`, `compression/`, `cli/` not implemented yet — only `core/` and `llm/` exist. `examples/` and `docs/` exist.
- Design decisions deferred (do not implement unless asked): middleware custom states (Phase is closed), middleware-owned state channels (keep state thin), tool streaming hook `on_tool_stream`.
