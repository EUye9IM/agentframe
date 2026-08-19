# AGENTS.md

> **Keep this file in sync with the code.** If you change architecture, hooks, commands, test infra, dependencies, or implement something listed under "Repo-state gaps", update this file in the same change. Stale AGENTS.md misleads every future session.

AgentFrame v0.2: a **pure-sync**, hook-driven agent framework on LangGraph. Master branch is the active v0.2 rewrite; `dev` holds the frozen v0.1 monolith (tagged v0.1.0) — do not port dev code into master; reference it only for behavior ideas.

## Architecture (read `docs/ARCHITECTURE.md` first)

- `BaseAgent` (`agentframe/core/base.py`) = thin engine + hook protocol. No features built in; everything is a middleware.
- **LangGraph StateGraph is the state machine**: nodes `Phase.LLM` / `Phase.TOOLS`, conditional edge to `Phase.END`. No other nodes. `Phase` (`core/phases.py`) is a closed `StrEnum`; StrEnum members are `str` so they pass LangGraph's `isinstance(x, str)` checks directly (no `.value`).
- `Agent` (`agentframe/agent.py`) = public class. `middlewares=[...]` stacks middleware **instances** via dynamic class inheritance (`type(m)`), MRO order = list order = execution order. Factories (e.g. `tools(...)`) return classes, so call them: `middlewares=[tools([run_bash])()]`. Duplicate factory calls are safe (fresh closure class each time). Only the middleware **class** is copied — an instance's `__init__` state (e.g. `self._other`) is lost on the final agent, so capture config in the factory closure or class attrs, not instance `__init__`.
- **State is deliberately thin**: `AgentState = {messages}` only. No token counters, no error fields. Errors flow via LangGraph `NodeError` → `error_handler` → `handle_error(error, node) -> Command(goto=...)` (never through state/messages).
- **Persistence is built in (memory by default)**: `checkpointer` (default `InMemorySaver()`) and `session_id` (default `"default"`) are constructor params on `Agent`/`BaseAgent`. `invoke(input_text)` maps `session_id` → LangGraph `thread_id` internally, so the same session automatically resumes prior context (checkpointer restores state per thread — this **is** the multi-turn/multi-agent history mechanism). Escape hatch: reassign `agent.checkpointer` to any `BaseCheckpointSaver`. `session_id` is also the hook-carried key (`before_trace`/`after_trace`).
- Hooks (`core/hooks.py`, 14 sync hooks): `before_*`/`after_*` = transforms (return modified data), `on_*` = events (no return), `handle_next`/`handle_error` = flow. `after_llm`, `after_tool_result`, and `after_turn` return `list[BaseMessage]` — they own converting response/result/turn → history; subclasses call `super()` then reorder. `after_tool_result(name, result, tool_call_id)` gets the id explicitly (no instance-var plumbing). One turn = one LLM call (incl. its tool outputs): `before_turn` fires before each LLM call, `after_turn` fires once per turn — with tools it receives `[that turn's AIMessage, *ToolMessages]`, and its return value is written back to state.
- **Trace lifecycle**: `before_trace(messages, session_id)` / `after_trace(state, session_id)` bracket `invoke` (the **only** public entry — text in, `str` out). `invoke` first reads the thread's history from the checkpointer, then calls `before_trace([*history, *new])` where `new` = `[system?, human]` (system deduped by `id="system"` when history already has it). `before_trace` may **rewrite the whole conversation** (e.g. compress an oversized history into a summary, inject long-term memory); the contract requires it to keep the user's `HumanMessage` as the last message (else `ValueError`). `invoke` then hoists the `id="system"` message to the front (`_hoist_system`); if the hook rewrote the list it seeds the graph via `update_state` (`RemoveMessage` all history + `start[:-1]`) and runs with input `[start[-1]]`, else it skips seeding and runs the graph's native restore path with `[fresh]` — so the rewrite is **persisted** and later turns resume from the rewritten conversation. `after_trace` returns the **last AI message's** content (empty string if none — a first-turn failure must not echo the user's own input back).
- Streaming hooks (`on_llm_reasoning`/`on_llm_content`) are **pure events** (no bool). Interruption = raise `StreamStop`; `_act_llm` attaches `partial` (+ `partial_reasoning`) and re-raises; the interrupter's own middleware `handle_error` override claims it. Base `handle_error` is uniform (`Command(goto=END)`), never type-switches.

## LLM layer

- `LLMClient` (`agentframe/llm/client.py`) is **raw httpx**, no openai SDK: struct-in/struct-out (`LLMRequest` → `LLMResponse`; `stream` yields `LLMStreamEvent`). SSE parsed manually. The client owns `model` (it's the model-bound endpoint); `LLMRequest` carries no model, and `_act_llm` fills `LLMResponse.model` from `self.llm_client.model`. Streaming `done` event carries aggregated `tool_calls` + `usage` + `finish_reason`; reasoning deltas accept `reasoning_content`/`reasoning`/`reasoning_text`.
- **Gotcha**: tool_calls format differs — OpenAI uses `arguments`, langchain `AIMessage` needs `args`. `BaseAgent._to_langchain_tool_calls` handles it; `_act_tools` reads `tc["args"]`. If you see "Unsupported message type" / duplicate-base / KeyError in the graph, it's usually a tool_calls format or middleware instance-vs-class bug.

## Commands

- Install: `uv sync --extra dev`. Python 3.12 (`.python-version`).
- Run all tests: `.venv/bin/python -m pytest tests/ -q` (currently 68, all pass).
- Run one test file: `.venv/bin/python -m pytest tests/test_errors.py -q`.
- Type check: `.venv/bin/basedpyright agentframe/ tests/` (installed by `uv sync --extra dev`). Rule calibration lives in `pyproject.toml [tool.basedpyright]` (noise rules off; `tests/` via `executionEnvironments` keeps unknown-* rules off). Syntax-only fallback: `.venv/bin/python -m compileall -q agentframe/ tests/`.
- Code style: **annotate all types** on public signatures (explicit function/parameter/return annotations), `from __future__ import annotations`, no comments unless they explain non-obvious design rationale.

## Tests (`tests/`, see `docs/TESTING.md`)

- No real network/LLM. `tests/conftest.py` provides `ScriptedLLMClient` (pre-scripted `LLMStreamEvent` sequences per call, `raise_at`/`exc` to inject failures), `RecordingAgent` (logs every hook call then `super()`s), and `make_agent` fixture. Helpers `content()`/`reasoning()`/`done()` build events.
- **Every code change that is testable and worth testing must ship with a regression test in the same commit.** Bug fixes get a test that fails before the fix and passes after; behavior changes get a test pinning the new semantics.
- Hook overrides in tests are passed as `hooks={"before_llm": fn}` — plain functions bound to the instance (no `self`), NOT via `RecordingAgent` subclasses. The `hooks` dict overrides replace `RecordingAgent`'s logging for that hook.
- `_act_llm` accumulates content **before** calling `on_llm_content`, so a `StreamStop` raised in the hook includes the triggering chunk in `partial`.
- LLMClient's `stream` is always used by `_act_llm`; a fake client needs a `model` attribute + `stream` implemented.

## Repo-state gaps (implementation in progress)

- `pyproject.toml` deps are aligned with code (`httpx`, no `openai`/`mcp`); `afcli` entry point removed until `cli/` exists.
- `middlewares/` exists with `log(logger)` (standard-logging observability factory), `tools(functions)` (register + inject OpenAI function schemas, with `register`/`unregister` for dynamic add/remove), and `compress(*, keep_recent=4)` (truncation-based history compression — keeps system prompt + recent N messages, removes older ones via `RemoveMessage`, no summarizer); `multiagent/`, `cli/` not implemented yet. `examples/` and `docs/` exist.
- Design decisions deferred (do not implement unless asked): middleware custom states (Phase is closed), middleware-owned state channels (keep state thin), tool streaming hook `on_tool_stream`.
