# agentframe — AGENTS.md

## Environment & Setup

```bash
# Python 3.12+, uv-managed venv
uv venv --python 3.12
uv pip install -e ".[dev]"
```

Package manager is **uv**, not pip. The `uv.lock` file is checked in and authoritative for dependency versions.

## Run / Verify

```bash
# All tests (mock-based, no API key needed) — 107 tests
.venv/bin/python -m pytest tests/ -v

# CLI (interactive)
.venv/bin/python -m agentframe
# or after `uv pip install -e .`:
afcli
```

## Architecture

```
agentframe/
  core/agent.py    # Agent class + 4 hooks + LangGraph StateGraph + MCP tools/prompts + invoke_messages
  llm/client.py    # openai wrapper: invoke, ainvoke, stream, astream
  tools/           # function_tool decorator, ToolRegistry, MCP client (tools + prompts)
  compression/     # Token-threshold summarization (Compressor: compress + acompress)
  memory/hooks.py  # doc-only: users pass langgraph BaseCheckpointSaver
  multiagent/      # Chatroom orchestrator: Member, Chatroom, round-robin + votes + summary approval
  cli/             # ChatAgent, ~/.afcli.toml config, UTF-8 input
tests/             # 107 tests across 9 files
examples/          # chatroom_315.py — real-LLM multi-agent consensus demo (needs DEEPSEEK_API_KEY)
```

The CLI does NOT use the LangGraph `StateGraph`. It calls `agent.ainvoke()` which routes through `_acall_agent` → `llm_client.astream` → hooks fire per token. The graph is for library users calling `invoke()`/`ainvoke()`/`stream()`/`astream()`.

### Agent internal flow

**Sync path**: `invoke()` → `_build_graph()` → `_call_agent` (uses `llm_client.invoke`) → `_should_continue` → `_call_tools` → loop

**Async path**: `ainvoke()` → `_abuild_graph()` → `_acall_agent` (uses `llm_client.astream` + hooks) → `_ashould_continue` → `_acall_tools` → loop

Shared helpers reduce sync/async duplication:
- `_build_graph_impl(agent_node, tools_node, should_continue_fn)` — graphs for both paths
- `_prepare_agent_state(state)` / `_prepare_agent_state_async(state)` — compression + tool list (both merge MCP tools when `mcp_configs`/`_mcp_clients` set)
- `_process_tool_calls(messages, last_message)` — approved_ids + rejection messages

**MCP routing (both paths)**: `_call_tools` / `_acall_tools` first check `FunctionTool` in `tool_registry`, then fall back to MCP clients (`_call_mcp_tool_sync` / `_call_mcp_tool`). MCP is supported in both sync and async paths — sync uses `MCPClient.connect_sync()`/`call_tool_sync()`, which bridge the async MCP SDK onto a persistent background event loop.

### Critical: `_acall_agent` event handling

The async path MUST extract `tool_calls` from the `"done"` stream event. Both sync and async `_call_tools` MUST call `on_tool_result`. Do NOT remove either — tests catch this.

### `invoke_messages` / `ainvoke_messages`

Public methods that start the graph from an explicit message list, bypassing `_build_input` (no system-prompt injection — the caller controls all messages). Used by the multiagent chatroom to feed each member its persona + the shared transcript. Compression/tools/MCP still apply. Without a `checkpointer` the call is stateless; with `session_id` messages merge into existing history via `add_messages`.

## Multiagent Chatroom

`agentframe/multiagent/` is a stateless orchestrator: the `Chatroom` owns the transcript, each `Member` is a plain `Agent` treated as a pure compute unit.

Flow: **discussion** (round-robin, members may reply `PASS`) → **vote** after each round (`VOTE:APPROVE|ABSTAIN|DISAGREE`, consensus when ≥1 APPROVE and 0 DISAGREE) → **summary** (summarizer drafts) → **approval loop** (`APPROVED`/`PASS` accept, otherwise summarizer revises, capped by `max_summary_iters`).

- `chatroom.discuss(topic) -> ChatroomResult` (turns, votes, summary, approvals, `all_approved`)
- `chatroom.stream_discussion(topic) -> AsyncIterator[Event]` for live UI rendering
- **Persona rule**: the agent's own `system_prompt` is bypassed by `ainvoke_messages`; the chatroom injects `Member.persona` (fallback: `agent.system_prompt`). Put strong instructions in `Member.persona`.
- Review parsing is tolerant: `PASS`/abstain counts as acceptance; unknown votes default to `ABSTAIN`.

`examples/chatroom_315.py` is a real-LLM demo: three gatekeepers accept only multiples of 5/7/9 and must reach consensus on a number (any valid answer is a multiple of 315). Run with `DEEPSEEK_API_KEY=... python examples/chatroom_315.py`.

Add `--secret` to forbid agents from DIRECTLY revealing their rules. The persona still tells each agent its own rule privately and requires strict verification, while allowing **indirect communication** via accepted-number lists ("我接受 5、10、15 这类数" — listing examples is not direct disclosure). Findings from real runs:
- Naive secret prompt (no rule, no strictness) → members stop verifying and conform socially, converging on wrong 42/12.
- Correct secret prompt → members hold positions (long DISAGREE streaks), communicate indirectly, negotiate honestly for 20+ rounds, but LLM arithmetic is inconsistent: a member can accept a number violating its own rule (e.g. M5 accepting 126, not a multiple of 5). No prompt fully fixes model arithmetic drift; a private `FunctionTool` checker per member would, but that's beyond the current demo.

`Chatroom(secret=True)` adjusts vote/review prompts to forbid rule-revealing ("does NOT reveal your secret rule") and allow indirect hints.

## Agent Hooks (override in subclass)

| Hook | Signature | Fires when |
|------|-----------|-----------|
| `on_llm_reasoning` | `(text: str)` | each reasoning chunk during streaming |
| `on_llm_content` | `(text: str)` | each content chunk during streaming |
| `on_tool_call` | `(tool_calls: list[dict]) -> list[dict]` | before tool execution, return approved subset |
| `on_tool_result` | `(name: str, result: str)` | after each tool execution |

Default implementations are no-ops. All four fire in both sync and async paths.

## Testing

### Patch rules

- **Agent tests**: `patch.object(agent.llm_client, "invoke", ...)` or `patch.object(agent.llm_client, "astream", ...)`. Do NOT patch `openai` module-level — always patch on the instance.
- **LLMClient tests**: `patch.object(client, "_get_client")` / `patch.object(client, "_get_aclient")` to mock the inner `openai` client.

### Mocking `astream` for async tests

`astream` returns an async generator. Mock with a closure that yields events per call:

```python
def _make_astream(*event_lists):
    calls = iter(event_lists)
    async def mock(messages, tools=None):
        for event in next(calls):
            yield event
    return mock
```

Conftest provides `make_response()` and `make_tool_call()` helpers. Both async and sync paths have test coverage (`test_agent.py` + `test_agent_async.py`).

### Mocking MCP for agent tests

Inject mock clients directly — no real MCP server needed:

```python
from tests.conftest import make_mock_mcp_tool, make_mock_mcp_client

mcp_client = make_mock_mcp_client(
    tools=[make_mock_mcp_tool("search")],
    prompts={"greeting": "You are helpful"},
)
agent._mcp_clients = [mcp_client]   # inject; patch agent._ensure_mcp_connected

# sync path: set call_tool_sync, async path: set call_tool (AsyncMock)
mcp_client.call_tool_sync = Mock(return_value="result")
```

`MCPClient` unit tests (`tests/test_mcp_client.py`) mock the SDK: `patch("mcp.client.stdio.stdio_client")` / `patch("mcp.client.sse.sse_client")` / `patch("mcp.ClientSession")`. The sync bridge tests exercise `connect_sync()`/`call_tool_sync()` which run on a background thread loop.

## Linting

代码修改完成后，运行基于 pyright 检查：

```bash
.venv/bin/basedpyright agentframe/
```

- 可修复的告警/报错直接修复
- 修复代价大或无法修复的，在行末加 `# type: ignore[<error-code>]` 屏蔽（如 `# type: ignore[arg-type]`、`# type: ignore[reportMissingImports]`）
- 防御性代码（如类型标注保证安全但仍保留的运行时检查）导致的告警也可用 `# type: ignore[unreachable]` 屏蔽
- 可选依赖导致的导入错误使用 `# type: ignore[reportMissingImports]` 屏蔽

## Known Quirks

- **LSP import errors**: LSP runs outside the venv, so `langchain_core`/`openai`/`langgraph` imports show as unresolved. Ignore them.
- **`base_url` support**: `LLMClient` accepts `base_url` for custom endpoints (Ollama, vLLM, etc.). Pass it to `Agent` and it flows through to `openai.OpenAI(base_url=...)`.
- **basedpyright false positive**: `_build_graph_impl` parameter types (`Callable[[AgentState], dict]`) cause a false-positive error on `workflow.add_node("agent", agent_node)` because LangGraph's `StateNode` Protocol expects `state` as a keyword param. Code runs correctly; this is a LangGraph type-stub limitation.
- **Config file**: `~/.afcli.toml` auto-created on first CLI run. `api_key` defaults to `""` — set it or use `LLM_AUTH_KEY` / `OPENAI_API_KEY` env var.
- **Session persistence**: pass `session_id="name"` to `invoke()`/`ainvoke()`. Must also pass a `checkpointer` (e.g. `MemorySaver` or `SqliteSaver`) when constructing Agent.
- **CLI builtin tools**: The CLI auto-registers `bash` tool (`agentframe/tools/builtin/bash.py`).
- **MCP connection caching**: `_ensure_mcp_connected()` (async) / `_ensure_mcp_connected_sync()` lazily connects and reuses MCP clients. Call `agent.aclose_mcp()` (async) or `agent.close_mcp_sync()` to clean up subprocesses.
- **`mcp_prompt` parameter**: pass `mcp_prompt="prompt_name"` (or `"server_name:prompt_name"`) to use an MCP server prompt template as the system prompt. Resolved once on first `ainvoke`/`astream`; falls back to `system_prompt` if not found. MCP servers without a `prompts` capability are handled gracefully.
- **Sync bridge is thread-based**: `MCPClient.connect_sync()`/`call_tool_sync()` run the async MCP SDK on a persistent daemon-thread event loop (`run_coroutine_threadsafe`). Connections survive across sync calls, and it's safe to call from within a running event loop. Do NOT add an `asyncio.run()` wrapper — it would break persistent MCP connections and raise `RuntimeError` when called from a running loop.
- **Dict tools in ToolRegistry**: registering a `dict` tool without `function.name` raises `ValueError` (no silent fallback).
- **Compressor dual path**: `Compressor.compress()` is sync, `Compressor.acompress()` is async. The async agent path uses `acompress` to avoid blocking the event loop.

<!-- CODEGRAPH_START -->
## CodeGraph

In repositories indexed by CodeGraph (a `.codegraph/` directory exists at the repo root), reach for it BEFORE grep/find or reading files when you need to understand or locate code:

- **MCP tool** (when available): `codegraph_explore` answers most code questions in one call — the relevant symbols' verbatim source plus the call paths between them, including dynamic-dispatch hops grep can't follow. Name a file or symbol in the query to read its current line-numbered source. If it's listed but deferred, load it by name via tool search.
- **Shell** (always works): `codegraph explore "<symbol names or question>"` prints the same output.

If there is no `.codegraph/` directory, skip CodeGraph entirely — indexing is the user's decision.
<!-- CODEGRAPH_END -->
