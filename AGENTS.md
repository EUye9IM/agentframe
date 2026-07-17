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
# All tests (mock-based, no API key needed) — 55 tests
.venv/bin/python -m pytest tests/ -v

# CLI (interactive)
.venv/bin/python -m agentframe
# or after `uv pip install -e .`:
afcli
```

## Architecture

```
agentframe/
  core/agent.py    # Agent class + 4 hooks + LangGraph StateGraph
  llm/client.py    # litellm wrapper: invoke, ainvoke, stream, astream
  tools/           # function_tool decorator, ToolRegistry, MCP client
  compression/     # Token-threshold summarization (Compressor: compress + acompress)
  memory/hooks.py  # doc-only: users pass langgraph BaseCheckpointSaver
  cli/             # ChatAgent, ~/.afcli.toml config, UTF-8 input
tests/             # 55 tests: test_agent.py, test_agent_async.py, test_tools.py, etc.
```

The CLI does NOT use the LangGraph `StateGraph`. It calls `agent.ainvoke()` which routes through `_acall_agent` → `llm_client.astream` → hooks fire per token. The graph is for library users calling `invoke()`/`ainvoke()`/`stream()`/`astream()`.

### Agent internal flow

**Sync path**: `invoke()` → `_build_graph()` → `_call_agent` (uses `llm_client.invoke`) → `_should_continue` → `_call_tools` → loop

**Async path**: `ainvoke()` → `_abuild_graph()` → `_acall_agent` (uses `llm_client.astream` + hooks) → `_ashould_continue` → `_acall_tools` → loop

Shared helpers reduce sync/async duplication:
- `_build_graph_impl(agent_node, tools_node, should_continue_fn)` — graphs for both paths
- `_prepare_agent_state(state)` / `_prepare_agent_state_async(state)` — compression + tool list
- `_process_tool_calls(messages, last_message)` — approved_ids + rejection messages

### Critical: `_acall_agent` event handling

The async path MUST extract `tool_calls` from the `"done"` stream event. Both sync and async `_call_tools` MUST call `on_tool_result`. Do NOT remove either — tests catch this.

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

- **Agent tests**: `patch.object(agent.llm_client, "invoke", ...)` or `patch.object(agent.llm_client, "astream", ...)`. Do NOT use `patch("litellm.completion")` — imported symbol aliasing breaks patching.
- **LLMClient tests**: patch `"agentframe.llm.client.completion"` (the bound module-level reference).

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

## Known Quirks

- **litellm import is slow (4–5s)**: one-time cost paid at CLI startup. Pytest sessions also pay this cost.
- **LSP import errors**: LSP runs outside the venv, so `langchain_core`/`litellm`/`langgraph` imports show as unresolved. Ignore them.
- **`fastapi` is a core dependency**: litellm 1.92.0 imports `fastapi` internally when tools are passed to `completion()`. Do not remove it from pyproject.toml.
- **Config file**: `~/.afcli.toml` auto-created on first CLI run. `api_key` defaults to `""` — set it or use `LLM_AUTH_KEY` / `OPENAI_API_KEY` env var.
- **Session persistence**: pass `session_id="name"` to `invoke()`/`ainvoke()`. Must also pass a `checkpointer` (e.g. `MemorySaver` or `SqliteSaver`) when constructing Agent.
- **CLI builtin tools**: The CLI auto-registers `bash` tool (`agentframe/tools/builtin/bash.py`).
- **MCP connection caching**: `_ensure_mcp_connected()` lazily connects and reuses MCP clients. Call `agent.aclose_mcp()` to clean up subprocesses.
- **Dict tools in ToolRegistry**: registering a `dict` tool without `function.name` raises `ValueError` (no silent fallback).
- **Compressor dual path**: `Compressor.compress()` is sync, `Compressor.acompress()` is async. The async agent path uses `acompress` to avoid blocking the event loop.

<!-- CODEGRAPH_START -->
## CodeGraph

In repositories indexed by CodeGraph (a `.codegraph/` directory exists at the repo root), reach for it BEFORE grep/find or reading files when you need to understand or locate code:

- **MCP tool** (when available): `codegraph_explore` answers most code questions in one call — the relevant symbols' verbatim source plus the call paths between them, including dynamic-dispatch hops grep can't follow. Name a file or symbol in the query to read its current line-numbered source. If it's listed but deferred, load it by name via tool search.
- **Shell** (always works): `codegraph explore "<symbol names or question>"` prints the same output.

If there is no `.codegraph/` directory, skip CodeGraph entirely — indexing is the user's decision.
<!-- CODEGRAPH_END -->
