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
# All tests (mock-based, no API key needed)
.venv/bin/python -m pytest tests/ -v

# CLI (interactive)
.venv/bin/python -m agentframe
# or after `uv pip install -e .`:
afcli
```

## Architecture — What Matters

```
agentframe/
  core/agent.py    # Agent class + 4 hooks + LangGraph StateGraph
  llm/client.py    # litellm wrapper (completion, stream, astream)
  tools/           # function_tool decorator, ToolRegistry, MCP client
  compression/     # Token-threshold summarization (Compressor)
  memory/hooks.py  # doc-only: users pass langgraph BaseCheckpointSaver
  cli/             # ChatAgent subclass, ~/.afcli.toml config, UTF-8 input
tests/             # 42 tests, pytest + pytest-asyncio
```

CLI does NOT use the LangGraph `StateGraph`. It uses Agent's hooks (`ainvoke` → `_acall_agent` → streaming via `llm_client.astream` → hooks fire per token). The graph is for library users calling `agent.invoke()` / `agent.ainvoke()`.

## Agent Hooks (override in subclass)

| Hook | Signature | Fires when |
|------|-----------|-----------|
| `on_llm_reasoning` | `(text: str)` | each reasoning chunk during streaming |
| `on_llm_content` | `(text: str)` | each content chunk during streaming |
| `on_tool_call` | `(tool_calls: list[dict]) -> list[dict]` | before tool execution, return approved subset |
| `on_tool_result` | `(name: str, result: str)` | after each tool execution |

Default implementations are no-ops. The CLI's `ChatAgent` overrides them.

## Testing

Tests mock with `unittest.mock.patch.object(agent.llm_client, "invoke", ...)`. Do NOT use `patch("litellm.completion")` — imported symbol aliasing breaks patching.

For LLMClient tests, patch `"agentframe.llm.client.completion"` (the bound module-level reference).

Conftest provides `make_response()` and `make_tool_call()` helpers.

## Known Quirks

- **litellm import is slow (4–5s)**: one-time cost paid at CLI startup (`main_loop()` does `import litellm` before showing the prompt). Pytest sessions also pay this cost.
- **LSP import errors**: LSP runs outside the venv, so all `langchain_core`/`litellm`/`langgraph` imports show as unresolved. Ignore them.
- **Tool calling needs `fastapi`**: litellm 1.92.0 imports `fastapi` when tools are passed (happens inside its `completion()` call). If missing, install manually: `uv pip install fastapi`. Not in project dependencies.
- **Config file**: `~/.afcli.toml` auto-created on first CLI run. `api_key` field defaults to `""` — set it or use `LLM_AUTH_KEY` / `OPENAI_API_KEY` env var.
- **Session persistence**: pass `session_id="name"` to `invoke()`/`ainvoke()`. Must also pass a `checkpointer` (e.g. `MemorySaver` or `SqliteSaver`) when constructing Agent.
- **`.pytest_cache`** and `**/__pycache__` are git-ignored.
