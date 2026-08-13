# AgentFrame v0.2 代码评审

> 日期：2026-08-13
> 范围：master 分支现有代码（`agentframe/core/`、`agentframe/llm/`、`agentframe/agent.py`、`tests/`）
> 基线：25 个测试全过，basedpyright 0 error / 0 warning。

## 结论

核心设计（极薄引擎 + 钩子协议 + LangGraph 状态机 + 结构体交换）实现干净、一致，类型注解强化后的代码质量良好。问题集中在**包元数据失配、流式解析丢失信息、错误路径语义、测试覆盖缺口**四类。均已按下列清单处理（✅ 已修）。

## A. 必修

| # | 问题 | 位置 | 状态 |
|---|------|------|------|
| A1 | `pyproject.toml` 与代码脱节：用了 `httpx` 未声明，声明了未用的 `openai`；`afcli` 入口指向不存在的 `agentframe.cli.main` | pyproject.toml | ✅ 对齐依赖，删 `afcli` 入口 |
| A2 | 流式 `usage` / `finish_reason` 丢失：末帧（无 choices、带 usage）被丢弃，`done` 事件不填 usage | llm/client.py | ✅ `stream()` 捕获 usage/finish_reason 并入 `done` 事件；`_act_llm` 透传到 `LLMResponse` |
| A3 | `_finish_tool_calls` 的 `json.loads` 无保护，截断的 tool_calls 会抛 `ValueError` | llm/client.py | ✅ `_safe_parse_arguments` 兜底（流式 + invoke 双路径） |
| A4 | （评审后发现）`_parse_completion_response` 用 `arguments` 键构造 `AIMessage.tool_calls`，langchain-core 1.4.9 需 `args` → 非流式 `invoke()` 的 tool_calls 直接 TypeError | llm/client.py | ✅ 改产 `args` 格式，与 `_to_langchain_tool_calls` 一致 |

## B. 健壮性（建议修）

| # | 问题 | 位置 | 状态 |
|---|------|------|------|
| B1 | LLM 首轮失败时 `invoke` 回显用户输入；`invoke_messages([])` IndexError | core/base.py | ✅ `after_trace` 取最后一条 AI 消息，无则空串；`invoke_messages` 走 `after_trace` |
| B2 | `LLMClient` 无 `close()` / 上下文管理器，httpx 连接池不释放 | llm/client.py | ✅ 增加 `close()` + `__enter__`/`__exit__`，并支持注入 `transport` 以便测试 |
| B3 | 推理字段只认 `reasoning_content`（DeepSeek 专属） | llm/client.py | ✅ 兼容 `reasoning`/`reasoning_text` |
| B4 | `StreamStop` 中断时 `partial` 不含 reasoning | core/base.py | ✅ 新增 `partial_reasoning`，`_act_llm` 一并挂载 |

## C. API 一致性

| # | 问题 | 位置 | 状态 |
|---|------|------|------|
| C1 | `invoke` 走 trace 钩子，`stream`/`invoke_messages` 绕过且 `session_id` 形同虚设 | core/base.py | ✅ `invoke_messages` 出口走 `after_trace`；`stream` 明确文档化为图级调试 API（绕过 trace） |
| C2 | `stream()` 返回 LangGraph 原始 state dict，与 `on_llm_content` 两套流式通道 | core/base.py | ✅ 文档化，`on_llm_*` 事件钩子为主流式通道 |
| C3 | README 的 `from agentframe.middlewares import tools, memory` import 即挂 | README.md | ✅ 标注为规划 API（模块未实现） |

## D. 可维护性

| # | 问题 | 位置 | 状态 |
|---|------|------|------|
| D1 | `LLMStreamEvent.type` 是裸 `str` | llm/types.py | ✅ 改 `Literal["reasoning","content","done"]` |
| D2 | `_act_llm` 手工累计可收敛为 accumulator | core/base.py | 保持现状（已随 A2 增加 usage/finish_reason 透传，结构清晰） |
| D3 | `_copy_state` 浅拷贝语义未显式说明 | core/base.py | ✅ 加注释说明共享 list 为刻意取舍 |

## E. 测试缺口

| # | 缺口 | 状态 |
|---|------|------|
| E1 | `LLMClient` 无测试 | ✅ 新增 `tests/test_llm_client.py`（invoke 解析 / SSE 流式 / usage/finish_reason / tool_calls 聚合与截断 / lifecycle），并借此发现并修复 A4 |
| E2 | `stream()` 无测试 | ✅ LLMClient 层覆盖；图级 `stream()` 保留为调试 API |
| E3 | 错误路径回显行为无固化 | ✅ `test_first_turn_failure_does_not_echo_user_input` + `test_streamstop_keeps_partial_reasoning` |

## 修复后基线

- 37 个测试全过；basedpyright 0 error。
