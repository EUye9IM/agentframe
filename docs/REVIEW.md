# AgentFrame v0.2 代码评审

> 日期：2026-08-13
> 范围：master 分支现有代码（`agentframe/core/`、`agentframe/llm/`、`agentframe/agent.py`、`tests/`）
> 基线：25 个测试全过，basedpyright 0 error / 0 warning。

## 结论

核心设计（极薄引擎 + 钩子协议 + LangGraph 状态机 + 结构体交换）实现干净、一致，类型注解强化后的代码质量良好。没有发现会导致核心流程崩溃的缺陷；问题集中在**包元数据失配、流式解析丢失信息、错误路径语义、测试覆盖缺口**四类。建议按「A 必修 / B 建议修 / C 一致性 / D 可维护性 / E 测试缺口」顺序处理。

## A. 必修

| # | 问题 | 位置 | 影响 |
|---|------|------|------|
| A1 | `pyproject.toml` 与代码脱节：用了 `httpx` 未声明，声明了未用的 `openai`；`afcli` 入口指向不存在的 `agentframe.cli.main` | pyproject.toml:8-12 | 装包缺依赖、入口坏 |
| A2 | 流式 `usage` / `finish_reason` 丢失：`_parse_chunk` 在 `choices` 为空时提前 return，丢弃 OpenAI 系末帧（无 choices、带 usage）的 usage；`done` 事件从不填 `LLMStreamEvent.usage` | llm/client.py:48, 141 | `usage` 字段形同虚设 |
| A3 | `_finish_tool_calls` 的 `json.loads` 无保护，供应商截断 tool_calls 时 arguments 残缺 → 流式抛 `ValueError` | llm/client.py:39 | 流式中途崩溃 |

## B. 健壮性（建议修）

| # | 问题 | 位置 | 影响 |
|---|------|------|------|
| B1 | LLM 首轮失败时 `invoke` 把用户输入当结果返回（`messages[-1]` 是 HumanMessage）；`invoke_messages([])` 直接 IndexError | core/base.py:23-24, 233 | 错误语义误导 |
| B2 | `LLMClient` 无 `close()` / 上下文管理器，httpx 连接池不释放 | llm/client.py | 长驻进程连接泄漏 |
| B3 | 推理字段只认 `reasoning_content`（DeepSeek 专属），`reasoning`/`reasoning_text` 不兼容 | llm/client.py:51 | 多厂商兼容性差 |
| B4 | `StreamStop` 在 reasoning 阶段中断时 `partial` 不含 reasoning（只累计 content） | core/base.py:96-101 | 中断恢复丢失思考内容 |

## C. API 一致性

| # | 问题 | 位置 | 影响 |
|---|------|------|------|
| C1 | `invoke` 走 `before_trace`/`after_trace`，`stream`/`invoke_messages` 完全绕过且 `session_id` 形同虚设 | core/base.py:207-240 | 生命周期语义不一致 |
| C2 | `stream()` 返回 LangGraph 原始 state dict，与 `on_llm_content` 是两套流式通道，且零测试 | core/base.py:220-229 | 通道冗余、易误用 |
| C3 | README 的 `from agentframe.middlewares import tools, memory` import 即挂（模块不存在） | README.md:47 | 文档示例不可运行 |

## D. 可维护性

| # | 问题 | 位置 |
|---|------|------|
| D1 | `LLMStreamEvent.type` 是裸 `str`，与 `Phase` StrEnum 风格不一致，建议 `Literal` | llm/types.py:48 |
| D2 | `_act_llm` 中 `full`/`reasoning` 手工累计可收敛为小 accumulator | core/base.py:87-99 |
| D3 | `_copy_state` 是浅拷贝，中间件原地改 `data["messages"]` 会污染真实状态（有文档提醒，保持现状 + 注释） | core/base.py:31-32 |

## E. 测试缺口

| # | 缺口 | 影响 |
|---|------|------|
| E1 | `LLMClient` 无任何测试：`_parse_chunk` / `_parse_completion_response` / SSE 解析 / tool_calls 聚合 / usage 全是纯函数 | 非流式 `invoke()` 路径零覆盖 |
| E2 | `stream()` 无测试 | C2 行为无固化 |
| E3 | 错误路径回显行为无测试固化 | B1 无回归保护 |

## 已确认通过

- 25 个测试全过；basedpyright 0 error。
- 未提交 diff（类型注解强化：`RunnableConfig`、`_AgentGraph` Protocol、`cast`、`@override`）质量良好，可提交。
