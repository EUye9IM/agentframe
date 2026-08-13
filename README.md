# AgentFrame

一个**纯同步**、**钩子驱动**的 Python Agent 框架，构建在 LangGraph 之上。

> 当前为 v0.2 重写版（`master` 分支）。核心引擎与 LLM 层已就绪，中间件 / 多智能体 / CLI 仍在实现中。

## 核心特性

- **钩子即架构**：`BaseAgent` 极薄，只提供 15 个同步钩子与 LangGraph 状态机，不内置任何功能
- **中间件叠加**：中间件是类，通过动态类继承（MRO）叠加到基类，顺序即执行顺序
- **状态机驱动**：LangGraph `StateGraph` 即状态机，错误是一等状态，可随时改道
- **纯同步**：全框架无 async，LLM 走裸 HTTP（httpx），流式用 SSE
- **类型固定**：状态用 `Phase` 枚举，LLM 交换用结构体，不散用 dict
- **状态最瘦**：`AgentState` 只有 `messages`，其他一切走钩子

## 安装

```bash
uv sync            # 或 pip install -e ".[dev]"
```

需要 Python 3.12。

## 快速开始

```python
from agentframe import Agent

agent = Agent(
    model="deepseek-chat",
    system_prompt="你是一个乐于助人的助手",
    base_url="https://api.deepseek.com",
    api_key="sk-...",
)

print(agent.invoke("你好"))
```

组合中间件（工厂函数返回中间件类，需实例化）：

```python
from agentframe import Agent
from agentframe.middlewares import tools, memory

agent = Agent(
    model="deepseek-chat",
    middlewares=[
        tools([my_function])(),   # 工具
        memory(),                 # 会话持久化
    ],
)
```

继承式自定义行为：

```python
from agentframe import Agent

class MyAgent(Agent):
    def on_content_end(self, content: str) -> None:
        super().on_content_end(content)
        print(content, end="", flush=True)

agent = MyAgent(model="deepseek-chat", middlewares=[memory()])
```

## 架构一览

```
Agent（公共类，可继承 + 可组合）
  └─ middlewares=[...] 动态类继承（MRO = 执行顺序）
        └─ BaseAgent（引擎 + 钩子协议）
              └─ LangGraph StateGraph 状态机：LLM ⇄ TOOLS 循环
                    └─ LLMClient（裸 httpx，结构体进出）
```

- `AgentState = {messages}`：跨节点唯一领域数据
- `Phase` 枚举：`LLM` / `TOOLS` / `END`
- 错误处理：`NodeError` → `handle_error(error, node) -> Command(goto=...)`，可修复状态 + 改道
- 流式中断：钩子抛 `StreamStop`，由打断者中间件的 `handle_error` 认领

## 钩子协议

| 类别 | 钩子 | 说明 |
|------|------|------|
| trace | `before_trace` / `after_trace` | 整个回合前后 |
| turn | `before_turn` / `after_turn` | 每次 LLM 调用前后 |
| LLM | `before_llm` / `on_llm_reasoning` / `on_reasoning_end` / `on_llm_content` / `on_content_end` / `after_llm` | 请求可改、流式事件、响应转消息历史 |
| 工具 | `before_tool_call` / `after_tool_result` | 审批子集、结果转消息历史 |
| 流程 | `handle_next` / `handle_error` | 状态机决策与错误改道 |
| 事件 | `on_state_changed` | 每次消息追加 |

## 文档

- [架构设计](docs/ARCHITECTURE.md)
- [测试设计](docs/TESTING.md)

## 测试

```bash
.venv/bin/python -m pytest tests/ -q
```

## 状态

- ✅ 核心：`BaseAgent` / 钩子协议 / `Phase` 枚举 / `AgentState`
- ✅ LLM 层：`LLMClient`（裸 httpx）/ `LLMRequest` / `LLMResponse` / `LLMStreamEvent`
- ✅ 测试：25 个状态转换 / 钩子 / 错误处理用例
- 🚧 实现中：middlewares（tools/mcp/compress/memory）、multiagent、CLI、examples

## License

MIT
