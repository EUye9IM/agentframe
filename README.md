# agentframe

基于 [LangGraph](https://github.com/langchain-ai/langgraph) 和 [litellm](https://github.com/BerriAI/litellm) 的轻量 Agent 框架。

## 特性

- **LangGraph 驱动** — StateGraph 构建 LLM → Tool → Compress 循环
- **统一 LLM 接口** — 通过 litellm 支持 100+ 模型供应商
- **工具系统** — Python 函数工具 + MCP 远程工具
- **上下文压缩** — 基于 Token 用量自动触发 LLM 摘要压缩
- **持久化钩子** — 直接兼容 LangGraph `BaseCheckpointSaver`，用户自行实现存储
- **同步 + 异步** — 同时提供 `invoke`/`stream` 和 `ainvoke`/`astream`

## 快速开始

```python
from agentframe import Agent, function_tool

@function_tool
def get_weather(city: str) -> str:
    """获取指定城市的当前天气。"""
    return f"{city}的天气是晴天，22°C。"

agent = Agent(
    model="gpt-4o",
    system_prompt="你是一个助手。",
    tools=[get_weather],
    compress_threshold=100_000,
)

response = agent.invoke("北京的天气怎么样？")
print(response)
```

## 安装

```bash
pip install agentframe
```

启用 MCP 支持：

```bash
pip install "agentframe[mcp]"
```

## 持久化

agentframe 不内置持久化实现。直接传入 LangGraph 的 `BaseCheckpointSaver`：

```python
from langgraph.checkpoint.sqlite import SqliteSaver
from agentframe import Agent

checkpointer = SqliteSaver.from_conn_string("sessions.db")
agent = Agent(model="gpt-4o", checkpointer=checkpointer)

agent.invoke("你好", session_id="session-1")
agent.invoke("刚才我说了什么？", session_id="session-1")  # 能看见历史
```

可用 checkpointer：

- `langgraph.checkpoint.memory.MemorySaver` — 内存存储
- `langgraph.checkpoint.sqlite.SqliteSaver` — SQLite 存储
- `langgraph.checkpoint.postgres.PostgresSaver` — PostgreSQL 存储

## MCP 工具（异步）

```python
import asyncio
from agentframe import Agent

agent = Agent(
    model="gpt-4o",
    mcp_configs=[
        {
            "name": "filesystem",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
            "transport": "stdio",
        }
    ],
)

async def main():
    response = await agent.ainvoke("列出 /tmp 目录中的文件")
    print(response)

asyncio.run(main())
```

## 架构

```
┌─────────────────────────────────────────────────────────┐
│              Agent  (public API)                         │
├─────────────────────────────────────────────────────────┤
│              LangGraph StateGraph                        │
│  ┌──────────┐     ┌──────────┐     ┌────────────────┐  │
│  │ agent    │ ──► │ tools    │ ──► │ agent (loop)   │  │
│  │ (LLM)    │ ◄── │ (exec)   │     │ (+ compress)   │  │
│  └──────────┘     └──────────┘     └────────────────┘  │
├───────────────┬─────────────────┬───────────────────────┤
│  LLM Client   │  ToolRegistry   │  Compressor           │
│  (litellm)    │  + MCP Client   │  (summary strategy)   │
└───────────────┴─────────────────┴───────────────────────┘
```

### Agent 循环

1. **agent 节点** — 将当前消息 + 工具定义发送给 LLM
2. **条件边** — 如果 AI 响应包含 tool_calls → 进入 tools 节点；否则 → 结束
3. **tools 节点** — 执行每个工具调用，将结果作为 `ToolMessage` 追加
4. **回到 agent** — 继续循环，携带工具结果
5. **压缩** — 每次 LLM 调用前，如果 `total_tokens > threshold`，将最旧的消息通过 LLM 摘要替换为一条 `SystemMessage`

## 模块

| 模块 | 说明 |
|------|------|
| `core.agent` | Agent 类 — 图构建、invoke/stream |
| `core.state` | AgentState TypedDict |
| `llm.client` | litellm 封装，OpenAI 消息格式转换 |
| `tools.function_tool` | `@function_tool` 装饰器 |
| `tools.registry` | 工具注册中心 |
| `tools.mcp_client` | MCP 客户端（SSE + STDIO） |
| `compression.summarizer` | Token 阈值驱动的对话摘要器 |
| `memory.hooks` | LangGraph checkpointer 集成说明 |
| `multi_agent` | （预留） |

## 依赖

- Python >= 3.12
- langgraph >= 0.2
- litellm >= 1.60
- langchain-core >= 0.3

## License

MIT
