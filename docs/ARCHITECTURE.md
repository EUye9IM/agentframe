# AgentFrame v0.2 架构设计

> 版本：v0.2.0（master 分支，完全重写）
> 状态：设计定稿，待实现
> 对比：v0.1 为单体 `Agent`（工具/MCP/压缩/checkpointer 全塞构造参数，sync/async 全线重复，钩子靠 monkey-patch）。v0.2 重构为 **钩子基类 + 状态机引擎 + 中间件叠加**。

## 1. 设计目标

| 目标 | 说明 |
|------|------|
| 钩子即架构 | `BaseAgent` 极薄，所有能力通过钩子注入，业务流通过状态机表达 |
| 中间件可叠加 | 工厂函数返回配置绑定的中间件类，`Agent` 用动态类继承拼装，MRO 即执行顺序 |
| 对外简单 | 用户只见 `Agent` 抽象类 + `middlewares=[...]`，感知不到 BaseAgent/MRO/钩子编排 |
| 纯同步 | 全框架无 async：LLM 只留 `invoke/stream`，MCP 走线程桥，multiagent/CLI/压缩全同步 |
| 状态机而非循环 | LangGraph StateGraph 即状态机，error 是一等状态，可通过 `Command(goto)` 改道 |
| 状态最瘦 | `AgentState` 只有 `messages`，私有状态走中间件实例属性或外部 Store |

## 2. 总体架构

```
┌─────────────────────────────────────────────────────────┐
│  Agent（公共抽象类）                                      │
│  ├─ 可继承：子类覆写钩子                                  │
│  └─ 可组合：middlewares=[tools(), mcp(), compress(), ...]│
│      └─ __init__ 动态类继承：                             │
│         Concrete = type(name, (*中间件类, 用户子类), {})  │
│         MRO 顺序 = 列表顺序 = 执行顺序                    │
├─────────────────────────────────────────────────────────┤
│  BaseAgent（引擎 + 钩子协议）                             │
│  └─ LangGraph StateGraph（状态机）                        │
│      LLM ⇄ TOOLS 循环 + error_handler 改道               │
├─────────────────────────────────────────────────────────┤
│  Middleware（钩子链成员）                                 │
│  tools / mcp / compress / memory / 自定义                │
└─────────────────────────────────────────────────────────┘
```

### 依赖关系

```
Agent ──继承──▶ BaseAgent ──使用──▶ LLMClient（同步）
                        │
                        ├──▶ StateGraph（LangGraph，仅作状态机引擎）
                        ├──▶ ToolRegistry / MCPClient（线程桥）
                        └──▶ Compressor（同步）
```

## 3. 核心状态模型

### 3.1 AgentState —— 只有 messages

```python
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
```

**设计决策**：
- 只有 `messages` 是跨节点流转的领域数据。
- 工具列表是 agent 配置（挂 `tool_registry`），不是流转数据。
- system_prompt / session_id 是构造参数。
- **错误信息不走 state**：LangGraph 将失败以 `NodeError` 函数参数注入 `error_handler`，经 `on_error` 钩子返回 `Command(update=..., goto=...)` 修复/改道。
- **无 total_tokens / input_text**：压缩改为从 messages 无状态估算；输入在 `invoke()` 构造好 `[system, human]` 再进图。

### 3.2 状态机（图）

```
LLM ──(有 tool_calls)──▶ TOOLS ──▶ LLM ──▶ ...
 │
 └──(无 tool_calls)──▶ END

任意节点异常 ──▶ 该节点 error_handler(注入 NodeError) ──▶ on_error 返回 Command(goto) ──▶ 重试/降级/自定义恢复状态
```

- 节点 = 状态，条件边 = 转换。
- 中间件可用 `add_state()` 声明自定义状态（编译期并入图）。
- 每个节点可挂 `error_handler` 和 `retry_policy`（LangGraph 原生）。

## 4. 钩子协议（BaseAgent）

### 4.1 语义约定

| 类别 | 前缀 | 返回值 | 链式行为 |
|------|------|--------|----------|
| 变换型 | `before_*` / `after_*` | 返回被修改的数据 | 每个中间件 `super()` 后变换，可短路 |
| 事件型 | `on_*` | 无（`None`） | 广播，逐个中间件触发 |
| 流程型 | `decide_next` / `on_error` | 状态名 / `Command` | 决定状态机下一步 |

**命名约定**：数据参数叫 `data`，状态（阶段）叫 `node`，不再混用 `state`。

### 4.2 钩子清单（全同步，默认空实现）

| 钩子 | 签名 | 类型 | 触发点 |
|------|------|------|--------|
| `before_trace` | `(input: str, session: str\|None) -> str` | 变换 | invoke() 入口，可改写输入 |
| `after_trace` | `(data: AgentState, session: str\|None) -> str` | 变换 | invoke() 出口，可改写结果 |
| `before_turn` | `(data: AgentState) -> AgentState` | 变换 | 每个回合（LLM 调用）开始 |
| `after_turn` | `(data: AgentState) -> AgentState` | 变换 | TOOLS 结束 / 无工具时 LLM 结束 |
| `before_llm` | `(messages, tools) -> (messages, tools)` | 变换 | LLM 调用前，可改上下文和工具集 |
| `on_llm_reasoning` | `(text: str) -> bool` | 事件/流式 | 每段 reasoning 流，返回 False 中断 |
| `on_llm_content` | `(text: str) -> bool` | 事件/流式 | 每段 content 流，返回 False 中断 |
| `after_llm` | `(ai_msg: AIMessage) -> AIMessage` | 变换 | LLM 调用后 |
| `before_tool_call` | `(tool_calls: list[dict]) -> list[dict]` | 变换 | 工具执行前，返回审批子集 |
| `after_tool_result` | `(name: str, result: str) -> str` | 变换 | 每个工具执行后，可改写结果 |
| `decide_next` | `(from_node: str, default: str) -> str` | 流程 | 条件边决策 |
| `on_error` | `(error: NodeError, node: str) -> Command` | 流程 | 错误处理，修复状态 + 改道 |
| `on_state_changed` | `(messages: list[BaseMessage]) -> None` | 事件 | 每次消息追加 |

### 4.3 流式中断（方案 A）

```python
def on_llm_content(self, text: str) -> bool:
    """返回 False 停止生成。默认 True。不负责任何提示。"""
    return True
```

```python
def _act_llm(self, state):
    messages, tools = self.before_llm(state["messages"], self._collect_tools())
    full = ""
    try:
        for event in self.llm_client.stream(messages, tools=tools):
            if event["type"] == "reasoning":
                if not self.on_llm_reasoning(event["content"]): break
            elif event["type"] == "content":
                if not self.on_llm_content(event["content"]): break
                full += event["content"]
    except KeyboardInterrupt:
        pass                                   # 同样：中断，保留 partial
    return {"messages": [AIMessage(content=full)]}   # 无 tool_calls → 自然到 END
```

- 钩子返回 `False` 或 `Ctrl+C` → 中断，**partial 内容保留为最后一条 AIMessage**，无 tool_calls，自然到 END。
- 提示由**打断者**（CLI/中间件）在 UI 层自行添加，**不进消息历史**。
- 中断后的后续动作留给下一轮 `before_llm` / `decide_next`，基类不干预。

### 4.4 错误处理（NodeError + Command 改道）

```python
def _default_error_handler(self, state: AgentState, error: NodeError) -> Command:
    return self.on_error(error, error.node)
```

```python
def on_error(self, error: NodeError, node: str) -> Command:
    """感知失败 + 修复状态 + 改道。默认中止。"""
    return Command(goto="END")
```

中间件覆写示例（限流重试，全程不碰 messages）：

```python
class RateLimitMiddleware:
    def on_error(self, error, node):
        if isinstance(error.error, RateLimitError):
            return Command(update={...}, goto="LLM")   # 修复 + 改道
        return super().on_error(error, node)
```

## 5. 中间件机制

### 5.1 工厂函数（返回闭包绑定配置的类）

```python
def tools(functions):
    class ToolsMiddleware(Middleware):
        def before_llm(self, messages, tools):
            messages, tools = super().before_llm(messages, tools)
            return messages, tools + [to_openai(f) for f in functions]
    return ToolsMiddleware
```

- 每次调用生成新类 → **同类可重复叠加**，无配置传递问题。
- 中间件类方法覆写钩子，用 `super()` 协作链式调用。

### 5.2 动态类继承（Agent.__init__）

```python
class Agent(BaseAgent):
    def __init__(self, model, *, system_prompt=None, middlewares=None, api_key=None, base_url=None):
        mws = list(middlewares or [])
        if mws:
            Concrete = type("_ConcreteAgent", (*[type(m) for m in mws], type(self)), {})
            self.__class__ = Concrete          # MRO = 中间件顺序 = 执行顺序
        super().__init__(model, system_prompt=system_prompt, api_key=api_key, base_url=base_url)
        for m in mws:
            m._register(self)                  # 中间件声明自定义状态/边
```

### 5.3 自定义状态

中间件在 `_register(agent)` 里调用 `agent.add_state(name, action, error_handler=None, retry_policy=None)` 声明新状态，编译期并入图。可同时声明到新状态的边。

### 5.4 model 切换

`_act_llm` 在 `before_llm` 之后才读 `self.llm_client`，所以中间件在 `before_llm` 里直接 `self.llm_client = fallback_client` 即可切换模型/降级兜底，零协议改动。

## 6. 公共 API

```python
# 组合式（默认用法）
agent = Agent(
    model="deepseek-chat",
    system_prompt="...",
    middlewares=[tools([run_bash]), mcp([{...}]), compress(100000), memory()],
)
agent.invoke("hi", session_id="s1")

# 继承式（自定义钩子）
class MyChat(Agent):
    def on_llm_content(self, text):
        if not super().on_llm_content(text): return False
        self.ui.write(text)
        return True

my = MyChat(model=..., middlewares=[memory()])
```

| 方法 | 签名 | 说明 |
|------|------|------|
| `invoke` | `(input_text, *, session_id=None) -> str` | 同步执行 |
| `stream` | `(input_text, *, session_id=None) -> Iterator[dict]` | 图事件流，调用方可外部停止 |
| `invoke_messages` | `(messages, *, session_id=None) -> str` | 显式消息列表（multiagent 用） |

## 7. 模块布局

```
agentframe/
  __init__.py            # 导出 Agent + middlewares
  core/
    base.py              # BaseAgent：引擎 + 钩子协议 + 图构建
    hooks.py             # Middleware 基类 + 钩子签名
    state.py             # AgentState（messages only）
  middlewares/
    __init__.py          # 工厂：tools / mcp / compress / memory
    tools.py             # ToolsMiddleware
    mcp.py               # MCPMiddleware
    compression.py       # CompressionMiddleware
    memory.py            # MemoryMiddleware（checkpointer + session_id）
  llm/
    client.py            # 仅同步：invoke / stream
  tools/
    registry.py
    function_tool.py
    mcp_client.py        # 同步（线程桥为唯一路径）
  compression/
    summarizer.py        # 同步
  multiagent/            # 同步编排（Member 基于 Agent + invoke_messages）
  cli/                   # Agent 子类示例（含原渲染逻辑）
examples/
  chatroom_315.py
pyproject.toml           # version 0.2.0，去 pytest-asyncio
```

## 8. 能力覆盖矩阵（泛用性评估）

| 能力 | 机制 | 覆盖 |
|------|------|------|
| tools | `ToolsMiddleware.before_llm` 注入 + `before_tool_call` 审批 + 分发 | ✅ |
| mcp | `MCPMiddleware` 同上，分发走线程桥 | ✅ |
| compress | `CompressionMiddleware.before_llm` 从 messages 估大小，超阈值摘要 | ✅ |
| 会话 memory | `MemoryMiddleware`：session_id → thread_id 喂 checkpointer | ✅ |
| 长期 memory | 中间件持外部 Store，`before_turn` 注入 / `after_turn` 写回 | ✅ |
| model 切换 | `before_llm` 里改 `self.llm_client` | ✅ |
| 子 Agent / multiagent | 子 Agent 包成 FunctionTool；multiagent 用 `invoke_messages` | ✅ |
| 错误重试 | `on_error` → `Command(goto)` + `retry_policy` | ✅ |
| 流式 UI | `on_llm_content` 在 invoke 内同步触发 | ✅ |
| 人工审批 | `before_tool_call` 阻塞式 y/n（纯同步） | ✅ |

## 9. 纯同步化要点

- `LLMClient`：只留 `invoke` / `stream`（同步生成器），删 `ainvoke/astream`。
- `MCPClient`：线程桥（`run_coroutine_threadsafe` + 后台常驻事件循环）成为**唯一**调用路径。
- `Compressor`：只留 `compress`。
- multiagent：`stream_discussion` 改同步生成器 `Iterator[Event]`。
- CLI：去 `asyncio.run`，`chat()` 全同步。
- pyproject：去 `pytest-asyncio` 依赖。

## 10. 版本与分支策略

- 分支：`master`（重写起点），dev 分支 v0.1 冻结不动。
- 版本：`v0.2.0`（tag 于实现完成时）。

## 11. 验证策略

- 不写单元测试（重构首版）。
- 冒烟验证：import 全通、`afcli` 可交互聊天、`chatroom_315` 示例可跑。

## 12. 后续扩展（不在首版）

- `StreamStop` 异常 + `Command(goto)` 路由（方案 B，中断可路由到自定义状态）。
- 中间件私有 channel（进 checkpointer）——当前明确**不做**，保持 state 最瘦。
- 工具流式输出钩子 `on_tool_stream`。
