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
| 纯同步 | 全框架无 async：LLM 走裸 HTTP（httpx），MCP 走线程桥，multiagent/CLI/压缩全同步 |
| 状态机而非循环 | LangGraph StateGraph 即状态机，error 是一等状态，可通过 `Command(goto)` 改道 |
| 状态最瘦 | `AgentState` 只有 `messages`，私有状态走中间件实例属性或外部 Store |
| 类型固定 | 状态用 `Phase` 枚举，LLM 交换用结构体（`LLMRequest`/`LLMResponse`），不散用 dict/裸字符串 |

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
│  LLMClient（裸 HTTP，结构体进/结构体出）                  │
│  ├─ LLMRequest → invoke/stream → LLMResponse/事件        │
│  └─ httpx 同步客户端，SSE 解析流式                       │
├─────────────────────────────────────────────────────────┤
│  Middleware（钩子链成员）                                 │
│  tools / mcp / compress / memory                        │
└─────────────────────────────────────────────────────────┘
```

### 依赖关系

```
Agent ──继承──▶ BaseAgent ──使用──▶ LLMClient（httpx 裸 HTTP）
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
- **错误信息不走 state**：LangGraph 将失败以 `NodeError` 函数参数注入 `error_handler`，经 `handle_error` 钩子返回 `Command(update=..., goto=...)` 修复/改道。
- **无 total_tokens / input_text**：压缩改为从 messages 无状态估算；输入在 `invoke()` 构造好 `[system, human]` 再进图。

### 3.2 状态机（图）与 Phase 枚举

```python
# core/phases.py
from enum import StrEnum

class Phase(StrEnum):
    LLM = "LLM"
    TOOLS = "TOOLS"
    END = "__end__"          # 与 LangGraph END 同值，路由目标而非可注册节点
```

```
LLM ──(有 tool_calls)──▶ TOOLS ──▶ LLM ──▶ ...
 │
 └──(无 tool_calls)──▶ END

任意节点异常 ──▶ 该节点 error_handler(注入 NodeError) ──▶ handle_error 返回 Command(goto) ──▶ 重试/降级/终止
```

- 节点 = 状态，条件边 = 转换。**状态数量固定**（LLM / TOOLS / END），`Phase` 为封闭 StrEnum，首版不支持中间件自定义状态。
- 每个节点可挂 `error_handler` 和 `retry_policy`（LangGraph 原生）。
- **枚举与 LangGraph 兼容**：StrEnum 成员本身是 `str` 子类，满足 LangGraph 所有 `isinstance(x, str)` 校验（add_node、`Command.goto`），可直接传参无需 `.value`。

## 4. LLM 结构体（llm/types.py）

```python
@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

@dataclass
class LLMRequest:
    """进入 LLMClient / before_llm 的请求结构体（model 归 LLMClient 端点持有）"""
    messages: list[BaseMessage]            # langchain 消息（传输层内部转 OpenAI dict）
    tools: list[dict] | None = None        # OpenAI 工具格式
    temperature: float | None = None
    max_tokens: int | None = None
    extra: dict = field(default_factory=dict)   # 厂商扩展参数逃生舱

@dataclass
class LLMResponse:
    """LLMClient / after_llm 产出的响应结构体"""
    message: AIMessage                     # content + tool_calls
    usage: Usage | None = None
    reasoning: str = ""
    finish_reason: str | None = None
    model: str | None = None
    raw: dict | None = None                # 原始 JSON（内部/调试用，非主接口）

@dataclass
class LLMStreamEvent:
    """stream() 逐事件产出"""
    type: str                              # "reasoning" | "content" | "done"
    content: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    usage: Usage | None = None
```

## 5. LLMClient（裸 HTTP，结构体进/结构体出）

```python
# llm/client.py
class LLMClient:
    def __init__(self, *, base_url: str, model: str, api_key: str | None, **defaults) -> None:
        self.model = model                           # 端点绑定模型
        self._http = httpx.Client(base_url=base_url,
                                  headers={"Authorization": f"Bearer {api_key}"}, ...)
        self._defaults = defaults                     # temperature 等默认值，与 request 合并

    def invoke(self, request: LLMRequest) -> LLMResponse: ...
    def stream(self, request: LLMRequest) -> Iterator[LLMStreamEvent]: ...   # SSE 解析
```

- 不再依赖 `openai` SDK，直接 `httpx` 调用 `/chat/completions`。
- `stream()` 内部做 SSE 解析（`data:` 行、`[DONE]`、delta 聚合），逐段 yield `LLMStreamEvent`，最后 yield `done`（含聚合后的 tool_calls 与 usage）。
- 流式完成后由 `_act_llm` 聚合回 `LLMResponse`，保证中间件在 `after_llm` 看到的形态与 `invoke` 一致。

## 6. 钩子协议（BaseAgent）

### 6.1 语义约定

| 类别 | 前缀 | 返回值 | 链式行为 |
|------|------|--------|----------|
| 变换型 | `before_*` / `after_*` | 返回被修改的数据 | 每个中间件 `super()` 后变换，可短路 |
| 事件型 | `on_*` | 无（`None`） | 广播，逐个中间件触发 |
| 流程型 | `handle_*` | `Phase` / `Command` | 决定状态机下一步 |

**命名约定**：数据参数叫 `data`，状态（阶段）叫 `node`；流程钩子统一 `handle_` 前缀。

### 6.2 钩子清单（全同步，默认空实现）

| 钩子 | 签名 | 类型 | 触发点 |
|------|------|------|--------|
| `before_trace` | `(input: str, session: str\|None) -> str` | 变换 | invoke() 入口（唯一入口），可改写输入 |
| `after_trace` | `(data: AgentState, session: str\|None) -> str` | 变换 | invoke() 出口，取最后一条 AI 消息（首轮失败时无 AI 消息则返回空串，不回显用户输入） |
| `before_turn` | `(data: AgentState) -> AgentState` | 变换 | 每个回合（LLM 调用）开始 |
| `after_turn` | `(data: AgentState) -> AgentState` | 变换 | TOOLS 结束 / 无工具时 LLM 结束 |
| `before_llm` | `(request: LLMRequest) -> LLMRequest` | 变换 | LLM 调用前，可改请求体（messages/tools/temperature...） |
| `on_llm_reasoning` | `(text: str) -> None` | 事件/流式 | 每段 reasoning 流 |
| `on_reasoning_end` | `(reasoning: str) -> None` | 事件 | 思考结束，通知完整思考内容 |
| `on_llm_content` | `(text: str) -> None` | 事件/流式 | 每段 content 流 |
| `on_content_end` | `(content: str) -> None` | 事件 | 消息结束，通知完整消息内容 |
| `after_llm` | `(response: LLMResponse) -> list[BaseMessage]` | 变换 | 响应转消息历史，可控制顺序 |
| `before_tool_call` | `(tool_calls: list[dict]) -> list[dict]` | 变换 | 工具执行前，返回审批子集 |
| `after_tool_result` | `(name: str, result: str) -> list[BaseMessage]` | 变换 | 每个工具执行后，结果转消息历史 |
| `handle_next` | `(from_node: Phase, default: Phase) -> Phase` | 流程 | 条件边决策 |
| `handle_error` | `(error: NodeError, node: Phase) -> Command` | 流程 | 错误处理，修复状态 + 改道 |
| `on_state_changed` | `(messages: list[BaseMessage]) -> None` | 事件 | 每次消息追加 |

### 6.3 `after_llm` / `after_tool_result`：响应/结果 → 消息历史

转换逻辑写在钩子内部，子类覆写后调 `super()` 拿父类解析结果，自行控制顺序：

```python
def after_llm(self, response: LLMResponse) -> list[BaseMessage]:
    """响应 → 消息历史。默认把 response.message 作为一条 AIMessage 返回。"""
    return [response.message]

# 子类示例：reasoning 前置成独立消息（控制顺序）
class MyAgent(BaseAgent):
    def after_llm(self, response):
        base = super().after_llm(response)               # 父类解析
        if response.reasoning:
            return [AIMessage(content=f"[thinking]\n{response.reasoning}")] + base
        return base

def after_tool_result(self, name: str, result: str) -> list[BaseMessage]:
    """工具结果 → 消息历史。默认包成一条 ToolMessage。"""
    return [ToolMessage(content=result, tool_call_id=self._last_call_id)]
```

### 6.4 流式中断（StreamStop 异常 + handle_error）

中断 = 钩子抛 `StreamStop` 异常，统一走错误路径，由**打断者自己写的中间件**在 `handle_error` 里认领处理：

```python
class StreamStop(Exception):
    def __init__(self, message: str = "", goto: Phase = Phase.END):
        self.message, self.goto = message, goto
        self.partial: str = ""            # 已累计的 content
        self.partial_reasoning: str = ""  # 已累计的 reasoning

def on_llm_content(self, text):          # 纯事件，无返回值
    if self._budget_exceeded():
        raise StreamStop(goto=Phase.LLM, message="预算超了")
```

```python
def _act_llm(self, state):
    request = self.before_llm(self._build_request(state["messages"]))
    full, reasoning, tool_calls, usage, finish_reason = "", "", [], None, None
    try:
        for event in self.llm_client.stream(request):
            if event.type == "reasoning":
                reasoning += event.content; self.on_llm_reasoning(event.content)
            elif event.type == "content":
                self.on_llm_content(event.content); full += event.content
            elif event.type == "done":
                tool_calls, usage, finish_reason = event.tool_calls, event.usage, event.finish_reason
    except StreamStop as stop:
        stop.partial = full                      # partial 内容挂到异常上
        stop.partial_reasoning = reasoning       # 思考内容一并保留
        raise
    except KeyboardInterrupt:
        raise StreamStop()
    self.on_reasoning_end(reasoning)             # 思考结束事件
    self.on_content_end(full)                    # 消息结束事件
    response = LLMResponse(message=AIMessage(content=full, tool_calls=tool_calls or []),
                           usage=usage, model=self.llm_client.model)
    return {"messages": self.after_llm(response)}
```

```python
# 基类 handle_error —— 统一，无类型判断，只默认中止
def handle_error(self, error: NodeError, node: Phase) -> Command:
    return Command(goto=Phase.END)

# 打断中间件 —— 复写 handle_error，自己认 StreamStop
class InterruptMiddleware(Middleware):
    def handle_error(self, error, node):
        if isinstance(error.error, StreamStop):
            return Command(update={"messages": [AIMessage(content=error.error.partial)]},
                           goto=error.error.goto)
        return super().handle_error(error, node)
```

职责划分：

| 层 | 职责 |
|----|------|
| `_act_llm`（核心） | 让中断变成带 partial 的异常——捕获 `StreamStop` 挂 `.partial` 重抛；捕获 `KeyboardInterrupt` 转 `StreamStop`。不做决策 |
| `handle_error`（基类） | 统一 `Command(goto=END)`，不认类型 |
| 打断中间件/CLI | 复写 `handle_error` 认领 `StreamStop` → 保留 partial、改道、加提示 |

- 提示由**打断者**在 UI 层自行添加，**不进消息历史**（`StreamStop.message` 字段供 UI 显示）。
- 无打断中间件时，Ctrl+C / `StreamStop` 仅中止（partial 丢弃），符合"打断者自己负责"。
- 中断后的后续动作留给下一轮 `before_llm` / `handle_next`，基类不干预。

### 6.5 错误处理（NodeError + Command 改道）

```python
def _default_error_handler(self, state: AgentState, error: NodeError) -> Command:
    return self.handle_error(error, error.node)
```

```python
def handle_error(self, error: NodeError, node: Phase) -> Command:
    """感知失败 + 修复状态 + 改道。默认中止。"""
    return Command(goto=Phase.END)
```

中间件覆写示例（限流重试，全程不碰 messages）：

```python
class RateLimitMiddleware:
    def handle_error(self, error, node):
        if isinstance(error.error, RateLimitError):
            return Command(update={...}, goto=Phase.LLM)   # 修复 + 改道
        return super().handle_error(error, node)
```

## 7. 中间件机制

### 7.1 工厂函数（返回闭包绑定配置的类）

```python
def tools(functions):
    class ToolsMiddleware(Middleware):
        def before_llm(self, request):
            request = super().before_llm(request)
            request.tools = (request.tools or []) + [to_openai(f) for f in functions]
            return request
    return ToolsMiddleware
```

- 每次调用生成新类 → **同类可重复叠加**，无配置传递问题。
- 中间件类方法覆写钩子，用 `super()` 协作链式调用。

### 7.2 动态类继承（Agent.__init__）

```python
class Agent(BaseAgent):
    def __init__(self, *, llm_client, system_prompt=None, middlewares=None):
        mws = list(middlewares or [])
        if mws:
            Concrete = type("_ConcreteAgent", (*[type(m) for m in mws], type(self)), {})
            self.__class__ = Concrete          # MRO = 中间件顺序 = 执行顺序
        super().__init__(llm_client=llm_client, system_prompt=system_prompt)
```

### 7.3 自定义状态

首版**不支持**中间件声明新状态——`Phase` 为封闭枚举，功能一律通过钩子注入。后续需要时再议。

### 7.4 model 切换

model 归 `LLMClient`（端点）持有，`Agent` 不感知模型；切换模型 = 换 client。中间件在 `before_llm` 里 `self.llm_client = fallback_client`（换供应商/兜底）：

```python
class FallbackMiddleware(Middleware):
    def before_llm(self, request):
        request = super().before_llm(request)
        if self._primary_down:
            self.llm_client = self._fallback_client
        return request
```

## 8. 公共 API

```python
# 组合式（默认用法）
client = LLMClient(model="deepseek-chat", base_url="...", api_key="...")
agent = Agent(
    llm_client=client,
    system_prompt="...",
    middlewares=[tools([run_bash]), mcp([{...}]), compress(100000), memory()],
)
agent.invoke("hi", session_id="s1")

# 继承式（自定义钩子）
class MyChat(Agent):
    def on_content_end(self, content):
        super().on_content_end(content)
        self.ui.write(content)

my = MyChat(llm_client=client, middlewares=[memory()])
```

| 方法 | 签名 | 说明 |
|------|------|------|
| `invoke` | `(input_text, *, session_id=None, config=None) -> str` | **唯一公开入口**：文本入、str 出，包在 before_trace/after_trace 内；`config` 原样透传 `graph.invoke`（checkpointer 逃生口）；流式体验走 `on_llm_content`/`on_llm_reasoning` 事件钩子 |

## 9. 模块布局

```
agentframe/
  __init__.py            # 导出 Agent + middlewares
  core/
    base.py              # BaseAgent：引擎 + 钩子协议 + 图构建
    hooks.py             # Middleware 基类 + 钩子签名
    phases.py            # Phase 枚举（LLM/TOOLS/END）
    state.py             # AgentState（messages only）
  llm/
    client.py            # LLMClient：裸 httpx，invoke/stream
    types.py             # LLMRequest / LLMResponse / LLMStreamEvent / Usage
  middlewares/
    __init__.py          # 工厂：tools / mcp / compress / memory
    tools.py             # ToolsMiddleware
    mcp.py               # MCPMiddleware
    compression.py       # CompressionMiddleware
    memory.py            # MemoryMiddleware（session_id → Store，未实现）
  tools/
    registry.py
    function_tool.py
    mcp_client.py        # 同步（线程桥为唯一路径）
  compression/
    summarizer.py        # 同步
  multiagent/            # 同步编排（Member 基于 Agent + invoke + checkpointer 线程）
  cli/                   # Agent 子类示例（含原渲染逻辑）
examples/
  chatroom_315.py
pyproject.toml           # version 0.2.0，去 pytest-asyncio
```

## 10. 能力覆盖矩阵（泛用性评估）

| 能力 | 机制 | 覆盖 |
|------|------|------|
| tools | `ToolsMiddleware.before_llm` 注入 + `before_tool_call` 审批 + 分发 | ✅ |
| mcp | `MCPMiddleware` 同上，分发走线程桥 | ✅ |
| compress | `CompressionMiddleware.before_llm` 从 messages 估大小，超阈值摘要 | ✅ |
| 会话 memory | `MemoryMiddleware`：session_id → Store（before_turn 注入 / after_turn 写回） | ❌ 未实现 |
| 长期 memory | 中间件持外部 Store，`before_turn` 注入 / `after_turn` 写回 | ❌ 未实现 |
| LangGraph 原生持久化 | 逃生口：`compile_kwargs={"checkpointer": ...}` + `invoke(..., config={"configurable": {"thread_id": ...}})` | ✅ |
| model 切换 | 中间件换 `self.llm_client`（model 归端点） | ✅ |
| 子 Agent / multiagent | 子 Agent 包成 FunctionTool；multiagent 成员经 `invoke` + 独立 checkpointer 线程隔离历史 | ✅ |
| 错误重试 | `handle_error` → `Command(goto)` + `retry_policy` | ✅ |
| 流式 UI | `on_llm_content` / `on_content_end` 在 invoke 内同步触发 | ✅ |
| 人工审批 | `before_tool_call` 阻塞式 y/n（纯同步） | ✅ |

## 11. 纯同步化要点

- `LLMClient`：裸 httpx，只留 `invoke` / `stream`（同步生成器），删 `ainvoke/astream`，去 `openai` SDK 依赖。
- `MCPClient`：线程桥（`run_coroutine_threadsafe` + 后台常驻事件循环）成为**唯一**调用路径。
- `Compressor`：只留 `compress`。
- multiagent：`stream_discussion` 改同步生成器 `Iterator[Event]`。
- CLI：去 `asyncio.run`，`chat()` 全同步。
- pyproject：去 `pytest-asyncio` 依赖，加 `httpx`。

## 12. 版本与分支策略

- 分支：`master`（重写起点），dev 分支 v0.1 冻结不动。
- 版本：`v0.2.0`（tag 于实现完成时）。

## 13. 验证策略

- 不写单元测试（重构首版）。
- 冒烟验证：import 全通、`afcli` 可交互聊天、`chatroom_315` 示例可跑。

## 14. 后续扩展（不在首版）

- 中间件自定义状态（`Phase` 扩展机制）——首版明确**不做**，`Phase` 封闭。
- 中间件私有 channel——保持 state 最瘦，不做。
- 工具流式输出钩子 `on_tool_stream`。
- 流式中断的 `StreamStop` 事件钩子（替代中间件覆写 `handle_error`）——当前由中间件覆写 `handle_error` 认领。
