# AgentFrame v0.2 测试设计（基类状态转换）

> 范围：仅基类（BaseAgent / Agent / LLMClient 结构体交换）。中间件、multiagent、CLI、compression 的测试后续补齐。
> 前提改造：`BaseAgent.__init__` 接收 `llm_client`（必填端点，持有 model），测试注入假客户端。

## 1. 测试基础设施（tests/conftest.py）

### ScriptedLLMClient

确定性驱动状态转换的假客户端。`_act_llm` 固定走 `stream`，所以只需实现 `stream`。

```python
class ScriptedLLMClient:
    def __init__(self, scripts, *, model="test-model", raise_at: int | None = None,
                 exc: BaseException | None = None) -> None:
        self.model = model
        self.scripts = list(scripts)          # 每次 stream 调用消耗一个脚本
        self.requests: list[LLMRequest] = []  # 记录收到的请求
        self.raise_at = raise_at              # 第 N 次调用抛异常
        self.exc = exc

    def stream(self, request: LLMRequest) -> Iterator[LLMStreamEvent]:
        self.requests.append(request)
        if self.raise_at is not None and len(self.requests) == self.raise_at:
            raise self.exc
        for event in self.scripts.pop(0):
            yield event
```

脚本元素为 `LLMStreamEvent`：`type="content"` 逐段文本、`type="reasoning"`、`type="done"`（带 tool_calls + usage）。可注入异常（含 `KeyboardInterrupt`）验证错误/中断路径。

### RecordingAgent

覆写全部 15 个钩子追加 `self.log` 后 `super()` 链，用于断言钩子调用顺序。

```python
class RecordingAgent(BaseAgent):
    def __init__(self, *, scripts, hooks=None, **kw):
        super().__init__(llm_client=ScriptedLLMClient(scripts), **kw)
        self.log: list[tuple[str, ...]] = []
        if hooks:
            for name, fn in hooks.items():
                setattr(self, name, fn.__get__(self, type(self)))

    def before_trace(self, input_text, session_id):      self.log.append(("before_trace",)); return super().before_trace(input_text, session_id)
    def after_trace(self, data, session_id):             self.log.append(("after_trace",));  return super().after_trace(data, session_id)
    def before_turn(self, messages):                     self.log.append(("before_turn",)); return super().before_turn(messages)
    def after_turn(self, messages):                      self.log.append(("after_turn",));  return super().after_turn(messages)
    def before_llm(self, request):                       self.log.append(("before_llm",)); return super().before_llm(request)
    def on_llm_reasoning(self, text):                    self.log.append(("on_llm_reasoning",)); super().on_llm_reasoning(text)
    def on_reasoning_end(self, reasoning):               self.log.append(("on_reasoning_end",)); super().on_reasoning_end(reasoning)
    def on_llm_content(self, text):                      self.log.append(("on_llm_content",)); super().on_llm_content(text)
    def on_content_end(self, content):                   self.log.append(("on_content_end",)); super().on_content_end(content)
    def after_llm(self, response):                       self.log.append(("after_llm",)); return super().after_llm(response)
    def before_tool_call(self, tool_calls):              self.log.append(("before_tool_call",)); return super().before_tool_call(tool_calls)
    def after_tool_result(self, name, result, tool_call_id): self.log.append(("after_tool_result",)); return super().after_tool_result(name, result, tool_call_id)
    def handle_next(self, from_node, default):           self.log.append(("handle_next",)); return super().handle_next(from_node, default)
    def handle_error(self, error, node):                 self.log.append(("handle_error",)); return super().handle_error(error, node)
```

### 辅助

- `make_agent(scripts, **kw) -> RecordingAgent`：组装注入假客户端。
- 工具执行需要注册：`agent.tool_registry.register(fn)`（`_act_tools` 经 registry 分发）。

## 2. 测试用例

### T1 状态转换（tests/test_state_transitions.py）

| # | 场景 | 脚本 | 断言 |
|---|------|------|------|
| 1 | 无工具直达 END | `[content("hi"), done]` | TOOLS 未执行；最终消息="hi"；钩子链 trace→turn→before_llm→content→on_content_end→after_llm→after_turn→after_trace |
| 2 | 一轮工具循环 | ①`[content("tool"), done(tool_calls=[bash])]` ②`[content("final"), done]` | 节点序列 LLM→TOOLS→LLM→END；ToolMessage 进历史；第二次 before_llm 的 request.messages 含 ToolMessage |
| 3 | 多轮工具循环 | 三个脚本，两次带工具 | TOOLS 执行两次；`before_turn`/`after_turn` 各触发与 LLM 调用数相同次 |
| 4 | 工具结果进历史 | 单次带工具 | `after_tool_result` 收到 name+result+tool_call_id；返回消息进历史 |

### T2 钩子链（tests/test_hooks.py）

| # | 场景 | 断言 |
|---|------|------|
| 5 | 记录全部钩子 | 单次 LLM 内顺序：before_llm→on_llm_reasoning→on_reasoning_end→on_llm_content→on_content_end→after_llm |
| 6 | 中间件在 `before_llm` 换 `self.llm_client` | 请求发到新端点，模型随之改变 |
| 7 | override `after_llm` 把 reasoning 前置 | 历史首条为 reasoning 消息，其后才是 AIMessage |
| 8 | 多块 content 流 | `on_content_end` 收到完整拼接文本；`on_reasoning_end` 同理 |
| 9 | 工具结果带 `tool_call_id` | `after_tool_result` 一轮多工具时各自收到对应 id |
| 10 | `after_turn` 写回 | 无工具回合改 AIMessage 内容，`invoke` 结果随之改变 |
| 11 | 动态类继承 | 测试内定义 A/B 测试中间件，`Agent(llm_client=..., middlewares=[a(), b()])` → 钩子链顺序 A→B→base；同类重复叠加不冲突 |

### T3 流程控制（tests/test_flow_control.py）

| # | 场景 | 断言 |
|---|------|------|
| 12 | 默认路由 | 有 tool_calls→TOOLS、无→END |
| 13 | override `handle_next` 强制 END | 即使有 tool_calls 也不执行工具 |
| 14 | 参数透传 | `handle_next` 收到 `(Phase.LLM, 默认目标)` |

### T4 错误与中断（tests/test_errors.py）

| # | 场景 | 断言 |
|---|------|------|
| 15 | 假客户端抛异常 | `handle_error` 收到 NodeError；默认 goto END；invoke 不抛，返回最后消息 |
| 16 | override `handle_error` 重试一次 | 第二次成功，最终正常 END |
| 17 | `on_llm_content` 抛 `StreamStop` + override | partial 进历史、消息=partial、goto 目标生效 |
| 18 | `StreamStop` 无 override | 默认 END、partial 丢弃 |
| 19 | 假客户端抛 `KeyboardInterrupt` | 转 `StreamStop`；有 override 时 partial 保留 |
| 20 | 工具 dispatch 抛异常 + 重试 | 后续 `after_tool_result` 仍拿到当次工具的正确 id |

### T5 公共 API（tests/test_invoke_api.py）

> `invoke` 是唯一公开入口（文本入、str 出）；历史场景 = `session_id` 构造参数（checkpointer 默认内存 InMemorySaver，session 映射 thread_id 自动恢复）。

| # | 场景 | 断言 |
|---|------|------|
| 21 | `invoke(input_text)` | 历史首条为 SystemMessage（有 system_prompt）/ HumanMessage |
| 22 | 同 session_id 两轮 invoke | 历史跨回合延续：第二轮请求带前轮 AIMessage；system 按 id 去重不重复 |
| 23 | 共享 checkpointer 不同 session_id | 历史互不可见，隔离生效 |
| 24 | 共享 checkpointer 同 session_id | 跨 agent 实例共享历史（a2 首轮请求带 a1 写入内容） |
| 25 | 自定义 checkpointer | 传入 saver 被采用，`agent.checkpointer is saver`，checkpoint 落库 |
| 26 | 编译后图 | 节点含 LLM/TOOLS，entry 为 LLM |

### T6 LLMClient 解析（tests/test_llm_client.py）

> 范围：`LLMClient` 的纯解析函数与流式协议（非流式 `invoke()` 与 SSE `stream()`）。
> 手段：`httpx.MockTransport` 注入，无真实网络；`LLMClient(transport=...)` 构造。

| # | 场景 | 断言 |
|---|------|------|
| 27 | `invoke()` 解析非流式响应 | content / usage / finish_reason / model 正确映射 |
| 28 | `invoke()` tool_calls | 转 langchain `args` 格式（A4 回归），`finish_reason="tool_calls"` |
| 29 | tool_calls arguments 截断 | 安全解析为 `{}`，不抛 `ValueError` |
| 30 | `stream()` 流式 content + 末帧 usage | `done` 事件携带聚合后的 usage（A2 回归） |
| 31 | `stream()` finish_reason | content 分片携带的 `finish_reason` 透传到 `done` |
| 32 | reasoning 多厂商字段 | `reasoning_content`/`reasoning`/`reasoning_text` 均产生 `reasoning` 事件 |
| 33 | tool_calls 分片聚合 | 多分片 id/name/arguments 拼接后进 `done.tool_calls` |
| 34 | 截断流式 arguments | 安全解析为 `{}` |
| 35 | 无 arguments 的 tool_calls | 空字符串安全解析为 `{}` |
| 36 | SSE 干扰行 | 空行 / `event:` 非 data 行被跳过，content 正常产出 |
| 37 | 请求体携带 tools/temperature/max_tokens | body 中字段正确透传，`stream` 标志正确 |
| 38 | `api_key` | 请求头带 `Authorization: Bearer <key>` |
| 39 | `defaults` 合并 | 默认参数进 body，request 显式字段优先 |
| 40 | 生命周期 | `close()` 幂等；上下文管理器可用 |

### T7 错误语义（tests/test_errors.py 补充）

| # | 场景 | 断言 |
|---|------|------|
| 41 | 首轮失败 `invoke` 结果 | 返回空串，不回显用户输入（B1 回归） |
| 42 | reasoning 阶段 `StreamStop` | `partial_reasoning` 含中断触发分片（B4 回归） |

### T8 补齐覆盖（test_hooks.py / test_invoke_api.py）

| # | 场景 | 断言 |
|---|------|------|
| 43 | agent 层 usage/finish_reason 透传 | `after_llm` 收到的 `response.usage` / `response.finish_reason` 由流式 `done` 事件填充（A2 回归） |

### T9 日志中间件（tests/test_logging.py）

> 手段：标准 `logging.Logger` + pytest `caplog`；工厂 `log(logger)()`，事件经 `super()` 续链。

| # | 场景 | 断言 |
|---|------|------|
| 44 | 无工具 invoke | 记录 `trace start/end`、`turn start/end`、`llm`（含 session_id） |
| 45 | 带工具 invoke | 记录 `tool start`（调用前，name/id 列表）与 `tool end`（执行后，result_len） |
| 46 | 客户端抛异常 | 记录 `error` 事件（node / 异常类型 / 消息） |

## 3. 运行

```bash
.venv/bin/python -m pytest tests/ -q
```

pyproject dev 依赖加回 `pytest >= 8`（去 pytest-asyncio）。
