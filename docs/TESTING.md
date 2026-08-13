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

    def before_trace(self, input_text, session):        self.log.append(("before_trace",)); return super().before_trace(input_text, session)
    def after_trace(self, data, session):               self.log.append(("after_trace",));  return super().after_trace(data, session)
    def before_turn(self, data):                        self.log.append(("before_turn",)); return super().before_turn(data)
    def after_turn(self, data):                         self.log.append(("after_turn",));  return super().after_turn(data)
    def before_llm(self, request):                      self.log.append(("before_llm",)); return super().before_llm(request)
    def on_llm_reasoning(self, text):                   self.log.append(("on_llm_reasoning",)); super().on_llm_reasoning(text)
    def on_reasoning_end(self, reasoning):              self.log.append(("on_reasoning_end",)); super().on_reasoning_end(reasoning)
    def on_llm_content(self, text):                     self.log.append(("on_llm_content",)); super().on_llm_content(text)
    def on_content_end(self, content):                  self.log.append(("on_content_end",)); super().on_content_end(content)
    def after_llm(self, response):                      self.log.append(("after_llm",)); return super().after_llm(response)
    def before_tool_call(self, tool_calls):             self.log.append(("before_tool_call",)); return super().before_tool_call(tool_calls)
    def after_tool_result(self, name, result):          self.log.append(("after_tool_result",)); return super().after_tool_result(name, result)
    def handle_next(self, from_node, default):          self.log.append(("handle_next",)); return super().handle_next(from_node, default)
    def handle_error(self, error, node):                self.log.append(("handle_error",)); return super().handle_error(error, node)
    def on_state_changed(self, messages):               self.log.append(("on_state_changed",)); super().on_state_changed(messages)
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
| 3 | 多轮工具循环 | 三个脚本，两次带工具 | TOOLS 执行两次；`after_turn` 触发两次 |
| 4 | 工具结果进历史 | 单次带工具 | `after_tool_result` 收到 name+result；返回消息进历史 |

### T2 钩子链（tests/test_hooks.py）

| # | 场景 | 断言 |
|---|------|------|
| 5 | 记录全部钩子 | 单次 LLM 内顺序：before_llm→on_llm_reasoning→on_reasoning_end→on_llm_content→on_content_end→after_llm |
| 6 | 中间件在 `before_llm` 换 `self.llm_client` | 请求发到新端点，模型随之改变 |
| 7 | override `after_llm` 把 reasoning 前置 | 历史首条为 reasoning 消息，其后才是 AIMessage |
| 8 | 多块 content 流 | `on_content_end` 收到完整拼接文本；`on_reasoning_end` 同理 |
| 9 | 每次追加消息 | `on_state_changed` 每次触发且参数含新增消息 |
| 10 | 动态类继承 | 测试内定义 A/B 测试中间件，`Agent(llm_client=..., middlewares=[a(), b()])` → 钩子链顺序 A→B→base；同类重复叠加不冲突 |

### T3 流程控制（tests/test_flow_control.py）

| # | 场景 | 断言 |
|---|------|------|
| 11 | 默认路由 | 有 tool_calls→TOOLS、无→END |
| 12 | override `handle_next` 强制 END | 即使有 tool_calls 也不执行工具 |
| 13 | 参数透传 | `handle_next` 收到 `(Phase.LLM, 默认目标)` |

### T4 错误与中断（tests/test_errors.py）

| # | 场景 | 断言 |
|---|------|------|
| 14 | 假客户端抛异常 | `handle_error` 收到 NodeError；默认 goto END；invoke 不抛，返回最后消息 |
| 15 | override `handle_error` 重试一次 | 第二次成功，最终正常 END |
| 16 | `on_llm_content` 抛 `StreamStop` + override | partial 进历史、消息=partial、goto 目标生效 |
| 17 | `StreamStop` 无 override | 默认 END、partial 丢弃 |
| 18 | 假客户端抛 `KeyboardInterrupt` | 转 `StreamStop`；有 override 时 partial 保留 |

### T5 公共 API（tests/test_invoke_api.py）

| # | 场景 | 断言 |
|---|------|------|
| 19 | `invoke(input_text)` | 历史首条为 SystemMessage（有 system_prompt）/ HumanMessage |
| 20 | `invoke_messages(messages)` | 直接用给定消息，不注入 system |
| 21 | `compile_kwargs={"checkpointer": InMemorySaver()}` + 同 `config` thread_id 两次 invoke | 历史跨会话延续（原生持久化逃生口） |
| 22 | 编译后图 | 节点含 LLM/TOOLS，entry 为 LLM |

## 3. 运行

```bash
.venv/bin/python -m pytest tests/ -q
```

pyproject dev 依赖加回 `pytest >= 8`（去 pytest-asyncio）。
