# Live MCP Environment MVP 方案

> 目标：在当前 `ReplayMCPExecutor` 之外，补一套 PROVE-style live MCP 功能等价子集。第一版减少 server 数量，但保留论文方案的必要机制：subprocess/stdio MCP server、session-scoped state isolation、grounded state-machine data synthesis、fresh reset replay validation、五组件 programmatic reward。

本文档是开发规格，不是研究笔记；Claude 实现时以 `必须`、`不得`、`验收` 为准，`可选` 只表示不阻塞第一版验收。

## 1. 背景判断

当前项目已有：

- `ReplayMCPExecutor`：基于 `EpisodeSeed.oracle_trace` 做 deterministic replay，不连接真实 MCP server。
- `MCPToolEnvironment`：把 replay executor、parser、reward info 封装成环境接口。
- `schemashift_reward_fn.py`：当前 GRPO 入口只评估第一个 oracle action，本质仍是 next-action reward。

第一版 live MCP 的完整闭环：

```text
model output
-> parse action
-> validate schema
-> call live MCP server
-> receive real observation / error
-> update session state
-> continue multi-turn rollout
-> compute execution-aware trajectory reward
```

第一版不是复刻论文的 20 个 server 规模，而是对齐论文的功能机制。server 数量可以少，但不能缺少 live execution、状态隔离、grounded query、state-machine orchestration、oracle validation 和五组件 reward。

与论文事实对齐：

- 论文的 MCP 环境是独立 subprocess，通过 stdio 通信。
- 同一套 live MCP environments 同时用于数据合成和 RL training。
- 每个 rollout 使用 session-scoped state isolation，并支持 deterministic reset。
- 数据合成由 dependency graph、live-state sampling 和 state-machine teacher 驱动。
- 每条 conversation 在 freshly reset environment 上 replay validate。
- reward 是 validity、coverage、adaptive efficiency、tool-name、argument-value 五组件。

## 2. 设计原则

1. 保留 replay 路线，不用 live 替换 replay。
2. 第一版做功能完整的纵向闭环，但只实现少量 server。
3. server 通过配置注册，不把 calendar/shopping 写死进 agent loop。
4. reward 不直接读各 server 原始返回，统一读 normalized execution result。
5. 所有 rollout 必须 session 隔离，支持 deterministic reset。
6. trace 必须完整记录，可用于 debug、replay、eval 和数据沉淀。
7. 训练脚本继续遵守项目约束：不写死 GPU、batch、TP、机器绝对路径。

## 3. 总体架构

```text
src/live_mcp/
  manager.py          # server lifecycle / session lifecycle / schema discovery
  transport.py        # subprocess stdio transport and test-only in-process transport
  executor.py         # tool_call -> normalized execution result
  agent_loop.py       # multi-turn rollout loop
  session.py          # session id / reset seed / state scope
  schema_registry.py  # tool schema registry and validation
  dependency_graph.py # tool dependency graph and chain extraction
  state_seeder.py     # deterministic initial state generation
  sampler.py          # live-state grounded task sampling
  query_generator.py  # structured task -> natural language query fallback
  teacher.py          # teacher adapter and deterministic fallback teacher
  oracle.py           # oracle planning and validation
  orchestrator.py     # data synthesis state machine
  reward.py           # execution-aware reward composer
  trace.py            # rollout trace recorder
  errors.py           # normalized error taxonomy

src/live_mcp/servers/
  calendar/
  shopping/

configs/live_mcp/
  calendar.yaml
  shopping.yaml
  suite_mvp.yaml
```

完整调用链：

```text
LiveMCPManager.start_suite()
-> manager.create_session(seed)
-> manager.discover_tools(session)
-> DependencyGraphBuilder.extract_chains(tools)
-> StateSeeder.reset(session, seed)
-> LiveStateSampler.sample_task(session)
-> TeacherAdapter.generate_query(structured_task, sampling_context)
-> QueryGenerator.validate_query(query, structured_task)
-> OraclePlanner.plan(structured_task)
-> OracleValidator.validate(live_task)
-> MCPToolsAgentLoop.rollout(task, tools, session)
-> LiveMCPExecutor.execute(tool_call, session)
-> TraceRecorder.append(...)
-> RewardComposer.compute(trace, task)
-> manager.reset_session(session)
```

## 4. 核心数据结构

### 4.1 SessionSpec

```python
@dataclass
class SessionSpec:
    session_id: str
    suite_name: str
    server_names: list[str]
    seed: int
    created_at: str
    max_turns: int = 8
    metadata: dict[str, Any] = field(default_factory=dict)
```

约束：

- `session_id` 是所有 tool call 的隔离边界。
- 同一个 `seed` 下 reset 后初始状态必须可复现。
- 一个 session 可以绑定一个或多个 server，为后续跨 server task 预留。

### 4.2 ToolCall

```python
@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    call_id: str
    raw_text: str = ""
```

约束：

- `arguments` 必须是 dict。非 dict 在 parser 层标记 invalid，executor 不尝试猜测。
- `name` 是模型看到的 tool name，可通过 schema perturbation 映射到 canonical name。

### 4.3 ToolExecutionResult

所有 server 返回必须归一化成这个结构，reward、trace、agent loop 只依赖它。

```python
@dataclass
class ToolExecutionResult:
    success: bool
    tool_name: str
    canonical_tool_name: str
    call_id: str
    session_id: str
    observation: dict[str, Any] | str | None
    error_type: str | None
    error_message: str
    schema_valid: bool
    state_changed: bool
    latency_ms: int
    metadata: dict[str, Any] = field(default_factory=dict)
```

`error_type` 统一取值：

```text
parse_error
unknown_tool
schema_invalid
argument_invalid
parallel_not_supported
precondition_failed
execution_error
timeout
permission_denied
state_conflict
server_unavailable
```

### 4.4 LiveTask

```python
@dataclass
class LiveTask:
    task_id: str
    source: str
    suite_name: str
    user_prompt: str
    session_id: str
    session_seed: int
    target_servers: list[str]
    visible_tools: list[dict[str, Any]]
    required_tools: list[str]
    expected_outcome: dict[str, Any]
    success_criteria: list[dict[str, Any]]
    oracle_program: "OracleProgram"
    sampling_context: dict[str, Any]
    max_turns: int
    difficulty: str = "easy"
    task_type: str = ""
    hidden_tools: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
```

说明：

- `oracle_program` 是系统验证用的标准工具计划，不进入模型 prompt。
- `sampling_context` 记录 query 中实体的 live-state provenance，必须可审计。
- `hidden_tools` 用于 missing-function/no-tool 样本，模型不可见。

### 4.5 RolloutTrace

```python
@dataclass
class RolloutTrace:
    trace_id: str
    task_id: str
    session_id: str
    model_name: str
    started_at: str
    ended_at: str | None
    turns: list["TraceTurn"]
    final_status: str
    reward: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)
```

```python
@dataclass
class TraceTurn:
    turn_idx: int
    prompt_hash: str
    model_output: str
    parsed_action_type: str
    tool_calls: list[ToolCall]
    execution_results: list[ToolExecutionResult]
    observation_text: str
    done: bool
```

`final_status` 取值：

```text
success
model_final
model_final_incorrect
max_turns
parse_error
execution_failed
reward_failed
```

## 5. 配置规范

### 5.1 单 server 配置

```yaml
name: calendar
type: local_subprocess
enabled: true

command:
  argv: ["python", "-m", "src.live_mcp.servers.calendar.server"]
  cwd: "."
  env:
    PYTHONUNBUFFERED: "1"

transport:
  kind: stdio
  startup_timeout_s: 20
  request_timeout_s: 10

session:
  reset_mode: deterministic_seed
  max_sessions: 64
  default_seed: 42

tools:
  discovery:
    - list_events
  readonly:
    - list_events
  mutating:
    - create_event
    - update_event
    - delete_event

sampler:
  class_path: src.live_mcp.servers.calendar.sampler.CalendarSampler

reward_profile: calendar_basic
```

### 5.2 Suite 配置

```yaml
suite_name: live_mcp_mvp
servers:
  - configs/live_mcp/calendar.yaml
  - configs/live_mcp/shopping.yaml

rollout:
  max_turns: 8
  max_parallel_tool_calls: 4
  observation_max_chars: 4096
  stop_on_execution_error: false

reward:
  profile: mvp_default
  weights:
    validity: 0.25
    coverage: 0.25
    efficiency: 0.10
    tool_selection: 0.20
    argument_value: 0.20

trace:
  output_dir: data/live_mcp/traces
  save_success: true
  save_failure: true
```

配置约束：

- `cwd` 必须是项目根目录相对路径。
- 不允许默认写死 `/data/...`、`/mnt/...` 等机器路径。
- 所有 timeout 必须显式配置。
- server 新增时只加配置、sampler、reward profile，不改 agent loop。

### 5.3 后端切换配置

训练和 smoke 必须通过配置切换 replay/live，不允许在业务代码里散落 `if use_live`。

```yaml
environment:
  backend: live  # replay | live

replay:
  episode_seeds_path: data/toucan/episode_seeds.jsonl

live:
  suite: configs/live_mcp/suite_mvp.yaml
  transport: subprocess_stdio
  task_file: data/live_mcp/tasks/live_mcp_mvp.jsonl
  trace_dir: data/live_mcp/traces
```

实现约束：

- 默认 backend 仍为 `replay`，不得破坏现有训练链路。
- `live.transport=subprocess_stdio` 是 smoke 验收默认值。
- `live.transport=in_process` 只能用于单测或显式 debug 参数。

### 5.4 第一版输出文件

数据生成脚本输出：

```text
data/live_mcp/tasks/live_mcp_mvp.jsonl
```

每行是一个 `LiveTask` JSON，必须包含：

```text
task_id
source
suite_name
session_seed
user_prompt
visible_tools
required_tools
hidden_tools
success_criteria
oracle_program
sampling_context
difficulty
task_type
metadata
```

rollout / validation trace 输出：

```text
data/live_mcp/traces/{suite_name}/{date}/{trace_id}.json
```

## 6. Manager 和 Transport 接口

```python
class LiveMCPManager:
    def __init__(self, suite_config: LiveMCPSuiteConfig):
        ...

    def start_suite(self) -> None:
        """启动 suite 中所有 enabled server，并做 healthcheck。"""

    def stop_suite(self) -> None:
        """停止所有 server，释放 subprocess / socket / temp state。"""

    def create_session(self, seed: int | None = None) -> SessionSpec:
        """创建隔离 session，并初始化每个 server 的 session state。"""

    def reset_session(self, session_id: str, seed: int | None = None) -> None:
        """重置 session 到 deterministic initial state。"""

    def close_session(self, session_id: str) -> None:
        """关闭 session 并清理状态。"""

    def discover_tools(self, session_id: str) -> list[dict[str, Any]]:
        """返回当前 session 可见工具 schema。"""

    def healthcheck(self) -> dict[str, bool]:
        """返回每个 server 的健康状态。"""
```

实现要求：

- `start_suite()` 不做训练逻辑，只管 server lifecycle。
- `create_session()` 必须返回全局唯一 `session_id`。
- `reset_session()` 是测试和 RL rollout 可复现的关键接口，第一版必须有单测。
- server 挂掉时 manager 返回 `server_unavailable`，不能让 agent loop 直接崩溃。

论文里的 MCP server 是独立 subprocess，并通过 stdio 通信。第一版必须实现 `SubprocessStdioTransport`，否则不能称为 live MCP 对齐实现。`InProcessTransport` 只允许用于单测和快速 debug，不能作为 smoke 验收的默认 transport。

```python
class MCPTransport(Protocol):
    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...

    def request(
        self,
        method: str,
        params: dict[str, Any],
        timeout_s: float,
    ) -> dict[str, Any]:
        ...
```

```python
class SubprocessStdioTransport:
    def __init__(
        self,
        argv: list[str],
        cwd: Path,
        env: dict[str, str],
        startup_timeout_s: float,
    ):
        ...
```

第一版验收要求：

- `calendar` 和 `shopping` server 的 smoke 必须通过 subprocess stdio transport 跑通。
- transport 必须支持 startup timeout、request timeout、stderr capture。
- server crash 必须归一化为 `server_unavailable`，不能让上层 traceback。
- in-process transport 只能出现在 `tests/live_mcp/` 或显式 `--transport in_process` debug 模式。

## 7. Executor 接口

```python
class LiveMCPExecutor:
    def __init__(
        self,
        manager: LiveMCPManager,
        schema_registry: SchemaRegistry,
        timeout_s: float,
    ):
        ...

    def execute(
        self,
        session_id: str,
        tool_call: ToolCall,
    ) -> ToolExecutionResult:
        """执行单个 tool call。"""

    def execute_many(
        self,
        session_id: str,
        tool_calls: list[ToolCall],
        mode: str = "sequential",
    ) -> list[ToolExecutionResult]:
        """执行多个 tool call。第一版支持 sequential。"""
```

执行顺序：

```text
lookup tool schema
-> map perturbed name to canonical name
-> validate arguments
-> send request to server
-> normalize response
-> classify error
-> return ToolExecutionResult
```

第一版要求：

- unknown tool 不调用 server，直接返回 `unknown_tool`。
- schema invalid 不调用 server，直接返回 `schema_invalid`。
- server timeout 返回 `timeout`，不抛到上层。
- mutating tool 必须标记 `state_changed`。

## 8. SchemaRegistry 接口

```python
class SchemaRegistry:
    def register_tools(
        self,
        server_name: str,
        tools: list[dict[str, Any]],
        name_map: dict[str, str] | None = None,
    ) -> None:
        ...

    def get_schema(self, tool_name: str) -> dict[str, Any] | None:
        ...

    def canonical_name(self, visible_name: str) -> str:
        ...

    def validate_arguments(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> "SchemaValidationResult":
        ...
```

```python
@dataclass
class SchemaValidationResult:
    valid: bool
    missing_required: list[str]
    unexpected_keys: list[str]
    type_errors: list[str]
    enum_errors: list[str]
```

约束：

- 复用当前 `ComponentReward` / `schema_perturber` 的 name_map、enum_map 思路。
- 校验逻辑不能依赖 server domain。
- validator 必须 bounded linear，避免复杂嵌套 schema 上出现 O(N^2) 扫描。

## 9. Agent Loop 接口

```python
class MCPToolsAgentLoop:
    def __init__(
        self,
        manager: LiveMCPManager,
        executor: LiveMCPExecutor,
        parser: ActionParser,
        trace_recorder: TraceRecorder,
        config: AgentLoopConfig,
    ):
        ...

    def rollout(
        self,
        task: LiveTask,
        model: "GenerationBackend",
    ) -> RolloutTrace:
        ...
```

```python
@dataclass
class AgentLoopConfig:
    max_turns: int = 8
    observation_max_chars: int = 4096
    stop_on_parse_error: bool = True
    stop_on_execution_error: bool = False
    allow_parallel_tool_calls: bool = False
```

每轮流程：

```text
build prompt
-> model.generate(prompt)
-> parse action
-> if final_answer: stop
-> if tool_call: executor.execute(...)
-> append observation
-> continue
```

与 verl 集成的第一版策略：

- 第一阶段只实现离线 smoke runner，不改 verl rollout worker。
- 第二阶段接 `SchemaShiftTaskRunner`，让 live backend 替代当前只评估第一个 oracle action 的路径。
- 第三阶段再做 batch sessions，减少 server startup/reset 开销。

第一版遇到模型输出多个 tool calls 时：

```text
allow_parallel_tool_calls=false:
  返回 schema_invalid 或 parallel_not_supported，由 reward 给低分。

allow_parallel_tool_calls=true:
  按输入顺序串行执行，trace 保留原始并行意图。
```

## 10. 数据合成组件

### 10.1 LiveStateSampler

```python
class LiveStateSampler(Protocol):
    def sample_task(
        self,
        session_id: str,
        manager: LiveMCPManager,
        difficulty: str,
        rng: random.Random,
    ) -> LiveTask:
        ...
```

规则：

- 只用 readonly/discovery 工具采样实体。
- task 中出现的 entity_id 必须来自当前 live session。
- 不允许编造 id。
- sampler 输出 `expected_outcome` 和 `success_criteria`，用于 reward。

### 10.2 DependencyGraphBuilder

第一版必须有 dependency graph，但不要求自动 LLM 探测。用配置化图即可，保证机制完整、结果可复现。

```python
class DependencyGraphBuilder:
    def build(
        self,
        server_name: str,
        tool_schemas: list[dict[str, Any]],
        config_edges: list["DependencyEdge"],
    ) -> "ToolDependencyGraph":
        ...

    def extract_chains(
        self,
        graph: "ToolDependencyGraph",
        min_len: int = 2,
        max_len: int = 5,
    ) -> list["ToolChain"]:
        ...
```

```python
@dataclass
class DependencyEdge:
    source_tool: str
    target_tool: str
    relation: str  # "explicit" | "implicit"
    source_output_path: str = ""
    target_argument_path: str = ""
    description: str = ""

@dataclass
class ToolChain:
    chain_id: str
    server_name: str
    tools: list[str]
    edges: list[DependencyEdge]
    difficulty: str
```

Calendar 第一版配置：

```yaml
dependency_graph:
  edges:
    - source_tool: list_events
      target_tool: update_event
      relation: explicit
      source_output_path: events[].event_id
      target_argument_path: event_id
    - source_tool: list_events
      target_tool: delete_event
      relation: explicit
      source_output_path: events[].event_id
      target_argument_path: event_id
    - source_tool: create_event
      target_tool: list_events
      relation: implicit
      description: created event should become visible in later listing
```

Shopping 第一版配置：

```yaml
dependency_graph:
  edges:
    - source_tool: search_products
      target_tool: add_to_cart
      relation: explicit
      source_output_path: products[].product_id
      target_argument_path: product_id
    - source_tool: add_to_cart
      target_tool: checkout
      relation: implicit
      description: checkout requires non-empty cart
    - source_tool: checkout
      target_tool: get_order
      relation: explicit
      source_output_path: order_id
      target_argument_path: order_id
```

### 10.3 StateSeeder

```python
class StateSeeder(Protocol):
    def seed_state(
        self,
        server_name: str,
        session_id: str,
        seed: int,
    ) -> dict[str, Any]:
        ...

    def reset_state(
        self,
        server_name: str,
        session_id: str,
        seed: int,
    ) -> dict[str, Any]:
        ...
```

要求：

- 同一 `server_name + seed` 必须生成相同初始状态。
- 不同 `session_id` 之间状态对象不能共享引用。
- 初始状态必须覆盖 sampler 需要的实体类型。

### 10.4 QueryGenerator / TeacherAdapter

```python
class QueryGenerator:
    def render(
        self,
        structured_task: "StructuredTask",
        style: str = "default",
    ) -> str:
        ...

    def validate_query(
        self,
        query: str,
        structured_task: "StructuredTask",
    ) -> "QueryValidationResult":
        ...
```

```python
@dataclass
class StructuredTask:
    task_id: str
    server_name: str
    tool_chain: ToolChain
    slots: dict[str, Any]
    user_visible_slots: list[str]
    hidden_slots: list[str]
    success_criteria: list[dict[str, Any]]
    required_tools: list[str]
    difficulty: str
```

`QueryGenerator` 是确定性 fallback。第一版同时必须提供 `TeacherAdapter`，使 state machine 可以按论文方式驱动 query generation、assistant turn、recovery 和 continuation。没有外部 teacher LLM 时，使用 `DeterministicTeacherAdapter` 包装模板能力，不能绕开状态机。

```python
class TeacherAdapter(Protocol):
    def generate_query(
        self,
        task: StructuredTask,
        sampling_context: dict[str, Any],
        persona: str,
        reference_date: str,
        information_level: str,
    ) -> str:
        ...

    def generate_assistant_turn(
        self,
        messages: list[dict[str, str]],
        tool_schemas: list[dict[str, Any]],
        state: "OrchestratorState",
    ) -> str:
        ...

    def decide_continuation(
        self,
        trace: RolloutTrace,
        min_turns: int,
        max_turns: int,
        rng: random.Random,
    ) -> str:
        """Return one of: 'end', 'follow_up', 'clarify'."""

    def recover(
        self,
        failed_result: ToolExecutionResult,
        messages: list[dict[str, str]],
        tool_schemas: list[dict[str, Any]],
    ) -> str:
        ...
```

第一版 teacher 策略：

```text
required: deterministic_template_teacher
optional_non_blocking: local_or_remote_llm_teacher
```

验收只依赖 `deterministic_template_teacher`。LLM teacher 可以实现，但不得成为生成数据、跑 smoke 或跑单测的必要条件。

确定性 fallback 使用模板渲染：

```yaml
query_templates:
  calendar_update_existing_event:
    - "请把{old_time}的{title}改到{new_time}，然后告诉我更新后的时间。"
    - "我想把{title}从{old_time}挪到{new_time}，帮我处理一下。"
  shopping_buy_product:
    - "帮我找一个{max_price}美元以内的{category}，加入购物车并完成结账。完成后告诉我订单号。"
```

验证要求：

- query 必须包含所有 `user_visible_slots`。
- query 不得包含 `hidden_slots`，例如内部 `event_id`、`product_id`，除非模板明确要求暴露。
- query 不得引用 sampler 没有返回的实体。
- teacher 生成后仍要跑 `validate_query`，防止实体幻觉、槽位丢失或 hidden id 泄漏。
- teacher 生成失败时允许 fallback regeneration，最多重试 `max_regenerations` 次。

### 10.5 OraclePlanner / OracleValidator

```python
class OraclePlanner:
    def plan(self, task: StructuredTask) -> "OracleProgram":
        ...

@dataclass
class OracleCall:
    tool_name: str
    arguments: dict[str, Any]
    save_as: str = ""

@dataclass
class OracleProgram:
    task_id: str
    calls: list[OracleCall]
    success_criteria: list[dict[str, Any]]
```

```python
class OracleValidator:
    def validate(
        self,
        task: StructuredTask,
        oracle_program: OracleProgram,
        manager: LiveMCPManager,
        executor: LiveMCPExecutor,
        seed: int,
    ) -> "OracleValidationResult":
        ...

@dataclass
class OracleValidationResult:
    valid: bool
    execution_results: list[ToolExecutionResult]
    final_state: dict[str, Any]
    failed_criteria: list[dict[str, Any]]
    error: str = ""
```

验证规则：

- 必须在 freshly reset session 上执行，不复用采样时的脏状态。
- 任一 oracle call schema invalid 或 execution failed，整条数据丢弃。
- 任一 success criteria 不满足，整条数据丢弃。
- validation trace 必须保存，便于定位模板或 server bug。

### 10.6 StateMachineOrchestrator

第一版需要有显式状态机，并且必须覆盖论文里的 query、turn、tool-execution、response、continuation、recovery 状态组。可以不依赖外部 LLM，但不能省略 teacher 接口；默认用 `DeterministicTeacherAdapter` 生成 query、assistant turn、recovery 和 continuation。

```python
class StateMachineOrchestrator:
    def generate_one(
        self,
        server_name: str,
        seed: int,
        difficulty: str,
    ) -> "LiveTask":
        ...

    def generate_many(
        self,
        server_name: str,
        count: int,
        seed: int,
        difficulty_mix: dict[str, float],
    ) -> list["LiveTask"]:
        ...
```

状态流：

```text
INIT_SESSION
-> DISCOVER_TOOLS
-> BUILD_DEPENDENCY_GRAPH
-> SAMPLE_STATE
-> SELECT_CHAIN
-> BIND_SLOTS
-> RENDER_QUERY
-> PLAN_ORACLE
-> VALIDATE_ORACLE
-> TEACHER_PROCESSING
-> TOOL_EXECUTION
-> RESPONSE_CLASSIFICATION
-> RECOVERY_OR_CONTINUATION
-> REPLAY_VALIDATE_TRACE
-> WRITE_TASK
```

状态说明：

```text
RENDER_QUERY:
  TeacherAdapter.generate_query 生成 grounded query，并通过 QueryGenerator.validate_query。

TEACHER_PROCESSING:
  TeacherAdapter.generate_assistant_turn 生成下一步 assistant action。

TOOL_EXECUTION:
  通过 LiveMCPExecutor 调 subprocess MCP server。

RESPONSE_CLASSIFICATION:
  将 execution result 分类为 success / partial_success / failure。

RECOVERY_OR_CONTINUATION:
  failure 时调用 TeacherAdapter.recover；
  success 时调用 TeacherAdapter.decide_continuation；
  最终受 min_turns / max_turns 约束。

REPLAY_VALIDATE_TRACE:
  在 freshly reset session 上重放完整 trace，错误率超过阈值则丢弃。
```

第一版不允许跳过 `VALIDATE_ORACLE` 或 `REPLAY_VALIDATE_TRACE`。

## 11. RewardComposer 接口

```python
class RewardComposer:
    def compute(
        self,
        task: LiveTask,
        trace: RolloutTrace,
        final_state: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        ...
```

第一版必须实现五个 reward components：

```text
validity:
  工具存在、schema 合法、server 执行成功

coverage:
  必要工具步骤是否按 dependency graph 顺序完成

efficiency:
  是否在复杂度自适应预算内完成，明显多余调用扣分

tool_selection:
  是否调用了任务所需工具，是否避免明显无关工具

argument_value:
  entity_id、枚举值、时间、数量等关键参数是否正确
```

第一版总分：

```python
score = (
    0.25 * validity
    + 0.25 * coverage
    + 0.10 * efficiency
    + 0.20 * tool_selection
    + 0.20 * argument_value
)
```

`validity`：

```text
每个 tool call 最高 1 分：
  0.33 tool name exists
  0.33 required arguments present and JSON types valid
  0.34 live execution succeeds
取所有 model tool calls 的平均值。
```

`coverage`：

```text
按 oracle_program / dependency graph 检查必要步骤是否覆盖。
一个 model call 匹配 GT step 需要：
  tool name 相同
  GT required argument keys 都存在
  依赖顺序不违反 dependency graph
coverage = matched_gt_steps / total_gt_steps
```

`adaptive efficiency`：

```python
budget = gt_call_count + ceil(alpha * gt_call_count)
excess = max(0, model_call_count - budget)
efficiency = -lambda_eff * excess
```

默认：

```python
alpha = 0.5
lambda_eff = 0.05
```

`tool_selection`：

```text
model 调用的工具名出现在 required_tools / oracle tool set 中就给分。
tool_selection = correct_tool_name_calls / max(1, model_call_count)
```

`argument_value`：

```text
对已按 tool name + keys 对齐的 call，比较关键参数值。
例如 event_id、product_id、quantity、time、category、max_price。
argument_value = matched_values / total_checked_values
```

最终状态检查不单独作为第六个 reward component，而是作为 `coverage` 和 validation 的硬门槛：

```text
如果 final_state 不满足 success_criteria：
  coverage 不能为 1.0
  trace 标记 final_status=execution_failed 或 model_final_incorrect
```

输出规范：

```python
{
    "score": 0.0,
    "component_validity": 0.0,
    "component_coverage": 0.0,
    "component_efficiency": 0.0,
    "component_tool_selection": 0.0,
    "component_argument_value": 0.0,
    "num_turns": 0.0,
    "num_tool_calls": 0.0,
    "num_execution_errors": 0.0,
}
```

注意：

- 返回给 verl validation aggregation 的字段必须是 scalar 或 string，不能返回 nested dict/list。
- 详细诊断写 trace，不写进 reward dict。

## 12. TraceRecorder 接口

```python
class TraceRecorder:
    def start(self, task: LiveTask, model_name: str) -> RolloutTrace:
        ...

    def append_turn(self, trace: RolloutTrace, turn: TraceTurn) -> None:
        ...

    def finish(
        self,
        trace: RolloutTrace,
        final_status: str,
        reward: dict[str, float],
    ) -> None:
        ...

    def save(self, trace: RolloutTrace) -> Path:
        ...
```

trace 文件必须包含：

- task
- visible tool schemas hash
- session seed
- every model output
- every parsed action
- every execution result
- final reward components

## 13. MVP Server 规范

### 13.1 Calendar server

工具：

```text
list_events(date_range?)
create_event(title, start_time, end_time, attendees?)
update_event(event_id, fields)
delete_event(event_id)
```

状态：

```python
{
    "events": {
        "evt_001": {
            "title": "team sync",
            "start_time": "...",
            "end_time": "...",
            "attendees": [...]
        }
    }
}
```

必须覆盖的任务：

- 查事件后修改事件。
- 创建新事件。
- 删除指定事件。
- 遇到不存在 event_id 时返回 `precondition_failed`。

### 13.2 Shopping server

工具：

```text
search_products(query?, category?, max_price?)
add_to_cart(product_id, quantity)
remove_from_cart(product_id)
checkout()
get_order(order_id)
```

状态：

```python
{
    "products": {...},
    "cart": [...],
    "orders": {...}
}
```

必须覆盖的任务：

- 搜索商品后加入购物车。
- 多商品加入购物车后结账。
- 查询订单。
- 库存不足时返回 `precondition_failed`。

## 14. 测试规范

第一版必须补以下测试：

```text
tests/live_mcp/test_manager.py
tests/live_mcp/test_executor.py
tests/live_mcp/test_schema_registry.py
tests/live_mcp/test_dependency_graph.py
tests/live_mcp/test_state_seeder.py
tests/live_mcp/test_sampler.py
tests/live_mcp/test_query_generator.py
tests/live_mcp/test_oracle.py
tests/live_mcp/test_orchestrator.py
tests/live_mcp/test_agent_loop.py
tests/live_mcp/test_reward.py
tests/live_mcp/test_trace.py
```

关键验收：

- `reset_session(seed=1)` 两次得到相同初始状态。
- 两个 session 并行修改状态互不影响。
- unknown tool 不触发 server 调用。
- schema invalid 不触发 server 调用。
- mutating tool 后 `state_changed=True`。
- dependency graph 能从配置提取 2-5 步 chain。
- sampler 生成的 entity_id 必须存在于当前 session。
- query generator 不泄漏 hidden id，除非模板显式要求。
- oracle planner 生成的 plan 能在 freshly reset session 验证通过。
- orchestrator 能生成至少 10 条 valid LiveTask。
- missing-function task 能隐藏必要工具，并触发 clarification / abstention 目标。
- distractor injection 能注入跨 server 无关工具，且 required_tools 不被污染。
- agent loop 能完成至少一个 calendar 多步任务。
- reward 包含 validity、coverage、efficiency、tool_selection、argument_value 五组件。
- reward dict 不包含 nested dict/list。
- trace 能落盘，且包含完整 turn 信息。

轻量命令：

```bash
python -m pytest tests/live_mcp/
python -m compileall src scripts tests
git diff --check
```

## 15. 与现有代码的衔接

短期不改现有 replay 训练链路：

```text
src/envs/replay_mcp_executor.py       # 保留
src/envs/mcp_tool_environment.py      # 保留
src/reward/schemashift_reward_fn.py   # 保留 next-action 路线
```

新增 live 路线：

```text
src/live_mcp/*                        # 新增
scripts/generate_live_mcp_tasks.py    # 新增
scripts/run_live_mcp_smoke.py         # 新增
configs/live_mcp/*.yaml               # 新增
tests/live_mcp/*                      # 新增
```

接入顺序：

```text
generate LiveTask dataset
-> Live MCP smoke
-> MCPToolsAgentLoop offline runner
-> live trace reward validation
-> custom SchemaShiftTaskRunner integration
-> GRPO live mini-batch smoke
```

## 16. Claude 开发顺序

Claude 开发时必须按阶段提交，每阶段都要有对应测试。不得先接 GRPO，不得先做 LLM teacher，不得先扩 server 数量。

### Phase A：框架骨架

- 新增 `src/live_mcp/` 包。
- 定义 dataclass、Protocol、error taxonomy。
- 实现 local subprocess stdio transport；in-process transport 只作为单测 mock。
- 完成 manager/executor/schema/dependency_graph/state_seeder/trace 单测。

### Phase B：两个 MVP server

- 实现 calendar server。
- 实现 shopping server。
- 实现 deterministic reset。
- 实现 readonly sampler。
- 实现配置化 dependency graph。
- 实现 query templates。
- 实现 `DeterministicTeacherAdapter`。
- 实现 oracle planner 和 validator。

### Phase C：数据合成闭环 + Agent loop + reward

- 实现 `StateMachineOrchestrator.generate_many()`。
- 实现 `scripts/generate_live_mcp_tasks.py`。
- 实现 state-machine trace replay validation。
- 实现 `MCPToolsAgentLoop.rollout()`。
- 实现五组件 `RewardComposer`。
- 实现 `scripts/run_live_mcp_smoke.py`。
- 产出 LiveTask 文件和 trace 文件。

### Phase D：接训练链路

- 先做离线 rollout 数据生成。
- 再接 `SchemaShiftTaskRunner`。
- 最后尝试 live GRPO smoke。

### Phase E：文档与回放沉淀

- 将成功 live trace 转换为 replay `EpisodeSeed` 的脚本设计清楚。
- 更新 `mcp_tools_rl_project_plan.md`，记录 live/replay 双后端边界。
- 不在第一版默认启用 live GRPO；只提供 smoke 入口。

## 17. 第一版范围

第一版不做：

- 不接真实公网 API。
- 不依赖外部不稳定服务。
- 不做 20 个 server。
- 不做复杂权限系统。
- 不做跨机器 distributed MCP server。
- 不直接替换现有 replay GRPO reward_fn。

第一版必须做，不能再推迟：

- `DependencyGraphBuilder`：至少支持配置化 dependency graph，能产出 2-5 步 tool chain。
- `StateSeeder`：能按 seed 生成 deterministic 初始状态。
- `StateSampler`：能调用 readonly tool 采样真实实体。
- `QueryGenerator`：能把结构化任务渲染成自然语言 query。
- `OraclePlanner`：能生成标准工具调用计划。
- `OracleValidator`：能在 freshly reset session 上执行 oracle plan 并验证最终状态。
- `TeacherAdapter`：必须提供 deterministic teacher；本地或远程 LLM teacher 是非阻塞扩展，不得作为第一版依赖。
- `StateMachineOrchestrator`：能串起 query generation、teacher/tool execution、response classification、recovery、continuation、trace replay validation。
- `RewardComposer`：必须包含 validity、coverage、adaptive efficiency、tool-name、argument-value 五类组件。
- `TraceRecorder`：必须保存完整生成轨迹和 rollout 轨迹。
- `RobustnessKnobs`：至少支持 missing-function/no-tool 和 distractor injection。
- `backend` 动态切换：至少支持 `replay` / `live` 两个后端配置。

第一版必须完整实现：

```text
calendar server
shopping server
dependency graph
state seeding
live-state sampling
grounded query generation
oracle planning
oracle validation
finite state-machine orchestration
five-component reward
trace recording
replay/live backend switch
```

可以暂缓扩展：

```text
20 个 server
LLM 自动发现 dependency graph
大规模 teacher 数据生成
distributed MCP server
live GRPO 大规模训练
```

## 18. 最终验收标准

第一版完成时，应该可以先生成数据：

```bash
python scripts/generate_live_mcp_tasks.py \
  --suite configs/live_mcp/suite_mvp.yaml \
  --output data/live_mcp/tasks/live_mcp_mvp.jsonl \
  --num-tasks 20 \
  --seed 42
```

并得到：

```text
tasks_requested = 20
tasks_written >= 20
oracle_valid_rate = 1.0
has_calendar_tasks = true
has_shopping_tasks = true
has_missing_function_tasks = true
has_distractor_tasks = true
no uncaught traceback
```

然后运行 live smoke：

```bash
python scripts/run_live_mcp_smoke.py \
  --suite configs/live_mcp/suite_mvp.yaml \
  --tasks data/live_mcp/tasks/live_mcp_mvp.jsonl \
  --server calendar \
  --num-tasks 10 \
  --seed 42
```

并得到：

```text
sessions_created = 10
rollouts_finished = 10
trace_files_written = 10
subprocess_stdio_used = true
reward_score_mean is finite
no uncaught traceback
```

工程上最重要的验收不是 reward 多高，而是：

```text
真实工具执行
状态确实变化
session 可隔离
trace 可审计
reward 可聚合
新增 server 不改 agent loop
```

## 19. 与 PROVE 的一致性和工程优化

第一版目标是 PROVE 的功能等价子集，而不是论文数值复现。

| 维度 | PROVE 论文 | 第一版方案 | 判断 |
|---|---|---|---|
| server 规模 | 20 个 stateful MCP servers，343 tools | 2 个 stateful MCP servers：calendar、shopping | 减少规模，不减少机制 |
| server 运行方式 | 独立 subprocess，stdio 通信 | 必须支持 subprocess stdio，in-process 仅测试 | 对齐 |
| session 隔离 | 每个 rollout 独立 session | `SessionSpec + deterministic reset` | 对齐 |
| schema | OpenAI-compatible function schemas | `SchemaRegistry` 统一发现和校验 | 对齐 |
| dependency graph | LLM 自动探测 tool pair 关系 | 第一版配置化 graph，后续可替换 LLM classifier | 工程优化 |
| query grounding | live-state sampling 后注入 teacher prompt | `StateSampler + TeacherAdapter + QueryValidator` | 对齐 |
| state machine | query / turn / tool-execution / response / continuation / recovery | `StateMachineOrchestrator` 强制覆盖这些状态 | 对齐 |
| replay validation | freshly reset env 上重放每条 conversation | `OracleValidator + REPLAY_VALIDATE_TRACE` | 对齐 |
| robustness knobs | distractor、enum stripping、irrelevance、missing-function | 第一版必须实现 missing-function/no-tool 和 distractor；enum stripping 可作为非阻塞配置开关 | 部分对齐 |
| reward | validity、coverage、adaptive efficiency、tool-name、argument-value | 同五组件，保留 complexity-adaptive call budget | 对齐 |
| RL 训练 | VERL + GRPO against same live env | 第一版先 smoke/offline runner，保留接 VERL 的 backend switch | 分阶段接入 |

工程优化的事实依据：

1. 自动 dependency graph 成本高且有不确定性。论文用 LLM 分类所有 tool pairs，并缓存 graph。第一版只有两个 server，手写配置图更可审计，也更容易单测；接口保持可替换，后续再接 LLM classifier。
2. 纯 teacher LLM 生成 query 容易引入实体幻觉。论文用 live-state sampling 和 anti-hallucination prompt 缓解。第一版进一步加 `QueryValidator`，强制检查 visible slots、hidden slots 和实体 provenance。
3. in-process server 不能暴露 subprocess/stdio 的真实失败模式。论文强调 live MCP subprocess。第一版 smoke 必须走 subprocess stdio，in-process 只用于单测加速。
4. 大规模 server 数量不是第一阶段瓶颈。论文价值来自 live execution + grounded synthesis + programmatic reward 的耦合，而不是 20 这个数字本身。因此第一版用少量高状态密度 server 验证机制，再横向扩展 domain。

第一版必须至少支持两类数据：

```text
multi-turn MCP conversations:
  calendar/shopping 的 2-5 步工具链。

missing-function / no-tool:
  隐藏必要工具或生成不可满足 query，训练 clarification / abstention。
```

第一版 robustness knobs 口径：

```text
distractor injection:
  必须实现。从另一个 server 注入 3-8 个无关工具。

missing-function / no-tool:
  必须实现。隐藏必要工具或生成不可满足 query，训练 clarification / abstention。

enum stripping:
  非阻塞配置项。移除部分 enum 列表，保留自然语言描述。
```

## 20. Live 数据合成中文流程

live 数据里的 query 不是让 LLM 凭空编出来的。正确流程是先有真实状态，再从真实状态中采样实体，然后用模板或 teacher 生成任务。

整体流程：

```text
初始化 server 状态
-> 用只读工具采样真实实体
-> 选择任务模板 / tool chain
-> 把实体填进模板
-> 生成用户 query
-> 生成 oracle program
-> 在 live server 上执行验证
-> 验证通过后写入 LiveTask 数据集
```

### 20.1 Calendar 例子

初始状态：

```json
{
  "events": {
    "evt_001": {
      "title": "Team Sync",
      "start_time": "2026-06-22T14:00",
      "end_time": "2026-06-22T15:00"
    }
  }
}
```

只读采样：

```text
调用 list_events
-> 得到真实存在的 evt_001
```

绑定实体：

```text
event_id = evt_001
title = Team Sync
new_start_time = 2026-06-23T10:00
```

生成 query：

```text
请把今天下午 2 点的 Team Sync 改到明天上午 10 点，然后告诉我更新后的时间。
```

oracle program：

```json
[
  {
    "tool": "list_events",
    "arguments": {"date_range": "this_week"}
  },
  {
    "tool": "update_event",
    "arguments": {
      "event_id": "evt_001",
      "fields": {"start_time": "2026-06-23T10:00"}
    }
  }
]
```

成功标准：

```text
events.evt_001.start_time == 2026-06-23T10:00
```

这个任务的重点是：query 里可以不直接暴露 `evt_001`，但这个 id 来自真实 server state。模型需要先调用 `list_events` 找到对应会议，再调用 `update_event` 修改。

### 20.2 Shopping 例子

初始状态：

```json
{
  "products": {
    "prd_007": {
      "name": "K3 Keyboard",
      "category": "keyboard",
      "price": 79,
      "stock": 5
    }
  },
  "cart": [],
  "orders": {}
}
```

只读采样：

```text
调用 search_products(category="keyboard", max_price=100)
-> 得到真实存在且有库存的 prd_007
```

生成 query：

```text
帮我找一个 100 美元以内的键盘，加入购物车并完成结账。完成后告诉我订单号。
```

oracle program：

```json
[
  {
    "tool": "search_products",
    "arguments": {"category": "keyboard", "max_price": 100}
  },
  {
    "tool": "add_to_cart",
    "arguments": {"product_id": "prd_007", "quantity": 1}
  },
  {
    "tool": "checkout",
    "arguments": {}
  }
]
```

成功标准：

```text
新订单已创建
订单 items 里包含 prd_007
购物车 checkout 后为空
```

## 21. 和当前 Toucan 数据的区别

当前 Toucan / EpisodeSeed 数据：

```text
用户问题、工具 schema、oracle_trace、预存 replay_observation 都已经在数据里。
模型调对了，就释放预存 observation。
server 状态不会真的改变。
```

新 live 数据：

```text
用户问题来自当前 live state。
工具 observation 是实时执行得到的。
模型调用写工具后，server state 真的改变。
reward 看最终状态是否满足 success criteria。
```

所以 live query 的来源可以总结成一句话：

```text
live query = 真实状态采样结果 + 任务模板 / teacher + 程序化成功标准
```

LLM 可以用于改写 query，让语言更自然；但实体、约束、oracle program 和成功标准必须由程序控制，不能让 LLM 自由虚构。
