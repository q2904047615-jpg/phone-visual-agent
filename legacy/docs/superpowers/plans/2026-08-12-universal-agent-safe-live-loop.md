> 历史/参考材料，不是当前运行权威。现行结论见[项目交接文档](../../../项目交接文档.md)，产品方向见[项目最终目标](../../../项目最终目标.md)。正文保留用于追溯，不据此恢复旧代码或执行旧步骤。

# Universal Agent Safe Live Loop Implementation Plan

> **ARCHIVED — DO NOT EXECUTE.** References to the deleted legacy runtime and DeepSeek v3 are historical only and must not be restored.

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 把 DeepSeek v3 动态任务图、Qwen v2 单步视觉决策、通用动作控制器和机械臂适配器接成首页默认运行链路，并以两个不同 App 的低风险导航任务验证“一次只执行一个动作、动作后重新观察与重规划”的通用闭环。

**Architecture:** 新增唯一的 `UniversalAgentOrchestrator` 纵向编排层，保留现有模型、视觉、动作和网页模块的职责边界。编排器只固定 `plan → observe → decide → policy gate → execute one → reobserve → verify → replan`，不包含 App 名称、固定命令或固定页面步骤。第一阶段本地策略仅放行可证明为导航的低风险动作，所有外部状态和未知影响动作在 Qwen 或机械臂之前失败关闭。

**Tech Stack:** Python 3.12、FastAPI、Pydantic、Pillow、现有 DeepSeek/Qwen provider、原生 `unittest`、Node.js 前端契约测试、Git。

---

## 实施边界

- 本计划必须服从根目录 `项目最终目标.md` 和已确认设计 `docs/superpowers/specs/2026-08-12-universal-agent-safe-live-loop-design.md`。
- 不新增微信、抖音或任何 App 的任务编排分支；App 名称只允许出现在测试样本和验收记录中。
- `start`、`next`、风险阻塞和协议失败路径的物理动作数必须为 `0`。
- 每次 `confirm` 最多执行一个物理动作；动作发出后失败也不得自动补点。
- 第一阶段禁止真实执行发送、关注、点赞、输入、支付、删除以及 `external_state/unknown` 子目标，即使前端提交 `confirmed=true`。
- 代码/模拟测试通过不等于真实闭环完成。真实机械臂验收是独立人工检查点。

## Task 1: 锁定第一阶段动作策略的失败关闭契约

**Files:**
- Create: `poc/universal_agent_orchestrator.py`
- Create: `poc/test_universal_agent_orchestrator.py`
- Reference: `poc/universal_action_controller.py`
- Reference: `poc/qwen_visual_decision.py`
- Reference: `poc/deepseek_task_graph.py`

**Step 1: 写出导航动作允许和禁止的失败测试**

在 `poc/test_universal_agent_orchestrator.py` 创建 `PhaseOneNavigationPolicyTests`，先覆盖：

```python
class PhaseOneNavigationPolicyTests(unittest.TestCase):
    def test_allows_swipe_for_navigation_only_subgoal(self): ...
    def test_allows_back_for_navigation_only_subgoal(self): ...
    def test_allows_canonical_navigation_tap(self): ...
    def test_rejects_toggle_input_and_keyboard_roles(self): ...
    def test_rejects_external_state_even_when_confirmed(self): ...
    def test_rejects_unknown_impact_before_qwen_or_robot(self): ...
    def test_rejects_account_effect_semantics(self): ...
    def test_rejects_noncanonical_or_ambiguous_tap_meaning(self): ...
    def test_rejects_out_of_bounds_or_stale_candidate(self): ...
```

构造最小合法 v3 当前子目标、`UIScene` 和 Qwen `GenericStepProposal`；不要在测试夹具里加入 App 专用判断。

**Step 2: 运行测试确认失败**

Run:

```powershell
python -m unittest poc.test_universal_agent_orchestrator.PhaseOneNavigationPolicyTests -v
```

Expected: `ModuleNotFoundError` 或缺少 `PhaseOneNavigationPolicy`，且没有硬件调用。

**Step 3: 实现最小策略对象**

在 `poc/universal_agent_orchestrator.py` 增加：

```python
@dataclass(frozen=True)
class NavigationPolicyDecision:
    allowed: bool
    reason: str
    canonical_class: str = ""


class PhaseOneNavigationPolicy:
    ALLOWED_ACTIONS = frozenset({
        "swipe", "back", "wait_for_change", "tap_semantic", "dismiss_overlay"
    })
    FORBIDDEN_ROLES = frozenset({"toggle", "input", "keyboard_key"})

    def evaluate(self, *, task_context, trusted_observation, decision) -> NavigationPolicyDecision:
        ...
```

实现要求：

- `swipe/back/wait_for_change` 依据结构化动作类型判定；
- `tap_semantic/dismiss_overlay` 只能接受通用导航语义分类，不接受任意按钮；
- 语义分类是安全策略 taxonomy，可包含中英文的“打开/进入/返回/关闭/取消/标签/列表/菜单”等通用类别，但不得据此生成业务步骤或坐标；
- 必须复用 `action_has_account_effect()`，并检查候选原始 `element_id/meaning/role/bounds` 与可信观察一致；
- 不能证明安全时返回 `allowed=False`，不得降级成猜测坐标。

**Step 4: 运行定向测试**

Run:

```powershell
python -m unittest poc.test_universal_agent_orchestrator.PhaseOneNavigationPolicyTests -v
```

Expected: 全部通过。

**Step 5: 提交**

```powershell
git add poc/universal_agent_orchestrator.py poc/test_universal_agent_orchestrator.py
git commit -m "feat: add phase one navigation safety policy"
```

## Task 2: 建立任务图与通用观察/动作层之间的桥接

**Files:**
- Modify: `poc/universal_agent_orchestrator.py`
- Modify: `poc/test_universal_agent_orchestrator.py`
- Reference: `poc/generic_intent.py`
- Reference: `poc/deepseek_task_graph.py`

**Step 1: 写桥接契约测试**

增加 `ObservationBridgeTests`：

```python
class ObservationBridgeTests(unittest.TestCase):
    def test_projects_dynamic_graph_to_generic_goal_without_business_steps(self): ...
    def test_projection_preserves_goal_entities_constraints_and_completion(self): ...
    def test_builds_observed_state_only_from_visible_evidence(self): ...
    def test_action_failure_is_not_reported_as_completion(self): ...
    def test_graph_and_observation_device_mismatch_fails_closed(self): ...
```

断言投影结果不包含 `operation`、固定坐标、App 路由或预制步骤。

**Step 2: 运行测试确认失败**

```powershell
python -m unittest poc.test_universal_agent_orchestrator.ObservationBridgeTests -v
```

Expected: 缺少 `ObservationBridge`。

**Step 3: 实现桥接器**

```python
class ObservationBridge:
    def goal_draft(self, graph: DynamicTaskGraph) -> GenericIntentDraft:
        """只做类型投影，绝不规划动作。"""
        ...

    def observed_state(
        self,
        *,
        scene: UIScene,
        action_outcome: str,
        verification: dict[str, Any],
    ) -> ObservedState:
        ...
```

映射规则：

- `objective` 来自任务图原始目标；
- `app_id/app_name` 只来自任务图目标应用，不做名称分支；
- `target/parameters` 只映射实体和约束；
- `success_criteria` 只映射任务图 completion conditions；
- `visible_evidence` 只能来自 `UIScene` 和控制器验证结果；
- `last_action_outcome` 只能是实际结果，不能由模型推断补全。

**Step 4: 运行测试并提交**

```powershell
python -m unittest poc.test_universal_agent_orchestrator.ObservationBridgeTests -v
git add poc/universal_agent_orchestrator.py poc/test_universal_agent_orchestrator.py
git commit -m "feat: bridge task graphs to trusted observations"
```

## Task 3: 让动作适配器返回与动作后场景完全匹配的四帧证据

**Files:**
- Modify: `poc/generic_action_adapter.py`
- Modify: `poc/test_generic_action_adapter.py`
- Modify: `poc/test_universal_agent_orchestrator.py`

**Step 1: 写结果帧契约测试**

在 `poc/test_generic_action_adapter.py` 增加：

```python
def test_execution_result_keeps_exact_four_verified_after_frames(self): ...
def test_after_frame_fingerprint_matches_after_scene(self): ...
def test_timeout_preserves_evidence_and_never_repeats_robot_action(self): ...
```

**Step 2: 运行测试确认失败**

```powershell
python -m unittest poc.test_generic_action_adapter -v
```

Expected: `GenericActionExecutionResult` 尚无 `after_frames`，新增测试失败。

**Step 3: 最小修改适配器数据流**

给 `GenericActionExecutionResult` 增加不进入 JSON 的运行时字段：

```python
after_frames: tuple[Image.Image, ...] = field(default_factory=tuple, repr=False)
after_frame_paths: tuple[str, ...] = ()
```

让 `_observe_stable_post_action_scene()` 返回场景、对应四帧和路径；`execute()` 必须把同一批帧放入结果。`to_dict()` 只输出路径和帧数量，不能序列化图像对象。

**Step 4: 验证旧行为无回归**

```powershell
python -m unittest poc.test_generic_action_adapter poc.test_universal_action_controller -v
```

Expected: 全部通过；所有失败路径的机械动作调用次数不超过 `1`。

**Step 5: 提交**

```powershell
git add poc/generic_action_adapter.py poc/test_generic_action_adapter.py poc/test_universal_agent_orchestrator.py
git commit -m "feat: preserve verified post-action observation frames"
```

## Task 4: 实现原子证据存储，写入失败时停止下一动作

**Files:**
- Modify: `poc/universal_agent_orchestrator.py`
- Modify: `poc/test_universal_agent_orchestrator.py`

**Step 1: 写证据目录和故障注入测试**

增加 `AgentEvidenceStoreTests`，覆盖：

```python
class AgentEvidenceStoreTests(unittest.TestCase):
    def test_writes_session_graph_risk_observation_qwen_controller_and_report(self): ...
    def test_json_write_uses_replace_not_partial_target_file(self): ...
    def test_pre_action_controller_evidence_failure_keeps_physical_actions_zero(self): ...
    def test_post_action_evidence_failure_marks_failed_and_blocks_next_action(self): ...
```

**Step 2: 运行测试确认失败**

```powershell
python -m unittest poc.test_universal_agent_orchestrator.AgentEvidenceStoreTests -v
```

**Step 3: 实现存储器**

```python
class AgentEvidenceStore:
    def write_json(self, name: str, payload: Mapping[str, Any]) -> Path:
        temp_path = self.run_dir / f".{name}.{uuid.uuid4().hex}.tmp"
        ...
        temp_path.replace(self.run_dir / name)

    def write_session(self, session: UniversalAgentSessionState) -> Path: ...
```

必须支持并按设计命名：

- `session.json`
- `task_graph_revision_*.json`
- `risk_audit_revision_*.json`
- `trusted_observation_step_*.json`
- `qwen_decision_step_*.json`
- `controller_decision_step_*.json`
- `verification_step_*.json`
- `report.json`

截图仍由适配器写入，但存储器必须把路径登记进会话报告。

**Step 4: 运行测试并提交**

```powershell
python -m unittest poc.test_universal_agent_orchestrator.AgentEvidenceStoreTests -v
git add poc/universal_agent_orchestrator.py poc/test_universal_agent_orchestrator.py
git commit -m "feat: add atomic universal agent evidence store"
```

## Task 5: 实现启动路径，确保规划和观察阶段绝不动作

**Files:**
- Modify: `poc/universal_agent_orchestrator.py`
- Modify: `poc/test_universal_agent_orchestrator.py`

**Step 1: 为编排器启动写端到端单元测试**

使用 `FakeDeepSeekPlanner`、`FakeQwenObserver`、`FakeAdapter` 和记录调用次数的 `FakeRobot`：

```python
class UniversalAgentStartTests(unittest.TestCase):
    def test_start_plans_observes_and_decides_with_zero_physical_actions(self): ...
    def test_start_uses_at_least_four_frames_for_trusted_observation(self): ...
    def test_external_state_blocks_before_qwen_and_robot(self): ...
    def test_unknown_impact_blocks_before_qwen_and_robot(self): ...
    def test_qwen_blocked_has_no_confirmation_entry(self): ...
    def test_qwen_finished_requires_deepseek_completion_revision(self): ...
    def test_protocol_or_identity_mismatch_fails_with_zero_actions(self): ...
```

**Step 2: 运行测试确认失败**

```powershell
python -m unittest poc.test_universal_agent_orchestrator.UniversalAgentStartTests -v
```

**Step 3: 实现会话状态与 `start()`**

```python
@dataclass
class UniversalAgentSessionState:
    session_id: str
    raw_goal: str
    device_id: str
    run_dir: Path
    task_graph: DynamicTaskGraph | None = None
    trusted_observation: TrustedObservation | None = None
    qwen_decision: QwenVisualDecision | None = None
    status: str = "created"
    step_number: int = 1
    physical_actions: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)
    failed_reason: str = ""

    def snapshot(self) -> dict[str, Any]: ...


class UniversalAgentOrchestrator:
    def start(self, *, session_id: str, raw_goal: str, device_id: str, run_dir: Path):
        ...
```

调用顺序必须固定为：

1. `DeepSeekTaskGraphPlanner.plan()`；
2. 检查当前子目标影响等级；
3. 安全子目标才允许 `adapter.capture_scene()`；
4. 从同一批四帧创建 `TrustedObservation`；
5. `graph.to_qwen_context()` → `QwenVisualDecisionObserver.decide()`；
6. `PhaseOneNavigationPolicy.evaluate()`；
7. 写证据并返回快照。

启动路径不得调用 `adapter.execute()`，并断言 `physical_actions == 0`。

**Step 4: 实现 Qwen `finished` 双重验证**

Qwen 返回 `finished` 时，把可信画面转换为 `ObservedState`，以 `completion_candidate` 触发一次无动作 `DeepSeek.replan()`：只有新 revision 的任务图确认为完成才置 `succeeded`，否则置 `blocked`，不得把 Qwen 单方判断当成成功。

**Step 5: 运行测试并提交**

```powershell
python -m unittest poc.test_universal_agent_orchestrator.UniversalAgentStartTests -v
git add poc/universal_agent_orchestrator.py poc/test_universal_agent_orchestrator.py
git commit -m "feat: orchestrate safe universal agent startup"
```

## Task 6: 实现严格确认、一次动作、重观察和 DeepSeek 重规划

**Files:**
- Modify: `poc/universal_agent_orchestrator.py`
- Modify: `poc/test_universal_agent_orchestrator.py`
- Reference: `poc/generic_supervised_runtime.py`

**Step 1: 写确认权限和完整闭环测试**

```python
class UniversalAgentConfirmTests(unittest.TestCase):
    def test_exact_confirmation_executes_one_action_then_pauses(self): ...
    def test_confirmation_requires_observation_id_and_fingerprint(self): ...
    def test_replay_and_task_device_revision_subgoal_risk_mismatches_fail(self): ...
    def test_policy_is_rechecked_immediately_before_execute(self): ...
    def test_new_observation_fingerprint_and_revision_are_required(self): ...
    def test_action_without_visible_change_fails_and_is_not_retried(self): ...
    def test_replan_failure_keeps_after_frames_and_blocks_old_plan(self): ...
    def test_next_qwen_decision_uses_only_new_revision_and_observation(self): ...
```

**Step 2: 运行测试确认失败**

```powershell
python -m unittest poc.test_universal_agent_orchestrator.UniversalAgentConfirmTests -v
```

**Step 3: 实现不可重放的精确确认对象**

```python
@dataclass
class ConfirmationAuthority:
    session_id: str
    task_id: str
    device_id: str
    revision: int
    subgoal_id: str
    risk_ids: tuple[str, ...]
    observation_id: str
    fingerprint: str
    consumed: bool = False
```

确认必须在进入适配器之前原子消费。任何异常、暂停、取消、设备状态变化、重新观察或新 revision 都使旧确认失效。

**Step 4: 实现 `confirm_one()`**

```python
def confirm_one(self, session, confirmation: Mapping[str, Any]) -> GenericActionExecutionResult:
    self._validate_and_consume_confirmation(session, confirmation)
    self.policy.require_allowed(...)
    self.evidence.write_controller_decision(...)
    result = session.adapter.execute(...)
    session.physical_actions += result.physical_actions
    new_observation = TrustedObservation.from_scene(
        frames=result.after_frames,
        device_id=session.device_id,
        scene=result.after_scene,
    )
    observed = self.bridge.observed_state(...)
    new_graph = self.deepseek.replan(...)
    ...
```

硬约束：

- `result.physical_actions > 1` 立即视为实现错误；
- 新 fingerprint 必须与动作前不同，除非动作类型是零物理动作的 `wait_for_change`；
- 新 graph 的 `revision` 必须严格大于旧 revision，且 task/device 不变；
- 新 Qwen 决策只能消费新 graph + 新 observation；
- 执行后无论成功、阻塞或失败都不能自动进入第二个物理动作。

**Step 5: 运行定向测试并提交**

```powershell
python -m unittest poc.test_universal_agent_orchestrator.UniversalAgentConfirmTests -v
git add poc/universal_agent_orchestrator.py poc/test_universal_agent_orchestrator.py
git commit -m "feat: execute one verified action and replan"
```

## Task 7: 增加按设备独占的会话注册表

**Files:**
- Modify: `poc/universal_agent_orchestrator.py`
- Modify: `poc/test_universal_agent_orchestrator.py`

**Step 1: 写并发与释放测试**

```python
class DeviceTaskRegistryTests(unittest.TestCase):
    def test_second_active_session_on_same_device_is_rejected(self): ...
    def test_terminal_session_releases_device(self): ...
    def test_pause_invalidates_confirmation_but_keeps_session_inspectable(self): ...
    def test_cancel_releases_device_and_never_calls_robot(self): ...
    def test_observe_execute_and_post_observe_share_one_device_lock(self): ...
```

**Step 2: 实现 `DeviceTaskRegistry`**

使用 `device_id -> RLock` 和 `device_id -> active_session_id` 两张映射。模型初始规划可在设备锁外，但捕获、确认校验、物理动作、动作后捕获必须在同一个设备锁上下文中。

不要移除现有 `_supervised_hardware_lock()`；第一台真实机械臂仍需全局串行，设备注册表负责会话语义上的独占。

**Step 3: 运行并发测试并提交**

```powershell
python -m unittest poc.test_universal_agent_orchestrator.DeviceTaskRegistryTests -v
git add poc/universal_agent_orchestrator.py poc/test_universal_agent_orchestrator.py
git commit -m "feat: lock universal agent sessions by device"
```

## Task 8: 将现有通用网页 API 切到新编排器

**Files:**
- Modify: `poc/web_app.py`
- Modify: `poc/test_web_platform.py`
- Modify: `poc/generic_supervised_runtime.py` only if compatibility serialization needs a narrow adapter
- Reference: `poc/deepseek_task_graph.py`
- Reference: `poc/qwen_visual_decision.py`

**Step 1: 先写 API 失败测试**

在 `poc/test_web_platform.py` 增加或更新：

```python
def test_generic_start_uses_deepseek_v3_and_qwen_v2_with_zero_actions(self): ...
def test_generic_start_does_not_call_legacy_intent_or_step_planner(self): ...
def test_external_state_start_returns_blocked_without_qwen_or_robot(self): ...
def test_confirm_requires_observation_id_and_fingerprint(self): ...
def test_confirm_returns_new_revision_and_exact_after_evidence(self): ...
def test_same_device_second_session_returns_409(self): ...
def test_auto_endpoint_is_phase_one_disabled_with_zero_actions(self): ...
```

**Step 2: 运行测试确认当前旧路径失败**

```powershell
python -m unittest poc.test_web_platform -v
```

Expected: 新断言显示 `/start` 仍在调用 `GenericIntentParser + GenericStepPlanner`。

**Step 3: 在 `Runtime` 注入真实编排器依赖**

初始化：

```python
self.deepseek_task_graph_planner = DeepSeekTaskGraphPlanner(self.intent_provider)
self.qwen_visual_decision_observer = QwenVisualDecisionObserver(self.vision_provider)
self.device_task_registry = DeviceTaskRegistry()
self.universal_agent_orchestrator = UniversalAgentOrchestrator(...)
```

`generic_intent_parser` 和 `generic_step_planner` 可为历史测试保留，但 `/api/agent/generic-supervised/*` 不再调用它们。

**Step 4: 保持 API 名称，替换内部实现**

- `/start` → `orchestrator.start()`；
- `GET` → 新会话 `snapshot()`；
- `/confirm` → `orchestrator.confirm_one()`；
- `/next` → 重新捕获四帧并用当前 graph 重新决策，`physical_actions=0`；
- `/pause` 和 `/cancel` → 编排器状态转换并使确认失效；
- `/auto` 第一阶段返回 `409` + `physical_actions=0` + 明确的 `phase_one_manual_confirmation_required`，保留路由但不提供旁路动作权限。

同时给 `GenericConfirmationScopeRequest` 增加：

```python
observation_id: StrictStr = Field(min_length=1, max_length=128)
fingerprint: StrictStr = Field(min_length=1, max_length=256)
```

**Step 5: 处理异常时保持真实动作计数**

任何异常响应都从异常和会话历史合并计算 `physical_actions`；不能在动作已经发出后错误返回 `0`。报告写入失败时状态置 `failed`，下一次确认被拒绝。

**Step 6: 运行 API 测试并提交**

```powershell
python -m unittest poc.test_web_platform -v
git add poc/web_app.py poc/test_web_platform.py poc/generic_supervised_runtime.py
git commit -m "feat: route supervised api through universal orchestrator"
```

如果 `poc/generic_supervised_runtime.py` 没有实际修改，不要把它加入暂存区。

## Task 9: 更新前端确认作用域和单步交互

**Files:**
- Modify: `poc/static/protocol_adapter.js`
- Modify: `poc/static/app.js`
- Modify: `poc/static/index.html` only if existing fields cannot display new evidence
- Modify: `poc/test_frontend_protocol.js`
- Modify: `poc/test_frontend_browser_contract.js`

**Step 1: 写前端协议失败测试**

增加断言：

```javascript
test("confirmation grant binds observation id and fingerprint", () => { ... });
test("changed observation invalidates confirmation grant", () => { ... });
test("phase one never auto advances physical actions", () => { ... });
test("renders graph revision controller gate and before-after evidence", () => { ... });
```

**Step 2: 运行测试确认失败**

```powershell
node poc/test_frontend_protocol.js
node poc/test_frontend_browser_contract.js
```

**Step 3: 扩展确认指纹**

`confirmationScope()`、`scopeFingerprint()`、`createConfirmationGrant()` 和 `consumeConfirmationGrant()` 必须携带：

```javascript
observation_id: session.visualAction.observationId,
fingerprint: session.visualAction.fingerprint,
```

任何为空或变化都拒绝提交。

**Step 4: 关闭默认自动连续推进**

首页不再调用 `runAutoAdvanceLoop()` 触发动作。可保留兼容函数和按钮提示，但点击后只提示“第一阶段需要逐步核对并确认”，不得请求 `/auto` 执行动作。

页面现有区域至少展示：任务 revision、影响等级、Qwen 置信度、本地策略允许/阻塞原因、动作前后证据路径、累计动作数和会话状态。

**Step 5: 运行前端测试并提交**

```powershell
node poc/test_frontend_protocol.js
node poc/test_frontend_browser_contract.js
git add poc/static/protocol_adapter.js poc/static/app.js poc/static/index.html poc/test_frontend_protocol.js poc/test_frontend_browser_contract.js
git commit -m "feat: bind web confirmation to exact observation"
```

只暂存实际修改的文件。

## Task 10: 增加不依赖 App 脚本的模拟闭环验收

**Files:**
- Create: `poc/test_universal_agent_mock_loop.py`
- Create: `poc/fixtures/universal_agent/README.md`
- Create: `poc/fixtures/universal_agent/navigation_open/` generated fixture metadata/images
- Create: `poc/fixtures/universal_agent/navigation_back/` generated fixture metadata/images
- Modify: `poc/README.md`

**Step 1: 写两个不同页面族的脚本化画面测试**

夹具按通用场景命名，不按 App 名称编排：

```python
class UniversalAgentMockLoopTests(unittest.TestCase):
    def test_unseen_open_goal_executes_one_navigation_tap_and_replans(self): ...
    def test_rephrased_back_goal_executes_one_back_and_replans(self): ...
    def test_non_navigation_button_is_blocked_without_robot_call(self): ...
    def test_unstable_after_frames_fail_after_one_action_without_retry(self): ...
```

两个用例必须使用不同 `app_id` 和不同自然语言措辞，但走完全相同的编排器代码。

**Step 2: 生成自有合成夹具**

用 Pillow 在测试辅助函数中生成简单 UI 图，不下载或复制第三方 App 截图。README 说明每组画面的预期可见变化和风险类别。

**Step 3: 运行模拟闭环**

```powershell
python -m unittest poc.test_universal_agent_mock_loop -v
```

Expected: 安全导航各执行一次；风险和不稳定路径为零次或一次且不重试。

**Step 4: 加架构退化扫描**

```powershell
rg -n "douyin|wechat|抖音|微信" poc/universal_agent_orchestrator.py
```

Expected: 无匹配，退出码 `1`。

同时检查官方入口不再调用旧规划器：

```powershell
rg -n "generic_intent_parser\.parse|generic_step_planner\.propose" poc/web_app.py
```

Expected: `/api/agent/generic-supervised/*` 函数体内无匹配；若其他历史兼容入口仍有匹配，人工核对行号并记录。

**Step 5: 更新运行说明并提交**

README 只说明通用数据流、第一阶段安全范围、测试命令和真实验收边界，不写某个 App 的固定步骤。

```powershell
git add poc/test_universal_agent_mock_loop.py poc/fixtures/universal_agent poc/README.md
git commit -m "test: add cross-app universal agent mock loop"
```

## Task 11: 全量回归、静态安全审计和证据检查

**Files:**
- Modify: only files needed to fix regressions within the approved design
- Create: `docs/validation/2026-08-12-universal-agent-offline-validation.md`

**Step 1: 运行 Python 全量测试**

```powershell
python -m unittest discover -s poc -p "test_*.py" -v
```

Expected: 全部通过；记录测试总数和耗时。

**Step 2: 运行前端全量测试**

```powershell
node poc/test_frontend_protocol.js
node poc/test_frontend_browser_contract.js
```

Expected: 全部通过。

**Step 3: 检查变更和敏感信息**

```powershell
git diff --check
git status --short
rg -n "sk-[A-Za-z0-9_-]{12,}|api[_-]?key\s*[:=]\s*['\"][^'\"]+" --glob "!*.lock" --glob "!*.log" .
```

Expected: `git diff --check` 无输出；敏感信息扫描无真实密钥。测试夹具中的明显占位符需人工说明。

**Step 4: 检查证据目录内容**

从模拟测试选一个成功会话和一个阻塞会话，确认存在：任务图、风险审计、动作前观察、Qwen 决策、控制器决策、动作后观察/验证（成功路径）和报告；确认阻塞路径没有动作后伪证据。

**Step 5: 写离线验证报告**

报告必须分开列出：

- 已实现代码能力；
- 通过的纯离线测试；
- 通过的模拟机械臂闭环；
- 尚未运行的真实设备项目；
- 已知限制：外部状态动作、文字输入、多台手机并行仍未开放。

**Step 6: 提交回归证据**

```powershell
git add docs/validation/2026-08-12-universal-agent-offline-validation.md
git commit -m "docs: record universal agent offline validation"
```

## Task 12: 真实设备预检与人工确认检查点

**Files:**
- Create: `poc/run_universal_agent_live_preflight.py`
- Create after execution: `docs/validation/2026-08-12-universal-agent-live-validation.md`
- Do not modify product logic during live acceptance unless a reproducible generic defect is found

**Step 1: 写只读预检测试与脚本**

脚本只检查并输出 JSON，不移动机械臂：

- 摄像头能稳定取得四帧；
- 机械臂控制端可连接但不发送动作；
- DeepSeek/Qwen provider 配置存在，密钥只输出布尔状态；
- `device_id` 没有其他活动会话；
- 当前编排器安全策略版本为第一阶段；
- `physical_actions=0`。

对应单元测试必须用假硬件证明预检不调用 `click/swipe/back`。

**Step 2: 运行只读预检**

```powershell
python poc/run_universal_agent_live_preflight.py --device-id default-device
```

Expected: 输出 `ready=true` 或具体阻塞项，且 `physical_actions=0`。

**Step 3: 停止并请求用户确认真实动作条件**

在执行任何真实动作前，向用户展示：

- 当前手机页面截图；
- 设备 ID 和独占状态；
- 第一个陌生导航目标；
- DeepSeek 当前子目标和影响等级；
- Qwen 候选动作、区域、置信度和本地允许理由；
- 明确声明只授权这一步，动作后自动暂停。

没有用户针对当前页面和当前候选的明确确认，不执行 Step 4。

**Step 4: 验收第一个 App 的一个安全动作**

通过网页 `/start` 确认启动阶段 `physical_actions=0`；用户确认后仅调用一次 `/confirm`。核对：

- 机械臂只执行一次；
- 动作前后各有四帧；
- fingerprint 改变；
- 控制器验证成功；
- DeepSeek revision 增加；
- 会话暂停且没有第二次动作。

**Step 5: 用另一种措辞和另一个 App 重复一次**

不修改代码，不增加坐标或 App 分支。若候选不安全或不唯一，应视为“正确阻塞”，另选安全页面，而不是现场加视觉补丁。

**Step 6: 写真实验收报告并提交**

报告分别列出两个 App 的原始目标、任务图 revision、前后证据路径、动作计数、验证结论和任何阻塞。只有两个 App 都完成一次安全闭环，才把第一轮状态写为“真实闭环通过”。

```powershell
git add poc/run_universal_agent_live_preflight.py docs/validation/2026-08-12-universal-agent-live-validation.md
git commit -m "test: validate universal agent on real device"
```

## Task 13: 最终复核并集成

**Files:**
- Review: all files changed by Tasks 1–12

**Step 1: 对照设计逐条验收**

确认：

- 首页默认链路是 DeepSeek v3 → Qwen v2 → 本地策略 → 单动作适配器 → 新观察 → DeepSeek replan；
- 编排器没有 App 名称分支、固定命令、固定坐标或固定业务步骤；
- 第一阶段禁止动作无法被前端字段绕过；
- 每个物理动作都有准确计数和前后证据；
- 真实设备结果和离线/模拟结果分开描述。

**Step 2: 再次运行完整验证**

```powershell
python -m unittest discover -s poc -p "test_*.py" -v
node poc/test_frontend_protocol.js
node poc/test_frontend_browser_contract.js
git diff --check
git status --short --branch
```

**Step 3: 请求代码复核**

使用 `superpowers:requesting-code-review` 检查安全边界、协议一致性、动作计数、异常路径和架构退化。先修复 P0/P1 和所有会造成错误动作的 P2，再重复 Step 2。

**Step 4: 完成分支**

使用 `superpowers:finishing-a-development-branch` 选择合并方式。只有最新全量验证通过，且用户确认真实验收范围已完成或明确延期，才合并到 `main` 并推送 GitHub。

若真实验收延期，合并说明必须写“代码与模拟闭环通过，真实设备双 App 验收未完成”，不得宣称第一轮真实闭环完成。
