# 通用手机视觉 Agent 第一轮低风险真实闭环设计

确认日期：2026-08-12
状态：用户已确认设计范围，等待书面设计复核
最高依据：根目录 `项目最终目标.md`

## 1. 目标

把已经合入 `main` 的 DeepSeek v3 动态任务图、Qwen v2 单步视觉决策、通用动作控制器、机械臂适配器和通用网页控制台接成默认运行路径，并以一条真实但无账号影响的导航任务证明完整闭环：

```text
自然语言目标
→ DeepSeek 动态任务图
→ 真实多帧观察
→ Qwen 唯一下一动作
→ 本地安全校验
→ 机械臂执行一个低风险动作
→ 重新观察并验证
→ DeepSeek 推进或重规划
→ 网页展示证据
```

第一轮的成功标准不是完成某个 App 的固定命令，而是证明新目标能够不修改代码地进入同一套闭环。

## 2. 本轮范围

### 2.1 允许真实执行

仅允许满足全部安全条件的无账号影响动作：

- `swipe`：上、下、左、右滑动；
- `back`：返回上一页；
- `wait_for_change`：等待页面稳定，不产生物理动作；
- `tap_semantic`：只允许本地策略明确认定为通用导航语义的唯一可信候选；
- `dismiss_overlay`：只允许关闭、取消或返回性质的唯一可信候选。

导航点击必须同时满足：

1. DeepSeek 当前子目标的 `external_impact` 为 `navigation_only`；
2. Qwen 使用本地可信候选的原始 `element_id/meaning/role/bounds`，不得改写候选；
3. 候选角色不是 `toggle`、`input`、`keyboard_key`；
4. 本地 `PhaseOneNavigationPolicy` 将目标语义判定为导航、打开、返回、标签切换、列表进入、关闭弹层之一；
5. `action_has_account_effect()` 没有发现账号影响标记；
6. 页面稳定、候选唯一、置信度不低于现有阈值，坐标处于 0～1000 范围；
7. 当前观察、Qwen 决策、任务 revision 和设备完全一致且未过期。

任何条件不满足都必须在物理动作前停止。

### 2.2 本轮禁止真实执行

以下动作可以由 DeepSeek 识别、在网页展示风险和草稿，但不得传给机械臂：

- 发送消息、评论、发布内容；
- 关注、取消关注、加好友、删除好友；
- 邀请入群、移出群、创建群；
- 修改权限、角色、资料或设置值；
- 点赞、收藏、订阅等账号状态改变；
- 输入文字、长按、拖动；
- 购买、支付、下单、删除数据；
- `external_state` 或 `unknown` 子目标；
- 任何无法被本地策略证明为低风险导航的动作。

服务端即使收到前端 `confirmed=true`，也不得在第一轮绕过这条硬限制。风险确认接口继续保留，用于验证作用域和未来扩展，但本轮确认只改变网页展示状态，不授予外部状态动作执行权。

## 3. 现有代码缺口

现有模块已经分别具备能力，但默认网页路径仍使用旧的 `GenericIntentDraft + GenericStepPlanner`：

- `deepseek_task_graph.py` 可以生成和重规划 v3 动态任务图；
- `qwen_visual_decision.py` 可以消费 v3 当前子目标和可信观察，返回唯一动作、完成或阻塞；
- `generic_action_adapter.py` 可以执行一个通用动作并在动作后重新观察；
- `universal_action_controller.py` 可以解析候选、检查边界并验证画面变化；
- `generic_supervised_runtime.py` 已经具备严格确认作用域，但 `bind_v3_confirmation_context()` 仍等待“未来 Agent 循环”提供真实任务图和 Qwen 决策；
- `web_app.py` 的 `/api/agent/generic-supervised/*` 仍从旧意图解析器和旧步骤规划器启动。

本轮不继续给这些模块分别堆功能，而是增加一个唯一的运行时编排层。

## 4. 选定架构

### 4.1 新增纵向编排器

新增 `poc/universal_agent_orchestrator.py`，其中包含：

- `UniversalAgentSessionState`：保存一台设备上的任务图、revision、当前可信观察、Qwen 决策、动作记录、状态和证据路径；
- `UniversalAgentOrchestrator`：按固定闭环调度模型和本地组件；
- `PhaseOneNavigationPolicy`：本轮低风险动作硬门禁；
- `ObservationBridge`：把动作后的可信观察和可见证据转换成 DeepSeek `ObservedState`，不生成业务步骤；
- `AgentEvidenceStore`：以原子写入方式保存每轮输入、输出、截图和验证结果。

编排器只固定“观察—决策—校验—一个动作—再观察—重规划”的循环，不固定 App、页面路径或用户命令。

### 4.2 依赖注入

编排器通过构造参数接收：

- `DeepSeekTaskGraphPlanner`；
- 只读场景观察器和 `TrustedObservation` 工厂；
- `QwenVisualDecisionObserver`；
- `GenericSingleActionAdapter`；
- `UniversalActionController`；
- 设备锁；
- 证据存储器。

测试使用确定性假实现，真实运行使用现有 DeepSeek、Qwen、摄像头和机械臂实现。编排器本身不读取密钥、不调用 Shell、不包含任何 App 名称分支。

## 5. 会话状态机

状态只允许如下转换：

```text
created
  → planning
  → observing
  → awaiting_confirmation
  → executing_one_action
  → verifying
  → replanning
  ├─ awaiting_confirmation
  ├─ succeeded
  ├─ blocked
  ├─ failed
  ├─ paused
  └─ cancelled
```

规则：

- `start` 只规划、观察和决策，物理动作数必须为 0；
- `confirm` 每次最多执行一个物理动作；
- 动作执行后必须在同一设备锁内完成重新观察；
- 旧观察、旧 revision、旧 subgoal 或旧确认不得复用；
- Qwen 返回 `blocked/finished` 时没有执行入口；
- DeepSeek 只有在收到动作后新观察证据时才能完成或推进子目标；
- 任何异常都不得自动重试物理动作；
- 暂停、取消、失败和设备断线立即使当前确认失效。

## 6. 单轮数据流

### 6.1 启动

1. 网页提交 `text + device_id`。
2. 服务端获取该 `device_id` 的独占任务锁。
3. DeepSeek `plan()` 生成 v3 任务图和语义风险审计。
4. 若当前子目标为 `external_state/unknown`，保存任务图并返回等待风险确认，Qwen 调用数和物理动作数均为 0。
5. 对 `read_only/navigation_only` 子目标采集至少四帧真实画面，完成稳定性和清晰度筛选。
6. 建立 `TrustedObservation`，随后调用 Qwen `decide()`。
7. Qwen 决策经身份、新鲜度、候选唯一性和第一轮导航策略校验。
8. 返回网页，状态为 `awaiting_confirmation` 或 `blocked/finished`；启动阶段物理动作数始终为 0。

### 6.2 确认执行一个动作

1. 前端提交当前 `session_id/task_id/device_id/revision/subgoal_id/risk_ids/observation_id/fingerprint`。
2. 服务端在设备锁内重新核对当前权威状态。
3. `PhaseOneNavigationPolicy` 再次检查动作是否属于本轮允许范围。
4. `GenericSingleActionAdapter` 执行一个动作。
5. 禁止重复执行；等待画面稳定并采集新的四帧。
6. 建立新的可信观察并验证预期画面变化。
7. 将新观察转换成 DeepSeek `ObservedState`，调用 `replan()`；revision 必须增加。
8. 使用新 revision 和当前子目标再次调用 Qwen，形成下一轮候选动作。
9. 会话暂停并返回网页；只有下一次用户确认才能执行下一动作。

### 6.3 完成与阻塞

- Qwen 的 `finished` 只作为候选完成信号；必须同时存在可信观察中的完成证据，并由 DeepSeek 新 revision 将目标标为完成；
- Qwen `blocked`、候选不唯一、模型失败、协议非法或置信度不足时，保存原因并停止；
- 动作已经发出但验证失败时记录 `physical_actions=1`，状态为 `failed`，严禁自动补点；
- DeepSeek 重规划失败时保留动作后的真实截图和原任务图，状态为 `blocked`，不回退旧计划继续执行。

## 7. 网页和 API

继续使用现有 `/api/agent/generic-supervised/*` 作为默认通用 Agent API，避免再增加第二套正式入口：

- `POST /start`：改为调用新编排器，只规划、观察和决策；
- `GET /{session_id}`：返回统一会话快照和证据；
- `POST /{session_id}/confirm`：只执行当前一个已确认的低风险动作；
- `POST /{session_id}/next`：只重新观察和重新决策，物理动作数为 0；
- `POST /{session_id}/pause`：使当前观察和确认失效；
- `POST /{session_id}/cancel`：取消任务并释放设备锁；
- `/auto`：第一轮保留接口但每次请求仍最多一个物理动作，默认网页不自动连续调用。

网页必须显示：

- 用户原始目标；
- DeepSeek 当前任务图和 revision；
- 当前子目标、影响等级和风险；
- 动作前真实截图；
- Qwen 唯一动作、目标区域、置信度和理由；
- 本地控制器允许或阻塞的原因；
- 动作后截图、验证结果和 DeepSeek 重规划记录；
- 累计物理动作数、暂停、停止和失败状态。

旧微信、抖音 API 和状态机可以作为历史兼容测试保留，但首页自然语言入口不得再路由到这些旧流程。

## 8. 设备锁与并发

- 第一轮只启用一台真实设备，但所有会话继续携带 `device_id`；
- `device_id` 对应一个独占执行锁，同一设备只能有一个非终态会话；
- 模型规划可以在锁外准备，但观察、确认校验、机械动作和动作后观察必须在同一设备锁内串行完成；
- 不同 Worktree 不得同时启动真实网页服务、摄像头或机械臂；
- 设备断线、摄像头断线或控制端异常时状态转为 `blocked`，物理动作数不增加。

## 9. 证据与报告

每个会话目录至少保存：

- `session.json`：当前权威状态；
- `task_graph_revision_*.json`：每次 DeepSeek 任务图；
- `risk_audit_revision_*.json`：风险审计结果；
- `before_step_*_frame_*.jpg`：动作前四帧；
- `trusted_observation_step_*.json`：可信观察、稳定性和清晰度；
- `qwen_decision_step_*.json`：Qwen 原始合法化决策；
- `controller_decision_step_*.json`：允许或阻塞依据；
- `after_step_*_frame_*.jpg`：动作后四帧；
- `verification_step_*.json`：预期变化验证；
- `report.json`：最终结果、物理动作数、失败阶段和完整轨迹。

写报告失败时不得继续执行下一物理动作。

## 10. 测试设计

### 10.1 纯离线测试

- 新自然语言目标可以生成任务图并进入 Qwen；
- 同一目标换一种措辞仍走相同协议；
- `start` 物理动作数为 0；
- 每次 `confirm` 最多一个动作；
- 动作后必有新观察、新 fingerprint 和更高 revision；
- 画面无变化时失败且不重复动作；
- Qwen `blocked/finished` 无执行入口；
- 外部状态和 unknown 在 Qwen 或机械臂之前阻塞；
- 旧确认跨 task/device/revision/subgoal/risk/observation 复用全部失败；
- 不出现微信、抖音或其他 App 名称分支；
- 报告写入失败会阻止动作；
- 同设备第二个任务返回冲突。

### 10.2 模拟机械臂闭环

使用脚本化画面验证：

- 滑动后页面变化，DeepSeek 推进到下一子目标；
- 返回后到达预期页面；
- 导航点击后出现新页面；
- 非导航按钮、toggle、输入框、风险语义全部被本地策略拦截；
- 动作后画面不稳定、模糊或候选变化时安全停止。

### 10.3 真实设备验收

真实验收只运行一个集成任务并独占设备：

1. 先只读检查摄像头、机械臂控制端、DeepSeek、Qwen和设备锁状态；
2. 手机停在用户确认的安全页面；
3. 输入项目中没有预设过的导航目标；
4. 启动后确认机械动作数为 0；
5. 用户核对截图和 Qwen 候选后确认一步；
6. 机械臂执行一个滑动、返回或已证明安全的导航点击；
7. 系统保存动作前后截图，验证页面变化，并让 DeepSeek产生新 revision；
8. 会话暂停，不自动执行第二步；
9. 用另一种措辞和另一个 App 的安全页面重复一次，不修改代码。

两个 App 都通过才能声明第一轮真实闭环完成；不能用某个固定 App 的一次动作代替通用闭环证明。

## 11. 验收标准

本设计完成必须同时满足：

- 首页自然语言入口默认进入 DeepSeek v3 → Qwen → 控制器 → 机械臂 → 重观察 → DeepSeek 重规划；
- 两个不同 App 的陌生安全导航目标不修改代码即可运行；
- 每轮严格一个物理动作并暂停；
- 每个物理动作都有前后真实画面和本地验证证据；
- 外部状态、unknown、协议错误、旧确认和不唯一目标均为 0 动作；
- 完整离线测试、模拟闭环测试和真实设备证据分开报告；
- 不新增任何 App 专用业务编排；
- 未达到上述条件时，不得描述为“通用手机视觉 Agent 已完成”。

## 12. 明确不在本轮实现

- 外部状态动作的真实执行；
- 无人值守连续多步执行；
- 文字输入、长按和拖动；
- 三个 App 的最终跨 App 验收；
- 多台手机并行调度；
- 删除旧 App 专用兼容代码；
- 对某张截图增加坐标补丁或视觉特判。

这些能力只能在第一轮真实闭环通过后，沿用同一编排器逐项开放。
