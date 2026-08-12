# 真机能力验收模式实施计划

> 实施方式：当前会话逐任务执行；每个任务先写失败测试，再做最小实现，完成后单独提交。

**Goal:** 实现通用、一次一动作、证据绑定且需要二次确认的真机能力验收与设备能力晋级。

**Architecture:** 独立 `CapabilityAcceptanceManager` 复用通用编排器和共享设备租约，通过未注册的临时控制器开放本次唯一候选动作。动作报告通过严格验证后，`CapabilityRegistryPromoter` 才能在第二次一次性确认下原子更新设备注册表；运行服务不热更新。

**Tech Stack:** Python 3.12、FastAPI、Pydantic v2、现有 DeepSeek/Qwen 协议、`UniversalAgentOrchestrator`、`GenericSingleActionAdapter`、OS 文件锁、原生 JavaScript、Node test、Playwright。

---

### Task 1: 验收报告和注册表晋级核心

**Files:**
- Create: `poc/capability_acceptance.py`
- Create: `poc/test_capability_acceptance.py`

- [ ] 写失败测试：报告缺少八张 trial 内 JPEG、动作数不为 1、动作类型不一致、`action_outcome` 非 matched 或 observation/fingerprint 未变化时，`validate_acceptance_report()` 拒绝。
- [ ] 运行 `poc\.venv\Scripts\python.exe -m unittest test_capability_acceptance -v`，确认因模块不存在而失败。
- [ ] 实现 `CapabilityAcceptanceError`、`AcceptanceReportScope`、`PromotionAuthority`、`validate_acceptance_report()` 和 SHA-256/安全路径辅助函数。
- [ ] 写失败测试：错误 report SHA、错误 registry SHA、错误设备/动作、已启用动作、确认重放和两个进程同时晋级均失败且目标 JSON 不变。
- [ ] 实现 `CapabilityRegistryPromoter.preview()` 与 `promote()`：同目录临时文件、flush、fsync、`os.replace`、`registry_before.json` 和 `promotion.json`。
- [ ] 运行模块测试并提交 `feat: add evidence-bound capability promotion`。

### Task 2: 临时单动作控制器和验收会话管理器

**Files:**
- Modify: `poc/web_app.py`
- Modify: `poc/robot_core.py`
- Create: `poc/test_capability_acceptance_runtime.py`
- Test: `poc/test_universal_agent_orchestrator.py`

- [ ] 写失败测试：`DeviceControllerRegistry.provisional_controller(device, action)` 返回未注册的新控制器，只多开放一个动作；原控制器和 descriptors 不变化。
- [ ] 写失败测试：已验证动作、未知动作、未知设备均在构造临时控制器前拒绝。
- [ ] 实现只读设备配置保存和 `provisional_controller()`，仅允许通用物理动作集合。
- [ ] 写失败测试：正式会话占用设备时，验收 start 在目录、相机、DeepSeek、Qwen 之前返回冲突；不同设备可分别存在 trial。
- [ ] 实现 `CapabilityAcceptanceManager`，共享正式 `DeviceTaskRegistry`，为每个 trial 创建专用 `UniversalAgentOrchestrator` 和临时 adapter。
- [ ] 写失败测试：Qwen 唯一动作与候选动作不同则 trial 失败且物理动作 0；相同时仍停在精确确认门。
- [ ] 写失败测试：一次确认最多一个物理动作；动作后观察失败时报告动作数 1、禁止重试和晋级。
- [ ] 实现 trial 状态、不可变元数据、报告生成和晋级 authority；提交 `feat: add supervised capability trials`。

### Task 3: 严格 FastAPI 接口

**Files:**
- Modify: `poc/web_app.py`
- Modify: `poc/test_web_platform.py`

- [ ] 增加严格 Pydantic 请求：start 只接收 `device_id/action/text`；risk/confirm 需要现有完整范围外加 `trial_id/action`；promote 需要 `trial_id/report_sha256/registry_sha256/confirmed=true`。
- [ ] 写失败测试：额外字段、布尔/字符串混淆、缺失作用域、错误设备或动作、重放均返回 409/422 且物理动作 0。
- [ ] 实现 `/api/capability-acceptance/*` 的 start/get/approve-risk/confirm/promotion-preview/promote/cancel 路由，全部复用本地控制令牌。
- [ ] 写测试证明 promote 从不调用相机、模型或机械臂，成功只返回 `requires_restart=true`。
- [ ] 运行 `poc\.venv\Scripts\python.exe -m unittest test_web_platform test_capability_acceptance_runtime -v` 并提交 `feat: expose capability acceptance api`。

### Task 4: 通用网页验收面板

**Files:**
- Modify: `poc/static/index.html`
- Modify: `poc/static/styles.css`
- Modify: `poc/static/app.js`
- Modify: `poc/static/protocol_adapter.js`
- Modify: `poc/test_frontend_protocol.js`
- Modify: `poc/test_frontend_browser_contract.js`

- [ ] 写 Node 协议失败测试：trial scope 缺少 trial/action、报告未通过、摘要变化或 confirmation 已消费时不生成晋级请求。
- [ ] 实现 `adaptCapabilityTrial()`、`createPromotionGrant()`、`consumePromotionGrant()` 和严格 payload 构造。
- [ ] 在控制台加入一个通用验收面板，动作下拉只显示未验证动作，启动文案明确“先观察，0 动作”。
- [ ] 复用风险对话框显示 trial/action/session/task/device/revision/subgoal/observation/fingerprint；不增加自动执行入口。
- [ ] 报告通过后显示证据列表和单独“确认启用该能力”按钮；成功后显示“等待安全重启”。
- [ ] 写 Playwright 契约：启动无动作、动作确认一次、失败无晋级、通过才有晋级按钮、晋级不触发动作接口。
- [ ] 运行 Node 语法、协议与浏览器测试并提交 `feat: add capability acceptance console`。

### Task 5: 全量验证、复审和交付记录

**Files:**
- Modify: `docs/validation/2026-08-12-live-acceptance-campaign.md`
- Modify: `docs/validation/2026-08-12-universal-agent-levels-progress.md`

- [ ] 运行 `poc\.venv\Scripts\python.exe -m unittest discover -s poc -p "test_*.py"` 并记录总数和耗时。
- [ ] 使用工作区 Node 运行 `node --check`、协议测试和 Playwright 浏览器契约。
- [ ] 运行 `git diff --check`，只读审查所有产品物理入口、trial 临时权限和晋级路径。
- [ ] 更新验证文档，明确代码/模拟通过不代表输入、长按、拖动已真机通过。
- [ ] 提交并推送分支；保持当前运行服务和待确认会话不变。

## 规格覆盖自检

- 临时权限与正式权限隔离：Task 2。
- 同设备互斥、不同设备隔离：Task 2、Task 3。
- 一次一动作、动作后四帧和失败不重试：Task 2。
- 二次确认、报告摘要和原子晋级：Task 1、Task 3。
- 通用网页入口与无自动执行：Task 4。
- 真实证据与离线测试分开：Task 5。
- 不包含 App 分支、固定坐标或固定步骤：所有任务共用动作协议和现有通用编排器。
