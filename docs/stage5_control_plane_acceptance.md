# 阶段 5 控制面网站验收

日期：2026-10-16（Asia/Shanghai）

本批把阶段 5 网站控制面剩余的配额、审计和通知入口接到现有 GitHub 会话与 Control Plane API。没有新增身份旁路，也没有把控制面权限直接变成 Runtime/Policy 的隐式放行。

## 已实现

- **配额页**：读取组织配额；`quota.manage` 主体可以修改周期、运行上限、物理动作上限和模型 Token 上限。保存沿用 `Idempotency-Key` 和现有主体审计。只有 `quota.read` 的主体为只读状态。
- **审计页**：按 `after_sequence` 游标读取组织审计，支持资源/结果/主体筛选；页面只渲染服务返回的脱敏 metadata。
- **通知页**：SQLite 持久化用户作用域通知。只从有意义的 Control Plane 写入审计生成，正文不复制任务、截图、令牌或认证断言；列表按 `organization_id + user_id` 过滤，支持标为已读。
- **多设备离线合同**：保留按 `device_id` 的租约、fencing epoch、设备锁、控制器/校准隔离和共享机械窗口锁。当前真实注册表仍只启用 `device-local-01`。
- **Runtime 配额与计费**：启动时按组织/主体原子预留 daily Run 和声明的物理动作预算；终态把实际物理动作、Qwen token 和 CNY 微单位估算结算到 SQLite，重复结算只返回已有账单，失败前退出释放预留。新增 billing 只读路由，不接支付或外部云账单。
- **网站计费入口**：owner 可在控制面打开只读“计费账本”，按组织隔离显示会话状态、模型、Token、物理动作、定价版本和 CNY 估算；无 `billing.read` 的成员不显示入口。
- **平台 GA 回滚**：SQLite 记录 active/previous/rolled_back 发布版本；回滚先生成计划并检查活动任务，提交前要求目标版本已由受控重载加载、无活动任务、所有者显式确认和幂等键。流程不会自动 checkout 或覆盖 Git 工作树。

## 验证

- Python 定向 Control Plane/Runtime 回归：23 项通过（含 Runtime 配额/计费/回滚合同 3 项）。
- Python 完整离线回归：1014 项通过，4 项跳过。
- 前端 `billing.js`、`app.js` Node 语法检查通过；本轮 npm 单元/浏览器命令被 Windows 沙箱的 Node `spawn EPERM` 阻断，不能计为前端完整回归通过。
- 服务已受控重载：监听 PID `7364`，OpenAPI 包含配额、审计和通知路由；typed client 核对配额 2 项、审计首批 50 条。
- 本批代码已受控重载为 `398e15f`，网站计费入口为 `a8c00d9`，当前监听 PID `1052`；启动自检 `verified=true`，`/api/session` 返回 `ready=true`，OpenAPI 的 billing、platform/releases、platform/rollback 路由方法与源码一致。SQLite 已创建 reservation/usage、billing ledger 和 platform releases 表，当前账单为 0、发布记录 3 条（active 1 条）。
- 所有验证使用临时数据库、Mock 或浏览器路由夹具；本批没有手机动作、第二台真机或付费模型调用。
- Runtime/回滚定向合同：`poc.test_runtime_quota_billing` 3 项通过，领域/客户端静态合同合计 23 项通过；当前 Windows 沙箱对部分历史 `tempfile.TemporaryDirectory` 清理存在权限错误，未把该环境错误归因于本批代码。匿名访问 billing/releases/device 均被预期的主体会话门禁返回 401；未执行手机动作。
- 前端新增计费账本模块已通过 Node 语法检查；既有 npm 单元/浏览器命令在当前沙箱被 Node `spawn EPERM` 阻断，未冒充为通过。

## 仍未完成的阶段 5 / GA 证据

- 第二台真机的独立标定、并行任务、同设备互斥和动作后业务证据尚未提供，不能把虚拟双设备测试写成真实多设备通过。
- 真实多设备并行动作、第二台真机独立标定和动作后业务证据尚未提供；这仍是第五阶段进入物理 GA 的阻断项。
- 计费当前是本地 CNY 估算账本，不是支付结算或云厂商最终账单；平台回滚已完成离线/单设备流程合同，真实发布环境的版本切换仍需按部署平台单独验收。
- 通知当前只覆盖 Control Plane 写入变更，不宣称外部邮件、Webhook 或移动推送已接入。

结论：**第五阶段的单设备软件子阶段已验证可行；Runtime 配额扣减、计费账本和 GA 回滚合同已补齐。由于尚无第二台真机和真实多设备证据，阶段 5 仍不能标记为物理 GA 全部完成。**
