# 架构迁移与旧入口退役验收记录

日期：2026-10-16

## 结论

**已验证可行（入口与单设备软件迁移子项）；完整公共 GA 仍有条件可行。**

本批把网页自然语言入口从旧的 `/api/qwen/chat` 迁移到平台入口 `/api/agent/chat`。入口现在只通过公开的 Model Gateway `chat` 合同取得 `chat_only` 或 `phone_task` 路由建议；它不能选择 canonical action、放行风险或结束 Task/Run。手机任务继续进入 Runtime → Planner → Policy → Executor → Observation → Verifier 链。

旧 `/api/qwen/chat` 路由已从生产 FastAPI 应用移除，前端和权限映射已切换到 `/api/agent/chat`，并加入旧路径返回 404 的回归。Qwen 仍保留为当前 Model Gateway 的一个供应商标识，不再作为网页入口或任务终态权威。

## 本批改动

- 新增 `AgentEntryRouter` 和 `AgentEntryRoute`，版本 `2026-10-16-agent-entry-route-v1`。
- 新增 `AgentEntryRequest`、`AgentChatRequest` 和 `AgentConversationMessage` 请求合同。
- Model Gateway 增加公开 `chat` 合同说明；入口不再调用 provider 私有 `_chat`。
- 前端入口改为 `/api/agent/chat`，用户可见名称改为 Agent 平台。
- 移除旧 `/api/qwen/chat` 路由、旧 Qwen 入口 schema 和旧权限特判。

## 离线验证

- Python 全量离线回归：`1029` 通过、`4` 跳过。
- 前端：`npm test` 通过，`19` 个单元合同和 `44` 个浏览器合同通过，共 `63` 项。
- 入口专项：新入口的 chat-only/phone-task、公开 Gateway 合同、无效 JSON、缺少 chat 合同和旧路径 `404` 均通过。
- 浏览器合同在受控 Edge 子进程环境中执行；普通受限沙箱的 `spawn EPERM` 只属于测试运行器限制，不是产品路由失败。

## 证据边界

- 本批不启用第二台真机、不执行手机动作、不调用支付或外部通知。
- 入口路由与权限/会话隔离可做离线和浏览器合同验证；真实公网登录、模型调用和运维证据沿用现有记录。
- 三 App 真机样本、多设备并行、多 worker、第二个真实账号、真实账单/支付和外部通知仍不属于本批完成项。
