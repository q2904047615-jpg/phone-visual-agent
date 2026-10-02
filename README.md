# 通用手机视觉操作 Agent

## 汽水音乐广告流程测试与人工确认

本分支在 `phone-visual-agent` 的 `main` 基线上增加一个受限的广告页面测试入口，用于检查广告加载、倒计时、返回控件和奖励/结束页状态。入口位于 `poc/features/qishui_ad_test/`，API 为 `GET /api/features/qishui-ad-test` 与 `POST /api/features/qishui-ad-test/start`。

流程由用户手动启动和观察：`manual_start → manual_observe → awaiting_confirmation → completed`。每轮最多处理一个广告；到达奖励/领取页后立即等待人工确认，确认范围只允许一次普通返回或结束。`claim_action_allowed` 始终为 `false`，不会自动刷广告、循环 farming、绕过限制、伪造观看完成或自动点击领取金币。

功能记录页面状态、耗时、失败原因和测试证据，不保存账号、Cookie 或支付信息；它不改变现有通用任务、福袋流程、设备执行器或主项目协议。

本地验证（2026-10-02）：汽水定向测试 8 项通过；完整 Python discovery 888 项通过（跳过 4 项）；compileall、pip check、git diff --check 通过；`poc` 目录 `npm test` 39 项通过（19 单元、16 浏览器、4 阶段二浏览器）。CI 将在分支推送后运行。

## 基线通用手机视觉操作 Agent

通过自然语言驱动不同 App：Qwen 结合整任务、实际动作历史和新截图选择一步动作或报告整任务完成，本地只负责单动作执行闭环。代码采用轻量 DDD 模块化单体；不是固定 App 脚本。

## 从哪里开始

- 产品方向：[项目最终目标](项目最终目标.md)；协作规则：[AGENTS](AGENTS.md)。
- 产品合同：[用户决策与协议边界](用户决策与协议边界.md)。
- 最新状态和唯一下一步：[项目交接文档](项目交接文档.md)、[当前验收台账](单一权威最小校验验收台账.md)。
- 使用与离线测试：[PoC 入口](poc/README.md)、[网页配置](poc/README_WEB.md)。
- 源码职责：[架构](poc/GENERIC_VISUAL_AGENT_ARCHITECTURE.md)；模型分工：[协议](通用Agent协议说明.md)。
- 清理结论：[项目清理清单](项目清理清单.md)；按需追溯：[历史文档索引](历史文档索引.md)。

现行状态只在交接和当前台账记录。历史设计、旧台账、离线通过和旧真机画面都不代表当前版本已验收。不要从旧文档恢复已退役的执行入口。
