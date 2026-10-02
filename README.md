# 汽水音乐广告流程测试与人工确认

这是一个独立保存的私有项目副本，基于原 `phone-visual-agent` 的 main 提交 `8425ed5aeb090efd625f832fc906cf5b2cbc11b0`。本仓库从源码快照建立独立 Git 历史，不继承原仓库的分支和提交历史；排除了本地备份、浏览器取证和运行输出。

汽水功能位于 `poc/features/qishui_ad_test/`，入口是 `GET /api/features/qishui-ad-test` 与 `POST /api/features/qishui-ad-test/start`。用户手动打开一次广告，再通过 `/{test_id}/observe` 请求观察；奖励/结束页进入 `awaiting_confirmation`，通过 `/{test_id}/confirm` 明确确认一次普通返回或结束。功能不会自动领取金币、不运行自动刷广告循环。现有通用 Agent 和福袋能力作为基线保留。

离线验证（2026-10-02）：本独立副本的 `npm test` 共 39 项通过；完整 Python discovery 共 873 项，结果 `OK (skipped=3)`；`compileall`、`pip check` 和 `git diff --check` 通过。浏览器测试使用 Windows 本机 Edge 与锁定版本 Playwright 1.62.1。未运行真实手机任务或付费模型调用，离线通过不代表广告页识别和真实设备流程已验收。

下方文档保留基线项目说明；其中历史服务状态和真机证据不代表本仓库的新验证结果。

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
