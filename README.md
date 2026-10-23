# Phone-first General Agent Platform

以手机为第一执行连接器的通用 Agent 平台。正式运行权威为 Runtime、Planner、Policy/Approval、Executor、Verifier 和事件账本；Qwen 是 Model Gateway 中的模型供应商，只提供结构化观察或规划建议，不能直接决定 canonical action、风险放行或 Task/Run 终态。代码采用轻量模块化单体。

**跨电脑交接请从 `codex/platform-policy-verifier-20261013` 分支开始。** 2026-10-24 核对时，GitHub 默认 `main` 仍是旧基线；`lean-rebuild` 从该旧基线创建，不代表用户已经批准替换当前平台。英文交接说明见 [Claude handoff](docs/CLAUDE_HANDOFF.md)。代码、部署与真机验收状态分别以交接文档和验收证据为准。

## 从哪里开始

- 产品方向：[项目最终目标](项目最终目标.md)；协作规则：[AGENTS](AGENTS.md)。
- 产品合同：[用户决策与协议边界](用户决策与协议边界.md)。
- 实施主方案：[项目首席架构师与验收执行提示词](docs/项目首席架构师与验收执行提示词.md)。
- 最新状态：[项目交接文档](项目交接文档.md)、[生产验证](docs/production_validation_20261024.md)、[远程工具验收](docs/remote_tools_acceptance_20261023.md)。旧台账仅用于追溯。
- 使用与离线测试：[PoC 入口](poc/README.md)、[网页配置](poc/README_WEB.md)。
- 源码职责：[架构](poc/GENERIC_VISUAL_AGENT_ARCHITECTURE.md)；模型分工：[协议](通用Agent协议说明.md)。
- 清理结论：[项目清理清单](项目清理清单.md)；按需追溯：[历史文档索引](历史文档索引.md)。

现行状态只在交接和当前台账记录。历史设计、旧台账、离线通过和旧真机画面都不代表当前版本已验收。不要从旧文档恢复已退役的执行入口。
