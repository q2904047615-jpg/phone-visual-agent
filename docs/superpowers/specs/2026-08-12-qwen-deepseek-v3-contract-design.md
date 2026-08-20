# Qwen 与 DeepSeek v3 交叉契约设计

> **已归档，不可执行。** v2/v3 均已退役，当前 Qwen 只接受 `2026-08-20-deepseek-typed-task-graph-v4`。

## 目标

让当前 Qwen 单动作视觉决策入口正式接受 DeepSeek 提交 `438cd2258cdca681abe42da11b70c399df58063e` 的 `2026-08-11-deepseek-task-graph-v3` 上下文，同时保留已有可信候选、单动作、新鲜度、失败关闭和硬件禁用规则。

## 协议策略

- `2026-08-11-deepseek-task-graph-v3` 是唯一默认正式协议。
- `2026-08-11-deepseek-task-graph-v2` 仅是明确的旧数据迁移输入。
- 两个版本共用同一组顶层必需字段；不接受未知顶层字段。
- v3 的 `confirmation_gate` 必须恰好包含 `required`、`state`、`risk_ids`、`scope`、`external_state_action_allowed`。
- v2 的 `confirmation_gate` 必须保持旧结构，不允许伪造 `scope`。
- v2 只有 `read_only` 和 `navigation_only` 可以继续工作；`external_state` 或 `unknown` 在任何视觉模型调用前阻塞。

## v3 确认作用域

`confirmation_gate.scope` 必须恰好包含：

- `task_id`
- `device_id`
- `revision`
- `subgoal_id`

四项分别与顶层上下文和 `current_subgoal.subgoal_id` 逐项一致。`gate.risk_ids`、`current_subgoal.risk_action_ids` 和 `risk_actions[].risk_id` 必须完整一致，不允许缺失、额外、重复或跨子目标引用。

`state=confirmed` 只有在以下条件全部满足时有效：

- 当前影响为 `external_state` 或 `unknown`；
- `required=true`；
- scope 完整且逐项匹配；
- 三处风险 ID 完整一致且非空；
- `external_state_action_allowed=true`。

任何缺失、过期或冲突均由本地协议层拒绝。未确认的外部状态上下文由评估/运行入口在观察前返回 `blocked`，观察和决策调用均为 0。

## 真实交叉契约样本

从 Git 提交 `438cd2258cdca681abe42da11b70c399df58063e` 导出原始 `deepseek_task_graph.py` 与测试构造输入，在临时目录运行该提交的 `DynamicTaskGraph.to_qwen_context()`，保存其实际输出为固定 JSON 样本。样本包含来源提交、生成方法和导航、未确认外部状态、已确认外部状态上下文。当前分支不复制或修改 DeepSeek 实现。

## 代码边界

- `poc/qwen_visual_decision.py`：按协议版本严格分派上下文与 confirmation gate 校验。
- `poc/eval_qwen_visual_decision.py`：保持解析/风险门在观察前执行并安全返回 blocked。
- `poc/evals/qwen_visual_decision/`：保存真实契约样本、更新七用例为 v3。
- 测试文件：覆盖真实 v3 契约、scope/risk 反例、v2 迁移边界和现有安全不变量。

不修改 DeepSeek 分支、App 流程、坐标、候选生成、硬件控制、摄像头或网页。

## 验证

- Qwen 协议专项测试。
- 评估器测试。
- DeepSeek v3→Qwen 交叉契约测试。
- 项目完整离线测试。
- 不使用 `resume-report` 的全新七用例在线 Qwen 离线截图评估。

所有在线评估仍固定 `hardware_actions_enabled=false`、`camera_enabled=false`、`web_console_enabled=false`、`robot_enabled=false`。
