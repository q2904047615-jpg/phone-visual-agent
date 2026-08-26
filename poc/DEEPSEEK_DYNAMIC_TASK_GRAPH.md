# DeepSeek 类型化动态任务图 v4

正式协议：`2026-08-20-deepseek-typed-task-graph-v4`

## 职责

`agent/application/deepseek_task_graph.py` 只把自然语言目标组织为可修订的高层任务图。它不输出坐标、不调用机械臂，也不能决定是否需要确认。用户可以直接使用“点击、滑动、输入、长按、拖动、返回、Home、发送、关注、评论”等自然动作词；只有坐标、ADB、Shell、keycode、卖家控制命令和绕过闭环的自由动作脚本会被拒绝。

## 模型输出

DeepSeek 的正式响应只能包含：

- `goal`、`constraints`、`completion_conditions`；
- `effect_intents`；
- `subgoals[].effect_ids`；
- `subgoals[].execution_class`，值为 `observe | navigate | effect | unknown`；
- `active_subgoal_id`、`status`、`clarification_questions`。

每个 effect 必须明确 `effect_id`、`kind`、目标实体角色、载荷实体角色、来源子目标和预期结果。模型不得输出风险等级、确认布尔值、机械权限或退役字段。额外字段、缺失字段、重复键、旧 v2/v3 版本，以及伪装成 v4 的旧字段都会在进入 Qwen 前失败关闭。

## 本地唯一策略权威

类型化响应先形成 `TaskSemanticIR`，再由版本化本地策略为每个 effect 生成 `local_policy`。模型不能自行升降风险。

- 普通发送、关注、评论、发布、收藏、订阅和一般数据修改默认自动；
- 只有登录/身份认证和资金交易需要一次效果确认；
- 敏感权限、账号/数据删除及其他普通效果按当前用户政策自动；
- `unknown` 不是风险类别，不能靠确认取得权限；若无法形成现有通用动作与可验证结果，应报告明确的语义或能力缺口。

效果确认使用 `effect_ids` 和 `/approve-effect`，绑定 session、task、device、revision、subgoal、当前观察、decision 和 action digest。旧确认、其他效果或其他 revision 不能复用。

## 提供给 Qwen 的上下文

`DynamicTaskGraph.to_qwen_context()` 只公开 typed v4 上下文：当前 `execution_class`、与活动子目标绑定的 `effect_intents`、本地 `effect_gate`、目标、约束、完成条件和实体。Qwen只能基于当前可信场景选择一个正式候选；它不能修改任务图、效果策略或确认状态。

## 重规划不变量

每次重规划返回完整的新 typed v4 快照，本地校验器强制：

1. task、device、目标和既有约束不能被改写或删除；
2. revision 必须精确增加一；
3. 已完成子目标和已消费 receipt 不能复活或重放；
4. visual claim 只证明当前画面，controller receipt 只证明已绑定动作执行；
5. effect、实体角色、来源子目标和预期结果不能漂移；
6. 依赖必须存在且无环，可推进状态只能有一个活动子目标；
7. 低层控制字段和退役协议字段不能进入任务图；
8. 结果缺证据时可以 0 动作重新观察，但不能重复 effect 动作。

## 已退役

DeepSeek v2/v3 schema、旧 transport 迁移器、旧远程自由文本风险审计、`risk_actions`、`risk_action_ids`、模型 `external_impact`、模型 `confirmation_required`、`confirmation_gate`、`risk_confirmation_scope` 和 `/approve-risk` 均不是兼容入口。历史文档和报告只作只读证据，不能进入正式 session、Qwen 或机械臂。

代码内部仍可能以历史 Python 属性名承载由 typed v4 **确定性生成**的运行时投影；这些对象不是模型输入协议，不能读取旧 payload、改变 effect 分类或产生确认权限。
