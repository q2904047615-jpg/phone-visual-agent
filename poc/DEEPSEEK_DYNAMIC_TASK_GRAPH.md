# DeepSeek 通用动态任务图 v1

协议版本：`2026-08-11-deepseek-task-graph-v3`

## 职责

`deepseek_task_graph.py` 只负责把自然语言目标组织为可修订的高层任务图，并在收到新的只读场景摘要后调整剩余子目标。模块不导入机械臂、摄像头、坐标转换、Qwen 或网页运行时代码，也没有任何执行入口。

任务图包含：

- 用户最终目标、目标 App 和目标实体；
- 不可在重规划时删除的全局约束；
- 必须由可见证据满足的全局完成条件；
- 每次只激活一个的高层子目标及其依赖；
- 会改变账号、数据、交易或外部状态的风险事项；
- 每轮重规划的触发原因、观察证据和子目标变化记录；
- 从第一版开始携带的 `task_id` 和 `device_id`，用于隔离任务与设备状态。

## 提供给 Qwen 的当前子目标

`DynamicTaskGraph.to_qwen_context()` 生成只读上下文，明确包含 `task_id`、`device_id`、最终目标、全局约束、全局完成条件和唯一的 `current_subgoal`。其中风险列表只保留与当前子目标关联的风险。Qwen据此结合真实画面提出一个下一视觉动作，但不能修改任务图，也不能直接执行设备动作。

上下文同时包含 `current_external_impact` 和 `confirmation_gate`。`external_state` 或 `unknown` 子目标成为 `current_subgoal` 时，任务图只能处于 `awaiting_confirmation`，默认门禁为 `required=true`、`external_state_action_allowed=false`。任务图之外的可信本地控制器在收到用户明确确认后，可把对应风险 ID 作为 `confirmed_risk_ids` 传入 `to_qwen_context()`；只有当前子目标的全部风险均在确认集合中，门禁才返回 `state=confirmed` 和允许值。DeepSeek 的模型响应没有这个字段，不能自行放行。

## 重规划不变量

DeepSeek 每轮返回完整的新图快照，本地校验器再决定是否接受。校验器强制保证：

1. 原目标、目标 App 和目标实体不能被改写；
2. 已有全局约束不能删除；
3. 已完成子目标必须保留，不能复活或改写；
4. 已满足完成条件不能撤销；新完成声明只能引用本轮观察提供的原文证据；
5. 已识别风险不能删除、降级或取消人工确认；
6. 依赖必须存在且不能形成环，活动子目标的依赖必须全部完成；
7. 可推进状态必须且只能有一个活动子目标；
8. 低层动作、裸坐标、系统命令和 `main.exe` 控制字段不能进入任务图。

## 两项强制校验

1. **DeepSeek 不输出低层动作**：除了拒绝 `action`、`steps`、坐标和命令字段，还会检查目标、子目标、完成条件及风险描述中的点击、滑动、长按、拖动、输入等指令性表达。命中后整张模型任务图作废，不能降级执行。
2. **外部状态动作必须确认**：每个子目标必须显式给出 `external_impact`，值只能是 `read_only`、`navigation_only`、`external_state` 或 `unknown`。持久化、发送、发布、账号、交易或数据变化必须标为 `external_state`；不能确定时标为 `unknown`。后两类必须双向关联 `risk_actions`，风险的 `confirmation_required` 必须为 `true`；成为当前子目标时任务状态只能是 `awaiting_confirmation`。

风险的 `risk_type` 只能使用跨 App 通用语义：消息或通信、内容发布、账号关系变化、成员关系变化、权限角色变化、数据修改、数据删除、交易或支付、账号或权限变化、未知外部影响。它不描述任何 App 页面路径或业务步骤。

本地分类不是固定命令解析器，而是组合判断“变化意图”和“影响对象”。例如通信、账号关系、成员关系、权限角色、内容/数据和交易对象一旦与相应变化意图共同出现，就不能声明成安全分类。反过来，`read_only` 必须包含查看、读取、检查等观察证据，`navigation_only` 必须包含打开、进入页面等导航证据；无法证明时必须使用 `unknown`。

用户确认的作用域包含 `task_id`、`device_id`、`revision`、`current_subgoal` 和关联风险 ID。旧 revision、其他子目标或无关风险的确认均不能复用。

## 与后续模块的接口

后续 Qwen 层读取 `to_qwen_context()` 中的 `current_subgoal`，结合真实画面提出一个视觉动作。动作结果重新观察后，调用 `replan()` 传入 `ObservedState`。本模块自身不会执行动作，也不会把 App 名称转换成固定步骤。
