# 通用手机视觉 Agent 正式协议

生效日期：2026-08-20（Asia/Shanghai）

当前用户边界的唯一清单见 `用户决策与协议边界.md`。未经用户批准不得扩展确认集合。

当前唯一正式高层协议为 `2026-08-20-deepseek-typed-task-graph-v4`。项目不再迁移、修补或执行 DeepSeek v2/v3，也不再让旧自由文本风险审计、旧风险字段或 App 固定流程参与正式会话。

运行时公开的配套协议为 `2026-08-20-task-semantic-ir-v3`、`2026-08-20-typed-effect-authority-v1` 和 `2026-08-19-visual-action-authority-v1`。`/api/device` 只公开 `typed_effect_authority`，不再公开旧 `semantic_risk_authority` 或 `formal_qwen_v3` 名称。

## 1. 用户目标

用户可以直接说“点击、滑动、输入、长按、拖动、返回、Home、发送、关注、评论”等自然动作。自然动作词不会被当成非法低层指令。协议只拒绝直接驱动设备的坐标、ADB、Shell、keycode、卖家控制命令和跳过观察闭环的自由动作脚本。

新 App 只要能由现有通用动作组合完成，就应当无需修改代码。App 名称、页面截图和固定步骤不能承担任务编排。

## 2. DeepSeek typed v4

DeepSeek 只输出：

- `goal`、`constraints`、`completion_conditions`；
- `effect_intents`；
- `subgoals[].effect_ids`；
- `subgoals[].execution_class`，值只允许 `observe | navigate | effect | unknown`；
- 当前唯一活动子目标和必要澄清。

每个 `EffectIntent` 必须包含：

- `effect_id` 和通用 `kind`；
- `target_entity_roles`、`payload_entity_roles`；
- `source_subgoal_ids`；
- `expected_results`。

模型不能输出风险等级、是否确认、坐标、视觉元素 ID 或机械臂权限。本地严格解析器拒绝额外字段、缺失字段和退役字段。没有 v2/v3 fallback；旧版本或把旧字段伪装成 v4 都是 `unsupported-protocol`，不会进入 Qwen 或机械臂。

## 3. 本地效果策略

`TaskSemanticIR` 把 surface、entity、effect、constraint、desired state、evidence 和 input field 分开。本地版本化策略是唯一确认权威，并把决定写入每个效果的 `local_policy`：

- 普通发送、关注、评论、发布、收藏、订阅和一般数据修改：默认自动；
- 只有登录/身份认证和资金交易：需要一次效果确认；
- 敏感权限、账号/数据删除和其他普通效果：按当前政策自动；
- `unknown`：不是风险类别，不能靠确认升级权限；无法映射到现有通用动作和可验证结果时报告语义或能力缺口。

约束、否定句、收件人、正文和完成条件不会再被旧正则或远程文本审计重新归类。出现分类问题只修改 typed effect 和本地策略，不恢复旧字段。

效果确认使用 `2026-08-20-typed-effect-confirmation-v1`，绑定 task、device、revision、subgoal、`effect_ids` 和 typed effect 摘要。公开接口为 `/approve-effect`；旧 `/approve-risk` 不存在。

## 4. Qwen 与可信观察

Qwen 只接受 typed v4 的 `current_execution_class`、当前 `effect_intents` 和 `effect_gate`。`effect_gate` 由本地生成，模型不能伪造。

Qwen 每轮只能从当前可信观察和本地正式候选中选择一个动作，或返回完成/阻塞。目标身份、逐字文本、元素状态和几何必须与观察绑定；确认前再次取得新帧并重做语义/几何绑定。画面或 scope 任一变化都会让旧确认失效。

## 5. 动作与执行闭环

正式动作包括 `tap_semantic`、`dismiss_overlay`、`swipe`、`back`、`home`、`reveal_system_navigation`、`input_verified_text`、`clear_verified_text`、`long_press`、`drag`、`wait_for_change`，以及设备已正式认证的扩展动作。

每次循环固定为：

1. 取得多帧可信观察；
2. DeepSeek 选定唯一活动子目标；
3. Qwen 选择唯一视觉动作；
4. 本地控制器验证 typed 语义、效果策略、设备能力、fresh scope 和几何；
5. 最多执行一个物理动作；
6. 重新观察并验证；
7. 写 typed receipt；
8. DeepSeek revision 精确加一并重规划。

物理动作失败不自动重复。自动执行取消不必要的人机确认；设备身份、目标唯一性、机械可达、一次一动作和动作后验证只承担正确执行职责，不得扩展产品风险或重复阻止合法任务。

## 6. 证据和完成

画面事实只能由绑定当前 scene 的 visual claim 证明；“动作已经执行”只能由 controller transition receipt 证明。两者不能互相替代，也不能跨会话或跨 revision 重放。

发送、付款等效果动作只允许执行一次。动作回执不能单独证明最终外部结果；若动作后首轮视觉漏检，系统只能进行 0 动作的新鲜观察以完成结果验证，不能重复效果动作。

## 7. 已彻底退役的正式入口

- DeepSeek v2/v3 schema、迁移器和兼容采样；
- `risk_actions`、`risk_action_ids`、`external_impact`、模型 `confirmation_required`；
- `current_external_impact`、`confirmation_gate`；
- `risk_confirmation_scope`、`risk_ids`、`/approve-risk`；
- 旧远程自由文本风险审计模块；
- 网页端旧协议兼容执行入口和旧协议夹具。

正式链内部为兼容现有 Python 数据结构而生成的确定性运行时对象不是模型协议，不能读取模型旧字段、改变 typed 分类或产生确认权限；后续清理内部命名不得改变这一权威边界。
