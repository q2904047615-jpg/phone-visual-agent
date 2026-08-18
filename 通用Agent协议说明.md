# 通用手机视觉 Agent 协议说明

生效日期：2026-08-19（Asia/Shanghai）

这份说明描述当前默认产品路径。默认入口只使用 DeepSeek 动态任务图、Qwen 单步视觉选择、本地确定性控制器和机械臂适配器；微信、抖音或其他固定 App 工作流不再参与默认执行。

## 一、用户可以怎么下达任务

用户可以直接说自然动作，不需要改写成抽象“状态句”。以下目标均为合法输入：

- “点击设置，找到隐私页面，再返回桌面。”
- “向上滑动，找到下载记录。”
- “在当前输入框输入 Agent 8，你好。”
- “长按当前唯一项目，再拖到目标区域。”
- “打开微信，进入文件传输助手，输入你好并发送。”

“点击、选择、打开、滑动、输入、清空、长按、拖动、返回、Home”等词只是意图，不直接产生机械臂权限。协议仍拒绝用户或模型绕过视觉闭环直接指定：

- 像素坐标、归一化坐标或数字点位；
- ADB、Shell、PowerShell、cmd、keycode；
- `main.exe`、卖家控制命令或连续自由动作脚本。

## 二、协议分层

### 1. DeepSeek 任务图

正式版本为 `2026-08-11-deepseek-task-graph-v3`。

DeepSeek 只负责：理解完整目标、拆分动态子目标、写完成条件、判断 `read_only`、`navigation_only`、`external_state` 或 `unknown` 影响等级，以及根据动作后证据重新规划。它可以保留用户说出的自然动作词，但不能输出坐标、系统命令或机械臂控制细节。

一次 plan/replan 只允许一次远程语义采样。只有唯一、局部且不改变语义的 JSON 规范化可以在本地修复；不会靠连续请求模型“抽到一个合法答案”。

能够影响执行权限的实体必须是正式 typed entity，但 entity 的业务名称不再受固定五键限制。例如：

- `recipient`：收件人或会话身份；
- `input_text`：需要输入的逐字文本；
- `target_ui_label`：用户明确指定的可见控件文字；
- `target_surface`：`device`、`system` 或 `current_surface`；
- `spatial_hint`：用户明确表达的相对区域。

任意用户明确给出的对象、文件、目录、联系人、字段或目标都可成为 typed entity。只有同时存在用户原文 source span、显式 relation/effect binding 和当前子目标引用时才能参与执行；旧 `goal.entities` 只作为非权威兼容上下文。没有目标 App 仅在目标明确属于设备、系统或当前表面时才合法。

DeepSeek 输出后必须经过正式 `2026-08-19-task-semantic-ir-v2` 投影。`TaskSemanticIR` 将 surface、entity、effect、constraint、desired state、evidence requirement、input field 和 semantic subgoal 分开，约束与完成条件不能再被当成风险字段。正式语义/风险权威为 `2026-08-19-semantic-risk-authority-v2`，权威范围仅是 `semantic_and_risk_only`，不能授予视觉动作或物理执行。

每个可执行子目标都绑定一个 typed surface。跨 App 任务若当前不在目标 App，正式选择器先产生 `home`，回到 Launcher 后再从首页寻找目标 App；不会从任意 App 内部猜测跨 App 入口。用户把常用 App 放在首页只是提高视觉可发现性，不形成 App 名称分支。

本地风险策略来自可替换配置 `poc/config/local_risk_policy.v1.json`。新旧语义切换会逐项生成结构化 diff；未声明的差异或无法类型化的 `generic_effect` 在调用 Qwen 前失败关闭。生产默认不再调用旧的远程自由文本风险审计；旧审计只保留在显式离线兼容测试中。

### 2. 可信观察

场景协议为 `2026-08-14-ui-scene-v3`，当前观察器为 `2026-08-18-generic-scene-observer-v53`。

观察器使用多帧真实画面生成有界候选集，记录角色、标签、meaning、状态、置信度、边界、遮挡和画面 fingerprint。模型不能自行写入本地可信标记；`fully_visible`、独立几何认证等权限必须由本地规则或严格几何审计产生。每条可见事实由本地生成绑定当前 scene ID 的 `visual_claim` 引用；任务图只能引用该 ID，不能复制一段自然语言就把它变成完成证据。

同一 fingerprint 下可复用只读证据；画面一旦变化，旧观察、旧几何和旧确认作用域全部失效。

### 3. Qwen 单步视觉选择

正式视觉决策版本为 `2026-08-14-qwen-visual-decision-v5`。

Qwen 每次只能从本地 `2026-08-19-visual-action-authority-v1` 报告中的正式候选选择一个下一动作，或返回完成/阻塞。候选由 claim、relation、affordance、typed transition 和当前 semantic subgoal 确定性生成；Qwen 不能新增候选、改写身份、扩大风险权限或直接给机械臂坐标。`goal_relevant` 仅保留为诊断字段，不能创造候选或执行权限。

当用户明确给出 `target_ui_label`、`recipient` 或 `input_text` 时，Qwen 必须逐字绑定相应实体。重复文字必须通过角色、容器、标题/身份锚点和区域消歧；无法得到唯一候选时为 0 动作阻塞。

### 4. 本地策略与动作控制器

控制器协议为 `2026-08-18-universal-action-v14`。本地控制器不再通过“微信、抖音、点赞”等关键词重新规划业务，只校验：

- action kind 是否为正式动作；
- 当前子目标影响等级是否允许该动作；
- 元素 ID、label、role、meaning、states 是否与可信观察原样绑定；
- 目标是否唯一、可见、未被遮挡且几何可执行；
- fresh 观察是否仍能语义重绑定，计划框与新框是否通过 IoU 门禁；
- 当前设备是否声明并已认证该动作能力；
- 本地策略标为 `confirmation_required` 的效果是否持有未消费、未过期的风险确认。

正式动作能力词汇：`tap_semantic`、`dismiss_overlay`、`swipe`、`back`、`home`、`reveal_system_navigation`、`input_verified_text`、`clear_verified_text`、`long_press`、`drag`、`wait_for_change`、`double_tap`、`press_enter`、`pinch`、`hardware_key`。词汇存在不等于当前机械设备支持；`2026-08-19-action-capability-v1` 逐项声明设备参数与支持状态，不支持或尚未安全接入的动作返回 `2026-08-19-capability-gap-v1`，不会进入视觉模型、相机或机械臂。

“发送、发布、点赞、评论、关注、收藏、订阅、支付、购买、删除、提交、保存、邀请、加入、确认、同意、授权”等可以被类型化为外部效果，不能伪装成普通导航；但“是外部效果”不再自动等于“必须向用户确认”。确认只由效果类型和本地策略决定，不读取收件人、正文、约束或完成条件中的关键词。

### 5. 独立几何审计

所有依赖控件落点的动作，在执行前必须用新确认帧重新观察。必要时只把单个 crop 交给独立几何审计，模型返回 crop-local 边界，本地使用真实像素变换映射回整帧。

几何审计只有在完整枚举、唯一匹配、高置信、完整控件、无邻居重叠且未触及内部裁剪边缘时，才生成私有本地凭据。无文字图标只有存在明确视觉形状证据并且全场语义唯一时可用；否则停止，不回退到粗框。

### 6. 输入与手势

输入内容来自 typed input field transaction，支持多个输入字段、多个接收者、多个段落和最多 4000 个 Unicode 字符；正文仍逐字来自用户原文，不由模型补写。执行采用“已有可信前缀 + 下一确定性分段”的事务：物理层每段最多 30 个字符，每段输入后重新观察并核对字段值，不正确时停止，不发送或提交。

输入法可以根据新观察在拼音、字母、数字、符号、大小写之间双向切换。换行被建模为独立 `press_enter` 能力，只有画面存在唯一可见 Enter 且能力被认证时才执行；否则整项任务在输入前返回 gap，不会先输入半段。清空文本只要求当前键盘存在唯一可见退格键，不再限定 QWERTY 或 direct-latin 布局。

长按、拖动、双击、Enter、pinch/多指、硬件键、文字长度和可用输入分段都来自设备能力描述。卖家工具中存在某个按钮或传输路径不等于已获得安全 ACK；无法证明时明确报告 capability gap。传输调用返回也不等于机械接触成功；最终仍必须由动作后画面证明 typed 后置条件。

## 三、自动执行与确认边界

### 安全动作

`read_only` 和 `navigation_only` 可以在同一任务中自动推进，不再逐步询问用户。默认一次自动批次最多 12 个物理动作、24 个循环；接口硬上限为 20 个物理动作、40 个循环。

预算不是一次发出多个动作。每次循环仍严格执行：

1. 取得当前可信观察；
2. Qwen 选择唯一下一动作；
3. 本地策略校验；
4. 只执行一个物理动作；
5. 重新取得多帧观察；
6. 验证结果、写 typed receipt；
7. DeepSeek revision 精确加一并决定下一步。

任何一步失败都停止；不会自动重试物理动作。

### 外部效果与风险确认

普通发送、关注、评论、发布、收藏和一般数据变更默认是 `automatic`，不再向用户追加风险确认。它们仍必须经过唯一视觉目标、精确实体/正文绑定、一次动作、动作后新观察和结果验证；“自动”只取消额外的人机确认，不降低视觉、几何、scope、设备能力或后置证据门禁。

默认仍要求确认的类型只有：登录/身份认证、资金交易、敏感权限变更、不可逆账号删除和不可逆数据删除。确认绑定 session、task、device、revision、subgoal、需要确认的 risk IDs 和类型化意图摘要；模型写入的 `confirmation_required` 只是兼容字段，不能覆盖本地策略。

例如“打开微信，进入文件传输助手，输入你好并发送”：App 导航、会话定位、精确正文输入和发送都可在同一自动闭环中推进，每一步仍只执行一个物理动作并重新观察。改成“登录工作账号”或“向商户支付20元”时，才会在对应效果前进入一次风险确认。

`unknown` 不会因用户确认而升级为权限；它在到达 Qwen 或机械臂前以“没有可执行语义类型”阻塞，必须先由高层任务图给出明确 EffectIntent。

## 四、确认作用域和动作回执

所有一次性 authority 都绑定 session、task、device、revision、subgoal、observation ID、fingerprint、decision node 和 action digest。任一字段变化、缺失、过期或已消费都会失败关闭。

动作后本地写 `VerifiedActionTransition`，绑定 requested、rebound、resolved 三层动作摘要、前后 observation/fingerprint、物理动作数、结果和错误。非等待动作必须恰好一个物理动作；revision 必须精确 `N -> N+1`；receipt 只能消费一次，不能跨会话或跨 revision 重放。

可见状态证据与历史动作回执严格分源：画面事实只能由绑定 scene 的 `visual_claim:<scene_id>:<digest>` 证明；“某动作已经执行”只能由一次性 controller transition receipt 证明，二者不能相互伪装、跨 revision 重放或靠复制自然语言升级为 authority。

## 五、默认入口和兼容层

默认控制台和 API 只创建 Universal Agent 会话。固定微信、抖音、固定步骤和旧任务队列默认返回 410，不参与产品执行。只有显式设置 `PHONE_AGENT_ENABLE_LEGACY_WORKFLOWS=1` 时，旧接口才用于离线兼容回归；它仍不是默认用户路径。

视觉模型、任务模型和设备能力均通过适配层提供，协议校验、风险权限、几何变换和动作回执由本地代码掌握。更换 Qwen 或 DeepSeek 提供方时，只需新适配器产生同一正式 schema，不得让模型供应商直接控制机械臂。

## 六、最终原则

协议的“严格”对象现在是权限、证据、唯一性、时效和设备能力，不再是禁止用户说自然动作。合法自然目标应尽量自动完成；不确定的落点、身份、风险或后置条件必须 0 动作停止。任何新 App 若能由现有动作组合完成，都不应再修改代码。
