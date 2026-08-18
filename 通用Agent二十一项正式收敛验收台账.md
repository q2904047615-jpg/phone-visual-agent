# 通用 Agent 二十一项正式收敛验收台账

更新时间：2026-08-19（Asia/Shanghai）

## 1. 本轮唯一验收缺口

阶段 2 已完成。原审计的 21 项现已全部完成生产代码收敛，并由跨措辞、跨画面、跨 surface
变化样本和完整离线回归证明默认链路不再依赖 App 名称、固定命令、固定步骤、固定坐标、
业务关键词或 `goal_relevant` 产生执行权限。**21/21 的状态是“已实现且完整离线回归通过”，
不是阶段 3 真机验收完成。**

本批不加载服务、不调用在线模型、不读取相机、不控制机械臂、不操作卖家 `main.exe`。
完整离线门关闭后，才另开最终三 App 真机验收批次。

## 2. 已观察事实与根因分类

1. `TaskSemanticIR` 已正式负责 effect 与风险，但仍只有 surface/entity/effect/binding；约束、子目标、
   完成状态和证据要求仍由旧任务图自由文本承担。
2. `VisualActionShadowReport` 已能生成 claim/relation/affordance/typed transition，
   但固定 `authoritative=false`、`execution_allowed=false`，正式 Qwen/控制器仍读取
   `states.goal_relevant` 和 `scene_changed/content_changed`。
3. 正式控制器仍含针对自然语言 meaning/label/state 的业务关键词判断，形成第二个业务规划器。
4. 设备能力 profile 已能描述输入、长按和拖动，但正式 action set 没有统一 `CapabilityGap`；
   双击、可变滑动、Enter、pinch/多指、硬件键等无法区分“设备不支持”和“协议未实现”。
5. `goal.entities.input_text` 与 Qwen context 仍硬限制 100 字、禁止换行；正式输入仍把一段文字
   当作一个实体，不能表达多字段、多段、长文本或多个接收者。
6. `EffectIntent` 能绑定 target/payload，但确认/自动执行预览没有统一 effect 实例视图；消息场景
   已较完整，其他 effect 仍可能只显示笼统风险文字。

主要类别：通用数据模型与正式 authority 切换缺口。不是微信、浏览器、某张截图、固定坐标或
某次 DeepSeek/Qwen 措辞问题。必须修改生产代码，但不得加入 App 名称、固定步骤或单命令分支。

## 3. 二十一项状态与正式关闭标准

| # | 问题 | 当前状态 | 本批正式关闭标准 |
|---|---|---|---|
| 1 | 风险判断混合 goal/constraints/completion 字段角色 | 已解决 | 风险只由 `EffectIntent` 与本地版本策略决定 |
| 2 | 本地规则可降低/删除风险 | 已解决 | 策略版本固定，模型/约束不能改写决定 |
| 3 | 风险类型未绑定具体 effect 实例 | 已解决 | 每个决定绑定唯一 `effect_id` |
| 4 | 权威实体只允许固定五键 | 已解决 | 任意 typed entity 可进入 IR；执行只引用显式 entity ID/role |
| 5 | 控制器使用业务语言关键词 | 已解决 | 正式控制器只读 action/effect/claim/capability，不重新理解业务文字 |
| 6 | 规范化混合结构修复与语义改写 | 已解决 | 结构修复、语义投影、authority 校验分层；authority digest 不被 normalizer 改写 |
| 7 | 无 App/surface 合同矛盾 | 已解决 | `device/system/current_surface/launcher/app` 统一且严格，空 App 只在非 App surface 合法 |
| 8 | 多 App 缺子目标级 surface 绑定 | 已解决 | 每个 executable subgoal 绑定 typed `surface_ref`；跨 App 先 Home 再从 Launcher 进入目标 App |
| 9 | constraints 仍是自由文本 | 已解决 | 关键约束成为 typed predicate；未类型化文字只能作非权威说明 |
| 10 | 完成/证据复制字符串，无 claim ID | 已解决 | desired state 与 evidence requirement 使用绑定 scene 的 visual claim ref 或 controller transition ref |
| 11 | `goal_relevant` 被当目标 authority | 已解决 | 正式候选由 entity/relation/affordance 产生；该布尔值只能诊断 |
| 12 | Qwen 弱候选与 `scene_changed` | 已解决 | 普通动作绑定 typed 后置状态；弱变化只允许显式 exploratory |
| 13 | 风险审计重复/同源 authority | 已解决 | 本地 effect policy 是唯一风险 authority |
| 14 | `unknown` 流程矛盾 | 已解决 | unknown 只允许只读重分类，物理动作前阻塞 |
| 15 | 非消息 effect 缺统一目标/载荷预览 | 已解决 | 所有 effect 用同一 typed preview，逐项展示 target/payload/policy/digest |
| 16 | 原命令是否授权动作规则不统一 | 已解决 | 原命令授权自动 ordinary effect；高风险 effect 按本地策略确认 |
| 17 | 动作能力集不完整且无正式 gap | 已解决 | 统一 action capability 描述和 `CapabilityGap`；支持项有 typed 参数，不支持项明确阻塞 |
| 18 | 输入单段/100字/无换行/单接收者 | 已解决 | typed input transaction 支持多字段/多段/多目标；长度和字符由设备 profile 决定 |
| 19 | scene role/state 缺正式关系模型 | 已解决 | claim/relation/affordance 正式接管候选绑定与验证 |
| 20 | 测试以已知样本为主 | 已解决 | 已有 metamorphic/property 基础，本批为每个切换补变化样本 |
| 21 | 文档与实现漂移 | 已解决 | 本台账、阶段 3、主交接与代码协议版本同批更新 |

## 4. 同类失败与变化样本集合

### 4.1 语义、surface 与约束

- 同一“在指定对象上产生指定效果”目标换成聊天、设置、文件和浏览器表面，typed IR 结构不变；
- 从 Launcher、目标 App 内页、系统弹层和 `current_surface` 开始，子目标 surface 必须不同且明确；
- “不要提交”“只读”“必须逐字等于 X”“保持开关关闭”成为 typed predicate；同义改写不改变摘要；
- 未知 entity 可作为 planner context，但不能成为执行 target/payload；重复 ID、悬空引用和跨 surface
  引用必须失败关闭。

### 4.2 视觉、候选与控制器

- 三个不同 App 的按钮、输入框、列表项和无文字图标，不改变 `goal_relevant` 即得到相同正式候选；
- 同名元素在不同容器、低置信、未完整可见、关系缺失时不得生成唯一候选；
- 普通点击、输入、选择和 effect 动作必须绑定具体 typed expectation；只有找寻/滚动等
  `exploratory=true` 动作可使用 viewport/observation changed；
- 改写 label/meaning 的近义词不能改变控制器结果；修改 action/effect/claim/capability ID 必须拒绝。

### 4.3 effect 预览、能力与输入

- 发送、关注、保存、权限、登录、付款、删除和 generic effect 使用同一 preview 结构；
- 普通 effect 自动，高风险 effect 确认；任一 target/payload/revision/observation/digest 漂移使旧 scope 失效；
- 设备支持双击/四向滑动/Enter 时生成对应候选；单指设备遇到 pinch/多指时返回 typed gap，绝不伪执行；
- 单字段短文本、多字段表单、长文本分段、换行、多个接收者、混合字符都由 input transaction
  表达；设备 profile 不支持的字符/长度/键位返回 typed gap；动作后仍逐段验证精确前缀。

## 5. 三个通用实现批次

### 批次 A：正式语义与任务结构

扩展 `TaskSemanticIR`：`ConstraintIntent`、`DesiredState`、`EvidenceRequirement`、
`SemanticSubgoal`、typed input/effect target 集合。旧 DeepSeek v3 JSON 暂作为远程传输格式，
本地只允许确定性编译一次；编译后的 IR 是正式语义 authority。normalizer 只修结构，
不得改变编译后 semantic digest。每个可执行子目标必须绑定 surface 和 desired state。

### 批次 B：正式视觉与控制器切换

把现有 shadow claim/relation/affordance/candidate 提升为正式本地视觉 authority。
Qwen 只选择 `candidate_id`；本地以原可信 scene 水合 geometry，控制器只校验 typed action、
effect、claim、relation、fresh observation、geometry 和 capability。`goal_relevant`、自然语言 evidence
和业务关键词不能创造权限。普通 candidate 禁止只有 `scene_changed`。

### 批次 C：effect 预览、能力与输入

新增统一 `EffectPreview`、`CapabilityGap` 和版本化 action/input capability。优先正式接入卖家工具
已有且能离线证明的单指动作；无法证明 ACK 或机械能力的动作只报告 gap。输入改为多 field/segment
事务，canonical 内容来自用户实体，Qwen 不生成正文；每段动作后重新观察并验证。

影响范围：`task_semantic_ir.py`、`deepseek_task_graph.py`、`visual_action_shadow.py`、
`qwen_visual_decision.py`、`universal_action_controller.py`、`generic_action_adapter.py`、
`robot_core.py`、`universal_agent_orchestrator.py`、Web 协议和对应测试。回滚方式是按批次整体撤销；
不得只恢复旧关键词、`goal_relevant` 或固定 App 路径。

## 6. 验证清单

1. 每批编辑循环只运行直接相关测试；不得在小改动间反复跑全量。
2. 重放语义 IR、视觉 shadow、阶段 2 typed receipt、消息风险和阶段 3 历史失败夹具。
3. 每个问题至少一个正向样本和一个换 App/换措辞/换状态的变化样本。
4. 属性测试覆盖 ID 稳定、顺序变化、约束位置变化、无关 wording 变化和 stale digest。
5. 能力测试覆盖 supported 与 unsupported/gap，证明不支持动作在相机/硬件前停止。
6. 输入测试覆盖多字段、分段、长文本、换行、多目标与设备差异。
7. 批次全部稳定后只运行一次 Python 完整 discover、前端协议、浏览器合同、静态编译和
   `git diff --check`。
8. 完整回归无断言失败后，更新 `项目交接文档.md` 并创建仅本地提交；不推送。

## 7. 停止条件

- 需要 App 名称、固定页面、固定步骤、固定坐标或单截图分支时停止并撤销该方向；
- 任何修改削弱 stale scope、fresh observation、geometry、设备独占或一动作一观察门禁时停止；
- shadow 提升后若模型能改写本地 candidate、effect 或 expectation，停止；
- 无法证明的卖家动作不得标记 supported；必须返回 capability gap；
- 相关测试未绿或完整回归有断言失败，不加载服务、不进行真机；
- 完成 21/21 离线合同之前，不启动最终三个不同 App 的正式安全验收。

## 8. 2026-08-19 最终离线结果

- 21 项全部进入正式默认链路；视觉 authority 为
  `2026-08-19-visual-action-authority-v1`，语义/风险 authority 为
  `2026-08-19-semantic-risk-authority-v2`。
- Python 完整回归：`1436/1436`。首次全量运行唯一失败是旧测试仍要求普通开关效果被阻塞；
  按已确认的本地风险政策更新为“先形成唯一候选、确认后只执行一次”的变化样本后，定向
  `1/1` 与最终完整回归均通过。生产逻辑没有为该样本增加特例。
- 前端协议、触控页、主浏览器与阶段 2 控制台合同：`43/43`。
- 已验证多 App/surface、多个任意实体、多字段/多目标/长文本/换行、typed evidence ref、
  typed effect preview、正式视觉候选和 supported/unsupported capability gap 的正反样本。
- 双击、Enter、pinch/多指和未认证硬件键没有被虚构为已支持：卖家链路不能提供安全 ACK
  或单指设备不能完成时，正式返回 `CapabilityGap`，并在模型、相机和机械臂前 0 动作停止。
- 本批没有加载或重启服务，没有访问在线模型、相机或机械臂，也没有操作卖家 `main.exe`。
  下一项决定性工作是加载本地提交后，以新会话完成三个不同真实 App 的阶段 3 验收。

## 9. 阶段 3 首次在线门发现的长否定作用域缺口

- 新版本加载后，Settings 新会话 `a789341d5a1941c8b66ae6f53edb600d` 在 DeepSeek 初始图阶段
  0 动作停止。真实约束是“不得包含坐标、Shell、ADB、keycode 或 main.exe 指令”，本地却报告
  `constraints` 含越权执行细节；无 Qwen、相机动作或机械臂动作，手机仍在 Launcher。
- 离线最小复现证明 `Shell/ADB/keycode` 均被识别为否定项，只有最后的 `main.exe` 因距离句首
  “不得”超过旧固定 40 字符回看窗口而失去否定作用域。这是跨语言、跨 App 的解析器长度缺陷，
  不是 Settings、模型随机措辞或手机状态问题。
- 通用修复：低层控制词的否定判定使用当前句/分句的完整前缀，不再截取固定字符数；句号、
  分号、换行以及“但/但是/然而/不过/然后/随后/接着”等转折或续行动词会重置否定作用域。
  不改任务、App、坐标权限、风险政策或视觉候选。
- 正向样本：上述真实长中文列表，以及长度和顺序变化的英文/中文否定列表。反向样本：
  “不得使用 Shell；然后调用 main.exe”和“Do not use ADB, but invoke main.exe”，后半句仍必须
  报越权执行细节。修复只在这些正反样本和 DeepSeek/编排相关回归通过后重新加载；不得直接
  重发在线请求试错。
- 离线结果：定向正反样本 `4/4`，DeepSeek+编排 `367/367`，Python 完整回归 `1438/1438`；
  只有既知测试子进程 `ResourceWarning`，无断言失败。修复已提交为 `bb4a0dd` 并只重载项目
  Uvicorn；卖家 `main.exe` PID 和启动时间未变化。原失败 session 已终止且不可复用。

## 10. 阶段 3 第二次在线门发现的几何标签集合错配

### 10.1 验收台账与根因证据

- 当前阶段合同缺口仍是：在一个全新 Settings 会话中完成“打开设置、只读确认设置主页、返回
  桌面”的同会话真实闭环；第二个会话 `634a02a077fd49e2b4a7fde6dd0a7d2a` 尚未产生物理动作。
- 已观察事实：DeepSeek 图、正式视觉候选和控制器策略均已通过，唯一候选是 `e1/设置`；可信
  observation 明确保存原始证据“灰色齿轮图标，下方文字‘设置’”。执行前独立几何审计却在
  本地、模型调用之前误报“缺少原始可见证据”，`physical_actions=0`。
- 离线最小复现：现场 scene 有 10 个非空文字标签。`element_geometry_audit_prompt` 的严格合同
  最多接收 8 个 `visible_literal_labels`，观察器却把整个 scene 的 10 个标签无差别传入；因此
  每条证据都因标签集合无效被跳过。传入前 8 个或只传目标证据实际提到的标签均通过。
- 根因分类：代码缺陷——调用方与严格几何子协议的输入集合错配。不是 Settings 特例、视觉
  漂移、证据缺失、模型随机输出或机械臂问题，因此必须修改通用生产代码。

### 10.2 同类失败与变化样本

- 现场样本：Launcher 上 10 个有文字 App，目标证据只提到“设置”；scene 密度不得让目标证据
  在模型调用前失效。
- 变化样本：任意列表/桌面含 12 个以上有文字控件，目标证据只提到目标及一个邻近文字；几何
  子协议只能收到这两个实际相关标签，不得泄露或枚举其余页面标签。
- 反向样本：一条不可信证据确实包含超过 8 个不同可见标签，或包含未被真实 label 覆盖的
  控制指令词时，仍必须失败关闭；不得截断后伪装成合法证据。

### 10.3 通用修复、影响与回滚

- 在 `GenericSceneObserver.audit_element_geometry` 中按每条原始证据独立构造 label allowlist：
  始终包含非空目标 label，并只追加该证据字符串中逐字出现的其他 scene labels；保持顺序稳定、
  去重且不超过严格协议上限。页面中未出现在该证据里的 App/控件 label 不进入几何 prompt。
- 不提高 8 项上限，不改变 evidence 控制信息过滤、唯一匹配、crop-local 坐标、置信度、完整
  可见性、fresh observation、IoU 或一动作一观察门禁。超过上限的真实相关证据继续被拒绝并可
  尝试元素的下一条独立证据。
- 影响范围仅 `generic_scene_observer.py` 与几何审计测试；回滚可整体撤销该批，不影响正式语义、
  风险、候选、控制器或硬件传输层。

### 10.4 验证清单与停止条件

1. 用 10 标签现场结构做正向测试，断言模型 prompt 只含目标证据实际提到的 label。
2. 用高密度跨 surface 变化样本验证无关 label 不进入 prompt；用超过 8 个真实相关 label 和
   未覆盖控制词做反向测试，断言 0 模型调用、0 动作。
3. 运行 element geometry 最小测试、observer/adapter 相关回归、一次 Python 完整回归和
   `git diff --check`；全部通过后本地提交。
4. 只重载项目 Uvicorn，核对 controller/camera/busy/device、卖家 `main.exe` PID/启动时间和
   新鲜 observation；随后只启动一个全新 Settings 验收会话。
5. 若相同根因再次出现或修复需要 App 名、固定位置/坐标、降低协议门禁，则停止，不再在线采样。

离线结果：几何审计定向测试 `36/36`，observer/adapter/通用 mock 相关回归 `320/320`，
Python 完整回归 `1441/1441`，静态编译与 `git diff --check` 均通过；只有既知测试子进程
`ResourceWarning`，无断言失败。待本地提交并只重载项目 Uvicorn 后，以一个全新 Settings
会话验证该通用修复；旧会话 `634a02a077fd49e2b4a7fde6dd0a7d2a` 已取消且不得复用。

## 11. Settings 首次真实动作后的中文主页身份语法缺口

### 11.1 验收台账与根因证据

- 提交 `929aee1` 加载后，0 动作观察 `ed9ebd9bbfdb24c30760` 证明 Launcher 稳定、唯一设置入口
  和原始证据完整。全新会话 `54c5961876e645a393b2740d09468568` 的独立几何审计通过，机械臂
  执行 1 次 `tap_semantic`；receipt `receipt_b5aec7fe3b0e4d55a8553656f8e2da50` 为 matched，
  前后 observation/fingerprint 均更新。没有自动重试，也没有执行后续 Home。
- 动作后可信 scene 明确为 `app_id=settings`、`screen_id=settings_home`，且有标题元素
  `label=设置 / meaning=page_title`；当前唯一缺口是 DeepSeek 重规划在本地身份门报：
  `completion_conditions.settings_home_visible` 缺少结构化画面身份锚点，session 最终 blocked。
- 离线复现证明：命名页面解析器只把“页面/界面/屏幕/视图”等当容器，不识别中文“主页/首页”。
  因而“设置主页的界面元素可见”会从后面的“界面”切分，错误提取锚点“设置主页的”，无法与
  结构化标题“设置”匹配。这是跨 App 的中文页面语法缺陷，不是视觉、硬件、Settings 特例或
  DeepSeek 无证据完成。

### 11.2 同类样本、通用修复与边界

- 现场正向：`设置主页的界面元素可见` 应提取命名锚点“设置”，由 `settings_home` 和标题“设置”
  共同支持；任意变化样本 `音乐首页的列表可见` 应提取“音乐”，不能依赖 App 分支。
- 反向：裸写“主页可见”或“首页可见”只是未命名页面，不得凭任意当前画面完成；Launcher 上
  只有某 App 入口仍不能证明该 App 已在前台，继续由编排器的 foreground App 独立门禁拒绝。
- 通用修复：把“主页/首页/主页面”加入命名视觉容器语法，并同步加入开头未命名容器与通用
  容器词清理；不新增 App alias，不读取 summary 自由文本作为身份，不改变 typed visual claim、
  controller receipt、前台 App、fresh observation、scope 或一动作一观察门禁。
- 影响范围仅 `deepseek_task_graph.py` 及其身份语法测试；回滚可整体撤销该批。若修复需要 App 名
  分支、跳过前台校验或让入口文字直接证明前台页面，则停止并撤销。

### 11.3 验证与下一在线停止条件

1. 加入现场“设置主页”、另一 App“音乐首页”正向样本，以及裸“主页/首页”和错误前台反向样本。
2. 运行 DeepSeek 身份测试、任务图/编排相关回归、一次 Python 完整回归、静态编译和
   `git diff --check`，全部通过后本地提交。
3. 只重载项目 Uvicorn；因手机当前停在设置主页，新的跨 App 正式会话必须先通过 typed `home`
   回到 Launcher，再打开目标 App 并在末尾返回 Launcher，不能手工点击或复用旧 session。
4. 下一会话任一步失败立即保存证据并停止；不得为完成 Settings 样本自动重试失败动作。

离线结果：现场“设置主页”与“音乐首页”变化样本、裸主页/错误 App 反向样本通过；DeepSeek+编排
相关回归 `369/369`，Python 完整回归 `1443/1443`，静态编译与 `git diff --check` 通过。
完整回归仍只有既知测试子进程 `ResourceWarning`，无断言失败。待本地提交并只重载项目
Uvicorn 后，用全新 session 从当前 Settings 前台先返回 Home，再完成正式跨 App 闭环。

## 12. 浏览器动作后目标精查容量合同缺口

### 12.1 已有证据与根因分类

- Settings 正式会话 `0769cfcc3b624ba2b1d0f5ef66e00b64` 已完成：同一 session 依次执行
  Home、打开设置、Home 三个真实动作，三条 receipt 均 matched，revision `1→2→3→4`，最终
  `status=succeeded`、graph completed、active_subgoal=null，最终可信 scene 为 Launcher。
- 第二个 App 会话 `c9720422f4fa4c09a7e0624230b82650` 已在 Launcher 唯一浏览器入口上执行
  1 次真实 `tap_semantic`；动作后的四帧已保存，但目标精查 JSON 解析失败，session 为 failed、
  `physical_actions=1`，没有 receipt、没有最终 Home、没有自动重试。
- 原始响应长度 1809 字符，在第 5 个元素的 `bounds.w` 值之前以 `"w":` 截断；JSON 大括号、
  数组和字符串均未闭合。目标精查协议允许最多 12 个完整元素，调用预算却只有 700 token，当前
  输出的元素数量没有越过协议上限。这是输出容量与 schema 最大规模不一致的通用代码缺陷。
- 历史目录 `generic_supervised_20260818_004715_00ae1a44` 已出现同一目标精查“单结构标点修复”
  失败；另有完整但最小增量协议不合格样本。按同类模型合同失败两次的规则，本轮停止在线采样，
  不通过改措辞或重试绕过，先修复子协议容量和诊断边界。

### 12.2 通用修复与样本

- 将 targeted delta 的输出预算与同样允许 12 个元素的 compact 场景预算对齐为 2600 token；
  仍按需调用，不增加模型轮次，不改变严格 schema、最多元素数或可执行候选权限。
- 提示继续要求只报告直接相关元素，并明确不得为填充数组枚举无关导航标签；即使模型仍多报，
  本地 relevance、唯一候选、fresh geometry 和控制器门禁仍独立裁决。
- 保留现有错误分类和失败关闭控制流；本次截断由落盘 raw 的未闭合结构直接证明，不把诊断标签
  改动带入 adapter 的格式重观察分支，避免为了标签准确而新增任何模型重试。
- 正向样本：允许上限内的长 targeted delta 能在 2600 预算下完整返回并继续严格解析；变化样本
  为标题读取、列表序数和多控件结构。反向样本：截断 raw 准确分类且 0 后续动作，完整协议外
  字段、重复键、越界或非唯一目标继续失败关闭。

### 12.3 影响、验证与停止条件

- 影响仅观察器 targeted 调用预算、提示和错误分类及对应测试；不改 DeepSeek、正式视觉 authority、
  controller、风险、坐标、硬件或 App 分支。回滚可整体撤销该批。
- 运行 Qwen error/observer 最小测试、观察/adapter/编排相关回归、一次 Python 完整回归、静态
  编译和 `git diff --check`；全部通过后本地提交并只重载 Uvicorn。
- 手机当前可能停在浏览器；下一全新 session 必须先 Home，再以原浏览器读取目标重新验收，不能
  复用旧会话或旧 scope。若完整响应仍违反 strict delta，本轮立即停止，不再扩大协议字段。

离线结果：targeted/error/observer 最小测试 `196/196`；观察、几何、adapter、通用 mock、编排
相关回归 `483/483`；Python 完整回归 `1443/1443`；静态编译与 `git diff --check` 通过。
完整回归仅有既知测试子进程 `ResourceWarning`。曾尝试把修复失败的最终错误标签重分类为
`truncated_json`，相关回归立即证明它会触发额外格式重观察；该改动已完整撤回且未进入提交，
因此生产变化只剩 token 容量对齐、无关元素提示和 observer v55 版本标识。

## 13. 多步骤目标向当前观察泄露后续子目标

### 13.1 验收台账与根因证据

- v55 加载后，全新浏览器会话 `a24a7b16858d40c5a83528351aa2b1ec` 先执行 Home 并形成 matched
  receipt，随后执行打开浏览器；第二个动作后的四帧清楚显示浏览器要闻页，但目标精查因协议外
  元素几何停止。session 最终 `failed`、`physical_actions=2`，没有最终 Home，也没有自动重试。
- 本次响应已完整结束，长度 1958 字符、finish reason 为 stop，排除了第 12 节的 token 截断。
  顶层四字段正确；无关底栏“主页”元素却给出 `y=1150,h=80`，违反独立归一化 0..1000 合同。
  同一响应还枚举多条新闻正文，而当前 active subgoal 仅是“打开浏览器”。
- 代码审计证明 compact/targeted/system-ui/icon/input 等证据选择提示直接序列化根目标 context；
  根目标包含未来“读取标题、最终返回桌面”，尽管严格 `active_subgoal_visual_context` 已存在。
  `_needs_targeted_refinement` 也用根 objective 判断“打开/进入/启动”，导致已在目标 App 前台时仍
  为后续步骤触发精查。这是跨 App、跨多步骤任务的语义数据流缺陷，不是浏览器坐标特例。

### 13.2 同类样本与通用修复

- 现场样本：当前节点“打开浏览器”不得向 Qwen 暴露未来“读取标题、返回桌面”；前台已是
  `browser` 且当前节点以“打开”开头时，稳定 `browser_home` 不需要目标精查。
- 变化样本：任意 App 的“打开 App → 读取内容 → 返回”与“重新加载 → 输入文字 → 收起键盘”
  都只能按当前 active node 选择观察证据；切换到后续读取/输入节点后才允许相应精查。
- 反向样本：当前 active node 本身要求读取标题时，没有唯一 grounded `page_title` 仍必须精查；
  没有结构化 active context 的旧只读调用继续使用完整单目标 context，不得丢失目标。
- 通用修复：新增一个只读 observation context 投影，只保留 active subgoal 的 id、objective、
  constraints、completion conditions、external impact 和 goal entities；所有模型证据选择 prompt
  使用该投影。targeted 判定和 goal terms 同样只解析当前节点；正式 App 身份、risk、authority、
  坐标、候选、fresh observation 和 controller 门禁保持不变。

### 13.3 影响、验证与停止条件

- 不接受、裁剪、换算或猜测 `y>1000`，不丢弃 malformed target，也不添加 App 名、固定页面、
  固定坐标或动作脚本。旧响应仍是失败样本；修复只消除不该发生的跨节点提示污染。
- 先用“打开浏览器/未来读取标题”现场结构和另一个多步骤变化样本验证 prompt 不泄露；验证当前
  open-app 节点不精查、切换 read-title 节点后仍精查。再运行 observer/adapter/编排相关回归、
  一次 Python 完整回归、静态编译和 `git diff --check`。
- 全部离线证据通过后才提交并只重载项目 Uvicorn。下一全新浏览器 session 必须仍从 Home 开始、
  使用原目标；若真正的 read-title 节点再次产生严格协议失败，则保存为新的当前缺口并停止，
  不放宽坐标或复用失败 session。

离线结果：observer 定向正反测试 `196/196`，观察、几何、adapter、通用 mock 与编排相关回归
`494/494`，Python 完整回归 `1445/1445`；完整回归只有既知测试子进程 `ResourceWarning`，无断言
失败。observer 版本更新为 v56，未放宽 targeted delta schema 或坐标边界。

## 14. Launcher 同名可见文字被误当成目标绑定已经成立

- v56 全新会话 `fbaccfae9a354bd3ba3c84a9a7453e02` 的 Home 动作 matched，Launcher 四帧和
  fingerprint 更新均完整；随后 0 次打开动作就 blocked。可信 scene 有唯一 `e3/浏览器/open_browser`
  和原始视觉证据，但 Qwen 将 12 个桌面元素全部写为 `goal_relevant:false` 且未提供完整可见性。
  本地正式视觉权威因此只产生 back/wait 候选，正确拒绝 tap；不能通过控制器放宽解决。
- 根因在 `_needs_targeted_refinement`：没有可信 goal element 时，只要任意 summary/label/meaning
  出现目标词就跳过精查。“看见同名文字”只能证明元素存在，不能证明它是当前子目标的完整唯一
  目标。这是任意 Launcher/App 入口和任意同名控件都会遇到的通用绑定缺口。
- 修复仅收紧“打开/进入/启动目标 App”的观察合同：目标 App 已在前台时不精查；目标 App 尚未
  在前台且没有通过现有可信条件的 goal element 时，必须进行一次当前子目标精查。精查仍必须由
  Qwen 明确给出 goal relevance、evidence、合法 bounds，之后还要独立 geometry/fresh/IoU/policy
  门禁；本地不把 false 改 true，也不新增 App 名、坐标或固定入口。
- 正向现场样本为 Launcher 上浏览器文字存在但 relevance=false；变化样本为音乐/设置等不同 App。
  反向样本为目标 App 已在前台，即使无 element 也不得为了 future goal 精查；已有低置信/不完整
  goal element 仍按原门禁精查。先完成 observer 定向、相关回归和一次完整回归，全部通过后提交、
  只重载 Uvicorn，再以原目标创建全新 session。旧 blocked session 不复用、不执行后续动作。

离线结果：observer 定向 `197/197`，观察、几何、adapter、通用 mock 与编排相关 `495/495`，
Python 完整回归 `1446/1446`；静态编译和差异检查通过。完整回归只有既知测试子进程
`ResourceWarning`，无断言失败。observer 版本更新为 v57。

## 15. Qwen 完成一个前缀子目标后被误要求整个任务结束

- v57 全新会话 `efd20d3a47e94e41b12fa3fee1d7ecc7` 在初始 Launcher 上 0 动作观察；
  Qwen `finished` 仅声明当前 `return_home_initial` 已满足。DeepSeek revision 2 正确把该子目标标为
  completed，并唯一激活 `open_browser`；本地 `_review_completion_candidate` 却只接受整个 graph
  completed，否则统一 blocked。这使正常的多步骤目标在第一个已满足前缀处停止。
- 根因是编排状态机把“当前 active subgoal 完成”与“整项任务完成”混为一个终态，不是 Qwen
  误判、浏览器特例或动作候选问题。现场 Qwen 证据只引用 Launcher scene，DeepSeek 没有把
  `open_browser` 误完成；两层证据本身正确。
- 通用修复：复核后若整个 graph completed，仍 succeeded；否则只有当旧 active subgoal 在精确
  `revision+1` 中确实 completed、且出现唯一不同的新 active subgoal 时，才接受为合法前缀推进。
  下一节点若需风险确认则进入对应确认态；其余一律 `needs_reobservation`，清除旧 controller 决定
  和 authority，用新观察生成新 Qwen 决策。旧子目标仍 active、缺失、跳过或图身份不匹配继续 blocked。
- 正向样本为“初始桌面已满足 → 打开 App”两节点；变化样本为任意已满足页面前缀后进入下一个
  navigation/read-only 节点。反向样本保留“DeepSeek 未完成旧子目标”和“直接声称整个任务完成但
  证据不足”，均 0 动作 blocked。修复不复用旧 Qwen 决策、不自动执行下一动作、不放宽风险或视觉门禁。
- 先运行 completion review 定向正反测试、DeepSeek/编排/web 相关回归和一次完整回归；全部通过后
  本地提交、只重载 Uvicorn，再以原目标创建全新 browser session。旧 blocked session 不复用。

离线结果：completion review、DeepSeek、编排与 web 相关回归 `540/540`，Python 完整回归
`1447/1447`；完整回归只有既知测试子进程 `ResourceWarning`，无断言失败。静态编译与差异检查
通过，未改变观察器、模型 schema、动作候选、风险或硬件层。

## 16. 命名目标 App 的诚实 unknown 身份没有进入独立审计

- 提交 `2547283` 加载后，全新会话 `137d2949f3064e1286bff17b969c07f2` 已通过前缀推进、严格
  浏览器入口精查、独立 geometry 与 fresh 复核，执行 1 次 `tap_semantic` 并形成 matched receipt
  `receipt_ffb43b88fcf94d1ab3f657ccd9c49101`。动作后 scene 稳定为 `screen_id=news_feed`，但
  `foreground_app_id=unknown`；DeepSeek 声明浏览器主页完成时被本地命名页面身份门正确阻断。
- 项目已有完全不接收用户目标的 foreground App identity audit，并严格要求纯 JSON、独立可见身份
  证据、confidence>=0.90；低置信或审计仍 unknown 会失败关闭。但调用条件只覆盖
  `current_foreground/current_app` 等非法引用占位符，未覆盖“模型诚实返回 unknown、任务却绑定了
  真实命名 App”的情况，导致专用安全审计永远没有机会工作。
- 通用修复：非法身份占位符继续始终审计；foreground 为 unknown 时，仅当根任务 `app_id` 是非空、
  非 unknown、非引用占位符的真实命名 App 才调用独立身份审计。prompt 不包含目标 App、用户目标、
  计划或前次答案；审计返回其它 App 会保留其它身份并由命名页面门拒绝，返回 unknown 也继续阻塞。
- 正向现场样本是 unknown+browser，变化样本为 unknown+settings；反向为 unknown+current_foreground，
  不增加无意义调用。已有结构化 foreground 不增加调用；低置信、重复键、协议外字段和不安全 evidence
  继续失败关闭。修复不把目标 App 当当前 App、不修改 screen、候选、动作或风险。
- 先运行 identity audit 定向与 observer/编排相关回归，再运行本批一次完整回归；全部通过后提交、
  只重载 Uvicorn，以原 browser 目标创建全新 session。旧 failed session 不复用、不自动补 Home。

离线结果：observer/identity 定向 `199/199`，观察、几何、adapter、通用 mock 与编排相关
`498/498`，Python 完整回归 `1449/1449`；完整回归只有既知测试子进程 `ResourceWarning`，无断言
失败。静态编译与差异检查通过，observer 版本更新为 v58。

## 17. 当前打开 App 节点没有形成逐字视觉目标

### 17.1 验收台账与根因证据

- 阶段 3 当前缺口仍是：在一个全新 Browser 会话中，从 Launcher 打开 Browser、读取打开后页面
  的主标题或错误提示、再返回 Launcher，并以同一会话 `succeeded` 和逐动作 matched receipt 证明。
- v58 全新会话 `54efb63b307040a1b0375d0e2725272b` 只执行了 1 次 Home 且 matched；进入
  `open_browser` 后，真实画面上“浏览器”入口可见，Qwen 决策也逐字说明看见它，但可信 scene 和
  本地 choices 只包含设置、电话、信息、相机，因此 Qwen 正确 blocked，Browser 点击为 0 动作。
- `ObservationBridge.goal_draft()` 把根 `goal.entities` 原样复制给每个活动节点。任务图虽然有类型化
  `target_apps=[browser/浏览器]`，当前节点也明确为“打开浏览器”，但活动观察上下文没有
  `target_ui_label=浏览器`；已有唯一逐字标签归一化和正式 `binds_surface` 因而没有可用的节点级绑定。
  这是多 App 任务中“当前节点指向哪个 App”的通用语义投影缺口，不是 Browser 名称或坐标特例。

### 17.2 同类样本、通用修复与回滚

- 现场样本：单目标 App 的当前节点“打开浏览器”；变化样本：多目标 App 任务中当前节点唯一写明
  “启动音乐”。两者都应只在当前节点观察上下文得到对应逐字 App 标签。
- 反向样本：当前节点是“读取浏览器标题”或“返回桌面”、节点同时提到多个目标 App、目标为
  `current_foreground` 等引用占位符、画面中同名元素缺失或重复，均不得形成可执行绑定。
- 通用修复：本地只读投影从任务图的目标 App 集合中找出“当前 navigation_only 节点以打开/进入/
  启动/切换到/前往语义唯一指向”的一个 App，把其 `app_name` 写入该节点副本的
  `goal_entities.target_ui_label`。不修改持久化任务图、根 entities、Qwen 决策或动作 authority。
- 观察器继续只接受唯一逐字同名元素；该修复不会补 geometry、confidence、fully_visible、evidence，
  更不会生成坐标。后续独立 geometry、确认前 fresh observation、IoU、policy 和一次一动作门禁保持不变。
  回滚只需移除节点级投影 helper，不影响任务图 schema、模型协议和历史会话。

### 17.3 验证清单与停止条件

- 正测当前“打开浏览器/启动音乐”得到节点级逐字标签；正测多 App 中仅当前唯一 App 被选中。
- 反测后续读取、Home、多个 App 同时被提及、引用占位 App 均不添加标签；持久化
  `graph.goal.entities` 不得被修改。复用 observer 现有正反合同证明唯一同名可纠正 relevance，重复同名
  仍不产生唯一可信目标。
- 先运行 ObservationBridge、observer、visual shadow 和编排最小回归，再运行本批一次完整 Python
  回归、静态编译和 `git diff --check`。全部通过后本地提交，只重载项目 Uvicorn。
- 重载后先做 0 动作设备/服务门检查，再用原 Browser 目标建立全新 session；旧 blocked session、
  observation 和 scope 全部不复用。若 Browser 仍未进入唯一完整候选，本轮停止并保存新的模型响应证据，
  不放宽唯一性或可见性，也不试探性点击。

离线结果：节点投影、唯一/重复标签及 surface binding 定向 `11/11`；ObservationBridge、observer、
visual shadow 与 TaskSemanticIR 相关回归 `409/409`；Python 完整回归 `1452/1452`。完整回归只有
既知测试子进程 `ResourceWarning`，无断言失败。静态编译与差异检查通过；未修改任务图 schema、
模型输出协议、坐标、风险或机械臂层。

## 18. 一次性导航回执与命名页面视觉门互相否定

### 18.1 验收台账与根因证据

- 提交 `00041c6` 加载后的全新 Browser 会话 `d91a3fb5a1ae4adab50f2e4cdfef5ae3` 成功取得
  唯一 Browser 入口、正式 `surface.active_ref=surface_browser` transition、确认前 fresh geometry 和
  policy 放行；只执行 1 次 `tap_semantic`，动作后新 fingerprint、4 帧、matched receipt
  `receipt_d47db791384b4b10b486849dcd8652b3` 均完整，没有自动重试。
- 动作后真实页面为 Browser 的新闻流首页，但目标无关身份审计按可见界面类别报告
  `foreground_app_id=news_aggregator`、`screen_id=unknown`，且没有 Browser 品牌标题。DeepSeek 正确使用
  严格绑定的 controller transition 完成 `open_browser` 时，本地 `_validate_revision()` 先接受该回执
  只能完成上一 `navigation_only` 节点，随后又无条件要求“浏览器主界面”的结构化视觉身份，最终以
  “命名页面完成声明缺少结构化画面身份锚点”阻塞。
- 这是 source-aware completion 合同内部冲突：一个本地类型化回执已经被定义为可完成其严格绑定的
  navigation-only 子目标，却又被只适用于视觉完成声明的命名页面门否定。不能通过把
  `news_aggregator` 改写为 Browser、给模型目标暗示或增加 Browser 首页特例解决。

### 18.2 通用修复、变化样本与边界

- 仅当新完成节点的 `completion_evidence` 实际引用本次 ObservedState 中的
  `controller_transition_evidence_refs`，且既有校验已经证明它绑定上一 active、impact 为
  `navigation_only`、outcome 为 matched、receipt/scope/新 observation 完整时，命名视觉身份门不再
  对同一节点重复裁决。该回执证明的是“一次导航转换已完成”，不伪造成视觉 App 身份。
- 现场样本：从唯一 Launcher App 入口进入一个没有品牌标题、外观像新闻流的首页。变化样本：从唯一
  命名入口进入无品牌的音乐/文件首页；只要严格回执绑定当前导航节点，都可推进到下一次独立观察。
- 反向样本：只用 summary/visible text 声称命名页完成、controller ref 未被 completion evidence 引用、
  wrong subgoal、mismatched、read_only、external_state/unknown、全局完成条件，全部继续失败关闭。
  后续“读取标题/错误提示”仍只认当前结构化画面，不能复用导航回执伪造读取结果。
- 不修改 foreground_app_id、视觉 identity audit、元素/坐标、风险或动作权限；回滚仅恢复一次无条件
  `_require_named_visual_identity_grounding()` 调用和提示说明。

### 18.3 验证与停止条件

- 新增与现场同构的 `browser -> news_aggregator` 正测：严格 matched controller ref 可完成
  `open_browser` 并激活下一 read-only 节点；同一结构改用普通 visible evidence 必须仍因身份不符拒绝。
- 复用已有 wrong subgoal、未消费回执、mismatched、external_state、receipt 重放和全局视觉条件反测；
  运行 DeepSeek/编排相关回归与一次 Python 完整回归，静态编译、diff-check 后本地提交。
- 只重载 Uvicorn 后，手机当前停在 Browser；下一全新原目标会先执行 Home，再重新进入 Browser。
  旧 blocked session 和 receipt 不复用。若下一 read-only 标题节点失败，保存为新的独立缺口并停止，
  不把导航回执用于读取结果。

离线结果：DeepSeek 任务图正反合同 `209/209`，DeepSeek 与通用编排相关回归 `375/375`，
Python 完整回归 `1454/1454`；完整回归只有既知测试子进程 `ResourceWarning`，无断言失败。
严格回执仍只完成其绑定的 navigation-only 转换，普通可见文字无法替代命名页面身份。静态编译与
差异检查通过后本地提交；未修改 observer、视觉身份、风险、坐标或机械臂层。

## 19. 无坐标系统 Home 被错误送入控件目标精查

### 19.1 验收台账与根因证据

- 提交 `ba28a51` 加载后，全新 Browser 会话 `aa59f302` 在第 1 轮动作前观察即失败，
  `physical_actions=0`，没有操作手机。当前 active 节点为 `return_home_initial`，objective 是
  “返回手机桌面”，后续 Browser、标题读取等内容没有进入当前观察焦点。
- compact 后因 `screen_id=unknown` 触发通用 target refinement；Qwen 把 Browser 内底栏“主页/视频/
  窗口/免费小说/我的”当成目标控件，输出的 x/y/w/h 中 y 达到 1850。严格 parser 正确拒绝该非
  0..1000 delta。若把 y 除以 2000 或放宽 schema，反而可能把 Browser 的“主页”标签授权为系统
  Home，属于危险误修。
- `home/back/reveal_system_navigation` 在正式 Qwen/action 合同中本来就是无元素、无坐标系统动作；
  targeted refinement 只能补元素几何，不能提高 base scene confidence，也不能证明系统 Home。
  因此根因是观察策略没有区分“元素绑定动作”和“明确无坐标系统动作”，不是某个 App 或坐标格式。

### 19.2 同类样本、通用修复与回滚

- 现场样本为 Browser 内存在“主页”标签时的“返回手机桌面”；变化样本为任意 App 内的“回到手机
  主屏幕”。两者均不得为了找可点击元素调用 targeted refinement。
- 反向样本为“打开应用主页”“进入首页”“点击主页标签”：这些仍是元素/页面语义，不能被识别成
  系统 Home，仍按原规则在缺少可信目标时精查或失败关闭。根目标未来提到“最后返回桌面”，但当前
  active 节点是“打开 Browser”时，也不能跳过当前元素精查。
- 通用修复：仅当 active_subgoal_visual_context 的 impact 为 navigation_only，且 objective 逐字表达
  “返回/回到手机桌面或主屏幕”的系统 Home 语义时，`_needs_targeted_refinement()` 直接返回 false。
  compact scene 原样进入 Qwen；Qwen 仍必须正式提出无元素 home，本地 policy、确认前同帧身份、动作后
  新观察和 matched receipt 全部保持。低置信 compact 不会被虚假提高，后续门禁仍可 0 动作拒绝。
- 不解析本次非法 delta、不增加 App/页面/坐标分支，不修改 action authority 或机械臂。回滚只需移除
  系统 Home 的 refinement 路由判断。

### 19.3 验证清单与停止条件

- 正测 `返回手机桌面`、`回到手机主屏幕` 在 unknown screen 且无目标元素时不触发第二次模型调用；
  变化样本即使存在 Browser “主页”tab 也不把它当系统 Home 精查目标。
- 反测 `打开应用主页` 和 active `打开浏览器`（根目标未来含返回桌面）仍触发必要精查；页面标题、输入、
  命名 App 入口等原 target refinement 测试全部保持。
- 运行 observer 定向与 Qwen/adapter/编排相关回归，再运行一次完整 Python 回归、静态编译和 diff-check。
  全绿后提交并只重载 Uvicorn。旧 `aa59f302` 会话不复用；重载后创建全新 Browser 会话。

离线结果：系统 Home 路由正反合同 `4/4`；observer、Qwen、adapter 与通用编排相关回归
`500/500`；Python 完整回归 `1456/1456`。完整回归只有既知测试子进程 `ResourceWarning`，无断言
失败。observer 版本更新为 v59；非法目标精查响应仍被严格拒绝，没有新增坐标归一化或控件权限。

## 20. 新活动子目标复用了旧目标条件下的动作后 scene

### 20.1 验收台账与根因证据

- v59 全新 Browser 会话 `e8ae082d1ab847f0830ab744651dfdb7` 执行 1 次 system Home，动作前后
  fingerprint `5110dcc5b5a47d155809 -> 3b3c22487ed7a2fe5f04`，新 4 帧、matched receipt
  `receipt_45948a6f860e4c6a8621a5f6088473e2` 完整，手机安全停在 Launcher；随后 0 动作 blocked，
  没有试探性点击 Browser。
- Home 动作后的 scene 是 observer 按旧 active 节点 `return_home_initial` 生成的；v59 正确不做控件
  精查，因此 scene elements 为空。DeepSeek 随即完成旧节点并激活 `open_browser`，但
  `_advance_after_observation()` 直接把这份旧节点 scene 交给新节点的 Qwen。虽然原图中 Browser 图标
  可见，新节点 choices 仍只有 back/wait，Qwen 正确拒绝。
- observer 输出包含目标相关元素，天然是 goal-conditioned observation。活动子目标改变后，同一 scene
  的元素集合不再对新节点完备；这同样会影响“打开 App -> 读取标题”“进入列表 -> 查找条目”等任意
  跨节点任务，不是 Browser 识别特例。

### 20.2 通用修复、变化样本与边界

- 当且仅当 DeepSeek 新 revision 将上一 active 子目标完成并激活了不同 subgoal_id，且新节点没有先
  进入风险确认，编排器不得直接在旧 goal-conditioned observation 上调用 Qwen；改为状态
  `needs_reobservation`，使既有安全自动循环用新 `goal_draft` 进行一次 0 动作只读重观察。
- 重观察继续捕获 4 帧、生成新 observation_id/fingerprint 绑定、新 Qwen 决策和新一次性 scope；旧
  confirmation、Qwen decision 和 controller authority 已在重规划前失效。重观察失败即 0 动作停止。
- 现场样本为 Home -> 打开 Browser；变化样本为打开无品牌 App -> 读取当前页面标题。反向样本为
  revision 增加但 active subgoal_id 未变的连续滚动/分页，此时可继续使用刚取得的动作后 scene，避免
  不必要重观察；新节点需要风险确认时仍先进入风险门，不提前调用视觉模型。
- 不改变 DeepSeek 图、observer schema、App 名称、坐标、风险或机械臂动作；复用现有
  `_refresh_decision_locked()`，不新建第二条观察实现。回滚仅移除 active 节点变化后的状态转换。

### 20.3 验证清单与停止条件

- 正测动作后 active ID 变化：`confirm_one` 后必须为 needs_reobservation，旧 scene 不得产生第二个
  Qwen 决策；`refresh_decision` 必须接收新节点 goal context、保持物理动作数不变并产生新 observation。
- 正测新 read-only 节点：重观察后可由新目标相关标题完成，不能复用上一导航回执伪造标题。
- 反测 active ID 不变的连续导航仍可直接生成下一确认；风险后继仍进入 risk confirmation；任何
  重观察异常为 0 新动作并失败关闭。
- 运行编排定向及 DeepSeek/Qwen/adapter/web 相关回归，再运行一次完整 Python 回归、静态编译和
  diff-check；全绿后提交、只重载 Uvicorn，并以全新 Browser 会话验收。旧会话不恢复、不复用 scope。

离线结果：活动节点切换与 read-only 后继定向 `3/3`，通用编排 `167/167`，DeepSeek、Qwen、adapter、
编排与 web 相关回归 `679/679`，Python 完整回归 `1457/1457`。完整回归只有既知测试子进程
`ResourceWarning`，无断言失败。同一 subgoal 的连续导航仍直接推进；不同 subgoal 必须先完成 0 动作
goal-conditioned 重观察。

## 21. 本地权威导航回执仍依赖 DeepSeek 自行消费

### 21.1 验收台账与根因证据

- 提交 `5d7721f` 加载后的全新 Browser 会话 `f5bfc428e32c407ab2270b54d1262bb0` 从桌面
  0 动作推进到 `open_browser`，经新目标重观察取得唯一 Browser 入口，执行 1 次 `tap_semantic`；
  fingerprint `0e767e6e3aa80bc88ea6 -> ddaa95a53923529c4964`、新 4 帧、matched receipt
  `receipt_508ad70a79a247e9b27763c63388c71e` 完整，随后因 DeepSeek 没有完成 `open_browser` 而停止，
  没有第二动作。
- 相同合同在会话 `d91a3fb5a1ae4adab50f2e4cdfef5ae3` 中曾被 DeepSeek 正确消费；本次却保留旧
  active。严格 parser、prompt 和本地验证相同，说明“语言模型是否把本地权威回执抄入 status/evidence”
  存在随机性。继续重采样违反单次确定合同，也无法形成通用能力。
- controller transition 已严格绑定 session/task/device/revision/subgoal/decision、三层 action digest、
  before/after observation+fingerprint、恰好 1 次物理动作、matched outcome 和一次性 receipt；让 DeepSeek
  决定是否承认它，权责关系倒置。

### 21.2 通用修复、变化样本与边界

- 在 DeepSeek 单次返回后、正式 graph 校验前，本地仅对严格匹配当前 graph 的 matched transition 生效：
  若上一 active 是 navigation_only 且至少有一个绑定该节点的 controller evidence ref，本地把该旧节点
  状态置为 completed，completion_evidence 置为这些 typed refs。模型不得改写该节点的 objective、依赖、
  条件、风险或 impact；任一变化直接拒绝。
- 若模型已经正确完成节点，本地保留其声明并继续用原 source-aware 校验裁决；只有模型原样保留旧
  active 时，才从既有依赖 DAG 中只在唯一、无风险且 impact 为 navigation_only/read_only 的 frontier
  激活后继。多个可运行后继、
  external_state/unknown 或需风险确认的后继仍交给 DeepSeek 正确表达，否则失败关闭，不由本地猜选。
- 现场样本为打开 Browser 后进入唯一 read-only 标题节点；变化样本为任意 App 的唯一导航后继。反向
  覆盖 wrong session/task/device/revision/subgoal/observation、mismatched、receipt 重放、旧节点语义被改、
  read_only/external_state 完成、多个安全 frontier，均不得本地完成或猜选。
- 回执只证明一次导航转换，不能满足全局视觉条件、后续标题/文字读取、发送/关注/删除/付款等外部结果；
  后续节点仍必须按新目标重观察。无第二次 DeepSeek 采样，不修改视觉、坐标、风险或机械臂。

### 21.3 验证清单与停止条件

- 正测模型保留旧 active：本地一次消费回执、完成旧 navigation 节点、唯一安全后继 active，provider 只调用
  一次；模型已正确完成时输出等价。
- 反测 unsafe/ambiguous frontier、节点语义改写、mismatch/wrong binding/replay；模型伪造普通 evidence
  必须继续被拒绝，不能由本地回执掩盖。
- 运行 DeepSeek 正反合同、编排闭环及 Qwen/adapter/web 相关回归，再运行一次完整 Python 回归、静态
  编译和 diff-check；全绿后提交、只重载 Uvicorn，以全新 Browser 会话验收，旧 session 不恢复。

离线结果：DeepSeek typed receipt 正反合同 `210/210`，编排、Qwen、adapter 与 web 相关回归
`470/470`，Python 完整回归 `1458/1458`。完整回归只有既知测试子进程 `ResourceWarning`，无断言
失败。单次模型返回被保留；本地只补严格回执已证明但模型遗漏的 navigation-only 状态转换。

## 22. 动作后观察仍按动作前源控件做目标精查

### 22.1 验收台账与根因证据

- 提交 `499461d` 加载后的 Browser 会话 `e78eee54bb8a4bd48fd023fc5496377e` 先完成 system Home，
  经新节点重观察后完成一次 Browser 入口点击；会话累计 2 个物理动作。第二动作后的 4 帧已采集，
  但 adapter 仍把动作前 `open_browser/target_ui_label=浏览器` goal 原样交给 observer。
- Browser 图标在目标页面中理应消失，目标精查却转而枚举 Browser 内“窗口”tab，并再次输出 y=1850 的
  非法 bounds；严格 parser 拒绝后会话停止，没有再次点击。该响应本身也明确说“当前无直接浏览器
  启动入口”，证明精查对象已经过期。
- `GenericSingleActionAdapter._observe_stable_post_action_scene()` 不区分 before-target 与 after-result；
  所有动作后观察都使用 `goal.to_dict()`。对于“点击入口 -> 进入新页面”的 navigation transition，源控件
  不再是动作后证据，继续寻找它会导致任意 App 的入口、列表项、菜单项跨页后出现同类误判。

### 22.2 通用修复、变化样本与边界

- adapter 仅在以下条件全部成立时构造本地 post-navigation observation context：active impact 是
  navigation_only；resolved kind 属于已支持导航原语；expected_effect 明确
  `scene_changed=true` 且 `goal_complete_on_success=true`；没有 `element_state` 等持续控件后置条件。
- 该上下文移除 active goal_entities 的源 `target_ui_label`，把 active objective/condition 改为“观察本次
  导航后的当前稳定画面”，并写入本地 `observation_phase=verified_navigation_result_v1`。observer 对此阶段
  只做 compact 全景和既有独立 App/方向审计，不做元素 target refinement；controller 仍验证 fingerprint/
  expected effect。DeepSeek 完成旧导航后，新 active 节点按第20项再做一次完整 goal-conditioned 重观察。
- 变化样本为任意 App 入口、列表详情、菜单跳页。反向样本：input/clear/long_press/drag、预期
  element_state 的持久控件、external_state/unknown、没有 goal_complete_on_success 的滚动探索，均保留原
  post-action 目标观察，不能借阶段标记跳过结果元素核对。
- marker 只由本地 adapter 在动作已经发生后生成，不进入 Qwen 动作 authority、确认 scope 或任务图；
  用户/DeepSeek 同名实体不能让不满足上述 resolved 条件的动作取得该模式。不放宽非法 delta parser。

### 22.3 验证清单与停止条件

- 正测 navigation tap 的确认前 observer 收到原 target，动作后收到移除源 label 的阶段上下文；unknown
  screen 即使存在 App 内“主页/窗口”tab 也不触发 target refinement。
- 反测 external impact、element_state、非完成型 swipe/input/long_press/drag 保持原上下文；动作后失败仍
  记录恰好一次物理动作且不重试。
- 运行 adapter/observer 定向和 Qwen/编排/web 相关回归，再运行一次完整 Python 回归、静态编译与
  diff-check；全绿后提交、只重载 Uvicorn，以全新 Browser 会话验收，旧会话不恢复。

离线结果：导航结果观察正反合同 `3/3`，adapter/observer `288/288`，DeepSeek、Qwen、编排、Web 与
确认作用域相关回归 `658/658`，Python 完整回归 `1461/1461`；5 个变更 Python 文件静态编译和
diff-check 通过。完整回归只有既知测试子进程 `ResourceWarning`，无断言失败。严格 targeted delta
parser 未放宽；不满足导航结果合同的动作继续使用原目标做动作后观察。

## 23. 云端内容审核拒绝会使坐标无关 Home 也无法退出

### 23.1 验收台账与根因证据

- 提交 `e6a9abc` 加载后的全新 Browser 会话 `4b1c7c1ad1a44b75ba7786077469024a` 在第一个
  `return_home_initial` 节点、0 个物理动作处失败。阿里云返回
  `data_inspection_failed`；本地保存的 4 帧稳定，当前 Browser 新闻流含政治新闻标题，图像编码和相机
  均正常。该失败发生在视觉请求进入模型之前，不是 scene JSON、坐标或动作后观察合同错误。
- 当前系统 Home 虽然没有元素坐标，仍先把整张 App 内容分别发送给 compact observer、Qwen action
  selector 和独立方向审计；任一任意 App 页面触发云端内容审核后，Agent 无法退出到桌面，形成跨 App
  的永久阻塞。重复发送相同整图既不会缩小缺口，也违反同类服务失败止损。

### 23.2 通用修复、边界与回滚

- 仅当当前 active subgoal 被严格识别为 `navigation_only` 的 Android 系统 Home、完成条件明确要求手机
  桌面可见时，构造本地最小披露图：中央 App 内容用固定中性色覆盖，只保留手机画布边缘与底部 Android
  系统导航结构。compact observer、Qwen 单步选择器和动作前独立方向审计三者使用同一最小披露视图；
  fingerprint、稳定性、新鲜度和物理执行门仍绑定未经修改的原始 4 帧。
- 该视图只能支持 `home`，不能形成元素候选，也不能用于 tap/input/swipe/back、App 身份、页面完成、
  标题读取或外部影响动作。用户/DeepSeek 仅写同名字段不能激活；必须同时满足本地 task context、唯一
  system-navigation action 和 controller resolved kind。Home 后必须用新的完整原图重新观察桌面。
- 这不是规避或分析被审核内容：中央内容不会发送给模型，模型只看执行系统导航所需的最小设备结构。
  若最小披露视图仍被拒、方向不唯一、返回动作不是 Home 或动作后桌面未被完整观察，保持 0/1 动作
  失败关闭，不再自动重试。回滚只需移除独立视图模块和三个明确接入点。

### 23.3 验证清单与停止条件

- 单测证明中央像素被固定覆盖、底部系统导航结构保留，原始帧不被修改；普通目标仍发送完整图。
- observer 明确 Home 时只调用一次 compact、跳过前台 App 身份与 target refinement；Qwen 只允许绑定
  Home；方向凭据仍绑定原始 frame fingerprint。伪造 marker、外部影响、App 内“主页”和其他动作反测。
- 相关模块与完整回归通过后提交并只重载 Uvicorn；然后从当前新闻页开启全新 Browser 会话。若首次
  最小披露请求仍被云端拒绝，停止真机链并保留为外部服务阻塞，不再扩大恢复通道。

离线结果：最小披露、Home-only 选择、方向凭据与普通路径正反定向 `7/7`，observer/Qwen/adapter/
orchestrator/web 相关回归 `732/732`，Python 完整回归 `1466/1466`。完整回归只有既知测试子进程
`ResourceWarning`，无断言失败。中央 App 内容会被固定遮罩且原帧不变；该视图产出的 App、页面、
元素和弹层声明均被本地清空，只能形成坐标无关 Home 候选。

## 24. 目标 App 启动成功后被功能页面分类误拒绝

### 24.1 验收台账与根因证据

- v61 全新 Browser 会话 `91d2b067fa394d848084a03a9147d6af` 先以 system Home 从新闻页返回
  Launcher，再由新目标重观察取得唯一“浏览器”入口并执行一次 `tap_semantic`。第二动作具有新的
  前后 4 帧、变化后的 observation/fingerprint、`physical_actions=1`、`outcome=matched` 和完整 typed
  receipt；手机实际进入了浏览器新闻流，没有第三次动作或自动重试。
- Qwen 动作逐字绑定桌面元素 `label=浏览器`、`meaning=open_browser`，正式转换要求
  `surface.active_ref equals surface_browser`。动作后 observer 根据当前页面功能把前台 App 分类为
  `news_aggregator/news_feed`。`_validate_newly_completed_named_app_surfaces()` 只比较动作后功能分类与
  goal 中的 `browser`，忽略已由本地控制器证明的启动转换，因此错误拒绝已经成功的导航节点。
- 这是“App 容器身份”和“当前功能页面类别”两个语义维度被错误当作同一字段，不是点击、坐标、
  Browser 特例或模型没有看到页面。任意 App 打开内容流、文档、媒体、会话或设置子页时都可能被按
  功能重新分类，继续添加 App 别名会形成不可收敛的专用补丁。

### 24.2 同类样本、通用修复与边界

- 现场样本为 `browser -> news_aggregator`；变化样本覆盖目标 App 打开后被分类为媒体页、文档页或内容
  流。共同合同是：动作前为 Launcher，唯一可信入口逐字绑定目标 App，Qwen 正式转换也绑定同一目标
  surface，控制器回执严格证明恰好一次 matched 动作和新的非 Launcher 观察。
- 仅在上述全部条件成立时，允许该 typed controller transition 完成它绑定的 navigation-only 启动节点；
  动作后功能分类无需冒充 App 包身份。回执必须严格绑定 session/task/device/revision/subgoal/decision、
  requested/rebound/resolved action digest、before/after observation 和 fingerprint，且对应 controller ref
  必须存在。
- 仍在 Launcher、目标 label/meaning 不绑定 App、正式转换缺失或指向其他 surface、回执 mismatched、
  0/2 次动作、错误 observation/fingerprint/subgoal，均继续拒绝。普通可见文字、Launcher 入口本身、
  read-only/external_state/unknown 节点不能取得该例外。
- 该回执只完成“打开目标 App”的导航转换；后续标题、错误提示、内容、发送或其他结果仍必须由新目标
  下的完整画面重新观察证明。不会改写 observer 的功能分类，不增加 App 名称映射、固定步骤或坐标。

### 24.3 验证清单与停止条件

- 正测两种不同目标 App 的“Launcher 入口 -> 不同功能类页面”：严格回执、目标 surface 和 controller
  ref 全部一致时允许完成导航节点。
- 反测仍在 Launcher、目标名称错误、正式 surface 缺失/错误、receipt 的 session/subgoal/after observation
  或 fingerprint 错误、mismatched/有 errors、controller ref 缺失；全部维持原身份错误并且 0 新动作。
- 复用原有“Launcher 入口不能证明前台 App”和“普通匹配前台 App 可证明”的视觉正反测试；运行
  DeepSeek/编排相关回归与一次 Python 完整回归、静态编译和 diff-check。
- 全绿后本地提交，只重载项目 Uvicorn，不操作卖家 `main.exe`。以全新 Browser 目标进行一次最终真机
  验收；若后续标题读取失败，保存为独立新缺口并停止，不扩大此身份修复。

离线结果：两种功能分类变化及完整绑定反例 `9/9`，DeepSeek 与编排核心 `379/379`，observer、Qwen、
adapter、DeepSeek、编排和 Web 相关回归 `887/887`，Python 完整回归 `1468/1468`。完整回归只有既知
测试子进程 `ResourceWarning`，无断言失败。原视觉身份门保持；只有本地严格回执证明的 Launcher 到
目标 App navigation-only 转换可使用该证据，后续页面内容仍须独立重观察。

## 25. 唯一可见文字已识别但只读取值仍依赖 Qwen 再次选择

### 25.1 验收台账与根因证据

- 提交 `6702cf8` 加载后的 Browser 会话 `f74af265f44542349c2d6724efe829ea` 完成 Home 和打开
  Browser 两次真实动作；`open_browser` 已由严格回执完成并推进到 `read_page_title`，证明第 24 项修复
  在线生效。随后新 4 帧只读观察得到唯一 `role=text`、`meaning=page_title`、`label=要闻`、
  `goal_relevant=true`、`fully_visible=true`、confidence 1.0 的候选。
- Qwen 在 read_only 节点仍返回一个动作；本地 parser 正确以“read_only 禁止物理动作”阻塞，最终
  `physical_actions=2`，没有第三动作。问题不是视觉缺少结果，而是确定性的结构化取值仍交给模型再次
  选择 finished，造成模型随机性阻塞。
- 现有本地可见推进只允许“定位/存在”类目标，并明确排除“读取文字”；因此即使可信观察已经给出唯一
  逐字结果，本地也不会把它作为候选证据交给 DeepSeek 复核。

### 25.2 通用修复、变化样本与边界

- 新增只读文字结果门：仅对 read_only、目标明确要求读取标题/题头/错误提示/状态提示，且没有预设
  exact expected value 时启用；要求唯一 goal-relevant 的 text/dialog/container 候选，meaning 明确为
  title/heading/error/status message，label 非空、完整可见、高置信、无冲突且位于安全画面内。
- 本地只把候选原始 `element_id/role/meaning/label` 写为 visible evidence，再让 DeepSeek 单次复核当前
  read_only 节点；不会自行生成用户未看到的值，也不会执行动作。当前值需要等于/包含某个指定文字、
  多候选、空 label、低置信、边缘裁切、非文字控件、external_state/unknown 或发生型结果继续失败关闭。
- 变化样本覆盖页面主标题、对话框错误提示和状态提示；不包含 Browser/App 名称分支。

## 26. 已验证 App surface 在立即只读重观察时缺少连续性载体

### 26.1 验收台账与根因证据

- 打开 Browser 的 typed receipt 严格证明 `surface_browser`，动作后功能分类为 `news_aggregator`；新目标
  下的 0 动作重观察仍为同一 `news_aggregator`，但 observation/fingerprint 因重新取帧而变化。
- 第 24 项回执只在完成 `open_browser` 当轮生效；若下一 read-only 节点完成，命名 App 身份门仍可能
  再次要求当前功能分类逐字等于 `browser`。用旧 fingerprint 冒充当前画面或永久信任旧 receipt 都不安全。

### 26.2 通用修复、失效边界与验证

- 在 session 内保存本地只读 `VerifiedAppSurfaceLineage`：来源只能是第 24 项严格 Launcher→目标 App
  matched 回执，绑定 session/task/device、目标 surface、source receipt/subgoal、动作后功能类和当时
  物理动作计数。它不改写当前 observer 分类。
- 仅当后继节点依赖该已完成启动节点、没有任何后续物理动作、当前仍为非 Launcher 且功能类与启动后
  一致时，可用于证明“这份当前功能页面仍属于刚打开的目标 App”。任何新物理动作、Home/Launcher、
  功能类变化、错误 task/device/session、缺 controller receipt 或无依赖关系立即失效。
- lineage 只补 App 容器归属；标题/错误文字仍必须由第 25 项当前新观察独立证明，不能从回执推断。
  session snapshot 保留脱敏结构化 lineage 供审计，终态不跨服务恢复。
- 测试覆盖两个 App 的正例，以及新动作后复用、功能类变化、回到 Launcher、无依赖、错 receipt、指定
  exact value、多候选和低置信反例。相关回归与一次完整回归全绿后才重载 Uvicorn；旧 blocked 会话不
  恢复，以全新 Browser 会话验收。

离线结果：唯一标题读取、exact/多候选拒绝及 surface lineage 正反定向 `4/4`，DeepSeek 与编排核心
`381/381`，observer、Qwen、adapter、DeepSeek、编排和 Web 相关回归 `889/889`，Python 完整回归
`1470/1470`。完整回归只有既知测试子进程 `ResourceWarning`，无断言失败。目标切换后的标题读取不再
调用 Qwen 选择动作；lineage 不改变 observer 功能分类，并在任何后续非 wait 物理动作前失效。

第一次在线复验 `e819788b17e44779bfe77af40d4c54df` 再次完成 Home 与打开 Browser 两动作，并在线
生成正确 lineage，但通用 refresh 在本来要求的新目标重观察中先按 fingerprint 变化调用了一次普通
DeepSeek replan；模型提前完成 read 节点时，DeepSeek 内部身份锚点尚看不到 lineage，因而 0 新动作
阻塞。修正后，唯一文字结果门在普通页面变化 replan 之前运行，并把本地 lineage 作为结构化 grounding
fact（不作为标题值）提供；本地另要求完成证据必须逐字包含当前唯一 `label` 结果，不能只引用 lineage。
相关回归仍为 `889/889`、完整回归 `1470/1470`；旧会话不恢复，下一次仍使用全新会话。

## 27. 时间指代结果页面被误当作固定命名页面

### 27.1 验收台账与根因证据

- 提交 `a36087f` 加载后的全新 session `9449b575749a40aeab214d06631c0848` 再次以 Home 和
  Launcher 上唯一 Browser 入口完成两个 matched 动作，累计 `physical_actions=2`，并生成绑定当前
  session/task/device、`surface_browser`、source receipt 和动作计数的 surface lineage。随后新观察取得
  唯一完整高置信 `role=text / meaning=page_title` 候选，逐字 label 为
  `朱镕基同志遗体在京火化 习近平等送别`；没有第三个物理动作。
- DeepSeek 用当前 observation 的三个 typed `visual_claim` 完成 `read_page_title`，但本地仍报
  `命名页面完成声明缺少结构化画面身份锚点：subgoals.read_page_title`。失败原文为
  `读取浏览器打开后页面的主标题或错误提示`。身份解析器只跳过以“打开后页面”开头的短语，无法识别
  动词前带 App/对象或动词与“后”之间带对象的同一时间指代，因此错误提取“浏览器”为固定页面名。
- 主要根因是通用自然语言时间指代边界，不是 App 身份、视觉结果、坐标、机械臂或模型证据缺失。稳定
  命名页面（例如“浏览器设置页面”“微信设置页面”）仍应要求结构化身份；“浏览器打开后页面”只描述
  已验证导航转换之后的当前结果页面，其 App 归属继续由 surface lineage 独立证明。

### 27.2 同类样本、通用修复与边界

- 同类正样本包括“读取浏览器打开后页面…”、“进入目标 App 后的页面…”、“跳转后界面…”以及原有
  “打开后页面…”；共同结构是在同一句内由打开、进入、加载、刷新、跳转、切换、返回等转换动词和
  “后/后的页面、界面、屏幕、视图”建立时间指代。
- 身份解析只在句内识别这类转换结果容器，不再要求转换动词位于字符串开头，也不假设动词与“后”
  相邻。逗号、句号、分号和换行截断匹配，防止跨句吞掉后续真实页面名。
- 稳定命名页面、逐字引号标题和系统 Launcher 继续使用原严格身份门；测试将原先错误把
  “浏览器打开后页面”当稳定页面的反例改为真正稳定的“浏览器设置页面”。surface lineage 与当前唯一
  typed visual claim 仍分别证明 App 容器归属和标题值，任一缺失都不能由本规则补造。

### 27.3 验证与停止条件

- 精确现场措辞和三种跨 App/换措辞时间指代必须不产生命名页面锚点；原“打开后页面”回归继续通过。
- “浏览器设置页面”“微信设置页面”“跨境订单结果页面”仍产生非空身份锚点；不匹配的功能页面事实
  不能完成这些稳定页面声明，逐字引号标题仍要求完整结构化匹配。
- DeepSeek 精确标题证据重放、surface lineage 编排测试和相关模块回归通过后，再运行一次完整 Python
  回归、静态编译和 diff-check。全绿后本地提交并只重载项目 Uvicorn；旧失败 session 不恢复。下一次
  真机只允许一个全新 Browser 会话，若再失败则按新证据停止，不继续扩展解析器。

离线结果：DeepSeek 与编排根因回归 `381/381`，observer、Qwen、adapter、DeepSeek、编排和 Web
关联回归 `889/889`，Python 完整回归 `1470/1470`。完整回归只有既知测试子进程
`ResourceWarning`，无断言失败；稳定命名页面与逐字标题反例继续失败关闭。

## 28. 已验证导航完成仍被模型自由证据抢先裁决

### 28.1 验收台账与根因证据

- 提交 `241be36` 加载后的标题任务 session `ee4f591867804f61b6199a144f347895` 再次完成 Home 与
  Browser 启动两个 matched 动作；第 27 项旧的时间指代身份错误没有复现。当前新闻流没有独立页面主
  标题，observer 正确只报告选中“要闻”tab 和多条 `news_headline`，因此该验收目标本身无法取得标题
  值；会话在 0 新动作处 blocked，没有把新闻标题伪装成页面标题。
- 随后改用不依赖内容标题的三步导航 session `07245a3dba6042ce88db5b6c9bb02add`。Home 与从
  Launcher 打开 Browser 两次动作均 matched，第二步 `verification_step_2` 为
  `physical_actions=1`、无 verification error、fingerprint
  `f2e09e91ad24468737ff -> e6b3e42828bd84e9d192`；会话总动作数为 2，没有第三动作或自动重试。
- DeepSeek 已把 `open_browser` 标为 completed，但没有使用本地 typed controller receipt；现有
  `_apply_verified_navigation_completion()` 仅在模型仍把节点留为 active 时才改用回执，模型已经标
  completed 时直接保留其自由证据。随后命名 App 身份门先于编排器 surface 验证拒绝，导致严格回执虽
  已存在却没有机会成为完成 authority。

### 28.2 通用修复、变化样本与边界

- 在 `action_result_matched` 且 transition、session/task/device/revision/subgoal、after observation、
  receipt 和 controller evidence ref 全部严格绑定时，本地控制器本来就是刚发生导航转换的唯一权威。
  无论 DeepSeek 是遗漏完成，还是已完成却引用当前视觉自由文本，本地都把该旧 navigation-only 节点的
  completion evidence 规范化为精确 typed receipt。
- 只改写同一旧活动节点的 controller-owned `status/completion_evidence`；objective、depends_on、
  constraints、completion conditions、risk IDs、impact 任一被模型改变仍直接拒绝。mismatched、纯观察、
  read_only/external_state、错 receipt/scene/revision、回执重放继续不能使用该规范化。
- 变化样本覆盖任意 App 启动后模型先写视觉证据、以及模型完全遗漏导航完成两种形状；稳定命名页面的
  普通视觉完成仍走原身份门，不会因为没有 matched receipt 而放宽。

### 28.3 验证与停止条件

- 正测模型遗漏完成与“已完成但引用非权威证据”均只调用一次 provider，并得到同一 controller ref；
  后继唯一安全 frontier 正常激活。
- 反测子目标语义改写、无 transition、错绑定、纯视觉稳定页面和 receipt 重放必须继续失败关闭。
- 运行 DeepSeek/编排根因测试、相关回归和一次完整 Python 回归，静态编译及 diff-check 全绿后本地
  提交并只重载项目 Uvicorn。旧 session 不恢复；下一次只建立一个全新 Browser 三步导航会话。

离线结果：两种 controller receipt 规范化正例及语义改写/纯视觉反例通过；DeepSeek 与编排根因回归
`381/381`，observer、Qwen、adapter、DeepSeek、编排和 Web 关联回归 `889/889`，Python 完整回归
`1470/1470`。完整回归只有既知测试子进程 `ResourceWarning`，无断言失败。

## 29. canonical 输入正文存在但常见完成状态句式未绑定

### 29.1 验收台账与根因证据

- Browser 正式 session `61f26496f1a046118468e922eacb5741` 已在同一会话完成 Home、从 Launcher
  打开 Browser、再次 Home 三个 matched 动作，revision `1→6`，最终 `succeeded`、
  `launcher/home_screen`；每步前后各 4 帧，无自动重试。它证明第 28 项 controller receipt 修复在线
  生效。
- 第二个 App 目标“打开设置，在搜索输入框输入 wifi，但不提交/选择结果，最后回桌面”在目录
  `generic_supervised_20260819_054627_5e5a5e67` 的 initial graph 阶段 0 动作失败。DeepSeek 已正确输出
  `goal.entities.input_text=wifi`、`input_wifi` 子目标和完成条件 `输入框中显示 'wifi'`，但本地输入状态
  识别只接受“输入框内容显示/内容为”等带中间名词句式，未接受同义且更常见的“输入框显示 literal”。
- 主要根因是 canonical literal 与载体状态的中文句式覆盖缺口，不是正文缺失、App、输入硬件、Qwen、
  坐标或风险分类；本地拒绝没有创建 session、调用视觉模型或产生物理动作。

### 29.2 通用修复、变化样本与边界

- 输入状态载体允许在同一标点子句内直接使用“输入框/文本框/搜索框/文本区域/输入区域/编辑区域 +
  为、是、变为、改为、修改为、替换为、显示 + canonical literal”。精确正文仍只来自
  `goal.entities.input_text`，解析器不从 prose 提取或改写文字。
- `_state_description_binds_canonical_input_text()` 继续要求逐字 literal 位于同一标点子句，并用字符边界
  拒绝 `wifi2`、`wifi.com`、相似词、跨句拼接和缺失正文。没有 canonical input_text 或载体状态时仍
  失败关闭。
- 变化样本覆盖搜索输入框、普通文本框、输入区域和编辑区域四种载体；不增加 Settings/App 名称、固定
  步骤、键位或坐标。执行阶段仍须由可信 input 元素、fresh 观察、精确前缀事务和动作后 value 核验授权。

### 29.3 验证与停止条件

- 精确现场完成条件和三种同义状态句式应通过；相似正文、跨句 literal、缺 canonical 值继续拒绝。
- 运行 DeepSeek/编排、输入事务、Qwen/observer/adapter/Web 相关回归及一次完整 Python 回归，静态编译
  与 diff-check 全绿后本地提交并只重载项目 Uvicorn。旧 0 动作失败不恢复；只用完全相同原始目标建立
  一个全新 Settings session，任何新失败立即停止真机链。

离线结果：载体直述状态与精确 literal 正反定向通过；DeepSeek 与编排核心 `382/382`，observer、Qwen、
adapter、DeepSeek、编排和 Web 关联回归 `890/890`，Python 完整回归 `1471/1471`。完整回归只有既知
测试子进程 `ResourceWarning`，无断言失败。

## 30. 根层执行实体泄漏到不相关前置子目标

### 30.1 验收台账与根因证据

- 提交 `e7eace1` 加载后，完全相同的 Settings 原目标已通过 initial graph；证明第 29 项 canonical
  literal 句式修复在线生效。新 session `3477939acd664dea95fb9a194c188ed2` 在 Launcher 的第一个
  `open_settings` 节点 0 动作 blocked，原因是“当前可信候选中不存在逐字一致文字：搜索输入框”。
- 同一 trusted scene 已正确识别唯一 `label=设置 / meaning=open_settings / role=button /`
  `goal_relevant=true` 候选；错误来自 Qwen context 仍把根层
  `goal.entities.target_ui_label=搜索输入框` 和 `input_text=wifi` 原样提供给每个子目标。打开 App 的前置
  节点因此被后续输入节点的精确 label 门错误覆盖。
- 主要根因是 canonical 执行实体缺少 current-subgoal 投影，不是视觉、App 图标、坐标、风险或模型
  没有找到设置。该 session 没有产生物理动作，设备保持 Launcher。

### 30.2 通用修复、变化样本与边界

- 完整 task graph 继续永久保存所有 canonical entities；只在生成当前 Qwen/本地策略上下文时，对可能
  授权执行的实体做 subgoal scope 投影。recipient(s)、input_text/fields、target_ui_label、spatial_hint、
  金额/币种/账户/文件/商品/日期/时间/target/value 等，只有其逐字值出现在当前 objective、constraints
  或 completion conditions 时才进入本轮 action context。
- `target_surface` 和目标 App 结构不属于后续控件 literal，继续保留；当前子目标、全局约束和原 task
  graph 均不被改写。Observer 已通过 active subgoal 上下文正确标记 Settings 入口，本批只防止后续实体
  在 Qwen 选择和 controller policy 中越级形成 exact-label 门。
- 变化样本覆盖同一四节点任务：打开 App 时 label/text 均不可见于 authority；定位输入框时只出现 label；
  输入节点同时出现 label/text；Home 节点再次全部移除。实体未逐字绑定当前节点时宁可失败关闭，不从
  根目标推断其可用于本轮动作。

### 30.3 验证与停止条件

- 正测上述四个 current-subgoal context；反测 `target_surface` 保留、确认 scope 和 external effect 的
  当前实体仍可用、完整 graph snapshot 未被修改。
- 运行 DeepSeek/Qwen/编排/policy/Web 相关回归和一次完整 Python 回归，静态编译及 diff-check 全绿后
  本地提交并只重载项目 Uvicorn。旧 blocked session 不恢复；完全相同 Settings 原目标只允许一个新会话，
  新失败后停止真机链，不继续加现场特例。

离线结果：四阶段实体投影定向通过；DeepSeek 与编排核心 `383/383`，observer、Qwen、adapter、
DeepSeek、编排和 Web 关联回归 `891/891`，Python 完整回归 `1472/1472`。完整回归只有既知测试
子进程 `ResourceWarning`，无断言失败。

## 31. 控件角色描述被误当作屏幕逐字标签

### 31.1 验收台账与根因证据

- 提交 `63950bf` 加载后的 Settings session `b0f4fd551b0d4f19b3b0061e2ce0e907` 已从 Launcher
  正确点击唯一“设置”入口；动作 matched，前后各 4 帧，revision `1→2`，证明第 30 项实体跨子目标
  泄漏修复在线生效。
- 当前设置页 scene 为 `com.android.settings/settings_main`，唯一可信输入候选是
  `role=input / label=搜索系统设置项 / placeholder=搜索系统设置项 / value="" /`
  `goal_relevant=true`。DeepSeek 把用户描述的控件类型“搜索输入框”写入 `target_ui_label`，Qwen context
  因而要求屏幕 label 逐字等于这五个字并在 0 新动作处 blocked；会话总 `physical_actions=1`，没有点击
  错误控件或重试。
- 主要根因是“控件角色描述”和“可见字面标签”没有区分，不是视觉漏掉输入框或几何不准。用户没有
  引号、‘名为/标有/文字为’等逐字指示；真实可见 placeholder 与语义描述不同是任意 App 的常见情况。

### 31.2 通用修复、变化样本与边界

- Qwen current-subgoal 投影把以输入框、文本框、搜索框、文本区域、输入区域、编辑区域、按钮、入口、
  选项、控件、元素、列表项、标签页或页签结尾的普通描述视为 role/semantic hint，不作为 exact label。
  目标仍须通过可信 role、meaning、goal relevance、唯一性和 fresh 几何门。
- 用户用中文/英文引号逐字引用，或明确写“名为/名称为/标有/标签为/文字为/显示文字为”时，原
  `target_ui_label` exact authority 保留。普通非角色字面标签（例如“设置”“下一步”）也不受影响。
- 只改变当前 Qwen/controller context，不删除 graph 中的原 entity、不改写 input_text、目标 App 或
  completion。没有 App 名称、placeholder 别名、固定步骤或坐标分支。

### 31.3 验证与停止条件

- 正测未加逐字标记的“搜索输入框”只保留 `input_text=wifi`，exact label 被移除；加引号/‘名为’时
  exact label 仍逐字保留。第 30 项四阶段 scope、literal 标签、输入事务和 policy 反例继续通过。
- 运行 DeepSeek/Qwen/observer/编排/policy/Web 相关回归和一次完整 Python 回归，静态编译及 diff-check
  全绿后本地提交并只重载 Uvicorn。旧会话不恢复；完全相同 Settings 原目标再用一个全新 session 验收。

离线结果：role description 与 explicit literal 正反定向通过；DeepSeek 与编排核心 `384/384`，
observer、Qwen、adapter、DeepSeek、编排和 Web 关联回归 `892/892`，Python 完整回归 `1473/1473`。
完整回归只有既知测试子进程 `ResourceWarning`，无断言失败。

## 32. 首次观察已满足导航目的地却仍要求执行旧动作

### 32.1 验收台账与根因证据

- 提交 `ba2437c` 加载后从设置首页启动完全相同目标，session
  `8d3e9d230ec24b01b15cf87339d78473` 为 0 物理动作 blocked。Qwen 明确报告“画面已显示设置主界面，
  open_settings 子目标已完成”，但当前节点仍停在 `open_settings`，合法候选只剩会离开设置的 Home，
  因此没有执行错误动作。
- 初始 start 路径已经调用 `_advance_visible_presence_prefix()`，但其词法门把 objective 和 completion
  conditions 拼在一起；objective 中的“打开”被统一视为必须发生的 transition，即使正式后置状态
  `设置主界面可见` 已被当前结构化 scene 证明，也禁止 0 动作完成。结果是系统无法适应“任务开始时
  用户已在目标 App/目标页”的不同起始状态。
- 主要根因是幂等导航目的地状态与必须发生的操作事件混为一类，不是 Qwen 没识别当前页；本次没有
  创建 confirmation scope、机械臂动作或旧 scope 复用。

### 32.2 通用修复、变化样本与边界

- presence 门优先审查正式 completion conditions。若它们只声明一个当前可见目的地，且后续已有的
  named-surface identity、foreground App、唯一候选、完整可见和冲突门全部通过，则“打开 App、进入
  页面、返回桌面”等幂等导航节点可用 0 动作完成并激活后继。
- 刷新/重新加载/重新获取/更新/同步等必须发生的事件，以及不可见、不存在、缺失、消失、移除等负
  状态仍不能由普通 presence 证明；现有 `reload` 和 dismiss/keyboard absence 反例保持 false。
- 不根据 App 名称、固定页面或截图判断；当前 completion 不具备 named identity 或 scene 不匹配目标
  surface 时仍不推进。DeepSeek 仍需用当前 typed visual evidence 生成新 revision。

### 32.3 验证与停止条件

- 正测已在 App 主界面、已在目标详情页、已在 Launcher 三类幂等目的地；反测 reload、键盘不可见、
  弹层消失、错误前台 App 和不匹配页面身份。
- 运行编排、DeepSeek/Qwen/observer/adapter/Web 相关回归和一次完整 Python 回归，静态编译及
  diff-check 全绿后本地提交并只重载 Uvicorn。旧会话不恢复；完全相同 Settings 原目标只建立一个
  新 session，从当前设置首页验证动态跳过与后续输入。

离线结果：幂等目的地与 occurrence/absence 正反定向通过；DeepSeek 与编排核心 `385/385`，
observer、Qwen、adapter、DeepSeek、编排和 Web 关联回归 `893/893`，Python 完整回归 `1474/1474`。
完整回归只有既知测试子进程 `ResourceWarning`，无断言失败。

## 33. 否定约束的共享词误杀合法目标

### 33.1 验收台账与根因证据

- 提交 `5ec7807` 加载后的 Settings session `afb3d1820d8e4423b8e09f8701ad3b9f` 已在 0 动作下
  正确跳过当前可见的 `open_settings` 和 `locate_search_box`，revision `1→3`，证明第 32 项动态起始
  状态修复在线生效；当前活动节点为 `input_wifi`。
- 当前可信画面有唯一 `role=input / meaning=application_text_input / label=搜索系统设置项 /`
  `goal_relevant=true / fully_visible=true / value="" / soft_keyboard_visible=false` 候选，设备动作能力也包含
  `tap_semantic` 和 `input_verified_text`，但 Qwen 收到的正式 choices 只剩 `home`，因此 0 动作 blocked。
- 离线最小复现确认 `constraint_excludes_candidate()` 会把“不得选择任何搜索结果”与“搜索系统设置项”
  仅凭共享二字词“搜索”判为同一禁止目标。正确输入框在 choices 构建前即被删除；主要根因是禁止动作的
  **宾语范围**没有被结构化绑定，不是 Qwen、输入事务、焦点前置、几何、App 或机械臂故障。

### 33.2 同类样本、通用修复与边界

- 已知现场样本：“不得选择搜索结果”不得排除搜索输入框，但仍必须排除真正的搜索结果；变化样本覆盖
  “不要点击广告”与非广告控件、“不要使用设置入口”与设置入口、英文 `do not open search results`
  与 search input，以及带“但/但是/but/however”的后续允许子句。
- 把元素级否定约束按标点和转折词切成独立子句，只从带禁止词且带点击/打开/选择/使用等目标动作的
  子句中抽取动作后的禁止宾语；候选必须与该宾语的规范化可见/语义短语存在完整包含或精确匹配才排除。
  不再用整句与候选任意一个中文二字片段相交就排除。
- 页面全部控件禁用、`除…之外` blanket exception、状态/外部效果约束继续走原独立门；真正的“搜索结果”、
  “发送按钮”“设置入口”等仍失败关闭。实现不包含 Settings、搜索框 label、固定步骤或坐标分支。

### 33.3 验证与停止条件

- 先为现场与变化样本补正反合同测试，再证明同一正式 input scene 产生唯一聚焦
  `tap_semantic` choice，而未聚焦时仍不产生 `input_verified_text`。
- 运行 constraint、Qwen、visual authority、observer、adapter、编排与 Web 相关回归及一次完整 Python
  回归，静态编译及 diff-check 全绿后本地提交并只重载项目 Uvicorn。旧 blocked session 不恢复；完全
  相同 Settings 原目标只建立一个新 session，任一真实动作失败立即停止，不自动重试。

离线结果：否定宾语范围现场与中英文变化样本 `16/16`，正式未聚焦 input choice 重放 `1/1`，
constraint、Qwen 与 visual authority `145/145`，observer、Qwen、adapter、DeepSeek、编排和 Web 关联
回归 `894/894`，Python 完整回归 `1479/1479`。完整回归只有既知测试子进程 `ResourceWarning`，
无断言失败。
