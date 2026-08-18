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
