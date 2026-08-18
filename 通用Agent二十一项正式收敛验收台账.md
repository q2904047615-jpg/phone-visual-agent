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
