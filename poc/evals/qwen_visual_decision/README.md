# Qwen 视觉能力离线验收集

本目录用已保存的真实截图验证“页面理解 → 目标定位 → 输入状态 → 完成判断 → 唯一下一动作”。它借鉴 AndroidWorld 的任务集和标准答案思路，但不启动 Android 模拟器：不连接摄像头、网页服务、`main.exe`、机械臂或 Companion IME。

## 输入边界

Qwen 的同一次响应接收当前高层目标、当前设备本轮实际可用动作和当前截图：

- `protocol_version`、`task_id`、`device_id`、`revision`；
- `current_subgoal`、`global_constraints`、`current_execution_class`；
- 与当前子目标绑定的正向完成条件；
- 当前稳定帧序列。响应同时发布 scene 和本轮唯一 decision。

正式共享协议只有 `2026-08-20-deepseek-typed-task-graph-v4`。Qwen 必须逐项复用
本地生成的计划定义；它不能生成确认布尔值、旧风险字段或机械权限。v2/v3、
额外退役字段和伪装成 v4 的旧结构都会被拒绝，不存在迁移 fallback。当前前端/离线交叉夹具为
`frontend_contract_fixtures/deepseek_typed_task_graph_v4.json`。

同一响应解析后才建立 `TrustedObservation`，保存本轮 `observation_id`、本地画面
`fingerprint`、多帧稳定性、所选帧和可信候选元素。decision只能引用同一响应中的元素，
不能另外创建或修改候选。

## 输出边界

- `status` 只能为 `action` 或 `finish`；
- `action` 只能携带一个通用动作，不能是列表，也不能包含后续计划；
- 点击或关闭动作必须绑定一个已有可信候选；滑动、返回和等待只允许整屏/系统语义区域；
- `finish` 必须引用同一响应中的可信 scene 或元素证据；
- 当前截图没有最终控件、需要中间导航或单步不能直接完成高层目标时，仍必须选择一个推进动作；
- scope 缺失、冲突、过期、格式错误、模型断连和服务超时统一记为具体技术失败，不能伪装成合法模型结论；
- 每张截图只调用一次模型，不做远程修复重试，也不由本地替模型改选动作。

可信观察会归并强重叠且文字/语义一致的重复候选，保留一个原始 `element_id` 和原始 `bounds`，不平均或生成坐标。重叠但语义冲突的候选保持分离并记录冲突。只读精确文字验证可以用 `goal.entities.expected_role/expected_meaning` 限定完成证据；点击文字仍必须是唯一可操作候选。

本地控制器只复核动作集合、唯一候选绑定、设备能力以及
task/revision/observation/fingerprint 新鲜度。任一项不匹配都不能产生可执行动作。

## 截图覆盖

当前 `2026-09-01-qwen-visual-capability-benchmark-v5` 包含 24 个“截图＋目标”用例，来自 6 组已保存画面：

- 稳定桌面与带反光变化的桌面；
- 脱敏系统设置列表；
- 包含 `.com` 的符号键盘输入弹层；
- 包含 `ihao` 预编辑的中文拼音键盘；
- 包含“三角洲行动”候选的裁切拼音画面。

同一截图可以绑定不同目标，这是有意设计：例如一张桌面图分别检查“识别桌面”“定位设置”“定位微信”，以区分页面识别能力和目标选择能力。已退役的候选词点击、普通提交确认和“模糊即模型停止”规则不作为成功合同；候选画面只用于只读识别评分。

每个用例都有可定位的标准答案 `expectations`。报告中的 `capability_metrics` 分别统计：

- `page`：页面/前台 App 与可见语义；
- `target`：模型动作绑定的同帧元素及其标签、角色、含义；
- `input`：输入框、正文、焦点、键盘布局、输入模式与预编辑；
- `finish`：是否在目标已满足时返回 `finish` 并引用同帧证据；
- `decision`：`action/finish` 与动作种类是否符合用例合同。

## 运行

清单、标准答案、评分器和报告测试不调用在线模型：

```powershell
cd poc
.\.venv\Scripts\python.exe -m unittest test_qwen_offline_eval
```

在已配置 `DASHSCOPE_API_KEY` 时，可运行真实 Qwen 的离线截图评估：

```powershell
poc\.venv\Scripts\python.exe poc\eval_qwen_visual_decision.py
```

完整 24 用例运行会调用 Qwen 24 次，每个“截图＋目标”恰好一次；不会调用第二个视觉判断，也不会执行任何手机动作。建议先用一个用例做低成本冒烟：

```powershell
poc\.venv\Scripts\python.exe poc\eval_qwen_visual_decision.py --case launcher_settings_action
```

默认单用例超时150秒、整套超时720秒。每个用例在隔离子进程中运行；超时会终止该用例并继续后续用例。运行开始时即创建包含全部待执行用例的 `report.json`，之后逐用例原子更新，因此整套内部超时后仍有完整汇总。

```powershell
poc\.venv\Scripts\python.exe poc\eval_qwen_visual_decision.py `
  --case-timeout-seconds 150 `
  --suite-timeout-seconds 720
```

只重跑某些用例：

```powershell
poc\.venv\Scripts\python.exe poc\eval_qwen_visual_decision.py --case launcher_settings_action
```

从已有报告恢复并只重跑未通过用例：

```powershell
poc\.venv\Scripts\python.exe poc\eval_qwen_visual_decision.py `
  --resume-report poc\output\offline_qwen_visual_decision\旧运行目录\report.json
```

恢复报告会把旧的成功结果标记为 `prior_run_reference`，注明来源报告和旧 `run_id`，并从本轮格式指标中排除，不能冒充本轮调用结果。

报告记录单次同响应模型调用次数、耗时、首次格式通过率、五项能力准确率、动作/完成/技术失败比例和具体失败原因。只有真实在线运行生成的报告才代表 Qwen 实际准确率；自动测试通过只证明验收工具本身可运行。
`hardware_actions_enabled`、`camera_enabled`、`web_console_enabled`、`robot_enabled` 均固定为 `false`。
