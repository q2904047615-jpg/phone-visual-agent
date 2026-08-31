# Qwen 单步视觉决策离线评估

本目录只使用既有截图验证“只读观察 → 可信候选 → Qwen 单步选择 → 本地协议校验”。不连接摄像头、网页服务、`main.exe` 或机械臂。

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

离线集包含真实不同帧的桌面序列，以及脱敏设置列表、输入弹层和已完成输入值画面。
已退役的候选词点击、普通提交确认和“模糊即模型停止”用例不再作为成功合同。

## 运行

自动协议测试不需要在线模型：

```powershell
python -m unittest discover -s poc -p test_qwen_visual_decision.py -v
```

在已配置 `DASHSCOPE_API_KEY` 时，可运行真实 Qwen 的离线截图评估：

```powershell
python poc\eval_qwen_visual_decision.py
```

默认单用例超时150秒、整套超时720秒。每个用例在隔离子进程中运行；超时会终止该用例并继续后续用例。运行开始时即创建包含全部待执行用例的 `report.json`，之后逐用例原子更新，因此整套内部超时后仍有完整汇总。

```powershell
python poc\eval_qwen_visual_decision.py `
  --case-timeout-seconds 150 `
  --suite-timeout-seconds 720
```

只重跑某些用例：

```powershell
python poc\eval_qwen_visual_decision.py --case launcher_text_icon_single_action
```

从已有报告恢复并只重跑未通过用例：

```powershell
python poc\eval_qwen_visual_decision.py `
  --resume-report poc\output\offline_qwen_visual_decision\旧运行目录\report.json
```

恢复报告会把旧的成功结果标记为 `prior_run_reference`，注明来源报告和旧 `run_id`，并从本轮格式指标中排除，不能冒充本轮调用结果。

报告记录单次同响应模型调用次数、耗时、首次格式通过率、动作/完成/技术失败比例和具体失败原因。
`hardware_actions_enabled`、`camera_enabled`、`web_console_enabled`、`robot_enabled` 均固定为 `false`。
