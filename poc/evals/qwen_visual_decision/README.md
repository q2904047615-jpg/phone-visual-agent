# Qwen 单步视觉决策离线评估

本目录只使用既有截图验证“只读观察 → 可信候选 → Qwen 单步选择 → 本地协议校验”。不连接摄像头、网页服务、`main.exe` 或机械臂。

## 输入边界

Qwen 决策入口接收 `TaskGraph.to_qwen_context()` 的完整上下文：

- `protocol_version`、`task_id`、`device_id`、`revision`；
- `current_subgoal`、`global_constraints`、`current_execution_class`；
- 与当前子目标绑定的 `effect_intents` 和本地 `effect_gate`；
- 当前稳定帧序列与本地只读观察阶段生成的 `TrustedObservation`。

正式共享协议只有 `2026-08-20-deepseek-typed-task-graph-v4`。Qwen 必须逐项复用
本地生成的 effect policy；它不能生成确认布尔值、旧风险字段或机械权限。v2/v3、
缺失 effect 绑定、额外退役字段和伪装成 v4 的旧结构都会在视觉模型调用前拒绝，
不存在迁移 fallback。当前前端/离线交叉夹具为
`frontend_contract_fixtures/deepseek_typed_task_graph_v4.json`。

`TrustedObservation` 保存本轮 `observation_id`、本地画面 `fingerprint`、多帧稳定性、所选帧和可信候选元素。模型不能在决策输出中创建元素，也不能修改已有候选的 `element_id`、文字、语义、角色、状态或原始 `bounds`。

## 输出边界

- 必须逐字回传 `task_id`、`device_id`、`revision`、`observation_id` 和 `fingerprint`；
- `status` 只能为 `action`、`finished` 或 `blocked`；
- `action` 只能携带一个通用动作，不能是列表，也不能包含后续计划；
- 点击或关闭动作必须绑定一个已有可信候选；滑动、返回和等待只允许整屏/系统语义区域；
- `page_state` 仅是语义描述，禁止携带候选元素，不能作为执行证据；
- `finished` 必须引用可信候选 ID 或本轮可信 scene；
- scope 缺失、冲突或过期，以及未满足本地 `effect_gate` 的效果子目标，
  都在调用观察或决策模型前直接 `blocked`。
- 决策格式错误最多修复重试一次；第一次非法输出不会形成候选动作，第二次仍非法则安全返回 `blocked`。
- 观察格式错误同样最多修复一次；网络断连和服务超时不会伪装成格式修复。

可信观察会归并强重叠且文字/语义一致的重复候选，保留一个原始 `element_id` 和原始 `bounds`，不平均或生成坐标。重叠但语义冲突的候选保持分离并记录冲突。只读精确文字验证可以用 `goal.entities.expected_role/expected_meaning` 限定完成证据；点击文字仍必须是唯一可操作候选。

本地控制器还会复核动作白名单、置信度、候选绑定、风险门，以及 task/revision/observation/fingerprint 新鲜度。任一项不匹配都不能产生可执行动作。

## 截图覆盖

离线集包含真实不同帧的桌面序列，以及脱敏设置列表、输入弹层、输入框、文字提交控件、拼音候选栏和模糊画面。测试按控件语义与页面类型工作，不按 App 名称编排流程。

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

报告记录观察/决策各自的模型调用次数、逐次耗时、首次格式通过率、修复重试率、错误类型、最终 blocked 比例和安全停止原因。`hardware_actions_enabled`、`camera_enabled`、`web_console_enabled`、`robot_enabled` 均固定为 `false`。
