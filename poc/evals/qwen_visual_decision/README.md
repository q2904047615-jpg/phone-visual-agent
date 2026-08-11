# Qwen 单步视觉决策离线评估

本目录只使用既有截图验证“只读观察 → 可信候选 → Qwen 单步选择 → 本地协议校验”。不连接摄像头、网页服务、`main.exe` 或机械臂。

## 输入边界

Qwen 决策入口接收 `TaskGraph.to_qwen_context()` 的完整上下文：

- `protocol_version`、`task_id`、`device_id`、`revision`；
- `current_subgoal`、`global_constraints`、`current_external_impact`；
- `risk_actions`、`confirmation_gate`；
- 当前稳定帧序列与本地只读观察阶段生成的 `TrustedObservation`。

`TrustedObservation` 保存本轮 `observation_id`、本地画面 `fingerprint`、多帧稳定性、所选帧和可信候选元素。模型不能在决策输出中创建元素，也不能修改已有候选的 `element_id`、文字、语义、角色、状态或原始 `bounds`。

## 输出边界

- 必须逐字回传 `task_id`、`device_id`、`revision`、`observation_id` 和 `fingerprint`；
- `status` 只能为 `action`、`finished` 或 `blocked`；
- `action` 只能携带一个通用动作，不能是列表，也不能包含后续计划；
- 点击或关闭动作必须绑定一个已有可信候选；滑动、返回和等待只允许整屏/系统语义区域；
- `page_state` 仅是语义描述，禁止携带候选元素，不能作为执行证据；
- `finished` 必须引用可信候选 ID 或本轮可信 scene；
- 未满足 `confirmation_gate` 的外部状态动作在调用决策模型前直接 `blocked`。
- 决策格式错误最多修复重试一次；第一次非法输出不会形成候选动作，第二次仍非法则安全返回 `blocked`。

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

报告写入忽略目录 `poc/output/offline_qwen_visual_decision/`，记录首次通过率、修复重试率、最终 blocked 比例。`hardware_actions_enabled` 固定为 `false`。
