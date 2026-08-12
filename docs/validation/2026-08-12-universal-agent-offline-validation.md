# 通用手机视觉 Agent 第一阶段离线验证报告

日期：2026-08-12
分支：`codex/universal-agent-safe-live-loop-design`
范围：纯离线代码、模拟机械臂、网页协议与浏览器契约；未执行真实机械臂动作。

## 结论

第一阶段通用安全闭环已在离线和模拟环境中通过：DeepSeek 生成动态任务图，Qwen 只根据当前子目标和四帧可信观察返回一个视觉动作，本地策略只允许低风险通用导航，用户确认精确绑定 `session/task/device/revision/subgoal/risk/observation/fingerprint`，控制器最多执行一次动作，随后使用四帧新画面验证并要求 DeepSeek 增加 revision。实现没有在新编排器中增加 App 名称分支或固定业务步骤。

这不代表真实设备已经完成验收。摄像头画质、机械臂落点、真实页面变化、两种真实 App/措辞以及多手机运行仍需后续只读预检和用户逐步确认。

## 已实现代码能力

- `deepseek-task-graph-v3` 作为高层任务和风险权威；DeepSeek 不产生坐标或低层动作。
- Qwen 决策必须绑定当前任务、设备、revision、观察 ID 和 fingerprint，且只输出一个候选动作。
- 第一阶段本地策略只允许通用导航类 `tap_semantic`、`dismiss_overlay`、`swipe`、`back` 和 `wait_for_change`。
- 外部状态、未知影响、输入、键盘、开关、账号效果、低置信度、候选冲突和过期画面均失败关闭。
- 确认票据一次性消费；执行前重新校验；一次请求最多一个物理动作；失败不自动重试物理动作。
- 动作后保存精确的四帧运行时画面，建立新可信观察、验证变化并触发 DeepSeek 新 revision。
- 单设备活动会话通过共享文件租约跨进程互斥；所有物理动作区段另有跨进程动作租约；暂停和取消使旧确认失效并释放设备。
- 确认后重新观察只能重绑定相同 meaning、label、role、states 且区域高度重合的目标；同名控件语义或位置变化时零动作失败关闭。
- 每次动作证据使用唯一前缀，不会被同一会话后续步骤覆盖；设备断线会立即废弃当前确认并要求重新观察。
- `/api/agent/generic-supervised/*` 已切换到新编排器；`/auto` 保留兼容路由但固定返回 `phase_one_manual_confirmation_required` 和 `physical_actions=0`。
- 网页展示 revision、影响等级、Qwen 置信度、本地策略结果、累计动作数和证据路径；前端不会请求 `/auto`。
- 权威 JSON 证据使用同目录临时文件和原子替换写入，写入失败时阻止继续执行。

## 纯离线测试

2026-08-12 使用项目虚拟环境运行：

```powershell
cd poc
.\.venv\Scripts\python.exe -m unittest discover -s . -p "test_*.py" -v
```

最终结果：`541 tests`，全部通过，耗时 `29.001s`。

前端协议和浏览器契约：

```powershell
node test_frontend_protocol.js
$env:NODE_PATH='C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\node_modules'
node test_frontend_browser_contract.js
```

结果：协议 `12/12`，浏览器 `3/3`，全部通过。浏览器测试验证页面不显示自动执行按钮、不发起 `/auto` 请求，并把 observation ID 与 fingerprint 放进确认请求。

Python 全量回归中出现一条历史 `ResourceWarning`，提示一个测试生成的子进程在对象回收时仍存在；测试本身和整套回归均通过。本次没有把该历史警告误报为新通用闭环已失败。

## 模拟机械臂闭环

`test_universal_agent_mock_loop.py` 使用 Pillow 生成自有 540×960 合成 UI，不使用任何真实 App 截图或专用脚本。结果 `4/4`：

- `synthetic.catalog`，自然语言“把眼前这个条目的内容页打开给我看”：执行一次通用语义点击，动作后四帧稳定，DeepSeek 从 revision 1 增至 2。
- `synthetic.reader`，自然语言“我不想停在这里，退回刚才那一层”：执行一次系统返回，动作后四帧稳定，DeepSeek 从 revision 1 增至 2。
- `synthetic.controls` 开关控件：本地策略阻断，机器人调用次数 0。
- `synthetic.unstable`：机器人已执行一次点击，但动作后合成画面持续不稳定；会话失败，动作计数为 1，Qwen 未再次调用，机器人未重试。

两个成功用例使用不同 `app_id`、不同措辞和不同动作类型，但经过相同的 `UniversalAgentOrchestrator` 与 `GenericSingleActionAdapter` 代码路径。

## 证据检查

成功模拟会话重复断言存在：

- `task_graph_revision_1.json`、`task_graph_revision_2.json`；
- `risk_audit_revision_1.json`；
- `trusted_observation_step_1.json`、`trusted_observation_step_2.json`；
- `qwen_decision_step_1.json`、`qwen_decision_step_2.json`；
- `controller_decision_step_1.json`；
- `verification_step_1.json`；
- `session.json`、`report.json`；
- 动作前和动作后的合成帧路径。

零动作阻断会话存在任务图、风险审计、动作前可信观察、Qwen 决策、控制器阻断、会话和报告；明确断言不存在 `verification_*` 或 `after_*` 伪证据。

## 静态安全审计

- `git diff --check`：通过，无空白错误。
- `poc/universal_agent_orchestrator.py` 中搜索 `douyin|wechat|抖音|微信`：无匹配。
- 执行入口不调用旧 `generic_step_planner.propose`；`generic_intent_parser.parse` 只剩 `/api/agent/generic-goal` 只读预览，不属于 `/generic-supervised/*` 执行路径。
- 密钥样式扫描未发现真实密钥。`task-*` 标识会被宽泛的 `sk-*` 子串规则误命中；`test-key` 是测试占位符；API key 代码行只读取参数或环境变量，不含明文凭据。

## 尚未完成的真实设备项目

- 已运行只读实机预检，但控制端窗口未启动，因此未能取得真实摄像头四帧；结果单独记录在真实验收报告。
- 未对真实机械臂发送点击、滑动或返回动作。
- 未验证真实页面动作前后 fingerprint、落点和视觉语义。
- 未用两个真实 App 和两种措辞重复一个安全动作验收。
- 未验证异常后人工恢复、服务重启后的会话恢复或长时间运行。

## 已知限制

- 外部状态动作仍全部阻断，包括发消息、关注、加好友、拉群、点赞、评论、保存、提交等。
- 文字输入、键盘、开关和账号效果动作仍未开放。
- 网页自动连续执行关闭，必须逐步观察和确认。
- 多台手机的数据模型、操作系统级跨进程设备租约和跨进程动作租约已具备；真实第二进程互斥测试通过，但没有真实多手机并行验收。同一设备仍只允许一个活动任务。
- 当前完成的是第一阶段低风险导航能力，不是“任意手机命令已经可实机执行”的最终产品。
