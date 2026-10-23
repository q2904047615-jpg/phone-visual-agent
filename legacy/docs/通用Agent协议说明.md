# 通用 Agent 当前协议说明

本文说明当前源码职责，不设立额外产品边界。产品最高权威为 [项目最终目标](项目最终目标.md)，明确决定见 [用户决策与协议边界](用户决策与协议边界.md)。

## 当前协议版本

| 合同 | 版本 |
| --- | --- |
| 整任务上下文 | 2026-09-06-single-visual-task-v1 |
| Qwen 单步结果 | 2026-09-06-qwen-whole-task-v19 |
| canonical 动作 | 2026-09-06-canonical-whole-task-v10 |
| Controller | 2026-09-06-universal-adb-text-v28 |
| 单步观察 | 2026-09-07-single-step-current-frame-v22 |
| 输入事实 | 2026-09-06-input-structure-field-preedit-v17 |

版本常量以源码为准；服务是否已加载见交接，不由本表推断。

## 模型与本地分工

2026-09-07用户批准 `2026-09-07-recent-navigation-v1` 有限例外：会话选择 open_recent_apps 后，本地从非 Launcher 先执行 Home，下一张当前图确认 Launcher 后才执行 open_recent_apps，再用新图交回 Qwen。当前已在 Launcher 可直接打开；未确认时共用整任务预算按新图纠正 Home。暂停保留进度、恢复重新观察；pending 期间其他 action/finish 不能跳过导航。原始模型建议保留在 model_binding 证据，实际决策标记 decision_source=local_recent_navigation，历史仅记录实际动作。只按 canonical 动作触发，不按任务关键词；不恢复其他子目标或完成清单。显式 open_recent_apps 能力范围在观察前包含真实支持的 Home 前置动作，底层原语不串行动作。下列一般 Qwen 权威以此唯一例外为限。

1. 网页原任务直接交给 Qwen，新会话历史为空。没有 DeepSeek 初始计划和固定子目标。
2. 本地按设备能力一次签发动作集合；Qwen 接收原任务、本会话真实动作/结果历史及新截图。历史同时保留 canonical_action 的完整语义目标/标签/正文和 action 的执行参数、次数、transport 回执及视觉结果。按用户选定的 Open-AutoGLM 方法只传本轮当前稳定帧，历史以文字保留。不缓存、传递或兼容上一动作前图，新会话历史为空，每步仍一次模型调用。Qwen finish 是唯一语义完成判断，不新增本地效果清单；解析/执行异常不转成功。
3. Qwen 同次响应独立报告 scene/input_structure，再给一个 action 或整任务 finish，并以 previous_action_outcome=matched/unmatched/uncertain 评价最后动作（初始无历史为 null）。
4. canonical 解析器唯一绑定该动作；Controller 检查当前输入结果、设备/观察、坐标/机械和一次执行凭据。
5. 执行后取新图。下一响应直接给下一动作或整任务 finish，不额外请求子目标完成图。

整任务语义完成由 Qwen 判断，用户已接受移除预计算效果清单的本地完成检查。本地不独立证明任务无遗漏；旧消息不能作为本次输入/发送的完成理由，这一语义要求由整任务和真实历史提示 Qwen 遵守，不伪称存在本地确定性完备证明。matched 不等于整任务完成。异常不能变成成功 finish。

## 点按、滑动和文字

点按类只用同帧 decision.target 的身份与 tap_point 坐标；scene.elements 附带内容不参与点按执行或否决。非点按元素规则仍适用。滚动容器用 scroll，操纵当前元素用 swipe_element 的同帧起终点。后台清理的用户专属顺序见产品合同，不用卡片滑动替代。

输入只消费同帧 input_structure.application_inputs 的正文、focused、完整边界及独立 preedit_text。模型不必填写本地元素编号或完成证据路径。非空文字动作须当前字段 focused=true；未知焦点可按新图聚焦。空清空无需执行；清空后验证正文/预编辑为空，不要求持续聚焦。Qwen 必须区分初始既有内容和本次效果，整任务 finish 的语义由模型负责。

唯一文字 transport 为 ADB Keyboard：输入和换行使用固定 ADB_INPUT_B64，清空使用固定 ADB_CLEAR_TEXT。input_verified_text.text 是当前选中字段应达到的完整逐字正文，旧正文是严格前缀时只追加差额；否则先单独清空、新图后再输入。显式 exact_input_text 必须完全一致。press_enter 在同帧 multiline=true 的字段中追加换行，不是卖家键盘 Enter 或隐式发送。广播回执不等于内容正确，结果仍需新图验证。

## 预算、风险和故障

整任务预算累计动作和观察，用尽保留进度暂停，继续先取新图。手动暂停也保留会话及设备归属，清除旧动作和登录/付款授权；执行中暂停在本次动作及新图结果记账后停下，不等待整个循环。paused 与 budget_paused 均列入活动会话；取消或真实终态才释放设备。不设置任务正文、字段数量和模型元素数量的人为容量阈值；实际资源失败如实记录。

仅 authentication、financial_transaction 需要产品风险确认。设备离线、图像不可用、scope 过期、目标无法唯一确定或真实机械不可达属于技术故障。confidence、旧元素编号、scene 重复输入内容不形成第二否决权。真实截图质量检查和防重复效果保持，本批不改阈值。

当前源码依据与限制必要性见 [架构](poc/GENERIC_VISUAL_AGENT_ARCHITECTURE.md)、[审查](运行限制必要性审查.md)。历史协议说明的完整快照见 [历史索引](历史文档索引.md)，不能恢复为运行规则。

## 确认与接口

HTTP 路径、方法及官方客户端入口保留。会话公开 task_context_protocol/task_id/revision/raw_goal/history，不再公开 task_graph。动作与登录/付款确认 scope 使用 step_id，不接受旧 subgoal_id。风险来自当前选中动作的正式效果语义，预览展示当前动作和摘要，不使用预先计划的 EffectIntent。新旧协议不同时运行，旧会话只能只读追溯。
