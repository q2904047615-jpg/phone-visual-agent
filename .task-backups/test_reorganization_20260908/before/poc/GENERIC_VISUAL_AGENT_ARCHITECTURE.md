# 当前通用视觉 Agent 架构

代码采用轻量 DDD 模块化单体。目录划分职责，不增加模型轮次或产品限制；完整产品合同见 [项目最终目标](../项目最终目标.md) 和 [用户决策与协议边界](../用户决策与协议边界.md)。

## 职责与唯一权威

| 层/入口 | 实际职责与源码 |
| --- | --- |
| interfaces / 组合根 | web_app.py 转换 HTTP/Pydantic 与装配依赖；官方客户端负责 API 合同 |
| application | universal_agent_orchestrator.py、universal_agent_sessions.py 管理串行会话；qwen_visual_decision.py 消费本轮唯一响应 |
| domain | qwen_task_context.py 整任务上下文；canonical_action_protocol.py 唯一动作解析；universal_action_controller.py 设备/scope/几何及一次执行结果；text_transport.py、confirmation_authority.py 等承载合同 |
| infrastructure | provider、generic_scene_observer.py、generic_action_adapter.py、device_executor.py、Robot、ADB Keyboard、相机、标定、独占与证据存储 |

依赖为 interfaces → application → domain；infrastructure 实现端口，由组合根注入。domain 不导入 Web、模型 SDK 或硬件实现，application 不反向导入 interfaces/infrastructure。独立标定和回放工具不是额外运行权威。

## 一次执行循环

原任务 + 本会话完整语义的实际动作/结果历史 + 当前新截图 → Qwen 同帧 scene/input/action 或 finish → 唯一 canonical 绑定 → 执行一个动作 → 新截图。

execution_history_entry 是立即动作后观察与后续重观察共用的纯领域文字历史投影。按 Open-AutoGLM 的当前图加文字历史方法，generic_action_adapter 不缓存上一动作前图，generic_scene_observer 每次请求只带本轮稳定帧；旧图仅保存在本地证据文件，不送入后续模型请求。保留本项目当前帧组及真实执行结果历史，不复制上游异常转 finish，不产生本地完成清单。

动作集合在观察前一次签发，模型、绑定和执行消费同一份 scope。动作匹配不等于完成；Qwen 的新图 finish 表示整任务完成，本地不按子目标或效果清单检查语义遗漏。DeepSeek 和固定计划已退役；本地不建立隐藏计划、候选排名或业务重排。

点按只使用 Qwen 同帧 decision.target 和 tap_point；不从元素框中心、OCR 或局部分割改点。非点按元素事实、当前输入、真实截图、机械几何、设备和一次凭据检查按既有合同保留。Qwen confidence 只作诊断。

## 文字、预算和效果

当前字段的 text、focused、preedit_text 来自同帧 input_structure。输入/清空/换行唯一使用 ADB Keyboard 固定广播；不保留机械文字、拼音选词、键盘 OCR、Companion 或回退。

整任务累计动作/观察预算跨请求保留，用尽暂停；新图纠正不重放旧 scope。输入和不确定外部效果不能自动重复。产品风险确认仅 authentication、financial_transaction；技术故障需说明实际原因。

架构/正向/变化样本测试只证明离线合同。当前加载及真机边界见根目录交接；不得把此架构说明当作功能验收结论。
