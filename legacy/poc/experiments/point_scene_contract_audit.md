# 点按额外 scene 元素：离线合同审查

> 2026-09-06：本文只保留历史反证。配套旧脚本已退役，源码恢复包见 `../output/retired_experiment_tools_20260906/retired-experiments.zip`；当前回归使用 `poc/test_point_scene_projection.py`，下面旧命令、字段要求和结果不得当作当前操作指南。

日期：2026-09-04。本报告正文保留实施前反证历史。用户随后批准方案，当前实施及测试结论以根目录交接文档与验收台账为准；不把下文历史“未实施”当作当前状态。

## 当前唯一能力缺口与证据

目标仍是网页自然任务“打开微信→打开文件传输助手→清空输入框→回主屏幕”的同会话真实闭环。
本报告只处理首步因重复 scene 元素而零动作失败的合同问题，不宣称完整任务成功。
原始证据：`poc/output/web/generic_supervised_20260904_160604_bd7674f5/before_step_1_frame_attempt_1_qwen_failure.json`。
保存响应未截断；原件未修改。前次“模型合同输出”分类描述触发位置，不等于根因全部在模型。

## 证实的触发链

1. Qwen 发布严格 `decision.target` 与 `tap_point=[450,830]`，坐标空间为 1000×1280。
2. 同响应又报告目标的 `scene.elements` 和 bounds。
3. `generic_scene_observer.py` 坐标规范化入口第603–604行硬拒绝点按非空 elements。
4. `test_flat_observation_contract.py` 第110–113行专门要求这个输入被拒绝。因此旧回归通过证明的是现有规则执行，不证明规则适合产品目标。
5. 请求 schema 的 elements 是通用数组，动作相关空数组要求写在 description/提示词中。当前响应违反这条语义合同，但不能据此认定模型没识别目标、设备断线或机械点偏。
6. 产品目标同时规定无关可选字段/重复摘要不得否决合法动作。把不参与点按的元素列表设成硬拒绝，构成需要明确调整的合同设计问题；不是未删除的另一套坐标执行器证据。

## 离线反证方法及结果

运行：`poc/.venv/Scripts/python.exe -X utf8 -B poc/experiments/audit_point_scene_contract.py`。
只在内存副本把 `scene.elements` 置空，其他响应字段保持原样；使用现有生产 envelope parser、scene parser 和 canonical binder，没有 monkeypatch 生产函数。

实施前13项检查全部符合预期（脚本现已升级为当前v14的14项重放，历史v13拒绝仅保留版本负样本）：

- 原始响应：复现相同拒绝。
- 原响应只去掉 elements：绑定原动作/原目标，标准化点击点为 `(0.45,0.648)`；这是既有坐标舍入结果，不是新点位。
- 四个变化样本：tap_semantic、dismiss_overlay、double_tap、long_press；目标改为设置按钮，点位变更，诊断列表故意含重复ID、错误输入角色与越界框。去掉非权威列表后仍只绑定明确 target 和给定点，未引入 bounds。
- 七个反例：缺 target、缺点、点越界、并存另一目标引用、严格 target 内夹带 bounds、缺目标证据、动作不在签发集合；均仍拒绝。

未覆盖：press_enter 的 typed 字段投影、真实帧验证、完整 Controller、机械执行、四段连续任务、模型下一次输出概率。
没有新增模型调用、API调用、设备动作或服务重载；没有跑全量回归，因为没有修改生产实现。

## 推荐方案及实施前反证

结论：**有条件可行**。

正式调整点按合同：`decision.target + tap_point` 仍是唯一动作目标与坐标权威；模型仍被指导不要重复输出元素，但若附带 scene.elements，入口仅保存原始诊断并在任何下游消费前排除它，不因此拒绝。
只对五种点按动作生效；非点按和 finish 的元素绑定规则不变。不从列表补目标、补点、取中心、重选或比较目标；缺失的必要动作字段不能靠忽略列表修好。

不能只删除第603–604行：后续 `_decision_element_ids` 可由 evidence_refs 选中元素，元素循环仍检查重复ID/bounds；`selected_scene_input` 仍读取原列表判断输入框；scene递归动作字段检查也在拒绝行之前。须在唯一入口投影点按 scene 后让所有下游只消费该投影，不能保存第二套可执行元素。

关键假设：在点按分支，完整动作身份已经由合法 target 唯一给出，列表不是完成证据或输入正文的必要来源。普通图标/按钮在本实验成立；typed 点按仍需正式测试。

会推翻方案的条件：丢弃列表后目标/点位发生变化；必须从列表恢复 typed 字段；finish 或非点按丢失必要元素；不合法动作因此通过；后端仍存在从原列表建立候选或否决的入口。出现任一条件应停止扩散修改，重新审查。

保留：JSON结构可解析与唯一动作、target必要身份、同帧 tap_point 合法范围、签发动作集合、typed input_structure 的字段/焦点/正文、scope/fingerprint、设备独占与标定、登录付款确认、每动作后新截图。

最小正式验证：纳入保存原响应及上述反例；补齐 press_enter/聚焦输入框、非点按/finish不变、完整观察器到Controller的组合测试。测试证明列表变化不影响相同权威动作，必要硬校验仍有效；稳定后再做一次相关回归和一次完整回归。真机另按授权安排，不能用本实验代替。

影响范围：观察器入口、提示词/schema说明、相关测试及正式协议/用户决策文档。预计不改机械 transport、ADB Keyboard、DeepSeek业务编排、公开API路径。若 wire接收语义变更需更新对应版本与展示断言。
回滚：记录实施前这几处工作树差异，只撤销新批次对应hunk，保留既有未提交修改；不得重置整个工作树。已产生真实动作不能靠代码回滚撤销。

不采用：整份响应全部宽松放行（丢失必要动作约束）；bounds中心或OCR回退（引入第二落点权威）；反复请求Qwen（未解决本地合同问题）；只删一条raise（下游仍消费列表）；直接照搬外部项目全部执行器（改变本项目机械与输入边界）。

## 参考实现核对

2026-09-04读取 Open-AutoGLM 官方 main 的 `phone_agent/actions/handler.py`：
https://github.com/zai-org/Open-AutoGLM/blob/main/phone_agent/actions/handler.py

`_handle_tap` 从 action.element 取模型点，`_convert_relative_to_absolute` 换算后交给 device_factory.tap；这条点击链不需要第二份 scene 元素框。本项目可借鉴单一点击点来源，但仍需相机/机械标定和明确目标身份，不能直接复制其ADB点击。

核对官方 issue #187、#376：
https://github.com/zai-org/Open-AutoGLM/issues/187
https://github.com/zai-org/Open-AutoGLM/issues/376

这些报告涉及标签/占位符等解析失败，不能视为本问题的现成修复；也不能从其页面显示“任务完成”推导真实执行成功。此次未找到官方针对本项目额外 scene 列表的直接方案，候选方案基于本地反事实重放和单一权威原则。

## 停止与交付边界

审查当轮只新增独立离线脚本与文档，未改生产代码、提示词、协议、API或手机。
后续用户已批准上述具体方案并实施；当前效果以根目录台账为准，本报告不签发真机成功结论。
