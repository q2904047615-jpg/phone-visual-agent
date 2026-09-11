# 隔离审查与验证工具

这些工具不是正式设备执行入口。历史输出保留在 `poc/output/`；旧实验通过不代表当前协议或真机通过。

## 保留的工具

- `audit_runtime_restrictions.py`：当前源码限制索引及文档链接一致性检查；`--refresh`重新生成索引。
- `project_cleanup_inventory.py`：Git跟踪及未忽略路径初筛，不是死代码证明或自动删除许可。
- `check_simple_contract.py`：离线unittest汇总，供指定模块或全量运行使用。
- `probe_simple_input_contract.py`：保存输入样本经当前观察/绑定/Controller的隔离验证；默认离线，`--online`会发送截图到模型，必须另有相应授权。
- `probe_explicit_input_focus.py`：历史明确焦点及同图换措辞的识别样本；默认会发送截图到模型，本次不运行。保留供对应历史证据复核，不用历史通过推定当前行为。
- `point_scene_contract_audit.md`：历史合同反证记录，不是现行运行规则。

## 已移除的过期实验入口（2026-09-06）

| 脚本 | 不再适用的依据 | 当前覆盖 |
| --- | --- | --- |
| qwen_action_union_contract.py | 独立oneOf及12元素上限，与当前扁平nullable协议不符 | test_flat_observation_contract.py |
| qwen_strict_direct_point_contract.py | 自建旧严格字段与固定样本坐标评分，不是现行canonical | test_point_scene_projection.py、通用eval_qwen_visual_decision.py |
| qwen_flat_contract_probe.py | 比较历史v12完整请求，读取已退役target_region副本 | test_flat_observation_contract.py、test_point_scene_projection.py |
| audit_point_scene_contract.py | 旧元素ID/必填证据断言已经被当前合同替代 | test_point_scene_projection.py |
| probe_runtime_restrictions.py | 要求已删除的500/4000容量、2%边缘及App后置门禁再次触发 | test_approved_restriction_choices.py、test_post_action_authority.py |

五份原脚本已逐文件SHA-256验证并保存为 `../output/retired_experiment_tools_20260906/retired-experiments.zip`。如需追溯，先解压到隔离临时目录并核对同目录backup_manifest.json；不要将旧脚本直接恢复成当前运行入口。原图片、模型响应、实验结果和现行测试未删除。
