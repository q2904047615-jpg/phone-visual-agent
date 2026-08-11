# Qwen 单步视觉决策离线评估

本目录只验证 Qwen 视觉层，不连接摄像头、网页服务、`main.exe` 或机械臂。

## 输入

- `device_id`：当前设备的稳定标识，输出必须逐字回传。
- `current_subgoal`：DeepSeek 任务图给出的当前一个高层子目标；不得包含坐标或动作计划。
- `constraints`：本轮必须遵守的通用限制。
- `frames`：同一手机画面的至少四帧；离线评估会把一张既有截图复制为四帧，只验证视觉协议，不证明真实画面稳定性。

## 输出

- `page_state`：当前前台 App、页面、弹层、可见元素和状态。
- `status`：`action`、`finished` 或 `blocked`。
- `next_action`：至多一个通用视觉动作；不得包含动作列表或裸坐标。
- `target_region`：动作的语义区域。点击区域必须逐项复用唯一目标元素的 `bounds`。
- `expected_result`：执行一个动作后可由下一张画面验证的变化。
- `confidence`：综合页面与目标元素后的安全置信度。
- `reason`：本轮画面支持该动作、完成或阻塞结论的依据。

## 运行

在已配置 `DASHSCOPE_API_KEY` 的环境中运行：

```powershell
python poc\eval_qwen_visual_decision.py
```

结果写入忽略目录 `poc/output/offline_qwen_visual_decision/`。报告中的
`hardware_actions_enabled` 固定为 `false`。
