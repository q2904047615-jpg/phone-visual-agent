# Beta unknown/已有物理动作恢复补偿（2026-10-14）

## 结论

`unknown` 与“已有物理动作”恢复补偿已完成最小通用合同：明确确认后只能选择取消且不重放，或创建全新观察会话；新会话继承原会话已消耗的物理动作和观察预算，不自动执行动作、不复用旧 scope。

## 策略

- `not_dispatched` 且物理动作数为 0：原有安全对账，可取消或新观察续跑。
- `unknown` 或物理动作数大于 0：不可按未派发处理；可记录 `unknown_effect_cancelled_no_replay`，或以 `unknown_effect_fresh_observation` 创建新会话。
- 新会话携带 `recovery_parent_session_id`、`recovery_mode`、`initial_physical_actions` 和 `initial_observation_attempts`，保留整任务预算。
- 设备未登记、设备不可用或观察预算耗尽时，恢复门保持不变，不自动重放。

## 真实服务核验

- API revision：`7aaf46351546a7b478088404feb97cf7503869ec`，PID `9120`。
- 使用官方 CLI 对历史 `api-scope` 会话执行 `confirmed=true + external_effect=unknown`，返回 `cancelled`。
- `recovery_resolution` 写入 `unknown_effect_cancelled_no_replay`，`physical_actions_before=0`，replacement 为 null。
- 服务最终 controller/camera 在线、`busy=false`；未执行物理动作，未操作卖家 `main.exe`。

## 离线验证

- 聚焦恢复回归：25 项通过。
- 完整离线回归：899 项，0 failures，0 errors，4 skipped。
- `compileall`、`git diff --check` 通过。

## 仍未闭合

`unknown + fresh_observation` 和已有物理动作的续跑只完成离线合同，尚未用真实服务执行带历史物理动作的样本；补偿不会自动判定业务成功，也不会自动重做原动作。由于当前只有一台真实设备，多设备调度延期；下一项按单设备主线，在产生真实恢复状态后执行 fresh-observation 验收。
