# 通用手机视觉 Agent 真实设备验收记录（2026-08-12）

## 结论

本次只完成了**只读实机预检**，没有进入真实动作验收。预检结果为
`ready=false`，且 `physical_actions=0`。

已通过的真实环境检查：

- `default-device` 没有其他活动会话，设备独占可用；
- DeepSeek provider 已配置；
- Qwen provider 已配置；
- 编排器使用第一阶段安全策略
  `2026-08-12-phase-one-navigation-v1`。

当前阻塞：

- 未检测到标题包含“智联新途”的机械臂控制端窗口；
- 因控制端离线，手机摄像头画面不可用，未能取得四帧和当前页面截图。

因此，本文档**不能证明真实机械臂闭环通过**，也不能证明双 App
真实验收完成。离线和模拟闭环的结果继续单独记录在
`docs/validation/2026-08-12-universal-agent-offline-validation.md`。

## 只读预检命令

```powershell
.\poc\.venv\Scripts\python.exe poc\run_universal_agent_live_preflight.py --device-id default-device
```

执行时间：`2026-08-12T14:39:52+08:00`

关键输出：

```json
{
  "ready": false,
  "physical_actions": 0,
  "device": {
    "device_id": "default-device",
    "exclusive_available": true,
    "active_session": null
  },
  "controller": {
    "online": false,
    "camera_online": false,
    "error": "没有找到标题包含‘智联新途’的窗口。请先打开 main.exe，并保持控制端窗口可见。"
  },
  "providers": {
    "deepseek_configured": true,
    "qwen_configured": true
  },
  "policy": {
    "version": "2026-08-12-phase-one-navigation-v1",
    "phase_one_matches": true
  },
  "blockers": [
    "机械臂控制端不可连接",
    "手机摄像头画面不可用"
  ]
}
```

## 零动作安全证明

`poc/test_universal_agent_live_preflight.py` 使用假硬件覆盖：

- 稳定四帧且所有依赖就绪；
- provider 缺失、画面不稳定、设备被其他会话占用；
- 控制端和摄像头离线；
- provider 状态读取异常且异常文本可能包含敏感信息。

测试把点击、滑动和返回方法设置为“一旦调用立即失败”。四项测试均通过，
所有路径的 `physical_actions` 均为 `0`，provider 密钥和异常中的敏感文本均未
进入 JSON 输出。

```text
Ran 4 tests in 0.072s
OK
```

## 下一次真实验收的开始条件

1. 用户启动 `main.exe` 并保持“智联新途”控制端窗口及手机画面可见；
2. 重新运行同一只读预检，必须得到四帧稳定画面和 `ready=true`；
3. 展示当前页面截图、设备独占状态、DeepSeek 当前子目标、Qwen 候选动作、
   区域、置信度与本地策略理由；
4. 用户只针对该页面和该候选明确授权一次动作后，才能调用一次 `/confirm`；
5. 动作后必须重新取得四帧、验证、replan 并暂停，不得自动执行第二个动作。

当前已停在第 1 项之前；未获得该页面和该候选的明确确认时，不执行真实动作。
