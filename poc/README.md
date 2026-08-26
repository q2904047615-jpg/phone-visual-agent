# 手机通用视觉操作 Agent PoC

本目录只维护通用视觉操作主路径：DeepSeek 生成类型化任务图，Qwen 从当前
可信画面的 canonical candidates 中选择一个动作，本地控制器复核并执行，随后
重新观察和验证。业务 App 名称不参与任务编排。

权威边界与阶段状态见根目录：

- `项目最终目标.md`
- `用户决策与协议边界.md`
- `项目交接文档.md`
- `复杂输入完整批次验收台账.md`

## 本地验证

在 `poc` 目录执行：

```powershell
python -m unittest discover -s . -p "test_*.py"
node .\test_frontend_protocol.js
node .\test_frontend_browser_contract.js
```

自动测试、离线截图和 Mock 结果只证明代码合同，不能替代真机验收。

## 控制器配置

通用底层控制参数位于 `controller_config.json`；触控标定由 `agent/infrastructure/tap_calibration.py`
和 `run_xy_calibration.py` 管理。`robot_gui_poc.py` 仅封装卖家控制端的画面捕获、
点击、长按、拖动、滑动和系统导航原语，不包含 App 识别、业务流程或任务计划。

任何真机动作都必须经过当前任务图、可信观察、scope、方向凭据和一次性执行权
校验。不得用本目录脚本绕过官方类型化 API 客户端直接驱动设备。

## 当前正式入口

网页服务只公开通用会话、通用场景观察、能力验收、设备状态与预览接口。
本地 Agent 调用必须使用 `local_agent_api_client.py` 或 `agent_api_cli.py`，并先从
当前服务 OpenAPI 核对路径和 schema。
