# 通用视觉操作 Agent PoC

Qwen 接收整任务、本会话实际执行历史和新截图，同次响应给出 scene/input_structure 与一个 canonical action 或整任务 finish；本地执行一次，再取新图。DeepSeek、固定子目标与本地预计算完成清单已退役。

当前权威：[项目最终目标](../项目最终目标.md)、[用户决策与协议边界](../用户决策与协议边界.md)、[项目交接文档](../项目交接文档.md)、[当前验收台账](../单一权威最小校验验收台账.md)。

## 配置与离线验证

启动、模型凭据、ADB Keyboard 和可信包名配置见 [网页说明](README_WEB.md)。不输出密钥。Agent 不操作卖家 main.exe。
仓库内的 `adb_keyboard_registry.json` 和 `app_package_registry.json` 已脱敏且默认关闭；真实设备请复制
`adb_keyboard_registry.example.json` / `app_package_registry.example.json` 到本机路径后，通过
`ROBOT_ADB_KEYBOARD_REGISTRY`、`ROBOT_APP_PACKAGE_REGISTRY` 指向它们，不要提交 serial、IP 或 Windows 用户路径。

在本目录已有依赖环境中运行（不需要 Windows、机械臂、ADB 设备或 Qwen 凭据）：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s . -p "test_*.py"
npm test
```

Linux/macOS 使用 `python3 -m unittest discover -s . -p 'test_*.py'`。CI 会在
`ROBOT_WEB_MOCK=1` 下执行同一套 Python 测试，并在 Ubuntu 安装 Playwright Chromium 后执行
前端单元和浏览器合同测试；Windows job 额外验证 Win32 适配器的导入、ABI 和边界。测试不会发起模型网络请求或物理动作。

npm test 覆盖协议、触摸页、浏览器合同和控制台四个测试入口。自动测试及 Mock 不替代真实任务。

## 八个正式入口

| 文件 | 用途 |
| --- | --- |
| web_app.py | FastAPI 接口与依赖组合根 |
| local_agent_api_client.py、agent_api_cli.py | 官方类型化 API 网关及 CLI；调用前核对源码/OpenAPI |
| eval_qwen_visual_decision.py | 保存截图的视觉评估；直接运行会调用远程模型，需相应授权 |
| eval_task_sequences.py | 无设备、无模型的动作序列回放 |
| capture_click_burst.py、run_xy_calibration.py | 人工维护用机械采集和标定，不是普通任务编排 |
| touch_calibration_server.py | 标定及动作验收页面服务 |

维护工具被保留，不代表自动授权运行。业务模块位于 agent/domain、agent/application、agent/infrastructure，职责见 [架构说明](GENERIC_VISUAL_AGENT_ARCHITECTURE.md)。controller_config.json、device_registry.json、tap_calibration.json 及本机 ADB/包名注册表属于设备配置，不是废弃代码。

协议版本分为三类：Qwen/canonical/device 输入等外部或安全边界合同；执行收据和证据存储合同；仅供内部诊断的实现版本。只有前两类参与兼容性判断，内部诊断版本不得成为第二套动作或完成权威。

## 证据和历史

输出在 output/；原截图、模型响应、报告、回滚快照保留。服务启动时会清理超过保留期的非活动运行目录，单个运行目录和总输出目录都有可配置配额；完整截图不会被静默截断，配额不足会报告存储错误。实验工具用途见 [experiments/README](experiments/README.md)，全项目清理处置见 [项目清理清单](../项目清理清单.md)。
