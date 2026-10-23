# 手机视觉 Agent

用一句自然语言让手机完成任务。千问视觉模型（Qwen-VL）看摄像头里的手机画面，每次只决定**一个**动作；机械臂（或 ADB Keyboard 文字输入）执行后重新截图，再交给千问判断下一步，直到任务完成。没有为某个 App 写死的流程——换 App、换说法都走同一个循环。

产品方向的唯一权威是 [项目最终目标.md](项目最终目标.md)。

```
自然语言任务 ──► 截图 ──► 千问：看图 + 已执行动作 + 全部历史截图 ──► 一个动作 / 完成
                  ▲                                                  │
                  └──────────── 机械臂 / ADB Keyboard 执行 ◄──────────┘
```

## 快速开始（Windows）

1. 安装 Python 3.11+（勾选 *Add to PATH*）。
2. 双击 **`设置千问密钥.cmd`**，粘贴阿里云百炼的 `DASHSCOPE_API_KEY`。
3. 先体验：双击 **`启动模拟演示.cmd`**。浏览器会打开控制台，所有设备换成内置的模拟手机（设置 / 备忘录 / 消息 / 相册），可以直接让千问操作它，比如“打开设置，把蓝牙打开”。
4. 真机：打开卖家控制端 `main.exe`（窗口标题含“智联新途”，相机画面和底部操作栏完整可见），然后双击 **`启动控制台.cmd`**。

首次运行会自动创建 `.venv` 并安装依赖。命令行方式：

```bash
python -m pva            # 真机
python -m pva --mock     # 模拟手机
```

## 控制台

- **设备**：切换机位/手机，实时画面，空闲时可勾选“手动控制”直接点画面、按返回/桌面/最近任务/滑动，切换卖家机位。
- **任务**：输入任务，选择“自动执行”或“逐步确认”（每个动作先给你看），设置动作/观察上限、是否深度思考。运行中可以暂停、继续、停止，或给千问发补充说明（下一步生效）。
- **确认**：只有**登录/身份认证**和**付款**会停下来等你点“执行”。发送、评论、关注、删除、设置等普通操作自动执行。
- **执行轨迹**：每一步的截图、动作、千问的画面描述和理由、执行结果。
- **历史记录**：所有任务都保存在 `output/tasks/<时间>_<任务ID>/`（截图、每步的模型请求与回复、session.json），服务重启后仍可查看。
- **插件**：抖音福袋监控、汽水音乐广告页测试（见下文）。

右上角 **全部停止** 立即停止所有任务；真机执行中按键盘 **Esc** 也会中断机械臂动作。

## 设备配置

第一次启动时会从 `config/devices.example.json` 生成 `config/devices.json`（不进 git），在里面改：

| 字段 | 说明 |
| --- | --- |
| `devices[].driver` | `robot`（机械臂 + 相机，默认）、`mock`（模拟手机）、`adb`（可选：纯 ADB 截图和点击，用于没有机械臂的手机） |
| `devices[].machine_position` | 卖家控制端的 1–10 号机位，任务开始前自动切换 |
| `devices[].adb_serial` | 填了才能用 ADB Keyboard 输入文字、清空、换行，以及按包名直启 App |
| `devices[].apps` | 可直启 App：`{"抖音": "com.ss.android.ugc.aweme"}`；不在列表里的 App 由千问在桌面点图标打开 |
| `devices[].calibration` | 触控标定文件；未启用时坐标不纠偏 |
| `controller` | 卖家窗口标题、点击按压时长、导航栏三键位置、滑动参数 |
| `agent` | 默认动作/观察上限（100/200）、动作后等待时间、截图尺寸、模型超时等 |

从旧版迁移（会读取旧的设备登记和你本机的 ADB/包名注册表）：

```bash
python -m pva.tools.migrate_config --adb-registry 你的adb_keyboard_registry.json --app-registry 你的app_package_registry.json
```

环境变量：`DASHSCOPE_API_KEY`（必需）、`VISION_MODEL`（默认 `qwen3-vl-plus`）、`VISION_MODEL_BASE_URL`、`PVA_PORT`（默认 8765）、`PVA_OUTPUT`、`PVA_MOCK=1`。

### 文字输入

手机安装 [ADB Keyboard](https://github.com/senzhk/ADBKeyBoard) 并设为当前输入法，电脑能 `adb devices` 看到手机，然后在设备配置里填 `adb_serial`。控制台“设备”信息里会显示“ADB Keyboard 就绪”。ADB 只用于固定的文字广播和包名启动，模型不能传任何 ADB 参数。

### 触控标定

机械臂落点有偏差时重新标定（每台手机/机位一次）：

```bash
python -m pva.tools.calibration_page                        # 1. 启动标定页服务（端口 8770）
#    手机浏览器打开 http://<电脑IP>:8770/ ，放在相机下
python -m pva.tools.calibrate collect --device device-local-01    # 2. 九点采集并拟合
#    刷新手机上的标定页
python -m pva.tools.calibrate validate --device device-local-01   # 3. 独立九点验证，通过后自动启用
```

## 插件

插件与通用 Agent 共用设备，但互不干扰；同一控制端同时只运行一个任务或插件。

- **抖音福袋监控**：手动打开直播间后启动，自动点福袋、发送预填评论、等待开奖，疑似中奖发邮件。使用本地 OCR，需要先安装额外依赖：`python -m pva.plugins.lucky_bag.setup`。识别模式在 `pva/plugins/lucky_bag/settings.json`（`upstream` = 上游检测器 + ADB，`local` = 模板 + 机械臂）。邮件用 `GMAIL_*` 环境变量配置，收件人默认 `LUCKY_BAG_NOTIFICATION_RECIPIENT`。
- **汽水音乐广告页测试**：手动打开一个广告后点“观察”，千问报告页面状态；最多经你确认执行一次“返回”，从不点击领取。

## 开发

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests -q        # 不需要硬件、模型密钥或网络
```

代码结构见 [docs/architecture.md](docs/architecture.md)。旧版代码与验收台账保存在 [legacy/](legacy/README.md)，只作追溯。

## 状态说明

- 离线测试覆盖：动作解析、标定数学、机械臂坐标换算（Win32 调用打桩）、任务循环、预算/暂停/确认、HTTP API、插件流程、模拟手机端到端。
- **尚未在真机上验证本版本**：机械臂、相机、ADB Keyboard 和真实千问调用需要在接好硬件的电脑上验收。底层机械臂驱动（`pva/devices/seller_window.py`）沿用旧版经过真机验证的实现。
