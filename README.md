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

每次截图都会先自检：画面全黑（相机没对准手机或手机熄屏），或配置了 `screen_region` 后屏幕明显移位，会隔一会儿再拍一次，仍不正常才暂停任务并说明原因，不会把这张图交给千问。真机任务开始时会占用与旧版控制台、平台连接器共用的跨进程设备锁（默认在系统临时目录的 `phone_visual_agent_device_leases/`，可用 `PHONE_VISUAL_AGENT_LEASE_DIR` 修改）；设备被其它程序占用时会拒绝启动。卖家控制端弹出提示框（例如“点击次数必须是一个整数”）时，任务会停下并显示提示原文，点掉后再继续；“连点次数”通过 Windows UI Automation 写入并回读，不模拟键盘，不会把按键打进别的窗口。

## 设备配置

第一次启动时会从 `config/devices.example.json` 生成 `config/devices.json`（不进 git），在里面改：

| 字段 | 说明 |
| --- | --- |
| `devices[].driver` | `robot`（机械臂 + 相机，默认）、`mock`（模拟手机）、`adb`（可选：纯 ADB 截图和点击，用于没有机械臂的手机） |
| `devices[].machine_position` | 卖家控制端的 1–10 号机位，任务开始前自动切换 |
| `devices[].adb_serial` | 填了才能用 ADB Keyboard 输入文字、清空、换行，以及按包名直启 App |
| `devices[].apps` | 可直启 App：`{"抖音": "com.ss.android.ugc.aweme"}`；不在列表里的 App 由千问在桌面点图标打开 |
| `devices[].calibration` | 触控标定文件；未启用时坐标不纠偏 |
| `devices[].screen_region` | 可选：标定时手机屏幕在相机画面中的位置 `[x0, y0, x1, y1]`（0~1）。填了以后，画面里的屏幕明显移位就暂停任务，避免点偏。请在亮色页面（如设置）上量，深色壁纸会让下边界量小 |
| `controller` | 卖家窗口标题、点击按压时长、滑动参数；`nav` 是手机导航栏三键在相机画面中的位置（0~1 比例，和点击一样经过触控标定修正；换手机或调整相机后要重新量） |
| `agent` | 默认动作/观察上限（100/200）、动作后等待时间、截图尺寸、模型超时等 |

从旧版迁移（会读取旧的设备登记和你本机的 ADB/包名注册表）：

```bash
python -m pva.tools.migrate_config --adb-registry 你的adb_keyboard_registry.json --app-registry 你的app_package_registry.json
```

环境变量：`DASHSCOPE_API_KEY`（必需）、`VISION_MODEL`（决策模型，默认 `qwen3.7-plus`）、`LOCATOR_MODEL`（按决策模型说的控件取点按坐标，默认 `qwen3-vl-plus`；设为空字符串则直接用决策模型的坐标）、`VISION_MODEL_BASE_URL`、`PVA_PORT`（默认 8765）、`PVA_OUTPUT`、`PVA_MOCK=1`。

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
- **真机验收（2026-11-04，1 号机位，小米 HyperOS 极简模式；决策 `qwen3.7-plus` + 定位 `qwen3-vl-plus`）**：点击、长按、滑动翻页、拖动（动作验收页回报 passed）、返回、桌面、最近任务（先回桌面再打开）、ADB Keyboard 输入中英文/清空/换行、按包名直启均通过；整任务循环通过——读取系统版本、进入蓝牙页后只用返回键退回桌面、查看最近任务里的后台 App、浏览器搜索框输入并清空、便签新建两行再删除、找不存在的 App 时如实报告失败。画面自检、跨进程设备锁、卖家弹窗检测在真机上生效。
- 已知现象：偶尔有一次点击没有生效（下一轮截图会看出来并重试，原因未查明）；极简模式桌面上，长按系统 App 图标没有任何菜单，长按普通 App 会弹出卸载确认。手机通过无线 ADB 连接，`adb tcpip 5555` 在手机重启后失效，需要重新打开无线调试并配对。
- 机械臂没有回零开关：控制板每次上电时触控笔所在的位置就是原点。main.exe 正常关闭时会让机械臂回原点（经卖家后台服务 JxbService 发 `X0Y0Z0`），但卡死后被强行结束、或开着 main.exe 直接重启电脑时不会回，下次上电原点就偏了。重启电脑、拔 USB、强行关闭 main.exe 之前，先双击 **`机械臂回原点.cmd`**（不依赖 main.exe，main.exe 卡死时也能用；`机械臂回原点.cmd -Center` 让镜头回到手机上方）。本机还在本地组策略里登记了关机脚本：每次关机/重启前以 SYSTEM 身份运行 `C:\ProgramData\PhoneVisualAgent\arm_home.ps1`（`pva/tools/arm_home.ps1` 的副本），结果记在同目录的 `arm_home.log`；要取消，在 gpedit.msc“计算机配置 → Windows 设置 → 脚本（启动/关机）→ 关机”里删除即可。已经偏了就断电、手动推回原点再上电。main.exe 卡死时，控制台会报“main.exe 没有响应”并停止任务。
