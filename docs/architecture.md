# 架构

```
pva/
  __main__.py          启动控制台（python -m pva）
  config.py            config/devices.json + 环境变量
  actions.py           唯一的动作词表：解析、校验、中文描述
  model/
    qwen.py            DashScope OpenAI 兼容客户端；从回复中提取 JSON
    prompt.py          系统提示词；把任务、历史、全部截图拼成一次请求
  agent/
    session.py         一个任务的循环与状态机
    manager.py         创建/查询任务；历史任务从 output/ 读取
  devices/
    base.py            Device 接口：capture() + execute(action)
    robot.py           机械臂：卖家窗口里的鼠标事件 → 机械臂触控
    seller_window.py   Win32 层（窗口查找、截图、点击/长按/滑动、机位、连点次数）
    calibration.py     九点标定的仿射纠偏
    adb.py             固定 ADB 命令族：ADB Keyboard 文字、包名启动、（adb 驱动）截图/点击
    adb_device.py      可选：纯 ADB 设备
    mock.py            模拟手机（真实像素 + 可交互状态），用于演示和测试
    __init__.py        DeviceManager：设备表、每个控制端一把锁、任务独占
  web/
    server.py          FastAPI：JSON API + 静态控制台
    static/            index.html / app.js / styles.css（无构建步骤）
  plugins/             可选的 App 专用插件（福袋、汽水广告页）
  tools/               标定页服务、九点标定、旧配置迁移
tests/                 离线测试（pytest）
config/                设备配置样例与标定文件
legacy/                旧版代码与文档（只读）
```

## 一次循环（`Session._iteration`）

1. 检查预算（动作上限 / 观察上限，用尽则暂停，可调高后继续）。
2. 在控制端锁内截图，保存为 `step_NNN.jpg`。
3. 构造请求：系统提示词 + 任务 + 用户补充说明 + 可直启 App + 设备支持的动作 + 已执行动作（真实记录）+ **本任务全部历史截图（缩略图，按时间顺序）** + CURRENT 截图（原尺寸）。请求（图片已省略）与回复写入任务目录。
4. 解析回复 JSON：`observation / last_result / reply / reason / action / risk`。动作由 `actions.parse_action` 校验（类型、坐标 0–1000、App 是否登记、设备是否支持）。无法解析时不执行任何动作，重新截图再问；连续 3 次失败则任务失败。
5. `finish`：`success=true` → 成功；`success=false` → 失败并显示原因。
6. `risk` 为 `authentication` / `payment` → 等待用户确认；逐步模式下每个动作都等待确认。
7. 在控制端锁内执行**一个**动作；设备故障直接停止任务并报告原因（不重试，避免重复发送）。
8. 等待 `settle_seconds` 后回到第 1 步。

`recents`（打开最近任务）按已批准的规则执行：先按 Home，截一张图存档并加入历史，再按最近任务键。

## 设备与独占

- 坐标统一是截图上的 0–1000 网格。机械臂设备：网格 → 标定纠偏 → 相机像素（相机区域从卖家窗口客户区左上角开始，因此像素即客户区坐标）→ 鼠标事件。
- 共用一个卖家窗口的多个机位共用一个 `hardware_key`：同一时间只有一个任务/插件持有（`reserve`），每个物理操作都在该控制端的锁内（`hardware`）。不同控制端、模拟手机之间可以并行。
- 任务开始时 `prepare()`：切换到该设备的机位、把卖家“连点次数”设为 1。

## 与旧版的差异

| 旧版 | 新版 |
| --- | --- |
| ~16k 行协议代码（canonical/scope/fingerprint/credential/多套版本号） | 一个动作词表 + 一个循环，约 2.3k 行核心代码（不含沿用的 Win32 驱动和模拟手机） |
| 80 行、自相矛盾的提示词（同时要求和禁止 element_id） | 一份简洁提示词，动作、坐标、输入、确认规则各一段 |
| 2.6k 行网页脚本，功能入口分散 | 三栏控制台：设备/任务/轨迹，历史与插件分页 |
| 任务只能在本次服务进程内查看 | 每个任务完整落盘，重启后可查看 |
| 模型无法表达“做不到” | `finish` 带 `success=false` 和原因，避免空耗预算 |
| 没有无硬件的演示方式 | 内置模拟手机，可用真实千问端到端体验 |

保留不变：只有登录/付款需要确认；每步全部截图交给模型；一次一个动作、动作后必须重新截图；文字只走 ADB Keyboard；包名直启只用本地登记表；动作/观察累计预算；`recents` 先回桌面。
