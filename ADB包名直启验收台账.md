# ADB 包名直启验收台账

更新时间：2026-08-30（Asia/Shanghai）

## 1. 当前唯一验收缺口

在不改变通用多 App、唯一 canonical 权威、每步一张当前截图、一次一动作和动作后新观察合同的前提下，新增可选 `launch_app`：只有当前设备的可信注册表能把当前 typed App 目标唯一解析为 Android 包时，才允许执行一次固定 ADB 包启动；否则继续使用当前画面上的 App 图标。

本批停止条件：

1. 正向与变化样本证明两种不同 App 都能由可信映射生成唯一 `launch_app`；
2. 映射/ADB/serial 缺失时不生成直启候选且视觉点击不受影响；
3. 非法包名、任意命令、Shell/Intent/组件透传在 transport 前 0 动作拒绝；
4. 直启后只有新观察的结构化 App 身份唯一证明 typed 目标已在前台才 matched；
5. 相关回归、一次完整回归、文档和本地提交完成；不推送。

## 2. 已有证据与根因分类

- 当前唯一 canonical 集合没有 `launch_app`，`DeviceExecutor` 也没有包启动 transport；自然语言“打开 App”只能在当前画面找到可见图标后点击。
- Open-AutoGLM 的可复用机制是 typed `Launch` 后继续截图循环；其 ADB Keyboard、裸坐标和宽泛动作处理不适合本项目。
- 本机只读检查当前 `PATH` 与常见 Android SDK 位置均没有 `adb.exe`。这是当前环境能力缺失，不是模型、机械臂、风险或 App 识别错误。
- 根因类别：通用能力缺口。不存在需要修补的单一 App、截图或固定坐标失败。

## 3. 同类样本

正向：两个不同 semantic App 目标，各自由当前设备可信映射签发不同 opaque `launch_ref`，只生成一个直启候选。

变化与反例：

- 同一目标没有映射；
- ADB 未配置、可执行文件不存在或 serial 为空；
- 包名为空、只有单词、含空格/换行/分号、命令替换、组件或 Intent 片段；
- transport 返回非零、超时或找不到目标；
- transport 返回成功但新截图仍是 Launcher、unknown 或其他包；
- 直启能力不可用但当前截图有唯一目标 App 图标。

## 4. 通用修复

- application 只查询 target-specific `LaunchTargetResolver` 端口；domain canonical 只接收本地签发的 opaque ref 与固定目标包，不读取配置或 subprocess。
- infrastructure 使用独立 App 包注册表绑定 `device_id + adb serial + aliases + package`，并以固定 argv、`shell=False`、固定超时执行一次启动。
- canonical 仅在当前活动目标是 App、当前未处于该 App、且 resolver 唯一成功时生成 `launch_app`；同一目标的图标点击在这一轮退出候选。resolver 失败时不添加 launch，不制造 blocked。
- Controller 与动作后验证继续消费同一 formal transition；包名只绑定可信 transport，新观察以真实运行包、typed App ID、App 名称或结构化页面身份之一唯一证明目标 App，App surface lineage 才能推进后继子目标。
- transport 已尝试后的超时、非零返回或异常不自动重发；Adapter 仍取得动作后四帧并由新观察裁决实际结果。公开 execution/report 保留 `transport`、`transport_status` 和 `mechanical_contact_ack=false`。
- `/api/device` 的 `protocol_physical_actions` 从唯一 canonical 动作集合派生；当前设备注册表实际启用时，`enabled_physical_actions` 才包含 `launch_app`，不再维护遗漏直启能力的第二份硬编码词表。
- 回滚方式：删除 `launch_app` canonical 成员、注册表/adapter 和对应测试；视觉图标路径本身不需要恢复或改写。

## 5. 验证清单

1. canonical 正向、跨 App 变化与视觉回退；
2. registry/transport 的包名与命令注入反例；
3. DeviceExecutor 恰好一次调用与 `physical_actions=1`；
4. Controller 动作后运行包、语义 App ID、App 名称三种正向身份与错 App/unknown/Launcher 反例；
5. App surface lineage 的 launch 正向与错包/错 scope 反例；
6. 能力禁用时 Qwen 0 个直启候选、视觉路径保持；
7. 相关回归、生产行数、静态编译、旧冲突扫描和 `git diff --check`；
8. 冻结后运行一次完整 Python 回归；更新交接并本地提交，不重载项目 API、不操作 `main.exe`。

## 6. 当前状态

- 产品边界：用户已批准，已写入 `用户决策与协议边界.md`。
- Open-AutoGLM 对照：已完成，只借鉴 `Launch → 新截图` 的通用循环。
- 运行环境：当前未找到 `adb.exe`，因此本批不做直启真机动作。
- 内部功能：已实现。可信 resolver、唯一 canonical 候选、固定 argv transport、一次执行、动作后新观察、typed receipt/lineage、公开能力状态和视觉回退均已接通。
- 相关回归：`611/611` 通过；直启专项 `12/12` 通过，其中成功、timeout、非零返回均证明一次 transport 后取得前后各四帧，且没有重发。
- 完整回归：生产代码冻结后 `922/922`（`87.769s`）通过；`compileall` 和 `git diff --check` 通过，公开路由与请求 schema 未改变。
- 生产规模：当前 `poc/agent/**/*.py` 为 `66` 个文件、`20142` 行。用户已明确要求先完成能力、暂不继续降代码，因此相较上一批 `19942` 行净增 `200` 行并暂时高于 20k；没有用隐藏实现或排版压缩掩盖。
- 真机结果：未新增。当前机器未找到 `adb.exe`，默认注册表也保持关闭，因此没有重载项目 API、没有创建真机会话、没有执行包启动或机械动作，也没有操作 `main.exe`。这不影响离线实现结论，但不能声称当前设备已完成直启真机验收。
- 提交边界：本批按用户授权本地提交，不推送。
