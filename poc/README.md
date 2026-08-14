# 手机机械臂自动控制 PoC

> 产品与架构最高优先级依据见根目录 `项目最终目标.md`。现有单功能PoC只能作为底层能力证据，不代表最终产品范围；后续实现不得退化为固定命令或App专用线性闭环。

> 新增：多 App 本机网页控制台请查看 [README_WEB.md](README_WEB.md)，或双击项目根目录的 `启动机械臂网页控制台.cmd`。

## 当前通用 Agent 主路径（第一阶段）

当前官方受监督入口是 `/api/agent/generic-supervised/*`，数据流固定为：自然语言目标 → DeepSeek 动态任务图 → 四帧可信观察 → Qwen 唯一下一视觉动作 → 本地通用导航策略 → 用户对当前 task/revision/subgoal/risk/observation/fingerprint 的精确确认 → 最多一个机械臂动作 → 四帧重新观察与验证 → DeepSeek 新 revision。新命令只要能由已开放的通用动作组合完成，就不应修改代码或增加 App 分支。

第一阶段只开放经过本地策略验证的低风险导航动作，包括通用语义入口点击、关闭遮挡层、滑动、系统返回和等待变化。外部状态、未知影响、输入、开关、账号效果及非导航按钮均阻断；网页自动连续执行关闭，同一设备同一时间只能有一个活动会话。

离线与模拟闭环测试：

```powershell
cd .\poc
.\.venv\Scripts\python.exe -m unittest test_universal_agent_orchestrator test_universal_agent_mock_loop -v
node test_frontend_protocol.js
```

合成模拟测试使用 Pillow 自有画面，不依赖任何真实 App 脚本。通过这些测试只证明代码协议、安全门和模拟机械臂闭环；不等于真实摄像头、真实机械臂、多台手机并行、文字输入或外部状态动作已经验收。下面保留的旧单功能 PoC 仅作为历史底层动作证据，不是通用 Agent 的任务编排方式。

这个 PoC 不修改卖家软件、不破解串口协议。它把现有
`智联新途机械臂控制端` 当作执行器：

1. 截取控制端中的实时手机画面；
2. 用本地模板匹配寻找指定 App 的按钮或图标；
3. 只在识别分数达到阈值时，点击控制端画面；
4. 控制端继续完成机械臂移动和物理点击。

不使用 OpenAI API，也不上传摄像头画面。运行依赖仅为本机已经具备的
Python 3.12、Pillow 和 NumPy。

## 安全边界

- 第一次只在手机桌面或测试页面运行，不要对支付、删除、发消息等页面测试。
- 机械臂必须已经按卖家教程标定成功，并且能够手动实时点击。
- 保持控制端窗口完整可见。项目按控制端实际客户区自动适配 100%～150% 缩放；若摄像区或
  底部操作栏被裁切，将在机械臂动作前失败关闭。
- 随时可以按 `Esc` 取消倒计时或动作序列。
- `find` 和默认的 `run-once` 都不会点击；必须明确加 `--execute`。
- 发生方向错误、撞边或异常动作时，立即拔掉机械臂电机电源。

## 第一次运行

先打开：

`D:\main软件发客户-20260705\main实时点击控制软件\main.exe`

确认摄像画面正常，并且手动点击画面能够驱动机械臂。

然后在 PowerShell 中进入项目：

```powershell
cd 'C:\Users\Administrator\Documents\Claude\Projects\手机自动点击器'
```

### 触控执行器单点探测

当控制端记录了正确坐标、但手机画面没有变化时，不要在真实 App 上重复点击。先在手机
浏览器打开本机安全触点页，用一次可回传的紫红靶点区分“触控笔没有接触”与“XY 偏差”。

在第一个 PowerShell 窗口启动触点页服务：

```powershell
cd .\poc
.\.venv\Scripts\python.exe touch_calibration_server.py --host 0.0.0.0 --port 8770
```

让手机与电脑处于同一局域网，并在手机浏览器打开
`http://电脑局域网IP:8770/`。随后在第二个 PowerShell 窗口先做零动作预检：

```powershell
cd .\poc
.\.venv\Scripts\python.exe run_xy_calibration.py probe
```

预检只在连续画面中定位靶点，必须显示 `physical_actions: 0`。取得针对当前靶点的新明确
确认后，才可执行最多一次真实点击：

```powershell
.\.venv\Scripts\python.exe run_xy_calibration.py probe --execute
```

结果写入 `poc\output\xy_calibration\probe_时间\`：

- `contact_detected=false`：手机没有回传触点，优先检查触控笔接触、落笔深度和执行器状态；
- `contact_detected=true`、`coordinate_passed=false`：触控有效但 XY 偏差，需要重新校准；
- 两项均为 `true`：单点执行器与当前标定通过，可回到全新 Agent 会话继续验收。

`probe` 不会修改 `tap_calibration.json`，也不会自动重试第二次点击。

### 边缘九点单步校准

`collect` 和 `validate` 已改为可恢复的单步命令。一次 CLI 调用最多执行一个物理动作：
中央全屏准备或当前 `sequence` 的一个边缘触点。省略 `--execute` 时只做零动作预检；
真实执行会先占用目标设备的共享任务 registry，再占用与网页 worker 相同的 physical lease，
并在锁内重新检查 `/api/device`、页面 heartbeat 和稳定靶点。任一门禁失败均保持 0 动作。

每一步先运行预检：

```powershell
cd .\poc
.\.venv\Scripts\python.exe run_xy_calibration.py collect --device-id device-local-01
```

核对输出中的 `page_session_id`、`next_action`、`sequence`、`target_frame` 和
`physical_actions: 0`，针对该新鲜状态取得一次明确确认后，只执行当前一步：

```powershell
.\.venv\Scripts\python.exe run_xy_calibration.py collect --device-id device-local-01 --execute
```

命令结束后必须退出并重新运行零动作预检；不得在一次确认内连续采集下一点。样本、动作意图、
前后截图和恢复状态保存在
`poc\output\xy_calibration\collect_页面session_id\`。若进程在动作后中断，下次调用只会恢复
上一步结果，不会自动重试或继续下一点。九个边缘触点齐全后，再运行一次不带 `--execute` 的
`collect` 完成零动作拟合。

拟合通过后，重置安全校准页形成新的页面 session：

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8770/api/reset
```

随后按同样方式逐步独立验证：

```powershell
.\.venv\Scripts\python.exe run_xy_calibration.py validate --device-id device-local-01
.\.venv\Scripts\python.exe run_xy_calibration.py validate --device-id device-local-01 --execute
```

九个验证触点齐全后，再运行一次不带 `--execute` 的 `validate`。只有该零动作最终化返回
`validation_complete`，标定才会设为 `validated=true`、`enabled=true`。

`tap_calibration.json` 中的仿射纠偏使用归一化画面坐标。经过独立验证的标定可以在
摄像画面做严格等比缩放时继续使用，例如 Windows 缩放使画面从 `540×960` 变为
`810×1440`；宽高缩放比例不一致时仍会安全停用纠偏，避免把裁切或变形画面误认为单纯
分辨率变化。

### 通用动作安全验收页

同一个8770服务还提供 `http://电脑局域网IP:8770/actions`。页面只包含向上滑动、系统返回、
输入并核对文字、长按和拖动五种通用底层动作，不包含任何 App 名称、账号写操作或业务流程。
每种模式在页面可见结果成立后向 `/api/action-event` 回传一个事件；电脑可通过
`/api/action-events` 读取当前会话证据，并用 `/api/action-reset` 清空模拟或上一轮事件。

浏览器模拟事件只能证明页面和回传协议可用，不能晋级设备能力。真机验收仍必须从当前真实
画面生成单一动作、执行一次、重新观察，并把动作前后画面和对应事件一起保存。

### 1. 离线自检

```powershell
python .\poc\robot_gui_poc.py selftest
```

### 2. 检查控制端窗口

```powershell
python .\poc\robot_gui_poc.py doctor
```

诊断截图会写入：

`poc\output\doctor_window.png`

诊断会同时输出窗口 DPI、卖家界面尺度和实际摄像区大小。当前电脑可把 Windows 全局缩放
设为 150%。卖家 `main.exe` 自身会在检测到非100%时主动退出，因此必须为它单独启用
“由系统执行高 DPI 缩放”（注册表兼容值 `~ DPIUNAWARE`），让它看到96 DPI；Windows其他
程序继续使用150%，本地控制器则按实际放大后的 `810×1440` 摄像区工作。改变缩放后必须
重启卖家软件和网页服务，先运行 `doctor`，再在安全触点页做一次单点验收；不得直接在真实
App 中试点。

### 3. 框选一个目标

让手机停在待识别页面，然后运行：

```powershell
python .\poc\robot_gui_poc.py select
```

在弹出的摄像画面上，用鼠标框住一个特征明显的图标或按钮。建议：

- 只框图标和少量文字；
- 不要把大面积背景框进去；
- 不要框动态数字、时间或角标。

默认保存为：

`poc\templates\target.png`

### 4. 只检查识别结果

```powershell
python .\poc\robot_gui_poc.py find
```

查看 `poc\output\last_match.png`。红框必须准确覆盖目标，匹配分数应达到
默认阈值 `0.82`。

### 5. Dry-run

```powershell
python .\poc\robot_gui_poc.py run-once
```

这一步仍然不会点击，只会打印它准备点击的位置。

### 6. 执行一次物理点击

先把机械臂速度调低，并把手放在电源插头旁，再运行：

```powershell
python .\poc\robot_gui_poc.py run-once --execute
```

程序倒计时 3 秒后点击一次。倒计时期间按 `Esc` 可取消。

实机验证需要让鼠标保持按下一小段时间，默认按住 `0.35` 秒。
更换设备后可用 `--hold 0.5` 调整，允许范围为 `0.1～2.0` 秒。

## 多步骤 PoC

分别把各页面目标保存为不同模板，例如：

```powershell
python .\poc\robot_gui_poc.py select --output .\poc\templates\app.png
python .\poc\robot_gui_poc.py select --output .\poc\templates\next.png
```

复制并修改 `sequence.example.json`，再执行：

```powershell
python .\poc\robot_gui_poc.py sequence --file .\poc\sequence.example.json
```

上面只是检查文件。确认每个模板都单独通过 `find` 后，才执行：

```powershell
python .\poc\robot_gui_poc.py sequence --file .\poc\sequence.example.json --execute
```

## 抖音自动上划 PoC

这一模式使用卖家控制端底部的“上划 → 动作”功能，不模拟摄像画面上的
鼠标拖动。后者会被卖家软件解释成单点，不能完成真正的手机滑动。

### 1. 一次性标定抖音首页

先让手机停在抖音推荐首页，关闭更新、登录、权限等弹窗，然后运行：

```powershell
python .\poc\robot_gui_poc.py douyin-calibrate
```

程序会显示摄像头画面。只框选抖音底部左下角稳定的“首页”图标和文字，
不要框视频内容、作者头像、点赞数或评论数。模板默认保存到：

`poc\templates\douyin_home.png`

### 2. Dry-run 检查

```powershell
python .\poc\robot_gui_poc.py douyin-auto --count 3 --interval 8
```

程序只检查以下内容，不驱动机械臂：

- 控制端窗口存在；
- 当前画面能够识别到抖音“首页”标记；
- 参数在安全范围内。

检查结果会保存到：

`poc\output\douyin_home_check.png`

### 3. 执行三次上划

确认手机在抖音首页、机械臂可以安全移动后运行：

```powershell
python .\poc\robot_gui_poc.py douyin-auto --count 3 --interval 8 --execute
```

执行逻辑：

1. 每次动作前重新识别“首页”标记；
2. 测量当前视频自身的自然画面变化；
3. 调用卖家软件的“上划”动作；
4. 对比动作前后画面，判断是否切换到新视频；
5. 连续两次无法确认换视频时自动停止。

运行中随时按 `Esc` 停止。每次运行的日志和前后截图会写入独立目录：

`poc\output\douyin_年月日_时分秒\`

如果出现隐私协议、手机号登录、权限申请、青少年模式或其他未知弹窗，
程序会因为识别不到“首页”标记而停止，不会尝试同意或提交。

## 抖音自动点赞闭环 PoC

这个模式解决了手动控制时的两个关键问题：

- 启动前强制把卖家软件的“连点次数”设为 `1`，避免第一次点赞后，
  第二次又把点赞取消；
- 点按后把鼠标移出摄像画面，在最多 4 秒内连续复查；识别到持续红心，
  或确认点击位置由白色明显变为红色动画时，才计为成功。

它不会把鼠标的蓝色/粉色点击高亮当成点赞成功，也不会对直播页面的未知
位置盲点。

### 1. 先运行只检测

先打开卖家控制端，让手机停在抖音推荐首页，然后双击：

`poc\01_只检测_抖音点赞PoC.cmd`

这一步不会点击或上划。检查终端应显示：

- 抖音首页标记分数达到阈值；
- 当前爱心状态为 `unliked`、`liked` 或 `missing`；
- 检查图保存到 `poc\output\douyin_like_check.png`。

检查图中黄色框应覆盖右侧爱心所在区域；绿色框应包住白色爱心。

### 2. 先执行 1 条实机测试

确认机械臂已在安全位置、抖音处于推荐页，然后双击：

`poc\02_执行1条抖音点赞测试.cmd`

程序会要求输入：

`确认执行1条点赞`

输入完全一致后才会开始。运行中按 `Esc` 可停止。

确认这一条确实变成红心且程序报告验证成功后，先双击：

`poc\03_执行3条连续点赞测试.cmd`

三条模式会要求输入：

`确认执行3条点赞`

三条模式验证自动上划和跨视频识别均正常后，再双击：

`poc\04_执行10条抖音点赞PoC.cmd`

十条模式会要求输入：

`确认执行10条点赞`

执行规则：

1. 白色爱心：单击一次，持续变红才计数；
2. 已经是红色：不重复点击，也不计入新增点赞数；
3. 推荐流直播预览：以左下方粉色“直播中”标签为最高优先级，不点击
   画面，直接上划；
4. 全屏直播：不点赞，识别并点击右上角关闭按钮，返回推荐流后再上划；
5. 其他未知页面：不点击、不再继续上划，立即保存截图并安全停止；
6. 点击后连续复查仍不能确认变红：立即停止，不继续盲点；
7. 每次上划后先进行页面分类；页面必须在至少 1.5 秒、至少 4 个清晰
   采样帧中始终保持同一类型，才允许执行点赞、关闭直播或继续上划；
8. 观察期间类型变化、画面模糊或无法分类时，不执行任何页面操作；
9. 成功达到 10 条后停在最后一条，不再上划。

每次运行的检测图和 JSONL 日志保存在：

`poc\output\douyin_like_年月日_时分秒\`

默认等待参数按当前实机设定，正常情况下 10 条约需 30～60 秒。视频加载慢
或遇到直播、已点赞页面时会更久。

命令行方式：

```powershell
# 只检测，不操作账号
python .\poc\robot_gui_poc.py douyin-like --count 10

# 真实执行，终端仍会要求本批确认
python .\poc\robot_gui_poc.py douyin-like --count 10 --execute
```

## PoC 的限制

- 依赖卖家控制端窗口及其当前布局；
- 控制端被遮挡、缩放变化或摄像头曝光变化会降低匹配分数；
- 模板匹配适合指定 App 的固定图标，不理解自然语言；
- 抖音视频本身持续变化，换视频验证属于启发式判断；所有动作都会保存
  前后截图，方便复核；
- 量产版本应直接读取摄像头并控制串口/GRBL，去掉 GUI 中间层。
