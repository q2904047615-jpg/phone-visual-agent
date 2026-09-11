# 通用手机视觉 Agent 网页控制台

网页控制台是 Universal Agent 的当前服务入口，不包含固定 App 工作流、后台兼容
worker、旧任务队列或旧语义执行接口。

## 启动与配置

用户自行双击根目录 `启动机械臂网页控制台.cmd`。Agent 不得启动、停止或重启卖家
`main.exe`。

视觉模型配置：

- `DASHSCOPE_API_KEY`：访问凭据；
- `VISION_MODEL`：可替换视觉模型，默认 `qwen3.7-plus`；
- `VISION_MODEL_BASE_URL`：可选服务地址。

正式运行只要求 Qwen 配置；DeepSeek 不再是启动依赖。密钥不写入报告、不输出到终端，也不通过临时
HTTP 命令传递。

### 本地回归依赖

网页层依赖由 `requirements-web.txt` 声明；机械/图像层还依赖本机已有的 Python 图像及 Windows 环境，不应将该文件当作从空系统安装全部硬件依赖的保证。前端浏览器合同测试的 Node 依赖由
`package.json` 与 `package-lock.json` 固定；首次运行前在 `poc` 目录执行：

```powershell
npm ci
npm test
```

`npm test` 依次运行不启动浏览器的协议测试和 Playwright 浏览器合同测试。真实模式启动脚本
使用 `agent_api_cli.py bootstrap` 校验当前服务的 OpenAPI 与认证，不再硬编码探测某个业务路由。

### ADB Keyboard 文字 transport

`ROBOT_ADB_KEYBOARD_REGISTRY` 可指向本机 ADB Keyboard JSON 注册表；未设置时读取
`adb_keyboard_registry.json`。每个启用设备只能登记一个固定 profile、一个固定 ADB 可执行文件和
一个固定 serial。最小结构为：

```json
{
  "version": "2026-09-02-adb-keyboard-runtime-v1",
  "devices": [{
    "profile": {
      "protocol_version": "2026-09-02-adb-keyboard-v1",
      "profile_id": "adb-keyboard-device-local-01",
      "device_id": "device-local-01",
      "adb_serial": "REPLACE_WITH_ADB_SERIAL",
      "enabled": true,
      "capabilities": ["append_text", "clear_text"],
      "command_timeout_seconds": 10.0
    },
    "adb_executable": "C:\\path\\to\\platform-tools\\adb.exe"
  }]
}
```

可从 [`adb_keyboard_registry.example.json`](adb_keyboard_registry.example.json) 复制本机注册表。ADB
Keyboard APK 必须先安装、在系统输入法设置中启用，并选为当前输入法；运行时只读核对这三个条件，
任一不成立都在广播前以 0 动作停止。输入只执行一次固定 `ADB_INPUT_B64 --es msg <UTF-8 Base64>`
广播，清空只执行一次固定 `ADB_CLEAR_TEXT` 广播。两者互不隐式组合，失败不重试、不回退机械键盘。
广播完成只算 transport 回执；正文和高层完成仍由动作后新截图中的 Qwen 当前帧证据判断。

Visual Agent Companion IME、TLS bridge、配对、editor session 和旧注册表自 2026-09-02 起永久退役，
不再由项目 API 导入、启动或作为回退路径。ADB 可执行文件、serial、IME ID 和广播 action 均由本地
可信代码与注册表固定，Qwen、网页和任务文本不能提供任意 ADB/Shell/Intent 参数。

可选 App 包名直启配置：

- `ROBOT_APP_PACKAGE_REGISTRY`：指向本机可信的 JSON 注册表；未设置时读取
  `app_package_registry.json`；
- 注册表按 `device_id` 绑定固定 `adb_executable`、`adb_serial`、App alias、opaque
  `launch_ref` 与 Android package；只有 `enabled=true`、ADB 文件存在、serial 非空且当前
  App 唯一匹配时才开放 `launch_app`；
- 用户目标、Qwen 和网页请求都不能传入 package、Shell、Intent、组件或任意命令。能力未配置或目标未登记时，
  不生成直启候选，仍按当前截图寻找 App 图标；
- 仓库默认注册表保持关闭。启用前应由设备维护者核对 ADB serial 和包名；可信 `launch_app` 只验证
  注册表绑定和 transport 回执。直启后必须重新截图，页面、下一动作和 `finish` 仍只由下一次 Qwen 响应
  决定。系统包名不在 Qwen `finish` 后形成第二完成裁决，也不能把 ADB 返回码当作高层任务成功。

## 当前公开能力

- `GET /api/session`：读取本地控制会话元数据；
- `GET /api/apps`：读取通用 Agent 能力说明；
- `GET /api/device`：读取设备、控制器、相机、占用和协议状态；
- `/api/agent/generic-scene/*`：四帧只读场景观察；
- `/api/agent/capability-acceptance/*`：通用动作能力验收；
- `/api/agent/generic-supervised/*`：一次一动作的通用任务闭环；
- `POST /api/stop`：请求当前项目控制器停止；
- `/api/preview.jpg`、`/api/preview.mjpg`：只读预览。

完整方法、路径和请求体必须以运行中服务的 `/openapi.json` 为准。本地命令行只使用
`local_agent_api_client.py` 或 `agent_api_cli.py`，不得手写通用 Agent HTTP 请求。

## 执行合同

每次 Qwen 调用只处理当前新截图，并在同一响应中发布 scene 与一个推进当前目标的 `action`，或以同帧
证据报告 `finish`。当前没有最终控件、需要中间导航或单步不能完成高层目标时仍必须选择推进动作；
Qwen 不再拥有普通语义停止状态。
点按 `action` 只绑定同帧 decision.target/tap_point；非点按绑定该动作需要的当前元素、字段或方向。本地不得另建候选目录、排序或替模型改选。执行一次后必须取得下一张
新截图再调用 Qwen；`matched` 只证明刚才动作符合预期，只有当前截图上的 `finish` 才推进高层目标。
目标变化后必须按新目标重新截图，禁止预取后继目标上下文、缓存上一轮 decision 或复用旧坐标。执行前
画面已变化且尚未产生物理动作时，丢弃旧 scope 并把新截图交给 Qwen；已经执行动作后同样只把新截图交给
Qwen 决定下一步，不进入本地 corrective selector。Qwen 同时接收整任务和实际执行历史，不再调用 DeepSeek 或固定子目标。
受信任包名直启只是可选的单次设备 transport，不绕过同一 canonical、scope、执行回执和动作后新观察；
App 内部操作继续完全使用当前视觉闭环和机械臂。
文字输入时，`input_structure.application_inputs[*].text` 是当前应用输入内容的唯一视觉权威；scene 输入元素只是可选页面上下文，不要求重复正文、标签、状态或几何重合，也不得否决正文。
配置 ADB Keyboard 的设备只使用该设备唯一的非机械文字 transport；广播回执只证明 transport 接受命令，
仍须通过动作后的新截图验证输入结果。输入失败不得自动改用机械键盘重输。
只有登录/身份认证和付款/资金交易要求用户确认，其余合法动作按任务授权自动执行。

输出证据默认位于 `output/web/`。报告必须区分代码测试、Mock/离线结果和真机结果。

## 当前代码分层

项目采用轻量、渐进式 DDD 模块化单体，不拆微服务。正式业务切片已经迁入 `agent/`，根目录只保留受 allowlist 约束的 interfaces/tools/evals：

- `agent/domain/` 定义轻量目标、当前 UI scene、canonical action、Controller 硬校验、瞬时 typed 文字事务及设备/会话/风险不变量；
- `agent/application/` 管理整任务及实际执行历史，把同一次 Qwen scene/input/decision 直接绑定为一个 action 或 finish，并编排一次观察、一次执行和下一张截图；
- `agent/infrastructure/` 提供 Qwen provider、场景观察、单动作执行 adapter、Robot/Replay 执行、设备独占、文件系统证据持久化、相机/动作协调、标定/方向门和设备控制器注册表；
- `web_app.py` 保留 HTTP/Pydantic 转换与组合根职责。

正式运行不存在 selector、动作后 replan、持久视觉/输入 lineage、decision cache、scene-only decision
回退或第二完成裁决。不得为目录整齐增加转发包装、兼容开关或第二套权威。
