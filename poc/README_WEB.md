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

DeepSeek 使用项目当前配置读取凭据。密钥不写入报告、不输出到终端，也不通过临时
HTTP 命令传递。

### 本地回归依赖

Python 运行依赖由 `requirements-web.txt` 声明。前端浏览器合同测试的 Node 依赖由
`package.json` 与 `package-lock.json` 固定；首次运行前在 `poc` 目录执行：

```powershell
npm ci
npm test
```

`npm test` 依次运行不启动浏览器的协议测试和 Playwright 浏览器合同测试。真实模式启动脚本
使用 `agent_api_cli.py bootstrap` 校验当前服务的 OpenAPI 与认证，不再硬编码探测某个业务路由。

### 可选 Companion IME 配置与首次配对

`ROBOT_COMPANION_IME_REGISTRY` 可指向本机 Companion IME JSON 注册表；未设置时读取
`companion_ime_registry.json`。注册表只保存设备 profile、监听 host/port 和 TLS 证书/私钥路径，
不得包含一次性 token、共享 key 或其明文副本。每个启用设备只能有一个 profile、pairing_id 和
TLS bridge。最小结构为：

```json
{
  "version": "2026-08-30-companion-ime-runtime-v1",
  "devices": [{
    "profile": {
      "protocol_version": "2026-08-30-companion-ime-v1",
      "profile_id": "companion-device-local-01",
      "device_id": "device-local-01",
      "pairing_id": "pairing-device-local-01",
      "enabled": true,
      "capabilities": ["append_text", "clear_text"],
      "ack_timeout_seconds": 5.0
    },
    "bind_host": "0.0.0.0",
    "bind_port": 18766,
    "tls_certificate_path": "tls/server.crt",
    "tls_private_key_path": "tls/server.key"
  }]
}
```

可从 [`companion_ime_registry.example.json`](companion_ime_registry.example.json) 复制一份本机注册表；
同时要在所填路径准备 PEM 格式的 TLS 证书和私钥。私钥、实际注册表、配对记录及 APK 均属于本机
运行产物，不应提交到 Git。若实际注册表不放在 `poc/companion_ime_registry.json`，启动项目 API 和
执行配对 CLI 时必须使用同一个 `ROBOT_COMPANION_IME_REGISTRY` 环境变量。

首次配对时先确认项目 API 未运行，再从 `poc` 目录执行：

```powershell
python companion_ime_setup.py --device-id device-local-01 --advertise-host 192.168.1.20
```

`--advertise-host` 是手机能访问的电脑地址；注册表 `bind_host` 为 `0.0.0.0` 或 `::` 时必须提供。
命令只为指定 device 启动注册表中的同一 TLS bridge，并以第一行 JSON 显示一次 host、port、证书
SHA-256 指纹和一次性 token。把这四项填入 Android Companion IME 后，命令会继续等待，只有新的
共享 key 已先由手机加密保存到不覆盖旧配对的 pending 槽、手机发回新 key 签名的 `pair_confirm`，收到
PC 签名确认后才提升为 active，再发出签名 `pair_commit`；PC 收到这份“手机已提升”证明后才切换 active
key，并且配对记录成功写入
`output/web/state/companion_ime_pairings/` 的 Windows 当前用户 DPAPI 存储后，才输出
`pairing_succeeded` 并退出；`pair_response` 和 `pair_confirm` 本身都不算成功，超时退出码为 1。token 只在内存中存在并
只显示一次。配对 CLI 与项目
API 不能同时占用同一 TLS 端口。Android 侧安装与输入法启用步骤见
[`android/companion-ime/README.md`](../android/companion-ime/README.md)。本项目不提供 raw text 或
pairing HTTP 路由。已有有效配对时，此流程只同步同一 active key，不在普通修复中隐式旋转；如需换 key，
且只接受原 `installation_id`。重装 App、换手机或需要换 key 时，必须先由用户显式撤销两端旧配对再重新
配对，不能让两个安装实例共享同一设备 key。

APK `0.2.0` 还需在系统的“有权查看使用情况的应用/Usage Access”中启用 **Visual Agent Companion
IME**。Companion 只在内存中选出最新 `ACTIVITY_RESUMED` 的包名和事件时间（输入连接活跃时优先使用
系统校验的 `EditorInfo.packageName`），经现有配对密钥签名后发送；不上传或保存使用历史、停留时长、
屏幕内容或 UI 节点。新观察有新鲜系统包名时，它是 App 身份的唯一权威，Qwen 仍只负责 App 内页面、
控件和下一 canonical 动作；状态不可用或超过 6 秒时才明确回退到当前截图视觉身份。旧 APK 没有该帧，
但仍保留原有文字 transport，不会因项目 API 加载新协议而失效；覆盖安装 `0.2.0` 可保留现有配对。

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
`action` 必须精确绑定同 scene 的当前元素引用或方向；本地不得另建候选目录、排序或替模型改选。执行一次后必须取得下一张
新截图再调用 Qwen；`matched` 只证明刚才动作符合预期，只有当前截图上的 `finish` 才推进高层目标。
目标变化后必须按新目标重新截图，禁止预取后继目标上下文、缓存上一轮 decision 或复用旧坐标。执行前
画面已变化且尚未产生物理动作时，丢弃旧 scope 并把新截图交给 Qwen；已经执行动作后同样只把新截图交给
Qwen 决定下一步，不进入本地 corrective selector。DeepSeek 只在会话开始时生成轻量计划，不参加动作后的正常循环。
受信任包名直启只是可选的单次设备 transport，不绕过同一 canonical、scope、执行回执和动作后新观察；
App 内部操作继续完全使用当前视觉闭环和机械臂。
文字输入时，`input_structure.application_inputs[*].text` 是当前应用输入内容的唯一视觉权威；scene 输入元素只是可选页面上下文，不要求重复正文、标签、状态或几何重合，也不得否决正文。
配置 Companion IME 的设备只使用该设备唯一的文字 transport；IME ACK 只证明 transport 接受命令，
仍须通过动作后的新截图验证输入结果。输入失败不得自动改用机械键盘重输。
只有登录/身份认证和付款/资金交易要求用户确认，其余合法动作按任务授权自动执行。

输出证据默认位于 `output/web/`。报告必须区分代码测试、Mock/离线结果和真机结果。

## 当前代码分层

项目采用轻量、渐进式 DDD 模块化单体，不拆微服务。正式业务切片已经迁入 `agent/`，根目录只保留受 allowlist 约束的 interfaces/tools/evals：

- `agent/domain/` 定义轻量目标、当前 UI scene、canonical action、Controller 硬校验、瞬时 typed 文字事务及设备/会话/风险不变量；
- `agent/application/` 在会话开始时调用一次 DeepSeek，把同一次 Qwen scene/input/decision 直接绑定为一个 action 或 finish，并编排一次观察、一次执行和下一张截图；
- `agent/infrastructure/` 提供 DeepSeek/Qwen provider、场景观察、单动作执行 adapter、Robot/Replay 执行、设备独占、文件系统证据持久化、相机/动作协调、OCR/标定/方向门和设备控制器注册表；
- `web_app.py` 保留 HTTP/Pydantic 转换与组合根职责。

正式运行不存在 selector、动作后 replan、持久视觉/输入 lineage、decision cache、scene-only decision
回退或第二完成裁决。不得为目录整齐增加转发包装、兼容开关或第二套权威。
