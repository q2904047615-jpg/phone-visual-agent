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

每次物理动作前必须按当前实际活动子目标取得新观察并核对设备身份、fingerprint、canonical candidate、
scope 和一次性执行权；动作后重新观察并验证。活动子目标不变时，动作后观察可以选择同一子目标的下一微动作；活动子目标一旦变化，必须进入 `needs_reobservation`，按新子目标重新截图并发起一次新的单步 Qwen 观察，禁止预取或复用后继子目标上下文。普通只读/导航动作无预期变化时，必须用新观察、新 candidate、新几何和新 scope 自动纠正一次，禁止直接复用旧坐标；纠正仍失败以及输入、外部效果、登录、付款动作失败时立即停止。
只有登录/身份认证和付款/资金交易要求用户确认，其余合法动作按任务授权自动执行。

输出证据默认位于 `output/web/`。报告必须区分代码测试、Mock/离线结果和真机结果。

## 当前代码分层

项目采用轻量、渐进式 DDD 模块化单体，不拆微服务。正式业务切片已经迁入 `agent/`，根目录只保留受 allowlist 约束的 interfaces/tools/evals：

- `agent/domain/` 定义任务图、会话、设备执行、会话证据、视觉场景、语义动作、确定性文字事务、canonical Controller/选择回执、一次性确认作用域和已验证 App 表面血缘合同；
- `agent/application/` 负责任务图规划、Qwen 单步决策、通用 Agent 编排、运行会话聚合以及开始、确认、重观察、自动推进、暂停和取消用例；
- `agent/infrastructure/` 提供 DeepSeek/Qwen provider、场景观察、单动作执行 adapter、Robot/Replay 执行、设备独占、文件系统证据持久化、相机/动作协调、OCR/标定/方向门和设备控制器注册表；
- `web_app.py` 保留 HTTP/Pydantic 转换与组合根职责。

不得为目录整齐增加转发包装、兼容开关或第二套权威。后续新增正式业务能力直接进入对应 DDD 层。
