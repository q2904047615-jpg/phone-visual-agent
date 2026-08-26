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

每次物理动作前必须取得新观察并核对设备身份、fingerprint、canonical candidate、
scope 和一次性执行权；动作后重新观察并验证。普通只读/导航动作无预期变化时，必须用新观察、新 candidate、新几何和新 scope 自动纠正一次，禁止直接复用旧坐标；纠正仍失败以及输入、外部效果、登录、付款动作失败时立即停止。
只有登录/身份认证和付款/资金交易要求用户确认，其余合法动作按任务授权自动执行。

输出证据默认位于 `output/web/`。报告必须区分代码测试、Mock/离线结果和真机结果。

## 当前代码分层

项目采用轻量、渐进式 DDD 模块化单体，不拆微服务，也不一次性重写。当前已迁入 `agent/` 的正式切片为：

- `agent/domain/` 定义会话、设备执行、会话证据、视觉场景、语义动作、canonical 选择回执、一次性确认作用域和已验证 App 表面血缘合同；
- `agent/application/` 负责运行会话聚合以及开始、确认、重观察、自动推进、暂停和取消用例；
- `agent/infrastructure/` 提供唯一的线程安全进程内会话仓储、Robot/Replay 执行、设备独占、文件系统证据持久化、每设备相机/动作协调资源和设备控制器注册表；
- `web_app.py` 保留 HTTP/Pydantic 转换与组合根职责。

尚未迁移的 DeepSeek、Qwen、canonical action 目录、Controller 和 typed input lineage 仍是正式实现；不得为目录整齐而增加转发包装或第二套权威。每个后续批次只迁移一个正在运行的业务切片，并在新入口接管后删除旧调用。
