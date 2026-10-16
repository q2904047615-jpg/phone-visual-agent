# 公共 GA 发布准备（不依赖第二台真机）

日期：2026-10-16（Asia/Shanghai）

## 当前结论

**有条件可行。** 单设备软件 GA、DNS、HTTPS、受控反向代理、GitHub OAuth 浏览器登录和服务端模型调用子项已经通过；完整公共 GA 仍受生产运维合同、第二台真机和多 worker 物理证据限制。应用继续只绑定 `127.0.0.1:8765`，公网请求由受控代理转发，来源校验只额外允许精确配置的 HTTPS origin。

公网登录与会话验收的逐项证据见 [公网 GitHub OAuth 与会话验收记录](public_ga_oauth_acceptance.md)。

## 本批锁定的公网入口

本项目暂定使用智能简历主域名下的独立子域名：

- `PUBLIC_BASE_URL=https://agent.jianlizhineng.cn`
- `GITHUB_OAUTH_REDIRECT_URI=https://agent.jianlizhineng.cn/api/control-plane/auth/github/callback`

该子域名不会覆盖现有的 `jianlizhineng.cn` 网站。DNS、证书和反向代理已经在线，服务仍保持 `127.0.0.1:8765` 本机监听。

## 本批已完成

- 单设备软件 GA 评估报告已经可以持久化，`offline_ga_status=pass`。
- 服务运行在最近一次干净 revision（以 `/api/session` 返回的 `revision` 为准），启动自检 `verified=true`，`/api/session` 为 `ready=true`。
- SQLite active release 与 Git HEAD 一致；未结算配额预留和活动 Runtime 均为 `0`，SQLite 保留离线回归产生的历史计费账本，未调用支付服务。
- 本批 Python 离线回归为 `1021` 通过、`4` 跳过；前端完整回归基线为 `64/64`。
- OpenAPI 的 billing、platform releases、platform rollback 路由已核对。
- 未启用第二台设备、未执行手机动作、未调用外部支付或推送。
- 已从当前 SQLite 沿用 1 个 GitHub owner 映射：`279507169 -> local-org/local-owner`；模型、模型地址和本机租约目录均使用代码默认值。
- 配置预检已收敛为只阻断公网入口和 OAuth App 外部配置，不再要求重复提供可推导的模型、本机路径或当前 owner 映射。
- `PUBLIC_BASE_URL=https://agent.jianlizhineng.cn` 的隔离 HTTP 回归已通过：配置的公网 origin 可访问，其他 origin 返回 `403`；该回归使用临时 SQLite，不代表公网 DNS/HTTPS 已验收。
- 公网真实验收已完成：GitHub OAuth 回调、`local-owner/local-org` 身份映射、刷新保持、退出登录和服务端会话撤销均通过；详见 [公网 GitHub OAuth 与会话验收记录](public_ga_oauth_acceptance.md)。
- 已补齐 WAL 安全的 SQLite 在线备份、完整性校验和显式恢复工具：`poc/evals/runtime_sqlite_backup.py`。备份使用 SQLite online backup API，不直接复制主库文件；恢复默认写入新路径，只有显式 `--replace` 才替换已有目标；目标仍有 `-wal/-shm` sidecar 时会拒绝覆盖，需先停写入并完成 checkpoint。
- 线上运维收尾已完成：服务重载后租约固定到 `/opt/phone-agent/state/leases`，备份固定到 `/opt/phone-agent/state/backups`；线上备份和独立恢复检查均为 `integrity=ok`。逐项证据见 [公共 GA 运维收尾验收记录](public_ga_ops_acceptance.md)。
- `DASHSCOPE_API_KEY` 已注入并完成一次真实 `qwen3-vl-plus` 文本调用；一次网络请求成功、`finish_reason=stop`，未启动手机任务。逐项证据见 [公共 GA 模型真实调用验收记录](public_ga_model_acceptance.md)。

## 公共 GA 配置合同（实现与验证状态）

在当前单机部署中，公网 HTTPS 基址、GitHub OAuth Client ID/回调/Secret、SQLite、模型密钥、租约目录和现有 owner 已在部署环境核对。当前配置预检为 `pass`；日志保留、健康检查和 SQLite 备份调度已通过，剩余是第二台真机、多 worker 等物理证据。

### 1. 公网入口与传输安全（必填）

- `PUBLIC_BASE_URL`：固定为 `https://agent.jianlizhineng.cn`；必须由 DNS 指向受控的公网反向代理，并使用有效 TLS 证书。
- DNS 所属域名和 TLS 终止位置（Cloudflare、Nginx、Caddy 或其他受控反向代理）。
- 反向代理到本机 API 的地址、端口和健康检查路径；应用仍建议只监听回环地址。
- 代理是否传递并且只信任 `Host`、`X-Forwarded-Proto` 等头部。不能让公网客户端伪造这些头部。
- 公网入口的来源策略：只允许该 HTTPS origin，不能继续使用“任意 Origin”或把本机控制令牌当作用户身份。

当前代码的 `verify_local_origin` 已支持显式 `PUBLIC_BASE_URL` 公网 HTTPS origin：未配置时继续只允许本机同源请求，配置后只额外允许该精确公网 origin，其他来源仍拒绝。本次线上验收已覆盖 HTTPS 公网来源、Secure/HttpOnly 会话 Cookie、刷新保持和退出撤销；逐项证据见 [公网 GitHub OAuth 与会话验收记录](public_ga_oauth_acceptance.md)。

### 2. GitHub OAuth（必填）

- `GITHUB_OAUTH_CLIENT_ID`：可以提供非敏感 client ID。
- `GITHUB_OAUTH_CLIENT_SECRET`：只通过部署环境的 Secret 管理器设置，不要发到聊天或提交到 Git。
- `GITHUB_OAUTH_REDIRECT_URI`：固定为 `https://agent.jianlizhineng.cn/api/control-plane/auth/github/callback`，并必须与 GitHub OAuth App 完全一致。
- `GITHUB_OAUTH_IDENTITY_MAP_FILE`：服务器上的受信映射文件绝对路径。
- 首个 owner 的 GitHub 稳定数字用户 ID，以及目标 `organization_id/user_id`。只提供 ID 和映射目标，不提供 access token。

### 3. 模型与运行时密钥（默认值已解析）

- `DASHSCOPE_API_KEY`：已由部署负责人注入服务器 Secret 环境，并通过真实模型调用验收；不保存到代码、浏览器或聊天。
- `VISION_MODEL`：已解析为默认 `qwen3-vl-plus`；若改动需重新做模型合同检查。
- `VISION_MODEL_BASE_URL`：已解析为默认 `https://dashscope.aliyuncs.com/compatible-mode/v1`。
- `PHONE_VISUAL_AGENT_LEASE_DIR`：线上已固定为 `/opt/phone-agent/state/leases`，由服务用户 `ubuntu` 持有并可写；多 worker 仍需独立并发验收。
- 证据保留策略：`ROBOT_EVIDENCE_RETENTION_SECONDS`、`ROBOT_EVIDENCE_MAX_BYTES`、`ROBOT_EVIDENCE_MAX_RUN_BYTES`。

### 4. 持久化、备份与运维（必填）

- Runtime SQLite 的持久路径，不放在临时目录或容器临时层。
- 备份目录、频率、保留周期和恢复演练负责人。
- 进程管理方式（Windows 服务、计划任务、容器或其他）以及停止/重启权限。
- 监控和日志保留位置：当前由 systemd/journald 管理，14 天、1GB journal 上限，健康检查每 5 分钟运行；日志必须继续脱敏，不保存 OAuth code、state、verifier、access token、client secret 或主体 token。

备份和恢复已在线演练通过；后续仍需按运维负责人确定的频率和保留周期设置调度。命令只输出路径、页数和 `integrity=ok`，不会输出密钥：

```powershell
python poc/evals/runtime_sqlite_backup.py verify `
  --database poc/output/web/runtime.sqlite3 `
  --required-table control_plane_meta `
  --required-table control_plane_platform_releases
python poc/evals/runtime_sqlite_backup.py backup `
  --source poc/output/web/runtime.sqlite3 `
  --destination poc/output/web/ga_readiness_20261014/runtime.sqlite3.backup `
  --required-table control_plane_meta `
  --required-table control_plane_platform_releases
python poc/evals/runtime_sqlite_backup.py restore `
  --backup poc/output/web/ga_readiness_20261014/runtime.sqlite3.backup `
  --destination poc/output/web/ga_readiness_20261014/runtime.sqlite3.restore-check `
  --required-table control_plane_meta `
  --required-table control_plane_platform_releases
```

恢复生产库前仍需由部署负责人确定持久备份目录、保留周期和停机/切换窗口；工具不会自动覆盖当前运行库。

### 5. 当前设备连接配置（单设备即可）

- 保持 `device-local-01` 为唯一启用设备和 `local-owner` 的归属。
- 如需文字输入，再提供服务器上的 `ROBOT_ADB_KEYBOARD_REGISTRY`；如需可信直启，再提供 `ROBOT_APP_PACKAGE_REGISTRY`。两者都不能把 serial、包名或 Shell 参数交给网页或模型。
- 第二台真机的 serial、校准和机位配置本阶段不需要提供。

### 6. 可选商业配置（当前可以不提供）

- 云账单/支付供应商、价格表、税务和结算账户。
- 外部邮件、Webhook 或移动推送供应商。
- 第二个真实 GitHub 账号、公开邀请/自助注册和多 API worker 的公共并发验收账号。

## 配置完成后的公共 GA 门禁

1. 在部署环境运行 `python poc/evals/check_public_ga_config.py`，确认所有必填项存在且不打印 secret；当前配置缺失时应返回阻断状态。
2. 通过 HTTPS 反向代理完成 GitHub OAuth 登录、退出、过期和回调重放拒绝检查。
3. 核对 Secure/HttpOnly/SameSite Cookie、同源策略、跨组织隔离和无主体访问拒绝。
4. 用干净 revision 启动并核对 `/api/session`、OpenAPI、SQLite active release 和备份恢复点；备份恢复点使用上述在线备份工具验证。
5. 在受控浏览器环境复跑前端 `64/64`，复跑 Python 全量离线回归 `1021`/`4 skipped`。
6. 用平台回滚合同做一次无活动任务的演练；不自动 checkout、不覆盖用户数据。

## 现在仍需要你或部署环境提供的配置

公网 DNS、HTTPS、反向代理、GitHub OAuth、单机持久化备份和模型真实调用已完成并通过验收。当前剩余：

1. 如需外部告警，再接入通知供应商；当前 systemd/journald 本机监控与备份调度已完成并通过演练。
3. 具备资源后再验收第二台真机、多 worker 并发、第二个真实 GitHub 账号、真实账单/支付和外部通知。

以下内容目前不需要提供：第二台真机、第二个 GitHub 账号、根域名替换、支付账号或外部通知账号。

## 可以暂缓但不能冒充已验收

以下事项可以延后到具备相应资源或进入商业化公共 GA 前：

- 第二台真机及真实多设备并行动作证据。
- 云账单、支付结算和生产发布平台切换。
- 外部通知渠道。
- 第二个真实 GitHub 账号、多 worker 并发、邀请和自助注册。

这些事项不阻断当前单设备软件发布；但在宣称完整公共 GA 前仍要分别验收。普通受限沙箱的 Edge `spawn EPERM` 只是测试运行器配置问题，受控浏览器环境已经通过；公共 CI 仍需使用允许创建 Edge 子进程的 runner。

## 当前仍未授权的动作

本准备阶段不接入第二台真机、不执行手机动作、不操作卖家 `main.exe`、不发送支付、不推送 GitHub，也不要求在聊天中提交任何 secret/token。
