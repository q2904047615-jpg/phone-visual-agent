# 公共 GA 发布准备（不依赖第二台真机）

日期：2026-10-16（Asia/Shanghai）

## 当前结论

**有条件可行。** 单设备软件 GA 的代码、离线回归、干净 revision、报告和本机服务状态已经通过；公共互联网 GA 还不能直接上线。当前服务仍绑定 `127.0.0.1:8765`，来源校验只接受 `127.0.0.1`/`localhost`，这是本机安全边界，不应通过端口映射或临时代理绕过。

公共上线需要先确定 HTTPS 公网入口、反向代理信任边界和 GitHub OAuth 回调地址，再实施对应的受信来源配置和部署验收。没有这些配置时，保持本机监听是正确状态。

## 本批已完成

- 单设备软件 GA 评估报告已经可以持久化，`offline_ga_status=pass`。
- 服务运行在最近一次干净 revision（以 `/api/session` 返回的 `revision` 为准），启动自检 `verified=true`，`/api/session` 为 `ready=true`。
- SQLite active release 与 Git HEAD 一致；未结算配额预留和活动 Runtime 均为 `0`，SQLite 保留离线回归产生的历史计费账本，未调用支付服务。
- 本批 Python 离线回归为 `1019` 通过、`4` 跳过；前端完整回归基线为 `64/64`。
- OpenAPI 的 billing、platform releases、platform rollback 路由已核对。
- 未启用第二台设备、未执行手机动作、未调用外部支付或推送。
- 已从当前 SQLite 沿用 1 个 GitHub owner 映射：`279507169 -> local-org/local-owner`；模型、模型地址和本机租约目录均使用代码默认值。
- 配置预检已收敛为只阻断公网入口和 OAuth App 外部配置，不再要求重复提供可推导的模型、本机路径或当前 owner 映射。

## 公共 GA 必须先确定的配置

在当前单机部署中，模型、SQLite、租约目录和现有 owner 已能从代码/数据库解析，不需要再次提供。公网接入前实际还缺四项：公网 HTTPS 基址、GitHub OAuth Client ID、与该基址同源的回调地址，以及通过 Secret 管理器注入的 GitHub OAuth Client Secret。

### 1. 公网入口与传输安全（必填）

- `PUBLIC_BASE_URL`：完整 HTTPS 地址，例如 `https://agent.example.com`。
- DNS 所属域名和 TLS 终止位置（Cloudflare、Nginx、Caddy 或其他受控反向代理）。
- 反向代理到本机 API 的地址、端口和健康检查路径；应用仍建议只监听回环地址。
- 代理是否传递并且只信任 `Host`、`X-Forwarded-Proto` 等头部。不能让公网客户端伪造这些头部。
- 公网入口的来源策略：只允许该 HTTPS origin，不能继续使用“任意 Origin”或把本机控制令牌当作用户身份。

当前代码的 `start_verified.py` 和 `verify_local_origin` 仍是本机验收合同；公共部署前必须针对上述域名完成一次受控来源配置、Cookie `Secure` 行为和跨站拒绝验收。

### 2. GitHub OAuth（必填）

- `GITHUB_OAUTH_CLIENT_ID`：可以提供非敏感 client ID。
- `GITHUB_OAUTH_CLIENT_SECRET`：只通过部署环境的 Secret 管理器设置，不要发到聊天或提交到 Git。
- `GITHUB_OAUTH_REDIRECT_URI`：必须与 GitHub OAuth App 完全一致，例如 `https://agent.example.com/api/control-plane/auth/github/callback`。
- `GITHUB_OAUTH_IDENTITY_MAP_FILE`：服务器上的受信映射文件绝对路径。
- 首个 owner 的 GitHub 稳定数字用户 ID，以及目标 `organization_id/user_id`。只提供 ID 和映射目标，不提供 access token。

### 3. 模型与运行时密钥（默认值已解析）

- `DASHSCOPE_API_KEY`：只注入 Secret 管理器，不发送给我。
- `VISION_MODEL`：已解析为默认 `qwen3-vl-plus`；若改动需重新做模型合同检查。
- `VISION_MODEL_BASE_URL`：已解析为默认 `https://dashscope.aliyuncs.com/compatible-mode/v1`。
- `PHONE_VISUAL_AGENT_LEASE_DIR`：当前使用代码默认临时租约目录；正式公网多 worker 部署时再切换为持久可写路径。
- 证据保留策略：`ROBOT_EVIDENCE_RETENTION_SECONDS`、`ROBOT_EVIDENCE_MAX_BYTES`、`ROBOT_EVIDENCE_MAX_RUN_BYTES`。

### 4. 持久化、备份与运维（必填）

- Runtime SQLite 的持久路径，不放在临时目录或容器临时层。
- 备份目录、频率、保留周期和恢复演练负责人。
- 进程管理方式（Windows 服务、计划任务、容器或其他）以及停止/重启权限。
- 监控和日志保留位置；日志必须继续脱敏，不保存 OAuth code、state、verifier、access token、client secret 或主体 token。

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
4. 用干净 revision 启动并核对 `/api/session`、OpenAPI、SQLite active release 和备份恢复点。
5. 在受控浏览器环境复跑前端 `64/64`，复跑 Python 全量离线回归 `1014`/`4 skipped`。
6. 用平台回滚合同做一次无活动任务的演练；不自动 checkout、不覆盖用户数据。

## 可以暂缓但不能冒充已验收

以下事项可以延后到具备相应资源或进入商业化公共 GA 前：

- 第二台真机及真实多设备并行动作证据。
- 云账单、支付结算和生产发布平台切换。
- 外部通知渠道。
- 第二个真实 GitHub 账号、多 worker 并发、邀请和自助注册。

这些事项不阻断当前单设备软件发布；但在宣称完整公共 GA 前仍要分别验收。普通受限沙箱的 Edge `spawn EPERM` 只是测试运行器配置问题，受控浏览器环境已经通过；公共 CI 仍需使用允许创建 Edge 子进程的 runner。

## 当前仍未授权的动作

本准备阶段不接入第二台真机、不执行手机动作、不操作卖家 `main.exe`、不发送支付、不推送 GitHub，也不要求在聊天中提交任何 secret/token。
