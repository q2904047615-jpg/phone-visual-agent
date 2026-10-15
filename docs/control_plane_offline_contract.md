# Control Plane 合同与本机线上接入（2026-10-14）

本文件记录 GA 控制面的合同。领域实现位于 `poc/agent/domain/control_plane.py`，持久化实现位于 `poc/agent/infrastructure/sqlite_control_plane_repository.py`，HTTP 路由位于 `poc/web_app.py`。

## 范围

- `ControlPlaneOrganization`、`ControlPlaneTeam`、`ControlPlanePrincipal`：组织、团队、用户、角色和团队归属必须在同一组织内，角色权限只提供可审计的静态映射。
- `QuotaLimit`、`QuotaUsage`、`evaluate_quota()`：按 daily/monthly 统计运行次数、物理动作和模型 token，返回纯函数式允许/超额结果及剩余量。
- `QuotaReservation`、Runtime quota bridge：Runtime 启动时原子预留 Run 和声明的物理动作预算；终态按实际物理动作和模型 token 结算，失败前退出可释放预留，重复结算只生成一条账单。
- `BillingRecord`：把现有 Qwen 用量账本中的 token、模型、定价版本和 CNY 微单位估算写入 SQLite；它是审计/成本投影，不是支付或外部云账单。
- `PlatformRollbackPlan`：发布版本账本、活动任务阻断、目标版本已加载校验和显式回滚提交；回滚合同不会自动 checkout 或覆盖 Git 工作树。
- `ApiRequestEnvelope`、`ApiResponseEnvelope`、`ApiError`：请求 ID、幂等键、组织/用户作用域、操作名、版本化响应和可重试错误。
- `SdkContractManifest`：SDK 版本、API 版本、支持操作、错误码和写操作幂等声明。

## 边界

当前已接入本机 Web API 的组织、团队、主体、配额和审计投影，均从与 Runtime 共用的 `runtime.sqlite3` 读取。服务首次启动时只创建缺失的本地演示投影 `local-org`，使用 `INSERT OR IGNORE`，不会覆盖已有数据。只读查询要求现有本机控制令牌；写入还要求由本机令牌引导签发的持久化主体会话和主体权限。身份会话只持久化令牌哈希，不保存明文令牌。审计记录请求 ID、资源、结果和数量等元数据，过滤令牌、授权头、密码、密钥和 Cookie。

写入合同包括主体会话签发、团队创建/更新、主体创建、配额更新和会话撤销。资源写入必须提供 `Idempotency-Key`；同组织、同主体、同键且请求摘要相同会复放原响应，同键但请求摘要不同返回冲突。首次创建返回 `201`，幂等重放返回 `200`。会话撤销支持撤销自己的当前会话；撤销其他会话需要 `identity.manage`。

外部身份接口为 `ExternalIdentityProvider` 端口和 `POST /api/control-plane/auth/external-sessions`。服务不会把任意断言当作身份，只有显式注册的 provider 完成断言验证并返回 `ExternalIdentity` 后才会签发本地主体会话。当前已接入 GitHub OAuth provider：服务端使用固定的 `https://api.github.com/applications/{client_id}/token` Check a token 接口验证 access token，并核对返回的应用 `client_id`、稳定数字用户 ID 和可选过期时间；GitHub 用户 ID 只通过受信本地映射文件映射到已存在的 `organization_id/user_id`，不会接受客户端提交的组织、角色或权限字段。未映射用户返回 `external_identity_mapping_not_found`，GitHub 无效令牌返回 `external_identity_rejected`，上游不可达或 5xx 返回 `external_provider_unavailable`。权限审计记录身份缺失、身份无效、组织不匹配、权限拒绝、权限通过、会话撤销、映射缺失和外部断言结果，且不记录断言正文、access token 或 client secret。

GitHub provider 只有在以下环境变量同时存在时才启用：`GITHUB_OAUTH_CLIENT_ID`、`GITHUB_OAUTH_CLIENT_SECRET`、`GITHUB_OAUTH_IDENTITY_MAP_FILE`；`GITHUB_OAUTH_REDIRECT_URI` 可选，默认是 `http://127.0.0.1:8765/api/control-plane/auth/github/callback`。映射文件使用 `{ "42": { "organization_id": "local-org", "user_id": "local-owner" } }`（也可包在 `mappings` 对象下）；它是部署受信配置，不由外部用户写入。缺少或部分配置时 provider fail-closed，并保持本机 Runtime 可启动。

浏览器授权码流程通过受本机控制令牌保护的 `GET /api/control-plane/auth/github/start` 启动，服务生成一次性 state、服务端 PKCE verifier 和 `S256` challenge，并只把 state 的哈希及回调地址写入 SQLite；verifier 只保留在当前 API 进程的短期内存中，进程重载会使未完成登录重新开始。GitHub 回调到 `GET /api/control-plane/auth/github/callback` 后，服务原子消费 state，向固定的 GitHub token endpoint 换取 code，再复用 Check a token 和本地映射签发主体会话。回调不会返回 GitHub access token，只返回本地 `principal_token` 会话封套；state 重放、过期、授权拒绝和未映射主体都有固定错误和审计结果。

当前完成的是浏览器授权码合同、PKCE/state 防重放、身份映射和离线回归；没有提交真实 GitHub client secret/access token，因此线上 GitHub 登录仍标记为“待凭据实测”，不能把 MockTransport 或离线 HTTP 合同当成真实 OAuth 验收。

当前身份认证仍支持本机控制令牌引导的主体会话；外部 provider 只负责验证外部身份和映射主体，不替代本地角色、权限或会话撤销。Runtime admission 已接入 daily 配额的原子预留/释放/终态结算；计费写入 CNY 微单位估算账本，不调用支付或外部计费服务。平台回滚通过 `/platform/releases` 和 `/platform/rollback` 提供可审计计划，只有目标代码已由受控重载加载且无活动任务时才允许提交状态。第二台真机、多设备并行动作和真实支付/云账单仍不在本批范围内。

## 验证

```text
python -m unittest poc.test_control_plane_contract -v
python -m unittest poc.test_control_plane_api -v
python -m unittest poc.test_runtime_quota_billing -v
python -m unittest poc.test_github_oauth_provider -v
python -m unittest poc.test_local_agent_api_client -v
```

当前验证证明数据隔离、权限映射、配额计算、Runtime 预留/结算/释放幂等、计费账本、平台回滚计划和 API/SDK 版本封套，以及 GitHub 授权 URL、PKCE code exchange、state 原子消费/重放拒绝、Check a token 请求、应用核对、过期/错误 token、未映射主体、上游故障、配置 fail-closed 和外部会话签发/审计可稳定回放；这不等于第二台真机、多设备并行或真实支付/云账单验收已完成。
