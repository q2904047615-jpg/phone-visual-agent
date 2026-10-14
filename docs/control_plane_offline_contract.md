# Control Plane 合同与本机线上接入（2026-10-14）

本文件记录 GA 控制面的合同。领域实现位于 `poc/agent/domain/control_plane.py`，持久化实现位于 `poc/agent/infrastructure/sqlite_control_plane_repository.py`，HTTP 路由位于 `poc/web_app.py`。

## 范围

- `ControlPlaneOrganization`、`ControlPlaneTeam`、`ControlPlanePrincipal`：组织、团队、用户、角色和团队归属必须在同一组织内，角色权限只提供可审计的静态映射。
- `QuotaLimit`、`QuotaUsage`、`evaluate_quota()`：按 daily/monthly 统计运行次数、物理动作和模型 token，返回纯函数式允许/超额结果及剩余量，不修改账本。
- `ApiRequestEnvelope`、`ApiResponseEnvelope`、`ApiError`：请求 ID、幂等键、组织/用户作用域、操作名、版本化响应和可重试错误。
- `SdkContractManifest`：SDK 版本、API 版本、支持操作、错误码和写操作幂等声明。

## 边界

当前已接入本机 Web API 的组织、团队、主体、配额和审计投影，均从与 Runtime 共用的 `runtime.sqlite3` 读取。服务首次启动时只创建缺失的本地演示投影 `local-org`，使用 `INSERT OR IGNORE`，不会覆盖已有数据。只读查询要求现有本机控制令牌；写入还要求由本机令牌引导签发的持久化主体会话和主体权限。身份会话只持久化令牌哈希，不保存明文令牌。审计记录请求 ID、资源、结果和数量等元数据，过滤令牌、授权头、密码、密钥和 Cookie。

写入合同包括主体会话签发、团队创建/更新、主体创建和配额更新。资源写入必须提供 `Idempotency-Key`；同组织、同主体、同键且请求摘要相同会复放原响应，同键但请求摘要不同返回冲突。首次创建返回 `201`，幂等重放返回 `200`。写入不会进入 Runtime admission、Policy 或设备执行链。

当前身份认证是本机控制令牌引导的主体会话，不是外部 OAuth/SSO；尚未接入并发配额扣减、计费、Runtime admission、Policy 或设备执行链。因此它不会替现有风险边界新增确认，也不会把配额或权限结果变成任务 `blocked`。第二台真机、多设备调度和 GA 发布仍不在本批范围内。

## 验证

```text
python -m unittest poc.test_control_plane_contract -v
python -m unittest poc.test_control_plane_api -v
python -m unittest poc.test_local_agent_api_client -v
```

当前验证证明数据隔离、权限映射、配额计算、API/SDK 版本封套，以及本机只读路由、SQLite 持久化和审计分页可稳定回放；这不等于身份认证、写入控制面或 GA 发布已完成。
