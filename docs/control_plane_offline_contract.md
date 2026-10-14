# Control Plane 离线合同（2026-10-14）

本文件记录 GA 控制面的第一批离线合同。实现位于 `poc/agent/domain/control_plane.py`，测试位于 `poc/test_control_plane_contract.py`。

## 范围

- `ControlPlaneOrganization`、`ControlPlaneTeam`、`ControlPlanePrincipal`：组织、团队、用户、角色和团队归属必须在同一组织内，角色权限只提供可审计的静态映射。
- `QuotaLimit`、`QuotaUsage`、`evaluate_quota()`：按 daily/monthly 统计运行次数、物理动作和模型 token，返回纯函数式允许/超额结果及剩余量，不修改账本。
- `ApiRequestEnvelope`、`ApiResponseEnvelope`、`ApiError`：请求 ID、幂等键、组织/用户作用域、操作名、版本化响应和可重试错误。
- `SdkContractManifest`：SDK 版本、API 版本、支持操作、错误码和写操作幂等声明。

## 边界

这些类型目前只用于离线合同和评测，尚未接入 `web_app`、Runtime、Policy 或设备执行链。因此它们不会替现有风险边界新增确认，也不会把配额或权限结果偷偷变成任务 `blocked`。接入运行时前需要单独定义数据存储、身份认证、审计、并发扣减和用户批准的产品策略。

## 验证

```text
python -m unittest poc.test_control_plane_contract -v
```

当前目标是先证明数据隔离、权限映射、配额计算和 API/SDK 版本封套可稳定回放；这不等于 GA 控制面已上线。
