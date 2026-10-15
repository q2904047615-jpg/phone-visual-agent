# GA 单设备离线准备验收

日期：2026-10-16（Asia/Shanghai）

## 结论

**已验证可行（不依赖第二台真机的 GA 软件范围）。** 当前评估器把 GA 控制面、Runtime 配额/计费、API/SDK、GitHub OAuth 身份映射和平台回滚作为独立离线门；唯一真机仍使虚拟双设备和物理多设备门保持阻断，不把 Mock 或静态注册表当成第二台真机证据。

## 本批完成

- `GAReadinessReport` 新增 `offline_ga_status`，把“单设备软件 GA 已通过”和“多设备物理证据未完成”分开表达。
- 用户/团队/角色权限、主体会话、会话撤销、资源隔离和 Runtime 主体上下文标为 `pass`。
- Runtime 配额原子预留/释放/终态结算、计费账本和本地 CNY 估算标为 `pass`。
- API/SDK 版本、错误封套、幂等、GitHub OAuth 映射和本机读写路由标为 `pass`。
- 平台发布版本账本、活动任务阻断、已加载目标版本校验、Owner 确认和幂等回滚标为 `pass`。
- 真实设备注册表仍只启用 `device-local-01`；没有启用未校准设备，没有修改生产注册表。

## 验证证据

- `poc/test_ga_readiness.py`：3/3 通过。
- 完整 Python 离线回归基线：1014 通过、4 skipped；本轮前端 npm 完整回归：64/64 通过（19 个单元 + 45 个浏览器合同）。
- 只读评估报告：`poc/output/web/ga_readiness_20261014/report.json`，其中 `offline_ga_status=pass`，`physical_multi_device_status=blocked_registry_not_ready`。
- 服务当前为 PID `20412`、监听 `127.0.0.1:8765`，`/api/session` 返回 `ready=true`；新 billing、platform releases、platform rollback 路由已在 OpenAPI 中登记。
- 本批没有手机动作、第二台真机、付费模型调用、外部支付或 GitHub 推送。

## 尚未完成

- 第二台真机的独立校准、并行运行、同设备互斥和动作后业务证据。
- 真实云厂商账单、支付结算和生产发布平台的版本切换。
- 浏览器合同需要创建 Edge 子进程；普通受限沙箱仍会返回 `spawn EPERM`，项目脚本已消除 Node 测试 worker 的额外进程，最终 64 项在允许浏览器子进程的受控环境完成。

## 回滚

本批代码提交为 `8b5ec5d`，可精确回退评估器和对应测试；报告和文档可单独回退。不会回滚设备注册表，不覆盖 Runtime SQLite，也不会恢复旧的 Qwen 终态权威。
