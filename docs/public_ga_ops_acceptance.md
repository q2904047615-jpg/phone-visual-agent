# 公共 GA 运维收尾验收记录

日期：2026-10-16（Asia/Shanghai）

## 结论

**已验证可行（单机持久化运维子项）；完整公共 GA 仍有条件可行。**

在不接入第二台真机、不执行手机动作和不调用支付的前提下，已将线上租约从临时目录切换到持久可写路径，并完成 SQLite 在线备份、完整性校验和独立恢复演练。服务重载后保持正常运行。

## 线上证据

- 服务用户：`ubuntu`；`phone-agent.service` 重载后为 `active`，`/api/session` 返回 `ready=true`、`mock=false`、`identity_required=true`。
- 持久租约目录：`/opt/phone-agent/state/leases`，属主 `ubuntu:ubuntu`，权限 `750`，服务用户可写。
- 备份目录：`/opt/phone-agent/state/backups`，属主 `ubuntu:ubuntu`，权限 `750`。
- SQLite 在线备份：`/opt/phone-agent/state/backups/runtime.sqlite3.backup`，`integrity=ok`，`page_count=70`，包含 `control_plane_meta` 和 `control_plane_platform_releases`。
- 独立恢复检查：`/opt/phone-agent/state/backups/runtime.sqlite3.restore-check`，`integrity=ok`，未覆盖运行中的数据库，`replaced=false`。
- 环境文件在修改前已保存为带时间戳的 `phone-agent.env.before-persistent-state-*` 备份；未输出或保存任何 Secret 到本地台账。

## 仍未完成

- `DASHSCOPE_API_KEY` 尚未注入，模型真实调用和完整公共 GA 预检仍为 `blocked`。
- 日志保留周期、监控告警和定期备份调度尚未形成正式运维负责人合同。
- 第二台真机、多 worker 物理并行、第二个真实 GitHub 账号、真实账单/支付和外部通知仍未验收。

以上缺口不否定持久化运维子项，也不把单机证据写成完整公共 GA。
