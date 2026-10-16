# 公共 GA 运维收尾验收记录

日期：2026-10-16（Asia/Shanghai）

## 结论

**已验证可行（单机持久化、日志保留与监控调度子项）；完整公共 GA 仍有条件可行。**

在不接入第二台真机、不执行手机动作和不调用支付的前提下，已将线上租约从临时目录切换到持久可写路径，并完成 SQLite 在线备份、完整性校验和独立恢复演练。服务重载后保持正常运行。

## 线上证据

- 服务用户：`ubuntu`；`phone-agent.service` 重载后为 `active`，`/api/session` 返回 `ready=true`、`mock=false`、`identity_required=true`。
- 持久租约目录：`/opt/phone-agent/state/leases`，属主 `ubuntu:ubuntu`，权限 `750`，服务用户可写。
- 备份目录：`/opt/phone-agent/state/backups`，属主 `ubuntu:ubuntu`，权限 `750`。
- SQLite 在线备份：`/opt/phone-agent/state/backups/runtime.sqlite3.backup`，`integrity=ok`，`page_count=70`，包含 `control_plane_meta` 和 `control_plane_platform_releases`。
- 独立恢复检查：`/opt/phone-agent/state/backups/runtime.sqlite3.restore-check`，`integrity=ok`，未覆盖运行中的数据库，`replaced=false`。
- 日志保留：`/etc/systemd/journald.conf.d/phone-agent-retention.conf` 已启用 14 天保留、1GB 持久 journal 上限、200MB 运行时上限和 2GB 最低空闲空间；应用后 journal 当前约 86.7MB。
- 健康监控：`phone-agent-healthcheck.timer` 已启用，每 5 分钟检查 `/api/session` 的 `ready=true` 且无 `startup_error`；立即演练 `Result=success`。
- 备份调度：`phone-agent-backup.timer` 已启用，每天 02:15 执行并随机延迟最多 5 分钟；备份脚本保留 14 天并清理非活动 sidecar；立即演练 `Result=success`，最新备份 `runtime-20261016T050122Z.sqlite3` 的 `integrity=ok`、`page_count=70`。
- 环境文件在修改前已保存为带时间戳的 `phone-agent.env.before-persistent-state-*` 备份；未输出或保存任何 Secret 到本地台账。

## 仍未完成

- 第二台真机、多 worker 物理并行、第二个真实 GitHub 账号、真实账单/支付和外部通知仍未验收。
- 当前监控为本机 systemd/journald 证据链；外部告警渠道仍未接入，失败会保留在 systemd 状态和 journal 中。

以上缺口不否定持久化运维子项，也不把单机证据写成完整公共 GA。
