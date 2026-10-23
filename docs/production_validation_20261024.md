# 生产环境验证记录

时间：2026-10-23 至 2026-10-24（Asia/Shanghai）。

## 结论与范围

**已验证可行：当前公网版本的有界只读负载、空闲进程自动恢复、三库在线备份与隔离恢复。完整生产验收仍未完成。**

本次 API 始终加载 `21696e1`，current 为 `/opt/phone-agent/releases/20261023-21696e1/poc`。没有部署上一批 `adcab73` 的远程工具协议 v3；该版本须与 Windows 连接器配套升级。本批只更新独立备份维护脚本，不修改业务 API、身份规则、设备配置或模型调用。

## 公网读取与权限拒绝

使用 `poc/evals/production_read_probe.py`，经官方 `LocalAgentApiClient` 与 HTTPS transport 访问已经核对的 `/api/session`、`/openapi.json` 和组织只读路由。transport 仅允许这三条 GET；不创建任务、不伪造主体、不打印令牌。

| 检查 | 实际结果 |
|---|---|
| 首次持续读取 | 2026-10-23 21:05:01 至 21:10:49；4 并发客户端，600/600 轮通过 |
| 匿名组织访问 | 60/60 次返回 401 |
| 每轮耗时 | P50 66.19ms、P95 1465.39ms、最大 2799.43ms |
| 恢复后复验 | 2026-10-24 01:18:48 至 01:19:01；2 并发，20/20 轮通过，2 次匿名访问均拒绝 |

每轮含 bootstrap 的两个 HTTP 请求，部分轮次附加权限拒绝检查；上述耗时不是单接口延迟。这是约 5 分 48 秒的小负载读取验收，不证明高负载容量、长时间稳定性、登录后的写入性能或业务吞吐。

## 备份故障、修复与加载

发现：多 worker 部署已将实际数据库迁至 `/opt/phone-agent/state/runtime/`，定时脚本仍读取 `$ROOT/output/web/runtime.sqlite3`，造成备份服务退出码 2。定时器 active 不能证明备份成功，原最新成功备份停留在 10 月 18 日。

修复：新增 `production_backup.py` 与 `phone-agent-backup.sh`，固定使用稳定状态目录，覆盖 `runtime.sqlite3`、`dispatch.sqlite3`、`connector.sqlite3`。各库使用 SQLite online backup，核对所需表和完整性；完成后写 manifest。14 天保留仅清理该工具生成、带正确 manifest、内容完全可识别的备份集；旧备份、额外用户文件及不完整集合保留。

同时修复公共 SQLite 拷贝工具：WAL 源库复制得到的私有目标先 checkpoint 并切到 DELETE journal，再原子改名，避免遗留临时 WAL/SHM 文件。源库 journal 和线上内容不变。

- 首次安装因新目录 root:root 750 导致 ubuntu 无读取权限，入口已自动回退；核对日志后改为 root:ubuntu 750，再以 ubuntu 检查可读。
- 正式维护脚本位于 `/opt/phone-agent/state/maintenance/production-backup-v1/`，systemd 入口仍为 `/opt/phone-agent/state/maintenance/phone-agent-backup`。
- 立即启动现有 `phone-agent-backup.service`，`Result=success`、`ExecMainStatus=0`。
- 实际产物：`/opt/phone-agent/state/backups/production-20261023T131209045954Z`。
- 从该产物恢复到新目录 `validation-20261023/scheduled-backup-restore-check`，三库 integrity 均为 ok，`live_replaced=false`。
- 定时器 active；复核时下一次触发为 2026-10-24 02:19:48 CST。该未来调度尚未观察执行，不能与已完成的立即服务演练混同。

每个库分别为一致快照，**三个库不构成跨库原子快照**。隔离恢复只验证文件和表完整性，不证明在途业务可以无损恢复或重放。

## 生产进程恢复

演练前重新检查：Runtime 会话为空，异步任务 active=0，派发 queued/dispatching=0，连接器 queued/claimed=0；三库 quick_check 均为 ok。只中断项目服务进程，没有操作卖家 main.exe 或手机。

1. SIGKILL 一个精确识别的 API worker（2355611），Uvicorn 父进程补回新 worker；约 0.58 秒完成进程补位及一次 HTTPS bootstrap。父进程与执行进程保持。
2. 再次核对空闲后，SIGKILL 唯一执行进程（2355608），systemd 按 Restart=on-failure 自动拉起为 2505826；约 4.87 秒完成公网 bootstrap。
3. 入口服务因现有 Requires=phone-agent-executor.service 依赖也被 systemd 停止并重新拉起，主 PID 变为 2505827。期间有两次探测错误，journal 明确出现 503。**此项是有短暂中断的恢复，不是零中断或高可用切换。**
4. 最终入口与执行服务均 active/Result=success；执行服务 NRestarts=1。现有健康检查立即运行成功，Result=success/ExecMainStatus=0；监控与备份定时器均 active。

演练前后账本一致：Runtime 0；dispatch responded=3846、not_dispatched=1；connector completed=3840、not_dispatched=32、unknown=1。历史 unknown 保留，没有新增或重放设备命令。空闲演练不替代在途真实业务恢复验收。

## 邮件与账号

用户确认没有第二个由其控制的 GitHub 账号，因此真实双账号登录与数据隔离保持待验收，隔离测试不冒充真实账号。

用户指定测试收件地址后，使用当前连接的 Gmail 发送仅含“按你的要求发送的测试告警邮件”的消息，工具返回成功及 SENT/INBOX 标签。原尝试含内部运维内容，被自动审批拒绝；未发送该内容。最终邮件回执 ID 为 `1a168e281c8dee67`。

这证明连接的 Gmail 完成了一次测试投递，不代表服务器具备自动发送能力。检查服务器配置未发现 Gmail/SMTP/告警凭据；现有 systemd 健康检查仍仅提供本机状态与 journal。不得从当前 Gmail 连接提取凭据给服务器。

自动告警后续需要在服务器安全配置可长期使用的发件渠道（例如现有适配器支持的 GMAIL_SENDER 与 GMAIL_APP_PASSWORD，或 OAuth 刷新凭据），再接通健康/备份失败触发、限频及恢复通知，完成一次由服务器触发的真实投递。不要把密码或令牌发送到聊天或写入 Git。

## 离线验证与证据

相关测试 31 项通过：

```powershell
python -m unittest test_production_validation test_runtime_sqlite_backup test_multi_worker_http test_multi_worker_deployment test_external_alert_config -q
```

新增测试覆盖三库 WAL 内容恢复、禁止覆盖线上目录、缺失源库、保留策略、额外用户文件保护、HTTPS origin 与只读探测边界。本批未改前端，不重复运行前端全量；也未以这 31 项声称完整 Python 回归。此前工具批次的全量测试仍是历史证据。

服务器脱敏证据保留在 `/opt/phone-agent/state/validation-20261023/`：

- `public-read-probe.json`：首次持续读取。
- `post-recovery-read-probe.json`：恢复后复验。
- `process-recovery.json`：中断、拉起、前后账本。
- `scheduled-backup-acceptance.json`：定时服务产物恢复结果与维护文件哈希。
- `phone-agent-backup.before`：原维护入口，只用于精确撤回脚本。

维护代码 SHA256：production_backup.py=`5690927cda2210615160a5dc63849212c98cd1576f567f5bf86f9a21c0935053`，runtime_sqlite_backup.py=`39a3e4ce6ea7043754dd254d95d632b2874310c11b53dd70dbf4c3667293e734`，Linux 包装脚本=`c1619a471f692eb2af3eb1dabf2e1aebde3ff04d4356120d60cd0713be41daef`。

回滚只恢复原备份入口，不替换数据库、不切换 API current；但旧脚本有已确认的路径故障，撤回后必须明确标记备份再次不可用。代码用本批独立 commit 精确追踪，不能 reset 掉其他批次。

## 尚未完成

- 第二个真实 GitHub 账号的登录、权限和内容隔离验收（用户目前没有账号）。
- 服务器自主外部告警及恢复通知（发件凭据与运维触发接线尚缺）。
- 新远程工具协议 v3 配套发布，ADB 手机连接后的真实工具与业务闭环；用户已决定后置手机连接。
- Windows 设备连接器常驻/自启、完整多设备真实任务及在途业务恢复证据。
- 长时间/写入负载与容量上限；本次有界读取不能替代。

上述缺口不影响本次已验证子项，但不能据此把整个 GA 或全部生产环境验证标成完成。
