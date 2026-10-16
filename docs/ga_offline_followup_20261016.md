# GA 离线收尾：多 worker、身份隔离与外部告警准备

日期：2026-10-16（Asia/Shanghai）

## 结论

**已验证可行（不依赖第二台真机的准备范围）；完整公共 GA 仍有条件可行。**

本批补齐了两个独立 Python 进程共享同一 SQLite 时的配额与身份绑定竞争验收，并增加外部告警的无 Secret 预检。测试使用稳定数字 ID fixture 验证身份唯一绑定竞争；没有把 fixture 当成真实 GitHub OAuth 登录。

## 多 worker 离线证据

- `poc/test_multi_worker_control_plane.py` 使用两个独立 Python 进程和同一 SQLite 文件并发竞争一个 Run 配额：只有一个进程取得 reservation，另一个收到 `quota_exceeded`；最终 `reserved_runs=1`。
- 两个独立进程同时竞争同一个 GitHub 稳定 subject，分别尝试绑定 `local-owner` 与 `worker-viewer`：只有一个绑定成功，另一个因版本/唯一绑定冲突失败；数据库最终只保留一个 user binding、revision=1。
- 该证据覆盖跨进程事务与隔离合同，不代表已经在线上启动多个 Uvicorn worker，也不替代多 worker 长时间压测。

## 第二 GitHub 身份隔离边界

- `worker-viewer` 是临时 SQLite fixture，只用于验证不同成员竞争同一稳定 subject 的唯一绑定。
- 未伪造第二个真实 GitHub 账号；真实双账号 OAuth、邀请和自助注册仍需用户提供第二个账号后再验收。
- 现有网站隔离合同继续要求任务、截图、回放、设备和通知按主体隔离，管理员不因管理权限获得他人内容读取权。

## 外部告警准备

- `poc/evals/check_external_alert_config.py` 提供不输出 Secret 的预检。
- 未提供完整 Gmail 配置时报告 `not_configured`；预检不验证 outbox 可写性、运行时接线或真实投递，不能据此宣称告警不丢失。
- 当前可选外部适配器是 Gmail SMTP 应用专用密码或 Gmail API refresh-token 组合；凭据只允许通过部署环境注入。
- Webhook、移动推送和外部告警真实投递尚未接入或验收；本批复用已有 Gmail 配置识别，新增脱敏预检，没有新增队列或启用外部供应商。

## 仍然阻断完整公共 GA 的事项

1. 第二台真机、独立校准和真实多设备动作证据。
2. 线上多 worker 长时间并发、故障恢复和部署平台级压测。
3. 第二个真实 GitHub 账号、邀请/自助注册与双账号浏览器验收。
4. 真实云账单/支付结算和外部告警渠道投递。

这些事项不阻断当前单设备软件发布；在没有对应外部资源和明确验收证据前，不把离线 fixture 或本地 outbox 写成已完成的公共 GA 能力。
