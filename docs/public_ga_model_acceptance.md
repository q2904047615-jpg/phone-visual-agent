# 公共 GA 模型真实调用验收记录

日期：2026-10-16（Asia/Shanghai）

## 结论

**已验证可行（服务端模型调用子项）。** `DASHSCOPE_API_KEY` 已由部署负责人注入服务器环境，服务重载后配置预检全部通过，并完成一次真实 DashScope 调用。该调用只验证模型连接，不读取手机画面、不启动任务、不执行物理动作。

## 验收证据

- `phone-agent.service`：重载后 `active`，`/api/session` 返回 `ready=true`、`mock=false`。
- 公共 GA 配置预检：`status=pass`；`required_secrets`、公网基址、GitHub 回调、身份映射和持久租约目录均通过；模型解析为 `qwen3-vl-plus`。
- 服务端 `DashScopeVisionProvider` 真实文本调用：`result=pass`、模型 `qwen3-vl-plus`、`successful_call_count=1`、`network_attempts=1`、`finish_reason=stop`，响应长度 9 字符；未记录回复正文或 API Key。
- 通过网站 `/api/qwen/chat` 的无主体请求返回 `401 identity_required`，证明控制令牌不能绕过网站用户会话；这不是模型调用失败。
- 本次没有读取摄像头、没有启动 Runtime Task、没有发送手机动作、没有调用支付或通知服务。

## 仍未完成

- 日志保留周期、监控告警和定期备份调度尚未形成正式运维负责人合同。
- 第二台真机、多 worker 物理并行、第二个真实 GitHub 账号、真实账单/支付和外部通知仍未验收。

以上缺口不否定模型调用子项，也不把单次模型调用写成完整公共 GA。
