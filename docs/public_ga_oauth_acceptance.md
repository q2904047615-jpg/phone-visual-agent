# 公网 GitHub OAuth 与会话验收记录

日期：2026-10-16（Asia/Shanghai）

## 结论

**已验证可行（公网登录子项）；完整公共 GA 仍有条件可行。**

`agent.jianlizhineng.cn` 的 DNS、HTTPS、Nginx 反向代理和 `phone-agent.service` 已在线；应用继续只监听 `127.0.0.1:8765`，公网请求由受控代理转发。公网来源白名单已加载，只接受配置的 HTTPS origin。

## 服务端证据

- `phone-agent.service`：`active (running)`。
- 公网 `/` 与 `/api/session`：HTTP `200`。
- systemd 环境预检：必需非 Secret 配置、`PUBLIC_BASE_URL`、GitHub 回调、稳定身份映射和租约目录均 `pass`。
- 当前预检唯一阻断：`DASHSCOPE_API_KEY` 未注入；它不阻断登录、刷新和退出验收，但阻断完整模型可用的公共 GA 预检。
- 远程加载文件哈希与本地已验证公网来源修复一致；旧文件保留在远程发布目录的备份文件中。

## 浏览器真实验收

在用户授权的 Edge 会话中，只操作 `agent.jianlizhineng.cn`：

1. GitHub OAuth 授权回调返回 `303`，网站显示 `local-owner` / `local-org`。
2. 完整页面刷新后仍显示已登录，证明主体会话 Cookie 可保持。
3. 点击“退出登录”后页面回到“未登录”。
4. 服务端审计记录 `identity.session.revoke | permission_allowed` 和 `identity.session.revoke | revoked`；退出后的主体查询返回 `authenticated=false`、`github_available=true`。

本次未读取或保存密码、验证码、OAuth code、state、access token、client secret 或主体 token。

## 仍未完成

- `DASHSCOPE_API_KEY` Secret 注入及模型真实调用回归。
- 持久化租约目录、多 worker 并发、备份保留和恢复负责人尚未作为生产运维合同正式核对。
- 第二台真机、物理多设备并行、第二个真实 GitHub 账号、真实云账单/支付和外部通知仍未验收。

以上缺口不否定本次公网登录子项，但完整公共 GA 不能据此宣称全部完成。
