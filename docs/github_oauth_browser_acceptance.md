# GitHub OAuth 浏览器授权验收准备

当前实现是 GitHub OAuth Web application authorization-code flow：浏览器授权、固定回调、服务端 code exchange、PKCE、Check a token、稳定用户 ID 映射和本地主体会话签发。

## 部署配置

在运行 API 的同一进程环境中设置：

```text
GITHUB_OAUTH_CLIENT_ID=<GitHub OAuth App 的 client_id>
GITHUB_OAUTH_CLIENT_SECRET=<仅保存在服务端的 client_secret>
GITHUB_OAUTH_IDENTITY_MAP_FILE=<部署侧受信 JSON 文件绝对路径>
GITHUB_OAUTH_REDIRECT_URI=http://127.0.0.1:8765/api/control-plane/auth/github/callback
```

映射文件示例：

```json
{
  "42": {
    "organization_id": "local-org",
    "user_id": "local-owner"
  }
}
```

`42` 必须替换为 GitHub 账号的稳定数字用户 ID，并且目标主体必须已经存在于 Control Plane SQLite 投影。不要把 client secret、access token 或 refresh token 写进仓库、聊天或日志。

GitHub OAuth App 的 callback URL 必须与 `GITHUB_OAUTH_REDIRECT_URI` 完全一致；不要启用通配符回调。当前默认只请求 `read:user`，不申请仓库写入权限。

## 验收步骤

1. 启动服务后，通过官方 typed client 调用 `GET /api/control-plane/auth/github/start`，请求带当前 `X-Control-Token` 和本机 Origin。
2. 在返回的 `authorization_url` 中完成 GitHub 登录和授权；浏览器应回到固定 callback。
3. callback 返回的 `data.session.user_id` 必须等于映射文件目标主体；响应正文不得出现 GitHub access token。
4. 使用返回的 `principal_token` 访问一个已授权的 Control Plane 读接口，再用会话撤销接口撤销自己；撤销后再次访问应返回身份无效。
5. 检查审计只出现 provider、subject、user_id、session 等必要元数据，不出现 code、state、verifier、access token 或 client secret。
6. 重放同一个 callback URL 应返回 `oauth_state_invalid`；改动 mapping 后重新登录，未映射用户应返回 `external_identity_mapping_not_found`。

离线回归已经覆盖授权 URL、PKCE、state 原子消费、callback code exchange、token revalidation、身份映射和会话签发。真实验收仍需要你在本机配置 GitHub OAuth App 和映射文件；只提供 client_id、issuer 或 callback URL 等非敏感信息即可协助核对，不要发送 secret/token。
