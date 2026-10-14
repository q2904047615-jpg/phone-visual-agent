# GitHub OAuth 浏览器授权验收准备

当前实现是 GitHub OAuth Web application authorization-code flow：浏览器授权、固定回调、服务端 code exchange、PKCE、Check a token、稳定用户 ID 映射和本地主体会话签发。

## 网站登录入口（2026-10-15）

工作台账号区域提供「使用 GitHub 登录」、当前主体/组织/有效期和「退出登录」。网页登录请求使用 `GET /api/control-plane/auth/github/start?browser=true`；服务设置短期 HttpOnly 浏览器绑定 Cookie，回调必须匹配发起浏览器，再复用同一 GitHub 验证和 SQLite 会话签发链。成功后 303 返回工作台，仅在 HttpOnly、SameSite=Lax Cookie 中保存本地主体令牌，Cookie 路径限定 `/api/control-plane/auth`，HTTPS 下设置 Secure，HTTP 仅用于现有本机验收地址。

`GET /api/control-plane/auth/browser-session` 从 SQLite 验证当前 Cookie，仅返回登录状态、主体、组织、角色和过期时间；无效/过期 Cookie 会被清除。`POST /api/control-plane/auth/browser-logout` 通过现有会话撤销权威撤销当前会话并清除 Cookie，重复退出安全。两接口保留本机控制令牌与来源校验；登录状态不接入手机 Runtime admission。回调失败回到工作台显示固定提示，不把授权码、令牌或供应商原始错误带进 URL；认证响应禁止缓存。JS/localStorage/sessionStorage 不保存主体令牌。

默认不带 `browser=true` 的 API 授权码流程保留 JSON 会话封套合同。网站使用 `http://127.0.0.1:8765/`，与部署回调同主机；服务重启会使进行中的 OAuth 请求失效，需要重新登录，已签发且未过期/撤销的 SQLite 会话仍有效。

本批离线证据：身份/API 定向 33 项通过；日常 Python 933 项、4 skipped，前端单元 19 项、浏览器 28 项通过。历史实验 120 项中 2 failures/38 errors，修改前 HEAD 隔离副本复现同样结果（缺失历史样本及旧断言），不得宣称完整历史回归通过。真实网站入口验收待本批加载后记录；已有下文真实 OAuth API 验收不能替代网站验收。

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
4. 通过官方类型化客户端核对映射目标主体，再携带返回的 `principal_token` 调用会话撤销接口撤销自己；用同一令牌再次调用该受主体认证保护的接口，应返回 `401 identity_invalid`。当前 Control Plane GET 路由只检查本机控制令牌，读取成功不能证明 OAuth 主体认证有效。
5. 检查审计只出现 provider、subject、user_id、session 等必要元数据，不出现 code、state、verifier、access token 或 client secret。
6. 重放同一个 callback URL 应返回 `oauth_state_invalid`；改动 mapping 后重新登录，未映射用户应返回 `external_identity_mapping_not_found`。

离线回归已经覆盖授权 URL、PKCE、state 原子消费、callback code exchange、token revalidation、身份映射和会话签发。部署配置和每次真实验收应分别记录，不要发送 secret/token。

## 2026-10-15 真实登录验收记录（Asia/Shanghai）

结论：本次真实 GitHub 浏览器登录、身份映射、本地会话撤销及相关审计链已验证可行。不是完整 GA 验收，也不是全部 OAuth 异常场景实测。

- 用户在 GitHub 浏览器中完成登录和授权；本机 callback 返回本地主体会话。
- GitHub 数字用户 ID `279507169` 映射到已存在的 `local-org / local-owner`。
- 验收会话 `a193476335aa4d19a627474c030e14ca` 于北京时间 00:20:56 签发，00:24:01 撤销。
- 官方类型化客户端调用自身会话撤销接口返回 HTTP 200、`revoked=true`；用同一令牌再次调用返回 HTTP 401、`identity_invalid`。只撤销本地主体会话，没有撤销 GitHub OAuth App 授权。
- SQLite 只读核对确认该会话已设置 `revoked_at`，令牌字段存储 64 字符哈希。
- 审计序号 21、23、24、25 分别记录 `issued`、`permission_allowed`、`revoked`、`identity_invalid`。最后一条因身份无效归入 `unknown`，通过只读 SQLite 查询核对；未猜测不存在的审计 API。
- 本次回调响应未含 GitHub access token；本次核对的审计元数据未出现 code/state/verifier/token/secret 等敏感字段，亦未含实际 client secret。
- 服务 bootstrap/OpenAPI 通过，controller/camera 在线，`busy=false`、`active_tasks=[]`；本批没有手机动作或模型调用。

本地脱敏证据：`poc/output/web/github_oauth_live_20261015/acceptance_report.json`（忽略提交）。API 验收启动关闭 HTTP access log，避免记录回调查询参数。辅助本机验收进程已经退出，浏览器中的会话正文已替换为验收状态。

未在本次真实登录中复测：回调重放、state 过期、未映射账号和 GitHub 故障路径；这些仍仅有既有离线合同证据。普通 GET 接口的主体认证、网站登录态界面、并发配额/计费和 GA 发布均不由本次结果推导完成。
