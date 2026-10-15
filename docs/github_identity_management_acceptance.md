# 网站 GitHub 身份关联管理验收

日期：2026-10-15。最新主方案阶段 5 的身份关联管理子项。用户明确批准权限层级、唯一关联、解绑撤销登录和保留可登录所有者；用户目前只有当前 GitHub 账号，新增/解绑双用户全流程使用隔离测试，真实验收限当前账号。不接第二台真机。

**已验证可行：实现、完整离线回归、服务加载与约定的单账号真实网站验收完成。** 双真实账号流程尚未验收，不计作 GA 完成。

## 实现与边界

网站“管理 GitHub 关联”支持按用户名或稳定 ID 查询、展示公开账号和主页、选择组织成员后确认关联、查看本组织关联、确认解除及错误重试。查询不自动授权，公开账号存在不代表管理员拥有该账号；真正登录仍须完成 GitHub OAuth。查询方式参考 [GitHub 官方用户 API](https://docs.github.com/en/rest/users/users#get-a-user-using-their-id)，不以可变用户名作为关联主键。

owner 可管理全部角色，admin 只能管理 operator/viewer。服务端对列表、查询、写入校验身份和组织；写入在事务内重新检查角色，不依赖按钮隐藏。GitHub 稳定 ID 全局唯一；已有有效关联不能直接改派，须先解除。未关联状态保留版本墓碑防止旧页面覆盖新关联。同键同请求返回原结果，不重复写入；网络失败后重试沿用幂等键。

解绑撤销目标成员全部网站会话，保留任务、设备和历史，不取消已开始任务。不能解除最后一名启用且已关联 GitHub 的 owner；同一保护接入成员降级/停用，不能用另一管理入口绕过。新关联可指向停用成员，但该成员恢复前仍不能登录，界面明确标记。

原文件只在首次启动时原子导入 SQLite，完成标记与全部映射同时提交；成员或 ID 无效则整体回滚。迁移完成后不再读取旧文件，即使文件变化、无效或环境变量移除，也不恢复旧关联。生产 Provider 只注入数据库 resolver，无静态文件登录回退。旧文件保留作迁移前资料，不是运行权威。

OAuth 验证后取得关联版本，签发网站会话时在 SQLite 写事务中再次核对 ID、组织、成员及版本。与解绑竞争时，签发先发生则随后被撤销，解绑先发生则拒绝旧回调；解绑后又关联回来也不能复用原版本。公开账号查询使用固定 GitHub API 主机、无应用密钥/用户令牌、禁止重定向，不接受组织账号、伪造 ID 或异常响应作为个人身份依据。

## API 与 SDK

- GET `/api/control-plane/organizations/{organization_id}/github-identities`：本组织关联及可管理标志，不返回令牌。
- GET 同路径 `/lookup?login=...` 或 `/lookup?subject=...`：二选一查询。其他组织占用仅返回“已有其他关联”，不泄露对方组织/成员 ID。
- PUT 同路径 `/{subject}`：必须提供 `user_id`（null 明确解除）和 `expected_revision`，要求主体、控制令牌、同源及 `Idempotency-Key`。403 越权，404 缺少成员/账号，409 已占用/旧版本/最后可登录所有者，422 字段错误，503 GitHub 查询暂不可用。
- 公开账号查询在 SQLite 写事务外执行；查询结束后再次验证操作者会话，撤销/降级后的在途查询不能继续写入。幂等成功重放不依赖 GitHub 可达性；解除不需 GitHub 网络。
- 官方 SDK 新增 `control_plane_github_identities`、`control_plane_github_lookup`、`control_plane_update_github_identity`，沿用 OpenAPI 校验和禁止自动重试写入的合同。

## 离线验收

Python `scripts/run_website_isolation_tests.py --full`：**997 tests，OK，4 skipped**。前端 `npm test`：**58 项通过**（19 单元、39 浏览器）；显式选择目标成员的最后界面调整再次通过相关 3 项浏览器回归。此前定向 **76 项通过**；最后新增独立连接竞争和伪造公开资料用例包含在完整回归中。

覆盖现有 GitHub token/code/PKCE/state、登录/退出、成员/设备/任务隔离回归，以及新增查询→关联→OAuth 登录→解绑→旧会话失效→拒绝登录→重新关联流程；权限层级/跨组织/禁止覆盖；最后登录 owner 及角色/停用入口；CAS/幂等/GitHub 故障重试/重开库/审计；查询期间撤销；旧回调/重新关联的 ABA 版本；SQLite 独立连接签发与解绑竞争；无认证公共查询/伪造返回；迁移原子性及无文件回退；390px、显式确认、失败重试、退出及修改查询后迟到响应清理。

浏览器测试第一次启动因自动审批超时未执行；按工具允许重试一次后通过，不属于代码测试失败。全部离线测试使用临时数据库及 Mock GitHub/设备，没有拿真实手机补调试。

## 加载与真实结果

代码提交 `92d27b28c80551a0762479eca6c298c0d0534cee` 已加载。最终监听 `127.0.0.1:8765`、PID 13624；版本 `0.2.0-trial-upstream`，官方客户端 bootstrap/OpenAPI 核对新增 list/lookup/update 路由成功。加载 revision 的 `+dirty` 来自保留原有未跟踪 Playwright 日志，受跟踪代码已提交。

真实网站使用现有 GitHub `q2904047615-jpg`、稳定 ID `279507169`，本地成员 `local-org/local-owner`：

1. 空闲检查通过后重载，旧映射原子迁入 SQLite，唯一关联保持 local-owner/revision=0，当前会话可用。
2. 网页按稳定 ID 查询成功，展示真实 GitHub 用户名、ID 和当前成员；确认原关联后显示用户名并更新 revision=1。随后按用户名查询也返回同一稳定 ID。
3. 通过页面确认解除唯一 owner 关联，返回 409/last_linked_owner_required，关联和会话保持，不会锁号。
4. 390px 下窗口 clientWidth/scrollWidth 均为 320。退出后关联入口隐藏、列表和查询资料清空；重新真实 GitHub OAuth 登录成功，回到 local-owner。
5. 再次 API 重载，关联和 revision=1 持久化。原 owner 仍 active/角色 owner/revision=0，前批验收成员仍 disabled/viewer/revision=4；设备仍归 local-owner/revision=2，无分配变化。
6. 本人历史 api-scope 仍 cancelled，任务表未新增且 physical_actions=0；status controller/camera 在线、busy=false、无活动任务/固定测试；doctor ready=true、blockers=[]、本地采集 4 帧、physical_actions=0。

SQLite 只读报告确认迁移完成标记及审计：135 导入一条身份、140 更新原关联、145 拒绝解除最后可登录 owner；业务变更 actor 为 local-owner。原映射文件未修改，迁移后它不再参与身份解析。

备份及证据位于 `poc/output/web/github_identity_management_20261015/`：`before_github_identity.sqlite3`（integrity_check=ok）、`github-link-full.log`、`github-link-frontend.log`、`github-link-targeted.log`、`github-link-browser-final.log`、`report.json`。没有真实新增其他人的关联、没有解除当前唯一账号；相应成功流程已有隔离证据。没有接第二台真机、发起手机动作/付费模型调用或操作卖家 main.exe，当前页面留在已登录的关联管理窗口。

## 回滚与剩余目标

本批以独立提交追踪代码，保留原有未跟踪日志。回滚管理页面或写入路由时必须保留数据库关联和停用检查；直接恢复旧文件认证可能让已解除关联重新生效，因此禁止盲目恢复旧认证入口。备份不用于覆盖后续用户/任务数据；旧实现回退需单独审核当前有效映射和停服恢复方案。

未完成双真实 GitHub 账号验收（用户目前只有一个账号）、邀请/自助注册、多 API worker 的完整网站并发、第二设备及完整 GA。数据库事务级跨连接验证不等于多进程端到端验收。
