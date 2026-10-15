# 网站组织成员与角色管理验收

日期：2026-10-15。最新主方案阶段 5 的组织成员/角色管理子项，不接第二台真机。

## 范围与合同

网站新增“管理组织成员”，支持查看组织成员、创建稳定成员 ID、修改角色、停用和恢复。角色中文说明、当前状态、失败原因、刷新和重试均在管理窗口内显示，390px 可使用。无设备的管理员仍可进入管理。

本批采用用户明确批准的权限：所有者管理全部角色；管理员仅管理操作员/只读成员，不能通过旧创建 API 或新更新 API 提权、修改管理员或所有者。组织必须至少保留一名启用的所有者。最后所有者检查、版本比较、更新、会话撤销、审计和幂等结果在一个 SQLite 事务中完成；指定所有者退出 owner 角色时，组织指针转向另一名启用的 owner。

角色权限仍只存于原 principals 表并由 ROLE_PERMISSIONS 定义；新增 member_state 表仅保存 active/disabled 和 revision。已有成员迁移默认 active/revision=0。停用撤销全部会话并拒绝新的 OAuth 登录；恢复不复活旧会话。停用不删除历史，不转移设备或任务归属。设备管理不允许选择停用成员。

停用前检查本人未结束任务、设备占用及在途写请求；当前单 API 进程内与请求派发使用同一锁和计数。请求在派发前重新检查最新身份及权限，画面逐帧复查。已开始的合法请求允许结束；角色修改不强制取消已开始的任务。多个 API worker 的跨进程任务派发互斥尚未验收；SQLite 独立连接间的最后所有者保护有离线验证。

创建成员不等于发送邀请或完成 GitHub 绑定。页面明确提示先由部署人员通过现有稳定 GitHub ID 映射关联身份，再按需分配设备。本批不新增公开注册、邮件邀请或第二身份权威。

## API 与 SDK

- GET `/api/control-plane/organizations/{organization_id}/principals`：返回成员 role/status/revision/permissions，以及由服务端角色权威计算的 create_roles 和 editable_roles；仅同组织 identity.manage 可读。
- POST 同一路径：沿用 user_id/role/team_ids，服务端补齐授予角色边界，事务内记录创建审计。
- PUT `/api/control-plane/organizations/{organization_id}/principals/{user_id}`：必须提供 role、status、expected_revision；要求主体身份、同源、控制令牌、Idempotency-Key。旧版本/最后所有者/未结束工作返回 409；越权 403；不存在 404；字段错误 422。
- 同键同请求复放结果，不重复更新/撤销/记录变更；同键不同请求返回冲突。停用成员再次 OAuth 验证身份后返回 principal_disabled，不签发网站会话。
- 官方 `LocalAgentApiClient.control_plane_update_principal()` 从当前 OpenAPI 校验方法和路由，不自动重试写入。

## 验证记录

新增验收覆盖：owner/admin/operator/viewer、旧创建入口提权、跨组织、最后所有者及组织指针转移、全部会话撤销/恢复、停用成员 OAuth 浏览器拒绝、任务历史和设备归属保持、停用成员不可分配设备、旧令牌角色即时生效、派发前重查、暂停任务/在途请求、CAS/幂等/重开库/审计、SQLite 独立连接竞争、官方 SDK、390px 界面、失败重试、退出后迟到响应清理。

完整离线验证：Python `scripts/run_website_isolation_tests.py --full` **986 tests，OK，4 skipped**；前端 `npm test` **55 项通过**（19 单元、36 浏览器）。离线 runner 强制 Mock 且使用临时 Runtime 数据库，不操作真实手机。真实网站结果在加载验收后补记。

初轮回归发现两个直接调用流式取图的旧测试请求缺少 HTTP method/route 元数据。设备复查改为直接检查 device.read/device.execute，测试补充真实 GET method；保留撤销后停止取图和后续取图异常用例，不降低生产权限校验。

## 加载与回滚

加载前备份运行 SQLite，确认当前项目 API 空闲，仅重载该 API。只使用真实 owner 会话管理一个验收成员；不更改现有 owner 权限，不转移设备，不创建手机任务。测试成员最终停用并保留审计。

用本批提交追踪代码；恢复代码不能覆盖后续用户数据。若需回滚界面或写入功能，应保留停用成员认证拒绝逻辑：直接退回不识别 disabled 的旧认证代码会重新允许该成员 OAuth 登录，因此不可作为安全回滚。数据库备份仅作故障恢复依据，不在线覆盖新数据。

本批完成不代表整个阶段 5/GA 完成；邀请/自助绑定、两名真实 GitHub 用户、多 worker 和多设备仍有独立验收缺口。
