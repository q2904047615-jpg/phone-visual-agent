# 网站团队管理与成员分组验收（2026-10-15）

范围：主方案阶段 5 的团队管理子项；暂不接入第二台真机。基线代码 `bd5ce4a`，工作树 f064。

## 实现与合同

网站增加“管理团队与分组”：创建、改名、勾选添加/移除成员、同一成员多团队、搜索团队/成员、查看未分组成员。停用成员明确标注，分组不恢复其登录。入口无需已分配设备。

沿用 ROLE_PERMISSIONS 的 team.manage：所有者/管理员管理本组织团队。成员角色、GitHub 登录、设备归属及本人任务/截图/回放隔离不随分组变化。普通成员无团队写权限，原有团队只读 API 合同保留。不新增邀请发送、共享任务或团队资源授权。

SQLite 团队 member_ids 是运行时成员归属来源，成员列表和现有登录会话派生 team_ids。旧成员 team_ids_json 保留为事务同步的回滚投影，不独立参与运行裁决；创建成员并指定团队的旧入口也同步团队名单并递增版本。创建/更新与包含真实 actor、修改前后成员的审计在同一事务内提交。

原 GET/POST/PATCH 路由和 SDK 保持，团队响应新增 revision，PATCH/SDK 可传 expected_revision。网站总是传版本；冲突返回 409 并刷新，不静默覆盖。旧 SDK 未提供版本仍保留既有更新行为，不宣称这些旧调用具有乐观并发保护。幂等摘要绑定组织、具体 team_id 和请求体，同键跨团队不能复放错误响应。

## 实施前反证与回滚

初步核对发现旧团队写入不会同步成员侧归属，旧成员创建也不会回填团队名单。当前真实库只有 local-team/local-owner 与一名已停用验收成员，两侧一致。首版启动时迁移并清空旧字段的补丁被自动审批拒绝，未应用。最终版本取消启动迁移和清空，保留原字段，仅在获准团队写入事务内同步投影。

加载前运行只读预检：`poc/.venv/Scripts/python.exe poc/scripts/check_team_memberships.py poc/output/web/runtime.sqlite3`。发现历史不一致应先备份、核对具体记录，不能直接加载新读取方式或自动合并。当前预检 consistent=true、teams=1、members=2、issues=[]。本批未验证其他历史库迁移。

代码按独立本地 commit 追踪。重载前做 SQLite 在线备份，回滚时先确认后续写入，再精确撤回本批代码并只重载项目 API；不得用旧备份覆盖后续任务/成员数据。旧字段保留可供回退读取，但旧代码再次写分组会失去本批一致性修复，需停止团队写入并复查后再恢复管理。

## 离线验证

- 完整 Python 隔离回归：1004 tests，OK，4 skipped。新加跨团队幂等反例及最终修正另跑 86 项定向回归全部通过。
- 前端完整回归：61 项通过（原 58 + 团队 3），最终刷新提示调整另跑团队 3 项通过。
- 覆盖创建/改名/加减成员、多团队、空团队、停用成员、非法/重复/跨组织成员、同组织层级、既有登录投影更新、本人资源隔离、幂等、两数据库连接版本竞争、审计事务失败整体回滚、重开持久化及官方 SDK。
- 浏览器覆盖无设备入口、390px 宽度、成员搜索/团队过滤、管理员操作、错误重试复用幂等键、过期编辑刷新、退出与延迟响应不恢复私有内容。
- 初次测试错误为 SDK 测试参数名、隔离 404 预期写成 403，以及 Mock 拦截器将 teams.js 当 API；修正后通过，未因此改动既有生产权限合同。

## 加载与真实验收

**已验证可行：本批实现、离线回归、加载和单真实账号网站验收完成。** 代码 `834eef3`、样式 `8299215` 均本地提交，最终重载 PID **18940**。重载请求 revision=`8299215eb1d2cb6f76e829a8a52f17e923f9ab44+dirty`，受跟踪代码干净，dirty 仅为保留的既有未跟踪 Playwright 日志；不据此进行真机能力晋级。官方客户端 bootstrap/OpenAPI 验证通过，版本 0.2.0-trial-upstream。

当前真实 local-owner 通过页面完成创建（含本人）→改名并加入已停用验收成员→移除两名成员并改名“验收留存（空组）”，版本依次 0→1→2。本人原有 local-team 始终保留；最终原成员分组、角色、状态及成员版本恢复/保持原值。测试组 ID=`team-9be86547-1e3b-4924-8f60-185bc487d902`，作为空组保留，不删除原团队。过期 expected_revision=0 写入返回 409/team_conflict，未覆盖当前组。

真实退出登录清空团队列表/分组概览并隐藏入口，再次 GitHub 登录成功；第二次 API 重载后团队名、空成员名单、revision=2 仍在。390×844 下对话框 clientWidth=scrollWidth=320，1280×900 桌面无横向溢出，截图已检查。真实跨账号权限拒绝没有第二账号实测，依据隔离回归，不混作真实证据。

SQLite 写入审计 **161/166/172** 记录创建和两次更新，actor 均 local-owner，包含实际成员及更新前后信息。最终只读预检 consistent=true、teams=2、members=2、issues=[]。唯一设备仍归 local-owner/revision=2；原验收成员仍 viewer/disabled/revision=4，本人 owner/active/revision=0。controller/camera 在线，busy=false、无活动任务及手动动作，doctor ready=true、blockers=[]、physical_actions=0，原历史 api-scope 仍 cancelled/0 物理动作。

证据目录 `poc/output/web/team_management_20261015/`（忽略入库）：`before_teams.sqlite3` 在线备份 integrity_check=ok、完整/定向测试日志、`team-mobile-final.png`、`team-desktop.png`、`report.json`。未接第二台真机、未执行手机动作或付费模型调用、未操作卖家 main.exe。真实 OAuth 重新登录沿用本会话授权。未推送 GitHub。

## 尚未验证的范围

第二真实 GitHub 账号、多 API worker 端到端、第二设备、团队删除/归档、邀请和团队共享资源不在本批完成证据内；不宣称整个阶段 5 或 GA 完成。
