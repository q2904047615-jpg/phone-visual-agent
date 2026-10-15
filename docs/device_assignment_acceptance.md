# 网站设备分配管理验收

日期：2026-10-15。主方案阶段 5 的设备分配子项；本批不接入第二台真机。

## 范围与结论

**已验证可行（离线）；服务加载与真实网页验收待记录。**

网站新增“管理设备分配”：owner/admin 可查看本组织已登记设备及现有成员，分配、改派、取消分配。operator/viewer 无管理权限。仅分配设备使用权，不转移历史任务、回放、截图或上下文，不启用其他机位，也不清理手机上的 App 账号或内容。后者在页面明确提示。

设备归属继续由 `control_plane_resource_owners` 唯一维护；新增表只记录设备分配版本，不保存第二份所有者。取消分配保留组织归属，避免其他组织认领。版本匹配、幂等结果和变更审计使用 SQLite 事务持久化，过期页面不能覆盖新分配。

所有涉及设备的 HTTP 请求在派发前重查归属并登记在途计数；分配变更与登记互斥。计数按设备隔离，不把不同设备的正常请求串行化。有在途请求、活动/暂停/待恢复任务、异步启动、监控或验收时返回 409，不强制停止或改派。MJPG 逐帧检查，变更后停止新帧。历史资源仍归原用户。

## API 与 SDK

- `GET /api/control-plane/organizations/{organization_id}/devices`：组织设备元数据，要求 `device.manage`，不返回任务内容或画面。
- `PUT /api/control-plane/organizations/{organization_id}/devices/{device_id}/assignment`：JSON `user_id`（null 取消分配）、`expected_revision`（非负整数）；要求身份、控制令牌、完整同源及 `Idempotency-Key`。
- 同键同请求复放原结果；同键不同内容、旧版本或占用返回 409；跨组织/无权限返回 403；未登记设备返回 404；不存在的组织成员返回 422。
- 官方客户端 `control_plane_devices()`、`control_plane_assign_device()` 通过当前 OpenAPI 校验，使用构造时的 `principal_token`；没有新增任意用户会话签发入口。

## 离线证据

项目 Python 执行 `scripts/run_website_isolation_tests.py --full`：**975 tests，OK，4 skipped**。整理导入后定向 **56 tests，OK**。`npm test`：**52 项通过**（19 单元、33 浏览器）。日志位于 `poc/output/device_assignment_full.log`、`device_assignment_targeted.log`、`device_assignment_frontend_full.log`。

新增验收覆盖：分配/改派/取消/恢复；角色、组织和成员边界；设备登记检查；历史任务不转移；旧用户拒绝取图；请求在途时不改派；旧授权在派发前重查；流式画面停止；事务版本与幂等；重开库保持分配；正式路由 SDK；390px 页面；无设备管理员仍能恢复分配；失败重试保持同一幂等键；退出清理管理界面。

第一次新增浏览器用例失败来自测试伪域名不是安全上下文、无法使用 `crypto.randomUUID()`；改为完全拦截的本机源后通过。没有关闭生产校验或使用真实设备作为调试器。历史实验套件本批未重跑，尚未证明所有未覆盖场景无退化。

## 真实运行与剩余边界

待备份库后重载项目 API，在真实 GitHub 登录下检查入口，执行唯一设备取消分配→确认无设备且拒绝取图→恢复 `local-owner`，核对审计及持久状态。不发起手机任务或付费模型请求，不操作卖家 `main.exe`。

两名真实 GitHub 用户间的转移和第二台真机仍未实测；相应跨用户/跨组织结论仅来自离线验收。本批不包含设备注册、硬件校准、成员邀请、计费或完整 GA。

## 回滚

以本批 commit 精确回滚代码，保留原有未提交日志。加载前在线备份 Runtime SQLite；真实验收结束恢复设备给 `local-owner`。回滚可保留版本元数据表，但不能通过还原旧数据库覆盖后来生成的任务/用户。任务历史与原有资源归属不改写。
