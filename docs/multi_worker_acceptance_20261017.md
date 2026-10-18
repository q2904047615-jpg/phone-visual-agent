# 多 API 入口与单 Runtime 执行进程验收

日期：2026-10-17（Asia/Shanghai）。本批只在隔离环境实施和验收，不修改公网部署。

## 结论与适用范围

**已验证可行：单机多个 HTTP API worker，共用一个业务执行进程。** 用户已明确批准采用单执行进程方案。本方案中的 API worker 接收和转发网站/API 请求；认证、控制面写入和 Runtime 仍由唯一执行进程处理。它不是多个完整业务 API 实例，也不是多主机调度或执行进程自动故障切换。

此前仅有两个独立 SQLite 客户端的竞争测试，不能证明完整 API 多 worker 可用。直接使用 `uvicorn web_app:app --workers 2` 会让每个进程创建 Runtime、控制器和恢复流程；第二个进程打开会话仓库时可能把另一个进程的活动任务标为待恢复。OAuth PKCE 临时状态和异步启动票据也原来各自留在进程内。因此该启动方式不能用于多 worker 部署。

## 实现边界

| 组件 | 职责 | 不承担的职责 |
| --- | --- | --- |
| `multi_worker_api:create_app` | 无 Runtime 的 ASGI 工厂；保留原 Host、Cookie、状态码、重复响应头和流式响应；转发到固定回环地址 | 不创建控制器、不独立认证、不判定任务完成、不自动重试写请求 |
| `single_executor:create_app` | 先取得状态目录的操作系统排他锁，再创建唯一 Runtime 和既有业务 API；锁覆盖进程生命周期 | 不允许第二个执行进程打开同一 Runtime；不把历史回执当作业务成功 |
| `dispatch.sqlite3 / dispatch_tickets` | API 派发前写入持久票据，执行端原子领取；绑定请求、身份材料及幂等键的摘要 | 不保存 Cookie、令牌、请求正文；不重新构造或重放请求 |
| `async_start_receipts` | 持久化异步启动状态、预分配 session ID 和结果；跨 worker 查询、重启后查询 | 不序列化 Python 闭包或认证材料，不在重启时自动运行旧操作 |
| 既有 Runtime / Control Plane | 唯一任务状态、权限、设备所有权、配额、幂等、审计和恢复权威 | 不由入口票据代替业务事务或 Verifier |

这里的持久命令通道是“持久派发票据 + 原请求传输”，覆盖非认证 `/api/` 写请求；读取和 `/api/control-plane/auth/` 认证路由直接转发到唯一业务进程。它不承诺入口进程在发送前死亡后还能自动送达，也不是离线积压任务系统。执行前再次校验排队请求的会话、成员权限和设备归属，沿用既有权限合同。

票据状态：

- `queued`：请求尚未被执行端领取。
- `dispatching`：执行端已领取，业务处理可能已开始。
- `responded`：HTTP 响应回执已落库；包括业务拒绝/错误，不代表任务成功。
- `not_dispatched`：连接前失败，或执行进程重启时发现未领取票据。
- `unknown`：执行端中断且未记录完整回执；必须结合关联 Run、事件和现有恢复合同判断，禁止自动重复动作。

执行进程重启时，异步启动票据区分 `async_start_not_dispatched` 与 `async_start_outcome_unknown`，保留 session ID。session ID 预分配不代表 Run 已创建。API worker 的重启不会触发 Runtime 恢复；执行进程重启则由既有 SQLite Runtime 合同标记活动 Run 待恢复。

OAuth start/callback 可以经过不同入口，但都到达唯一执行进程。执行进程重启时尚未完成的 OAuth 流程需要重新开始；已有持久化登录会话仍由原会话仓库验证。

## 验收方法与证据

专项测试实际启动两个独立 Uvicorn API 进程和一个执行进程，使用真实本地 TCP、真实路由、官方 OpenAPI 客户端、SQLite 和 Runtime。测试身份、GitHub 上游、模型、设备及画面是隔离夹具；没有外部账号授权、付费调用或真实手机动作。

覆盖项：

1. 不同 API PID、不导入 `web_app`、共享引导信息与控制面状态。
2. worker A 发起 OAuth、worker B 回调、Cookie 跨 worker 使用、退出后另一入口拒绝旧会话。
3. 相同幂等键竞争、不同请求体冲突、版本 CAS 竞争。
4. 成员降权/停用跨入口生效、保留最后一名所有者、排队期间撤销会话后禁止派发。
5. A 创建任务、B 查询和取消；其他用户不可见；事件游标无重复。
6. 同模拟设备竞争只有一个任务；不同模拟设备独立运行；最后一个配额竞争只有一个成功。
7. API 进程死亡并替换后活动 Run 保持原状态，不被误判为待恢复。
8. 动作执行期间可从另一入口请求暂停，记录当前结果后停止后续动作。
9. 第二执行进程在导入 Runtime 前被排他锁拒绝。
10. 分别中断同步执行和异步启动的执行进程，重启后回执/Run 关联保持，恢复操作不重放已发生的模拟动作。
11. 并发读写压力检查、票据单次领取、摘要绑定、故障分类、异步票据保留及失败提交清理。
12. 受控重载保留工厂入口和配置端口；待启动任务也计入重载忙状态。

暂停测试最初使用了动作后立即成功的单步夹具，错误期待“已成功”再变“暂停”；按现有终态合同改为仍有下一步的样本。重启夹具最初为已停用成员签发会话而失败，已改为只给启用成员签发。这两项未通过放宽生产合同解决。

测试命令（在 `poc` 目录，所有数据库为临时目录）：

```powershell
python -m unittest test_command_inbox test_durable_async_tasks test_multi_worker_http -v
python scripts/run_website_isolation_tests.py
python scripts/run_website_isolation_tests.py --full
npm test
```

最终计数和压力规模在本文件的“最终结果”中登记。默认压力样本为 4 个并发客户端完成 100 轮创建/读取；本次最终全量回归在命令前设置 `$env:MULTI_WORKER_SOAK_ROUNDS='1000'`，扩展为 1,000 轮。这不是生产长时间压力测试。

## 启动与上线步骤（尚未在公网执行）

前提：同一主机、同一服务账号、同一现有 Runtime 数据目录、共享本机 SQLite；Nginx 继续代理 API 入口。先确认真实部署路径，不把新空目录冒充原状态目录。

两个服务共同设置：

```text
PHONE_AGENT_STATE_DIR=<现有 WEB_OUTPUT_DIR 的绝对路径>
PHONE_AGENT_EXECUTOR_URL=http://127.0.0.1:8767
ROBOT_WEB_NO_BROWSER=1
```

执行服务另设 `ROBOT_API_HOST=127.0.0.1`、`ROBOT_API_PORT=8767`，使受控自重载保持执行端端口；保留当前已核对的 OAuth、PUBLIC_BASE_URL、模型配置、设备注册和租约目录。入口服务不需要模型/OAuth 密钥；两类进程都须能访问同一 `dispatch.sqlite3`。

从 `poc` 目录分别启动（端口仅为规划示例，上线前先核对占用）：

```text
python -m uvicorn single_executor:create_app --factory --host 127.0.0.1 --port 8767 --workers 1 --proxy-headers --forwarded-allow-ips 127.0.0.1
python -m uvicorn multi_worker_api:create_app --factory --host 127.0.0.1 --port 8765 --workers 2 --proxy-headers --forwarded-allow-ips 127.0.0.1
```

切换前分别在两套服务环境运行只读预检；它不会导入 Runtime、连接公网或打印 Secret：

```text
python evals/check_multi_worker_deployment.py --role owner --output /var/lib/phone-agent/preflight-owner.json
python evals/check_multi_worker_deployment.py --role ingress --output /var/lib/phone-agent/preflight-ingress.json
```

owner 预检必须看到 `owner_mode=pass`；入口预检必须看到 `executor_url=pass`、`port_separation=pass` 和 `ingress_mode=pass`。如果已有 `runtime.sqlite3` 却没有 `dispatch.sqlite3`，预检会阻断，必须先由 owner 初始化派发库并完成备份。

1. 在预发布环境复用部署 OS、Nginx 和服务账号；核对精确 Git 版本及现有状态路径、权限、端口。
2. 停止接收新任务，等待执行及异步启动退出，列出 unresolved Run；一致性备份 Runtime、证据、配置和派发数据库。运行中的 SQLite 使用 backup API，不能只复制主文件而遗漏 WAL。
3. 停止原 `web_app:app` 服务后再启动唯一执行进程，然后启动 API 入口。旧入口不参与新 owner 锁，禁止同时运行两种模式。
4. 通过官方客户端核对 OpenAPI、实际版本、会话、任务与审计。先做只读检查；切流后完成真实网站登录、刷新、退出和已授权任务的验收。
5. 故障演练、持续压力和部署平台的服务管理重启验证通过后，才可把线上多 worker 标为完成。

不可直接复用给全部进程加载 `web_app` 的旧 systemd 命令；也不可将执行端端口暴露公网。反向代理信任范围只包含实际受控代理。当前实现没有跨主机共享 SQLite、WebSocket 转发或执行端自动故障切换合同。

## 数据保留和回滚

派发数据库属于新的运行数据，需要加入部署备份/恢复清单。`dispatch_tickets` 在 owner 启动时清理超过 14 天的 `responded/not_dispatched`，保留 `unknown`。异步终态回执按既有 TTL 清理，活动和恢复未决回执保留；未决回执不会因 Run 恢复而自动删除。当前不是在线定时清理任务，持续运行的数据增长需在预发布压力验收中核对。

回滚先停止入口接单，等待当前操作退出；如必须强制停止 owner，按未知结果恢复合同处理在途动作。确认 owner 已退出后才能恢复原单 worker 启动命令，保留原 Runtime 库和新派发库，不删库、不回放旧请求、不同时启动新旧执行权威。旧版本不提供新增异步回执查询，因此回滚前导出未决票据与 Run 的对应关系用于核对。

## 最终结果

- Python 全量：**1,062 项，1,058 通过、4 项跳过，0 失败**，208.986 秒。跳过项为 1 项外部源文件缺失及 3 项仅针对非 Windows/Linux 的环境测试，未算通过。初轮唯一失败是根目录架构入口白名单未登记新增两个组合入口；同步清单后全量重跑通过。
- 新增专项包含 **16 项真实本地 HTTP 多进程测试、5 项派发票据测试、5 项异步持久回执测试**；另补派发前撤销、重载入口和未启动任务重载保护回归，均纳入上述全量。
- 网站权限/官方客户端定向回归：94 项通过。
- 前端：`npm test` **64 项通过**，包含单元及浏览器合同测试；本批未改网页布局，不把 Mock 浏览器合同当成公网真实登录。
- 压力样本：4 个并发客户端、两个 API worker、一个执行进程，**1,000 轮创建/读取通过**；压测结束时所有已写派发票据均为 `responded`。读请求不建立票据，`responded` 只代表 HTTP 回执完整。
- 本地原始日志：`poc/output/multi-worker-full-python.log`、`poc/output/multi-worker-frontend.log`（忽略目录，不提交运行数据）。差异格式检查通过。
- 加载状态：只启动并销毁了隔离测试进程；未重载公网/原本机服务。没有真实设备动作、真实外部 OAuth 或付费模型调用。

本批不代表完整公共 GA 完成。公网多 worker 切换与持续压测、第二真实 GitHub 账号、第二台真机以及外部告警真实投递仍须各自验收。
