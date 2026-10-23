# 公网到 Windows 手机工具链：软件验收

日期：2026-10-23；实施工作树：f064；分支：codex/platform-policy-verifier-20261013。

## 结论与范围

**已验证可行（软件与隔离 HTTP 范围）。真实手机效果尚未验证。** 用户要求补齐公网到本机工具链，随后明确选择“先完成软件部分，手机稍后连接”。本批不操作手机，不修改卖家 main.exe 或机位下压值，不接入付费模型，不推送、不切换公网版本。

原问题：远程控制器已转发机械动作与图片，但 Runtime 的文字、可信 App 启动工厂仍创建云端本机 ADB 适配器，Linux 无法使用 Windows 的设备连接。Windows 的旧能力表还会对已通过云端授权的单次能力验收再次否决。

## 实现与保持的合同

- Runtime 对远程设备创建 RemoteTextTransport / RemoteAppLauncher；普通本机设备仍走原工厂。云端不执行 Windows ADB，也不读取云端路径作为远程手机的备用配置。
- 文字输入、换行、清空复用原 TextTransportActionScope、摘要和 TextTransportResult。Windows 只执行固定 ADB Keyboard 命令族；启动只执行本机注册表中的 launch_ref。没有任意 shell、任意包名、隐式清空或失败自动重输。
- 发现能力与派发绑定当前机位及本机配置实例；原 device_id 继续是云端授权资源，Windows 将机位映射到本机已启用设备配置。未登记机位不借用一号 serial；切机位或重启连接器后，旧绑定不能继续执行。
- 每个有副作用的工具调用消费本机持久观察记录；文字 scope nonce 也持久消费。命令重复只返回原回执，结果丢失只补交原结果，未知状态不重放。文字回执严格核对 device/action/nonce/operation，仍要求动作后视觉验证。
- 机械动作继续使用原单步凭据与几何检查。Windows 仅对当前认证命令临时授予其云端已批准能力，finally 恢复本机能力集合；不写入本机晋级结果，不建立第二审批权威。
- Windows 重启从本机相同 connector/device 的最新成功机位回执恢复状态，不重新发送选择；后续正式任务仍经过原 prepare_machine_position。观察与动作机位不一致时不得误操作另一台手机。
- 专用工具超时覆盖既有 ADB profile 的预检及广播上限。未领取为 not_dispatched，已领取但结果未确认仍为 unknown。

## 验证

使用临时 SQLite、Mock ADB、模拟机械控制器；无真实手机动作。

- 新增远程工具测试覆盖中文/emoji、换行、清空、可信启动、二号独立 serial、禁用及未登记机位、正文摘要/设备归属、配置变化、错误回执、超时、重启后 nonce 防重放、回执丢失补交，以及 Runtime 工厂不落回云端 ADB。
- 真实本地 TCP：两个无 Runtime 的 API worker、一个 Runtime owner、一个模拟 Windows worker。输入/换行/清空和 App 启动均经过签名 poll/result，核对只派发到二号 fixture serial。测试替代 TLS 终止，不冒充公网 TLS 或真实手机验收。
- 最终定向回归 71 项通过；前端 npm test 65 项通过。
- 最终完整 Python 回归 1,109 项：1,105 通过、4 skipped、0 failures / 0 errors（123.918 秒）。包含最后新增的机位恢复、失败和回执样本；此前首轮 1,104 项通过记录不冒充最终数量。
- 首次沙箱运行因临时目录/SQLite 权限失败；获得工具审批后在隔离临时目录重跑通过。新增重复投递测试最初未登记云端命令而补交回执，产生 fixture 409；修正夹具后通过，不归因为产品故障。

## 本机配置和加载准备

Windows CLI 增加 `--text-registry`、`--app-registry`，默认分别读取 `--registry` 同目录的 `adb_keyboard_registry.json`、`app_package_registry.json`。文件格式保持原合同；不上传 adb.exe 路径或可执行参数，不将私有配置提交 Git。

1. 设备注册表必须用 machine_position 唯一绑定同一控制窗口下的已启用设备；每台手机的 text/app profile 使用对应 device_id 和该手机自己的 serial。启用新设备须核对独立标定，不能仅改 enabled 冒充验收。
2. 本轮 f064 没有两份私有 ADB 配置。只读找到 6e64 旧的一号配置（微信、抖音直启），未复制、修改或扩展白名单。该 profile 只读状态为 device_offline；二号未配置。恢复手机连接后再绑定本机配置。
3. 无 ADB 配置时，机械动作与预览仍可用；输入和可信直启不会伪装为已就绪。远程手机配对在 Windows 主机完成；已有网站配对路由对远程设备返回明确设置提示，不在云端执行 adb pair。这是部署设置方式，不是新增任务确认规则。

协议升至 `2026-10-23-remote-device-connector-v3`，不允许 v2/v3 混用。两端必须同批升级：

1. 重新核对无活动 Runtime、无 queued/claimed 副作用命令、本机无未确认/未结束交付。停止本批连接器，释放设备租约；不操作卖家 main.exe。
2. 在线备份 Runtime/dispatch/connector 三库、本机交付库、受限配置及原发布指针。保留旧源码和旧发布目录；不得输出密钥。
3. 发布包包含根目录 trial.json、poc 源码及依赖声明。先在新发布目录运行隔离测试，再同批加载云端执行器/API 与 Windows v3 连接器。
4. 密钥复用现有受限渠道，仅在进程内加载，不出现在命令行或日志。启动前检查已有连接器进程及设备租约。
5. 使用官方 SDK bootstrap 核对 OpenAPI；核对服务实际 revision、签名轮询和机位/预览；没有手机连接时不得发送文字/启动命令冒充验收。
6. 回滚时停止本批连接器及项目服务，恢复原发布指针/配置并加载配套 v2 两端；保留新增账本。新表 used_tool_scopes 是增量结构，不能删除它或重放未确认命令。

本轮只读公网检查：current 仍为 `/opt/phone-agent/releases/20261023-21696e1/poc`，两项服务 active，Runtime 无会话，在途命令为空。本机交付库无未确认/未结束记录，连接器进程未运行。这些是检查时状态，不作为未来加载许可的空闲证据。

## 后置验收与剩余边界

手机连接后分别验证：当前目标机位 → 新观察 → 已授权单次输入/换行/清空/可信启动 → 新图；记录实际文字、页面和动作次数。不得把 broadcast/monkey 成功回执当作目标完成。

二号机位下压与标定、真实第二账号、常驻服务、密钥轮换、长期压力、外部告警不属于本批软件完成证据。当前仍未证明公网新版本无退化，尚未完成整套三 App/GA 验收。
