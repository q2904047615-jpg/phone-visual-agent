# Companion IME 验收台账

更新时间：2026-08-30
状态：内部实现与完整自动回归完成；Android 构建和真机验收受环境阻塞
正式边界：`用户决策与协议边界.md` 第 54 节

## 1. 本轮唯一能力缺口

在不改变通用手机视觉 Agent、公开 API、canonical 动作和真机串行执行语义的前提下，为允许安装第三方输入法的 Android 设备增加一个项目自有 Companion IME 文字 transport，使现有 `input_verified_text` 与 `clear_verified_text` 能把已授权 Unicode 文本准确交给当前已聚焦的 typed 输入字段，并继续由动作后新截图完成精确验证。

本轮不新增动作类型，不让 Companion IME 观察页面、选择字段、规划任务或判断成功，也不恢复 ADB Keyboard、卖家文字对话框、剪贴板或 Accessibility 输入。

## 2. 必须出现的证据

### 2.1 合同与架构

- `项目最终目标.md`、`AGENTS.md` 与 `用户决策与协议边界.md` 只有一套正式文字 transport 边界。
- canonical 仍只有 `input_verified_text` 与 `clear_verified_text`；Qwen 不直接选择 transport。
- application 声明文字 transport 端口，infrastructure 实现；domain 不依赖网络、Android、Web 或设备 SDK。
- 每台设备显式配置一个文字 transport；一次动作只调用一个 transport，失败后不自动重试或切换。

### 2.2 授权与防重放

每个命令必须绑定并校验：

- `device_id`
- `session_id`
- `task_id`
- `revision`
- `action_id`
- typed `input_field_id`
- observation fingerprint
- prior / fragment / expected digest
- 有效期
- 单次 nonce

错误设备、错误字段、错误摘要、过期、重复 nonce、乱序 revision 或无当前编辑连接均须 fail closed。日志和回执不得记录输入正文、配对密钥或完整签名。

### 2.3 Android Companion IME

- 使用系统 `InputMethodService`，不注册公开文字广播入口。
- 只在当前编辑连接可用时执行 `finishComposingText()` 后的追加，或独立清空命令。
- 追加命令不得隐式清空或替换；清空命令不得携带新正文。
- 支持中文、拉丁字母、标点、Emoji、换行和长文本。
- 用户手动安装、启用、选择并配对；不静默安装或切换输入法。

### 2.4 PC 端与现有闭环

- 现有 typed 输入事务生成授权命令，设备 transport 只消费该命令。
- transport ACK 只证明命令被当前 IME 接受或拒绝，不证明输入成功。
- 动作后仍取得新四帧观察，以同一 `input_field_id` 的精确文本结构作为唯一成功证据。
- Companion 离线、未配对、无焦点或拒绝命令时报告具体运行故障，且不产生机械键盘回退。

### 2.5 测试与交付

- 正向：中文、Emoji、复杂标点、换行、长文本、连续追加、独立清空、多字段身份绑定。
- 变化：换设备、换字段、换 observation、换 revision、过期与重复 nonce。
- 失败：离线、未配对、ACK 超时、无编辑连接、摘要不一致，均无重试和回退。
- 相关 Python 测试与历史输入失败回放通过。
- Android 单元测试、lint 与 debug APK 构建通过。
- 批次稳定后只运行一次完整 Python 回归。
- 更新 README、交接和本台账，并创建一个本地提交；不推送。

## 3. 已有证据

- canonical typed 输入动作、typed `input_field_id`、prior/fragment/expected 事务和动作后精确验证已经存在。
- 当前输入正文唯一视觉权威已经固定为单步 `input_structure.application_inputs[*].text`。
- 当前闭环已经要求一次动作后重新观察，不允许输入动作自动重复。
- 用户已明确批准项目自有 Companion IME 方案，并接受首次安装、启用、选择和配对的设备条件。
- **已实现**：transport-neutral domain/application 合同、PC 端 Companion IME TLS transport、单次 scope 与
  nonce 防重放、Windows DPAPI 配对存储、同端口一次性配对、每设备组合根接线、本地 standalone setup CLI
  以及 Android `InputMethodService` 源码均已落盘。Companion 只消费当前 canonical
  `input_verified_text` / `clear_verified_text`，ACK 不替代动作后视觉验证；未新增 raw text HTTP API。
- **相关测试通过**：当前 Companion IME 相关 Python 测试运行 `449` 项，结果 `OK (skipped=1)`；
  `compileall` 退出码为 `0`；本机 Windows 当前用户 DPAPI 加密/解密 roundtrip 已实际通过。这些证据只证明
  内部合同、适配器、配对存储和组合根的自动验证，不等于 Android APK 或真机能力已验收。
- **未完成**：Android 源码中已有 `38` 个 JUnit 测试方法，但本机未发现 JDK、Gradle、Android SDK
  platform/build-tools/cmdline-tools 或 Android Studio，因此这些 JUnit 尚未运行，Android lint 与 debug APK
  构建也未进行；尚未生成 APK、安装、启用、配对或执行真机输入验收。
- 本批没有启动、停止或重载项目 API，没有启动、停止、重启或操作卖家 `main.exe`，也没有执行任何真机
  动作。

## 4. 当前唯一缺口

核心 Companion IME 链路、Android pending/active 双槽与 `pair_confirm -> promote -> pair_commit` 配对确认、
PC 仅在 commit 后落盘/启用、已有配对 repair 不隐式旋转 key，以及完整 envelope 发送前 frame 上限和
canonical 显式分段均已实现并通过相关 Python 测试；repair 还必须匹配原 `installation_id`，不同安装
实例不得共享同一 key。当前唯一缺口是运行 Android JUnit/lint/APK 构建（受本机工具链缺失阻塞），
随后安装、启用、配对并完成正式真机输入验收。内部实现、文档、相关回归与完整 Python 回归由本批同一
本地提交固化；Android 与真机证据未完成前，阶段状态不得写成真机验收完成。

## 5. 实施设计

```text
当前新截图 + 当前 typed 子目标
  -> canonical input_verified_text / clear_verified_text
  -> Controller 校验 scope、字段、设备与输入事务
  -> 当前设备唯一配置的 TextTransportPort
       - mechanical_keyboard
       - companion_ime
  -> Companion 一次性授权命令（仅当配置为 companion_ime）
  -> 当前 InputConnection 执行一次
  -> 最小 ACK
  -> 新四帧截图
  -> 现有同字段精确验证
```

配对建立设备密钥；运行命令使用短期、单次、带上下文摘要的认证信封。Transport 不能构造输入目标，也不能根据 ACK 推进任务。

## 6. 停止条件

- 若实现需要新增 App/截图/固定坐标特例，立即停止并撤回该方向。
- 若需要让 Companion 读取 Accessibility、屏幕内容或自行寻找字段，立即停止；这会形成第二权威。
- 若一个已尝试的输入命令失败或结果未知，立即停止该动作，不重发、不切换 transport。
- 若本机工具链仍缺失，Android 源码与合同测试可以完成，但必须把 APK 构建和真机安装验收标为环境阻塞，禁止宣称完成。
- 真机验收前必须重新核对设备、相机、控制器、busy、活动会话和当前 observation；不得操作卖家 `main.exe`。

## 7. 当前结论

内部机制：**已实现**核心 domain/application/infrastructure、Windows 配对存储、组合根、setup CLI、
手机 pending/active 配对提交、完整 frame 发送前校验、显式长文本分段与 Android 源码。
自动测试：**完整回归通过**，相关 Python `449 tests OK (skipped=1)`；最终全目录 Python
`981/981`（`116.544s`，`skipped=1`）；`compileall=0`、四份 Android XML 可解析、Windows DPAPI
roundtrip 通过。
Android：**未完成**，`38` 个 JUnit 方法尚未运行，未 lint、未构建 APK；本机缺少 JDK/Gradle/Android
SDK。
真机与交付：**未完成**，未生成或安装 APK，未配对、未执行真机输入验收；本批只创建本地提交、不推送，且未操作
项目 API、`main.exe` 或真机。
