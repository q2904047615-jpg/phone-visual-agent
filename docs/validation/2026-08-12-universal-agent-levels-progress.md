# 通用手机视觉 Agent 后续交付层级验证记录

日期：2026-08-12
分支：`codex/complete-universal-agent`
最高依据：根目录 `项目最终目标.md`

## 当前结论

本轮已把后续层级的代码主路径接入同一个通用编排器：通用文字输入、长按和拖动协议；外部状态的风险范围确认与具体动作确认；动作后重新观察和 DeepSeek 新 revision；有界低风险连续推进；网页控制台；三种合成 App 的陌生命令与改写验收；按 `device_id` 隔离的控制窗口、标定、会话租约和物理动作租约。

这份记录区分代码能力、模拟验证与真机验证。没有真实硬件证据的项目不标记为真机完成。

## 已通过的代码与模拟能力

- DeepSeek 仍只产生动态高层任务图，不包含 App 流程或坐标。
- Qwen 每轮只返回一个绑定当前观察的动作，支持点击、关闭弹层、四向滑动、返回、等待、已聚焦输入、长按和双候选拖动。
- 输入文字只能逐字来自 `goal.entities.input_text`；模型不能自行发明要输入的内容。
- 外部状态先确认 `task/device/revision/subgoal/risk_ids`，这一步不拍摄、不调用 Qwen、不执行机械臂；随后才产生绑定 `observation_id/fingerprint` 的具体动作并要求第二次确认。
- 风险确认和动作确认均一次性消费，不能跨任务、设备、revision、子目标、风险或画面复用。
- 安全连续推进只允许 `read_only/navigation_only` 的点击、关闭、滑动、返回和等待；最多 8 个物理动作、16 轮观察；输入、长按、拖动、外部状态、未知影响、失败或重复画面立即暂停。
- 每个物理动作仍由单动作适配器执行，随后重新采集稳定画面、验证变化并要求 DeepSeek revision 增加。
- 网页显示动态计划、风险范围确认、具体动作确认、连续安全导航入口、证据、暂停和取消。
- 三个合成 App 以同一编排器完成点击、返回和滑动；两种不同措辞复用同一 `tap_semantic` 协议；没有新增 App 名称分支。
- 设备注册表为每个 `device_id` 配置独立控制窗口和标定文件。相同设备跨进程互斥，不同设备使用不同物理租约；重复绑定同一窗口失败关闭。
- 跨进程反例已经覆盖：子进程持有设备 A 时，父进程重复申请 A 会失败，但可同时取得设备 B；两个设备使用不同租约文件。
- 每台设备显式声明 `verified_actions`。Qwen 的动作候选集合、本地策略和硬件适配器都只接受该设备已验证且有可调用方法的动作；协议支持不再等同于真机可用。
- 只读检查厂商控制端确认其摄像头区域内部存在“右键按下 → 移动 → 抬起”的通用触摸路径。代码已实现两点标定、插值移动和异常时强制抬起，但默认设备尚未把 `drag` 加入 `verified_actions`。

## 测试结果

Python 全量（加入安全语义重绑定与 Qwen 等价 JSON 结构反例后）：

```text
Ran 613 tests in 32.411s
OK
```

在线 Qwen 脱敏截图用例 `launcher_text_icon_single_action` 与
`settings_list_scroll_single_action` 已使用当前代码同批复跑通过：

```text
report_status=complete, passed=2, failed=0
hardware_actions_enabled=false
```

报告位于 `poc/output/offline_qwen_visual_decision/20260812_224044_945173/report.json`。
解析器兼容模型常见的 `type + params` 等价动作结构，但只保留当前动作类型
实际生效的本地参数。滑动距离提示和模型局部区域不进入控制器；`x/y`、裸坐标、
冲突字段和未知字段继续失败关闭。
Qwen 的 `foreground_app_id/new_foreground_app_id/new_screen_id/screen_change`
等已知预期结果别名会归一成控制器实际校验的 `app_id/screen_id/scene_changed`；
无法由控制器证明的自然语言结果键会在动作前被拒绝，不再退化成“只要页面有变化就算匹配”。

前端协议与真实浏览器契约：

```text
18 tests passed
```

浏览器契约验证了：风险确认不会发送动作确认字段；风险确认后才显示具体动作；安全导航可选择有界连续推进；外部状态不能出现连续推进入口。

## 真机当前状态

- 用户明确确认后，旧服务会话 `acc53be0caff41ccb6779db89cbaf062` 执行了 1 次“点击浏览器”物理动作。
- 动作前后画面仍是同一个 Android 桌面，浏览器没有打开；真实结果不通过。
- 动作前 fingerprint 为 `f7a89af49da849d7e8e4`，动作后为 `a7f9b74d9c5b769aa872`。fingerprint 虽变化，但可见元素、角色、语义和状态没有变化，证明单独依赖摄像头 fingerprint 会产生假阳性。
- 动作前证据：`poc/output/web/generic_supervised_20260812_180711_acc53be0/qwen_visual_revision_1_23637210a3b14a708afb2f3a53523a9f_before_4.jpg`。
- 动作后证据：`poc/output/web/generic_supervised_20260812_180711_acc53be0/qwen_visual_revision_1_23637210a3b14a708afb2f3a53523a9f_after_attempt_1_4.jpg`。
- 旧服务曾把 fingerprint 变化写成 `matched=true`，但 DeepSeek 没有结束子目标，新的精确确认门也阻止了自动重复点击；第二次动作未执行。
- 通用控制器现同时比较 fingerprint 和场景语义签名。语义未变化会记录为 `mismatched`、写入失败证据、触发 `action_result_mismatch` 重规划，并在安全循环中立即停止；不会重试机械臂。
- 旧会话已暂停并失效其第二次确认权限。新代码服务已启动，机械臂控制端、摄像头在线且当前不忙。
- 同一句陌生命令在新服务进行了多轮“只观察、零物理动作”的在线协议修复。修复均位于通用边界：空可选 `input_text` 归一为缺失；Qwen 动作/区域的已知字段别名与嵌套等价结构归一；目标区域及候选语义字段由本地可信观察构造；冲突字段仍失败关闭；“启动/launch/start”纳入通用导航词汇且外部状态禁词仍优先拦截。
- 会话 `fbe331637189465c852fbdc77734f83b` 的确认在执行前重新观察时安全停止，原因是同一浏览器图标被 Qwen 用另一种同义措辞描述；机械臂动作数保持 0，旧确认已经失效。
- 通用重绑定现只允许“未知页面信息变得更明确”和同一个低风险导航语义类别内的模型措辞变化，并继续逐项要求相同 label、role、states 与至少 0.60 IoU。`return -> save_and_return` 等风险语义变化仍以 0 动作拒绝。确认失败证据也会包含确认时重新采集的四帧。
- 修复并重启后，新会话 `bbdcfc2253c24552be0f43104e77498a` 已停在 `awaiting_confirmation`，物理动作数为 0，动作是绑定可信候选 `browser_app_icon` 的 `tap_semantic`；revision 为 1，observation 为 `obs_3680de47e9c74697b963133c0617ac63`，fingerprint 为 `d9317f1ca5b56ee43b3b`。
- 网页只读实测已从运行服务恢复该会话，展示自然语言入口、动态任务图、当前真实观察、0 个物理动作、精确确认、重新观察、暂停和取消入口；未点击任何执行按钮。
- `/api/device` 的产品主路径标识已改为 `universal_agent`。旧队列 worker、旧 generic orchestrator 和旧语义适配器只标记为 `compatibility_only/default_user_path=false`，不再把保留的 `legacy` worker 模式误报为网页主路径。
- 多设备协调锁已从全局锁拆为按 `device_id` 隔离：不同设备可同时取得各自的进程租约、协调锁和控制器锁；同一设备的第二次占用仍失败关闭。两个不同设备也可同时保持独立活动会话和确认范围。

## 尚需真实硬件完成的验收

- 浏览器点击已有一次真实失败证据；最新会话已经重新形成合法确认门，仍需针对该新会话的精确确认、成功页面变化和 DeepSeek 新 revision 才算通过。更早确认不能复用。
- 真实输入和真实长按各一次；每次都要单独确认并保存前后证据。
- 任意两点拖动代码路径已经接入，但尚未进行一次受监督真机动作和动作后画面验证；当前设备的 `verified_actions` 因此不含 `drag`，Qwen 不能提出该动作。
- 至少三个真实 App 的陌生命令、多步路径变化和结果验证。
- 第二套手机、机械臂控制窗口和独立标定接入后，进行两设备并行及同设备互斥的真机验收。

因此当前可以声明“后续层级的通用代码骨架和离线/模拟验收已完成”，不能声明“最终目标已在真实多手机上全部完成”。
