# 通用 Agent 合成夹具

这些夹具只描述通用页面变化，不包含任何真实 App 截图、品牌素材或固定业务流程。测试运行时由 Pillow 生成 540×960 的简单 UI 画面。

- `navigation_open/metadata.json`：入口页经一次通用语义点击变为详情页，影响等级为 `navigation_only`。
- `navigation_back/metadata.json`：内容页经一次系统返回动作变为上一层页面，影响等级为 `navigation_only`。

每条成功路径都应严格执行一个物理动作，随后使用四张新的稳定画面建立可信观察并触发 DeepSeek revision 增加。非导航控件应在本地策略处零动作阻断；动作后画面不稳定时允许记录已发生的一次动作，但不得重试。
