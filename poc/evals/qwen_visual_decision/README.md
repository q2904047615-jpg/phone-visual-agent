# Qwen 保存截图评估

这里验证当前截图的页面、目标、输入事实、完成和唯一动作。不连接摄像头、网页服务、机械臂或 ADB Keyboard。注意：保存截图评估仍可能调用远程模型，不等于无网络测试。

## 当前合同与历史画面

评估调用正式观察器和单步解析，不另建模型协议。当前协议及点按/输入/finish 规则见 [根目录协议说明](../../../通用Agent协议说明.md)。点按只消费同帧 target/tap_point，输入只评分当前字段的正文、focused 和 preedit，不要求机械键盘布局或拼音模式元数据。

cases.json 的 24 个目标用例覆盖桌面/反光桌面、脱敏设置列表、符号输入弹层、拼音预编辑和裁切画面。图中出现旧键盘不表示仍支持机械文字：这些是只读视觉样本。没有清单引用的 motion_blur_after_pinyin.jpg 保留为历史模糊截图证据，不算当前覆盖。

页面语义仍可要求识别可见键盘或候选栏；这是图片内容评分，不生成选词动作、不启用机械输入。布局/模式旧字段评分已随正式字段退役；字段正文、焦点、预编辑错误仍计为失败。

## 无网络测试

在 poc 目录：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest test_qwen_offline_eval
```

此命令只验证清单、载图、评分和 Mock 请求，不调用真实模型。

## 真实模型评估

只有取得对应图片/远程调用授权后，在项目根目录运行：

```powershell
poc\.venv\Scripts\python.exe -X utf8 poc\eval_qwen_visual_decision.py --case launcher_settings_action
```

不指定 --case 会请求整个清单。每个目标只调用一次，不远程修复采样。--case-timeout-seconds、--suite-timeout-seconds 控制隔离进程期限；--resume-report 只复跑未通过样本，旧通过标记 prior_run_reference，不计入本轮新证据。

报告保留页面/目标/输入/finish/decision 评分与具体错误。工具测试通过不等于实际 Qwen 准确率通过，保存截图的模型结果不等于真机任务成功。密钥不写入报告。
