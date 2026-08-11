# 手机视觉离线回放题库

这个题库用于在修改视觉提示词或相关状态规则之前，先用历史截图做回归。
回放只调用千问视觉模型，不操作 `main.exe`、机械臂或真实手机。

`history/history_index.json` 收录所有历史报告中实际保存的截图。`review_status=golden`
的图片已经人工确认并进入回放门禁；`success_candidate` 和 `failure_candidate` 只进入
待审核队列，不会把历史模型回答直接当作正确答案。

## 门禁顺序

1. 先运行现有单元测试。
2. 用 `--validate-only` 检查题库和图片，不产生 token。
3. 用 `--tag` 只回放本次改动涉及的题目。
4. 受影响题目全通过后，再跑全部题目。
5. 全部通过后，只允许进行一次“不发送”的实机测试。

## 命令

在 `poc` 目录运行：

```powershell
.\.venv\Scripts\python.exe replay_vision_eval.py --list
.\.venv\Scripts\python.exe replay_vision_eval.py --validate-only
.\.venv\Scripts\python.exe replay_vision_eval.py --run-model --tag pinyin
.\.venv\Scripts\python.exe replay_vision_eval.py --run-model
.\.venv\Scripts\python.exe build_vision_history_index.py
.\.venv\Scripts\python.exe build_vision_review_queue.py
```

只有显式加入 `--run-model` 才会调用千问并产生 token。相同代码、题目、图片和模型
会复用 `cache` 中的结果；修改 `vision_agent.py` 后缓存自动失效。

`build_vision_review_queue.py` 使用保守的近重复阈值整理待审核截图，并生成
`history/review_queue.html`。聚类只用于把相似画面放在一起查看，页面会列出组内
所有成员；不会把代表图标签自动传播给其他截图。

## 标注规则

- `accepted` 可以列出多个安全答案，命中任意一个即通过。
- `forbidden_actions` 命中即失败。
- `fields` 按原值精确比较。
- `normalized_fields` 会忽略拼音中的空格、单引号和大小写。
- `nested_fields` 用点号检查嵌套字段，例如 `keyboard_layout.type`。
- `coordinate_region` 只检查坐标是否落在安全区域，不要求完全相同的像素。

历史报告只保存了每一步最后一帧，所以首批题目会把同一张图复制为四帧，并按线上
阶段生成相同的候选栏或输入框第五张放大图。后续采集应保存真实四帧，以便题库覆盖
页面稳定性判断。
