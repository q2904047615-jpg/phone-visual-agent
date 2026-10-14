# 任务级动作序列回放

这里保存跨 App 的离线只读回放样本，用于检测 `DeviceExecutor` 动作名称、参数和顺序是否发生回归，并输出可审计的任务成功率、动作匹配率和失败主分类。

它不会调用 DeepSeek、Qwen、真实控制端或机械臂，报告固定记录 `physical_actions=0`、`remote_model_calls=0`。因此通过只表示离线任务序列合同未回归，不能替代真机验收。

运行：

```powershell
python eval_task_sequences.py --manifest evals/task_sequences/cases.json --output output/task_sequence_benchmark.json
```

评测输出包含 `evaluation_protocol=2026-10-14-task-sequence-evaluation-v1`、`failure_categories`、每个 case 的 `failure_category/failure_code`，并固定声明 `physical_actions=0`、`remote_model_calls=0`。报告只证明离线动作序列和终态合同，不能替代视觉模型、真实 App 或机械臂真机验收。
