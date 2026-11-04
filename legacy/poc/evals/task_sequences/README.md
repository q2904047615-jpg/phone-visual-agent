# 任务级动作序列回放

这里保存跨 App 的离线只读回放样本，用于检测 `DeviceExecutor` 动作名称、参数和顺序是否发生回归。

它不会调用 DeepSeek、Qwen、真实控制端或机械臂，报告固定记录 `physical_actions=0`、`remote_model_calls=0`。因此通过只表示离线任务序列合同未回归，不能替代真机验收。

运行：

```powershell
python eval_task_sequences.py --manifest evals/task_sequences/cases.json --output output/task_sequence_benchmark.json
```
