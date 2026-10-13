# 轻量 DDD 模块化单体

本目录只承载正式运行链实际调用的业务代码。依赖方向固定为
`interfaces -> application -> domain`；`infrastructure` 实现端口并由组合根注入。

当前运行链只有一条：

```text
整任务与本次实际动作/结果历史、新鲜截图进入 Runtime/Planner
→ Model Gateway（可用 Qwen）返回 scene/input_structure 与一个候选 action 或 finish
→ Planner 绑定 observation、scope/fingerprint 和模型建议，签发唯一 ToolIntent
→ 本地校验 device、scope/fingerprint、当前目标/字段身份、合法坐标、机械/可信 transport 和必要确认
→ 执行一次
→ 取得新截图并回到 Runtime/Planner
```

- `domain`：整任务上下文、当前 UI scene、canonical action、Controller 硬校验、瞬时 typed 文字事务、设备/会话/风险不变量。
- `application`：维护本次实际历史；把模型网关建议交给 Planner，签发一个 ToolIntent；编排一次观察、一次执行和下一张截图。
- `infrastructure`：Model Gateway（当前可用 Qwen）、新鲜截图观察、单动作 adapter、相机、机械臂、有界 ADB Keyboard、可信包名直启、租约和证据存储。
- `web_app.py`：HTTP/Pydantic 转换和依赖装配。

正式运行不存在 TaskSemanticIR、candidate selector、动作后 DeepSeek replan、持久视觉或输入
lineage、decision cache、App lineage 完成裁决、scene-only decision 回退或第二套动作权威。
旧任务图和子目标协议不进入运行兼容；模型的 finish 只能作为候选，整任务终态由 Runtime/Verifier 根据绑定的 ToolIntent 与新观察推进。
