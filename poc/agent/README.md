# 轻量 DDD 模块化单体

本目录只承载已经迁移并被正式运行链调用的业务切片，不用空目录或转发包装制造“看起来像 DDD”的结构。

依赖方向：

- `domain`：会话聚合所需的最小合同、业务不变量和仓储端口；不依赖 Web、模型 SDK、摄像头或机械臂实现。
- `application`：编排一个用户用例；依赖 domain 端口，不决定 canonical 动作，也不直接访问 FastAPI。
- `infrastructure`：实现仓储等端口，由组合根注入 application。
- `web_app.py`：当前接口与组合根；只负责认证、Pydantic/HTTP 转换和依赖装配，并随迁移批次逐步变薄。

第一批迁移的是通用 Agent 会话生命周期。DeepSeek、Qwen、canonical action、Controller、typed input lineage 与物理执行仍使用现有正式模块；后续只有在一个真实切片完成迁移并删除旧入口后，才继续移动下一批。
