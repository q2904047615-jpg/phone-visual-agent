# 夹具与历史证据

当前自动回归的正式输入主要在 test_fixtures/、frontend_contract_fixtures/ 和 evals/（相对 poc）。

本目录 semantic_ir/role_aware_risk_cases.json 是旧语义风险台账引用的原始证据，无运行或测试加载。保留用于核对历史，不恢复为 TaskSemanticIR 运行入口或当前风险策略。

原 universal_agent 的两个 metadata.json 和说明没有任何测试读取，且说明错误要求动作后 DeepSeek 重规划，已删除。实际合成图由 test_universal_agent_mock_loop.py 的 synthetic_frame 现场生成，测试完整保留。三份原文件已保存于 ../output/final_project_cleanup_20260906/before/poc/fixtures/universal_agent/，扩展名为 .snapshot，SHA-256 清单在证据目录根部。
