> 历史/参考材料，不是当前运行权威。现行结论见[项目交接文档](../../../项目交接文档.md)，产品方向见[项目最终目标](../../../项目最终目标.md)。正文保留用于追溯，不据此恢复旧代码或执行旧步骤。

# Qwen DeepSeek v3 Contract Implementation Plan

> **ARCHIVED — DO NOT EXECUTE.** DeepSeek v3 and every migration step in this plan are retired. The only formal protocol is `2026-08-20-deepseek-typed-task-graph-v4`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make DeepSeek task-graph v3 the formal Qwen input protocol with exact confirmation scope validation and a safe read-only/navigation-only v2 migration path.

**Architecture:** `QwenTaskContext` keeps one strict top-level schema and dispatches confirmation-gate validation by protocol version. v3 validates exact scope and risk coverage; v2 never synthesizes scope and blocks external or unknown impact before observation. A generated fixture from DeepSeek commit `438cd22` anchors the cross-branch contract without modifying or copying the DeepSeek module.

**Tech Stack:** Python 3.12, dataclasses, unittest, JSON fixtures, existing Qwen offline evaluator.

---

### Task 1: Generate and lock the DeepSeek v3 contract sample

**Files:**
- Create: `poc/evals/qwen_visual_decision/deepseek_v3_contract_438cd22.json`
- Create: `poc/test_qwen_deepseek_v3_contract.py`

- [ ] **Step 1: Export commit source to a temporary directory**

Use `git show 438cd22:poc/deepseek_task_graph.py` and the existing `generic_intent.py` dependency. Do not alter the DeepSeek branch.

- [ ] **Step 2: Run the actual `to_qwen_context()` implementation**

Generate navigation, awaiting-confirmation, and confirmed contexts from the commit's own task graph classes and save them with:

```json
{
  "fixture_protocol": "deepseek-to-qwen-contract-fixture-v1",
  "source_commit": "438cd2258cdca681abe42da11b70c399df58063e",
  "source_method": "DynamicTaskGraph.to_qwen_context",
  "contexts": {}
}
```

- [ ] **Step 3: Write the first failing contract test**

The test must load the generated navigation context and call:

```python
parsed = QwenTaskContext.from_dict(sample)
self.assertEqual(parsed.protocol_version, "2026-08-11-deepseek-task-graph-v3")
```

- [ ] **Step 4: Run the test and verify RED**

Run:

```powershell
python -m unittest test_qwen_deepseek_v3_contract.py
```

Expected: fail because Qwen only supports v2 and rejects v3/scope.

### Task 2: Add strict v3 and constrained v2 parsing

**Files:**
- Modify: `poc/qwen_visual_decision.py`
- Test: `poc/test_qwen_deepseek_v3_contract.py`
- Test: `poc/test_qwen_visual_decision.py`

- [ ] **Step 1: Add failing v3 scope tests**

Cover exact gate shape, missing scope field, extra scope field, and task/device/revision/subgoal mismatches. Each malformed context must raise before provider calls.

- [ ] **Step 2: Add failing risk and confirmation tests**

Cover mismatched risk ID sets, invalid confirmed/allowed combinations, awaiting confirmation, and valid confirmed external state.

- [ ] **Step 3: Add failing v2 migration tests**

Assert v2 navigation/read-only parses, while v2 external/unknown returns a pre-observation block reason and never becomes externally allowed.

- [ ] **Step 4: Run targeted tests and verify RED**

```powershell
python -m unittest test_qwen_deepseek_v3_contract.py test_qwen_visual_decision.py
```

- [ ] **Step 5: Implement minimal version dispatch**

Define explicit constants for v3 default and v2 migration, keep strict top-level fields, and validate gate fields by version. Never synthesize a v3 scope for v2.

- [ ] **Step 6: Run targeted tests and verify GREEN**

Use the same unittest command and require exit code 0.

### Task 3: Move all offline cases to real v3 shape

**Files:**
- Modify: `poc/evals/qwen_visual_decision/cases.json`
- Modify: `poc/evals/qwen_visual_decision/README.md`
- Modify: `poc/test_qwen_visual_decision.py`
- Modify: `poc/test_qwen_offline_eval.py`

- [ ] **Step 1: Add failing manifest assertions**

Every case must use v3 and contain an exact scope matching its task, device, revision, and active subgoal.

- [ ] **Step 2: Run manifest/evaluator tests and verify RED**

```powershell
python -m unittest test_qwen_visual_decision.py test_qwen_offline_eval.py
```

- [ ] **Step 3: Update all seven contexts to v3**

Add exact scope to each gate; do not change page-specific action expectations or loosen candidate selection.

- [ ] **Step 4: Update protocol documentation**

Document v3 default, v2 migration-only behavior, exact scope, source fixture, and zero model calls for pre-observation blocks.

- [ ] **Step 5: Run tests and verify GREEN**

Use the same command and require exit code 0.

### Task 4: Regress all safety invariants

**Files:**
- Test: `poc/test_qwen_visual_decision.py`
- Test: `poc/test_qwen_deepseek_v3_contract.py`
- Test: `poc/test_qwen_offline_eval.py`

- [ ] **Step 1: Run the Qwen protocol suite**

```powershell
python -m unittest test_qwen_visual_decision.py
```

- [ ] **Step 2: Run evaluator and cross-contract suites**

```powershell
python -m unittest test_qwen_offline_eval.py test_qwen_deepseek_v3_contract.py
```

- [ ] **Step 3: Run explicit invariant tests**

Run the named tests for forged elements, modified bounds, stale revision/fingerprint/observation, multiple actions, and confirmation gate zero calls.

- [ ] **Step 4: Run complete offline discovery**

```powershell
python -m unittest discover -p "test_*.py"
```

Use the existing project virtual environment if the system Python lacks FastAPI or pypinyin.

### Task 5: Run a fresh online seven-case screenshot evaluation

**Files:**
- Runtime output only: `poc/output/offline_qwen_visual_decision/<run_id>/report.json`

- [ ] **Step 1: Run without resume**

```powershell
python eval_qwen_visual_decision.py --case-timeout-seconds 120 --suite-timeout-seconds 480
```

- [ ] **Step 2: Verify report provenance**

Require `report_status=complete`, `full_case_report_complete=true`, `current_run_case_count=7`, and `reused_prior_reference_count=0`.

- [ ] **Step 3: Record metrics and failures**

Read pass count, combined first-pass rate, repair retry rate, final blocked rate, and every failure's exact error type/reason.

### Task 6: Final verification and commit

**Files:**
- All files listed above.

- [ ] **Step 1: Inspect the diff**

Run `git diff --check`, scan additions for App-name branching/hardware enablement, and verify only intended files changed.

- [ ] **Step 2: Re-run fresh required tests**

Run protocol, evaluator, contract, and full offline suites after the final edit.

- [ ] **Step 3: Commit current branch**

Stage only the v3 contract implementation and commit on `agent/qwen-next-action`. Do not merge or push.
