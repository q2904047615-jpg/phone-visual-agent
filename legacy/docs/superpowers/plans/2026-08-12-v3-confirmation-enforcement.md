> 历史/参考材料，不是当前运行权威。现行结论见[项目交接文档](../../../项目交接文档.md)，产品方向见[项目最终目标](../../../项目最终目标.md)。正文保留用于追溯，不据此恢复旧代码或执行旧步骤。

# DeepSeek v3 Console and Scoped Confirmation Implementation Plan

> **ARCHIVED — DO NOT EXECUTE.** v3 confirmation and migration behavior are retired; current confirmation is typed effect confirmation under v4.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the universal console treat DeepSeek v3 as the formal shared protocol and enforce one-time confirmation scope on the FastAPI server immediately before one physical action.

**Architecture:** Keep all browser compatibility logic inside `protocol_adapter.js`, with v3 selected before explicit v2/legacy fallbacks. Add a server-owned v3 confirmation authority to `GenericSupervisedSession`; strict Pydantic request models and the existing session/hardware locks validate and atomically consume that authority before the adapter can execute. The existing backend remains fail-closed until a real DeepSeek/Qwen loop installs authoritative v3 context.

**Tech Stack:** JavaScript/Node test runner, Playwright, Python 3, FastAPI, Pydantic 1.x, unittest.

---

### Task 1: Real v3 cross-contract fixtures

**Files:**
- Create: `poc/frontend_contract_fixtures/deepseek_task_graph_v3.json`
- Modify: `poc/frontend_contract_fixtures/qwen_visual_decision_v2.json`
- Modify: `poc/test_frontend_protocol.js`
- Modify: `poc/test_frontend_browser_contract.js`

- [ ] **Step 1: Generate DeepSeek output from `438cd22`**

Run the real `DynamicTaskGraph.to_dict()` and `to_qwen_context()` serializers with `active_external_payload()` and record commit, serializer and exact output. Verify the protocol equals `2026-08-11-deepseek-task-graph-v3` and `confirmation_gate.scope` contains task, device, revision and subgoal.

- [ ] **Step 2: Generate the current Qwen decision output**

Run `QwenVisualDecisionObserver(...).decide(...).to_dict()` from `dca0df2` with matching task/device/revision identity and copy the complete serializer output, including trusted observation aliases/conflicts.

- [ ] **Step 3: Write failing v3 frontend tests**

Assert v3 is formal (`compatibilityFallback === false`), v2 is explicit compatibility data, the goal and gate scope render, the Qwen decision remains executable only for `action`, and automatic payloads contain neither `confirmed` nor `confirmation`.

- [ ] **Step 4: Run the Node tests and observe the v2/fallback failures**

Run: `node --test poc/test_frontend_protocol.js`

Expected before implementation: failures mentioning the v2 protocol expectation or missing v3 formal classification/scope.

### Task 2: Server-owned one-time v3 confirmation

**Files:**
- Modify: `poc/generic_supervised_runtime.py`
- Modify: `poc/web_app.py`
- Create: `poc/test_v3_confirmation_scope.py`
- Modify: `poc/test_web_platform.py`

- [ ] **Step 1: Write failing confirmation authority tests**

Create an offline session with a fake adapter, exact v3 task context and matching Qwen action. Test success once, replay, task/device/revision/subgoal/risk mutations, stale observation/decision, no authority, blocked/finished decisions, invalidation and cross-device requests. Every rejected case must retain zero adapter executions.

- [ ] **Step 2: Write failing FastAPI schema tests**

Assert missing `confirmation` returns a rejection with zero physical actions, extra top-level/nested fields are forbidden, and the automatic model has only `device_id` plus `max_physical_actions=1`.

- [ ] **Step 3: Run targeted Python tests and observe failures**

Run: `python -m unittest poc.test_v3_confirmation_scope poc.test_web_platform.WebPlatformTests.test_generic_supervised_auto_request_allows_exactly_one_physical_action`

Expected before implementation: missing authority APIs and permissive request-model failures.

- [ ] **Step 4: Implement the minimum authority boundary**

Add an exact v3 authority snapshot bound to session/task/device/revision/subgoal/risk IDs plus observation/fingerprint. Validate current state again while holding the session and hardware locks, mark the grant consumed before calling the adapter, and invalidate it on every terminal or state-changing path. Reject legacy/no-authority sessions rather than treating `confirmed=true` as permission.

- [ ] **Step 5: Implement strict request models and device checks**

Use Pydantic `extra = "forbid"`; declare the nested confirmation fields explicitly; remove confirmation fields from auto requests; bind start/next/auto/cancel/pause to the session device; return 409 with zero physical actions for semantic scope failures.

- [ ] **Step 6: Run targeted tests to green**

Run: `python -m unittest poc.test_v3_confirmation_scope`

Expected: all confirmation replay, TOCTOU and multi-device cases pass.

### Task 3: Unified v3 browser adapter and display

**Files:**
- Modify: `poc/static/protocol_adapter.js`
- Modify: `poc/static/app.js`
- Modify: `poc/test_frontend_protocol.js`
- Modify: `poc/test_frontend_browser_contract.js`

- [ ] **Step 1: Add formal v3 normalization**

Recognize only the exact v3 version as formal, normalize `goal.objective`, identities, current subgoal, risks and `confirmation_gate.scope`, and preserve exact v3 fields before any v2/legacy branch.

- [ ] **Step 2: Keep v2 and generic legacy explicit**

Mark v2 and older shapes as compatibility fallback without allowing legacy data to overwrite v3 fields.

- [ ] **Step 3: Remove root confirmation/device duplication**

Send only `confirmed` plus the nested confirmation object to the strict confirm endpoint. Keep device-scoped start/next/auto/cancel/stop and make pause clear browser grants and notify the server to invalidate its authority.

- [ ] **Step 4: Run protocol and Playwright tests**

Run: `node --check poc/static/protocol_adapter.js`, `node --check poc/static/app.js`, `node --test poc/test_frontend_protocol.js`, and `node --test poc/test_frontend_browser_contract.js`.

Expected: syntax and all browser contract tests pass without any real device.

### Task 4: Regression verification and commit

**Files:**
- Modify only files listed above if a regression reveals an in-scope issue.

- [ ] **Step 1: Run webpage API tests**

Run: `python -m unittest poc.test_web_platform`

- [ ] **Step 2: Run all runnable offline tests**

Run: `python -m unittest discover -s poc -p "test_*.py"`

Record historical missing-image failures separately; do not describe exclusions as a full pass.

- [ ] **Step 3: Review scope and repository state**

Confirm no App-specific UI or execution branch was added, no real-device command ran, current branch is `agent/universal-console`, and only intended files changed.

- [ ] **Step 4: Commit without merging or pushing**

Stage only this implementation and commit with a message describing v3 protocol and scoped confirmation enforcement. Re-run `git status --short --branch` and report the exact commit plus remaining backend/hardware boundaries.
