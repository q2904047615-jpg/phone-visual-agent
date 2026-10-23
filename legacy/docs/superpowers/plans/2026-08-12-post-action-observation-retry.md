> 历史/参考材料，不是当前运行权威。现行结论见[项目交接文档](../../../项目交接文档.md)，产品方向见[项目最终目标](../../../项目最终目标.md)。正文保留用于追溯，不据此恢复旧代码或执行旧步骤。

# Post-Action Observation Retry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover once from a Qwen JSON/protocol failure during post-action verification by recapturing fresh frames, without ever repeating the physical action.

**Architecture:** Keep `GenericSceneObserver` responsible for one observation and its internal compact repair. Extend `GenericSingleActionAdapter` so its existing bounded post-action observation loop treats only classified format failures as retryable, gives each observation attempt its own frame-stability deadline, and aggregates evidence across attempts.

**Tech Stack:** Python 3, `unittest`, Pillow images, existing `qwen_runtime_errors` classification and universal action controller.

---

## File map

- Modify `poc/test_generic_step_planner.py`: add adapter-level regression fixtures and three safety tests.
- Modify `poc/generic_action_adapter.py`: classify post-action observation errors and consume at most one remaining observation attempt without invoking the robot again.
- Verify `poc/test_generic_scene_observer.py`, `poc/test_universal_agent_orchestrator.py`, and `poc/test_web_platform.py`: ensure the surrounding observer, orchestrator, and HTTP safety contracts remain intact.

### Task 1: Add failing adapter recovery tests

**Files:**
- Modify: `poc/test_generic_step_planner.py`
- Test: `poc/test_generic_step_planner.py`

- [ ] **Step 1: Make the existing fake observer able to raise queued errors**

Change the fake's `observe` method to return scenes normally but raise queued exceptions:

```python
def observe(self, *, frames, goal_context=None):
    self.calls += 1
    result = self.scenes.pop(0)
    if isinstance(result, BaseException):
        raise result
    return result
```

- [ ] **Step 2: Write the recovery test**

Add a test that queues confirmation scene, malformed-JSON error, and successful post-action scene. Use twelve stable frames and an evidence directory. Assert one robot action, three observer calls, twelve captures, eight action-after evidence paths, and four final verified frames.

```python
def test_post_action_format_failure_recaptures_once_without_repeating_action(self):
    planned = scene("planned")
    fresh = scene("before", element_id="fresh")
    after = scene("after", screen_id="app_home", element_id="after")
    observer = FakeSceneObserver(
        [fresh, VisionAgentError("模型返回的 JSON 无法解析"), after]
    )
    robot = FakeRobot()
    capture = SequenceCapture(["gray"] * 12)
    adapter = GenericSingleActionAdapter(
        capture=capture,
        observer=observer,
        robot=robot,
        frame_interval=0,
        post_action_settle=0,
        post_action_timeout=1,
    )
    action = SemanticAction(
        node_id="generic_step_1",
        action="tap_semantic",
        params={"element_id": "e1", "target": "app_icon"},
    )
    with tempfile.TemporaryDirectory() as temp:
        result = adapter.execute(
            requested_action=action,
            planned_scene=planned,
            goal=goal(),
            confirmed=True,
            evidence_dir=Path(temp),
        )
    self.assertEqual(robot.actions, [("tap", 300, 400)])
    self.assertEqual(result.physical_actions, 1)
    self.assertEqual(observer.calls, 3)
    self.assertEqual(capture.calls, 12)
    self.assertEqual(len(result.evidence), 12)
    self.assertEqual(len(result.after_frame_paths), 4)
```

- [ ] **Step 3: Write the exhausted-format-retry safety test**

Queue two format failures after the confirmation scene. Assert failure reports one physical action, both post-action frame sets are retained, and the robot still has one action.

```python
def test_two_post_action_format_failures_stop_after_one_robot_action(self):
    observer = FakeSceneObserver(
        [
            scene("before", element_id="fresh"),
            VisionAgentError("模型返回的 JSON 无法解析"),
            VisionAgentError("模型返回的 JSON 无法解析"),
        ]
    )
    robot = FakeRobot()
    adapter = GenericSingleActionAdapter(
        capture=SequenceCapture(["gray"] * 12),
        observer=observer,
        robot=robot,
        frame_interval=0,
        post_action_settle=0,
        post_action_timeout=1,
    )
    with tempfile.TemporaryDirectory() as temp:
        with self.assertRaises(GenericActionAdapterError) as caught:
            adapter.execute(
                requested_action=SemanticAction(
                    "generic_step_1",
                    "tap_semantic",
                    {"element_id": "e1", "target": "app_icon"},
                ),
                planned_scene=scene("planned"),
                goal=goal(),
                confirmed=True,
                evidence_dir=Path(temp),
            )
    self.assertEqual(caught.exception.physical_actions, 1)
    self.assertEqual(len(caught.exception.evidence), 12)
    self.assertEqual(observer.calls, 3)
    self.assertEqual(len(robot.actions), 1)
```

- [ ] **Step 4: Write the non-format failure test**

Queue a timeout after the confirmation scene. Assert it fails immediately with two observer calls and eight captures.

```python
def test_post_action_non_format_failure_is_not_retried(self):
    observer = FakeSceneObserver(
        [scene("before", element_id="fresh"), VisionAgentError("请求超时")]
    )
    robot = FakeRobot()
    capture = SequenceCapture(["gray"] * 8)
    adapter = GenericSingleActionAdapter(
        capture=capture,
        observer=observer,
        robot=robot,
        frame_interval=0,
        post_action_settle=0,
        post_action_timeout=1,
    )
    with self.assertRaises(GenericActionAdapterError) as caught:
        adapter.execute(
            requested_action=SemanticAction(
                "generic_step_1",
                "tap_semantic",
                {"element_id": "e1", "target": "app_icon"},
            ),
            planned_scene=scene("planned"),
            goal=goal(),
            confirmed=True,
        )
    self.assertEqual(caught.exception.physical_actions, 1)
    self.assertEqual(observer.calls, 2)
    self.assertEqual(capture.calls, 8)
    self.assertEqual(len(robot.actions), 1)
```

- [ ] **Step 5: Run the three tests and verify RED**

Run:

```powershell
python -m unittest `
  poc.test_generic_step_planner.GenericActionAdapterTests.test_post_action_format_failure_recaptures_once_without_repeating_action `
  poc.test_generic_step_planner.GenericActionAdapterTests.test_two_post_action_format_failures_stop_after_one_robot_action `
  poc.test_generic_step_planner.GenericActionAdapterTests.test_post_action_non_format_failure_is_not_retried -v
```

Expected: the first recovery test fails because the adapter immediately wraps the first JSON error; the exhausted-format test observes only one post-action attempt instead of two. The non-format test already passes and protects the no-retry boundary.

### Task 2: Implement bounded classified re-observation

**Files:**
- Modify: `poc/generic_action_adapter.py`
- Test: `poc/test_generic_step_planner.py`

- [ ] **Step 1: Import the existing shared error classifier**

```python
from qwen_runtime_errors import FORMAT_ERROR_TYPES, classify_qwen_error
```

- [ ] **Step 2: Add a focused predicate**

Add this method to `GenericSingleActionAdapter`:

```python
def _post_observation_retryable(self, error: RuntimeError) -> bool:
    diagnostics = getattr(self.observer, "last_diagnostics", {})
    error_type = (
        diagnostics.get("error_type")
        if isinstance(diagnostics, dict)
        else None
    )
    return (error_type or classify_qwen_error(error)) in FORMAT_ERROR_TYPES
```

- [ ] **Step 3: Give each post-action observation its own frame deadline**

Move the frame acquisition deadline inside the loop:

```python
for attempt in range(1, self.post_action_max_observations + 1):
    attempt_deadline = time.monotonic() + self.post_action_timeout
    frames, paths = self._capture_stable_post_action_frames(
        deadline=attempt_deadline,
        evidence_dir=evidence_dir,
        prefix=f"{evidence_prefix}_after_attempt_{attempt}",
    )
```

The bounded attempt count remains the global safety limit; the provider retains its own request timeout.

- [ ] **Step 4: Continue only for classified format failures**

Replace the immediate wrap in the observer `except` block:

```python
except RuntimeError as exc:
    last_error = exc
    if (
        attempt >= self.post_action_max_observations
        or not self._post_observation_retryable(exc)
    ):
        raise GenericActionAdapterError(
            f"通用页面观察失败：{exc}",
            evidence=all_paths,
        ) from exc
    continue
```

Keep the physical action outside this loop and unchanged.

- [ ] **Step 5: Use the current attempt deadline for page-transition retry**

In the `UniversalActionError` branch, compare and sleep against `attempt_deadline`. This preserves the existing bounded transition behavior without sharing an expired deadline between independent observations.

- [ ] **Step 6: Run the focused tests and verify GREEN**

Run the Task 1 command again.

Expected: `Ran 3 tests ... OK`; all robot-action assertions remain exactly one.

- [ ] **Step 7: Run the complete adapter test class**

Run:

```powershell
python -m unittest poc.test_generic_step_planner.GenericActionAdapterTests -v
```

Expected: all adapter tests pass.

- [ ] **Step 8: Commit the implementation**

```powershell
git add -- poc/generic_action_adapter.py poc/test_generic_step_planner.py
git commit -m "fix: retry malformed post-action observations"
```

### Task 3: Verify surrounding contracts and runtime readiness

**Files:**
- Verify: `poc/test_generic_scene_observer.py`
- Verify: `poc/test_universal_agent_orchestrator.py`
- Verify: `poc/test_web_platform.py`
- Verify: `poc/test_generic_step_planner.py`

- [ ] **Step 1: Run targeted surrounding suites**

```powershell
python -m unittest `
  poc.test_generic_scene_observer `
  poc.test_generic_step_planner `
  poc.test_universal_agent_orchestrator `
  poc.test_web_platform -v
```

Expected: all tests pass with no physical hardware access; the suites use fake controllers and providers.

- [ ] **Step 2: Run the complete Python regression suite**

```powershell
python -m unittest discover -s poc -p "test_*.py"
```

Expected: all tests pass and the total is at least the previous 553 tests plus the three new cases.

- [ ] **Step 3: Check diff hygiene and forbidden specialization**

```powershell
git diff --check HEAD~1..HEAD
git diff HEAD~1..HEAD -- poc/generic_action_adapter.py poc/test_generic_step_planner.py
```

Expected: no whitespace errors; production diff contains no App names, fixed coordinates, or second robot invocation.

- [ ] **Step 4: Restart only the local web service and run read-only readiness checks**

Restart the current branch's web service so it loads the new code, then call `/api/device` and an observation-only/preflight endpoint. Do not call `/confirm`, do not invoke the robot, and report the live browser state separately from the code regression result.

- [ ] **Step 5: Commit any plan checkbox updates only if the repository convention tracks them**

No production change is required in this step. Leave runtime logs and generated evidence untracked.
