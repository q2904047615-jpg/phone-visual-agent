from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Generator, Iterable

from task_orchestrator import PlanCondition, PlanNode, TaskPlan


MIN_OBSERVATION_CONFIDENCE = 0.72


class SemanticExecutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class PageObservation:
    """Device-independent page report consumed by the semantic controller."""

    page_state: str
    properties: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    stable: bool = True
    frame_id: str = ""

    def validate(self) -> None:
        if not self.page_state.strip():
            raise SemanticExecutionError("页面观察缺少 page_state。")
        if isinstance(self.confidence, bool) or not isinstance(
            self.confidence, (int, float)
        ):
            raise SemanticExecutionError("页面观察置信度格式无效。")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise SemanticExecutionError("页面观察置信度必须在0～1之间。")
        if not isinstance(self.stable, bool):
            raise SemanticExecutionError("页面稳定状态格式无效。")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class SemanticAction:
    node_id: str
    action: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActionResult:
    node_id: str
    success: bool
    observation: PageObservation
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StepDecision:
    status: str
    action: SemanticAction | None = None
    reason: str = ""
    counters: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "action": self.action.to_dict() if self.action else None,
            "reason": self.reason,
            "counters": dict(self.counters),
        }


class SingleStepSemanticExecutor:
    """Interpret a validated plan and expose exactly one action per advance.

    The executor knows no camera, model, coordinates, vendor process or robot
    controller.  A future adapter may execute the returned semantic action and
    must return a fresh PageObservation before the plan can advance.
    """

    def __init__(
        self,
        plan: TaskPlan,
        *,
        min_confidence: float = MIN_OBSERVATION_CONFIDENCE,
    ) -> None:
        plan.validate()
        self.plan = plan
        self.min_confidence = float(min_confidence)
        self.status = "ready"
        self.reason = ""
        self.counters: dict[str, int] = {}
        self.current_observation: PageObservation | None = None
        self.pending_action: SemanticAction | None = None
        self._runner: Generator[SemanticAction, ActionResult, None] | None = None

    def start(self, observation: PageObservation) -> StepDecision:
        if self.status != "ready":
            return self._fail("单步执行器只能启动一次。")
        observation.validate()
        self.current_observation = observation
        self.status = "running"
        self._runner = self._run_node(self.plan.root)
        return self._drive()

    def advance(self, result: ActionResult) -> StepDecision:
        if self.status != "running" or self._runner is None:
            return self._fail("执行器当前没有可推进的任务。")
        if self.pending_action is None:
            return self._fail("执行器没有等待中的语义动作。")
        if result.node_id != self.pending_action.node_id:
            return self._fail(
                "动作结果与等待节点不一致："
                f"{result.node_id} != {self.pending_action.node_id}"
            )
        result.observation.validate()
        try:
            next_action = self._runner.send(result)
        except StopIteration:
            self.pending_action = None
            self.status = "finished"
            return StepDecision(
                status="finished",
                reason="目标计划已完成并通过语义验收。",
                counters=dict(self.counters),
            )
        except SemanticExecutionError as exc:
            return self._fail(str(exc))
        self.pending_action = next_action
        return StepDecision(
            status="action",
            action=next_action,
            counters=dict(self.counters),
        )

    def _drive(self) -> StepDecision:
        assert self._runner is not None
        try:
            action = next(self._runner)
        except StopIteration:
            self.status = "finished"
            return StepDecision(
                status="finished",
                reason="目标计划无需更多动作。",
                counters=dict(self.counters),
            )
        except SemanticExecutionError as exc:
            return self._fail(str(exc))
        self.pending_action = action
        return StepDecision(
            status="action",
            action=action,
            counters=dict(self.counters),
        )

    def _fail(self, reason: str) -> StepDecision:
        self.status = "failed"
        self.reason = reason
        self.pending_action = None
        return StepDecision(
            status="failed",
            reason=reason,
            counters=dict(self.counters),
        )

    def _run_node(
        self,
        node: PlanNode,
    ) -> Generator[SemanticAction, ActionResult, None]:
        if node.kind == "action":
            yield from self._run_action(node)
            return
        if node.kind == "sequence":
            for child in node.children:
                yield from self._run_node(child)
            return
        if node.kind == "branch":
            branch = node.children if self._condition_is_true(node.condition) else node.otherwise
            for child in branch:
                yield from self._run_node(child)
            return
        if node.kind == "repeat":
            iterations = 0
            while self._condition_is_true(node.condition):
                if iterations >= int(node.max_iterations or 0):
                    raise SemanticExecutionError(
                        f"循环 {node.node_id} 已达到安全上限，目标仍未完成。"
                    )
                for child in node.children:
                    yield from self._run_node(child)
                iterations += 1
            return
        raise SemanticExecutionError(f"不支持的计划节点：{node.kind}")

    def _run_action(
        self,
        node: PlanNode,
    ) -> Generator[SemanticAction, ActionResult, None]:
        action_name = str(node.action or "")
        if action_name != "observe":
            self._require_safe_observation()
        if action_name in {"record_verified_result", "finish"}:
            self._validate_semantic_evidence(node)

        result = yield SemanticAction(
            node_id=node.node_id,
            action=action_name,
            params=dict(node.params),
        )
        if result.success is not True:
            detail = str(result.details.get("reason") or "动作适配器报告失败")
            raise SemanticExecutionError(f"节点 {node.node_id} 执行失败：{detail}")
        self.current_observation = result.observation

        if action_name == "record_verified_result":
            counter = str(node.params.get("counter") or "").strip()
            if not counter:
                raise SemanticExecutionError("计数动作缺少 counter。")
            self.counters[counter] = self.counters.get(counter, 0) + 1

    def _condition_is_true(self, condition: PlanCondition | None) -> bool:
        if condition is None:
            raise SemanticExecutionError("计划条件缺失。")
        self._require_safe_observation()
        params = condition.params
        if condition.kind == "page_is":
            states = params.get("states")
            if not isinstance(states, list):
                raise SemanticExecutionError("page_is 条件缺少 states。")
            return self._observation().page_state in {str(item) for item in states}
        if condition.kind == "property_equals":
            name = str(params.get("name") or "").strip()
            return self._observation().properties.get(name) == params.get("value")
        if condition.kind == "counter_less_than":
            counter = str(params.get("counter") or "").strip()
            value = params.get("value")
            if isinstance(value, bool) or not isinstance(value, int):
                raise SemanticExecutionError("counter_less_than 的 value 必须是整数。")
            return self.counters.get(counter, 0) < value
        raise SemanticExecutionError(f"不支持的计划条件：{condition.kind}")

    def _validate_semantic_evidence(self, node: PlanNode) -> None:
        observation = self._observation()
        expected_state = str(node.params.get("expected_state") or "").strip()
        if expected_state and observation.page_state != expected_state:
            raise SemanticExecutionError(
                f"节点 {node.node_id} 缺少页面证据："
                f"{observation.page_state} != {expected_state}"
            )
        requirements = node.params.get("requirements") or {}
        if not isinstance(requirements, dict):
            raise SemanticExecutionError("语义验收 requirements 格式无效。")
        for name, expected in requirements.items():
            actual = observation.properties.get(str(name))
            if actual != expected:
                raise SemanticExecutionError(
                    f"节点 {node.node_id} 缺少属性证据："
                    f"{name}={actual!r}，期望 {expected!r}"
                )
        if "expected_count" in node.params:
            counter = str(node.params.get("counter") or "").strip()
            expected_count = node.params.get("expected_count")
            actual_count = self.counters.get(counter, 0)
            if actual_count != expected_count:
                raise SemanticExecutionError(
                    f"节点 {node.node_id} 计数未达标："
                    f"{actual_count} != {expected_count}"
                )

    def _require_safe_observation(self) -> None:
        observation = self._observation()
        observation.validate()
        if not observation.stable:
            raise SemanticExecutionError("当前页面仍在变化，禁止执行下一动作。")
        if float(observation.confidence) < self.min_confidence:
            raise SemanticExecutionError(
                "当前页面置信度不足："
                f"{float(observation.confidence):.2f} < {self.min_confidence:.2f}"
            )

    def _observation(self) -> PageObservation:
        if self.current_observation is None:
            raise SemanticExecutionError("尚未提供页面观察。")
        return self.current_observation


@dataclass(frozen=True)
class FakeTransition:
    action: str
    observation_after: PageObservation
    node_id: str | None = None
    success: bool = True
    details: dict[str, Any] = field(default_factory=dict)


class ScriptedFakeEnvironment:
    """Offline-only action adapter backed by deterministic fake observations."""

    def __init__(
        self,
        initial_observation: PageObservation,
        transitions: Iterable[FakeTransition],
    ) -> None:
        initial_observation.validate()
        self.current_observation = initial_observation
        self._transitions = deque(transitions)

    @property
    def remaining(self) -> int:
        return len(self._transitions)

    def execute(self, action: SemanticAction) -> ActionResult:
        if not self._transitions:
            return ActionResult(
                node_id=action.node_id,
                success=False,
                observation=self.current_observation,
                details={"reason": f"假环境没有为 {action.action} 配置下一状态"},
            )
        expected = self._transitions.popleft()
        if expected.action != action.action:
            return ActionResult(
                node_id=action.node_id,
                success=False,
                observation=self.current_observation,
                details={
                    "reason": (
                        f"假环境期望 {expected.action}，实际收到 {action.action}"
                    )
                },
            )
        if expected.node_id and expected.node_id != action.node_id:
            return ActionResult(
                node_id=action.node_id,
                success=False,
                observation=self.current_observation,
                details={
                    "reason": (
                        f"假环境期望节点 {expected.node_id}，"
                        f"实际收到 {action.node_id}"
                    )
                },
            )
        expected.observation_after.validate()
        self.current_observation = expected.observation_after
        return ActionResult(
            node_id=action.node_id,
            success=expected.success,
            observation=self.current_observation,
            details=dict(expected.details),
        )


@dataclass(frozen=True)
class OfflineLoopReport:
    status: str
    actions: tuple[SemanticAction, ...]
    counters: dict[str, int]
    reason: str = ""
    remaining_fake_transitions: int = 0


class OfflineSemanticLoop:
    """Run the single-step controller against fake observations only."""

    def __init__(self, *, max_steps: int = 100) -> None:
        if not 1 <= max_steps <= 500:
            raise ValueError("离线循环步数上限必须是1～500。")
        self.max_steps = max_steps

    def run(
        self,
        executor: SingleStepSemanticExecutor,
        environment: ScriptedFakeEnvironment,
    ) -> OfflineLoopReport:
        actions: list[SemanticAction] = []
        decision = executor.start(environment.current_observation)
        while decision.status == "action":
            if len(actions) >= self.max_steps:
                return OfflineLoopReport(
                    status="failed",
                    actions=tuple(actions),
                    counters=dict(executor.counters),
                    reason="离线循环达到全局步数上限。",
                    remaining_fake_transitions=environment.remaining,
                )
            assert decision.action is not None
            actions.append(decision.action)
            result = environment.execute(decision.action)
            decision = executor.advance(result)
        return OfflineLoopReport(
            status=decision.status,
            actions=tuple(actions),
            counters=dict(decision.counters),
            reason=decision.reason,
            remaining_fake_transitions=environment.remaining,
        )
