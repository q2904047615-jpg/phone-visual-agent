from __future__ import annotations

from agent.domain.validation import reject_if
import json
from functools import lru_cache
from importlib.resources import files
import uuid
from dataclasses import replace
from typing import Any, Protocol

from agent.domain.generic_goal import GenericIntentError, _parse_json_object
from agent.domain.task_semantic_ir import (
    SemanticRiskAuthorityReport,
    TaskSemanticIRError,
    apply_formal_semantic_risk_policy,
    compile_formal_semantic_authority,
)
from agent.domain import task_graph as task_graph_domain
from agent.domain.task_graph import (
    DynamicTaskGraph,
    ObservedState,
    REPLAN_TRIGGERS,
    ReplanRecord,
    TaskGraphError,
    _apply_verified_navigation_completion,
    _graph_from_payload,
    _normalize_single_effect_result_string,
    _normalize_unique_planner_transport_aliases,
    _planner_transport_snapshot,
    _project_terminal_single_navigation_candidate,
    _require_text,
    _restore_completed_history_evidence,
    _validate_device_id,
    _validate_task_id,
)

__all__ = ("DeepSeekTaskGraphPlanner", "JsonTaskGraphProvider")

class JsonTaskGraphProvider(Protocol):
    configured: bool

    def chat_json(self, messages: list[dict[str, Any]], max_tokens: int = 2000) -> str: ...

class DeepSeekTaskGraphPlanner:
    """Create and revise high-level task graphs without any execution capability."""

    def __init__(self, provider: JsonTaskGraphProvider) -> None:
        self.provider = provider
        self.last_raw_response = ""
        self.last_semantic_authority: SemanticRiskAuthorityReport | None = None
        self.last_semantic_authority_error = ""

    def plan(
        self,
        raw_goal: str,
        *,
        device_id: str,
        task_id: str | None = None,
    ) -> DynamicTaskGraph:
        # Internal whitespace can be literal user payload.  In particular, a
        # line feed is an authorized input character that must survive into
        # the typed graph and semantic source spans unchanged.
        text = str(raw_goal or "").strip()
        reject_if(not text, TaskGraphError("用户目标不能为空。"))
        _validate_device_id(device_id)
        resolved_task_id = task_id or uuid.uuid4().hex
        _validate_task_id(resolved_task_id)
        self._reset_semantic_authority()
        self._require_provider()
        prompt = _initial_prompt(text)
        graph = self._request_graph(prompt, task_id=resolved_task_id, device_id=device_id, revision=1,
            raw_user_goal=text, validate=False)
        # The typed graph is the sole planning authority. Only the syntax-only
        # wire normalizers in ``_request_graph`` run before this strict check;
        # local code never rewrites its surface, frontier, status or semantics.
        graph.validate()
        graph = self._apply_formal_semantic_authority(graph)
        graph.validate()
        reject_if(graph.status == 'completed' or any((item.status == 'completed' for item in graph.subgoals)) or any((item.satisfied for item in graph.completion_conditions)), TaskGraphError("初始规划没有观察证据，不能宣称目标或子目标已完成。"))
        return graph

    def replan(self, graph: DynamicTaskGraph, observation: ObservedState, *, trigger: str,
        reason: str) -> DynamicTaskGraph:
        graph.validate()
        observation.validate()
        reject_if(trigger not in REPLAN_TRIGGERS, TaskGraphError(f"不支持的重规划触发原因：{trigger}"))
        _require_text(reason, "replan.reason")
        self._reset_semantic_authority()
        self._require_provider()
        prompt = _replan_prompt(graph, observation, trigger=trigger, reason=reason)
        candidate = self._request_graph(prompt, task_id=graph.task_id, device_id=graph.device_id,
            revision=graph.revision + 1, raw_user_goal=graph.raw_user_goal or graph.goal.objective, validate=False)
        candidate = _restore_completed_history_evidence(graph, candidate)
        candidate = _project_terminal_single_navigation_candidate(graph, candidate, observation, trigger=trigger)
        candidate = _apply_verified_navigation_completion(graph, candidate, observation, trigger=trigger)
        # Validate the raw typed transport once before local projection.  The
        # revision-specific execution-class and EffectIntent invariants are
        # owned by _validate_replan_candidate below and must not be duplicated
        # on the same candidate.
        candidate.validate()
        candidate = self._apply_formal_semantic_authority(candidate)
        self._validate_replan_candidate(graph, candidate, observation, trigger=trigger)
        previous_ids = {item.subgoal_id for item in graph.subgoals}
        completed_ids = tuple((item.subgoal_id for item in graph.subgoals if item.status == 'completed'))
        added_ids = tuple((item.subgoal_id for item in candidate.subgoals if item.subgoal_id not in previous_ids))
        skipped_ids = tuple((item.subgoal_id for item in candidate.subgoals if item.status == 'skipped'
            and next((old.status for old in graph.subgoals if old.subgoal_id == item.subgoal_id), None) != 'skipped'))
        record = ReplanRecord(
            revision=candidate.revision,
            trigger=trigger,
            reason=reason.strip(),
            scene_id=observation.scene_id,
            evidence=observation.visible_evidence,
            retained_completed_subgoal_ids=completed_ids,
            added_subgoal_ids=added_ids,
            skipped_subgoal_ids=skipped_ids,
            consumed_action_transition_receipt_id=(
                observation.verified_action_transition.receipt_id
                if observation.verified_action_transition is not None
                and trigger in {
                    "action_result_matched",
                    "action_result_mismatch",
                }
                else ""
            ),
        )
        revised = replace(candidate, replan_history=graph.replan_history + (record,))
        revised.validate()
        return revised

    def _apply_formal_semantic_authority(self, graph: DynamicTaskGraph) -> DynamicTaskGraph:
        """Apply typed field roles and local confirmation policy fail-closed."""

        self.last_semantic_authority = None
        self.last_semantic_authority_error = ""
        try:
            authority = compile_formal_semantic_authority(graph)
            projected = apply_formal_semantic_risk_policy(graph, authority)
        except TaskSemanticIRError as exc:
            self.last_semantic_authority_error = str(exc)[:1000]
            raise TaskGraphError(f"正式语义风险权威拒绝任务图：{exc}") from exc
        self.last_semantic_authority = authority
        return projected

    def _reset_semantic_authority(self) -> None:
        self.last_semantic_authority = None
        self.last_semantic_authority_error = ""

    def _validate_replan_candidate(self, graph: DynamicTaskGraph, candidate: DynamicTaskGraph,
        observation: ObservedState, *, trigger: str) -> None:
        """Apply every safety and evidence check to one replan candidate."""

        reject_if(candidate.revision != graph.revision + 1, TaskGraphError('重规划 revision 必须严格等于上一 revision + 1。'))
        transition = observation.verified_action_transition
        if trigger in {'action_result_matched', 'action_result_mismatch'}:
            reject_if(transition is None, TaskGraphError("动作结果重规划缺少本地 verified action transition。"))
            expected_outcome = 'matched' if trigger == 'action_result_matched' else 'mismatched'
            reject_if(transition.outcome != expected_outcome, TaskGraphError("重规划触发与本地动作转换回执 outcome 不一致。"))
            previous_current = graph.active_subgoal()
            reject_if(transition.task_id != graph.task_id or transition.device_id != graph.device_id or transition.prior_revision != graph.revision or (transition.subgoal_id != graph.active_subgoal_id) or (transition.after_observation_id != observation.scene_id) or (previous_current is None), TaskGraphError("动作转换回执未严格绑定上一任务图及当前观察。"))
            consumed_receipts = {item.consumed_action_transition_receipt_id for item
                in graph.replan_history if item.consumed_action_transition_receipt_id}
            reject_if(transition.receipt_id in consumed_receipts, TaskGraphError("动作转换回执已经消费，禁止跨 revision 重放。"))
        elif trigger == 'observation_changed' and transition is not None:
            raise TaskGraphError("纯观察变化不得携带动作执行回执。")

        task_graph_domain._validate_execution_class_revision(graph, candidate)
        task_graph_domain._validate_preserved_effect_intents(graph, candidate)
        candidate.validate()
        previous_current = graph.active_subgoal()
        candidate_current = candidate.active_subgoal()
        reject_if(trigger == 'action_result_mismatch' and previous_current is not None and (next((item.status for item in candidate.subgoals if item.subgoal_id == previous_current.subgoal_id), None) == 'completed'), TaskGraphError('动作结果不匹配时不能完成回执绑定的上一活动子目标。'))
        reject_if(trigger == 'subgoal_completed' and previous_current is not None and (previous_current.external_impact == 'read_only') and (candidate_current is not None) and (candidate_current.external_impact == 'read_only'), TaskGraphError('read_only 完成复核不能继续保留 read_only 活动子目标；当前证据足够时应完成，证据不足时应阻塞，或推进到后续非只读子目标。'))
        task_graph_domain._validate_revision(graph, candidate, observation)

    def _request_graph(self, prompt: str, *, task_id: str, device_id: str, revision: int, raw_user_goal: str,
        validate: bool=True) -> DynamicTaskGraph:
        raw = self.provider.chat_json([{'role': 'user', 'content': prompt}], max_tokens=2400)
        self.last_raw_response = raw
        try:
            payload = _parse_json_object(raw)
        except GenericIntentError as exc:
            raise TaskGraphError(str(exc)) from exc
        payload = _normalize_single_effect_result_string(payload)
        payload = _normalize_unique_planner_transport_aliases(payload)
        graph = _graph_from_payload(payload, task_id=task_id, device_id=device_id, revision=revision,
            raw_user_goal=raw_user_goal)
        if validate:
            graph.validate()
        return graph

    def _require_provider(self) -> None:
        reject_if(not self.provider.configured, TaskGraphError("DeepSeek 动态任务图尚未配置。"))

@lru_cache(maxsize=3)
def _prompt_template(name: str) -> str:
    with files(__package__).joinpath("prompts", name).open("r", encoding="utf-8") as stream:
        return stream.read()


def _render_prompt(name: str, **values: str) -> str:
    template = _prompt_template(name)
    for key, value in values.items():
        template = template.replace("{{" + key + "}}", value)
    return template


def _initial_prompt(raw_goal: str) -> str:
    return _render_prompt("deepseek_initial.txt", RAW_GOAL=json.dumps(raw_goal, ensure_ascii=False),
        SCHEMA=_schema_prompt())


def _replan_prompt(graph: DynamicTaskGraph, observation: ObservedState, *, trigger: str, reason: str) -> str:
    return _render_prompt("deepseek_replan.txt",
        GRAPH=json.dumps(_planner_transport_snapshot(graph), ensure_ascii=False),
        TRIGGER=json.dumps(trigger, ensure_ascii=False), REASON=json.dumps(reason, ensure_ascii=False),
        OBSERVATION=json.dumps(observation.to_dict(), ensure_ascii=False), SCHEMA=_schema_prompt())


def _schema_prompt() -> str:
    return _prompt_template("deepseek_schema.txt").rstrip("\n")
