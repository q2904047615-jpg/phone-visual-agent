"""Application orchestration for the generic one-action visual loop."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any, Callable
import uuid

from agent.domain.task_graph import (
    ControllerTransitionEvidenceRef,
    TaskGraphError,
    DynamicTaskGraph,
    ObservedState,
    VerifiedActionTransition,
    VisualClaimEvidenceRef,
    _named_visual_identity_anchor,
    build_exact_action_task_graph,
    build_exact_input_task_graph,
    named_visual_identity_is_grounded,
)
from agent.application.runtime_session import (
    CORRECTIVE_RETRY_PROTOCOL_VERSION,
    POST_ACTION_TRANSITION_PROTOCOL_VERSION,
    UniversalAgentSessionState,
)
from agent.domain.action_capabilities import build_device_capability_snapshot
from agent.domain.canonical_action_kinds import CANONICAL_ACTION_KINDS
from agent.domain import (
    AgentEvidenceStoreFactory,
    CANONICAL_SELECTION_RECEIPT_VERSION,
    CanonicalSelectionReceipt,
    ConfirmationAuthority,
    DeviceTaskRegistryPort,
    EffectConfirmationAuthority,
    EvidenceStoreError,
    VerifiedAppSurfaceLineage,
)
from agent.application.action_adapter import (
    GenericActionAdapterError,
    GenericSingleActionAdapterPort,
)
from agent.domain.generic_goal import GenericIntentDraft
from agent.domain.canonical_action_protocol import (
    CanonicalActionProtocolError,
    GenericStepProposal,
    expected_idempotent_system_surface_kind,
    scene_matches_target_app_surface,
)
from agent.domain.trusted_observation import TrustedObservation
from agent.domain.qwen_task_context import QwenTaskContext
from agent.domain.ui_scene import (
    MIN_TARGET_CONFIDENCE,
    UISceneError,
    scene_surface_kind,
)
from agent.domain.task_semantic_ir import (
    TaskSemanticIRError,
    compile_formal_semantic_authority,
)
from agent.domain.verified_text_transaction import (
    VerifiedTextTransactionError,
    keyboard_layout_switch_advances,
    plan_next_verified_input,
    preferred_keyboard_layout,
    required_keyboard_input_mode_for_step,
)
from agent.application.vision_usage import VisionSessionUsageLedger


POST_ACTION_OUTCOMES = frozenset({"matched", "mismatched"})
CORRECTIVE_RETRY_IMPACTS = frozenset({"read_only", "navigation_only"})
CORRECTIVE_RETRY_ACTION_KINDS = frozenset(
    {
        "back",
        "dismiss_overlay",
        "double_tap",
        "drag",
        "home",
        "open_recent_apps",
        "long_press",
        "swipe",
        "tap_semantic",
    }
)
MAX_VISIBLE_PRESENCE_ADVANCES_PER_OBSERVATION = 4
_TRANSIENT_ACTION_KEYS = frozenset(
    {
        "node_id",
        "element_id",
        "source_element_id",
        "destination_element_id",
        "bounds",
        "source_bounds",
        "destination_bounds",
        "point",
        "normalized_point",
        "before_fingerprint",
        "observation_id",
        "fingerprint",
    }
)


def _allows_fresh_observation_corrective_retry(
    *,
    impact: str,
    action_kind: str,
) -> bool:
    """Return whether one freshly replanned physical correction is allowed."""

    return (
        str(impact or "").strip() in CORRECTIVE_RETRY_IMPACTS
        and str(action_kind or "").strip() in CORRECTIVE_RETRY_ACTION_KINDS
    )


def _validate_visible_completion_condition_progress(
    previous: DynamicTaskGraph,
    revised: DynamicTaskGraph,
    observed: ObservedState,
) -> None:
    """Allow evidence-backed condition state progress without semantic rewrites."""

    old_conditions = tuple(previous.completion_conditions)
    new_conditions = tuple(revised.completion_conditions)
    old_ids = tuple(item.condition_id for item in old_conditions)
    new_ids = tuple(item.condition_id for item in new_conditions)
    if old_ids != new_ids:
        raise UniversalAgentOrchestratorError(
            "可见状态证据推进不得增加、删除或重排全局完成条件。"
        )
    typed_refs = {
        item.ref_id for item in observed.visual_claim_evidence_refs
    }
    current_evidence = (
        typed_refs
        if typed_refs
        else set(observed.visible_evidence).union(observed.grounded_visual_facts)
    )
    for old, new in zip(old_conditions, new_conditions):
        if (
            new.description != old.description
            or new.evidence_required != old.evidence_required
        ):
            raise UniversalAgentOrchestratorError(
                "可见状态证据推进不得改写全局完成条件定义："
                f"{old.condition_id}。"
            )
        if old.satisfied and not new.satisfied:
            raise UniversalAgentOrchestratorError(
                "可见状态证据推进不得撤销已满足的全局完成条件："
                f"{old.condition_id}。"
            )
        old_evidence = set(old.evidence)
        new_evidence = set(new.evidence)
        if not old_evidence.issubset(new_evidence):
            raise UniversalAgentOrchestratorError(
                "可见状态证据推进不得删除既有全局完成证据："
                f"{old.condition_id}。"
            )
        if not (new_evidence - old_evidence).issubset(current_evidence):
            raise UniversalAgentOrchestratorError(
                "可见状态证据推进使用了当前观察之外的全局完成证据："
                f"{old.condition_id}。"
            )


def _stable_action_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _stable_action_payload(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in _TRANSIENT_ACTION_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_stable_action_payload(item) for item in value]
    return value


def _action_digest(action: Any) -> str:
    if action is None:
        raise UniversalAgentOrchestratorError("动作摘要缺少语义动作。")
    payload = action.to_dict() if callable(getattr(action, "to_dict", None)) else action
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _action_equivalence_digest(action: Any) -> str:
    if action is None:
        raise UniversalAgentOrchestratorError("动作等价摘要缺少语义动作。")
    payload = action.to_dict() if callable(getattr(action, "to_dict", None)) else action
    canonical = json.dumps(
        _stable_action_payload(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _subgoal_progress_signature(subgoal: Any) -> str:
    if subgoal is None:
        return ""
    payload = {
        "subgoal_id": str(getattr(subgoal, "subgoal_id", "")),
        "objective": str(getattr(subgoal, "objective", "")),
        "completion_conditions": list(
            getattr(subgoal, "completion_conditions", ())
        ),
        "external_impact": str(getattr(subgoal, "external_impact", "")),
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _confirmation_effect_ids(
    graph: DynamicTaskGraph,
    current: Any | None,
) -> tuple[str, ...]:
    """Return only locally-policy-bound confirmation risks for one subgoal."""

    if current is None:
        return ()
    active_ids = set(tuple(getattr(current, "risk_action_ids", ()) or ()))
    return tuple(
        sorted(
            risk.risk_id
            for risk in graph.risk_actions
            if risk.risk_id in active_ids and risk.confirmation_required
        )
    )


def _requires_effect_confirmation(
    graph: DynamicTaskGraph,
    current: Any | None,
) -> bool:
    return bool(_confirmation_effect_ids(graph, current))


def _effect_confirmation_material(
    graph: DynamicTaskGraph,
    current: Any,
) -> tuple[str, dict[str, Any]]:
    effect_ids = _confirmation_effect_ids(graph, current)
    risks = [
        risk
        for risk in graph.risk_actions
        if risk.risk_id in set(effect_ids)
    ]
    if {risk.risk_id for risk in risks} != set(effect_ids):
        raise UniversalAgentOrchestratorError(
            "效果确认引用了任务图中不存在的 EffectIntent。"
        )
    serialized_effects = {
        item["effect_id"]: item for item in graph.to_dict()["effect_intents"]
    }
    selected_effects = [serialized_effects[effect_id] for effect_id in effect_ids]
    preview: dict[str, Any] = {
        "kind": "typed_effects",
        "effect_ids": list(effect_ids),
        "effects": [
            {
                "effect_id": item["effect_id"],
                "kind": item["kind"],
                "expected_results": list(item["expected_results"]),
                "policy_level": item["local_policy"]["policy_level"],
            }
            for item in selected_effects
        ],
    }
    payload = {
        "protocol_version": "2026-08-20-typed-effect-confirmation-v1",
        "task_id": graph.task_id,
        "device_id": graph.device_id,
        "revision": graph.revision,
        "subgoal_id": current.subgoal_id,
        "effect_ids": list(effect_ids),
        "effect_intents": selected_effects,
        "goal": {
            "target_apps": [
                {"app_id": app.app_id, "app_name": app.app_name}
                for app in graph.goal.target_apps
            ],
            "entities": graph.goal.entities,
        },
        "preview": preview,
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest(), preview


class UniversalAgentOrchestratorError(RuntimeError):
    pass


def _discard_deepseek_failure_diagnostic(
    *_args: Any,
    **_kwargs: Any,
) -> tuple[str, ...]:
    """Default application port when no diagnostic sink is configured."""

    return ()


class ObservationBridge:
    """Translate protocol objects without inventing actions or business flow."""

    _APP_REFERENCE_IDS = frozenset(
        {
            "unknown",
            "current_foreground",
            "current_app",
            "foreground_app",
            "target_app",
            "active_app",
        }
    )
    _SURFACE_OBSERVATION_IDENTITIES = {
        "device": ("device", "设备界面"),
        "system": ("system", "系统界面"),
        "current_surface": ("current_surface", "当前界面"),
    }

    @classmethod
    def _active_app_entry_target_label(
        cls,
        graph: DynamicTaskGraph,
        active: Any,
    ) -> str:
        """Project one typed App name into the current observation node only.

        The task graph owns App identity while the scene owns geometry and
        visibility.  This projection only preserves which named App the active
        navigation node refers to; it cannot create an element or authorize an
        action.  Missing or ambiguous references remain empty and fail closed.
        """

        if active is None or active.external_impact != "navigation_only":
            return ""
        objective = str(active.objective or "").strip()
        if not objective:
            return ""

        eligible_apps = [
            app
            for app in graph.goal.target_apps
            if str(app.app_name or "").strip()
            and str(app.app_id or "").strip().casefold() not in cls._APP_REFERENCE_IDS
        ]
        mentioned_apps = [
            app
            for app in eligible_apps
            if str(app.app_name or "").strip().casefold() in objective.casefold()
        ]
        if len(mentioned_apps) != 1:
            return ""

        matches: list[str] = []
        for app in mentioned_apps:
            app_id = str(app.app_id or "").strip().casefold()
            app_name = str(app.app_name or "").strip()
            assert app_name and app_id not in cls._APP_REFERENCE_IDS
            literal = re.escape(app_name)
            patterns = (
                rf"(?:打开|进入|启动|切换到|切至|前往)\s*(?:应用|app)?\s*{literal}",
                rf"{literal}\s*(?:应用)?\s*(?:已打开|已启动|主界面可见|首页可见)",
                rf"(?<![a-z0-9_])(?:open|launch|enter|go\s+to|switch\s+to)\s+"
                rf"(?:the\s+)?(?:app\s+)?{literal}(?![a-z0-9_])",
            )
            if any(re.search(pattern, objective, flags=re.IGNORECASE) for pattern in patterns):
                matches.append(app_name)
        unique = tuple(dict.fromkeys(matches))
        return unique[0] if len(unique) == 1 else ""

    @staticmethod
    def _active_input_transaction(
        graph: DynamicTaskGraph,
        active: Any,
    ) -> dict[str, Any]:
        """Project one typed input field into read-only observation.

        The projection is minted only from TaskSemanticIR and overwritten after
        copying model-authored entities.  ``field_label`` remains a literal
        visual selector, while ``field_id`` is only a stable local identity.
        Neither field grants geometry or action authority.
        """

        if active is None or active.external_impact not in {
            "navigation_only",
            "read_only",
        }:
            return {}
        try:
            semantic_ir = compile_formal_semantic_authority(graph).semantic_ir
        except TaskSemanticIRError:
            # The normal formal-authority gate reports the exact error later.
            # Observation projection must not create a fallback authority.
            return {}
        typed_subgoal = next(
            (
                item
                for item in semantic_ir.subgoals
                if item.subgoal_id == active.subgoal_id
            ),
            None,
        )
        if typed_subgoal is None:
            return {}
        entities_by_id = {
            item.entity_id: item for item in semantic_ir.entities
        }
        fields = tuple(
            item
            for item in semantic_ir.input_fields
            if typed_subgoal.subgoal_id in item.source_subgoal_ids
        )
        constraints_by_id = {
            item.constraint_id: item for item in semantic_ir.constraints
        }
        required_actions = {
            str(constraints_by_id[constraint_ref].value)
            for constraint_ref in typed_subgoal.constraint_refs
            if constraint_ref in constraints_by_id
            and constraints_by_id[constraint_ref].kind == "required_action"
        }
        if (
            not fields
            and "clear_verified_text" in required_actions
            and "input_verified_text" not in required_actions
        ):
            # A clear-only goal has no new text payload by design.  Give its
            # sole active input target a stable local identity so the same
            # single-step visual audit must still enumerate the application
            # field, IME preedit and visible backspace key.  This marker grants
            # no text or geometry authority; the fresh audit and canonical
            # clear candidate remain mandatory.
            return {
                "text": "",
                "field_id": "input_field_clear_target",
                "field_label": "",
                "multiline": False,
                "target_only": True,
            }
        if len(fields) != 1:
            return {}
        field = fields[0]
        payload = entities_by_id.get(field.payload_ref)
        if (
            payload is None
            or payload.role != "input_text"
            or not isinstance(payload.value, str)
            or not payload.value
        ):
            return {}
        return {
            "text": payload.value,
            "field_id": field.field_id,
            "field_label": field.field_label,
            "multiline": field.multiline,
        }

    @staticmethod
    def _active_input_predecessor_transaction(
        graph: DynamicTaskGraph, active: Any,
    ) -> dict[str, Any]:
        """Project one direct typed predecessor needed for a Next-key focus step."""

        try:
            semantic_ir = compile_formal_semantic_authority(graph).semantic_ir
        except TaskSemanticIRError:
            return {}
        typed = next((item for item in semantic_ir.subgoals
                      if item.subgoal_id == getattr(active, "subgoal_id", "")), None)
        if typed is None:
            return {}
        fields = tuple(item for item in semantic_ir.input_fields
                       if set(item.source_subgoal_ids).intersection(typed.depends_on))
        if len(fields) != 1:
            return {}
        field = fields[0]
        payload = next((item for item in semantic_ir.entities
                        if item.entity_id == field.payload_ref), None)
        if payload is None or payload.role != "input_text" or not payload.value:
            return {}
        return {"field_id": field.field_id, "field_label": field.field_label,
                "text": payload.value}

    @classmethod
    def _active_input_transaction_text(
        cls,
        graph: DynamicTaskGraph,
        active: Any,
    ) -> str:
        transaction = cls._active_input_transaction(graph, active)
        value = transaction.get("text")
        return value if isinstance(value, str) else ""

    @staticmethod
    def _active_input_verification_projection(
        graph: DynamicTaskGraph,
        active: Any,
    ) -> dict[str, Any]:
        """Project exact typed values needed by the current read-only node.

        A multi-field graph keeps its complete ``input_fields`` authority at
        the task root.  Copying that nested array into the active visual
        context both exposes future writes to the current observation and
        exceeds the observer's deliberately bounded context depth.  A final
        verification node still needs every exact desired value, so expose
        parallel typed maps whose leaves stay within that existing bound.
        """

        if active is None or active.external_impact != "read_only":
            return {}
        try:
            semantic_ir = compile_formal_semantic_authority(graph).semantic_ir
        except TaskSemanticIRError:
            return {}
        typed_subgoal = next(
            (
                item
                for item in semantic_ir.subgoals
                if item.subgoal_id == active.subgoal_id
            ),
            None,
        )
        if typed_subgoal is None:
            return {}
        desired_by_id = {
            item.state_id: item for item in semantic_ir.desired_states
        }
        fields_by_payload = {
            item.payload_ref: item for item in semantic_ir.input_fields
        }
        values: dict[str, str] = {}
        labels: dict[str, str] = {}
        for state_ref in typed_subgoal.desired_state_refs:
            state = desired_by_id.get(state_ref)
            if (
                state is None
                or state.predicate != "input.value_equals"
                or not isinstance(state.value, str)
            ):
                continue
            field = fields_by_payload.get(state.subject_ref)
            if field is None:
                continue
            values[field.field_id] = state.value
            if field.field_label:
                labels[field.field_id] = field.field_label
        if not values:
            return {}
        result: dict[str, Any] = {"desired_input_values": values}
        if labels:
            result["desired_input_labels"] = labels
        return result

    @classmethod
    def _subgoal_visual_context(
        cls,
        graph: DynamicTaskGraph,
        subgoal: Any,
    ) -> dict[str, Any]:
        """Project one typed subgoal into the bounded visual prompt shape."""

        goal_entities = {
            key: value
            for key, value in graph.goal.entities.items()
            if key != "input_fields"
        }
        for local_marker in (
            "active_input_transaction_text",
            "active_input_field_id",
            "active_input_field_label",
            "active_input_multiline",
        ):
            goal_entities.pop(local_marker, None)
        active_app_label = cls._active_app_entry_target_label(graph, subgoal)
        if active_app_label:
            goal_entities["target_ui_label"] = active_app_label
        active_input = cls._active_input_transaction(graph, subgoal)
        active_input_text = active_input.get("text")
        target_only_input = active_input.get("target_only") is True
        if (
            isinstance(active_input_text, str)
            and (bool(active_input_text) or target_only_input)
        ):
            if active_input_text:
                goal_entities["active_input_transaction_text"] = active_input_text
            goal_entities["active_input_field_id"] = active_input["field_id"]
            if target_only_input:
                goal_entities["active_input_target_only"] = True
            if active_input.get("field_label"):
                goal_entities["active_input_field_label"] = active_input[
                    "field_label"
                ]
            goal_entities["active_input_multiline"] = bool(
                active_input.get("multiline")
            )
            predecessor = cls._active_input_predecessor_transaction(
                graph,
                subgoal,
            )
            if predecessor:
                goal_entities.update(
                    {
                        "active_input_predecessor_field_id": predecessor[
                            "field_id"
                        ],
                        "active_input_predecessor_field_label": predecessor[
                            "field_label"
                        ],
                        "active_input_predecessor_text": predecessor["text"],
                    }
                )
        else:
            goal_entities.pop("input_text", None)
            goal_entities.update(
                cls._active_input_verification_projection(graph, subgoal)
            )
        return {
            "subgoal_id": subgoal.subgoal_id,
            "objective": subgoal.objective,
            "constraints": list(subgoal.constraints),
            "completion_conditions": list(subgoal.completion_conditions),
            "execution_class": {
                "read_only": "observe",
                "navigation_only": "navigate",
                "external_state": "effect",
                "unknown": "unknown",
            }.get(subgoal.external_impact, "unknown"),
            "goal_entities": goal_entities,
        }

    def goal_draft(self, graph: DynamicTaskGraph) -> GenericIntentDraft:
        graph.validate()
        target_surface = str(
            graph.goal.entities.get("target_surface") or ""
        ).strip()
        if graph.goal.target_apps:
            primary_app_id = graph.goal.target_apps[0].app_id
            primary_app_name = graph.goal.target_apps[0].app_name
        elif target_surface in self._SURFACE_OBSERVATION_IDENTITIES:
            primary_app_id, primary_app_name = (
                self._SURFACE_OBSERVATION_IDENTITIES[target_surface]
            )
        else:
            raise UniversalAgentOrchestratorError(
                "任务图没有目标 App 或正式目标 surface，不能建立通用观察上下文。"
            )
        active = graph.active_subgoal()
        constraints = list(graph.constraints)
        if active is not None:
            constraints.extend(active.constraints)
        entities = dict(graph.goal.entities)
        # DeepSeek intentionally abstracts low-level wording out of the task
        # graph. Preserve the user's original visual descriptors only inside
        # the read-only scene-observation draft so labels, colors, shapes and
        # coarse positions are not lost. The Qwen action context is still
        # produced directly from ``DynamicTaskGraph.to_qwen_context()``, so
        # this value cannot authorize or specify an action.
        if graph.raw_user_goal.strip():
            entities["original_goal_visual_context"] = graph.raw_user_goal.strip()
        if active is not None:
            # The complete typed graph remains at the root, while the visual
            # request is conditioned only on the actual active subgoal.  A
            # successor becomes observable only after DeepSeek activates it
            # and the orchestrator captures a new scene for that revision.
            entities["active_subgoal_visual_context"] = (
                self._subgoal_visual_context(graph, active)
            )
        entities["target_apps"] = [
            {"app_id": item.app_id, "app_name": item.app_name}
            for item in graph.goal.target_apps
        ]
        success_criteria = {
            item.condition_id: {
                "description": item.description,
                "evidence_required": list(item.evidence_required),
                "satisfied": item.satisfied,
            }
            for item in graph.completion_conditions
        }
        account_effects = tuple(
            dict.fromkeys(item.risk_type for item in graph.risk_actions)
        )
        draft = GenericIntentDraft(
            understood=True,
            app_id=primary_app_id,
            app_name=primary_app_name,
            objective=graph.goal.objective,
            entities=entities,
            constraints=tuple(dict.fromkeys(constraints)),
            success_criteria=success_criteria,
            account_effects=account_effects,
            needs_confirmation=True,
        )
        draft.validate()
        return draft

    @staticmethod
    def _text_items(value: Any) -> tuple[str, ...]:
        if not isinstance(value, (list, tuple)):
            return ()
        return tuple(
            text
            for item in value
            if (text := str(item or "").strip())
        )

    def observed_state(
        self,
        *,
        graph: DynamicTaskGraph,
        trusted_observation: Any,
        action_outcome: str,
        verification: dict[str, Any],
        verified_action_transition: VerifiedActionTransition | None = None,
        controller_transition_evidence_refs: tuple[
            ControllerTransitionEvidenceRef, ...
        ] = (),
    ) -> ObservedState:
        graph.validate()
        observation_device = str(
            getattr(trusted_observation, "device_id", "")
        ).strip()
        if observation_device != graph.device_id:
            raise UniversalAgentOrchestratorError(
                "任务图与可信观察 device_id 不一致。"
            )
        scene = getattr(trusted_observation, "scene", None)
        if scene is None:
            raise UniversalAgentOrchestratorError("可信观察缺少 UIScene。")
        scene.validate()
        fingerprint = str(
            getattr(trusted_observation, "fingerprint", "")
        ).strip()
        if not fingerprint or scene.fingerprint != fingerprint:
            raise UniversalAgentOrchestratorError(
                "可信观察与 UIScene fingerprint 不一致。"
            )
        if not isinstance(verification, dict):
            raise UniversalAgentOrchestratorError("控制器验证结果必须是对象。")

        evidence: list[str] = []
        grounded_visual_facts: list[str] = []

        def add(items: Any) -> None:
            for item in self._text_items(items):
                if item not in evidence:
                    evidence.append(item)

        def add_grounded(item: str) -> None:
            value = str(item or "").strip()
            if value and value not in grounded_visual_facts:
                grounded_visual_facts.append(value)

        if scene.summary.strip():
            add((scene.summary.strip(),))
        add(scene.overlays)
        add_grounded(
            json.dumps(
                {
                    "app_id": scene.app_id,
                    "screen_id": scene.screen_id,
                    "overlays": list(scene.overlays),
                    "stable": scene.stable,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        for element in scene.elements:
            add(element.evidence)
            visible = " / ".join(
                item for item in (element.label.strip(), element.meaning.strip()) if item
            )
            if visible:
                add((visible,))
            add_grounded(
                json.dumps(
                    {
                        "element_id": element.element_id,
                        "role": element.role,
                        "label": element.label,
                        "meaning": element.meaning,
                        "states": dict(element.states),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        add(verification.get("visible_evidence"))
        if action_outcome == "matched":
            add(verification.get("completion_evidence"))
        if not evidence:
            raise UniversalAgentOrchestratorError(
                "当前观察没有可交给 DeepSeek 的可见证据。"
            )

        scene_id = str(
            getattr(trusted_observation, "observation_id", "")
        ).strip() or f"{scene.screen_id}:{fingerprint[:16]}"
        visual_claim_evidence_refs: list[VisualClaimEvidenceRef] = []
        typed_fact_sources = [
            (fact, "scene.visible_literal", "") for fact in evidence
        ] + [
            (fact, "grounded.snapshot", "grounded")
            for fact in grounded_visual_facts
        ]
        for fact, default_predicate, source_kind in typed_fact_sources:
            try:
                payload = json.loads(fact)
            except json.JSONDecodeError:
                payload = {}
            element_id = str(payload.get("element_id") or "").strip()
            subject_ref = (
                f"element:{element_id}" if element_id else f"scene:{scene_id}"
            )
            predicate = (
                "element.snapshot"
                if element_id
                else "scene.snapshot"
                if source_kind == "grounded"
                else default_predicate
            )
            claim_id = hashlib.sha256(
                json.dumps(
                    {
                        "scene_id": scene_id,
                        "subject_ref": subject_ref,
                        "predicate": predicate,
                        "fact": fact,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            visual_claim_evidence_refs.append(
                VisualClaimEvidenceRef(
                    ref_id=f"visual_claim:{scene_id}:{claim_id}",
                    claim_id=claim_id,
                    scene_id=scene_id,
                    subject_ref=subject_ref,
                    predicate=predicate,
                    fact=fact,
                )
            )
        observed = ObservedState(
            scene_id=scene_id,
            summary=scene.summary.strip() or "当前可信页面观察",
            visible_evidence=tuple(evidence),
            grounded_visual_facts=tuple(grounded_visual_facts),
            last_action_outcome=str(action_outcome or "not_applicable"),
            blocked_reasons=self._text_items(verification.get("blocked_reasons")),
            verified_action_transition=verified_action_transition,
            controller_transition_evidence_refs=(
                controller_transition_evidence_refs
            ),
            visual_claim_evidence_refs=tuple(visual_claim_evidence_refs),
        )
        observed.validate()
        return observed


@dataclass(frozen=True)
class TaskGraphTransitionReport:
    """Non-action progress/completion report from the validated task graph."""

    status: str
    reason: str
    completion_evidence: tuple[str, ...]
    action: None = field(default=None, init=False)
    authority: str = field(default="deepseek_task_graph", init=False)

    def __post_init__(self) -> None:
        if self.status not in {"progressed", "completed"}:
            raise UniversalAgentOrchestratorError("任务图报告状态无效。")
        if not self.reason.strip():
            raise UniversalAgentOrchestratorError("任务图报告缺少原因。")
        if not self.completion_evidence:
            raise UniversalAgentOrchestratorError("任务图报告缺少可见证据。")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "action": None,
            "reason": self.reason,
            "completion_evidence": list(self.completion_evidence),
            "authority": self.authority,
        }


class UniversalAgentOrchestrator:
    """Coordinate the generic one-action visual loop without App workflows."""

    def __init__(
        self,
        *,
        deepseek_planner: Any,
        qwen_observer: Any,
        adapter_factory: Callable[[str], GenericSingleActionAdapterPort],
        evidence_store_factory: AgentEvidenceStoreFactory,
        trusted_observation_factory: Callable[..., Any],
        bridge: ObservationBridge | None = None,
        device_registry: DeviceTaskRegistryPort,
        deepseek_failure_diagnostic_writer: Callable[..., tuple[str, ...]] | None = None,
    ) -> None:
        self.deepseek_planner = deepseek_planner
        self.qwen_observer = qwen_observer
        self.adapter_factory = adapter_factory
        self.trusted_observation_factory = trusted_observation_factory
        self.evidence_store_factory = evidence_store_factory
        self.bridge = bridge or ObservationBridge()
        self.device_registry = device_registry
        self.deepseek_failure_diagnostic_writer = (
            deepseek_failure_diagnostic_writer
            or _discard_deepseek_failure_diagnostic
        )

    def _vision_usage_scope(
        self,
        ledger: VisionSessionUsageLedger | None,
    ):
        provider = getattr(self.qwen_observer, "provider", None)
        scope_factory = getattr(provider, "session_usage_scope", None)
        return scope_factory(ledger) if callable(scope_factory) else nullcontext()

    def _release_if_terminal(self, session: UniversalAgentSessionState) -> None:
        if session.status in self.device_registry.TERMINAL_STATUSES:
            self.device_registry.release(session.device_id, session.session_id)

    @staticmethod
    def _available_action_kinds(
        session: UniversalAgentSessionState,
    ) -> frozenset[str]:
        provider = getattr(session.adapter, "supported_action_kinds", None)
        if not callable(provider):
            return CANONICAL_ACTION_KINDS
        actions = frozenset(str(item or "").strip() for item in provider())
        if not actions or "" in actions:
            raise UniversalAgentOrchestratorError(
                "设备动作能力为空或包含无效动作。"
            )
        unexpected = actions - CANONICAL_ACTION_KINDS
        if unexpected:
            raise UniversalAgentOrchestratorError(
                "设备报告了协议外动作：" + ", ".join(sorted(unexpected))
            )
        return actions

    @staticmethod
    def _lineage_matches_observed_foreground(
        lineage: VerifiedAppSurfaceLineage,
        foreground_app_id: str,
    ) -> bool:
        foreground = str(foreground_app_id or "").strip().casefold()
        if not foreground or foreground == "launcher":
            return False
        return foreground in {
            str(lineage.functional_foreground_app_id or "").strip().casefold(),
            str(lineage.app_id or "").strip().casefold(),
            str(lineage.app_name or "").strip().casefold(),
        }

    @classmethod
    def _bind_verified_lineage_to_qwen_context(
        cls,
        session: UniversalAgentSessionState,
        context: QwenTaskContext,
        trusted_observation: Any,
    ) -> QwenTaskContext:
        """Rebind a typed App surface to its receipt-proven runtime package.

        The App entry receipt is the only source of this alias.  It is scoped to
        the same session/task/device/action count and only remains usable by a
        descendant of the completed entry subgoal on the same observed App.
        """

        lineage = session.verified_app_surface_lineage
        graph = session.task_graph
        semantic_ir = context.semantic_ir
        scene = getattr(trusted_observation, "scene", None)
        if lineage is None or graph is None or semantic_ir is None or scene is None:
            return context
        if (
            lineage.session_id != session.session_id
            or lineage.task_id != graph.task_id
            or lineage.task_id != context.task_id
            or lineage.device_id != session.device_id
            or lineage.device_id != graph.device_id
            or lineage.device_id != context.device_id
            or lineage.physical_actions != session.physical_actions
            or not cls._lineage_matches_observed_foreground(
                lineage,
                str(getattr(scene, "foreground_app_id", "")),
            )
        ):
            return context

        by_id = {item.subgoal_id: item for item in graph.subgoals}
        source = by_id.get(lineage.source_subgoal_id)
        current = graph.active_subgoal()
        if (
            source is None
            or source.status != "completed"
            or current is None
        ):
            return context
        pending = list(current.depends_on)
        visited: set[str] = set()
        lineage_is_ancestor = False
        while pending:
            dependency_id = pending.pop()
            if dependency_id == lineage.source_subgoal_id:
                lineage_is_ancestor = True
                break
            if dependency_id in visited:
                continue
            visited.add(dependency_id)
            dependency = by_id.get(dependency_id)
            if dependency is not None:
                pending.extend(dependency.depends_on)
        if not lineage_is_ancestor:
            return context

        matching_surfaces = tuple(
            surface
            for surface in semantic_ir.surfaces
            if surface.surface_id == lineage.surface_id
            and surface.kind == "app"
            and surface.app_id.casefold() == lineage.app_id.casefold()
            and surface.app_name.casefold() == lineage.app_name.casefold()
        )
        if len(matching_surfaces) != 1:
            return context
        target_surface = matching_surfaces[0]
        # A receipt-proven runtime package is stronger than a structured
        # screen-ID alias.  Keep the precise lineage upgrade unless the typed
        # surface already carries that exact foreground identity.
        if (
            str(getattr(scene, "foreground_app_id", "") or "").strip().casefold()
            == str(target_surface.app_id or "").strip().casefold()
        ):
            return context
        rebound_surface = replace(
            target_surface,
            app_id=lineage.functional_foreground_app_id,
        )
        rebound_ir = replace(
            semantic_ir,
            surfaces=tuple(
                rebound_surface
                if item.surface_id == target_surface.surface_id
                else item
                for item in semantic_ir.surfaces
            ),
        )
        rebound_context = replace(context, semantic_ir=rebound_ir)
        try:
            rebound_context.validate()
        except Exception:
            return context
        return rebound_context

    def _decide_next_action(
        self,
        session: UniversalAgentSessionState,
        *,
        frames: list[Any],
        task_context: Any,
        trusted_observation: Any,
    ) -> Any:
        context = (
            task_context
            if isinstance(task_context, QwenTaskContext)
            else QwenTaskContext.from_dict(dict(task_context))
        )
        if context.semantic_ir is None:
            graph = session.task_graph
            semantic_authority = None
            if graph is not None:
                try:
                    semantic_authority = compile_formal_semantic_authority(graph)
                except TaskSemanticIRError as exc:
                    raise UniversalAgentOrchestratorError(
                        f"正式 TaskSemanticIR authority 拒绝：{exc}"
                    ) from exc
                semantic_ir = semantic_authority.semantic_ir
            else:
                semantic_authority = getattr(
                    self.deepseek_planner,
                    "last_semantic_authority",
                    None,
                )
                semantic_ir = getattr(semantic_authority, "semantic_ir", None)
                if semantic_ir is None:
                    raise UniversalAgentOrchestratorError(
                        "正式视觉决策缺少当前任务图。"
                    )
            context = replace(context, semantic_ir=semantic_ir)
            context.validate()
            if semantic_authority is not None and hasattr(
                semantic_authority,
                "effect_previews",
            ):
                session.effect_previews = tuple(
                    {
                        **preview.to_dict(),
                        "preview_digest": preview.preview_digest,
                    }
                    for preview in semantic_authority.effect_previews
                )
        context = self._bind_verified_lineage_to_qwen_context(
            session,
            context,
            trusted_observation,
        )
        session.semantic_task_context = context
        available_actions = self._available_action_kinds(session)
        semantic_ir = context.semantic_ir
        assert semantic_ir is not None
        active_id = str(context.current_subgoal.get("subgoal_id") or "")
        typed_subgoal = next(
            (item for item in semantic_ir.subgoals if item.subgoal_id == active_id),
            None,
        )
        constraints = {item.constraint_id: item for item in semantic_ir.constraints}
        required_actions = tuple(
            dict.fromkeys(
                str(constraints[ref].value)
                for ref in (
                    typed_subgoal.constraint_refs if typed_subgoal is not None else ()
                )
                if constraints[ref].kind == "required_action"
            )
        )
        unsupported = tuple(
            action for action in required_actions if action not in available_actions
        )
        if unsupported:
            provider = getattr(session.adapter, "capability_snapshot", None)
            capability = (
                provider()
                if callable(provider)
                else build_device_capability_snapshot(
                    device_id=context.device_id,
                    supported_actions=available_actions,
                )
            )
            gap = capability.gap(unsupported[0])
            assert gap is not None
            session.capability_gap = gap.to_dict()
            reason = "当前设备能力不支持 typed required_action：" + unsupported[0]
            proposal = GenericStepProposal(status="blocked", reason=reason)
            decision = SimpleNamespace(
                task_id=context.task_id,
                device_id=context.device_id,
                revision=context.revision,
                observation_id=trusted_observation.observation_id,
                fingerprint=trusted_observation.fingerprint,
                trusted_observation=trusted_observation,
                proposal=proposal,
                target_region=None,
                expected_result={},
                confidence=1.0,
                reason=reason,
            )
            decision.to_dict = lambda: {
                "task_id": decision.task_id,
                "device_id": decision.device_id,
                "revision": decision.revision,
                "observation_id": decision.observation_id,
                "fingerprint": decision.fingerprint,
                "status": "blocked",
                "next_action": None,
                "reason": reason,
                "capability_gap": dict(session.capability_gap),
            }
            return decision
        return self.qwen_observer.decide(
            frames=frames,
            task_context=context,
            trusted_observation=trusted_observation,
            decision_number=session.step_number,
            available_action_kinds=available_actions,
        )

    @staticmethod
    def _context_value(source: Any, name: str, default: Any = None) -> Any:
        if isinstance(source, Mapping):
            return source.get(name, default)
        return getattr(source, name, default)

    @staticmethod
    def _is_idempotent_app_foreground_completion(value: Any) -> bool:
        """Recognize one completed App-foreground state, never an action receipt."""

        text = str(value or "").strip().casefold()
        if not text or len(text) > 96:
            return False
        chinese = re.fullmatch(
            r"[\w\u4e00-\u9fff·._ -]{1,64}(?:应用|程序)"
            r"(?:已经|已)?(?:打开|启动|在前台|处于前台)(?:可见)?[。.]?",
            text,
        )
        english = re.fullmatch(
            r"[a-z0-9][a-z0-9 ._-]{0,63}\s+(?:app|application)\s+"
            r"(?:is\s+)?(?:open|opened|launched|in the foreground|foreground)"
            r"(?:\s+and\s+visible)?[.]?",
            text,
        )
        return bool(chinese or english)

    @staticmethod
    def _is_presence_only_read_only_subgoal(subgoal: Any) -> bool:
        """Allow zero-action completion only for locating one visible object.

        A visible element can prove that an object exists, but it cannot prove
        its exact value, text, state, or a result produced elsewhere.  Keep the
        lexical gate deliberately narrow so ambiguous read-only work fails
        closed and can be retried with a fresh observation instead.
        """

        text = " ".join(
            str(item or "").strip()
            for item in (
                getattr(subgoal, "objective", ""),
                *tuple(getattr(subgoal, "completion_conditions", ()) or ()),
            )
            if str(item or "").strip()
        ).casefold()
        if not text:
            return False
        presence_markers = (
            "定位",
            "找到",
            "寻找",
            "识别",
            "可见",
            "存在",
            "locate",
            "find",
            "identify",
            "visible",
            "present",
            "exists",
        )
        value_verification_markers = (
            "内容",
            "文字",
            "文本",
            "数值",
            "字段值",
            "包含",
            "等于",
            "是否为",
            "状态为",
            "验证",
            "核对",
            "读取",
            "content",
            "text equals",
            "contains",
            "value",
            "verify",
            "read the",
        )
        transition_markers = (
            "刷新",
            "重新加载",
            "重新载入",
            "重新获取",
            "重新读取",
            "重新连接",
            "加载完成",
            "更新完成",
            "同步完成",
            "导航",
            "跳转",
            "进入",
            "返回",
            "切换",
            "打开",
            "启动",
            "收起",
            "隐藏",
            "关闭",
            "消失",
            "移除",
            "不可见",
            "不存在",
            "缺失",
            "refresh",
            "reload",
            "reloaded",
            "updated",
            "synchronized",
            "navigated",
            "redirected",
            "entered",
            "returned",
            "switched",
            "opened",
            "launched",
            "dismiss",
            "hide",
            "hidden",
            "close",
            "closed",
            "disappear",
            "remove",
            "not visible",
            "absent",
            "missing",
            "retrieved",
            "refetched",
            "reconnected",
        )
        completion_text = " ".join(
            str(item or "").strip()
            for item in tuple(
                getattr(subgoal, "completion_conditions", ()) or ()
            )
            if str(item or "").strip()
        ).casefold()
        completion_conditions = tuple(
            str(item or "").strip()
            for item in tuple(
                getattr(subgoal, "completion_conditions", ()) or ()
            )
            if str(item or "").strip()
        )
        if (
            len(completion_conditions) == 1
            and UniversalAgentOrchestrator._is_idempotent_app_foreground_completion(
                completion_conditions[0]
            )
        ):
            return True
        occurrence_or_absence_markers = (
            "刷新",
            "重新加载",
            "重新载入",
            "重新获取",
            "重新读取",
            "重新连接",
            "加载完成",
            "更新完成",
            "同步完成",
            "不可见",
            "不存在",
            "缺失",
            "消失",
            "移除",
            "refresh",
            "reload",
            "reloaded",
            "updated",
            "synchronized",
            "not visible",
            "absent",
            "missing",
            "disappear",
            "remove",
            "retrieved",
            "refetched",
            "reconnected",
        )
        if (
            completion_text
            and any(marker in completion_text for marker in presence_markers)
            and not any(
                marker in completion_text for marker in value_verification_markers
            )
            and not any(
                marker in completion_text
                for marker in occurrence_or_absence_markers
            )
        ):
            # An already-visible destination may satisfy an idempotent
            # navigation node such as opening an App or returning Home.  The
            # caller still requires grounded named-surface identity and unique
            # current visual evidence.  Occurrence claims such as refresh and
            # negative/absence states remain ineligible.
            return True
        return any(marker in text for marker in presence_markers) and not any(
            marker in text for marker in value_verification_markers
        ) and not any(marker in text for marker in transition_markers)

    @staticmethod
    def _candidate_has_unresolved_conflict(
        trusted_observation: Any,
        element_id: str,
    ) -> bool:
        for conflict in getattr(trusted_observation, "candidate_conflicts", ()) or ():
            if not isinstance(conflict, Mapping):
                if element_id in str(conflict):
                    return True
                continue
            conflict_ids = conflict.get("element_ids") or []
            resolved_duplicate = (
                conflict.get("kind") == "duplicate_visual_object_collapsed"
                and conflict.get("canonical_element_id") == element_id
                and element_id in conflict_ids
            )
            if not resolved_duplicate and (
                element_id in conflict_ids or element_id in str(conflict)
            ):
                return True
        return False

    @classmethod
    def _verified_focused_input_fact(
        cls,
        trusted_observation: Any,
    ) -> str | None:
        """Return one local fact only when focus is uniquely scene-proven."""

        scene = getattr(trusted_observation, "scene", None)
        if scene is None:
            return None
        candidates = []
        for item in tuple(getattr(scene, "elements", ()) or ()):
            states = getattr(item, "states", {}) or {}
            left, top, right, bottom = getattr(item, "bounds", (0, 0, 0, 0))
            if (
                str(getattr(item, "role", "") or "") == "input"
                and states.get("goal_relevant") is True
                and states.get("focused") is True
                and float(getattr(item, "confidence", 0.0)) >= 0.90
                and 0.02 <= left < right <= 0.98
                and 0.02 <= top < bottom <= 0.98
                and not cls._candidate_has_unresolved_conflict(
                    trusted_observation,
                    str(getattr(item, "element_id", "") or ""),
                )
            ):
                candidates.append(item)
        if len(candidates) != 1:
            return None
        item = candidates[0]
        return (
            "当前可信画面的局部控件状态："
            f"element_id={item.element_id}, role=input, focused=true。"
        )

    @classmethod
    def _zero_action_visible_state_fact(
        cls,
        subgoal: Any,
        trusted_observation: Any,
    ) -> str | None:
        """Bind a safe state-only checkpoint to an exact local scene fact.

        This deliberately recognizes only focus as a reversible, structured UI
        state.  Input values, keyboard modes, external results, and other state
        claims keep their existing dedicated evidence contracts.
        """

        conditions = tuple(
            str(item or "").strip().casefold()
            for item in tuple(getattr(subgoal, "completion_conditions", ()) or ())
            if str(item or "").strip()
        )
        if not conditions:
            return None
        focus_patterns = (
            re.compile(
                r"(?:输入框|文本框|输入区域).{0,10}"
                r"(?:已|处于|保持|获得)?(?:聚焦|焦点)"
            ),
            re.compile(
                r"焦点.{0,10}(?:位于|保持在|处于)?"
                r"(?:输入框|文本框|输入区域)"
            ),
            re.compile(
                r"(?:input|textbox|text field).{0,20}"
                r"(?:is |remains |has )?(?:focused|focus)"
            ),
        )
        if any(
            not any(pattern.search(condition) for pattern in focus_patterns)
            for condition in conditions
        ):
            return None
        return cls._verified_focused_input_fact(trusted_observation)

    @staticmethod
    def _presence_binding_terms(*values: Any) -> frozenset[str]:
        """Return bounded literal terms for a zero-action presence check."""

        text = " ".join(
            str(value or "").casefold().replace("_", " ") for value in values
        )
        generic = {
            "action", "button", "control", "current", "element", "image",
            "item", "page", "screen", "target", "view", "visible",
            "当前", "页面", "画面", "目标", "元素", "控件", "可见", "出现",
        }
        terms = {
            token
            for token in re.findall(r"[a-z0-9]{3,}", text)
            if token not in generic
        }
        for run in re.findall(r"[\u4e00-\u9fff]{2,}", text):
            for size in range(2, min(6, len(run)) + 1):
                terms.update(
                    run[index:index + size]
                    for index in range(0, len(run) - size + 1)
                )
        return frozenset(term for term in terms if term not in generic)

    @staticmethod
    def _presence_title_prefixes(*values: Any) -> tuple[str, ...]:
        """Extract explicit visible title-prefix selectors, never infer one."""

        text = " ".join(str(value or "").strip() for value in values)
        selectors: list[str] = []
        patterns = (
            re.compile(
                r"标题(?:文字)?(?:开头|起始)(?:为|是|[:：])?\s*[“\"']?"
                r"([A-Za-z0-9\u4e00-\u9fff·._-]{1,64}?)"
                r"(?=的(?:唯一)?(?:卡片|列表项|条目|按钮|菜单项)|[”\"'，,。；;]|$)"
            ),
            re.compile(
                r"title\s+(?:starts?|begins?)\s+with\s+[\"']?"
                r"([A-Za-z0-9][A-Za-z0-9 ._\-]{0,63}?)"
                r"(?=(?:\s+(?:card|item|button|entry))|[\"',.;]|$)",
                re.IGNORECASE,
            ),
        )
        for pattern in patterns:
            for match in pattern.finditer(text):
                value = match.group(1).strip().casefold()
                if value and value not in selectors:
                    selectors.append(value)
        return tuple(selectors)

    @classmethod
    def _presence_surface_classes(cls, *values: Any) -> frozenset[str]:
        """Keep destination/container nouns from collapsing into ordinal overlap."""

        text = " ".join(
            str(value or "").casefold().replace("_", " ").replace("-", " ")
            for value in values
        )
        classes = {
            name
            for name, markers in {
                "page": (
                    "页面",
                    "网页",
                    "界面",
                    "首页",
                    "主界面",
                    " page",
                    "screen",
                    "view",
                    "interface",
                    "app home",
                ),
                "title": ("标题", "题头", "title", "heading"),
                "list": ("列表", "清单", " list"),
                "input": ("输入框", "文本框", "input field", "textbox"),
                "menu": ("菜单", " menu"),
                "dialog": ("对话框", "弹窗", "dialog", "modal"),
                "destination": (
                    "对应页面",
                    "目标页面",
                    "下一页",
                    "详情页",
                    "详情",
                    "destination page",
                    "target page",
                    "next page",
                    "detail page",
                    "details",
                ),
                "foreground_app": (
                    "应用在前台",
                    "前台应用",
                    "前台可见",
                    "foreground app",
                    "in the foreground",
                    "is foreground",
                ),
            }.items()
            if any(marker in text for marker in markers)
        }
        if any(cls._is_idempotent_app_foreground_completion(value) for value in values):
            classes.add("foreground_app")
        return frozenset(classes)

    @classmethod
    def _target_app_identity_terms(cls, *values: Any) -> frozenset[str]:
        return cls._presence_binding_terms(*values).difference(
            {"app", "application", "android", "com", "应用", "程序"}
        )

    @staticmethod
    def _compact_app_surface_phrase(value: Any) -> str:
        return "".join(
            re.findall(
                r"[a-z0-9]+|[\u4e00-\u9fff]+",
                str(value or "").casefold(),
            )
        )

    @classmethod
    def _presence_names_only_target_app_surface(
        cls,
        presence_text: str,
        target_app: Any,
    ) -> bool:
        """Bind App foreground only when no named child surface remains.

        Full App names/IDs may be followed by generic state words such as
        ``main screen`` or ``visible``.  If removing those identities leaves a
        recipient, order, settings section or any other named qualifier, App
        foreground cannot prove that more specific destination page.
        """

        compact = cls._compact_app_surface_phrase(presence_text)
        identities = tuple(
            dict.fromkeys(
                identity
                for identity in (
                    cls._compact_app_surface_phrase(
                        getattr(target_app, "app_id", "")
                    ),
                    cls._compact_app_surface_phrase(
                        getattr(target_app, "app_name", "")
                    ),
                )
                if identity
                and identity not in {"app", "application", "应用", "程序"}
            )
        )
        if not compact or not identities or not any(
            identity in compact for identity in identities
        ):
            return False
        residual = compact
        for identity in sorted(identities, key=len, reverse=True):
            residual = residual.replace(identity, "")
        generic_state_tokens = (
            "处于前台", "已经打开", "已经启动", "应用程序", "主界面", "主页面",
            "当前", "目标", "应用", "程序", "主页", "首页", "页面", "界面",
            "屏幕", "视图", "打开", "启动", "进入", "前台", "可见", "显示",
            "已经", "已", "在", "的", "并", "and", "application", "foreground",
            "launched", "opened", "visible", "current", "target", "screen",
            "interface", "page", "view", "home", "main", "launch", "open",
            "app", "is", "in", "the",
        )
        for token in sorted(generic_state_tokens, key=len, reverse=True):
            residual = residual.replace(token, "")
        return not residual

    @classmethod
    def _presence_references_target_app_identity(
        cls,
        presence_text: str,
        target_app: Any,
    ) -> bool:
        """Return whether the text contains one complete target-App identity."""

        compact = cls._compact_app_surface_phrase(presence_text)
        identities = (
            cls._compact_app_surface_phrase(getattr(target_app, "app_id", "")),
            cls._compact_app_surface_phrase(getattr(target_app, "app_name", "")),
        )
        return bool(
            compact
            and any(
                identity in compact
                for identity in identities
                if identity
                and identity not in {"app", "application", "应用", "程序"}
            )
        )

    @staticmethod
    def _subgoal_targets_launcher_surface(
        graph: DynamicTaskGraph,
        subgoal_id: str,
    ) -> bool:
        """Use the formal semantic surface to distinguish a Launcher destination."""

        if not str(subgoal_id or "").strip():
            return False
        try:
            semantic_ir = compile_formal_semantic_authority(graph).semantic_ir
        except TaskSemanticIRError:
            return False
        typed_subgoal = next(
            (
                item
                for item in semantic_ir.subgoals
                if item.subgoal_id == subgoal_id
            ),
            None,
        )
        if typed_subgoal is None:
            return False
        surfaces = {item.surface_id: item for item in semantic_ir.surfaces}
        surface = surfaces.get(typed_subgoal.surface_ref)
        return bool(surface is not None and surface.kind == "launcher")

    @staticmethod
    def _typed_idempotent_system_surface_fact(
        graph: DynamicTaskGraph,
        subgoal_id: str,
        scene: Any,
    ) -> str | None:
        """Bind one canonical system action to its already-reached surface."""

        if not str(subgoal_id or "").strip():
            return None
        try:
            semantic_ir = compile_formal_semantic_authority(graph).semantic_ir
            actual_surface_kind = scene_surface_kind(scene)
        except (TaskSemanticIRError, UISceneError):
            return None
        typed_subgoal = next(
            (
                item
                for item in semantic_ir.subgoals
                if item.subgoal_id == subgoal_id
            ),
            None,
        )
        if typed_subgoal is None:
            return None
        constraints_by_id = {
            item.constraint_id: item for item in semantic_ir.constraints
        }
        required_actions = {
            str(constraints_by_id[constraint_ref].value)
            for constraint_ref in typed_subgoal.constraint_refs
            if constraint_ref in constraints_by_id
            and constraints_by_id[constraint_ref].kind == "required_action"
        }
        if len(required_actions) != 1:
            return None
        action_kind = next(iter(required_actions))
        expected_surface_kind = expected_idempotent_system_surface_kind(
            action_kind
        )
        if (
            expected_surface_kind is None
            or actual_surface_kind != expected_surface_kind
        ):
            return None
        return json.dumps(
            {
                "action_kind": action_kind,
                "operator": "equals",
                "predicate": "surface.kind",
                "source": "canonical_action_protocol",
                "value": actual_surface_kind,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def _referenced_target_app_pages(
        cls,
        *,
        graph: DynamicTaskGraph,
        presence_text: str,
        subgoal_id: str = "",
    ) -> tuple[Any, ...]:
        """Return target Apps whose named page is the claimed visible state.

        A launcher affordance labelled with an App name proves that the App can
        be opened; it does not prove that the named App page is already in the
        foreground.  Keep the binding structural and graph-derived so the same
        rule applies to every App and every natural-language goal.
        """

        if cls._subgoal_targets_launcher_surface(graph, subgoal_id):
            return ()
        required_surfaces = cls._presence_surface_classes(presence_text)
        if not required_surfaces.intersection({"page", "foreground_app"}):
            return ()
        referenced = []
        for target_app in graph.goal.target_apps:
            if str(target_app.app_id or "").strip().casefold() == "current_foreground":
                continue
            if cls._presence_references_target_app_identity(
                presence_text,
                target_app,
            ):
                referenced.append(target_app)
        return tuple(referenced)

    @classmethod
    def _scene_foreground_matches_target_app_page(
        cls,
        *,
        scene: Any,
        target_apps: tuple[Any, ...],
    ) -> bool:
        # The canonical action protocol owns App-surface identity.  Reuse that
        # exact contract for zero-action task progress so DeepSeek/Qwen cannot
        # disagree about whether the current structured surface is the target.
        return any(
            scene_matches_target_app_surface(scene, target_app)
            for target_app in target_apps
        )

    @staticmethod
    def _scene_page_identity_facts(scene: Any) -> tuple[str, ...]:
        facts = [
            json.dumps(
                {
                    "app_id": str(getattr(scene, "app_id", "") or ""),
                    "foreground_app_id": str(
                        getattr(scene, "foreground_app_id", "") or ""
                    ),
                    "screen_id": str(getattr(scene, "screen_id", "") or ""),
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        ]
        for element in tuple(getattr(scene, "elements", ()) or ()):
            role = str(getattr(element, "role", "") or "").casefold()
            meaning = str(getattr(element, "meaning", "") or "").casefold()
            if role not in {"text", "container"}:
                continue
            if role != "container" and not any(
                marker in meaning
                for marker in ("page", "screen", "view", "home", "title", "heading")
            ):
                continue
            facts.append(
                json.dumps(
                    {
                        "role": role,
                        "meaning": meaning,
                        "label": str(getattr(element, "label", "") or ""),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        return tuple(facts)

    @classmethod
    def _scene_named_presence_is_grounded(
        cls,
        *,
        scene: Any,
        texts: tuple[str, ...],
    ) -> bool:
        return named_visual_identity_is_grounded(
            texts,
            cls._scene_page_identity_facts(scene),
        )

    @classmethod
    def _intrinsic_presence_surface_classes(cls, item: Any) -> frozenset[str]:
        """Return surface types owned by an element, not words near it."""

        role = str(getattr(item, "role", "") or "").casefold()
        classes = set(
            cls._presence_surface_classes(
                role,
                getattr(item, "meaning", ""),
            )
        )
        role_classes = {
            "input": "input",
            "textbox": "input",
            "text_input": "input",
            "list": "list",
            "list_item": "list",
            "menu": "menu",
            "menu_item": "menu",
            "dialog": "dialog",
            "modal": "dialog",
            "title": "title",
            "heading": "title",
        }
        if role in role_classes:
            classes.add(role_classes[role])
        return frozenset(classes)

    def _multi_presence_candidates(
        self,
        *,
        subgoal: Any,
        scene: Any,
        trusted_observation: Any,
    ) -> tuple[Any, ...] | None:
        """Bind two to four explicitly conjoined visible objects, fail closed."""

        text = " ".join(
            str(item or "").strip()
            for item in (
                getattr(subgoal, "objective", ""),
                *tuple(getattr(subgoal, "completion_conditions", ()) or ()),
            )
            if str(item or "").strip()
        ).casefold()
        if not self._is_explicit_multi_presence_text(text):
            return None
        text_terms = self._presence_binding_terms(text)
        if not text_terms:
            return None
        matched: list[tuple[Any, frozenset[str]]] = []
        for candidate in scene.elements:
            terms = self._presence_binding_terms(
                candidate.label,
                candidate.meaning,
                *candidate.evidence,
            ).intersection(text_terms)
            if not terms:
                continue
            if (
                float(candidate.confidence) < MIN_TARGET_CONFIDENCE
                or candidate.states.get("visible") is False
                or candidate.states.get("fully_visible") is False
                or self._candidate_has_unresolved_conflict(
                    trusted_observation,
                    candidate.element_id,
                )
            ):
                return None
            left, top, right, bottom = candidate.bounds
            if not (
                0.02 <= left < right <= 0.98
                and 0.02 <= top < bottom <= 0.98
            ):
                return None
            matched.append((candidate, frozenset(terms)))
        # An instruction card may repeat every endpoint name. It is aggregate
        # evidence, not either endpoint. Remove it only when two or more
        # smaller candidates together cover all of its bound terms.
        reduced: list[tuple[Any, frozenset[str]]] = []
        for index, item in enumerate(matched):
            _candidate, terms = item
            others = [
                other_terms
                for other_index, (_other, other_terms) in enumerate(matched)
                if other_index != index and other_terms.intersection(terms)
            ]
            aggregate = False
            if len(others) >= 2:
                for first in range(len(others)):
                    for second in range(first + 1, len(others)):
                        combined = others[first].union(others[second])
                        if combined and combined.issubset(terms):
                            aggregate = True
                            break
                    if aggregate:
                        break
            if not aggregate:
                reduced.append(item)
        if not 2 <= len(reduced) <= 4:
            return None
        goal_candidates = tuple(item[0] for item in reduced)
        candidate_terms = [item[1] for item in reduced]
        for index, terms in enumerate(candidate_terms):
            other_terms = frozenset().union(
                *(item for other_index, item in enumerate(candidate_terms)
                  if other_index != index)
            )
            if not terms.difference(other_terms):
                return None
        return goal_candidates

    @staticmethod
    def _is_explicit_multi_presence_text(text: str) -> bool:
        return bool(
            re.search(
                r"(?:和|与|及|同时|均|都|两者|两个|多个|分别|"
                r"\bboth\b|\band\b|\ball\b|\btwo\b|\bmultiple\b)",
                str(text or "").casefold(),
            )
        )

    @staticmethod
    def _is_visible_text_read_subgoal(subgoal: Any) -> bool:
        if str(getattr(subgoal, "external_impact", "")) != "read_only":
            return False
        text = " ".join(
            (
                str(getattr(subgoal, "objective", "") or ""),
                *tuple(getattr(subgoal, "completion_conditions", ()) or ()),
            )
        ).casefold()
        read_markers = ("读取", "获取", "读出", "read", "report", "get the")
        value_markers = (
            "标题", "题头", "错误提示", "错误信息", "状态提示",
            "title", "heading", "error message", "status message",
        )
        exact_markers = (
            "等于", "包含", "逐字", "指定文字", "是否为",
            "equals", "contains", "exactly", "whether",
        )
        return (
            any(marker in text for marker in read_markers)
            and any(marker in text for marker in value_markers)
            and not any(marker in text for marker in exact_markers)
        )

    def _try_advance_visible_text_read_subgoal(
        self,
        session: UniversalAgentSessionState,
        *,
        graph: DynamicTaskGraph,
        trusted_observation: Any,
    ) -> DynamicTaskGraph | None:
        current = graph.active_subgoal()
        scene = getattr(trusted_observation, "scene", None)
        if current is None or scene is None or not self._is_visible_text_read_subgoal(current):
            return None
        allowed_meanings = ("title", "heading", "error", "status_message")
        candidates = []
        for item in tuple(getattr(scene, "elements", ()) or ()):
            meaning = str(getattr(item, "meaning", "") or "").casefold()
            label = str(getattr(item, "label", "") or "").strip()
            left, top, right, bottom = getattr(item, "bounds", (0, 0, 0, 0))
            if (
                str(getattr(item, "role", "")) in {"text", "dialog", "container"}
                and any(marker in meaning for marker in allowed_meanings)
                and label
                and item.states.get("goal_relevant") is True
                and item.states.get("fully_visible") is True
                and float(item.confidence) >= MIN_TARGET_CONFIDENCE
                and not self._candidate_has_unresolved_conflict(
                    trusted_observation, item.element_id
                )
                and 0.02 <= left < right <= 0.98
                and 0.02 <= top < bottom <= 0.98
            ):
                candidates.append(item)
        if len(candidates) != 1:
            return None
        item = candidates[0]
        visible_fact = (
            "当前可信画面读取结果："
            f"element_id={item.element_id}, role={item.role}, "
            f"meaning={item.meaning}, label={item.label}。"
        )
        observed = self.bridge.observed_state(
            graph=graph,
            trusted_observation=trusted_observation,
            action_outcome="not_applicable",
            verification={"visible_evidence": [scene.summary, visible_fact]},
        )
        lineage = session.verified_app_surface_lineage
        if lineage is not None:
            lineage_fact = json.dumps(
                {
                    "source": "verified_app_surface_lineage",
                    "app_id": lineage.app_id,
                    "app_name": lineage.app_name,
                    "surface_id": lineage.surface_id,
                    "functional_foreground_app_id": (
                        lineage.functional_foreground_app_id
                    ),
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            observed = replace(
                observed,
                grounded_visual_facts=(
                    *observed.grounded_visual_facts,
                    lineage_fact,
                ),
            )
        revised = self.deepseek_planner.replan(
            graph,
            observed,
            trigger="subgoal_completed",
            reason=(
                "当前 read_only 子目标具有唯一、完整、高置信且无冲突的"
                "文字结果候选；只能用 visible_evidence 中逐字结果完成当前节点，"
                "不得推断预设值、外部状态或执行动作。"
            ),
        )
        self._validate_graph_identity(
            revised,
            device_id=session.device_id,
            previous=graph,
            trusted_observation=trusted_observation,
            session_id=session.session_id,
            verified_app_surface_lineage=session.verified_app_surface_lineage,
            physical_actions=session.physical_actions,
        )
        old = graph.active_subgoal()
        new_old = next(
            (candidate for candidate in revised.subgoals if candidate.subgoal_id == old.subgoal_id),
            None,
        )
        if (
            new_old is None
            or new_old.status != "completed"
            or visible_fact not in new_old.completion_evidence
        ):
            raise UniversalAgentOrchestratorError(
                "DeepSeek 未使用唯一可信文字结果完成当前 read_only 子目标。"
            )
        return revised

    def _try_advance_visible_presence_subgoal(
        self,
        session: UniversalAgentSessionState,
        *,
        graph: DynamicTaskGraph,
        trusted_observation: Any,
    ) -> DynamicTaskGraph | None:
        """Use bounded visible evidence to advance one safe presence checkpoint."""

        current = graph.active_subgoal()
        if (
            current is None
            or current.external_impact not in {"read_only", "navigation_only"}
            or not self._is_presence_only_read_only_subgoal(current)
        ):
            return None
        scene = getattr(trusted_observation, "scene", None)
        if scene is None:
            return None
        candidates: tuple[Any, ...]
        presence_text = " ".join(
            (
                current.objective,
                *tuple(current.completion_conditions or ()),
            )
        )
        typed_system_surface_fact = self._typed_idempotent_system_surface_fact(
            graph,
            current.subgoal_id,
            scene,
        )
        referenced_app_pages = self._referenced_target_app_pages(
            graph=graph,
            presence_text=presence_text,
            subgoal_id=current.subgoal_id,
        )
        foreground_matches_referenced_app = bool(
            referenced_app_pages
            and self._scene_foreground_matches_target_app_page(
                scene=scene,
                target_apps=referenced_app_pages,
            )
        )
        unique_goal_candidate = scene.unique_trusted_goal_element(
            min_confidence=MIN_TARGET_CONFIDENCE,
        )
        if unique_goal_candidate is None:
            completion_reader = getattr(
                scene,
                "trusted_completion_evidence",
                None,
            )
            completion_candidates = (
                tuple(
                    completion_reader(
                        min_confidence=MIN_TARGET_CONFIDENCE,
                    )
                )
                if callable(completion_reader)
                else ()
            )
            if len(completion_candidates) == 1:
                completion_candidate = completion_candidates[0]
                competing_goal_evidence = any(
                    other.element_id != completion_candidate.element_id
                    and other.states.get("goal_relevant") is True
                    and float(other.confidence) >= MIN_TARGET_CONFIDENCE
                    for other in tuple(getattr(scene, "elements", ()) or ())
                )
                if not competing_goal_evidence:
                    # Containers and dialogs may prove one already-visible
                    # presence checkpoint, but they remain excluded from
                    # TARGET_LOCAL_ACTION_ROLES and therefore grant no action
                    # or geometry authority.
                    unique_goal_candidate = completion_candidate
        presence_terms = self._presence_binding_terms(presence_text)
        unique_candidate_terms = (
            self._presence_binding_terms(
                unique_goal_candidate.label,
                unique_goal_candidate.meaning,
                *unique_goal_candidate.evidence,
            )
            if unique_goal_candidate is not None
            else frozenset()
        )
        title_prefixes = self._presence_title_prefixes(
            current.objective,
            *tuple(current.completion_conditions or ()),
        )
        candidate_label = str(
            getattr(unique_goal_candidate, "label", "") or ""
        ).strip().casefold()
        title_prefix_grounded = bool(
            unique_goal_candidate is not None
            and title_prefixes
            and all(candidate_label.startswith(prefix) for prefix in title_prefixes)
        )
        candidate_identity_grounded = bool(
            unique_goal_candidate is not None
            and (
                title_prefix_grounded
                if title_prefixes
                else presence_terms.intersection(unique_candidate_terms)
            )
        )
        completion_identity_texts = tuple(
            str(item or "").strip()
            for item in tuple(current.completion_conditions or ())
            if str(item or "").strip()
        )
        required_surfaces = self._presence_surface_classes(
            *completion_identity_texts
        )
        destination_surface_claim = bool(
            current.external_impact == "navigation_only"
            and required_surfaces
            and required_surfaces.issubset(
                {"page", "destination", "foreground_app"}
            )
            and required_surfaces.intersection(
                {"page", "destination", "foreground_app"}
            )
        )
        named_destination_grounded = bool(
            _named_visual_identity_anchor(completion_identity_texts)
            and self._scene_named_presence_is_grounded(
                scene=scene,
                texts=completion_identity_texts,
            )
        )
        if destination_surface_claim:
            # A link, button, or list item naming a destination proves that the
            # destination can be entered; it never proves that the destination
            # is the current page.  Zero-action idempotence is allowed only
            # when the completion condition itself is grounded by the current
            # structured page identity (or by an exact target-App foreground).
            app_foreground_is_complete_destination = bool(
                foreground_matches_referenced_app
                and any(
                    self._presence_names_only_target_app_surface(
                        presence_text,
                        target_app,
                    )
                    for target_app in referenced_app_pages
                )
            )
            if not (
                app_foreground_is_complete_destination
                or named_destination_grounded
                or typed_system_surface_fact is not None
            ):
                return None
        elif not (
            foreground_matches_referenced_app
            or self._scene_named_presence_is_grounded(
                scene=scene,
                texts=(
                    current.objective,
                    *tuple(current.completion_conditions or ()),
                ),
            )
            or candidate_identity_grounded
        ):
            return None
        if referenced_app_pages and not foreground_matches_referenced_app:
            return None
        scene_identity_facts: tuple[str, ...] = ()
        if destination_surface_claim:
            # The named App/page container has already been grounded by the
            # foreground identity or page-title/container-only facts above. A
            # business control or navigation affordance that repeats the
            # destination name must neither grant nor veto this page claim.
            candidates = ()
            scene_identity_facts = (
                *self._scene_page_identity_facts(scene),
                *(
                    (typed_system_surface_fact,)
                    if typed_system_surface_fact is not None
                    else ()
                ),
            )
        elif self._is_explicit_multi_presence_text(presence_text):
            candidates = self._multi_presence_candidates(
                subgoal=current,
                scene=scene,
                trusted_observation=trusted_observation,
            ) or ()
            if not candidates:
                return None
        else:
            completion_terms = self._presence_binding_terms(
                *tuple(current.completion_conditions or ())
            )
            candidate = unique_goal_candidate
            if candidate is not None:
                candidate_terms = self._presence_binding_terms(
                    candidate.label,
                    candidate.meaning,
                    *candidate.evidence,
                )
                scene_terms = self._presence_binding_terms(
                    scene.screen_id,
                    scene.summary,
                )
                candidate_surfaces = self._intrinsic_presence_surface_classes(
                    candidate
                )
                scene_container_surfaces = self._presence_surface_classes(
                    scene.screen_id,
                    scene.summary,
                ).intersection({"page"})
                if referenced_app_pages:
                    scene_container_surfaces = scene_container_surfaces.union(
                        required_surfaces.intersection({"foreground_app"})
                    )
                required_element_surfaces = required_surfaces.difference(
                    scene_container_surfaces
                )
                # In "a card whose title starts with X", title is a visible
                # selector carried by the list item's label, not a demand that
                # the whole card itself have role=text/title.
                if (
                    "title" in required_element_surfaces
                    and title_prefix_grounded
                ):
                    required_element_surfaces = required_element_surfaces.difference(
                        {"title"}
                    )
                if (
                    not completion_terms
                    or not (
                        candidate_terms.intersection(completion_terms)
                        or scene_terms.intersection(completion_terms)
                    )
                    or not required_element_surfaces.issubset(candidate_surfaces)
                    or candidate.states.get("fully_visible") is not True
                    or self._candidate_has_unresolved_conflict(
                        trusted_observation,
                        candidate.element_id,
                    )
                ):
                    return None
                candidates = (candidate,)
            elif current.external_impact == "navigation_only":
                presence_terms = self._presence_binding_terms(presence_text)
                summary_terms = self._presence_binding_terms(scene.summary)
                if not presence_terms or not presence_terms.intersection(summary_terms):
                    return None
                scene_container_surfaces = self._presence_surface_classes(
                    scene.screen_id,
                    scene.summary,
                ).intersection({"page"})
                if referenced_app_pages:
                    scene_container_surfaces = scene_container_surfaces.union(
                        required_surfaces.intersection({"foreground_app"})
                    )
                required_element_surfaces = required_surfaces.difference(
                    scene_container_surfaces
                )
                matched = []
                for item in scene.elements:
                    item_terms = self._presence_binding_terms(
                        item.label,
                        item.meaning,
                        *item.evidence,
                    )
                    # Container identity (for example, "the current page") is
                    # a scene-level fact.  The required control type must be
                    # intrinsic to the candidate itself; a nearby instruction
                    # merely mentioning an input must not become that input.
                    intrinsic_item_surfaces = (
                        self._intrinsic_presence_surface_classes(item)
                    )
                    if (
                        not presence_terms.intersection(item_terms)
                        or not required_element_surfaces.issubset(
                            intrinsic_item_surfaces
                        )
                    ):
                        continue
                    if (
                        float(item.confidence) < MIN_TARGET_CONFIDENCE
                        or item.states.get("fully_visible") is not True
                        or self._candidate_has_unresolved_conflict(
                            trusted_observation,
                            item.element_id,
                        )
                    ):
                        return None
                    left, top, right, bottom = item.bounds
                    if not (
                        0.02 <= left < right <= 0.98
                        and 0.02 <= top < bottom <= 0.98
                    ):
                        return None
                    matched.append(item)
                if not 1 <= len(matched) <= 4:
                    return None
                candidates = tuple(matched)
            else:
                return None

        candidate_facts = tuple(
            "当前可信画面的目标元素："
            f"element_id={item.element_id}, role={item.role}, "
            f"label={item.label or '[empty]'}, meaning={item.meaning}, "
            f"confidence={float(item.confidence):.3f}, "
            f"fully_visible={item.states.get('fully_visible', 'unknown')}, "
            "bounds_inside_safe_frame=true。"
            for item in candidates
        )
        focused_input_fact = self._verified_focused_input_fact(
            trusted_observation
        )
        local_state_facts = (
            (focused_input_fact,) if focused_input_fact is not None else ()
        )
        observed = self.bridge.observed_state(
            graph=graph,
            trusted_observation=trusted_observation,
            action_outcome="not_applicable",
            verification={
                "visible_evidence": [
                    scene.summary,
                    *scene_identity_facts,
                    *candidate_facts,
                    *local_state_facts,
                    *(fact for item in candidates for fact in item.evidence),
                ]
            },
        )
        revised = self.deepseek_planner.replan(
            graph,
            observed,
            trigger="subgoal_completed",
            reason=(
                "当前可信画面已经以严格 scene identity 或逐项语义绑定、"
                f"高置信且无冲突的目标元素证明定位类 {current.external_impact} 子目标；"
                f"本轮必须先把当前 subgoal_id={current.subgoal_id} 标为 completed，"
                "其 completion_evidence 必须逐字选择 visible_evidence 中至少一项。"
                "只有直接依赖连续、且 visible_evidence 已提供对应逐字本地控件状态事实的"
                "可逆状态节点可以同时完成；最多激活一个直接后继，其他节点不得越级完成。"
                "不得推断元素值、"
                "外部状态或执行动作；无法满足这些约束时必须 blocked。"
            ),
        )
        self._validate_graph_identity(
            revised,
            device_id=session.device_id,
            previous=graph,
            trusted_observation=trusted_observation,
            session_id=session.session_id,
            verified_app_surface_lineage=session.verified_app_surface_lineage,
            physical_actions=session.physical_actions,
        )
        if revised.revision != graph.revision + 1:
            raise UniversalAgentOrchestratorError(
                "可见状态证据推进必须且只能产生一个新 revision。"
            )
        old_ids = tuple(item.subgoal_id for item in graph.subgoals)
        new_ids = tuple(item.subgoal_id for item in revised.subgoals)
        if old_ids != new_ids:
            raise UniversalAgentOrchestratorError(
                "可见状态证据推进不得增加、删除或重排子目标。"
            )
        old_by_id = {item.subgoal_id: item for item in graph.subgoals}
        new_by_id = {item.subgoal_id: item for item in revised.subgoals}
        if (
            revised.goal != graph.goal
            or revised.constraints != graph.constraints
            or revised.risk_actions != graph.risk_actions
        ):
            raise UniversalAgentOrchestratorError(
                "可见状态证据推进不得修改目标、约束或效果定义。"
            )
        _validate_visible_completion_condition_progress(graph, revised, observed)
        for subgoal_id in old_ids:
            old_item = old_by_id[subgoal_id]
            new_item = new_by_id[subgoal_id]
            if (
                new_item.objective != old_item.objective
                or new_item.depends_on != old_item.depends_on
                or new_item.constraints != old_item.constraints
                or new_item.completion_conditions != old_item.completion_conditions
                or new_item.risk_action_ids != old_item.risk_action_ids
                or new_item.external_impact != old_item.external_impact
            ):
                raise UniversalAgentOrchestratorError(
                    "可见状态证据推进只能改变子目标状态和完成证据。"
                )
        newly_completed = tuple(
            subgoal_id
            for subgoal_id in old_ids
            if old_by_id[subgoal_id].status != "completed"
            and new_by_id[subgoal_id].status == "completed"
        )
        completed_current = new_by_id[current.subgoal_id]
        completed_before = {
            item.subgoal_id for item in graph.subgoals if item.status == "completed"
        }
        accepted_prefix: list[str] = []
        prefix_valid = bool(
            newly_completed and newly_completed[0] == current.subgoal_id
        )
        unsupported_subgoal_id = ""
        for subgoal_id in newly_completed:
            old_item = old_by_id[subgoal_id]
            new_item = new_by_id[subgoal_id]
            dependencies_ready = all(
                dependency in completed_before or dependency in accepted_prefix
                for dependency in old_item.depends_on
            )
            state_fact = self._zero_action_visible_state_fact(
                old_item,
                trusted_observation,
            )
            presence_eligible = self._is_presence_only_read_only_subgoal(
                old_item
            )
            state_eligible = bool(
                state_fact
                and state_fact in new_item.completion_evidence
            )
            if (
                old_item.external_impact not in {"read_only", "navigation_only"}
                or not (presence_eligible or state_eligible)
                or not dependencies_ready
                or not new_item.completion_evidence
            ):
                prefix_valid = False
                unsupported_subgoal_id = subgoal_id
                break
            accepted_prefix.append(subgoal_id)
        if not prefix_valid or not completed_current.completion_evidence:
            narrowed = self._narrow_unproven_visible_successor(
                previous=graph,
                revised=revised,
                current_subgoal_id=current.subgoal_id,
                accepted_prefix=tuple(accepted_prefix),
                unsupported_subgoal_id=unsupported_subgoal_id,
            )
            if narrowed is None:
                raise UniversalAgentOrchestratorError(
                    "可见状态证据只能完成从当前节点开始、依赖连续满足的安全定位前缀，"
                    "且每个节点必须记录可见证据："
                    f"current={current.subgoal_id}, newly_completed={newly_completed}, "
                    f"current_evidence_count={len(completed_current.completion_evidence)}。"
                )
            revised = narrowed
            new_by_id = {item.subgoal_id: item for item in revised.subgoals}
            newly_completed = tuple(
                subgoal_id
                for subgoal_id in old_ids
                if old_by_id[subgoal_id].status != "completed"
                and new_by_id[subgoal_id].status == "completed"
            )
            completed_current = new_by_id[current.subgoal_id]
            prefix_valid = True
        for subgoal_id in old_ids:
            old_status = old_by_id[subgoal_id].status
            new_status = new_by_id[subgoal_id].status
            if subgoal_id in newly_completed:
                continue
            if old_status == "completed" and new_status != "completed":
                raise UniversalAgentOrchestratorError(
                    "可见状态证据推进不得回退已完成子目标。"
                )
            if old_status == "pending" and new_status not in {"pending", "active"}:
                raise UniversalAgentOrchestratorError(
                    "可见状态证据推进不得越过后续子目标。"
                )
        newly_active = tuple(
            subgoal_id
            for subgoal_id in old_ids
            if old_by_id[subgoal_id].status == "pending"
            and new_by_id[subgoal_id].status == "active"
        )
        if len(newly_active) > 1:
            raise UniversalAgentOrchestratorError(
                "可见状态证据推进最多只能激活一个后续子目标。"
            )
        if revised.status != "completed" and (
            len(newly_active) != 1
            or revised.active_subgoal_id != newly_active[0]
        ):
            raise UniversalAgentOrchestratorError(
                "可见状态证据推进后必须精确激活一个后续子目标。"
            )
        return revised

    @staticmethod
    def _narrow_unproven_visible_successor(
        *,
        previous: DynamicTaskGraph,
        revised: DynamicTaskGraph,
        current_subgoal_id: str,
        accepted_prefix: tuple[str, ...],
        unsupported_subgoal_id: str,
    ) -> DynamicTaskGraph | None:
        """Keep only the locally proven part of a model-completed prefix.

        The projection is deliberately one-way: it may revoke an unsupported
        completion, but it can never complete a node, add evidence, or widen
        action authority.  This lets a valid current visible checkpoint survive
        when model prose over-claims one directly dependent reversible state.
        """

        if (
            not accepted_prefix
            or accepted_prefix[0] != current_subgoal_id
            or not unsupported_subgoal_id
        ):
            return None
        old_by_id = {item.subgoal_id: item for item in previous.subgoals}
        new_by_id = {item.subgoal_id: item for item in revised.subgoals}
        unsupported = old_by_id.get(unsupported_subgoal_id)
        last_accepted_id = accepted_prefix[-1]
        accepted = set(accepted_prefix)
        previously_completed = {
            item.subgoal_id
            for item in previous.subgoals
            if item.status == "completed"
        }
        if (
            unsupported is None
            or unsupported_subgoal_id not in new_by_id
            or unsupported.status != "pending"
            or new_by_id[unsupported_subgoal_id].status != "completed"
            or unsupported.external_impact not in {"read_only", "navigation_only"}
            or last_accepted_id not in unsupported.depends_on
            or not all(
                dependency in previously_completed or dependency in accepted
                for dependency in unsupported.depends_on
            )
        ):
            return None

        normalized = []
        for old_item in previous.subgoals:
            subgoal_id = old_item.subgoal_id
            if subgoal_id in accepted or old_item.status == "completed":
                normalized.append(new_by_id[subgoal_id])
            elif subgoal_id == unsupported_subgoal_id:
                normalized.append(
                    replace(
                        old_item,
                        status="active",
                        completion_evidence=(),
                    )
                )
            else:
                normalized.append(
                    replace(old_item, completion_evidence=())
                )
        narrowed = replace(
            revised,
            status="ready",
            subgoals=tuple(normalized),
            active_subgoal_id=unsupported_subgoal_id,
            clarification_questions=previous.clarification_questions,
        )
        narrowed.validate()
        return narrowed

    def _advance_visible_presence_prefix(
        self,
        session: UniversalAgentSessionState,
        *,
        graph: DynamicTaskGraph,
        trusted_observation: Any,
    ) -> tuple[DynamicTaskGraph, int]:
        """Consume a bounded, independently validated visible-state prefix."""

        current_graph = graph
        advances = 0
        while advances < MAX_VISIBLE_PRESENCE_ADVANCES_PER_OBSERVATION:
            revised = self._try_advance_visible_presence_subgoal(
                session,
                graph=current_graph,
                trusted_observation=trusted_observation,
            )
            if revised is None:
                break
            self._store_revised_graph(session, revised)
            current_graph = revised
            advances += 1
            if revised.status == "completed":
                break
        return current_graph, advances

    def _store_revised_graph(
        self,
        session: UniversalAgentSessionState,
        revised: DynamicTaskGraph,
    ) -> None:
        session.task_graph = revised
        session.goal_draft = self.bridge.goal_draft(revised)
        session.confirmation_authority = None
        session.effect_confirmation_authority = None
        session.confirmed_effect_ids = ()
        self._remember(
            session,
            session.evidence_store.write_task_graph(revised),
            session.evidence_store.write_effect_policy_snapshot(revised),
        )

    @staticmethod
    def _selection_receipt_payload(
        decision: CanonicalSelectionReceipt,
    ) -> dict[str, Any]:
        return decision.to_dict()

    @classmethod
    def _selection_receipt(
        cls,
        session: UniversalAgentSessionState,
        decision: Any,
    ) -> CanonicalSelectionReceipt:
        """Record the already-validated canonical choice without judging it again."""

        proposal = getattr(decision, "proposal", None)
        action = getattr(proposal, "action", None)
        if proposal is None or getattr(proposal, "status", "") != "action":
            return CanonicalSelectionReceipt(
                allowed=False,
                reason="当前单步决策没有唯一 canonical 动作。",
            )
        action_kind = str(getattr(action, "action", "") or "").strip()
        available = cls._available_action_kinds(session)
        if action_kind not in available:
            return CanonicalSelectionReceipt(
                allowed=False,
                reason=f"当前设备没有 canonical 动作能力：{action_kind or 'missing'}。",
            )
        return CanonicalSelectionReceipt(
            allowed=True,
            reason=(
                "单步 Qwen 决策已绑定当前 observation、canonical candidate "
                "和 typed transition；后续只消费同一 scope。"
            ),
            canonical_class=action_kind,
        )

    @classmethod
    def _validate_newly_completed_named_app_surfaces(
        cls,
        *,
        previous: DynamicTaskGraph,
        revised: DynamicTaskGraph,
        trusted_observation: Any,
        session_id: str = "",
        verified_transition: VerifiedActionTransition | None = None,
        controller_transition_evidence_refs: tuple[
            ControllerTransitionEvidenceRef, ...
        ] = (),
        before_observation: Any | None = None,
        previous_decision: Any | None = None,
        execution_result: Any | None = None,
        verified_app_surface_lineage: VerifiedAppSurfaceLineage | None = None,
        physical_actions: int = 0,
    ) -> None:
        scene = getattr(trusted_observation, "scene", None)
        if scene is None:
            raise UniversalAgentOrchestratorError(
                "DeepSeek revision 缺少可复核的可信场景。"
            )
        old_by_id = {item.subgoal_id: item for item in previous.subgoals}
        for item in revised.subgoals:
            old = old_by_id.get(item.subgoal_id)
            if item.status != "completed" or (
                old is not None and old.status == "completed"
            ):
                continue
            presence_text = " ".join(
                (item.objective, *tuple(item.completion_conditions or ()))
            )
            referenced_app_pages = cls._referenced_target_app_pages(
                graph=previous,
                presence_text=presence_text,
                subgoal_id=item.subgoal_id,
            )
            if referenced_app_pages and not cls._scene_foreground_matches_target_app_page(
                scene=scene,
                target_apps=referenced_app_pages,
            ) and not cls._verified_transition_proves_named_app_surface(
                previous=previous,
                completed_subgoal=item,
                target_apps=referenced_app_pages,
                trusted_observation=trusted_observation,
                session_id=session_id,
                verified_transition=verified_transition,
                controller_transition_evidence_refs=(
                    controller_transition_evidence_refs
                ),
                before_observation=before_observation,
                previous_decision=previous_decision,
                execution_result=execution_result,
            ) and not cls._verified_lineage_proves_named_app_surface(
                previous=previous,
                completed_subgoal=item,
                target_apps=referenced_app_pages,
                trusted_observation=trusted_observation,
                session_id=session_id,
                verified_app_surface_lineage=verified_app_surface_lineage,
                physical_actions=physical_actions,
            ):
                raise UniversalAgentOrchestratorError(
                    "Launcher 或其他页面中的 App 入口不能证明目标 App 页面已在前台："
                    f"subgoal_id={item.subgoal_id}。"
                )

    @classmethod
    def _verified_transition_proves_named_app_surface(
        cls,
        *,
        previous: DynamicTaskGraph,
        completed_subgoal: Any,
        target_apps: tuple[Any, ...],
        trusted_observation: Any,
        session_id: str,
        verified_transition: VerifiedActionTransition | None,
        controller_transition_evidence_refs: tuple[
            ControllerTransitionEvidenceRef, ...
        ],
        before_observation: Any | None,
        previous_decision: Any | None,
        execution_result: Any | None,
    ) -> bool:
        """Accept only a controller-proven launcher-to-App surface transition.

        A visual observer may describe the destination by its current function
        (for example, a news feed) rather than by the enclosing App identity.
        That functional classification is not rewritten here.  This bounded
        proof completes only the navigation node whose exact launcher action,
        typed surface expectation and one-action receipt all agree.
        """

        receipt = verified_transition
        before_scene = getattr(before_observation, "scene", None)
        after_scene = getattr(trusted_observation, "scene", None)
        proposal = getattr(previous_decision, "proposal", None)
        action = getattr(proposal, "action", None)
        if (
            receipt is None
            or before_scene is None
            or after_scene is None
            or action is None
            or execution_result is None
            or not session_id
            or str(getattr(completed_subgoal, "external_impact", ""))
            != "navigation_only"
            or str(getattr(action, "action", "")) != "tap_semantic"
        ):
            return False
        try:
            receipt.validate()
            for ref in controller_transition_evidence_refs:
                ref.validate()
        except TaskGraphError:
            return False
        old_active = previous.active_subgoal()
        if old_active is None or old_active.subgoal_id != completed_subgoal.subgoal_id:
            return False
        if (
            receipt.session_id != session_id
            or receipt.task_id != previous.task_id
            or receipt.device_id != previous.device_id
            or receipt.prior_revision != previous.revision
            or receipt.subgoal_id != completed_subgoal.subgoal_id
            or receipt.decision_node_id != str(getattr(action, "node_id", ""))
            or receipt.action_kind != "tap_semantic"
            or receipt.action_digest != _action_digest(action)
            or receipt.rebound_action_digest
            != _action_digest(getattr(execution_result, "rebound_action", None))
            or receipt.resolved_action_digest
            != _action_digest(getattr(execution_result, "resolved_action", None))
            or receipt.outcome != "matched"
            or receipt.physical_actions != 1
            or receipt.errors
            or receipt.before_observation_id
            != str(getattr(before_observation, "observation_id", ""))
            or receipt.before_fingerprint
            != str(getattr(before_observation, "fingerprint", ""))
            or receipt.after_observation_id
            != str(getattr(trusted_observation, "observation_id", ""))
            or receipt.after_fingerprint
            != str(getattr(trusted_observation, "fingerprint", ""))
            or receipt.before_fingerprint == receipt.after_fingerprint
        ):
            return False
        # The current single-step visual contract records the verified outcome
        # directly on the transition receipt.  It intentionally no longer emits
        # a second controller-completion sentence for ordinary navigation.  Do
        # not restore that retired duplicate proof as a hidden veto: the exact
        # session/task/revision/action digests, fresh before/after observation
        # identities, one physical action, matched outcome and launcher exit are
        # already checked above and below.  If optional controller refs exist,
        # they were validated above, but their absence is not a failure.
        if (
            str(getattr(before_scene, "foreground_app_id", "")).casefold()
            != "launcher"
            or str(getattr(after_scene, "foreground_app_id", "")).casefold()
            == "launcher"
        ):
            return False

        params = getattr(action, "params", None)
        if not isinstance(params, Mapping):
            return False
        element_id = str(params.get("element_id") or "").strip()
        before_elements = tuple(getattr(before_scene, "elements", ()) or ())
        matches = tuple(
            element
            for element in before_elements
            if str(getattr(element, "element_id", "")) == element_id
        )
        if len(matches) != 1:
            return False
        element = matches[0]
        if (
            str(params.get("label") or "").strip()
            != str(getattr(element, "label", "") or "").strip()
            or str(params.get("role") or "").strip()
            != str(getattr(element, "role", "") or "").strip()
            or str(params.get("target") or params.get("meaning") or "").strip()
            != str(getattr(element, "meaning", "") or "").strip()
        ):
            return False

        action_terms = cls._presence_binding_terms(
            params.get("label"),
            params.get("target"),
            params.get("meaning"),
            getattr(element, "label", ""),
            getattr(element, "meaning", ""),
        )
        bound_targets = tuple(
            target_app
            for target_app in target_apps
            if cls._target_app_identity_terms(
                target_app.app_id,
                target_app.app_name,
            ).intersection(action_terms)
        )
        if len(bound_targets) != 1:
            return False
        try:
            semantic_ir = compile_formal_semantic_authority(previous).semantic_ir
        except Exception:
            return False
        target = bound_targets[0]
        surface_ids = {
            surface.surface_id
            for surface in semantic_ir.surfaces
            if surface.kind == "app"
            and (
                surface.app_id.casefold() == str(target.app_id).casefold()
                or surface.app_name.casefold() == str(target.app_name).casefold()
            )
        }
        formal_transition = params.get("formal_transition")
        expectations = (
            formal_transition.get("expectations")
            if isinstance(formal_transition, Mapping)
            else None
        )
        if (
            not isinstance(expectations, list)
            or formal_transition.get("exploratory") is not False
        ):
            return False
        matching_expectations = [
            expectation
            for expectation in expectations
            if isinstance(expectation, Mapping)
            and expectation.get("subject_ref") == "surface_current"
            and expectation.get("predicate") == "surface.active_ref"
            and expectation.get("operator") == "equals"
            and expectation.get("value") in surface_ids
        ]
        return bool(surface_ids) and len(matching_expectations) == 1

    @classmethod
    def _verified_lineage_proves_named_app_surface(
        cls,
        *,
        previous: DynamicTaskGraph,
        completed_subgoal: Any,
        target_apps: tuple[Any, ...],
        trusted_observation: Any,
        session_id: str,
        verified_app_surface_lineage: VerifiedAppSurfaceLineage | None,
        physical_actions: int,
    ) -> bool:
        lineage = verified_app_surface_lineage
        scene = getattr(trusted_observation, "scene", None)
        if lineage is None or scene is None:
            return False
        if (
            not session_id
            or lineage.session_id != session_id
            or lineage.task_id != previous.task_id
            or lineage.device_id != previous.device_id
            or lineage.physical_actions != physical_actions
            or str(getattr(scene, "foreground_app_id", "")).casefold()
            != lineage.functional_foreground_app_id.casefold()
            or lineage.functional_foreground_app_id.casefold() == "launcher"
        ):
            return False
        if not any(
            str(app.app_id).casefold() == lineage.app_id.casefold()
            and str(app.app_name).casefold() == lineage.app_name.casefold()
            for app in target_apps
        ):
            return False
        by_id = {item.subgoal_id: item for item in previous.subgoals}
        source = by_id.get(lineage.source_subgoal_id)
        if (
            source is None
            or source.status != "completed"
        ):
            return False
        pending = list(getattr(completed_subgoal, "depends_on", ()) or ())
        visited: set[str] = set()
        while pending:
            dependency_id = pending.pop()
            if dependency_id == lineage.source_subgoal_id:
                return True
            if dependency_id in visited:
                continue
            visited.add(dependency_id)
            dependency = by_id.get(dependency_id)
            if dependency is not None:
                pending.extend(dependency.depends_on)
        return False

    @classmethod
    def _build_verified_app_surface_lineage(
        cls,
        *,
        session: UniversalAgentSessionState,
        previous: DynamicTaskGraph,
        revised: DynamicTaskGraph,
        trusted_observation: Any,
        receipt: VerifiedActionTransition | None,
        controller_refs: tuple[ControllerTransitionEvidenceRef, ...],
        before_observation: Any,
        previous_decision: Any,
        execution_result: Any,
    ) -> VerifiedAppSurfaceLineage | None:
        old_by_id = {item.subgoal_id: item for item in previous.subgoals}
        for item in revised.subgoals:
            old = old_by_id.get(item.subgoal_id)
            if item.status != "completed" or old is None or old.status == "completed":
                continue
            text = " ".join((item.objective, *item.completion_conditions))
            target_apps = cls._referenced_target_app_pages(
                graph=previous,
                presence_text=text,
                subgoal_id=item.subgoal_id,
            )
            if not target_apps or not cls._verified_transition_proves_named_app_surface(
                previous=previous,
                completed_subgoal=item,
                target_apps=target_apps,
                trusted_observation=trusted_observation,
                session_id=session.session_id,
                verified_transition=receipt,
                controller_transition_evidence_refs=controller_refs,
                before_observation=before_observation,
                previous_decision=previous_decision,
                execution_result=execution_result,
            ):
                continue
            action = previous_decision.proposal.action
            terms = cls._presence_binding_terms(
                action.params.get("label"), action.params.get("target")
            )
            bound = [
                app for app in target_apps
                if cls._target_app_identity_terms(
                    app.app_id, app.app_name
                ).intersection(terms)
            ]
            expectations = action.params["formal_transition"]["expectations"]
            surface_id = next(
                str(expectation["value"])
                for expectation in expectations
                if expectation.get("predicate") == "surface.active_ref"
                and expectation.get("operator") == "equals"
            )
            if len(bound) != 1 or receipt is None:
                return None
            return VerifiedAppSurfaceLineage(
                session_id=session.session_id,
                task_id=previous.task_id,
                device_id=previous.device_id,
                app_id=str(bound[0].app_id),
                app_name=str(bound[0].app_name),
                surface_id=surface_id,
                source_receipt_id=receipt.receipt_id,
                source_subgoal_id=receipt.subgoal_id,
                functional_foreground_app_id=str(
                    trusted_observation.scene.foreground_app_id
                ),
                physical_actions=session.physical_actions,
            )
        return None

    @classmethod
    def _carry_verified_app_surface_lineage(
        cls,
        *,
        session: UniversalAgentSessionState,
        previous: DynamicTaskGraph,
        revised: DynamicTaskGraph,
        trusted_observation: Any,
        prior_lineage: VerifiedAppSurfaceLineage | None,
        prior_physical_actions: int,
        execution_result: Any,
    ) -> VerifiedAppSurfaceLineage | None:
        """Advance one internally minted App identity across a verified step.

        The launcher-to-App receipt establishes the only alias between the
        task's typed App identity and the runtime foreground package.  A later
        action makes the old physical-action scope stale, but it must not erase
        that alias when fresh post-action evidence proves that execution stayed
        inside the same App and the current subgoal is still a descendant of
        the receipt-proven entry node.
        """

        lineage = prior_lineage
        scene = getattr(trusted_observation, "scene", None)
        physical_delta = int(getattr(execution_result, "physical_actions", 0))
        resolved_kind = str(
            getattr(getattr(execution_result, "resolved_action", None), "kind", "")
        )
        valid_delta = physical_delta == 1 or (
            physical_delta == 0 and resolved_kind == "wait_for_change"
        )
        if (
            lineage is None
            or scene is None
            or str(getattr(execution_result, "action_outcome", ""))
            not in POST_ACTION_OUTCOMES
            or not valid_delta
            or lineage.session_id != session.session_id
            or lineage.task_id != previous.task_id
            or lineage.task_id != revised.task_id
            or lineage.device_id != session.device_id
            or lineage.device_id != previous.device_id
            or lineage.device_id != revised.device_id
            or lineage.physical_actions != prior_physical_actions
            or session.physical_actions != prior_physical_actions + physical_delta
            or not cls._lineage_matches_observed_foreground(
                lineage,
                str(getattr(scene, "foreground_app_id", "")),
            )
        ):
            return None

        revised_by_id = {item.subgoal_id: item for item in revised.subgoals}
        source = revised_by_id.get(lineage.source_subgoal_id)
        if source is None or source.status != "completed":
            return None
        current = revised.active_subgoal()
        if current is None:
            return (
                replace(lineage, physical_actions=session.physical_actions)
                if revised.status == "completed"
                else None
            )

        pending = list(current.depends_on)
        visited: set[str] = set()
        while pending:
            dependency_id = pending.pop()
            if dependency_id == lineage.source_subgoal_id:
                return replace(
                    lineage,
                    physical_actions=session.physical_actions,
                )
            if dependency_id in visited:
                continue
            visited.add(dependency_id)
            dependency = revised_by_id.get(dependency_id)
            if dependency is not None:
                pending.extend(dependency.depends_on)
        return None

    @classmethod
    def _refresh_verified_app_surface_lineage(
        cls,
        *,
        session: UniversalAgentSessionState,
        graph: DynamicTaskGraph,
        prior_observation: Any,
        new_observation: Any,
    ) -> VerifiedAppSurfaceLineage | None:
        """Upgrade one App alias across a read-only observer identity change.

        A post-action model scene may name the foreground by the typed App ID,
        while the next local audit reports the runtime package.  With no
        intervening physical action, a unique shared page title proves surface
        continuity strongly enough to replace only the runtime identity.  It
        does not grant a new App entry or action authority.
        """

        lineage = session.verified_app_surface_lineage
        prior_scene = getattr(prior_observation, "scene", None)
        new_scene = getattr(new_observation, "scene", None)
        if (
            lineage is None
            or prior_scene is None
            or new_scene is None
            or lineage.session_id != session.session_id
            or lineage.task_id != graph.task_id
            or lineage.device_id != session.device_id
            or lineage.device_id != graph.device_id
            or lineage.physical_actions != session.physical_actions
            or not cls._lineage_matches_observed_foreground(
                lineage,
                str(getattr(prior_scene, "foreground_app_id", "")),
            )
            or str(getattr(new_scene, "foreground_app_id", "")).casefold()
            == "launcher"
            or not bool(getattr(prior_scene, "stable", False))
            or not bool(getattr(new_scene, "stable", False))
        ):
            return None
        if cls._lineage_matches_observed_foreground(
            lineage,
            str(getattr(new_scene, "foreground_app_id", "")),
        ):
            return lineage

        def page_titles(scene: Any) -> frozenset[str]:
            values = []
            for element in tuple(getattr(scene, "elements", ()) or ()):
                meaning = str(getattr(element, "meaning", "") or "").casefold()
                label = str(getattr(element, "label", "") or "").strip()
                states = getattr(element, "states", {}) or {}
                if (
                    label
                    and (
                        meaning == "page_title"
                        or meaning.endswith("_page_title")
                        or meaning.endswith("_screen_title")
                    )
                    and states.get("fully_visible") is not False
                    and float(getattr(element, "confidence", 0.0))
                    >= MIN_TARGET_CONFIDENCE
                ):
                    values.append(label.casefold())
            return frozenset(values)

        shared_titles = page_titles(prior_scene) & page_titles(new_scene)
        if len(shared_titles) != 1:
            return None

        by_id = {item.subgoal_id: item for item in graph.subgoals}
        source = by_id.get(lineage.source_subgoal_id)
        current = graph.active_subgoal()
        if source is None or source.status != "completed" or current is None:
            return None
        pending = list(current.depends_on)
        visited: set[str] = set()
        while pending:
            dependency_id = pending.pop()
            if dependency_id == lineage.source_subgoal_id:
                return replace(
                    lineage,
                    functional_foreground_app_id=str(
                        getattr(new_scene, "foreground_app_id", "")
                    ),
                )
            if dependency_id in visited:
                continue
            visited.add(dependency_id)
            dependency = by_id.get(dependency_id)
            if dependency is not None:
                pending.extend(dependency.depends_on)
        return None

    @classmethod
    def _validate_graph_identity(
        cls,
        graph: DynamicTaskGraph,
        *,
        device_id: str,
        previous: DynamicTaskGraph | None = None,
        trusted_observation: Any | None = None,
        session_id: str = "",
        verified_transition: VerifiedActionTransition | None = None,
        controller_transition_evidence_refs: tuple[
            ControllerTransitionEvidenceRef, ...
        ] = (),
        before_observation: Any | None = None,
        previous_decision: Any | None = None,
        execution_result: Any | None = None,
        verified_app_surface_lineage: VerifiedAppSurfaceLineage | None = None,
        physical_actions: int = 0,
    ) -> None:
        graph.validate()
        if graph.device_id != device_id:
            raise UniversalAgentOrchestratorError(
                "DeepSeek 任务图 device_id 与会话设备不一致。"
            )
        if previous is not None:
            if graph.task_id != previous.task_id or graph.device_id != previous.device_id:
                raise UniversalAgentOrchestratorError(
                    "DeepSeek 重规划改变了 task_id 或 device_id。"
                )
            if graph.revision != previous.revision + 1:
                raise UniversalAgentOrchestratorError(
                    "DeepSeek 重规划 revision 必须严格等于上一 revision + 1。"
                )
            if trusted_observation is not None:
                cls._validate_newly_completed_named_app_surfaces(
                    previous=previous,
                    revised=graph,
                    trusted_observation=trusted_observation,
                    session_id=session_id,
                    verified_transition=verified_transition,
                    controller_transition_evidence_refs=(
                        controller_transition_evidence_refs
                    ),
                    before_observation=before_observation,
                    previous_decision=previous_decision,
                    execution_result=execution_result,
                    verified_app_surface_lineage=verified_app_surface_lineage,
                    physical_actions=physical_actions,
                )

    @staticmethod
    def _validate_decision_binding(
        graph: DynamicTaskGraph,
        observation: Any,
        decision: Any,
    ) -> None:
        expected = {
            "task_id": graph.task_id,
            "device_id": graph.device_id,
            "revision": graph.revision,
            "observation_id": str(
                getattr(observation, "observation_id", "")
            ),
            "fingerprint": str(getattr(observation, "fingerprint", "")),
        }
        for field_name, expected_value in expected.items():
            actual = getattr(decision, field_name, None)
            if actual != expected_value:
                raise UniversalAgentOrchestratorError(
                    f"Qwen 决策 {field_name} 与当前权威状态不一致。"
                )
        bound = getattr(decision, "trusted_observation", None)
        if bound is None or (
            str(getattr(bound, "observation_id", ""))
            != expected["observation_id"]
            or str(getattr(bound, "fingerprint", ""))
            != expected["fingerprint"]
        ):
            raise UniversalAgentOrchestratorError(
                "Qwen 决策没有绑定当前可信观察。"
            )
        proposal = getattr(decision, "proposal", None)
        if proposal is None:
            raise UniversalAgentOrchestratorError("Qwen 决策缺少 proposal。")
        try:
            proposal.validate(observation.scene)
        except (CanonicalActionProtocolError, AttributeError, TypeError) as exc:
            raise UniversalAgentOrchestratorError(
                f"Qwen 决策 proposal 不符合 canonical 合同：{exc}"
            ) from exc

    @staticmethod
    def _remember(session: UniversalAgentSessionState, *paths: Any) -> None:
        for path in paths:
            if path is None:
                continue
            values = path if isinstance(path, (list, tuple)) else (path,)
            for item in values:
                text = str(item or "").strip()
                if text and text not in session.evidence_paths:
                    session.evidence_paths.append(text)

    def _record_deepseek_failure(
        self,
        session: UniversalAgentSessionState,
        error: Exception,
        *,
        stage: str,
        previous_graph: DynamicTaskGraph | None = None,
    ) -> None:
        if not isinstance(error, TaskGraphError):
            return
        try:
            paths = self.deepseek_failure_diagnostic_writer(
                self.deepseek_planner,
                evidence_dir=session.run_dir,
                prefix=f"deepseek_{stage}_step_{session.step_number}",
                failed_stage=stage,
                error=error,
                previous_graph=previous_graph,
            )
        except Exception:
            return
        self._remember(session, paths)

    def _write_terminal_snapshot(self, session: UniversalAgentSessionState) -> None:
        if session.vision_usage is not None:
            self._remember(
                session,
                session.evidence_store.write_json(
                    "qwen_usage.json",
                    session.vision_usage.to_dict(),
                ),
            )
        session_path = session.evidence_store.write_session(session)
        self._remember(session, session_path)
        report_path = session.evidence_store.write_report(
            {
                "mode": "universal_agent_safe_live_loop",
                "policy_version": CANONICAL_SELECTION_RECEIPT_VERSION,
                "session": session.snapshot(),
            }
        )
        self._remember(session, report_path)

    def _ensure_terminal_snapshot(self, session: UniversalAgentSessionState) -> None:
        """Write a terminal report only when the persisted one is missing/stale."""

        try:
            report = session.evidence_store.read_report()
            if report is None:
                raise KeyError("report")
            persisted = report["session"]
            current = session.snapshot()
            compared_fields = (
                "status",
                "failed_reason",
                "confirm_stage",
                "physical_actions",
                "qwen_usage",
                "last_post_action_transition",
                "last_confirmation_failure",
            )
            if all(persisted.get(key) == current.get(key) for key in compared_fields):
                return
        except (EvidenceStoreError, KeyError, TypeError):
            pass
        self._write_terminal_snapshot(session)

    def _current_confirmation_scope(
        self,
        session: UniversalAgentSessionState,
    ) -> dict[str, Any]:
        graph = session.task_graph
        observation = session.trusted_observation
        decision = session.qwen_decision
        if graph is None or observation is None or decision is None:
            raise UniversalAgentOrchestratorError(
                "当前会话没有完整的确认权威状态。"
            )
        self._validate_graph_identity(graph, device_id=session.device_id)
        self._validate_decision_binding(graph, observation, decision)
        if decision.proposal.status != "action":
            raise UniversalAgentOrchestratorError(
                "非 action 决策没有可确认动作。"
            )
        current = graph.active_subgoal()
        if current is None:
            raise UniversalAgentOrchestratorError("当前任务没有活动子目标。")
        return {
            "session_id": session.session_id,
            "task_id": graph.task_id,
            "device_id": graph.device_id,
            "revision": graph.revision,
            "subgoal_id": current.subgoal_id,
            "effect_ids": sorted(current.risk_action_ids),
            "observation_id": str(observation.observation_id),
            "fingerprint": str(observation.fingerprint),
            "decision_node_id": str(decision.proposal.action.node_id),
            "action_digest": _action_digest(decision.proposal.action),
        }

    def _bind_confirmation(self, session: UniversalAgentSessionState) -> None:
        scope = self._current_confirmation_scope(session)
        session.confirmation_authority = ConfirmationAuthority(
            session_id=scope["session_id"],
            task_id=scope["task_id"],
            device_id=scope["device_id"],
            revision=scope["revision"],
            subgoal_id=scope["subgoal_id"],
            effect_ids=tuple(scope["effect_ids"]),
            observation_id=scope["observation_id"],
            fingerprint=scope["fingerprint"],
            decision_node_id=scope["decision_node_id"],
            action_digest=scope["action_digest"],
        )

    @staticmethod
    def _normalize_confirmation(value: Mapping[str, Any]) -> dict[str, Any]:
        required = {
            "session_id",
            "task_id",
            "device_id",
            "revision",
            "subgoal_id",
            "effect_ids",
            "observation_id",
            "fingerprint",
            "decision_node_id",
            "action_digest",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise UniversalAgentOrchestratorError(
                "确认作用域字段缺失或包含额外字段。"
            )
        effect_ids = value.get("effect_ids")
        if not isinstance(effect_ids, list):
            raise UniversalAgentOrchestratorError("确认作用域 effect_ids 必须是数组。")
        revision = value.get("revision")
        if isinstance(revision, bool) or not isinstance(revision, int):
            raise UniversalAgentOrchestratorError("确认作用域 revision 格式无效。")
        decision_node_id = str(value.get("decision_node_id") or "").strip()
        action_digest = str(value.get("action_digest") or "").strip()
        if not decision_node_id:
            raise UniversalAgentOrchestratorError(
                "确认作用域 decision_node_id 不能为空。"
            )
        if not re.fullmatch(r"[0-9a-f]{64}", action_digest):
            raise UniversalAgentOrchestratorError(
                "确认作用域 action_digest 必须是 64 位小写 SHA-256。"
            )
        return {
            "session_id": str(value.get("session_id") or ""),
            "task_id": str(value.get("task_id") or ""),
            "device_id": str(value.get("device_id") or ""),
            "revision": revision,
            "subgoal_id": str(value.get("subgoal_id") or ""),
            "effect_ids": sorted(str(item) for item in effect_ids),
            "observation_id": str(value.get("observation_id") or ""),
            "fingerprint": str(value.get("fingerprint") or ""),
            "decision_node_id": decision_node_id,
            "action_digest": action_digest,
        }

    def _bind_effect_confirmation(self, session: UniversalAgentSessionState) -> None:
        graph = session.task_graph
        if graph is None:
            raise UniversalAgentOrchestratorError("效果确认缺少任务图。")
        current = graph.active_subgoal()
        effect_ids = _confirmation_effect_ids(graph, current)
        if current is None or not effect_ids:
            raise UniversalAgentOrchestratorError("当前子目标没有可确认风险。")
        intent_digest, intent_preview = _effect_confirmation_material(graph, current)
        session.effect_confirmation_authority = EffectConfirmationAuthority(
            session_id=session.session_id,
            task_id=graph.task_id,
            device_id=graph.device_id,
            revision=graph.revision,
            subgoal_id=current.subgoal_id,
            effect_ids=effect_ids,
            intent_digest=intent_digest,
            intent_preview=intent_preview,
        )

    @staticmethod
    def _normalize_effect_confirmation(value: Mapping[str, Any]) -> dict[str, Any]:
        required = {
            "session_id",
            "task_id",
            "device_id",
            "revision",
            "subgoal_id",
            "effect_ids",
            "intent_digest",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise UniversalAgentOrchestratorError(
                "效果确认作用域字段缺失或包含额外字段。"
            )
        effect_ids = value.get("effect_ids")
        revision = value.get("revision")
        if not isinstance(effect_ids, list):
            raise UniversalAgentOrchestratorError("效果确认 effect_ids 必须是数组。")
        if isinstance(revision, bool) or not isinstance(revision, int):
            raise UniversalAgentOrchestratorError("效果确认 revision 格式无效。")
        intent_digest = str(value.get("intent_digest") or "").strip()
        if not re.fullmatch(r"[0-9a-f]{64}", intent_digest):
            raise UniversalAgentOrchestratorError(
                "效果确认 intent_digest 必须是 64 位小写 SHA-256。"
            )
        return {
            "session_id": str(value.get("session_id") or ""),
            "task_id": str(value.get("task_id") or ""),
            "device_id": str(value.get("device_id") or ""),
            "revision": revision,
            "subgoal_id": str(value.get("subgoal_id") or ""),
            "effect_ids": sorted(str(item) for item in effect_ids),
            "intent_digest": intent_digest,
        }

    def _validate_and_consume_confirmation(
        self,
        session: UniversalAgentSessionState,
        confirmation: Mapping[str, Any],
    ) -> None:
        authority = session.confirmation_authority
        if session.status != "awaiting_confirmation":
            raise UniversalAgentOrchestratorError(
                f"会话已推进，当前状态不能确认：{session.status}。"
            )
        if authority is None:
            raise UniversalAgentOrchestratorError("当前会话没有可用确认作用域。")
        if authority.consumed:
            raise UniversalAgentOrchestratorError("当前确认已使用，禁止重放。")
        current_scope = self._current_confirmation_scope(session)
        if current_scope != authority.scope():
            authority.consumed = True
            authority.invalid_reason = "authoritative_state_changed"
            raise UniversalAgentOrchestratorError(
                "任务、画面或视觉决策已经变化，当前确认已失效。"
            )
        try:
            requested = self._normalize_confirmation(confirmation)
        except UniversalAgentOrchestratorError:
            authority.consumed = True
            authority.invalid_reason = "invalid_confirmation_shape"
            raise
        if requested != current_scope:
            authority.consumed = True
            authority.invalid_reason = "confirmation_scope_mismatch"
            raise UniversalAgentOrchestratorError(
                "确认作用域与当前 task/device/revision/subgoal/risk/observation 不一致。"
            )

    @staticmethod
    def _verified_target_app_home_reset_microstep(
        *,
        graph: DynamicTaskGraph,
        previous_decision: Any,
        result: Any,
        before_observation: Any,
        new_observation: Any,
    ) -> bool:
        """Keep the active App goal after a verified reset to Launcher.

        Home is an intermediate, coordinate-free reset when the active typed
        surface is an App but the stable foreground is a different App.  The
        reset cannot complete the App-opening subgoal, so it must retain the
        same task revision and obtain a fresh Launcher decision instead of
        asking DeepSeek to reinterpret the high-level graph.
        """

        current = graph.active_subgoal()
        action = getattr(
            getattr(previous_decision, "proposal", None),
            "action",
            None,
        )
        resolved = getattr(result, "resolved_action", None)
        before_scene = getattr(result, "before_scene", None)
        after_scene = getattr(result, "after_scene", None)
        if (
            current is None
            or current.external_impact != "navigation_only"
            or action is None
            or str(getattr(action, "action", "")) != "home"
            or resolved is None
            or str(getattr(resolved, "kind", "")) != "home"
            or before_scene is None
            or after_scene is None
            or str(getattr(result, "action_outcome", "")) != "matched"
            or int(getattr(result, "physical_actions", 0)) != 1
            or tuple(getattr(result, "verification_errors", ()))
            or str(getattr(before_scene, "foreground_app_id", "")).casefold()
            == "launcher"
            or str(getattr(after_scene, "foreground_app_id", "")).casefold()
            != "launcher"
            or str(getattr(new_observation, "fingerprint", ""))
            != str(getattr(after_scene, "fingerprint", ""))
            or str(getattr(before_observation, "fingerprint", ""))
            != str(getattr(before_scene, "fingerprint", ""))
        ):
            return False
        try:
            semantic_ir = compile_formal_semantic_authority(graph).semantic_ir
        except TaskSemanticIRError:
            return False
        active = next(
            (
                item
                for item in semantic_ir.subgoals
                if item.subgoal_id == current.subgoal_id
            ),
            None,
        )
        surfaces = {item.surface_id: item for item in semantic_ir.surfaces}
        target_surface = (
            surfaces.get(active.surface_ref) if active is not None else None
        )
        if target_surface is None or target_surface.kind != "app":
            return False
        params = getattr(action, "params", None)
        transition = (
            params.get("formal_transition")
            if isinstance(params, Mapping)
            else None
        )
        expectations = (
            transition.get("expectations")
            if isinstance(transition, Mapping)
            else None
        )
        return bool(
            transition.get("exploratory") is False
            and isinstance(expectations, list)
            and len(expectations) == 1
            and expectations[0]
            == {
                "subject_ref": "surface_current",
                "predicate": "surface.kind",
                "operator": "equals",
                "value": "launcher",
            }
        )

    @staticmethod
    def _verified_input_transaction_microstep(
        *,
        graph: DynamicTaskGraph,
        previous_decision: Any,
        result: Any,
        before_observation: Any,
        new_observation: Any,
        allow_terminal: bool = False,
    ) -> bool:
        """Recognize one controller-verified step inside canonical text input.

        DeepSeek owns the high-level task graph, but it must not reinterpret an
        unfinished deterministic text transaction after every locally audited
        fragment or keyboard-mode switch.  This predicate grants no action
        authority: it runs only after the one-shot action was consumed and the
        adapter verified a fresh after-scene.  The next action still requires a
        new Qwen decision, policy decision and confirmation scope.
        """

        current = graph.active_subgoal()
        canonical = graph.goal.entities.get("input_text")
        if not isinstance(canonical, str) or not canonical:
            canonical = ObservationBridge._active_input_transaction_text(
                graph,
                current,
            )
        resolved = getattr(result, "resolved_action", None)
        before_scene = getattr(result, "before_scene", None)
        after_scene = getattr(result, "after_scene", None)
        if (
            current is None
            or current.external_impact != "navigation_only"
            or not isinstance(canonical, str)
            or not canonical
            or resolved is None
            or before_scene is None
            or after_scene is None
            or str(getattr(result, "action_outcome", "")) != "matched"
            or int(getattr(result, "physical_actions", 0)) != 1
            or tuple(getattr(result, "verification_errors", ()))
            or str(getattr(resolved, "before_fingerprint", ""))
            != str(getattr(before_scene, "fingerprint", ""))
            or str(getattr(new_observation, "fingerprint", ""))
            != str(getattr(after_scene, "fingerprint", ""))
        ):
            return False

        proposal_action = getattr(
            getattr(previous_decision, "proposal", None),
            "action",
            None,
        )
        if proposal_action is None:
            return False

        resolved_kind = str(getattr(resolved, "kind", ""))
        if resolved_kind == "tap_semantic":
            target_id = str(
                getattr(resolved, "target_element_id", "") or ""
            ).strip()
            try:
                focus_target = before_scene.get_element(
                    target_id,
                    min_confidence=MIN_TARGET_CONFIDENCE,
                )
            except UISceneError:
                focus_target = None
            if focus_target is not None and focus_target.role == "input":
                expected_effect = getattr(resolved, "expected_effect", None)
                expected_element = (
                    expected_effect.get("element_state")
                    if isinstance(expected_effect, Mapping)
                    else None
                )
                expected_states = (
                    expected_element.get("states")
                    if isinstance(expected_element, Mapping)
                    else None
                )
                raw_value = focus_target.states.get("value")
                placeholder = focus_target.states.get("placeholder")
                prior_value = (
                    ""
                    if isinstance(raw_value, str)
                    and isinstance(placeholder, str)
                    and raw_value == placeholder
                    and focus_target.states.get("focused") is not True
                    else raw_value
                )
                field_id = str(
                    focus_target.states.get("input_field_id") or ""
                ).strip()
                after_inputs = tuple(
                    element
                    for element in after_scene.elements
                    if element.role == "input"
                    and float(element.confidence) >= MIN_TARGET_CONFIDENCE
                    and element.states.get("visible") is not False
                    and element.states.get("focused") is True
                    and element.meaning == focus_target.meaning
                    and element.states.get("value") == prior_value
                    and (
                        not field_id
                        or str(element.states.get("input_field_id") or "").strip()
                        == field_id
                    )
                )
                return bool(
                    str(getattr(proposal_action, "action", "")) == "tap_semantic"
                    and isinstance(prior_value, str)
                    and canonical.startswith(prior_value)
                    and prior_value != canonical
                    and isinstance(expected_element, Mapping)
                    and str(expected_element.get("meaning") or "").strip()
                    == focus_target.meaning
                    and expected_states == {"focused": True}
                    and len(after_inputs) == 1
                )

        auxiliary_meanings = {
            "ime_exact_candidate",
            "input_exact_literal_key",
            "input_exact_enter_key",
            "switch_keyboard_layout",
            "switch_keyboard_case",
            "switch_keyboard_input_mode",
        }
        before_input_id = ""
        auxiliary = None
        if resolved_kind == "input_verified_text":
            before_input_id = str(
                getattr(resolved, "target_element_id", "") or ""
            ).strip()
            if (
                str(getattr(proposal_action, "action", ""))
                != "input_verified_text"
                or str(getattr(resolved, "text", "")) != canonical
            ):
                return False
        elif resolved_kind in {"tap_semantic", "press_enter"}:
            target_id = str(
                getattr(resolved, "target_element_id", "") or ""
            ).strip()
            try:
                auxiliary = before_scene.get_element(
                    target_id,
                    min_confidence=MIN_TARGET_CONFIDENCE,
                )
            except UISceneError:
                return False
            if (
                str(getattr(proposal_action, "action", "")) != resolved_kind
                or auxiliary.meaning not in auxiliary_meanings
                or auxiliary.states.get("fully_visible") is not True
            ):
                return False
            before_input_id = str(
                auxiliary.states.get("input_element_id") or ""
            ).strip()
        else:
            return False

        expected_effect = getattr(resolved, "expected_effect", None)
        expected_element = (
            expected_effect.get("element_state")
            if isinstance(expected_effect, Mapping)
            else None
        )
        expected_states = (
            expected_element.get("states")
            if isinstance(expected_element, Mapping)
            else None
        )
        expected_meaning = (
            str(expected_element.get("meaning") or "").strip()
            if isinstance(expected_element, Mapping)
            else ""
        )
        if (
            not before_input_id
            or not expected_meaning
            or not isinstance(expected_states, dict)
            or not isinstance(expected_states.get("value"), str)
        ):
            return False
        try:
            before_input = before_scene.get_element(
                before_input_id,
                min_confidence=MIN_TARGET_CONFIDENCE,
            )
        except UISceneError:
            return False
        prior_value = before_input.states.get("value")
        if (
            before_input.role != "input"
            or before_input.states.get("focused") is not True
            or not isinstance(prior_value, str)
            or not canonical.startswith(prior_value)
        ):
            return False
        try:
            input_step = plan_next_verified_input(canonical, prior_value)
        except (ValueError, VerifiedTextTransactionError):
            return False
        if input_step is None:
            return False

        expected_value = expected_states["value"]
        if not canonical.startswith(expected_value):
            return False
        if auxiliary is None:
            if (
                getattr(resolved, "prior_input_value", None) != prior_value
                or getattr(resolved, "input_fragment", None)
                != input_step.segment
                or getattr(resolved, "input_method", None) != input_step.kind
                or getattr(resolved, "expected_input_value", None)
                != input_step.expected_value
            ):
                return False
            direct_preedit_inputs = tuple(
                element
                for element in after_scene.elements
                if input_step.kind == "direct_latin"
                and element.role == "input"
                and float(element.confidence) >= MIN_TARGET_CONFIDENCE
                and element.states.get("visible") is not False
                and element.states.get("focused") is True
                and element.meaning == expected_meaning
                and element.states.get("value") == input_step.current_text
                and element.states.get("ime_preedit_text") == input_step.segment
                and element.states.get("ime_exact_candidate_text")
                == input_step.segment
            )
            direct_preedit_candidates = tuple(
                element
                for element in after_scene.elements
                if len(direct_preedit_inputs) == 1
                and element.meaning == "ime_exact_candidate"
                and element.label == input_step.segment
                and float(element.confidence) >= MIN_TARGET_CONFIDENCE
                and element.states.get("goal_relevant") is True
                and element.states.get("fully_visible") is True
                and element.states.get("ime_candidate") is True
                and element.states.get("input_element_id")
                == direct_preedit_inputs[0].element_id
                and element.states.get("prior_input_value")
                == input_step.current_text
                and element.states.get("expected_input_value")
                == input_step.expected_value
                and element.states.get("pinyin") == input_step.segment
            )
            direct_preedit_observed = bool(
                len(direct_preedit_inputs) == 1
                and len(direct_preedit_candidates) == 1
            )
            exact_expected_states = (
                {
                    "value": input_step.current_text,
                    "ime_preedit_text": input_step.pinyin,
                    "ime_exact_candidate_text": input_step.segment,
                }
                if input_step.kind == "chinese_pinyin"
                else {"value": input_step.expected_value}
            )
            if expected_states != exact_expected_states:
                return False
            after_match_states = (
                {
                    "value": input_step.current_text,
                    "ime_preedit_text": input_step.segment,
                    "ime_exact_candidate_text": input_step.segment,
                }
                if direct_preedit_observed
                else exact_expected_states
            )
        else:
            direct_preedit_observed = False
            after_match_states = expected_states
            states = auxiliary.states
            if states.get("prior_input_value") != prior_value:
                return False
            if auxiliary.meaning in {
                "input_exact_literal_key",
                "input_exact_enter_key",
            }:
                if (
                    input_step.kind != "literal_key"
                    or states.get("key_value") != input_step.segment
                    or states.get("expected_input_value")
                    != input_step.expected_value
                    or expected_states != {"value": input_step.expected_value}
                ):
                    return False
            elif auxiliary.meaning == "ime_exact_candidate":
                if (
                    input_step.kind not in {"chinese_pinyin", "direct_latin"}
                    or auxiliary.label != input_step.segment
                    or states.get("pinyin")
                    != (
                        input_step.pinyin
                        if input_step.kind == "chinese_pinyin"
                        else input_step.segment
                    )
                    or states.get("expected_input_value")
                    != input_step.expected_value
                    or expected_states != {"value": input_step.expected_value}
                ):
                    return False
            elif auxiliary.meaning == "switch_keyboard_layout":
                desired_layout = preferred_keyboard_layout(
                    input_step.segment[0]
                )
                current_layout = states.get("current_layout")
                target_layout = states.get("target_layout")
                if (
                    not keyboard_layout_switch_advances(
                        current_layout=current_layout,
                        target_layout=target_layout,
                        desired_layout=desired_layout,
                    )
                    or expected_states
                    != {
                        "value": prior_value,
                        "keyboard_layout": target_layout,
                    }
                ):
                    return False
            elif auxiliary.meaning == "switch_keyboard_case":
                if (
                    not input_step.required_case_mode
                    or expected_states
                    != {
                        "value": prior_value,
                        "keyboard_case_mode": input_step.required_case_mode,
                    }
                ):
                    return False
            elif auxiliary.meaning == "switch_keyboard_input_mode":
                required_input_mode = required_keyboard_input_mode_for_step(
                    input_step
                )
                if (
                    required_input_mode is None
                    or states.get("target_mode") != required_input_mode
                    or expected_states
                    != {
                        "value": prior_value,
                        "keyboard_input_mode": required_input_mode,
                    }
                ):
                    return False

        after_inputs = tuple(
            element
            for element in after_scene.elements
            if element.role == "input"
            and float(element.confidence) >= MIN_TARGET_CONFIDENCE
            and element.states.get("visible") is not False
            and element.states.get("focused") is True
            and element.meaning == expected_meaning
            and all(
                element.states.get(key) == value
                for key, value in after_match_states.items()
            )
        )
        # Only an unfinished verified prefix remains inside the deterministic
        # input transaction.  Once the fresh after-scene proves the canonical
        # value itself, this is a terminal input step: return to the normal
        # typed receipt / DeepSeek completion path instead of asking Qwen to
        # enumerate the same text again (for example, input value and a stale
        # IME candidate carrying an identical literal).
        if len(after_inputs) != 1:
            return False
        if direct_preedit_observed:
            return True
        if expected_value == canonical:
            return bool(allow_terminal)
        if UniversalAgentOrchestrator._input_step_reaches_formal_successor(
            graph=graph,
            current_subgoal_id=current.subgoal_id,
            canonical=canonical,
            expected_value=expected_value,
        ):
            return False
        return True

    @staticmethod
    def _input_transaction_reached_canonical(
        graph: DynamicTaskGraph,
        result: Any,
    ) -> bool:
        canonical = graph.goal.entities.get("input_text")
        resolved = getattr(result, "resolved_action", None)
        expected_effect = getattr(resolved, "expected_effect", None)
        expected_element = (
            expected_effect.get("element_state")
            if isinstance(expected_effect, Mapping)
            else None
        )
        expected_states = (
            expected_element.get("states")
            if isinstance(expected_element, Mapping)
            else None
        )
        after_scene = getattr(result, "after_scene", None)
        target_id = str(
            getattr(resolved, "input_element_id", "")
            or getattr(resolved, "target_element_id", "")
            or ""
        ).strip()
        before_scene = getattr(result, "before_scene", None)
        if target_id and before_scene is not None:
            try:
                before_target = before_scene.get_element(
                    target_id,
                    min_confidence=MIN_TARGET_CONFIDENCE,
                )
            except UISceneError:
                before_target = None
            if before_target is not None and before_target.role != "input":
                target_id = str(
                    before_target.states.get("input_element_id") or ""
                ).strip()
        if not (
            isinstance(canonical, str)
            and canonical
            and isinstance(expected_states, Mapping)
            and expected_states.get("value") == canonical
            and after_scene is not None
            and target_id
        ):
            return False
        try:
            after_input = after_scene.get_element(
                target_id,
                min_confidence=MIN_TARGET_CONFIDENCE,
            )
        except UISceneError:
            return False
        return bool(
            after_input.role == "input"
            and after_input.states.get("value") == canonical
            and after_input.states.get("ime_preedit_text") in {None, ""}
        )

    @staticmethod
    def _complete_local_exact_input_graph(
        graph: DynamicTaskGraph,
        *,
        new_observation: Any,
    ) -> DynamicTaskGraph:
        current = graph.active_subgoal()
        if (
            graph.status not in {"ready", "running", "awaiting_confirmation"}
            or current is None
            or len(graph.subgoals) != 1
            or graph.subgoals[0].subgoal_id != current.subgoal_id
            or current.external_impact != "navigation_only"
            or graph.risk_actions
            or not isinstance(graph.goal.entities.get("input_text"), str)
        ):
            raise UniversalAgentOrchestratorError(
                "本地 exact_input_text 完成只允许单一、无效果的输入子目标。"
            )
        observation_id = str(
            getattr(new_observation, "observation_id", "") or ""
        ).strip()
        fingerprint = str(
            getattr(new_observation, "fingerprint", "") or ""
        ).strip()
        if not observation_id or not fingerprint:
            raise UniversalAgentOrchestratorError(
                "本地 exact_input_text 完成缺少新 observation/fingerprint。"
            )
        evidence = (
            f"动作后观察 {observation_id} 已验证输入框精确值，fingerprint={fingerprint}",
        )
        completed = replace(
            graph,
            revision=graph.revision + 1,
            status="completed",
            completion_conditions=tuple(
                replace(condition, satisfied=True, evidence=evidence)
                for condition in graph.completion_conditions
            ),
            subgoals=(
                replace(
                    current,
                    status="completed",
                    completion_evidence=evidence,
                ),
            ),
            active_subgoal_id=None,
        )
        completed.validate()
        return completed

    @staticmethod
    def _input_step_reaches_formal_successor(
        *,
        graph: DynamicTaskGraph,
        current_subgoal_id: str,
        canonical: str,
        expected_value: str,
    ) -> bool:
        """Detect one typed input boundary already split by the formal graph.

        A deterministic text transaction normally remains inside one high-level
        input subgoal.  When the graph instead has one direct successor whose
        required action exactly matches the next deterministic input step, the
        matched current step must use the normal receipt/replan path so that
        the successor becomes active.  This is structural: no App, screenshot,
        coordinate or free-form completion phrase decides the transition.
        """

        try:
            semantic_ir = compile_formal_semantic_authority(graph).semantic_ir
            next_step = plan_next_verified_input(canonical, expected_value)
        except (TaskSemanticIRError, ValueError, VerifiedTextTransactionError):
            return False
        if next_step is None:
            return False
        next_required_action = (
            "press_enter"
            if next_step.kind == "literal_key" and next_step.segment == "\n"
            else "input_verified_text"
        )
        constraints = {
            item.constraint_id: item for item in semantic_ir.constraints
        }

        def required_actions(subgoal: Any) -> frozenset[str]:
            return frozenset(
                str(constraints[ref].value)
                for ref in subgoal.constraint_refs
                if ref in constraints
                and constraints[ref].kind == "required_action"
            )

        current = next(
            (
                item
                for item in semantic_ir.subgoals
                if item.subgoal_id == current_subgoal_id
            ),
            None,
        )
        successors = tuple(
            item
            for item in semantic_ir.subgoals
            if item.status == "pending"
            and current_subgoal_id in item.depends_on
        )
        if current is None or len(successors) != 1:
            return False
        current_actions = required_actions(current)
        successor_actions = required_actions(successors[0])
        return bool(
            "input_verified_text" in current_actions
            and next_required_action not in current_actions
            and next_required_action in successor_actions
        )

    @staticmethod
    def _build_effect_verification(
        session: UniversalAgentSessionState,
        *,
        graph: DynamicTaskGraph,
        subgoal: Any,
        receipt: VerifiedActionTransition,
        consumed_revision: int,
    ) -> dict[str, Any]:
        """Bind one matched external effect to a read-only result check.

        The receipt proves only that the exact effect action ran once.  It is
        deliberately not exposed as visual completion evidence; a later fresh
        scene still has to provide the result claim.
        """

        receipt.validate()
        semantic_ir = getattr(session.semantic_task_context, "semantic_ir", None)
        effects = tuple(
            effect
            for effect in tuple(getattr(semantic_ir, "effects", ()) or ())
            if subgoal.subgoal_id in tuple(effect.source_subgoal_ids)
        )
        if len(effects) != 1:
            raise UniversalAgentOrchestratorError(
                "外部效果结果复核要求当前子目标唯一绑定一个 EffectIntent。"
            )
        effect = effects[0]
        previews = tuple(
            item
            for item in session.effect_previews
            if str(item.get("effect_id") or "") == effect.effect_id
        )
        if len(previews) != 1:
            raise UniversalAgentOrchestratorError(
                "外部效果结果复核缺少唯一 EffectPreview。"
            )
        preview = previews[0]
        preview_digest = str(preview.get("preview_digest") or "")
        if (
            subgoal.external_impact != "external_state"
            or receipt.outcome != "matched"
            or receipt.physical_actions != 1
            or receipt.session_id != session.session_id
            or receipt.task_id != graph.task_id
            or receipt.device_id != graph.device_id
            or receipt.prior_revision != graph.revision
            or receipt.subgoal_id != subgoal.subgoal_id
            or str(preview.get("task_id") or "") != graph.task_id
            or str(preview.get("device_id") or "") != graph.device_id
            or int(preview.get("revision") or 0) != graph.revision
            or str(preview.get("effect_kind") or "") != effect.kind
            or not re.fullmatch(r"[0-9a-f]{64}", preview_digest)
        ):
            raise UniversalAgentOrchestratorError(
                "外部效果结果复核的 receipt/EffectIntent/preview 绑定不一致。"
            )
        return {
            "protocol_version": "2026-08-19-effect-result-verification-v1",
            "status": "pending",
            "session_id": session.session_id,
            "task_id": graph.task_id,
            "device_id": graph.device_id,
            "effect_id": effect.effect_id,
            "effect_kind": effect.kind,
            "effect_preview_digest": preview_digest,
            "subgoal_id": subgoal.subgoal_id,
            "receipt_id": receipt.receipt_id,
            "receipt_prior_revision": receipt.prior_revision,
            "receipt_after_observation_id": receipt.after_observation_id,
            "receipt_after_fingerprint": receipt.after_fingerprint,
            "consumed_revision": consumed_revision,
            "verification_attempts": 0,
        }

    @staticmethod
    def _validate_pending_effect_verification(
        session: UniversalAgentSessionState,
        graph: DynamicTaskGraph,
    ) -> dict[str, Any]:
        pending = session.effect_verification
        if not isinstance(pending, dict) or pending.get("status") != "pending":
            raise UniversalAgentOrchestratorError("当前没有待处理的外部效果只读复核。")
        previews = tuple(
            item
            for item in session.effect_previews
            if str(item.get("effect_id") or "") == pending.get("effect_id")
        )
        subgoal = next(
            (
                item
                for item in graph.subgoals
                if item.subgoal_id == pending.get("subgoal_id")
            ),
            None,
        )
        transition = session.last_post_action_transition or {}
        receipt = transition.get("receipt") or {}
        if (
            pending.get("protocol_version")
            != "2026-08-19-effect-result-verification-v1"
            or pending.get("session_id") != session.session_id
            or pending.get("task_id") != graph.task_id
            or pending.get("device_id") != graph.device_id
            or pending.get("consumed_revision") != graph.revision
            or pending.get("verification_attempts") != 0
            or subgoal is None
            or subgoal.external_impact != "external_state"
            or len(previews) != 1
            or previews[0].get("preview_digest")
            != pending.get("effect_preview_digest")
            or receipt.get("receipt_id") != pending.get("receipt_id")
            or receipt.get("outcome") != "matched"
            or receipt.get("physical_actions") != 1
            or receipt.get("subgoal_id") != pending.get("subgoal_id")
            or receipt.get("after_observation_id")
            != pending.get("receipt_after_observation_id")
            or receipt.get("after_fingerprint")
            != pending.get("receipt_after_fingerprint")
        ):
            raise UniversalAgentOrchestratorError(
                "待复核外部效果与当前 session/graph/receipt/preview 不一致。"
            )
        return dict(pending)

    def _advance_after_observation(
        self,
        session: UniversalAgentSessionState,
        *,
        result: Any,
        before_observation: Any,
        new_observation: Any,
        prior_verified_app_surface_lineage: VerifiedAppSurfaceLineage | None = None,
        prior_physical_actions: int | None = None,
    ) -> None:
        previous_graph = session.task_graph
        assert previous_graph is not None
        action_outcome = str(getattr(result, "action_outcome", "matched"))
        if action_outcome not in POST_ACTION_OUTCOMES:
            raise UniversalAgentOrchestratorError(
                f"动作后验证返回了不支持的 outcome：{action_outcome}。"
            )
        matched = action_outcome == "matched"
        verification_errors = tuple(
            str(item)
            for item in getattr(result, "verification_errors", ())
            if str(item).strip()
        )
        if matched and verification_errors:
            raise UniversalAgentOrchestratorError(
                "动作后验证同时返回 matched 与 verification_errors。"
            )
        if not matched and not verification_errors:
            raise UniversalAgentOrchestratorError(
                "动作后验证返回 mismatched 但没有结构化错误。"
            )
        previous_current = previous_graph.active_subgoal()
        previous_decision = session.qwen_decision
        authority = session.confirmation_authority
        if (
            previous_current is None
            or previous_decision is None
            or previous_decision.proposal.action is None
            or authority is None
        ):
            raise UniversalAgentOrchestratorError(
                "动作后重规划缺少上一子目标、决策或确认权威。"
            )
        input_transaction_microstep = self._verified_input_transaction_microstep(
            graph=previous_graph,
            previous_decision=previous_decision,
            result=result,
            before_observation=before_observation,
            new_observation=new_observation,
            allow_terminal=session.local_exact_input_authority,
        )
        input_transaction_terminal = bool(
            input_transaction_microstep
            and session.local_exact_input_authority
            and self._input_transaction_reached_canonical(previous_graph, result)
        )
        target_app_home_reset_microstep = (
            self._verified_target_app_home_reset_microstep(
                graph=previous_graph,
                previous_decision=previous_decision,
                result=result,
                before_observation=before_observation,
                new_observation=new_observation,
            )
        )
        wait_transition = (
            result.resolved_action.kind == "wait_for_change"
            and int(result.physical_actions) == 0
        )
        receipt: VerifiedActionTransition | None = None
        controller_refs: tuple[ControllerTransitionEvidenceRef, ...] = ()
        if not wait_transition:
            receipt = VerifiedActionTransition(
                receipt_id=f"receipt_{uuid.uuid4().hex}",
                session_id=session.session_id,
                task_id=previous_graph.task_id,
                device_id=previous_graph.device_id,
                prior_revision=previous_graph.revision,
                subgoal_id=previous_current.subgoal_id,
                decision_node_id=authority.decision_node_id,
                action_digest=authority.action_digest,
                rebound_action_digest=_action_digest(result.rebound_action),
                resolved_action_digest=_action_digest(result.resolved_action),
                action_kind=str(previous_decision.proposal.action.action),
                before_observation_id=str(before_observation.observation_id),
                before_fingerprint=str(before_observation.fingerprint),
                after_observation_id=str(new_observation.observation_id),
                after_fingerprint=str(new_observation.fingerprint),
                physical_actions=int(result.physical_actions),
                outcome=action_outcome,
                errors=verification_errors,
                controller_transition_evidence=tuple(
                    str(item)
                    for item in getattr(
                        result, "controller_transition_evidence", ()
                    )
                    if str(item).strip()
                ),
            )
            receipt.validate()
            if (
                previous_current.external_impact == "navigation_only"
                and not input_transaction_microstep
                and not target_app_home_reset_microstep
            ):
                controller_refs = tuple(
                    ControllerTransitionEvidenceRef(
                        ref_id=(
                            f"controller_transition:{receipt.receipt_id}:{index}"
                        ),
                        receipt_id=receipt.receipt_id,
                        subgoal_id=receipt.subgoal_id,
                        text=text,
                    )
                    for index, text in enumerate(
                        receipt.controller_transition_evidence,
                        start=1,
                    )
                )
        verification = {
            "matched": matched,
            "action_outcome": action_outcome,
            "physical_actions": result.physical_actions,
            "before_fingerprint": before_observation.fingerprint,
            "execution_before_fingerprint": result.before_scene.fingerprint,
            "after_fingerprint": result.after_scene.fingerprint,
            "visible_evidence": [result.after_scene.summary],
            "blocked_reasons": list(verification_errors),
            "after_frame_paths": list(result.after_frame_paths),
            "controller_transition_evidence": list(
                receipt.controller_transition_evidence if receipt is not None else ()
            ),
            "verified_action_transition": (
                receipt.to_dict() if receipt is not None else None
            ),
            "transition_kind": (
                "wait_observation" if wait_transition else "physical_action"
            ),
        }
        self._remember(
            session,
            session.evidence_store.write_verification(
                max(1, session.step_number - 1),
                verification,
            ),
        )
        observed = self.bridge.observed_state(
            graph=previous_graph,
            trusted_observation=new_observation,
            action_outcome=("not_applicable" if wait_transition else action_outcome),
            verification=verification,
            verified_action_transition=receipt,
            controller_transition_evidence_refs=controller_refs,
        )
        previous_signature = _subgoal_progress_signature(previous_current)
        transition_record = {
            "protocol_version": POST_ACTION_TRANSITION_PROTOCOL_VERSION,
            "receipt": receipt.to_dict() if receipt is not None else None,
            "transition_kind": (
                "wait_observation" if wait_transition else "physical_action"
            ),
            "prior_subgoal_signature": previous_signature,
            "prior_action_equivalence_digest": _action_equivalence_digest(
                previous_decision.proposal.action
            ),
            "disposition": "replanning",
        }
        if input_transaction_terminal:
            transition_record["input_transaction_completed"] = True
        elif input_transaction_microstep:
            transition_record["input_transaction_progress"] = True
        if target_app_home_reset_microstep:
            transition_record["target_app_home_reset_progress"] = True

        def persist_transition() -> None:
            session.last_post_action_transition = dict(transition_record)
            self._remember(
                session,
                session.evidence_store.write_post_action_transition(
                    max(1, session.step_number - 1),
                    transition_record,
                ),
            )

        persist_transition()
        # The consumed decision and every authority derived from the old frame
        # become unusable before any model replan attempt.
        session.qwen_decision = None
        session.controller_decision = None
        session.confirmation_authority = None
        session.effect_confirmation_authority = None
        session.confirmed_effect_ids = ()
        try:
            if target_app_home_reset_microstep:
                revised = previous_graph
            elif input_transaction_microstep:
                revised = (
                    self._complete_local_exact_input_graph(
                        previous_graph,
                        new_observation=new_observation,
                    )
                    if input_transaction_terminal
                    else previous_graph
                )
            else:
                revised = self.deepseek_planner.replan(
                    previous_graph,
                    observed,
                    trigger=(
                        "observation_changed"
                        if wait_transition
                        else "action_result_matched"
                        if matched
                        else "action_result_mismatch"
                    ),
                    reason=(
                        "wait_for_change 未产生物理动作；仅依据新的可信画面重规划。"
                        if wait_transition
                        else "一个动作已经执行并由新的可信画面验证。"
                        if matched
                        else "动作已执行，但新画面没有证明预期语义变化，必须重规划。"
                    ),
                )
            next_lineage = None
            if not input_transaction_microstep and not target_app_home_reset_microstep:
                next_lineage = self._build_verified_app_surface_lineage(
                    session=session,
                    previous=previous_graph,
                    revised=revised,
                    trusted_observation=new_observation,
                    receipt=receipt,
                    controller_refs=controller_refs,
                    before_observation=before_observation,
                    previous_decision=previous_decision,
                    execution_result=result,
                )
            if next_lineage is None and prior_physical_actions is not None:
                next_lineage = self._carry_verified_app_surface_lineage(
                    session=session,
                    previous=previous_graph,
                    revised=revised,
                    trusted_observation=new_observation,
                    prior_lineage=prior_verified_app_surface_lineage,
                    prior_physical_actions=prior_physical_actions,
                    execution_result=result,
                )
            if not input_transaction_microstep and not target_app_home_reset_microstep:
                self._validate_graph_identity(
                    revised,
                    device_id=session.device_id,
                    previous=previous_graph,
                    trusted_observation=new_observation,
                    session_id=session.session_id,
                    verified_transition=receipt,
                    controller_transition_evidence_refs=controller_refs,
                    before_observation=before_observation,
                    previous_decision=previous_decision,
                    execution_result=result,
                    verified_app_surface_lineage=next_lineage,
                    physical_actions=session.physical_actions,
                )
            session.verified_app_surface_lineage = next_lineage
        except Exception as exc:
            session.status = "blocked"
            session.failed_reason = f"DeepSeek 重规划失败：{exc}"
            self._record_deepseek_failure(
                session,
                exc,
                stage="post_action_replan",
                previous_graph=previous_graph,
            )
            transition_record["disposition"] = "blocked_replan_failure"
            transition_record["diagnostic"] = session.failed_reason
            persist_transition()
            return

        session.task_graph = revised
        session.goal_draft = self.bridge.goal_draft(revised)
        revised_current = revised.active_subgoal()
        transition_record["revised_revision"] = revised.revision
        transition_record["revised_subgoal_id"] = (
            revised_current.subgoal_id if revised_current is not None else None
        )
        transition_record["revised_subgoal_signature"] = (
            _subgoal_progress_signature(revised_current)
        )
        if receipt is not None:
            transition_record["receipt_consumed_revision"] = revised.revision
        if (
            not input_transaction_microstep
            and not target_app_home_reset_microstep
        ) or input_transaction_terminal:
            self._remember(
                session,
                session.evidence_store.write_task_graph(revised),
                session.evidence_store.write_effect_policy_snapshot(revised),
            )
        if revised.status == "completed":
            session.status = "succeeded"
            transition_record["disposition"] = "task_completed"
            persist_transition()
            return
        if (
            matched
            and receipt is not None
            and previous_current.external_impact == "external_state"
        ):
            try:
                session.effect_verification = self._build_effect_verification(
                    session,
                    graph=previous_graph,
                    subgoal=previous_current,
                    receipt=receipt,
                    consumed_revision=revised.revision,
                )
            except Exception as exc:
                session.status = "blocked"
                session.failed_reason = f"外部效果只读复核绑定失败：{exc}"
                transition_record["disposition"] = (
                    "blocked_effect_verification_binding"
                )
                transition_record["diagnostic"] = session.failed_reason
                persist_transition()
                return
            session.status = "needs_effect_verification"
            session.failed_reason = ""
            session.qwen_decision = None
            session.controller_decision = None
            session.confirmation_authority = None
            session.effect_confirmation_authority = None
            session.confirmed_effect_ids = ()
            transition_record["disposition"] = (
                "pending_read_only_effect_result_verification"
            )
            transition_record["effect_verification"] = dict(
                session.effect_verification
            )
            persist_transition()
            self._complete_pending_effect_verification(
                session,
                graph=revised,
                observation=new_observation,
                before_actions=session.physical_actions,
            )
            transition_record["disposition"] = (
                "effect_verified_from_same_post_action_observation"
                if session.status == "succeeded"
                else "blocked_same_post_action_effect_verification"
            )
            transition_record["effect_verification"] = dict(
                session.effect_verification or {}
            )
            if session.task_graph is not None:
                verified_current = session.task_graph.active_subgoal()
                transition_record["effect_verification_revision"] = (
                    session.task_graph.revision
                )
                transition_record["effect_verification_subgoal_id"] = (
                    verified_current.subgoal_id
                    if verified_current is not None
                    else None
                )
            if session.failed_reason:
                transition_record["diagnostic"] = session.failed_reason
            persist_transition()
            return
        if controller_refs:
            prior_in_revised = next(
                (
                    item
                    for item in revised.subgoals
                    if item.subgoal_id == previous_current.subgoal_id
                ),
                None,
            )
            if prior_in_revised is None or prior_in_revised.status != "completed":
                session.status = "blocked"
                session.failed_reason = (
                    "本地一次性 controller_transition 完成证据已满足，"
                    "但重规划未完成其绑定的 navigation_only 子目标；"
                    "禁止继续产生动作或第二确认。"
                )
                transition_record["disposition"] = (
                    "blocked_unconsumed_controller_completion"
                )
                transition_record["diagnostic"] = session.failed_reason
                persist_transition()
                return
        current = revised.active_subgoal()
        impact = current.external_impact if current is not None else "unknown"
        if current is None:
            session.status = "blocked"
            session.failed_reason = "重规划后的任务图没有活动子目标。"
            transition_record["disposition"] = "blocked_missing_active_subgoal"
            persist_transition()
            return
        current_focus_changed = (
            _subgoal_progress_signature(current) != previous_signature
        )
        if _requires_effect_confirmation(revised, current):
            session.status = "awaiting_effect_confirmation"
            session.failed_reason = ""
            transition_record["disposition"] = "advanced_to_effect_confirmation"
            self._bind_effect_confirmation(session)
            persist_transition()
            return
        if current_focus_changed:
            # The post-action pixels remain authoritative for verifying the
            # action that just ran, but their structured scene was conditioned
            # on the previous active subgoal.  Every newly activated or
            # rewritten subgoal starts with its own observation/model step.
            session.status = "needs_reobservation"
            session.failed_reason = ""
            session.qwen_decision = None
            session.controller_decision = None
            session.confirmation_authority = None
            transition_record["disposition"] = (
                "advanced_to_current_subgoal_reobservation"
            )
            transition_record["reobservation_subgoal_id"] = current.subgoal_id
            persist_transition()
            return
        if (
            not matched
            and _allows_fresh_observation_corrective_retry(
                impact=impact,
                action_kind=previous_decision.proposal.action.action,
            )
        ):
            # A navigation mismatch is verified by this action-result scene,
            # but its corrective candidate must come from one independent
            # current-subgoal observation.  Do not mint a provisional choice
            # here only for the autonomous loop to discard it immediately.
            session.status = "needs_reobservation"
            session.failed_reason = ""
            session.qwen_decision = None
            session.controller_decision = None
            session.confirmation_authority = None
            transition_record["disposition"] = (
                "navigation_mismatch_needs_fresh_observation"
            )
            transition_record["reobservation_subgoal_id"] = current.subgoal_id
            persist_transition()
            return
        if impact == "read_only":
            try:
                reviewed = self.deepseek_planner.replan(
                    revised,
                    replace(
                        observed,
                        last_action_outcome="not_applicable",
                        verified_action_transition=None,
                        controller_transition_evidence_refs=(),
                    ),
                    trigger="subgoal_completed",
                    reason=(
                        "当前 read_only 子目标只能用已经采集的当前可信画面"
                        "完成或阻塞；不得请求任何新的物理动作。"
                    ),
                )
                self._validate_graph_identity(
                    reviewed,
                    device_id=session.device_id,
                    previous=revised,
                    trusted_observation=new_observation,
                )
                session.task_graph = reviewed
                session.goal_draft = self.bridge.goal_draft(reviewed)
                self._remember(
                    session,
                    session.evidence_store.write_task_graph(reviewed),
                    session.evidence_store.write_effect_policy_snapshot(reviewed),
                )
            except Exception as exc:
                session.status = "blocked"
                session.failed_reason = f"只读完成复核失败：{exc}"
                transition_record["disposition"] = "blocked_read_only_review"
                transition_record["diagnostic"] = session.failed_reason
                persist_transition()
                return
            if reviewed.status == "completed":
                session.status = "succeeded"
                session.failed_reason = ""
                transition_record["disposition"] = "task_completed_after_read_only_review"
                persist_transition()
                return
            reviewed_current = reviewed.active_subgoal()
            reviewed_impact = (
                reviewed_current.external_impact
                if reviewed_current is not None
                else "unknown"
            )
            if reviewed_current is None:
                session.status = "blocked"
                session.failed_reason = "只读复核后的任务图没有活动子目标。"
                transition_record["disposition"] = "blocked_read_only_no_active"
                transition_record["diagnostic"] = session.failed_reason
                persist_transition()
                return
            if _requires_effect_confirmation(reviewed, reviewed_current):
                session.status = "awaiting_effect_confirmation"
                session.failed_reason = ""
                self._bind_effect_confirmation(session)
                transition_record["disposition"] = (
                    "advanced_to_effect_confirmation_after_read_only"
                )
                persist_transition()
                return
            if (
                _subgoal_progress_signature(reviewed_current)
                != _subgoal_progress_signature(current)
            ):
                session.status = "needs_reobservation"
                session.failed_reason = ""
                session.qwen_decision = None
                session.controller_decision = None
                session.confirmation_authority = None
                transition_record["disposition"] = (
                    "read_only_advanced_to_current_subgoal_reobservation"
                )
                transition_record["reobservation_subgoal_id"] = (
                    reviewed_current.subgoal_id
                )
                persist_transition()
                return
            if reviewed_impact == "read_only":
                session.status = "blocked"
                session.failed_reason = (
                    "当前可信画面没有让 DeepSeek 完成 read_only 子目标；"
                    "禁止为只读验证请求物理动作。"
                )
                transition_record["disposition"] = "blocked_read_only_incomplete"
                transition_record["diagnostic"] = session.failed_reason
                persist_transition()
                return
            revised = reviewed
            transition_record["revised_revision"] = revised.revision
            transition_record["revised_subgoal_id"] = revised.active_subgoal_id
            transition_record["revised_subgoal_signature"] = (
                _subgoal_progress_signature(revised.active_subgoal())
            )

        frames = list(result.after_frames)
        context = revised.to_qwen_context()
        decision = self._decide_next_action(
            session,
            frames=frames,
            task_context=context,
            trusted_observation=new_observation,
        )
        self._validate_decision_binding(revised, new_observation, decision)
        session.qwen_decision = decision
        self._remember(
            session,
            session.evidence_store.write_qwen_decision(
                session.step_number,
                decision,
            ),
        )
        if decision.proposal.status != "action":
            session.status = "blocked"
            session.failed_reason = (
                decision.proposal.reason
                if decision.proposal.status == "blocked"
                else f"本地动作选择器返回了不支持的状态：{decision.proposal.status}"
            )
            session.controller_decision = CanonicalSelectionReceipt(
                allowed=False,
                reason=session.failed_reason,
            )
            session.confirmation_authority = None
            transition_record["disposition"] = "blocked_qwen_no_action"
            transition_record["diagnostic"] = session.failed_reason
            persist_transition()
            return
        new_equivalence_digest = _action_equivalence_digest(
            decision.proposal.action
        )
        transition_record["next_action_equivalence_digest"] = (
            new_equivalence_digest
        )
        if (
            bool(controller_refs)
            and matched
            and transition_record.get("revised_subgoal_signature")
            == previous_signature
            and new_equivalence_digest
            == transition_record["prior_action_equivalence_digest"]
        ):
            session.status = "blocked"
            session.failed_reason = (
                "一次性 controller_transition 完成证据已满足，"
                "但同一活动子目标仍提出等价动作；禁止生成第二确认。"
            )
            session.controller_decision = CanonicalSelectionReceipt(
                allowed=False,
                reason=session.failed_reason,
            )
            session.confirmation_authority = None
            transition_record["disposition"] = "blocked_equivalent_repeat"
            transition_record["diagnostic"] = session.failed_reason
            persist_transition()
            return
        selection_receipt = self._selection_receipt(session, decision)
        session.controller_decision = selection_receipt
        self._remember(
            session,
            session.evidence_store.write_controller_decision(
                session.step_number,
                self._selection_receipt_payload(selection_receipt),
            ),
        )
        if not selection_receipt.allowed:
            session.status = "blocked"
            session.failed_reason = selection_receipt.reason
            session.confirmation_authority = None
            transition_record["disposition"] = "blocked_canonical_selection"
            transition_record["diagnostic"] = session.failed_reason
            persist_transition()
            return
        session.status = "awaiting_confirmation"
        self._bind_confirmation(session)
        transition_record["disposition"] = "advanced_to_new_confirmation"
        transition_record["next_confirmation_scope"] = (
            session.confirmation_authority.scope()
        )
        persist_transition()

    def _complete_pending_effect_verification(
        self,
        session: UniversalAgentSessionState,
        *,
        graph: DynamicTaskGraph,
        observation: Any,
        before_actions: int,
    ) -> Any:
        pending = self._validate_pending_effect_verification(session, graph)
        observed = self.bridge.observed_state(
            graph=graph,
            trusted_observation=observation,
            action_outcome="not_applicable",
            verification={
                "visible_evidence": [observation.scene.summary],
                "blocked_reasons": [],
            },
        )
        try:
            revised = self.deepseek_planner.replan(
                graph,
                observed,
                trigger="observation_changed",
                reason=(
                    "外部效果动作已有严格一次性 matched receipt；本轮仅用新鲜"
                    " typed visual claim 复核结果，禁止规划或重复任何效果动作。"
                ),
            )
            self._validate_graph_identity(
                revised,
                device_id=session.device_id,
                previous=graph,
                trusted_observation=observation,
            )
        except Exception as exc:
            failed = {
                **pending,
                "status": "failed",
                "verification_attempts": 1,
                "verification_observation_id": observation.observation_id,
                "verification_fingerprint": observation.fingerprint,
                "reason": f"外部效果只读结果复核失败：{exc}",
            }
            session.effect_verification = failed
            session.status = "blocked"
            session.failed_reason = failed["reason"]
            session.qwen_decision = None
            session.controller_decision = CanonicalSelectionReceipt(
                allowed=False,
                reason=session.failed_reason,
            )
            if session.physical_actions != before_actions:
                raise UniversalAgentOrchestratorError(
                    "外部效果只读复核失败路径错误地改变了物理动作计数。"
                )
            decision = SimpleNamespace(
                proposal=GenericStepProposal(
                    status="blocked",
                    reason=session.failed_reason,
                )
            )
            self._write_terminal_snapshot(session)
            return decision

        session.qwen_decision = None
        session.controller_decision = None
        session.confirmation_authority = None
        session.effect_confirmation_authority = None
        session.confirmed_effect_ids = ()
        result_subgoal = next(
            (
                item
                for item in revised.subgoals
                if item.subgoal_id == pending["subgoal_id"]
            ),
            None,
        )
        current_visual_refs = {
            item.ref_id for item in observed.visual_claim_evidence_refs
        }
        visual_result_proven = bool(
            revised.status == "completed"
            and result_subgoal is not None
            and result_subgoal.status == "completed"
            and set(result_subgoal.completion_evidence).intersection(
                current_visual_refs
            )
        )
        if visual_result_proven or revised.status != "completed":
            session.task_graph = revised
            session.goal_draft = self.bridge.goal_draft(revised)
            self._remember(
                session,
                session.evidence_store.write_task_graph(revised),
                session.evidence_store.write_effect_policy_snapshot(revised),
            )
        final = {
            **pending,
            "status": "verified" if visual_result_proven else "failed",
            "verification_attempts": 1,
            "verification_observation_id": observation.observation_id,
            "verification_fingerprint": observation.fingerprint,
            "visual_claim_refs": sorted(
                set(result_subgoal.completion_evidence).intersection(
                    current_visual_refs
                )
                if result_subgoal is not None
                else ()
            ),
        }
        session.effect_verification = final
        if visual_result_proven:
            session.status = "succeeded"
            session.failed_reason = ""
            proposal = TaskGraphTransitionReport(
                status="completed",
                reason="新的可信画面已证明一次性外部效果结果。",
                completion_evidence=tuple(final["visual_claim_refs"][:3]),
            )
        else:
            session.status = "blocked"
            session.failed_reason = (
                "新的只读观察仍未以当前 typed visual claim 证明外部效果结果；"
                "效果动作不会重试。"
            )
            final["reason"] = session.failed_reason
            session.effect_verification = final
            proposal = GenericStepProposal(
                status="blocked",
                reason=session.failed_reason,
            )
        if session.physical_actions != before_actions:
            raise UniversalAgentOrchestratorError(
                "外部效果只读复核错误地改变了物理动作计数。"
            )
        decision = SimpleNamespace(proposal=proposal)
        self._write_terminal_snapshot(session)
        return decision

    def refresh_decision(self, session: UniversalAgentSessionState) -> Any:
        """Capture a fresh trusted scene and replace the pending decision.

        This is the read-only implementation behind the web ``/next`` route.
        It deliberately never calls ``adapter.execute`` and always invalidates
        the previous confirmation before touching the camera.
        """

        if self.device_registry.active_session(session.device_id) != session.session_id:
            raise UniversalAgentOrchestratorError(
                "当前会话已不再拥有该设备，禁止重新观察。"
            )
        try:
            with self._vision_usage_scope(
                session.vision_usage
            ), self.device_registry.device_lock(session.device_id):
                return self._refresh_decision_locked(session)
        finally:
            self._release_if_terminal(session)

    def _refresh_decision_locked(self, session: UniversalAgentSessionState) -> Any:
        graph = session.task_graph
        goal = session.goal_draft
        if graph is None or goal is None:
            raise UniversalAgentOrchestratorError("当前会话缺少任务图或目标投影。")
        self._validate_graph_identity(graph, device_id=session.device_id)
        pending_effect = None
        if session.status == "needs_effect_verification":
            pending_effect = self._validate_pending_effect_verification(
                session,
                graph,
            )
        current = graph.active_subgoal()
        observed_subgoal_signature = _subgoal_progress_signature(current)
        impact = current.external_impact if current is not None else "unknown"
        if pending_effect is None and session.status == "awaiting_effect_confirmation":
            raise UniversalAgentOrchestratorError(
                "当前子目标必须先满足本地效果策略，禁止提前调用 Qwen。"
            )
        if pending_effect is None and (current is None or (
            _requires_effect_confirmation(graph, current)
            and not session.confirmed_effect_ids
        )):
            raise UniversalAgentOrchestratorError(
                f"当前 {impact} 子目标缺少有效效果确认。"
            )

        prior_observation = session.trusted_observation
        authority = session.confirmation_authority
        if authority is not None:
            authority.consumed = True
            authority.invalid_reason = "fresh_observation_requested"
        session.confirmation_authority = None
        before_actions = session.physical_actions
        try:
            session.status = "observing"
            observation_id = f"obs_{uuid.uuid4().hex}"
            scene, frames, frame_paths = session.adapter.capture_scene(
                goal,
                evidence_dir=session.run_dir,
                prefix=(
                    f"refresh_step_{session.step_number}_"
                    f"{observation_id[-8:]}_frame"
                ),
            )
            self._remember(session, frame_paths)
            try:
                observation = self.trusted_observation_factory(
                    frames=frames,
                    device_id=session.device_id,
                    scene=scene,
                    observation_id=observation_id,
                )
            except Exception as exc:
                session.status = "blocked"
                session.failed_reason = f"重新观察证据不足：{exc}"
                session.qwen_decision = None
                session.controller_decision = CanonicalSelectionReceipt(
                    allowed=False,
                    reason=session.failed_reason,
                )
                if session.physical_actions != before_actions:
                    raise UniversalAgentOrchestratorError(
                        "重新观察证据失败路径错误地改变了物理动作计数。"
                    )
                blocked_decision = SimpleNamespace(
                    proposal=GenericStepProposal(
                        status="blocked",
                        reason=session.failed_reason,
                    )
                )
                self._write_terminal_snapshot(session)
                return blocked_decision
            session.trusted_observation = observation
            session.trusted_frames = tuple(frames)
            lineage = session.verified_app_surface_lineage
            if lineage is not None and (
                lineage.physical_actions != session.physical_actions
                or not self._lineage_matches_observed_foreground(
                    lineage,
                    str(scene.foreground_app_id),
                )
            ):
                session.verified_app_surface_lineage = (
                    self._refresh_verified_app_surface_lineage(
                        session=session,
                        graph=graph,
                        prior_observation=prior_observation,
                        new_observation=observation,
                    )
                )
            self._remember(
                session,
                session.evidence_store.write_trusted_observation(
                    session.step_number,
                    observation,
                ),
            )

            if pending_effect is not None:
                return self._complete_pending_effect_verification(
                    session,
                    graph=graph,
                    observation=observation,
                    before_actions=before_actions,
                )

            current = graph.active_subgoal()
            if current is not None and current.external_impact in {
                "read_only",
                "navigation_only",
            }:
                visible_revised, visible_advances = (
                    self._advance_visible_presence_prefix(
                        session,
                        graph=graph,
                        trusted_observation=observation,
                    )
                )
                if visible_advances:
                    graph = visible_revised
                    if visible_revised.status == "completed":
                        session.status = "succeeded"
                        session.failed_reason = ""
                    else:
                        next_subgoal = visible_revised.active_subgoal()
                        if next_subgoal is None:
                            session.status = "blocked"
                            session.failed_reason = (
                                "可见状态证据推进后没有活动子目标。"
                            )
                        elif _requires_effect_confirmation(
                            visible_revised,
                            next_subgoal,
                        ):
                            session.status = "awaiting_effect_confirmation"
                            session.failed_reason = ""
                            self._bind_effect_confirmation(session)
                        else:
                            session.status = "needs_reobservation"
                            session.failed_reason = ""
                    decision = SimpleNamespace(
                        proposal=TaskGraphTransitionReport(
                            status=(
                                "completed"
                                if visible_revised.status == "completed"
                                else "progressed"
                            ),
                            reason=(
                                "当前可见状态已由 DeepSeek 任务图和本地"
                                "完成条件共同复核。"
                            ),
                            completion_evidence=(scene.summary,),
                        )
                    )
                    session.qwen_decision = None
                    session.controller_decision = None
                    self._write_terminal_snapshot(session)
                    return decision

            if current is not None and current.external_impact == "read_only":
                text_revised = self._try_advance_visible_text_read_subgoal(
                    session,
                    graph=graph,
                    trusted_observation=observation,
                )
                if text_revised is not None:
                    self._store_revised_graph(session, text_revised)
                    if text_revised.status == "completed":
                        session.status = "succeeded"
                    else:
                        next_subgoal = text_revised.active_subgoal()
                        if next_subgoal is None:
                            session.status = "blocked"
                            session.failed_reason = "只读文字结果推进后没有活动子目标。"
                        elif _requires_effect_confirmation(text_revised, next_subgoal):
                            session.status = "awaiting_effect_confirmation"
                            session.failed_reason = ""
                            self._bind_effect_confirmation(session)
                        else:
                            session.status = "needs_reobservation"
                            session.failed_reason = ""
                    decision = SimpleNamespace(
                        proposal=TaskGraphTransitionReport(
                            status=(
                                "completed"
                                if text_revised.status == "completed"
                                else "progressed"
                            ),
                            reason="唯一可信可见文字已由 DeepSeek 复核。",
                            completion_evidence=(scene.summary,),
                        )
                    )
                    session.qwen_decision = None
                    session.controller_decision = None
                    self._write_terminal_snapshot(session)
                    return decision

            if (
                prior_observation is not None
                and str(getattr(prior_observation, "fingerprint", ""))
                != str(observation.fingerprint)
            ):
                observed = self.bridge.observed_state(
                    graph=graph,
                    trusted_observation=observation,
                    action_outcome="not_applicable",
                    verification={
                        "visible_evidence": [scene.summary],
                        "blocked_reasons": [],
                    },
                )
                try:
                    revised = self.deepseek_planner.replan(
                        graph,
                        observed,
                        trigger="observation_changed",
                        reason=(
                            "只读重新观察发现页面指纹变化；必须先修订高层状态，"
                            "再允许 Qwen 规划下一动作。"
                        ),
                    )
                    self._validate_graph_identity(
                        revised,
                        device_id=session.device_id,
                        previous=graph,
                        trusted_observation=observation,
                    )
                except Exception as exc:
                    session.status = "blocked"
                    session.failed_reason = f"页面变化重规划失败：{exc}"
                    session.qwen_decision = None
                    session.controller_decision = CanonicalSelectionReceipt(
                        allowed=False,
                        reason=session.failed_reason,
                    )
                    session.confirmed_effect_ids = ()
                    session.effect_confirmation_authority = None
                    if session.physical_actions != before_actions:
                        raise UniversalAgentOrchestratorError(
                            "页面变化重规划失败路径错误地改变了物理动作计数。"
                        )
                    blocked_decision = SimpleNamespace(
                        proposal=GenericStepProposal(
                            status="blocked",
                            reason=session.failed_reason,
                        )
                    )
                    self._write_terminal_snapshot(session)
                    return blocked_decision
                session.task_graph = revised
                session.goal_draft = self.bridge.goal_draft(revised)
                session.qwen_decision = None
                session.controller_decision = None
                session.confirmed_effect_ids = ()
                session.effect_confirmation_authority = None
                graph = revised
                goal = session.goal_draft
                self._remember(
                    session,
                    session.evidence_store.write_task_graph(revised),
                    session.evidence_store.write_effect_policy_snapshot(revised),
                )
                if revised.status == "completed":
                    session.status = "succeeded"
                    session.failed_reason = ""
                    terminal_decision = SimpleNamespace(
                        proposal=TaskGraphTransitionReport(
                            status="completed",
                            reason="DeepSeek 已依据新的可信画面确认任务完成。",
                            completion_evidence=observed.visible_evidence[:3],
                        )
                    )
                    if session.physical_actions != before_actions:
                        raise UniversalAgentOrchestratorError(
                            "重新观察路径错误地改变了物理动作计数。"
                        )
                    self._write_terminal_snapshot(session)
                    return terminal_decision
                current = revised.active_subgoal()
                impact = (
                    current.external_impact if current is not None else "unknown"
                )
                if current is None:
                    session.status = "blocked"
                    session.failed_reason = "页面变化重规划后没有活动子目标。"
                    blocked_decision = SimpleNamespace(
                        proposal=GenericStepProposal(
                            status="blocked",
                            reason=session.failed_reason,
                        )
                    )
                    self._write_terminal_snapshot(session)
                    return blocked_decision
                if _requires_effect_confirmation(graph, current):
                    session.status = "awaiting_effect_confirmation"
                    session.failed_reason = ""
                    self._bind_effect_confirmation(session)
                    risk_decision = SimpleNamespace(
                        proposal=GenericStepProposal(
                            status="blocked",
                            reason="页面变化后必须重新确认当前效果作用域。",
                        )
                    )
                    self._write_terminal_snapshot(session)
                    return risk_decision
                if (
                    _subgoal_progress_signature(current)
                    != observed_subgoal_signature
                ):
                    session.status = "needs_reobservation"
                    session.failed_reason = ""
                    session.qwen_decision = None
                    session.controller_decision = None
                    session.confirmation_authority = None
                    progressed_decision = SimpleNamespace(
                        proposal=TaskGraphTransitionReport(
                            status="progressed",
                            reason=(
                                "页面变化已推进活动子目标；必须按新子目标"
                                "重新截图后再选择动作。"
                            ),
                            completion_evidence=(scene.summary,),
                        )
                    )
                    self._write_terminal_snapshot(session)
                    return progressed_decision

            if session.confirmed_effect_ids:
                context = graph.to_qwen_context(
                    confirmed_effect_ids=session.confirmed_effect_ids,
                    confirmed_task_id=graph.task_id,
                    confirmed_device_id=graph.device_id,
                    confirmed_subgoal_id=graph.active_subgoal_id,
                    confirmed_revision=graph.revision,
                )
            else:
                context = graph.to_qwen_context()
            decision = self._decide_next_action(
                session,
                frames=frames,
                task_context=context,
                trusted_observation=observation,
            )
            self._validate_decision_binding(graph, observation, decision)
            session.qwen_decision = decision
            self._remember(
                session,
                session.evidence_store.write_qwen_decision(
                    session.step_number,
                    decision,
                ),
            )

            if decision.proposal.status == "action":
                selection_receipt = self._selection_receipt(session, decision)
                session.controller_decision = selection_receipt
                self._remember(
                    session,
                    session.evidence_store.write_controller_decision(
                        session.step_number,
                        self._selection_receipt_payload(selection_receipt),
                    ),
                )
                if selection_receipt.allowed:
                    session.status = "awaiting_confirmation"
                    session.failed_reason = ""
                    self._bind_confirmation(session)
                else:
                    session.status = "blocked"
                    session.failed_reason = selection_receipt.reason
            else:
                session.status = "blocked"
                session.failed_reason = (
                    decision.proposal.reason
                    if decision.proposal.status == "blocked"
                    else f"不支持的 Qwen 状态：{decision.proposal.status}"
                )
                session.controller_decision = CanonicalSelectionReceipt(
                    allowed=False,
                    reason=session.failed_reason,
                )
            if session.physical_actions != before_actions:
                raise UniversalAgentOrchestratorError(
                    "重新观察路径错误地改变了物理动作计数。"
                )
            self._write_terminal_snapshot(session)
            return decision
        except Exception as exc:
            session.status = "failed"
            session.failed_reason = str(exc)
            self._record_deepseek_failure(
                session,
                exc,
                stage="refresh_decision",
            )
            try:
                self._write_terminal_snapshot(session)
            except Exception:
                pass
            raise

    def confirm_one(
        self,
        session: UniversalAgentSessionState,
        confirmation: Mapping[str, Any],
    ) -> Any:
        if self.device_registry.active_session(session.device_id) != session.session_id:
            raise UniversalAgentOrchestratorError(
                "当前会话已不再拥有该设备，禁止执行。"
            )
        before_actions = session.physical_actions
        authority_before = session.confirmation_authority
        post_transition_before = session.last_post_action_transition
        try:
            with self._vision_usage_scope(
                session.vision_usage
            ), self.device_registry.device_lock(session.device_id):
                try:
                    return self._confirm_one_locked(session, confirmation)
                except Exception as exc:
                    self._finalize_confirm_failure(
                        session,
                        exc,
                        before_actions=before_actions,
                        authority_before=authority_before,
                        post_transition_before=post_transition_before,
                    )
                    raise
        finally:
            self._release_if_terminal(session)

    def _finalize_confirm_failure(
        self,
        session: UniversalAgentSessionState,
        error: Exception,
        *,
        before_actions: int,
        authority_before: Any,
        post_transition_before: Any,
    ) -> None:
        """Best-effort memory/artifact convergence after a consumed confirm.

        The original exception remains authoritative and is always re-raised by
        the caller.  This method never invokes the adapter or any observation
        source and never invents unavailable receipt/frame fields.
        """

        authority = session.confirmation_authority or authority_before
        authority_consumed = bool(
            authority is not None and getattr(authority, "consumed", False)
        )
        request_actions = max(0, session.physical_actions - int(before_actions))
        if not (authority_consumed or request_actions):
            return

        # A transition produced by the normal after-observation path is more
        # specific than this exception boundary.  Never replace it; only make a
        # best-effort attempt to converge the terminal report with memory.
        if session.last_post_action_transition is not post_transition_before:
            try:
                self._ensure_terminal_snapshot(session)
            except Exception:
                pass
            return

        failed_stage = session.confirm_stage or "confirmation"
        reason = str(error).strip() or error.__class__.__name__
        if request_actions == 0 and session.status == "blocked":
            # Policy recheck already produced the authoritative blocked terminal
            # state and snapshot.  Consuming the confirmation token alone does
            # not turn that pre-action rejection into an execution failure.
            try:
                self._ensure_terminal_snapshot(session)
            except Exception:
                pass
            return
        recoverable_reobservation = bool(
            request_actions == 0 and session.status == "needs_reobservation"
        )
        if not recoverable_reobservation:
            session.status = "failed"
        session.failed_reason = reason
        if request_actions == 0:
            requested_kind = str(
                getattr(
                    getattr(
                        getattr(session.qwen_decision, "proposal", None),
                        "action",
                        None,
                    ),
                    "action",
                    "",
                )
                or ""
            )
            if failed_stage == "validating_confirmation":
                transition_kind = "confirmation_failure"
            elif requested_kind == "wait_for_change":
                transition_kind = "wait_observation_failure"
            else:
                transition_kind = "pre_action_failure"
            transition = {
                "protocol_version": POST_ACTION_TRANSITION_PROTOCOL_VERSION,
                "transition_kind": transition_kind,
                "disposition": (
                    "needs_reobservation"
                    if recoverable_reobservation
                    else "failed"
                ),
                "failed_stage": failed_stage,
                "error_type": error.__class__.__name__,
                "error": reason,
                "authority_consumed": authority_consumed,
                "physical_actions_before": int(before_actions),
                "physical_actions": int(session.physical_actions),
                "request_physical_actions": 0,
                "evidence": list(dict.fromkeys(session.evidence_paths)),
            }
            if authority is not None and callable(getattr(authority, "scope", None)):
                try:
                    transition["authority_scope"] = authority.scope()
                except Exception:
                    pass
            session.last_confirmation_failure = transition
            failure_step = max(1, session.step_number)
            try:
                self._remember(
                    session,
                    session.evidence_store.write_confirmation_failure(
                        failure_step,
                        transition,
                    ),
                )
            except Exception:
                pass
            try:
                self._write_terminal_snapshot(session)
            except Exception:
                pass
            return

        transition: dict[str, Any] = {
            "protocol_version": POST_ACTION_TRANSITION_PROTOCOL_VERSION,
            "transition_kind": "post_action_failure",
            "disposition": "failed",
            "failed_stage": failed_stage,
            "error_type": error.__class__.__name__,
            "error": reason,
            "authority_consumed": authority_consumed,
            "physical_actions_before": int(before_actions),
            "physical_actions": int(session.physical_actions),
            "request_physical_actions": request_actions,
            "evidence": list(dict.fromkeys(session.evidence_paths)),
        }
        if authority is not None and callable(getattr(authority, "scope", None)):
            try:
                transition["authority_scope"] = authority.scope()
            except Exception:
                pass
        session.last_post_action_transition = transition
        failure_step = max(1, session.step_number)
        if session.history and authority is not None:
            latest = session.history[-1]
            if latest.get("task_revision") == getattr(authority, "revision", None):
                failure_step = max(1, int(latest.get("step_number") or failure_step))
        try:
            self._remember(
                session,
                session.evidence_store.write_post_action_transition(
                    failure_step,
                    transition,
                ),
            )
        except Exception:
            pass
        try:
            self._write_terminal_snapshot(session)
        except Exception:
            pass

    def _confirm_one_locked(
        self,
        session: UniversalAgentSessionState,
        confirmation: Mapping[str, Any],
    ) -> Any:
        session.confirm_stage = "validating_confirmation"
        self._validate_and_consume_confirmation(session, confirmation)
        authority = session.confirmation_authority
        assert authority is not None
        graph = session.task_graph
        observation = session.trusted_observation
        decision = session.qwen_decision
        assert graph is not None and observation is not None and decision is not None

        session.confirm_stage = "scope_consumed"
        selection_receipt = session.controller_decision
        if selection_receipt is None or not selection_receipt.allowed:
            raise UniversalAgentOrchestratorError(
                "确认作用域缺少已验证的 canonical selection receipt。"
            )
        authority.consumed = True
        authority.invalid_reason = "consumed_before_execution"

        session.confirm_stage = "pre_execute_evidence"
        try:
            self._remember(
                session,
                session.evidence_store.write_controller_decision(
                    session.step_number,
                    {
                        **self._selection_receipt_payload(selection_receipt),
                        "phase": "pre_execute_scope_consume",
                    },
                ),
            )
        except Exception:
            session.status = "failed"
            session.failed_reason = "执行前控制器证据写入失败。"
            raise

        session.status = "executing_one_action"
        session.confirm_stage = "executing"
        prior_verified_app_surface_lineage = (
            session.verified_app_surface_lineage
        )
        prior_physical_actions = session.physical_actions
        if decision.proposal.action.action != "wait_for_change":
            session.verified_app_surface_lineage = None
        try:
            result = session.adapter.execute(
                requested_action=decision.proposal.action,
                planned_scene=observation.scene,
                goal=session.goal_draft,
                confirmed=True,
                evidence_dir=session.run_dir,
                planned_frames=session.trusted_frames,
            )
        except GenericActionAdapterError as exc:
            failed_physical_actions = max(0, int(exc.physical_actions))
            if failed_physical_actions == 0:
                session.verified_app_surface_lineage = (
                    prior_verified_app_surface_lineage
                )
            session.physical_actions += failed_physical_actions
            self._remember(session, exc.evidence)
            session.status = (
                "needs_reobservation"
                if failed_physical_actions == 0
                else "failed"
            )
            session.failed_reason = str(exc)
            try:
                self._write_terminal_snapshot(session)
            except Exception:
                pass
            raise

        self._remember(
            session,
            getattr(result, "evidence", ()),
            getattr(result, "after_frame_paths", ()),
        )
        session.confirm_stage = "validating_execution_result"
        physical_actions = int(result.physical_actions)
        wait_transition = result.resolved_action.kind == "wait_for_change"
        if physical_actions != 1 and not (wait_transition and physical_actions == 0):
            session.physical_actions += max(0, physical_actions)
            session.status = "failed"
            session.failed_reason = (
                "已确认动作必须产生一次物理动作，或仅 wait_for_change 产生零动作；"
                f"实际返回：{physical_actions}。"
            )
            raise UniversalAgentOrchestratorError(session.failed_reason)
        session.physical_actions += physical_actions

        requested_action = decision.proposal.action
        rebound_params = dict(getattr(result.rebound_action, "params", {}) or {})
        resolved_kind = str(getattr(result.resolved_action, "kind", ""))
        resolved_target_binding_ok = True
        if resolved_kind in {
            "tap_semantic",
            "dismiss_overlay",
            "input_verified_text",
            "press_enter",
            "clear_verified_text",
            "long_press",
        }:
            resolved_target_binding_ok = (
                str(getattr(result.resolved_action, "target_element_id", "") or "")
                == str(rebound_params.get("element_id") or "")
            )
        elif resolved_kind == "drag":
            resolved_target_binding_ok = (
                str(getattr(result.resolved_action, "target_element_id", "") or "")
                == str(rebound_params.get("source_element_id") or "")
                and str(
                    getattr(result.resolved_action, "destination_element_id", "")
                    or ""
                )
                == str(rebound_params.get("destination_element_id") or "")
            )
        if (
            requested_action is None
            or _action_digest(requested_action) != authority.action_digest
            or _action_digest(result.requested_action) != authority.action_digest
            or str(getattr(result.requested_action, "node_id", ""))
            != authority.decision_node_id
            or str(getattr(result.rebound_action, "node_id", ""))
            != authority.decision_node_id
            or str(getattr(result.resolved_action, "node_id", ""))
            != authority.decision_node_id
            or str(getattr(result.resolved_action, "kind", ""))
            != str(getattr(result.rebound_action, "action", ""))
            or not resolved_target_binding_ok
            or dict(getattr(result.resolved_action, "expected_effect", {}) or {})
            != dict(
                getattr(result.rebound_action, "params", {}).get(
                    "expected_effect", {}
                )
                or {}
            )
        ):
            session.status = "failed"
            session.failed_reason = (
                "执行结果没有严格绑定 confirmed/requested/rebound/resolved 动作链。"
            )
            raise UniversalAgentOrchestratorError(session.failed_reason)
        session.status = "verifying"
        session.confirm_stage = "validating_post_action_evidence"
        action_outcome = str(getattr(result, "action_outcome", ""))
        verification_errors = tuple(
            str(item)
            for item in getattr(result, "verification_errors", ())
            if str(item).strip()
        )
        if action_outcome not in POST_ACTION_OUTCOMES:
            session.status = "failed"
            session.failed_reason = f"动作结果 outcome 无效：{action_outcome}。"
            raise UniversalAgentOrchestratorError(session.failed_reason)
        if (action_outcome == "matched") == bool(verification_errors):
            session.status = "failed"
            session.failed_reason = (
                "动作结果 outcome 与 verification_errors 不一致。"
            )
            raise UniversalAgentOrchestratorError(session.failed_reason)
        if (
            result.planned_scene_fingerprint != observation.fingerprint
            or result.confirmation_frame_identity_verified is not True
        ):
            session.status = "failed"
            session.failed_reason = "动作结果没有绑定确认 scope 的规划画面。"
            raise UniversalAgentOrchestratorError(session.failed_reason)
        if result.resolved_action.before_fingerprint != result.before_scene.fingerprint:
            session.status = "failed"
            session.failed_reason = "动作结果没有绑定复核后的执行前画面。"
            raise UniversalAgentOrchestratorError(session.failed_reason)
        after_frames = tuple(getattr(result, "after_frames", ()))
        after_paths = tuple(
            str(item).strip() for item in getattr(result, "after_frame_paths", ())
        )
        if (
            len(after_frames) < 4
            or len(after_paths) != len(after_frames)
            or any(not item for item in after_paths)
            or len(set(after_paths)) != len(after_paths)
        ):
            session.status = "failed"
            session.failed_reason = "动作后可信观察缺少完整且唯一的原始帧证据。"
            raise UniversalAgentOrchestratorError(session.failed_reason)
        if (
            action_outcome == "matched"
            and
            result.resolved_action.kind != "wait_for_change"
            and result.after_scene.fingerprint == observation.fingerprint
        ):
            session.status = "failed"
            session.failed_reason = "动作后 fingerprint 没有变化，禁止继续。"
            try:
                self._write_terminal_snapshot(session)
            except Exception:
                pass
            raise UniversalAgentOrchestratorError(session.failed_reason)

        session.confirm_stage = "building_trusted_observation"
        new_observation = self.trusted_observation_factory(
            frames=list(after_frames),
            device_id=session.device_id,
            scene=result.after_scene,
            observation_id=f"obs_{uuid.uuid4().hex}",
        )
        if (
            str(getattr(new_observation, "device_id", "")) != session.device_id
            or str(getattr(new_observation, "fingerprint", ""))
            != result.after_scene.fingerprint
            or str(getattr(getattr(new_observation, "scene", None), "fingerprint", ""))
            != result.after_scene.fingerprint
        ):
            session.status = "failed"
            session.failed_reason = "动作后可信观察未严格绑定 device/after scene fingerprint。"
            raise UniversalAgentOrchestratorError(session.failed_reason)
        if (
            new_observation.observation_id == observation.observation_id
            or (
                action_outcome == "matched"
                and
                result.resolved_action.kind != "wait_for_change"
                and new_observation.fingerprint == observation.fingerprint
            )
        ):
            session.status = "failed"
            session.failed_reason = "动作后可信观察 observation/fingerprint 未更新。"
            raise UniversalAgentOrchestratorError(session.failed_reason)
        session.confirm_stage = "persisting_post_observation"
        session.trusted_observation = new_observation
        session.trusted_frames = after_frames
        session.step_number += 1
        self._remember(
            session,
            session.evidence_store.write_trusted_observation(
                session.step_number,
                new_observation,
            ),
        )
        session.history.append(
            {
                "step_number": session.step_number - 1,
                "task_revision": graph.revision,
                "qwen_decision": UniversalAgentSessionState._serialize(decision),
                "execution": result.to_dict(),
                "before_observation_id": observation.observation_id,
                "before_fingerprint": observation.fingerprint,
                "after_observation_id": new_observation.observation_id,
                "after_fingerprint": new_observation.fingerprint,
            }
        )
        try:
            session.status = "replanning"
            session.confirm_stage = "replanning"
            self._advance_after_observation(
                session,
                result=result,
                before_observation=observation,
                new_observation=new_observation,
                prior_verified_app_surface_lineage=(
                    prior_verified_app_surface_lineage
                ),
                prior_physical_actions=prior_physical_actions,
            )
            session.confirm_stage = "completed"
            self._write_terminal_snapshot(session)
            return result
        except EvidenceStoreError as exc:
            session.status = "failed"
            session.failed_reason = str(exc)
            raise
        except Exception as exc:
            session.status = "failed"
            session.failed_reason = str(exc)
            self._record_deepseek_failure(
                session,
                exc,
                stage="post_action_replan",
            )
            try:
                self._write_terminal_snapshot(session)
            except Exception:
                pass
            raise

    def approve_effects(
        self,
        session: UniversalAgentSessionState,
        confirmation: Mapping[str, Any],
    ) -> Any:
        """Consume one effect approval and execute at most its one bound effect.

        The approval binds the canonical task/subgoal/typed-effect intent. Observation,
        Qwen selection and the controller still create a separate exact action
        scope internally; when that scope is valid it is consumed immediately,
        so the user is not asked to confirm the same external effect twice.
        """

        if self.device_registry.active_session(session.device_id) != session.session_id:
            raise UniversalAgentOrchestratorError(
                "当前会话已不再拥有该设备，禁止确认风险。"
            )
        try:
            with self._vision_usage_scope(
                session.vision_usage
            ), self.device_registry.device_lock(session.device_id):
                if session.status != "awaiting_effect_confirmation":
                    raise UniversalAgentOrchestratorError(
                        f"当前状态不能确认风险：{session.status}。"
                    )
                authority = session.effect_confirmation_authority
                if authority is None or authority.consumed:
                    raise UniversalAgentOrchestratorError("当前效果确认已失效或已使用。")
                requested = self._normalize_effect_confirmation(confirmation)
                if requested != authority.scope():
                    authority.consumed = True
                    authority.invalid_reason = "effect_scope_mismatch"
                    raise UniversalAgentOrchestratorError(
                        "效果确认与当前 task/device/revision/subgoal/effect 不一致。"
                    )
                authority.consumed = True
                authority.invalid_reason = "consumed_before_observation"
                session.confirmed_effect_ids = tuple(authority.effect_ids)
                graph = session.task_graph
                assert graph is not None
                context = graph.to_qwen_context(
                    confirmed_effect_ids=session.confirmed_effect_ids,
                    confirmed_task_id=graph.task_id,
                    confirmed_device_id=graph.device_id,
                    confirmed_subgoal_id=graph.active_subgoal_id,
                    confirmed_revision=graph.revision,
                )
                result = self._observe_after_effect_confirmation(session, context)
                if session.status == "awaiting_confirmation":
                    action_authority = session.confirmation_authority
                    if action_authority is None or action_authority.consumed:
                        raise UniversalAgentOrchestratorError(
                            "效果确认后没有形成一次性精确动作作用域。"
                        )
                    result = self._confirm_one_locked(
                        session,
                        action_authority.scope(),
                    )
                self._write_terminal_snapshot(session)
                return result
        finally:
            self._release_if_terminal(session)

    def run_autonomous_safe_loop(
        self,
        session: UniversalAgentSessionState,
        *,
        max_physical_actions: int = 12,
        max_iterations: int = 24,
    ) -> dict[str, Any]:
        """Advance read-only/navigation work with fresh one-shot scopes.

        Every iteration consumes the exact authority already bound to the
        latest revision/observation/decision, executes at most one physical
        action, then re-observes and replans through ``_confirm_one_locked``.
        External/unknown work and exhausted budgets stop the loop.  A semantic
        mismatch may schedule exactly one corrective physical action, but only
        after a new observation, new canonical candidate, new geometry and new
        one-shot scope; the consumed action/coordinate is never replayed.
        """

        if self.device_registry.active_session(session.device_id) != session.session_id:
            raise UniversalAgentOrchestratorError(
                "当前会话已不再拥有该设备，禁止自动推进。"
            )
        if (
            isinstance(max_physical_actions, bool)
            or not isinstance(max_physical_actions, int)
            or not 1 <= max_physical_actions <= 20
        ):
            raise UniversalAgentOrchestratorError("安全动作预算必须是1～20。")
        if (
            isinstance(max_iterations, bool)
            or not isinstance(max_iterations, int)
            or not 1 <= max_iterations <= 40
        ):
            raise UniversalAgentOrchestratorError("安全迭代预算必须是1～40。")

        start_actions = session.physical_actions
        iterations = 0
        pending_corrective_retry: dict[str, Any] | None = None
        session.automatic_loop_enabled = True
        session.auto_pause_reason = ""
        try:
            with self._vision_usage_scope(
                session.vision_usage
            ), self.device_registry.device_lock(session.device_id):
                while iterations < max_iterations:
                    if session.status in {
                        "succeeded",
                        "blocked",
                        "failed",
                        "cancelled",
                        "awaiting_effect_confirmation",
                    }:
                        break
                    graph = session.task_graph
                    current = graph.active_subgoal() if graph is not None else None
                    impact = current.external_impact if current is not None else "unknown"
                    if impact not in {"read_only", "navigation_only"}:
                        if pending_corrective_retry is not None:
                            pending_corrective_retry["status"] = (
                                "stopped_before_corrective_action"
                            )
                            pending_corrective_retry["stop_reason"] = (
                                "重新规划后的子目标不再属于普通只读或导航动作。"
                            )
                            pending_corrective_retry = None
                        session.auto_pause_reason = (
                            "下一子目标可能产生外部影响或仍未知，已在物理动作前停止。"
                        )
                        break
                    if session.status == "needs_reobservation":
                        prior_observation_id = str(
                            getattr(session.trusted_observation, "observation_id", "")
                            or ""
                        )
                        self._refresh_decision_locked(session)
                        iterations += 1
                        if pending_corrective_retry is not None:
                            refreshed_observation_id = str(
                                getattr(
                                    session.trusted_observation,
                                    "observation_id",
                                    "",
                                )
                                or ""
                            )
                            refreshed_fingerprint = str(
                                getattr(
                                    session.trusted_observation,
                                    "fingerprint",
                                    "",
                                )
                                or ""
                            )
                            pending_corrective_retry.update(
                                {
                                    "refresh_observation_id": (
                                        refreshed_observation_id
                                    ),
                                    "refresh_fingerprint": refreshed_fingerprint,
                                }
                            )
                            if (
                                not refreshed_observation_id
                                or refreshed_observation_id == prior_observation_id
                            ):
                                pending_corrective_retry["status"] = (
                                    "stopped_before_corrective_action"
                                )
                                pending_corrective_retry["stop_reason"] = (
                                    "纠正重观察没有形成新的 observation。"
                                )
                                session.status = "failed"
                                session.failed_reason = (
                                    pending_corrective_retry["stop_reason"]
                                )
                                session.auto_pause_reason = session.failed_reason
                                pending_corrective_retry = None
                                break
                            refreshed_graph = session.task_graph
                            refreshed_current = (
                                refreshed_graph.active_subgoal()
                                if refreshed_graph is not None
                                else None
                            )
                            refreshed_subgoal_id = str(
                                getattr(refreshed_current, "subgoal_id", "") or ""
                            )
                            if session.status == "succeeded":
                                pending_corrective_retry["status"] = (
                                    "resolved_by_reobservation"
                                )
                                pending_corrective_retry = None
                            elif (
                                refreshed_subgoal_id
                                and refreshed_subgoal_id
                                != pending_corrective_retry.get(
                                    "source_subgoal_id"
                                )
                            ):
                                pending_corrective_retry["status"] = (
                                    "resolved_by_replan"
                                )
                                pending_corrective_retry[
                                    "replanned_subgoal_id"
                                ] = refreshed_subgoal_id
                                pending_corrective_retry = None
                            elif session.status == "awaiting_confirmation":
                                pending_corrective_retry["status"] = (
                                    "ready_for_corrective_action"
                                )
                            elif session.status == "needs_reobservation":
                                pending_corrective_retry["status"] = (
                                    "stopped_before_corrective_action"
                                )
                                pending_corrective_retry["stop_reason"] = (
                                    "一次新观察仍未形成唯一可执行动作。"
                                )
                                session.status = "failed"
                                session.failed_reason = (
                                    pending_corrective_retry["stop_reason"]
                                )
                                session.auto_pause_reason = session.failed_reason
                                pending_corrective_retry = None
                                break
                            else:
                                pending_corrective_retry["status"] = (
                                    "stopped_before_corrective_action"
                                )
                                pending_corrective_retry["stop_reason"] = (
                                    session.failed_reason
                                    or f"重新观察后状态为 {session.status}。"
                                )
                                pending_corrective_retry = None
                        continue
                    if session.status != "awaiting_confirmation":
                        session.auto_pause_reason = (
                            f"会话状态 {session.status} 没有可执行的安全动作。"
                        )
                        break
                    authority = session.confirmation_authority
                    if authority is None or authority.consumed:
                        raise UniversalAgentOrchestratorError(
                            "安全自动推进缺少当前一次性动作作用域。"
                        )
                    before = session.physical_actions
                    action_kind = str(
                        getattr(
                            getattr(
                                getattr(session.qwen_decision, "proposal", None),
                                "action",
                                None,
                            ),
                            "action",
                            "",
                        )
                        or ""
                    )
                    is_corrective_action = bool(
                        pending_corrective_retry is not None
                        and pending_corrective_retry.get("status")
                        == "ready_for_corrective_action"
                    )
                    if is_corrective_action and not _allows_fresh_observation_corrective_retry(
                        impact=impact,
                        action_kind=action_kind,
                    ):
                        assert pending_corrective_retry is not None
                        pending_corrective_retry["status"] = (
                            "stopped_before_corrective_action"
                        )
                        pending_corrective_retry["stop_reason"] = (
                            "新计划不再是允许自动纠正的普通导航动作。"
                        )
                        session.auto_pause_reason = pending_corrective_retry[
                            "stop_reason"
                        ]
                        pending_corrective_retry = None
                        break
                    if is_corrective_action:
                        assert pending_corrective_retry is not None
                        pending_corrective_retry.update(
                            {
                                "corrective_action_kind": action_kind,
                                "corrective_observation_id": str(
                                    getattr(
                                        session.trusted_observation,
                                        "observation_id",
                                        "",
                                    )
                                    or ""
                                ),
                                "corrective_fingerprint": str(
                                    getattr(
                                        session.trusted_observation,
                                        "fingerprint",
                                        "",
                                    )
                                    or ""
                                ),
                            }
                        )
                    result = self._confirm_one_locked(session, authority.scope())
                    iterations += 1
                    delta = session.physical_actions - before
                    if delta not in {0, 1}:
                        raise UniversalAgentOrchestratorError(
                            "单轮安全自动推进产生了超过一个物理动作。"
                        )
                    if getattr(result, "action_outcome", "matched") != "matched":
                        if is_corrective_action:
                            assert pending_corrective_retry is not None
                            pending_corrective_retry["status"] = "exhausted"
                            pending_corrective_retry["corrective_outcome"] = (
                                "mismatched"
                            )
                            pending_corrective_retry["stop_reason"] = (
                                "新观察重新规划后的唯一纠正动作仍未产生预期语义变化。"
                            )
                            session.status = "failed"
                            session.failed_reason = pending_corrective_retry[
                                "stop_reason"
                            ]
                            session.qwen_decision = None
                            session.controller_decision = None
                            session.confirmation_authority = None
                            session.auto_pause_reason = session.failed_reason
                            pending_corrective_retry = None
                            break
                        if not (
                            session.status
                            in {"needs_reobservation", "awaiting_confirmation"}
                            and _allows_fresh_observation_corrective_retry(
                                impact=impact,
                                action_kind=action_kind,
                            )
                        ):
                            session.auto_pause_reason = (
                                "当前动作不属于一次新观察纠正范围，已按具体结果停止。"
                            )
                            break
                        if (
                            session.physical_actions - start_actions
                            >= max_physical_actions
                        ):
                            session.status = "failed"
                            session.failed_reason = (
                                "动作未产生预期变化，但本次物理动作预算不足以执行一次纠正。"
                            )
                            session.auto_pause_reason = session.failed_reason
                            break
                        # A mismatched action may already have produced a
                        # provisional next decision from the same post-action
                        # response.  It is not fresh enough to authorize the
                        # corrective action: invalidate it and force one new
                        # observation/candidate/scope.
                        session.status = "needs_reobservation"
                        session.qwen_decision = None
                        session.controller_decision = None
                        session.confirmation_authority = None
                        transition = dict(
                            session.last_post_action_transition or {}
                        )
                        receipt = dict(transition.get("receipt") or {})
                        pending_corrective_retry = {
                            "protocol_version": (
                                CORRECTIVE_RETRY_PROTOCOL_VERSION
                            ),
                            "correction_id": f"correction_{uuid.uuid4().hex}",
                            "status": "needs_reobservation",
                            "source_receipt_id": str(
                                receipt.get("receipt_id") or ""
                            ),
                            "source_subgoal_id": str(
                                receipt.get("subgoal_id")
                                or getattr(current, "subgoal_id", "")
                                or ""
                            ),
                            "source_action_kind": str(
                                receipt.get("action_kind") or action_kind
                            ),
                            "source_before_observation_id": str(
                                receipt.get("before_observation_id") or ""
                            ),
                            "source_after_observation_id": str(
                                receipt.get("after_observation_id") or ""
                            ),
                            "source_action_digest": str(
                                receipt.get("action_digest") or ""
                            ),
                            "scheduled_after_physical_action": (
                                session.physical_actions
                            ),
                        }
                        session.corrective_retry_history.append(
                            pending_corrective_retry
                        )
                        session.auto_pause_reason = ""
                        continue
                    if is_corrective_action:
                        assert pending_corrective_retry is not None
                        pending_corrective_retry["status"] = "matched"
                        pending_corrective_retry["corrective_outcome"] = "matched"
                        pending_corrective_retry = None
                    if session.physical_actions - start_actions >= max_physical_actions:
                        session.auto_pause_reason = "已达到本次安全物理动作预算。"
                        break
                if pending_corrective_retry is not None:
                    pending_corrective_retry["status"] = (
                        "stopped_before_corrective_action"
                    )
                    pending_corrective_retry["stop_reason"] = (
                        "自动循环迭代预算耗尽，纠正动作未执行。"
                    )
                    session.status = "failed"
                    session.failed_reason = pending_corrective_retry["stop_reason"]
                    session.auto_pause_reason = session.failed_reason
                if not session.auto_pause_reason:
                    session.auto_pause_reason = {
                        "awaiting_effect_confirmation": "下一子目标需要一次效果确认。",
                        "succeeded": "目标已由新观察和任务图修订证明完成。",
                        "blocked": "当前视觉或本地门禁已阻止继续。",
                        "failed": "当前执行或验证失败，禁止自动重试。",
                    }.get(session.status, "已达到本次安全迭代预算。")
        finally:
            session.automatic_loop_enabled = False
            self._write_terminal_snapshot(session)
            self._release_if_terminal(session)
        return {
            "physical_actions": session.physical_actions - start_actions,
            "iterations": iterations,
            "status": session.status,
            "pause_reason": session.auto_pause_reason,
        }

    def _observe_after_effect_confirmation(
        self,
        session: UniversalAgentSessionState,
        task_context: Mapping[str, Any],
    ) -> Any:
        graph = session.task_graph
        assert graph is not None and session.goal_draft is not None
        before_actions = session.physical_actions
        session.status = "observing"
        scene, frames, frame_paths = session.adapter.capture_scene(
            session.goal_draft,
            evidence_dir=session.run_dir,
            prefix=f"before_step_{session.step_number}_frame",
        )
        self._remember(session, frame_paths)
        observation = self.trusted_observation_factory(
            frames=frames,
            device_id=session.device_id,
            scene=scene,
        )
        session.trusted_observation = observation
        session.trusted_frames = tuple(frames)
        self._remember(
            session,
            session.evidence_store.write_trusted_observation(
                session.step_number,
                observation,
            ),
        )
        decision = self._decide_next_action(
            session,
            frames=frames,
            task_context=task_context,
            trusted_observation=observation,
        )
        self._validate_decision_binding(graph, observation, decision)
        session.qwen_decision = decision
        self._remember(
            session,
            session.evidence_store.write_qwen_decision(
                session.step_number,
                decision,
            ),
        )
        if decision.proposal.status == "action":
            selection_receipt = self._selection_receipt(session, decision)
            session.controller_decision = selection_receipt
            self._remember(
                session,
                session.evidence_store.write_controller_decision(
                    session.step_number,
                    self._selection_receipt_payload(selection_receipt),
                ),
            )
            if selection_receipt.allowed:
                session.status = "awaiting_confirmation"
                session.failed_reason = ""
                self._bind_confirmation(session)
            else:
                session.status = "blocked"
                session.failed_reason = selection_receipt.reason
        else:
            session.status = "blocked"
            session.failed_reason = (
                decision.proposal.reason
                if decision.proposal.status == "blocked"
                else "外部状态目标的完成候选必须由 DeepSeek 新 revision 复核。"
            )
        if session.physical_actions != before_actions:
            raise UniversalAgentOrchestratorError("效果确认路径错误地产生了额外物理动作。")
        return decision

    def start(
        self,
        *,
        session_id: str,
        raw_goal: str,
        exact_input_text: str | None = None,
        exact_action_kind: str | None = None,
        exact_target_label: str = "",
        device_id: str,
        run_dir: Path,
    ) -> UniversalAgentSessionState:
        resolved_session = str(session_id or "").strip()
        resolved_device = str(device_id or "").strip()
        self.device_registry.reserve(resolved_device, resolved_session)
        try:
            vision_usage = VisionSessionUsageLedger(session_id=resolved_session)
            with self.device_registry.device_lock(resolved_device):
                with self._vision_usage_scope(vision_usage):
                    session = self._start_reserved(
                        session_id=resolved_session,
                        raw_goal=raw_goal,
                        exact_input_text=exact_input_text,
                        exact_action_kind=exact_action_kind,
                        exact_target_label=exact_target_label,
                        device_id=resolved_device,
                        run_dir=run_dir,
                        vision_usage=vision_usage,
                    )
        except Exception:
            self.device_registry.release(resolved_device, resolved_session)
            raise
        self._release_if_terminal(session)
        return session

    def _start_reserved(
        self,
        *,
        session_id: str,
        raw_goal: str,
        exact_input_text: str | None,
        exact_action_kind: str | None,
        exact_target_label: str,
        device_id: str,
        run_dir: Path,
        vision_usage: VisionSessionUsageLedger | None = None,
    ) -> UniversalAgentSessionState:
        adapter = self.adapter_factory(device_id)
        store = self.evidence_store_factory(Path(run_dir))
        session = UniversalAgentSessionState(
            session_id=str(session_id or "").strip(),
            # Preserve literal payload whitespace (especially a real LF).
            # The typed planner is responsible for validating any input_text
            # entity derived from this original user authority.
            raw_goal=str(raw_goal or "").strip(),
            device_id=str(device_id or "").strip(),
            run_dir=Path(run_dir),
            adapter=adapter,
            evidence_store=store,
            vision_usage=vision_usage,
            local_exact_input_authority=exact_input_text is not None,
        )
        if not session.session_id or not session.raw_goal or not session.device_id:
            raise UniversalAgentOrchestratorError(
                "启动通用 Agent 需要 session_id、目标和 device_id。"
            )
        try:
            session.status = "planning"
            if exact_input_text is not None and exact_action_kind is not None:
                raise UniversalAgentOrchestratorError(
                    "exact_input_text 与 exact_action_kind 不能同时使用。"
                )
            graph = (
                build_exact_input_task_graph(
                    session.raw_goal,
                    exact_input_text=exact_input_text,
                    device_id=session.device_id,
                )
                if exact_input_text is not None
                else build_exact_action_task_graph(
                    session.raw_goal,
                    action_kind=exact_action_kind,
                    target_label=exact_target_label,
                    device_id=session.device_id,
                )
                if exact_action_kind is not None
                else self.deepseek_planner.plan(
                    session.raw_goal,
                    device_id=session.device_id,
                )
            )
            self._validate_graph_identity(graph, device_id=session.device_id)
            session.task_graph = graph
            session.goal_draft = self.bridge.goal_draft(graph)
            self._remember(
                session,
                store.write_task_graph(graph),
                store.write_effect_policy_snapshot(graph),
            )

            current = graph.active_subgoal()
            impact = current.external_impact if current is not None else "unknown"
            if current is None:
                session.status = "blocked"
                session.failed_reason = "任务图没有活动子目标。"
                self._write_terminal_snapshot(session)
                return session
            session.status = "observing"
            scene, frames, frame_paths = adapter.capture_scene(
                session.goal_draft,
                evidence_dir=session.run_dir,
                prefix=f"before_step_{session.step_number}_frame",
            )
            self._remember(session, frame_paths)
            observation = self.trusted_observation_factory(
                frames=frames,
                device_id=session.device_id,
                scene=scene,
            )
            session.trusted_observation = observation
            session.trusted_frames = tuple(frames)
            self._remember(
                session,
                store.write_trusted_observation(session.step_number, observation),
            )

            if impact == "unknown":
                observed = self.bridge.observed_state(
                    graph=graph,
                    trusted_observation=observation,
                    action_outcome="not_applicable",
                    verification={
                        "visible_evidence": [scene.summary],
                        "blocked_reasons": [],
                    },
                )
                revised = self.deepseek_planner.replan(
                    graph,
                    observed,
                    trigger="observation_changed",
                    reason=(
                        "初始只读观察已经可用；请仅依据当前结构化画面事实"
                        "重新分类 unknown 子目标。不能因此宣称动作已执行。"
                    ),
                )
                self._validate_graph_identity(
                    revised,
                    device_id=session.device_id,
                    previous=graph,
                    trusted_observation=observation,
                )
                self._store_revised_graph(session, revised)
                graph = revised
                if graph.status == "completed":
                    session.status = "succeeded"
                    session.failed_reason = ""
                    self._write_terminal_snapshot(session)
                    return session
                current = graph.active_subgoal()
                if current is None:
                    session.status = "blocked"
                    session.failed_reason = "unknown 子目标重分类后没有活动子目标。"
                    self._write_terminal_snapshot(session)
                    return session
                impact = current.external_impact if current is not None else "unknown"

            if _requires_effect_confirmation(graph, current):
                session.status = "awaiting_effect_confirmation"
                session.failed_reason = ""
                session.confirmed_effect_ids = ()
                session.confirmation_authority = None
                self._bind_effect_confirmation(session)
                self._write_terminal_snapshot(session)
                return session

            if impact in {"read_only", "navigation_only"}:
                initial_safe = graph.active_subgoal()
                revised, visible_advances = self._advance_visible_presence_prefix(
                    session,
                    graph=graph,
                    trusted_observation=observation,
                )
                if visible_advances:
                    graph = revised
                elif impact == "read_only":
                    text_revised = self._try_advance_visible_text_read_subgoal(
                        session,
                        graph=graph,
                        trusted_observation=observation,
                    )
                    if text_revised is not None:
                        self._store_revised_graph(session, text_revised)
                        revised = text_revised
                        graph = text_revised
                        visible_advances = 1
                if (
                    impact == "read_only"
                    and not visible_advances
                    and initial_safe is not None
                    and not self._is_presence_only_read_only_subgoal(
                        initial_safe
                    )
                ):
                    observed = self.bridge.observed_state(
                        graph=graph,
                        trusted_observation=observation,
                        action_outcome="not_applicable",
                        verification={
                            "visible_evidence": [scene.summary],
                            "blocked_reasons": [],
                        },
                    )
                    revised = self.deepseek_planner.replan(
                        graph,
                        observed,
                        trigger="observation_changed",
                        reason=(
                            "初始可信画面不能直接证明当前 read_only 结果。"
                            "如果目标页面或区域尚未出现，必须先修订为一个"
                            " navigation_only 中间状态；不得请求低层动作、"
                            "不得直接宣称结果完成。"
                        ),
                    )
                    self._validate_graph_identity(
                        revised,
                        device_id=session.device_id,
                        previous=graph,
                        trusted_observation=observation,
                    )
                    self._store_revised_graph(session, revised)
                    graph = revised
                    visible_advances = 1
                if not visible_advances and impact == "read_only":
                    session.status = "blocked"
                    session.failed_reason = (
                        "当前 read_only 子目标不是可由唯一完整可见元素证明的"
                        "定位目标，或当前画面证据不唯一；未请求物理动作。"
                    )
                    self._write_terminal_snapshot(session)
                    return session
                if visible_advances:
                    if revised.status == "completed":
                        session.status = "succeeded"
                        session.failed_reason = ""
                        self._write_terminal_snapshot(session)
                        return session
                    current = revised.active_subgoal()
                    impact = (
                        current.external_impact if current is not None else "unknown"
                    )
                    if current is None:
                        session.status = "blocked"
                        session.failed_reason = "可见状态证据推进后没有活动子目标。"
                        self._write_terminal_snapshot(session)
                        return session
                    if _requires_effect_confirmation(revised, current):
                        session.status = "awaiting_effect_confirmation"
                        session.failed_reason = ""
                        self._bind_effect_confirmation(session)
                        self._write_terminal_snapshot(session)
                        return session
                    session.status = "needs_reobservation"
                    session.failed_reason = (
                        "可见状态证据已切换活动子目标；必须按新子目标重新观察，"
                        "不得复用旧目标条件下的候选清单。"
                    )
                    session.controller_decision = None
                    session.confirmation_authority = None
                    self._write_terminal_snapshot(session)
                    return session

            task_context_payload = graph.to_qwen_context()
            task_context = QwenTaskContext.from_dict(task_context_payload)
            try:
                semantic_authority = compile_formal_semantic_authority(graph)
                semantic_ir = semantic_authority.semantic_ir
            except TaskSemanticIRError as exc:
                session.status = "blocked"
                session.failed_reason = f"正式 TaskSemanticIR authority 拒绝：{exc}"
                self._write_terminal_snapshot(session)
                return session
            task_context = replace(task_context, semantic_ir=semantic_ir)
            task_context.validate()
            session.effect_previews = tuple(
                {
                    **preview.to_dict(),
                    "preview_digest": preview.preview_digest,
                }
                for preview in semantic_authority.effect_previews
            )
            active_typed_subgoal = next(
                (
                    item
                    for item in semantic_ir.subgoals
                    if item.subgoal_id
                    == str(task_context.current_subgoal.get("subgoal_id") or "")
                ),
                None,
            )
            constraints = {
                item.constraint_id: item for item in semantic_ir.constraints
            }
            required_actions = tuple(
                dict.fromkeys(
                    str(constraints[constraint_ref].value)
                    for constraint_ref in (
                        active_typed_subgoal.constraint_refs
                        if active_typed_subgoal is not None
                        else ()
                    )
                    if constraints[constraint_ref].kind == "required_action"
                )
            )
            available_actions = self._available_action_kinds(session)
            unsupported_actions = tuple(
                action for action in required_actions if action not in available_actions
            )
            if unsupported_actions:
                capability_provider = getattr(
                    session.adapter,
                    "capability_snapshot",
                    None,
                )
                snapshot = (
                    capability_provider()
                    if callable(capability_provider)
                    else build_device_capability_snapshot(
                        device_id=session.device_id,
                        supported_actions=available_actions,
                    )
                )
                gap = snapshot.gap(unsupported_actions[0])
                session.capability_gap = gap.to_dict() if gap is not None else None
                session.status = "blocked"
                session.failed_reason = (
                    "当前设备能力不支持 typed required_action："
                    + unsupported_actions[0]
                )
                self._write_terminal_snapshot(session)
                return session
            decision = self._decide_next_action(
                session,
                frames=frames,
                task_context=task_context,
                trusted_observation=observation,
            )
            self._validate_decision_binding(graph, observation, decision)
            session.qwen_decision = decision
            self._remember(
                session,
                store.write_qwen_decision(session.step_number, decision),
            )

            proposal = decision.proposal
            if proposal.status == "action":
                selection_receipt = self._selection_receipt(session, decision)
                session.controller_decision = selection_receipt
                self._remember(
                    session,
                    store.write_controller_decision(
                        session.step_number,
                        self._selection_receipt_payload(selection_receipt),
                    ),
                )
                if selection_receipt.allowed:
                    session.status = "awaiting_confirmation"
                    self._bind_confirmation(session)
                else:
                    session.status = "blocked"
                    session.failed_reason = selection_receipt.reason
            elif proposal.status == "blocked":
                session.status = "blocked"
                session.failed_reason = proposal.reason
                session.controller_decision = CanonicalSelectionReceipt(
                    allowed=False,
                    reason=proposal.reason,
                )
            else:
                raise UniversalAgentOrchestratorError(
                    f"不支持的 Qwen 状态：{proposal.status}"
                )

            if session.physical_actions != 0:
                raise UniversalAgentOrchestratorError(
                    "start 路径错误地触发了物理动作。"
                )
            self._write_terminal_snapshot(session)
            return session
        except Exception as exc:
            session.status = "failed"
            session.failed_reason = str(exc)
            self._record_deepseek_failure(
                session,
                exc,
                stage=(
                    "initial_task_graph"
                    if session.task_graph is None
                    else "start_replan"
                ),
            )
            try:
                self._write_terminal_snapshot(session)
            except Exception:
                pass
            raise

    def pause(self, session: UniversalAgentSessionState) -> None:
        try:
            with self.device_registry.device_lock(session.device_id):
                for authority in (
                    session.confirmation_authority,
                    session.effect_confirmation_authority,
                ):
                    if authority is not None:
                        authority.consumed = True
                        authority.invalid_reason = "paused"
                session.confirmed_effect_ids = ()
                session.status = "paused"
                session.failed_reason = "用户已暂停；旧确认和旧观察不可复用。"
                self._write_terminal_snapshot(session)
        finally:
            self.device_registry.release(session.device_id, session.session_id)

    def invalidate_confirmation(
        self,
        session: UniversalAgentSessionState,
        *,
        reason: str,
    ) -> None:
        """Invalidate a pending scope while retaining the device for re-observation."""

        if self.device_registry.active_session(session.device_id) != session.session_id:
            return
        with self.device_registry.device_lock(session.device_id):
            for authority in (
                session.confirmation_authority,
                session.effect_confirmation_authority,
            ):
                if authority is not None:
                    authority.consumed = True
                    authority.invalid_reason = str(reason or "invalidated")
            session.confirmation_authority = None
            session.effect_confirmation_authority = None
            session.confirmed_effect_ids = ()
            session.status = "needs_reobservation"
            session.failed_reason = (
                "设备或摄像头状态变化；旧确认已失效，必须重新观察后再确认。"
            )
            self._write_terminal_snapshot(session)

    def cancel(self, session: UniversalAgentSessionState) -> None:
        try:
            with self.device_registry.device_lock(session.device_id):
                for authority in (
                    session.confirmation_authority,
                    session.effect_confirmation_authority,
                ):
                    if authority is not None:
                        authority.consumed = True
                        authority.invalid_reason = "cancelled"
                session.confirmed_effect_ids = ()
                session.status = "cancelled"
                session.failed_reason = "用户已取消任务。"
                self._write_terminal_snapshot(session)
        finally:
            self.device_registry.release(session.device_id, session.session_id)
