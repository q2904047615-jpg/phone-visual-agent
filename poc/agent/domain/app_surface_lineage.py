"""Receipt-scoped App surface identity carried across fresh observations."""

from __future__ import annotations

from .validation import DataclassWire, canonical_digest, reject_if
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from .generic_goal import VisibleGoalEvidence
from .task_graph import ControllerTransitionEvidenceRef, DynamicTaskGraph, TaskGraphError, VerifiedActionTransition
from .task_semantic_ir import compile_formal_semantic_authority
from .ui_scene import MIN_TARGET_CONFIDENCE, scene_matches_app_identity


POST_ACTION_OUTCOMES = frozenset({"matched", "mismatched"})


class AppSurfaceLineageError(RuntimeError):
    """Raised when a graph claims an App surface without typed evidence."""


def _action_digest(action: Any) -> str:
    reject_if(action is None, AppSurfaceLineageError("动作摘要缺少语义动作。"))
    payload = action.to_dict() if callable(getattr(action, "to_dict", None)) else action
    return canonical_digest(payload)


def _is_descendant(graph: DynamicTaskGraph, subgoal: Any, ancestor_id: str) -> bool:
    by_id = {item.subgoal_id: item for item in graph.subgoals}
    pending = list(getattr(subgoal, "depends_on", ()) or ())
    visited: set[str] = set()
    while pending:
        dependency_id = pending.pop()
        if dependency_id == ancestor_id:
            return True
        if dependency_id in visited:
            continue
        visited.add(dependency_id)
        dependency = by_id.get(dependency_id)
        if dependency is not None:
            pending.extend(dependency.depends_on)
    return False


def _newly_completed(previous: DynamicTaskGraph, revised: DynamicTaskGraph, *, existing_only: bool=False) -> tuple[Any,
    ...]:
    old_by_id = {item.subgoal_id: item for item in previous.subgoals}
    return tuple((item for item in revised.subgoals if item.status == 'completed' and (not existing_only
        or item.subgoal_id in old_by_id) and (old_by_id.get(item.subgoal_id) is None
        or old_by_id[item.subgoal_id].status != 'completed')))


def _find_subgoal(graph: DynamicTaskGraph, subgoal_id: str) -> Any | None:
    return next((item for item in graph.subgoals if item.subgoal_id == subgoal_id), None)


@dataclass(frozen=True)
class VerifiedAppSurfaceLineage(DataclassWire):
    session_id: str
    task_id: str
    device_id: str
    app_id: str
    app_name: str
    surface_id: str
    source_receipt_id: str
    source_subgoal_id: str
    functional_foreground_app_id: str
    physical_actions: int

    def matches_foreground(self, foreground_app_id: str) -> bool:
        foreground = str(foreground_app_id or "").strip().casefold()
        return bool(foreground and foreground != 'launcher') and foreground in {self.functional_foreground_app_id.strip(
            ).casefold(), self.app_id.strip().casefold(), self.app_name.strip().casefold()}

    def is_scoped_to(self, *, session: Any, graph: DynamicTaskGraph, physical_actions: int) -> bool:
        return self.session_id == session.session_id and self.task_id == graph.task_id and (self.device_id ==
            session.device_id == graph.device_id) and (self.physical_actions == physical_actions)


class AppSurfaceLineageAuthority:
    """Mint, validate and carry the single typed App-surface alias."""

    @staticmethod
    def transition_proves(*, previous: DynamicTaskGraph, completed_subgoal: Any, target_apps: tuple[Any, ...],
        trusted_observation: Any, session_id: str, verified_transition: VerifiedActionTransition | None,
        controller_transition_evidence_refs: tuple[ControllerTransitionEvidenceRef, ...],
        before_observation: Any | None, previous_decision: Any | None, execution_result: Any | None) -> bool:
        receipt = verified_transition
        before_scene = getattr(before_observation, "scene", None)
        after_scene = getattr(trusted_observation, "scene", None)
        proposal = getattr(previous_decision, "proposal", None)
        action = getattr(proposal, "action", None)
        action_kind = str(getattr(action, 'action', ''))
        if (any((value is None for value in (receipt, before_scene, after_scene, action,
            execution_result))) or not session_id or str(getattr(completed_subgoal, 'external_impact',
            '')) != 'navigation_only' or action_kind not in {'tap_semantic', 'launch_app'}):
            return False
        try:
            receipt.validate()
            for ref in controller_transition_evidence_refs:
                ref.validate()
        except TaskGraphError:
            return False
        active = previous.active_subgoal()
        if active is None or active.subgoal_id != completed_subgoal.subgoal_id:
            return False
        actual_scope = (receipt.session_id, receipt.task_id, receipt.device_id, receipt.prior_revision,
            receipt.subgoal_id, receipt.decision_node_id, receipt.action_kind, receipt.action_digest,
            receipt.rebound_action_digest, receipt.resolved_action_digest, receipt.outcome, receipt.physical_actions,
            receipt.before_observation_id, receipt.before_fingerprint, receipt.after_observation_id,
            receipt.after_fingerprint)
        expected_scope = (session_id, previous.task_id, previous.device_id, previous.revision,
            completed_subgoal.subgoal_id, str(getattr(action, 'node_id', '')), action_kind, _action_digest(action),
            _action_digest(getattr(execution_result, 'rebound_action', None)), _action_digest(getattr(execution_result,
            'resolved_action', None)), 'matched', 1, str(getattr(before_observation, 'observation_id', '')),
            str(getattr(before_observation, 'fingerprint', '')), str(getattr(trusted_observation, 'observation_id',
            '')), str(getattr(trusted_observation, 'fingerprint', '')))
        if (actual_scope != expected_scope or receipt.errors or receipt.before_fingerprint == receipt.after_fingerprint
            or str(getattr(after_scene, 'foreground_app_id', '')).casefold() == 'launcher'):
            return False

        params = getattr(action, "params", None)
        if not isinstance(params, Mapping):
            return False
        if action_kind == 'launch_app':
            expected_app_id = str(params.get('expected_app_id') or '').strip().casefold()
            observed_app_id = str(getattr(after_scene, 'foreground_app_id', '') or '').strip().casefold()
            if (not expected_app_id or not str(params.get('launch_ref') or '').strip()
                or (observed_app_id != expected_app_id and not scene_matches_app_identity(after_scene,
                str(params.get('target_app_id') or ''), str(params.get('target_app_name') or '')))):
                return False
            bound_targets = tuple((app for app in target_apps if str(app.app_id).casefold() == str(params.get(
                'target_app_id') or '').casefold() and str(app.app_name).casefold() == str(params.get(
                'target_app_name') or '').casefold()))
        else:
            if str(getattr(before_scene, 'foreground_app_id', '')).casefold() != 'launcher':
                return False
            element_id = str(params.get("element_id") or "").strip()
            matches = tuple((element for element in tuple(getattr(before_scene, 'elements',
                ()) or ()) if str(getattr(element, 'element_id', '')) == element_id))
            if len(matches) != 1:
                return False
            element = matches[0]
            action_identity = tuple((str(value or '').strip() for value in (params.get('label'), params.get('role'),
                params.get('target') or params.get('meaning'))))
            element_identity = tuple((str(value or '').strip() for value in (getattr(element, 'label', ''), getattr(
                element, 'role', ''), getattr(element, 'meaning', ''))))
            if action_identity != element_identity:
                return False
            action_terms = VisibleGoalEvidence.binding_terms(params.get('label'), params.get('target'),
                params.get('meaning'), getattr(element, 'label', ''), getattr(element, 'meaning', ''))
            bound_targets = tuple((app for app in target_apps if VisibleGoalEvidence.target_app_terms(app.app_id,
                app.app_name).intersection(action_terms)))
        if len(bound_targets) != 1:
            return False
        try:
            semantic_ir = compile_formal_semantic_authority(previous).semantic_ir
        except Exception:
            return False
        target = bound_targets[0]
        surface_ids = {surface.surface_id for surface in semantic_ir.surfaces if surface.kind == 'app'
            and (surface.app_id.casefold() == str(target.app_id).casefold()
            or surface.app_name.casefold() == str(target.app_name).casefold())}
        if action_kind == 'launch_app' and params.get('target_surface_id') not in surface_ids:
            return False
        formal_transition = params.get("formal_transition")
        expectations = formal_transition.get("expectations") if isinstance(formal_transition, Mapping) else None
        if not isinstance(expectations, list) or formal_transition.get('exploratory') is not False:
            return False
        matching = [expectation for expectation in expectations if isinstance(expectation,
            Mapping) and expectation.get('subject_ref') == 'surface_current'
            and (expectation.get('predicate') == 'surface.active_ref') and (expectation.get('operator') == 'equals')
            and (expectation.get('value') in surface_ids)]
        return bool(surface_ids) and len(matching) == 1

    @staticmethod
    def lineage_proves(*, previous: DynamicTaskGraph, completed_subgoal: Any, target_apps: tuple[Any, ...],
        trusted_observation: Any, session_id: str, verified_app_surface_lineage: VerifiedAppSurfaceLineage | None,
        physical_actions: int) -> bool:
        lineage = verified_app_surface_lineage
        scene = getattr(trusted_observation, "scene", None)
        if lineage is None or scene is None:
            return False
        if (not session_id or lineage.session_id != session_id or lineage.task_id != previous.task_id
            or (lineage.device_id != previous.device_id) or (lineage.physical_actions != physical_actions)
            or (str(getattr(scene, 'foreground_app_id', '')).casefold() !=
            lineage.functional_foreground_app_id.casefold())
            or (lineage.functional_foreground_app_id.casefold() == 'launcher')
            or (not any((str(app.app_id).casefold() == lineage.app_id.casefold()
            and str(app.app_name).casefold() == lineage.app_name.casefold() for app in target_apps)))):
            return False
        source = _find_subgoal(previous, lineage.source_subgoal_id)
        return bool(source and source.status == 'completed' and _is_descendant(previous, completed_subgoal,
            lineage.source_subgoal_id))

    @classmethod
    def validate_newly_completed(cls, *, previous: DynamicTaskGraph, revised: DynamicTaskGraph,
        trusted_observation: Any, session_id: str='', verified_transition: VerifiedActionTransition | None=None,
        controller_transition_evidence_refs: tuple[ControllerTransitionEvidenceRef, ...]=(),
        before_observation: Any | None=None, previous_decision: Any | None=None, execution_result: Any | None=None,
        verified_app_surface_lineage: VerifiedAppSurfaceLineage | None=None, physical_actions: int=0) -> None:
        scene = getattr(trusted_observation, "scene", None)
        reject_if(scene is None, AppSurfaceLineageError("DeepSeek revision 缺少可复核的可信场景。"))
        for item in _newly_completed(previous, revised):
            text = " ".join((item.objective, *tuple(item.completion_conditions or ())))
            target_apps = VisibleGoalEvidence.referenced_target_apps(previous, text, item.subgoal_id)
            reject_if(target_apps and (not VisibleGoalEvidence.foreground_matches(scene, target_apps)) and (not cls.transition_proves(previous=previous, completed_subgoal=item, target_apps=target_apps, trusted_observation=trusted_observation, session_id=session_id, verified_transition=verified_transition, controller_transition_evidence_refs=controller_transition_evidence_refs, before_observation=before_observation, previous_decision=previous_decision, execution_result=execution_result)) and (not cls.lineage_proves(previous=previous, completed_subgoal=item, target_apps=target_apps, trusted_observation=trusted_observation, session_id=session_id, verified_app_surface_lineage=verified_app_surface_lineage, physical_actions=physical_actions)), AppSurfaceLineageError(f'Launcher 或其他页面中的 App 入口不能证明目标 App 页面已在前台：subgoal_id={item.subgoal_id}。'))

    @classmethod
    def build(cls, *, session: Any, previous: DynamicTaskGraph, revised: DynamicTaskGraph, trusted_observation: Any,
        receipt: VerifiedActionTransition | None, controller_refs: tuple[ControllerTransitionEvidenceRef, ...],
        before_observation: Any, previous_decision: Any, execution_result: Any) -> VerifiedAppSurfaceLineage | None:
        for item in _newly_completed(previous, revised, existing_only=True):
            text = " ".join((item.objective, *item.completion_conditions))
            target_apps = VisibleGoalEvidence.referenced_target_apps(previous, text, item.subgoal_id)
            if (not target_apps or not cls.transition_proves(previous=previous, completed_subgoal=item,
                target_apps=target_apps, trusted_observation=trusted_observation, session_id=session.session_id,
                verified_transition=receipt, controller_transition_evidence_refs=controller_refs,
                before_observation=before_observation, previous_decision=previous_decision,
                execution_result=execution_result)):
                continue
            action = previous_decision.proposal.action
            if action.action == 'launch_app':
                bound = [app for app in target_apps if str(app.app_id).casefold() == str(action.params.get(
                    'target_app_id') or '').casefold() and str(app.app_name).casefold() == str(action.params.get(
                    'target_app_name') or '').casefold()]
            else:
                terms = VisibleGoalEvidence.binding_terms(action.params.get("label"), action.params.get("target"))
                bound = [app for app in target_apps if VisibleGoalEvidence.target_app_terms(app.app_id,
                    app.app_name).intersection(terms)]
            expectations = action.params["formal_transition"]["expectations"]
            surface_id = next((str(expectation['value']) for expectation
                in expectations if expectation.get('predicate') == 'surface.active_ref'
                and expectation.get('operator') == 'equals'))
            if len(bound) != 1 or receipt is None:
                return None
            return VerifiedAppSurfaceLineage(session_id=session.session_id, task_id=previous.task_id,
                device_id=previous.device_id, app_id=str(bound[0].app_id), app_name=str(bound[0].app_name),
                surface_id=surface_id, source_receipt_id=receipt.receipt_id, source_subgoal_id=receipt.subgoal_id,
                functional_foreground_app_id=str(trusted_observation.scene.foreground_app_id),
                physical_actions=session.physical_actions)
        return None

    @staticmethod
    def carry(*, session: Any, previous: DynamicTaskGraph, revised: DynamicTaskGraph, trusted_observation: Any,
        prior_lineage: VerifiedAppSurfaceLineage | None, prior_physical_actions: int,
        execution_result: Any) -> VerifiedAppSurfaceLineage | None:
        lineage = prior_lineage
        scene = getattr(trusted_observation, "scene", None)
        physical_delta = int(getattr(execution_result, "physical_actions", 0))
        resolved_kind = str(getattr(getattr(execution_result, "resolved_action", None), "kind", ""))
        valid_delta = physical_delta == 1 or (physical_delta == 0 and resolved_kind == "wait_for_change")
        if (lineage is None or scene is None or str(getattr(execution_result, 'action_outcome',
            '')) not in POST_ACTION_OUTCOMES or (not valid_delta) or (not lineage.is_scoped_to(session=session,
            graph=previous, physical_actions=prior_physical_actions)) or (lineage.task_id != revised.task_id)
            or (lineage.device_id != revised.device_id) or (session.physical_actions != prior_physical_actions +
            physical_delta) or (not lineage.matches_foreground(str(getattr(scene, 'foreground_app_id', ''))))):
            return None
        source = _find_subgoal(revised, lineage.source_subgoal_id)
        if source is None or source.status != 'completed':
            return None
        current = revised.active_subgoal()
        if current is None:
            return replace(lineage,
                physical_actions=session.physical_actions) if revised.status == 'completed' else None
        if _is_descendant(revised, current, lineage.source_subgoal_id):
            return replace(lineage, physical_actions=session.physical_actions)
        return None

    @staticmethod
    def refresh(*, session: Any, graph: DynamicTaskGraph, prior_observation: Any,
        new_observation: Any) -> VerifiedAppSurfaceLineage | None:
        lineage = session.verified_app_surface_lineage
        prior_scene = getattr(prior_observation, "scene", None)
        new_scene = getattr(new_observation, "scene", None)
        if (lineage is None or prior_scene is None or new_scene is None or (not lineage.is_scoped_to(session=session,
            graph=graph, physical_actions=session.physical_actions))
            or (not lineage.matches_foreground(str(getattr(prior_scene, 'foreground_app_id',
            '')))) or (str(getattr(new_scene, 'foreground_app_id',
            '')).casefold() == 'launcher') or (not bool(getattr(prior_scene, 'stable',
            False))) or (not bool(getattr(new_scene, 'stable', False)))):
            return None
        if lineage.matches_foreground(str(getattr(new_scene, 'foreground_app_id', ''))):
            return lineage

        def page_titles(scene: Any) -> frozenset[str]:
            return frozenset((str(getattr(element, 'label', '') or '').strip().casefold() for element
                in tuple(getattr(scene, 'elements', ()) or ()) if str(getattr(element, 'label',
                '') or '').strip() and (str(getattr(element, 'meaning',
                '') or '').casefold() == 'page_title' or str(getattr(element, 'meaning',
                '') or '').casefold().endswith(('_page_title', '_screen_title'))) and ((getattr(element, 'states',
                {}) or {}).get('fully_visible') is not False) and (float(getattr(element, 'confidence',
                0.0)) >= MIN_TARGET_CONFIDENCE)))

        if len(page_titles(prior_scene) & page_titles(new_scene)) != 1:
            return None
        source = _find_subgoal(graph, lineage.source_subgoal_id)
        current = graph.active_subgoal()
        if (source is None or source.status != 'completed' or current is None or (not _is_descendant(graph, current,
            lineage.source_subgoal_id))):
            return None
        return replace(lineage, functional_foreground_app_id=str(getattr(new_scene, "foreground_app_id", "")))
