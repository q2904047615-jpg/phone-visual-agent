"""Infrastructure adapter for one observed, verified device action."""

from __future__ import annotations

from agent.domain.validation import NormalizedPoint, canonical_digest, dataclass_wire, reject_if
import statistics
import time
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from agent.domain import DeviceActionRequest, DeviceExecutionError, DeviceExecutor
from agent.domain.confirmation_authority import ConfirmationAuthority
from agent.domain.qwen_task_context import execution_history_entry
from agent.domain.text_transport import text_digest
from agent.application.text_transport import TrustedTextTransportPort
from agent.infrastructure import RobotDeviceExecutor
from agent.domain.generic_goal import GenericIntentDraft
from agent.infrastructure.generic_scene_observer import SingleStepGenericSceneObserver
from agent.infrastructure.task_screenshot_history import save_task_frames
from agent.application.action_adapter import GenericActionAdapterError
from agent.domain.action_capabilities import build_device_capability_snapshot
from agent.domain.action_catalog import PHYSICAL_ACTION_KINDS
from agent.infrastructure.observation_images import (
    measure_frame_sharpness,
    measure_local_stability,
    measure_material_visual_transition,
)
from agent.infrastructure.orientation_safety import (
    OrientationCredential,
    OrientationSafetyError,
    _mint_single_step_scene_credential,
    frame_fingerprint,
    validate_device_id,
)
from agent.infrastructure.qwen_runtime_errors import classify_qwen_error
from agent.infrastructure.model_failure_diagnostics import model_failure_payload, persist_model_failure_payload
from agent.domain.semantic_action import SemanticAction
from agent.domain.ui_scene import UIScene
from agent.domain.universal_action_controller import (
    ResolvedSemanticAction,
    UniversalActionController,
    UniversalActionError,
)


QWEN_FAILURE_DIAGNOSTIC_VERSION = "2026-08-17-qwen-failure-diagnostic-v1"


def _persist_qwen_failure_diagnostic(*, evidence_dir: Path | None, prefix: str, raw_response: str, error: Exception,
    diagnostics: dict[str, Any] | None=None) -> Path | None:
    """Persist bounded redacted model output without changing failure policy."""

    raw = str(raw_response or "")
    if evidence_dir is None or not raw:
        return None
    details = diagnostics if isinstance(diagnostics, dict) else {}
    payload = model_failure_payload(
        artifact_version=QWEN_FAILURE_DIAGNOSTIC_VERSION,
        raw=raw,
        failed_stage=str(details.get('failed_stage') or 'unknown'),
        error_type=str(details.get('error_type') or classify_qwen_error(error, raw_response=raw)),
        error_message=str(error),
    )
    return persist_model_failure_payload(
        payload,
        evidence_dir=evidence_dir,
        prefix=prefix,
        default_prefix='observation',
        suffix='qwen_failure',
    )


def persist_observer_failure_diagnostic(observer: Any, *, evidence_dir: Path | None, prefix: str,
    error: Exception) -> tuple[str, ...]:
    diagnostics = getattr(observer, "last_diagnostics", {})
    raw_response = getattr(observer, "last_raw_response", "")
    try:
        path = _persist_qwen_failure_diagnostic(evidence_dir=evidence_dir, prefix=prefix, raw_response=raw_response,
            error=error, diagnostics=diagnostics if isinstance(diagnostics, dict) else None)
    except Exception as diagnostic_error:
        if isinstance(diagnostics, dict):
            diagnostics['diagnostic_persistence_error'] = type(diagnostic_error).__name__
        return ()
    return (str(path),) if path is not None else ()


@dataclass(frozen=True)
class GenericActionExecutionResult:
    requested_action: SemanticAction
    rebound_action: SemanticAction
    resolved_action: ResolvedSemanticAction
    before_scene: UIScene
    after_scene: UIScene
    planned_scene_fingerprint: str
    # Public diagnostic fields: pixel identity is no longer an execution gate.
    # Keep false/null rather than claim that geometry checks prove UI identity.
    confirmation_frame_identity_verified: bool
    confirmation_frame_delta: float | None
    physical_actions: int
    after_model_decision: dict[str, Any] = field(repr=False, compare=False)
    primary_input_confirmation_reused: bool = False
    # Transport/hard-check result only. Visual success belongs to Qwen's new frame.
    action_outcome: str = "executed"
    verification_errors: tuple[str, ...] = ()
    robot_result: Any = None
    hardware_receipt: dict[str, Any] | None = None
    execution_metadata: dict[str, Any] = field(default_factory=dict)
    evidence: tuple[str, ...] = ()
    after_frames: tuple[Image.Image, ...] = field(default_factory=tuple, repr=False, compare=False)
    after_frame_paths: tuple[str, ...] = ()
    observation_errors: tuple[str, ...] = ()
    controller_transition_evidence: tuple[str, ...] = ()
    before_frames: tuple[Image.Image, ...] = field(default_factory=tuple, repr=False, compare=False)
    before_frame_paths: tuple[str, ...] = ()
    orientation_credential: OrientationCredential | None = None

    def __post_init__(self) -> None:
        # Freeze sampling lists because a completed result is a live promotion source.
        object.__setattr__(self, "after_frames", tuple(self.after_frames))
        object.__setattr__(self, "before_frames", tuple(self.before_frames))
        reject_if(not isinstance(self.after_model_decision, Mapping),
            ValueError("动作结果缺少同一动作后截图的 Qwen decision。"))
        object.__setattr__(self, "after_model_decision", dict(self.after_model_decision))
        object.__setattr__(self, 'controller_transition_evidence', tuple(self.controller_transition_evidence))
        object.__setattr__(self, 'execution_metadata', dict(self.execution_metadata))

    def to_dict(self) -> dict[str, Any]:
        value = dataclass_wire(self, omit=('after_frames', 'before_frames', 'after_model_decision'))
        value.update(after_frame_count=len(self.after_frames), before_frame_count=len(self.before_frames),
            robot_result=self.robot_result, visual_outcome=self.after_model_decision.get('previous_action_outcome'))
        return value


class GenericSingleActionAdapter:
    """The only generic bridge from a verified scene to one robot action."""

    PHYSICAL_KINDS = PHYSICAL_ACTION_KINDS
    DEVICE_ACTION_KINDS = PHYSICAL_KINDS | {'launch_app'}

    @staticmethod
    def _post_action_goal(goal: GenericIntentDraft, *, authority: ConfirmationAuthority | None,
        requested: SemanticAction, resolved: ResolvedSemanticAction, physical_actions: int,
        execution_metadata: Mapping[str, Any] | None = None) -> GenericIntentDraft:
        """Project one scoped execution fact into the immediate post-action Qwen call."""

        if authority is None:
            return goal
        entities = dict(goal.entities)
        entities["history"] = [*entities.get("history", []), execution_history_entry(
            step=authority.revision, requested_action=requested.to_dict(), resolved_action=resolved.to_dict(),
            physical_actions=physical_actions, transport_outcome="executed",
            execution_metadata=execution_metadata)]
        return replace(goal, entities=entities)

    def _single_step_scene_orientation_credential(self, *, scene: UIScene,
        frames: list[Image.Image]) -> OrientationCredential:
        """Mint a local frame binding without consuming optional Qwen metadata."""

        reject_if(not frames, OrientationSafetyError("单步方向绑定缺少当前稳定帧。"))
        return _mint_single_step_scene_credential(device_id=self.device_id, scene_fingerprint=scene.fingerprint,
            frame=frames[-1].convert('RGB'))

    def supported_action_kinds(self) -> frozenset[str]:
        """Return only actions backed by callable methods on this device."""

        capability_provider = getattr(self.robot, "hardware_capabilities", None)
        declared = capability_provider() if callable(capability_provider) else {}
        if not isinstance(declared, dict):
            declared = {}
        methods = {'tap_semantic': 'vision_tap_relative', 'dismiss_overlay': 'vision_dismiss_overlay_relative',
            'double_tap': 'vision_double_tap_relative', 'reveal_system_navigation': 'vision_reveal_system_navigation',
            'back': 'vision_android_back', 'home': 'vision_android_home',
            'open_recent_apps': 'vision_android_recent_apps',
            'long_press': 'vision_long_press_relative', 'drag': 'vision_drag_relative'}
        supported = {'wait_for_change'} if bool(declared.get('wait_for_change', True)) else set()
        supported.update((action for action, method in methods.items() if bool(declared.get(action,
            True)) and callable(getattr(self.robot, method, None))))
        if (bool(declared.get('swipe', True)) and any((callable(getattr(self.robot, f'vision_swipe_{direction}',
            None)) for direction in ('up', 'down', 'left', 'right')))):
            supported.update({"scroll", "swipe_element"})
        if self.text_transport is not None:
            profile = self.text_transport.profile
            profile.validate()
            if profile.enabled and 'append_text' in profile.capabilities:
                supported.update({'input_verified_text', 'press_enter'})
            else:
                supported.difference_update({'input_verified_text', 'press_enter'})
            if profile.enabled and 'clear_text' in profile.capabilities:
                supported.add('clear_verified_text')
            else:
                supported.discard('clear_verified_text')
        if self.app_launcher is not None and bool(getattr(self.app_launcher, 'enabled', False)):
            supported.add('launch_app')
        return frozenset(supported)

    def text_transport_profile(self) -> Any | None:
        return self.text_transport.profile if self.text_transport is not None else None

    def supported_app_aliases(self) -> tuple[str, ...]:
        provider = getattr(self.app_launcher, "aliases", None)
        return tuple(provider()) if callable(provider) else ()

    def resolve_app_launch_target(self, app_id: str, app_name: str) -> Any | None:
        resolver = getattr(self.app_launcher, 'resolve', None)
        return resolver(app_id, app_name) if callable(resolver) else None

    def capability_snapshot(self) -> Any:
        provider = getattr(self.robot, "hardware_capability_profile", None)
        raw_profile = provider() if callable(provider) else None
        return build_device_capability_snapshot(device_id=str(getattr(self.robot, 'device_id', '') or 'unknown-device'),
            supported_actions=self.supported_action_kinds(), raw_profile=raw_profile)

    def __init__(self, *, capture: Callable[[], Image.Image], observer: SingleStepGenericSceneObserver, robot: Any,
        device_executor: DeviceExecutor | None=None, app_launcher: Any=None,
        text_transport: TrustedTextTransportPort | None=None,
        controller: UniversalActionController | None=None,
        frame_interval: float=0.37, post_action_settle: float=1.5, post_action_timeout: float | None=None,
        post_action_continuous_timeout: float | None=None,
        post_action_min_relative_sharpness: float=0.8, post_action_min_reference_sharpness: float=2.0,
        device_id: str) -> None:
        self.capture = capture
        self.observer = observer
        self.robot = robot
        self.app_launcher = app_launcher
        self.text_transport = text_transport
        if text_transport is not None:
            text_transport.profile.validate()
            reject_if(text_transport.profile.device_id != device_id,
                ValueError('ADB Keyboard profile 与 adapter device_id 不一致。'))
        self.device_executor = device_executor or RobotDeviceExecutor(robot, app_launcher=app_launcher,
            text_transport=text_transport)
        self.controller = controller or UniversalActionController()
        self.frame_interval = max(0.0, float(frame_interval))
        self.post_action_settle = max(0.0, float(post_action_settle))
        self.post_action_timeout = max(0.0, 10.0 if post_action_timeout is None else float(post_action_timeout))
        self.post_action_continuous_timeout = max(self.post_action_timeout,
            45.0 if post_action_continuous_timeout is None and post_action_timeout
            is None else self.post_action_timeout if post_action_continuous_timeout
            is None else float(post_action_continuous_timeout))
        self.post_action_min_relative_sharpness = max(0.0, min(1.0, float(post_action_min_relative_sharpness)))
        self.post_action_min_reference_sharpness = max(0.0, float(post_action_min_reference_sharpness))
        try:
            self.device_id = validate_device_id(device_id)
        except OrientationSafetyError as exc:
            raise ValueError(str(exc)) from exc

    @staticmethod
    def _validate_confirmation_frame_dimensions(planned_frames: tuple[Image.Image, ...] | list[Image.Image],
        fresh_frames: list[Image.Image]) -> None:
        """Preserve the coordinate canvas, not pixel equality across dynamic UI."""
        reject_if(not planned_frames or not fresh_frames, GenericActionAdapterError("确认前缺少本地真实帧，不能验证画面尺寸。"))
        planned_sizes = {frame.size for frame in planned_frames}
        fresh_sizes = {frame.size for frame in fresh_frames}
        reject_if(len(planned_sizes) != 1 or len(fresh_sizes) != 1 or planned_sizes != fresh_sizes, GenericActionAdapterError("确认前真实画面尺寸发生变化。"))

    def _capture_frame(self) -> Image.Image:
        frame = self.capture().convert("RGB")
        reject_if(frame.width < 400 or frame.height < 700, GenericActionAdapterError("摄像头返回残缺画面，停止单步动作。"))
        return frame

    def _capture_frame_burst(self) -> list[Image.Image]:
        frames: list[Image.Image] = []
        for index in range(4):
            frames.append(self._capture_frame())
            if index < 3 and self.frame_interval:
                time.sleep(self.frame_interval)
        return frames

    def _capture_confirmation_frames(self, *, evidence_dir: Path | None, prefix: str) -> tuple[list[Image.Image],
        tuple[str, ...]]:
        """Capture four fresh local frames without re-interpreting the scene."""

        frames = self._capture_frame_burst()
        measure_local_stability(frames)  # Dimension validation; pixel variation is diagnostic only.
        paths = self._save_frames(frames, evidence_dir, prefix)
        return frames, paths

    def _capture_scene_once(self, goal: GenericIntentDraft, *, evidence_dir: Path | None=None,
        prefix: str, available_action_kinds: frozenset[str] | None=None
        ) -> tuple[UIScene, list[Image.Image], tuple[str, ...], dict[str, Any]]:
        frames = self._capture_frame_burst()
        paths = self._save_frames(frames, evidence_dir, prefix)
        try:
            scene, model_decision = self._observe_scene(frames, goal.to_dict(),
                available_action_kinds=available_action_kinds,
                response_evidence_dir=evidence_dir, response_evidence_prefix=prefix, current_frame_paths=paths)
            if getattr(self.observer, 'last_response_evidence_path', None):
                paths += (self.observer.last_response_evidence_path,)
        except RuntimeError as exc:
            if getattr(self.observer, 'last_response_evidence_path', None):
                paths += (self.observer.last_response_evidence_path,)
            diagnostic_paths = persist_observer_failure_diagnostic(self.observer, evidence_dir=evidence_dir,
                prefix=prefix, error=exc)
            raise GenericActionAdapterError(f'通用页面观察失败：{exc}', evidence=paths + diagnostic_paths) from exc
        return scene, frames, paths, model_decision

    def _observe_scene(self, frames: list[Image.Image] | tuple[Image.Image, ...], goal_context: dict[str, Any], *,
        response_evidence_dir: Path | None=None, response_evidence_prefix: str='observation',
        current_frame_paths: tuple[str, ...]=(),
        available_action_kinds: frozenset[str] | None=None
        ) -> tuple[UIScene, dict[str, Any]]:
        kwargs: dict[str, Any] = {'frames': list(frames), 'goal_context': goal_context}
        if getattr(self.observer, 'supports_response_evidence', False):
            kwargs.update(device_id=self.device_id, response_evidence_dir=response_evidence_dir,
                response_evidence_prefix=response_evidence_prefix)
        if getattr(self.observer, 'supports_task_screenshots', False) is True:
            kwargs['current_frame_paths'] = current_frame_paths
        if getattr(self.observer, 'supports_runtime_action_contract', False) is True:
            supported = self.supported_action_kinds()
            scoped = supported if available_action_kinds is None else frozenset(available_action_kinds)
            reject_if(not scoped or scoped - supported,
                GenericActionAdapterError('当前观察动作集合为空或超出设备能力。'))
            kwargs['available_action_kinds'] = scoped
        observe_with_decision = getattr(self.observer, "observe_with_decision", None)
        reject_if(not callable(observe_with_decision),
            GenericActionAdapterError("当前观察器不支持同一截图响应中的 scene + decision 合同。"))
        scene, model_decision = observe_with_decision(**kwargs)
        reject_if(not isinstance(model_decision, Mapping),
            GenericActionAdapterError("当前观察缺少同一截图响应中的 Qwen decision。"))
        return scene, dict(model_decision)

    def capture_scene(
        self,
        goal: GenericIntentDraft,
        *,
        evidence_dir: Path | None,
        prefix: str,
        available_action_kinds: frozenset[str] | None=None,
    ) -> tuple[UIScene, list[Image.Image], tuple[str, ...], dict[str, Any]]:
        # One step permits one Qwen request; stability sampling never triggers model resampling.
        try:
            return self._capture_scene_once(goal, evidence_dir=evidence_dir, prefix=f'{prefix}_attempt_1',
                available_action_kinds=available_action_kinds)
        except GenericActionAdapterError as exc:
            error = f"第1轮动作前观察失败：{exc}"
            raise GenericActionAdapterError('动作前通用页面观察失败：' + error, evidence=tuple(exc.evidence),
                observation_errors=(error,)) from exc

    @staticmethod
    def _requires_post_action_relative_clarity(resolved: ResolvedSemanticAction) -> bool:
        """Use relative clarity only when an action preserves a comparable input surface."""

        return resolved.kind in {'input_verified_text', 'press_enter', 'clear_verified_text'}

    @staticmethod
    def _requires_extended_post_action_timeout(resolved: ResolvedSemanticAction) -> bool:
        return resolved.kind in {'clear_verified_text', 'double_tap', 'drag', 'input_verified_text', 'long_press'}

    def _post_action_timeout_for(self, resolved: ResolvedSemanticAction) -> float:
        if self._requires_extended_post_action_timeout(resolved):
            return self.post_action_continuous_timeout
        return self.post_action_timeout

    def _capture_stable_post_action_frames(self, *, deadline: float, evidence_dir: Path | None, prefix: str,
        clarity_reference_frames: tuple[Image.Image, ...]=(), require_relative_clarity: bool=False
        ) -> tuple[list[Image.Image], tuple[str, ...]]:
        """Collect current frames; pixel changes do not delay or veto observation."""
        frames: list[Image.Image] = []
        reference_sharpness = statistics.median(measure_frame_sharpness(frame) for frame
            in clarity_reference_frames) if require_relative_clarity and clarity_reference_frames else None
        clarity_is_comparable = bool(reference_sharpness is not None
            and reference_sharpness >= self.post_action_min_reference_sharpness)
        last_candidate_sharpness = None
        last_relative_sharpness = None
        while True:
            frames.append(self._capture_frame())
            if len(frames) > 4:
                frames.pop(0)
            if len(frames) == 4:
                measure_local_stability(frames)  # Only invalid coordinate dimensions may fail.
                if clarity_is_comparable:
                    last_candidate_sharpness = statistics.median(measure_frame_sharpness(frame) for frame in frames)
                    last_relative_sharpness = last_candidate_sharpness / reference_sharpness
                if last_relative_sharpness is None or last_relative_sharpness >= self.post_action_min_relative_sharpness:
                    return list(frames), self._save_frames(frames, evidence_dir, prefix)
            if time.monotonic() >= deadline:
                paths = self._save_frames(frames, evidence_dir, f"{prefix}_timeout")
                if last_relative_sharpness is not None:
                    raise GenericActionAdapterError(
                        "动作后画面在限定时间内仍不够清晰："
                        f"参考清晰度{reference_sharpness:.3f}，候选清晰度{last_candidate_sharpness:.3f}，"
                        f"相对值{last_relative_sharpness:.3f}，要求至少{self.post_action_min_relative_sharpness:.3f}",
                        evidence=paths)
                raise GenericActionAdapterError("动作后未能采集到连续4帧。", evidence=paths)
            if self.frame_interval:
                time.sleep(min(self.frame_interval, max(0.0, deadline - time.monotonic())))

    def _observe_stable_post_action_scene(self, goal: GenericIntentDraft, *, before: UIScene,
        before_frames: tuple[Image.Image, ...], resolved: ResolvedSemanticAction,
        evidence_dir: Path | None, evidence_prefix: str,
        available_action_kinds: frozenset[str] | None=None) -> tuple[UIScene, tuple[Image.Image, ...], tuple[str, ...], tuple[str, ...], tuple[str,
        ...], tuple[str, ...], dict[str, Any]]:
        action_timeout = self._post_action_timeout_for(resolved)
        if self.post_action_settle:
            time.sleep(min(self.post_action_settle, action_timeout))

        prefix = f"{evidence_prefix}_after_attempt_1"
        try:
            frames, paths = self._capture_stable_post_action_frames(deadline=time.monotonic() + action_timeout,
                evidence_dir=evidence_dir, prefix=prefix, clarity_reference_frames=before_frames,
                require_relative_clarity=self._requires_post_action_relative_clarity(resolved))
        except GenericActionAdapterError as exc:
            raise GenericActionAdapterError(f'动作后画面采集失败：{exc}', evidence=tuple(exc.evidence)) from exc
        all_paths = paths
        try:
            after, model_decision = self._observe_scene(frames, goal.to_dict(),
                available_action_kinds=available_action_kinds,
                response_evidence_dir=evidence_dir, response_evidence_prefix=prefix, current_frame_paths=paths)
            if getattr(self.observer, 'last_response_evidence_path', None):
                all_paths += (self.observer.last_response_evidence_path,)
        except (RuntimeError, OSError) as exc:
            if getattr(self.observer, 'last_response_evidence_path', None):
                all_paths += (self.observer.last_response_evidence_path,)
            observation_errors = (f"第1轮动作后观察失败：{exc}",)
            try:
                all_paths += persist_observer_failure_diagnostic(self.observer, evidence_dir=evidence_dir,
                    prefix=prefix, error=exc)
            except OSError as diagnostic_error:
                # A second storage failure must not erase the first error or captured evidence.
                observation_errors += (f"动作后诊断保存失败：{diagnostic_error}",)
            raise GenericActionAdapterError('通用页面观察失败：' + observation_errors[0], evidence=all_paths,
                observation_errors=observation_errors) from exc

        try:
            controller_evidence = self.controller.verify_after_action(resolved, before, after)
        except UniversalActionError as exc:
            raise GenericActionAdapterError(f"必要动作后硬校验失败：{exc}", evidence=all_paths,
                verification_errors=(str(exc),)) from exc
        return (after, tuple(frames), paths, all_paths, (), controller_evidence, model_decision)


    def _arm_physical_execution(self, resolved: ResolvedSemanticAction, scene: UIScene,
        frames: tuple[Image.Image, ...] | list[Image.Image], paths: tuple[str,
        ...]) -> tuple[OrientationCredential | None, Callable[[], Any] | None]:
        if resolved.text_transport == 'adb_keyboard' and resolved.kind in {'input_verified_text',
            'clear_verified_text', 'press_enter'}:
            return None, None
        clear = getattr(self.robot, "clear_physical_execution_authorization", None)
        if resolved.kind not in self.PHYSICAL_KINDS:
            return None, clear if callable(clear) else None
        arm = getattr(self.robot, "arm_physical_execution", None)
        reject_if(not callable(arm) or not callable(clear), GenericActionAdapterError("机械臂控制器未提供共享物理执行门禁，拒绝动作。", evidence=paths))
        clear()
        try:
            credential = self._single_step_scene_orientation_credential(scene=scene, frames=frames)
            diagnostic_flag = "local_canvas_binding_verified"
            selected = len(frames) - 1
            try:
                self.observer.last_orientation_audit_diagnostics = {'audit_source': credential.source, 'model_calls': 0,
                    diagnostic_flag: True, 'selected_frame_index': selected, 'scene_fingerprint': scene.fingerprint}
            except Exception:
                pass
            if 0 <= selected < len(paths):
                with Image.open(paths[selected]) as persisted:
                    credential = replace(credential, evidence_frame_fingerprint=frame_fingerprint(persisted.convert(
                        'RGB')))
            hardware_action = {'scroll': 'swipe',
                'swipe_element': 'swipe'}.get(resolved.kind, resolved.kind)
            credential.assert_authorizes(device_id=self.device_id, scene_fingerprint=scene.fingerprint,
                frame_size=credential.frame_size, action=hardware_action)
            arm(credential, action=hardware_action, scene_fingerprint=scene.fingerprint)
            return credential, clear
        except (OrientationSafetyError, RuntimeError, ValueError) as exc:
            clear()
            raise GenericActionAdapterError(f'动作前单步画面方向凭据校验失败：{exc}', evidence=paths) from exc

    def _prepare_execution(
        self,
        requested_action: SemanticAction,
        planned_scene: UIScene,
        *,
        confirmed: bool,
        evidence_dir: Path | None,
        planned_frames: tuple[Image.Image, ...] | list[Image.Image],
    ) -> tuple[str, list[Image.Image], tuple[str, ...], UIScene, SemanticAction,
               ResolvedSemanticAction, OrientationCredential | None, Callable[[], Any] | None]:
        reject_if(confirmed is not True, GenericActionAdapterError("必须明确确认当前这一个语义动作。"))
        safe_node = re.sub(r"[^a-zA-Z0-9_-]+", "_", requested_action.node_id)[:48]
        evidence_prefix = f"{safe_node or 'action'}_{uuid.uuid4().hex}"
        reject_if(
            not planned_frames,
            GenericActionAdapterError("执行动作必须携带产生该 Qwen 动作的当前截图帧。"),
        )
        before_frames, before_paths = self._capture_confirmation_frames(
            evidence_dir=evidence_dir, prefix=f"{evidence_prefix}_before"
        )
        try:
            self._validate_confirmation_frame_dimensions(planned_frames, before_frames)
        except GenericActionAdapterError as exc:
            raise GenericActionAdapterError(str(exc), evidence=before_paths) from exc
        before = planned_scene
        # Qwen's action is immutable after selection.  Local code may validate
        # device/scope/geometry, but it may not rewrite its semantic fields.
        try:
            resolved = self.controller.resolve_one(requested_action, before, confirmed=True)
        except UniversalActionError as exc:
            raise GenericActionAdapterError(f'确认前控制器拒绝动作：{exc}', evidence=before_paths) from exc
        reject_if(
            resolved.kind not in self.DEVICE_ACTION_KINDS and resolved.kind != 'wait_for_change',
            GenericActionAdapterError(f'当前通用硬件适配器尚未开放：{resolved.kind}', evidence=before_paths),
        )
        orientation_credential, clear_authorization = self._arm_physical_execution(resolved, before,
            before_frames, before_paths)
        return (evidence_prefix, before_frames, before_paths, before, requested_action,
            resolved, orientation_credential, clear_authorization)

    @staticmethod
    def _executor_point(point: NormalizedPoint | None) -> tuple[int, int] | None:
        if point is None:
            return None
        return (
            max(0, min(1000, round(point[0] * 1000))),
            max(0, min(1000, round(point[1] * 1000))),
        )

    def _mint_text_scope(
        self,
        resolved: ResolvedSemanticAction,
        requested_action: SemanticAction,
        before: UIScene,
        action_authority: ConfirmationAuthority | None,
        before_paths: tuple[str, ...],
    ) -> Any:
        if resolved.text_transport != 'adb_keyboard':
            return None
        transport = self.text_transport
        reject_if(
            transport is None or action_authority is None,
            GenericActionAdapterError(
                "ADB Keyboard 动作缺少 transport 或已消费的一次性 authority。",
                evidence=before_paths,
            ),
        )
        assert transport is not None and action_authority is not None
        reject_if(
            action_authority.device_id != self.device_id
            or action_authority.fingerprint != before.fingerprint
            or action_authority.decision_node_id != requested_action.node_id
            or action_authority.action_digest != canonical_digest(requested_action.to_dict()),
            GenericActionAdapterError(
                "ADB Keyboard 动作 authority 与当前设备、画面或 canonical 动作不一致。",
                evidence=before_paths,
            ),
        )
        prior = resolved.prior_input_value
        fragment = resolved.input_fragment if resolved.kind != 'clear_verified_text' else ''
        expected = resolved.expected_input_value
        reject_if(
            not isinstance(prior, str)
            or not isinstance(fragment, str)
            or not isinstance(expected, str)
            or not resolved.input_field_id,
            GenericActionAdapterError(
                "ADB Keyboard 动作缺少精确 typed 文字事务。", evidence=before_paths
            ),
        )
        if resolved.text_transport == 'adb_keyboard':
            try:
                return transport.mint_action_scope(session_id=action_authority.session_id,
                    task_id=action_authority.task_id, revision=action_authority.revision,
                    action_id=action_authority.decision_node_id, input_field_id=resolved.input_field_id,
                    observation_fingerprint=before.fingerprint, prior_text_digest=text_digest(prior),
                    fragment_text_digest=text_digest(fragment), expected_text_digest=text_digest(expected))
            except (RuntimeError, ValueError) as exc:
                raise GenericActionAdapterError(f"ADB Keyboard 单动作 scope 签发失败：{exc}",
                    evidence=before_paths) from exc
        return None

    def _execute_device_action(
        self,
        resolved: ResolvedSemanticAction,
        text_scope: Any,
        clear_authorization: Callable[[], Any] | None,
        before_paths: tuple[str, ...],
    ) -> tuple[int, Any, dict[str, Any] | None, dict[str, Any]]:
        execution_request = DeviceActionRequest(
            kind=resolved.kind,
            point=self._executor_point(resolved.normalized_point),
            end_point=self._executor_point(resolved.normalized_end_point),
            direction=resolved.direction,
            hold_seconds=resolved.hold_seconds,
            input_fragment=resolved.input_fragment,
            text_transport=resolved.text_transport,
            text_scope=text_scope,
            wait_seconds=max(0.5, self.post_action_settle) if resolved.kind == 'wait_for_change' else None,
            launch_ref=resolved.launch_ref,
        )
        try:
            execution_result = self.device_executor.execute(execution_request)
            return (
                execution_result.physical_actions,
                execution_result.transport_result,
                execution_result.hardware_receipt,
                dict(execution_result.metadata),
            )
        except DeviceExecutionError as exc:
            raise GenericActionAdapterError(f'设备执行器拒绝动作：{exc}', physical_actions=exc.physical_actions,
                evidence=before_paths, execution_metadata=getattr(exc, 'metadata', {})) from exc
        except OrientationSafetyError as exc:
            raise GenericActionAdapterError(f'共享物理执行门在控制端原语前拒绝动作：{exc}', physical_actions=0,
                evidence=before_paths) from exc
        except Exception as exc:
            raise GenericActionAdapterError(f'设备单步动作调用失败：{exc}', physical_actions=0,
                evidence=before_paths) from exc
        finally:
            if callable(clear_authorization):
                clear_authorization()

    def _observe_after_execution(
        self,
        goal: GenericIntentDraft,
        requested_action: SemanticAction,
        resolved: ResolvedSemanticAction,
        action_authority: ConfirmationAuthority | None,
        physical_actions: int,
        execution_metadata: dict[str, Any],
        before: UIScene,
        before_frames: list[Image.Image],
        before_paths: tuple[str, ...],
        evidence_dir: Path | None,
        evidence_prefix: str,
        available_action_kinds: frozenset[str] | None,
        post_action_available_action_kinds: frozenset[str] | None,
    ) -> tuple[UIScene, tuple[Image.Image, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...], dict[str, Any]]:
        try:
            (
                after,
                after_frames,
                after_frame_paths,
                all_after_paths,
                observation_errors,
                controller_transition_evidence,
                after_model_decision,
            ) = self._observe_stable_post_action_scene(
                self._post_action_goal(
                    goal, authority=action_authority, requested=requested_action, resolved=resolved,
                    physical_actions=physical_actions, execution_metadata=execution_metadata,
                ),
                before=before,
                before_frames=tuple(before_frames),
                resolved=resolved,
                evidence_dir=evidence_dir,
                evidence_prefix=evidence_prefix,
                available_action_kinds=(
                    post_action_available_action_kinds
                    if post_action_available_action_kinds is not None
                    else available_action_kinds
                ),
            )
        except Exception as exc:
            # The device has already executed. Every ordinary post-action failure must
            # carry its receipt count, including filesystem and unexpected parser errors.
            evidence = before_paths + tuple(getattr(exc, "evidence", ()))
            raise GenericActionAdapterError(
                f'单步动作后验证失败：{exc}', physical_actions=physical_actions, evidence=evidence,
                observation_errors=tuple(getattr(exc, 'observation_errors', ())), verification_errors=tuple(getattr(exc,
                'verification_errors', ())), execution_metadata=execution_metadata) from exc
        return (
            after, after_frames, after_frame_paths, all_after_paths,
            observation_errors, controller_transition_evidence, after_model_decision,
        )

    @staticmethod
    def _verify_effect_finish(
        action_authority: ConfirmationAuthority | None,
        after_model_decision: dict[str, Any],
        before_frames: list[Image.Image],
        after_frames: tuple[Image.Image, ...],
        before_paths: tuple[str, ...],
        all_after_paths: tuple[str, ...],
        physical_actions: int,
        execution_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        if not (action_authority is not None and action_authority.effect_ids
            and str(after_model_decision.get('status') or '').strip() == 'finish'):
            return execution_metadata
        transition = measure_material_visual_transition(before_frames, after_frames)
        execution_metadata = {**execution_metadata, 'effect_visual_transition': transition}
        reject_if(
            not transition['material'],
            GenericActionAdapterError(
                '外部效果动作后的真实帧没有可归因的新变化；动作前已有画面不能作为本次 finish 证据，'
                '本次已执行1次且不会自动重复。',
                physical_actions=physical_actions,
                evidence=before_paths + all_after_paths,
                verification_errors=('外部效果缺少动作前后真实帧变化',),
                execution_metadata=execution_metadata,
            ),
        )
        return execution_metadata

    def execute(self, *, requested_action: SemanticAction, planned_scene: UIScene, goal: GenericIntentDraft,
        confirmed: bool, evidence_dir: Path | None=None, planned_frames: tuple[Image.Image,
        ...] | list[Image.Image]=(), action_authority: ConfirmationAuthority | None=None,
        available_action_kinds: frozenset[str] | None=None,
        post_action_available_action_kinds: frozenset[str] | None=None
        ) -> GenericActionExecutionResult:
        (
            evidence_prefix, before_frames, before_paths, before, rebound, resolved,
            orientation_credential, clear_authorization,
        ) = self._prepare_execution(
            requested_action, planned_scene, confirmed=confirmed, evidence_dir=evidence_dir,
            planned_frames=planned_frames,
        )
        text_scope = self._mint_text_scope(
            resolved, requested_action, before, action_authority, before_paths
        )
        physical_actions, robot_result, hardware_receipt, execution_metadata = self._execute_device_action(
            resolved, text_scope, clear_authorization, before_paths
        )
        (
            after, after_frames, after_frame_paths, all_after_paths,
            observation_errors, controller_transition_evidence, after_model_decision,
        ) = self._observe_after_execution(
            goal, requested_action, resolved, action_authority, physical_actions, execution_metadata,
            before, before_frames, before_paths, evidence_dir, evidence_prefix,
            available_action_kinds, post_action_available_action_kinds,
        )
        execution_metadata = self._verify_effect_finish(
            action_authority, after_model_decision, before_frames, after_frames,
            before_paths, all_after_paths, physical_actions, execution_metadata,
        )
        return GenericActionExecutionResult(requested_action=requested_action, rebound_action=rebound,
            resolved_action=resolved, before_scene=before, after_scene=after,
            planned_scene_fingerprint=planned_scene.fingerprint,
            confirmation_frame_identity_verified=False,
            confirmation_frame_delta=None, physical_actions=physical_actions,
            primary_input_confirmation_reused=False,
            action_outcome='executed', verification_errors=(),
            robot_result=robot_result, hardware_receipt=hardware_receipt, execution_metadata=execution_metadata,
            evidence=before_paths + all_after_paths,
            after_frames=after_frames, after_frame_paths=after_frame_paths, observation_errors=observation_errors,
            controller_transition_evidence=controller_transition_evidence,
            after_model_decision=after_model_decision, before_frames=before_frames,
            before_frame_paths=before_paths, orientation_credential=orientation_credential)


    def _save_frames(self, frames: list[Image.Image], evidence_dir: Path | None, prefix: str) -> tuple[str, ...]:
        return save_task_frames(frames, evidence_dir, prefix, self.device_id)
