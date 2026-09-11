from __future__ import annotations

"""Regression coverage for the canonical single-action adapter."""

import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageDraw, ImageFilter, ImageOps

from agent.infrastructure.capability_acceptance import _validate_live_promotion_source
from agent.domain import (
    DeviceActionRequest,
    DeviceExecutionError,
)
from agent.domain.confirmation_authority import ConfirmationAuthority
from agent.domain.text_transport import (
    TEXT_TRANSPORT_PROTOCOL,
    TextTransportActionScope,
    TextTransportProfile,
    TextTransportResult,
)
from agent.domain.validation import canonical_digest
from agent.infrastructure import RobotDeviceExecutor
from agent.infrastructure.generic_action_adapter import (
    GenericSingleActionAdapter as _GenericSingleActionAdapter,
)
from agent.application.action_adapter import GenericActionAdapterError
from agent.domain.generic_goal import GenericIntentDraft
from agent.infrastructure.generic_scene_observer import (
    SINGLE_STEP_OUTPUT_TOKENS,
    SingleStepGenericSceneObserver,
)
from agent.infrastructure.observation_images import local_frame_fingerprint
from agent.infrastructure.observation_images import measure_frame_sharpness
from agent.infrastructure.orientation_safety import (
    OrientationSafetyError,
    _claim_audit_seal,
)
from agent.domain.semantic_action import SemanticAction as _SemanticAction
from agent.domain.ui_scene import (
    CameraAlignmentFacts,
    SystemUIFacts,
    UIElement,
    UIScene,
)
from agent.domain.universal_action_controller import (
    ResolvedSemanticAction,
    UniversalActionController,
    UniversalActionError,
)
from agent.domain.vision_model import VisionAgentError


def SemanticAction(*, node_id: str, action: str, params: dict) -> _SemanticAction:
    """Upgrade historical adapter fixtures to the current explicit-point wire contract."""

    current = dict(params)
    if action in {"tap_semantic", "dismiss_overlay", "press_enter", "double_tap", "long_press"}:
        current.setdefault("tap_point", (0.3, 0.4))
    return _SemanticAction(node_id=node_id, action=action, params=current)


TEST_QWERTY_GEOMETRY = {
    "type": "qwerty",
    "anchors": {
        "q": [115, 704],
        "p": [875, 704],
        "a": [157, 773],
        "l": [832, 773],
        "z": [241, 844],
        "m": [747, 844],
        "backspace": [875, 844],
    },
    "source": "input_structure_audit",
}


class RawSceneProvider:
    configured = True

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.max_tokens_seen = []

    def _chat(self, messages, *, max_tokens, **_kwargs):
        self.calls += 1
        self.max_tokens_seen.append(max_tokens)
        value = self.responses.pop(0)
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        return value

    def status(self):
        return {"configured": True, "model": "offline-sequence"}


class FakeSceneObserver:
    def __init__(self, scenes):
        self.scenes = list(scenes)
        self.calls = 0
        self.goal_contexts = []

    def _next_scene(self, *, goal_context=None):
        self.calls += 1
        self.goal_contexts.append(goal_context)
        result = self.scenes.pop(0)
        if isinstance(result, BaseException):
            raise result
        # Historical generic frame tests predate typed fields. Supply fixture
        # identity only; text, focus, geometry and post-action results stay exact.
        result = replace(result, elements=tuple(replace(item,
            states={"input_field_id": "field_primary", **item.states})
            if item.role == "input" else item for item in result.elements))
        if result.camera_alignment.phone_content_rotation == "unknown":
            result = replace(
                result,
                camera_alignment=aligned_camera_facts(),
            )
        return result

    def observe_with_decision(self, *, frames, goal_context=None, **_kwargs):
        del frames
        result = self._next_scene(goal_context=goal_context)
        return result, {
            "status": "finish",
            "evidence_refs": ["scene.summary"],
            "confidence": 1.0,
            "reason": "测试观察器在同一帧返回 scene 与 decision。",
        }


class RuntimeActionRecordingObserver(FakeSceneObserver):
    supports_runtime_action_contract = True

    def __init__(self, scenes):
        super().__init__(scenes)
        self.available_action_sets = []

    def observe_with_decision(self, *, available_action_kinds, **kwargs):
        self.available_action_sets.append(frozenset(available_action_kinds))
        return super().observe_with_decision(**kwargs)

class RawFailureSceneObserver(FakeSceneObserver):
    def __init__(self, raw_response: str) -> None:
        super().__init__([RuntimeError("目标精查严格协议拒绝")])
        self.last_raw_response = raw_response
        self.last_diagnostics = {
            "failed_stage": "parsing_targeted_refinement",
            "error_type": "schema_validation",
        }


class FakeRobot:
    def __init__(self):
        self.actions = []
        self.device_id = "test-device"
        self._armed = None
        self._long_press_receipt = None
        self._click_receipt = None
        self._swipe_receipt = None

    def _record_click_receipt(self, click_count=1):
        self._click_receipt = {
            "version": "2026-08-19-seller-gui-click-barrier-v1",
            "channel": "left_button_atomic_click",
            "input_events_dispatched": True,

            "mechanical_contact_ack": False,
            "click_count": click_count,
        }

    def arm_physical_execution(self, credential, *, action, scene_fingerprint):
        credential.assert_authorizes(
            device_id=self.device_id,
            scene_fingerprint=scene_fingerprint,
            frame_size=credential.frame_size,
            action=action,
        )
        _claim_audit_seal(credential)
        self._armed = action

    def clear_physical_execution_authorization(self):
        self._armed = None

    def _consume(self, action):
        if self._armed != action:
            raise RuntimeError("missing test physical authorization")
        self._armed = None

    def vision_tap_relative(self, x, y):
        self._consume("tap_semantic")
        self.actions.append(("tap", x, y))
        self._record_click_receipt()
        return (x, y)

    def vision_dismiss_overlay_relative(self, x, y):
        self._consume("dismiss_overlay")
        self.actions.append(("dismiss", x, y))
        self._record_click_receipt()
        return (x, y)

    def vision_double_tap_relative(self, x, y):
        self._consume("double_tap")
        self.actions.append(("double_tap", x, y))
        self._record_click_receipt(2)
        return (x, y)

    def vision_swipe_up(self):
        self._consume("swipe")
        self.actions.append(("swipe", "up"))

    def vision_swipe_relative(
        self,
        start_x,
        start_y,
        end_x,
        end_y,
        direction,
    ):
        self._consume("swipe")
        self.actions.append(
            (
                "swipe_relative",
                direction,
                start_x,
                start_y,
                end_x,
                end_y,
            )
        )
        self._swipe_receipt = {
            "right_button_down_dispatched": True, "input_events_dispatched": True,
            "right_button_up_dispatched": True,


            "mechanical_contact_ack": False,
            "requested_direction": direction,
            "step_count": 6,
            "interpolation_steps_completed": 6,
        }
        return (start_x, start_y), (end_x, end_y)

    def vision_reveal_system_navigation(self):
        self._consume("reveal_system_navigation")
        self.actions.append(("reveal_system_navigation",))
        return (500, 950)

    def vision_android_back(self):
        self._consume("back")
        self.actions.append(("back",))
        self._record_click_receipt()
        return (500, 950)

    def vision_android_home(self):
        self._consume("home")
        self.actions.append(("home",))
        self._record_click_receipt()
        return (500, 950)

    def consume_last_click_receipt(self):
        receipt = self._click_receipt
        self._click_receipt = None
        return receipt

    def consume_last_swipe_receipt(self):
        receipt = self._swipe_receipt
        self._swipe_receipt = None
        return receipt

    def vision_long_press_relative(self, x, y, hold_seconds):
        self._consume("long_press")
        self.actions.append(("long_press", x, y, hold_seconds))
        self._long_press_receipt = {
            "version": "2026-08-16-seller-gui-contact-barrier-v3",
            "channel": "right_button_stationary_touch",
            "input_events_dispatched": True,

            "hold_started_after_barrier": True,
            "requested_hold_seconds": hold_seconds,
            "barrier_offset_pixels": 3,
            "changed_pixels": 240,
            "return_changed_pixels": 240,
            "barrier_elapsed_ms": 35.0,
            "post_barrier_settle_seconds": 0.45,
        }
        return (x, y)

    def consume_last_long_press_receipt(self):
        receipt = self._long_press_receipt
        self._long_press_receipt = None
        return receipt

    def vision_drag_relative(self, start_x, start_y, end_x, end_y):
        self._consume("drag")
        self.actions.append(("drag", start_x, start_y, end_x, end_y))


class PhysicalGateCanvasErrorRobot(FakeRobot):
    def vision_swipe_up(self):
        self._consume("swipe")
        raise OrientationSafetyError("方向凭据与本地画布尺寸不匹配。")


class ClickReceiptRobot(FakeRobot):
    def __init__(self, *, valid=True):
        super().__init__()
        self.valid = valid
        self._click_receipt = None

    def _record_click_receipt(self):
        self._click_receipt = {
            "version": "2026-08-19-seller-gui-click-barrier-v1",
            "channel": "left_button_atomic_click",
            "input_events_dispatched": self.valid,

            "mechanical_contact_ack": False,
        }

    def vision_tap_relative(self, x, y):
        result = super().vision_tap_relative(x, y)
        self._record_click_receipt()
        return result

    def vision_android_home(self):
        result = super().vision_android_home()
        self._record_click_receipt()
        return result

    def consume_last_click_receipt(self):
        receipt = self._click_receipt
        self._click_receipt = None
        return receipt

class GenericSingleActionAdapter(_GenericSingleActionAdapter):
    def __init__(self, *args, device_id="test-device", **kwargs):
        if "text_transport" not in kwargs:
            kwargs["text_transport"] = FakeAdbKeyboardTextTransport(event_trace=kwargs.get("robot"))
        super().__init__(*args, device_id=device_id, **kwargs)

    def execute(self, *args, planned_frames=(), **kwargs):
        # Production receives the exact Qwen observation frames from the
        # orchestrator.  Direct adapter tests emulate that caller explicitly;
        # the production adapter itself no longer observes or reselects here.
        if not planned_frames:
            observed, frames, _paths, _model_decision = self.capture_scene(
                kwargs["goal"],
                evidence_dir=kwargs.get("evidence_dir"),
                prefix="test_orchestrator_before",
            )
            kwargs["planned_scene"] = observed
            planned_frames = tuple(frames)
        requested = kwargs["requested_action"]
        if requested.action in {"input_verified_text", "clear_verified_text"} and "text_transport" not in requested.params:
            # Explicit test-fixture migration from mechanical requests to the
            # current typed ADB contract. Never alter already typed test cases.
            before = kwargs["planned_scene"]
            before = replace(before, elements=tuple(replace(item,
                states={"input_field_id": "field_primary", **item.states})
                if item.role == "input" else item for item in before.elements))
            kwargs["planned_scene"] = before
            target = before.get_element(requested.params["element_id"])
            prior = target.states.get("value", "")
            expected = requested.params.get("text", "") if requested.action == "input_verified_text" else ""
            params = {**requested.params, "text_transport": "adb_keyboard",
                "input_field_id": target.states.get("input_field_id", "field_primary"),
                "prior_input_value": prior, "expected_input_value": expected}
            if requested.action == "input_verified_text":
                params["input_fragment"] = expected[len(prior):]
            requested = replace(requested, params=params)
            kwargs["requested_action"] = requested
            kwargs.setdefault("action_authority", ConfirmationAuthority(session_id="fixture-session",
                task_id="fixture-task", device_id=self.device_id, revision=1, step_id="fixture-subgoal",
                effect_ids=(), observation_id="fixture-observation", fingerprint=before.fingerprint,
                decision_node_id=requested.node_id, action_digest=canonical_digest(requested.to_dict()), consumed=True))
        return super().execute(*args, planned_frames=planned_frames, **kwargs)


class FakeAdbKeyboardTextTransport:
    def __init__(self, event_trace=None) -> None:
        self.event_trace = event_trace
        self.profile = TextTransportProfile(protocol_version=TEXT_TRANSPORT_PROTOCOL,
            profile_id="profile-test-device", device_id="test-device", adb_serial="serial-test-device",
            enabled=True, capabilities=("append_text", "clear_text"), command_timeout_seconds=5.0)
        self.minted = []
        self.calls = []

    def mint_action_scope(self, **values):
        self.minted.append(dict(values))
        return TextTransportActionScope(protocol_version=TEXT_TRANSPORT_PROTOCOL,
            device_id=self.profile.device_id, issued_at_epoch=100.0,
            nonce="nonce-0000000000001", **values)

    @staticmethod
    def _result(scope, operation):
        return TextTransportResult(protocol_version=TEXT_TRANSPORT_PROTOCOL, device_id=scope.device_id,
            action_id=scope.action_id, nonce=scope.nonce, operation=operation, status="accepted",
            attempted=True, accepted=True, reason_code=None, command_digest="a" * 64,
            receipt_digest="b" * 64)

    def append_text(self, scope, text):
        self.calls.append(("append_text", scope, text))
        if self.event_trace is not None:
            # Shared test event trace, not a Robot method or mechanical call.
            self.event_trace.actions.append(("input", text))
        return self._result(scope, "append_text")

    def clear_text(self, scope):
        self.calls.append(("clear_text", scope))
        if self.event_trace is not None:
            self.event_trace.actions.append(("clear",))
        return self._result(scope, "clear_text")


class SequenceCapture:
    def __init__(self, frames):
        self.frames = list(frames)
        self.calls = 0

    def __call__(self):
        index = min(self.calls, len(self.frames) - 1)
        self.calls += 1
        value = self.frames[index]
        if isinstance(value, Image.Image):
            return value.copy()
        return Image.new("RGB", (540, 960), value)


def textured_phone_frame() -> Image.Image:
    """Return a deterministic high-frequency frame for clarity-gate tests."""

    frame = Image.new("RGB", (540, 960), "white")
    draw = ImageDraw.Draw(frame)
    for y in range(0, 960, 24):
        color = "black" if (y // 24) % 2 == 0 else "navy"
        draw.line((0, y, 539, y), fill=color, width=3)
    for x in range(0, 540, 30):
        draw.line((x, 0, x, 959), fill="gray", width=2)
    draw.rectangle((80, 240, 460, 350), outline="black", width=5)
    draw.text((100, 280), "generic input surface 2026", fill="black")
    return frame


def ocr_text_payload(
    text: str,
    *,
    left: int,
    top: int,
    width: int = 180,
    height: int = 30,
) -> dict:
    word = {
        "text": text,
        "left": left,
        "top": top,
        "width": width,
        "height": height,
    }
    return {
        "lines": [
            {
                **word,
                "words": [dict(word)],
            }
        ]
    }


class SecondPostCaptureFailureAdapter(GenericSingleActionAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.post_capture_calls = 0

    def _capture_stable_post_action_frames(self, **kwargs):
        self.post_capture_calls += 1
        if self.post_capture_calls == 2:
            raise GenericActionAdapterError(
                "动作后画面在未能采集到连续4帧",
                evidence=("second_capture_timeout.jpg",),
            )
        return super()._capture_stable_post_action_frames(**kwargs)


def goal():
    return GenericIntentDraft(
        understood=True,
        app_id="settings",
        app_name="设置",
        objective="打开蓝牙设置",
        success_criteria={"screen": "蓝牙设置"},
    )


def navigation_goal(*, execution_class="navigate"):
    return GenericIntentDraft(
        understood=True,
        app_id="browser",
        app_name="浏览器",
        objective="打开浏览器后读取当前页面标题",
        entities={
            "active_subgoal_visual_context": {
                "step_id": "open_browser",
                "objective": "打开浏览器",
                "constraints": ["仅使用当前可见入口"],
                "completion_conditions": ["浏览器结果页面已显示"],
                "execution_class": execution_class,
                "goal_entities": {
                    "target_ui_label": "浏览器",
                    "target_surface": "device",
                },
            }
        },
        success_criteria={"screen": "浏览器结果页面"},
    )


def exact_tap_goal(label="两个字段分别输入"):
    return GenericIntentDraft(
        understood=True,
        app_id="current_surface",
        app_name="当前界面",
        objective="点击当前画面中的目标控件",
        entities={
            "target_ui_label": label,
            "active_subgoal_visual_context": {
                "step_id": "exact_tap_semantic",
                "objective": "点击当前画面中的目标控件",
                "constraints": [],
                "completion_conditions": ["动作后出现新的稳定画面"],
                "execution_class": "navigate",
                "goal_entities": {
                    "target_surface": "current_surface",
                    "target_ui_label": label,
                },
            },
        },
        success_criteria={"action_completed": "动作后出现新的稳定画面"},
    )


def aligned_camera_facts() -> CameraAlignmentFacts:
    return CameraAlignmentFacts(
        camera_layout_orientation="portrait",
        phone_content_rotation="upright",
        confidence=0.96,
        evidence=("手机界面轴线与相机画布正向一致",),
    )


def scene(
    fingerprint,
    *,
    screen_id="android_home",
    element_id="e1",
    bounds=(0.2, 0.3, 0.4, 0.5),
    app_id=None,
    system_ui=None,
    camera_alignment=None,
):
    current = UIScene(
        app_id=(
            app_id
            if app_id is not None
            else ("settings" if screen_id != "android_home" else "unknown")
        ),
        screen_id=screen_id,
        summary="测试页面",
        elements=(
            UIElement(
                element_id=element_id,
                role="icon",
                meaning="app_icon",
                label="设置",
                bounds=bounds,
                confidence=0.96,
            ),
        ),
        stable=True,
        confidence=0.95,
        fingerprint=fingerprint,
        system_ui=system_ui or SystemUIFacts(),
        camera_alignment=camera_alignment or aligned_camera_facts(),
    )
    return current


def compact_scene_raw(*, label="设置"):
    payload = scene("model-scene").to_dict()
    payload["elements"][0]["label"] = label
    for element in payload["elements"]:
        element["bounds"] = [value * 1000 for value in element["bounds"]]
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


class RevealSystemNavigationControllerTests(unittest.TestCase):
    @staticmethod
    def action() -> SemanticAction:
        return SemanticAction(
            node_id="reveal-navigation",
            action="reveal_system_navigation",
            params={},
        )

    def test_requires_pre_facts_but_does_not_rejudge_post_navigation(self):
        controller = UniversalActionController()
        before = scene(
            "same",
            system_ui=SystemUIFacts(
                immersive_or_fullscreen=True,
                navigation_bar_visible=False,
            ),
        )
        resolved = controller.resolve_one(self.action(), before, confirmed=True)

        controller.verify_after_action(
            resolved,
            before,
            scene(
                "same",
                system_ui=SystemUIFacts(
                    immersive_or_fullscreen=True,
                    navigation_bar_visible=True,
                ),
            ),
        )

        self.assertEqual((), controller.verify_after_action(
                resolved,
                before,
                scene("different", screen_id="navigation_bar_visible_summary_only"),
            ))

    def test_rejects_unknown_or_already_visible_system_ui_precondition(self):
        controller = UniversalActionController()
        for facts in (
            SystemUIFacts(
                immersive_or_fullscreen="unknown",
                navigation_bar_visible="unknown",
            ),
            SystemUIFacts(
                immersive_or_fullscreen=True,
                navigation_bar_visible=True,
            ),
        ):
            with self.subTest(facts=facts), self.assertRaisesRegex(
                UniversalActionError,
                "沉浸态且导航栏隐藏",
            ):
                controller.resolve_one(
                    self.action(),
                    scene("before", system_ui=facts),
                    confirmed=True,
                )


class ElementBoundSwipeControllerTests(unittest.TestCase):
    @staticmethod
    def before_scene() -> UIScene:
        return UIScene(
            app_id="system",
            screen_id="recent_tasks",
            summary="唯一应用预览卡片可见",
            elements=(
                UIElement(
                    element_id="preview-card",
                    role="list_item",
                    meaning="application_preview_card",
                    label="示例应用",
                    bounds=(0.27, 0.29, 0.73, 0.81),
                    confidence=0.99,
                    states={
                        "goal_relevant": True,
                        "fully_visible": True,
                    },
                    evidence=("唯一完整可见的应用预览卡片",),
                ),
            ),
            stable=True,
            confidence=0.99,
            fingerprint="before-card",
            camera_alignment=aligned_camera_facts(),
        )

    @classmethod
    def action(cls) -> SemanticAction:
        element = cls.before_scene().elements[0]
        return SemanticAction(
            node_id="dismiss-card",
            action="swipe_element",
            params={
                "element_id": element.element_id,
                "target": element.meaning,
                "role": element.role,
                "label": element.label,
                "states": dict(element.states),
                "start": (0.5, 0.68),
                "end": (0.5, 0.08),
            },
        )


    def test_controller_derives_path_and_accepts_fresh_stable_receipt(self):
        controller = UniversalActionController()
        before = self.before_scene()
        resolved = controller.resolve_one(self.action(), before, confirmed=True)

        self.assertAlmostEqual(0.5, resolved.normalized_point[0])
        self.assertAlmostEqual(0.68, resolved.normalized_point[1])
        self.assertEqual((0.5, 0.08), resolved.normalized_end_point)
        self.assertEqual("preview-card", resolved.target_element_id)
        self.assertEqual("up", resolved.direction)

        unchanged = replace(before, fingerprint="fresh-stable-receipt")
        self.assertEqual((), controller.verify_after_action(resolved, before, unchanged))

    def test_targeted_transport_uses_relative_path_but_viewport_keeps_preset(self):
        robot = FakeRobot()
        executor = RobotDeviceExecutor(robot)

        robot._armed = "swipe"
        targeted = executor.execute(
            DeviceActionRequest(
                kind="swipe_element",
                point=(500, 680),
                end_point=(500, 80),
                direction="up",
            )
        )
        self.assertEqual(1, targeted.physical_actions)
        self.assertEqual(
            [("swipe_relative", "up", 500, 680, 500, 80)],
            robot.actions,
        )

        robot._armed = "swipe"
        viewport = executor.execute(
            DeviceActionRequest(kind="scroll", direction="up")
        )
        self.assertEqual(1, viewport.physical_actions)
        self.assertEqual(("swipe", "up"), robot.actions[-1])

        with self.assertRaisesRegex(DeviceExecutionError, "请求方向不一致"):
            DeviceActionRequest(
                kind="swipe_element",
                point=(500, 80),
                end_point=(500, 680),
                direction="up",
            ).validate()


class ExactTypedInputControllerTests(unittest.TestCase):




    def test_adb_keyboard_clear_binds_typed_field_without_keyboard_geometry(self):
        controller = UniversalActionController()
        before = UIScene(app_id="generic_app", screen_id="editor", summary="唯一聚焦输入框",
            elements=(UIElement(element_id="field", role="input", meaning="application_text_input",
                label="草稿", bounds=(0.1, 0.1, 0.9, 0.2), confidence=0.98, states={
                    "focused": True, "goal_relevant": True, "fully_visible": True,
                    "value": "草稿🙂", "input_field_id": "field_primary", "ime_preedit_text": "",
                }),), stable=True, confidence=0.98, fingerprint="before-companion-clear")
        action = SemanticAction(node_id="companion-clear", action="clear_verified_text", params={
            "element_id": "field", "target": "application_text_input", "role": "input", "label": "草稿",
            "states": before.elements[0].states, "text_transport": "adb_keyboard",
            "input_field_id": "field_primary", "prior_input_value": "草稿🙂", "expected_input_value": "",
        })

        resolved = controller.resolve_one(action, before, confirmed=True)
        self.assertEqual("adb_keyboard", resolved.text_transport)
        self.assertIsNone(resolved.normalized_point)
        self.assertNotIn("delete_count", resolved.to_dict())
        after = replace(before, fingerprint="after-companion-clear", elements=(replace(before.elements[0],
            label="", states={**before.elements[0].states, "value": ""}),))
        controller.verify_after_action(resolved, before, after)

        stale_preedit = replace(after, fingerprint="after-companion-stale", elements=(replace(after.elements[0],
            states={**after.elements[0].states, "ime_preedit_text": "stale"}),))
        with self.assertRaisesRegex(UniversalActionError, "预编辑"):
            controller.verify_after_action(resolved, before, stale_preedit)




class GenericActionAdapterTests(unittest.TestCase):
    def _adapter(self, observer, robot, **kwargs):
        return GenericSingleActionAdapter(
            capture=lambda: Image.new("RGB", (540, 960), "gray"),
            observer=observer,
            robot=robot,
            frame_interval=0,
            post_action_settle=0,
            **kwargs,
        )

    def test_retired_confirmation_input_recovery_helpers_stay_absent(self) -> None:
        source = (
            Path(__file__).resolve().parent
            / "agent"
            / "infrastructure"
            / "generic_action_adapter.py"
        ).read_text(encoding="utf-8")
        for retired_name in (
            "_local_input_auxiliary_recovery_target",
            "_confirmation_allows_omitted_local_input_auxiliary",
            "_recover_omitted_verified_input_scene",
            "_recover_conflicting_clear_input_scene",
        ):
            self.assertNotIn(retired_name, source)

    def test_post_action_goal_appends_actual_action_without_mutating_history(self) -> None:
        goal = GenericIntentDraft(understood=True, app_id="messenger", app_name="消息工具",
            objective="输入新文字", entities={"history": [{"action": "home"}]})
        projected = _GenericSingleActionAdapter._post_action_goal(goal, authority=SimpleNamespace(revision=1),
            requested=SemanticAction(node_id='node-1', action='input_verified_text', params={'text': '新文字'}),
            resolved=ResolvedSemanticAction(node_id="node-1", kind="input_verified_text"), physical_actions=1)
        self.assertEqual(1, len(goal.entities["history"]))
        self.assertEqual(2, len(projected.entities["history"]))
        receipt = projected.entities["history"][-1]
        self.assertEqual("executed", receipt["transport_outcome"])
        self.assertEqual("input_verified_text", receipt["action"]["kind"])
        self.assertIsNone(receipt["visual_outcome"])

    def test_effect_finish_rejects_unchanged_raw_frames_after_one_action(self) -> None:
        frame = textured_phone_frame()
        fingerprint = local_frame_fingerprint(frame)
        planned = scene(fingerprint)
        target = planned.elements[0]
        action = SemanticAction(node_id='effect-without-visual-transition', action='tap_semantic', params={
            'element_id': target.element_id,
            'target': target.meaning,
            'role': target.role,
            'label': target.label,
            'states': dict(target.states),
        })
        authority = ConfirmationAuthority(session_id='session-effect', task_id='task-effect',
            device_id='test-device', revision=1, step_id='effect-subgoal', effect_ids=('effect-1',),
            observation_id='observation-effect', fingerprint=fingerprint, decision_node_id=action.node_id,
            action_digest=canonical_digest(action.to_dict()), consumed=True)
        robot = FakeRobot()
        adapter = GenericSingleActionAdapter(
            capture=SequenceCapture([frame.copy() for _ in range(8)]),
            observer=FakeSceneObserver([replace(planned, fingerprint='after-same-pixels')]),
            robot=robot,
            frame_interval=0,
            post_action_settle=0,
        )

        with self.assertRaisesRegex(GenericActionAdapterError, '真实帧没有可归因的新变化') as raised:
            adapter.execute(requested_action=action, planned_scene=planned,
                planned_frames=tuple(frame.copy() for _ in range(4)), goal=goal(), confirmed=True,
                action_authority=authority)

        self.assertEqual(1, raised.exception.physical_actions)
        self.assertEqual([('tap', 300, 400)], robot.actions)
        credential = raised.exception.execution_metadata['effect_visual_transition']
        self.assertFalse(credential['material'])
        self.assertLess(credential['max_tile_median_delta'], credential['minimum_tile_delta'])

    def test_capture_scene_forwards_current_subgoal_action_scope_to_qwen(self) -> None:
        observer = RuntimeActionRecordingObserver([scene("scoped")])
        adapter = self._adapter(observer, FakeRobot())
        scoped = frozenset({"tap_semantic", "home", "back"})

        adapter.capture_scene(
            goal(),
            evidence_dir=None,
            prefix="scoped-actions",
            available_action_kinds=scoped,
        )

        self.assertEqual([scoped], observer.available_action_sets)

    def test_production_execute_requires_the_current_qwen_frames(self) -> None:
        robot = FakeRobot()
        adapter = _GenericSingleActionAdapter(
            capture=lambda: Image.new("RGB", (540, 960), "gray"),
            observer=FakeSceneObserver([]),
            robot=robot,
            device_id="test-device",
            frame_interval=0,
            post_action_settle=0,
        )

        with self.assertRaisesRegex(GenericActionAdapterError, "产生该 Qwen 动作的当前截图帧"):
            adapter.execute(
                requested_action=SemanticAction(
                    node_id="must-bind-current-frame",
                    action="tap_semantic",
                    params={"element_id": "e1", "target": "app_icon"},
                ),
                planned_scene=scene("planned"),
                goal=goal(),
                confirmed=True,
            )

        self.assertEqual([], robot.actions)


    @staticmethod
    def _literal_input_scene(
        fingerprint,
        *,
        value="live",
        include_key=True,
        audited=False,
    ):
        input_element = UIElement(
            element_id="local_audited_input_1",
            role="input",
            meaning="application_text_input",
            label=value,
            bounds=(0.15, 0.53, 0.70, 0.59),
            confidence=1.0,
            states={
                "goal_relevant": False,
                "fully_visible": True,
                "focused": True,
                "visible": True,
                "value": value,
                "keyboard_layout": "qwerty",
                "keyboard_input_mode": "direct_latin",
                "keyboard_case_mode": "lower",
            },
            evidence=(f"应用输入框当前文字：{value}", "caret"),
        )
        elements = [input_element]
        if include_key:
            elements.append(
                UIElement(
                    element_id="local_audited_literal_key_1",
                    role="button",
                    meaning="input_exact_literal_key",
                    label="2",
                    bounds=(0.18, 0.70, 0.26, 0.77),
                    confidence=1.0,
                    states={
                        "goal_relevant": True,
                        "fully_visible": True,
                        "input_literal_key": True,
                        "key_value": "2",
                        "prior_input_value": "live",
                        "expected_input_value": "live2",
                        "input_element_id": "local_audited_input_1",
                        **(
                            {
                                "independent_geometry_verified": True,
                                "geometry_audit_source": "input_structure_audit",
                            }
                            if audited
                            else {}
                        ),
                    },
                    evidence=("输入结构审计确认下一字符对应唯一完整可见键位",),
                )
            )
        result = UIScene(
            app_id="wechat",
            screen_id="conversation",
            summary="聚焦输入框与键盘可见",
            elements=tuple(elements),
            stable=True,
            confidence=1.0,
            fingerprint=fingerprint,
            camera_alignment=aligned_camera_facts(),
        )
        result.validate()
        return result



    @staticmethod
    def _strict_primary_input_scene(fingerprint="strict-input"):
        states = {
            "goal_relevant": True,
            "fully_visible": True,
            "focused": True,
            "value": "",
            "input_field_id": "body_field",
            "input_field_label": "正文",
            "input_multiline": False,
            "keyboard_layout": "qwerty",
            "keyboard_input_mode": "direct_latin",
            "keyboard_case_mode": "lower",
            "keyboard_geometry": TEST_QWERTY_GEOMETRY,
            "primary_input_geometry_verified": True,
            "geometry_audit_source": "input_structure_audit",
        }
        result = UIScene(
            app_id="generic_app",
            screen_id="editor",
            summary="专用输入审计建立唯一聚焦输入框",
            elements=(
                UIElement(
                    element_id="local_audited_input_1",
                    role="input",
                    meaning="application_text_input",
                    label="",
                    bounds=(0.14, 0.27, 0.86, 0.45),
                    confidence=1.0,
                    states=states,
                    evidence=("正文输入框完整可见且光标位于框内",),
                ),
            ),
            stable=True,
            confidence=1.0,
            fingerprint=fingerprint,
            camera_alignment=aligned_camera_facts(),
        )
        result.validate()
        return result


    def test_strict_input_execute_uses_zero_duplicate_pre_action_model_audits(self):
        gray = Image.new("RGB", (540, 960), "gray")
        planned = self._strict_primary_input_scene(
            local_frame_fingerprint(gray)
        )
        field = planned.elements[0]
        after_field = replace(
            field,
            label="agent",
            states={**field.states, "value": "agent"},
            evidence=("正文输入框逐字显示 agent",),
        )
        after = replace(
            planned,
            summary="正文输入框逐字显示 agent",
            elements=(after_field,),
            fingerprint="after-agent",
        )
        observer = FakeSceneObserver([after])
        robot = FakeRobot()
        anchors = {
            key: list(value)
            for key, value in TEST_QWERTY_GEOMETRY["anchors"].items()
        }
        adapter = GenericSingleActionAdapter(
            capture=SequenceCapture(["gray"] * 4 + ["white"] * 4),
            observer=observer,
            robot=robot,
            frame_interval=0,
            post_action_settle=0,
        )
        result = adapter.execute(
            requested_action=SemanticAction(
                node_id="type-body",
                action="input_verified_text",
                params={


                    "element_id": field.element_id,
                    "target": field.meaning,
                    "role": field.role,
                    "label": field.label,
                    "states": dict(field.states),
                    "text": "agent",

                },
            ),
            planned_scene=planned,
            planned_frames=(gray, gray.copy(), gray.copy(), gray.copy()),
            goal=GenericIntentDraft(
                understood=True,
                app_id="generic_app",
                app_name="当前应用",
                objective="正文逐字等于 agent 且不发送",
                entities={"input_text": "agent"},
                success_criteria={"input": "agent"},
            ),
            confirmed=True,
        )

        self.assertEqual([("input", "agent")], robot.actions)
        self.assertEqual(1, observer.calls)
        self.assertFalse(result.primary_input_confirmation_reused)
        self.assertEqual("executed", result.action_outcome)
        self.assertFalse(hasattr(observer, "last_orientation_audit_diagnostics"))

    def test_post_action_scene_is_never_rewritten_from_transport_expectations(self):
        self.assertFalse(hasattr(GenericSingleActionAdapter, "_primary_input_confirmation_reusable"))
        self.assertFalse(hasattr(GenericSingleActionAdapter, "_rebind_action"))
        self.assertFalse(hasattr(GenericSingleActionAdapter, "_reconcile_literal_key_visual_wrap"))
        self.assertFalse(hasattr(GenericSingleActionAdapter, "_reconcile_verified_text_horizontal_suffix"))

        prior = "releasecandidate"
        fragment = "continuation"
        observed_suffix = "didatecontinuation"
        before = self._literal_input_scene("before", value=prior, include_key=False)
        after = self._literal_input_scene("after", value=observed_suffix, include_key=False)
        resolved = ResolvedSemanticAction(
            node_id="no-post-rewrite",
            kind="input_verified_text",
            text=prior + fragment,
            input_fragment=fragment,
            prior_input_value=prior,
            expected_input_value=prior + fragment,
            target_element_id="local_audited_input_1",
            before_fingerprint=before.fingerprint,
        )
        adapter = self._adapter(FakeSceneObserver([after]), FakeRobot())
        stable_frames = [Image.new("RGB", (540, 960), "white") for _ in range(4)]

        with patch.object(
            adapter,
            "_capture_stable_post_action_frames",
            return_value=(stable_frames, ()),
        ), self.assertRaisesRegex(GenericActionAdapterError, "文字不匹配"):
            adapter._observe_stable_post_action_scene(
                goal(),
                before=before,
                before_frames=tuple(stable_frames),
                resolved=resolved,
                evidence_dir=None,
                evidence_prefix="no-post-rewrite",
            )

        self.assertEqual(
            observed_suffix,
            after.get_element("local_audited_input_1").states["value"],
        )
    def test_placeholder_loss_recovery_executes_only_authorized_suffix(self):
        gray = Image.new("RGB", (540, 960), "gray")

        def typed_scene(fingerprint, value, *, bounds=(0.15, 0.53, 0.70, 0.59)):
            source = self._literal_input_scene(
                fingerprint,
                value=value,
                include_key=False,
            )
            field = source.elements[0]
            return replace(
                source,
                elements=(
                    replace(
                        field,
                        label="first",
                        bounds=bounds,
                        states={
                            **field.states,
                            "input_field_id": "input_field_1",
                            "input_multiline": True,
                            "keyboard_geometry": TEST_QWERTY_GEOMETRY,
                        },
                    ),
                ),
            )

        planned = typed_scene("planned", "first\n")
        fresh_missing = replace(planned, fingerprint="fresh", elements=())
        planned_audited = typed_scene("planned-audited", "first\n")
        fresh_audited = typed_scene(
            "fresh-audited",
            "first\n",
            bounds=(0.14, 0.52, 0.71, 0.60),
        )
        after = typed_scene("after", "first\nsecond")
        observer = FakeSceneObserver([after])
        robot = FakeRobot()
        adapter = GenericSingleActionAdapter(
            capture=SequenceCapture(["gray"] * 4 + ["white"] * 4),
            observer=observer,
            robot=robot,
            frame_interval=0,
            post_action_settle=0,
        )
        field = planned.elements[0]
        expected = "first\nsecond"

        result = adapter.execute(
            requested_action=SemanticAction(
                node_id="append-second",
                action="input_verified_text",
                params={


                    "element_id": field.element_id,
                    "target": field.meaning,
                    "role": field.role,
                    "label": field.label,
                    "states": dict(field.states),
                    "text": expected,

                },
            ),
            planned_scene=planned,
            planned_frames=tuple(gray.copy() for _ in range(4)),
            goal=goal(),
            confirmed=True,
        )

        self.assertEqual([("input", "second")], robot.actions)
        self.assertEqual(1, result.physical_actions)
        self.assertEqual("executed", result.action_outcome)
        self.assertEqual("input_field_1", result.before_scene.elements[0].states["input_field_id"])















    def _assert_public_observation_failure_before_robot(self, responses):
        provider = RawSceneProvider(responses)
        robot = FakeRobot()
        adapter = self._adapter(SingleStepGenericSceneObserver(provider), robot)
        action = SemanticAction(
            node_id="blocked-observer-json",
            action="tap_semantic",
            params={"element_id": "e1", "target": "app_icon"},
        )

        with self.assertRaises(GenericActionAdapterError) as caught:
            adapter.execute(
                requested_action=action,
                planned_scene=scene("planned"),
                goal=goal(),
                confirmed=True,
            )

        self.assertEqual(0, caught.exception.physical_actions)
        self.assertEqual([], robot.actions)
        self.assertIsNone(robot._armed)
        self.assertEqual([None], provider.max_tokens_seen)

    def test_failed_observation_persists_bounded_redacted_qwen_response(self):
        raw = (
            '{"api_key":"secret-api-value","Authorization":"Bearer bearer-value",'
            '"image":"data:image/jpeg;base64,QUJDREVGRw==",'
            '"url":"https://example.test/path?token=query-secret",'
            '"text":"visible"} password=plain-secret '
            + ("x" * 17000)
        )
        observer = RawFailureSceneObserver(raw)
        with tempfile.TemporaryDirectory() as temp:
            evidence_dir = Path(temp)
            with self.assertRaises(GenericActionAdapterError) as caught:
                self._adapter(observer, FakeRobot()).capture_scene(
                    goal(),
                    evidence_dir=evidence_dir,
                    prefix="before_step_1",
                )
            diagnostic_paths = [
                Path(path)
                for path in caught.exception.evidence
                if path.endswith("_qwen_failure.json")
            ]
            self.assertEqual(1, len(diagnostic_paths))
            artifact = json.loads(diagnostic_paths[0].read_text(encoding="utf-8"))

        self.assertEqual("parsing_targeted_refinement", artifact["failed_stage"])
        self.assertEqual("schema_validation", artifact["error_type"])
        self.assertEqual(len(raw), artifact["raw_response_length"])
        self.assertEqual(
            hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            artifact["raw_response_sha256"],
        )
        self.assertTrue(artifact["redacted_response_truncated"])
        self.assertLessEqual(len(artifact["redacted_raw_response"]), 16000)
        serialized = json.dumps(artifact, ensure_ascii=False)
        for secret in (
            "secret-api-value",
            "bearer-value",
            "query-secret",
            "plain-secret",
            "QUJDREVGRw==",
        ):
            self.assertNotIn(secret, serialized)
        self.assertNotIn("data:image", serialized)
        self.assertIn("[REDACTED_SECRET]", serialized)
        self.assertIn("[REDACTED_IMAGE_DATA_URL]", serialized)

    def test_successful_observation_does_not_write_qwen_failure_artifact(self):
        with tempfile.TemporaryDirectory() as temp:
            evidence_dir = Path(temp)
            self._adapter(FakeSceneObserver([scene("fresh")]), FakeRobot()).capture_scene(
                goal(),
                evidence_dir=evidence_dir,
                prefix="before_step_1",
            )
            self.assertEqual([], list(evidence_dir.glob("*_qwen_failure.json")))

    def test_scene_only_observer_cannot_reenter_the_runtime_contract(self):
        class ObsoleteSceneOnlyObserver:
            def observe(self, *, frames, goal_context=None):
                del frames, goal_context
                return scene("obsolete-scene-only")

        with self.assertRaisesRegex(
            GenericActionAdapterError,
            r"不支持同一截图响应中的 scene \+ decision 合同",
        ):
            self._adapter(ObsoleteSceneOnlyObserver(), FakeRobot()).capture_scene(
                goal(),
                evidence_dir=None,
                prefix="before_step_1",
            )

    def test_adapter_passes_the_device_runtime_action_set_to_qwen(self):
        class RuntimeActionObserver(FakeSceneObserver):
            supports_runtime_action_contract = True

            def __init__(self, scenes):
                super().__init__(scenes)
                self.available_action_sets = []

            def observe_with_decision(
                self,
                *,
                frames,
                goal_context=None,
                available_action_kinds=None,
            ):
                self.available_action_sets.append(frozenset(available_action_kinds or ()))
                return super().observe_with_decision(frames=frames, goal_context=goal_context)

        observer = RuntimeActionObserver([scene("fresh")])
        adapter = self._adapter(observer, FakeRobot())

        adapter.capture_scene(
            goal(),
            evidence_dir=None,
            prefix="before_step_1",
        )

        self.assertEqual(
            [adapter.supported_action_kinds()],
            observer.available_action_sets,
        )

    def test_diagnostic_write_failure_preserves_primary_observation_error(self):
        observer = RawFailureSceneObserver('{"broken":true}')
        with tempfile.TemporaryDirectory() as temp, patch(
            "agent.infrastructure.generic_action_adapter._persist_qwen_failure_diagnostic",
            side_effect=OSError("disk unavailable"),
        ):
            with self.assertRaisesRegex(
                GenericActionAdapterError, "目标精查严格协议拒绝"
            ):
                self._adapter(observer, FakeRobot()).capture_scene(
                    goal(),
                    evidence_dir=Path(temp),
                    prefix="before_step_1",
                )

        self.assertEqual("OSError", observer.last_diagnostics["diagnostic_persistence_error"])

    def test_public_execute_irreparable_observation_never_calls_robot(self):
        self._assert_public_observation_failure_before_robot(
            ["not-json"]
        )

    def test_public_execute_duplicate_key_observation_never_calls_robot(self):
        valid = compact_scene_raw()
        duplicate = valid.replace(
            '"summary":',
            '"summary":"duplicate","summary":',
            1,
        )
        self._assert_public_observation_failure_before_robot(
            [duplicate]
        )

    def test_reports_only_callable_device_actions(self):
        adapter = self._adapter(FakeSceneObserver([]), FakeRobot())

        supported = adapter.supported_action_kinds()

        self.assertIn("tap_semantic", supported)
        self.assertIn("reveal_system_navigation", supported)
        self.assertIn("input_verified_text", supported)
        self.assertIn("press_enter", supported)
        self.assertIn("clear_verified_text", supported)
        self.assertIn("long_press", supported)
        self.assertIn("drag", supported)
        self.assertIn("back", supported)
        self.assertIn("home", supported)
        self.assertIn("wait_for_change", supported)
        self.assertIn("scroll", supported)
        self.assertIn("swipe_element", supported)

    def test_optional_camera_alignment_metadata_does_not_veto_ordinary_action(self):
        cases = (
            ("omitted", CameraAlignmentFacts()),
            (
                "low_model_confidence",
                CameraAlignmentFacts(
                    camera_layout_orientation="portrait",
                    phone_content_rotation="upright",
                    confidence=0.01,
                    evidence=("模型仅提供诊断性方向描述",),
                ),
            ),
            (
                "model_layout_mismatch",
                CameraAlignmentFacts(
                    camera_layout_orientation="landscape",
                    phone_content_rotation="unknown",
                    confidence=0.0,
                    evidence=(),
                ),
            ),
            (
                "model_reports_rotated_content",
                CameraAlignmentFacts(
                    camera_layout_orientation="portrait",
                    phone_content_rotation="rotated_90",
                    confidence=0.99,
                    evidence=("模型诊断为旋转画面",),
                ),
            ),
        )
        frames = tuple(Image.new("RGB", (540, 960), "gray") for _ in range(4))

        class PreserveOptionalAlignmentObserver(FakeSceneObserver):
            def _next_scene(self, *, goal_context=None):
                self.calls += 1
                self.goal_contexts.append(goal_context)
                result = self.scenes.pop(0)
                if isinstance(result, BaseException):
                    raise result
                return result

        for label, alignment in cases:
            with self.subTest(label=label):
                planned = scene("planned", camera_alignment=alignment)
                after = scene("after", screen_id="app_home", element_id="after")
                robot = FakeRobot()
                adapter = GenericSingleActionAdapter(
                    capture=SequenceCapture(["gray"] * 4 + ["white"] * 4),
                    observer=PreserveOptionalAlignmentObserver([after]),
                    robot=robot,
                    frame_interval=0,
                    post_action_settle=0,
                )

                result = adapter.execute(
                    requested_action=SemanticAction(
                        node_id="ordinary-tap",
                        action="tap_semantic",
                        params={"element_id": "e1", "target": "app_icon"},
                    ),
                    planned_scene=planned,
                    planned_frames=frames,
                    goal=goal(),
                    confirmed=True,
                )

                self.assertEqual(1, result.physical_actions)
                self.assertEqual([("tap", 300, 400)], robot.actions)
                self.assertEqual("portrait", result.orientation_credential.camera_layout_orientation)
                self.assertEqual(1.0, result.orientation_credential.confidence)

    def test_reveal_system_navigation_calls_one_dedicated_robot_action(self):
        hidden = SystemUIFacts(
            immersive_or_fullscreen=True,
            navigation_bar_visible=False,
        )
        visible = SystemUIFacts(
            immersive_or_fullscreen=True,
            navigation_bar_visible=True,
        )
        planned = scene("planned", system_ui=hidden)
        after = scene("planned", system_ui=visible)
        robot = FakeRobot()
        adapter = GenericSingleActionAdapter(
            capture=SequenceCapture(["gray"] * 4 + ["white"] * 4),
            observer=FakeSceneObserver([after]),
            robot=robot,
            frame_interval=0,
            post_action_settle=0,
        )

        result = adapter.execute(
            requested_action=SemanticAction(
                node_id="generic_step_1",
                action="reveal_system_navigation",
                params={

                },
            ),
            planned_scene=planned,
            planned_frames=tuple(
                Image.new("RGB", (540, 960), "gray") for _ in range(4)
            ),
            goal=goal(),
            confirmed=True,
        )

        self.assertEqual([("reveal_system_navigation",)], robot.actions)
        self.assertEqual(1, result.physical_actions)

    def test_reveal_system_navigation_leaves_summary_only_result_to_qwen_without_fallback(self):
        hidden = SystemUIFacts(
            immersive_or_fullscreen=True,
            navigation_bar_visible=False,
        )
        planned = scene("planned", system_ui=hidden)
        invalid_after = scene(
            "after",
            screen_id="navigation_bar_visible_summary_only",
        )
        robot = FakeRobot()
        adapter = GenericSingleActionAdapter(
            capture=SequenceCapture(["gray"] * 4 + ["white"] * 8),
            observer=FakeSceneObserver([invalid_after, invalid_after]),
            robot=robot,
            frame_interval=0,
            post_action_settle=0,
        )

        result = adapter.execute(
                requested_action=SemanticAction(
                    node_id="generic_step_1",
                    action="reveal_system_navigation",
                    params={},
                ),
                planned_scene=planned,
                planned_frames=tuple(
                    Image.new("RGB", (540, 960), "gray") for _ in range(4)
                ),
                goal=goal(),
                confirmed=True,
            )

        self.assertEqual([("reveal_system_navigation",)], robot.actions)
        self.assertEqual("finish", result.after_model_decision["status"])
        self.assertEqual((), result.controller_transition_evidence)

    def test_confirmed_home_executes_exactly_once_and_reobserves(self):
        planned = scene(
            "planned",
            screen_id="settings_home",
            element_id="settings-title",
            app_id="settings",
        )
        fresh = scene(
            "before",
            screen_id="settings_home",
            element_id="settings-title",
            app_id="settings",
        )
        after = scene(
            "after",
            screen_id="android_home",
            element_id="launcher-icon",
            app_id="launcher",
        )
        observer = FakeSceneObserver([fresh, after])
        robot = FakeRobot()
        action = SemanticAction(
            node_id="return-to-launcher",
            action="home",
            params={},
        )

        result = self._adapter(observer, robot).execute(
            requested_action=action,
            planned_scene=planned,
            goal=goal(),
            confirmed=True,
        )

        self.assertEqual([("home",)], robot.actions)
        self.assertEqual("home", result.resolved_action.kind)
        self.assertEqual(1, result.physical_actions)
        self.assertEqual(2, observer.calls)
        self.assertEqual(
            "single_step_scene_orientation",
            result.orientation_credential.source,
        )

    def test_fixed_system_home_uses_frame_binding_when_app_content_is_hidden(self):
        class ExactAlignmentObserver(FakeSceneObserver):
            def _next_scene(self, *, goal_context=None):
                self.calls += 1
                self.goal_contexts.append(goal_context)
                result = self.scenes.pop(0)
                if isinstance(result, BaseException):
                    raise result
                return result

        hidden_content_alignment = CameraAlignmentFacts(
            camera_layout_orientation="portrait",
            phone_content_rotation="unknown",
            confidence=0.95,
            evidence=("中央App内容被隐私遮罩，当前完整画布保持稳定",),
        )
        planned = scene(
            "planned-hidden-content",
            screen_id="unknown",
            app_id="unknown",
            camera_alignment=hidden_content_alignment,
        )
        fresh = scene(
            "fresh-hidden-content",
            screen_id="unknown",
            app_id="unknown",
            camera_alignment=hidden_content_alignment,
        )
        after = scene(
            "launcher-after-fixed-home",
            screen_id="android_home",
            app_id="launcher",
        )
        observer = ExactAlignmentObserver([fresh, after])
        robot = FakeRobot()

        result = self._adapter(observer, robot).execute(
            requested_action=SemanticAction(
                node_id="fixed-system-home",
                action="home",
                params={

                },
            ),
            planned_scene=planned,
            goal=goal(),
            confirmed=True,
        )

        self.assertEqual([("home",)], robot.actions)
        self.assertEqual(1, result.physical_actions)
        self.assertEqual(
            "unknown",
            result.orientation_credential.phone_content_rotation,
        )
        self.assertEqual(2, observer.calls)

    def test_home_records_single_click_transport_receipt(self):
        planned = scene("planned", screen_id="settings_home", app_id="settings")
        fresh = scene("before", screen_id="settings_home", app_id="settings")
        after = scene("after", screen_id="android_home", app_id="launcher")
        observer = FakeSceneObserver([fresh, after])
        robot = ClickReceiptRobot()

        result = self._adapter(observer, robot).execute(
            requested_action=SemanticAction(
                node_id="return-to-launcher",
                action="home",
                params={},
            ),
            planned_scene=planned,
            goal=goal(),
            confirmed=True,
        )

        self.assertEqual([("home",)], robot.actions)
        self.assertTrue(result.hardware_receipt["input_events_dispatched"])
        self.assertFalse(result.hardware_receipt["mechanical_contact_ack"])
        self.assertIsNone(robot.consume_last_click_receipt())

    def test_invalid_home_click_receipt_stops_before_fresh_screenshot_observation(self):
        planned = scene("planned", screen_id="settings_home", app_id="settings")
        fresh = scene("before", screen_id="settings_home", app_id="settings")
        observer = FakeSceneObserver([fresh])
        robot = ClickReceiptRobot(valid=False)

        with self.assertRaisesRegex(GenericActionAdapterError, "单击事件派发凭据"):
            self._adapter(observer, robot).execute(
                requested_action=SemanticAction(
                    node_id="return-to-launcher",
                    action="home",
                    params={},
                ),
                planned_scene=planned,
                goal=goal(),
                confirmed=True,
            )

        self.assertEqual([("home",)], robot.actions)
        self.assertEqual(1, observer.calls)

    def test_confirmed_tap_executes_exactly_once_and_reobserves(self):
        planned = scene("planned", bounds=(0.1, 0.2, 0.3, 0.4))
        fresh = scene("before", bounds=(0.11, 0.21, 0.31, 0.41))
        after = scene("after", screen_id="app_home", element_id="after")
        observer = FakeSceneObserver([fresh, after])
        robot = FakeRobot()
        action = SemanticAction(
            node_id="generic_step_1",
            action="tap_semantic",
            params={
                "element_id": "e1",
                "target": "app_icon",
                "role": "icon",
                "label": "设置",
                "tap_point": (0.21, 0.31),
            },
        )
        with tempfile.TemporaryDirectory() as temp:
            result = self._adapter(observer, robot).execute(
                requested_action=action,
                planned_scene=planned,
                goal=goal(),
                confirmed=True,
                evidence_dir=Path(temp),
            )
        self.assertEqual(robot.actions, [("tap", 210, 310)])
        self.assertEqual(result.physical_actions, 1)
        self.assertEqual(observer.calls, 2)


    def test_typed_field_identity_bridges_optional_visual_wording_and_mode_drift(self):
        states = {
            "focused": True,
            "fully_visible": True,
            "value": "first\n",
            "input_field_id": "input_field_1",
            "input_multiline": True,
            "keyboard_layout": "qwerty",
            "keyboard_input_mode": "direct_latin",
            "goal_relevant": True,
        }
        before = UIScene(
            app_id="unknown",
            screen_id="通用动作具机验收页",
            summary="同一多行输入框",
            elements=(
                UIElement(
                    element_id="before-field",
                    role="input",
                    meaning="application_text_input",
                    label="first",
                    bounds=(0.14, 0.52, 0.71, 0.60),
                    confidence=1.0,
                    states=states,
                ),
            ),
            stable=True,
            confidence=1.0,
            fingerprint="before",
            camera_alignment=aligned_camera_facts(),
        )
        after = replace(
            before,
            screen_id="通用动作真机验收页",
            fingerprint="after",
            elements=(
                replace(
                    before.elements[0],
                    element_id="after-field",
                    bounds=(0.13, 0.51, 0.72, 0.61),
                    label="first\nsecond",
                    states={**states, "value": "first\nsecond"},
                ),
            ),
        )
        resolved = ResolvedSemanticAction(
            node_id="append-second",
            kind="input_verified_text",
            text="first\nsecond",
            input_fragment="second",
            prior_input_value="first\n",
            expected_input_value="first\nsecond",
            input_field_id="input_field_1",
            target_element_id="before-field",
            before_fingerprint=before.fingerprint,
        )

        UniversalActionController().verify_after_action(resolved, before, after)

        different_field = replace(
            after,
            elements=(
                replace(
                    after.elements[0],
                    states={
                        **after.elements[0].states,
                        "input_field_id": "input_field_2",
                    },
                ),
            ),
        )
        changed_mode = replace(
            after,
            elements=(
                replace(
                    after.elements[0],
                    states={
                        **after.elements[0].states,
                        "keyboard_input_mode": "chinese_pinyin",
                    },
                ),
            ),
        )
        for unsafe_before, unsafe_after in ((before, different_field),):
            with self.subTest(
                before_app_id=unsafe_before.app_id,
                app_id=unsafe_after.app_id,
                states=unsafe_after.elements[0].states,
            ):
                with self.assertRaises(UniversalActionError):
                    UniversalActionController().verify_after_action(
                        resolved,
                        unsafe_before,
                        unsafe_after,
                    )

        UniversalActionController().verify_after_action(resolved, before, changed_mode)



    def test_direct_input_and_clear_use_current_exact_value_without_persistent_lineage(self):
        def input_scene(fingerprint, value, *, goal_relevant=True):
            return UIScene(
                app_id="generic_app",
                screen_id="editor",
                summary="唯一聚焦输入框",
                elements=(
                    UIElement(
                        element_id="field",
                        role="input",
                        meaning="application_text_input",
                        label=value,
                        bounds=(0.13, 0.54, 0.69, 0.61),
                        confidence=1.0,
                        states={
                            "focused": True,
                            "fully_visible": True,
                            "value": value,
                            "keyboard_layout": "qwerty",
                            "keyboard_input_mode": "direct_latin",
                            "keyboard_geometry": TEST_QWERTY_GEOMETRY,
                            "input_field_id": "field",
                            "goal_relevant": goal_relevant,
                        },
                        evidence=(f"应用输入框当前文字：{value}",),
                    ),
                ),
                stable=True,
                confidence=1.0,
                fingerprint=fingerprint,
                camera_alignment=aligned_camera_facts(),
            )

        before = input_scene("before", "")
        after = input_scene("after", "longinput")
        robot = FakeRobot()
        input_action = SemanticAction(
            node_id="type-segment",
            action="input_verified_text",
            params={
                "element_id": "field",
                "target": "application_text_input",
                "role": "input",
                "label": "",
                "states": dict(before.elements[0].states),
                "text": "longinput",
            },
        )
        result = self._adapter(FakeSceneObserver([before, after]), robot).execute(
            requested_action=input_action,
            planned_scene=before,
            goal=goal(),
            confirmed=True,
        )
        self.assertEqual("executed", result.action_outcome)

        clear_after = input_scene("cleared", "", goal_relevant=False)
        clear_result = self._adapter(FakeSceneObserver([after, clear_after]), FakeRobot()).execute(
            requested_action=SemanticAction(
                node_id="clear-segment",
                action="clear_verified_text",
                params={
                    "element_id": "field",
                    "target": "application_text_input",
                    "role": "input",
                    "label": "longinput",
                    "states": dict(after.elements[0].states),
                },
            ),
            planned_scene=after,
            goal=goal(),
            confirmed=True,
        )
        self.assertEqual("executed", clear_result.action_outcome)

        wrong_after = input_scene("wrong-after", "longinpuw")
        mismatch_robot = FakeRobot()
        with self.assertRaisesRegex(GenericActionAdapterError, "文字不匹配") as caught:
            self._adapter(FakeSceneObserver([before, wrong_after]), mismatch_robot).execute(
                requested_action=input_action,
                planned_scene=before,
                goal=goal(),
                confirmed=True,
            )
        self.assertEqual(1, caught.exception.physical_actions)
        self.assertEqual(1, len(mismatch_robot.actions))

    def test_confirmed_input_uses_qwen_selected_id_before_exact_post_receipt(self):
        def input_scene(
            fingerprint,
            element_id,
            meaning,
            label,
            value,
            *,
            app_id="browser",
            screen_id="search",
            bounds=(0.1, 0.1, 0.9, 0.2),
            keyboard_input_mode="direct_latin",
        ):
            return UIScene(
                app_id=app_id,
                screen_id=screen_id,
                summary="唯一输入框",
                elements=(
                    UIElement(
                        element_id=element_id,
                        role="input",
                        meaning=meaning,
                        label=label,
                        bounds=bounds,
                        confidence=0.98,
                        states={
                            "focused": True,
                            "value": value,
                            "keyboard_layout": "qwerty",
                            "keyboard_input_mode": keyboard_input_mode,
                            "keyboard_geometry": TEST_QWERTY_GEOMETRY,
                            "goal_relevant": True,
                        },
                    ),
                ),
                stable=True,
                confidence=0.98,
                fingerprint=fingerprint,
            )

        planned = input_scene(
            "planned", "planned-input", "target_text_input", "", ""
        )
        fresh = input_scene(
            "before", "planned-input", "target_text_input", "", ""
        )
        after = input_scene(
            "after", "audited-input", "application_text_input", "agent", "agent"
        )
        robot = FakeRobot()
        result = self._adapter(FakeSceneObserver([fresh, after]), robot).execute(
            requested_action=SemanticAction(
                node_id="input-alias",
                action="input_verified_text",
                params={
                    "element_id": "planned-input",
                    "target": "target_text_input",
                    "role": "input",
                    "label": "",
                    "states": {
                        "focused": True,
                        "value": "",
                        "keyboard_layout": "qwerty",
                        "keyboard_input_mode": "direct_latin",
                        "keyboard_geometry": TEST_QWERTY_GEOMETRY,
                        "goal_relevant": True,
                    },
                    "text": "agent",

                },
            ),
            planned_scene=planned,
            goal=goal(),
            confirmed=True,
        )

        self.assertEqual([("input", "agent")], robot.actions)
        self.assertEqual("executed", result.action_outcome)

    def test_confirmed_input_accepts_identity_degradation_to_unknown_for_same_input(self):
        def input_scene(fingerprint, app_id, screen_id, value):
            return UIScene(
                app_id=app_id,
                screen_id=screen_id,
                summary="本地输入页",
                elements=(
                    UIElement(
                        element_id="audited-input",
                        role="input",
                        meaning="target_text_input",
                        label="验收输入框",
                        bounds=(0.12, 0.38, 0.87, 0.47),
                        confidence=0.98,
                        states={
                            "focused": True,
                            "value": value,
                            "keyboard_layout": "qwerty",
                            "keyboard_input_mode": "direct_latin",
                            "keyboard_geometry": TEST_QWERTY_GEOMETRY,
                            "goal_relevant": True,
                        },
                    ),
                ),
                stable=True,
                confidence=0.98,
                fingerprint=fingerprint,
                camera_alignment=aligned_camera_facts(),
            )

        planned = input_scene("planned", "current_foreground", "通用动作真机验收页", "")
        fresh = input_scene("before", "current_foreground", "通用动作真机验收页", "")
        after = input_scene("after", "unknown", "unknown", "agent")
        robot = FakeRobot()

        result = self._adapter(FakeSceneObserver([fresh, after]), robot).execute(
            requested_action=SemanticAction(
                node_id="input-identity-degradation",
                action="input_verified_text",
                params={
                    "element_id": "audited-input",
                    "target": "target_text_input",
                    "role": "input",
                    "label": "验收输入框",
                    "states": {
                        "focused": True,
                        "value": "",
                        "keyboard_layout": "qwerty",
                        "keyboard_input_mode": "direct_latin",
                        "keyboard_geometry": TEST_QWERTY_GEOMETRY,
                        "goal_relevant": True,
                    },
                    "text": "agent",

                },
            ),
            planned_scene=planned,
            goal=goal(),
            confirmed=True,
        )

        self.assertEqual([("input", "agent")], robot.actions)
        self.assertEqual("executed", result.action_outcome)

    def test_companion_unicode_input_uses_authorized_transport_once_then_visual_verifies(self):
        before = UIScene(
            app_id="sample.app",
            screen_id="editor",
            summary="唯一 typed 输入框已聚焦",
            elements=(UIElement(element_id="field", role="input", meaning="application_text_input",
                label="前缀", bounds=(0.1, 0.2, 0.9, 0.3), confidence=0.99, states={
                    "focused": True, "goal_relevant": True, "fully_visible": True,
                    "value": "前缀", "input_field_id": "field_primary", "ime_preedit_text": "",
                }, evidence=("应用输入框当前文字：前缀",)),),
            stable=True,
            confidence=0.99,
            fingerprint="before-companion",
            camera_alignment=aligned_camera_facts(),
        )
        expected = "前缀🙂\nsecond@例"
        after = replace(before, fingerprint="after-companion", elements=(replace(before.elements[0],
            label=expected, states={**before.elements[0].states, "value": expected},
            evidence=(f"应用输入框当前文字：{expected}",)),))
        action = SemanticAction(node_id="companion-unicode", action="input_verified_text", params={
            "element_id": "field", "target": "application_text_input", "role": "input", "label": "前缀",
            "states": before.elements[0].states, "text": expected, "text_transport": "adb_keyboard",
            "input_field_id": "field_primary", "prior_input_value": "前缀",
            "input_fragment": "🙂\nsecond@例", "expected_input_value": expected,

        })
        authority = ConfirmationAuthority(session_id="session-1", task_id="task-1", device_id="test-device",
            revision=3, step_id="subgoal-1", effect_ids=(), observation_id="observation-1",
            fingerprint=before.fingerprint, decision_node_id=action.node_id,
            action_digest=canonical_digest(action.to_dict()), consumed=True)
        transport = FakeAdbKeyboardTextTransport()
        robot = FakeRobot()

        result = self._adapter(FakeSceneObserver([before, after, after]), robot,
            text_transport=transport).execute(requested_action=action, planned_scene=before, goal=goal(),
            confirmed=True, action_authority=authority)

        self.assertEqual("executed", result.action_outcome)
        self.assertEqual(1, result.physical_actions)
        self.assertEqual([], robot.actions)
        self.assertEqual(1, len(transport.calls))
        self.assertEqual(("append_text", "🙂\nsecond@例"), (transport.calls[0][0], transport.calls[0][2]))
        minted = transport.minted[0]
        self.assertEqual("field_primary", minted["input_field_id"])
        self.assertEqual(before.fingerprint, minted["observation_fingerprint"])
        self.assertNotIn(expected, json.dumps(result.execution_metadata, ensure_ascii=False))
        self.assertEqual("accepted", result.execution_metadata["transport_status"])

    def test_adb_keyboard_transport_accepts_focused_typed_field_without_visible_keyboard(self):
        before = UIScene(
            app_id="sample.app", screen_id="editor", summary="唯一 typed 输入框可见",
            elements=(UIElement(element_id="field", role="input", meaning="application_text_input",
                label="", bounds=(0.1, 0.2, 0.9, 0.3), confidence=0.99, states={
                    "goal_relevant": True, "fully_visible": True, "focused": True,
                    "soft_keyboard_visible": False,
                    "value": "", "input_field_id": "field_primary", "ime_preedit_text": "",
                }),), stable=True, confidence=0.99, fingerprint="before-adb-no-keyboard",
            camera_alignment=aligned_camera_facts(),
        )
        expected = "ADB测试？你好"
        after = replace(before, fingerprint="after-adb-no-keyboard", elements=(replace(before.elements[0],
            label=expected, states={**before.elements[0].states, "value": expected}),))
        action = SemanticAction(node_id="adb-unicode", action="input_verified_text", params={
            "element_id": "field", "target": "application_text_input", "role": "input", "label": "",
            "states": before.elements[0].states, "text": expected, "text_transport": "adb_keyboard",
            "input_field_id": "field_primary", "prior_input_value": "",
            "input_fragment": expected, "expected_input_value": expected,
        })
        authority = ConfirmationAuthority(session_id="session-1", task_id="task-1", device_id="test-device",
            revision=3, step_id="subgoal-1", effect_ids=(), observation_id="observation-1",
            fingerprint=before.fingerprint, decision_node_id=action.node_id,
            action_digest=canonical_digest(action.to_dict()), consumed=True)
        transport = FakeAdbKeyboardTextTransport()
        robot = FakeRobot()

        result = self._adapter(FakeSceneObserver([before, after, after]), robot,
            text_transport=transport).execute(requested_action=action, planned_scene=before, goal=goal(),
            confirmed=True, action_authority=authority)

        self.assertEqual("executed", result.action_outcome)
        self.assertEqual(("append_text", expected), (transport.calls[0][0], transport.calls[0][2]))

    def test_confirmed_input_accepts_optional_page_wording_change_with_exact_value(self):
        before = UIScene(
            app_id="browser",
            screen_id="page-a",
            summary="输入页",
            elements=(
                UIElement(
                    element_id="field",
                    role="input",
                    meaning="target_text_input",
                    label="输入框",
                    bounds=(0.1, 0.1, 0.9, 0.2),
                    confidence=0.98,
                    states={
                        "focused": True,
                        "value": "",
                        "keyboard_layout": "qwerty",
                        "keyboard_input_mode": "direct_latin",
                        "keyboard_geometry": TEST_QWERTY_GEOMETRY,
                        "goal_relevant": True,
                    },
                ),
            ),
            stable=True,
            confidence=0.98,
            fingerprint="before",
            camera_alignment=aligned_camera_facts(),
        )
        after = replace(
            before,
            screen_id="page-b",
            fingerprint="after",
            elements=(replace(before.elements[0], states={**before.elements[0].states, "value": "agent"}),),
        )
        robot = FakeRobot()

        result = self._adapter(FakeSceneObserver([before, after]), robot).execute(
            requested_action=SemanticAction(
                node_id="accept-page-wording-change",
                action="input_verified_text",
                params={
                    "element_id": "field",
                    "target": "target_text_input",
                    "role": "input",
                    "label": "输入框",
                    "states": before.elements[0].states,
                    "text": "agent",
                },
            ),
            planned_scene=before,
            goal=goal(),
            confirmed=True,
        )
        self.assertEqual("executed", result.action_outcome)
        self.assertEqual(1, result.physical_actions)


    def test_execution_result_keeps_exact_four_verified_after_frames(self):
        gray = Image.new("RGB", (540, 960), "gray")
        white = Image.new("RGB", (540, 960), "white")
        before_fingerprint = local_frame_fingerprint(gray)
        after_fingerprint = local_frame_fingerprint(white)
        planned = scene(before_fingerprint)
        after = scene(
            after_fingerprint,
            screen_id="app_home",
            element_id="after",
        )
        observer = FakeSceneObserver([after])
        robot = FakeRobot()
        adapter = GenericSingleActionAdapter(
            capture=SequenceCapture(["gray"] * 4 + ["white"] * 4),
            observer=observer,
            robot=robot,
            frame_interval=0,
            post_action_settle=0,
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
                planned_frames=(gray, gray.copy(), gray.copy(), gray.copy()),
                goal=goal(),
                confirmed=True,
                evidence_dir=Path(temp),
            )

        self.assertEqual(4, len(result.after_frames))
        self.assertEqual(4, len(result.before_frames))
        self.assertIsInstance(result.after_frames, tuple)
        self.assertIsInstance(result.before_frames, tuple)
        self.assertEqual(4, len(result.before_frame_paths))
        self.assertTrue(
            all(frame.getpixel((0, 0)) == (255, 255, 255) for frame in result.after_frames)
        )
        self.assertEqual(4, len(result.after_frame_paths))
        credential = result.orientation_credential
        self.assertIsNotNone(credential)
        _validate_live_promotion_source(
            report={
                "candidate_action": "tap_semantic",
                "device_id": credential.device_id,
                "physical_actions": 1,
                "action_outcome": "executed",
                "execution": result.to_dict(),
                "before_frame_paths": list(result.before_frame_paths),
            },
            orientation_credential=credential,
            execution_result=result,
        )

    def test_after_frame_fingerprint_matches_after_scene(self):
        gray = Image.new("RGB", (540, 960), "gray")
        white = Image.new("RGB", (540, 960), "white")
        before_fingerprint = local_frame_fingerprint(gray)
        after_fingerprint = local_frame_fingerprint(white)
        adapter = GenericSingleActionAdapter(
            capture=SequenceCapture(["gray"] * 4 + ["white"] * 4),
            observer=FakeSceneObserver(
                [scene(after_fingerprint, screen_id="app_home", element_id="after")]
            ),
            robot=FakeRobot(),
            frame_interval=0,
            post_action_settle=0,
        )
        action = SemanticAction(
            node_id="generic_step_1",
            action="tap_semantic",
            params={"element_id": "e1", "target": "app_icon"},
        )

        result = adapter.execute(
            requested_action=action,
            planned_scene=scene(before_fingerprint),
            planned_frames=(gray, gray.copy(), gray.copy(), gray.copy()),
            goal=goal(),
            confirmed=True,
        )

        selected_fingerprint = local_frame_fingerprint(result.after_frames[0])
        self.assertEqual(result.after_scene.fingerprint, selected_fingerprint)

    def test_changed_non_authoritative_state_does_not_reselect_or_veto(self):
        planned = UIScene(
            app_id="unknown",
            screen_id="android_home",
            summary="主屏幕",
            elements=(
                UIElement(
                    element_id="e1",
                    role="icon",
                    meaning="app_icon",
                    label="设置",
                    bounds=(0.2, 0.3, 0.4, 0.5),
                    confidence=0.96,
                    states={"enabled": True},
                ),
            ),
            fingerprint="planned",
        )
        fresh = UIScene(
            app_id="unknown",
            screen_id="android_home",
            summary="主屏幕",
            elements=(
                UIElement(
                    element_id="e1",
                    role="icon",
                    meaning="app_icon",
                    label="设置",
                    bounds=(0.2, 0.3, 0.4, 0.5),
                    confidence=0.96,
                    states={"enabled": False},
                ),
            ),
            fingerprint="fresh",
        )
        robot = FakeRobot()
        result = self._adapter(
            FakeSceneObserver([fresh, scene("after", screen_id="app_home")]),
            robot,
        ).execute(
            requested_action=SemanticAction(
                node_id="generic_step_1",
                action="tap_semantic",
                params={"element_id": "e1", "target": "app_icon", "tap_point": (0.3, 0.4)},
            ),
            planned_scene=planned,
            goal=goal(),
            confirmed=True,
        )
        self.assertEqual([("tap", 300, 400)], robot.actions)
        self.assertEqual("executed", result.action_outcome)

    def test_each_execution_uses_unique_evidence_paths(self):
        action = SemanticAction(
            node_id="generic_step_1",
            action="tap_semantic",
            params={"element_id": "e1", "target": "app_icon"},
        )
        with tempfile.TemporaryDirectory() as temp:
            evidence_dir = Path(temp)
            first = self._adapter(
                FakeSceneObserver(
                    [scene("same"), scene("after-1", screen_id="app_home")]
                ),
                FakeRobot(),
            ).execute(
                requested_action=action,
                planned_scene=scene("same"),
                goal=goal(),
                confirmed=True,
                evidence_dir=evidence_dir,
            )
            first_bytes = {path: Path(path).read_bytes() for path in first.evidence}
            second = self._adapter(
                FakeSceneObserver(
                    [scene("same"), scene("after-2", screen_id="app_home")]
                ),
                FakeRobot(),
            ).execute(
                requested_action=action,
                planned_scene=scene("same"),
                goal=goal(),
                confirmed=True,
                evidence_dir=evidence_dir,
            )

            self.assertTrue(set(first.evidence).isdisjoint(second.evidence))
            self.assertEqual(
                first_bytes,
                {path: Path(path).read_bytes() for path in first.evidence},
            )

    def test_matching_local_frames_override_model_screen_id_wording_drift(self):
        planned = scene("planned", screen_id="generic_action_verification_page")
        after = scene("after", screen_id="blue_endpoint_visible")
        observer = FakeSceneObserver([after])
        robot = FakeRobot()
        capture = SequenceCapture(["gray"] * 4 + ["white"] * 4)
        adapter = GenericSingleActionAdapter(
            capture=capture,
            observer=observer,
            robot=robot,
            frame_interval=0,
            post_action_settle=0,
        )
        action = SemanticAction(
            node_id="generic_step_1",
            action="scroll",
            params={"direction": "up", },
        )

        result = adapter.execute(
            requested_action=action,
            planned_scene=planned,
            planned_frames=tuple(Image.new("RGB", (540, 960), "gray") for _ in range(4)),
            goal=goal(),
            confirmed=True,
        )

        self.assertEqual([("swipe", "up")], robot.actions)
        self.assertEqual(1, result.physical_actions)
        self.assertEqual(1, observer.calls)

    def test_physical_gate_canvas_error_preserves_zero_action_and_evidence(self):
        planned = scene("planned", screen_id="generic_action_verification_page")
        observer = FakeSceneObserver([])
        robot = PhysicalGateCanvasErrorRobot()
        adapter = GenericSingleActionAdapter(
            capture=SequenceCapture(["gray"] * 4),
            observer=observer,
            robot=robot,
            frame_interval=0,
            post_action_settle=0,
        )
        action = SemanticAction(
            node_id="generic_step_1",
            action="scroll",
            params={"direction": "up", },
        )

        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(
                GenericActionAdapterError,
                "共享物理执行门.*画布尺寸不匹配",
            ) as caught:
                adapter.execute(
                    requested_action=action,
                    planned_scene=planned,
                    planned_frames=tuple(
                        Image.new("RGB", (540, 960), "gray") for _ in range(4)
                    ),
                    goal=goal(),
                    confirmed=True,
                    evidence_dir=Path(temp),
                )
            self.assertEqual(4, len(caught.exception.evidence))
            self.assertTrue(all(Path(item).is_file() for item in caught.exception.evidence))
        self.assertEqual(0, caught.exception.physical_actions)
        self.assertEqual([], robot.actions)

    def test_matching_planned_frames_require_fresh_geometry_interpretation(self):
        planned = scene("planned")
        fresh = scene("fresh")
        after = scene("after", screen_id="app_home", element_id="after")
        observer = FakeSceneObserver([fresh, after])
        robot = FakeRobot()
        adapter = GenericSingleActionAdapter(
            capture=SequenceCapture(["gray"] * 4 + ["white"] * 4),
            observer=observer,
            robot=robot,
            frame_interval=0,
            post_action_settle=0,
        )

        result = adapter.execute(
            requested_action=SemanticAction(
                node_id="generic_step_1",
                action="tap_semantic",
                params={"element_id": "e1", "target": "app_icon"},
            ),
            planned_scene=planned,
            planned_frames=tuple(
                Image.new("RGB", (540, 960), "gray") for _ in range(4)
            ),
            goal=goal(),
            confirmed=True,
        )

        self.assertEqual(1, result.physical_actions)
        self.assertEqual([("tap", 300, 400)], robot.actions)
        self.assertEqual(1, observer.calls)
        self.assertEqual("planned", result.before_scene.fingerprint)

    def test_matching_frame_identity_reuses_original_geometry(self):
        planned = scene("planned", bounds=(0.12, 0.46, 0.58, 0.51))
        fresh = scene("fresh", bounds=(0.12, 0.225, 0.45, 0.265))
        observer = FakeSceneObserver(
            [scene("after", screen_id="app_home", element_id="after")]
        )
        robot = FakeRobot()
        adapter = GenericSingleActionAdapter(
            capture=SequenceCapture(["gray"] * 4),
            observer=observer,
            robot=robot,
            frame_interval=0,
            post_action_settle=0,
        )

        result = adapter.execute(
            requested_action=SemanticAction(
                node_id="generic_step_1",
                action="tap_semantic",
                params={"element_id": "e1", "target": "app_icon", "tap_point": (0.35, 0.485)},
            ),
            planned_scene=planned,
            planned_frames=tuple(
                Image.new("RGB", (540, 960), "gray") for _ in range(4)
            ),
            goal=goal(),
            confirmed=True,
        )

        self.assertEqual([("tap", 350, 485)], robot.actions)
        self.assertEqual("planned", result.before_scene.fingerprint)
        self.assertEqual(1, observer.calls)

    def test_stable_frames_allow_input_field_bounds_drift_with_fresh_keyboard_geometry(self):
        sharp_frame = textured_phone_frame()
        blurred_frame = sharp_frame.filter(ImageFilter.GaussianBlur(2))
        states = {
            "goal_relevant": True,
            "fully_visible": True,
            "focused": True,
            "value": "",
            "keyboard_layout": "qwerty",
            "keyboard_input_mode": "direct_latin",
            "keyboard_case_mode": "lower",
            "keyboard_geometry": TEST_QWERTY_GEOMETRY,
        }

        def input_scene(fingerprint, bounds, *, value=""):
            return UIScene(
                app_id="wechat",
                screen_id="chat_file_transfer_helper",
                summary="同一会话中的唯一聚焦输入框和 QWERTY 键盘",
                elements=(
                    UIElement(
                        element_id="local_audited_input_1",
                        role="input",
                        meaning="application_text_input",
                        label="",
                        bounds=bounds,
                        confidence=1.0,
                        states={**states, "value": value},
                    ),
                ),
                stable=True,
                confidence=1.0,
                fingerprint=fingerprint,
                camera_alignment=aligned_camera_facts(),
            )

        planned = input_scene("planned", (0.14, 0.53, 0.69, 0.58))
        fresh = input_scene("fresh", (0.24, 0.623, 0.80, 0.68))
        target_text = "longinput2026abcdefghijklmnopqrstuvwxyz"
        first_segment = target_text
        after = input_scene(
            "after",
            (0.15, 0.53, 0.70, 0.60),
            value=first_segment,
        )
        observer = FakeSceneObserver([after])
        robot = FakeRobot()
        snapped_anchors = {
            key: list(value)
            for key, value in TEST_QWERTY_GEOMETRY["anchors"].items()
        }
        adapter = GenericSingleActionAdapter(
            capture=SequenceCapture(
                [sharp_frame] * 4 + [blurred_frame] * 4 + [sharp_frame] * 4
            ),
            observer=observer,
            robot=robot,
            frame_interval=0,
            post_action_settle=0,
        )

        result = adapter.execute(
            requested_action=SemanticAction(
                node_id="input-long-text",
                action="input_verified_text",
                params={
                    "element_id": "local_audited_input_1",
                    "target": "application_text_input",
                    "role": "input",
                    "label": "",
                    "states": states,
                    "text": target_text,

                },
            ),
            planned_scene=planned,
            planned_frames=tuple(
                sharp_frame.copy() for _ in range(4)
            ),
            goal=goal(),
            confirmed=True,
        )

        self.assertEqual([("input", first_segment)], robot.actions)
        self.assertEqual(1, result.physical_actions)
        self.assertEqual("executed", result.action_outcome)
        self.assertGreater(adapter.capture.calls, 8)
        self.assertEqual(1, observer.calls)
        self.assertEqual("planned", result.before_scene.fingerprint)
        self.assertFalse(hasattr(robot, "vision_type_text_with_layout"))

    def test_matching_frames_do_not_accept_second_model_input_mode_veto(self):
        planned_states = {
            "goal_relevant": True,
            "fully_visible": True,
            "focused": True,
            "value": "",
            "keyboard_layout": "qwerty",
            "keyboard_input_mode": "direct_latin",
            "keyboard_case_mode": "lower",
            "keyboard_geometry": TEST_QWERTY_GEOMETRY,
        }
        planned = UIScene(
            app_id="generic_app",
            screen_id="editor",
            summary="唯一聚焦输入框",
            elements=(
                UIElement(
                    element_id="input",
                    role="input",
                    meaning="application_text_input",
                    label="",
                    bounds=(0.14, 0.53, 0.69, 0.58),
                    confidence=1.0,
                    states=planned_states,
                ),
            ),
            stable=True,
            confidence=1.0,
            fingerprint="planned",
            camera_alignment=aligned_camera_facts(),
        )
        fresh = replace(
            planned,
            elements=(
                replace(
                    planned.elements[0],
                    bounds=(0.24, 0.623, 0.80, 0.68),
                    states={
                        **planned_states,
                        "keyboard_input_mode": "chinese_pinyin",
                    },
                ),
            ),
            fingerprint="fresh",
        )
        after = replace(
            planned,
            elements=(
                replace(
                    planned.elements[0],
                    states={**planned_states, "value": "agent"},
                ),
            ),
            fingerprint="after",
        )
        observer = FakeSceneObserver([after])
        robot = FakeRobot()
        adapter = GenericSingleActionAdapter(
            capture=SequenceCapture(["gray"] * 4),
            observer=observer,
            robot=robot,
            frame_interval=0,
            post_action_settle=0,
        )

        result = adapter.execute(
                requested_action=SemanticAction(
                    node_id="input-mode-changed",
                    action="input_verified_text",
                    params={
                        "element_id": "input",
                        "target": "application_text_input",
                        "role": "input",
                        "label": "",
                        "states": planned_states,
                        "text": "agent",
                    },
                ),
                planned_scene=planned,
                planned_frames=tuple(
                    Image.new("RGB", (540, 960), "gray") for _ in range(4)
                ),
                goal=goal(),
                confirmed=True,
            )

        self.assertEqual(1, result.physical_actions)
        self.assertEqual([("input", "agent")], robot.actions)

    def test_drag_reuses_single_step_endpoint_geometry(self):
        def drag_scene(fingerprint, source_bounds, destination_bounds):
            return UIScene(
                app_id="local.acceptance",
                screen_id="drag-board",
                summary="本地拖动验收页面",
                elements=(
                    UIElement(
                        element_id="source",
                        role="image",
                        meaning="draggable_purple_block",
                        label="起点",
                        bounds=source_bounds,
                        confidence=0.98,
                        states={"fully_visible": True},
                        evidence=("紫色圆角方块，中心写有起点二字",),
                    ),
                    UIElement(
                        element_id="destination",
                        role="container",
                        meaning="drop_target_green_zone",
                        label="绿色终点",
                        bounds=destination_bounds,
                        confidence=0.98,
                        states={"fully_visible": True},
                        evidence=("绿色虚线框区域，内部写有绿色终点",),
                    ),
                ),
                stable=True,
                confidence=0.98,
                fingerprint=fingerprint,
                camera_alignment=aligned_camera_facts(),
            )

        destination_bounds = (0.62, 0.66, 0.78, 0.82)
        planned = drag_scene(
            "planned",
            (0.12, 0.46, 0.58, 0.51),
            destination_bounds,
        )
        fresh = drag_scene(
            "fresh",
            (0.12, 0.225, 0.45, 0.265),
            destination_bounds,
        )
        audited_source = (0.18, 0.20, 0.32, 0.30)
        planned_audited = drag_scene(
            "planned",
            audited_source,
            destination_bounds,
        )
        fresh_audited = drag_scene(
            "fresh",
            audited_source,
            destination_bounds,
        )
        after = drag_scene(
            "after",
            (0.63, 0.68, 0.73, 0.78),
            destination_bounds,
        )
        observer = FakeSceneObserver([after])
        robot = FakeRobot()
        adapter = GenericSingleActionAdapter(
            capture=SequenceCapture(["gray"] * 4 + ["white"] * 4),
            observer=observer,
            robot=robot,
            frame_interval=0,
            post_action_settle=0,
        )

        result = adapter.execute(
            requested_action=SemanticAction(
                node_id="drag-audited",
                action="drag",
                params={
                    "source_element_id": "source",
                    "source_target": "draggable_purple_block",
                    "source_role": "image",
                    "source_label": "起点",
                    "source_states": {"fully_visible": True},
                    "destination_element_id": "destination",
                    "destination_target": "drop_target_green_zone",
                    "destination_role": "container",
                    "destination_label": "绿色终点",
                    "destination_states": {"fully_visible": True},

                },
            ),
            planned_scene=planned,
            planned_frames=tuple(
                Image.new("RGB", (540, 960), "gray") for _ in range(4)
            ),
            goal=goal(),
            confirmed=True,
        )

        self.assertEqual(
            [("drag", 350, 485, 700, 740)],
            robot.actions,
        )
        self.assertEqual(
            planned.get_element("source").bounds,
            result.before_scene.get_element("source").bounds,
        )
        self.assertEqual(1, result.physical_actions)

    def test_long_press_reuses_single_step_target_geometry(self):
        states = {"goal_relevant": True, "fully_visible": True}

        def long_press_scene(
            fingerprint,
            bounds,
            *,
            meaning="long_press_target_area",
            passed=False,
        ):
            return UIScene(
                app_id="local.acceptance",
                screen_id="long-press-board",
                summary="本地长按验收页面",
                elements=(
                    UIElement(
                        element_id="target",
                        role="button",
                        meaning=meaning,
                        label="长按我 · 不要移动",
                        bounds=bounds,
                        confidence=1.0,
                        states=states,
                        evidence=("黄色虚线框内逐字显示长按目标",),
                    ),
                ),
                overlays=(("long_press 验收通过",) if passed else ()),
                stable=True,
                confidence=1.0,
                fingerprint=fingerprint,
                camera_alignment=aligned_camera_facts(),
            )

        planned = long_press_scene("planned", (0.09, 0.53, 0.91, 0.76))
        fresh = long_press_scene(
            "fresh",
            (0.09, 0.47, 0.91, 0.69),
            meaning="interaction_zone",
        )
        audited_bounds = (0.10, 0.48, 0.90, 0.70)
        planned_audited = long_press_scene("planned", audited_bounds)
        fresh_audited = long_press_scene(
            "fresh",
            audited_bounds,
            meaning="interaction_zone",
        )
        after = long_press_scene(
            "after",
            audited_bounds,
            meaning="interaction_zone",
            passed=True,
        )
        observer = FakeSceneObserver([after])
        robot = FakeRobot()
        adapter = GenericSingleActionAdapter(
            capture=SequenceCapture(["gray"] * 4 + ["white"] * 4),
            observer=observer,
            robot=robot,
            frame_interval=0,
            post_action_settle=0,
        )

        result = adapter.execute(
            requested_action=SemanticAction(
                node_id="long-press-audited",
                action="long_press",
                params={
                    "element_id": "target",
                    "target": "long_press_target",
                    "role": "button",
                    "label": "长按我 · 不要移动",
                    "states": states,
                    "duration_ms": 800,
                    "tap_point": (0.5, 0.645),

                },
            ),
            planned_scene=planned,
            planned_frames=tuple(
                Image.new("RGB", (540, 960), "gray") for _ in range(4)
            ),
            goal=goal(),
            confirmed=True,
        )

        self.assertEqual(
            [("long_press", 500, 645, 0.8)],
            robot.actions,
        )
        self.assertEqual(
            planned.get_element("target").bounds,
            result.before_scene.get_element("target").bounds,
        )
        self.assertEqual(1, result.physical_actions)

    def test_matching_pixels_do_not_accept_second_model_long_press_veto(self):
        states = {"goal_relevant": True, "fully_visible": True}
        target = UIElement(
            element_id="target",
            role="button",
            meaning="long_press_target_area",
            label="长按我 · 不要移动",
            bounds=(0.10, 0.48, 0.90, 0.70),
            confidence=1.0,
            states=states,
            evidence=("黄色虚线框内逐字显示长按目标",),
        )
        planned = UIScene(
            app_id="local.acceptance",
            screen_id="long-press-board",
            summary="本地长按验收页面",
            elements=(target,),
            stable=True,
            confidence=1.0,
            fingerprint="planned",
            camera_alignment=aligned_camera_facts(),
        )
        after = replace(
            planned,
            overlays=("long_press 验收通过",),
            fingerprint="after",
        )
        robot = FakeRobot()
        adapter = GenericSingleActionAdapter(
            capture=SequenceCapture(["gray"] * 4),
            observer=FakeSceneObserver([after]),
            robot=robot,
            frame_interval=0,
            post_action_settle=0,
        )

        result = adapter.execute(
                requested_action=SemanticAction(
                    node_id="long-press-risky",
                    action="long_press",
                    params={
                        "element_id": "target",
                        "target": "long_press_target_area",
                        "role": "button",
                        "label": "长按我 · 不要移动",
                        "states": states,
                        "duration_ms": 800,
                        "tap_point": (0.5, 0.59),

                    },
                ),
                planned_scene=planned,
                planned_frames=tuple(
                    Image.new("RGB", (540, 960), "gray") for _ in range(4)
                ),
                goal=goal(),
                confirmed=True,
            )

        self.assertEqual(1, result.physical_actions)
        self.assertEqual([("long_press", 500, 590, 0.8)], robot.actions)

    def test_double_tap_crosses_adapter_as_one_canonical_action(self):
        states = {"goal_relevant": True, "fully_visible": True}
        target = UIElement(
            element_id="target",
            role="list_item",
            meaning="image_preview",
            label="预览图",
            bounds=(0.20, 0.30, 0.80, 0.70),
            confidence=1.0,
            states=states,
            evidence=("预览图完整可见",),
        )
        planned = UIScene(
            app_id="sample.app",
            screen_id="preview-list",
            summary="预览列表",
            elements=(target,),
            stable=True,
            confidence=1.0,
            fingerprint="planned",
            camera_alignment=aligned_camera_facts(),
        )
        after = replace(
            planned,
            screen_id="preview-detail",
            overlays=("预览已打开",),
            fingerprint="after",
        )
        robot = FakeRobot()
        result = GenericSingleActionAdapter(
            capture=SequenceCapture(["gray"] * 4),
            observer=FakeSceneObserver([after]),
            robot=robot,
            frame_interval=0,
            post_action_settle=0,
        ).execute(
            requested_action=SemanticAction(
                node_id="double-preview",
                action="double_tap",
                params={
                    "element_id": "target",
                    "target": "image_preview",
                    "role": "list_item",
                    "label": "预览图",
                    "states": states,
                    "tap_point": (0.5, 0.5),

                },
            ),
            planned_scene=planned,
            planned_frames=tuple(
                Image.new("RGB", (540, 960), "gray") for _ in range(4)
            ),
            goal=goal(),
            confirmed=True,
        )

        self.assertEqual([("double_tap", 500, 500)], robot.actions)
        self.assertEqual(1, result.physical_actions)
        self.assertEqual(2, result.hardware_receipt["click_count"])

    def test_drag_rebind_reuses_exact_endpoint_ids_despite_wording_drift(self):
        def make_scene(fingerprint, source_meaning, destination_meaning):
            return UIScene(
                app_id="local.acceptance",
                screen_id="drag-board",
                summary="通用拖动验收页面",
                elements=(
                    UIElement(
                        element_id="source",
                        role="container",
                        meaning=source_meaning,
                        label="起点",
                        bounds=(0.18, 0.68, 0.38, 0.82),
                        confidence=0.98,
                        states={"goal_relevant": True, "fully_visible": True},
                        evidence=("紧凑方块内逐字显示起点",),
                    ),
                    UIElement(
                        element_id="destination",
                        role="container",
                        meaning=destination_meaning,
                        label="绿色终点",
                        bounds=(0.55, 0.65, 0.85, 0.88),
                        confidence=0.98,
                        states={"goal_relevant": True, "fully_visible": True},
                        evidence=("绿色虚线区域内逐字显示绿色终点",),
                    ),
                ),
                stable=True,
                confidence=0.98,
                fingerprint=fingerprint,
                camera_alignment=aligned_camera_facts(),
            )

        planned = make_scene("planned", "drag_source_object", "drop_target_zone")
        fresh = make_scene("fresh", "draggable_source_block", "drag_target_region")
        after = make_scene("after", "draggable_source_block", "drag_target_region")
        moved_source = replace(
            after.elements[0],
            bounds=(0.62, 0.68, 0.78, 0.82),
        )
        after = replace(after, elements=(moved_source, after.elements[1]))
        observer = FakeSceneObserver([after])
        robot = FakeRobot()
        adapter = GenericSingleActionAdapter(
            capture=SequenceCapture(["gray"] * 4 + ["white"] * 4),
            observer=observer,
            robot=robot,
            frame_interval=0,
            post_action_settle=0,
        )

        result = adapter.execute(
            requested_action=SemanticAction(
                node_id="drag-synonyms",
                action="drag",
                params={
                    "source_element_id": "source",
                    "source_target": "drag_source_object",
                    "source_role": "container",
                    "source_label": "起点",
                    "source_states": {"goal_relevant": True, "fully_visible": True},
                    "destination_element_id": "destination",
                    "destination_target": "drop_target_zone",
                    "destination_role": "container",
                    "destination_label": "绿色终点",
                    "destination_states": {"goal_relevant": True, "fully_visible": True},

                },
            ),
            planned_scene=planned,
            planned_frames=tuple(
                Image.new("RGB", (540, 960), "gray") for _ in range(4)
            ),
            goal=goal(),
            confirmed=True,
        )

        self.assertEqual(1, result.physical_actions)
        self.assertEqual([("drag", 280, 750, 700, 765)], robot.actions)

    def test_changed_local_pixels_do_not_veto_or_claim_identity_verified(self):
        planned = scene("planned", screen_id="same_screen")
        fresh = scene("before", screen_id="same_screen")
        observer = FakeSceneObserver([fresh])
        robot = FakeRobot()
        adapter = GenericSingleActionAdapter(
            capture=SequenceCapture(["white"] * 4),
            observer=observer,
            robot=robot,
            frame_interval=0,
            post_action_settle=0,
        )
        action = SemanticAction(
            node_id="generic_step_1",
            action="scroll",
            params={"direction": "up", },
        )

        result = adapter.execute(
            requested_action=action,
            planned_scene=planned,
            planned_frames=tuple(Image.new("RGB", (540, 960), "black") for _ in range(4)),
            goal=goal(),
            confirmed=True,
        )
        self.assertEqual([("swipe", "up")], robot.actions)
        self.assertEqual(1, observer.calls)
        self.assertFalse(result.confirmation_frame_identity_verified)
        self.assertIsNone(result.confirmation_frame_delta)

    def test_observer_exception_stops_without_retry_or_action(self):
        planned = scene("planned")
        fresh = scene("before", element_id="fresh")
        after = scene("after", screen_id="app_home", element_id="after")
        observer = FakeSceneObserver(
            [
                RuntimeError("观察器返回不可解析响应。"),
                fresh,
                after,
            ]
        )
        robot = FakeRobot()
        action = SemanticAction(
            node_id="generic_step_1",
            action="tap_semantic",
            params={"element_id": "e1", "target": "app_icon"},
        )

        with self.assertRaisesRegex(GenericActionAdapterError, "第1轮动作前观察失败"):
            self._adapter(observer, robot).execute(
                requested_action=action,
                planned_scene=planned,
                goal=goal(),
                confirmed=True,
            )

        self.assertEqual(1, observer.calls)
        self.assertEqual(0, len(robot.actions))

    def test_goal_polluted_home_app_is_normalized_before_confirmation(self):
        planned = scene("planned", app_id="douyin")
        fresh = scene("before", app_id="unknown")
        after = scene(
            "after",
            screen_id="app_home",
            element_id="after",
            app_id="douyin",
        )
        observer = FakeSceneObserver([fresh, after])
        robot = FakeRobot()
        action = SemanticAction(
            node_id="generic_step_1",
            action="tap_semantic",
            params={"element_id": "e1", "target": "app_icon"},
        )
        result = self._adapter(observer, robot).execute(
            requested_action=action,
            planned_scene=planned,
            goal=goal(),
            confirmed=True,
        )
        self.assertEqual(planned.foreground_app_id, "launcher")
        self.assertEqual(fresh.foreground_app_id, "launcher")
        self.assertEqual(result.physical_actions, 1)
        self.assertEqual(len(robot.actions), 1)

    def test_ordinary_unchanged_screen_is_a_valid_fresh_receipt(self):
        planned = scene("same")
        fresh = scene("same")
        unchanged = scene("same", element_id="after")
        observer = FakeSceneObserver([fresh, unchanged, unchanged])
        robot = FakeRobot()
        action = SemanticAction(
            node_id="generic_step_1",
            action="tap_semantic",
            params={"element_id": "e1", "target": "app_icon"},
        )
        result = self._adapter(observer, robot).execute(
            requested_action=action,
            planned_scene=planned,
            goal=goal(),
            confirmed=True,
        )
        self.assertEqual(result.physical_actions, 1)
        self.assertEqual(result.action_outcome, "executed")
        self.assertEqual(result.verification_errors, ())
        self.assertEqual(observer.calls, 2)
        self.assertEqual(len(robot.actions), 1)

    def test_navigation_receipt_does_not_depend_on_next_input_readiness(self):
        planned = scene("before")
        wrong_conversation = scene(
            "wrong-after",
            screen_id="wrong_named_conversation",
            app_id="com.example.messaging",
        )
        current_goal = navigation_goal()
        current_goal.entities["next_subgoal_visual_context"] = {
            "step_id": "input_message",
            "objective": "在消息输入框输入abc",
            "constraints": [],
            "completion_conditions": ["消息输入框内容为abc"],
            "execution_class": "navigate",
            "goal_entities": {
                "active_input_transaction_text": "abc",
                "active_input_field_id": "message_field",
                "active_input_field_label": "消息",
                "active_input_multiline": False,
            },
        }
        robot = FakeRobot()
        adapter = GenericSingleActionAdapter(
            capture=SequenceCapture(["gray"] * 8),
            observer=FakeSceneObserver([wrong_conversation]),
            robot=robot,
            frame_interval=0,
            post_action_settle=0,
        )

        result = adapter.execute(
            requested_action=SemanticAction(
                node_id="open-target",
                action="tap_semantic",
                params={
                    "element_id": "e1",
                    "target": "app_icon",

                },
            ),
            planned_scene=planned,
            planned_frames=tuple(
                Image.new("RGB", (540, 960), "gray") for _ in range(4)
            ),
            goal=current_goal,
            confirmed=True,
        )

        self.assertEqual(1, result.physical_actions)
        self.assertEqual("executed", result.action_outcome)
        self.assertEqual((), result.verification_errors)
        self.assertEqual(1, len(robot.actions))

    def test_post_action_accepts_first_complete_changing_window(self):
        planned = scene("planned")
        after = scene("after", screen_id="app_home", element_id="after")
        observer = FakeSceneObserver([after])
        robot = FakeRobot()
        # Four gray frames are used by confirmation.  The post-action window
        # then sees a black/white transition before four consecutive white
        # frames finally settle.
        capture = SequenceCapture(
            ["gray"] * 4 + ["black", "white", "white", "white", "white"]
        )
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
        result = adapter.execute(
            requested_action=action,
            planned_scene=planned,
            planned_frames=tuple(
                Image.new("RGB", (540, 960), "gray") for _ in range(4)
            ),
            goal=goal(),
            confirmed=True,
        )
        self.assertEqual(result.physical_actions, 1)
        self.assertEqual(robot.actions, [("tap", 300, 400)])
        self.assertEqual(capture.calls, 8)
        self.assertEqual(observer.calls, 1)

    def test_input_post_action_waits_past_stable_blur_until_clarity_recovers(self):
        sharp = textured_phone_frame()
        blurred = sharp.filter(ImageFilter.GaussianBlur(2))
        capture = SequenceCapture([blurred] * 4 + [sharp] * 4)
        adapter = GenericSingleActionAdapter(
            capture=capture,
            observer=FakeSceneObserver([]),
            robot=FakeRobot(),
            frame_interval=0,
            post_action_settle=0,
            post_action_timeout=1,
        )

        frames, _paths = adapter._capture_stable_post_action_frames(
            deadline=10**9,
            evidence_dir=None,
            prefix="clarity_recovers",
            clarity_reference_frames=tuple(sharp.copy() for _ in range(4)),
            require_relative_clarity=True,
        )

        reference = measure_frame_sharpness(sharp)
        candidate = sorted(measure_frame_sharpness(frame) for frame in frames)
        self.assertGreater(capture.calls, 4)
        self.assertGreaterEqual(
            (candidate[1] + candidate[2]) / 2 / reference,
            0.80,
        )

    def test_continuous_action_does_not_reject_changed_background(self):
        reference = textured_phone_frame()
        off_phone = ImageOps.invert(reference)
        recovered = reference.copy()
        ImageDraw.Draw(recovered).rectangle(
            (180, 300, 360, 340),
            fill="white",
            outline="black",
        )
        capture = SequenceCapture([off_phone] * 4 + [recovered] * 4)
        adapter = GenericSingleActionAdapter(
            capture=capture,
            observer=FakeSceneObserver([]),
            robot=FakeRobot(),
            frame_interval=0,
            post_action_settle=0,
            post_action_timeout=1,
            post_action_continuous_timeout=45,
        )

        frames, _paths = adapter._capture_stable_post_action_frames(
            deadline=10**9,
            evidence_dir=None,
            prefix="camera_returns",
            clarity_reference_frames=tuple(reference.copy() for _ in range(4)),
            require_relative_clarity=True,
        )

        self.assertEqual(4, capture.calls)
        self.assertEqual(4, len(frames))
        self.assertEqual(off_phone.tobytes(), frames[-1].tobytes())

    def test_continuous_actions_receive_adaptive_post_action_budget(self):
        production_defaults = GenericSingleActionAdapter(
            capture=SequenceCapture(["gray"]),
            observer=FakeSceneObserver([]),
            robot=FakeRobot(),
        )
        self.assertEqual(10, production_defaults.post_action_timeout)
        self.assertEqual(45, production_defaults.post_action_continuous_timeout)

        adapter = GenericSingleActionAdapter(
            capture=SequenceCapture(["gray"]),
            observer=FakeSceneObserver([]),
            robot=FakeRobot(),
            post_action_timeout=10,
            post_action_continuous_timeout=45,
        )
        for kind in (
            "clear_verified_text",
            "drag",
            "input_verified_text",
            "long_press",
        ):
            with self.subTest(kind=kind):
                resolved = ResolvedSemanticAction(
                    node_id="continuous",
                    kind=kind,
                )
                self.assertTrue(
                    adapter._requires_extended_post_action_timeout(resolved)
                )
                self.assertEqual(45, adapter._post_action_timeout_for(resolved))

        navigation = ResolvedSemanticAction(
            node_id="navigation",
            kind="tap_semantic",
        )
        self.assertFalse(
            adapter._requires_extended_post_action_timeout(navigation)
        )
        self.assertEqual(10, adapter._post_action_timeout_for(navigation))

        explicit_test_budget = GenericSingleActionAdapter(
            capture=SequenceCapture(["gray"]),
            observer=FakeSceneObserver([]),
            robot=FakeRobot(),
            post_action_timeout=1,
        )
        self.assertEqual(1, explicit_test_budget.post_action_timeout)
        self.assertEqual(1, explicit_test_budget.post_action_continuous_timeout)

    def test_relative_clarity_threshold_separates_observed_input_samples(self):
        adapter = GenericSingleActionAdapter(
            capture=SequenceCapture(["gray"]),
            observer=FakeSceneObserver([]),
            robot=FakeRobot(),
        )
        observed_success_ratios = (
            0.934,
            1.007,
            0.985,
            0.945,
            1.005,
            1.018,
            1.276,
        )

        self.assertLess(0.568, adapter.post_action_min_relative_sharpness)
        self.assertTrue(
            all(
                ratio >= adapter.post_action_min_relative_sharpness
                for ratio in observed_success_ratios
            )
        )

    def test_relative_clarity_is_scoped_to_comparable_input_mutations(self):
        input_focus = ResolvedSemanticAction(
            node_id="focus-input",
            kind="tap_semantic",
        )
        navigation = ResolvedSemanticAction(
            node_id="open-page",
            kind="tap_semantic",
        )

        self.assertFalse(
            GenericSingleActionAdapter._requires_post_action_relative_clarity(
                input_focus
            )
        )
        self.assertFalse(
            GenericSingleActionAdapter._requires_post_action_relative_clarity(
                navigation
            )
        )
        for kind in (
            "input_verified_text",
            "press_enter",
            "clear_verified_text",
        ):
            with self.subTest(kind=kind):
                mutation = ResolvedSemanticAction(
                    node_id=kind,
                    kind=kind,
                )
                self.assertTrue(
                    GenericSingleActionAdapter._requires_post_action_relative_clarity(
                        mutation
                    )
                )

    def test_input_post_action_stable_blur_times_out_before_model_call(self):
        sharp = textured_phone_frame()
        blurred = sharp.filter(ImageFilter.GaussianBlur(2))
        observer = FakeSceneObserver([])
        adapter = GenericSingleActionAdapter(
            capture=SequenceCapture([blurred] * 4),
            observer=observer,
            robot=FakeRobot(),
            frame_interval=0,
            post_action_settle=0,
            post_action_timeout=1,
        )

        with patch(
            "agent.infrastructure.generic_action_adapter.time.monotonic",
            side_effect=[0.0, 0.0, 0.0, 1.0],
        ):
            with self.assertRaisesRegex(
                GenericActionAdapterError,
                "仍不够清晰.*相对值.*要求至少0.800",
            ):
                adapter._capture_stable_post_action_frames(
                    deadline=0.5,
                    evidence_dir=None,
                    prefix="clarity_timeout",
                    clarity_reference_frames=tuple(
                        sharp.copy() for _ in range(4)
                    ),
                    require_relative_clarity=True,
                )

        self.assertEqual(observer.calls, 0)

    def test_navigation_post_action_does_not_compare_changed_page_sharpness(self):
        sharp = textured_phone_frame()
        blurred = sharp.filter(ImageFilter.GaussianBlur(2))
        capture = SequenceCapture([blurred] * 4)
        adapter = GenericSingleActionAdapter(
            capture=capture,
            observer=FakeSceneObserver([]),
            robot=FakeRobot(),
            frame_interval=0,
            post_action_settle=0,
            post_action_timeout=1,
        )

        frames, _paths = adapter._capture_stable_post_action_frames(
            deadline=10**9,
            evidence_dir=None,
            prefix="navigation_changed_page",
            clarity_reference_frames=tuple(sharp.copy() for _ in range(4)),
            require_relative_clarity=False,
        )

        self.assertEqual(capture.calls, 4)
        self.assertEqual(len(frames), 4)

    def test_low_texture_input_reference_keeps_existing_stability_contract(self):
        adapter = GenericSingleActionAdapter(
            capture=SequenceCapture(["white"] * 4),
            observer=FakeSceneObserver([]),
            robot=FakeRobot(),
            frame_interval=0,
            post_action_settle=0,
            post_action_timeout=1,
        )

        frames, _paths = adapter._capture_stable_post_action_frames(
            deadline=10**9,
            evidence_dir=None,
            prefix="low_texture_reference",
            clarity_reference_frames=tuple(
                Image.new("RGB", (540, 960), "gray") for _ in range(4)
            ),
            require_relative_clarity=True,
        )

        self.assertEqual(len(frames), 4)


    def test_post_action_format_failure_does_not_resample_model(self):
        planned = scene("planned")
        fresh = scene("before")
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
            with self.assertRaises(GenericActionAdapterError) as caught:
                adapter.execute(
                    requested_action=action,
                    planned_scene=planned,
                    goal=goal(),
                    confirmed=True,
                    evidence_dir=Path(temp),
                )

        self.assertEqual(robot.actions, [("tap", 300, 400)])
        self.assertEqual(caught.exception.physical_actions, 1)
        self.assertEqual(observer.calls, 2)
        self.assertEqual(capture.calls, 12)
        self.assertEqual(len(caught.exception.evidence), 8)
        self.assertEqual(
            caught.exception.observation_errors,
            ("第1轮动作后观察失败：模型返回的 JSON 无法解析",),
        )

    def test_first_post_action_format_failure_stops_after_one_robot_action(self):
        observer = FakeSceneObserver(
            [
                scene("before"),
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
                        node_id="generic_step_1",
                        action="tap_semantic",
                        params={"element_id": "e1", "target": "app_icon"},
                    ),
                    planned_scene=scene("planned"),
                    goal=goal(),
                    confirmed=True,
                    evidence_dir=Path(temp),
                )

        self.assertEqual(caught.exception.physical_actions, 1)
        self.assertEqual(len(caught.exception.evidence), 8)
        self.assertEqual(observer.calls, 2)
        self.assertEqual(len(robot.actions), 1)
        self.assertEqual(
            caught.exception.observation_errors,
            ("第1轮动作后观察失败：模型返回的 JSON 无法解析",),
        )
        self.assertIn("第1轮动作后观察失败", str(caught.exception))
        self.assertNotIn("第2轮动作后观察失败", str(caught.exception))

    def test_each_action_uses_exactly_one_fresh_screenshot_observation(self):
        observer = FakeSceneObserver(
            [
                scene("before"),
                VisionAgentError("模型返回的 JSON 无法解析：first"),
                VisionAgentError("模型返回的 JSON 无法解析：second"),
                scene("after", screen_id="app_home", element_id="after"),
            ]
        )
        robot = FakeRobot()
        capture = SequenceCapture(["gray"] * 16)
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
                    node_id="generic_step_1",
                    action="tap_semantic",
                    params={"element_id": "e1", "target": "app_icon"},
                ),
                planned_scene=scene("planned"),
                goal=goal(),
                confirmed=True,
            )

        self.assertEqual(caught.exception.physical_actions, 1)
        self.assertEqual(observer.calls, 2)
        self.assertEqual(capture.calls, 12)
        self.assertEqual(robot.actions, [("tap", 300, 400)])

    def test_format_error_stops_before_any_second_capture(self):
        observer = FakeSceneObserver(
            [
                scene("before"),
                VisionAgentError("模型返回的 JSON 无法解析：first"),
            ]
        )
        robot = FakeRobot()
        adapter = SecondPostCaptureFailureAdapter(
            capture=SequenceCapture(["gray"] * 8),
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
                        node_id="generic_step_1",
                        action="tap_semantic",
                        params={"element_id": "e1", "target": "app_icon"},
                    ),
                    planned_scene=scene("planned"),
                    goal=goal(),
                    confirmed=True,
                    evidence_dir=Path(temp),
                )

        self.assertEqual(caught.exception.physical_actions, 1)
        self.assertEqual(
            caught.exception.observation_errors,
            ("第1轮动作后观察失败：模型返回的 JSON 无法解析：first",),
        )
        self.assertEqual(len(caught.exception.evidence), 8)
        self.assertNotIn("second_capture_timeout.jpg", caught.exception.evidence)
        self.assertEqual(observer.calls, 2)
        self.assertEqual(robot.actions, [("tap", 300, 400)])

    def test_confirmed_dismiss_uses_dedicated_physical_entry(self):
        planned = scene("planned", bounds=(0.1, 0.2, 0.3, 0.4))
        fresh = scene("before", bounds=(0.11, 0.21, 0.31, 0.41))
        after = scene("after", screen_id="app_home", element_id="after")
        robot = FakeRobot()
        action = SemanticAction(
            node_id="generic_step_1",
            action="dismiss_overlay",
            params={"element_id": "e1", "target": "close", "tap_point": (0.21, 0.31)},
        )

        result = self._adapter(
            FakeSceneObserver([fresh, after]),
            robot,
        ).execute(
            requested_action=action,
            planned_scene=planned,
            goal=goal(),
            confirmed=True,
        )

        self.assertEqual(result.physical_actions, 1)
        self.assertEqual(robot.actions, [("dismiss", 210, 310)])


    def test_post_action_non_format_failure_is_not_retried(self):
        observer = FakeSceneObserver(
            [scene("before"), VisionAgentError("请求超时")]
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
                    node_id="generic_step_1",
                    action="tap_semantic",
                    params={"element_id": "e1", "target": "app_icon"},
                ),
                planned_scene=scene("planned"),
                goal=goal(),
                confirmed=True,
            )

        self.assertEqual(caught.exception.physical_actions, 1)
        self.assertEqual(observer.calls, 2)
        self.assertEqual(capture.calls, 12)
        self.assertEqual(len(robot.actions), 1)

    def test_post_action_timeout_stops_without_calling_model_or_tapping_again(self):
        planned = scene("planned")
        observer = FakeSceneObserver([])
        robot = FakeRobot()
        capture = SequenceCapture(
            ["gray"] * 4 + ["black", "white", "black", "white"]
        )
        adapter = GenericSingleActionAdapter(
            capture=capture,
            observer=observer,
            robot=robot,
            frame_interval=0,
            post_action_settle=0,
            post_action_timeout=0,
        )
        action = SemanticAction(
            node_id="generic_step_1",
            action="tap_semantic",
            params={"element_id": "e1", "target": "app_icon"},
        )
        with self.assertRaisesRegex(
            GenericActionAdapterError,
            "未能采集到连续4帧",
        ) as ctx:
            adapter.execute(
                requested_action=action,
                planned_scene=planned,
                planned_frames=tuple(
                    Image.new("RGB", (540, 960), "gray") for _ in range(4)
                ),
                goal=goal(),
                confirmed=True,
            )
        self.assertEqual(ctx.exception.physical_actions, 1)
        self.assertEqual(observer.calls, 0)
        self.assertEqual(robot.actions, [("tap", 300, 400)])


if __name__ == "__main__":
    unittest.main()
