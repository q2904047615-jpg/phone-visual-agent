from __future__ import annotations


"""Regression coverage for the canonical single-action adapter."""


import json
import unittest
from unittest.mock import patch
from dataclasses import replace
from PIL import Image, ImageDraw
from agent.domain.confirmation_authority import ConfirmationAuthority
from agent.domain.text_transport import (
    TEXT_TRANSPORT_PROTOCOL,
    TextTransportActionScope,
    TextTransportProfile,
    TextTransportResult,
)
from agent.domain.validation import canonical_digest
from agent.infrastructure.generic_action_adapter import (
    GenericSingleActionAdapter as _GenericSingleActionAdapter,
)
from agent.application.action_adapter import GenericActionAdapterError
from agent.domain.generic_goal import GenericIntentDraft
from agent.infrastructure.generic_scene_observer import (
    SingleStepGenericSceneObserver,
)
from agent.infrastructure.observation_images import local_frame_fingerprint
from agent.infrastructure.orientation_safety import (
    OrientationSafetyError,
    _claim_audit_seal,
)
from agent.domain.semantic_action import SemanticAction
from agent.domain.ui_scene import (
    CameraAlignmentFacts,
    SystemUIFacts,
    UIElement,
    UIScene,
)
from agent.domain.universal_action_controller import (
    UniversalActionController,
)
from agent.domain.vision_model import VisionAgentError


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


def consumed_authority(action, before):
    """Build the explicit adapter-test precondition; never rewrite an action."""
    return ConfirmationAuthority(session_id="fixture-session", task_id="fixture-task",
        device_id="test-device", revision=1, step_id="fixture-step", effect_ids=(),
        observation_id="fixture-observation", fingerprint=before.fingerprint,
        decision_node_id=action.node_id, action_digest=canonical_digest(action.to_dict()), consumed=True)


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


class _BaseRevealSystemNavigationControllerTests(unittest.TestCase):
    @staticmethod
    def action() -> SemanticAction:
        return SemanticAction(
            node_id="reveal-navigation",
            action="reveal_system_navigation",
            params={},
        )


class _BaseElementBoundSwipeControllerTests(unittest.TestCase):
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


class _BaseGenericActionAdapterTests(unittest.TestCase):
    def _adapter(self, observer, robot, **kwargs):
        return GenericSingleActionAdapter(
            capture=lambda: Image.new("RGB", (540, 960), "gray"),
            observer=observer,
            robot=robot,
            frame_interval=0,
            post_action_settle=0,
            **kwargs,
        )
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
            states={"input_field_id": "field_primary", **({
                "goal_relevant": False,
                "fully_visible": True,
                "focused": True,
                "visible": True,
                "value": value,
                "keyboard_layout": "qwerty",
                "keyboard_input_mode": "direct_latin",
                "keyboard_case_mode": "lower",
            })},
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
                    states={"input_field_id": "field_primary", **(states)},
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
    def _assert_public_observation_failure_before_robot(self, responses):
        provider = RawSceneProvider(responses)
        robot = FakeRobot()
        adapter = self._adapter(SingleStepGenericSceneObserver(provider), robot)
        action = SemanticAction(
            node_id="blocked-observer-json",
            action="tap_semantic",
            params={"tap_point": (0.3, 0.4), "element_id": "e1", "target": "app_icon"},
        )

        with self.assertRaises(GenericActionAdapterError) as caught:
            prepared_15_adapter = adapter
            prepared_15_goal = goal()
            prepared_15_scene, prepared_15_frames, _, _ = prepared_15_adapter.capture_scene(
                prepared_15_goal, prefix="test_orchestrator_before", evidence_dir=None)
            prepared_15_adapter.execute(
                requested_action=action,
                planned_scene=prepared_15_scene, planned_frames=prepared_15_frames,
                goal=prepared_15_goal,
                confirmed=True,
            )

        self.assertEqual(0, caught.exception.physical_actions)
        self.assertEqual([], robot.actions)
        self.assertIsNone(robot._armed)
        self.assertEqual([None], provider.max_tokens_seen)

