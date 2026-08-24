from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageDraw, ImageFilter, ImageOps

from capability_acceptance import _validate_live_promotion_source
from generic_action_adapter import (
    FUSED_POST_ACTION_NEXT_STEP_OBSERVATION_PHASE,
    GenericActionAdapterError,
    GenericSingleActionAdapter as _GenericSingleActionAdapter,
    _post_action_observation_context,
    _typed_exact_tap_target_label,
    stable_qwerty_ocr_anchors,
)
from generic_goal import GenericIntentDraft
from generic_scene_observer import GenericSceneObserver, _local_frame_fingerprint
from input_value_lineage import TypedInputLineageStore
from observation_images import measure_frame_sharpness
from orientation_safety import (
    LOCAL_QWERTY_ORIENTATION_SOURCE,
    OrientationFrameMismatchError,
    _claim_audit_seal,
    _mint_audited_credential,
)
from generic_step_planner import (
    GenericStepPlanner,
    GenericStepPlanningError,
    GenericStepProposal,
)
from semantic_action import SemanticAction
from ui_scene import CameraAlignmentFacts, SystemUIFacts, UIElement, UIScene
from universal_action_controller import (
    ResolvedSemanticAction,
    UniversalActionController,
    UniversalActionError,
)
from vision_agent import VisionAgentError


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


class FakeTextProvider:
    configured = True

    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def chat_json(self, messages, max_tokens=700):
        self.calls += 1
        self.messages = messages
        return json.dumps(self.payload, ensure_ascii=False)


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
    def __init__(
        self,
        scenes,
        *,
        audit_rotation="upright",
        audit_confidence=0.96,
        geometry_scenes=None,
    ):
        self.scenes = list(scenes)
        self.calls = 0
        self.geometry_audit_calls = []
        self.geometry_scenes = list(geometry_scenes or ())
        self.goal_contexts = []
        self.home_audit_calls = 0
        self.audit_rotation = audit_rotation
        self.audit_confidence = audit_confidence

    def observe(self, *, frames, goal_context=None):
        self.calls += 1
        self.goal_contexts.append(goal_context)
        result = self.scenes.pop(0)
        if isinstance(result, BaseException):
            raise result
        if result.camera_alignment.phone_content_rotation == "unknown":
            result = replace(
                result,
                camera_alignment=aligned_camera_facts(),
            )
        return result

    def audit_camera_alignment(self, *, frames, device_id, scene_fingerprint):
        return _mint_audited_credential(
            device_id=device_id,
            scene_fingerprint=scene_fingerprint,
            frame=frames[-1],
            phone_content_rotation=self.audit_rotation,
            confidence=self.audit_confidence,
            evidence=("测试手机界面轴线",),
        )

    def audit_coordinate_free_system_navigation_alignment(
        self, *, frames, device_id, scene_fingerprint
    ):
        self.home_audit_calls += 1
        return self.audit_camera_alignment(
            frames=frames,
            device_id=device_id,
            scene_fingerprint=scene_fingerprint,
        )

    def audit_element_geometry(self, *, frames, scene, element_ids):
        self.geometry_audit_calls.append(tuple(element_ids))
        return self.geometry_scenes.pop(0) if self.geometry_scenes else scene


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
        self.keyboard_layouts = []
        self.calibrated_target_requests = []
        self.device_id = "test-device"
        self._armed = None
        self._long_press_receipt = None

    def arm_physical_execution(self, credential, *, action, scene_fingerprint):
        credential.assert_authorizes(
            device_id=self.device_id,
            scene_fingerprint=scene_fingerprint,
            frame_size=credential.frame_size,
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
        return (x, y)

    def resolve_calibrated_target_grid_point(
        self, x, y, target_bounds, frame_size
    ):
        self.calibrated_target_requests.append(
            (x, y, tuple(target_bounds), tuple(frame_size))
        )
        return x, y

    def vision_dismiss_overlay_relative(self, x, y):
        self._consume("dismiss_overlay")
        self.actions.append(("dismiss", x, y))
        return (x, y)

    def vision_swipe_up(self):
        self._consume("swipe")
        self.actions.append(("swipe", "up"))

    def vision_reveal_system_navigation(self):
        self._consume("reveal_system_navigation")
        self.actions.append(("reveal_system_navigation",))
        return (500, 950)

    def vision_android_back(self):
        self._consume("back")
        self.actions.append(("back",))
        return (500, 950)

    def vision_android_home(self):
        self._consume("home")
        self.actions.append(("home",))
        return (500, 950)

    def vision_type_text_with_layout(self, text, keyboard_layout):
        self._consume("input_verified_text")
        self.keyboard_layouts.append(keyboard_layout)
        self.actions.append(("input", text))

    def vision_type_pinyin(self, text, pinyin, keyboard_layout):
        self._consume("input_verified_text")
        self.keyboard_layouts.append(keyboard_layout)
        self.actions.append(("pinyin", text, pinyin))

    def vision_clear_text(self, keyboard_layout, delete_count):
        self._consume("input_verified_text")
        self.keyboard_layouts.append(keyboard_layout)
        self.actions.append(("clear", delete_count))

    def vision_long_press_relative(self, x, y, hold_seconds):
        self._consume("long_press")
        self.actions.append(("long_press", x, y, hold_seconds))
        self._long_press_receipt = {
            "version": "2026-08-16-seller-gui-contact-barrier-v3",
            "channel": "right_button_stationary_touch",
            "seller_event_barrier_confirmed": True,
            "round_trip_position_confirmed": True,
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


class PhysicalGateDriftRobot(FakeRobot):
    def vision_swipe_up(self):
        self._consume("swipe")
        raise OrientationFrameMismatchError(
            "动作前实际捕获帧与独立方向审计帧发生视觉漂移："
            "亮度差22.66，结构差73.09。",
            actual_frame=Image.new("RGB", (540, 960), "white"),
            brightness_delta=22.66,
            centered_mae=73.09,
        )


class ClickReceiptRobot(FakeRobot):
    def __init__(self, *, valid=True):
        super().__init__()
        self.valid = valid
        self._click_receipt = None

    def _record_click_receipt(self):
        self._click_receipt = {
            "version": "2026-08-19-seller-gui-click-barrier-v1",
            "channel": "left_button_atomic_click",
            "seller_event_barrier_confirmed": self.valid,
            "round_trip_position_confirmed": True,
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
        super().__init__(*args, device_id=device_id, **kwargs)


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


class SecondPostCaptureFailureAdapter(GenericSingleActionAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.post_capture_calls = 0

    def _capture_stable_post_action_frames(self, **kwargs):
        self.post_capture_calls += 1
        if self.post_capture_calls == 2:
            raise GenericActionAdapterError(
                "动作后画面在限定时间内没有稳定",
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
                "subgoal_id": "open_browser",
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
                "subgoal_id": "exact_tap_semantic",
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


class GenericStepPlannerTests(unittest.TestCase):
    def test_reveal_system_navigation_has_no_geometry_parameters(self):
        provider = FakeTextProvider(
            {
                "status": "action",
                "action": {
                    "kind": "reveal_system_navigation",
                    "expected_effect": {
                        "system_ui": {"navigation_bar_visible": True}
                    },
                },
                "reason": "当前处于沉浸态且系统导航栏隐藏",
                "completion_evidence": [],
            }
        )
        current = scene(
            "a",
            system_ui=SystemUIFacts(
                immersive_or_fullscreen=True,
                navigation_bar_visible=False,
            ),
        )

        proposal = GenericStepPlanner(provider).propose(goal(), current)

        self.assertEqual("reveal_system_navigation", proposal.action.action)
        self.assertEqual(
            {"expected_effect": {"system_ui": {"navigation_bar_visible": True}}},
            proposal.action.params,
        )

    def test_reveal_system_navigation_rejects_direction(self):
        provider = FakeTextProvider(
            {
                "status": "action",
                "action": {
                    "kind": "reveal_system_navigation",
                    "direction": "up",
                    "expected_effect": {
                        "system_ui": {"navigation_bar_visible": True}
                    },
                },
                "reason": "bad",
                "completion_evidence": [],
            }
        )
        current = scene(
            "a",
            system_ui=SystemUIFacts(
                immersive_or_fullscreen=True,
                navigation_bar_visible=False,
            ),
        )

        with self.assertRaisesRegex(GenericStepPlanningError, "不能携带"):
            GenericStepPlanner(provider).propose(goal(), current)

    def test_proposes_only_one_existing_element(self):
        provider = FakeTextProvider(
            {
                "status": "action",
                "action": {
                    "kind": "tap_semantic",
                    "element_id": "e1",
                    "target": "app_icon",
                    "role": "icon",
                    "label": "设置",
                    "states": {},
                    "expected_effect": {},
                },
                "reason": "设置图标清晰可见",
                "completion_evidence": [],
            }
        )
        proposal = GenericStepPlanner(provider).propose(goal(), scene("a"))
        self.assertEqual(proposal.status, "action")
        self.assertEqual(proposal.action.params["element_id"], "e1")
        self.assertEqual(provider.calls, 1)

    def test_rejects_raw_coordinates(self):
        provider = FakeTextProvider(
            {
                "status": "action",
                "action": {
                    "kind": "tap_semantic",
                    "element_id": "e1",
                    "target": "app_icon",
                    "x": 300,
                },
                "reason": "bad",
                "completion_evidence": [],
            }
        )
        with self.assertRaisesRegex(GenericStepPlanningError, "协议外字段"):
            GenericStepPlanner(provider).propose(goal(), scene("a"))

    def test_finished_requires_visible_evidence(self):
        provider = FakeTextProvider(
            {
                "status": "finished",
                "action": None,
                "reason": "完成",
                "completion_evidence": [],
            }
        )
        with self.assertRaisesRegex(GenericStepPlanningError, "可见证据"):
            GenericStepPlanner(provider).propose(goal(), scene("a"))


class RevealSystemNavigationControllerTests(unittest.TestCase):
    @staticmethod
    def action() -> SemanticAction:
        return SemanticAction(
            node_id="reveal-navigation",
            action="reveal_system_navigation",
            params={
                "expected_effect": {
                    "system_ui": {"navigation_bar_visible": True}
                }
            },
        )

    def test_requires_structured_system_ui_pre_and_post_facts(self):
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

        with self.assertRaisesRegex(UniversalActionError, "结构化导航栏可见证据"):
            controller.verify_after_action(
                resolved,
                before,
                scene("different", screen_id="navigation_bar_visible_summary_only"),
            )

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


class FormalTypedTransitionControllerTests(unittest.TestCase):
    def test_home_requires_typed_launcher_postcondition(self):
        controller = UniversalActionController()
        before = scene("before", screen_id="settings_main", app_id="settings")
        action = SemanticAction(
            node_id="formal-home",
            action="home",
            params={
                "formal_candidate_id": "candidate.home",
                "formal_report_digest": "a" * 64,
                "formal_transition": {
                    "transition_id": "transition.home",
                    "precondition_claim_ids": ["claim.surface"],
                    "expectations": [
                        {
                            "subject_ref": "surface.current",
                            "predicate": "surface.kind",
                            "operator": "equals",
                            "value": "launcher",
                        }
                    ],
                    "exploratory": False,
                },
            },
        )
        resolved = controller.resolve_one(action, before, confirmed=True)
        self.assertEqual("candidate.home", resolved.formal_candidate_id)
        self.assertEqual(
            "surface.kind",
            resolved.formal_transition["expectations"][0]["predicate"],
        )

        controller.verify_after_action(
            resolved,
            before,
            scene("after", screen_id="android_home", app_id="unknown"),
        )
        with self.assertRaisesRegex(UniversalActionError, "typed surface.kind"):
            controller.verify_after_action(
                resolved,
                before,
                scene("wrong", screen_id="settings_detail", app_id="settings"),
            )

    def test_formal_input_uses_exact_element_identity_not_goal_relevant_flag(self):
        controller = UniversalActionController()
        before = UIScene(
            app_id="generic_app",
            screen_id="editor",
            summary="唯一聚焦输入框",
            elements=(
                UIElement(
                    element_id="field",
                    role="input",
                    meaning="application_text_input",
                    label="输入",
                    bounds=(0.1, 0.1, 0.9, 0.2),
                    confidence=0.97,
                    states={
                        "focused": True,
                        "value": "",
                        "keyboard_layout": "qwerty",
                        "keyboard_input_mode": "direct_latin",
                        "keyboard_case_mode": "lower",
                        "keyboard_geometry": TEST_QWERTY_GEOMETRY,
                        "goal_relevant": False,
                    },
                ),
            ),
            stable=True,
            confidence=0.96,
            fingerprint="before-input",
        )
        action = SemanticAction(
            node_id="formal-input",
            action="input_verified_text",
            params={
                "element_id": "field",
                "target": "application_text_input",
                "role": "input",
                "label": "输入",
                "states": {"focused": True},
                "text": "agent",
                "expected_effect": {
                    "element_state": {
                        "meaning": "application_text_input",
                        "states": {"value": "agent"},
                    }
                },
                "formal_candidate_id": "candidate.input",
                "formal_report_digest": "b" * 64,
                "formal_transition": {
                    "transition_id": "transition.input",
                    "precondition_claim_ids": ["claim.field"],
                    "expectations": [
                        {
                            "subject_ref": "element.field",
                            "predicate": "element.state.value",
                            "operator": "equals",
                            "value": "agent",
                        }
                    ],
                    "exploratory": False,
                },
            },
        )
        resolved = controller.resolve_one(action, before, confirmed=True)
        after = UIScene(
            app_id="generic_app",
            screen_id="editor",
            summary="输入完成",
            elements=(
                UIElement(
                    element_id="field",
                    role="input",
                    meaning="application_text_input",
                    label="输入",
                    bounds=(0.1, 0.1, 0.9, 0.2),
                    confidence=0.97,
                    states={"focused": True, "value": "agent"},
                ),
            ),
            stable=True,
            confidence=0.96,
            fingerprint="after-input",
        )
        controller.verify_after_action(resolved, before, after)

    def test_formal_chinese_input_verifies_preedit_before_candidate_selection(self):
        controller = UniversalActionController()
        before_states = {
            "focused": True,
            "value": "",
            "keyboard_layout": "qwerty",
            "keyboard_input_mode": "chinese_pinyin",
            "keyboard_case_mode": "lower",
            "keyboard_geometry": TEST_QWERTY_GEOMETRY,
            "input_field_id": "field_primary",
            "goal_relevant": False,
        }
        before = UIScene(
            app_id="generic_app",
            screen_id="editor",
            summary="唯一聚焦中文输入框",
            elements=(
                UIElement(
                    element_id="field",
                    role="input",
                    meaning="application_text_input",
                    label="消息",
                    bounds=(0.1, 0.1, 0.9, 0.2),
                    confidence=0.97,
                    states=before_states,
                ),
            ),
            stable=True,
            confidence=0.96,
            fingerprint="before-chinese-input",
        )
        expected_states = {
            "value": "",
            "ime_preedit_text": "nihao",
            "ime_exact_candidate_text": "你好",
        }
        action = SemanticAction(
            node_id="formal-chinese-input",
            action="input_verified_text",
            params={
                "element_id": "field",
                "target": "application_text_input",
                "role": "input",
                "label": "消息",
                "states": before_states,
                "text": "你好",
                "expected_effect": {
                    "element_state": {
                        "meaning": "application_text_input",
                        "states": expected_states,
                    }
                },
                "formal_candidate_id": "candidate.chinese-input",
                "formal_report_digest": "d" * 64,
                "formal_transition": {
                    "transition_id": "transition.chinese-input",
                    "precondition_claim_ids": ["claim.field"],
                    "expectations": [
                        {
                            "subject_ref": "element.field",
                            "predicate": f"element.state.{name}",
                            "operator": "equals",
                            "value": value,
                        }
                        for name, value in expected_states.items()
                    ],
                    "exploratory": False,
                },
            },
        )

        resolved = controller.resolve_one(action, before, confirmed=True)
        after = replace(
            before,
            summary="拼音组合和唯一候选可见",
            fingerprint="after-chinese-input",
            elements=(
                replace(
                    before.elements[0],
                    states={**before_states, **expected_states},
                ),
            ),
        )
        controller.verify_after_action(resolved, before, after)

        wrong_candidate = replace(
            after,
            fingerprint="wrong-chinese-candidate",
            elements=(
                replace(
                    after.elements[0],
                    states={
                        **after.elements[0].states,
                        "ime_exact_candidate_text": "您好",
                    },
                ),
            ),
        )
        with self.assertRaisesRegex(UniversalActionError, "唯一逐字一致的中文候选"):
            controller.verify_after_action(resolved, before, wrong_candidate)

    def test_formal_direct_latin_input_accepts_only_exact_visible_preedit_candidate(self):
        controller = UniversalActionController()
        before_states = {
            "focused": True,
            "value": "",
            "keyboard_layout": "qwerty",
            "keyboard_input_mode": "direct_latin",
            "keyboard_case_mode": "lower",
            "keyboard_geometry": TEST_QWERTY_GEOMETRY,
            "input_field_id": "field_primary",
            "goal_relevant": False,
        }
        before = UIScene(
            app_id="generic_app",
            screen_id="editor",
            summary="唯一聚焦英文输入框",
            elements=(
                UIElement(
                    element_id="field",
                    role="input",
                    meaning="application_text_input",
                    label="消息",
                    bounds=(0.1, 0.1, 0.9, 0.2),
                    confidence=0.97,
                    states=before_states,
                ),
            ),
            stable=True,
            confidence=0.96,
            fingerprint="before-direct-preedit",
        )
        action = SemanticAction(
            node_id="formal-direct-preedit",
            action="input_verified_text",
            params={
                "element_id": "field",
                "target": "application_text_input",
                "role": "input",
                "label": "消息",
                "states": before_states,
                "text": "first",
                "expected_effect": {
                    "element_state": {
                        "meaning": "application_text_input",
                        "states": {"value": "first"},
                    }
                },
                "formal_candidate_id": "candidate.direct-preedit",
                "formal_report_digest": "e" * 64,
                "formal_transition": {
                    "transition_id": "transition.direct-preedit",
                    "precondition_claim_ids": ["claim.field"],
                    "expectations": [
                        {
                            "subject_ref": "element.field",
                            "predicate": "element.state.value",
                            "operator": "equals",
                            "value": "first",
                        }
                    ],
                    "exploratory": False,
                },
            },
        )
        resolved = controller.resolve_one(action, before, confirmed=True)
        field = replace(
            before.elements[0],
            states={
                **before_states,
                "keyboard_input_mode": "chinese_pinyin",
                "ime_preedit_text": "first",
                "ime_exact_candidate_text": "first",
            },
        )
        candidate = UIElement(
            element_id="local_audited_ime_candidate_1",
            role="button",
            meaning="ime_exact_candidate",
            label="first",
            bounds=(0.1, 0.6, 0.25, 0.64),
            confidence=1.0,
            states={
                "goal_relevant": True,
                "fully_visible": True,
                "ime_candidate": True,
                "input_element_id": "field",
                "prior_input_value": "",
                "expected_input_value": "first",
                "pinyin": "first",
            },
            evidence=("逐字相同的英文联想候选",),
        )
        after = replace(
            before,
            summary="英文预编辑及逐字相同候选可见",
            fingerprint="after-direct-preedit",
            elements=(field, candidate),
        )
        controller.verify_after_action(resolved, before, after)

        wrong = replace(
            after,
            fingerprint="wrong-direct-preedit",
            elements=(field, replace(candidate, label="firstly")),
        )
        with self.assertRaisesRegex(UniversalActionError, "文字不匹配"):
            controller.verify_after_action(resolved, before, wrong)

    def test_press_enter_requires_newline_key_and_verifies_exact_multiline_value(self):
        controller = UniversalActionController()
        before = UIScene(
            app_id="generic_app",
            screen_id="editor",
            summary="正文多行输入框和换行键可见",
            elements=(
                UIElement(
                    element_id="field",
                    role="input",
                    meaning="application_text_input",
                    label="正文",
                    bounds=(0.1, 0.1, 0.9, 0.3),
                    confidence=0.98,
                    states={
                        "focused": True,
                        "value": "first",
                        "input_multiline": True,
                    },
                ),
                UIElement(
                    element_id="enter",
                    role="button",
                    meaning="input_exact_enter_key",
                    label="↵",
                    bounds=(0.78, 0.78, 0.94, 0.9),
                    confidence=0.98,
                    states={
                        "goal_relevant": True,
                        "fully_visible": True,
                        "input_enter_key": True,
                        "key_action": "newline",
                        "key_value": "\n",
                        "prior_input_value": "first",
                        "expected_input_value": "first\n",
                        "input_element_id": "field",
                    },
                ),
            ),
            stable=True,
            confidence=0.98,
            fingerprint="before-enter",
        )
        action = SemanticAction(
            node_id="formal-enter",
            action="press_enter",
            params={
                "element_id": "enter",
                "target": "input_exact_enter_key",
                "role": "button",
                "label": "↵",
                "states": {"key_action": "newline"},
                "expected_effect": {
                    "element_state": {
                        "meaning": "application_text_input",
                        "states": {"value": "first\n"},
                    }
                },
                "formal_candidate_id": "candidate.enter",
                "formal_report_digest": "c" * 64,
                "formal_transition": {
                    "transition_id": "transition.enter",
                    "precondition_claim_ids": ["claim.enter"],
                    "expectations": [
                        {
                            "subject_ref": "element.field",
                            "predicate": "element.state.value",
                            "operator": "equals",
                            "value": "first\n",
                        }
                    ],
                    "exploratory": False,
                },
            },
        )
        resolved = controller.resolve_one(action, before, confirmed=True)
        self.assertEqual("press_enter", resolved.kind)
        self.assertEqual("field", resolved.input_element_id)
        after = UIScene(
            app_id="generic_app",
            screen_id="editor",
            summary="正文已有真实换行",
            elements=(
                UIElement(
                    element_id="field",
                    role="input",
                    meaning="application_text_input",
                    label="正文",
                    bounds=(0.1, 0.1, 0.9, 0.3),
                    confidence=0.98,
                    states={"focused": True, "value": "first\n"},
                ),
            ),
            stable=True,
            confidence=0.98,
            fingerprint="after-enter",
        )
        controller.verify_after_action(resolved, before, after)

        send_key = replace(
            before.elements[1],
            states={**before.elements[1].states, "key_action": "send"},
        )
        with self.assertRaisesRegex(UniversalActionError, "newline"):
            controller.resolve_one(
                replace(
                    action,
                    params={**action.params, "states": {"key_action": "send"}},
                ),
                replace(before, elements=(before.elements[0], send_key)),
                confirmed=True,
            )


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
                                "geometry_audit_source": "element_geometry_audit",
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

    def test_literal_key_reuses_single_step_scene_on_matching_frames(self):
        gray = Image.new("RGB", (540, 960), "gray")
        planned_fingerprint = _local_frame_fingerprint(gray)
        planned = self._literal_input_scene(planned_fingerprint)
        fresh_missing = self._literal_input_scene(
            planned_fingerprint,
            include_key=False,
        )
        planned_audited = self._literal_input_scene(
            planned_fingerprint,
            audited=True,
        )
        fresh_audited = self._literal_input_scene(
            planned_fingerprint,
            audited=True,
        )
        planned_audited = replace(
            planned_audited,
            elements=tuple(
                replace(element, bounds=(0.40, 0.68, 0.60, 0.74))
                if element.element_id == "local_audited_literal_key_1"
                else element
                for element in planned_audited.elements
            ),
        )
        fresh_audited = replace(
            fresh_audited,
            elements=tuple(
                replace(element, bounds=(0.32, 0.664, 0.52, 0.724))
                if element.element_id == "local_audited_literal_key_1"
                else element
                for element in fresh_audited.elements
            ),
        )
        after = self._literal_input_scene(
            "after-live2",
            value="live2",
            include_key=False,
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
        key = planned.get_element("local_audited_literal_key_1")
        expected_effect = {
            "element_state": {
                "meaning": "application_text_input",
                "states": {"value": "live2"},
            }
        }

        result = adapter.execute(
            requested_action=SemanticAction(
                node_id="literal-2",
                action="tap_semantic",
                params={
                    "formal_candidate_id": "candidate-literal-2",
                    "formal_report_digest": "a" * 64,
                    "formal_transition": {
                        "transition_id": "transition-literal-2",
                        "precondition_claim_ids": ["claim-input-live"],
                        "expectations": [
                            {
                                "subject_ref": "element.local_audited_input_1",
                                "predicate": "element.state.value",
                                "operator": "equals",
                                "value": "live2",
                            }
                        ],
                        "exploratory": False,
                    },
                    "element_id": key.element_id,
                    "target": key.meaning,
                    "role": key.role,
                    "label": key.label,
                    "states": dict(key.states),
                    "expected_effect": expected_effect,
                },
            ),
            planned_scene=planned,
            planned_frames=(gray, gray.copy(), gray.copy(), gray.copy()),
            goal=GenericIntentDraft(
                understood=True,
                app_id="current_foreground",
                app_name="当前应用",
                objective="当前输入框显示 live21 且尚未提交",
                entities={"input_text": "live21"},
                success_criteria={"input": "live21"},
            ),
            confirmed=True,
        )

        self.assertEqual(1, result.physical_actions)
        self.assertEqual((), result.verification_errors)
        self.assertEqual("matched", result.action_outcome)
        self.assertEqual([("tap", 220, 735)], robot.actions)
        self.assertEqual([], robot.calibrated_target_requests)
        self.assertEqual([], observer.geometry_audit_calls)

    def test_press_enter_executes_one_verified_tap_and_matches_exact_newline(self):
        gray = Image.new("RGB", (540, 960), "gray")
        fingerprint = _local_frame_fingerprint(gray)

        def enter_scene(value, *, audited=False, include_key=True, fp=fingerprint):
            source = self._literal_input_scene(
                fp,
                value=value,
                include_key=include_key,
                audited=audited,
            )
            elements = []
            for item in source.elements:
                if item.role == "input":
                    elements.append(
                        replace(
                            item,
                            label="正文",
                            states={**item.states, "input_multiline": True},
                        )
                    )
                else:
                    elements.append(
                        replace(
                            item,
                            element_id="local_audited_enter_key_1",
                            meaning="input_exact_enter_key",
                            label="↵",
                            states={
                                **item.states,
                                "input_literal_key": False,
                                "input_enter_key": True,
                                "key_action": "newline",
                                "key_value": "\n",
                                "prior_input_value": "first",
                                "expected_input_value": "first\n",
                            },
                        )
                    )
            return replace(source, elements=tuple(elements))

        planned = enter_scene("first")
        fresh_without_key = enter_scene("first", include_key=False)
        audited = enter_scene("first", audited=True)
        after = enter_scene(
            "first\n",
            include_key=False,
            fp="after-enter",
        )
        observer = FakeSceneObserver([after])

        robot = ClickReceiptRobot()
        adapter = GenericSingleActionAdapter(
            capture=SequenceCapture(["gray"] * 4 + ["white"] * 4),
            observer=observer,
            robot=robot,
            frame_interval=0,
            post_action_settle=0,
        )
        key = planned.get_element("local_audited_enter_key_1")
        expected_effect = {
            "element_state": {
                "meaning": "application_text_input",
                "states": {"value": "first\n"},
            }
        }
        result = adapter.execute(
            requested_action=SemanticAction(
                node_id="enter",
                action="press_enter",
                params={
                    "formal_candidate_id": "candidate-enter",
                    "formal_report_digest": "a" * 64,
                    "formal_transition": {
                        "transition_id": "transition-enter",
                        "precondition_claim_ids": ["claim-enter"],
                        "expectations": [
                            {
                                "subject_ref": "element.local_audited_input_1",
                                "predicate": "element.state.value",
                                "operator": "equals",
                                "value": "first\n",
                            }
                        ],
                        "exploratory": False,
                    },
                    "element_id": key.element_id,
                    "target": key.meaning,
                    "role": key.role,
                    "label": key.label,
                    "states": dict(key.states),
                    "expected_effect": expected_effect,
                },
            ),
            planned_scene=planned,
            planned_frames=(gray, gray.copy(), gray.copy(), gray.copy()),
            goal=GenericIntentDraft(
                understood=True,
                app_id="current_foreground",
                app_name="当前应用",
                objective="正文逐字为 first 换行 second",
                entities={"input_text": "first\nsecond"},
                success_criteria={"input": "first\nsecond"},
            ),
            confirmed=True,
        )
        self.assertEqual(1, result.physical_actions)
        self.assertEqual("matched", result.action_outcome)
        self.assertEqual("press_enter", result.resolved_action.kind)
        self.assertEqual([("tap", 220, 735)], robot.actions)
        self.assertTrue(result.hardware_receipt["seller_event_barrier_confirmed"])
        self.assertNotIn(
            "_allow_omitted_local_input_auxiliary_confirmation",
            observer.goal_contexts[0],
        )
        self.assertEqual([], observer.geometry_audit_calls)

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

    def test_only_strict_primary_input_audit_can_reuse_confirmation(self):
        planned = self._strict_primary_input_scene()
        field = planned.elements[0]
        requested = SemanticAction(
            node_id="type-body",
            action="input_verified_text",
            params={
                "formal_candidate_id": "candidate-type-body",
                "element_id": field.element_id,
                "target": field.meaning,
                "role": field.role,
                "label": field.label,
                "states": dict(field.states),
                "text": "agent",
            },
        )

        self.assertTrue(
            GenericSingleActionAdapter._primary_input_confirmation_reusable(
                requested,
                planned,
            )
        )

        ordinary_field = replace(
            field,
            states={
                **field.states,
                "primary_input_geometry_verified": False,
            },
        )
        ordinary = replace(planned, elements=(ordinary_field,))
        ordinary_request = replace(
            requested,
            params={**requested.params, "states": dict(ordinary_field.states)},
        )
        self.assertFalse(
            GenericSingleActionAdapter._primary_input_confirmation_reusable(
                ordinary_request,
                ordinary,
            )
        )

    def test_local_qwerty_rows_replace_only_direction_model_call(self):
        planned = self._strict_primary_input_scene()
        field = planned.elements[0]
        requested = SemanticAction(
            node_id="type-body",
            action="input_verified_text",
            params={
                "formal_candidate_id": "candidate-type-body",
                "element_id": field.element_id,
                "target": field.meaning,
                "role": field.role,
                "label": field.label,
                "states": dict(field.states),
                "text": "agent",
            },
        )
        anchors = {
            key: list(value)
            for key, value in TEST_QWERTY_GEOMETRY["anchors"].items()
        }
        adapter = self._adapter(
            FakeSceneObserver([]),
            FakeRobot(),
            qwerty_row_snapper=lambda _frames, _anchors: anchors,
        )
        frames = [Image.new("RGB", (540, 960), "gray") for _ in range(4)]

        credential = adapter._local_qwerty_orientation_credential(
            requested=requested,
            scene=planned,
            frames=frames,
        )

        self.assertIsNotNone(credential)
        self.assertEqual(LOCAL_QWERTY_ORIENTATION_SOURCE, credential.source)
        self.assertEqual("upright", credential.phone_content_rotation)

        rotated = {
            **anchors,
            "q": [115, 844],
            "p": [875, 844],
            "z": [241, 704],
            "m": [747, 704],
            "backspace": [875, 704],
        }
        rejected = self._adapter(
            FakeSceneObserver([]),
            FakeRobot(),
            qwerty_row_snapper=lambda _frames, _anchors: rotated,
        )._local_qwerty_orientation_credential(
            requested=requested,
            scene=planned,
            frames=frames,
        )
        self.assertIsNone(rejected)

    def test_strict_input_execute_uses_zero_duplicate_pre_action_model_audits(self):
        gray = Image.new("RGB", (540, 960), "gray")
        planned = self._strict_primary_input_scene(
            _local_frame_fingerprint(gray)
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
            qwerty_row_snapper=lambda _frames, _anchors: anchors,
        )
        result = adapter.execute(
            requested_action=SemanticAction(
                node_id="type-body",
                action="input_verified_text",
                params={
                    "formal_candidate_id": "candidate-type-body",
                    "formal_report_digest": "a" * 64,
                    "formal_transition": {
                        "transition_id": "transition-type-body",
                        "precondition_claim_ids": ["claim-body-empty"],
                        "expectations": [
                            {
                                "subject_ref": "element.local_audited_input_1",
                                "predicate": "element.state.value",
                                "operator": "equals",
                                "value": "agent",
                            }
                        ],
                        "exploratory": False,
                    },
                    "element_id": field.element_id,
                    "target": field.meaning,
                    "role": field.role,
                    "label": field.label,
                    "states": dict(field.states),
                    "text": "agent",
                    "expected_effect": {
                        "element_state": {
                            "meaning": "application_text_input",
                            "states": {"value": "agent"},
                        }
                    },
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
        self.assertEqual([], observer.geometry_audit_calls)
        self.assertTrue(result.primary_input_confirmation_reused)
        self.assertEqual("matched", result.action_outcome)
        self.assertEqual(
            LOCAL_QWERTY_ORIENTATION_SOURCE,
            observer.last_orientation_audit_diagnostics["audit_source"],
        )
        self.assertEqual(
            0,
            observer.last_orientation_audit_diagnostics["model_calls"],
        )

    def test_literal_key_receipt_reconciles_only_proven_visual_soft_wrap(self):
        before = self._literal_input_scene("before", value="live", audited=True)
        expected_effect = {
            "element_state": {
                "meaning": "application_text_input",
                "states": {"value": "live2"},
            }
        }
        resolved = ResolvedSemanticAction(
            node_id="literal-2",
            kind="tap_semantic",
            normalized_point=(0.22, 0.735),
            target_element_id="local_audited_literal_key_1",
            before_fingerprint=before.fingerprint,
            expected_effect=expected_effect,
        )
        wrapped = self._literal_input_scene(
            "after-live2",
            value="li\nve2",
            include_key=False,
        )

        reconciled = GenericSingleActionAdapter._reconcile_literal_key_visual_wrap(
            resolved,
            before,
            wrapped,
        )

        input_element = reconciled.get_element("local_audited_input_1")
        self.assertEqual("live2", input_element.states["value"])
        self.assertEqual("live2", input_element.label)
        self.assertIn("本地逐键回执确认该换行为控件视觉软折行", input_element.evidence)
        UniversalActionController().verify_after_action(resolved, before, reconciled)

        wrong_text = self._literal_input_scene(
            "after-wrong",
            value="li\nve3",
            include_key=False,
        )
        missing_evidence_input = wrapped.get_element("local_audited_input_1")
        missing_evidence = replace(
            wrapped,
            elements=(
                replace(missing_evidence_input, evidence=("可见输入框",)),
            ),
        )
        for unsafe_after in (wrong_text, missing_evidence):
            with self.subTest(fingerprint=unsafe_after.fingerprint):
                self.assertIs(
                    unsafe_after,
                    GenericSingleActionAdapter._reconcile_literal_key_visual_wrap(
                        resolved,
                        before,
                        unsafe_after,
                    ),
                )

        newline_target = replace(
            resolved,
            expected_effect={
                "element_state": {
                    "meaning": "application_text_input",
                    "states": {"value": "live\n2"},
                }
            },
        )
        self.assertIs(
            wrapped,
            GenericSingleActionAdapter._reconcile_literal_key_visual_wrap(
                newline_target,
                before,
                wrapped,
            ),
        )

    def test_verified_text_reconciles_only_cross_boundary_horizontal_suffix(self):
        cases = (
            (
                "abcdefghijklmnopqrst",
                "uvwxyzabcdefghijk",
                "nopqrstuvwxyzabcdefghijk",
                "input_field_1",
            ),
            (
                "releasecandidate",
                "continuation",
                "didatecontinuation",
                "notes_field",
            ),
        )
        for prior, fragment, observed, field_id in cases:
            with self.subTest(field_id=field_id):
                expected = prior + fragment
                before = self._literal_input_scene(
                    "before-" + field_id,
                    value=prior,
                    include_key=False,
                )
                before_input = before.get_element("local_audited_input_1")
                before = replace(
                    before,
                    elements=(
                        replace(
                            before_input,
                            states={
                                **before_input.states,
                                "input_field_id": field_id,
                                "input_multiline": False,
                            },
                        ),
                    ),
                )
                after = self._literal_input_scene(
                    "after-" + field_id,
                    value=observed,
                    include_key=False,
                )
                after_input = after.get_element("local_audited_input_1")
                after = replace(
                    after,
                    screen_id="unknown",
                    elements=(
                        replace(
                            after_input,
                            bounds=(0.13, 0.68, 0.87, 0.77),
                            states={
                                **after_input.states,
                                "input_field_id": field_id,
                                "input_multiline": False,
                            },
                        ),
                    ),
                )
                resolved = ResolvedSemanticAction(
                    node_id="verified-text-" + field_id,
                    kind="input_verified_text",
                    text=expected,
                    input_fragment=fragment,
                    input_method="direct_latin",
                    prior_input_value=prior,
                    expected_input_value=expected,
                    target_element_id="local_audited_input_1",
                    before_fingerprint=before.fingerprint,
                    expected_effect={
                        "element_state": {
                            "meaning": "application_text_input",
                            "states": {"value": expected},
                        }
                    },
                    formal_candidate_id="candidate-" + field_id,
                    formal_transition={
                        "transition_id": "transition-" + field_id,
                        "precondition_claim_ids": ["claim-" + field_id],
                        "expectations": [
                            {
                                "subject_ref": "element." + field_id,
                                "predicate": "element.state.value",
                                "operator": "equals",
                                "value": expected,
                            }
                        ],
                        "exploratory": False,
                    },
                )

                reconciled = (
                    GenericSingleActionAdapter._reconcile_verified_text_horizontal_suffix(
                        resolved,
                        before,
                        after,
                    )
                )

                reconciled_input = reconciled.get_element("local_audited_input_1")
                self.assertEqual(expected, reconciled_input.states["value"])
                self.assertEqual(
                    observed,
                    reconciled_input.states["visible_value_suffix"],
                )
                self.assertEqual(
                    "horizontal_suffix",
                    reconciled_input.states["value_visibility"],
                )
                UniversalActionController().verify_after_action(
                    resolved,
                    before,
                    reconciled,
                )

                wrong_field_input = replace(
                    after_input,
                    states={
                        **after_input.states,
                        "input_field_id": field_id + "-other",
                        "input_multiline": False,
                    },
                )
                no_overlap_input = replace(
                    after_input,
                    label=fragment,
                    states={
                        **after_input.states,
                        "value": fragment,
                        "input_field_id": field_id,
                        "input_multiline": False,
                    },
                    evidence=(f"应用输入框当前文字：{fragment}", "caret"),
                )
                wrong_prefix = "x" + observed[1:]
                wrong_prefix_input = replace(
                    after_input,
                    label=wrong_prefix,
                    states={
                        **after_input.states,
                        "value": wrong_prefix,
                        "input_field_id": field_id,
                        "input_multiline": False,
                    },
                    evidence=(f"应用输入框当前文字：{wrong_prefix}", "caret"),
                )
                missing_evidence_input = replace(
                    after.get_element("local_audited_input_1"),
                    evidence=("可见输入框", "caret"),
                )
                for unsafe_input in (
                    wrong_field_input,
                    no_overlap_input,
                    wrong_prefix_input,
                    missing_evidence_input,
                ):
                    unsafe_after = replace(after, elements=(unsafe_input,))
                    self.assertIs(
                        unsafe_after,
                        GenericSingleActionAdapter._reconcile_verified_text_horizontal_suffix(
                            resolved,
                            before,
                            unsafe_after,
                        ),
                    )

    def test_missing_ordinary_button_cannot_use_input_auxiliary_recovery(self):
        planned = scene("ordinary-planned")
        requested = SemanticAction(
            node_id="ordinary",
            action="tap_semantic",
            params={
                "formal_candidate_id": "candidate-ordinary",
                "element_id": planned.elements[0].element_id,
                "target": planned.elements[0].meaning,
                "role": planned.elements[0].role,
                "label": planned.elements[0].label,
                "states": dict(planned.elements[0].states),
                "expected_effect": {"scene_changed": True},
            },
        )

        self.assertIsNone(
            GenericSingleActionAdapter._local_input_auxiliary_recovery_target(
                requested,
                planned,
                replace(planned, elements=(), fingerprint="fresh-empty"),
            )
        )
        self.assertFalse(
            GenericSingleActionAdapter._confirmation_allows_omitted_local_input_auxiliary(
                requested,
                planned,
            )
        )

    def test_typed_field_identity_recovers_placeholder_loss_only_for_exact_prefix(self):
        def typed_scene(fingerprint, value="first\n"):
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
                        states={
                            **field.states,
                            "input_field_id": "input_field_1",
                            "input_multiline": True,
                            "keyboard_geometry": TEST_QWERTY_GEOMETRY,
                        },
                    ),
                ),
            )

        planned = typed_scene("planned")
        field = planned.elements[0]
        expected_effect = {
            "element_state": {
                "meaning": "application_text_input",
                "states": {"value": "first\nsecond"},
            }
        }
        requested = SemanticAction(
            node_id="append-second",
            action="input_verified_text",
            params={
                "formal_candidate_id": "candidate-append-second",
                "element_id": field.element_id,
                "target": field.meaning,
                "role": field.role,
                "label": field.label,
                "states": dict(field.states),
                "text": "first\nsecond",
                "expected_effect": expected_effect,
            },
        )
        empty_fresh = replace(planned, fingerprint="fresh", elements=())

        recovered = GenericSingleActionAdapter._recover_omitted_verified_input_scene(
            requested,
            planned,
            empty_fresh,
        )

        self.assertIsNotNone(recovered)
        recovered_field = recovered.get_element("local_audited_input_1")
        self.assertEqual("input_field_1", recovered_field.states["input_field_id"])
        self.assertEqual("first\n", recovered_field.states["value"])
        self.assertIn("授权文字精确前缀", recovered_field.evidence[-1])

        visible_without_typed_id = replace(
            empty_fresh,
            elements=(
                replace(
                    field,
                    element_id="fresh-visible-input",
                    bounds=(0.14, 0.51, 0.72, 0.61),
                    states={
                        key: ("first" if key == "value" else value)
                        for key, value in field.states.items()
                        if key != "input_field_id"
                    },
                    evidence=("应用输入框当前文字：first", "caret"),
                ),
                UIElement(
                    element_id="fresh-enter",
                    role="button",
                    meaning="input_exact_enter_key",
                    label="↵",
                    bounds=(0.82, 0.89, 0.94, 0.96),
                    confidence=1.0,
                    states={
                        "input_enter_key": True,
                        "input_field_id": "input_field_1",
                    },
                    evidence=("可见换行键",),
                ),
            ),
        )
        varied = GenericSingleActionAdapter._recover_omitted_verified_input_scene(
            requested,
            planned,
            visible_without_typed_id,
        )
        self.assertIsNotNone(varied)
        varied_field = varied.get_element("local_audited_input_1")
        self.assertEqual((0.14, 0.51, 0.72, 0.61), varied_field.bounds)
        self.assertEqual("input_field_1", varied_field.states["input_field_id"])

        wrong_prefix = replace(
            requested,
            params={
                **requested.params,
                "text": "other\nsecond",
            },
        )
        duplicate = replace(
            empty_fresh,
            elements=(field, replace(field, element_id="duplicate-field")),
        )
        for unsafe_request, unsafe_fresh in (
            (wrong_prefix, empty_fresh),
            (requested, duplicate),
        ):
            with self.subTest(
                text=unsafe_request.params["text"],
                element_count=len(unsafe_fresh.elements),
            ):
                self.assertIsNone(
                    GenericSingleActionAdapter._recover_omitted_verified_input_scene(
                        unsafe_request,
                        planned,
                        unsafe_fresh,
                    )
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
            qwerty_row_snapper=lambda _frames, anchors: anchors,
            require_local_qwerty_row_snap=True,
        )
        field = planned.elements[0]
        expected = "first\nsecond"

        result = adapter.execute(
            requested_action=SemanticAction(
                node_id="append-second",
                action="input_verified_text",
                params={
                    "formal_candidate_id": "candidate-append-second",
                    "formal_report_digest": "a" * 64,
                    "formal_transition": {
                        "transition_id": "transition-append-second",
                        "precondition_claim_ids": ["claim-append-second"],
                        "expectations": [
                            {
                                "subject_ref": "element.input_field_1",
                                "predicate": "element.state.value",
                                "operator": "equals",
                                "value": expected,
                            }
                        ],
                        "exploratory": False,
                    },
                    "element_id": field.element_id,
                    "target": field.meaning,
                    "role": field.role,
                    "label": field.label,
                    "states": dict(field.states),
                    "text": expected,
                    "expected_effect": {
                        "element_state": {
                            "meaning": field.meaning,
                            "states": {"value": expected},
                        }
                    },
                },
            ),
            planned_scene=planned,
            planned_frames=tuple(gray.copy() for _ in range(4)),
            goal=goal(),
            confirmed=True,
        )

        self.assertEqual([("input", "second")], robot.actions)
        self.assertEqual(1, result.physical_actions)
        self.assertEqual("matched", result.action_outcome)
        self.assertEqual("input_field_1", result.before_scene.elements[0].states["input_field_id"])
        self.assertEqual([], observer.geometry_audit_calls)

    def test_completed_navigation_observes_result_without_source_target(self):
        planned = scene("planned")
        fresh = replace(planned, fingerprint="fresh")
        after = scene(
            "after",
            screen_id="browser_home",
            app_id="browser",
            element_id="browser-content",
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
                node_id="open-browser",
                action="tap_semantic",
                params={
                    "element_id": "e1",
                    "target": "app_icon",
                    "role": "icon",
                    "label": "设置",
                    "states": {},
                    "expected_effect": {
                        "scene_changed": True,
                        "goal_complete_on_success": True,
                    },
                },
            ),
            planned_scene=planned,
            planned_frames=tuple(
                Image.new("RGB", (540, 960), "gray") for _ in range(4)
            ),
            goal=navigation_goal(),
            confirmed=True,
        )

        self.assertEqual(1, result.physical_actions)
        self.assertEqual([], robot.calibrated_target_requests)
        self.assertEqual(1, len(observer.goal_contexts))
        post_focus = observer.goal_contexts[0]["entities"][
            "active_subgoal_visual_context"
        ]
        self.assertNotIn("target_ui_label", post_focus["goal_entities"])
        self.assertEqual(
            "verified_navigation_result_v1",
            post_focus["goal_entities"]["observation_phase"],
        )
        self.assertEqual("观察本次导航后的当前稳定画面", post_focus["objective"])

    def test_post_navigation_result_context_is_fail_closed(self):
        safe = ResolvedSemanticAction(
            node_id="open-browser",
            kind="tap_semantic",
            expected_effect={
                "scene_changed": True,
                "goal_complete_on_success": True,
            },
        )
        original = navigation_goal().to_dict()
        self.assertNotEqual(
            original,
            _post_action_observation_context(
                navigation_goal(),
                safe,
                physical_action_executed=True,
            ),
        )

        self.assertEqual(
            original,
            _post_action_observation_context(navigation_goal(), safe),
        )

        unsafe_cases = (
            (
                navigation_goal(execution_class="effect"),
                safe,
            ),
            (
                navigation_goal(),
                replace(
                    safe,
                    expected_effect={
                        "scene_changed": True,
                        "goal_complete_on_success": True,
                        "element_state": {
                            "meaning": "toggle",
                            "states": {"checked": True},
                        },
                    },
                ),
            ),
            (navigation_goal(), replace(safe, kind="input_verified_text")),
            (
                navigation_goal(),
                replace(
                    safe,
                    expected_effect={
                        "scene_changed": True,
                        "goal_complete_on_success": True,
                        "system_ui": {"navigation_bar_visible": True},
                    },
                ),
            ),
        )
        for case_goal, resolved in unsafe_cases:
            with self.subTest(
                execution_class=case_goal.entities["active_subgoal_visual_context"][
                    "execution_class"
                ],
                kind=resolved.kind,
                expected=resolved.expected_effect,
            ):
                self.assertEqual(
                    case_goal.to_dict(),
                    _post_action_observation_context(
                        case_goal,
                        resolved,
                        physical_action_executed=True,
                    ),
                )

        spoofed = navigation_goal(execution_class="effect")
        spoofed_focus = spoofed.entities["active_subgoal_visual_context"]
        spoofed_focus["objective"] = "观察本次导航后的当前稳定画面"
        spoofed_focus["completion_conditions"] = [
            "当前稳定结果画面已被重新观察"
        ]
        spoofed_focus["goal_entities"]["observation_phase"] = (
            "verified_navigation_result_v1"
        )
        sanitized = _post_action_observation_context(spoofed, safe)
        self.assertNotIn(
            "observation_phase",
            sanitized["entities"]["active_subgoal_visual_context"][
                "goal_entities"
            ],
        )

    def test_post_action_context_reuses_unique_next_focus_in_same_observation(self):
        current_goal = navigation_goal()
        current_goal.entities["next_subgoal_visual_context"] = {
            "subgoal_id": "read_title",
            "objective": "读取当前页面主标题",
            "constraints": ["仅读取"],
            "completion_conditions": ["页面主标题已读取"],
            "execution_class": "observe",
            "goal_entities": {"target_surface": "current_surface"},
        }
        resolved = ResolvedSemanticAction(
            node_id="open-browser",
            kind="tap_semantic",
            expected_effect={"scene_changed": True},
        )

        result = _post_action_observation_context(
            current_goal,
            resolved,
            physical_action_executed=True,
        )
        focus = result["entities"]["active_subgoal_visual_context"]

        self.assertEqual("read_title", focus["subgoal_id"])
        self.assertEqual(
            FUSED_POST_ACTION_NEXT_STEP_OBSERVATION_PHASE,
            focus["goal_entities"]["observation_phase"],
        )

    def test_input_focus_switches_to_next_only_at_typed_terminal_value(self):
        current_goal = navigation_goal()
        active = current_goal.entities["active_subgoal_visual_context"]
        active["objective"] = "在主题字段输入 first"
        active["goal_entities"].update(
            {
                "active_input_transaction_text": "first",
                "active_input_field_id": "subject_field",
                "active_input_field_label": "主题",
                "active_input_multiline": False,
            }
        )
        current_goal.entities["next_subgoal_visual_context"] = {
            "subgoal_id": "input_body",
            "objective": "在正文字段输入 second",
            "constraints": ["不要发送"],
            "completion_conditions": ["正文字段逐字为 second"],
            "execution_class": "navigate",
            "goal_entities": {
                "active_input_transaction_text": "second",
                "active_input_field_id": "body_field",
                "active_input_field_label": "正文",
                "active_input_multiline": False,
            },
        }
        partial = ResolvedSemanticAction(
            node_id="type-partial",
            kind="input_verified_text",
            prior_input_value="",
            expected_input_value="fir",
            expected_effect={
                "element_state": {
                    "meaning": "application_text_input",
                    "states": {"value": "fir"},
                }
            },
        )
        terminal = replace(partial, expected_input_value="first")

        partial_context = _post_action_observation_context(
            current_goal,
            partial,
            physical_action_executed=True,
        )
        terminal_context = _post_action_observation_context(
            current_goal,
            terminal,
            physical_action_executed=True,
        )

        self.assertEqual(
            "open_browser",
            partial_context["entities"]["active_subgoal_visual_context"][
                "subgoal_id"
            ],
        )
        self.assertEqual(
            "input_body",
            terminal_context["entities"]["active_subgoal_visual_context"][
                "subgoal_id"
            ],
        )

    def test_executed_back_and_swipe_use_result_focused_compact_observation(self):
        cases = (
            (
                "back-live",
                "back",
                {"scene_changed": True},
                "按一次返回键",
                ["动作后出现新的稳定画面"],
            ),
            (
                "swipe-live",
                "swipe",
                {"content_changed": True},
                "在当前模式选择页向上滑动一次",
                ["验收模式选项出现在当前画面中"],
            ),
            (
                "back-content-variation",
                "back",
                {"content_changed": True},
                "返回上一层并观察内容变化",
                ["当前内容已变化"],
            ),
            (
                "swipe-scene-variation",
                "swipe",
                {"scene_changed": True},
                "横向滑动到下一页",
                ["下一页稳定画面已显示"],
            ),
        )
        for node_id, kind, expected_effect, objective, conditions in cases:
            with self.subTest(kind=kind, expected_effect=expected_effect):
                current_goal = navigation_goal()
                focus = current_goal.entities["active_subgoal_visual_context"]
                focus["objective"] = objective
                focus["completion_conditions"] = conditions
                resolved = ResolvedSemanticAction(
                    node_id=node_id,
                    kind=kind,
                    expected_effect=expected_effect,
                )

                result = _post_action_observation_context(
                    current_goal,
                    resolved,
                    physical_action_executed=True,
                )

                result_focus = result["entities"]["active_subgoal_visual_context"]
                self.assertNotIn("target_ui_label", result_focus["goal_entities"])
                self.assertEqual(
                    "verified_navigation_result_v1",
                    result_focus["goal_entities"]["observation_phase"],
                )
                self.assertEqual(
                    "观察本次导航后的当前稳定画面",
                    result_focus["objective"],
                )

    def test_executed_back_and_swipe_attest_result_context_in_adapter(self):
        cases = (
            ("back", {"scene_changed": True}, {}),
            ("swipe", {"content_changed": True}, {"direction": "up"}),
        )
        for kind, expected_effect, action_params in cases:
            with self.subTest(kind=kind):
                planned = scene("planned", screen_id="verification_list")
                after = scene("after", screen_id="verification_list_after")
                observer = FakeSceneObserver([after])
                robot = FakeRobot()
                adapter = GenericSingleActionAdapter(
                    capture=SequenceCapture(["gray"] * 4 + ["white"] * 4),
                    observer=observer,
                    robot=robot,
                    frame_interval=0,
                    post_action_settle=0,
                )
                params = dict(action_params)
                params["expected_effect"] = expected_effect

                result = adapter.execute(
                    requested_action=SemanticAction(
                        node_id=f"live-{kind}",
                        action=kind,
                        params=params,
                    ),
                    planned_scene=planned,
                    planned_frames=tuple(
                        Image.new("RGB", (540, 960), "gray") for _ in range(4)
                    ),
                    goal=navigation_goal(),
                    confirmed=True,
                )

                self.assertEqual(1, result.physical_actions)
                post_focus = observer.goal_contexts[-1]["entities"][
                    "active_subgoal_visual_context"
                ]
                self.assertEqual(
                    "verified_navigation_result_v1",
                    post_focus["goal_entities"]["observation_phase"],
                )

    def test_back_and_swipe_result_marker_rejects_unexecuted_or_wrong_effect(self):
        base_goal = navigation_goal()
        cases = (
            (
                False,
                ResolvedSemanticAction(
                    node_id="unexecuted-back",
                    kind="back",
                    expected_effect={"scene_changed": True},
                ),
            ),
            (
                True,
                ResolvedSemanticAction(
                    node_id="unchanged-swipe",
                    kind="swipe",
                    expected_effect={"content_changed": False},
                ),
            ),
            (
                True,
                ResolvedSemanticAction(
                    node_id="wrong-effect-back",
                    kind="back",
                    expected_effect={
                        "scene_changed": True,
                        "element_state": {"meaning": "toggle"},
                    },
                ),
            ),
            (
                True,
                ResolvedSemanticAction(
                    node_id="unknown-effect-swipe",
                    kind="swipe",
                    expected_effect={
                        "content_changed": True,
                        "coordinates_changed": True,
                    },
                ),
            ),
        )
        for executed, resolved in cases:
            with self.subTest(node_id=resolved.node_id):
                self.assertEqual(
                    base_goal.to_dict(),
                    _post_action_observation_context(
                        base_goal,
                        resolved,
                        physical_action_executed=executed,
                    ),
                )

    def test_stable_local_ocr_snaps_qwerty_row_heights(self):
        payload = {
            "lines": [
                {
                    "words": [
                        {"text": "q", "top": 1000, "height": 24},
                        {"text": "w", "top": 1002, "height": 22},
                        {"text": "e", "top": 1001, "height": 24},
                        {"text": "c", "top": 1204, "height": 22},
                        {"text": "v", "top": 1203, "height": 24},
                        {"text": "b", "top": 1205, "height": 22},
                    ]
                }
            ]
        }
        frames = [Image.new("RGB", (810, 1440), "gray") for _ in range(3)]
        anchors = {
            "q": [120, 730],
            "p": [880, 730],
            "a": [180, 810],
            "l": [820, 810],
            "z": [280, 890],
            "m": [720, 890],
            "backspace": [880, 890],
        }

        snapped = stable_qwerty_ocr_anchors(
            frames,
            anchors,
            ocr_recognizer=lambda *_args, **_kwargs: payload,
        )

        self.assertEqual(703, snapped["q"][1])
        self.assertEqual(774, snapped["a"][1])
        self.assertEqual(844, snapped["z"][1])
        self.assertEqual(844, snapped["backspace"][1])

    def test_stable_local_ocr_repairs_vertically_compressed_model_rows(self):
        payload = {
            "lines": [
                {
                    "words": [
                        {"text": "q", "top": 1000, "height": 24},
                        {"text": "w", "top": 1002, "height": 22},
                        {"text": "e", "top": 1001, "height": 24},
                        {"text": "c", "top": 1204, "height": 22},
                        {"text": "v", "top": 1203, "height": 24},
                        {"text": "b", "top": 1205, "height": 22},
                    ]
                }
            ]
        }
        frames = [Image.new("RGB", (810, 1440), "gray") for _ in range(3)]
        compressed = {
            "q": [110, 895],
            "p": [890, 895],
            "a": [160, 945],
            "l": [840, 945],
            "z": [260, 990],
            "m": [740, 990],
            "backspace": [890, 990],
        }

        snapped = stable_qwerty_ocr_anchors(
            frames,
            compressed,
            ocr_recognizer=lambda *_args, **_kwargs: payload,
        )

        self.assertIsNotNone(snapped)
        self.assertEqual(703, snapped["q"][1])
        self.assertEqual(774, snapped["a"][1])
        self.assertEqual(844, snapped["z"][1])

    def test_stable_local_ocr_rebuilds_untrusted_portrait_grid_geometry(self):
        payload = {
            "lines": [
                {
                    "words": [
                        {"text": "q", "left": 90, "width": 20, "top": 1000, "height": 24},
                        {"text": "w", "left": 170, "width": 20, "top": 1002, "height": 22},
                        {"text": "e", "left": 250, "width": 20, "top": 1001, "height": 24},
                        {"text": "c", "left": 370, "width": 20, "top": 1204, "height": 22},
                        {"text": "v", "left": 450, "width": 20, "top": 1203, "height": 24},
                        {"text": "b", "left": 530, "width": 20, "top": 1205, "height": 22},
                    ]
                }
            ]
        }
        frames = [Image.new("RGB", (1000, 1440), "gray") for _ in range(3)]
        portrait_grid = {
            "q": [110, 1430],
            "p": [890, 1430],
            "a": [160, 1580],
            "l": [840, 1580],
            "z": [260, 1730],
            "m": [740, 1730],
            "backspace": [890, 1730],
        }

        snapped = stable_qwerty_ocr_anchors(
            frames,
            portrait_grid,
            ocr_recognizer=lambda *_args, **_kwargs: payload,
        )

        self.assertEqual([100, 703], snapped["q"])
        self.assertEqual([820, 703], snapped["p"])
        self.assertEqual([140, 774], snapped["a"])
        self.assertEqual([780, 774], snapped["l"])
        self.assertEqual([220, 844], snapped["z"])
        self.assertEqual([700, 844], snapped["m"])
        self.assertEqual([820, 844], snapped["backspace"])

    def test_stable_local_ocr_rejects_untrusted_grid_without_local_row_layout(self):
        payload = {
            "lines": [
                {
                    "words": [
                        {"text": "q", "left": 90, "width": 20, "top": 1000, "height": 24},
                        {"text": "w", "left": 390, "width": 20, "top": 1002, "height": 22},
                        {"text": "e", "left": 250, "width": 20, "top": 1001, "height": 24},
                        {"text": "c", "left": 370, "width": 20, "top": 1204, "height": 22},
                        {"text": "v", "left": 150, "width": 20, "top": 1203, "height": 24},
                        {"text": "b", "left": 530, "width": 20, "top": 1205, "height": 22},
                    ]
                }
            ]
        }
        frames = [Image.new("RGB", (1000, 1440), "gray") for _ in range(3)]
        portrait_grid = {
            "q": [110, 1430],
            "p": [890, 1430],
            "a": [160, 1580],
            "l": [840, 1580],
            "z": [260, 1730],
            "m": [740, 1730],
            "backspace": [890, 1730],
        }

        self.assertIsNone(
            stable_qwerty_ocr_anchors(
                frames,
                portrait_grid,
                ocr_recognizer=lambda *_args, **_kwargs: payload,
            )
        )

    def test_stable_local_ocr_allows_one_frame_row_dropout(self):
        complete = {
            "lines": [
                {
                    "words": [
                        {"text": "q", "top": 1000, "height": 24},
                        {"text": "w", "top": 1002, "height": 22},
                        {"text": "e", "top": 1001, "height": 24},
                        {"text": "c", "top": 1204, "height": 22},
                        {"text": "v", "top": 1203, "height": 24},
                        {"text": "b", "top": 1205, "height": 22},
                    ]
                }
            ]
        }
        top_only = {
            "lines": [
                {
                    "words": [
                        {"text": "q", "top": 1000, "height": 24},
                        {"text": "w", "top": 1002, "height": 22},
                        {"text": "e", "top": 1001, "height": 24},
                    ]
                }
            ]
        }
        payloads = iter((complete, complete, top_only))
        frames = [Image.new("RGB", (810, 1440), "gray") for _ in range(3)]
        compressed = {
            "q": [110, 900],
            "p": [890, 900],
            "a": [150, 945],
            "l": [850, 945],
            "z": [250, 985],
            "m": [750, 985],
            "backspace": [890, 985],
        }

        snapped = stable_qwerty_ocr_anchors(
            frames,
            compressed,
            ocr_recognizer=lambda *_args, **_kwargs: next(payloads),
        )

        self.assertIsNotNone(snapped)
        self.assertEqual(703, snapped["q"][1])
        self.assertEqual(844, snapped["z"][1])

    def test_stable_local_ocr_rejects_only_one_complete_frame(self):
        complete = {
            "lines": [
                {
                    "words": [
                        {"text": "q", "top": 1000, "height": 24},
                        {"text": "w", "top": 1002, "height": 22},
                        {"text": "c", "top": 1204, "height": 22},
                        {"text": "v", "top": 1203, "height": 24},
                    ]
                }
            ]
        }
        empty = {"lines": []}
        payloads = iter((complete, empty, empty))
        frames = [Image.new("RGB", (810, 1440), "gray") for _ in range(3)]
        compressed = {
            "q": [110, 900],
            "p": [890, 900],
            "a": [150, 945],
            "l": [850, 945],
            "z": [250, 985],
            "m": [750, 985],
            "backspace": [890, 985],
        }

        self.assertIsNone(
            stable_qwerty_ocr_anchors(
                frames,
                compressed,
                ocr_recognizer=lambda *_args, **_kwargs: next(payloads),
            )
        )

    def test_stable_local_ocr_rejects_incomplete_row_evidence(self):
        payload = {
            "lines": [
                {
                    "words": [
                        {"text": "q", "top": 1000, "height": 24},
                        {"text": "w", "top": 1002, "height": 22},
                        {"text": "z", "top": 1204, "height": 22},
                    ]
                }
            ]
        }
        frames = [Image.new("RGB", (810, 1440), "gray") for _ in range(3)]
        compressed = {
            "q": [110, 895],
            "p": [890, 895],
            "a": [160, 945],
            "l": [840, 945],
            "z": [260, 990],
            "m": [740, 990],
            "backspace": [890, 990],
        }

        self.assertIsNone(
            stable_qwerty_ocr_anchors(
                frames,
                compressed,
                ocr_recognizer=lambda *_args, **_kwargs: payload,
            )
        )

    def _assert_public_observation_failure_before_robot(self, responses):
        provider = RawSceneProvider(responses)
        robot = FakeRobot()
        adapter = self._adapter(GenericSceneObserver(provider), robot)
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
        self.assertEqual([2600], provider.max_tokens_seen)

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

    def test_diagnostic_write_failure_preserves_primary_observation_error(self):
        observer = RawFailureSceneObserver('{"broken":true}')
        with tempfile.TemporaryDirectory() as temp, patch(
            "generic_action_adapter._persist_qwen_failure_diagnostic",
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

    def test_public_execute_ambiguous_observation_never_calls_robot(self):
        first = compact_scene_raw(label="设置")
        second = compact_scene_raw(label="蓝牙")
        with patch(
            "generic_scene_observer._single_json_structural_edits",
            side_effect=lambda _raw: iter((first, second)),
        ):
            self._assert_public_observation_failure_before_robot(
                ['{"bad":}']
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
        self.assertIn("swipe", supported)

    def test_orientation_mismatch_blocks_tap_back_home_and_drag_before_robot(self):
        mismatch = CameraAlignmentFacts(
            camera_layout_orientation="portrait",
            phone_content_rotation="rotated_90",
            confidence=0.95,
            evidence=("手机界面文字需旋转九十度才正向",),
        )
        ordinary_scene = scene("planned", camera_alignment=mismatch)
        drag_scene = UIScene(
            app_id="settings",
            screen_id="arrange",
            summary="两个可拖动对象",
            elements=(
                UIElement(
                    element_id="source",
                    role="icon",
                    meaning="source_item",
                    label="源",
                    bounds=(0.15, 0.25, 0.25, 0.35),
                    confidence=0.96,
                ),
                UIElement(
                    element_id="destination",
                    role="icon",
                    meaning="destination_slot",
                    label="目标",
                    bounds=(0.60, 0.25, 0.70, 0.35),
                    confidence=0.96,
                ),
            ),
            confidence=0.96,
            fingerprint="planned",
            camera_alignment=mismatch,
        )
        cases = (
            (
                "tap_semantic",
                ordinary_scene,
                SemanticAction(
                    node_id="blocked-tap",
                    action="tap_semantic",
                    params={"element_id": "e1", "target": "app_icon"},
                ),
            ),
            (
                "back",
                ordinary_scene,
                SemanticAction(node_id="blocked-back", action="back", params={}),
            ),
            (
                "home",
                ordinary_scene,
                SemanticAction(node_id="blocked-home", action="home", params={}),
            ),
            (
                "drag",
                drag_scene,
                SemanticAction(
                    node_id="blocked-drag",
                    action="drag",
                    params={
                        "source_element_id": "source",
                        "destination_element_id": "destination",
                        "expected_effect": {"scene_changed": True},
                    },
                ),
            ),
        )
        frames = tuple(Image.new("RGB", (540, 960), "gray") for _ in range(4))
        for kind, planned, action in cases:
            with self.subTest(kind=kind):
                robot = FakeRobot()
                adapter = GenericSingleActionAdapter(
                    capture=SequenceCapture(["gray"] * 4),
                    observer=FakeSceneObserver(
                        (
                            [planned]
                            if action.action
                            in GenericSingleActionAdapter.GEOMETRY_BOUND_KINDS
                            else []
                        ),
                        audit_rotation="rotated_90",
                    ),
                    robot=robot,
                    frame_interval=0,
                    post_action_settle=0,
                )
                with self.assertRaisesRegex(
                    GenericActionAdapterError,
                    "方向不一致或未知",
                ) as caught:
                    adapter.execute(
                        requested_action=action,
                        planned_scene=planned,
                        planned_frames=frames,
                        goal=goal(),
                        confirmed=True,
                    )
                self.assertEqual(0, caught.exception.physical_actions)
                self.assertEqual([], robot.actions)

    def test_orientation_gate_rejects_unknown_low_confidence_or_local_mismatch(self):
        cases = (
            ("rotated_90", 0.95, "方向不一致或未知"),
            ("rotated_180", 0.95, "方向不一致或未知"),
            ("rotated_270", 0.95, "方向不一致或未知"),
            ("unknown", 0.95, "方向不一致或未知"),
            ("upright", 0.4, "置信度不足"),
        )
        frames = tuple(Image.new("RGB", (540, 960), "gray") for _ in range(4))
        for rotation, confidence, error in cases:
            with self.subTest(rotation=rotation, confidence=confidence):
                robot = FakeRobot()
                adapter = GenericSingleActionAdapter(
                    capture=SequenceCapture(["gray"] * 4),
                    observer=FakeSceneObserver(
                        [],
                        audit_rotation=rotation,
                        audit_confidence=confidence,
                    ),
                    robot=robot,
                    frame_interval=0,
                    post_action_settle=0,
                )
                with self.assertRaisesRegex(GenericActionAdapterError, error) as caught:
                    planned = scene(
                        "planned",
                        camera_alignment=CameraAlignmentFacts(
                            camera_layout_orientation="portrait",
                            phone_content_rotation=rotation,
                            confidence=confidence,
                            evidence=("手机界面方向由本轮完整画面报告",),
                        ),
                    )
                    adapter.execute(
                        requested_action=SemanticAction(
                            node_id="blocked-back",
                            action="back",
                            params={},
                        ),
                        planned_scene=planned,
                        planned_frames=frames,
                        goal=goal(),
                        confirmed=True,
                    )
                self.assertEqual(0, caught.exception.physical_actions)
                self.assertEqual([], robot.actions)

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
                    "expected_effect": {
                        "system_ui": {"navigation_bar_visible": True}
                    }
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

    def test_reveal_system_navigation_failure_does_not_fallback_to_swipe(self):
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
                params={
                    "expected_effect": {
                        "system_ui": {"navigation_bar_visible": True}
                    }
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
        self.assertEqual("mismatched", result.action_outcome)
        self.assertTrue(result.verification_errors)
        self.assertEqual([("reveal_system_navigation",)], robot.actions)

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
            params={"expected_effect": {"scene_changed": True, "app_id": "launcher"}},
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
        self.assertEqual(0, observer.home_audit_calls)
        self.assertEqual(
            "single_step_scene_orientation",
            result.orientation_credential.source,
        )

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
                params={"expected_effect": {"scene_changed": True, "app_id": "launcher"}},
            ),
            planned_scene=planned,
            goal=goal(),
            confirmed=True,
        )

        self.assertEqual([("home",)], robot.actions)
        self.assertTrue(result.hardware_receipt["seller_event_barrier_confirmed"])
        self.assertFalse(result.hardware_receipt["mechanical_contact_ack"])
        self.assertIsNone(robot.consume_last_click_receipt())

    def test_invalid_home_click_receipt_stops_before_post_action_observation(self):
        planned = scene("planned", screen_id="settings_home", app_id="settings")
        fresh = scene("before", screen_id="settings_home", app_id="settings")
        observer = FakeSceneObserver([fresh])
        robot = ClickReceiptRobot(valid=False)

        with self.assertRaisesRegex(GenericActionAdapterError, "单击事件栅栏凭据"):
            self._adapter(observer, robot).execute(
                requested_action=SemanticAction(
                    node_id="return-to-launcher",
                    action="home",
                    params={"expected_effect": {"scene_changed": True, "app_id": "launcher"}},
                ),
                planned_scene=planned,
                goal=goal(),
                confirmed=True,
            )

        self.assertEqual([("home",)], robot.actions)
        self.assertEqual(1, observer.calls)

    def test_confirmed_tap_executes_exactly_once_and_reobserves(self):
        planned = scene("planned", bounds=(0.1, 0.2, 0.3, 0.4))
        fresh = scene("before", element_id="fresh", bounds=(0.11, 0.21, 0.31, 0.41))
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

    def test_confirmed_input_executes_exact_text_once_and_reobserves(self):
        def input_scene(fingerprint, element_id):
            return UIScene(
                app_id="browser",
                screen_id="search",
                summary="输入框已聚焦",
                elements=(
                    UIElement(
                        element_id=element_id,
                        role="input",
                        meaning="搜索输入框",
                        label="搜索",
                        bounds=(0.1, 0.1, 0.9, 0.2),
                        confidence=0.96,
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
                confidence=0.95,
                fingerprint=fingerprint,
            )

        planned = input_scene("planned", "field-planned")
        fresh = input_scene("before", "field-fresh")
        after = UIScene(
            app_id="browser",
            screen_id="search",
            summary="输入框已显示新文字",
            elements=(
                UIElement(
                    element_id="field-after",
                    role="input",
                    meaning="搜索输入框",
                    label="搜索",
                    bounds=(0.1, 0.1, 0.9, 0.2),
                    confidence=0.96,
                    states={"focused": True, "value": "agent"},
                ),
            ),
            stable=True,
            confidence=0.95,
            fingerprint="after",
        )
        observer = FakeSceneObserver([fresh, after])
        robot = FakeRobot()
        action = SemanticAction(
            node_id="input-step",
            action="input_verified_text",
            params={
                "element_id": "field-planned",
                "target": "搜索输入框",
                "role": "input",
                "label": "搜索",
                "states": {"focused": True},
                "text": "agent",
                "expected_effect": {
                    "element_state": {
                        "meaning": "搜索输入框",
                        "states": {"value": "agent"},
                    }
                },
            },
        )

        snapped_anchors = {
            key: list(value)
            for key, value in TEST_QWERTY_GEOMETRY["anchors"].items()
        }
        result = self._adapter(
            observer,
            robot,
            qwerty_row_snapper=lambda _frames, _anchors: snapped_anchors,
            require_local_qwerty_row_snap=True,
        ).execute(
            requested_action=action,
            planned_scene=planned,
            goal=goal(),
            confirmed=True,
        )

        self.assertEqual([("input", "agent")], robot.actions)
        self.assertEqual(
            {
                **TEST_QWERTY_GEOMETRY,
                "anchors": snapped_anchors,
                "row_snap_source": "stable_local_ocr",
            },
            robot.keyboard_layouts[0],
        )
        self.assertEqual(1, result.physical_actions)

        blocked_robot = FakeRobot()
        blocked_observer = FakeSceneObserver([fresh, after])
        with self.assertRaisesRegex(
            GenericActionAdapterError,
            "本地 OCR 未能稳定确认 QWERTY 三行中心",
        ) as caught:
            self._adapter(
                blocked_observer,
                blocked_robot,
                qwerty_row_snapper=lambda _frames, _anchors: None,
                require_local_qwerty_row_snap=True,
            ).execute(
                requested_action=action,
                planned_scene=planned,
                goal=goal(),
                confirmed=True,
            )
        self.assertEqual(0, caught.exception.physical_actions)
        self.assertEqual([], blocked_robot.actions)
        self.assertEqual(2, observer.calls)

    def test_confirmed_chinese_input_types_pinyin_then_requires_exact_candidate(self):
        class ContinuationObserver(FakeSceneObserver):
            input_lineage_store = object()
            supports_typed_input_continuation = True

            def __init__(self, scenes):
                super().__init__(scenes)
                self.lineage_overrides = []
                self.prior_scenes = []

            def observe(
                self,
                *,
                frames,
                goal_context=None,
                device_id=None,
                input_lineage_override=None,
                prior_scene=None,
            ):
                self.lineage_overrides.append(input_lineage_override)
                self.prior_scenes.append(prior_scene)
                return super().observe(frames=frames, goal_context=goal_context)

        def input_scene(fingerprint, *, ime=False):
            states = {
                "focused": True,
                "fully_visible": True,
                "value": "",
                "input_field_id": "input_field_1",
                "keyboard_layout": "qwerty",
                "keyboard_input_mode": "chinese_pinyin",
                "keyboard_geometry": TEST_QWERTY_GEOMETRY,
                "goal_relevant": True,
            }
            if ime:
                states.update(
                    {
                        "ime_preedit_text": "nihao",
                        "ime_exact_candidate_text": "你好",
                    }
                )
            return UIScene(
                app_id="chat",
                screen_id="conversation",
                summary="消息输入框和中文键盘可见",
                elements=(
                    UIElement(
                        element_id="field",
                        role="input",
                        meaning="application_text_input",
                        label="消息",
                        bounds=(0.1, 0.1, 0.9, 0.2),
                        confidence=0.97,
                        states=states,
                        evidence=("输入框与中文拼音键盘可见",),
                    ),
                ),
                stable=True,
                confidence=0.96,
                fingerprint=fingerprint,
            )

        before = input_scene("before")
        after = input_scene("after", ime=True)
        robot = FakeRobot()
        action = SemanticAction(
            node_id="input-chinese",
            action="input_verified_text",
            params={
                "element_id": "field",
                "target": "application_text_input",
                "role": "input",
                "label": "消息",
                "states": before.elements[0].states,
                "text": "你好",
                "expected_effect": {
                    "element_state": {
                        "meaning": "application_text_input",
                        "states": {
                            "value": "",
                            "ime_preedit_text": "nihao",
                            "ime_exact_candidate_text": "你好",
                        },
                    }
                },
            },
        )

        observer = ContinuationObserver([before, after])
        result = self._adapter(observer, robot).execute(
            requested_action=action,
            planned_scene=before,
            goal=goal(),
            confirmed=True,
        )

        self.assertEqual([("pinyin", "你好", "nihao")], robot.actions)
        self.assertEqual("matched", result.action_outcome)
        self.assertEqual(1, result.physical_actions)
        self.assertEqual(
            [None, "pending_verified_chinese_preedit_action"],
            [
                getattr(item, "source", None)
                for item in observer.lineage_overrides
            ],
        )
        self.assertEqual("before", observer.prior_scenes[-1].fingerprint)
        self.assertEqual(
            "input_field_1",
            observer.prior_scenes[-1].elements[0].states["input_field_id"],
        )

    def test_confirmed_chinese_input_accepts_localized_same_surface_identity(self):
        before_states = {
            "focused": True,
            "value": "复杂输入",
            "keyboard_layout": "qwerty",
            "keyboard_input_mode": "chinese_pinyin",
            "keyboard_geometry": TEST_QWERTY_GEOMETRY,
            "goal_relevant": True,
        }
        before = UIScene(
            app_id="微信",
            screen_id="聊天界面",
            summary="输入框显示既有中文，键盘已聚焦",
            elements=(
                UIElement(
                    element_id="field",
                    role="input",
                    meaning="application_text_input",
                    label="复杂输入",
                    bounds=(0.11, 0.52, 0.70, 0.57),
                    confidence=1.0,
                    states=before_states,
                ),
            ),
            stable=True,
            confidence=1.0,
            fingerprint="before",
            camera_alignment=aligned_camera_facts(),
        )
        after = UIScene(
            app_id="当前会话标题",
            screen_id="chat_input",
            summary="同一输入框显示验收的拼音和候选",
            elements=(
                replace(
                    before.elements[0],
                    bounds=(0.13, 0.535, 0.70, 0.58),
                    states={
                        **before_states,
                        "ime_preedit_text": "yanshou",
                        "ime_exact_candidate_text": "验收",
                    },
                ),
            ),
            stable=True,
            confidence=1.0,
            fingerprint="after",
            camera_alignment=aligned_camera_facts(),
        )
        robot = FakeRobot()
        action = SemanticAction(
            node_id="input-next-chinese-segment",
            action="input_verified_text",
            params={
                "element_id": "field",
                "target": "application_text_input",
                "role": "input",
                "label": "复杂输入",
                "states": before_states,
                "text": "复杂输入验收",
                "expected_effect": {
                    "element_state": {
                        "meaning": "application_text_input",
                        "states": {
                            "value": "复杂输入",
                            "ime_preedit_text": "yanshou",
                            "ime_exact_candidate_text": "验收",
                        },
                    }
                },
            },
        )

        result = self._adapter(
            FakeSceneObserver([before, after, after]),
            robot,
        ).execute(
            requested_action=action,
            planned_scene=before,
            goal=goal(),
            confirmed=True,
        )

        self.assertEqual([("pinyin", "验收", "yanshou")], robot.actions)
        self.assertEqual(
            "matched",
            result.action_outcome,
            result.verification_errors,
        )
        self.assertEqual(1, result.physical_actions)

    def test_input_surface_family_does_not_bridge_different_page_families(self):
        self.assertTrue(
            UniversalActionController._input_screen_identity_is_compatible(
                "聊天界面",
                "chat_input",
            )
        )
        self.assertFalse(
            UniversalActionController._input_screen_identity_is_compatible(
                "聊天界面",
                "search_input",
            )
        )
        self.assertEqual(
            "",
            UniversalActionController._input_screen_identity_family("unknown"),
        )

    def test_typed_field_identity_bridges_only_model_screen_wording_drift(self):
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
            input_method="direct_latin",
            prior_input_value="first\n",
            expected_input_value="first\nsecond",
            target_element_id="before-field",
            before_fingerprint=before.fingerprint,
            expected_effect={
                "element_state": {
                    "meaning": "application_text_input",
                    "states": {"value": "first\nsecond"},
                }
            },
            formal_candidate_id="candidate-append-second",
            formal_transition={
                "transition_id": "transition-append-second",
                "precondition_claim_ids": ["claim-append-second"],
                "expectations": [
                    {
                        "subject_ref": "element.input_field_1",
                        "predicate": "element.state.value",
                        "operator": "equals",
                        "value": "first\nsecond",
                    }
                ],
                "exploratory": False,
            },
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
        concrete_before = replace(before, app_id="com.example.source")
        different_package = replace(after, app_id="com.example.other")
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
        for unsafe_before, unsafe_after in (
            (before, different_field),
            (concrete_before, different_package),
            (before, changed_mode),
        ):
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

    def test_post_action_transient_mismatch_is_not_resampled(self):
        states = {
            "focused": True,
            "value": "",
            "keyboard_layout": "qwerty",
            "keyboard_input_mode": "chinese_pinyin",
            "keyboard_geometry": TEST_QWERTY_GEOMETRY,
            "goal_relevant": True,
        }
        before = UIScene(
            app_id="chat",
            screen_id="conversation",
            summary="消息输入框和中文键盘可见",
            elements=(
                UIElement(
                    element_id="field",
                    role="input",
                    meaning="application_text_input",
                    label="消息",
                    bounds=(0.1, 0.1, 0.9, 0.2),
                    confidence=0.97,
                    states=states,
                ),
            ),
            stable=True,
            confidence=0.96,
            fingerprint="before",
        )
        transient = replace(
            before,
            summary="首轮暂态观察遗漏输入框",
            elements=(),
            fingerprint="transient",
        )
        after = replace(
            before,
            summary="输入框与逐字中文候选均已验证",
            elements=(
                replace(
                    before.elements[0],
                    states={
                        **states,
                        "ime_preedit_text": "fuzashuru",
                        "ime_exact_candidate_text": "复杂输入",
                    },
                ),
            ),
            fingerprint="after",
        )
        robot = FakeRobot()
        action = SemanticAction(
            node_id="input-chinese-segment",
            action="input_verified_text",
            params={
                "element_id": "field",
                "target": "application_text_input",
                "role": "input",
                "label": "消息",
                "states": states,
                "text": "复杂输入",
                "expected_effect": {
                    "element_state": {
                        "meaning": "application_text_input",
                        "states": {
                            "value": "",
                            "ime_preedit_text": "fuzashuru",
                            "ime_exact_candidate_text": "复杂输入",
                        },
                    }
                },
            },
        )
        observer = FakeSceneObserver([before, transient, after])

        result = self._adapter(observer, robot).execute(
            requested_action=action,
            planned_scene=before,
            goal=goal(),
            confirmed=True,
        )

        self.assertEqual([("pinyin", "复杂输入", "fuzashuru")], robot.actions)
        self.assertEqual(2, observer.calls)
        self.assertEqual(1, result.physical_actions)
        self.assertEqual("mismatched", result.action_outcome)
        self.assertTrue(result.verification_errors)

    def test_confirmed_clear_uses_exact_observed_count_and_fresh_qwerty_geometry(self):
        def input_scene(fingerprint, element_id, value):
            return UIScene(
                app_id="browser",
                screen_id="draft",
                summary="唯一已聚焦输入框",
                elements=(
                    UIElement(
                        element_id=element_id,
                        role="input",
                        meaning="draft_input",
                        label="",
                        bounds=(0.1, 0.1, 0.9, 0.2),
                        confidence=0.97,
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
                confidence=0.97,
                fingerprint=fingerprint,
            )

        planned = input_scene("planned", "planned-field", "lxs,")
        fresh = input_scene("before", "fresh-field", "lxs,")
        after = input_scene("after", "after-field", "")
        robot = FakeRobot()
        snapped_anchors = {
            key: list(value)
            for key, value in TEST_QWERTY_GEOMETRY["anchors"].items()
        }

        def snap_before_orientation_arm(_frames, _anchors):
            self.assertIsNone(robot._armed)
            return snapped_anchors

        result = self._adapter(
            FakeSceneObserver([fresh, after]),
            robot,
            qwerty_row_snapper=snap_before_orientation_arm,
            require_local_qwerty_row_snap=True,
        ).execute(
            requested_action=SemanticAction(
                node_id="clear-wrong-draft",
                action="clear_verified_text",
                params={
                    "element_id": "planned-field",
                    "target": "draft_input",
                    "role": "input",
                    "label": "",
                    "states": dict(planned.elements[0].states),
                    "expected_effect": {
                        "element_state": {
                            "meaning": "draft_input",
                            "states": {"value": ""},
                        }
                    },
                },
            ),
            planned_scene=planned,
            goal=goal(),
            confirmed=True,
        )

        self.assertEqual([("clear", 4)], robot.actions)
        self.assertEqual("stable_local_ocr", robot.keyboard_layouts[0]["row_snap_source"])
        self.assertEqual(1, result.physical_actions)
        self.assertEqual("matched", result.action_outcome)

    def test_clear_recovery_keeps_typed_field_across_preedit_read_drift(self):
        planned_states = {
            "goal_relevant": True,
            "fully_visible": True,
            "focused": True,
            "value": "longinp",
            "input_field_id": "input_field_1",
            "input_multiline": False,
            "keyboard_layout": "qwerty",
            "keyboard_input_mode": "chinese_pinyin",
            "keyboard_case_mode": "lower",
            "keyboard_geometry": TEST_QWERTY_GEOMETRY,
        }
        planned = UIScene(
            app_id="chat",
            screen_id="conversation",
            summary="输入框中显示带下划线的 longinp",
            elements=(UIElement(
                element_id="planned-input",
                role="input",
                meaning="application_text_input",
                label="longinp",
                bounds=(0.13, 0.54, 0.70, 0.59),
                confidence=1.0,
                states=planned_states,
            ),),
            fingerprint="planned",
        )
        fresh_states = {
            **planned_states,
            "goal_relevant": False,
            "value": "",
            "keyboard_geometry": {
                **TEST_QWERTY_GEOMETRY,
                "anchors": {
                    **TEST_QWERTY_GEOMETRY["anchors"],
                    "backspace": [879, 853],
                },
            },
        }
        fresh_input = replace(
            planned.elements[0],
            element_id="fresh-input",
            label="",
            bounds=(0.13, 0.54, 0.70, 0.60),
            states=fresh_states,
        )
        fresh = replace(
            planned,
            elements=(fresh_input,),
            fingerprint="fresh",
        )
        requested = SemanticAction(
            node_id="clear-conflicting-preedit",
            action="clear_verified_text",
            params={
                "element_id": "planned-input",
                "target": "application_text_input",
                "role": "input",
                "label": "longinp",
                "states": planned_states,
                "formal_candidate_id": "candidate-clear-longinp",
                "expected_effect": {
                    "element_state": {
                        "meaning": "application_text_input",
                        "states": {"value": ""},
                    }
                },
            },
        )

        recovered = GenericSingleActionAdapter._recover_conflicting_clear_input_scene(
            requested,
            planned,
            fresh,
        )

        self.assertIsNotNone(recovered)
        recovered_input = recovered.get_element("planned-input")
        self.assertEqual("longinp", recovered_input.states["value"])
        self.assertEqual(
            [879, 853],
            recovered_input.states["keyboard_geometry"]["anchors"]["backspace"],
        )
        self.assertEqual((0.13, 0.54, 0.70, 0.60), recovered_input.bounds)

        for name, conflicting in (
            (
                "different-field",
                replace(
                    fresh_input,
                    states={**fresh_states, "input_field_id": "input_field_2"},
                ),
            ),
            (
                "different-visible-value",
                replace(fresh_input, states={**fresh_states, "value": "useful"}),
            ),
        ):
            with self.subTest(name=name):
                rejected = replace(fresh, elements=(conflicting,))
                self.assertIsNone(
                    GenericSingleActionAdapter._recover_conflicting_clear_input_scene(
                        requested,
                        planned,
                        rejected,
                    )
                )

        ambiguous = replace(
            fresh,
            elements=(fresh_input, replace(fresh_input, element_id="other-input")),
        )
        self.assertIsNone(
            GenericSingleActionAdapter._recover_conflicting_clear_input_scene(
                requested,
                planned,
                ambiguous,
            )
        )

    def test_matched_direct_input_persists_lineage_and_clear_discards_it(self):
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

        with tempfile.TemporaryDirectory() as temp:
            store = TypedInputLineageStore(Path(temp))
            before = input_scene("before", "")
            after = input_scene("after", "longinput")
            robot = FakeRobot()
            result = self._adapter(
                FakeSceneObserver([before, after]),
                robot,
                input_lineage_store=store,
            ).execute(
                requested_action=SemanticAction(
                    node_id="type-segment",
                    action="input_verified_text",
                    params={
                        "element_id": "field",
                        "target": "application_text_input",
                        "role": "input",
                        "label": "",
                        "states": dict(before.elements[0].states),
                        "text": "longinput",
                        "expected_effect": {
                            "element_state": {
                                "meaning": "application_text_input",
                                "states": {"value": "longinput"},
                            }
                        },
                    },
                ),
                planned_scene=before,
                goal=goal(),
                confirmed=True,
            )
            self.assertEqual("matched", result.action_outcome)
            self.assertEqual("longinput", store.load("test-device").exact_value)

            clear_after = input_scene("cleared", "", goal_relevant=False)
            clear_result = self._adapter(
                FakeSceneObserver([after, clear_after]),
                FakeRobot(),
                input_lineage_store=store,
            ).execute(
                requested_action=SemanticAction(
                    node_id="clear-segment",
                    action="clear_verified_text",
                    params={
                        "element_id": "field",
                        "target": "application_text_input",
                        "role": "input",
                        "label": "longinput",
                        "states": dict(after.elements[0].states),
                        "expected_effect": {
                            "element_state": {
                                "meaning": "application_text_input",
                                "states": {"value": ""},
                            }
                        },
                    },
                ),
                planned_scene=after,
                goal=goal(),
                confirmed=True,
            )
            self.assertEqual("matched", clear_result.action_outcome)
            self.assertIsNone(store.load("test-device"))

            bad_store = TypedInputLineageStore(Path(temp) / "bad")
            wrong_after = input_scene("wrong-after", "longinpuw")
            mismatch = self._adapter(
                FakeSceneObserver([before, wrong_after, wrong_after]),
                FakeRobot(),
                input_lineage_store=bad_store,
            ).execute(
                requested_action=SemanticAction(
                    node_id="type-mismatch",
                    action="input_verified_text",
                    params={
                        "element_id": "field",
                        "target": "application_text_input",
                        "role": "input",
                        "label": "",
                        "states": dict(before.elements[0].states),
                        "text": "longinput",
                        "expected_effect": {
                            "element_state": {
                                "meaning": "application_text_input",
                                "states": {"value": "longinput"},
                            }
                        },
                    },
                ),
                planned_scene=before,
                goal=goal(),
                confirmed=True,
            )
            self.assertEqual("mismatched", mismatch.action_outcome)
            self.assertIsNone(bad_store.load("test-device"))

    def test_confirmed_input_accepts_unique_overlapping_post_input_alias(self):
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
            "before", "fresh-input", "target_text_input", "", ""
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
                    "expected_effect": {
                        "element_state": {
                            "meaning": "target_text_input",
                            "states": {"value": "agent"},
                        }
                    },
                },
            ),
            planned_scene=planned,
            goal=goal(),
            confirmed=True,
        )

        self.assertEqual([("input", "agent")], robot.actions)
        self.assertEqual("matched", result.action_outcome)

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
                    "expected_effect": {
                        "element_state": {
                            "meaning": "target_text_input",
                            "states": {"value": "agent"},
                        }
                    },
                },
            ),
            planned_scene=planned,
            goal=goal(),
            confirmed=True,
        )

        self.assertEqual([("input", "agent")], robot.actions)
        self.assertEqual("matched", result.action_outcome)

    def test_confirmed_input_rejects_different_concrete_page_identity(self):
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

        result = self._adapter(FakeSceneObserver([before, after, after]), robot).execute(
            requested_action=SemanticAction(
                node_id="reject-page-switch",
                action="input_verified_text",
                params={
                    "element_id": "field",
                    "target": "target_text_input",
                    "role": "input",
                    "label": "输入框",
                    "states": before.elements[0].states,
                    "text": "agent",
                    "expected_effect": {
                        "element_state": {
                            "meaning": "target_text_input",
                            "states": {"value": "agent"},
                        }
                    },
                },
            ),
            planned_scene=before,
            goal=goal(),
            confirmed=True,
        )

        self.assertEqual("mismatched", result.action_outcome)
        self.assertTrue(
            all("App 或页面身份发生变化" in error for error in result.verification_errors)
        )
        self.assertEqual(1, result.physical_actions)

    def test_confirmed_input_rejects_unknown_identity_when_keyboard_mode_changes(self):
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
            app_id="unknown",
            screen_id="unknown",
            fingerprint="after",
            elements=(
                replace(
                    before.elements[0],
                    states={
                        **before.elements[0].states,
                        "value": "agent",
                        "keyboard_input_mode": "chinese_pinyin",
                    },
                ),
            ),
        )
        robot = FakeRobot()

        result = self._adapter(FakeSceneObserver([before, after, after]), robot).execute(
            requested_action=SemanticAction(
                node_id="reject-mode-drift",
                action="input_verified_text",
                params={
                    "element_id": "field",
                    "target": "target_text_input",
                    "role": "input",
                    "label": "输入框",
                    "states": before.elements[0].states,
                    "text": "agent",
                    "expected_effect": {
                        "element_state": {
                            "meaning": "target_text_input",
                            "states": {"value": "agent"},
                        }
                    },
                },
            ),
            planned_scene=before,
            goal=goal(),
            confirmed=True,
        )

        self.assertEqual("mismatched", result.action_outcome)
        self.assertTrue(
            all("App 或页面身份发生变化" in error for error in result.verification_errors)
        )
        self.assertEqual(1, result.physical_actions)

    def test_execution_result_keeps_exact_four_verified_after_frames(self):
        gray = Image.new("RGB", (540, 960), "gray")
        white = Image.new("RGB", (540, 960), "white")
        before_fingerprint = _local_frame_fingerprint(gray)
        after_fingerprint = _local_frame_fingerprint(white)
        planned = scene(before_fingerprint)
        fresh = scene(before_fingerprint, element_id="fresh")
        after = scene(
            after_fingerprint,
            screen_id="app_home",
            element_id="after",
        )
        observer = FakeSceneObserver([fresh, after])
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
                "action_outcome": "matched",
                "execution": {
                    "orientation_credential": credential.to_dict(),
                },
                "before_frame_paths": list(result.before_frame_paths),
            },
            orientation_credential=credential,
            execution_result=result,
        )

    def test_after_frame_fingerprint_matches_after_scene(self):
        gray = Image.new("RGB", (540, 960), "gray")
        white = Image.new("RGB", (540, 960), "white")
        before_fingerprint = _local_frame_fingerprint(gray)
        after_fingerprint = _local_frame_fingerprint(white)
        adapter = GenericSingleActionAdapter(
            capture=SequenceCapture(["gray"] * 4 + ["white"] * 4),
            observer=FakeSceneObserver(
                [
                    scene(before_fingerprint, element_id="fresh"),
                    scene(after_fingerprint, screen_id="app_home", element_id="after"),
                ]
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
            goal=goal(),
            confirmed=True,
        )

        selected_fingerprint = _local_frame_fingerprint(result.after_frames[0])
        self.assertEqual(result.after_scene.fingerprint, selected_fingerprint)

    def test_rebind_rejects_same_label_when_meaning_changes(self):
        planned = UIScene(
            app_id="unknown",
            screen_id="android_home",
            summary="主屏幕",
            elements=(
                UIElement(
                    element_id="douyin_app_icon",
                    role="icon",
                    meaning="抖音应用启动入口",
                    label="抖音",
                    bounds=(0.69, 0.63, 0.84, 0.76),
                    confidence=0.98,
                    states={"goal_relevant": True},
                ),
            ),
            stable=True,
            confidence=0.97,
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
                    meaning="抖音应用图标",
                    label="抖音",
                    bounds=(0.70, 0.64, 0.85, 0.77),
                    confidence=0.97,
                    states={"goal_relevant": True},
                ),
            ),
            stable=True,
            confidence=0.96,
            fingerprint="planned",
        )
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
            params={
                "element_id": "douyin_app_icon",
                "target": "抖音应用启动入口",
                "role": "icon",
                "label": "抖音",
                "states": {"goal_relevant": True},
            },
        )
        with self.assertRaisesRegex(GenericActionAdapterError, "语义"):
            self._adapter(observer, robot).execute(
                requested_action=action,
                planned_scene=planned,
                goal=goal(),
                confirmed=True,
            )
        self.assertEqual(robot.actions, [])

    def test_formal_rebind_accepts_same_navigation_class_after_model_wording_drift(self):
        planned = UIScene(
            app_id="unknown",
            screen_id="unknown",
            summary="主屏幕",
            elements=(
                UIElement(
                    element_id="browser_app_icon",
                    role="icon",
                    meaning="启动浏览器应用",
                    label="浏览器",
                    bounds=(0.12, 0.03, 0.33, 0.17),
                    confidence=0.98,
                    states={"goal_relevant": True},
                ),
            ),
            stable=True,
            confidence=0.97,
            fingerprint="planned",
        )
        fresh = UIScene(
            app_id="launcher",
            screen_id="home_screen",
            summary="主屏幕",
            elements=(
                UIElement(
                    element_id="browser_icon",
                    role="icon",
                    meaning="launch_browser_app",
                    label="浏览器",
                    bounds=(0.13, 0.03, 0.32, 0.16),
                    confidence=0.98,
                    states={"goal_relevant": True},
                ),
            ),
            stable=True,
            confidence=0.96,
            fingerprint="fresh",
        )
        after = UIScene(
            app_id="browser",
            screen_id="browser_home",
            summary="浏览器首页",
            stable=True,
            confidence=0.95,
            fingerprint="after",
        )
        robot = FakeRobot()
        rebound = self._adapter(
            FakeSceneObserver([]),
            robot,
        )._rebind_action(
            SemanticAction(
                node_id="open-browser",
                action="tap_semantic",
                params={
                    "element_id": "browser_app_icon",
                    "target": "启动浏览器应用",
                    "role": "icon",
                    "label": "浏览器",
                    "states": {"goal_relevant": True},
                    "formal_candidate_id": "candidate-browser",
                },
            ),
            planned,
            fresh,
        )

        self.assertEqual([], robot.actions)
        self.assertEqual("launch_browser_app", rebound.params["target"])

    def test_rebind_accepts_exact_local_gesture_mode_selector_label(self):
        states = {"goal_relevant": True, "fully_visible": True}
        planned = UIScene(
            app_id="unknown",
            screen_id="unknown",
            summary="本地验收模式列表",
            elements=(
                UIElement(
                    element_id="mode",
                    role="button",
                    meaning="long_press_target",
                    label="长按目标",
                    bounds=(0.12, 0.76, 0.88, 0.85),
                    confidence=1.0,
                    states=states,
                ),
            ),
            fingerprint="planned",
        )
        fresh = UIScene(
            app_id="unknown",
            screen_id="unknown",
            summary="本地验收模式列表",
            elements=(
                UIElement(
                    element_id="fresh-mode",
                    role="button",
                    meaning="select_long_press_target_mode",
                    label="长按目标",
                    bounds=(0.12, 0.77, 0.88, 0.86),
                    confidence=1.0,
                    states=states,
                ),
            ),
            fingerprint="fresh",
        )
        robot = FakeRobot()
        result = self._adapter(
            FakeSceneObserver([fresh, scene("after", screen_id="long-press")]),
            robot,
        ).execute(
            requested_action=SemanticAction(
                node_id="open-mode",
                action="tap_semantic",
                params={
                    "element_id": "mode",
                    "target": "long_press_target",
                    "role": "button",
                    "label": "长按目标",
                    "states": states,
                },
            ),
            planned_scene=planned,
            goal=goal(),
            confirmed=True,
        )

        self.assertEqual([("tap", 500, 815)], robot.actions)
        self.assertEqual(
            "select_long_press_target_mode",
            result.rebound_action.params["target"],
        )

    def test_rebind_uses_planned_selector_semantics_when_fresh_wording_is_minimal(self):
        states = {"goal_relevant": True, "fully_visible": True}
        planned = UIScene(
            app_id="unknown",
            screen_id="acceptance_modes",
            summary="本地模式列表",
            elements=(
                UIElement(
                    element_id="planned-mode",
                    role="list_item",
                    meaning="acceptance_mode_option",
                    label="语义点击",
                    bounds=(0.12, 0.35, 0.88, 0.43),
                    confidence=1.0,
                    states=states,
                ),
            ),
            fingerprint="planned",
        )
        fresh = replace(
            planned,
            elements=(
                replace(
                    planned.elements[0],
                    element_id="fresh-mode",
                    meaning="语义动作控件",
                ),
            ),
            fingerprint="fresh",
        )
        robot = FakeRobot()

        result = self._adapter(
            FakeSceneObserver([fresh, scene("after", screen_id="semantic-mode")]),
            robot,
        ).execute(
            requested_action=SemanticAction(
                node_id="open-mode",
                action="tap_semantic",
                params={
                    "element_id": "planned-mode",
                    "target": "acceptance_mode_option",
                    "role": "list_item",
                    "label": "语义点击",
                    "states": states,
                },
            ),
            planned_scene=planned,
            goal=goal(),
            confirmed=True,
        )

        self.assertEqual(1, result.physical_actions)
        self.assertEqual([("tap", 500, 390)], robot.actions)
        self.assertEqual("语义动作控件", result.rebound_action.params["target"])

    def test_rebind_uses_chinese_planned_selector_semantics(self):
        states = {"goal_relevant": True, "fully_visible": True}
        planned = UIScene(
            app_id="unknown",
            screen_id="acceptance_modes",
            summary="本地模式列表",
            elements=(
                UIElement(
                    element_id="planned-mode",
                    role="list_item",
                    meaning="验收模式选项",
                    label="语义点击",
                    bounds=(0.12, 0.35, 0.88, 0.43),
                    confidence=1.0,
                    states=states,
                ),
            ),
            fingerprint="planned",
        )
        fresh = replace(
            planned,
            elements=(
                replace(
                    planned.elements[0],
                    element_id="fresh-mode",
                    meaning="语义动作控件",
                ),
            ),
            fingerprint="fresh",
        )
        robot = FakeRobot()

        result = self._adapter(
            FakeSceneObserver([fresh, scene("after", screen_id="semantic-mode")]),
            robot,
        ).execute(
            requested_action=SemanticAction(
                node_id="open-mode",
                action="tap_semantic",
                params={
                    "element_id": "planned-mode",
                    "target": "验收模式选项",
                    "role": "list_item",
                    "label": "语义点击",
                    "states": states,
                },
            ),
            planned_scene=planned,
            goal=goal(),
            confirmed=True,
        )

        self.assertEqual(1, result.physical_actions)
        self.assertEqual([("tap", 500, 390)], robot.actions)
        self.assertEqual("语义动作控件", result.rebound_action.params["target"])

    def test_rebind_accepts_exact_navigation_label_across_selector_roles(self):
        states = {"goal_relevant": True, "fully_visible": True}
        label = "连续闭环：滑动→点击→返回"
        planned = UIScene(
            app_id="unknown",
            screen_id="acceptance_modes",
            summary="本地验收模式列表",
            elements=(
                UIElement(
                    element_id="mode",
                    role="list_item",
                    meaning="continuous_loop_acceptance_mode_entry",
                    label=label,
                    bounds=(0.13, 0.79, 0.87, 0.86),
                    confidence=1.0,
                    states=states,
                ),
            ),
            fingerprint="planned",
        )
        fresh = replace(
            planned,
            elements=(
                replace(
                    planned.elements[0],
                    element_id="fresh-mode",
                    role="button",
                    meaning="target_entry",
                ),
            ),
            fingerprint="fresh",
        )
        robot = FakeRobot()

        result = self._adapter(
            FakeSceneObserver([fresh, scene("after", screen_id="sequence")]),
            robot,
        ).execute(
            requested_action=SemanticAction(
                node_id="open-mode",
                action="tap_semantic",
                params={
                    "element_id": "mode",
                    "target": "continuous_loop_acceptance_mode_entry",
                    "role": "list_item",
                    "label": label,
                    "states": states,
                },
            ),
            planned_scene=planned,
            goal=goal(),
            confirmed=True,
        )

        self.assertEqual(1, result.physical_actions)
        self.assertEqual([("tap", 500, 825)], robot.actions)
        self.assertEqual("button", result.rebound_action.params["role"])
        self.assertEqual("target_entry", result.rebound_action.params["target"])

    def test_rebind_accepts_exact_text_link_after_button_role_drift(self):
        states = {"fully_visible": True}
        label = "返回验收模式选择"
        planned = UIScene(
            app_id="unknown",
            screen_id="unknown",
            summary="本地验收详情页",
            elements=(
                UIElement(
                    element_id="planned-link",
                    role="button",
                    meaning="back_navigation",
                    label=label,
                    bounds=(0.118, 0.095, 0.415, 0.125),
                    confidence=1.0,
                    states=states,
                ),
            ),
            fingerprint="planned",
        )
        fresh = replace(
            planned,
            elements=(
                replace(
                    planned.elements[0],
                    element_id="fresh-link",
                    role="text",
                    meaning="navigation_link",
                    bounds=(0.13, 0.095, 0.43, 0.125),
                ),
            ),
            fingerprint="fresh",
        )
        for action in ("tap_semantic", "dismiss_overlay"):
            with self.subTest(action=action):
                adapter = self._adapter(FakeSceneObserver([fresh]), FakeRobot())
                rebound = adapter._rebind_action(
                    SemanticAction(
                        node_id="return-to-list",
                        action=action,
                        params={
                            "element_id": "planned-link",
                            "target": "back_navigation",
                            "role": "button",
                            "label": label,
                            "states": states,
                        },
                    ),
                    planned,
                    fresh,
                )

                self.assertEqual("fresh-link", rebound.params["element_id"])
                self.assertEqual("text", rebound.params["role"])
                self.assertEqual("navigation_link", rebound.params["target"])

    def test_rebind_accepts_unique_unlabeled_icon_button_role_drift(self):
        states = {"goal_relevant": True, "fully_visible": True}
        planned = UIScene(
            app_id="browser",
            screen_id="page",
            summary="浏览器页面",
            elements=(
                UIElement(
                    element_id="planned-home",
                    role="icon",
                    meaning="home",
                    label="",
                    bounds=(0.83, 0.91, 0.93, 0.97),
                    confidence=1.0,
                    states=states,
                ),
            ),
            fingerprint="planned",
        )
        fresh = replace(
            planned,
            elements=(
                replace(
                    planned.elements[0],
                    element_id="fresh-home",
                    role="button",
                ),
            ),
            fingerprint="fresh",
        )

        rebound = self._adapter(
            FakeSceneObserver([fresh]), FakeRobot()
        )._rebind_action(
            SemanticAction(
                node_id="open-home",
                action="tap_semantic",
                params={
                    "element_id": "planned-home",
                    "target": "home",
                    "role": "icon",
                    "label": "",
                    "states": states,
                    "formal_candidate_id": "candidate-home",
                },
            ),
            planned,
            fresh,
        )

        self.assertEqual("fresh-home", rebound.params["element_id"])
        self.assertEqual("button", rebound.params["role"])

    def test_rebind_rejects_ambiguous_unlabeled_icon_button_role_drift(self):
        states = {"goal_relevant": True, "fully_visible": True}
        planned = UIScene(
            app_id="browser",
            screen_id="page",
            summary="浏览器页面",
            elements=(
                UIElement(
                    element_id="planned-icon",
                    role="icon",
                    meaning="navigation",
                    label="",
                    bounds=(0.1, 0.9, 0.2, 0.97),
                    confidence=1.0,
                    states=states,
                ),
            ),
            fingerprint="planned",
        )
        fresh = replace(
            planned,
            elements=(
                replace(planned.elements[0], element_id="fresh-a", role="button"),
                replace(
                    planned.elements[0],
                    element_id="fresh-b",
                    role="button",
                    bounds=(0.3, 0.9, 0.4, 0.97),
                ),
            ),
            fingerprint="fresh",
        )

        with self.assertRaisesRegex(
            GenericActionAdapterError,
            "目标语义不再严格唯一",
        ):
            self._adapter(
                FakeSceneObserver([fresh]), FakeRobot()
            )._rebind_action(
                SemanticAction(
                    node_id="ambiguous-navigation",
                    action="tap_semantic",
                    params={
                        "element_id": "planned-icon",
                        "target": "navigation",
                        "role": "icon",
                        "label": "",
                        "states": states,
                        "formal_candidate_id": "candidate-navigation",
                    },
                ),
                planned,
                fresh,
            )

    def test_rebind_accepts_fresh_positive_fully_visible_attestation(self):
        planned = scene("planned")
        fresh_element = replace(
            planned.elements[0],
            element_id="fresh-target",
            states={**planned.elements[0].states, "fully_visible": True},
        )
        fresh = replace(
            planned,
            elements=(fresh_element,),
            fingerprint="fresh",
        )
        after = scene("after", screen_id="next-screen", element_id="after")
        robot = FakeRobot()

        result = self._adapter(
            FakeSceneObserver([fresh, after]),
            robot,
        ).execute(
            requested_action=SemanticAction(
                node_id="open-target",
                action="tap_semantic",
                params={
                    "element_id": planned.elements[0].element_id,
                    "target": planned.elements[0].meaning,
                    "role": planned.elements[0].role,
                    "label": planned.elements[0].label,
                    "states": dict(planned.elements[0].states),
                },
            ),
            planned_scene=planned,
            goal=goal(),
            confirmed=True,
        )

        self.assertEqual(1, result.physical_actions)
        self.assertEqual([("tap", 300, 400)], robot.actions)
        self.assertTrue(result.rebound_action.params["states"]["fully_visible"])

    def test_rebind_ignores_goal_relevant_context_drift(self):
        planned = UIScene(
            app_id="unknown",
            screen_id="acceptance_modes",
            summary="本地验收模式列表",
            elements=(
                UIElement(
                    element_id="planned-second-item",
                    role="list_item",
                    meaning="验收模式选项",
                    label="语义点击",
                    bounds=(0.13, 0.39, 0.87, 0.47),
                    confidence=1.0,
                    states={"goal_relevant": True, "fully_visible": True},
                ),
            ),
            fingerprint="planned",
        )
        fresh = replace(
            planned,
            elements=(
                replace(
                    planned.elements[0],
                    element_id="fresh-second-item",
                    states={"goal_relevant": False, "fully_visible": True},
                ),
            ),
            fingerprint="fresh",
        )
        robot = FakeRobot()

        result = self._adapter(
            FakeSceneObserver([fresh, scene("after", screen_id="tap-mode")]),
            robot,
        ).execute(
            requested_action=SemanticAction(
                node_id="open-second-item",
                action="tap_semantic",
                params={
                    "element_id": "planned-second-item",
                    "target": "验收模式选项",
                    "role": "list_item",
                    "label": "语义点击",
                    "states": {"goal_relevant": True, "fully_visible": True},
                },
            ),
            planned_scene=planned,
            goal=goal(),
            confirmed=True,
        )

        self.assertEqual(1, result.physical_actions)
        self.assertEqual([("tap", 500, 430)], robot.actions)
        self.assertFalse(result.rebound_action.params["states"]["goal_relevant"])

    def test_tap_reuses_single_step_geometry_when_pixels_match(self):
        planned = UIScene(
            app_id="unknown",
            screen_id="acceptance_modes",
            summary="本地验收模式列表",
            elements=(
                UIElement(
                    element_id="planned-second-item",
                    role="list_item",
                    meaning="验收模式选项",
                    label="语义点击",
                    bounds=(0.13, 0.39, 0.87, 0.47),
                    confidence=1.0,
                    states={"goal_relevant": True, "fully_visible": True},
                    evidence=("第二项语义点击完整可见",),
                ),
            ),
            fingerprint="planned",
            camera_alignment=aligned_camera_facts(),
        )
        fresh = replace(
            planned,
            elements=(
                replace(
                    planned.elements[0],
                    element_id="fresh-second-item",
                    bounds=(0.13, 0.50, 0.87, 0.58),
                    states={"goal_relevant": False, "fully_visible": True},
                ),
            ),
            fingerprint="fresh",
        )
        planned_audited = replace(
            planned,
            elements=(replace(planned.elements[0], bounds=(0.15, 0.40, 0.85, 0.47)),),
        )
        fresh_audited = replace(
            fresh,
            elements=(replace(fresh.elements[0], bounds=(0.15, 0.40, 0.85, 0.47)),),
        )
        observer = FakeSceneObserver([scene("after", screen_id="tap-mode")])
        robot = FakeRobot()

        result = self._adapter(observer, robot).execute(
            requested_action=SemanticAction(
                node_id="open-second-item",
                action="tap_semantic",
                params={
                    "element_id": "planned-second-item",
                    "target": "验收模式选项",
                    "role": "list_item",
                    "label": "语义点击",
                    "states": {"goal_relevant": True, "fully_visible": True},
                },
            ),
            planned_scene=planned,
            goal=goal(),
            confirmed=True,
            planned_frames=tuple(
                Image.new("RGB", (540, 960), "gray") for _ in range(4)
            ),
        )

        self.assertEqual(1, result.physical_actions)
        self.assertEqual([("tap", 500, 430)], robot.actions)
        self.assertEqual([], observer.geometry_audit_calls)

    def test_tap_reuses_private_local_geometry_attestation_without_third_audit(self):
        states = {
            "goal_relevant": True,
            "fully_visible": True,
            "reload_visual_audit": True,
            "independent_geometry_verified": True,
            "geometry_audit_source": "icon_cluster_localization",
        }

        def reload_scene(fingerprint, *, screen_id="page"):
            return UIScene(
                app_id="browser",
                screen_id=screen_id,
                summary="本地页面",
                elements=(
                    UIElement(
                        element_id="local_audited_reload_control_1",
                        role="icon",
                        meaning="reload",
                        label="",
                        bounds=(0.80, 0.10, 0.86, 0.16),
                        confidence=0.97,
                        states=states,
                        evidence=("独立图标簇定位与本地几何校验",),
                    ),
                ),
                stable=True,
                confidence=0.98,
                fingerprint=fingerprint,
                camera_alignment=aligned_camera_facts(),
            )

        planned = reload_scene("planned")
        fresh = reload_scene("fresh")
        after = reload_scene("after", screen_id="reloaded")
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
                node_id="reload-locally-attested",
                action="tap_semantic",
                params={
                    "element_id": "local_audited_reload_control_1",
                    "target": "reload",
                    "role": "icon",
                    "label": "",
                    "states": states,
                    "expected_effect": {
                        "scene_changed": True,
                        "goal_complete_on_success": True,
                    },
                },
            ),
            planned_scene=planned,
            planned_frames=tuple(
                Image.new("RGB", (540, 960), "gray") for _ in range(4)
            ),
            goal=goal(),
            confirmed=True,
        )

        self.assertEqual([], observer.geometry_audit_calls)
        self.assertEqual([("tap", 830, 130)], robot.actions)
        self.assertEqual(1, result.physical_actions)
        self.assertEqual(
            (
                "控制器确认动作前后场景指纹发生变化：planned -> after",
            ),
            result.controller_completion_evidence,
        )

    def test_rebind_accepts_tight_loose_audit_boxes_for_same_static_target(self):
        planned = scene(
            "planned",
            bounds=(0.11, 0.235, 0.43, 0.27),
        )
        fresh = replace(planned, fingerprint="fresh")
        planned_audited = replace(
            planned,
            elements=(
                replace(
                    planned.elements[0],
                    bounds=(0.11, 0.235, 0.43, 0.27),
                ),
            ),
        )
        fresh_audited = replace(
            fresh,
            elements=(
                replace(
                    fresh.elements[0],
                    bounds=(0.0828, 0.225, 0.3372, 0.255),
                ),
            ),
        )
        after = scene("after", screen_id="acceptance_modes", element_id="after")
        observer = FakeSceneObserver([after])
        robot = FakeRobot()

        result = self._adapter(observer, robot).execute(
            requested_action=SemanticAction(
                node_id="return-to-list",
                action="tap_semantic",
                params={
                    "element_id": "e1",
                    "target": "app_icon",
                    "role": planned.elements[0].role,
                    "label": planned.elements[0].label,
                    "states": dict(planned.elements[0].states),
                },
            ),
            planned_scene=planned,
            goal=goal(),
            confirmed=True,
            planned_frames=tuple(
                Image.new("RGB", (540, 960), "gray") for _ in range(4)
            ),
        )

        self.assertEqual(1, result.physical_actions)
        self.assertEqual([("tap", 270, 252)], robot.actions)

    def test_rebind_accepts_narrow_strict_local_input_audit_jitter(self):
        states = {
            "goal_relevant": True,
            "fully_visible": True,
            "ime_candidate": True,
            "input_element_id": "local_audited_input_1",
            "prior_input_value": "",
            "expected_input_value": "你好",
            "pinyin": "nihao",
            "independent_geometry_verified": True,
            "geometry_audit_source": "element_geometry_audit",
        }

        def candidate_scene(fingerprint, bounds):
            return UIScene(
                app_id="chat",
                screen_id="conversation",
                summary="唯一逐字输入法候选可见",
                elements=(
                    UIElement(
                        element_id="local_audited_ime_candidate_1",
                        role="button",
                        meaning="ime_exact_candidate",
                        label="你好",
                        bounds=bounds,
                        confidence=1.0,
                        states=states,
                        evidence=("输入结构审计确认唯一逐字候选你好",),
                    ),
                ),
                stable=True,
                confidence=0.98,
                fingerprint=fingerprint,
            )

        planned = candidate_scene("planned", (0.10, 0.61, 0.24, 0.635))
        fresh = candidate_scene("fresh", (0.09, 0.626, 0.192, 0.654))
        requested = SemanticAction(
            node_id="select-exact-candidate",
            action="tap_semantic",
            params={
                "element_id": planned.elements[0].element_id,
                "target": planned.elements[0].meaning,
                "role": planned.elements[0].role,
                "label": planned.elements[0].label,
                "states": dict(planned.elements[0].states),
                "formal_candidate_id": "candidate-exact-nihao",
            },
        )
        adapter = self._adapter(FakeSceneObserver([]), FakeRobot())

        rebound = adapter._rebind_action(
            requested,
            planned,
            fresh,
            local_frame_identity_verified=True,
        )

        self.assertEqual(
            "local_audited_ime_candidate_1",
            rebound.params["element_id"],
        )
        self.assertEqual(fresh.elements[0].states, rebound.params["states"])

        with self.assertRaisesRegex(
            GenericActionAdapterError,
            "目标区域已明显移动",
        ):
            adapter._rebind_action(
                requested,
                planned,
                fresh,
                local_frame_identity_verified=False,
            )

    def test_rebind_accepts_dual_audited_literal_key_frame_jitter(self):
        states = {
            "goal_relevant": True,
            "fully_visible": True,
            "input_literal_key": True,
            "key_value": "1",
            "input_element_id": "local_audited_input_1",
            "prior_input_value": "复杂输入验收2026:",
            "expected_input_value": "复杂输入验收2026:1",
            "independent_geometry_verified": True,
            "geometry_audit_source": "element_geometry_audit",
        }

        def literal_key_scene(fingerprint, bounds):
            return UIScene(
                app_id="editor",
                screen_id="input",
                summary="唯一下一字符键位可见",
                elements=(
                    UIElement(
                        element_id="local_audited_literal_key_1",
                        role="button",
                        meaning="input_exact_literal_key",
                        label="1",
                        bounds=bounds,
                        confidence=1.0,
                        states=states,
                        evidence=("输入结构审计确认下一字符对应唯一完整可见键位",),
                    ),
                ),
                stable=True,
                confidence=0.98,
                fingerprint=fingerprint,
            )

        planned = literal_key_scene("planned", (0.40, 0.68, 0.60, 0.74))
        # The same low-profile key can move by 8% of full-frame width between
        # two independent model crops while retaining a substantial consensus.
        fresh = literal_key_scene("fresh", (0.32, 0.664, 0.52, 0.724))
        requested = SemanticAction(
            node_id="append-next-literal",
            action="tap_semantic",
            params={
                "element_id": planned.elements[0].element_id,
                "target": planned.elements[0].meaning,
                "role": planned.elements[0].role,
                "label": planned.elements[0].label,
                "states": dict(planned.elements[0].states),
            },
        )
        adapter = self._adapter(FakeSceneObserver([]), FakeRobot())

        consensus_scene = adapter._apply_local_input_geometry_consensus(
            requested,
            planned,
            fresh,
            local_frame_identity_verified=True,
        )
        self.assertEqual(
            (0.40, 0.68, 0.52, 0.724),
            consensus_scene.elements[0].bounds,
        )
        rebound = adapter._rebind_action(
            requested,
            planned,
            consensus_scene,
            local_frame_identity_verified=True,
        )

        self.assertEqual(fresh.elements[0].element_id, rebound.params["element_id"])
        self.assertEqual(fresh.elements[0].states, rebound.params["states"])

        for moved_bounds in (
            (0.52, 0.696, 0.68, 0.756),
            (0.60, 0.68, 0.80, 0.74),
            (0.40, 0.75, 0.60, 0.81),
        ):
            with self.subTest(moved_bounds=moved_bounds):
                with self.assertRaisesRegex(
                    GenericActionAdapterError,
                    "目标区域已明显移动",
                ):
                    adapter._rebind_action(
                        requested,
                        planned,
                        literal_key_scene("moved", moved_bounds),
                        local_frame_identity_verified=True,
                    )

    def test_rebind_keeps_global_geometry_gate_for_nonlocal_audited_control(self):
        attested_states = {
            "independent_geometry_verified": True,
            "geometry_audit_source": "element_geometry_audit",
        }
        planned = replace(
            scene("planned", bounds=(0.10, 0.61, 0.24, 0.635)),
            elements=(
                replace(
                    scene("planned").elements[0],
                    states=attested_states,
                    evidence=("独立几何审计",),
                    bounds=(0.10, 0.61, 0.24, 0.635),
                ),
            ),
        )
        fresh = replace(
            planned,
            fingerprint="fresh",
            elements=(
                replace(planned.elements[0], bounds=(0.09, 0.626, 0.192, 0.654)),
            ),
        )
        requested = SemanticAction(
            node_id="ordinary-button",
            action="tap_semantic",
            params={
                "element_id": planned.elements[0].element_id,
                "target": planned.elements[0].meaning,
                "role": planned.elements[0].role,
                "label": planned.elements[0].label,
                "states": dict(planned.elements[0].states),
            },
        )

        with self.assertRaisesRegex(
            GenericActionAdapterError,
            "目标区域已明显移动",
        ):
            self._adapter(FakeSceneObserver([]), FakeRobot())._rebind_action(
                requested,
                planned,
                fresh,
                local_frame_identity_verified=True,
            )

    def test_matching_pixels_do_not_accept_a_second_model_geometry_veto(self):
        planned = scene("planned", bounds=(0.08, 0.225, 0.34, 0.255))
        fresh = replace(planned, fingerprint="fresh")
        planned_audited = planned
        fresh_audited = replace(
            fresh,
            elements=(
                replace(
                    fresh.elements[0],
                    bounds=(0.083, 0.27, 0.337, 0.30),
                ),
            ),
        )
        observer = FakeSceneObserver(
            [scene("after", screen_id="acceptance_modes", element_id="after")]
        )
        robot = FakeRobot()

        result = self._adapter(observer, robot).execute(
            requested_action=SemanticAction(
                node_id="return-text-link",
                action="tap_semantic",
                params={
                    "element_id": "e1",
                    "target": "app_icon",
                    "role": planned.elements[0].role,
                    "label": planned.elements[0].label,
                    "states": dict(planned.elements[0].states),
                },
            ),
            planned_scene=planned,
            goal=goal(),
            confirmed=True,
            planned_frames=tuple(
                Image.new("RGB", (540, 960), "gray") for _ in range(4)
            ),
        )

        self.assertEqual(1, result.physical_actions)
        self.assertEqual([("tap", 210, 240)], robot.actions)

    def test_rebind_rejects_fresh_negative_fully_visible_attestation(self):
        planned = scene("planned")
        fresh = replace(
            planned,
            elements=(
                replace(
                    planned.elements[0],
                    element_id="fresh-target",
                    states={**planned.elements[0].states, "fully_visible": False},
                ),
            ),
            fingerprint="fresh",
        )
        robot = FakeRobot()

        with self.assertRaisesRegex(GenericActionAdapterError, "状态"):
            self._adapter(FakeSceneObserver([fresh]), robot).execute(
                requested_action=SemanticAction(
                    node_id="open-target",
                    action="tap_semantic",
                    params={
                        "element_id": planned.elements[0].element_id,
                        "target": planned.elements[0].meaning,
                        "role": planned.elements[0].role,
                        "label": planned.elements[0].label,
                        "states": dict(planned.elements[0].states),
                    },
                ),
                planned_scene=planned,
                goal=goal(),
                confirmed=True,
            )

        self.assertEqual([], robot.actions)

    def test_rebind_rejects_risky_gesture_label_even_for_mode_selector(self):
        states = {"goal_relevant": True, "fully_visible": True}
        planned = UIScene(
            app_id="unknown",
            screen_id="unknown",
            summary="模式列表",
            elements=(
                UIElement(
                    element_id="mode",
                    role="button",
                    meaning="long_press_target",
                    label="长按删除",
                    bounds=(0.12, 0.76, 0.88, 0.85),
                    confidence=1.0,
                    states=states,
                ),
            ),
            fingerprint="planned",
        )
        fresh = replace(
            planned,
            elements=(
                replace(
                    planned.elements[0],
                    element_id="fresh-mode",
                    meaning="select_long_press_delete_mode",
                ),
            ),
            fingerprint="fresh",
        )
        robot = FakeRobot()

        with self.assertRaisesRegex(GenericActionAdapterError, "语义"):
            self._adapter(FakeSceneObserver([fresh]), robot).execute(
                requested_action=SemanticAction(
                    node_id="open-mode",
                    action="tap_semantic",
                    params={
                        "element_id": "mode",
                        "target": "long_press_target",
                        "role": "button",
                        "label": "长按删除",
                        "states": states,
                    },
                ),
                planned_scene=planned,
                goal=goal(),
                confirmed=True,
            )
        self.assertEqual([], robot.actions)

    def test_rebind_rejects_risky_meaning_drift_even_when_label_and_region_match(self):
        planned = UIScene(
            app_id="settings",
            screen_id="edit",
            summary="编辑页",
            elements=(
                UIElement(
                    element_id="return_button",
                    role="button",
                    meaning="return",
                    label="返回",
                    bounds=(0.1, 0.1, 0.3, 0.2),
                    confidence=0.96,
                    states={"enabled": True},
                ),
            ),
            fingerprint="planned",
        )
        fresh = UIScene(
            app_id="settings",
            screen_id="edit",
            summary="编辑页",
            elements=(
                UIElement(
                    element_id="save_return_button",
                    role="button",
                    meaning="save_and_return",
                    label="返回",
                    bounds=(0.1, 0.1, 0.3, 0.2),
                    confidence=0.96,
                    states={"enabled": True},
                ),
            ),
            fingerprint="fresh",
        )
        robot = FakeRobot()

        with self.assertRaisesRegex(GenericActionAdapterError, "语义"):
            self._adapter(FakeSceneObserver([fresh]), robot).execute(
                requested_action=SemanticAction(
                    node_id="return",
                    action="tap_semantic",
                    params={
                        "element_id": "return_button",
                        "target": "return",
                        "role": "button",
                        "label": "返回",
                        "states": {"enabled": True},
                        "formal_candidate_id": "candidate-return",
                    },
                ),
                planned_scene=planned,
                goal=goal(),
                confirmed=True,
            )

        self.assertEqual([], robot.actions)

    def test_exact_typed_label_identity_survives_free_meaning_drift(self):
        states = {"goal_relevant": True, "fully_visible": True, "enabled": True}

        def target_scene(
            fingerprint,
            *,
            label,
            meaning,
            element_id,
            bounds=(0.13, 0.45, 0.87, 0.53),
        ):
            return UIScene(
                app_id="unknown",
                screen_id="acceptance_modes",
                summary="唯一目标完整可见",
                elements=(
                    UIElement(
                        element_id=element_id,
                        role="button",
                        meaning=meaning,
                        label=label,
                        bounds=bounds,
                        confidence=1.0,
                        states=states,
                    ),
                ),
                stable=True,
                confidence=1.0,
                fingerprint=fingerprint,
            )

        cases = (
            (
                "两个字段分别输入",
                "two_fields_input_test",
                "two_fields_input",
            ),
            ("Open details", "open_details_card", "details_entry"),
        )
        adapter = self._adapter(FakeSceneObserver([]), FakeRobot())
        for label, planned_meaning, fresh_meaning in cases:
            with self.subTest(label=label):
                planned = target_scene(
                    "planned",
                    label=label,
                    meaning=planned_meaning,
                    element_id="planned-target",
                )
                fresh = target_scene(
                    "fresh",
                    label=label,
                    meaning=fresh_meaning,
                    element_id="fresh-target",
                    bounds=(0.132, 0.451, 0.868, 0.531),
                )
                requested = SemanticAction(
                    node_id="exact-target",
                    action="tap_semantic",
                    params={
                        "element_id": "planned-target",
                        "target": planned_meaning,
                        "role": "button",
                        "label": label,
                        "states": states,
                    },
                )
                exact_label = _typed_exact_tap_target_label(
                    exact_tap_goal(label),
                    requested,
                )

                rebound = adapter._rebind_action(
                    requested,
                    planned,
                    fresh,
                    typed_exact_target_label=exact_label,
                )

                self.assertEqual(label, exact_label)
                self.assertEqual("fresh-target", rebound.params["element_id"])
                self.assertEqual(fresh_meaning, rebound.params["target"])

    def test_exact_typed_label_does_not_bypass_other_rebind_gates(self):
        label = "两个字段分别输入"
        states = {"goal_relevant": True, "fully_visible": True, "enabled": True}
        planned_element = UIElement(
            element_id="planned-target",
            role="button",
            meaning="two_fields_input_test",
            label=label,
            bounds=(0.13, 0.45, 0.87, 0.53),
            confidence=1.0,
            states=states,
        )

        def target_scene(
            fingerprint,
            *,
            element=planned_element,
            app_id="settings",
            screen_id="acceptance_modes",
            extra_elements=(),
        ):
            return UIScene(
                app_id=app_id,
                screen_id=screen_id,
                summary="验收模式列表",
                elements=(element, *extra_elements),
                stable=True,
                confidence=1.0,
                fingerprint=fingerprint,
            )

        planned = target_scene("planned")
        requested = SemanticAction(
            node_id="exact-target",
            action="tap_semantic",
            params={
                "element_id": "planned-target",
                "target": planned_element.meaning,
                "role": planned_element.role,
                "label": label,
                "states": states,
            },
        )
        exact_label = _typed_exact_tap_target_label(
            exact_tap_goal(label),
            requested,
        )
        drifted = replace(
            planned_element,
            element_id="fresh-target",
            meaning="two_fields_input",
        )
        duplicate = replace(
            drifted,
            element_id="duplicate-target",
            bounds=(0.13, 0.55, 0.87, 0.63),
        )
        cases = (
            (
                "different-label",
                target_scene(
                    "fresh-label",
                    element=replace(drifted, label="另一个入口"),
                ),
            ),
            (
                "different-role",
                target_scene(
                    "fresh-role",
                    element=replace(drifted, role="text"),
                ),
            ),
            (
                "duplicate-label",
                target_scene(
                    "fresh-duplicate",
                    element=drifted,
                    extra_elements=(duplicate,),
                ),
            ),
            (
                "state-conflict",
                target_scene(
                    "fresh-state",
                    element=replace(
                        drifted,
                        states={**states, "enabled": False},
                    ),
                ),
            ),
            (
                "fresh-not-goal-relevant",
                target_scene(
                    "fresh-goal",
                    element=replace(
                        drifted,
                        states={**states, "goal_relevant": False},
                    ),
                ),
            ),
            (
                "fresh-not-fully-visible",
                target_scene(
                    "fresh-visible",
                    element=replace(
                        drifted,
                        states={**states, "fully_visible": False},
                    ),
                ),
            ),
            (
                "app-switch",
                target_scene("fresh-app", element=drifted, app_id="browser"),
            ),
            (
                "screen-switch",
                target_scene(
                    "fresh-screen",
                    element=drifted,
                    screen_id="other_screen",
                ),
            ),
            (
                "geometry-conflict",
                target_scene(
                    "fresh-geometry",
                    element=replace(
                        drifted,
                        bounds=(0.05, 0.75, 0.35, 0.84),
                    ),
                ),
            ),
        )
        adapter = self._adapter(FakeSceneObserver([]), FakeRobot())
        for name, fresh in cases:
            with self.subTest(name=name), self.assertRaises(GenericActionAdapterError):
                adapter._rebind_action(
                    requested,
                    planned,
                    fresh,
                    typed_exact_target_label=exact_label,
                )

        self.assertEqual(
            "",
            _typed_exact_tap_target_label(goal(), requested),
        )
        with self.assertRaisesRegex(GenericActionAdapterError, "语义"):
            adapter._rebind_action(
                requested,
                planned,
                target_scene("fresh-ordinary", element=drifted),
            )

    def test_rebind_allows_unique_overlapping_input_meaning_alias(self):
        states = {"goal_relevant": True, "fully_visible": True, "value": ""}
        planned = UIScene(
            app_id="unknown",
            screen_id="input_page",
            summary="唯一空输入框可见",
            elements=(
                UIElement(
                    element_id="planned_input",
                    role="input",
                    meaning="application_text_input",
                    label="",
                    bounds=(0.13, 0.51, 0.87, 0.60),
                    confidence=1.0,
                    states=states,
                ),
            ),
            fingerprint="planned",
        )
        fresh = replace(
            planned,
            elements=(
                replace(
                    planned.elements[0],
                    element_id="fresh_input",
                    meaning="text_input_field",
                ),
            ),
            fingerprint="fresh",
        )
        adapter = self._adapter(FakeSceneObserver([fresh]), FakeRobot())

        rebound = adapter._rebind_action(
            SemanticAction(
                node_id="focus-input",
                action="tap_semantic",
                params={
                    "element_id": "planned_input",
                    "target": "application_text_input",
                    "role": "input",
                    "label": "",
                    "states": states,
                },
            ),
            planned,
            fresh,
        )

        self.assertEqual("fresh_input", rebound.params["element_id"])
        self.assertEqual("text_input_field", rebound.params["target"])

    def test_rebind_accepts_unknown_to_known_keyboard_case_enrichment(self):
        planned_states = {
            "goal_relevant": True,
            "fully_visible": True,
            "value": "",
            "focused": True,
            "keyboard_layout": "qwerty",
            "keyboard_input_mode": "chinese_pinyin",
            "keyboard_case_mode": "unknown",
        }
        planned = UIScene(
            app_id="chat",
            screen_id="conversation",
            summary="唯一空白输入框已聚焦",
            elements=(
                UIElement(
                    element_id="planned_input",
                    role="input",
                    meaning="application_text_input",
                    label="",
                    bounds=(0.15, 0.54, 0.69, 0.59),
                    confidence=1.0,
                    states=planned_states,
                ),
            ),
            fingerprint="planned",
        )
        fresh_states = dict(planned_states)
        fresh_states.update(
            {
                "goal_relevant": False,
                "keyboard_case_mode": "lower",
            }
        )
        fresh = replace(
            planned,
            elements=(
                replace(
                    planned.elements[0],
                    element_id="fresh_input",
                    bounds=(0.153, 0.54, 0.69, 0.585),
                    states=fresh_states,
                ),
            ),
            fingerprint="fresh",
        )
        adapter = self._adapter(FakeSceneObserver([]), FakeRobot())

        rebound = adapter._rebind_action(
            SemanticAction(
                node_id="input-long-text",
                action="input_verified_text",
                params={
                    "element_id": "planned_input",
                    "target": "application_text_input",
                    "role": "input",
                    "label": "",
                    "states": planned_states,
                    "text": "复杂输入验收2026",
                    "formal_candidate_id": "candidate-long-text",
                },
            ),
            planned,
            fresh,
        )

        self.assertEqual("fresh_input", rebound.params["element_id"])
        self.assertEqual("lower", rebound.params["states"]["keyboard_case_mode"])

    def test_rebind_rejects_known_keyboard_case_change(self):
        planned_states = {
            "fully_visible": True,
            "value": "",
            "focused": True,
            "keyboard_layout": "qwerty",
            "keyboard_input_mode": "direct_latin",
            "keyboard_case_mode": "lower",
        }
        planned = UIScene(
            app_id="form",
            screen_id="edit",
            summary="输入框已聚焦",
            elements=(
                UIElement(
                    element_id="planned_input",
                    role="input",
                    meaning="application_text_input",
                    label="",
                    bounds=(0.1, 0.2, 0.9, 0.3),
                    confidence=1.0,
                    states=planned_states,
                ),
            ),
            fingerprint="planned",
        )
        fresh_states = dict(planned_states)
        fresh_states["keyboard_case_mode"] = "upper"
        fresh = replace(
            planned,
            elements=(
                replace(
                    planned.elements[0],
                    element_id="fresh_input",
                    states=fresh_states,
                ),
            ),
            fingerprint="fresh",
        )
        adapter = self._adapter(FakeSceneObserver([]), FakeRobot())

        with self.assertRaisesRegex(
            GenericActionAdapterError,
            "目标语义不再严格唯一",
        ):
            adapter._rebind_action(
                SemanticAction(
                    node_id="input-latin",
                    action="input_verified_text",
                    params={
                        "element_id": "planned_input",
                        "target": "application_text_input",
                        "role": "input",
                        "label": "",
                        "states": planned_states,
                        "text": "agent",
                        "formal_candidate_id": "candidate-latin",
                    },
                ),
                planned,
                fresh,
            )

    def test_changed_target_region_before_confirmation_stops_without_robot_action(self):
        planned = scene("planned")
        fresh = scene("fresh", bounds=(0.65, 0.65, 0.85, 0.85))
        observer = FakeSceneObserver([fresh])
        robot = FakeRobot()
        action = SemanticAction(
            node_id="generic_step_1",
            action="tap_semantic",
            params={"element_id": "e1", "target": "app_icon"},
        )

        with self.assertRaisesRegex(GenericActionAdapterError, "区域"):
            self._adapter(observer, robot).execute(
                requested_action=action,
                planned_scene=planned,
                goal=goal(),
                confirmed=True,
            )
        self.assertEqual(robot.actions, [])

    def test_changed_target_state_before_confirmation_stops_without_robot_action(self):
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
                    element_id="fresh",
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
        with self.assertRaisesRegex(GenericActionAdapterError, "语义|状态"):
            self._adapter(FakeSceneObserver([fresh]), robot).execute(
                requested_action=SemanticAction(
                    node_id="generic_step_1",
                    action="tap_semantic",
                    params={"element_id": "e1", "target": "app_icon"},
                ),
                planned_scene=planned,
                goal=goal(),
                confirmed=True,
            )
        self.assertEqual([], robot.actions)

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
                    [scene("same", element_id="fresh-1"), scene("after-1", screen_id="app_home")]
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
                    [scene("same", element_id="fresh-2"), scene("after-2", screen_id="app_home")]
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

    def test_changed_screen_before_confirmation_stops_without_robot_action(self):
        planned = scene("planned")
        changed = scene("before", screen_id="app_home")
        observer = FakeSceneObserver([changed])
        robot = FakeRobot()
        action = SemanticAction(
            node_id="generic_step_1",
            action="tap_semantic",
            params={"element_id": "e1", "target": "app_icon"},
        )
        with self.assertRaisesRegex(GenericActionAdapterError, "已变化"):
            self._adapter(observer, robot).execute(
                requested_action=action,
                planned_scene=planned,
                goal=goal(),
                confirmed=True,
            )
        self.assertEqual(robot.actions, [])

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
            action="swipe",
            params={"direction": "up", "expected_effect": {"scene_changed": True}},
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

    def test_physical_gate_drift_persists_exact_actual_frame_without_action(self):
        planned = scene("planned", screen_id="generic_action_verification_page")
        observer = FakeSceneObserver([])
        robot = PhysicalGateDriftRobot()
        adapter = GenericSingleActionAdapter(
            capture=SequenceCapture(["gray"] * 4),
            observer=observer,
            robot=robot,
            frame_interval=0,
            post_action_settle=0,
        )
        action = SemanticAction(
            node_id="generic_step_1",
            action="swipe",
            params={"direction": "up", "expected_effect": {"scene_changed": True}},
        )

        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(
                GenericActionAdapterError,
                "共享物理执行门.*视觉漂移",
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
            actual_paths = [
                Path(item)
                for item in caught.exception.evidence
                if "physical_gate_actual" in Path(item).name
            ]
            self.assertEqual(1, len(actual_paths))
            self.assertTrue(actual_paths[0].is_file())
            with Image.open(actual_paths[0]) as persisted:
                self.assertEqual((540, 960), persisted.size)
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
                params={"element_id": "e1", "target": "app_icon"},
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
        first_segment = "longinput"
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
            qwerty_row_snapper=lambda _frames, _anchors: snapped_anchors,
            require_local_qwerty_row_snap=True,
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
                    "expected_effect": {
                        "element_state": {
                            "meaning": "application_text_input",
                            "states": {"value": first_segment},
                        }
                    },
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
        self.assertEqual("matched", result.action_outcome)
        self.assertGreater(adapter.capture.calls, 8)
        self.assertEqual(1, observer.calls)
        self.assertEqual("planned", result.before_scene.fingerprint)
        self.assertEqual(
            "stable_local_ocr",
            robot.keyboard_layouts[0]["row_snap_source"],
        )
        self.assertEqual([], observer.geometry_audit_calls)

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
        self.assertEqual([], observer.geometry_audit_calls)

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
                    "expected_effect": {"scene_changed": True},
                },
            ),
            planned_scene=planned,
            planned_frames=tuple(
                Image.new("RGB", (540, 960), "gray") for _ in range(4)
            ),
            goal=goal(),
            confirmed=True,
        )

        self.assertEqual([], observer.geometry_audit_calls)
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
                    "expected_effect": {"scene_changed": True},
                },
            ),
            planned_scene=planned,
            planned_frames=tuple(
                Image.new("RGB", (540, 960), "gray") for _ in range(4)
            ),
            goal=goal(),
            confirmed=True,
        )

        self.assertEqual([], observer.geometry_audit_calls)
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
                        "expected_effect": {"scene_changed": True},
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

    def test_drag_rebind_accepts_safe_meaning_synonyms_for_exact_labelled_endpoints(self):
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
        observer = FakeSceneObserver(
            [fresh, after],
            geometry_scenes=[fresh, fresh],
        )
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
                    "expected_effect": {"scene_changed": True},
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

    def test_changed_local_frames_stop_even_when_model_screen_id_matches(self):
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
            action="swipe",
            params={"direction": "up", "expected_effect": {"scene_changed": True}},
        )

        with self.assertRaisesRegex(GenericActionAdapterError, "本地真实画面已变化"):
            adapter.execute(
                requested_action=action,
                planned_scene=planned,
                planned_frames=tuple(
                    Image.new("RGB", (540, 960), "black") for _ in range(4)
                ),
                goal=goal(),
                confirmed=True,
            )

        self.assertEqual([], robot.actions)

    def test_low_confidence_observation_stops_without_model_retry(self):
        planned = scene("planned")
        fresh = scene("before", element_id="fresh")
        after = scene("after", screen_id="app_home", element_id="after")
        observer = FakeSceneObserver(
            [
                RuntimeError("页面不稳定或整体置信度不足，不能建立可信候选。"),
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

    def test_first_low_confidence_failure_stops_without_action(self):
        observer = FakeSceneObserver(
            [
                RuntimeError("页面不稳定或整体置信度不足，不能建立可信候选。"),
                RuntimeError("页面不稳定或整体置信度不足，不能建立可信候选。"),
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
                planned_scene=scene("planned"),
                goal=goal(),
                confirmed=True,
            )

        self.assertEqual(1, observer.calls)
        self.assertEqual([], robot.actions)

    def test_goal_polluted_home_app_is_normalized_before_confirmation(self):
        planned = scene("planned", app_id="douyin")
        fresh = scene("before", element_id="fresh", app_id="unknown")
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

    def test_failed_post_verification_never_retries_physical_action(self):
        planned = scene("same")
        fresh = scene("same", element_id="fresh")
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
        self.assertEqual(result.action_outcome, "mismatched")
        self.assertTrue(result.verification_errors)
        self.assertIn("没有可验证", result.verification_errors[-1])
        self.assertEqual(len(robot.actions), 1)

    def test_post_action_waits_until_four_frame_window_is_locally_stable(self):
        planned = scene("planned")
        fresh = scene("before", element_id="fresh")
        after = scene("after", screen_id="app_home", element_id="after")
        observer = FakeSceneObserver([fresh, after])
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
            goal=goal(),
            confirmed=True,
        )
        self.assertEqual(result.physical_actions, 1)
        self.assertEqual(robot.actions, [("tap", 300, 400)])
        self.assertEqual(capture.calls, 9)
        self.assertEqual(observer.calls, 2)

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

    def test_continuous_action_waits_until_camera_returns_to_phone_view(self):
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
            require_phone_view_identity=True,
        )

        self.assertEqual(8, capture.calls)
        self.assertEqual(4, len(frames))
        self.assertEqual(recovered.tobytes(), frames[-1].tobytes())

    def test_stable_off_phone_view_times_out_before_model_observation(self):
        reference = textured_phone_frame()
        off_phone = ImageOps.invert(reference)
        observer = FakeSceneObserver([])
        adapter = GenericSingleActionAdapter(
            capture=SequenceCapture([off_phone] * 4),
            observer=observer,
            robot=FakeRobot(),
            frame_interval=0,
            post_action_settle=0,
            post_action_timeout=1,
            post_action_continuous_timeout=45,
        )

        with patch(
            "generic_action_adapter.time.monotonic",
            side_effect=[0.0, 0.0, 0.0, 1.0],
        ):
            with self.assertRaisesRegex(
                GenericActionAdapterError,
                "相机尚未回到手机取景.*取景差异.*要求最多45.0",
            ):
                adapter._capture_stable_post_action_frames(
                    deadline=0.5,
                    evidence_dir=None,
                    prefix="off_phone_timeout",
                    clarity_reference_frames=tuple(
                        reference.copy() for _ in range(4)
                    ),
                    require_relative_clarity=True,
                    require_phone_view_identity=True,
                )

        self.assertEqual(0, observer.calls)

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
                    expected_effect={},
                )
                self.assertTrue(
                    adapter._requires_post_action_phone_view_identity(resolved)
                )
                self.assertEqual(45, adapter._post_action_timeout_for(resolved))

        navigation = ResolvedSemanticAction(
            node_id="navigation",
            kind="tap_semantic",
            expected_effect={"scene_changed": True},
        )
        self.assertFalse(
            adapter._requires_post_action_phone_view_identity(navigation)
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
            expected_effect={
                "element_state": {
                    "meaning": "application_text_input",
                    "states": {"focused": True},
                }
            },
        )
        navigation = ResolvedSemanticAction(
            node_id="open-page",
            kind="tap_semantic",
            expected_effect={"scene_changed": True},
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
                    expected_effect={
                        "element_state": {
                            "meaning": "application_text_input",
                        }
                    },
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
            "generic_action_adapter.time.monotonic",
            side_effect=[0.0, 0.0, 0.0, 1.0],
        ):
            with self.assertRaisesRegex(
                GenericActionAdapterError,
                "虽已稳定但仍不够清晰.*相对值.*要求至少0.800",
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

    def test_post_action_mismatch_stops_after_one_observation(self):
        planned = scene("planned")
        fresh = scene("before", element_id="fresh")
        transitional = UIScene(
            app_id="unknown",
            screen_id="loading",
            summary="过渡中",
            elements=(),
            stable=False,
            confidence=0.30,
            fingerprint="transition",
        )
        after = scene("after", screen_id="app_home", element_id="after")
        observer = FakeSceneObserver([fresh, transitional, after])
        robot = FakeRobot()
        adapter = GenericSingleActionAdapter(
            capture=lambda: Image.new("RGB", (540, 960), "gray"),
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
            goal=goal(),
            confirmed=True,
        )
        self.assertEqual(result.physical_actions, 1)
        self.assertEqual(robot.actions, [("tap", 300, 400)])
        self.assertEqual(observer.calls, 2)
        self.assertEqual(result.action_outcome, "mismatched")

    def test_post_action_format_failure_does_not_resample_model(self):
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
        self.assertEqual(capture.calls, 8)
        self.assertEqual(len(caught.exception.evidence), 8)
        self.assertEqual(
            caught.exception.observation_errors,
            ("第1轮动作后观察失败：模型返回的 JSON 无法解析",),
        )

    def test_first_post_action_format_failure_stops_after_one_robot_action(self):
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

    def test_post_action_observation_limit_is_hard_capped_at_one_call(self):
        observer = FakeSceneObserver(
            [
                scene("before", element_id="fresh"),
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
            post_action_max_observations=5,
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

        self.assertEqual(adapter.post_action_max_observations, 2)
        self.assertEqual(caught.exception.physical_actions, 1)
        self.assertEqual(observer.calls, 2)
        self.assertEqual(capture.calls, 8)
        self.assertEqual(robot.actions, [("tap", 300, 400)])

    def test_format_error_stops_before_any_second_capture(self):
        observer = FakeSceneObserver(
            [
                scene("before", element_id="fresh"),
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
        fresh = scene("before", element_id="fresh", bounds=(0.11, 0.21, 0.31, 0.41))
        after = scene("after", screen_id="app_home", element_id="after")
        robot = FakeRobot()
        action = SemanticAction(
            node_id="generic_step_1",
            action="dismiss_overlay",
            params={"element_id": "e1", "target": "close"},
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

    def test_semantic_mismatch_returns_without_any_second_capture(self):
        unchanged = scene("camera-noise-only", element_id="after")
        observer = FakeSceneObserver(
            [scene("before", element_id="fresh"), unchanged]
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
            result = adapter.execute(
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

        self.assertEqual(result.physical_actions, 1)
        self.assertEqual(len(result.verification_errors), 1)
        self.assertIn("语义变化", result.verification_errors[0])
        self.assertEqual(len(result.evidence), 8)
        self.assertEqual(observer.calls, 2)
        self.assertEqual(robot.actions, [("tap", 300, 400)])

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
        self.assertEqual(capture.calls, 8)
        self.assertEqual(len(robot.actions), 1)

    def test_post_action_timeout_stops_without_calling_model_or_tapping_again(self):
        planned = scene("planned")
        fresh = scene("before", element_id="fresh")
        observer = FakeSceneObserver([fresh])
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
            "限定时间内没有稳定",
        ) as ctx:
            adapter.execute(
                requested_action=action,
                planned_scene=planned,
                goal=goal(),
                confirmed=True,
            )
        self.assertEqual(ctx.exception.physical_actions, 1)
        self.assertEqual(observer.calls, 1)
        self.assertEqual(robot.actions, [("tap", 300, 400)])


if __name__ == "__main__":
    unittest.main()
