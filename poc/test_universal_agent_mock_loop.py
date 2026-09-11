from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from PIL import Image, ImageDraw
from agent.infrastructure import DeviceTaskRegistry, FileSystemAgentEvidenceStore
from agent.application.action_adapter import GenericActionAdapterError
from agent.infrastructure.generic_action_adapter import GenericSingleActionAdapter
from agent.domain.canonical_action_protocol import (
    GenericStepProposal,
    bind_same_response_action,
)
from agent.infrastructure.orientation_safety import _claim_audit_seal
from agent.domain.ui_scene import CameraAlignmentFacts, UIElement, UIScene
from agent.application.universal_agent_orchestrator import (
    UniversalAgentOrchestrator,
)
from test_universal_agent_orchestrator import CountingQwen
from test_single_visual_loop import Observation as _trusted_factory

def _confirmation(session):
    return session.confirmation_authority.scope()


def synthetic_frame(*, page: str, unstable_variant: int = 0) -> Image.Image:
    """Create an owned, generic phone UI frame with no third-party assets."""

    colors = {
        "before": (244, 247, 250),
        "after": (220, 238, 255),
        "unstable": (
            (255, 255, 255) if unstable_variant % 2 else (0, 0, 0)
        ),
    }
    image = Image.new("RGB", (540, 960), colors[page])
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((36, 80, 504, 170), radius=18, fill=(28, 38, 54))
    draw.rounded_rectangle((64, 260, 476, 390), radius=22, fill=(255, 255, 255))
    draw.rounded_rectangle((64, 430, 476, 560), radius=22, fill=(255, 255, 255))
    draw.rectangle((80, 300, 320, 326), fill=(80, 100, 130))
    if page == "after":
        draw.rectangle((80, 470, 420, 500), fill=(40, 118, 205))
    elif page == "unstable":
        draw.rectangle(
            (0, 0, 540, 220),
            fill=(0, 0, 0) if unstable_variant % 2 else (255, 255, 255),
        )
        x = 80 if unstable_variant % 2 else 280
        draw.rectangle((x, 650, x + 160, 720), fill=(210, 56, 72))
    return image


def scene(
    *,
    app_id: str,
    fingerprint: str,
    action_kind: str,
    unsafe: bool = False,
) -> UIScene:
    elements = ()
    if action_kind == "tap_semantic":
        elements = (
            UIElement(
                element_id="generic-entry",
                role="toggle" if unsafe else "button",
                meaning="enable_setting" if unsafe else "open_details",
                label="启用" if unsafe else "查看内容",
                bounds=(0.12, 0.27, 0.88, 0.42),
                confidence=0.98,
                states={"goal_relevant": True, "fully_visible": True},
                evidence=("合成画面中的唯一候选",),
            ),
        )
    elif action_kind == "scroll":
        elements = (
            UIElement(
                element_id="generic-scroll-surface",
                role="container",
                meaning="scrollable_content",
                label="",
                bounds=(0.05, 0.18, 0.95, 0.9),
                confidence=0.98,
                states={
                    "fully_visible": True,
                    "scrollable": True,
                    "scroll_axis": "vertical",
                },
                evidence=("合成内容区域仍可继续浏览",),
            ),
        )
    return UIScene(
        app_id=app_id,
        screen_id="after" if fingerprint.endswith("after") else "before",
        summary="合成详情页" if fingerprint.endswith("after") else "合成入口页",
        elements=elements,
        stable=True,
        confidence=0.98,
        fingerprint=fingerprint,
        camera_alignment=CameraAlignmentFacts(
            camera_layout_orientation="portrait",
            phone_content_rotation="upright",
            confidence=0.98,
            evidence=("合成手机界面与相机画布正向一致",),
        ),
    )


def _same_frame_model_decision(scene: UIScene, *, status: str, action_kind: str) -> dict:
    if status == "finish":
        return {
            "status": "finish", "previous_action_outcome": "matched",
            "evidence_refs": ["scene.summary"],
            "confidence": 0.97,
            "reason": "合成的新截图已经证明当前目标完成。",
        }
    payload = {
        "status": "action",
        "action": action_kind,
        "confidence": 0.97,
        "reason": "合成观察在同一帧直接选择一个动作。",`n        "postcondition": {"status": "unknown", "fact": "等待动作后重新观察"},
    }
    if action_kind in {
        "tap_semantic",
        "dismiss_overlay",
        "input_verified_text",
        "press_enter",
        "clear_verified_text",
        "double_tap",
        "long_press",
    }:
        if len(scene.elements) != 1:
            raise AssertionError("测试 scene 必须只有一个同帧动作目标。")
        element = scene.elements[0]
        if action_kind in {"tap_semantic", "dismiss_overlay", "press_enter", "double_tap", "long_press"}:
            payload["target"] = {
                "element_id": element.element_id,
                "role": element.role,
                "meaning": element.meaning,
                "label": element.label,
                "evidence": list(element.evidence),
            }
            center_x, center_y = element.center
            payload["tap_point"] = [round(center_x * 1000), round(center_y * 1000)]
        else:
            payload["element_id"] = element.element_id
    elif action_kind == "scroll":
        payload["direction"] = "up"
    return payload


class ScriptedObserver:
    def __init__(self, before_scene: UIScene, after_scene: UIScene, action_kind: str) -> None:
        self.before_scene = before_scene
        self.after_scene = after_scene
        self.action_kind = action_kind
        self.calls = 0

    def observe_with_decision(self, *, frames, goal_context, **_comparison):
        del goal_context
        self.calls += 1
        pixel = frames[-1].getpixel((0, 0))
        is_after = pixel == (220, 238, 255)
        selected = self.after_scene if is_after else self.before_scene
        return selected, _same_frame_model_decision(
            selected,
            status="finish" if is_after else "action",
            action_kind=self.action_kind,
        )

class ScriptedCapture:
    def __init__(self, *, unstable_after: bool = False) -> None:
        self.calls = 0
        self.unstable_after = unstable_after

    def __call__(self) -> Image.Image:
        self.calls += 1
        if self.calls <= 8:
            return synthetic_frame(page="before")
        if self.unstable_after:
            return synthetic_frame(
                page="unstable",
                unstable_variant=self.calls,
            )
        return synthetic_frame(page="after")


class RecordingRobot:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[int, ...]]] = []
        self._armed = None
        self._click_receipt = None

    def arm_physical_execution(self, credential, *, action, scene_fingerprint):
        credential.assert_authorizes(
            device_id="mock-device",
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

    def vision_tap_relative(self, x: int, y: int):
        self._consume("tap_semantic")
        self.calls.append(("tap", (x, y)))
        self._click_receipt = {
            "input_events_dispatched": True,

            "mechanical_contact_ack": False,
            "click_count": 1,
        }
        return {"ok": True, "kind": "tap"}

    def consume_last_click_receipt(self):
        receipt = self._click_receipt
        self._click_receipt = None
        return receipt

    def vision_android_back(self):
        self._consume("back")
        self.calls.append(("back", ()))
        self._click_receipt = {
            "input_events_dispatched": True,

            "mechanical_contact_ack": False,
            "click_count": 1,
        }
        return {"ok": True, "kind": "back"}

    def vision_swipe_up(self):
        self._consume("swipe")
        self.calls.append(("swipe_up", ()))
        return {"ok": True, "kind": "swipe"}


class UniversalAgentMockLoopTests(unittest.TestCase):
    def _session(
        self,
        temp: str,
        *,
        app_id: str,
        app_name: str,
        raw_goal: str,
        action_kind: str,
        unsafe: bool = False,
        unstable_after: bool = False,
    ):
        before = scene(
            app_id=app_id,
            fingerprint=f"{app_id}-before",
            action_kind=action_kind,
            unsafe=unsafe,
        )
        after = scene(
            app_id=app_id,
            fingerprint=f"{app_id}-after",
            action_kind=action_kind,
            unsafe=unsafe,
        )
        capture = ScriptedCapture(unstable_after=unstable_after)
        observer = ScriptedObserver(before, after, action_kind)
        robot = RecordingRobot()
        adapter = GenericSingleActionAdapter(
            capture=capture,
            observer=observer,
            robot=robot,
            device_id="mock-device",
            frame_interval=0,
            post_action_settle=0,
            post_action_timeout=0.02,
        )
        qwen = CountingQwen()
        orchestrator = UniversalAgentOrchestrator(
            qwen_observer=qwen,
            adapter_factory=lambda _device_id: adapter,
            trusted_observation_factory=_trusted_factory,
            evidence_store_factory=FileSystemAgentEvidenceStore,
            device_registry=DeviceTaskRegistry(),
        )
        session = orchestrator.start(
            session_id=f"session-{app_id}",
            raw_goal=raw_goal,
            device_id="device-1",
            run_dir=Path(temp),
        )
        return orchestrator, session, None, qwen, capture, robot

    def test_unseen_open_goal_executes_one_navigation_tap_then_finishes(self):
        with tempfile.TemporaryDirectory() as temp:
            orchestrator, session, planner, qwen, _capture, robot = self._session(
                temp,
                app_id="synthetic.catalog",
                app_name="合成目录",
                raw_goal="把眼前这个条目的内容页打开给我看",
                action_kind="tap_semantic",
            )
            result = orchestrator.confirm_one(session, _confirmation(session))
            evidence_names = {item.name for item in Path(temp).iterdir()}
            self.assertTrue(
                {
                    "trusted_observation_step_1.json",
                    "trusted_observation_step_2.json",
                    "qwen_decision_step_1.json",
                    "controller_decision_step_1.json",
                    "verification_step_1.json",
                    "session.json",
                    "report.json",
                }.issubset(evidence_names)
            )

        self.assertEqual(1, result.physical_actions)
        self.assertEqual(["tap"], [item[0] for item in robot.calls])
        self.assertEqual(2, session.step_number)
        self.assertEqual(2, len(qwen.calls))
        self.assertEqual("current_surface", session.goal_draft.app_id)

    def test_rephrased_back_goal_executes_one_back_then_finishes(self):
        with tempfile.TemporaryDirectory() as temp:
            orchestrator, session, planner, qwen, _capture, robot = self._session(
                temp,
                app_id="synthetic.reader",
                app_name="合成阅读器",
                raw_goal="我不想停在这里，退回刚才那一层",
                action_kind="back",
            )
            result = orchestrator.confirm_one(session, _confirmation(session))

        self.assertEqual(1, result.physical_actions)
        self.assertEqual([("back", ())], robot.calls)
        self.assertEqual(2, session.step_number)
        self.assertEqual(2, len(qwen.calls))
        self.assertEqual("current_surface", session.goal_draft.app_id)

    def test_third_unseen_app_combines_generic_scroll_without_code_branch(self):
        with tempfile.TemporaryDirectory() as temp:
            orchestrator, session, _planner, qwen, _capture, robot = self._session(
                temp,
                app_id="synthetic.timeline",
                app_name="合成时间线",
                raw_goal="这页没有我要的公开条目，往下翻一屏再判断",
                action_kind="scroll",
            )
            result = orchestrator.confirm_one(session, _confirmation(session))

        self.assertEqual(1, result.physical_actions)
        self.assertEqual([("swipe_up", ())], robot.calls)
        self.assertEqual(2, session.step_number)
        self.assertEqual(2, len(qwen.calls))
        self.assertEqual("current_surface", session.goal_draft.app_id)

    def test_two_phrasings_use_the_same_generic_action_contract(self):
        outcomes = []
        for index, wording in enumerate(
            ("把眼前条目的内容页打开", "进入当前唯一可见项目看看详情"),
            start=1,
        ):
            with tempfile.TemporaryDirectory() as temp:
                orchestrator, session, _planner, _qwen, _capture, robot = self._session(
                    temp,
                    app_id=f"synthetic.paraphrase-{index}",
                    app_name="合成目录",
                    raw_goal=wording,
                    action_kind="tap_semantic",
                )
                result = orchestrator.confirm_one(session, _confirmation(session))
                outcomes.append((result.resolved_action.kind, robot.calls[0][0]))

        self.assertEqual([("tap_semantic", "tap"), ("tap_semantic", "tap")], outcomes)

    def test_ordinary_toggle_effect_is_offered_and_executes_once(self):
        with tempfile.TemporaryDirectory() as temp:
            orchestrator, session, _planner, _qwen, _capture, robot = self._session(
                temp,
                app_id="synthetic.controls",
                app_name="合成控制页",
                raw_goal="把这个开关启用",
                action_kind="tap_semantic",
                unsafe=True,
            )
            evidence_names = {item.name for item in Path(temp).iterdir()}
            self.assertTrue(
                {
                    "trusted_observation_step_1.json",
                    "qwen_decision_step_1.json",
                    "controller_decision_step_1.json",
                    "session.json",
                    "report.json",
                }.issubset(evidence_names)
            )
            self.assertFalse(
                any(name.startswith("verification_") for name in evidence_names)
            )
            self.assertFalse(any(name.startswith("after_") for name in evidence_names))

            self.assertEqual("awaiting_confirmation", session.status)
            self.assertEqual(0, session.physical_actions)
            self.assertEqual([], robot.calls)

            result = orchestrator.confirm_one(session, _confirmation(session))

        self.assertEqual(1, result.physical_actions)
        self.assertEqual(1, session.physical_actions)
        self.assertEqual(["tap"], [item[0] for item in robot.calls])

    def test_changing_after_frames_continue_after_one_action_without_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            orchestrator, session, _planner, qwen, _capture, robot = self._session(
                temp,
                app_id="synthetic.unstable",
                app_name="合成动态页",
                raw_goal="进入这个公开信息入口",
                action_kind="tap_semantic",
                unstable_after=True,
            )
            orchestrator.confirm_one(session, _confirmation(session))

        self.assertEqual(1, session.physical_actions)
        self.assertEqual(["tap"], [item[0] for item in robot.calls])
        self.assertEqual(2, len(qwen.calls))
        self.assertEqual("awaiting_confirmation", session.status)  # new model action; never force finish


if __name__ == "__main__":
    unittest.main()


