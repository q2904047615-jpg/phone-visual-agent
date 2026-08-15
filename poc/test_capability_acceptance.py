import hashlib
import json
from dataclasses import replace
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest

from PIL import Image

from capability_acceptance import (
    CapabilityAcceptanceError,
    CapabilityRegistryPromoter,
    validated_calibration_evidence,
    validate_acceptance_report,
)
from device_exclusivity import InterProcessLease
from orientation_safety import (
    ORIENTATION_AUDIT_SOURCE,
    ORIENTATION_CREDENTIAL_VERSION,
    OrientationCredential,
    PhysicalExecutionGate,
    _mint_audited_credential,
    frame_fingerprint,
)


class CapabilityAcceptanceCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.trial_dir = self.root / "trial-001"
        self.trial_dir.mkdir()
        self.registry_path = self.root / "device_registry.json"
        self.calibration_path = self.root / "tap-a.json"
        self.calibration_path.write_text(
            json.dumps(
                {
                    "version": 2,
                    "enabled": True,
                    "validated": True,
                    "accepted_fit": True,
                    "frame_size": [540, 960],
                    "coverage": {
                        "sufficient": True,
                        "normalized_bounds": [0.05, 0.05, 0.95, 0.95],
                        "normalized_hull": [
                            [0.05, 0.05],
                            [0.95, 0.05],
                            [0.95, 0.95],
                            [0.05, 0.95],
                        ],
                    },
                    "validation": {
                        "passed": True,
                        "coverage_passed": True,
                        "coverage": {
                            "sufficient": True,
                            "normalized_bounds": [0.05, 0.05, 0.95, 0.95],
                            "normalized_hull": [
                                [0.05, 0.05],
                                [0.95, 0.05],
                                [0.95, 0.95],
                                [0.05, 0.95],
                            ],
                        },
                    },
                    "target_to_command": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        self.registry_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "default_device_id": "device-a",
                    "devices": [
                        {
                            "device_id": "device-a",
                            "enabled": True,
                            "window_title": "controller-a",
                            "calibration_path": "tap-a.json",
                            "verified_actions": [
                                "tap_semantic",
                                "swipe",
                                "back",
                            ],
                        }
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        self.report_path = self.trial_dir / "acceptance_report.json"
        self._write_valid_report()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _frame(self, name: str, color: str) -> str:
        path = self.trial_dir / name
        Image.new("RGB", (16, 16), color).save(path, format="JPEG")
        return str(path)

    def _valid_report(self) -> dict:
        before = [self._frame(f"before_{index}.jpg", "black") for index in range(1, 5)]
        after = [self._frame(f"after_{index}.jpg", "white") for index in range(1, 5)]
        return {
            "version": 3,
            "trial_id": "trial-001",
            "session_id": "session-001",
            "task_id": "task-001",
            "device_id": "device-a",
            "candidate_action": "drag",
            "calibration_evidence": validated_calibration_evidence(
                self.calibration_path
            ),
            "status": "passed",
            "code_revision": "86b63d8",
            "physical_actions": 1,
            "action_outcome": "matched",
            "before_observation": {
                "observation_id": "obs-before",
                "fingerprint": "fingerprint-confirmed",
            },
            "after_observation": {
                "observation_id": "obs-after",
                "fingerprint": "fingerprint-after",
            },
            "confirmation_scope": {
                "session_id": "session-001",
                "task_id": "task-001",
                "device_id": "device-a",
                "revision": 1,
                "subgoal_id": "subgoal-001",
                "risk_ids": [],
                "observation_id": "obs-before",
                "fingerprint": "fingerprint-confirmed",
            },
            "execution": {
                "orientation_credential": {
                    "version": ORIENTATION_CREDENTIAL_VERSION,
                    "credential_id": "credential-001",
                    "source": ORIENTATION_AUDIT_SOURCE,
                    "device_id": "device-a",
                    "scene_fingerprint": "fingerprint-execution-before",
                    "frame_fingerprint": frame_fingerprint(
                        Image.open(before[0]).convert("RGB")
                    ),
                    "evidence_frame_fingerprint": frame_fingerprint(
                        Image.open(before[0]).convert("RGB")
                    ),
                    "frame_size": [16, 16],
                    "camera_layout_orientation": "square",
                    "phone_content_rotation": "upright",
                    "confidence": 0.95,
                    "evidence": ["手机状态文字正向"],
                },
                "resolved_action": {
                    "node_id": "drag-001",
                    "kind": "drag",
                    "normalized_point": [0.15, 0.25],
                    "normalized_end_point": [0.8, 0.3],
                    "hold_seconds": 0.8,
                    "path_distance": 0.6519202405202649,
                    "target_element_id": "source",
                    "destination_element_id": "destination",
                    "before_fingerprint": "fingerprint-execution-before",
                    "expected_effect": {"scene_changed": True},
                },
                "before_scene": {
                    "foreground_app_id": "test-app",
                    "screen_id": "board",
                    "summary": "拖动前",
                    "elements": [
                        {
                            "element_id": "source",
                            "role": "button",
                            "meaning": "draggable_item",
                            "label": "项目",
                            "bounds": [0.1, 0.2, 0.2, 0.3],
                            "confidence": 0.95,
                            "states": {},
                        },
                        {
                            "element_id": "destination",
                            "role": "container",
                            "meaning": "drop_zone",
                            "label": "目标",
                            "bounds": [0.7, 0.2, 0.9, 0.4],
                            "confidence": 0.95,
                            "states": {},
                        },
                    ],
                    "overlays": [],
                    "stable": True,
                    "confidence": 0.95,
                    "fingerprint": "fingerprint-execution-before",
                },
                "after_scene": {
                    "foreground_app_id": "test-app",
                    "screen_id": "board",
                    "summary": "拖动后",
                    "elements": [
                        {
                            "element_id": "source",
                            "role": "button",
                            "meaning": "draggable_item",
                            "label": "项目",
                            "bounds": [0.55, 0.2, 0.65, 0.3],
                            "confidence": 0.95,
                            "states": {},
                        },
                        {
                            "element_id": "destination",
                            "role": "container",
                            "meaning": "drop_zone",
                            "label": "目标",
                            "bounds": [0.7, 0.2, 0.9, 0.4],
                            "confidence": 0.95,
                            "states": {},
                        },
                    ],
                    "overlays": [],
                    "stable": True,
                    "confidence": 0.95,
                    "fingerprint": "fingerprint-after",
                },
                "observation_errors": [],
                "verification_errors": [],
                "robot_result": [[2, 4], [12, 8]],
            },
            "before_frame_paths": before,
            "after_frame_paths": after,
            "before_frame_sha256": [
                hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in before
            ],
            "after_frame_sha256": [
                hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in after
            ],
        }

    def _write_valid_report(self) -> dict:
        report = self._valid_report()
        self.report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return report

    def _valid_reveal_system_navigation_report(self) -> dict:
        report = self._valid_report()
        report["candidate_action"] = "reveal_system_navigation"
        report["before_observation"]["fingerprint"] = (
            "fingerprint-execution-before"
        )
        report["confirmation_scope"]["fingerprint"] = (
            "fingerprint-execution-before"
        )
        report["execution"].update(
            {
                "resolved_action": {
                    "node_id": "reveal-system-navigation-001",
                    "kind": "reveal_system_navigation",
                    "normalized_point": None,
                    "normalized_end_point": None,
                    "text": None,
                    "direction": None,
                    "hold_seconds": None,
                    "path_distance": None,
                    "target_element_id": None,
                    "destination_element_id": None,
                    "before_fingerprint": "fingerprint-execution-before",
                    "expected_effect": {
                        "system_ui": {"navigation_bar_visible": True}
                    },
                },
                "before_scene": {
                    "foreground_app_id": "test-app",
                    "screen_id": "immersive-page",
                    "summary": "沉浸页面，导航栏隐藏",
                    "system_ui": {
                        "immersive_or_fullscreen": True,
                        "navigation_bar_visible": False,
                    },
                    "elements": [],
                    "overlays": [],
                    "stable": True,
                    "confidence": 0.95,
                    "fingerprint": "fingerprint-execution-before",
                },
                "after_scene": {
                    "foreground_app_id": "test-app",
                    "screen_id": "immersive-page",
                    "summary": "系统导航栏已经可见",
                    "system_ui": {
                        "immersive_or_fullscreen": True,
                        "navigation_bar_visible": True,
                    },
                    "elements": [],
                    "overlays": [],
                    "stable": True,
                    "confidence": 0.95,
                    "fingerprint": "fingerprint-after",
                },
                "robot_result": {
                    "action": "reveal_system_navigation",
                    "edge": "bottom",
                    "frame_size": [16, 16],
                    "requested_grid": [[120, 491], [331, 490]],
                    "corrected_grid": [[94, 486], [314, 485]],
                    "client_path": [[2, 15], [8, 10]],
                },
            }
        )
        return report

    def _mutate_report(self, mutation) -> None:
        report = json.loads(self.report_path.read_text(encoding="utf-8"))
        mutation(report)
        self.report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _live_promotion_source(self, *, write_report: bool = True):
        report = json.loads(self.report_path.read_text(encoding="utf-8"))
        before_frames = tuple(
            Image.open(path).convert("RGB") for path in report["before_frame_paths"]
        )
        credential = replace(
            _mint_audited_credential(
                device_id=report["device_id"],
                scene_fingerprint=report["execution"]["before_scene"]["fingerprint"],
                frame=before_frames[0],
                phone_content_rotation="upright",
                confidence=0.95,
                evidence=("手机状态文字正向",),
            ),
            evidence_frame_fingerprint=frame_fingerprint(before_frames[0]),
        )
        PhysicalExecutionGate(report["device_id"]).arm(
            credential,
            action=report["candidate_action"],
            scene_fingerprint=credential.scene_fingerprint,
        )
        result = SimpleNamespace(
            orientation_credential=credential,
            physical_actions=1,
            action_outcome="matched",
            resolved_action=SimpleNamespace(kind=report["candidate_action"]),
            before_scene=SimpleNamespace(fingerprint=credential.scene_fingerprint),
            before_frames=before_frames,
            before_frame_paths=tuple(report["before_frame_paths"]),
        )
        if write_report:
            report["execution"]["orientation_credential"] = credential.to_dict()
            self.report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        return credential, result

    def _preview_live(self, promoter: CapabilityRegistryPromoter):
        credential, result = self._live_promotion_source()
        return promoter.preview(
            self.report_path,
            orientation_credential=credential,
            execution_result=result,
        )

    def test_valid_report_requires_one_matched_action_and_eight_trial_frames(self) -> None:
        report = validate_acceptance_report(self.report_path)

        self.assertEqual(report["trial_id"], "trial-001")
        self.assertEqual(report["candidate_action"], "drag")

    def test_v2_report_remains_readable_but_cannot_promote(self) -> None:
        self._mutate_report(lambda report: report.__setitem__("version", 2))
        readable = json.loads(self.report_path.read_text(encoding="utf-8"))
        self.assertEqual(2, readable["version"])
        with self.assertRaisesRegex(CapabilityAcceptanceError, "版本无效"):
            validate_acceptance_report(self.report_path)

    def test_missing_low_conflicting_or_tampered_orientation_cannot_promote(self):
        cases = (
            (
                lambda report: report["execution"].pop("orientation_credential"),
                "独立方向凭据",
            ),
            (
                lambda report: report["execution"]["orientation_credential"].__setitem__(
                    "confidence", 0.4
                ),
                "置信度不足",
            ),
            (
                lambda report: report["execution"]["orientation_credential"].__setitem__(
                    "device_id", "device-b"
                ),
                "设备不匹配",
            ),
            (
                lambda report: report["execution"]["orientation_credential"].__setitem__(
                    "evidence_frame_fingerprint", "tampered"
                ),
                "未绑定动作前保存",
            ),
            (
                lambda report: report["execution"]["before_scene"].__setitem__(
                    "camera_alignment",
                    {
                        "camera_layout_orientation": "square",
                        "phone_content_rotation": "rotated_90",
                        "confidence": 0.95,
                        "evidence": ["手机文字旋转九十度"],
                    },
                ),
                "主场景方向事实.*冲突",
            ),
        )
        for mutation, message in cases:
            with self.subTest(message=message):
                self._write_valid_report()
                self._mutate_report(mutation)
                with self.assertRaisesRegex(CapabilityAcceptanceError, message):
                    validate_acceptance_report(self.report_path)

    def test_reveal_system_navigation_report_requires_structured_system_ui(self) -> None:
        report = self._valid_reveal_system_navigation_report()
        self.report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        validated = validate_acceptance_report(self.report_path)

        self.assertEqual(
            "reveal_system_navigation", validated["candidate_action"]
        )

    def test_reveal_system_navigation_rejects_summary_only_success(self) -> None:
        report = self._valid_reveal_system_navigation_report()
        report["execution"]["after_scene"]["system_ui"][
            "navigation_bar_visible"
        ] = "unknown"
        self.report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(CapabilityAcceptanceError, "导航栏"):
            validate_acceptance_report(self.report_path)

    def test_reveal_system_navigation_rejects_model_geometry(self) -> None:
        report = self._valid_reveal_system_navigation_report()
        report["execution"]["resolved_action"]["direction"] = "up"
        self.report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(CapabilityAcceptanceError, "模型坐标"):
            validate_acceptance_report(self.report_path)

    def test_reveal_system_navigation_rejects_bad_robot_receipt(self) -> None:
        mutations = (
            (
                "实际像素端点",
                lambda result: result.__setitem__("client_path", [[2, 15]]),
            ),
            (
                "相机尺寸",
                lambda result: result.__setitem__("frame_size", [15, 16]),
            ),
        )
        for expected, mutation in mutations:
            with self.subTest(expected=expected):
                report = self._valid_reveal_system_navigation_report()
                mutation(report["execution"]["robot_result"])
                self.report_path.write_text(
                    json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(CapabilityAcceptanceError, expected):
                    validate_acceptance_report(self.report_path)

    def test_report_rejects_wrong_physical_action_count(self) -> None:
        for value in (0, 2, True, "1"):
            with self.subTest(value=value):
                self._write_valid_report()
                self._mutate_report(lambda report: report.__setitem__("physical_actions", value))
                with self.assertRaisesRegex(CapabilityAcceptanceError, "物理动作数"):
                    validate_acceptance_report(self.report_path)

    def test_report_rejects_mismatched_action_or_outcome(self) -> None:
        mutations = (
            ("动作类型", lambda report: report["execution"]["resolved_action"].__setitem__("kind", "long_press")),
            ("结果", lambda report: report.__setitem__("action_outcome", "mismatched")),
            ("观察错误", lambda report: report["execution"].__setitem__("observation_errors", ["bad frame"])),
            ("验证错误", lambda report: report["execution"].__setitem__("verification_errors", ["not changed"])),
        )
        for message, mutation in mutations:
            with self.subTest(message=message):
                self._write_valid_report()
                self._mutate_report(mutation)
                with self.assertRaisesRegex(CapabilityAcceptanceError, message):
                    validate_acceptance_report(self.report_path)

    def test_input_report_requires_exact_structured_value(self) -> None:
        report = self._valid_report()
        report["candidate_action"] = "input_verified_text"
        report["calibration_evidence"] = None
        report["execution"].update(
            {
                "resolved_action": {
                    "kind": "input_verified_text",
                    "text": "agent",
                    "target_element_id": "field",
                    "before_fingerprint": "fingerprint-execution-before",
                },
                "before_scene": {
                    "foreground_app_id": "test-app",
                    "screen_id": "input",
                    "summary": "输入前",
                    "elements": [
                        {
                            "element_id": "field",
                            "role": "input",
                            "meaning": "search_field",
                            "label": "搜索",
                            "bounds": [0.1, 0.1, 0.9, 0.2],
                            "confidence": 0.95,
                            "states": {
                                "focused": True,
                                "value": "",
                                "keyboard_layout": "qwerty",
                                "keyboard_input_mode": "direct_latin",
                                "goal_relevant": True,
                            },
                        }
                    ],
                    "stable": True,
                    "confidence": 0.95,
                    "fingerprint": "fingerprint-execution-before",
                },
                "after_scene": {
                    "foreground_app_id": "test-app",
                    "screen_id": "input",
                    "summary": "输入后",
                    "elements": [
                        {
                            "element_id": "field-after",
                            "role": "input",
                            "meaning": "search_field",
                            "label": "搜索",
                            "bounds": [0.1, 0.1, 0.9, 0.2],
                            "confidence": 0.95,
                            "states": {"value": "agent.com"},
                        }
                    ],
                    "stable": True,
                    "confidence": 0.95,
                    "fingerprint": "fingerprint-after",
                },
            }
        )
        self.report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(CapabilityAcceptanceError, "文字不匹配"):
            validate_acceptance_report(self.report_path)

        report["execution"]["after_scene"]["elements"][0]["states"]["value"] = "agent"
        self.report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        validated = validate_acceptance_report(self.report_path)
        self.assertEqual("input_verified_text", validated["candidate_action"])

        report["execution"]["after_scene"]["screen_id"] = "other-input-screen"
        self.report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(CapabilityAcceptanceError, "App 或页面身份"):
            validate_acceptance_report(self.report_path)

    def test_input_report_requires_goal_relevant_unique_target(self) -> None:
        report = self._valid_report()
        report["candidate_action"] = "input_verified_text"
        report["calibration_evidence"] = None
        report["execution"].update(
            {
                "resolved_action": {
                    "kind": "input_verified_text",
                    "text": "agent",
                    "target_element_id": "field",
                    "before_fingerprint": "fingerprint-execution-before",
                },
                "before_scene": {
                    "foreground_app_id": "test-app",
                    "screen_id": "input",
                    "summary": "输入前",
                    "elements": [
                        {
                            "element_id": "field",
                            "role": "input",
                            "meaning": "search_field",
                            "label": "搜索",
                            "bounds": [0.1, 0.1, 0.9, 0.2],
                            "confidence": 0.95,
                            "states": {
                                "focused": True,
                                "value": "",
                                "keyboard_layout": "qwerty",
                                "keyboard_input_mode": "direct_latin",
                                "goal_relevant": False,
                            },
                        }
                    ],
                    "stable": True,
                    "confidence": 0.95,
                    "fingerprint": "fingerprint-execution-before",
                },
                "after_scene": {
                    "foreground_app_id": "test-app",
                    "screen_id": "input",
                    "summary": "输入后",
                    "elements": [
                        {
                            "element_id": "field",
                            "role": "input",
                            "meaning": "search_field",
                            "label": "搜索",
                            "bounds": [0.1, 0.1, 0.9, 0.2],
                            "confidence": 0.95,
                            "states": {"value": "agent"},
                        }
                    ],
                    "stable": True,
                    "confidence": 0.95,
                    "fingerprint": "fingerprint-after",
                },
            }
        )
        self.report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(CapabilityAcceptanceError, "缺少可信关联"):
            validate_acceptance_report(self.report_path)

    def test_input_report_rejects_missing_direct_latin_precondition(self) -> None:
        report = self._valid_report()
        report["candidate_action"] = "input_verified_text"
        report["calibration_evidence"] = None
        report["execution"].update(
            {
                "resolved_action": {
                    "kind": "input_verified_text",
                    "text": "agent",
                    "target_element_id": "field",
                    "before_fingerprint": "fingerprint-execution-before",
                },
                "before_scene": {
                    "foreground_app_id": "test-app",
                    "screen_id": "input",
                    "summary": "输入前",
                    "elements": [
                        {
                            "element_id": "field",
                            "role": "input",
                            "meaning": "search_field",
                            "label": "搜索",
                            "bounds": [0.1, 0.1, 0.9, 0.2],
                            "confidence": 0.95,
                            "states": {
                                "focused": True,
                                "value": "",
                                "keyboard_layout": "qwerty",
                                "keyboard_input_mode": "chinese_pinyin",
                                "goal_relevant": True,
                            },
                        }
                    ],
                    "stable": True,
                    "confidence": 0.95,
                    "fingerprint": "fingerprint-execution-before",
                },
                "after_scene": {
                    "foreground_app_id": "test-app",
                    "screen_id": "input",
                    "summary": "输入后",
                    "elements": [
                        {
                            "element_id": "field",
                            "role": "input",
                            "meaning": "search_field",
                            "label": "搜索",
                            "bounds": [0.1, 0.1, 0.9, 0.2],
                            "confidence": 0.95,
                            "states": {"value": "agent"},
                        }
                    ],
                    "stable": True,
                    "confidence": 0.95,
                    "fingerprint": "fingerprint-after",
                },
            }
        )
        self.report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(CapabilityAcceptanceError, "direct_latin"):
            validate_acceptance_report(self.report_path)

    def test_report_rejects_confirmation_scope_fingerprint_drift(self) -> None:
        self._mutate_report(
            lambda report: report["confirmation_scope"].__setitem__(
                "fingerprint",
                "stale-fingerprint",
            )
        )

        with self.assertRaisesRegex(CapabilityAcceptanceError, "确认作用域不一致"):
            validate_acceptance_report(self.report_path)

    def test_drag_report_rejects_tampered_distance_or_duration(self) -> None:
        mutations = (
            ("路径距离", lambda report: report["execution"]["resolved_action"].__setitem__("path_distance", 0.1)),
            ("固定的0.8秒", lambda report: report["execution"]["resolved_action"].__setitem__("hold_seconds", 1.2)),
        )
        for message, mutation in mutations:
            with self.subTest(message=message):
                self._write_valid_report()
                self._mutate_report(mutation)
                with self.assertRaisesRegex(CapabilityAcceptanceError, message):
                    validate_acceptance_report(self.report_path)

    def test_gesture_report_requires_calibration_and_physical_result_evidence(self) -> None:
        mutations = (
            (
                "触控标定证据",
                lambda report: report.__setitem__("calibration_evidence", None),
            ),
            (
                "实际像素端点",
                lambda report: report["execution"].__setitem__("robot_result", None),
            ),
        )
        for message, mutation in mutations:
            with self.subTest(message=message):
                self._write_valid_report()
                self._mutate_report(mutation)
                with self.assertRaisesRegex(CapabilityAcceptanceError, message):
                    validate_acceptance_report(self.report_path)

    def test_calibration_requires_independent_validation_record(self) -> None:
        original = json.loads(self.calibration_path.read_text(encoding="utf-8"))
        mutations = (
            (
                "独立验证记录",
                lambda payload: payload.pop("validation"),
            ),
            (
                "独立验证记录",
                lambda payload: payload["validation"].__setitem__("passed", False),
            ),
            (
                "独立验证记录",
                lambda payload: payload["validation"].__setitem__(
                    "coverage_passed", False
                ),
            ),
            (
                "独立验证覆盖",
                lambda payload: payload["validation"]["coverage"].__setitem__(
                    "sufficient", False
                ),
            ),
        )
        for message, mutation in mutations:
            with self.subTest(message=message):
                payload = json.loads(json.dumps(original))
                mutation(payload)
                self.calibration_path.write_text(
                    json.dumps(payload),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(CapabilityAcceptanceError, message):
                    validated_calibration_evidence(self.calibration_path)
        self.calibration_path.write_text(json.dumps(original), encoding="utf-8")

    def test_report_rejects_unsafe_calibration_evidence_bounds(self) -> None:
        mutations = (
            (
                "归一化屏幕范围",
                "coverage_bounds",
                [-0.01, 0.05, 0.95, 0.95],
            ),
            (
                "归一化屏幕范围",
                "coverage_bounds",
                [0.2, 0.05, 0.7, 0.95],
            ),
            (
                "归一化屏幕范围",
                "validation_coverage_bounds",
                [0.05, 0.2, 0.95, 0.7],
            ),
        )
        for message, field, value in mutations:
            with self.subTest(field=field, value=value):
                self._write_valid_report()
                self._mutate_report(
                    lambda report: report["calibration_evidence"].__setitem__(
                        field,
                        value,
                    )
                )
                with self.assertRaisesRegex(CapabilityAcceptanceError, message):
                    validate_acceptance_report(self.report_path)

    def test_promotion_rejects_calibration_changed_after_report(self) -> None:
        promoter = CapabilityRegistryPromoter(self.registry_path)
        payload = json.loads(self.calibration_path.read_text(encoding="utf-8"))
        payload["frame_size"] = [720, 1280]
        self.calibration_path.write_text(json.dumps(payload), encoding="utf-8")
        credential, result = self._live_promotion_source()

        with self.assertRaisesRegex(CapabilityAcceptanceError, "标定.*不一致"):
            promoter.preview(
                self.report_path,
                orientation_credential=credential,
                execution_result=result,
            )

    def test_report_rejects_uncommitted_code_revision(self) -> None:
        self._mutate_report(
            lambda report: report.__setitem__("code_revision", "86b63d8+dirty")
        )

        with self.assertRaisesRegex(CapabilityAcceptanceError, "未提交代码"):
            validate_acceptance_report(self.report_path)

    def test_report_rejects_unchanged_observation_or_fingerprint(self) -> None:
        mutations = (
            lambda report: report["after_observation"].__setitem__("observation_id", "obs-before"),
            lambda report: report["after_observation"].__setitem__("fingerprint", "fingerprint-confirmed"),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                self._write_valid_report()
                self._mutate_report(mutation)
                with self.assertRaisesRegex(CapabilityAcceptanceError, "动作后.*未变化"):
                    validate_acceptance_report(self.report_path)

    def test_report_rejects_missing_or_outside_frame(self) -> None:
        outside = self.root / "outside.jpg"
        Image.new("RGB", (16, 16), "red").save(outside, format="JPEG")
        for path_value in (str(self.trial_dir / "missing.jpg"), str(outside)):
            with self.subTest(path=path_value):
                self._write_valid_report()
                self._mutate_report(
                    lambda report: report["before_frame_paths"].__setitem__(0, path_value)
                )
                with self.assertRaisesRegex(CapabilityAcceptanceError, "证据"):
                    validate_acceptance_report(self.report_path)

    def test_report_hash_binds_each_evidence_frame(self) -> None:
        report = self._write_valid_report()
        Image.new("RGB", (16, 16), "blue").save(
            report["after_frame_paths"][0],
            format="JPEG",
        )

        with self.assertRaisesRegex(CapabilityAcceptanceError, "证据摘要"):
            validate_acceptance_report(self.report_path)

    def test_promotion_scope_binds_report_and_registry_hashes(self) -> None:
        promoter = CapabilityRegistryPromoter(self.registry_path)
        authority = self._preview_live(promoter)
        scope = authority.scope

        self.assertEqual(scope.trial_id, "trial-001")
        self.assertEqual(scope.device_id, "device-a")
        self.assertEqual(scope.action, "drag")
        self.assertEqual(
            scope.report_sha256,
            hashlib.sha256(self.report_path.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            scope.registry_sha256,
            hashlib.sha256(self.registry_path.read_bytes()).hexdigest(),
        )

    def test_report_only_preview_fails_closed_without_registry_write(self) -> None:
        promoter = CapabilityRegistryPromoter(self.registry_path)
        before = self.registry_path.read_bytes()

        with self.assertRaisesRegex(CapabilityAcceptanceError, "live manager"):
            promoter.preview(self.report_path)

        self.assertEqual(self.registry_path.read_bytes(), before)

    def test_deserialized_or_different_credential_cannot_preview(self) -> None:
        promoter = CapabilityRegistryPromoter(self.registry_path)
        credential, result = self._live_promotion_source()
        serialized = OrientationCredential.from_dict(credential.to_dict())
        serialized_result = SimpleNamespace(**vars(result))
        serialized_result.orientation_credential = serialized
        before = self.registry_path.read_bytes()

        with self.assertRaisesRegex(CapabilityAcceptanceError, "live-trial"):
            promoter.preview(
                self.report_path,
                orientation_credential=serialized,
                execution_result=serialized_result,
            )

        cloned = replace(credential)
        cloned_result = SimpleNamespace(**vars(result))
        cloned_result.orientation_credential = cloned
        with self.assertRaisesRegex(CapabilityAcceptanceError, "live-trial"):
            promoter.preview(
                self.report_path,
                orientation_credential=cloned,
                execution_result=cloned_result,
            )
        authority = promoter.preview(
            self.report_path,
            orientation_credential=credential,
            execution_result=result,
        )
        authority.invalidate()

        other_credential, other_result = self._live_promotion_source(
            write_report=False
        )

        with self.assertRaisesRegex(CapabilityAcceptanceError, "序列化视图"):
            promoter.preview(
                self.report_path,
                orientation_credential=other_credential,
                execution_result=other_result,
            )

        self.assertIsNot(credential, other_credential)
        self.assertIs(result.orientation_credential, credential)
        self.assertEqual(self.registry_path.read_bytes(), before)

    def test_copied_report_cannot_reissue_live_authority(self) -> None:
        promoter = CapabilityRegistryPromoter(self.registry_path)
        credential, result = self._live_promotion_source()
        promoter.preview(
            self.report_path,
            orientation_credential=credential,
            execution_result=result,
        )
        copied_report = self.trial_dir / "copied_acceptance_report.json"
        copied_report.write_bytes(self.report_path.read_bytes())
        before = self.registry_path.read_bytes()

        with self.assertRaisesRegex(CapabilityAcceptanceError, "live-trial"):
            promoter.preview(
                copied_report,
                orientation_credential=credential,
                execution_result=result,
            )

        self.assertEqual(self.registry_path.read_bytes(), before)

    def test_scope_mismatch_consumes_authority_without_changing_registry(self) -> None:
        promoter = CapabilityRegistryPromoter(self.registry_path)
        authority = self._preview_live(promoter)
        scope = authority.scope
        before = self.registry_path.read_bytes()
        bad_confirmation = scope.to_dict()
        bad_confirmation["report_sha256"] = "0" * 64

        with self.assertRaisesRegex(CapabilityAcceptanceError, "确认范围"):
            promoter.promote(
                self.report_path,
                confirmation=bad_confirmation,
                authority=authority,
            )

        self.assertTrue(authority.consumed)
        self.assertEqual(self.registry_path.read_bytes(), before)

    def test_changed_registry_after_preview_consumes_authority_and_fails(self) -> None:
        promoter = CapabilityRegistryPromoter(self.registry_path)
        authority = self._preview_live(promoter)
        scope = authority.scope
        self.registry_path.write_bytes(self.registry_path.read_bytes() + b" ")

        with self.assertRaisesRegex(CapabilityAcceptanceError, "注册表.*变化"):
            promoter.promote(
                self.report_path,
                confirmation=scope.to_dict(),
                authority=authority,
            )

        self.assertTrue(authority.consumed)

    def test_changed_report_after_preview_consumes_authority_and_fails(self) -> None:
        promoter = CapabilityRegistryPromoter(self.registry_path)
        authority = self._preview_live(promoter)
        scope = authority.scope
        before = self.registry_path.read_bytes()
        self._mutate_report(
            lambda report: report.__setitem__("code_revision", "changed-revision")
        )

        with self.assertRaisesRegex(CapabilityAcceptanceError, "报告.*变化"):
            promoter.promote(
                self.report_path,
                confirmation=scope.to_dict(),
                authority=authority,
            )

        self.assertTrue(authority.consumed)
        self.assertIsNone(authority._orientation_credential)
        self.assertIsNone(authority._execution_result)
        self.assertEqual(self.registry_path.read_bytes(), before)

    def test_registry_lease_contention_consumes_authority_without_writing(self) -> None:
        lease_path = self.root / "promotion.lease"
        promoter = CapabilityRegistryPromoter(
            self.registry_path,
            lease_path=lease_path,
        )
        authority = self._preview_live(promoter)
        scope = authority.scope
        before = self.registry_path.read_bytes()
        held = InterProcessLease(
            lease_path,
            owner_id="other-process",
            metadata={"kind": "test"},
        )
        self.assertTrue(held.acquire())
        try:
            with self.assertRaisesRegex(CapabilityAcceptanceError, "锁.*占用"):
                promoter.promote(
                    self.report_path,
                    confirmation=scope.to_dict(),
                    authority=authority,
                )
        finally:
            held.release()

        self.assertTrue(authority.consumed)
        self.assertEqual(self.registry_path.read_bytes(), before)

    def test_atomic_replace_failure_keeps_registry_unchanged(self) -> None:
        def fail_replace(_source: Path, _target: Path) -> None:
            raise OSError("simulated replace failure")

        promoter = CapabilityRegistryPromoter(
            self.registry_path,
            replace_file=fail_replace,
        )
        authority = self._preview_live(promoter)
        scope = authority.scope
        before = self.registry_path.read_bytes()

        with self.assertRaisesRegex(CapabilityAcceptanceError, "原子替换失败"):
            promoter.promote(
                self.report_path,
                confirmation=scope.to_dict(),
                authority=authority,
            )

        self.assertTrue(authority.consumed)
        self.assertEqual(self.registry_path.read_bytes(), before)
        self.assertFalse((self.trial_dir / "registry_before.json").exists())
        self.assertFalse((self.trial_dir / "promotion.json").exists())
        self.assertEqual(list(self.root.glob(".device_registry.json.*.tmp")), [])

    def test_successful_promotion_adds_one_action_and_is_not_replayable(self) -> None:
        promoter = CapabilityRegistryPromoter(self.registry_path)
        authority = self._preview_live(promoter)
        scope = authority.scope

        result = promoter.promote(
            self.report_path,
            confirmation=scope.to_dict(),
            authority=authority,
        )

        payload = json.loads(self.registry_path.read_text(encoding="utf-8"))
        self.assertEqual(
            payload["devices"][0]["verified_actions"],
            ["back", "drag", "swipe", "tap_semantic"],
        )
        self.assertTrue(result["requires_restart"])
        self.assertEqual(result["action"], "drag")
        self.assertTrue((self.trial_dir / "registry_before.json").is_file())
        self.assertTrue((self.trial_dir / "promotion.json").is_file())
        promoted_registry = self.registry_path.read_bytes()

        with self.assertRaisesRegex(CapabilityAcceptanceError, "已使用"):
            promoter.promote(
                self.report_path,
                confirmation=scope.to_dict(),
                authority=authority,
            )
        self.assertEqual(self.registry_path.read_bytes(), promoted_registry)
        self.assertIsNone(authority._orientation_credential)
        self.assertIsNone(authority._execution_result)

    def test_concurrent_promote_has_one_writer_and_no_early_release(self) -> None:
        replace_entered = threading.Event()
        allow_replace = threading.Event()
        replace_calls = 0

        def blocking_replace(source: Path, target: Path) -> None:
            nonlocal replace_calls
            replace_calls += 1
            replace_entered.set()
            if not allow_replace.wait(timeout=5):
                raise OSError("timed out waiting to replace registry")
            os.replace(source, target)

        promoter = CapabilityRegistryPromoter(
            self.registry_path,
            replace_file=blocking_replace,
        )
        authority = self._preview_live(promoter)
        confirmation = authority.scope.to_dict()
        outcomes: list[tuple[str, object]] = []
        outcomes_lock = threading.Lock()

        def promote_once() -> None:
            try:
                outcome = (
                    "success",
                    promoter.promote(
                        self.report_path,
                        confirmation=confirmation,
                        authority=authority,
                    ),
                )
            except Exception as exc:
                outcome = ("failure", exc)
            with outcomes_lock:
                outcomes.append(outcome)

        first = threading.Thread(target=promote_once)
        second = threading.Thread(target=promote_once)
        first.start()
        self.assertTrue(replace_entered.wait(timeout=5))
        second.start()
        second.join(timeout=2)
        second_finished_before_release = not second.is_alive()
        allow_replace.set()
        first.join(timeout=5)
        second.join(timeout=5)

        self.assertTrue(second_finished_before_release)
        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(1, replace_calls)
        self.assertEqual(1, sum(kind == "success" for kind, _ in outcomes))
        failures = [value for kind, value in outcomes if kind == "failure"]
        self.assertEqual(1, len(failures))
        self.assertIsInstance(failures[0], CapabilityAcceptanceError)
        self.assertRegex(str(failures[0]), "正在使用")
        payload = json.loads(self.registry_path.read_text(encoding="utf-8"))
        self.assertEqual(1, payload["devices"][0]["verified_actions"].count("drag"))
        self.assertTrue(authority.consumed)
        self.assertIsNone(authority._orientation_credential)
        self.assertIsNone(authority._execution_result)

    def test_already_enabled_action_cannot_be_promoted(self) -> None:
        self._mutate_report(
            lambda report: (
                report.__setitem__("candidate_action", "back"),
                report.__setitem__("calibration_evidence", None),
                report["execution"]["resolved_action"].__setitem__("kind", "back"),
                report["execution"]["after_scene"].__setitem__("screen_id", "other"),
            )
        )

        with self.assertRaisesRegex(CapabilityAcceptanceError, "已经启用"):
            promoter = CapabilityRegistryPromoter(self.registry_path)
            credential, result = self._live_promotion_source()
            promoter.preview(
                self.report_path,
                orientation_credential=credential,
                execution_result=result,
            )


if __name__ == "__main__":
    unittest.main()
