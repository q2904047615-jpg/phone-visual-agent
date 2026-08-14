import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from capability_acceptance import (
    CapabilityAcceptanceError,
    CapabilityRegistryPromoter,
    PromotionAuthority,
    validate_acceptance_report,
)
from device_exclusivity import InterProcessLease


class CapabilityAcceptanceCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.trial_dir = self.root / "trial-001"
        self.trial_dir.mkdir()
        self.registry_path = self.root / "device_registry.json"
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
            "version": 1,
            "trial_id": "trial-001",
            "session_id": "session-001",
            "task_id": "task-001",
            "device_id": "device-a",
            "candidate_action": "drag",
            "status": "passed",
            "code_revision": "86b63d8",
            "physical_actions": 1,
            "action_outcome": "matched",
            "before_observation": {
                "observation_id": "obs-before",
                "fingerprint": "fingerprint-before",
            },
            "after_observation": {
                "observation_id": "obs-after",
                "fingerprint": "fingerprint-after",
            },
            "execution": {
                "resolved_action": {"kind": "drag"},
                "observation_errors": [],
                "verification_errors": [],
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

    def _mutate_report(self, mutation) -> None:
        report = json.loads(self.report_path.read_text(encoding="utf-8"))
        mutation(report)
        self.report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def test_valid_report_requires_one_matched_action_and_eight_trial_frames(self) -> None:
        report = validate_acceptance_report(self.report_path)

        self.assertEqual(report["trial_id"], "trial-001")
        self.assertEqual(report["candidate_action"], "drag")

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
        report["execution"].update(
            {
                "resolved_action": {
                    "kind": "input_verified_text",
                    "text": "agent",
                    "target_element_id": "field",
                },
                "before_scene": {
                    "elements": [
                        {
                            "element_id": "field",
                            "role": "input",
                            "meaning": "search_field",
                            "label": "搜索",
                            "confidence": 0.95,
                            "states": {"value": ""},
                        }
                    ]
                },
                "after_scene": {
                    "elements": [
                        {
                            "element_id": "field-after",
                            "role": "input",
                            "meaning": "search_field",
                            "label": "搜索",
                            "confidence": 0.95,
                            "states": {"value": "agent.com"},
                        }
                    ]
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

    def test_report_rejects_uncommitted_code_revision(self) -> None:
        self._mutate_report(
            lambda report: report.__setitem__("code_revision", "86b63d8+dirty")
        )

        with self.assertRaisesRegex(CapabilityAcceptanceError, "未提交代码"):
            validate_acceptance_report(self.report_path)

    def test_report_rejects_unchanged_observation_or_fingerprint(self) -> None:
        mutations = (
            lambda report: report["after_observation"].__setitem__("observation_id", "obs-before"),
            lambda report: report["after_observation"].__setitem__("fingerprint", "fingerprint-before"),
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
        scope = promoter.preview(self.report_path)

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

    def test_scope_mismatch_consumes_authority_without_changing_registry(self) -> None:
        promoter = CapabilityRegistryPromoter(self.registry_path)
        scope = promoter.preview(self.report_path)
        authority = PromotionAuthority(scope)
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
        scope = promoter.preview(self.report_path)
        authority = PromotionAuthority(scope)
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
        scope = promoter.preview(self.report_path)
        authority = PromotionAuthority(scope)
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

    def test_registry_lease_contention_consumes_authority_without_writing(self) -> None:
        lease_path = self.root / "promotion.lease"
        promoter = CapabilityRegistryPromoter(
            self.registry_path,
            lease_path=lease_path,
        )
        scope = promoter.preview(self.report_path)
        authority = PromotionAuthority(scope)
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
        scope = promoter.preview(self.report_path)
        authority = PromotionAuthority(scope)
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
        scope = promoter.preview(self.report_path)
        authority = PromotionAuthority(scope)

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

        with self.assertRaisesRegex(CapabilityAcceptanceError, "已使用"):
            promoter.promote(
                self.report_path,
                confirmation=scope.to_dict(),
                authority=authority,
            )

    def test_already_enabled_action_cannot_be_promoted(self) -> None:
        self._mutate_report(
            lambda report: (
                report.__setitem__("candidate_action", "back"),
                report["execution"]["resolved_action"].__setitem__("kind", "back"),
            )
        )

        with self.assertRaisesRegex(CapabilityAcceptanceError, "已经启用"):
            CapabilityRegistryPromoter(self.registry_path).preview(self.report_path)


if __name__ == "__main__":
    unittest.main()
