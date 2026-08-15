from __future__ import annotations

import json
import re
import unittest

from PIL import Image

try:
    from element_geometry_audit import (
        ELEMENT_GEOMETRY_AUDIT_PROTOCOL_VERSION,
        CropTransform,
        ElementGeometryAuditError,
        build_candidate_crop_transform,
        element_geometry_audit_prompt,
        parse_element_geometry_audit,
        select_unique_audited_geometry,
    )
except ModuleNotFoundError:  # Support ``python -m unittest poc/test_...py``.
    from poc.element_geometry_audit import (
        ELEMENT_GEOMETRY_AUDIT_PROTOCOL_VERSION,
        CropTransform,
        ElementGeometryAuditError,
        build_candidate_crop_transform,
        element_geometry_audit_prompt,
        parse_element_geometry_audit,
        select_unique_audited_geometry,
    )

try:
    from generic_scene_observer import GenericSceneObserver, _local_frame_fingerprint
    from ui_scene import UIElement, UIScene
except ModuleNotFoundError:
    from poc.generic_scene_observer import GenericSceneObserver, _local_frame_fingerprint
    from poc.ui_scene import UIElement, UIScene


SOURCE_REF = "source-ee017-geometry"


def audit_payload(**overrides):
    match = {
        "match_id": "match-1",
        "literal_label": "返回验收模式选择",
        "visual_role": "button",
        "bounds": [120, 100, 450, 170],
        "confidence": 0.97,
        "fully_visible": True,
        "whole_control": True,
        "evidence": ["逐字可见返回验收模式选择"],
    }
    match.update(overrides.pop("match", {}))
    payload = {
        "protocol_version": ELEMENT_GEOMETRY_AUDIT_PROTOCOL_VERSION,
        "source_ref": SOURCE_REF,
        "crop_clear": True,
        "enumeration_complete": True,
        "matches": [match],
    }
    payload.update(overrides)
    return payload


class CropTransformTests(unittest.TestCase):
    def test_broad_roi_contains_target_far_from_wrong_rough_center(self):
        transform = build_candidate_crop_transform(
            (810, 1515),
            (0.12, 0.46, 0.58, 0.51),
        )
        left, top, right, bottom = transform.pixel_bounds
        actual = (0.12, 0.225, 0.45, 0.265)
        self.assertLessEqual(left / 810, actual[0])
        self.assertLessEqual(top / 1515, actual[1])
        self.assertGreaterEqual(right / 810, actual[2])
        self.assertGreaterEqual(bottom / 1515, actual[3])
        self.assertGreaterEqual(right - left, round(810 * 0.60) - 1)
        self.assertGreaterEqual(bottom - top, round(1515 * 0.60) - 1)

    def test_roi_clamps_near_full_frame_edge(self):
        transform = build_candidate_crop_transform(
            (810, 1515),
            (0.01, 0.01, 0.08, 0.06),
        )
        self.assertEqual(0, transform.pixel_bounds[0])
        self.assertEqual(0, transform.pixel_bounds[1])
        self.assertGreater(transform.pixel_bounds[2], 0)
        self.assertGreater(transform.pixel_bounds[3], 0)

    def test_large_target_expands_to_full_width(self):
        transform = build_candidate_crop_transform(
            (1000, 1600),
            (0.10, 0.30, 0.90, 0.70),
        )
        self.assertEqual((0, 1000), (transform.pixel_bounds[0], transform.pixel_bounds[2]))
        self.assertGreaterEqual(transform.crop_size[1], round(1600 * 0.60) - 1)

    def test_crop_uses_the_exact_pixel_rectangle(self):
        image = Image.new("RGB", (810, 1515), "white")
        transform = CropTransform((810, 1515), (81, 303, 648, 1212))
        crop = transform.crop(image)
        self.assertEqual((567, 909), crop.size)
        with self.assertRaisesRegex(ElementGeometryAuditError, "尺寸"):
            transform.crop(Image.new("RGB", (811, 1515), "white"))

    def test_exact_pixel_mapping_handles_non_divisible_frame_dimensions(self):
        transform = CropTransform((810, 1515), (81, 303, 648, 1212))
        mapped = transform.map_bounds_to_full((100, 200, 900, 800))
        expected = (0.17, 0.32, 0.73, 0.68)
        for actual, wanted in zip(mapped, expected):
            self.assertAlmostEqual(wanted, actual, places=12)

    def test_invalid_rough_or_local_bounds_never_map(self):
        with self.assertRaisesRegex(ElementGeometryAuditError, "粗候选"):
            build_candidate_crop_transform((810, 1515), (0.2, 0.4, 1.2, 0.5))
        transform = CropTransform((810, 1515), (81, 303, 648, 1212))
        with self.assertRaisesRegex(ElementGeometryAuditError, "crop-local"):
            transform.map_bounds_to_full((0, 0, 1200, 100))


class StrictGeometryAuditProtocolTests(unittest.TestCase):
    def setUp(self):
        self.transform = CropTransform((810, 1515), (80, 200, 730, 1100))

    def render(self, payload=None):
        return json.dumps(payload or audit_payload(), ensure_ascii=False, separators=(",", ":"))

    def select(self, raw=None, **kwargs):
        return select_unique_audited_geometry(
            raw or self.render(),
            expected_source_ref=kwargs.pop("expected_source_ref", SOURCE_REF),
            expected_label=kwargs.pop("expected_label", "返回验收模式选择"),
            expected_role=kwargs.pop("expected_role", "button"),
            transform=kwargs.pop("transform", self.transform),
            **kwargs,
        )

    def test_valid_unique_match_maps_and_preserves_literal_identity(self):
        result = self.select()
        self.assertEqual("返回验收模式选择", result.literal_label)
        self.assertEqual("button", result.visual_role)
        self.assertEqual((120.0, 100.0, 450.0, 170.0), result.local_bounds)
        self.assertEqual(SOURCE_REF, result.source_ref)
        self.assertEqual(
            self.transform.map_bounds_to_full(result.local_bounds),
            result.full_bounds,
        )

    def test_duplicate_json_keys_are_rejected_at_top_and_nested_levels(self):
        top = self.render()[:-1] + ',"matches":[]}'
        with self.assertRaisesRegex(ElementGeometryAuditError, "重复JSON字段"):
            parse_element_geometry_audit(top)
        nested = self.render().replace(
            '"confidence":0.97',
            '"confidence":0.97,"confidence":0.98',
        )
        with self.assertRaisesRegex(ElementGeometryAuditError, "重复JSON字段"):
            parse_element_geometry_audit(nested)

    def test_extra_top_or_match_fields_are_rejected(self):
        top = audit_payload(debug=True)
        with self.assertRaisesRegex(ElementGeometryAuditError, "顶层字段"):
            parse_element_geometry_audit(self.render(top))
        nested = audit_payload(match={"action": "tap"})
        with self.assertRaisesRegex(ElementGeometryAuditError, "match 字段"):
            parse_element_geometry_audit(self.render(nested))

    def test_markdown_or_non_object_response_is_rejected(self):
        with self.assertRaisesRegex(ElementGeometryAuditError, "JSON对象"):
            parse_element_geometry_audit("```json\n{}\n```")
        with self.assertRaisesRegex(ElementGeometryAuditError, "顶层"):
            parse_element_geometry_audit("[]")

    def test_zero_or_duplicate_matches_have_no_rough_fallback(self):
        with self.assertRaisesRegex(ElementGeometryAuditError, "唯一"):
            self.select(self.render(audit_payload(matches=[])))
        duplicate = audit_payload()["matches"] * 2
        duplicate[1] = dict(duplicate[1], match_id="match-2")
        with self.assertRaisesRegex(ElementGeometryAuditError, "唯一"):
            self.select(self.render(audit_payload(matches=duplicate)))

    def test_source_label_and_role_must_match_exactly(self):
        with self.assertRaisesRegex(ElementGeometryAuditError, "source_ref"):
            self.select(expected_source_ref="source-another-geometry")
        with self.assertRaisesRegex(ElementGeometryAuditError, r"label\+role"):
            self.select(expected_label="返回")
        with self.assertRaisesRegex(ElementGeometryAuditError, r"label\+role"):
            self.select(expected_role="icon")

    def test_confidence_visibility_and_whole_control_are_hard_gates(self):
        cases = (
            ({"confidence": 0.919}, "confidence"),
            ({"fully_visible": False}, "完整可见"),
            ({"whole_control": False}, "完整可见"),
        )
        for changes, error in cases:
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(ElementGeometryAuditError, error):
                    self.select(self.render(audit_payload(match=changes)))

    def test_crop_and_enumeration_must_be_complete(self):
        for field in ("crop_clear", "enumeration_complete"):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ElementGeometryAuditError, "不清晰|不完整"):
                    self.select(self.render(audit_payload(**{field: False})))

    def test_internal_crop_edges_are_rejected(self):
        cases = (
            [10, 100, 300, 200],
            [100, 10, 300, 200],
            [100, 100, 990, 200],
            [100, 100, 300, 990],
        )
        for bounds in cases:
            with self.subTest(bounds=bounds):
                with self.assertRaisesRegex(ElementGeometryAuditError, "内部 crop 边缘"):
                    self.select(self.render(audit_payload(match={"bounds": bounds})))

    def test_full_frame_edge_is_not_misclassified_as_internal_crop_edge(self):
        transform = CropTransform((810, 1515), (0, 0, 730, 1100))
        result = self.select(
            self.render(audit_payload(match={"bounds": [0, 0, 300, 200]})),
            transform=transform,
        )
        self.assertEqual(0.0, result.full_bounds[0])
        self.assertEqual(0.0, result.full_bounds[1])

    def test_protocol_rejects_invalid_bounds_and_control_evidence(self):
        with self.assertRaisesRegex(ElementGeometryAuditError, "crop-local"):
            parse_element_geometry_audit(
                self.render(audit_payload(match={"bounds": [0, 0, 1200, 100]}))
            )
        with self.assertRaisesRegex(ElementGeometryAuditError, "控制信息"):
            parse_element_geometry_audit(
                self.render(audit_payload(match={"evidence": ["点击这个按钮"]}))
            )

    def test_prompt_exposes_one_crop_local_coordinate_contract_only(self):
        prompt = element_geometry_audit_prompt(
            source_ref=SOURCE_REF,
            literal_label="返回验收模式选择",
            visual_role="button",
            visible_evidence="逐字可见返回验收模式选择",
        )
        self.assertIn("exactly one image", prompt)
        self.assertIn("only coordinate space", prompt)
        self.assertIn("left/top is 0 and right/bottom is 1000", prompt)
        self.assertIn("Never infer or return full-frame coordinates", prompt)
        self.assertNotIn("0.12", prompt)
        self.assertNotIn("0.46", prompt)


class GeometryAuditProvider:
    configured = True

    def __init__(self, local_bounds):
        self.local_bounds = list(local_bounds)
        self.messages = []

    def _chat(self, messages, *, max_tokens, **_kwargs):
        self.messages.append(messages)
        prompt = messages[-1]["content"][0]["text"]
        source_ref = re.search(r"^source_ref=(.+)$", prompt, re.MULTILINE).group(1)
        label = json.loads(
            re.search(r"^literal_label=(.+)$", prompt, re.MULTILINE).group(1)
        )
        role = re.search(r"^visual_role=(.+)$", prompt, re.MULTILINE).group(1)
        bounds = self.local_bounds.pop(0)
        matches = []
        if bounds is not None:
            matches.append(
                {
                    "match_id": "match-1",
                    "literal_label": label,
                    "visual_role": role,
                    "bounds": bounds,
                    "confidence": 0.98,
                    "fully_visible": True,
                    "whole_control": True,
                    "evidence": ["裁剪画面中逐字标签和完整区域清晰可见"],
                }
            )
        return json.dumps(
            {
                "protocol_version": ELEMENT_GEOMETRY_AUDIT_PROTOCOL_VERSION,
                "source_ref": source_ref,
                "crop_clear": True,
                "enumeration_complete": True,
                "matches": matches,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def status(self):
        return {"configured": True, "model": "offline-geometry"}


class GenericSceneGeometryAuditIntegrationTests(unittest.TestCase):
    @staticmethod
    def scene(frame):
        return UIScene(
            app_id="local.acceptance",
            screen_id="drag-board",
            summary="本地动作验收页面",
            elements=(
                UIElement(
                    element_id="source",
                    role="image",
                    meaning="draggable_purple_block",
                    label="起点",
                    bounds=(0.12, 0.46, 0.58, 0.51),
                    confidence=0.98,
                    evidence=("紫色圆角方块，中心写有起点二字",),
                ),
                UIElement(
                    element_id="destination",
                    role="container",
                    meaning="drop_target_green_zone",
                    label="绿色终点",
                    bounds=(0.55, 0.63, 0.85, 0.83),
                    confidence=0.98,
                    evidence=("绿色虚线框区域，内部写有绿色终点",),
                ),
            ),
            stable=True,
            confidence=0.98,
            fingerprint=_local_frame_fingerprint(frame),
        )

    def test_two_endpoints_use_two_independent_single_crop_calls(self):
        frame = Image.new("RGB", (810, 1515), "gray")
        current = self.scene(frame)
        local_bounds = ([120, 100, 450, 170], [300, 250, 800, 650])
        provider = GeometryAuditProvider(local_bounds)
        observer = GenericSceneObserver(provider)

        audited = observer.audit_element_geometry(
            frames=[frame.copy() for _ in range(4)],
            scene=current,
            element_ids=("source", "destination"),
        )

        self.assertEqual(2, len(provider.messages))
        for messages in provider.messages:
            content = messages[-1]["content"]
            self.assertEqual(2, len(content))
            self.assertEqual("text", content[0]["type"])
            self.assertEqual("image_url", content[1]["type"])
        for element_id, raw_local in zip(
            ("source", "destination"),
            local_bounds,
        ):
            original = current.get_element(element_id)
            transform = build_candidate_crop_transform(frame.size, original.bounds)
            expected = transform.map_bounds_to_full(tuple(raw_local))
            actual = audited.get_element(element_id).bounds
            for expected_part, actual_part in zip(expected, actual):
                self.assertAlmostEqual(expected_part, actual_part, places=8)
        self.assertEqual(current.elements[0].meaning, audited.elements[0].meaning)
        self.assertEqual(current.elements[1].states, audited.elements[1].states)

    def test_zero_match_never_returns_the_rough_scene(self):
        frame = Image.new("RGB", (810, 1515), "gray")
        current = self.scene(frame)
        observer = GenericSceneObserver(GeometryAuditProvider([None]))

        with self.assertRaisesRegex(ElementGeometryAuditError, "唯一"):
            observer.audit_element_geometry(
                frames=[frame.copy() for _ in range(4)],
                scene=current,
                element_ids=("source",),
            )


if __name__ == "__main__":
    unittest.main()
