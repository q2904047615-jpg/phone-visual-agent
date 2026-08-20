from __future__ import annotations

import json
import re
import unittest

from PIL import Image, ImageDraw

try:
    from element_geometry_audit import (
        ELEMENT_GEOMETRY_AUDIT_PROTOCOL_VERSION,
        CropTransform,
        ElementGeometryAuditError,
        build_candidate_crop_transform,
        build_literal_candidate_crop_transform,
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
        build_literal_candidate_crop_transform,
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

    def test_literal_selector_roi_is_wide_but_excludes_adjacent_rows(self):
        transform = build_literal_candidate_crop_transform(
            (810, 1515),
            (0.13, 0.225, 0.56, 0.265),
        )
        left, top, right, bottom = transform.pixel_bounds
        self.assertGreaterEqual(right - left, round(810 * 0.60) - 1)
        self.assertEqual(round(1515 * 0.20), bottom - top)
        self.assertLessEqual(top / 1515, 0.225)
        self.assertGreaterEqual(bottom / 1515, 0.265)
        self.assertLess(bottom / 1515, 0.40)

    def test_literal_selector_roi_is_resolution_independent(self):
        for full_size in ((540, 960), (810, 1515), (1080, 2400)):
            with self.subTest(full_size=full_size):
                transform = build_literal_candidate_crop_transform(
                    full_size,
                    (0.10, 0.42, 0.38, 0.47),
                )
                width, height = full_size
                self.assertGreaterEqual(
                    transform.crop_size[0], round(width * 0.60) - 1
                )
                self.assertGreaterEqual(
                    transform.crop_size[1], round(height * 0.20) - 1
                )
                self.assertLessEqual(
                    transform.crop_size[1], round(height * 0.20) + 1
                )

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

    def test_one_to_four_safe_evidence_items_are_bounded_and_accepted(self):
        facts = [
            "绿色文件夹图标完整可见",
            "文件传输助手标题逐字可见",
            "副标题位于同一完整列表项内",
            "列表项左右边缘均完整可见",
        ]
        for count in (1, 2, 3, 4):
            with self.subTest(count=count):
                result = self.select(
                    self.render(audit_payload(match={"evidence": facts[:count]}))
                )
                self.assertEqual(tuple(facts[:count]), result.evidence)

    def test_geometry_evidence_count_and_contents_remain_fail_closed(self):
        cases = (
            ([], "格式无效"),
            (["安全可见事实"] * 5, "格式无效"),
            (["安全可见事实", "点击该列表项"], "控制信息"),
            ([123], "格式无效"),
        )
        for evidence, error in cases:
            with self.subTest(evidence=evidence):
                with self.assertRaisesRegex(ElementGeometryAuditError, error):
                    self.select(
                        self.render(audit_payload(match={"evidence": evidence}))
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

    def test_protocol_and_prompt_allow_literal_text_link_role(self):
        payload = audit_payload(
            match={
                "visual_role": "text",
                "literal_label": "返回验收模式选择",
                "evidence": ["白色下划线文字，位于说明文字下方"],
            }
        )
        parsed = parse_element_geometry_audit(self.render(payload))
        self.assertEqual("text", parsed.matches[0].visual_role)

        prompt = element_geometry_audit_prompt(
            source_ref=SOURCE_REF,
            literal_label="返回验收模式选择",
            visual_role="text",
            visible_evidence="白色下划线文字，位于说明文字下方",
        )
        self.assertIn("visual_role=text", prompt)

        payload["matches"][0]["visual_role"] = "unknown_role"
        with self.assertRaisesRegex(ElementGeometryAuditError, "visual_role"):
            parse_element_geometry_audit(self.render(payload))

    def test_protocol_allows_shape_bound_unlabelled_input_icon_and_button(self):
        payload = audit_payload(
            match={
                "visual_role": "input",
                "literal_label": "",
                "evidence": ["空输入框四边完整可见"],
            }
        )
        parsed = parse_element_geometry_audit(self.render(payload))
        self.assertEqual("", parsed.matches[0].literal_label)
        prompt = element_geometry_audit_prompt(
            source_ref=SOURCE_REF,
            literal_label="",
            visual_role="input",
            visible_evidence="空输入框四边完整可见",
        )
        self.assertIn(
            "literal_label is empty and visual_role is input, icon, or button",
            prompt,
        )

        payload["matches"][0]["visual_role"] = "button"
        payload["matches"][0]["evidence"] = ["无文字圆形按钮轮廓完整可见"]
        parsed_button = parse_element_geometry_audit(self.render(payload))
        self.assertEqual("button", parsed_button.matches[0].visual_role)
        button_prompt = element_geometry_audit_prompt(
            source_ref=SOURCE_REF,
            literal_label="",
            visual_role="button",
            visible_evidence="无文字圆形按钮轮廓完整可见",
        )
        self.assertIn("visual_role=button", button_prompt)

        payload["matches"][0]["visual_role"] = "text"
        with self.assertRaisesRegex(ElementGeometryAuditError, "literal_label"):
            parse_element_geometry_audit(self.render(payload))

    def test_protocol_allows_non_numeric_boundary_fact_but_rejects_coordinates(self):
        payload = audit_payload(
            match={
                "evidence": [
                    "Text is clearly legible within the button bounds."
                ]
            }
        )
        parsed = parse_element_geometry_audit(self.render(payload))
        self.assertEqual(1, len(parsed.matches))

        payload["matches"][0]["evidence"] = ["bounds=[1,2,3,4]"]
        with self.assertRaisesRegex(ElementGeometryAuditError, "控制信息"):
            parse_element_geometry_audit(self.render(payload))

    def test_protocol_allows_bounded_descriptive_evidence_but_rejects_overlong_text(self):
        descriptive = (
            "Underlined white text located below the description paragraph and "
            "above the '等待动作' button, visually distinct as a clickable "
            "navigation link."
        )
        self.assertEqual(140, len(descriptive))
        payload = audit_payload(match={"evidence": [descriptive]})
        parsed = parse_element_geometry_audit(
            self.render(payload),
            visible_literal_labels=("等待动作",),
        )
        self.assertEqual((descriptive,), parsed.matches[0].evidence)

        payload["matches"][0]["evidence"] = ["a" * 201]
        with self.assertRaisesRegex(ElementGeometryAuditError, "过长"):
            parse_element_geometry_audit(self.render(payload))

    def test_protocol_allows_control_word_inside_exact_literal_label_only(self):
        payload = audit_payload(
            match={
                "literal_label": "语义点击",
                "evidence": ["文字清晰可辨为“语义点击”，四边完整可见"],
            }
        )
        parsed = parse_element_geometry_audit(self.render(payload))
        self.assertEqual("语义点击", parsed.matches[0].literal_label)

        payload["matches"][0]["evidence"] = ["请点击文字清晰可辨的“语义点击”"]
        with self.assertRaisesRegex(ElementGeometryAuditError, "控制信息"):
            parse_element_geometry_audit(self.render(payload))

    def test_protocol_allows_exact_sibling_label_but_not_extra_control_text(self):
        payload = audit_payload(
            match={
                "literal_label": "语义点击",
                "evidence": [
                    "位于‘向上滑动’正下方，文字‘语义点击’清晰可见"
                ],
            }
        )
        parsed = parse_element_geometry_audit(
            self.render(payload),
            visible_literal_labels=("向上滑动", "语义点击"),
        )
        self.assertEqual("语义点击", parsed.matches[0].literal_label)

        payload["matches"][0]["evidence"] = [
            "请滑动到向上滑动后点击语义点击"
        ]
        with self.assertRaisesRegex(ElementGeometryAuditError, "控制信息"):
            parse_element_geometry_audit(
                self.render(payload),
                visible_literal_labels=("向上滑动", "语义点击"),
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

    def test_prompt_allows_exact_action_word_label_but_rejects_extra_instruction(self):
        prompt = element_geometry_audit_prompt(
            source_ref=SOURCE_REF,
            literal_label="语义点击",
            visual_role="list_item",
            visible_evidence="文字清晰可辨为“语义点击”，四边完整可见",
        )
        self.assertIn('literal_label="语义点击"', prompt)

        with self.assertRaisesRegex(ElementGeometryAuditError, "visible_evidence"):
            element_geometry_audit_prompt(
                source_ref=SOURCE_REF,
                literal_label="语义点击",
                visual_role="list_item",
                visible_evidence="请点击文字清晰可辨的“语义点击”",
            )

    def test_prompt_allows_other_exact_scene_label_but_not_control_text(self):
        prompt = element_geometry_audit_prompt(
            source_ref=SOURCE_REF,
            literal_label="语义点击",
            visual_role="list_item",
            visible_evidence="列表第二项紧接‘向上滑动’下方，文字清晰可见",
            visible_literal_labels=("向上滑动", "语义点击"),
        )
        self.assertIn('visible_literal_labels=["语义点击", "向上滑动"]', prompt)

        with self.assertRaisesRegex(ElementGeometryAuditError, "visible_evidence"):
            element_geometry_audit_prompt(
                source_ref=SOURCE_REF,
                literal_label="语义点击",
                visual_role="list_item",
                visible_evidence="请滑动到向上滑动后点击语义点击",
                visible_literal_labels=("向上滑动", "语义点击"),
            )


class GeometryAuditProvider:
    configured = True

    def __init__(self, local_bounds):
        self.local_bounds = list(local_bounds)
        self.messages = []
        self.kwargs = []

    def _chat(self, messages, *, max_tokens, **kwargs):
        self.messages.append(messages)
        self.kwargs.append(kwargs)
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
        self.assertEqual(
            [{"type": "json_object"}, {"type": "json_object"}],
            [item["response_format"] for item in provider.kwargs],
        )
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
        self.assertEqual(current.elements[0].meaning, audited.elements[0].meaning)
        for element in audited.elements:
            self.assertTrue(element.states["fully_visible"])
            self.assertTrue(element.states["independent_geometry_verified"])
            self.assertEqual(
                "element_geometry_audit",
                element.states["geometry_audit_source"],
            )

    def test_dense_scene_exposes_only_labels_cited_by_target_evidence(self):
        frame = Image.new("RGB", (1000, 1600), "gray")
        labels = (
            "设置",
            "浏览器",
            "音乐",
            "小爱视频",
            "手电筒",
            "手机管家",
            "应用商店",
            "支付宝",
            "微信",
            "相机",
        )
        elements = tuple(
            UIElement(
                element_id=f"e{index + 1}",
                role="button",
                meaning=f"open_app_{index + 1}",
                label=label,
                bounds=(
                    0.05 + (index % 2) * 0.45,
                    0.10 + (index // 2) * 0.15,
                    0.35 + (index % 2) * 0.45,
                    0.18 + (index // 2) * 0.15,
                ),
                confidence=0.98,
                evidence=((
                    "灰色齿轮图标，下方文字‘设置’"
                    if index == 0
                    else f"图标下方文字‘{label}’"
                ),),
            )
            for index, label in enumerate(labels)
        )
        scene = UIScene(
            app_id="launcher",
            screen_id="dense-home",
            summary="多个带文字入口可见",
            elements=elements,
            stable=True,
            confidence=0.98,
            fingerprint=_local_frame_fingerprint(frame),
        )
        provider = GeometryAuditProvider([[300, 300, 700, 700]])

        GenericSceneObserver(provider).audit_element_geometry(
            frames=[frame.copy() for _ in range(4)],
            scene=scene,
            element_ids=("e1",),
        )

        prompt = provider.messages[0][-1]["content"][0]["text"]
        self.assertIn('visible_literal_labels=["设置"]', prompt)
        for unrelated in labels[1:]:
            self.assertNotIn(unrelated, prompt)

    def test_evidence_cited_sibling_label_is_preserved_without_page_wide_labels(self):
        frame = Image.new("RGB", (1000, 1600), "gray")
        scene = UIScene(
            app_id="unknown",
            screen_id="dense-list",
            summary="多个文字入口可见",
            elements=(
                UIElement(
                    element_id="target",
                    role="list_item",
                    meaning="semantic_tap_entry",
                    label="语义点击",
                    bounds=(0.10, 0.35, 0.55, 0.42),
                    confidence=0.98,
                    evidence=("位于‘向上滑动’正下方，文字‘语义点击’清晰可见",),
                ),
                UIElement(
                    element_id="sibling",
                    role="list_item",
                    meaning="swipe_up_entry",
                    label="向上滑动",
                    bounds=(0.10, 0.25, 0.55, 0.32),
                    confidence=0.98,
                    evidence=("文字‘向上滑动’清晰可见",),
                ),
                UIElement(
                    element_id="unrelated",
                    role="button",
                    meaning="other_entry",
                    label="执行建议",
                    bounds=(0.60, 0.60, 0.90, 0.68),
                    confidence=0.98,
                    evidence=("右下方文字入口完整可见",),
                ),
            ),
            stable=True,
            confidence=0.98,
            fingerprint=_local_frame_fingerprint(frame),
        )
        provider = GeometryAuditProvider([[250, 300, 750, 700]])

        GenericSceneObserver(provider).audit_element_geometry(
            frames=[frame.copy() for _ in range(4)],
            scene=scene,
            element_ids=("target",),
        )

        prompt = provider.messages[0][-1]["content"][0]["text"]
        self.assertIn(
            'visible_literal_labels=["语义点击", "向上滑动"]',
            prompt,
        )
        self.assertNotIn("执行建议", prompt)

    def test_evidence_with_more_than_eight_actual_labels_still_fails_closed(self):
        frame = Image.new("RGB", (1000, 1600), "gray")
        labels = tuple(f"入口{index}" for index in range(1, 10))
        evidence = "、".join(labels) + "均清晰可见"
        scene = UIScene(
            app_id="unknown",
            screen_id="overloaded-evidence",
            summary="密集入口可见",
            elements=tuple(
                UIElement(
                    element_id=f"e{index}",
                    role="button",
                    meaning=f"entry_{index}",
                    label=label,
                    bounds=(0.05, 0.05 + index * 0.08, 0.45, 0.10 + index * 0.08),
                    confidence=0.98,
                    evidence=(evidence,) if index == 1 else (f"{label}清晰可见",),
                )
                for index, label in enumerate(labels, start=1)
            ),
            stable=True,
            confidence=0.98,
            fingerprint=_local_frame_fingerprint(frame),
        )
        provider = GeometryAuditProvider([[250, 300, 750, 700]])

        with self.assertRaisesRegex(ElementGeometryAuditError, "缺少"):
            GenericSceneObserver(provider).audit_element_geometry(
                frames=[frame.copy() for _ in range(4)],
                scene=scene,
                element_ids=("e1",),
            )
        self.assertEqual([], provider.messages)

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

    def test_literal_button_uses_shallow_profile_without_changing_other_roles(self):
        frame = Image.new("RGB", (810, 1515), "gray")
        current = UIScene(
            app_id="unknown",
            screen_id="dense-list",
            summary="文字入口与相邻列表项可见",
            elements=(
                UIElement(
                    element_id="literal",
                    role="button",
                    meaning="return_to_selection",
                    label="返回验收模式选择",
                    bounds=(0.13, 0.225, 0.56, 0.265),
                    confidence=1.0,
                    states={"goal_relevant": True, "fully_visible": True},
                    evidence=("白色下划线文字完整可见",),
                ),
            ),
            stable=True,
            confidence=1.0,
            fingerprint=_local_frame_fingerprint(frame),
        )
        observer = GenericSceneObserver(GeometryAuditProvider([[180, 350, 700, 550]]))

        observer.audit_element_geometry(
            frames=[frame.copy() for _ in range(4)],
            scene=current,
            element_ids=("literal",),
        )

        diagnostics = observer.last_geometry_audit_diagnostics["audits"][0]
        self.assertEqual("literal_selector", diagnostics["crop_profile"])
        left, top, right, bottom = diagnostics["pixel_bounds"]
        self.assertGreaterEqual(right - left, round(810 * 0.60) - 1)
        self.assertEqual(round(1515 * 0.20), bottom - top)

    def test_unlabelled_input_still_requires_unique_single_crop_audit(self):
        frame = Image.new("RGB", (810, 1515), "gray")
        current = UIScene(
            app_id="local.acceptance",
            screen_id="input-page",
            summary="唯一空输入框可见",
            elements=(
                UIElement(
                    element_id="input",
                    role="input",
                    meaning="text_input_field",
                    label="",
                    bounds=(0.13, 0.51, 0.87, 0.60),
                    confidence=1.0,
                    states={"goal_relevant": True, "value": ""},
                    evidence=("空输入框四边完整可见",),
                ),
            ),
            stable=True,
            confidence=1.0,
            fingerprint=_local_frame_fingerprint(frame),
        )
        provider = GeometryAuditProvider([[120, 180, 880, 320]])
        audited = GenericSceneObserver(provider).audit_element_geometry(
            frames=[frame.copy() for _ in range(4)],
            scene=current,
            element_ids=("input",),
        )

        self.assertEqual(1, len(provider.messages))
        self.assertEqual("", audited.get_element("input").label)
        self.assertNotEqual(current.get_element("input").bounds, audited.get_element("input").bounds)

    def test_unlabelled_input_geometry_snaps_to_unique_local_border(self):
        frame = Image.new("RGB", (810, 1440), "black")
        draw = ImageDraw.Draw(frame)
        draw.rounded_rectangle(
            (89, 540, 725, 982), radius=30, outline=(100, 255, 255), width=5
        )
        draw.rounded_rectangle(
            (122, 717, 690, 847), radius=24, outline=(120, 255, 255), width=7
        )
        # A light keyboard-like lower half makes any crop-wide median
        # background ambiguous; border localization must be polarity-neutral.
        draw.rectangle((0, 900, 809, 1439), fill="white")
        current = UIScene(
            app_id="unknown",
            screen_id="input-page",
            summary="一个完整空输入框可见",
            elements=(
                UIElement(
                    element_id="input",
                    role="input",
                    meaning="text_input_field",
                    label="",
                    bounds=(0.13, 0.50, 0.87, 0.60),
                    confidence=1.0,
                    states={"goal_relevant": True, "value": ""},
                    evidence=("空输入框四边完整可见",),
                ),
            ),
            stable=True,
            confidence=1.0,
            fingerprint=_local_frame_fingerprint(frame),
        )
        # The semantic audit recognizes the unique input but returns a loose,
        # downward-shifted box. Local pixels may tighten geometry only after
        # that strict semantic attestation.
        provider = GeometryAuditProvider([[140, 530, 860, 710]])
        observer = GenericSceneObserver(provider)

        audited = observer.audit_element_geometry(
            frames=[frame.copy() for _ in range(4)],
            scene=current,
            element_ids=("input",),
        )

        bounds = audited.get_element("input").bounds
        self.assertAlmostEqual(122 / 810, bounds[0], delta=0.01)
        self.assertAlmostEqual(717 / 1440, bounds[1], delta=0.01)
        self.assertAlmostEqual(690 / 810, bounds[2], delta=0.01)
        self.assertAlmostEqual(847 / 1440, bounds[3], delta=0.01)
        diagnostics = observer.last_geometry_audit_diagnostics["audits"][0]
        self.assertEqual("broad_structural", diagnostics["crop_profile"])
        self.assertTrue(diagnostics["local_border_snap_used"])
        self.assertNotEqual(
            diagnostics["full_bounds"], diagnostics["snapped_full_bounds"]
        )


if __name__ == "__main__":
    unittest.main()
