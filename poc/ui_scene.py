from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


UI_SCENE_PROTOCOL_VERSION = "2026-08-14-ui-scene-v3"
MIN_TARGET_CONFIDENCE = 0.72
MIN_CAMERA_ALIGNMENT_CONFIDENCE = 0.80

# A low-confidence dynamic background must never authorize a screen-wide action.
# It may only expose one locally trustworthy, goal-relevant element for the
# downstream exact-element gates.
TARGET_LOCAL_ACTION_ROLES = frozenset(
    {"button", "icon", "input", "text", "tab", "toggle", "image", "list_item"}
)
COMPLETION_EVIDENCE_ROLES = frozenset({"container", "dialog"})

ALLOWED_ROLES = {
    "button",
    "icon",
    "input",
    "text",
    "tab",
    "toggle",
    "image",
    "list_item",
    "dialog",
    "keyboard_key",
    "container",
    "unknown",
}


class UISceneError(ValueError):
    pass


SYSTEM_UI_UNKNOWN = "unknown"
CAMERA_ALIGNMENT_UNKNOWN = "unknown"
CAMERA_LAYOUT_ORIENTATIONS = frozenset(
    {"portrait", "landscape", "square", CAMERA_ALIGNMENT_UNKNOWN}
)
PHONE_CONTENT_ROTATIONS = frozenset(
    {
        "upright",
        "rotated_90",
        "rotated_180",
        "rotated_270",
        CAMERA_ALIGNMENT_UNKNOWN,
    }
)

_CAMERA_ALIGNMENT_EVIDENCE_FORBIDDEN = re.compile(
    r"(?:coordinates?|coords?|bounds?|\bx\s*[=:]|\by\s*[=:]|"
    r"\bpx\s*(?::|/\s*mm\b)|\bmm\s*:|"
    r"\b(?:robot[-_ ]?controller|controller|calibration)\b|"
    r"机械臂|控制端|校准|底部(?:按钮|控件)|"
    r"\(\s*\d+\s*,\s*\d+\s*\)|"
    r"\b(?:tap|click|press|swipe|drag|execute|suggest)\b|"
    r"点击|滑动|拖动|按下|坐标|执行|建议)",
    re.IGNORECASE,
)


def camera_alignment_evidence_is_safe(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and value.strip()
        and len(value) <= 160
        and not _CAMERA_ALIGNMENT_EVIDENCE_FORBIDDEN.search(value)
    )


@dataclass(frozen=True)
class SystemUIFacts:
    """Read-only system UI facts; unknown never satisfies a visual gate."""

    immersive_or_fullscreen: bool | str = SYSTEM_UI_UNKNOWN
    navigation_bar_visible: bool | str = SYSTEM_UI_UNKNOWN

    def validate(self) -> None:
        for field_name, value in (
            ("immersive_or_fullscreen", self.immersive_or_fullscreen),
            ("navigation_bar_visible", self.navigation_bar_visible),
        ):
            if isinstance(value, bool) or value == SYSTEM_UI_UNKNOWN:
                continue
            raise UISceneError(
                f"system_ui.{field_name} 必须是布尔值或明确的 unknown。"
            )

    def to_dict(self) -> dict[str, bool | str]:
        self.validate()
        return {
            "immersive_or_fullscreen": self.immersive_or_fullscreen,
            "navigation_bar_visible": self.navigation_bar_visible,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "SystemUIFacts":
        if not isinstance(value, dict):
            raise UISceneError("scene.system_ui 必须是 JSON 对象。")
        required = {"immersive_or_fullscreen", "navigation_bar_visible"}
        missing = required - set(value)
        unexpected = set(value) - required
        if missing:
            raise UISceneError(
                "scene.system_ui 缺少字段：" + ", ".join(sorted(missing))
            )
        if unexpected:
            raise UISceneError(
                "scene.system_ui 包含协议外字段："
                + ", ".join(sorted(map(str, unexpected)))
            )
        facts = cls(
            immersive_or_fullscreen=value["immersive_or_fullscreen"],
            navigation_bar_visible=value["navigation_bar_visible"],
        )
        facts.validate()
        return facts


@dataclass(frozen=True)
class CameraAlignmentFacts:
    """Read-only relation between the camera canvas and the phone's UI axes."""

    camera_layout_orientation: str = CAMERA_ALIGNMENT_UNKNOWN
    phone_content_rotation: str = CAMERA_ALIGNMENT_UNKNOWN
    confidence: float = 0.0
    evidence: tuple[str, ...] = ()

    def validate(self) -> None:
        if (
            not isinstance(self.camera_layout_orientation, str)
            or self.camera_layout_orientation not in CAMERA_LAYOUT_ORIENTATIONS
        ):
            raise UISceneError(
                "camera_alignment.camera_layout_orientation 必须是 "
                "portrait、landscape、square 或 unknown。"
            )
        if (
            not isinstance(self.phone_content_rotation, str)
            or self.phone_content_rotation not in PHONE_CONTENT_ROTATIONS
        ):
            raise UISceneError(
                "camera_alignment.phone_content_rotation 必须是 upright、"
                "rotated_90、rotated_180、rotated_270 或 unknown。"
            )
        if isinstance(self.confidence, bool) or not isinstance(
            self.confidence, (int, float)
        ):
            raise UISceneError("camera_alignment.confidence 格式无效。")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise UISceneError("camera_alignment.confidence 必须在0到1之间。")
        if not isinstance(self.evidence, tuple) or len(self.evidence) > 2:
            raise UISceneError("camera_alignment.evidence 最多包含两个短字符串。")
        for item in self.evidence:
            if not isinstance(item, str) or not item.strip() or len(item) > 160:
                raise UISceneError(
                    "camera_alignment.evidence 只允许非空短字符串。"
                )
            if not camera_alignment_evidence_is_safe(item):
                raise UISceneError(
                    "camera_alignment.evidence 包含坐标或控制指令。"
                )
        if (
            self.phone_content_rotation != CAMERA_ALIGNMENT_UNKNOWN
            and not self.evidence
        ):
            raise UISceneError("明确的手机内容方向必须附带只读视觉证据。")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "camera_layout_orientation": self.camera_layout_orientation,
            "phone_content_rotation": self.phone_content_rotation,
            "confidence": float(self.confidence),
            "evidence": list(self.evidence),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "CameraAlignmentFacts":
        if not isinstance(value, dict):
            raise UISceneError("scene.camera_alignment 必须是 JSON 对象。")
        required = {
            "camera_layout_orientation",
            "phone_content_rotation",
            "confidence",
            "evidence",
        }
        missing = required - set(value)
        unexpected = set(value) - required
        if missing:
            raise UISceneError(
                "scene.camera_alignment 缺少字段：" + ", ".join(sorted(missing))
            )
        if unexpected:
            raise UISceneError(
                "scene.camera_alignment 包含协议外字段："
                + ", ".join(sorted(map(str, unexpected)))
            )
        evidence = value["evidence"]
        if not isinstance(evidence, list):
            raise UISceneError("scene.camera_alignment.evidence 必须是数组。")
        facts = cls(
            camera_layout_orientation=value["camera_layout_orientation"],
            phone_content_rotation=value["phone_content_rotation"],
            confidence=value["confidence"],
            evidence=tuple(evidence),
        )
        facts.validate()
        return facts


@dataclass(frozen=True)
class UIElement:
    """A perceived semantic element. It contains evidence, never an action."""

    element_id: str
    role: str
    meaning: str
    bounds: tuple[float, float, float, float]
    confidence: float
    label: str = ""
    states: dict[str, Any] = field(default_factory=dict)
    evidence: tuple[str, ...] = ()

    def validate(self) -> None:
        if not self.element_id.strip():
            raise UISceneError("元素缺少 element_id。")
        if self.role not in ALLOWED_ROLES:
            raise UISceneError(f"不支持的元素角色：{self.role}")
        if not self.meaning.strip():
            raise UISceneError("元素缺少语义 meaning。")
        if _is_system_navigation_bar_fact(self.meaning):
            raise UISceneError(
                "系统导航栏只能写入 scene.system_ui，不得进入 elements。"
            )
        if len(self.bounds) != 4:
            raise UISceneError("元素 bounds 必须包含4个归一化数值。")
        left, top, right, bottom = self.bounds
        if not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in self.bounds
        ):
            raise UISceneError("元素 bounds 格式无效。")
        if not (0.0 <= left < right <= 1.0 and 0.0 <= top < bottom <= 1.0):
            raise UISceneError(f"元素 bounds 超出归一化画面：{self.bounds}")
        if isinstance(self.confidence, bool) or not isinstance(
            self.confidence, (int, float)
        ):
            raise UISceneError("元素置信度格式无效。")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise UISceneError("元素置信度必须在0到1之间。")
        if "value" in self.states:
            value = self.states["value"]
            if self.role != "input" or not isinstance(value, str):
                raise UISceneError("只有 input 元素的 states.value 可以保存可见字符串。")
            if len(value) > 200:
                raise UISceneError("input 元素的 states.value 最多200个字符。")
        if "value_visibility" in self.states:
            visible_suffix = self.states.get("visible_value_suffix")
            full_value = self.states.get("value")
            if (
                self.role != "input"
                or self.states["value_visibility"] != "horizontal_suffix"
                or not isinstance(visible_suffix, str)
                or not visible_suffix
                or not isinstance(full_value, str)
                or visible_suffix == full_value
                or not full_value.endswith(visible_suffix)
                or self.states.get("focused") is not True
                or self.states.get("input_multiline") is not False
            ):
                raise UISceneError(
                    "horizontal_suffix 只允许标记聚焦单行输入框的非空可见尾段。"
                )
        if "keyboard_layout" in self.states:
            layout = self.states["keyboard_layout"]
            if self.role != "input" or layout not in {
                "qwerty",
                "numeric",
                "symbol",
                "unknown",
            }:
                raise UISceneError(
                    f"元素 {self.element_id}（role={self.role}）的 "
                    "states.keyboard_layout 只允许 input 使用，且值必须是 "
                    "qwerty、numeric、symbol 或 unknown。"
                )
        if "keyboard_input_mode" in self.states:
            input_mode = self.states["keyboard_input_mode"]
            if self.role != "input" or input_mode not in {
                "direct_latin",
                "chinese_pinyin",
                "unknown",
            }:
                raise UISceneError(
                    f"元素 {self.element_id}（role={self.role}）的 "
                    "states.keyboard_input_mode 只允许 input 使用，且值必须是 "
                    "direct_latin、chinese_pinyin 或 unknown。"
                )
        if "keyboard_geometry" in self.states:
            geometry = self.states["keyboard_geometry"]
            anchors = geometry.get("anchors") if isinstance(geometry, dict) else None
            if (
                self.role != "input"
                or self.states.get("focused") is not True
                or not isinstance(geometry, dict)
                or set(geometry) != {"type", "anchors", "source"}
                or geometry.get("source") != "input_structure_audit"
                or not isinstance(anchors, dict)
            ):
                raise UISceneError(
                    "states.keyboard_geometry 只允许保存输入结构审计绑定的聚焦键盘几何。"
                )
            geometry_type = geometry.get("type")
            if geometry_type == "qwerty":
                expected_anchors = {"q", "p", "a", "l", "z", "m", "backspace"}
                if (
                    self.states.get("keyboard_layout") != "qwerty"
                    or set(anchors) != expected_anchors
                ):
                    raise UISceneError(
                        "QWERTY keyboard_geometry 必须绑定完整七点 anchors。"
                    )
            elif geometry_type == "generic":
                if (
                    self.states.get("keyboard_layout")
                    not in {"qwerty", "numeric", "symbol"}
                    or set(anchors) != {"backspace"}
                ):
                    raise UISceneError(
                        "generic keyboard_geometry 只允许绑定完整可见的唯一退格键。"
                    )
            else:
                raise UISceneError(
                    "states.keyboard_geometry.type 只允许 qwerty 或 generic。"
                )
            for key, point in anchors.items():
                if (
                    not isinstance(point, (list, tuple))
                    or len(point) != 2
                    or any(
                        isinstance(part, bool)
                        or not isinstance(part, (int, float))
                        or not 0 <= float(part) <= 1000
                        for part in point
                    )
                ):
                    raise UISceneError(f"keyboard_geometry anchor {key} 无效。")
        if "local_text_clear" in self.states:
            if self.role not in {"button", "icon"} or self.states["local_text_clear"] is not True:
                raise UISceneError(
                    "states.local_text_clear=true 只允许标记独立的 button 或 icon。"
                )
            if self.meaning != "clear_local_text":
                raise UISceneError(
                    "states.local_text_clear=true 的 meaning 必须是 clear_local_text。"
                )
            if self.label.strip().casefold() not in {"×", "✕", "✖", "x"}:
                raise UISceneError(
                    "clear_local_text 必须在 label 逐字保存真实可见的 ×/✕/✖/x 图形。"
                )
        if "keyboard_input_mode_switch" in self.states:
            modes = {"direct_latin", "chinese_pinyin"}
            current_mode = self.states.get("current_mode")
            target_mode = self.states.get("target_mode")
            if (
                self.role not in {"button", "icon"}
                or self.meaning != "switch_keyboard_input_mode"
                or self.states["keyboard_input_mode_switch"] is not True
                or current_mode not in modes
                or target_mode not in modes
                or current_mode == target_mode
            ):
                raise UISceneError(
                    "keyboard_input_mode_switch 必须是方向明确的独立模式切换按钮。"
                )
        if "page_index" in self.states or "page_count" in self.states:
            page_index = self.states.get("page_index")
            page_count = self.states.get("page_count")
            if (
                self.role != "container"
                or self.meaning != "paged_viewport"
                or isinstance(page_index, bool)
                or not isinstance(page_index, int)
                or isinstance(page_count, bool)
                or not isinstance(page_count, int)
                or page_count < 2
                or not 0 <= page_index < page_count
                or self.states.get("scrollable") is not True
                or self.states.get("scroll_axis") not in {"horizontal", "vertical"}
                or self.states.get("fully_visible") is not True
                or not self.evidence
            ):
                raise UISceneError(
                    "分页视口必须用 paged_viewport container 保存有证据的零基页码、"
                    "总页数和滚动轴。"
                )
        if "focus_only_input_surface" in self.states:
            allowed_focus_only_states = {
                "enabled",
                "visible",
                "fully_visible",
                "goal_relevant",
                "focus_only_input_surface",
            }
            if (
                self.states.get("focus_only_input_surface") is not True
                or self.role != "input"
                or self.element_id.startswith("local_audited_")
                or self.states.get("goal_relevant") is not True
                or self.states.get("fully_visible") is not True
                or set(self.states) - allowed_focus_only_states
                or not any(str(item).strip() for item in self.evidence)
            ):
                raise UISceneError(
                    "focus_only_input_surface 只能标记唯一完整可见的粗输入面，"
                    "且不得携带正文、typed字段身份、键盘状态或本地审计权威。"
                )
        _reject_action_data(self.states, "states")

    @property
    def center(self) -> tuple[float, float]:
        left, top, right, bottom = self.bounds
        return ((left + right) / 2.0, (top + bottom) / 2.0)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        value = asdict(self)
        value["bounds"] = list(self.bounds)
        value["evidence"] = list(self.evidence)
        return value

    @classmethod
    def from_dict(
        cls,
        value: dict[str, Any],
        *,
        coordinate_scale: float = 1.0,
    ) -> "UIElement":
        if not isinstance(value, dict):
            raise UISceneError("元素必须是 JSON 对象。")
        _reject_action_data(value, "element")
        allowed = {
            "element_id",
            "role",
            "meaning",
            "bounds",
            "confidence",
            "label",
            "states",
            "evidence",
        }
        unexpected = set(value) - allowed
        if unexpected:
            raise UISceneError(
                "元素包含协议外字段：" + ", ".join(sorted(map(str, unexpected)))
            )
        raw_bounds = value.get("bounds")
        if not isinstance(raw_bounds, (list, tuple)) or len(raw_bounds) != 4:
            raise UISceneError("元素 bounds 必须包含4个数值。")
        if coordinate_scale <= 0:
            raise UISceneError("coordinate_scale 必须大于0。")
        try:
            bounds = tuple(float(item) / coordinate_scale for item in raw_bounds)
        except (TypeError, ValueError) as exc:
            raise UISceneError("元素 bounds 含有非数值。") from exc
        states = value.get("states") or {}
        if not isinstance(states, dict):
            raise UISceneError("元素 states 必须是 JSON 对象。")
        evidence = value.get("evidence") or []
        if not isinstance(evidence, (list, tuple)):
            raise UISceneError("元素 evidence 必须是数组。")
        element = cls(
            element_id=str(value.get("element_id") or "").strip(),
            role=str(value.get("role") or "unknown").strip(),
            meaning=str(value.get("meaning") or "").strip(),
            bounds=bounds,  # type: ignore[arg-type]
            confidence=float(value.get("confidence") or 0.0),
            label=str(value.get("label") or "").strip()[:200],
            states=dict(states),
            evidence=tuple(str(item).strip()[:200] for item in evidence if str(item).strip()),
        )
        element.validate()
        return element


def compact_drag_source_container_error(scene: Any, source: UIElement) -> str:
    """Return why a container cannot safely represent one draggable object.

    Vision role names are descriptive, not hardware authority. A compact card
    or block may be reported as either ``image`` or ``container``. This shared
    gate lets only one labelled, fully visible, goal-bound object proceed to
    confirmation-time geometry auditing; broad grouping containers remain
    denied in every policy/controller phase.
    """

    if source.role != "container":
        return ""
    label = str(source.label or "").strip()
    if not label:
        return "容器型拖动起点必须有逐字可见标签。"
    if source.states.get("goal_relevant") is not True:
        return "容器型拖动起点必须明确绑定当前目标。"
    if source.states.get("fully_visible") is not True:
        return "容器型拖动起点必须完整可见。"
    left, top, right, bottom = (float(value) for value in source.bounds)
    width = right - left
    height = bottom - top
    if width > 0.45 or height > 0.45 or width * height > 0.12:
        return "拖动起点是过大的页面容器，不能视为单个可拖动物体。"
    for candidate in getattr(scene, "elements", ()):
        if candidate.element_id == source.element_id:
            continue
        c_left, c_top, c_right, c_bottom = (
            float(value) for value in candidate.bounds
        )
        center_x = (c_left + c_right) / 2.0
        center_y = (c_top + c_bottom) / 2.0
        if not (left <= center_x <= right and top <= center_y <= bottom):
            continue
        same_literal_text = (
            candidate.role == "text" and str(candidate.label or "").strip() == label
        )
        if not same_literal_text:
            return "拖动起点容器包含其他可见元素，不能证明它是单个物体。"
    return ""


@dataclass(frozen=True)
class UIScene:
    """App-independent visual scene graph consumed by the controller."""

    # Compatibility storage name. In protocol v2 this value means only the
    # currently visible foreground App. The user's target App belongs to the
    # goal/intent object and must never be copied into a scene.
    app_id: str
    screen_id: str
    summary: str
    elements: tuple[UIElement, ...] = ()
    overlays: tuple[str, ...] = ()
    stable: bool = True
    confidence: float = 1.0
    fingerprint: str = ""
    protocol_version: str = UI_SCENE_PROTOCOL_VERSION
    system_ui: SystemUIFacts = field(default_factory=SystemUIFacts)
    camera_alignment: CameraAlignmentFacts = field(
        default_factory=CameraAlignmentFacts
    )

    @property
    def foreground_app_id(self) -> str:
        """Current foreground App; kept separate from the goal's target App."""

        return _normalize_foreground_app_id(self.app_id, self.screen_id)

    def validate(self) -> None:
        if not self.foreground_app_id.strip():
            raise UISceneError(
                "场景缺少 foreground_app_id；未知时必须明确写 unknown。"
            )
        if not self.screen_id.strip():
            raise UISceneError("场景缺少 screen_id。")
        if not isinstance(self.system_ui, SystemUIFacts):
            raise UISceneError("scene.system_ui 必须是 SystemUIFacts。")
        self.system_ui.validate()
        if not isinstance(self.camera_alignment, CameraAlignmentFacts):
            raise UISceneError(
                "scene.camera_alignment 必须是 CameraAlignmentFacts。"
            )
        self.camera_alignment.validate()
        if isinstance(self.confidence, bool) or not isinstance(
            self.confidence, (int, float)
        ):
            raise UISceneError("场景置信度格式无效。")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise UISceneError("场景置信度必须在0到1之间。")
        if not isinstance(self.stable, bool):
            raise UISceneError("场景 stable 格式无效。")
        seen: set[str] = set()
        for element in self.elements:
            element.validate()
            if element.element_id in seen:
                raise UISceneError(f"元素ID重复：{element.element_id}")
            seen.add(element.element_id)

    def find_elements(
        self,
        *,
        label: str | None = None,
        meaning: str | None = None,
        role: str | None = None,
        states: dict[str, Any] | None = None,
        min_confidence: float = MIN_TARGET_CONFIDENCE,
    ) -> tuple[UIElement, ...]:
        self.validate()
        expected_label = (label or "").strip().casefold()
        expected_meaning = (meaning or "").strip().casefold()
        expected_states = states or {}
        matches: list[UIElement] = []
        for element in self.elements:
            if float(element.confidence) < min_confidence:
                continue
            if role and element.role != role:
                continue
            if expected_label and element.label.casefold() != expected_label:
                continue
            if expected_meaning and expected_meaning not in {
                element.meaning.casefold(),
                element.label.casefold(),
            }:
                continue
            if any(element.states.get(key) != value for key, value in expected_states.items()):
                continue
            matches.append(element)
        return tuple(matches)

    def get_element(
        self,
        element_id: str,
        *,
        min_confidence: float = MIN_TARGET_CONFIDENCE,
    ) -> UIElement:
        """Resolve one model element ID inside this exact observation only."""

        self.validate()
        expected = str(element_id or "").strip()
        if not expected:
            raise UISceneError("元素 ID 不能为空。")
        for element in self.elements:
            if element.element_id != expected:
                continue
            if float(element.confidence) < min_confidence:
                raise UISceneError(f"元素置信度不足：{expected}")
            return element
        raise UISceneError(f"当前场景不存在元素：{expected}")

    def unique_trusted_goal_element(
        self,
        *,
        min_confidence: float = MIN_TARGET_CONFIDENCE,
    ) -> UIElement | None:
        """Return the sole strong goal element without trusting the whole scene.

        This is deliberately narrower than ``resolve_unique``: it only supports
        exact element-bound actions.  Screen actions still require trustworthy
        scene-level confidence in the controller policy.
        """

        self.validate()
        matches = tuple(
            element
            for element in self.elements
            if element.role in TARGET_LOCAL_ACTION_ROLES
            and element.states.get("goal_relevant") is True
            and element.states.get("enabled") is not False
            and element.states.get("visible") is not False
            and float(element.confidence) >= min_confidence
        )
        if len(matches) != 1:
            return None
        candidate = matches[0]
        for other in self.elements:
            if other.element_id == candidate.element_id:
                continue
            if (
                other.states.get("goal_relevant") is True
                and float(other.confidence) >= min_confidence
            ):
                return None
            if (
                other.role in TARGET_LOCAL_ACTION_ROLES
                and float(other.confidence) >= min_confidence
                and _bounds_iou(candidate.bounds, other.bounds) >= 0.5
            ):
                return None
        return candidate

    def trusted_completion_evidence(
        self,
        *,
        min_confidence: float = MIN_TARGET_CONFIDENCE,
    ) -> tuple[UIElement, ...]:
        """Return strong read-only facts; these never authorize an action."""

        self.validate()
        return tuple(
            element
            for element in self.elements
            if element.role in COMPLETION_EVIDENCE_ROLES
            and element.states.get("goal_relevant") is True
            and element.states.get("visible") is not False
            and float(element.confidence) >= min_confidence
        )

    def resolve_unique(
        self,
        *,
        meaning: str,
        label: str | None = None,
        role: str | None = None,
        states: dict[str, Any] | None = None,
        min_confidence: float = MIN_TARGET_CONFIDENCE,
    ) -> UIElement:
        if not self.stable:
            raise UISceneError("页面仍在变化，禁止定位控件。")
        if float(self.confidence) < min_confidence:
            raise UISceneError("页面整体置信度不足，禁止定位控件。")
        matches = self.find_elements(
            meaning=meaning,
            label=label,
            role=role,
            states=states,
            min_confidence=min_confidence,
        )
        if not matches:
            raise UISceneError(f"未找到可信的语义控件：{meaning}")
        if len(matches) != 1:
            raise UISceneError(f"语义控件不唯一：{meaning}，共{len(matches)}个")
        return matches[0]

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "protocol_version": self.protocol_version,
            "foreground_app_id": self.foreground_app_id,
            # Temporary compatibility alias for existing reports and UI.
            "app_id": self.foreground_app_id,
            "screen_id": self.screen_id,
            "summary": self.summary,
            "system_ui": self.system_ui.to_dict(),
            "camera_alignment": self.camera_alignment.to_dict(),
            "elements": [element.to_dict() for element in self.elements],
            "overlays": list(self.overlays),
            "stable": self.stable,
            "confidence": float(self.confidence),
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(
        cls,
        value: dict[str, Any],
        *,
        coordinate_scale: float = 1.0,
        stable_override: bool | None = None,
        fingerprint_override: str | None = None,
    ) -> "UIScene":
        if not isinstance(value, dict):
            raise UISceneError("视觉场景必须是 JSON 对象。")
        _reject_action_data(value, "scene")
        allowed = {
            "protocol_version",
            "foreground_app_id",
            "app_id",
            "screen_id",
            "summary",
            "system_ui",
            "camera_alignment",
            "elements",
            "overlays",
            "stable",
            "confidence",
            "fingerprint",
        }
        unexpected = set(value) - allowed
        if unexpected:
            raise UISceneError(
                "视觉场景包含协议外字段：" + ", ".join(sorted(map(str, unexpected)))
            )
        raw_elements = value.get("elements") or []
        if not isinstance(raw_elements, list):
            raise UISceneError("场景 elements 必须是数组。")
        if len(raw_elements) > 60:
            raise UISceneError("单个场景元素超过60个，拒绝不受控的视觉输出。")
        elements = tuple(
            UIElement.from_dict(item, coordinate_scale=coordinate_scale)
            for item in raw_elements
        )
        overlays = value.get("overlays") or []
        if not isinstance(overlays, list):
            raise UISceneError("场景 overlays 必须是数组。")
        if any(not isinstance(item, str) for item in overlays):
            raise UISceneError(
                "场景 overlays 只允许字符串描述；可交互候选必须放入 elements。"
            )
        screen_id = str(value.get("screen_id") or "unknown").strip().lower()
        if (
            value.get("foreground_app_id")
            and value.get("app_id")
            and str(value.get("foreground_app_id")).strip().lower()
            != str(value.get("app_id")).strip().lower()
        ):
            raise UISceneError(
                "foreground_app_id 与兼容字段 app_id 冲突，拒绝含糊场景。"
            )
        raw_foreground_app_id = str(
            value.get("foreground_app_id")
            or value.get("app_id")
            or "unknown"
        ).strip().lower()
        scene = cls(
            protocol_version=str(
                value.get("protocol_version") or UI_SCENE_PROTOCOL_VERSION
            ).strip(),
            app_id=_normalize_foreground_app_id(raw_foreground_app_id, screen_id),
            screen_id=screen_id,
            summary=str(value.get("summary") or "").strip()[:500],
            system_ui=(
                SystemUIFacts.from_dict(value["system_ui"])
                if "system_ui" in value
                else SystemUIFacts()
            ),
            camera_alignment=(
                CameraAlignmentFacts.from_dict(value["camera_alignment"])
                if "camera_alignment" in value
                else CameraAlignmentFacts()
            ),
            elements=elements,
            overlays=tuple(item.strip()[:120] for item in overlays if item.strip()),
            stable=(
                bool(value.get("stable"))
                if stable_override is None
                else bool(stable_override)
            ),
            confidence=float(value.get("confidence") or 0.0),
            fingerprint=(
                str(value.get("fingerprint") or "").strip()
                if fingerprint_override is None
                else str(fingerprint_override)
            ),
        )
        scene.validate()
        return scene


def scene_surface_kind(scene: UIScene) -> str:
    """Return the one typed surface class used by catalog and receipt checks."""

    scene.validate()
    foreground = scene.foreground_app_id.strip().casefold()
    screen = scene.screen_id.strip().casefold()
    if (
        screen
        in {
            "system_recent_tasks",
            "android_recent_tasks",
            "recent_tasks",
            "recent_apps",
            "recents",
        }
        and foreground in {"system", "android_system", "launcher", "unknown"}
    ):
        return "recent_tasks"
    identity = f"{foreground} {screen}"
    if any(token in identity for token in ("launcher", "home_screen", "desktop")):
        return "launcher"
    if scene.overlays:
        return "system_dialog" if "system" in identity else "app"
    return "app"


def _infer_app_id(screen_id: str) -> str:
    normalized_screen = screen_id.strip().lower()
    if normalized_screen in {
        "android_home",
        "ios_home",
        "launcher",
        "home_screen",
    }:
        return "launcher"
    prefix = screen_id.split("_", 1)[0].strip().lower()
    return prefix if prefix and prefix not in {"android", "unknown"} else "unknown"


def _is_system_navigation_bar_fact(meaning: str) -> bool:
    normalized = meaning.strip().casefold().replace("-", "_").replace(" ", "_")
    return normalized in {
        "navigation_bar",
        "system_navigation_bar",
        "system_nav_bar",
        "system_nav_bar_stub",
        "android_navigation_bar",
    }


def _normalize_foreground_app_id(app_id: str, screen_id: str) -> str:
    """Apply deterministic facts that must not depend on model interpretation."""

    normalized_screen = screen_id.strip().lower()
    if normalized_screen in {
        "android_home",
        "ios_home",
        "launcher",
        "home_screen",
    }:
        return "launcher"
    normalized_app = app_id.strip().lower()
    return normalized_app or "unknown"


def _bounds_iou(
    left_bounds: tuple[float, float, float, float],
    right_bounds: tuple[float, float, float, float],
) -> float:
    left = max(left_bounds[0], right_bounds[0])
    top = max(left_bounds[1], right_bounds[1])
    right = min(left_bounds[2], right_bounds[2])
    bottom = min(left_bounds[3], right_bounds[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    left_area = (left_bounds[2] - left_bounds[0]) * (
        left_bounds[3] - left_bounds[1]
    )
    right_area = (right_bounds[2] - right_bounds[0]) * (
        right_bounds[3] - right_bounds[1]
    )
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _reject_action_data(value: Any, path: str) -> None:
    forbidden = {
        "action",
        "tap",
        "swipe",
        "command",
        "shell",
        "next_action",
        "execution_plan",
    }
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).strip().lower() in forbidden:
                raise UISceneError(f"视觉场景包含动作字段：{path}.{key}")
            _reject_action_data(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_action_data(item, f"{path}[{index}]")


def ensure_scene_elements(elements: Iterable[UIElement]) -> tuple[UIElement, ...]:
    result = tuple(elements)
    for element in result:
        element.validate()
    return result
