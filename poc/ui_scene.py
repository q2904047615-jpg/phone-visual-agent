from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


UI_SCENE_PROTOCOL_VERSION = "2026-08-10-ui-scene-v2"
MIN_TARGET_CONFIDENCE = 0.72

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

    @classmethod
    def from_legacy_observation(
        cls,
        observed: Any,
        *,
        frame_size: tuple[int, int],
    ) -> "UIScene":
        """Compatibility adapter; new perception should emit UIScene directly."""
        width, height = frame_size
        if width <= 0 or height <= 0:
            raise UISceneError("frame_size 无效。")
        targets = dict(getattr(observed, "targets", {}) or {})
        target_bounds = dict(getattr(observed, "target_bounds", {}) or {})
        elements: list[UIElement] = []
        for index, (meaning, point) in enumerate(targets.items(), start=1):
            bounds = target_bounds.get(meaning)
            if bounds is None:
                x, y = point
                radius = max(4, min(width, height) // 100)
                bounds = (x - radius, y - radius, x + radius, y + radius)
            left, top, right, bottom = bounds
            normalized = (
                max(0.0, left / width),
                max(0.0, top / height),
                min(1.0, right / width),
                min(1.0, bottom / height),
            )
            if normalized[0] >= normalized[2] or normalized[1] >= normalized[3]:
                continue
            elements.append(
                UIElement(
                    element_id=f"legacy-{index}-{meaning}",
                    role="unknown",
                    meaning=str(meaning),
                    label=str(meaning),
                    bounds=normalized,
                    confidence=float(getattr(observed, "confidence", 0.0)),
                    evidence=("legacy_target_adapter",),
                )
            )
        return cls(
            app_id=_infer_app_id(str(getattr(observed, "state", "unknown"))),
            screen_id=str(getattr(observed, "state", "unknown")),
            summary=str(getattr(observed, "reason", "")),
            elements=tuple(elements),
            overlays=tuple(getattr(observed, "overlays", ()) or ()),
            stable=bool(getattr(observed, "stable", False)),
            confidence=float(getattr(observed, "confidence", 0.0)),
            fingerprint=str(getattr(observed, "page_fingerprint", "")),
        )


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
