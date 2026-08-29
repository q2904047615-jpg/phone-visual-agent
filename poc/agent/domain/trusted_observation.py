"""Pure trusted-observation snapshot and candidate canonicalization."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .ui_scene import UIElement, UIScene
from .validation import NormalizedBounds, dataclass_wire
from .visual_evidence import LocalFrameStability


OBSERVATION_ID_PATTERN = re.compile(r"^obs_[A-Za-z0-9]{16,64}$")
ROLE_PRIORITY = {'input': 100, 'button': 95, 'icon': 90, 'keyboard_key': 85, 'list_item': 80, 'tab': 75, 'toggle': 75,
    'dialog': 60, 'text': 40, 'image': 35, 'container': 10, 'unknown': 0}


@dataclass(frozen=True)
class TrustedObservation:
    observation_id: str
    device_id: str
    fingerprint: str
    scene: UIScene
    local_stability: LocalFrameStability
    selected_frame_index: int
    frame_sharpness_scores: tuple[float, ...]
    candidate_aliases: tuple[tuple[str, str], ...] = ()
    candidate_conflicts: tuple[dict[str, Any], ...] = ()

    def get_candidate(self, element_id: str) -> UIElement:
        return self.scene.get_element(element_id)

    def target_local_candidate(self) -> UIElement | None:
        """Return the sole conflict-free goal element usable on a dynamic page."""

        return trusted_target_local_candidate(self.scene, self.candidate_conflicts)

    def to_dict(self) -> dict[str, Any]:
        value = dataclass_wire(self)
        scene = value['scene']
        system_ui = structured_system_ui(self.scene)
        if system_ui is not None:
            scene["system_ui"] = system_ui
        value.update(frame_sharpness_scores=[round(score, 3) for score in self.frame_sharpness_scores],
            candidate_aliases=dict(self.candidate_aliases))
        return value


def structured_system_ui(scene: UIScene) -> dict[str, Any] | None:
    facts = getattr(scene, "system_ui", None)
    return facts.to_dict() if facts is not None else None


def canonicalize_trusted_scene(scene: UIScene) -> tuple[UIScene, tuple[tuple[str, str], ...], tuple[dict[str, Any],
    ...]]:
    """Collapse duplicate descriptions while preserving original bounds."""

    elements = list(scene.elements)
    if len(elements) < 2:
        return scene, (), ()
    parents = list(range(len(elements)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    conflicts: list[dict[str, Any]] = []
    for left in range(len(elements)):
        for right in range(left + 1, len(elements)):
            overlap = bounds_overlap(elements[left].bounds, elements[right].bounds)
            compatible = elements_semantically_compatible(elements[left], elements[right])
            exact_same_role = bool(elements[left].label.strip()
                and elements[left].label.strip().casefold() == elements[right].label.strip().casefold()
                and (elements[left].role == elements[right].role))
            if (compatible and (overlap['intersection_over_smaller'] >= 0.85 or (exact_same_role
                and overlap['iou'] >= 0.5))):
                union(left, right)
            elif overlap['iou'] >= 0.5:
                conflicts.append({'kind': 'overlapping_semantic_conflict', 'element_ids': [elements[left].element_id,
                    elements[right].element_id], 'iou': round(overlap['iou'], 4)})

    groups: dict[int, list[UIElement]] = {}
    for (index, element) in enumerate(elements):
        groups.setdefault(find(index), []).append(element)
    canonical: list[UIElement] = []
    aliases: list[tuple[str, str]] = []
    for group in groups.values():
        selected = max(group, key=canonical_element_rank)
        canonical.append(selected)
        if len(group) > 1:
            duplicate_ids = sorted(item.element_id for item in group)
            conflicts.append({'kind': 'duplicate_visual_object_collapsed', 'canonical_element_id': selected.element_id,
                'element_ids': duplicate_ids})
            aliases.extend(((item.element_id, selected.element_id) for item
                in group if item.element_id != selected.element_id))
    canonical.sort(key=lambda item: elements.index(item))
    if len(canonical) == len(elements):
        return scene, tuple(sorted(aliases)), tuple(conflicts)
    canonical_scene = UIScene(app_id=scene.app_id, screen_id=scene.screen_id, summary=scene.summary,
        elements=tuple(canonical), overlays=scene.overlays, stable=scene.stable, confidence=scene.confidence,
        fingerprint=scene.fingerprint, protocol_version=scene.protocol_version, system_ui=scene.system_ui,
        camera_alignment=scene.camera_alignment)
    return canonical_scene, tuple(sorted(aliases)), tuple(conflicts)


def trusted_target_local_candidate(scene: UIScene, conflicts: tuple[dict[str, Any], ...]) -> UIElement | None:
    """Resolve one strong goal element and fail closed on unresolved overlap."""

    candidate = scene.unique_trusted_goal_element()
    if candidate is None:
        return None
    for conflict in conflicts:
        conflict_ids = tuple(str(item) for item in conflict.get("element_ids") or ())
        if candidate.element_id not in conflict_ids:
            continue
        if (conflict.get('kind') == 'duplicate_visual_object_collapsed'
            and conflict.get('canonical_element_id') == candidate.element_id):
            continue
        return None
    return candidate


def canonical_element_rank(element: UIElement) -> tuple[int, int, float, float]:
    left, top, right, bottom = element.bounds
    area = (right - left) * (bottom - top)
    locally_audited_input_control = int(element.element_id.startswith('local_audited_')
        and (element.meaning == 'ime_exact_candidate' and element.states.get('ime_candidate') is True
        or (element.meaning == 'input_exact_literal_key' and element.states.get('input_literal_key') is True)
        or element.states.get('keyboard_layout_switch') is True or (element.states.get('keyboard_case_switch') is True)
        or (element.states.get('keyboard_input_mode_switch') is True)))
    return (locally_audited_input_control, ROLE_PRIORITY.get(element.role, 0), float(element.confidence), -area)


def elements_semantically_compatible(left: UIElement, right: UIElement) -> bool:
    left_texts = {text.strip().casefold() for text in (left.label, *left.evidence) if text.strip()}
    right_texts = {text.strip().casefold() for text in (right.label, *right.evidence) if text.strip()}
    if left_texts and right_texts and left_texts.intersection(right_texts):
        return True
    return left.meaning.strip().casefold() == right.meaning.strip().casefold()


def bounds_overlap(left: NormalizedBounds, right: NormalizedBounds) -> dict[str, float]:
    intersection_width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    smaller = min(left_area, right_area)
    return {'iou': intersection / union if union > 0 else 0.0,
        'intersection_over_smaller': intersection / smaller if smaller > 0 else 0.0}
