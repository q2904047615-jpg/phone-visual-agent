import unittest
from copy import deepcopy
from dataclasses import replace
from agent.domain.semantic_action import SemanticAction
from agent.domain.ui_scene import (
    SystemUIFacts,
    UIElement,
    UIScene,
)
from agent.domain.universal_action_controller import (
    UniversalActionController,
)


def element(
    element_id: str,
    meaning: str,
    *,
    role: str = "button",
    confidence: float = 0.95,
    states=None,
    bounds=(0.2, 0.3, 0.4, 0.5),
) -> UIElement:
    return UIElement(
        element_id=element_id,
        role=role,
        meaning=meaning,
        label=meaning,
        bounds=bounds,
        confidence=confidence,
        states=states or {},
        evidence=("visible",),
    )


def scene(
    *elements: UIElement,
    app_id="calculator",
    screen_id="home",
    fingerprint="a",
    confidence=0.95,
):
    return UIScene(
        app_id=app_id,
        screen_id=screen_id,
        summary="test",
        elements=tuple(elements),
        confidence=confidence,
        stable=True,
        fingerprint=fingerprint,
    )

