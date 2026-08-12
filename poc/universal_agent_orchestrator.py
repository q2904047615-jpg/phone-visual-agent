from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from deepseek_task_graph import DynamicTaskGraph, ObservedState
from generic_intent import GenericIntentDraft
from ui_scene import MIN_TARGET_CONFIDENCE, UISceneError
from universal_action_controller import action_has_account_effect


class UniversalAgentOrchestratorError(RuntimeError):
    pass


class ObservationBridge:
    """Translate protocol objects without inventing actions or business flow."""

    def goal_draft(self, graph: DynamicTaskGraph) -> GenericIntentDraft:
        graph.validate()
        if not graph.goal.target_apps:
            raise UniversalAgentOrchestratorError(
                "任务图没有目标 App，不能建立通用观察上下文。"
            )
        active = graph.active_subgoal()
        constraints = list(graph.constraints)
        if active is not None:
            constraints.extend(active.constraints)
        entities = dict(graph.goal.entities)
        entities["target_apps"] = [
            {"app_id": item.app_id, "app_name": item.app_name}
            for item in graph.goal.target_apps
        ]
        success_criteria = {
            item.condition_id: {
                "description": item.description,
                "evidence_required": list(item.evidence_required),
                "satisfied": item.satisfied,
            }
            for item in graph.completion_conditions
        }
        account_effects = tuple(
            dict.fromkeys(item.risk_type for item in graph.risk_actions)
        )
        primary_app = graph.goal.target_apps[0]
        draft = GenericIntentDraft(
            understood=True,
            app_id=primary_app.app_id,
            app_name=primary_app.app_name,
            objective=graph.goal.objective,
            entities=entities,
            constraints=tuple(dict.fromkeys(constraints)),
            success_criteria=success_criteria,
            account_effects=account_effects,
            needs_confirmation=True,
        )
        draft.validate()
        return draft

    @staticmethod
    def _text_items(value: Any) -> tuple[str, ...]:
        if not isinstance(value, (list, tuple)):
            return ()
        return tuple(
            text
            for item in value
            if (text := str(item or "").strip())
        )

    def observed_state(
        self,
        *,
        graph: DynamicTaskGraph,
        trusted_observation: Any,
        action_outcome: str,
        verification: dict[str, Any],
    ) -> ObservedState:
        graph.validate()
        observation_device = str(
            getattr(trusted_observation, "device_id", "")
        ).strip()
        if observation_device != graph.device_id:
            raise UniversalAgentOrchestratorError(
                "任务图与可信观察 device_id 不一致。"
            )
        scene = getattr(trusted_observation, "scene", None)
        if scene is None:
            raise UniversalAgentOrchestratorError("可信观察缺少 UIScene。")
        scene.validate()
        fingerprint = str(
            getattr(trusted_observation, "fingerprint", "")
        ).strip()
        if not fingerprint or scene.fingerprint != fingerprint:
            raise UniversalAgentOrchestratorError(
                "可信观察与 UIScene fingerprint 不一致。"
            )
        if not isinstance(verification, dict):
            raise UniversalAgentOrchestratorError("控制器验证结果必须是对象。")

        evidence: list[str] = []

        def add(items: Any) -> None:
            for item in self._text_items(items):
                if item not in evidence:
                    evidence.append(item)

        if scene.summary.strip():
            add((scene.summary.strip(),))
        add(scene.overlays)
        for element in scene.elements:
            add(element.evidence)
            visible = " / ".join(
                item for item in (element.label.strip(), element.meaning.strip()) if item
            )
            if visible:
                add((visible,))
        add(verification.get("visible_evidence"))
        if action_outcome == "matched":
            add(verification.get("completion_evidence"))
        if not evidence:
            raise UniversalAgentOrchestratorError(
                "当前观察没有可交给 DeepSeek 的可见证据。"
            )

        scene_id = str(
            getattr(trusted_observation, "observation_id", "")
        ).strip() or f"{scene.screen_id}:{fingerprint[:16]}"
        observed = ObservedState(
            scene_id=scene_id,
            summary=scene.summary.strip() or "当前可信页面观察",
            visible_evidence=tuple(evidence),
            last_action_outcome=str(action_outcome or "not_applicable"),
            blocked_reasons=self._text_items(verification.get("blocked_reasons")),
        )
        observed.validate()
        return observed


@dataclass(frozen=True)
class NavigationPolicyDecision:
    allowed: bool
    reason: str
    canonical_class: str = ""


class PhaseOneNavigationPolicy:
    """Fail-closed gate for the first real-device navigation milestone.

    This class classifies one already proposed visual action.  It never plans
    a task, chooses an App, invents an element, or changes coordinates.
    """

    VERSION = "2026-08-12-phase-one-navigation-v1"
    ALLOWED_ACTIONS = frozenset(
        {"swipe", "back", "wait_for_change", "tap_semantic", "dismiss_overlay"}
    )
    FORBIDDEN_ROLES = frozenset({"toggle", "input", "keyboard_key"})
    NAVIGATION_ROLES = frozenset(
        {"button", "icon", "text", "tab", "image", "list_item"}
    )
    FORBIDDEN_ENGLISH = frozenset(
        {
            "send",
            "publish",
            "post",
            "comment",
            "follow",
            "unfollow",
            "like",
            "favorite",
            "subscribe",
            "pay",
            "purchase",
            "buy",
            "order",
            "delete",
            "remove",
            "submit",
            "save",
            "invite",
            "join",
            "input",
            "type",
            "drag",
            "longpress",
        }
    )
    FORBIDDEN_CHINESE = (
        "发送",
        "发布",
        "评论",
        "关注",
        "取关",
        "点赞",
        "收藏",
        "订阅",
        "支付",
        "购买",
        "下单",
        "删除",
        "移除",
        "提交",
        "保存",
        "邀请",
        "加入",
        "输入",
        "长按",
        "拖动",
    )
    NAVIGATION_CLASSES = (
        ("back", frozenset({"back", "return", "previous"}), ("返回", "后退", "上一页")),
        ("close", frozenset({"close", "cancel", "dismiss"}), ("关闭", "取消", "收起")),
        ("tab", frozenset({"tab", "switch"}), ("标签", "切换")),
        ("menu", frozenset({"menu", "more"}), ("菜单", "更多")),
        ("list", frozenset({"list", "item"}), ("列表", "条目")),
        ("search", frozenset({"search"}), ("搜索",)),
        ("open", frozenset({"open", "enter", "navigate"}), ("打开", "进入")),
        ("view", frozenset({"view", "details", "detail"}), ("查看", "详情")),
    )

    def __init__(self, *, min_confidence: float = MIN_TARGET_CONFIDENCE) -> None:
        self.min_confidence = float(min_confidence)

    @staticmethod
    def _value(source: Any, name: str, default: Any = "") -> Any:
        if isinstance(source, dict):
            return source.get(name, default)
        return getattr(source, name, default)

    @staticmethod
    def _deny(reason: str) -> NavigationPolicyDecision:
        return NavigationPolicyDecision(allowed=False, reason=reason)

    @staticmethod
    def _tokens(value: str) -> set[str]:
        return {
            token
            for token in re.split(r"[^a-z0-9]+", value.casefold())
            if token
        }

    def _semantic_class(self, *values: str) -> str:
        combined = " ".join(str(value or "").strip() for value in values)
        tokens = self._tokens(combined)
        if tokens.intersection(self.FORBIDDEN_ENGLISH) or any(
            marker in combined for marker in self.FORBIDDEN_CHINESE
        ):
            return "forbidden"
        for canonical, english, chinese in self.NAVIGATION_CLASSES:
            if tokens.intersection(english) or any(marker in combined for marker in chinese):
                return canonical
        return ""

    def evaluate(
        self,
        *,
        task_context: Any,
        trusted_observation: Any,
        decision: Any,
    ) -> NavigationPolicyDecision:
        impact = str(
            self._value(task_context, "current_external_impact", "unknown")
        ).strip()
        if impact in {"external_state", "unknown"}:
            return self._deny(f"第一阶段禁止 {impact} 子目标进入视觉或机械臂执行。")

        proposal = self._value(decision, "proposal", None)
        if proposal is None or str(self._value(proposal, "status", "")) != "action":
            return self._deny("当前 Qwen 决策没有唯一可执行动作。")
        action = self._value(proposal, "action", None)
        if action is None:
            return self._deny("当前 Qwen 决策缺少动作。")
        action_kind = str(self._value(action, "action", "")).strip()
        if action_kind not in self.ALLOWED_ACTIONS:
            return self._deny(f"第一阶段不允许动作：{action_kind or 'missing'}。")
        if action_kind == "wait_for_change":
            if impact not in {"read_only", "navigation_only"}:
                return self._deny(f"等待动作不能用于 {impact} 子目标。")
        elif impact != "navigation_only":
            return self._deny(f"物理导航动作要求 navigation_only，当前为 {impact}。")

        for field in ("task_id", "device_id", "revision"):
            expected = self._value(task_context, field, None)
            actual = self._value(decision, field, None)
            if expected is not None and actual != expected:
                return self._deny(f"Qwen 决策 {field} 与任务上下文不一致。")

        observation_device = str(
            self._value(trusted_observation, "device_id", "")
        ).strip()
        context_device = str(self._value(task_context, "device_id", "")).strip()
        decision_device = str(self._value(decision, "device_id", "")).strip()
        if not observation_device or observation_device not in {
            context_device,
            decision_device,
        } or context_device != decision_device:
            return self._deny("可信观察、任务和 Qwen 决策的 device_id 不一致。")

        observation_fingerprint = str(
            self._value(trusted_observation, "fingerprint", "")
        ).strip()
        decision_fingerprint = str(
            self._value(decision, "fingerprint", "")
        ).strip()
        if not observation_fingerprint or decision_fingerprint != observation_fingerprint:
            return self._deny("Qwen 决策 fingerprint 与当前可信观察不一致。")

        decision_observation = self._value(decision, "trusted_observation", None)
        if decision_observation is not None:
            bound_fingerprint = str(
                self._value(decision_observation, "fingerprint", "")
            ).strip()
            if bound_fingerprint != observation_fingerprint:
                return self._deny("Qwen 决策绑定了不同的可信观察。")

        scene = self._value(trusted_observation, "scene", None)
        if scene is None:
            return self._deny("可信观察缺少页面场景。")
        try:
            scene.validate()
        except (AttributeError, UISceneError) as exc:
            return self._deny(f"可信页面场景无效：{exc}")
        if not scene.stable or float(scene.confidence) < self.min_confidence:
            return self._deny("页面不稳定或整体置信度不足。")
        if scene.fingerprint != observation_fingerprint:
            return self._deny("页面 fingerprint 与可信观察不一致。")
        if float(self._value(decision, "confidence", 0.0)) < self.min_confidence:
            return self._deny("Qwen 决策置信度不足。")

        if action_has_account_effect(action):
            return self._deny("动作语义可能改变账号或外部状态。")

        if action_kind == "swipe":
            direction = str(action.params.get("direction") or "").strip()
            if direction not in {"up", "down", "left", "right"}:
                return self._deny("滑动方向无效。")
            return NavigationPolicyDecision(True, "允许一个四向导航滑动。", "swipe")
        if action_kind == "back":
            return NavigationPolicyDecision(True, "允许一个系统返回动作。", "back")
        if action_kind == "wait_for_change":
            return NavigationPolicyDecision(True, "允许等待页面变化，不产生物理动作。", "wait")

        element_id = str(action.params.get("element_id") or "").strip()
        try:
            element = scene.get_element(element_id, min_confidence=self.min_confidence)
        except UISceneError as exc:
            return self._deny(f"当前可信观察不能唯一解析候选：{exc}")
        if element.role in self.FORBIDDEN_ROLES or element.role not in self.NAVIGATION_ROLES:
            return self._deny(f"候选角色 {element.role} 不允许作为第一阶段导航点击。")
        expected_fields = {
            "target": element.meaning,
            "role": element.role,
            "label": element.label,
        }
        for field, expected in expected_fields.items():
            if str(action.params.get(field) or "") != expected:
                return self._deny(f"动作 {field} 没有逐字复用可信候选。")

        region = self._value(decision, "target_region", None)
        if region is None:
            return self._deny("点击动作缺少可信目标区域。")
        if (
            str(self._value(region, "kind", "")) != "element"
            or str(self._value(region, "element_id", "")) != element.element_id
            or tuple(self._value(region, "bounds", ())) != tuple(element.bounds)
        ):
            return self._deny("目标区域没有逐项复用可信候选 bounds。")

        conflicts = self._value(trusted_observation, "candidate_conflicts", ()) or ()
        if any(element.element_id in str(conflict) for conflict in conflicts):
            return self._deny("当前候选存在语义冲突或不唯一。")

        canonical = self._semantic_class(
            element.meaning,
            element.label,
            str(action.params.get("target") or ""),
        )
        if canonical == "forbidden":
            return self._deny("候选包含外部状态、输入或破坏性语义。")
        if not canonical:
            return self._deny("本地策略无法证明候选属于通用导航语义。")
        if action_kind == "dismiss_overlay" and canonical not in {"close", "back"}:
            return self._deny("关闭弹层动作只能指向关闭、取消或返回语义。")

        return NavigationPolicyDecision(
            allowed=True,
            reason="当前唯一候选通过第一阶段低风险导航策略。",
            canonical_class=canonical,
        )
