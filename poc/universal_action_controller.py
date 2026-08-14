from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import re
from typing import Any

from semantic_executor import SemanticAction
from ui_scene import MIN_TARGET_CONFIDENCE, UIElement, UIScene, UISceneError


UNIVERSAL_CONTROLLER_PROTOCOL_VERSION = "2026-08-14-universal-action-v5"


class UniversalActionError(RuntimeError):
    pass


ACCOUNT_EFFECT_MARKERS = frozenset(
    {
        "send",
        "send_message",
        "publish",
        "like",
        "heart",
        "comment",
        "follow",
        "delete",
        "发送",
        "发布",
        "点赞",
        "爱心",
        "评论",
        "关注",
        "删除",
    }
)
ACCOUNT_EFFECT_STATE_KEYS = frozenset(
    {
        "is_liked",
        "liked",
        "is_following",
        "following",
        "message_sent",
        "sent",
        "published",
        "deleted",
    }
)

FORBIDDEN_SEMANTIC_ENGLISH = frozenset(
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
        "confirm",
        "approve",
        "accept",
        "agree",
        "authorize",
    }
)
FORBIDDEN_SEMANTIC_CHINESE = (
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
    "确认",
    "确定",
    "同意",
    "批准",
    "授权",
)
NAVIGATION_SEMANTIC_CLASSES = (
    ("back", frozenset({"back", "return", "previous"}), ("返回", "后退", "上一页")),
    ("forward", frozenset({"forward", "next"}), ("前进", "下一页")),
    ("close", frozenset({"close", "cancel", "dismiss"}), ("关闭", "取消", "收起")),
    ("tab", frozenset({"tab", "switch"}), ("标签", "切换")),
    ("menu", frozenset({"menu", "more"}), ("菜单", "更多")),
    ("list", frozenset({"list", "item"}), ("列表", "条目")),
    ("search", frozenset({"search"}), ("搜索",)),
    (
        "open",
        frozenset({"open", "enter", "navigate", "entry", "launcher", "launch", "start"}),
        ("打开", "进入", "入口", "启动"),
    ),
    ("view", frozenset({"view", "details", "detail"}), ("查看", "详情")),
)


def navigation_semantic_class(*values: str) -> str:
    """Classify only locally known navigation semantics, failing risky text closed."""

    combined = " ".join(str(value or "").strip() for value in values)
    tokens = {
        token
        for token in re.split(r"[^a-z0-9]+", combined.casefold())
        if token
    }
    if tokens.intersection(FORBIDDEN_SEMANTIC_ENGLISH) or any(
        marker in combined for marker in FORBIDDEN_SEMANTIC_CHINESE
    ):
        return "forbidden"
    for canonical, english, chinese in NAVIGATION_SEMANTIC_CLASSES:
        if tokens.intersection(english) or any(marker in combined for marker in chinese):
            return canonical
    return ""


def action_account_effect_marker(action: SemanticAction) -> str:
    """Return the concrete marker proving that an action may change an account.

    Semantic targets are allowed to be natural labels such as ``点赞按钮`` or
    identifiers such as ``like_button``.  Exact string matching is therefore
    unsafe: it can silently treat a real account-changing action as navigation.
    """

    params = action.params
    text_values = (
        params.get("target"),
        params.get("label"),
        params.get("meaning"),
        params.get("element_meaning"),
    )
    for raw_value in text_values:
        value = str(raw_value or "").strip().casefold()
        if not value:
            continue
        english_tokens = {
            token for token in re.split(r"[^a-z0-9]+", value) if token
        }
        for marker in ACCOUNT_EFFECT_MARKERS:
            normalized = marker.casefold()
            if any("\u4e00" <= char <= "\u9fff" for char in normalized):
                if normalized in value:
                    return marker
            elif normalized in english_tokens or value == normalized:
                return marker

    expected_effect = params.get("expected_effect") or {}
    if isinstance(expected_effect, dict):
        element_state = expected_effect.get("element_state") or {}
        if isinstance(element_state, dict):
            states = element_state.get("states") or {}
            if isinstance(states, dict):
                for key in states:
                    normalized_key = str(key).strip().casefold()
                    if normalized_key in ACCOUNT_EFFECT_STATE_KEYS:
                        return normalized_key
    return ""


def action_has_account_effect(action: SemanticAction) -> bool:
    return bool(action_account_effect_marker(action))


@dataclass(frozen=True)
class ActionSafetyPolicy:
    """Safety policy is data, not App-specific orchestration code."""

    account_changing_meanings: frozenset[str] = frozenset(
        {"send", "publish", "like", "follow", "pay", "delete", "confirm_purchase"}
    )
    blocked_meanings: frozenset[str] = frozenset({"pay", "confirm_purchase"})

    def check(
        self,
        action: SemanticAction,
        *,
        confirmed: bool,
    ) -> None:
        meaning = str(action.params.get("target") or "").strip().lower()
        if meaning in self.blocked_meanings:
            raise UniversalActionError(f"当前安全策略禁止动作：{meaning}")
        marker = action_account_effect_marker(action)
        if marker and not confirmed:
            raise UniversalActionError(f"账号变更动作尚未确认：{marker}")


@dataclass(frozen=True)
class ResolvedSemanticAction:
    """One device-independent action resolved from one fresh scene."""

    node_id: str
    kind: str
    normalized_point: tuple[float, float] | None = None
    normalized_end_point: tuple[float, float] | None = None
    text: str | None = None
    direction: str | None = None
    hold_seconds: float | None = None
    target_element_id: str | None = None
    destination_element_id: str | None = None
    before_fingerprint: str = ""
    expected_effect: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        if self.normalized_point is not None:
            value["normalized_point"] = list(self.normalized_point)
        if self.normalized_end_point is not None:
            value["normalized_end_point"] = list(self.normalized_end_point)
        return value


class UniversalActionController:
    """Resolve semantic actions without knowing WeChat, Douyin or any App UI."""

    def __init__(
        self,
        *,
        min_confidence: float = MIN_TARGET_CONFIDENCE,
        safety_policy: ActionSafetyPolicy | None = None,
    ) -> None:
        self.min_confidence = float(min_confidence)
        self.safety_policy = safety_policy or ActionSafetyPolicy()

    def resolve_one(
        self,
        action: SemanticAction,
        scene: UIScene,
        *,
        confirmed: bool = False,
    ) -> ResolvedSemanticAction:
        scene.validate()
        if not scene.stable:
            raise UniversalActionError("页面仍在变化，不能执行动作。")
        if float(scene.confidence) < self.min_confidence:
            target_local_candidate = scene.unique_trusted_goal_element(
                min_confidence=self.min_confidence
            )
            action_element_id = str(action.params.get("element_id") or "").strip()
            if (
                action.action
                not in {
                    "tap_semantic",
                    "dismiss_overlay",
                    "input_verified_text",
                    "long_press",
                }
                or target_local_candidate is None
                or target_local_candidate.element_id != action_element_id
            ):
                raise UniversalActionError(
                    "页面整体置信度不足，且没有唯一可信的目标局部证据。"
                )
        self.safety_policy.check(action, confirmed=confirmed)
        expected_effect = dict(action.params.get("expected_effect") or {})

        if action.action == "tap_semantic":
            element = self._resolve_target(action, scene)
            return self._point_action(
                action,
                element,
                expected_effect,
                scene.fingerprint,
            )
        if action.action == "dismiss_overlay":
            if not action.params.get("target"):
                action = SemanticAction(
                    node_id=action.node_id,
                    action=action.action,
                    params={**action.params, "target": "close"},
                )
            element = self._resolve_target(action, scene)
            return self._point_action(
                action,
                element,
                expected_effect,
                scene.fingerprint,
            )
        if action.action == "input_verified_text":
            text = str(action.params.get("text") or "")
            if not text:
                raise UniversalActionError("文字输入动作缺少 text。")
            if len(text) > 100 or "\n" in text or "\r" in text:
                raise UniversalActionError("文字输入必须为1～100个无换行字符。")
            element = self._resolve_target(action, scene, required_role="input")
            if element.states.get("focused") is not True:
                raise UniversalActionError("文字输入前必须有当前画面证明输入框已聚焦。")
            return ResolvedSemanticAction(
                node_id=action.node_id,
                kind="input_verified_text",
                normalized_point=element.center,
                text=text,
                target_element_id=element.element_id,
                before_fingerprint=scene.fingerprint,
                expected_effect=expected_effect,
            )
        if action.action == "long_press":
            element = self._resolve_target(action, scene)
            duration_ms = action.params.get("duration_ms", 800)
            if isinstance(duration_ms, bool) or not isinstance(duration_ms, (int, float)):
                raise UniversalActionError("长按 duration_ms 格式无效。")
            if not 500 <= float(duration_ms) <= 2000:
                raise UniversalActionError("长按 duration_ms 必须在500～2000之间。")
            resolved = self._point_action(
                action,
                element,
                expected_effect,
                scene.fingerprint,
            )
            return replace(
                resolved,
                hold_seconds=float(duration_ms) / 1000.0,
            )
        if action.action == "drag":
            source = self._resolve_target(action, scene, prefix="source_")
            destination = self._resolve_target(
                action,
                scene,
                prefix="destination_",
            )
            if source.element_id == destination.element_id:
                raise UniversalActionError("拖动起点和终点不能是同一元素。")
            return ResolvedSemanticAction(
                node_id=action.node_id,
                kind="drag",
                normalized_point=source.center,
                normalized_end_point=destination.center,
                target_element_id=source.element_id,
                destination_element_id=destination.element_id,
                before_fingerprint=scene.fingerprint,
                expected_effect=expected_effect,
            )
        if action.action == "swipe":
            direction = str(action.params.get("direction") or "").strip().lower()
            if direction not in {"up", "down", "left", "right"}:
                raise UniversalActionError(f"不支持的滑动方向：{direction}")
            return ResolvedSemanticAction(
                node_id=action.node_id,
                kind="swipe",
                direction=direction,
                before_fingerprint=scene.fingerprint,
                expected_effect=expected_effect,
            )
        if action.action in {
            "back",
            "home",
            "observe",
            "wait_for_change",
            "verify",
            "finish",
        }:
            return ResolvedSemanticAction(
                node_id=action.node_id,
                kind=action.action,
                before_fingerprint=scene.fingerprint,
                expected_effect=expected_effect,
            )
        if action.action == "ensure_app":
            app_id = str(action.params.get("app_id") or "").strip().lower()
            if not app_id:
                raise UniversalActionError("ensure_app 缺少 app_id。")
            return ResolvedSemanticAction(
                node_id=action.node_id,
                kind="ensure_app",
                before_fingerprint=scene.fingerprint,
                expected_effect={"app_id": app_id, **expected_effect},
            )
        raise UniversalActionError(f"通用动作控制器尚不支持：{action.action}")

    def verify_after_action(
        self,
        resolved: ResolvedSemanticAction,
        before: UIScene,
        after: UIScene,
    ) -> None:
        before.validate()
        after.validate()
        if not after.stable or float(after.confidence) < self.min_confidence:
            raise UniversalActionError("动作后的页面不稳定或置信度不足。")
        if (
            resolved.kind not in {"observe", "verify", "finish", "wait_for_change"}
            and resolved.expected_effect.get("allow_unchanged") is not True
            and (
                (
                    before.fingerprint
                    and after.fingerprint
                    and before.fingerprint == after.fingerprint
                )
                or self.scenes_semantically_equivalent(before, after)
            )
        ):
            raise UniversalActionError("动作后页面没有可验证的语义变化。")
        expected = resolved.expected_effect
        expected_app = str(expected.get("app_id") or "").strip()
        if expected_app and after.foreground_app_id != expected_app:
            raise UniversalActionError(
                "动作后前台 App 不符合预期："
                f"{after.foreground_app_id} != {expected_app}"
            )
        expected_screen = str(expected.get("screen_id") or "").strip()
        if expected_screen and after.screen_id != expected_screen:
            raise UniversalActionError(
                f"动作后页面不符合预期：{after.screen_id} != {expected_screen}"
            )
        element_state = expected.get("element_state")
        if element_state is not None:
            if not isinstance(element_state, dict):
                raise UniversalActionError("expected_effect.element_state 格式无效。")
            meaning = str(element_state.get("meaning") or "").strip()
            states = dict(element_state.get("states") or {})
            try:
                after.resolve_unique(
                    meaning=meaning,
                    states=states,
                    min_confidence=self.min_confidence,
                )
            except UISceneError as exc:
                raise UniversalActionError(f"动作结果缺少元素状态证据：{exc}") from exc
        if resolved.kind == "input_verified_text":
            self._verify_exact_input_value(resolved, before, after)

    def _verify_exact_input_value(
        self,
        resolved: ResolvedSemanticAction,
        before: UIScene,
        after: UIScene,
    ) -> None:
        expected = resolved.text
        target_id = str(resolved.target_element_id or "").strip()
        if not expected or not target_id:
            raise UniversalActionError("输入动作缺少精确文字或目标输入框身份。")
        try:
            before_input = before.get_element(
                target_id,
                min_confidence=self.min_confidence,
            )
        except UISceneError as exc:
            raise UniversalActionError(f"输入前目标证据无效：{exc}") from exc
        if before_input.role != "input":
            raise UniversalActionError("输入前目标不是 input 元素。")

        exact_id = tuple(
            element
            for element in after.elements
            if element.element_id == target_id
            and element.role == "input"
            and float(element.confidence) >= self.min_confidence
            and element.states.get("visible") is not False
        )
        if exact_id:
            candidates = exact_id
        else:
            candidates = tuple(
                element
                for element in after.elements
                if element.role == "input"
                and float(element.confidence) >= self.min_confidence
                and element.states.get("visible") is not False
                and element.meaning.casefold() == before_input.meaning.casefold()
                and element.label.casefold() == before_input.label.casefold()
            )
        if len(candidates) != 1:
            raise UniversalActionError("动作后无法唯一绑定原目标输入框。")
        states = candidates[0].states
        if "value" not in states or not isinstance(states["value"], str):
            raise UniversalActionError("动作后缺少输入框 states.value 精确文字证据。")
        actual = states["value"]
        if actual != expected:
            raise UniversalActionError(
                f"动作后输入框文字不匹配：实际 {actual!r}，预期 {expected!r}。"
            )

    @classmethod
    def scenes_semantically_equivalent(
        cls,
        before: UIScene,
        after: UIScene,
    ) -> bool:
        """Ignore camera noise and model box jitter when comparing scenes."""

        def freeze(value: Any) -> Any:
            if isinstance(value, dict):
                return tuple(
                    sorted((str(key), freeze(item)) for key, item in value.items())
                )
            if isinstance(value, (list, tuple)):
                return tuple(freeze(item) for item in value)
            return value

        def signature(scene: UIScene) -> tuple[Any, ...]:
            elements = tuple(
                sorted(
                    (
                        element.role.casefold(),
                        element.label.casefold(),
                        # Visible text plus role/states is stronger cross-frame
                        # identity than model-authored meaning wording.  For an
                        # unlabeled icon we still need meaning to identify it.
                        (
                            ""
                            if element.label.strip()
                            else element.meaning.casefold()
                        ),
                        freeze(element.states),
                    )
                    for element in scene.elements
                )
            )
            return (
                scene.foreground_app_id.casefold(),
                scene.screen_id.casefold(),
                tuple(sorted(item.casefold() for item in scene.overlays)),
                elements,
            )

        before.validate()
        after.validate()
        return signature(before) == signature(after)

    def completion_evidence_after_action(
        self,
        resolved: ResolvedSemanticAction,
        before: UIScene,
        after: UIScene,
    ) -> tuple[str, ...]:
        """Return controller-owned proof that this action completed the goal.

        A model may declare that one action is terminal, but that declaration is
        never sufficient by itself.  At least one concrete expected effect must
        also be proven from the before/after scenes.
        """

        expected = resolved.expected_effect
        if expected.get("goal_complete_on_success") is not True:
            return ()

        # Reuse the same safety checks that accepted the physical action result.
        self.verify_after_action(resolved, before, after)
        evidence: list[str] = []

        scene_change_requested = any(
            expected.get(key) is True
            for key in ("scene_changed", "content_changed", "current_video_changed")
        )
        if scene_change_requested:
            if not before.fingerprint or not after.fingerprint:
                return ()
            if before.fingerprint == after.fingerprint:
                return ()
            evidence.append(
                "控制器确认动作前后场景指纹发生变化："
                f"{before.fingerprint} -> {after.fingerprint}"
            )

        expected_app = str(expected.get("app_id") or "").strip()
        if expected_app:
            if after.foreground_app_id != expected_app:
                return ()
            evidence.append(f"控制器确认前台 App 为 {expected_app}")

        expected_screen = str(expected.get("screen_id") or "").strip()
        if expected_screen:
            if after.screen_id != expected_screen:
                return ()
            evidence.append(f"控制器确认页面为 {expected_screen}")

        element_state = expected.get("element_state")
        if element_state is not None:
            if not isinstance(element_state, dict):
                return ()
            meaning = str(element_state.get("meaning") or "").strip()
            states = dict(element_state.get("states") or {})
            try:
                element = after.resolve_unique(
                    meaning=meaning,
                    states=states,
                    min_confidence=self.min_confidence,
                )
            except UISceneError:
                return ()
            evidence.append(
                "控制器确认目标元素状态："
                f"{element.meaning} {element.states}"
            )

        # A terminal flag with only free-form text is not machine-verifiable.
        return tuple(evidence)

    def _resolve_target(
        self,
        action: SemanticAction,
        scene: UIScene,
        *,
        prefix: str = "",
        required_role: str | None = None,
    ) -> UIElement:
        target = str(action.params.get(f"{prefix}target") or "").strip()
        element_id = str(action.params.get(f"{prefix}element_id") or "").strip()
        if not target and not element_id:
            raise UniversalActionError(
                f"{action.action} 缺少 {prefix}target 或 {prefix}element_id。"
            )
        role = str(action.params.get(f"{prefix}role") or "").strip() or None
        if required_role is not None:
            if role is not None and role != required_role:
                raise UniversalActionError(f"动作目标角色必须为 {required_role}。")
            role = required_role
        label = str(action.params.get(f"{prefix}label") or "").strip() or None
        states = action.params.get(f"{prefix}states") or {}
        if not isinstance(states, dict):
            raise UniversalActionError(f"{action.action}.{prefix}states 格式无效。")
        try:
            if element_id:
                element = scene.get_element(
                    element_id,
                    min_confidence=self.min_confidence,
                )
                if target and target.casefold() not in {
                    element.meaning.casefold(),
                    element.label.casefold(),
                }:
                    raise UISceneError(
                        f"元素 {element_id} 的语义与目标不一致：{target}"
                    )
                if role and element.role != role:
                    raise UISceneError(
                        f"元素 {element_id} 的角色与目标不一致：{role}"
                    )
                if label and element.label.casefold() != label.casefold():
                    raise UISceneError(
                        f"元素 {element_id} 的文字与目标不一致：{label}"
                    )
                if any(element.states.get(key) != value for key, value in states.items()):
                    raise UISceneError(f"元素 {element_id} 的状态与目标不一致。")
                return element
            return scene.resolve_unique(
                meaning=target,
                label=label,
                role=role,
                states=states,
                min_confidence=self.min_confidence,
            )
        except UISceneError as exc:
            raise UniversalActionError(str(exc)) from exc

    @staticmethod
    def _point_action(
        action: SemanticAction,
        element: UIElement,
        expected_effect: dict[str, Any],
        before_fingerprint: str,
    ) -> ResolvedSemanticAction:
        if element.role == "container":
            raise UniversalActionError("页面容器不是可点击控件，禁止执行点击。")
        return ResolvedSemanticAction(
            node_id=action.node_id,
            kind=action.action,
            normalized_point=element.center,
            target_element_id=element.element_id,
            before_fingerprint=before_fingerprint,
            expected_effect=expected_effect,
        )
