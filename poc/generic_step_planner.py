from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from generic_intent import GenericIntentDraft, GenericIntentError, _parse_json_object
from semantic_executor import SemanticAction
from ui_scene import UIScene


GENERIC_STEP_PROTOCOL_VERSION = "2026-08-10-generic-step-v1"
ALLOWED_STEP_ACTIONS = frozenset(
    {
        "tap_semantic",
        "dismiss_overlay",
        "swipe",
        "back",
        "wait_for_change",
    }
)


class JsonStepProvider(Protocol):
    configured: bool

    def chat_json(self, messages: list[dict[str, Any]], max_tokens: int = 700) -> str: ...


class GenericStepPlanningError(ValueError):
    pass


@dataclass(frozen=True)
class GenericStepProposal:
    status: str
    action: SemanticAction | None = None
    reason: str = ""
    completion_evidence: tuple[str, ...] = ()
    protocol_version: str = GENERIC_STEP_PROTOCOL_VERSION

    def validate(self, scene: UIScene) -> None:
        scene.validate()
        if self.status not in {"action", "finished", "blocked"}:
            raise GenericStepPlanningError(f"不支持的单步状态：{self.status}")
        if self.status == "action":
            if self.action is None:
                raise GenericStepPlanningError("action 状态缺少唯一动作。")
            if self.action.action not in ALLOWED_STEP_ACTIONS:
                raise GenericStepPlanningError(
                    f"单步动作不在通用白名单：{self.action.action}"
                )
            if self.action.action in {"tap_semantic", "dismiss_overlay"}:
                element_id = str(self.action.params.get("element_id") or "").strip()
                if not element_id:
                    raise GenericStepPlanningError("点击动作必须引用当前场景 element_id。")
                scene.get_element(element_id)
            if self.action.action == "swipe":
                direction = str(self.action.params.get("direction") or "").strip()
                if direction not in {"up", "down", "left", "right"}:
                    raise GenericStepPlanningError("滑动动作方向无效。")
        elif self.action is not None:
            raise GenericStepPlanningError("finished/blocked 状态不能携带动作。")
        if self.status == "finished" and not self.completion_evidence:
            raise GenericStepPlanningError("完成判断缺少当前画面的可见证据。")
        if self.status == "blocked" and not self.reason.strip():
            raise GenericStepPlanningError("阻塞判断必须说明原因。")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["action"] = self.action.to_dict() if self.action else None
        value["completion_evidence"] = list(self.completion_evidence)
        return value


class GenericStepPlanner:
    """DeepSeek proposes one semantic step; the controller remains authoritative."""

    def __init__(self, provider: JsonStepProvider) -> None:
        self.provider = provider
        self.last_raw_response = ""

    def propose(
        self,
        goal: GenericIntentDraft,
        scene: UIScene,
        *,
        step_number: int = 1,
    ) -> GenericStepProposal:
        goal.validate()
        scene.validate()
        if not goal.understood:
            raise GenericStepPlanningError("未理解的目标不能规划动作。")
        if not self.provider.configured:
            raise GenericStepPlanningError("DeepSeek 单步规划尚未配置。")
        prompt = f"""
你是通用手机视觉操作 Agent 的“单步语义建议层”。目标理解和页面观察已经由其他模块完成。
你只能根据当前这一帧场景，提出最多一个语义动作；不能输出坐标、不能输出动作列表、
不能规划后续多步，更不能假设画面中不存在的控件。你的建议还会经过本地控制器校验和用户确认。

用户目标：
{json.dumps(goal.to_dict(), ensure_ascii=False)}

当前只读场景：
{json.dumps(scene.to_dict(), ensure_ascii=False)}

只返回一个 JSON 对象，只允许以下字段：
{{
  "status":"action|finished|blocked",
  "action":{{
    "kind":"tap_semantic|dismiss_overlay|swipe|back|wait_for_change",
    "element_id":"点击时必须是当前场景已有的 element_id",
    "target":"元素 meaning",
    "role":"元素 role",
    "label":"元素可见文字",
    "states":{{}},
    "direction":"仅 swipe 使用 up|down|left|right",
    "expected_effect":{{
      "scene_changed":false,
      "app_id":"可选，动作后应在的 App",
      "screen_id":"可选，动作后应在的页面",
      "element_state":{{"meaning":"可选","states":{{}}}},
      "goal_complete_on_success":false
    }}
  }},
  "reason":"为什么当前唯一动作推进目标，或为什么必须停止",
  "completion_evidence":["仅 finished 使用，抄录当前画面中证明已完成的证据"]
}}

严格规则：
1. action 必须只有一个对象，禁止 steps/actions/plan/coordinates/x/y/tap_point。
2. 点击只能引用 elements 中真实存在且置信度不低于0.72的 element_id；同时逐字复制其
   meaning、role、label 和必要 states，禁止自己创造目标。
3. 当前页面已有最上层弹窗时，只能关闭弹层、返回或 blocked，不能点击被遮挡页面。
4. 若目标 App 尚未打开，只能点击当前画面真实可见的对应 App 图标；看不见就 blocked。
5. 不得把“需要多步”当作 blocked；只提出眼前这一步，执行后系统会重新观察。
6. 当前阶段尚未开放通用文字输入动作。可以先点击输入框使其聚焦；若下一步必须输入文字，
   但键盘/输入引擎尚未接入，则 blocked，不能用点击键盘猜文字。
7. 只有当前画面已经直接满足 success_criteria 才能 finished，并给出证据。
8. 不确定、页面模糊、元素不唯一或所需控件不可见时 blocked。
9. expected_effect 只能描述动作后可由画面验证的事实。若本动作的效果一旦被验证就会直接
   满足整个用户目标，设置 goal_complete_on_success=true；中间步骤必须为 false。
10. 内容切换使用 scene_changed=true，不要创造 current_video_changed 等 App 专用字段。
"""
        raw = self.provider.chat_json(
            [{"role": "user", "content": prompt}],
            max_tokens=900,
        )
        self.last_raw_response = raw
        payload = _parse_json_object(raw)
        proposal = self._from_payload(payload, step_number=step_number)
        proposal.validate(scene)
        return proposal

    @staticmethod
    def _from_payload(
        payload: dict[str, Any],
        *,
        step_number: int,
    ) -> GenericStepProposal:
        allowed = {"status", "action", "reason", "completion_evidence"}
        unexpected = set(payload) - allowed
        if unexpected:
            raise GenericStepPlanningError(
                "单步模型返回协议外字段：" + ", ".join(sorted(unexpected))
            )
        status = str(payload.get("status") or "").strip().lower()
        raw_action = payload.get("action")
        action: SemanticAction | None = None
        if status == "action":
            if not isinstance(raw_action, dict):
                raise GenericStepPlanningError("单步模型缺少 action 对象。")
            action_allowed = {
                "kind",
                "element_id",
                "target",
                "role",
                "label",
                "states",
                "direction",
                "expected_effect",
            }
            action_unexpected = set(raw_action) - action_allowed
            if action_unexpected:
                raise GenericStepPlanningError(
                    "单步动作包含协议外字段："
                    + ", ".join(sorted(action_unexpected))
                )
            kind = str(raw_action.get("kind") or "").strip().lower()
            params = {
                key: raw_action[key]
                for key in action_allowed - {"kind"}
                if key in raw_action and raw_action[key] not in (None, "", {}, [])
            }
            if not isinstance(params.get("states", {}), dict):
                raise GenericStepPlanningError("动作 states 必须是对象。")
            if not isinstance(params.get("expected_effect", {}), dict):
                raise GenericStepPlanningError("动作 expected_effect 必须是对象。")
            _reject_raw_control_data(params)
            action = SemanticAction(
                node_id=f"generic_step_{max(1, int(step_number))}",
                action=kind,
                params=params,
            )
        elif raw_action not in (None, {}):
            raise GenericStepPlanningError("非 action 状态不能携带 action。")
        evidence = payload.get("completion_evidence") or []
        if not isinstance(evidence, list):
            raise GenericStepPlanningError("completion_evidence 必须是数组。")
        return GenericStepProposal(
            status=status,
            action=action,
            reason=str(payload.get("reason") or "").strip()[:500],
            completion_evidence=tuple(
                str(item).strip()[:200] for item in evidence if str(item).strip()
            ),
        )


def _reject_raw_control_data(value: Any) -> None:
    forbidden = {
        "actions",
        "steps",
        "plan",
        "coordinate",
        "coordinates",
        "tap_point",
        "x",
        "y",
        "shell",
        "command",
    }
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).strip().lower() in forbidden:
                raise GenericStepPlanningError(f"单步动作包含禁止字段：{key}")
            _reject_raw_control_data(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_raw_control_data(item)
    elif not isinstance(value, (str, int, float, bool, type(None))):
        raise GenericStepPlanningError("单步动作参数类型无效。")
