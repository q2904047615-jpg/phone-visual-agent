from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from PIL import Image

from generic_scene_observer import _local_frame_fingerprint, _safe_goal_context
from generic_step_planner import (
    ALLOWED_STEP_ACTIONS,
    GenericStepPlanningError,
    GenericStepProposal,
    _reject_raw_control_data,
)
from observation_images import measure_frame_sharpness, measure_local_stability
from semantic_executor import SemanticAction
from ui_scene import ALLOWED_ROLES, UIScene, UISceneError
from vision_agent import VisionAgentError, _extract_json_object, _image_data_url


QWEN_VISUAL_DECISION_PROTOCOL_VERSION = "2026-08-11-qwen-visual-decision-v1"
QWEN_VISUAL_DECISION_MODEL_ROLE = "current_subgoal_visual_single_step"
DECISION_TIMEOUT_SECONDS = 60.0
DECISION_OUTPUT_TOKENS = 1800
DECISION_RETRY_TOKENS = 1200
MIN_DECISION_CONFIDENCE = 0.72
DEVICE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")


@dataclass(frozen=True)
class VisualTargetRegion:
    kind: str
    bounds: tuple[float, float, float, float]
    description: str
    element_id: str = ""

    def validate(self, scene: UIScene, action: SemanticAction) -> None:
        if self.kind not in {"element", "screen", "system_navigation"}:
            raise GenericStepPlanningError(f"不支持的目标区域类型：{self.kind}")
        if len(self.bounds) != 4:
            raise GenericStepPlanningError("目标区域 bounds 必须包含4个数值。")
        left, top, right, bottom = self.bounds
        if not (0.0 <= left < right <= 1.0 and 0.0 <= top < bottom <= 1.0):
            raise GenericStepPlanningError(f"目标区域超出归一化画面：{self.bounds}")
        if not self.description.strip():
            raise GenericStepPlanningError("目标区域缺少可读描述。")

        if action.action in {"tap_semantic", "dismiss_overlay"}:
            if self.kind != "element":
                raise GenericStepPlanningError("点击动作的目标区域必须绑定当前元素。")
            action_element_id = str(action.params.get("element_id") or "").strip()
            if not self.element_id or self.element_id != action_element_id:
                raise GenericStepPlanningError("目标区域 element_id 与唯一下一动作不一致。")
            element = scene.get_element(self.element_id)
            if any(abs(a - b) > 0.002 for a, b in zip(self.bounds, element.bounds)):
                raise GenericStepPlanningError(
                    "点击目标区域必须逐项复用当前页面元素 bounds，禁止另造落点。"
                )
        elif action.action == "swipe" and self.kind != "screen":
            raise GenericStepPlanningError("滑动动作的目标区域必须是当前屏幕区域。")
        elif action.action == "back" and self.kind != "system_navigation":
            raise GenericStepPlanningError("返回动作的目标区域必须是系统导航。")
        elif action.action == "wait_for_change" and self.kind != "screen":
            raise GenericStepPlanningError("等待动作的目标区域必须是当前屏幕。")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "element_id": self.element_id or None,
            "bounds": list(self.bounds),
            "description": self.description,
        }


@dataclass(frozen=True)
class QwenVisualDecision:
    device_id: str
    page_state: UIScene
    proposal: GenericStepProposal
    target_region: VisualTargetRegion | None
    expected_result: dict[str, Any]
    confidence: float
    reason: str
    protocol_version: str = QWEN_VISUAL_DECISION_PROTOCOL_VERSION

    def validate(self) -> None:
        self.page_state.validate()
        self.proposal.validate(self.page_state)
        if not DEVICE_ID_PATTERN.fullmatch(self.device_id):
            raise GenericStepPlanningError(f"device_id 格式无效：{self.device_id!r}")
        if self.protocol_version != QWEN_VISUAL_DECISION_PROTOCOL_VERSION:
            raise GenericStepPlanningError(
                f"Qwen视觉决策协议版本无效：{self.protocol_version}"
            )
        if isinstance(self.confidence, bool) or not isinstance(
            self.confidence, (int, float)
        ):
            raise GenericStepPlanningError("Qwen视觉决策置信度格式无效。")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise GenericStepPlanningError("Qwen视觉决策置信度必须在0到1之间。")
        if not isinstance(self.expected_result, dict):
            raise GenericStepPlanningError("预期结果必须是JSON对象。")

        action = self.proposal.action
        if self.proposal.status == "action":
            if action is None or self.target_region is None:
                raise GenericStepPlanningError("唯一下一动作缺少目标区域。")
            if not self.expected_result:
                raise GenericStepPlanningError("唯一下一动作缺少可验证预期结果。")
            self.target_region.validate(self.page_state, action)
            if action.action in {"tap_semantic", "dismiss_overlay"}:
                element = self.page_state.get_element(
                    str(action.params.get("element_id") or "")
                )
                copied_fields = {
                    "target": element.meaning,
                    "role": element.role,
                    "label": element.label,
                }
                for key, expected in copied_fields.items():
                    if str(action.params.get(key) or "") != expected:
                        raise GenericStepPlanningError(
                            f"唯一下一动作未逐字复制目标元素 {key}。"
                        )
                requested_states = action.params.get("states") or {}
                if any(element.states.get(key) != value for key, value in requested_states.items()):
                    raise GenericStepPlanningError(
                        "唯一下一动作 states 与目标元素可见状态不一致。"
                    )
            if dict(action.params.get("expected_effect") or {}) != self.expected_result:
                raise GenericStepPlanningError("动作 expected_effect 与顶层预期结果不一致。")
            if float(self.confidence) < MIN_DECISION_CONFIDENCE:
                raise GenericStepPlanningError("动作置信度不足，Qwen必须返回 blocked。")
        elif self.target_region is not None:
            raise GenericStepPlanningError("finished/blocked 不能携带动作目标区域。")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "protocol_version": self.protocol_version,
            "device_id": self.device_id,
            "page_state": self.page_state.to_dict(),
            "status": self.proposal.status,
            "next_action": (
                self.proposal.action.to_dict() if self.proposal.action else None
            ),
            "target_region": (
                self.target_region.to_dict() if self.target_region else None
            ),
            "expected_result": dict(self.expected_result),
            "confidence": float(self.confidence),
            "reason": self.reason,
            "completion_evidence": list(self.proposal.completion_evidence),
        }


class QwenVisualDecisionObserver:
    """Qwen reads one stable phone image and proposes at most one visual action.

    This component never imports a robot controller and has no execution method.
    Its target region is descriptive evidence; tap regions must exactly reuse a
    validated page element before any downstream controller can consume them.
    """

    def __init__(self, provider: Any) -> None:
        self.provider = provider
        self.last_raw_response = ""
        self.last_diagnostics: dict[str, Any] = {}

    def status(self) -> dict[str, Any]:
        value = dict(self.provider.status())
        value.update(
            {
                "visual_decision_protocol": QWEN_VISUAL_DECISION_PROTOCOL_VERSION,
                "model_role": QWEN_VISUAL_DECISION_MODEL_ROLE,
                "hardware_actions_enabled": False,
                "decision_timeout_seconds": DECISION_TIMEOUT_SECONDS,
                "decision_output_tokens": DECISION_OUTPUT_TOKENS,
                "decision_retry_tokens": DECISION_RETRY_TOKENS,
            }
        )
        return value

    def decide(
        self,
        *,
        frames: list[Image.Image],
        device_id: str,
        current_subgoal: dict[str, Any],
        constraints: list[str] | tuple[str, ...],
        decision_number: int = 1,
    ) -> QwenVisualDecision:
        self.last_raw_response = ""
        self.last_diagnostics = {}
        if len(frames) < 4:
            raise VisionAgentError("Qwen视觉单步决策至少需要4帧。")
        stability = measure_local_stability(frames)
        if not stability.stable:
            self.last_diagnostics = {
                "model_calls": 0,
                "local_stability": stability.to_dict(),
                "hardware_actions_enabled": False,
            }
            raise VisionAgentError(
                f"本地多帧稳定性检查未通过：{stability.reason}；不调用Qwen。"
            )

        safe_device_id = str(device_id or "").strip()
        if not DEVICE_ID_PATTERN.fullmatch(safe_device_id):
            raise VisionAgentError(f"device_id 格式无效：{safe_device_id!r}")
        safe_subgoal = _safe_goal_context(current_subgoal)
        if not safe_subgoal:
            raise VisionAgentError("当前子目标不能为空。")
        if not isinstance(constraints, (list, tuple)):
            raise VisionAgentError("constraints 必须是字符串数组。")
        safe_constraints = tuple(str(item).strip()[:300] for item in constraints)
        if any(not item for item in safe_constraints):
            raise VisionAgentError("constraints 不能包含空字符串。")
        _safe_goal_context({"constraints": list(safe_constraints)})
        sharpness_scores = [measure_frame_sharpness(item) for item in frames]
        selected_frame_index = max(
            range(len(frames)), key=sharpness_scores.__getitem__
        )
        frame = frames[selected_frame_index].convert("RGB")
        fingerprint = _local_frame_fingerprint(frame)
        base_diagnostics = {
            "visual_decision_protocol": QWEN_VISUAL_DECISION_PROTOCOL_VERSION,
            "model_calls": 0,
            "protocol_retry_used": False,
            "local_stability": stability.to_dict(),
            "selected_frame_index": selected_frame_index,
            "frame_sharpness_scores": [round(value, 3) for value in sharpness_scores],
            "frame_size": list(frame.size),
            "fingerprint": fingerprint,
            "device_id": safe_device_id,
            "hardware_actions_enabled": False,
        }
        self.last_diagnostics = dict(base_diagnostics)
        prompt = _decision_prompt(
            safe_subgoal,
            device_id=safe_device_id,
            constraints=safe_constraints,
            decision_number=max(1, int(decision_number)),
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": _image_data_url(frame)},
                    },
                ],
            }
        ]
        raw = self._provider_chat(messages, max_tokens=DECISION_OUTPUT_TOKENS)
        base_diagnostics["model_calls"] = 1
        self.last_diagnostics = dict(base_diagnostics)
        self.last_raw_response = raw
        model_calls = 1
        retry_used = False
        try:
            decision = _parse_decision(
                raw,
                fingerprint=fingerprint,
                expected_device_id=safe_device_id,
                decision_number=max(1, int(decision_number)),
            )
        except VisionAgentError as first_error:
            retry_used = True
            retry_messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": _decision_retry_prompt(
                                safe_subgoal,
                                device_id=safe_device_id,
                                constraints=safe_constraints,
                                error=first_error,
                                decision_number=max(1, int(decision_number)),
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": _image_data_url(frame)},
                        },
                    ],
                }
            ]
            raw = self._provider_chat(
                retry_messages,
                max_tokens=DECISION_RETRY_TOKENS,
            )
            model_calls += 1
            base_diagnostics.update(
                {"model_calls": model_calls, "protocol_retry_used": True}
            )
            self.last_diagnostics = dict(base_diagnostics)
            self.last_raw_response = raw
            try:
                decision = _parse_decision(
                    raw,
                    fingerprint=fingerprint,
                    expected_device_id=safe_device_id,
                    decision_number=max(1, int(decision_number)),
                )
            except VisionAgentError as retry_error:
                self.last_diagnostics.update(
                    {
                        "failed_stage": "parsing_protocol_retry",
                        "error": str(retry_error),
                    }
                )
                raise
        self.last_diagnostics = {
            "visual_decision_protocol": QWEN_VISUAL_DECISION_PROTOCOL_VERSION,
            "model_calls": model_calls,
            "protocol_retry_used": retry_used,
            "local_stability": stability.to_dict(),
            "selected_frame_index": selected_frame_index,
            "frame_sharpness_scores": [round(value, 3) for value in sharpness_scores],
            "frame_size": list(frame.size),
            "fingerprint": fingerprint,
            "device_id": safe_device_id,
            "decision_status": decision.proposal.status,
            "decision_confidence": decision.confidence,
            "hardware_actions_enabled": False,
        }
        return decision

    def _provider_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
    ) -> str:
        try:
            return self.provider._chat(
                messages,
                max_tokens=max_tokens,
                timeout=DECISION_TIMEOUT_SECONDS,
                max_attempts=1,
            )
        except TypeError as exc:
            text = str(exc)
            if "unexpected keyword" not in text and "keyword argument" not in text:
                raise
            return self.provider._chat(messages, max_tokens=max_tokens)


def _decision_prompt(
    current_subgoal: dict[str, Any],
    *,
    device_id: str,
    constraints: tuple[str, ...],
    decision_number: int,
) -> str:
    return f"""
你是通用手机视觉操作 Agent 的 Qwen 单步视觉决策层。你只负责结合“当前子目标”和
“当前真实手机画面”输出页面状态与唯一下一视觉动作；不得拆分任务图，不得规划后续多步，
不得执行机械臂，也不得输出系统命令或可直接执行的裸点击坐标。

当前子目标：
{json.dumps(current_subgoal, ensure_ascii=False, separators=(',', ':'))}
设备ID：{json.dumps(device_id, ensure_ascii=False)}
本轮约束：{json.dumps(list(constraints), ensure_ascii=False, separators=(',', ':'))}

只返回一个JSON对象，字段和语义如下：
{{
  "protocol_version":"{QWEN_VISUAL_DECISION_PROTOCOL_VERSION}",
  "device_id":"逐字复制输入的设备ID",
  "page_state":{{
    "foreground_app_id":"unknown",
    "screen_id":"unknown",
    "summary":"当前页面和关键状态",
    "elements":[],
    "overlays":[],
    "stable":true,
    "confidence":0.0,
    "fingerprint":""
  }},
  "status":"action|finished|blocked",
  "next_action":{{
    "kind":"tap_semantic|dismiss_overlay|swipe|back|wait_for_change",
    "element_id":"点击时必须引用page_state.elements中唯一元素",
    "target":"逐字复制该元素meaning",
    "role":"逐字复制该元素role",
    "label":"逐字复制该元素label",
    "states":{{}},
    "direction":"仅swipe使用up|down|left|right"
  }},
  "target_region":{{
    "kind":"element|screen|system_navigation",
    "element_id":"点击时与next_action.element_id一致，否则空字符串",
    "bounds":[0,0,1000,1000],
    "description":"区域的可见语义"
  }},
  "expected_result":{{"scene_changed":true}},
  "confidence":0.0,
  "reason":"只解释当前画面为何支持这一个动作或停止",
  "completion_evidence":[]
}}

严格规则：
1. page_state.elements最多12个，只保留当前子目标、最上层弹层、关闭/返回和必要导航相关元素。
   device_id必须逐字复制输入值{device_id}，不得改写或推断其他设备。
2. 元素字段只允许element_id、role、meaning、label、bounds、confidence、states、evidence；
   bounds使用0..1000的[left,top,right,bottom]，只框真实可见区域。
3. role仅限button/icon/input/text/tab/toggle/image/list_item/dialog/keyboard_key/container/unknown；
   meaning使用lower_snake_case；目标相关元素写states.goal_relevant=true。
4. status=action时next_action必须且只能是一个对象，禁止actions、steps、plan或第二个动作。
5. tap_semantic/dismiss_overlay只能引用当前elements中置信度不低于0.72的唯一element_id；
   target_region必须逐项复制该元素bounds，不能另造中心点或缩小框。
6. swipe的target_region.kind=screen；back使用system_navigation；wait_for_change使用screen。
7. expected_result只描述执行一个动作后能够从下一张手机画面验证的变化。
8. 当前画面已直接满足子目标时status=finished，next_action和target_region为null，并提供可见完成证据。
9. 画面模糊、目标不唯一、置信度低于0.72或所需控件不可见时status=blocked；
   next_action和target_region为null，绝不猜测。
10. 不得依据常识补出画面中看不见的文字、控件、App或状态；fingerprint留空。
11. 当前子目标明确给出文字时，目标元素label必须逐字相同；同音字、近似词、缺字、多字或
    “最接近的候选”都不算匹配，找不到逐字相同元素必须blocked。
12. 这是第{decision_number}轮，只能根据本轮画面决策，不得延续旧坐标。
不要Markdown，只返回完整JSON。
"""


def _decision_retry_prompt(
    current_subgoal: dict[str, Any],
    *,
    device_id: str,
    constraints: tuple[str, ...],
    error: Exception,
    decision_number: int,
) -> str:
    return f"""
上一次Qwen视觉决策没有通过本地协议校验，系统没有执行任何动作。
错误：{str(error)[:400]}
当前子目标：{json.dumps(current_subgoal, ensure_ascii=False, separators=(',', ':'))}
设备ID：{json.dumps(device_id, ensure_ascii=False)}
本轮约束：{json.dumps(list(constraints), ensure_ascii=False, separators=(',', ':'))}

请重新独立观察同一张图，只返回一个完整JSON，不要Markdown。必须遵守：
1. protocol_version只能是{QWEN_VISUAL_DECISION_PROTOCOL_VERSION}。
2. 顶层只能有protocol_version、device_id、page_state、status、next_action、target_region、
   expected_result、confidence、reason、completion_evidence。
   device_id必须逐字复制为{device_id}。
3. status=finished或blocked时next_action和target_region必须为null。
4. status=action时只能有一个next_action；点击必须引用page_state.elements中的唯一element_id，
   target_region.bounds必须逐项复制该元素bounds。
5. 低于0.72、模糊、不唯一或看不清时返回blocked，绝不猜。
6. page_state元素与动作字段格式沿用上一轮要求；这是第{decision_number}轮，禁止输出后续计划。
7. 子目标明确给出的文字必须与可见label逐字相同；同音字或近似词必须blocked，禁止替代。
8. page_state每个元素只允许element_id、role、meaning、label、bounds、confidence、states、evidence；
   next_action只能用kind、element_id、target、role、label、states、direction；
   target_region只能用kind、element_id、bounds、description；expected_result必须是JSON对象，
   completion_evidence必须是数组。
"""


def _parse_decision(
    raw: str,
    *,
    fingerprint: str,
    expected_device_id: str,
    decision_number: int,
) -> QwenVisualDecision:
    try:
        payload = _extract_json_object(raw)
        allowed = {
            "protocol_version",
            "device_id",
            "page_state",
            "status",
            "next_action",
            "target_region",
            "expected_result",
            "confidence",
            "reason",
            "completion_evidence",
        }
        unexpected = set(payload) - allowed
        if unexpected:
            raise GenericStepPlanningError(
                "Qwen视觉决策包含协议外字段：" + ", ".join(sorted(unexpected))
            )
        output_device_id = str(payload.get("device_id") or "").strip()
        if output_device_id != expected_device_id:
            raise GenericStepPlanningError(
                f"Qwen返回的device_id不匹配：{output_device_id!r}"
            )
        status = str(payload.get("status") or "").strip().lower()
        page_payload = payload.get("page_state")
        if not isinstance(page_payload, dict):
            raise GenericStepPlanningError("Qwen视觉决策缺少 page_state 对象。")
        _normalize_page_elements(
            page_payload,
            discard_invalid_elements=status in {"finished", "blocked"},
        )
        scene = UIScene.from_dict(
            page_payload,
            coordinate_scale=1000.0,
            stable_override=True,
            fingerprint_override=fingerprint,
        )
        raw_action = payload.get("next_action")
        expected_result = payload.get("expected_result") or {}
        if not isinstance(expected_result, dict):
            raise GenericStepPlanningError("expected_result 必须是JSON对象。")
        _reject_raw_control_data(expected_result)
        action = _parse_action(
            raw_action,
            status=status,
            expected_result=expected_result,
            decision_number=decision_number,
        )
        evidence = payload.get("completion_evidence") or []
        if not isinstance(evidence, list):
            raise GenericStepPlanningError("completion_evidence 必须是数组。")
        proposal = GenericStepProposal(
            status=status,
            action=action,
            reason=str(payload.get("reason") or "").strip()[:500],
            completion_evidence=tuple(
                str(item).strip()[:200] for item in evidence if str(item).strip()
            ),
        )
        target_region = _parse_target_region(payload.get("target_region"))

        raw_confidence = payload.get("confidence", 0.0)
        if isinstance(raw_confidence, bool):
            raise GenericStepPlanningError("confidence 不能是布尔值。")
        confidence = float(raw_confidence)
        confidence_ceiling = float(scene.confidence)
        if action and action.action in {"tap_semantic", "dismiss_overlay"}:
            element = scene.get_element(str(action.params.get("element_id") or ""))
            confidence_ceiling = min(confidence_ceiling, float(element.confidence))
        confidence = min(confidence, confidence_ceiling)

        decision = QwenVisualDecision(
            protocol_version=str(payload.get("protocol_version") or "").strip(),
            device_id=output_device_id,
            page_state=scene,
            proposal=proposal,
            target_region=target_region,
            expected_result=dict(expected_result),
            confidence=confidence,
            reason=str(payload.get("reason") or "").strip()[:500],
        )
        decision.validate()
        return decision
    except (UISceneError, GenericStepPlanningError, ValueError, TypeError) as exc:
        raise VisionAgentError(f"Qwen视觉单步决策不符合协议：{exc}") from exc


def _parse_action(
    raw_action: Any,
    *,
    status: str,
    expected_result: dict[str, Any],
    decision_number: int,
) -> SemanticAction | None:
    if status != "action":
        if raw_action not in (None, {}):
            raise GenericStepPlanningError("finished/blocked 不能携带 next_action。")
        return None
    if not isinstance(raw_action, dict):
        raise GenericStepPlanningError("action 状态缺少唯一 next_action 对象。")
    allowed = {
        "kind",
        "element_id",
        "target",
        "role",
        "label",
        "states",
        "direction",
    }
    unexpected = set(raw_action) - allowed
    if unexpected:
        raise GenericStepPlanningError(
            "next_action 包含协议外字段：" + ", ".join(sorted(unexpected))
        )
    kind = str(raw_action.get("kind") or "").strip().lower()
    if kind not in ALLOWED_STEP_ACTIONS:
        raise GenericStepPlanningError(f"唯一下一动作不在通用白名单：{kind}")
    params = {
        key: raw_action[key]
        for key in allowed - {"kind"}
        if key in raw_action and raw_action[key] not in (None, "", {}, [])
    }
    params["expected_effect"] = dict(expected_result)
    if not isinstance(params.get("states", {}), dict):
        raise GenericStepPlanningError("next_action.states 必须是对象。")
    _reject_raw_control_data(params)
    return SemanticAction(
        node_id=f"qwen_visual_decision_{max(1, int(decision_number))}",
        action=kind,
        params=params,
    )


def _parse_target_region(value: Any) -> VisualTargetRegion | None:
    if value in (None, {}):
        return None
    if not isinstance(value, dict):
        raise GenericStepPlanningError("target_region 必须是对象或null。")
    allowed = {"kind", "element_id", "bounds", "description"}
    unexpected = set(value) - allowed
    if unexpected:
        raise GenericStepPlanningError(
            "target_region 包含协议外字段：" + ", ".join(sorted(unexpected))
        )
    raw_bounds = value.get("bounds")
    if not isinstance(raw_bounds, (list, tuple)) or len(raw_bounds) != 4:
        raise GenericStepPlanningError("target_region.bounds 必须包含4个数值。")
    try:
        bounds = tuple(float(item) / 1000.0 for item in raw_bounds)
    except (TypeError, ValueError) as exc:
        raise GenericStepPlanningError("target_region.bounds 含有非数值。") from exc
    return VisualTargetRegion(
        kind=str(value.get("kind") or "").strip().lower(),
        element_id=str(value.get("element_id") or "").strip(),
        bounds=bounds,  # type: ignore[arg-type]
        description=str(value.get("description") or "").strip()[:200],
    )


def _normalize_page_elements(
    page_payload: dict[str, Any],
    *,
    discard_invalid_elements: bool,
) -> None:
    elements = page_payload.get("elements")
    if not isinstance(elements, list):
        return
    allowed_fields = {
        "element_id",
        "role",
        "meaning",
        "label",
        "bounds",
        "confidence",
        "states",
        "evidence",
    }
    accepted: list[Any] = []
    for item in elements:
        if not isinstance(item, dict):
            if not discard_invalid_elements:
                accepted.append(item)
            continue
        evidence = item.get("evidence")
        if isinstance(evidence, str):
            text = evidence.strip()
            item["evidence"] = [text] if text else []
        invalid = (
            str(item.get("role") or "").strip().lower() not in ALLOWED_ROLES
            or not isinstance(item.get("states", {}), dict)
            or bool(set(item) - allowed_fields)
        )
        if discard_invalid_elements and invalid:
            continue
        accepted.append(item)
    page_payload["elements"] = accepted
