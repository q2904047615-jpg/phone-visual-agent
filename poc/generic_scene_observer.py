from __future__ import annotations

import hashlib
import json
import re
import threading
from typing import Any

from PIL import Image

from observation_images import measure_frame_sharpness, measure_local_stability
from ui_scene import ALLOWED_ROLES, UI_SCENE_PROTOCOL_VERSION, UIScene, UISceneError
from vision_agent import VisionAgentError, _extract_json_object, _image_data_url


GENERIC_SCENE_OBSERVER_VERSION = "2026-08-10-generic-scene-observer-v5"
COMPACT_OUTPUT_TOKENS = 800
COMPACT_RETRY_TOKENS = 600
TARGETED_OUTPUT_TOKENS = 800
OBSERVATION_TIMEOUT_SECONDS = 60.0
MAX_COMPACT_ELEMENTS = 12

STAGE_LABELS = {
    "idle": "空闲",
    "checking_stability": "检查画面稳定性",
    "waiting_compact_observation": "等待千问快速观察",
    "parsing_compact_observation": "解析快速观察结果",
    "waiting_compact_retry": "等待千问修正观察格式",
    "parsing_compact_retry": "解析修正结果",
    "waiting_targeted_refinement": "等待千问目标精查",
    "parsing_targeted_refinement": "解析目标精查结果",
    "completed": "观察完成",
    "failed": "观察安全停止",
}


class GenericSceneObserver:
    """Qwen reports the current scene; it never chooses or executes actions."""

    def __init__(self, provider: Any) -> None:
        self.provider = provider
        self.last_raw_response = ""
        self.last_diagnostics: dict[str, Any] = {}
        self._stage_lock = threading.RLock()
        self._current_stage = "idle"
        self._last_stage = "idle"

    def _set_stage(self, stage: str) -> None:
        with self._stage_lock:
            self._current_stage = stage
            if stage != "idle":
                self._last_stage = stage

    def status(self) -> dict[str, Any]:
        value = dict(self.provider.status())
        with self._stage_lock:
            current_stage = self._current_stage
            last_stage = self._last_stage
        value.update(
            {
                "observer_version": GENERIC_SCENE_OBSERVER_VERSION,
                "scene_protocol": UI_SCENE_PROTOCOL_VERSION,
                "model_role": "generic_observation_only",
                "supported_app_scope": "dynamic",
                "hardware_actions_enabled": False,
                "current_stage": current_stage,
                "current_stage_label": STAGE_LABELS.get(current_stage, current_stage),
                "last_stage": last_stage,
                "last_stage_label": STAGE_LABELS.get(last_stage, last_stage),
                "compact_output_tokens": COMPACT_OUTPUT_TOKENS,
                "observation_timeout_seconds": OBSERVATION_TIMEOUT_SECONDS,
                "max_compact_elements": MAX_COMPACT_ELEMENTS,
            }
        )
        return value

    def observe(
        self,
        *,
        frames: list[Image.Image],
        goal_context: dict[str, Any] | None = None,
    ) -> UIScene:
        self.last_diagnostics = {}
        self._set_stage("checking_stability")
        model_calls = 0
        compact_retry_used = False
        targeted_refinement_used = False
        try:
            if len(frames) < 4:
                raise VisionAgentError("通用页面观察至少需要4帧。")
            stability = measure_local_stability(frames)
            if not stability.stable:
                self.last_diagnostics = {
                    "observer_version": GENERIC_SCENE_OBSERVER_VERSION,
                    "model_calls": 0,
                    "local_stability": stability.to_dict(),
                    "failed_stage": "checking_stability",
                }
                raise VisionAgentError(
                    f"本地多帧稳定性检查未通过：{stability.reason}；不调用模型。"
                )

            sharpness_scores = [measure_frame_sharpness(item) for item in frames]
            selected_frame_index = max(
                range(len(frames)),
                key=sharpness_scores.__getitem__,
            )
            frame = frames[selected_frame_index].convert("RGB")
            fingerprint = _local_frame_fingerprint(frame)
            context = _safe_goal_context(goal_context or {})
            image_part = {
                "type": "image_url",
                "image_url": {"url": _image_data_url(frame)},
            }
            first_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _compact_prompt(context)},
                        image_part,
                    ],
                }
            ]

            self._set_stage("waiting_compact_observation")
            try:
                model_calls += 1
                raw = self._provider_chat(
                    first_messages,
                    max_tokens=COMPACT_OUTPUT_TOKENS,
                )
                self.last_raw_response = raw
                self._set_stage("parsing_compact_observation")
                scene = _parse_scene(raw, fingerprint=fingerprint)
            except VisionAgentError as first_error:
                if not _compact_retry_allowed(first_error):
                    raise
                compact_retry_used = True
                self._set_stage("waiting_compact_retry")
                retry_messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": _compact_retry_prompt(context, first_error),
                            },
                            image_part,
                        ],
                    }
                ]
                model_calls += 1
                raw = self._provider_chat(
                    retry_messages,
                    max_tokens=COMPACT_RETRY_TOKENS,
                )
                self.last_raw_response = raw
                self._set_stage("parsing_compact_retry")
                scene = _parse_scene(raw, fingerprint=fingerprint)

            if _needs_targeted_refinement(scene, context):
                targeted_refinement_used = True
                self._set_stage("waiting_targeted_refinement")
                detail_messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": _targeted_prompt(
                                    context,
                                    first_scene=scene.to_dict(),
                                ),
                            },
                            image_part,
                        ],
                    }
                ]
                model_calls += 1
                raw = self._provider_chat(
                    detail_messages,
                    max_tokens=TARGETED_OUTPUT_TOKENS,
                )
                self.last_raw_response = raw
                self._set_stage("parsing_targeted_refinement")
                # A failed refinement must stop the controller. Returning the
                # earlier ambiguous scene would allow action on stale evidence.
                scene = _parse_scene(raw, fingerprint=fingerprint)

            self.last_diagnostics = {
                "observer_version": GENERIC_SCENE_OBSERVER_VERSION,
                "strategy": "compact_then_targeted_on_demand",
                "model_calls": model_calls,
                "compact_retry_used": compact_retry_used,
                "targeted_refinement_used": targeted_refinement_used,
                "local_stability": stability.to_dict(),
                "selected_frame_index": selected_frame_index,
                "frame_sharpness_scores": [
                    round(value, 3) for value in sharpness_scores
                ],
                "frame_size": list(frame.size),
                "fingerprint": fingerprint,
                "element_count": len(scene.elements),
                "output_token_budget": (
                    TARGETED_OUTPUT_TOKENS
                    if targeted_refinement_used
                    else (
                        COMPACT_RETRY_TOKENS
                        if compact_retry_used
                        else COMPACT_OUTPUT_TOKENS
                    )
                ),
            }
            self._set_stage("completed")
            return scene
        except Exception:
            self._set_stage("failed")
            if not self.last_diagnostics:
                self.last_diagnostics = {
                    "observer_version": GENERIC_SCENE_OBSERVER_VERSION,
                    "model_calls": model_calls,
                    "compact_retry_used": compact_retry_used,
                    "targeted_refinement_used": targeted_refinement_used,
                    "failed_stage": self.status()["last_stage"],
                }
            raise
        finally:
            self._set_stage("idle")

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
                timeout=OBSERVATION_TIMEOUT_SECONDS,
                max_attempts=1,
            )
        except TypeError as exc:
            # Keep simple test providers and local replay providers compatible.
            # Production DashScopeVisionProvider accepts the explicit limits.
            text = str(exc)
            if "unexpected keyword" not in text and "keyword argument" not in text:
                raise
            return self.provider._chat(messages, max_tokens=max_tokens)


def _compact_prompt(context: dict[str, Any]) -> str:
    return f"""
你是通用手机页面观察器，只报告画面事实，不规划也不执行动作。
用户目标只用于选择需要读清的控件，不能让你幻读：
{json.dumps(context, ensure_ascii=False, separators=(',', ':'))}

用最短JSON报告：当前前台App、页面类型、最上层弹层，以及与目标直接相关的可见控件。
规则：
1. 桌面写 launcher；不确定写 unknown。不得把目标App当成当前App。
2. elements最多{MAX_COMPACT_ELEMENTS}个，只保留目标相关控件、关闭/返回、当前输入框和必要导航。
3. bounds使用0..1000的[left,top,right,bottom]，必须只框真实清晰控件。
4. role仅限button/icon/input/text/tab/toggle/image/list_item/dialog/keyboard_key/container/unknown。
5. meaning用lower_snake_case。与目标直接相关的控件在states中写goal_relevant:true。
6. evidence只抄画面短文字或明确外观。看不清就降低confidence或省略元素。
7. 禁止action、plan、step、tap、swipe、command、coordinates等动作字段。

只返回下列完整JSON，不要Markdown：
{{"protocol_version":"{UI_SCENE_PROTOCOL_VERSION}","foreground_app_id":"unknown",
"screen_id":"unknown","summary":"当前画面短描述","elements":[],"overlays":[],
"stable":true,"confidence":0.0,"fingerprint":""}}
每个element只允许：
{{"element_id":"e1","role":"button","meaning":"open_search","label":"搜索",
"bounds":[0,0,1000,1000],"confidence":0.0,"states":{{"goal_relevant":true}},"evidence":[]}}
"""


def _compact_retry_prompt(context: dict[str, Any], error: Exception) -> str:
    return f"""
上一次快速观察超时或JSON不完整，控制器没有执行任何动作。请重新独立观察同一张图。
目标上下文：{json.dumps(context, ensure_ascii=False, separators=(',', ':'))}
只返回一个最小、完整、可解析JSON；elements最多6个。没有把握就写unknown和空elements，禁止猜。
格式必须是：
{{"protocol_version":"{UI_SCENE_PROTOCOL_VERSION}","foreground_app_id":"unknown",
"screen_id":"unknown","summary":"短描述","elements":[],"overlays":[],
"stable":true,"confidence":0.0,"fingerprint":""}}
元素格式仅允许element_id、role、meaning、label、bounds、confidence、states、evidence。
role仅限button/icon/input/text/tab/toggle/image/list_item/dialog/keyboard_key/container/unknown。
container仅表示与目标有关的页面内容区域；tab_group、tab_bar、navigation_bar、toolbar等其他非点击结构只写进summary，不要放入elements。
与目标直接相关的元素写states.goal_relevant=true。禁止任何动作或计划字段。不要Markdown。
"""


def _targeted_prompt(
    context: dict[str, Any],
    *,
    first_scene: dict[str, Any],
) -> str:
    # Keep the first scene short to avoid anchoring the model with many labels.
    compact_scene = {
        "foreground_app_id": first_scene.get("foreground_app_id"),
        "screen_id": first_scene.get("screen_id"),
        "summary": first_scene.get("summary"),
        "overlays": first_scene.get("overlays"),
        "confidence": first_scene.get("confidence"),
    }
    return f"""
你是通用手机页面观察器。快速观察没有找到足够明确的目标相关控件，现在只做目标精查，仍然不能规划或执行动作。
用户目标：{json.dumps(context, ensure_ascii=False, separators=(',', ':'))}
快速观察摘要：{json.dumps(compact_scene, ensure_ascii=False, separators=(',', ':'))}

重新检查原图中与目标直接相关的文字、图标、输入框、列表项和最上层弹层。
只保留最多8个最相关元素；目标元素必须states.goal_relevant=true。看不清或不唯一就不要输出，
并降低场景confidence。坐标0..1000，只框元素自身。禁止任何动作、计划或建议字段。
只返回完整JSON：
{{"protocol_version":"{UI_SCENE_PROTOCOL_VERSION}","foreground_app_id":"unknown",
"screen_id":"unknown","summary":"目标精查后的当前画面","elements":[],"overlays":[],
"stable":true,"confidence":0.0,"fingerprint":""}}
元素仅允许element_id、role、meaning、label、bounds、confidence、states、evidence。不要Markdown。
role仅限button/icon/input/text/tab/toggle/image/list_item/dialog/keyboard_key/container/unknown。
container仅表示与目标有关的页面内容区域；tab_group、tab_bar、navigation_bar、toolbar等其他非点击结构只写进summary，不要放入elements。
"""


def _parse_scene(raw: str, *, fingerprint: str) -> UIScene:
    try:
        payload = _extract_json_object(raw)
        _normalize_compact_scene_payload(payload)
        return UIScene.from_dict(
            payload,
            coordinate_scale=1000.0,
            stable_override=True,
            fingerprint_override=fingerprint,
        )
    except (UISceneError, ValueError, TypeError) as exc:
        raise VisionAgentError(f"通用页面观察结果不符合协议：{exc}") from exc


def _normalize_compact_scene_payload(payload: dict[str, Any]) -> None:
    """Normalize harmless compact-model shorthand before strict validation.

    The scene protocol still validates every field afterwards.  This only accepts
    the common JSON shorthand where a single evidence string is emitted instead
    of the requested one-item array; it does not repair coordinates, confidence,
    states or any action-like fields.
    """

    collection = payload.get("elements")
    if not isinstance(collection, list):
        return

    accepted: list[Any] = []
    for item in collection:
        if not isinstance(item, dict):
            accepted.append(item)
            continue

        evidence = item.get("evidence")
        if isinstance(evidence, str):
            text = evidence.strip()
            item["evidence"] = [text] if text else []

        role = str(item.get("role") or "").strip().lower()
        if role in ALLOWED_ROLES:
            accepted.append(item)
            continue

        states = item.get("states")
        goal_relevant = (
            isinstance(states, dict) and states.get("goal_relevant") is True
        )
        if goal_relevant:
            # Never coerce or discard an invalid target. Keeping it lets strict
            # protocol validation stop the controller before any action.
            accepted.append(item)
        # Unsupported peripheral structure is intentionally discarded. It is
        # not converted into a clickable role and therefore cannot be targeted.

    payload["elements"] = accepted


def _compact_retry_allowed(error: VisionAgentError) -> bool:
    text = str(error)
    non_retryable = (
        "未配置 DASHSCOPE_API_KEY",
        "HTTP 400",
        "HTTP 401",
        "HTTP 403",
        "HTTP 404",
        "目标上下文包含",
    )
    return not any(marker in text for marker in non_retryable)


def _needs_targeted_refinement(scene: UIScene, context: dict[str, Any]) -> bool:
    if not context:
        return False
    if scene.confidence < 0.72 or scene.screen_id == "unknown":
        return True
    if any(element.states.get("goal_relevant") is True for element in scene.elements):
        return False

    target_app = str(context.get("app_id") or "").strip().casefold()
    objective = str(context.get("objective") or "").strip()
    if target_app and scene.foreground_app_id.casefold() == target_app:
        if re.search(r"^(打开|进入|启动)", objective):
            return False

    terms = _goal_terms(context)
    if not terms:
        return False
    visible = " ".join(
        [scene.summary]
        + [
            " ".join([element.label, element.meaning, *element.evidence])
            for element in scene.elements
        ]
    ).casefold()
    return not any(term in visible for term in terms)


def _goal_terms(context: dict[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    for key in ("app_id", "app_name"):
        value = str(context.get(key) or "").strip().casefold()
        if value:
            values.append(value)
    entities = context.get("entities")
    if isinstance(entities, dict):
        values.extend(str(value).strip().casefold() for value in entities.values())
    objective = str(context.get("objective") or "").strip().casefold()
    if objective:
        simplified = re.sub(
            r"打开|进入|启动|点击|选择|查找|搜索|关闭|返回|当前|页面|应用|app|然后|请|帮我",
            " ",
            objective,
        )
        values.extend(re.findall(r"[a-z0-9_]{1,40}|[\u4e00-\u9fff]{1,12}", simplified))
        values.extend(re.findall(r"[0-9]+", objective))
    return tuple(dict.fromkeys(value for value in values if value))


def _local_frame_fingerprint(frame: Image.Image) -> str:
    compact = frame.convert("L").resize((64, 96), Image.Resampling.BILINEAR)
    return hashlib.sha256(compact.tobytes()).hexdigest()[:20]


def _safe_goal_context(value: dict[str, Any]) -> dict[str, Any]:
    """Keep goal data useful to OCR while refusing hidden control instructions."""

    forbidden = {
        "action",
        "actions",
        "step",
        "steps",
        "tap",
        "swipe",
        "coordinate",
        "coordinates",
        "x",
        "y",
        "command",
        "shell",
        "execution_plan",
    }

    def clean(item: Any, depth: int = 0) -> Any:
        if depth > 5:
            raise VisionAgentError("目标上下文嵌套过深。")
        if isinstance(item, dict):
            result: dict[str, Any] = {}
            for raw_key, raw_value in item.items():
                key = str(raw_key).strip()
                if key.lower() in forbidden:
                    raise VisionAgentError(f"目标上下文包含控制字段：{key}")
                result[key[:80]] = clean(raw_value, depth + 1)
            return result
        if isinstance(item, (list, tuple)):
            return [clean(part, depth + 1) for part in list(item)[:50]]
        if isinstance(item, str):
            return item[:1000]
        if isinstance(item, (int, float, bool)) or item is None:
            return item
        raise VisionAgentError("目标上下文包含不支持的数据类型。")

    cleaned = clean(value)
    if not isinstance(cleaned, dict):
        raise VisionAgentError("目标上下文必须是对象。")
    return cleaned
