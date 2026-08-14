from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from typing import Any

from PIL import Image

from observation_images import measure_frame_sharpness, measure_local_stability
from qwen_runtime_errors import (
    FORMAT_ERROR_TYPES,
    classify_qwen_error,
    failure_diagnostics,
)
from ui_scene import (
    ALLOWED_ROLES,
    MIN_TARGET_CONFIDENCE,
    UI_SCENE_PROTOCOL_VERSION,
    UIScene,
    UISceneError,
)
from vision_agent import VisionAgentError, _extract_json_object, _image_data_url


GENERIC_SCENE_OBSERVER_VERSION = "2026-08-14-generic-scene-observer-v10"
COMPACT_OUTPUT_TOKENS = 800
COMPACT_RETRY_TOKENS = 800
TARGETED_OUTPUT_TOKENS = 1200
INPUT_STRUCTURE_AUDIT_TOKENS = 700
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
    "waiting_input_structure_audit": "等待输入结构只读审计",
    "parsing_input_structure_audit": "解析输入结构只读审计",
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
        self.last_raw_response = ""
        self.last_diagnostics = {}
        self._set_stage("checking_stability")
        started = time.perf_counter()
        model_calls = 0
        compact_retry_used = False
        format_retry_used = False
        targeted_refinement_used = False
        input_structure_audit_used = False
        targeted_roi_bounds: tuple[int, int, int, int] | None = None
        model_call_elapsed_seconds: list[float] = []
        model_call_token_budgets: list[int] = []

        def model_chat(messages: list[dict[str, Any]], *, max_tokens: int) -> str:
            nonlocal model_calls
            model_calls += 1
            model_call_token_budgets.append(max_tokens)
            call_started = time.perf_counter()
            try:
                return self._provider_chat(messages, max_tokens=max_tokens)
            finally:
                model_call_elapsed_seconds.append(
                    round(time.perf_counter() - call_started, 3)
                )

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
            detail_image_part = image_part
            first_messages = [
                _json_only_system_message(),
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
                raw = model_chat(
                    first_messages,
                    max_tokens=COMPACT_OUTPUT_TOKENS,
                )
                self.last_raw_response = raw
                self._set_stage("parsing_compact_observation")
                scene = _parse_scene(
                    raw,
                    fingerprint=fingerprint,
                    goal_context=context,
                )
            except VisionAgentError as first_error:
                first_error_type = classify_qwen_error(
                    first_error,
                    raw_response=self.last_raw_response,
                )
                if (
                    first_error_type not in FORMAT_ERROR_TYPES
                    or not _compact_retry_allowed(first_error)
                ):
                    raise
                compact_retry_used = True
                format_retry_used = True
                self._set_stage("waiting_compact_retry")
                retry_messages = [
                    _json_only_system_message(),
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
                raw = model_chat(
                    retry_messages,
                    max_tokens=COMPACT_RETRY_TOKENS,
                )
                self.last_raw_response = raw
                self._set_stage("parsing_compact_retry")
                scene = _parse_scene(
                    raw,
                    fingerprint=fingerprint,
                    goal_context=context,
                )

            if _needs_targeted_refinement(scene, context):
                targeted_refinement_used = True
                targeted_roi_bounds = _goal_directed_roi_bounds(context)
                detail_image_part = image_part
                if targeted_roi_bounds is not None:
                    detail_frame = _crop_normalized(frame, targeted_roi_bounds)
                    detail_image_part = {
                        "type": "image_url",
                        "image_url": {"url": _image_data_url(detail_frame)},
                    }
                self._set_stage("waiting_targeted_refinement")
                detail_messages = [
                    _json_only_system_message(),
                    {
                        "role": "user",
                        "content": (
                            [
                                {
                                "type": "text",
                                "text": _targeted_prompt(
                                    context,
                                    first_scene=scene.to_dict(),
                                    roi_bounds=targeted_roi_bounds,
                                ),
                                },
                                image_part,
                            ]
                            + ([detail_image_part] if targeted_roi_bounds is not None else [])
                        ),
                    }
                ]
                try:
                    raw = model_chat(
                        detail_messages,
                        max_tokens=TARGETED_OUTPUT_TOKENS,
                    )
                    self.last_raw_response = raw
                    self._set_stage("parsing_targeted_refinement")
                    # A failed refinement must stop the controller. Returning the
                    # earlier ambiguous scene would allow action on stale evidence.
                    scene = _parse_scene(
                        raw,
                        fingerprint=fingerprint,
                        goal_context=context,
                    )

                except VisionAgentError as targeted_error:
                    targeted_error_type = classify_qwen_error(
                        targeted_error,
                        raw_response=self.last_raw_response,
                    )
                    if (
                        format_retry_used
                        or targeted_error_type not in FORMAT_ERROR_TYPES
                    ):
                        raise
                    format_retry_used = True
                    self._set_stage("waiting_compact_retry")
                    targeted_retry_messages = [
                        _json_only_system_message(),
                        {
                            "role": "user",
                            "content": (
                                [
                                    {
                                    "type": "text",
                                    "text": _targeted_retry_prompt(
                                        context,
                                        targeted_error,
                                        roi_bounds=targeted_roi_bounds,
                                    ),
                                    },
                                    image_part,
                                ]
                                + ([detail_image_part] if targeted_roi_bounds is not None else [])
                            ),
                        }
                    ]
                    raw = model_chat(
                        targeted_retry_messages,
                        max_tokens=TARGETED_OUTPUT_TOKENS,
                    )
                    self.last_raw_response = raw
                    self._set_stage("parsing_compact_retry")
                    scene = _parse_scene(
                        raw,
                        fingerprint=fingerprint,
                        goal_context=context,
                    )

            if _should_audit_prefilled_input(scene, context):
                input_structure_audit_used = True
                self._set_stage("waiting_input_structure_audit")
                audit_content: list[dict[str, Any]] = [
                    {
                        "type": "text",
                        "text": _input_structure_audit_prompt(
                            context,
                            roi_bounds=targeted_roi_bounds,
                        ),
                    },
                    image_part,
                ]
                if targeted_roi_bounds is not None:
                    audit_content.append(detail_image_part)
                raw = model_chat(
                    [
                        _json_only_system_message(),
                        {"role": "user", "content": audit_content},
                    ],
                    max_tokens=INPUT_STRUCTURE_AUDIT_TOKENS,
                )
                self.last_raw_response = raw
                self._set_stage("parsing_input_structure_audit")
                scene = _apply_input_structure_audit(
                    scene,
                    raw,
                    fingerprint=fingerprint,
                )

            target_local_candidate = scene.unique_trusted_goal_element()
            completion_evidence = scene.trusted_completion_evidence()
            if not scene.stable or (
                float(scene.confidence) < MIN_TARGET_CONFIDENCE
                and target_local_candidate is None
                and not completion_evidence
            ):
                self.last_diagnostics = {
                    "observer_version": GENERIC_SCENE_OBSERVER_VERSION,
                    "strategy": "compact_then_targeted_on_demand",
                    "model_calls": model_calls,
                    "compact_retry_used": compact_retry_used,
                    "format_retry_used": format_retry_used,
                    "targeted_refinement_used": targeted_refinement_used,
                    "local_stability": stability.to_dict(),
                    "scene_confidence": float(scene.confidence),
                    "candidate_summary": [
                        {
                            "element_id": item.element_id,
                            "role": item.role,
                            "meaning": item.meaning,
                            "label": item.label,
                            "confidence": float(item.confidence),
                            "goal_relevant": item.states.get("goal_relevant") is True,
                        }
                        for item in scene.elements
                    ],
                }
                raise VisionAgentError(
                    "页面不稳定或整体置信度不足，不能建立可信候选。"
                )

            self.last_diagnostics = {
                "observer_version": GENERIC_SCENE_OBSERVER_VERSION,
                "strategy": "compact_then_targeted_on_demand",
                "model_calls": model_calls,
                "compact_retry_used": compact_retry_used,
                "format_retry_used": format_retry_used,
                "first_pass_success": not format_retry_used,
                "repair_retry_success": format_retry_used,
                "targeted_refinement_used": targeted_refinement_used,
                "input_structure_audit_used": input_structure_audit_used,
                "targeted_roi_bounds": (
                    list(targeted_roi_bounds)
                    if targeted_roi_bounds is not None
                    else None
                ),
                "prefilled_input_structure_inferred": any(
                    item.element_id.startswith("local_structured_input_")
                    for item in scene.elements
                ),
                "model_call_elapsed_seconds": model_call_elapsed_seconds,
                "model_call_token_budgets": model_call_token_budgets,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "local_stability": stability.to_dict(),
                "selected_frame_index": selected_frame_index,
                "confidence_basis": (
                    "scene"
                    if float(scene.confidence) >= MIN_TARGET_CONFIDENCE
                    else "unique_goal_element"
                    if target_local_candidate is not None
                    else "completion_evidence_only"
                ),
                "frame_sharpness_scores": [
                    round(value, 3) for value in sharpness_scores
                ],
                "frame_size": list(frame.size),
                "fingerprint": fingerprint,
                "element_count": len(scene.elements),
                "output_token_budget": model_call_token_budgets[-1],
            }
            self._set_stage("completed")
            return scene
        except Exception as exc:
            failed_stage = self.status()["last_stage"]
            self._set_stage("failed")
            base = dict(self.last_diagnostics)
            base.update(
                {
                    "observer_version": GENERIC_SCENE_OBSERVER_VERSION,
                    "model_calls": model_calls,
                    "compact_retry_used": compact_retry_used,
                    "format_retry_used": format_retry_used,
                    "first_pass_success": False,
                    "repair_retry_success": False,
                    "targeted_refinement_used": targeted_refinement_used,
                    "input_structure_audit_used": input_structure_audit_used,
                    "model_call_elapsed_seconds": model_call_elapsed_seconds,
                    "model_call_token_budgets": model_call_token_budgets,
                }
            )
            base.update(
                failure_diagnostics(
                    exc,
                    raw_response=self.last_raw_response,
                    stage=failed_stage,
                    model_calls=model_calls,
                    elapsed_seconds=time.perf_counter() - started,
                    safe_stop_reason="观察阶段未建立可信候选，决策模型与控制器均未执行动作。",
                )
            )
            base["raw_response_length"] = len(self.last_raw_response)
            base["raw_response_excerpt"] = self.last_raw_response[:1000]
            self.last_diagnostics = base
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
                max_attempts=2,
            )
        except TypeError as exc:
            # Keep simple test providers and local replay providers compatible.
            # Production DashScopeVisionProvider accepts the explicit limits.
            text = str(exc)
            if "unexpected keyword" not in text and "keyword argument" not in text:
                raise
            return self.provider._chat(messages, max_tokens=max_tokens)


def _json_only_system_message() -> dict[str, str]:
    return {
        "role": "system",
        "content": (
            "你是只读页面观察器。只输出一个语法完整的JSON对象；禁止Markdown、解释、"
            "思考过程、代码围栏、JSON字符串套壳或对象前后的任何文字。"
        ),
    }


PREFILLED_INPUT_OBSERVATION_RULE = (
    "输入框可能为空，也可能已经含有文字；预填充且未聚焦时可以没有光标或占位提示。"
    "当一个有清晰独立边界的横向矩形内含查询/地址/表单文字，并带有边界独立的尾部功能控件"
    "（例如搜索、提交、清除、语音或扫描图标）时，"
    "这组结构本身就是role=input的可靠视觉证据，不得仅因没有光标而降级成text或container；"
    "尾部功能控件必须作为另一个控件观察，不能把输入框和功能控件合成横幅。这个判断只报告"
    "页面事实，绝不表示可以激活尾部控件。框内文字的内容或主题不能改变控件角色；其他没有"
    "上述成组结构的带文字区域仍不得仅因含有文字就被认作输入框。"
)

INPUT_VALUE_OBSERVATION_RULE = (
    "role=input且框内文字清晰可读时，必须在states.value中逐字填写当前可见文字；空框写空字符串，"
    "看不清才省略value，禁止根据目标补写。软键盘可见时还必须在states.keyboard_layout写"
    "qwerty、numeric、symbol或unknown，并在states.keyboard_input_mode写direct_latin、"
    "chinese_pinyin或unknown。QWERTY只描述按键排列，绝不等于英文直输：画面出现中文候选、"
    "拼音分词撇号或明确中文模式时必须写chinese_pinyin；只有明确显示英文/Latin直输模式时才能写"
    "direct_latin；看不清写unknown。这些都只是画面事实，不授权输入。若键盘底部清楚可见独立的"
    "中/英模式切换键，必须另建role=button元素，meaning写switch_keyboard_input_mode，label逐字抄"
    "可见键面文字，states写keyboard_input_mode_switch:true、current_mode和target_mode；不确定当前"
    "模式或切换方向时不得编造该元素。字母、数字、退格、回车等普通键仍必须role=keyboard_key。"
    "若非空输入框内部或紧邻右侧清楚可见独立的圆形×/清空图标，必须另建role=button或icon元素，"
    "meaning写clear_local_text，states写local_text_clear:true，label必须逐字写图标本身的×/✕/✖/x；"
    "若看不清真实叉号图形或只能自由描述为叉号，就不得标记local_text_clear。只框该图标自身，不能与输入框合并，"
    "也绝不能把键盘退格键/删除键标成local_text_clear。页面右侧的文字‘取消’/cancel是取消编辑或"
    "退出控件，不是本地清空图标；必须meaning=cancel且goal_relevant:false，绝不能标成clear_local_text。"
)


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
   overlays只允许简短字符串名称；任何带边界、角色或ID的可交互候选必须放入elements，
   不得把对象放入overlays。
8. 场景confidence只评价当前画面本身是否清楚、稳定、可描述，不评价目标是否已完成或目标控件
   是否存在。清晰稳定的页面即使没有目标控件，也应保持与画面质量一致的高confidence并返回空
   elements；只有模糊、遮挡、过渡或无法判断页面事实时才降低confidence。
9. {PREFILLED_INPUT_OBSERVATION_RULE}
10. {INPUT_VALUE_OBSERVATION_RULE}

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
只返回一个最小、完整、可解析JSON；不要转义成字符串，不要输出reasoning或说明文字。
summary最多40字，elements最多2个，evidence每个元素最多1条且最多30字；禁止罗列非目标内容。
没有把握就写unknown和空elements，禁止猜。务必在token耗尽前闭合全部括号。
画面清晰稳定但目标控件不存在时，空elements不等于低置信；confidence仍只按画面质量填写。
格式修复不能靠删除真实候选通过：若原图清楚存在与目标直接相关的可交互入口，即使目标结果
尚未出现，也必须在elements中报告该入口；只有重新观察后仍无法确认时才返回空elements。
格式必须是：
{{"protocol_version":"{UI_SCENE_PROTOCOL_VERSION}","foreground_app_id":"unknown",
"screen_id":"unknown","summary":"短描述","elements":[],"overlays":[],
"stable":true,"confidence":0.0,"fingerprint":""}}
元素格式仅允许element_id、role、meaning、label、bounds、confidence、states、evidence。
bounds必须是恰好4个0..1000数值的数组[left,top,right,bottom]；不能是x/y/width/height对象、
两个点或嵌套数组。
role仅限button/icon/input/text/tab/toggle/image/list_item/dialog/keyboard_key/container/unknown。
container仅表示与目标有关的页面内容区域；tab_group、tab_bar、navigation_bar、toolbar等其他非点击结构只写进summary，不要放入elements。
overlays只能是字符串数组；带bounds、role、element_id或overlay_id的对象必须改写成elements，
并使用element_id。禁止把对象序列化成字符串塞入overlays。
与目标直接相关的元素写states.goal_relevant=true。禁止任何动作或计划字段。不要Markdown。
输入框识别规则：{PREFILLED_INPUT_OBSERVATION_RULE}
输入框文字与键盘规则：{INPUT_VALUE_OBSERVATION_RULE}
"""


def _targeted_retry_prompt(
    context: dict[str, Any],
    error: Exception,
    *,
    roi_bounds: tuple[int, int, int, int] | None = None,
) -> str:
    return f"""
上一次目标精查输出不是完整、合法的页面观察JSON，控制器没有产生任何候选动作。
错误摘要：{str(error)[:300]}
目标上下文：{json.dumps(context, ensure_ascii=False, separators=(',', ':'))}
{_roi_observation_note(roi_bounds)}
这是本轮观察唯一一次格式修复。请重新独立观察原图，只返回最小完整JSON；没有可靠目标就返回空elements并降低confidence。
格式修复不能靠删除真实候选通过；原图中清楚可见且与目标直接相关的入口必须改写为elements，
即使目标最终结果尚未出现。只有重新观察后仍无法确认时才返回空elements。
格式：
{{"protocol_version":"{UI_SCENE_PROTOCOL_VERSION}","foreground_app_id":"unknown",
"screen_id":"unknown","summary":"短描述","elements":[],"overlays":[],
"stable":true,"confidence":0.0,"fingerprint":""}}
元素仅允许element_id、role、meaning、label、bounds、confidence、states、evidence；禁止动作、计划和裸坐标。不要Markdown。
bounds必须是恰好4个0..1000数值的数组[left,top,right,bottom]；不能是x/y/width/height对象、两个点或嵌套数组。
overlays只能是字符串数组；可交互候选必须放入elements并使用element_id，不能把对象放入overlays。
输入框识别规则：{PREFILLED_INPUT_OBSERVATION_RULE}
输入框文字与键盘规则：{INPUT_VALUE_OBSERVATION_RULE}
"""


def _targeted_prompt(
    context: dict[str, Any],
    *,
    first_scene: dict[str, Any],
    roi_bounds: tuple[int, int, int, int] | None = None,
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
{_roi_observation_note(roi_bounds)}

重新检查原图中与目标直接相关的文字、图标、输入框、列表项和最上层弹层。
只保留最多4个最相关元素；目标元素必须states.goal_relevant=true。看不清或不唯一就不要输出，
并降低场景confidence。坐标0..1000，只框元素自身。禁止任何动作、计划或建议字段。
目标相关元素既包括已经满足完成条件的可见结果，也包括画面上清楚可见、能使该结果进入视野
的入口控件；这里只报告控件事实，不建议也不授权使用它。
置信度只评价当前画面观察本身是否可靠，不能因为目标尚未完成而降低；例如清晰桌面上唯一目标
应用入口可形成高可信观察，即使应用尚未打开。模糊、遮挡或不唯一时仍必须降低，禁止虚增。
目标相关控件确实不存在时返回空elements，但只要页面事实清楚稳定，场景confidence仍应保持高值；
不得因为系统级动作没有屏内按钮、或因为未找到目标控件，就把清晰页面写成低置信。
    输入框识别规则：{PREFILLED_INPUT_OBSERVATION_RULE}
    输入框文字与键盘规则：{INPUT_VALUE_OBSERVATION_RULE}
如果能清楚看见相关横向边框、框内文字和右侧独立搜索/提交按钮，但仍无法判断边框是否可编辑，
不得因此返回空elements：请分别报告container、其内部text和右侧button的真实边界与证据；
这三个元素都必须在states中明确写fully_visible:true或false。若画面边缘还有被裁切的相似结构，
只能在summary说明，不能把它标成目标；优先报告四边完整可见的结构。完整container和text写
goal_relevant:true，相邻button写goal_relevant:false。本地只会在三者都fully_visible:true且严格
几何关系成立时把这组只读事实归一化，绝不会因此激活按钮。
只返回完整JSON：
{{"protocol_version":"{UI_SCENE_PROTOCOL_VERSION}","foreground_app_id":"unknown",
"screen_id":"unknown","summary":"目标精查后的当前画面","elements":[],"overlays":[],
"stable":true,"confidence":0.0,"fingerprint":""}}
元素仅允许element_id、role、meaning、label、bounds、confidence、states、evidence。不要Markdown。
role仅限button/icon/input/text/tab/toggle/image/list_item/dialog/keyboard_key/container/unknown。
container仅表示与目标有关的页面内容区域；tab_group、tab_bar、navigation_bar、toolbar等其他非点击结构只写进summary，不要放入elements。
overlays只能是字符串数组；任何可交互候选都必须放入elements并使用element_id，
不得把带bounds、role或ID的对象放入overlays。
"""


def _input_structure_audit_prompt(
    context: dict[str, Any],
    *,
    roi_bounds: tuple[int, int, int, int] | None,
) -> str:
    return f"""
You are a read-only generic UI structure auditor. The normal scene observer did not establish an input target.
Goal context (evidence selection only): {json.dumps(context, ensure_ascii=False, separators=(',', ':'))}
Image 1 is always the complete phone frame. {_input_audit_detail_note(roi_bounds)}
Enumerate every horizontal search/address/form-like editable structure relevant to the goal, including structures clipped by an image edge.
Do not plan, suggest, authorize, or perform any action. All bounds MUST use Image 1 full-frame normalized coordinates 0..1000.
For each structure report whether all four outer edges are fully visible, its current text, confidence, and its separate trailing utility control (for example search, submit, clear, voice, or scan). A trailing control is structural evidence only and is never authorized for activation.
Return exactly this JSON schema and no other fields:
{{"structures":[{{"structure_id":"s1","bounds":[0,0,1000,1000],"fully_visible":true,
"text":"current visible text","confidence":0.0,"right_button":{{"label":"button text",
"bounds":[0,0,1000,1000],"confidence":0.0}}}}]}}
Return an empty structures array when the geometry is not visible. Never merge a clipped structure with a complete structure.
"""


def _input_audit_detail_note(
    roi_bounds: tuple[int, int, int, int] | None,
) -> str:
    if roi_bounds is None:
        return "No detail crop is provided."
    return (
        f"Image 2 is only a magnified read-only crop of Image 1 at {list(roi_bounds)}. "
        "Use it to read details, but never use Image 2 as a coordinate system."
    )


def _parse_scene(
    raw: str,
    *,
    fingerprint: str,
    goal_context: dict[str, Any] | None = None,
) -> UIScene:
    try:
        payload = _extract_json_object(raw)
        _normalize_compact_scene_payload(payload)
        _normalize_prefilled_input_structure(payload, goal_context or {})
        _normalize_local_text_clear_structure(payload, goal_context or {})
        _normalize_unique_input_focus(payload)
        return UIScene.from_dict(
            payload,
            coordinate_scale=1000.0,
            stable_override=True,
            fingerprint_override=fingerprint,
        )
    except (UISceneError, ValueError, TypeError) as exc:
        raise VisionAgentError(f"通用页面观察结果不符合协议：{exc}") from exc


def _normalize_unique_input_focus(payload: dict[str, Any]) -> None:
    """Derive focus only from one target input plus visible soft keyboard facts."""

    elements = payload.get("elements")
    if not isinstance(elements, list):
        return
    candidates = [
        item
        for item in elements
        if isinstance(item, dict)
        and str(item.get("role") or "").strip() == "input"
        and isinstance(item.get("states"), dict)
        and item["states"].get("goal_relevant") is True
        and float(item.get("confidence") or 0.0) >= MIN_TARGET_CONFIDENCE
    ]
    if len(candidates) != 1:
        return
    visible_text = " ".join(
        [
            str(payload.get("summary") or ""),
            *(
                str(value)
                for value in (payload.get("overlays") or [])
                if isinstance(value, str)
            ),
            *(
                " ".join(
                    str(item.get(key) or "")
                    for key in ("role", "meaning", "label")
                )
                for item in elements
                if isinstance(item, dict)
            ),
        ]
    ).casefold()
    keyboard_visible = bool(
        re.search(
            r"(?:软键盘|输入法|键盘|keyboard|ime)",
            visible_text,
            re.IGNORECASE,
        )
    )
    if not keyboard_visible:
        return
    candidate = candidates[0]
    states = dict(candidate.get("states") or {})
    if states.get("focused") is False:
        # An explicit contradictory visual fact always wins.
        return
    states["focused"] = True
    candidate["states"] = states


def _normalize_local_text_clear_structure(
    payload: dict[str, Any],
    goal_context: dict[str, Any],
) -> None:
    """Bind one observed clear control to one non-empty input for clear goals.

    This normalization grants no action permission.  It only restores omitted
    goal-relevance facts when the model has already reported the complete
    high-confidence structure required by the controller.  Focus is still
    derived separately and only when the same scene also reports a visible
    soft keyboard.
    """

    if not _goal_requests_local_text_clear(goal_context):
        return
    elements = payload.get("elements")
    if not isinstance(elements, list):
        return

    all_inputs = [
        item
        for item in elements
        if isinstance(item, dict)
        and str(item.get("role") or "").strip() == "input"
        and isinstance(item.get("states"), dict)
        and isinstance(item["states"].get("value"), str)
        and bool(item["states"]["value"])
        and isinstance(item.get("confidence"), (int, float))
        and not isinstance(item.get("confidence"), bool)
        and float(item["confidence"]) >= 0.9
        and _valid_1000_bounds(item.get("bounds"))
    ]
    claimed_clear_controls = [
        item
        for item in elements
        if isinstance(item, dict)
        and str(item.get("role") or "").strip() in {"button", "icon"}
        and isinstance(item.get("states"), dict)
        and (
            str(item.get("meaning") or "").strip() == "clear_local_text"
            or item["states"].get("local_text_clear") is True
        )
    ]
    inputs = [
        item
        for item in all_inputs
        if item["states"].get("goal_relevant") is not False
    ]
    clear_controls = [
        item
        for item in claimed_clear_controls
        if item["states"].get("local_text_clear") is True
        and item["states"].get("goal_relevant") is not False
        and isinstance(item.get("confidence"), (int, float))
        and not isinstance(item.get("confidence"), bool)
        and float(item["confidence"]) >= 0.9
        and _valid_1000_bounds(item.get("bounds"))
        and not _has_cancel_semantics(item)
        and _has_exact_clear_glyph(item)
    ]
    matches: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for input_element in inputs:
        il, it, ir, ib = (float(value) for value in input_element["bounds"])
        input_height = max(1e-9, ib - it)
        for clear_control in clear_controls:
            cl, ct, cr, cb = (float(value) for value in clear_control["bounds"])
            clear_width = cr - cl
            clear_height = max(1e-9, cb - ct)
            vertical_overlap = max(0.0, min(ib, cb) - max(it, ct))
            gap = max(0.0, cl - ir)
            geometrically_bound = (
                vertical_overlap / clear_height >= 0.6
                and cl >= il + 0.4 * (ir - il)
                and cr <= min(1000.0, ir + 200.0)
                and gap <= max(40.0, input_height)
                and clear_width <= 2.0 * input_height
                and clear_height <= 1.5 * input_height
            )
            if geometrically_bound:
                matches.append((input_element, clear_control))

    if len(matches) == 1 and len(inputs) == 1 and len(clear_controls) == 1:
        input_element, clear_control = matches[0]
        input_element["states"] = dict(input_element["states"])
        input_element["states"]["goal_relevant"] = True
        clear_control["states"] = dict(clear_control["states"])
        clear_control["states"]["goal_relevant"] = True
        return

    # Fail closed and let targeted refinement re-observe the exact visual
    # structure. A model semantic claim alone grants no clear-button authority.
    for item in all_inputs:
        item["states"] = dict(item["states"])
        item["states"]["goal_relevant"] = False
        item["states"].pop("focused", None)
    for item in claimed_clear_controls:
        item["states"] = dict(item["states"])
        item["states"].pop("local_text_clear", None)
        item["states"]["goal_relevant"] = False


def _has_cancel_semantics(item: dict[str, Any]) -> bool:
    visible = " ".join(
        [
            str(item.get("meaning") or ""),
            str(item.get("label") or ""),
            *(str(value) for value in (item.get("evidence") or [])),
        ]
    ).casefold()
    return "取消" in visible or bool(re.search(r"\bcancel(?:led|ing)?\b", visible))


def _has_exact_clear_glyph(item: dict[str, Any]) -> bool:
    return str(item.get("label") or "").strip().casefold() in {"×", "✕", "✖", "x"}


def _goal_directed_roi_bounds(
    context: dict[str, Any],
) -> tuple[int, int, int, int] | None:
    """Select at most one coarse ROI from explicit spatial words in the goal."""

    visible = json.dumps(context, ensure_ascii=False).casefold()
    top = any(term in visible for term in ("顶部", "上方", "顶端", "top"))
    bottom = any(term in visible for term in ("底部", "下方", "底端", "bottom"))
    left = any(term in visible for term in ("左侧", "左边", "left"))
    right = any(term in visible for term in ("右侧", "右边", "right"))
    if top and bottom:
        top = bottom = False
    if left and right:
        left = right = False
    horizontal = (0, 1000)
    vertical = (0, 1000)
    if left:
        horizontal = (0, 560)
    elif right:
        horizontal = (440, 1000)
    if top:
        vertical = (0, 420)
    elif bottom:
        vertical = (580, 1000)
    if horizontal == (0, 1000) and vertical == (0, 1000):
        return None
    return horizontal[0], vertical[0], horizontal[1], vertical[1]


def _crop_normalized(
    image: Image.Image,
    bounds: tuple[int, int, int, int],
) -> Image.Image:
    left, top, right, bottom = bounds
    x0 = round(left * image.width / 1000)
    y0 = round(top * image.height / 1000)
    x1 = round(right * image.width / 1000)
    y1 = round(bottom * image.height / 1000)
    return image.crop((x0, y0, x1, y1))


def _roi_observation_note(
    bounds: tuple[int, int, int, int] | None,
) -> str:
    if bounds is None:
        return "本次仍提供完整手机画面；bounds相对于完整画面。"
    return (
        f"本次依次提供完整手机画面和根据目标明确方位词裁出的高清局部，局部在原图范围为{list(bounds)}。"
        "第一张只用于理解页面上下文；必须在第二张高清局部中重新辨认目标。"
        "第二张只提供放大细节，绝不能作为坐标系；所有bounds必须回到第一张完整手机画面，"
        "相对于第一张使用0..1000坐标。"
        "局部图只用于看清事实，不增加任何动作权限。"
    )


def _normalize_prefilled_input_structure(
    payload: dict[str, Any],
    goal_context: dict[str, Any],
) -> None:
    """Infer an input only from a strict model-reported field/button structure."""

    if not _goal_requests_input(goal_context):
        return
    elements = payload.get("elements")
    if not isinstance(elements, list) or any(
        isinstance(item, dict) and str(item.get("role") or "").strip() == "input"
        for item in elements
    ):
        return

    def trusted(item: Any, role: str) -> bool:
        return (
            isinstance(item, dict)
            and str(item.get("role") or "").strip() == role
            and isinstance(item.get("confidence"), (int, float))
            and not isinstance(item.get("confidence"), bool)
            and float(item["confidence"]) >= 0.9
            and _valid_1000_bounds(item.get("bounds"))
            and isinstance(item.get("states"), dict)
            and item["states"].get("fully_visible") is True
        )

    containers = [
        item
        for item in elements
        if trusted(item, "container")
        and isinstance(item.get("states"), dict)
        and item["states"].get("goal_relevant") is True
        and _has_any_semantic_term(item, ("search", "query", "input", "form", "搜索", "查询", "输入"))
    ]
    texts = [
        item
        for item in elements
        if trusted(item, "text")
        and isinstance(item.get("states"), dict)
        and item["states"].get("goal_relevant") is True
    ]
    buttons = [
        item
        for item in elements
        if trusted(item, "button")
        and isinstance(item.get("states"), dict)
        and item["states"].get("goal_relevant") is False
        and _has_any_semantic_term(item, ("search", "submit", "go", "搜索", "提交", "查找"))
    ]
    matches: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for container in containers:
        cb = tuple(float(value) for value in container["bounds"])
        if cb[2] - cb[0] < 240 or cb[3] - cb[1] > 300:
            continue
        for button in buttons:
            bb = tuple(float(value) for value in button["bounds"])
            button_inside_group = (
                _bounds_inside(bb, cb, tolerance=35)
                and bb[0] > cb[0] + 0.45 * (cb[2] - cb[0])
            )
            button_adjacent_right = (
                bb[0] >= cb[2] - 0.15 * (cb[2] - cb[0])
                and bb[0] <= cb[2] + 80
                and bb[2] > cb[2]
            )
            if not (button_inside_group or button_adjacent_right):
                continue
            if _vertical_overlap_ratio(bb, cb) < 0.65:
                continue
            for text in texts:
                tb = tuple(float(value) for value in text["bounds"])
                if not _bounds_inside(tb, cb, tolerance=35) or tb[2] > bb[0] + 20:
                    continue
                if _vertical_overlap_ratio(tb, cb) < 0.45:
                    continue
                matches.append((container, text, button))
    if len(matches) != 1:
        return
    container, text, button = matches[0]
    cb = tuple(float(value) for value in container["bounds"])
    bb = tuple(float(value) for value in button["bounds"])
    input_bounds = [round(cb[0]), round(cb[1]), round(min(cb[2], bb[0])), round(cb[3])]
    if input_bounds[2] - input_bounds[0] < 120:
        return
    normalized_input_box = tuple(float(value) for value in input_bounds)
    for item in elements:
        if not isinstance(item, dict) or item is button:
            continue
        raw_item_bounds = item.get("bounds")
        if item in (container, text) or (
            _valid_1000_bounds(raw_item_bounds)
            and _bounds_inside(
                tuple(float(value) for value in raw_item_bounds),
                normalized_input_box,
                tolerance=20,
            )
        ):
            item["states"] = dict(item.get("states") or {})
            item["states"]["goal_relevant"] = False
    label = str(text.get("label") or "").strip()[:200]
    evidence = []
    for source in (container, text, button):
        for value in source.get("evidence") or []:
            value = str(value).strip()
            if value and value not in evidence:
                evidence.append(value[:200])
    elements.append(
        {
            "element_id": "local_structured_input_1",
            "role": "input",
            "meaning": "prefilled_text_input",
            "label": label,
            "bounds": input_bounds,
            "confidence": min(
                float(container["confidence"]),
                float(text["confidence"]),
                float(button["confidence"]),
            ),
            "states": {"goal_relevant": True, "value": label},
            "evidence": evidence[:6],
        }
    )


def _valid_1000_bounds(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return False
    if not all(
        isinstance(part, (int, float)) and not isinstance(part, bool)
        for part in value
    ):
        return False
    left, top, right, bottom = (float(part) for part in value)
    return 0 <= left < right <= 1000 and 0 <= top < bottom <= 1000


def _goal_requests_input(context: dict[str, Any]) -> bool:
    visible = json.dumps(context, ensure_ascii=False).casefold()
    return any(
        term in visible
        for term in (
            "输入框",
            "搜索框",
            "编辑框",
            "地址栏",
            "字段进入编辑",
            "字段获得焦点",
            "字段内容",
            "input field",
            "search box",
            "text field",
            "editable field",
            "address bar",
            "textbox",
        )
    )


def _goal_requests_local_text_clear(context: dict[str, Any]) -> bool:
    if not _goal_requests_input(context):
        return False
    visible = json.dumps(context, ensure_ascii=False).casefold()
    return any(
        term in visible
        for term in (
            "清空",
            "清除",
            "置空",
            "文字变为空",
            "内容变为空",
            "clear text",
            "clear the text",
            "empty the input",
            "empty the field",
            "remove the text",
        )
    )


def _should_audit_prefilled_input(scene: UIScene, context: dict[str, Any]) -> bool:
    if not _goal_requests_input(context):
        return False
    return not any(
        item.role == "input"
        and item.states.get("goal_relevant") is True
        and float(item.confidence) >= 0.9
        for item in scene.elements
    )


def _apply_input_structure_audit(
    scene: UIScene,
    raw: str,
    *,
    fingerprint: str,
) -> UIScene:
    try:
        payload = _extract_json_object(raw)
        if set(payload) != {"structures"}:
            raise UISceneError("输入结构审计包含协议外字段。")
        structures = payload.get("structures")
        if not isinstance(structures, list) or len(structures) > 4:
            raise UISceneError("输入结构审计 structures 必须是最多4项的数组。")
        matches: list[dict[str, Any]] = []
        for item in structures:
            if not isinstance(item, dict) or set(item) != {
                "structure_id",
                "bounds",
                "fully_visible",
                "text",
                "confidence",
                "right_button",
            }:
                raise UISceneError("输入结构审计结构字段不符合协议。")
            button = item.get("right_button")
            if not isinstance(button, dict) or set(button) != {
                "label",
                "bounds",
                "confidence",
            }:
                raise UISceneError("输入结构审计按钮字段不符合协议。")
            if not isinstance(item.get("fully_visible"), bool):
                raise UISceneError("输入结构审计 fully_visible 必须是布尔值。")
            if not _valid_1000_bounds(item.get("bounds")) or not _valid_1000_bounds(
                button.get("bounds")
            ):
                raise UISceneError("输入结构审计 bounds 不符合0..1000协议。")
            if any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in (item.get("confidence"), button.get("confidence"))
            ):
                raise UISceneError("输入结构审计 confidence 格式无效。")
            confidence = float(item["confidence"])
            button_confidence = float(button["confidence"])
            if not 0.0 <= confidence <= 1.0 or not 0.0 <= button_confidence <= 1.0:
                raise UISceneError("输入结构审计 confidence 超出0..1。")
            if not item["fully_visible"] or confidence < 0.9 or button_confidence < 0.9:
                continue
            text = str(item.get("text") or "").strip()
            label = str(button.get("label") or "").strip()
            if not text or not label:
                continue
            bounds = tuple(float(value) for value in item["bounds"])
            button_bounds = tuple(float(value) for value in button["bounds"])
            width = bounds[2] - bounds[0]
            height = bounds[3] - bounds[1]
            if (
                bounds[1] <= 10
                or bounds[3] >= 990
                or width < 240
                or not 20 <= height <= 180
                or not _bounds_inside(button_bounds, bounds, tolerance=20)
                or button_bounds[0] <= bounds[0] + 0.55 * width
                or _vertical_overlap_ratio(button_bounds, bounds) < 0.8
            ):
                continue
            input_bounds = [
                round(bounds[0]),
                round(bounds[1]),
                round(button_bounds[0]),
                round(bounds[3]),
            ]
            if input_bounds[2] - input_bounds[0] < 120:
                continue
            matches.append(
                {
                    "text": text,
                    "button_label": label,
                    "input_bounds": input_bounds,
                    "button_bounds": [round(value) for value in button_bounds],
                    "confidence": min(confidence, button_confidence),
                }
            )
        if len(matches) != 1:
            return scene
        match = matches[0]
        value = scene.to_dict()
        elements = list(value.get("elements") or [])
        for element in elements:
            if isinstance(element, dict):
                element["states"] = dict(element.get("states") or {})
                element["states"]["goal_relevant"] = False
        elements.extend(
            [
                {
                    "element_id": "local_audited_input_1",
                    "role": "input",
                    "meaning": "prefilled_text_input",
                    "label": match["text"],
                    "bounds": [value / 1000.0 for value in match["input_bounds"]],
                    "confidence": match["confidence"],
                    "states": {"goal_relevant": True, "fully_visible": True},
                    "evidence": [
                        f"完整横向输入结构，当前文字：{match['text']}",
                        f"右侧独立按钮：{match['button_label']}",
                    ],
                },
                {
                    "element_id": "local_audited_adjacent_button_1",
                    "role": "button",
                    "meaning": "adjacent_submit_button",
                    "label": match["button_label"],
                    "bounds": [value / 1000.0 for value in match["button_bounds"]],
                    "confidence": match["confidence"],
                    "states": {"goal_relevant": False, "fully_visible": True},
                    "evidence": ["输入结构审计中的相邻独立按钮；不具备目标权限"],
                },
            ]
        )
        value["elements"] = elements
        return UIScene.from_dict(
            value,
            coordinate_scale=1.0,
            stable_override=True,
            fingerprint_override=fingerprint,
        )
    except (UISceneError, ValueError, TypeError) as exc:
        raise VisionAgentError(f"输入结构只读审计结果不符合协议：{exc}") from exc


def _has_any_semantic_term(item: dict[str, Any], terms: tuple[str, ...]) -> bool:
    visible = " ".join(
        str(item.get(key) or "") for key in ("meaning", "label")
    ).casefold()
    return any(term in visible for term in terms)


def _bounds_inside(
    inner: tuple[float, float, float, float],
    outer: tuple[float, float, float, float],
    *,
    tolerance: float,
) -> bool:
    return (
        inner[0] >= outer[0] - tolerance
        and inner[1] >= outer[1] - tolerance
        and inner[2] <= outer[2] + tolerance
        and inner[3] <= outer[3] + tolerance
    )


def _vertical_overlap_ratio(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    overlap = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    smaller = min(first[3] - first[1], second[3] - second[1])
    return overlap / smaller if smaller > 0 else 0.0


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
    if scene.confidence < 0.72:
        return True
    if any(element.states.get("goal_relevant") is True for element in scene.elements):
        return False
    if scene.screen_id == "unknown":
        return True

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
