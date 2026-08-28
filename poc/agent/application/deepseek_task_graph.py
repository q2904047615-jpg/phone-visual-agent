from __future__ import annotations

import json
import uuid
from dataclasses import replace
from typing import Any, Callable, Protocol

from agent.domain.generic_goal import GenericIntentError, _parse_json_object
from agent.domain.task_semantic_ir import (
    SemanticRiskAuthorityReport,
    TaskSemanticIRError,
    LocalRiskPolicyConfig,
    apply_formal_semantic_risk_policy,
    compile_formal_semantic_authority,
)
from agent.domain import task_graph as task_graph_domain
from agent.domain.task_graph import (
    DynamicTaskGraph,
    ObservedState,
    REPLAN_TRIGGERS,
    ReplanRecord,
    TaskGraphError,
    _apply_verified_navigation_completion,
    _canonicalize_literal_visible_evidence_clauses,
    _graph_from_payload,
    _normalize_explicit_target_surface,
    _normalize_explicit_ui_label_payload,
    _normalize_initial_premature_completed_status,
    _normalize_local_navigation_execution_classes,
    _normalize_redundant_prohibited_effect_conditions,
    _normalize_single_effect_result_string,
    _normalize_terminal_single_navigation_payload,
    _normalize_unique_active_frontier,
    _normalize_unique_planner_transport_aliases,
    _planner_transport_snapshot,
    _project_terminal_single_navigation_candidate,
    _require_text,
    _restore_completed_history_evidence,
    _validate_device_id,
    _validate_task_id,
)

__all__ = ("DeepSeekTaskGraphPlanner", "JsonTaskGraphProvider")

class JsonTaskGraphProvider(Protocol):
    configured: bool

    def chat_json(self, messages: list[dict[str, Any]], max_tokens: int = 2000) -> str: ...

class DeepSeekTaskGraphPlanner:
    """Create and revise high-level task graphs without any execution capability."""

    def __init__(self, provider: JsonTaskGraphProvider, *,
        semantic_risk_policy: LocalRiskPolicyConfig | None=None) -> None:
        self.provider = provider
        self.semantic_risk_policy = (
            semantic_risk_policy if semantic_risk_policy is not None else LocalRiskPolicyConfig()
        )
        self.last_raw_response = ""
        self.last_semantic_authority: SemanticRiskAuthorityReport | None = None
        self.last_semantic_authority_error = ""

    def plan(
        self,
        raw_goal: str,
        *,
        device_id: str,
        task_id: str | None = None,
    ) -> DynamicTaskGraph:
        # Internal whitespace can be literal user payload.  In particular, a
        # line feed is an authorized input character that must survive into
        # the typed graph and semantic source spans unchanged.
        text = str(raw_goal or "").strip()
        if not text:
            raise TaskGraphError("用户目标不能为空。")
        _validate_device_id(device_id)
        resolved_task_id = task_id or uuid.uuid4().hex
        _validate_task_id(resolved_task_id)
        self._reset_semantic_authority()
        self._require_provider()
        prompt = _initial_prompt(text)
        graph = self._request_graph(prompt, task_id=resolved_task_id, device_id=device_id, revision=1,
            raw_user_goal=text, validate=False)
        # Only deterministic, semantics-preserving local normalization is
        # allowed.  A malformed or unsafe semantic answer is never repaired by
        # another remote sample.
        graph = _normalize_explicit_target_surface(graph, text)
        graph = _normalize_redundant_prohibited_effect_conditions(graph)
        graph = _normalize_initial_premature_completed_status(graph)
        graph = _normalize_unique_active_frontier(graph)
        # Reject malformed planner transport before semantic cutover so the
        # formal projector never masks missing IDs, invalid enums or broken
        # graph structure with a later migration error.
        graph.validate()
        graph = self._apply_formal_semantic_authority(graph)
        graph.validate()
        if (graph.status == 'completed' or any((item.status == 'completed' for item in graph.subgoals))
            or any((item.satisfied for item in graph.completion_conditions))):
            raise TaskGraphError("初始规划没有观察证据，不能宣称目标或子目标已完成。")
        return graph

    def replan(self, graph: DynamicTaskGraph, observation: ObservedState, *, trigger: str,
        reason: str) -> DynamicTaskGraph:
        graph.validate()
        observation.validate()
        if trigger not in REPLAN_TRIGGERS:
            raise TaskGraphError(f"不支持的重规划触发原因：{trigger}")
        _require_text(reason, "replan.reason")
        self._reset_semantic_authority()
        self._require_provider()
        prompt = _replan_prompt(graph, observation, trigger=trigger, reason=reason)
        candidate = self._request_graph(prompt, task_id=graph.task_id, device_id=graph.device_id,
            revision=graph.revision + 1, raw_user_goal=graph.raw_user_goal or graph.goal.objective, validate=False,
            payload_normalizer=lambda payload: _normalize_terminal_single_navigation_payload(graph, observation,
            trigger=trigger, payload=payload))
        candidate = _normalize_redundant_prohibited_effect_conditions(candidate)
        candidate = _restore_completed_history_evidence(graph, candidate)
        candidate = _canonicalize_literal_visible_evidence_clauses(graph, candidate, observation)
        candidate = _project_terminal_single_navigation_candidate(graph, candidate, observation, trigger=trigger)
        candidate = _apply_verified_navigation_completion(graph, candidate, observation, trigger=trigger)
        candidate = _normalize_unique_active_frontier(candidate)
        # Validate the raw typed transport once before local projection.  The
        # revision-specific execution-class and EffectIntent invariants are
        # owned by _validate_replan_candidate below and must not be duplicated
        # on the same candidate.
        candidate.validate()
        candidate = self._apply_formal_semantic_authority(candidate)
        self._validate_replan_candidate(graph, candidate, observation, trigger=trigger)
        previous_ids = {item.subgoal_id for item in graph.subgoals}
        completed_ids = tuple((item.subgoal_id for item in graph.subgoals if item.status == 'completed'))
        added_ids = tuple((item.subgoal_id for item in candidate.subgoals if item.subgoal_id not in previous_ids))
        skipped_ids = tuple((item.subgoal_id for item in candidate.subgoals if item.status == 'skipped'
            and next((old.status for old in graph.subgoals if old.subgoal_id == item.subgoal_id), None) != 'skipped'))
        record = ReplanRecord(
            revision=candidate.revision,
            trigger=trigger,
            reason=reason.strip(),
            scene_id=observation.scene_id,
            evidence=observation.visible_evidence,
            retained_completed_subgoal_ids=completed_ids,
            added_subgoal_ids=added_ids,
            skipped_subgoal_ids=skipped_ids,
            consumed_action_transition_receipt_id=(
                observation.verified_action_transition.receipt_id
                if observation.verified_action_transition is not None
                and trigger in {
                    "action_result_matched",
                    "action_result_mismatch",
                }
                else ""
            ),
        )
        revised = replace(candidate, replan_history=graph.replan_history + (record,))
        revised.validate()
        return revised

    def _apply_formal_semantic_authority(self, graph: DynamicTaskGraph) -> DynamicTaskGraph:
        """Apply typed field roles and local confirmation policy fail-closed."""

        self.last_semantic_authority = None
        self.last_semantic_authority_error = ""
        try:
            authority = compile_formal_semantic_authority(graph, risk_policy=self.semantic_risk_policy)
            projected = apply_formal_semantic_risk_policy(graph, authority)
        except TaskSemanticIRError as exc:
            self.last_semantic_authority_error = str(exc)[:1000]
            raise TaskGraphError(f"正式语义风险权威拒绝任务图：{exc}") from exc
        self.last_semantic_authority = authority
        return projected

    def _reset_semantic_authority(self) -> None:
        self.last_semantic_authority = None
        self.last_semantic_authority_error = ""

    def _validate_replan_candidate(self, graph: DynamicTaskGraph, candidate: DynamicTaskGraph,
        observation: ObservedState, *, trigger: str) -> None:
        """Apply every safety and evidence check to one replan candidate."""

        if candidate.revision != graph.revision + 1:
            raise TaskGraphError('重规划 revision 必须严格等于上一 revision + 1。')
        transition = observation.verified_action_transition
        if trigger in {'action_result_matched', 'action_result_mismatch'}:
            if transition is None:
                raise TaskGraphError("动作结果重规划缺少本地 verified action transition。")
            expected_outcome = 'matched' if trigger == 'action_result_matched' else 'mismatched'
            if transition.outcome != expected_outcome:
                raise TaskGraphError("重规划触发与本地动作转换回执 outcome 不一致。")
            previous_current = graph.active_subgoal()
            if (transition.task_id != graph.task_id or transition.device_id != graph.device_id
                or transition.prior_revision != graph.revision or (transition.subgoal_id != graph.active_subgoal_id)
                or (transition.after_observation_id != observation.scene_id) or (previous_current is None)):
                raise TaskGraphError("动作转换回执未严格绑定上一任务图及当前观察。")
            consumed_receipts = {item.consumed_action_transition_receipt_id for item
                in graph.replan_history if item.consumed_action_transition_receipt_id}
            if transition.receipt_id in consumed_receipts:
                raise TaskGraphError("动作转换回执已经消费，禁止跨 revision 重放。")
        elif trigger == 'observation_changed' and transition is not None:
            raise TaskGraphError("纯观察变化不得携带动作执行回执。")

        task_graph_domain._validate_execution_class_revision(graph, candidate)
        task_graph_domain._validate_preserved_effect_intents(graph, candidate)
        candidate.validate()
        previous_current = graph.active_subgoal()
        candidate_current = candidate.active_subgoal()
        if (trigger == 'action_result_mismatch' and previous_current is not None and (next((item.status for item
            in candidate.subgoals if item.subgoal_id == previous_current.subgoal_id), None) == 'completed')):
            raise TaskGraphError('动作结果不匹配时不能完成回执绑定的上一活动子目标。')
        if (trigger == 'subgoal_completed' and previous_current is not None
            and (previous_current.external_impact == 'read_only') and (candidate_current is not None)
            and (candidate_current.external_impact == 'read_only')):
            raise TaskGraphError('read_only 完成复核不能继续保留 read_only 活动子目标；当前证据足够时应完成，证据不足时应阻塞，或推进到后续非只读子目标。')
        task_graph_domain._validate_revision(graph, candidate, observation)

    def _request_graph(self, prompt: str, *, task_id: str, device_id: str, revision: int, raw_user_goal: str,
        validate: bool=True, payload_normalizer: Callable[[dict[str, Any]], dict[str,
        Any]] | None=None) -> DynamicTaskGraph:
        raw = self.provider.chat_json([{'role': 'user', 'content': prompt}], max_tokens=2400)
        self.last_raw_response = raw
        try:
            payload = _parse_json_object(raw)
        except GenericIntentError as exc:
            raise TaskGraphError(str(exc)) from exc
        payload = _normalize_single_effect_result_string(payload)
        payload = _normalize_unique_planner_transport_aliases(payload)
        payload = _normalize_explicit_ui_label_payload(payload, raw_user_goal)
        payload = _normalize_local_navigation_execution_classes(payload)
        if payload_normalizer is not None:
            payload = payload_normalizer(payload)
        graph = _graph_from_payload(payload, task_id=task_id, device_id=device_id, revision=revision,
            raw_user_goal=raw_user_goal)
        if validate:
            graph.validate()
        return graph

    def _require_provider(self) -> None:
        if not self.provider.configured:
            raise TaskGraphError("DeepSeek 动态任务图尚未配置。")

def _initial_prompt(raw_goal: str) -> str:
    return f"""
你是通用手机视觉操作 Agent 的 DeepSeek 高层任务图规划器。你只维护目标和高层子目标，
不观察图片、不选择控件。用户可以直接要求点击、滑动、输入、长按、拖动、返回或回到主页；
这些自然动作意图可以原样进入目标和子目标，但它们绝不构成执行授权。你不能输出坐标、
Shell、ADB、keycode、main.exe 指令或其他可直接驱动设备的控制细节。

用户原始目标：{json.dumps(raw_goal, ensure_ascii=False)}

{_schema_prompt()}

初始规划规则：
1. 适用于任意 App 和跨 App 目标，不得生成任何 App 专用固定流程。
2. 子目标可以同时保留“用户要做什么动作”和“动作后必须出现什么状态”。动作名称只是任务语义，
   不能携带坐标、控件索引、系统命令或可执行脚本；实际下一步控件和动作仍由 Qwen 基于真实画面
   提议，并由本地控制器以独立观察、作用域和能力门禁裁决。用户对方向、次数、文字原文和禁止事项
   必须逐字保留，不得为了满足协议而改写成另一项任务。
   如果用户明确指“当前页面”“当前应用”或“当前前台”但没有说 App 名称，target_apps 使用
   [{{"app_id":"current_foreground","app_name":"当前前台应用"}}]；不能只因未重复 App 名称而阻塞。
   如果目标界面的字面标签含动作词，可将逐字标签保存在 goal.entities.target_ui_label；同一个动作词
   也可以出现在 objective，但两者含义必须分开。entities 可使用稳定、描述角色的任意键保存
   用户明确给出的对象、内容、字段、文件、日期或其他 JSON 值；本地会把每项编译成 typed entity，
   只有带明确 role、用户字面来源和显式 relation/effect binding 的 entity 才能进入动作 authority，
   未绑定键只能作 planner context，模型不能借它扩大 Qwen 或控制器权限。recipient/input_text、
   recipients/input_fields、target_ui_label、target_surface、spatial_hint 是通用常见结构，不是封闭白名单。
   多字段输入时，input_fields 每项使用 field_id、field_label、text：field_id 是稳定ASCII身份，
   field_label 必须逐字复制该字段在页面上的可见标签或占位提示，text 是用户要求写入的逐字正文；
   不得用“第一个/第二个”替代可见字段标签。每个写入子目标只能逐字引用并绑定其中一个字段标签和
   对应 text；多个字段必须拆成有依赖关系的多个写入子目标，最后再用独立 observe 子目标同时逐字
   核对所有字段，不能把多个字段合并成一个可执行输入子目标。
   device/system/current_surface 目标可将 target_apps 留空并设置 target_surface；target_surface 只能放在
   goal.entities 内，禁止作为 goal 的直辖字段；
   App 目标仍应使用 target_apps。
3. 只能有一个 active 子目标；其依赖必须已经 completed（初始图通常无依赖）。
4. 初始规划没有画面证据，所有完成条件 satisfied=false，任何子目标都不能 completed。
5. 每个子目标只用 execution_class 标为 observe、navigate、effect 或 unknown；不得输出内部运行态名称
   read_only、navigation_only 或 external_state；模型不得输出风险等级、
   confirmation_required、external_impact 或 risk_actions。真正产生外部结果的子目标必须声明 typed
   effect_intents，并让 expected_results 逐字引用该子目标的正向完成条件；若整个任务只有一个
   effect_intent，也可逐字引用唯一的最终正向完成条件。禁止、未发生、保持不变、
   按钮可见但未触发等约束或状态不得声明为 effect。纯粹的“不要发送、未提交、未保存、未登录”等
   效果禁令只保留在 constraints，不得再重复建立 completion_conditions；completion_conditions 只写
   最终需要由当前画面或正式效果回执证明的正向结果。确认只由 EffectIntent.kind 与本地版本化策略决定。
6. observe 只能描述查看、读取、检查等纯观察结果；navigate 只能描述打开或进入页面等
   导航结果。仅改变本机临时界面层级、前后台页面或临时标签页也属于 navigation_only，不得为它
   虚构 effect_intents；系统最近任务/任务概览中划掉、关闭或移除应用预览卡片，只改变本机临时
   任务层级，也属于 navigate，不得为卡片对应 App 的内容虚构 effect_intents；关闭或删除 App 内
   真实数据仍属于 effect。登录/退出账号、修改账号数据或云端同步状态也仍属于 effect。
   当用户要查看、读取或核对某个目标页面的结果，但没有明确说明该结果已经在当前画面中时，
   必须先建立一个 navigation_only 子目标描述“目标页面或目标区域可见”，再建立 read_only 子目标
   描述要核对的结果；不得把潜在导航需求隐藏在单个 read_only 子目标中。
   如果 goal.entities.target_ui_label 是具名入口/分类，而最终完成条件要求另一个结果文字或状态，
   必须再拆分为“具名入口在列表中可见”与“入口对应的目标页面可见”两个 navigation_only 状态，
   最后才是 read_only 结果核对。入口可见绝不能证明其对应页面或最终结果已经可见。
   只改变当前可见输入框中的未提交临时文字，也可归入 navigation_only。写入文字时目标文字必须
   明确非空；将当前唯一未发送/未提交临时草稿恢复为空白时，必须把空白状态写成明确结果而不能
   虚构空字符串 input_text。两者都要求用户直接禁止该上下文中的搜索、提交、发送、保存或发布等
   效果，且句中没有任何未被否定的外部效果。输入并搜索/发送/保存、清除云端或已保存数据、未明确
   禁止提交效果、或含义不清时仍必须标为 effect 或 unknown。
   不能证明属于这些安全类别时必须标为 unknown，不能为了免确认而猜成安全类别。
7. effect_intents.kind 只用 schema 中的跨 App typed effect，不得描述 App 页面路径；target/payload role
   必须引用 goal.entities 中实际存在的用户字面实体。
   用户指定已有收件人和文字消息时，必须把收件人逐字写入 goal.entities.recipient，把消息原文
   逐字写入 goal.entities.input_text；不得翻译、纠错、补标点或改写。打开目标 App、查找并进入
   已有收件人的聊天页面属于 navigate，但相关子目标和完成条件必须逐字包含 canonical
   recipient，供本地唯一身份核对。只在输入框保留未发送草稿也属于 navigation_only；真正发送
   才是 effect，send_message 只能关联发送子目标，不能提前污染 App 导航、收件人定位
   或未提交草稿准备。联系人重名、身份不唯一或缺少消息原文时必须 blocked 并提出澄清问题。
8. 信息不足时 status=blocked、active_subgoal_id=null，并填写 clarification_questions。
9. 一次给出完整、严格 JSON。不要 Markdown，也不要要求通过第二次远程采样修复格式。
"""

def _replan_prompt(graph: DynamicTaskGraph, observation: ObservedState, *, trigger: str, reason: str) -> str:
    return f"""
你是通用手机视觉操作 Agent 的 DeepSeek 高层任务图重规划器。根据新的只读观察，返回修订后的
完整高层任务图快照。可以保留用户的自然点击、滑动、输入、长按和拖动意图，但不能输出
具体控件选择、坐标、按键码、Shell、ADB、main.exe 或其他可直接驱动设备的细节。

 当前正式任务计划（仅新 typed transport，不含运行时风险投影视图）：
 {json.dumps(_planner_transport_snapshot(graph), ensure_ascii=False)}

重规划触发：{json.dumps(trigger, ensure_ascii=False)}
重规划原因：{json.dumps(reason, ensure_ascii=False)}
新的只读观察：
{json.dumps(observation.to_dict(), ensure_ascii=False)}

{_schema_prompt()}

重规划规则：
1. goal 必须逐字段保持不变；constraints 必须保留已有约束，可追加新发现的约束。纯粹的禁止效果或
   “某动作未发生”继续只作为 constraints，不得新增或满足同义 completion_conditions；正向最终状态
   仍必须使用后述 typed evidence。
2. 已 completed 的子目标必须原样保留且仍为 completed；已满足的全局条件不得撤销。
3. 可修改、跳过或替换尚未完成的子目标，并新增子目标；不要坚持已失效的旧路径。
4. 新宣称 completed/satisfied 时，如果 visual_claim_evidence_refs 非空，视觉证据必须逐字复制
   其中的 ref_id；只有旧观察没有 typed visual refs 时才允许逐字复制 visible_evidence。
   历史完成节点继续保留自己的历史证据。
5. 既有 effect_intents 必须原样保留，除非其 source subgoal 已由可信视觉结果完成；不得输出风险等级、
   confirmation_required、risk_actions、risk_action_ids 或 external_impact。
6. effect 子目标必须引用 typed effect_intents；unknown 不得成为 active。确认状态由本地策略生成，模型
   不得返回 awaiting_confirmation。每轮只选择一个 active 高层子目标；不要提出下一视觉动作。
7. 既有 effect 不能降级，unknown 没有新的可靠证据时不能改成 observe 或 navigate；observe/navigate
   必须分别有纯观察或纯导航依据。
   trigger=observation_changed 且当前 read_only 结果无法由新画面直接证明时，如果目标页面或区域
   尚未出现，应把未完成路径改写为先达到 navigation_only 的目标页面可见状态，再保留后续
   read_only 结果核对；可以保留用户原始导航动作意图，但不得写入具体控件或坐标，也不得凭空
   宣称结果完成。
8. 只返回 JSON 对象，不要 Markdown，也不要返回 task_id、device_id、revision、协议版本、
    current_subgoal 或历史记录；这些字段由本地协议层生成。
9. 当 trigger=subgoal_completed 且当前子目标是 read_only 时，本轮必须用当前 typed visual claim 完成
   该只读子目标及匹配的全局条件，或明确阻塞，或推进到后续非只读子目标；不得继续保留任何
   read_only 活动子目标，避免只读复核再次请求视觉动作或形成循环。
10. visual_claim_evidence_refs[].fact 仅用于理解当前事实，输出证据必须选择对应 ref_id，不能复制 fact。
    completion_conditions[].evidence 只能选择 visual_claim_evidence_refs[].ref_id；旧观察没有该数组时
    才兼容 visible_evidence 完整短字符串。subgoals[].completion_evidence 也遵守同一规则；唯一例外是当前严格绑定的
     navigate 旧子目标可选择 controller_transition_evidence_refs[].ref_id。每个数组最多3项，
    不得拼接多项、不得复制整个观察对象或 JSON。没有匹配证据时保持未完成或阻塞。
11. verified_action_transition 是本地控制器生成、严格绑定上一 revision/子目标/决策/动作和
    前后观察的动作回执；它与 visible_evidence 分离，不能当作页面可见事实或全局完成证据。
    outcome=matched 只证明该受控动作已执行并获得匹配验证，不代表任意子目标自动完成。
12. trigger=action_result_matched 时，可以结合回执和当前 visual claim 完成其严格绑定的
     navigate 旧子目标，或推进到不同的剩余状态目标；若证据不足，应明确重写剩余目标
     或阻塞。不得让同一活动子目标原样存活后再次请求等价动作。effect/unknown 不能
    仅凭回执完成，仍必须由当前 visual claim 证明真实外部结果。
    如果旧子目标要求目标页面/结果区域可见，而新画面只出现了具名入口或分类项，绝不能完成旧
    子目标；应把未完成路径修订为先达到“具名入口可见”的 navigation_only 状态，再保留目标页面
    和结果核对状态。控制器回执只证明本轮受控动作及其可见变化，不能把入口冒充结果页面。
13. trigger=action_result_mismatch 时，不得完成回执绑定的旧子目标；必须根据当前画面重规划、
    阻塞或提出高层澄清。revision 必须严格增加 1。
14. 任何包含具名页面、卡片、区域或结果身份的 completed/satisfied 声明，其名称必须
    能从 grounded_visual_facts 的 screen_id、overlay 或可见元素 label/meaning 中找到结构化支持。
    visible_evidence 或 summary 中的自由文本描述不能单独证明具名身份。若旧路径使用了
    未被结构化画面支持的具名页面，不得硬完成；应跳过或替换该未完成节点，
    改为基于 grounded_visual_facts 中实际可见的结构化状态（例如唯一目标输入元素可见）
    继续高层规划；不得把入口名称、动作成功或场景变化冒充为目标页面身份。
"""

def _schema_prompt() -> str:
    return """JSON 只允许以下结构：
{
  "status":"ready|running|completed|blocked",
  "goal":{
    "objective":"用户目标，可保留点击、滑动、输入、长按、拖动等自然动作意图，但不得含坐标或系统命令",
    "target_apps":[{"app_id":"稳定小写英文ID","app_name":"App名称；设备或当前表面任务可为空数组"}],
    "entities":{"recipient":"发送文字消息时逐字复制用户指定收件人；否则省略","input_text":"需要输入时逐字复制原文；否则省略","target_ui_label":"具名字面目标；否则省略","target_surface":"仅 device|system|current_surface；App任务省略","spatial_hint":"仅用户明确给出的上中下左右提示；否则省略","其他键":"仅供规划说明，不能扩大执行权限"}
  },
  "constraints":["全局约束"],
  "completion_conditions":[{
    "condition_id":"小写稳定ID",
    "description":"最终完成条件",
    "evidence_required":["需要从画面看到的事实"],
    "satisfied":false,
    "evidence":[]
  }],
  "effect_intents":[{
    "effect_id":"小写稳定ID",
    "kind":"send_message|publish_content|relationship_change|membership_change|data_mutation|authentication|financial_transaction|sensitive_permission_change|irreversible_account_deletion|irreversible_data_deletion",
    "target_entity_roles":["goal.entities 中作为效果对象的键"],
    "payload_entity_roles":["goal.entities 中作为效果正文或值的键"],
    "source_subgoal_ids":["实际产生该效果的子目标ID"],
    "expected_results":["逐字复制绑定子目标中的正向完成条件；单一效果任务也可复制唯一最终正向完成条件"]
  }],
  "subgoals":[{
    "subgoal_id":"小写稳定ID",
    "objective":"要执行的自然动作意图及其可验证后置状态",
    "status":"pending|active|completed|blocked|skipped",
    "depends_on":["前置子目标ID"],
    "constraints":["本子目标约束"],
    "completion_conditions":["本子目标完成条件"],
    "completion_evidence":[],
    "effect_ids":["本子目标实际产生的 effect_id；纯观察或导航为空"],
    "execution_class":"observe|navigate|effect|unknown"
  }],
  "active_subgoal_id":"活动子目标ID或null",
  "clarification_questions":["阻塞时需要用户补充的信息"]
}"""
