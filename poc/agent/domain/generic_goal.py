from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from .canonical_action_kinds import expected_idempotent_system_surface_kind
from .task_graph import DynamicTaskGraph, _named_visual_identity_anchor, named_visual_identity_is_grounded
from .task_semantic_ir import TaskSemanticIRError, compile_formal_semantic_authority
from .ui_scene import MIN_TARGET_CONFIDENCE, UISceneError, scene_matches_target_app_surface, scene_surface_kind
from .vision_model import VisionAgentError


GOAL_PROJECTION_PROTOCOL = "2026-08-20-typed-goal-projection-v1"
APP_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")


class GenericIntentError(ValueError):
    pass


@dataclass(frozen=True)
class GenericIntentDraft:
    """Read-only projection of the authoritative typed task graph.

    This carrier has no risk or action veto.  Risk is decided by the typed
    policy and action availability by the canonical action catalog.
    """

    understood: bool
    app_id: str = ""
    app_name: str = ""
    objective: str = ""
    entities: dict[str, Any] = field(default_factory=dict)
    constraints: tuple[str, ...] = ()
    success_criteria: dict[str, Any] = field(default_factory=dict)
    account_effects: tuple[str, ...] = ()
    message: str = ""
    needs_confirmation: bool = False
    protocol_version: str = GOAL_PROJECTION_PROTOCOL

    def validate(self) -> None:
        if not isinstance(self.understood, bool):
            raise GenericIntentError("understood 格式无效。")
        if not self.understood:
            if not self.message.strip():
                raise GenericIntentError("未理解任务时必须说明缺少的信息。")
            return
        if not APP_ID_PATTERN.fullmatch(self.app_id):
            raise GenericIntentError(f"App ID 无效：{self.app_id!r}")
        if not self.app_name.strip() or not self.objective.strip():
            raise GenericIntentError("通用任务缺少 App 名称或目标。")
        if not isinstance(self.needs_confirmation, bool):
            raise GenericIntentError("needs_confirmation 格式无效。")
        _validate_json_value(self.entities, "entities")
        _validate_json_value(self.success_criteria, "success_criteria")
        for value in (*self.constraints, *self.account_effects):
            if not isinstance(value, str) or not value.strip():
                raise GenericIntentError("约束和账号影响必须是非空字符串。")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        value = asdict(self)
        value["constraints"] = list(self.constraints)
        value["account_effects"] = list(self.account_effects)
        return value


_INPUT_TERMS = ('输入框', '文本框', '搜索框', '编辑框', '地址栏', '字段进入编辑', '字段获得焦点', '字段内容', '草稿区域', '草稿字段', 'input field',
    'search box', 'text field', 'editable field', 'draft field', 'draft area', 'address bar', 'textbox', 'input_text',
    '输入模式', '直输模式', '键盘模式', '软键盘', 'input mode', 'keyboard mode', 'soft keyboard', ' ime ', 'direct_latin',
    'chinese_pinyin')
_MODE_SWITCH_TERMS = ('切换输入模式', '输入模式切换', '切换到英文', '切到英文', '英文直输', '切换到中文', '切到中文', '切换直输模式', '切换为直输模式',
    'switch input mode', 'switch keyboard mode', 'direct_latin', 'chinese_pinyin')
_CLEAR_TERMS = ('清空', '清除', '置空', '删除', '文字变为空', '内容变为空', '恢复为空', '恢复为空白', 'clear text', 'clear the text',
    'clear draft', 'empty the input', 'empty the field', 'remove the text', 'delete')
_ACTIVE_VISUAL_FIELDS = {'subgoal_id', 'objective', 'constraints', 'completion_conditions', 'execution_class',
    'goal_entities'}


@dataclass(frozen=True)
class ActiveVisualGoal:
    """Read-only view of the current typed graph node for visual observation."""

    root: dict[str, Any]
    focus: dict[str, Any]
    root_entities: dict[str, Any]
    goal_entities: dict[str, Any]
    has_active_focus: bool

    @classmethod
    def from_context(cls, context: dict[str, Any]) -> 'ActiveVisualGoal':
        root_entities = context.get("entities")
        root_entities = root_entities if isinstance(root_entities, dict) else {}
        candidate = root_entities.get("active_subgoal_visual_context")
        valid = bool(isinstance(candidate, dict) and set(candidate) == _ACTIVE_VISUAL_FIELDS
            and str(candidate.get('subgoal_id') or '').strip() and str(candidate.get('objective') or '').strip()
            and isinstance(candidate.get('constraints'), list) and isinstance(candidate.get('completion_conditions'),
            list) and isinstance(candidate.get('goal_entities'), dict))
        focus = candidate if valid else context
        entities = focus.get("goal_entities") if valid else root_entities
        return cls(context, focus, root_entities, entities if isinstance(entities, dict) else {}, valid)

    @property
    def observation_context(self) -> dict[str, Any]:
        if not self.has_active_focus:
            return self.root
        return {key: list(value) if key in {'constraints',
            'completion_conditions'} else dict(value) if key == 'goal_entities' else value for key,
            value in self.focus.items()}

    @property
    def explicit_text(self) -> str:
        value = self.goal_entities.get("input_text")
        return value.strip() if isinstance(value, str) else ""

    @property
    def transaction_text(self) -> str:
        value = self.goal_entities.get("active_input_transaction_text") if self.has_active_focus else None
        return value if isinstance(value, str) and value else ""

    @property
    def field(self) -> tuple[str, str, bool]:
        if not self.has_active_focus:
            return "", "", False
        field_id = self.goal_entities.get("active_input_field_id")
        label = self.goal_entities.get("active_input_field_label", "")
        multiline = self.goal_entities.get("active_input_multiline", False)
        if (not isinstance(field_id, str) or not field_id or (not isinstance(label, str)) or (not isinstance(multiline,
            bool))):
            return "", "", False
        return field_id, label, multiline

    @property
    def target_only(self) -> bool:
        return bool(self.has_active_focus and self.goal_entities.get('active_input_target_only') is True
            and self.field[0])

    @property
    def unique_typed_field(self) -> bool:
        fields = self.root_entities.get("input_fields")
        field_id, label, _ = self.field
        text = self.transaction_text
        if not isinstance(fields, list) or not field_id or (not label) or (not text):
            return False
        exact = sum((isinstance(item, dict) and item.get('field_id') == field_id and (item.get('field_label') == label)
            and (item.get('text') == text) for item in fields))
        labels = sum(isinstance(item, dict) and item.get("field_label") == label for item in fields)
        return exact == labels == 1

    @property
    def predecessor(self) -> tuple[str, str, str]:
        if not self.has_active_focus:
            return "", "", ""
        values = tuple((self.goal_entities.get(key) for key in ('active_input_predecessor_field_id',
            'active_input_predecessor_field_label', 'active_input_predecessor_text')))
        fields = self.root_entities.get("input_fields")
        if not all((isinstance(value, str) and value for value in values)) or not isinstance(fields, list):
            return "", "", ""
        field_id, label, text = values
        active_id, active_label, _ = self.field
        active_text = self.transaction_text
        matches = lambda fid, flabel, value: sum((isinstance(item,
            dict) and item.get('field_id') == fid and (item.get('field_label',
            '') == flabel) and (item.get('text') == value) for item in fields))
        return (field_id, label, text) if field_id != active_id and matches(field_id, label, text) == matches(active_id,
            active_label, active_text) == 1 else ('', '', '')

    @property
    def mode_switch_requested(self) -> bool:
        selectors: list[Any] = [self.focus]
        if (self.has_active_focus and str(self.focus.get('subgoal_id') or '').strip() == 'exact_tap_semantic'
            and str(self.goal_entities.get('target_ui_label') or '').strip()):
            original = self.root_entities.get("original_goal_visual_context")
            if isinstance(original, str) and original.strip():
                selectors.append(original)
        visible = json.dumps(selectors, ensure_ascii=False).casefold()
        return any(term in visible for term in _MODE_SWITCH_TERMS)

    @property
    def input_requested(self) -> bool:
        if self.target_only or self.transaction_text or self.mode_switch_requested:
            return True
        source = self.root if not self.has_active_focus else {'objective': self.focus.get('objective'),
            'completion_conditions': self.focus.get('completion_conditions')}
        visible = json.dumps(source, ensure_ascii=False).casefold()
        return any(term in visible for term in _INPUT_TERMS)

    @property
    def clear_requested(self) -> bool:
        visible = str(self.focus.get("objective") or "").casefold()
        return self.input_requested and any(term in visible for term in _CLEAR_TERMS)

    @property
    def has_explicit_text(self) -> bool:
        return bool(self.explicit_text or self.transaction_text)


class VisibleGoalEvidence:
    """The sole current-frame, zero-action goal evidence evaluator."""

    @staticmethod
    def idempotent_app_foreground(value: Any) -> bool:
        text = str(value or "").strip().casefold()
        if not text or len(text) > 96:
            return False
        return bool(
            re.fullmatch(
                r"[\w\u4e00-\u9fff·._ -]{1,64}(?:应用|程序)(?:已经|已)?"
                r"(?:打开|启动|在前台|处于前台)(?:可见)?[。.]?",
                text,
            )
            or re.fullmatch(
                r"[a-z0-9][a-z0-9 ._-]{0,63}\s+(?:app|application)\s+"
                r"(?:is\s+)?(?:open|opened|launched|in the foreground|foreground)"
                r"(?:\s+and\s+visible)?[.]?",
                text,
            )
        )

    @classmethod
    def presence_only(cls, subgoal: Any) -> bool:
        conditions = tuple(filter(None, (str(item or '').strip() for item in getattr(subgoal, 'completion_conditions',
            ()) or ())))
        if len(conditions) == 1 and cls.idempotent_app_foreground(conditions[0]):
            return True
        completion = " ".join(conditions).casefold()
        text = f'{getattr(subgoal, "objective", "")} {completion}'.casefold()
        if not text or not re.search(
            r"定位|找到|寻找|识别|可见|存在|\blocat(?:e|ed)\b|\bfind\b|\bidentif(?:y|ied)\b|"
            r"\bvisible\b|\bpresent\b|\bexists?\b",
            text,
        ):
            return False
        if re.search(
            r"内容|文字|文本|数值|字段值|包含|等于|是否为|状态为|验证|核对|读取|刷新|重新(?:加载|载入|获取|读取|连接)|"
            r"(?:加载|更新|同步)完成|不可见|不存在|缺失|消失|移除|\b(?:content|value|verify|contains?|equals?|read the|"
            r"refresh|reload(?:ed)?|updated|synchronized|not visible|absent|missing|disappear|remove|retrieved|refetched|reconnected)\b",
            text,
        ):
            return False
        return bool(completion) or not re.search(
            r"导航|跳转|进入|返回|切换|打开|启动|收起|隐藏|关闭|"
            r"\b(?:navigate|redirect|enter|return|switch|open|launch|dismiss|hide|close)\w*\b",
            text,
        )

    @staticmethod
    def has_conflict(observation: Any, element_id: str) -> bool:
        for conflict in getattr(observation, 'candidate_conflicts', ()) or ():
            if not isinstance(conflict, Mapping):
                if element_id in str(conflict):
                    return True
                continue
            ids = conflict.get("element_ids") or []
            resolved = conflict.get('kind') == 'duplicate_visual_object_collapsed' and conflict.get(
                'canonical_element_id') == element_id and (element_id in ids)
            if not resolved and (element_id in ids or element_id in str(conflict)):
                return True
        return False

    @classmethod
    def safe_element(cls, item: Any, observation: Any, *, roles: frozenset[str] | None=None,
        goal_relevant: bool | None=None) -> bool:
        states = getattr(item, "states", {}) or {}
        left, top, right, bottom = getattr(item, "bounds", (0, 0, 0, 0))
        return bool((roles is None or str(getattr(item, 'role',
            '')) in roles) and (goal_relevant is None or states.get('goal_relevant') is goal_relevant)
            and (states.get('visible') is not False) and (states.get('fully_visible') is True) and (float(getattr(item,
            'confidence', 0.0)) >= MIN_TARGET_CONFIDENCE) and (0.02 <= left < right <= 0.98)
            and (0.02 <= top < bottom <= 0.98) and (not cls.has_conflict(observation, str(getattr(item, 'element_id',
            '') or ''))))

    @classmethod
    def focused_input_fact(cls, observation: Any) -> str | None:
        scene = getattr(observation, "scene", None)
        candidates = [item for item in getattr(scene, 'elements', ()) or () if (getattr(item, 'states',
            {}) or {}).get('focused') is True and cls.safe_element(item, observation, roles=frozenset({'input'}),
            goal_relevant=True)]
        if len(candidates) != 1:
            return None
        item = candidates[0]
        return f"当前可信画面的局部控件状态：element_id={item.element_id}, role=input, focused=true。"

    @classmethod
    def zero_action_fact(cls, subgoal: Any, observation: Any) -> str | None:
        conditions = tuple((str(item or '').strip().casefold() for item in getattr(subgoal, 'completion_conditions',
            ()) or () if str(item or '').strip()))
        focus = re.compile(
            r"(?:输入框|文本框|输入区域).{0,10}(?:聚焦|焦点)|焦点.{0,10}(?:输入框|文本框|输入区域)|"
            r"(?:input|textbox|text field).{0,20}(?:focused|focus)"
        )
        return cls.focused_input_fact(observation) if conditions and all((focus.search(item) for item
            in conditions)) else None

    @staticmethod
    def binding_terms(*values: Any) -> frozenset[str]:
        text = " ".join(str(value or "").casefold().replace("_", " ") for value in values)
        generic = {'action', 'button', 'control', 'current', 'display', 'element', 'foreground', 'image', 'item',
            'page', 'screen', 'show', 'stable', 'target', 'view', 'visible', '当前', '前台', '页面', '画面', '目标', '元素', '控件',
            '可见', '出现', '显示', '稳定', '完整', '唯一'}
        terms = {token for token in re.findall(r"[a-z0-9]{3,}", text) if token not in generic}
        for run in re.findall('[\\u4e00-\\u9fff]{2,}', text):
            terms.update((run[index:index + size] for size in range(2, min(6,
                len(run)) + 1) for index in range(len(run) - size + 1)))
        return frozenset(terms.difference(generic))

    @staticmethod
    def title_prefixes(*values: Any) -> tuple[str, ...]:
        text = " ".join(str(value or "").strip() for value in values)
        patterns = (
            re.compile(r"标题(?:文字)?(?:开头|起始)(?:为|是|[:：])?\s*[“\"']?([A-Za-z0-9\u4e00-\u9fff·._-]{1,64}?)(?=的(?:唯一)?(?:卡片|列表项|条目|按钮|菜单项)|[”\"'，,。；;]|$)"),
            re.compile(r"title\s+(?:starts?|begins?)\s+with\s+[\"']?([A-Za-z0-9][A-Za-z0-9 ._\-]{0,63}?)(?=(?:\s+(?:card|item|button|entry))|[\"',.;]|$)", re.IGNORECASE),
        )
        return tuple(dict.fromkeys((match.group(1).strip().casefold() for pattern in patterns for match
            in pattern.finditer(text) if match.group(1).strip())))

    @classmethod
    def surface_classes(cls, *values: Any) -> frozenset[str]:
        text = " ".join(str(value or "").casefold().replace("_", " ").replace("-", " ") for value in values)
        markers = {'page': ('页面', '网页', '界面', '首页', ' page', 'screen', 'view', 'interface', 'app home'), 'title': ('标题',
            '题头', 'title', 'heading'), 'list': ('列表', '清单', ' list'), 'input': ('输入框', '文本框', 'input field', 'textbox'),
            'menu': ('菜单', ' menu'), 'dialog': ('对话框', '弹窗', 'dialog', 'modal'), 'destination': ('对应页面', '目标页面', '下一页',
            '详情', 'destination page', 'target page', 'next page', 'detail'), 'foreground_app': ('应用在前台', '前台应用', '前台可见',
            'foreground app', 'in the foreground', 'is foreground')}
        classes = {name for name, words in markers.items() if any(word in text for word in words)}
        if any((cls.idempotent_app_foreground(value) for value in values)):
            classes.add("foreground_app")
        return frozenset(classes)

    @classmethod
    def target_app_terms(cls, *values: Any) -> frozenset[str]:
        return cls.binding_terms(*values).difference({"app", "application", "android", "com", "应用", "程序"})

    @staticmethod
    def compact_app_phrase(value: Any) -> str:
        return "".join(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", str(value or "").casefold()))

    @classmethod
    def references_target_app(cls, text: str, target_app: Any) -> bool:
        compact = cls.compact_app_phrase(text)
        identities = (cls.compact_app_phrase(getattr(target_app, 'app_id', '')),
            cls.compact_app_phrase(getattr(target_app, 'app_name', '')))
        return bool(compact and any((identity in compact for identity in identities if identity and identity not
            in {'app', 'application', '应用', '程序'})))

    @classmethod
    def names_only_target_app(cls, text: str, target_app: Any) -> bool:
        compact = cls.compact_app_phrase(text)
        identities = tuple(dict.fromkeys((identity for identity in (cls.compact_app_phrase(getattr(target_app, 'app_id',
            '')), cls.compact_app_phrase(getattr(target_app, 'app_name', ''))) if identity and identity not in {'app',
            'application', '应用', '程序'})))
        if not compact or not identities or (not any((identity in compact for identity in identities))):
            return False
        for identity in sorted(identities, key=len, reverse=True):
            compact = compact.replace(identity, "")
        generic = ('处于前台', '已经打开', '已经启动', '应用程序', '主界面', '主页面', '当前', '目标', '应用', '程序', '主页', '首页', '页面', '界面', '屏幕',
            '视图', '打开', '启动', '进入', '前台', '可见', '显示', '已经', '已', '在', '的', '并', 'and', 'application', 'foreground',
            'launched', 'opened', 'visible', 'current', 'target', 'screen', 'interface', 'page', 'view', 'home', 'main',
            'launch', 'open', 'app', 'is', 'in', 'the')
        for token in sorted(generic, key=len, reverse=True):
            compact = compact.replace(token, "")
        return not compact

    @staticmethod
    def subgoal_targets_launcher(graph: DynamicTaskGraph, subgoal_id: str) -> bool:
        if not str(subgoal_id or '').strip():
            return False
        try:
            semantic_ir = compile_formal_semantic_authority(graph).semantic_ir
        except TaskSemanticIRError:
            return False
        subgoal = next((item for item in semantic_ir.subgoals if item.subgoal_id == subgoal_id), None)
        surfaces = {item.surface_id: item for item in semantic_ir.surfaces}
        return bool(subgoal is not None and surfaces.get(subgoal.surface_ref) is not None
            and (surfaces[subgoal.surface_ref].kind == 'launcher'))

    @staticmethod
    def typed_system_surface_fact(graph: DynamicTaskGraph, subgoal_id: str, scene: Any) -> str | None:
        if not str(subgoal_id or '').strip():
            return None
        try:
            ir = compile_formal_semantic_authority(graph).semantic_ir
            actual = scene_surface_kind(scene)
        except (TaskSemanticIRError, UISceneError):
            return None
        subgoal = next((item for item in ir.subgoals if item.subgoal_id == subgoal_id), None)
        constraints = {item.constraint_id: item for item in ir.constraints}
        actions = {str(constraints[ref].value) for ref in getattr(subgoal, 'constraint_refs',
            ()) if ref in constraints and constraints[ref].kind == 'required_action'}
        action = next(iter(actions)) if len(actions) == 1 else ""
        if not action or expected_idempotent_system_surface_kind(action) != actual:
            return None
        return json.dumps({'action_kind': action, 'operator': 'equals', 'predicate': 'surface.kind',
            'source': 'canonical_action_protocol', 'value': actual}, ensure_ascii=False, sort_keys=True,
            separators=(',', ':'))

    @classmethod
    def referenced_target_apps(cls, graph: DynamicTaskGraph, text: str, subgoal_id: str='') -> tuple[Any, ...]:
        if (cls.subgoal_targets_launcher(graph, subgoal_id) or not cls.surface_classes(text).intersection({'page',
            'foreground_app'})):
            return ()
        return tuple((app for app in graph.goal.target_apps if str(app.app_id
            or '').strip().casefold() != 'current_foreground' and cls.references_target_app(text, app)))

    @staticmethod
    def foreground_matches(scene: Any, target_apps: tuple[Any, ...]) -> bool:
        return any(scene_matches_target_app_surface(scene, app) for app in target_apps)

    @staticmethod
    def page_identity_facts(scene: Any) -> tuple[str, ...]:
        values = [{'app_id': str(getattr(scene, 'app_id', '') or ''), 'foreground_app_id': str(getattr(scene,
            'foreground_app_id', '') or ''), 'screen_id': str(getattr(scene, 'screen_id', '') or '')}]
        for item in tuple(getattr(scene, 'elements', ()) or ()):
            role = str(getattr(item, "role", "") or "").casefold()
            meaning = str(getattr(item, "meaning", "") or "").casefold()
            if (role in {'text', 'container'} and (role == 'container' or any((marker in meaning for marker in ('page',
                'screen', 'view', 'home', 'title', 'heading'))))):
                values.append({"role": role, "meaning": meaning, "label": str(getattr(item, "label", "") or "")})
        return tuple(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) for value in values)

    @classmethod
    def named_presence_grounded(cls, scene: Any, texts: tuple[str, ...]) -> bool:
        return named_visual_identity_is_grounded(texts, cls.page_identity_facts(scene))

    @classmethod
    def intrinsic_surface_classes(cls, item: Any) -> frozenset[str]:
        role = str(getattr(item, "role", "") or "").casefold()
        classes = set(cls.surface_classes(role, getattr(item, "meaning", "")))
        mapped = {'input': 'input', 'textbox': 'input', 'text_input': 'input', 'list': 'list', 'list_item': 'list',
            'menu': 'menu', 'menu_item': 'menu', 'dialog': 'dialog', 'modal': 'dialog', 'title': 'title',
            'heading': 'title'}.get(role)
        if mapped:
            classes.add(mapped)
        return frozenset(classes)

    @staticmethod
    def is_multi_text(text: str) -> bool:
        return bool(re.search('(?:和|与|及|同时|均|都|两者|两个|多个|分别|\\bboth\\b|\\band\\b|\\ball\\b|\\btwo\\b|\\bmultiple\\b)',
            str(text or '').casefold()))

    @classmethod
    def multi_candidates(cls, subgoal: Any, scene: Any, observation: Any) -> tuple[Any, ...] | None:
        text = ' '.join(map(str, (getattr(subgoal, 'objective', ''), *tuple(getattr(subgoal, 'completion_conditions',
            ()) or ())))).casefold()
        terms = cls.binding_terms(text)
        if not cls.is_multi_text(text) or not terms:
            return None
        matched = [(item, cls.binding_terms(item.label, item.meaning,
            *item.evidence).intersection(terms)) for item in scene.elements]
        matched = [(item, item_terms) for item, item_terms in matched if item_terms]
        if any((not cls.safe_element(item, observation) for item, _ in matched)):
            return None
        reduced = []
        for (index, pair) in enumerate(matched):
            peers = [other_terms for other_index, (_, other_terms) in enumerate(matched) if other_index != index]
            if (not any((left.union(right).issubset(pair[1]) for left_index,
                left in enumerate(peers) for right in peers[left_index + 1:]))):
                reduced.append(pair)
        reduced_terms = [item[1] for item in reduced]
        if (not 2 <= len(reduced) <= 4 or any((not item.difference(frozenset().union(*reduced_terms[:index] +
            reduced_terms[index + 1:])) for index, item in enumerate(reduced_terms)))):
            return None
        return tuple(item[0] for item in reduced)

    @staticmethod
    def visible_text_read(subgoal: Any) -> bool:
        if str(getattr(subgoal, 'external_impact', '')) != 'read_only':
            return False
        text = ' '.join((str(getattr(subgoal, 'objective', '') or ''), *tuple(getattr(subgoal, 'completion_conditions',
            ()) or ()))).casefold()
        return bool(any((marker in text for marker in ('读取', '获取', '读出', 'read', 'report',
            'get the'))) and any((marker in text for marker in ('标题', '题头', '错误提示', '错误信息', '状态提示', 'title', 'heading',
            'error message', 'status message'))) and (not any((marker in text for marker in ('等于', '包含', '逐字', '指定文字',
            '是否为', 'equals', 'contains', 'exactly', 'whether')))))

    @classmethod
    def unique_candidate(cls, scene: Any, observation: Any) -> Any | None:
        candidate = scene.unique_trusted_goal_element(min_confidence=MIN_TARGET_CONFIDENCE)
        if candidate is not None:
            return candidate
        reader = getattr(scene, "trusted_completion_evidence", None)
        candidates = tuple(reader(min_confidence=MIN_TARGET_CONFIDENCE)) if callable(reader) else ()
        if len(candidates) != 1:
            return None
        candidate = candidates[0]
        competing = any((item.element_id != candidate.element_id and item.states.get('goal_relevant') is True
            and (float(item.confidence) >= MIN_TARGET_CONFIDENCE) for item in getattr(scene, 'elements', ()) or ()))
        return None if competing else candidate

    @classmethod
    def evidence(cls, graph: DynamicTaskGraph, subgoal: Any, observation: Any) -> tuple[str, ...] | None:
        scene = getattr(observation, "scene", None)
        if scene is None:
            return None
        conditions = tuple(filter(None, (str(item or '').strip() for item in getattr(subgoal, 'completion_conditions',
            ()) or ())))
        text = " ".join((str(subgoal.objective), *conditions))
        surfaces = cls.surface_classes(*conditions)
        typed_surface = cls.typed_system_surface_fact(graph, subgoal.subgoal_id, scene)
        target_apps = cls.referenced_target_apps(graph, text, subgoal.subgoal_id)
        app_matches = bool(target_apps and cls.foreground_matches(scene, target_apps))
        if target_apps and (not app_matches):
            return None
        page_facts = cls.page_identity_facts(scene)
        named_surface = cls.named_presence_grounded(scene, conditions)
        destination = subgoal.external_impact == 'navigation_only' and bool(surfaces) and surfaces.issubset({'page',
            'destination', 'foreground_app'})
        app_destination = app_matches and any(cls.names_only_target_app(text, app) for app in target_apps)
        if destination:
            if not (app_destination or (_named_visual_identity_anchor(conditions) and named_surface) or typed_surface):
                return None
            return (scene.summary, *page_facts, *((typed_surface,) if typed_surface else ()))
        if (subgoal.external_impact == 'read_only' and (not surfaces)
            and (not _named_visual_identity_anchor(conditions)) and (not cls.is_multi_text(text)) and named_surface):
            return (scene.summary, *page_facts)
        candidates: tuple[Any, ...] = ()
        if cls.is_multi_text(text):
            candidates = cls.multi_candidates(subgoal, scene, observation) or ()
            if not candidates:
                return None
        else:
            candidate = cls.unique_candidate(scene, observation)
            if candidate is not None:
                terms = cls.binding_terms(*conditions)
                prefixes = cls.title_prefixes(subgoal.objective, *conditions)
                candidate_terms = cls.binding_terms(candidate.label, candidate.meaning, *candidate.evidence)
                scene_terms = cls.binding_terms(scene.screen_id, scene.summary)
                prefix_matches = bool(prefixes and all((candidate.label.casefold().startswith(item) for item
                    in prefixes)))
                scene_surfaces = cls.surface_classes(scene.screen_id, scene.summary).intersection({"page"})
                scene_surfaces = scene_surfaces.union({"foreground_app"}) if app_matches else scene_surfaces
                element_surfaces = surfaces.difference(scene_surfaces)
                element_surfaces = element_surfaces.difference({
                    'title'}) if 'title' in element_surfaces and prefix_matches else element_surfaces
                if (not cls.safe_element(candidate, observation) or not terms
                    or (not (prefix_matches if prefixes else terms.intersection(candidate_terms.union(scene_terms))))
                    or (not element_surfaces.issubset(cls.intrinsic_surface_classes(candidate)))):
                    return None
                candidates = (candidate,)
            elif subgoal.external_impact == 'navigation_only':
                terms = cls.binding_terms(text)
                scene_terms = cls.binding_terms(scene.summary)
                scene_surfaces = cls.surface_classes(scene.screen_id, scene.summary).intersection({"page"})
                scene_surfaces = scene_surfaces.union({"foreground_app"}) if app_matches else scene_surfaces
                element_surfaces = surfaces.difference(scene_surfaces)
                candidates = tuple((item for item in scene.elements if terms.intersection(cls.binding_terms(item.label,
                    item.meaning, *item.evidence)) and element_surfaces.issubset(cls.intrinsic_surface_classes(item))))
                if (not terms.intersection(scene_terms) or not 1 <= len(candidates) <= 4
                    or any((not cls.safe_element(item, observation) for item in candidates))):
                    return None
            elif not (app_matches or named_surface):
                return None
        facts = tuple(
            f"当前可信画面的目标元素：element_id={item.element_id}, role={item.role}, label={item.label or '[empty]'}, "
            f"meaning={item.meaning}, confidence={float(item.confidence):.3f}, fully_visible=true, bounds_inside_safe_frame=true。"
            for item in candidates
        )
        focus = cls.focused_input_fact(observation)
        return (scene.summary, *((focus,) if focus else ()), *facts,
            *(fact for item in candidates for fact in item.evidence), *((*page_facts,
            typed_surface) if typed_surface else ()))


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    if text.startswith('```'):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise GenericIntentError(f"文本模型没有返回有效 JSON：{exc}") from exc
    if not isinstance(value, dict):
        raise GenericIntentError("文本模型返回内容不是 JSON 对象。")
    return value


def _validate_json_value(value: Any, path: str) -> None:
    if isinstance(value, dict):
        for (key, item) in value.items():
            if not isinstance(key, str) or not key.strip():
                raise GenericIntentError(f"目标参数字段无效：{path}")
            _validate_json_value(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for (index, item) in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]")
    elif not isinstance(value, (str, int, float, bool, type(None))):
        raise GenericIntentError(f"目标参数类型不受支持：{path}")


def safe_goal_context(value: dict[str, Any]) -> dict[str, Any]:
    """Keep goal data useful to observation while refusing control fields."""

    forbidden = {'action', 'actions', 'step', 'steps', 'tap', 'swipe', 'coordinate', 'coordinates', 'x', 'y', 'command',
        'shell', 'execution_plan'}

    def clean(item: Any, depth: int=0) -> Any:
        if depth > 5:
            raise VisionAgentError("目标上下文嵌套过深。")
        if isinstance(item, dict):
            result: dict[str, Any] = {}
            for (raw_key, raw_value) in item.items():
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
