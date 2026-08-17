from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


GENERIC_BINDING_TERMS = frozenset(
    {
        "action",
        "button",
        "control",
        "current",
        "element",
        "icon",
        "image",
        "item",
        "page",
        "screen",
        "setup",
        "target",
        "view",
        "元素",
        "图标",
        "当前",
        "按钮",
        "控件",
        "入口",
        "操作",
        "目标",
        "画面",
        "视图",
        "页面",
    }
)
_PROHIBITION = re.compile(
    r"(?:不要|不得|禁止|不可|不能|勿|避免|避开|不再)"
    r"|\b(?:do\s+not|don't|must\s+not|never|avoid|exclude)\b",
    re.IGNORECASE,
)
_PAGE_ELEMENT_SCOPE = re.compile(
    r"(?:(?:任何|所有|一切|任意|全部)\s*)"
    r"(?:页面(?:正文|内容|内)?|正文)"
    r".{0,24}(?:链接|按钮|元素|入口|控件)"
    r"|(?:页面正文|页面内容|页面内|正文)"
    r".{0,24}(?:(?:任何|所有|一切|任意|全部)\s*)?"
    r"(?:链接|按钮|元素|入口|控件)"
    r"|\b(?:any|all|every)\s+(?:page\s+(?:body|content)|in[- ]page)"
    r".{0,24}(?:link|button|element|control)s?\b"
    r"|\b(?:page\s+(?:body|content)|in[- ]page)"
    r".{0,24}(?:any|all|every)\s+(?:link|button|element|control)s?\b",
    re.IGNORECASE,
)
_ELEMENT_TARGETING_PROHIBITION = re.compile(
    r"(?:不要|不得|禁止|不可|不能|勿|不再)"
    r"[^，。；;]{0,32}?"
    r"(?:点击|点按|触碰|触摸|打开|进入|选择|勾选|切换|操作|使用|访问|启动|按下|长按|滑动|拖动)"
    r"|\b(?:do\s+not|don't|must\s+not|never)\b"
    r"[^,.;]{0,48}?"
    r"\b(?:click|tap|touch|open|enter|select|choose|toggle|operate|use|visit|launch|press|long[- ]press|swipe|drag)\b"
    r"|(?:避免|避开|\b(?:avoid|exclude)\b)",
    re.IGNORECASE,
)
_STRONG_ELEMENT_TARGETING_PROHIBITION = re.compile(
    r"(?:不要|不得|禁止|不可|不能|勿|不再)"
    r"[^，。；;]{0,32}?"
    r"(?:点击|点按|触碰|触摸|打开|进入|选择|勾选|切换|使用|访问|启动|按下|长按|滑动|拖动)"
    r"|\b(?:do\s+not|don't|must\s+not|never)\b"
    r"[^,.;]{0,48}?"
    r"\b(?:click|tap|touch|open|enter|select|choose|toggle|use|visit|launch|press|long[- ]press|swipe|drag)\b"
    r"|(?:避免|避开|\b(?:avoid|exclude)\b)",
    re.IGNORECASE,
)
_STATE_EFFECT_OPERATION_PROHIBITION = re.compile(
    r"(?:不要|不得|禁止|不可|不能|勿|避免|不再)"
    r"[^，。；;]{0,16}(?:执行|进行)"
    r"[^，。；;]{0,28}(?:改变|修改|保存|提交|发送|发布|删除|移除|创建|新增|上传|分享|支付|购买)"
    r"[^，。；;]{0,28}(?:的)?操作"
    r"|\b(?:do\s+not|don't|must\s+not|never|avoid)\b"
    r"[^,.;]{0,24}\b(?:perform|carry\s+out)\b"
    r"[^,.;]{0,40}\b(?:change|modify|save|submit|send|publish|delete|remove|create|upload|share|pay|purchase)\b"
    r"[^,.;]{0,32}\b(?:operation|action)s?\b",
    re.IGNORECASE,
)
_EXCEPTION_SCOPE = re.compile(
    r"除(?P<zh>[^，。；;]{1,80}?)之外"
    r"|\b(?:except|other\s+than)\s+(?P<en>[^,.;]{1,80})",
    re.IGNORECASE,
)
_BLANKET_OPERATION_SCOPE = re.compile(
    r"(?:任何|所有|一切|其他|其它|其余)(?:的)?(?:操作|动作|行为)"
    r"|\b(?:any|all|every|other)\s+(?:action|operation)s?\b"
    r"|\b(?:anything|everything)\b",
    re.IGNORECASE,
)
ELEMENT_BOUND_ROLES = frozenset(
    {"button", "icon", "text", "tab", "image", "list_item", "input", "toggle"}
)


def exception_scope_terms(constraint: str) -> set[str]:
    """Return the explicitly allowed target terms of a blanket prohibition."""

    if not _BLANKET_OPERATION_SCOPE.search(constraint):
        return set()
    match = _EXCEPTION_SCOPE.search(constraint)
    if match is None:
        return set()
    allowed = match.group("zh") or match.group("en") or ""
    return binding_terms(allowed)


def structured_strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else ()
    if isinstance(value, Mapping):
        result: list[str] = []
        for item in value.values():
            result.extend(structured_strings(item))
        return tuple(result)
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.extend(structured_strings(item))
        return tuple(result)
    return ()


def binding_terms(values: Any) -> set[str]:
    terms: set[str] = set()
    chinese_generic_terms = tuple(
        term
        for term in GENERIC_BINDING_TERMS
        if re.fullmatch(r"[\u3400-\u9fff]+", term)
    )
    for value in structured_strings(values):
        normalized = value.casefold()
        terms.update(
            token
            for token in re.findall(r"[a-z0-9]+", normalized)
            if len(token) >= 2 and token not in GENERIC_BINDING_TERMS
        )
        for sequence in re.findall(r"[\u3400-\u9fff]+", normalized):
            for generic in chinese_generic_terms:
                sequence = sequence.replace(generic, "")
            terms.update(
                sequence[index : index + 2]
                for index in range(len(sequence) - 1)
                if sequence[index : index + 2] not in GENERIC_BINDING_TERMS
            )
    return terms


def constraint_excludes_candidate(
    constraints: Any,
    candidate_values: Any,
    *,
    candidate_role: str = "",
) -> bool:
    candidate_terms = binding_terms(candidate_values)
    if not candidate_terms:
        return False
    for constraint in dict.fromkeys(structured_strings(constraints)):
        if not _PROHIBITION.search(constraint):
            continue
        allowed_terms = exception_scope_terms(constraint)
        if allowed_terms:
            if candidate_terms.intersection(allowed_terms):
                continue
            if str(candidate_role or "").strip() in ELEMENT_BOUND_ROLES:
                return True
        if (
            str(candidate_role or "").strip() in ELEMENT_BOUND_ROLES
            and _PAGE_ELEMENT_SCOPE.search(constraint)
        ):
            return True
        if (
            _STATE_EFFECT_OPERATION_PROHIBITION.search(constraint)
            and not _STRONG_ELEMENT_TARGETING_PROHIBITION.search(constraint)
        ):
            # In phrases such as "do not perform an operation that may modify
            # settings", "operation" names the prohibited effect scope; it is
            # not an instruction to ban a visible element sharing that noun.
            continue
        if not _ELEMENT_TARGETING_PROHIBITION.search(constraint):
            # State/result constraints such as "do not change any setting"
            # are enforced by the task-risk and action layers.  A shared word
            # with an App or control label is not enough to make that visible
            # element itself a forbidden target.
            continue
        if candidate_terms.intersection(binding_terms(constraint)):
            return True
    return False
