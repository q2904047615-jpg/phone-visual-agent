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
ELEMENT_BOUND_ROLES = frozenset(
    {"button", "icon", "text", "tab", "image", "list_item", "input", "toggle"}
)


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
        if (
            str(candidate_role or "").strip() in ELEMENT_BOUND_ROLES
            and _PAGE_ELEMENT_SCOPE.search(constraint)
        ):
            return True
        if candidate_terms.intersection(binding_terms(constraint)):
            return True
    return False
