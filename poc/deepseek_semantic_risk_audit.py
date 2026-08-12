from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from generic_intent import _parse_json_object


SEMANTIC_RISK_AUDIT_PROTOCOL = "semantic-risk-audit-v1"
MIN_SEMANTIC_AUDIT_CONFIDENCE = 0.80
EXTERNAL_IMPACTS = frozenset(
    {"read_only", "navigation_only", "external_state", "unknown"}
)
RISK_TYPES = frozenset(
    {
        "message_or_communication",
        "content_publication",
        "account_relationship_change",
        "membership_change",
        "permission_role_change",
        "data_mutation",
        "data_deletion",
        "transaction_or_payment",
        "account_or_permission_change",
        "unknown_external_effect",
    }
)
class JsonRiskAuditProvider(Protocol):
    configured: bool

    def chat_json(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 4000,
    ) -> str: ...


@dataclass(frozen=True)
class AuditSource:
    source_id: str
    source_kind: str
    text: str
    subgoal_id: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_kind": self.source_kind,
            "subgoal_id": self.subgoal_id,
            "text": self.text,
        }


@dataclass(frozen=True)
class RiskAuditAssessment:
    source_id: str
    source_kind: str
    subgoal_id: str | None
    external_impact: str
    risk_types: tuple[str, ...]
    reason: str
    confidence: float
    model_external_impact: str


@dataclass(frozen=True)
class SemanticRiskAuditReport:
    assessments: tuple[RiskAuditAssessment, ...]
    failed_closed: bool = False
    failure_reason: str = ""
    protocol: str = SEMANTIC_RISK_AUDIT_PROTOCOL


class SemanticRiskAuditor:
    """Run a separate, generic semantic safety judgment over isolated text fields."""

    def __init__(self, provider: JsonRiskAuditProvider) -> None:
        self.provider = provider
        self.call_count = 0
        self.last_raw_response = ""

    def audit(self, sources: tuple[AuditSource, ...]) -> SemanticRiskAuditReport:
        self.call_count += 1
        if not sources:
            return SemanticRiskAuditReport(())
        if not getattr(self.provider, "configured", False):
            return _failed_closed_report(sources, "语义风险审计提供方尚未配置")
        prompt = _audit_prompt(sources)
        try:
            raw = self.provider.chat_json(
                [{"role": "user", "content": prompt}],
                max_tokens=4000,
            )
            self.last_raw_response = raw
            payload = _parse_json_object(raw)
            return _report_from_payload(payload, sources)
        except Exception as exc:
            return _failed_closed_report(
                sources,
                f"语义风险审计调用或解析失败：{type(exc).__name__}: {exc}",
            )


def _audit_prompt(sources: tuple[AuditSource, ...]) -> str:
    source_json = json.dumps(
        [item.to_payload() for item in sources],
        ensure_ascii=False,
        indent=2,
    )
    return f"""你是独立于任务规划结果的通用语义风险审计器。
审计协议：{SEMANTIC_RISK_AUDIT_PROTOCOL}

逐项判断下面每个原始字段的外部影响。每个 source 是独立边界，禁止把不同 source 的文本拼接后再解释。
你必须为每个 source_id 返回且只返回一个 assessment，并原样返回 source_kind 和 subgoal_id。

分类：read_only | navigation_only | external_state | unknown
通用风险类型：message_or_communication | content_publication |
account_relationship_change | membership_change | permission_role_change |
data_mutation | data_deletion | transaction_or_payment |
account_or_permission_change | unknown_external_effect

规则：
1. 只有能明确证明不改变外部状态的纯观察才是 read_only。
2. 只有能明确证明仅改变查看位置的页面导航才是 navigation_only。
3. 消息沟通、内容发布、关系/成员/权限变化、数据修改或删除、交易支付等属于 external_state。
4. 句子先观察、后产生外部效果时，整项属于 external_state；不能因包含观察或导航用语而忽略后半句。
5. 含义不清或无法可靠判断时返回 unknown，不能猜成安全类别。
6. external_state 必须返回至少一个匹配的通用风险类型；unknown 使用 unknown_external_effect。
7. read_only/navigation_only 的 risk_types 必须为空数组。
8. 只输出结构化风险判断。禁止输出点击、坐标、输入内容、Shell、系统命令或任何操作步骤。
9. 不得返回确认结果；确认只能来自本地控制器。
10. 明确否定或禁止的效果词只是安全约束，不会因为出现效果词就变成 external_state。例如
    “不登录”“不要发送”“禁止删除”“不搜索、不登录”应判断句中剩余的正向目标；并列否定
    对同一短语中的各并列项生效。
11. 否定词没有直接否定效果时不能降级风险。例如“不要忘记登录”“不能只查看而要发送”仍然
    包含正向外部效果，必须按 external_state 判断。

输入 sources：
{source_json}

只返回以下 JSON，不要 Markdown：
{{
  "assessments": [
    {{
      "source_id": "输入中的 source_id",
      "source_kind": "输入中的 source_kind",
      "subgoal_id": "输入中的 subgoal_id 或 null",
      "external_impact": "read_only|navigation_only|external_state|unknown",
      "risk_types": ["通用风险类型"],
      "reason": "高层语义判断理由，不含操作步骤",
      "confidence": 0.0
    }}
  ]
}}"""


def _report_from_payload(
    payload: dict[str, Any],
    sources: tuple[AuditSource, ...],
) -> SemanticRiskAuditReport:
    _expect_exact_keys(payload, {"assessments"}, "语义风险审计")
    raw_assessments = payload.get("assessments")
    if not isinstance(raw_assessments, list):
        raise ValueError("语义风险审计 assessments 必须是数组")
    source_map = {item.source_id: item for item in sources}
    if len(source_map) != len(sources):
        raise ValueError("语义风险审计输入 source_id 必须唯一")
    parsed: dict[str, RiskAuditAssessment] = {}
    for raw in raw_assessments:
        if not isinstance(raw, dict):
            raise ValueError("语义风险审计 assessment 必须是对象")
        _expect_exact_keys(
            raw,
            {
                "source_id",
                "source_kind",
                "subgoal_id",
                "external_impact",
                "risk_types",
                "reason",
                "confidence",
            },
            "语义风险审计 assessment",
        )
        source_id = _required_text(raw.get("source_id"), "source_id")
        if source_id in parsed:
            raise ValueError(f"语义风险审计重复 source_id：{source_id}")
        source = source_map.get(source_id)
        if source is None:
            raise ValueError(f"语义风险审计返回未知 source_id：{source_id}")
        source_kind = _required_text(raw.get("source_kind"), "source_kind")
        subgoal_id = raw.get("subgoal_id")
        if subgoal_id is not None:
            subgoal_id = _required_text(subgoal_id, "subgoal_id")
        if source_kind != source.source_kind or subgoal_id != source.subgoal_id:
            raise ValueError(f"语义风险审计篡改来源：{source_id}")
        model_impact = _required_text(
            raw.get("external_impact"),
            "external_impact",
        ).lower()
        if model_impact not in EXTERNAL_IMPACTS:
            raise ValueError(f"语义风险审计分类无效：{model_impact}")
        risk_types = _risk_types(raw.get("risk_types"))
        if model_impact == "external_state" and not risk_types:
            raise ValueError(f"external_state 缺少风险类型：{source_id}")
        if model_impact == "unknown" and risk_types != ("unknown_external_effect",):
            raise ValueError(f"unknown 必须使用 unknown_external_effect：{source_id}")
        if model_impact in {"read_only", "navigation_only"} and risk_types:
            raise ValueError(f"安全分类不得携带风险类型：{source_id}")
        confidence = raw.get("confidence")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0 <= confidence <= 1
        ):
            raise ValueError(f"语义风险审计 confidence 无效：{source_id}")
        confidence = float(confidence)
        reason = _required_text(raw.get("reason"), "reason")
        if confidence < MIN_SEMANTIC_AUDIT_CONFIDENCE:
            effective_impact = "unknown"
            effective_risk_types = ("unknown_external_effect",)
            reason = f"置信度不足，失败关闭为 unknown：{reason}"
        else:
            effective_impact = model_impact
            effective_risk_types = risk_types
        parsed[source_id] = RiskAuditAssessment(
            source_id=source_id,
            source_kind=source_kind,
            subgoal_id=subgoal_id,
            external_impact=effective_impact,
            risk_types=effective_risk_types,
            reason=reason,
            confidence=confidence,
            model_external_impact=model_impact,
        )
    missing = set(source_map) - set(parsed)
    if missing:
        raise ValueError(
            "语义风险审计遗漏 source_id：" + ", ".join(sorted(missing))
        )
    return SemanticRiskAuditReport(
        assessments=tuple(parsed[item.source_id] for item in sources)
    )


def _failed_closed_report(
    sources: tuple[AuditSource, ...],
    reason: str,
) -> SemanticRiskAuditReport:
    return SemanticRiskAuditReport(
        assessments=tuple(
            RiskAuditAssessment(
                source_id=item.source_id,
                source_kind=item.source_kind,
                subgoal_id=item.subgoal_id,
                external_impact="unknown",
                risk_types=("unknown_external_effect",),
                reason=reason,
                confidence=0.0,
                model_external_impact="unknown",
            )
            for item in sources
        ),
        failed_closed=True,
        failure_reason=reason,
    )


def _expect_exact_keys(value: dict[str, Any], allowed: set[str], path: str) -> None:
    unexpected = set(value) - allowed
    missing = allowed - set(value)
    if unexpected:
        raise ValueError(
            f"{path} 包含协议外字段：" + ", ".join(sorted(map(str, unexpected)))
        )
    if missing:
        raise ValueError(
            f"{path} 缺少字段：" + ", ".join(sorted(map(str, missing)))
        )


def _required_text(value: Any, path: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{path} 不能为空")
    return text


def _risk_types(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError("risk_types 必须是数组")
    items = tuple(str(item or "").strip().lower() for item in value)
    if any(not item for item in items) or len(set(items)) != len(items):
        raise ValueError("risk_types 必须是非空唯一字符串")
    invalid = set(items) - RISK_TYPES
    if invalid:
        raise ValueError("通用风险类型无效：" + ", ".join(sorted(invalid)))
    return items
