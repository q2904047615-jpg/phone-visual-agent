from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping


VISION_USAGE_LEDGER_VERSION = "2026-08-24-single-step-qwen-usage-v2"
QWEN_PLUS_MODEL = "qwen3.7-plus"
SINGLE_STEP_ALLOWED_REQUEST_STAGES = frozenset({"single_step_observation"})
DEFAULT_MAX_MODEL_REQUESTS = 16
DEFAULT_MAX_TOTAL_TOKENS = 50_000
# Six-action acceptance normally needs one initial observation plus exactly one
# post-action observation per action.  The hard ceiling remains a fail-safe for
# genuine replans; it is not the expected production topology.
CHINESE_ACCEPTANCE_TARGET_REQUESTS = 7
CHINESE_ACCEPTANCE_TARGET_TOKENS = 30_000

# Official Model Studio pricing page, verified 2026-08-24 for China (Beijing),
# non-thinking qwen3.7-plus requests with <=256K input tokens.  The promotion
# can change, so reports retain both values and label them as estimates rather
# than claiming to be the Alibaba Cloud invoice.
QWEN_PLUS_PRICING_VERSION = "cn-beijing-qwen3.7-plus-2026-08-24"
QWEN_PLUS_PRICING_SOURCE = (
    "https://help.aliyun.com/zh/model-studio/model-pricing"
)
QWEN_PLUS_LIST_INPUT_CNY_PER_MILLION = 2.0
QWEN_PLUS_LIST_OUTPUT_CNY_PER_MILLION = 8.0
QWEN_PLUS_PROMO_INPUT_CNY_PER_MILLION = 1.6
QWEN_PLUS_PROMO_OUTPUT_CNY_PER_MILLION = 6.4


class VisionUsageError(RuntimeError):
    error_code = "vision_usage_error"


class VisionModelBudgetExceeded(VisionUsageError):
    error_code = "model_budget_exhausted"


class VisionModelIdentityMismatch(VisionUsageError):
    error_code = "vision_model_identity_mismatch"


class VisionStepContractViolation(VisionUsageError):
    error_code = "vision_step_contract_violation"


def _cost_cny(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    input_rate: float,
    output_rate: float,
) -> float:
    return round(
        prompt_tokens * input_rate / 1_000_000
        + completion_tokens * output_rate / 1_000_000,
        6,
    )


def _non_negative_int(value: Any) -> int:
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        else 0
    )


@dataclass
class VisionSessionUsageLedger:
    """Thread-safe, session-scoped Qwen request and token ledger."""

    session_id: str
    expected_model: str = QWEN_PLUS_MODEL
    max_model_requests: int = DEFAULT_MAX_MODEL_REQUESTS
    max_total_tokens: int = DEFAULT_MAX_TOTAL_TOKENS
    target_model_requests: int = CHINESE_ACCEPTANCE_TARGET_REQUESTS
    target_total_tokens: int = CHINESE_ACCEPTANCE_TARGET_TOKENS
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        )
    )
    _events: list[dict[str, Any]] = field(default_factory=list, repr=False)
    _request_count: int = field(default=0, repr=False)
    _successful_count: int = field(default=0, repr=False)
    _network_attempts: int = field(default=0, repr=False)
    _prompt_tokens: int = field(default=0, repr=False)
    _completion_tokens: int = field(default=0, repr=False)
    _total_tokens: int = field(default=0, repr=False)
    _cache_hits: int = field(default=0, repr=False)
    _budget_rejections: int = field(default=0, repr=False)
    _contract_rejections: int = field(default=0, repr=False)
    _lock: threading.RLock = field(
        default_factory=threading.RLock,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        self.session_id = str(self.session_id or "").strip()
        self.expected_model = str(self.expected_model or "").strip()
        if not self.session_id:
            raise ValueError("Qwen 用量账本必须绑定 session_id。")
        if self.expected_model != QWEN_PLUS_MODEL:
            raise ValueError("正式 Qwen 用量账本只允许 qwen3.7-plus。")
        for name in (
            "max_model_requests",
            "max_total_tokens",
            "target_model_requests",
            "target_total_tokens",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"Qwen 用量账本 {name} 必须是正整数。")
        if self.target_model_requests > self.max_model_requests:
            raise ValueError("Qwen 目标请求数不能超过硬上限。")
        if self.target_total_tokens > self.max_total_tokens:
            raise ValueError("Qwen 目标 Token 不能超过硬上限。")

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="milliseconds")

    @staticmethod
    def _metadata(stage: str, fingerprint: str) -> tuple[str, str]:
        resolved_stage = str(stage or "unscoped").strip()[:120] or "unscoped"
        resolved_fingerprint = str(fingerprint or "").strip()[:256]
        return resolved_stage, resolved_fingerprint

    def reserve_request(
        self,
        *,
        model: str,
        stage: str,
        fingerprint: str,
        max_completion_tokens: int,
    ) -> str:
        stage, fingerprint = self._metadata(stage, fingerprint)
        model = str(model or "").strip()
        with self._lock:
            if model != self.expected_model:
                self._budget_rejections += 1
                self._events.append(
                    {
                        "event": "request_rejected",
                        "outcome": "model_identity_mismatch",
                        "timestamp": self._timestamp(),
                        "stage": stage,
                        "fingerprint": fingerprint,
                        "model": model,
                        "network_attempts": 0,
                    }
                )
                raise VisionModelIdentityMismatch(
                    "vision_model_identity_mismatch: 正式会话固定使用 "
                    f"{self.expected_model}，拒绝自动切换为 {model or 'unknown'}。"
                )
            if stage not in SINGLE_STEP_ALLOWED_REQUEST_STAGES:
                self._contract_rejections += 1
                self._events.append(
                    {
                        "event": "request_rejected",
                        "outcome": "vision_step_contract_violation",
                        "timestamp": self._timestamp(),
                        "stage": stage,
                        "fingerprint": fingerprint,
                        "model": model,
                        "network_attempts": 0,
                    }
                )
                raise VisionStepContractViolation(
                    "vision_step_contract_violation: 正式会话每个闭环步骤"
                    "只允许 single_step_observation；旧视觉审计或动作选择"
                    f"阶段 {stage} 已在联网前拒绝。"
                )
            if (
                self._request_count >= self.max_model_requests
                or self._total_tokens >= self.max_total_tokens
            ):
                self._budget_rejections += 1
                self._events.append(
                    {
                        "event": "request_rejected",
                        "outcome": "model_budget_exhausted",
                        "timestamp": self._timestamp(),
                        "stage": stage,
                        "fingerprint": fingerprint,
                        "model": model,
                        "network_attempts": 0,
                        "request_count": self._request_count,
                        "total_tokens": self._total_tokens,
                    }
                )
                raise VisionModelBudgetExceeded(
                    "model_budget_exhausted: 当前会话 Qwen 请求或 Token 预算已耗尽，"
                    "禁止发起下一次网络请求。"
                )
            local_request_id = f"qwen_local_{uuid.uuid4().hex}"
            self._request_count += 1
            self._events.append(
                {
                    "event": "model_request",
                    "outcome": "started",
                    "timestamp": self._timestamp(),
                    "local_request_id": local_request_id,
                    "provider_request_id": "",
                    "stage": stage,
                    "fingerprint": fingerprint,
                    "model": model,
                    "max_completion_tokens": max(0, int(max_completion_tokens)),
                    "network_attempts": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }
            )
            return local_request_id

    def _request_event(self, local_request_id: str) -> dict[str, Any]:
        for event in reversed(self._events):
            if event.get("local_request_id") == local_request_id:
                return event
        raise VisionUsageError("Qwen 用量账本找不到当前本地请求。")

    def record_success(
        self,
        local_request_id: str,
        *,
        provider_request_id: str,
        response_model: str,
        network_attempts: int,
        usage: Mapping[str, Any] | None,
        finish_reason: str,
    ) -> None:
        raw = usage if isinstance(usage, Mapping) else {}
        prompt = _non_negative_int(raw.get("prompt_tokens"))
        completion = _non_negative_int(raw.get("completion_tokens"))
        total = _non_negative_int(raw.get("total_tokens")) or prompt + completion
        prompt_details = raw.get("prompt_tokens_details")
        cached = (
            _non_negative_int(prompt_details.get("cached_tokens"))
            if isinstance(prompt_details, Mapping)
            else 0
        )
        with self._lock:
            event = self._request_event(local_request_id)
            if event.get("outcome") != "started":
                raise VisionUsageError("Qwen 请求用量被重复结算。")
            event.update(
                {
                    "outcome": "succeeded",
                    "completed_at": self._timestamp(),
                    "provider_request_id": str(provider_request_id or "")[:256],
                    "response_model": str(response_model or "")[:128],
                    "network_attempts": max(1, int(network_attempts)),
                    "prompt_tokens": prompt,
                    "completion_tokens": completion,
                    "total_tokens": total,
                    "cached_prompt_tokens": cached,
                    "finish_reason": str(finish_reason or "")[:120],
                    "estimated_list_cost_cny": _cost_cny(
                        prompt_tokens=prompt,
                        completion_tokens=completion,
                        input_rate=QWEN_PLUS_LIST_INPUT_CNY_PER_MILLION,
                        output_rate=QWEN_PLUS_LIST_OUTPUT_CNY_PER_MILLION,
                    ),
                    "estimated_promotional_cost_cny": _cost_cny(
                        prompt_tokens=prompt,
                        completion_tokens=completion,
                        input_rate=QWEN_PLUS_PROMO_INPUT_CNY_PER_MILLION,
                        output_rate=QWEN_PLUS_PROMO_OUTPUT_CNY_PER_MILLION,
                    ),
                }
            )
            self._successful_count += 1
            self._network_attempts += max(1, int(network_attempts))
            self._prompt_tokens += prompt
            self._completion_tokens += completion
            self._total_tokens += total

    def record_failure(
        self,
        local_request_id: str,
        *,
        network_attempts: int,
        error: BaseException | str,
    ) -> None:
        with self._lock:
            event = self._request_event(local_request_id)
            if event.get("outcome") != "started":
                return
            event.update(
                {
                    "outcome": "failed",
                    "completed_at": self._timestamp(),
                    "network_attempts": max(0, int(network_attempts)),
                    "error_type": (
                        error.__class__.__name__
                        if isinstance(error, BaseException)
                        else "error"
                    ),
                    "error": str(error)[:500],
                }
            )
            self._network_attempts += max(0, int(network_attempts))

    def record_cache_hit(self, *, stage: str, fingerprint: str) -> None:
        stage, fingerprint = self._metadata(stage, fingerprint)
        with self._lock:
            self._cache_hits += 1
            self._events.append(
                {
                    "event": "observation_cache_hit",
                    "outcome": "reused",
                    "timestamp": self._timestamp(),
                    "stage": stage,
                    "fingerprint": fingerprint,
                    "model": self.expected_model,
                    "network_attempts": 0,
                }
            )

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            list_cost = _cost_cny(
                prompt_tokens=self._prompt_tokens,
                completion_tokens=self._completion_tokens,
                input_rate=QWEN_PLUS_LIST_INPUT_CNY_PER_MILLION,
                output_rate=QWEN_PLUS_LIST_OUTPUT_CNY_PER_MILLION,
            )
            promotional_cost = _cost_cny(
                prompt_tokens=self._prompt_tokens,
                completion_tokens=self._completion_tokens,
                input_rate=QWEN_PLUS_PROMO_INPUT_CNY_PER_MILLION,
                output_rate=QWEN_PLUS_PROMO_OUTPUT_CNY_PER_MILLION,
            )
            return {
                "version": VISION_USAGE_LEDGER_VERSION,
                "session_id": self.session_id,
                "created_at": self.created_at,
                "model": self.expected_model,
                "downgrade_allowed": False,
                "budget": {
                    "max_model_requests": self.max_model_requests,
                    "max_total_tokens": self.max_total_tokens,
                    "target_model_requests": self.target_model_requests,
                    "target_total_tokens": self.target_total_tokens,
                },
                "totals": {
                    "model_requests": self._request_count,
                    "successful_requests": self._successful_count,
                    "network_attempts": self._network_attempts,
                    "prompt_tokens": self._prompt_tokens,
                    "completion_tokens": self._completion_tokens,
                    "total_tokens": self._total_tokens,
                    "observation_cache_hits": self._cache_hits,
                    "budget_rejections": self._budget_rejections,
                    "contract_rejections": self._contract_rejections,
                    "remaining_model_requests": max(
                        0, self.max_model_requests - self._request_count
                    ),
                    "remaining_tokens": max(
                        0, self.max_total_tokens - self._total_tokens
                    ),
                    "target_request_count_exceeded": (
                        self._request_count > self.target_model_requests
                    ),
                    "target_token_count_exceeded": (
                        self._total_tokens > self.target_total_tokens
                    ),
                    "budget_exhausted": (
                        self._request_count >= self.max_model_requests
                        or self._total_tokens >= self.max_total_tokens
                    ),
                    "estimated_list_cost_cny": list_cost,
                    "estimated_promotional_cost_cny": promotional_cost,
                },
                "pricing": {
                    "version": QWEN_PLUS_PRICING_VERSION,
                    "source": QWEN_PLUS_PRICING_SOURCE,
                    "scope": "China Beijing, non-thinking, input <=256K",
                    "list_input_cny_per_million": (
                        QWEN_PLUS_LIST_INPUT_CNY_PER_MILLION
                    ),
                    "list_output_cny_per_million": (
                        QWEN_PLUS_LIST_OUTPUT_CNY_PER_MILLION
                    ),
                    "promotional_input_cny_per_million": (
                        QWEN_PLUS_PROMO_INPUT_CNY_PER_MILLION
                    ),
                    "promotional_output_cny_per_million": (
                        QWEN_PLUS_PROMO_OUTPUT_CNY_PER_MILLION
                    ),
                    "disclaimer": (
                        "估算值未计缓存折扣、免费额度、资源包和活动变化；"
                        "阿里云最终账单为准。"
                    ),
                },
                "events": [dict(event) for event in self._events],
            }
