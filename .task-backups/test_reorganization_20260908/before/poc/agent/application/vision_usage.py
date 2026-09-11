"""Session-scoped in-memory accounting for visual-model requests."""

from __future__ import annotations

from agent.domain.validation import reject_if
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

from agent.domain.vision_model import DEFAULT_VISION_MODEL


VISION_USAGE_LEDGER_VERSION = "2026-08-25-single-step-qwen-usage-v4"
QWEN_PLUS_MODEL = DEFAULT_VISION_MODEL
SINGLE_STEP_ALLOWED_REQUEST_STAGES = frozenset({"single_step_observation"})

# Beijing non-thinking prices verified 2026-08-24; reports keep list/promo estimates, not invoice claims.
QWEN_PLUS_PRICING_VERSION = "cn-beijing-qwen3.7-plus-2026-08-24"
QWEN_PLUS_PRICING_SOURCE = 'https://help.aliyun.com/zh/model-studio/model-pricing'
QWEN_PLUS_LIST_INPUT_CNY_PER_MILLION = 2.0
QWEN_PLUS_LIST_OUTPUT_CNY_PER_MILLION = 8.0
QWEN_PLUS_PROMO_INPUT_CNY_PER_MILLION = 1.6
QWEN_PLUS_PROMO_OUTPUT_CNY_PER_MILLION = 6.4

_ZERO_USAGE_TOTALS: dict[str, int | float] = {
    'model_requests': 0, 'successful_requests': 0, 'network_attempts': 0, 'prompt_tokens': 0,
    'completion_tokens': 0, 'total_tokens': 0, 'observation_cache_hits': 0, 'identity_rejections': 0,
    'contract_rejections': 0, 'timed_requests': 0, 'total_elapsed_seconds': 0.0, 'max_elapsed_seconds': 0.0,
}


class VisionUsageError(RuntimeError):
    error_code = "vision_usage_error"


class VisionModelIdentityMismatch(VisionUsageError):
    error_code = "vision_model_identity_mismatch"


class VisionStepContractViolation(VisionUsageError):
    error_code = "vision_step_contract_violation"


def _cost_cny(*, prompt_tokens: int, completion_tokens: int, input_rate: float, output_rate: float) -> float:
    return round(prompt_tokens * input_rate / 1000000 + completion_tokens * output_rate / 1000000, 6)


def _non_negative_int(value: Any) -> int:
    return value if isinstance(value, int) and (not isinstance(value, bool)) and (value >= 0) else 0


@dataclass
class VisionSessionUsageLedger:
    """Thread-safe, session-scoped Qwen request and token ledger."""

    session_id: str
    expected_model: str = QWEN_PLUS_MODEL
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec='seconds'))
    _events: list[dict[str, Any]] = field(default_factory=list, repr=False)
    _totals: dict[str, int | float] = field(default_factory=lambda: dict(_ZERO_USAGE_TOTALS), repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.session_id = str(self.session_id or "").strip()
        self.expected_model = str(self.expected_model or "").strip()
        reject_if(not self.session_id, ValueError("Qwen 用量账本必须绑定 session_id。"))
        reject_if(self.expected_model != QWEN_PLUS_MODEL, ValueError("正式 Qwen 用量账本只允许 qwen3.7-plus。"))

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="milliseconds")

    @staticmethod
    def _metadata(stage: str, fingerprint: str) -> tuple[str, str]:
        resolved_stage = str(stage or "unscoped").strip()[:120] or "unscoped"
        resolved_fingerprint = str(fingerprint or "").strip()[:256]
        return resolved_stage, resolved_fingerprint

    def _add_elapsed(self, elapsed: float | None) -> None:
        if elapsed is None:
            return
        self._totals['timed_requests'] += 1
        self._totals['total_elapsed_seconds'] += elapsed
        self._totals['max_elapsed_seconds'] = max(self._totals['max_elapsed_seconds'], elapsed)

    def reserve_request(self, *, model: str, stage: str, fingerprint: str,
        max_completion_tokens: int | None) -> str:
        stage, fingerprint = self._metadata(stage, fingerprint)
        model = str(model or "").strip()
        with self._lock:
            if model != self.expected_model:
                self._totals['identity_rejections'] += 1
                self._events.append({'event': 'request_rejected', 'outcome': 'model_identity_mismatch',
                    'timestamp': self._timestamp(), 'stage': stage, 'fingerprint': fingerprint, 'model': model,
                    'network_attempts': 0})
                raise VisionModelIdentityMismatch(
                    "vision_model_identity_mismatch: 正式会话固定使用 "
                    f"{self.expected_model}，拒绝自动切换为 {model or 'unknown'}。"
                )
            if stage not in SINGLE_STEP_ALLOWED_REQUEST_STAGES:
                self._totals['contract_rejections'] += 1
                self._events.append({'event': 'request_rejected', 'outcome': 'vision_step_contract_violation',
                    'timestamp': self._timestamp(), 'stage': stage, 'fingerprint': fingerprint, 'model': model,
                    'network_attempts': 0})
                raise VisionStepContractViolation(
                    "vision_step_contract_violation: 正式会话每个闭环步骤"
                    "只允许 single_step_observation；旧视觉审计或动作选择"
                    f"阶段 {stage} 已在联网前拒绝。"
                )
            local_request_id = f"qwen_local_{uuid.uuid4().hex}"
            self._totals['model_requests'] += 1
            self._events.append({'event': 'model_request', 'outcome': 'started', 'timestamp': self._timestamp(),
                'local_request_id': local_request_id, 'provider_request_id': '', 'stage': stage,
                'fingerprint': fingerprint, 'model': model, 'max_completion_tokens': max(0,
                    int(max_completion_tokens or 0)),
                'network_attempts': 0, 'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0})
            return local_request_id

    def _request_event(self, local_request_id: str) -> dict[str, Any]:
        for event in reversed(self._events):
            if event.get('local_request_id') == local_request_id:
                return event
        raise VisionUsageError("Qwen 用量账本找不到当前本地请求。")

    def record_success(self, local_request_id: str, *, provider_request_id: str, response_model: str,
        network_attempts: int, usage: Mapping[str, Any] | None, finish_reason: str,
        elapsed_seconds: float | None=None) -> None:
        raw = usage if isinstance(usage, Mapping) else {}
        prompt = _non_negative_int(raw.get("prompt_tokens"))
        completion = _non_negative_int(raw.get("completion_tokens"))
        total = _non_negative_int(raw.get("total_tokens")) or prompt + completion
        prompt_details = raw.get("prompt_tokens_details")
        cached = _non_negative_int(prompt_details.get('cached_tokens')) if isinstance(prompt_details, Mapping) else 0
        elapsed = max(0.0, float(elapsed_seconds)) if isinstance(elapsed_seconds, (int,
            float)) and (not isinstance(elapsed_seconds, bool)) else None
        with self._lock:
            event = self._request_event(local_request_id)
            reject_if(event.get('outcome') != 'started', VisionUsageError("Qwen 请求用量被重复结算。"))
            event.update({'outcome': 'succeeded', 'completed_at': self._timestamp(),
                'provider_request_id': str(provider_request_id or '')[:256],
                'response_model': str(response_model or '')[:128], 'network_attempts': max(1, int(network_attempts)),
                'prompt_tokens': prompt, 'completion_tokens': completion, 'total_tokens': total,
                'cached_prompt_tokens': cached, 'finish_reason': str(finish_reason or '')[:120],
                'elapsed_seconds': round(elapsed, 6) if elapsed is not None else None,
                'estimated_list_cost_cny': _cost_cny(prompt_tokens=prompt, completion_tokens=completion,
                input_rate=QWEN_PLUS_LIST_INPUT_CNY_PER_MILLION, output_rate=QWEN_PLUS_LIST_OUTPUT_CNY_PER_MILLION),
                'estimated_promotional_cost_cny': _cost_cny(prompt_tokens=prompt, completion_tokens=completion,
                input_rate=QWEN_PLUS_PROMO_INPUT_CNY_PER_MILLION, output_rate=QWEN_PLUS_PROMO_OUTPUT_CNY_PER_MILLION)})
            for key, amount in {'successful_requests': 1, 'network_attempts': max(1, int(network_attempts)),
                'prompt_tokens': prompt, 'completion_tokens': completion, 'total_tokens': total}.items():
                self._totals[key] += amount
            self._add_elapsed(elapsed)

    def record_failure(self, local_request_id: str, *, network_attempts: int, error: BaseException | str,
        elapsed_seconds: float | None=None) -> None:
        elapsed = max(0.0, float(elapsed_seconds)) if isinstance(elapsed_seconds, (int,
            float)) and (not isinstance(elapsed_seconds, bool)) else None
        with self._lock:
            event = self._request_event(local_request_id)
            if event.get('outcome') != 'started':
                return
            event.update({'outcome': 'failed', 'completed_at': self._timestamp(), 'network_attempts': max(0,
                int(network_attempts)), 'error_type': error.__class__.__name__ if isinstance(error,
                BaseException) else 'error', 'error': str(error)[:500], 'elapsed_seconds': round(elapsed,
                6) if elapsed is not None else None})
            self._totals['network_attempts'] += max(0, int(network_attempts))
            self._add_elapsed(elapsed)

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            totals = dict(self._totals)
            prompt_tokens = int(totals['prompt_tokens'])
            completion_tokens = int(totals['completion_tokens'])
            list_cost = _cost_cny(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                input_rate=QWEN_PLUS_LIST_INPUT_CNY_PER_MILLION, output_rate=QWEN_PLUS_LIST_OUTPUT_CNY_PER_MILLION)
            promotional_cost = _cost_cny(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                input_rate=QWEN_PLUS_PROMO_INPUT_CNY_PER_MILLION, output_rate=QWEN_PLUS_PROMO_OUTPUT_CNY_PER_MILLION)
            timed_requests = int(totals['timed_requests'])
            total_elapsed = float(totals['total_elapsed_seconds'])
            totals.update({'estimated_list_cost_cny': list_cost,
                'estimated_promotional_cost_cny': promotional_cost,
                'total_elapsed_seconds': round(total_elapsed, 6),
                'average_elapsed_seconds': round(total_elapsed / timed_requests, 6) if timed_requests else None,
                'max_elapsed_seconds': round(float(totals['max_elapsed_seconds']), 6) if timed_requests else None})
            return {
                "version": VISION_USAGE_LEDGER_VERSION,
                "session_id": self.session_id,
                "created_at": self.created_at,
                "model": self.expected_model,
                "downgrade_allowed": False,
                "session_limits_enforced": False,
                "totals": totals,
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
