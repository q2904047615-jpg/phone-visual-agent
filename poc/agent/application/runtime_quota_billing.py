"""Runtime admission, usage settlement and billing bridge.

The bridge is intentionally small: SQLite owns atomic reservations and the
Runtime session owns the measured evidence.  No payment provider is called and
no currency balance is mutated; the ledger is an auditable CNY estimate.
"""

from __future__ import annotations

from typing import Any, Mapping


class RuntimeQuotaExceeded(RuntimeError):
    error_code = "quota_exceeded"


class RuntimeQuotaBilling:
    def __init__(self, repository: Any) -> None:
        self.repository = repository

    def reserve(self, *, organization_id: str, user_id: str, session_id: str,
                requested_physical_actions: int, requested_model_tokens: int = 0,
                quota_key: str = "daily") -> dict[str, Any]:
        try:
            return self.repository.reserve_runtime_quota(
                organization_id=organization_id,
                user_id=user_id,
                session_id=session_id,
                quota_key=quota_key,
                requested_physical_actions=requested_physical_actions,
                requested_model_tokens=requested_model_tokens,
            )
        except RuntimeError as exc:
            if str(exc).startswith("quota_exceeded:"):
                raise RuntimeQuotaExceeded(str(exc)) from exc
            raise

    @staticmethod
    def _usage(session: Any) -> tuple[int, int, int, int, str, str]:
        snapshot = session.snapshot() if callable(getattr(session, "snapshot", None)) else {}
        usage = snapshot.get("qwen_usage") if isinstance(snapshot, Mapping) else {}
        usage = usage if isinstance(usage, Mapping) else {}
        totals = usage.get("totals") if isinstance(usage, Mapping) else {}
        totals = totals if isinstance(totals, Mapping) else {}
        prompt = max(0, int(totals.get("prompt_tokens") or 0))
        completion = max(0, int(totals.get("completion_tokens") or 0))
        total = max(0, int(totals.get("total_tokens") or prompt + completion))
        cost_cny = float(totals.get("estimated_list_cost_cny") or 0.0)
        cost_micros = max(0, int(round(cost_cny * 1_000_000)))
        model = str(usage.get("model") or "unknown")[:128]
        pricing = usage.get("pricing") if isinstance(usage, Mapping) else {}
        pricing_version = str(pricing.get("version") or "")[:128] if isinstance(pricing, Mapping) else ""
        return prompt, completion, total, cost_micros, model, pricing_version

    def settle(self, *, session: Any, organization_id: str, user_id: str) -> dict[str, Any]:
        prompt, completion, total, cost_micros, model, pricing_version = self._usage(session)
        return self.repository.settle_runtime_usage(
            session_id=str(session.session_id),
            status=str(getattr(session, "status", "unknown")),
            physical_actions=max(0, int(getattr(session, "physical_actions", 0))),
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
            model=model,
            cost_micros=cost_micros,
            pricing_version=pricing_version,
        )

    def release(self, session_id: str, *, reason: str = "") -> dict[str, Any] | None:
        return self.repository.release_runtime_quota(session_id, reason=reason)

