from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

# Support `python poc/evals/check_external_alert_config.py` from the repository root.
PROJECT_ROOT = Path(__file__).parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.lucky_bag.gmail import gmail_configuration_status


EXTERNAL_ALERT_PROTOCOL = "2026-10-16-external-alert-preflight-v1"


@dataclass(frozen=True)
class ExternalAlertReport:
    protocol_version: str
    status: str
    external_delivery: str
    message: str
    secret_values_emitted: bool = False

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_external_alert_config() -> ExternalAlertReport:
    """Report provider readiness without requiring or echoing credentials.

    This checks credential presence only. It does not validate credentials,
    inspect outbox writability, enable a provider, or send a notification.
    """

    configured = gmail_configuration_status()
    if bool(configured.get("configured")):
        return ExternalAlertReport(
            protocol_version=EXTERNAL_ALERT_PROTOCOL,
            status="configured",
            external_delivery=str(configured.get("mode") or "configured"),
            message="Gmail 适配器所需配置字段已提供；凭据有效性、告警接线和真实投递尚未验证。",
        )
    return ExternalAlertReport(
        protocol_version=EXTERNAL_ALERT_PROTOCOL,
        status="not_configured",
        external_delivery="optional_not_configured",
        message="未检测到完整的 Gmail 配置；本预检不验证 outbox 可写性或消息投递。",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="只读检查外部告警准备状态，不输出 Secret。")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate_external_alert_config()
    encoded = json.dumps(report.as_dict(), ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
