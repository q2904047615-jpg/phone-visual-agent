from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Support both `python poc/evals/check_ga_readiness.py` from the repository
# root and `python -m evals.check_ga_readiness` from `poc`.
PROJECT_ROOT = Path(__file__).parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.domain.ga_readiness import evaluate_device_registry


def main() -> int:
    parser = argparse.ArgumentParser(description="输出只读多设备/GA 准备评估。")
    parser.add_argument("--registry", type=Path, default=Path(__file__).parents[1] / "device_registry.json")
    parser.add_argument("--real-evidence", action="store_true", help="仅在已有真实第二机位证据时显式传入。")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate_device_registry(args.registry, real_evidence=args.real_evidence).as_dict()
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
