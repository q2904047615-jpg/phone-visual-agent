"""Read-only preflight for the single-executor / multi-ingress topology.

It validates the process contract without importing ``web_app`` or printing
any environment value that may contain a credential.  The command is safe to
run on a staging or production host before stopping the existing service.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlsplit


PROTOCOL = "2026-10-19-multi-worker-deployment-preflight-v1"


@dataclass(frozen=True)
class Check:
    check_id: str
    status: str
    detail: str


@dataclass(frozen=True)
class Report:
    protocol_version: str
    role: str
    status: str
    checks: tuple[Check, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "protocol_version": self.protocol_version,
            "role": self.role,
            "status": self.status,
            "checks": [asdict(check) for check in self.checks],
        }


def _port(value: str, key: str, checks: list[Check]) -> int | None:
    try:
        result = int(value)
    except (TypeError, ValueError):
        checks.append(Check(key, "blocked", f"{key} 必须是 1-65535 的端口。"))
        return None
    if not 1 <= result <= 65535:
        checks.append(Check(key, "blocked", f"{key} 必须是 1-65535 的端口。"))
        return None
    checks.append(Check(key, "pass", "端口格式有效。"))
    return result


def _state_dir(values: dict[str, str], checks: list[Check]) -> Path | None:
    raw = values.get("PHONE_AGENT_STATE_DIR", "").strip()
    if not raw:
        checks.append(Check("state_dir", "blocked", "缺少 PHONE_AGENT_STATE_DIR。"))
        return None
    path = Path(raw)
    if not path.is_absolute():
        checks.append(Check("state_dir", "blocked", "PHONE_AGENT_STATE_DIR 必须是绝对路径。"))
        return None
    if not path.is_dir():
        checks.append(Check("state_dir", "blocked", "PHONE_AGENT_STATE_DIR 不存在或不是目录。"))
        return None
    if not os.access(path, os.W_OK):
        checks.append(Check("state_dir", "blocked", "PHONE_AGENT_STATE_DIR 不可写。"))
        return None
    if path.joinpath("runtime.sqlite3").exists() and not path.joinpath("dispatch.sqlite3").exists():
        checks.append(Check("shared_dispatch_store", "blocked",
                            "已有 Runtime 状态但缺少 dispatch.sqlite3；上线前需由 owner 初始化并备份。"))
    else:
        checks.append(Check("state_dir", "pass", "状态目录为绝对路径且可写。"))
    return path


def evaluate(values: dict[str, str] | None = None, *, role: str = "both") -> Report:
    env = dict(os.environ if values is None else values)
    checks: list[Check] = []
    if role not in {"owner", "ingress", "both"}:
        return Report(PROTOCOL, role, "blocked", (Check("role", "blocked", "role 必须是 owner、ingress 或 both。"),))
    path = _state_dir(env, checks)
    owner_host = env.get("ROBOT_API_HOST", "127.0.0.1").strip()
    owner_port = _port(env.get("ROBOT_API_PORT", ""), "executor_port", checks)
    if owner_host != "127.0.0.1":
        checks.append(Check("executor_host", "blocked", "执行端必须只绑定 127.0.0.1。"))
    else:
        checks.append(Check("executor_host", "pass", "执行端绑定回环地址。"))

    if role != "owner":
        target = env.get("PHONE_AGENT_EXECUTOR_URL", "").strip()
        parsed = urlsplit(target)
        try:
            parsed_port = parsed.port
        except ValueError:
            parsed_port = None
        if (parsed.scheme, parsed.hostname, parsed.path, parsed.query, parsed.fragment,
                parsed.username, parsed.password) != ("http", "127.0.0.1", "", "", "", None, None) or not parsed_port:
            checks.append(Check("executor_url", "blocked", "PHONE_AGENT_EXECUTOR_URL 必须是无路径的回环 HTTP 地址。"))
        else:
            checks.append(Check("executor_url", "pass", "入口指向显式回环 HTTP 执行端口。"))
        ingress_port = _port(env.get("ROBOT_API_PORT", ""), "ingress_port", checks)
        if ingress_port is not None and parsed_port == ingress_port:
            checks.append(Check("port_separation", "blocked", "入口端口不能与执行端端口相同。"))
        elif ingress_port is not None and parsed_port is not None:
            checks.append(Check("port_separation", "pass", "入口和执行端端口已分离。"))
    if role == "owner" and env.get("PHONE_AGENT_EXECUTOR_MODE") != "1":
        checks.append(Check("owner_mode", "blocked", "owner 必须设置 PHONE_AGENT_EXECUTOR_MODE=1。"))
    elif role == "owner":
        checks.append(Check("owner_mode", "pass", "owner 使用 single_executor 组合根。"))
    if role == "ingress" and env.get("PHONE_AGENT_EXECUTOR_MODE") == "1":
        checks.append(Check("ingress_mode", "blocked", "ingress 不得加载 PHONE_AGENT_EXECUTOR_MODE=1。"))
    elif role == "ingress":
        checks.append(Check("ingress_mode", "pass", "ingress 不加载 Runtime 组合根。"))
    if path is not None and role == "both":
        checks.append(Check("shared_state", "pass", "当前环境可访问共享状态目录；两个服务必须使用同一路径。"))
    status = "pass" if all(item.status == "pass" for item in checks) else "blocked"
    return Report(PROTOCOL, role, status, tuple(checks))


def main() -> int:
    parser = argparse.ArgumentParser(description="只读检查单执行进程/多入口部署合同，不输出 Secret。")
    parser.add_argument("--role", choices=("owner", "ingress", "both"), default="both")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate(role=args.role)
    encoded = json.dumps(report.as_dict(), ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report.status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
