from __future__ import annotations

import argparse
import json
import os
import socket
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping
from urllib.parse import urlparse


REQUIRED_SECRET_KEYS = (
    "GITHUB_OAUTH_CLIENT_SECRET",
    "DASHSCOPE_API_KEY",
)
REQUIRED_VALUE_KEYS = (
    "PUBLIC_BASE_URL",
    "GITHUB_OAUTH_CLIENT_ID",
    "GITHUB_OAUTH_IDENTITY_MAP_FILE",
    "GITHUB_OAUTH_REDIRECT_URI",
    "VISION_MODEL",
    "VISION_MODEL_BASE_URL",
    "PHONE_VISUAL_AGENT_LEASE_DIR",
)
GITHUB_CALLBACK_PATH = "/api/control-plane/auth/github/callback"


@dataclass(frozen=True)
class ConfigCheck:
    check_id: str
    status: str
    detail: str
    sensitive: bool = False


@dataclass(frozen=True)
class PublicGaConfigReport:
    protocol_version: str
    status: str
    checks: tuple[ConfigCheck, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "protocol_version": self.protocol_version,
            "status": self.status,
            "checks": [asdict(check) for check in self.checks],
        }


PUBLIC_GA_CONFIG_PROTOCOL = "2026-10-16-public-ga-config-v1"


def _present(environ: Mapping[str, str], key: str) -> bool:
    return bool(str(environ.get(key, "")).strip())


def _is_local_host(hostname: str | None) -> bool:
    if not hostname:
        return True
    normalized = hostname.strip("[]").lower().rstrip(".")
    if normalized in {"localhost", "127.0.0.1", "::1"}:
        return True
    try:
        return ip_is_loopback(normalized)
    except ValueError:
        return False


def ip_is_loopback(hostname: str) -> bool:
    return bool(__import__("ipaddress").ip_address(hostname).is_loopback)


def _parse_url(value: str, *, key: str, require_https: bool) -> ConfigCheck | None:
    parsed = urlparse(value)
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        return ConfigCheck(key.lower(), "blocked", f"{key} 不能包含用户信息、查询参数或片段。")
    if not parsed.hostname or parsed.path not in {"", "/"} and key == "PUBLIC_BASE_URL":
        return ConfigCheck(key.lower(), "blocked", f"{key} 必须是 origin，不应包含路径。")
    if require_https and parsed.scheme != "https":
        return ConfigCheck(key.lower(), "blocked", f"{key} 必须使用 HTTPS。")
    if not require_https and parsed.scheme not in {"https", "http"}:
        return ConfigCheck(key.lower(), "blocked", f"{key} 必须使用 HTTP 或 HTTPS。")
    if require_https and _is_local_host(parsed.hostname):
        return ConfigCheck(key.lower(), "blocked", f"{key} 不能指向本机回环地址。")
    return None


def _check_identity_map(environ: Mapping[str, str]) -> ConfigCheck:
    key = "github_oauth_identity_map"
    raw_path = str(environ.get("GITHUB_OAUTH_IDENTITY_MAP_FILE", "")).strip()
    if not raw_path:
        return ConfigCheck(key, "blocked", "缺少 GITHUB_OAUTH_IDENTITY_MAP_FILE。")
    path = Path(raw_path)
    if not path.is_absolute():
        return ConfigCheck(key, "blocked", "身份映射文件必须使用绝对路径。")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return ConfigCheck(key, "blocked", "身份映射文件不存在、不可读或不是合法 JSON。")
    mappings = payload.get("mappings", payload) if isinstance(payload, dict) else None
    if not isinstance(mappings, dict) or not mappings:
        return ConfigCheck(key, "blocked", "身份映射必须是非空对象。")
    for subject, target in mappings.items():
        if not str(subject).isdigit() or not isinstance(target, dict):
            return ConfigCheck(key, "blocked", "身份映射键必须是 GitHub 数字用户 ID，值必须是对象。")
        if not str(target.get("organization_id", "")).strip() or not str(target.get("user_id", "")).strip():
            return ConfigCheck(key, "blocked", "每个身份映射都必须包含 organization_id 和 user_id。")
    return ConfigCheck(key, "pass", f"映射文件可读，已发现 {len(mappings)} 个稳定用户 ID。")


def _check_lease_dir(environ: Mapping[str, str]) -> ConfigCheck:
    key = "runtime_lease_dir"
    raw_path = str(environ.get("PHONE_VISUAL_AGENT_LEASE_DIR", "")).strip()
    if not raw_path:
        return ConfigCheck(key, "blocked", "缺少 PHONE_VISUAL_AGENT_LEASE_DIR。")
    path = Path(raw_path)
    if not path.is_absolute():
        return ConfigCheck(key, "blocked", "租约目录必须使用绝对路径。")
    if not path.is_dir():
        return ConfigCheck(key, "blocked", "租约目录不存在或不是目录。")
    if not os.access(path, os.W_OK):
        return ConfigCheck(key, "blocked", "租约目录不可写。")
    return ConfigCheck(key, "pass", "租约目录存在且可写。")


def evaluate_public_ga_config(environ: Mapping[str, str] | None = None) -> PublicGaConfigReport:
    values = os.environ if environ is None else environ
    checks: list[ConfigCheck] = []
    missing_values = [key for key in REQUIRED_VALUE_KEYS if not _present(values, key)]
    if missing_values:
        checks.append(ConfigCheck("required_values", "blocked", "缺少非 Secret 配置：" + ", ".join(missing_values)))
    else:
        checks.append(ConfigCheck("required_values", "pass", "必需的非 Secret 配置均已提供。"))
    missing_secrets = [key for key in REQUIRED_SECRET_KEYS if not _present(values, key)]
    if missing_secrets:
        checks.append(ConfigCheck("required_secrets", "blocked", "缺少 Secret 注入：" + ", ".join(missing_secrets), True))
    else:
        checks.append(ConfigCheck("required_secrets", "pass", "必需 Secret 已注入；值不会输出。", True))

    base = str(values.get("PUBLIC_BASE_URL", "")).strip()
    if base:
        check = _parse_url(base, key="PUBLIC_BASE_URL", require_https=True)
        checks.append(check or ConfigCheck("public_base_url", "pass", "公网基址为 HTTPS 且不是本机回环地址。"))
    redirect = str(values.get("GITHUB_OAUTH_REDIRECT_URI", "")).strip()
    if redirect:
        check = _parse_url(redirect, key="GITHUB_OAUTH_REDIRECT_URI", require_https=True)
        if check:
            checks.append(check)
        elif base:
            base_url = urlparse(base)
            redirect_url = urlparse(redirect)
            if (redirect_url.scheme, redirect_url.netloc) != (base_url.scheme, base_url.netloc):
                checks.append(ConfigCheck("github_oauth_redirect", "blocked", "OAuth 回调必须与 PUBLIC_BASE_URL 同源。"))
            elif redirect_url.path != GITHUB_CALLBACK_PATH:
                checks.append(ConfigCheck("github_oauth_redirect", "blocked", f"OAuth 回调路径必须为 {GITHUB_CALLBACK_PATH}。"))
            else:
                checks.append(ConfigCheck("github_oauth_redirect", "pass", "OAuth 回调与公网基址同源且路径正确。"))
    checks.append(_check_identity_map(values))
    checks.append(_check_lease_dir(values))
    model_base = str(values.get("VISION_MODEL_BASE_URL", "")).strip()
    if model_base:
        check = _parse_url(model_base, key="VISION_MODEL_BASE_URL", require_https=False)
        checks.append(check or ConfigCheck("vision_model_base_url", "pass", "视觉模型地址格式有效。"))
    else:
        checks.append(ConfigCheck("vision_model_base_url", "blocked", "缺少 VISION_MODEL_BASE_URL。"))
    status = "pass" if all(check.status == "pass" for check in checks) else "blocked"
    return PublicGaConfigReport(PUBLIC_GA_CONFIG_PROTOCOL, status, tuple(checks))


def main() -> int:
    parser = argparse.ArgumentParser(description="只读检查公共 GA 配置，不输出 Secret。")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate_public_ga_config()
    encoded = json.dumps(report.as_dict(), ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report.status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
