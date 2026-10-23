"""Windows connector CLI; protocol and delivery live in infrastructure."""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import httpx
from agent.infrastructure.remote_connector_worker import DeviceConnector

def main():
    parser = argparse.ArgumentParser(description="Phone Agent Windows device connector")
    parser.add_argument("--base-url", default=os.environ.get("PHONE_AGENT_CONNECTOR_BASE_URL", ""))
    parser.add_argument("--connector-id", default=os.environ.get("PHONE_AGENT_CONNECTOR_ID", "windows-local-01"))
    parser.add_argument("--registry", type=Path, default=Path(__file__).with_name("device_registry.json"))
    parser.add_argument("--text-registry", type=Path, help="Windows 本机可信 ADB Keyboard 注册表")
    parser.add_argument("--app-registry", type=Path, help="Windows 本机可信 App 启动注册表")
    parser.add_argument("--device-id", default="device-local-01")
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    secret = os.environ.get("PHONE_AGENT_CONNECTOR_SECRET", "")
    with DeviceConnector(base_url=args.base_url, connector_id=args.connector_id, secret=secret,
                         registry_path=args.registry, device_id=args.device_id, journal_path=args.journal,
                         text_registry_path=args.text_registry, app_registry_path=args.app_registry) as worker:
        delay = 0.25
        while True:
            try:
                worker.run_once()
                delay = 0.25
            except httpx.TransportError:
                delay = min(15, max(1, delay * 2))
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code < 500:
                    raise SystemExit("连接器认证或协议被拒绝，请核对配置。") from None
                delay = min(15, max(1, delay * 2))
            if args.once:
                return
            time.sleep(delay)


if __name__ == "__main__":
    main()
