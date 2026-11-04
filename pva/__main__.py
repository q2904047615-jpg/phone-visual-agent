"""Start the web console:  python -m pva [--mock] [--port 8765] [--no-browser]"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import webbrowser

USER_ENV_NAMES = ("DASHSCOPE_API_KEY", "VISION_MODEL", "VISION_MODEL_BASE_URL", "LUCKY_BAG_NOTIFICATION_RECIPIENT",
                  "GMAIL_SENDER", "GMAIL_APP_PASSWORD", "GMAIL_TOKEN_FILE", "GMAIL_CREDENTIALS_FILE",
                  "GMAIL_REFRESH_TOKEN", "GMAIL_ACCESS_TOKEN", "GMAIL_CLIENT_ID", "GMAIL_CLIENT_SECRET")


def load_windows_user_environment() -> None:
    """``setx`` values only reach new consoles; read them directly so a stale shell still works."""
    if os.name != "nt":
        return
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            for name in USER_ENV_NAMES:
                if not os.environ.get(name):
                    try:
                        os.environ[name] = str(winreg.QueryValueEx(key, name)[0])
                    except FileNotFoundError:
                        pass
    except OSError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="手机视觉 Agent 网页控制台")
    parser.add_argument("--mock", action="store_true", help="所有设备使用模拟手机（无硬件）")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PVA_PORT", "8765")))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if args.mock:
        os.environ["PVA_MOCK"] = "1"
    load_windows_user_environment()
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    import uvicorn
    from pva.web.server import create_app

    app = create_app()
    url = f"http://127.0.0.1:{args.port}/"
    print(f"控制台：{url}   （Ctrl+C 退出）")
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
