"""Authorize Gmail once and save access/refresh tokens for the feature runtime.

Usage:
  python gmail_authorize.py --credentials credentials.json --token gmail_token.json
"""

from __future__ import annotations

import argparse
import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen


SCOPE = "https://www.googleapis.com/auth/gmail.send"
TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"


class _CallbackHandler(BaseHTTPRequestHandler):
    code = ""
    error = ""

    def do_GET(self) -> None:
        query = parse_qs(urlparse(self.path).query)
        type(self).code = str((query.get("code") or [""])[0])
        type(self).error = str((query.get("error") or [""])[0])
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write("授权完成，可以关闭此窗口。".encode("utf-8"))

    def log_message(self, *_args) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--credentials", required=True)
    parser.add_argument("--token", default="gmail_token.json")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()

    raw = json.loads(Path(args.credentials).read_text(encoding="utf-8"))
    installed = raw.get("installed") or raw.get("web") or {}
    client_id = str(installed["client_id"])
    client_secret = str(installed["client_secret"])
    redirect_uri = f"http://127.0.0.1:{args.port}/callback"
    server = ThreadingHTTPServer(("127.0.0.1", args.port), _CallbackHandler)
    auth_url = "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode({
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": SCOPE,
        "access_type": "offline",
        "prompt": "consent",
    })
    print("正在打开 Google 授权页面……")
    webbrowser.open(auth_url)
    while not _CallbackHandler.code and not _CallbackHandler.error:
        server.handle_request()
    server.server_close()
    if _CallbackHandler.error:
        raise SystemExit("授权失败：" + _CallbackHandler.error)

    request = Request(
        TOKEN_ENDPOINT,
        data=urlencode({
            "code": _CallbackHandler.code,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        }).encode("utf-8"),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urlopen(request, timeout=30) as response:
        token = json.loads(response.read().decode("utf-8"))
    Path(args.token).write_text(json.dumps(token, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"授权凭据已保存到：{Path(args.token).resolve()}")
    print("不要把该文件上传、提交或发送到聊天中。")


if __name__ == "__main__":
    main()
