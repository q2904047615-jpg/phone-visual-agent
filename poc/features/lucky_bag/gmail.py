"""Gmail API notification sinks with refresh-token support."""

from __future__ import annotations

import base64
from email.message import EmailMessage
import json
import os
from pathlib import Path
import smtplib
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from features.notifications import JsonlNotificationOutbox, NotificationEvent, NotificationSink


class NotificationDeliveryError(RuntimeError):
    """A configured notification provider could not accept the event."""


class GmailSmtpNotificationSink:
    """Send through Gmail SMTP using an app password kept outside the repo."""

    def __init__(
        self,
        *,
        sender: str,
        app_password: str,
        smtp_factory: Callable[..., object] = smtplib.SMTP,
        host: str = "smtp.gmail.com",
        port: int = 587,
        timeout: float = 20.0,
    ) -> None:
        self.sender = str(sender or "").strip()
        self.app_password = str(app_password or "").strip()
        self.smtp_factory = smtp_factory
        self.host = host
        self.port = int(port)
        self.timeout = timeout
        if not self.sender or not self.app_password:
            raise ValueError("Gmail sender 和应用专用密码不能为空。")

    def publish(self, event: NotificationEvent) -> None:
        message = EmailMessage()
        message["To"] = event.recipient
        message["From"] = self.sender
        message["Subject"] = event.subject
        message.set_content(event.body)
        try:
            with self.smtp_factory(self.host, self.port, timeout=self.timeout) as server:
                server.starttls()
                server.login(self.sender, self.app_password)
                server.send_message(message)
        except (OSError, smtplib.SMTPException) as exc:
            raise NotificationDeliveryError("Gmail SMTP 发送失败。") from exc

class GmailApiNotificationSink:
    """Send through Gmail API using an access token or refresh-token credentials."""

    endpoint = "https://gmail.googleapis.com/gmail/v1/users/me/messages/send"
    token_endpoint = "https://oauth2.googleapis.com/token"

    def __init__(
        self,
        *,
        sender: str,
        access_token: str = "",
        refresh_token: str = "",
        client_id: str = "",
        client_secret: str = "",
        opener: Callable[..., object] = urlopen,
        timeout: float = 20.0,
    ) -> None:
        self.access_token = str(access_token or "").strip()
        self.refresh_token = str(refresh_token or "").strip()
        self.client_id = str(client_id or "").strip()
        self.client_secret = str(client_secret or "").strip()
        self.sender = str(sender or "").strip()
        self.opener = opener
        self.timeout = timeout
        if not self.sender:
            raise ValueError("Gmail sender 不能为空。")
        if not self.access_token and not all((self.refresh_token, self.client_id, self.client_secret)):
            raise ValueError("必须提供 Gmail access token，或 refresh token/client id/client secret。")

    def _refresh_access_token(self) -> None:
        request = Request(
            self.token_endpoint,
            data=urlencode({
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "refresh_token": self.refresh_token,
                "grant_type": "refresh_token",
            }).encode("utf-8"),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        try:
            response = self.opener(request, timeout=self.timeout)
            payload = json.loads(response.read().decode("utf-8"))
            token = str(payload.get("access_token") or "").strip()
        except (HTTPError, URLError, OSError, ValueError, TypeError) as exc:
            raise NotificationDeliveryError("Gmail OAuth token 刷新失败。") from exc
        if not token:
            raise NotificationDeliveryError("Gmail OAuth token 刷新响应缺少 access_token。")
        self.access_token = token

    def _send_request(self, request: Request) -> None:
        try:
            response = self.opener(request, timeout=self.timeout)
            response.read()
        except HTTPError as exc:
            if exc.code == 401 and self.refresh_token:
                self._refresh_access_token()
                retry = Request(
                    request.full_url,
                    data=request.data,
                    headers={
                        "Authorization": "Bearer " + self.access_token,
                        "Content-Type": "application/json",
                    },
                    method="POST",
                )
                try:
                    response = self.opener(retry, timeout=self.timeout)
                    response.read()
                    return
                except (HTTPError, URLError, OSError) as retry_exc:
                    raise NotificationDeliveryError("Gmail API 发送失败。") from retry_exc
            raise NotificationDeliveryError("Gmail API 发送失败。") from exc
        except (URLError, OSError) as exc:
            raise NotificationDeliveryError("Gmail API 发送失败。") from exc

    def publish(self, event: NotificationEvent) -> None:
        if not self.access_token and self.refresh_token:
            self._refresh_access_token()
        message = EmailMessage()
        message["To"] = event.recipient
        message["From"] = self.sender
        message["Subject"] = event.subject
        message.set_content(event.body)
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode("ascii").rstrip("=")
        request = Request(
            self.endpoint,
            data=json.dumps({"raw": raw}).encode("utf-8"),
            headers={
                "Authorization": "Bearer " + self.access_token,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        self._send_request(request)


class DurableNotificationRouter:
    """Persist every event, then optionally deliver it through Gmail."""

    def __init__(
        self,
        *,
        outbox: JsonlNotificationOutbox,
        remote: NotificationSink | None = None,
    ) -> None:
        self.outbox = outbox
        self.remote = remote

    def publish(self, event: NotificationEvent) -> None:
        self.outbox.publish(event)
        if self.remote is not None:
            self.remote.publish(event)


def _load_file_credentials() -> dict[str, str]:
    token_path = os.environ.get("GMAIL_TOKEN_FILE", "").strip()
    credentials_path = os.environ.get("GMAIL_CREDENTIALS_FILE", "").strip()
    values: dict[str, str] = {}
    if token_path and Path(token_path).is_file():
        try:
            token = json.loads(Path(token_path).read_text(encoding="utf-8"))
            values["access_token"] = str(token.get("access_token") or "")
            values["refresh_token"] = str(token.get("refresh_token") or "")
        except (OSError, ValueError, TypeError):
            pass
    if credentials_path and Path(credentials_path).is_file():
        try:
            raw = json.loads(Path(credentials_path).read_text(encoding="utf-8"))
            installed = raw.get("installed") or raw.get("web") or {}
            values["client_id"] = str(installed.get("client_id") or "")
            values["client_secret"] = str(installed.get("client_secret") or "")
        except (OSError, ValueError, TypeError):
            pass
    return values
def configured_gmail_sink() -> NotificationSink | None:
    sender = os.environ.get("GMAIL_SENDER", "").strip()
    app_password = os.environ.get("GMAIL_APP_PASSWORD", "").strip()
    if sender and app_password:
        return GmailSmtpNotificationSink(sender=sender, app_password=app_password)

    values = _load_file_credentials()
    access_token = os.environ.get("GMAIL_ACCESS_TOKEN", "").strip() or values.get("access_token", "")
    refresh_token = os.environ.get("GMAIL_REFRESH_TOKEN", "").strip() or values.get("refresh_token", "")
    client_id = os.environ.get("GMAIL_CLIENT_ID", "").strip() or values.get("client_id", "")
    client_secret = os.environ.get("GMAIL_CLIENT_SECRET", "").strip() or values.get("client_secret", "")
    if not sender:
        return None
    if not access_token and not all((refresh_token, client_id, client_secret)):
        return None
    return GmailApiNotificationSink(
        sender=sender,
        access_token=access_token,
        refresh_token=refresh_token,
        client_id=client_id,
        client_secret=client_secret,
    )


def gmail_configuration_status() -> dict[str, str | bool]:
    """Return secret-free preflight information for the lucky-bag UI."""

    sender = bool(os.environ.get("GMAIL_SENDER", "").strip())
    smtp = sender and bool(os.environ.get("GMAIL_APP_PASSWORD", "").strip())
    values = _load_file_credentials()
    api = sender and bool(
        os.environ.get("GMAIL_ACCESS_TOKEN", "").strip() or values.get("access_token", "")
        or (
            (os.environ.get("GMAIL_REFRESH_TOKEN", "").strip() or values.get("refresh_token", ""))
            and (os.environ.get("GMAIL_CLIENT_ID", "").strip() or values.get("client_id", ""))
            and (os.environ.get("GMAIL_CLIENT_SECRET", "").strip() or values.get("client_secret", ""))
        )
    )
    configured = bool(smtp or api)
    return {
        "configured": configured,
        "mode": "smtp_app_password" if smtp else "gmail_api" if api else "local_outbox_only",
        "message": "Gmail 已配置，可发送疑似中奖通知。" if configured else "Gmail 未配置，中奖通知只能写入本地队列。",
    }


__all__ = [
    "DurableNotificationRouter",
    "GmailApiNotificationSink",
    "GmailSmtpNotificationSink",
    "NotificationDeliveryError",
    "configured_gmail_sink",
    "gmail_configuration_status",
]
