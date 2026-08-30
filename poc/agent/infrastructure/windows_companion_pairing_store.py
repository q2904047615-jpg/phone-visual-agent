"""Windows DPAPI persistence for Companion IME pairing credentials."""

from __future__ import annotations

import base64
import ctypes
from ctypes import wintypes
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any, Protocol

from agent.domain.text_transport import (
    TEXT_TRANSPORT_PROTOCOL,
    TextTransportAuthenticationError,
    TextTransportContractError,
)
from agent.infrastructure.atomic_files import atomic_replace_bytes, json_bytes


WINDOWS_PAIRING_STORE_VERSION = "2026-08-30-windows-companion-pairing-v1"
_RECORD_FIELDS = frozenset({"record_version", "protocol_version", "device_id", "pairing_id",
    "installation_id", "created_at_epoch", "protected_shared_key"})


class PairingSecretProtector(Protocol):
    def protect(self, plaintext: bytes, *, entropy: bytes) -> bytes: ...

    def unprotect(self, ciphertext: bytes, *, entropy: bytes) -> bytes: ...


class _DataBlob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_ubyte))]


def _input_blob(value: bytes) -> tuple[_DataBlob, Any]:
    buffer = ctypes.create_string_buffer(value)
    return _DataBlob(len(value), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte))), buffer


class WindowsDpapiProtector:
    """Current-user DPAPI adapter; key material never leaves protected memory on disk."""

    _UI_FORBIDDEN = 0x1

    @staticmethod
    def _libraries() -> tuple[Any, Any]:
        if os.name != "nt":
            raise RuntimeError("Windows DPAPI 只可在 Windows 使用。")
        return ctypes.WinDLL("crypt32", use_last_error=True), ctypes.WinDLL("kernel32", use_last_error=True)

    def protect(self, plaintext: bytes, *, entropy: bytes) -> bytes:
        if not isinstance(plaintext, bytes) or not plaintext:
            raise TextTransportAuthenticationError("待保护的 Companion IME 密钥无效。")
        crypt32, kernel32 = self._libraries()
        source, source_buffer = _input_blob(plaintext)
        entropy_blob, entropy_buffer = _input_blob(entropy)
        output = _DataBlob()
        crypt32.CryptProtectData.argtypes = [ctypes.POINTER(_DataBlob), wintypes.LPCWSTR,
            ctypes.POINTER(_DataBlob), ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD,
            ctypes.POINTER(_DataBlob)]
        crypt32.CryptProtectData.restype = wintypes.BOOL
        result = crypt32.CryptProtectData(ctypes.byref(source), "Visual Agent Companion IME",
            ctypes.byref(entropy_blob), None, None, self._UI_FORBIDDEN, ctypes.byref(output))
        del source_buffer, entropy_buffer
        if not result:
            raise OSError(ctypes.get_last_error(), "CryptProtectData failed")
        try:
            return ctypes.string_at(output.pbData, output.cbData)
        finally:
            kernel32.LocalFree(output.pbData)

    def unprotect(self, ciphertext: bytes, *, entropy: bytes) -> bytes:
        if not isinstance(ciphertext, bytes) or not ciphertext:
            raise TextTransportAuthenticationError("Companion IME 密钥密文无效。")
        crypt32, kernel32 = self._libraries()
        source, source_buffer = _input_blob(ciphertext)
        entropy_blob, entropy_buffer = _input_blob(entropy)
        output = _DataBlob()
        crypt32.CryptUnprotectData.argtypes = [ctypes.POINTER(_DataBlob), ctypes.c_void_p,
            ctypes.POINTER(_DataBlob), ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD,
            ctypes.POINTER(_DataBlob)]
        crypt32.CryptUnprotectData.restype = wintypes.BOOL
        result = crypt32.CryptUnprotectData(ctypes.byref(source), None, ctypes.byref(entropy_blob), None, None,
            self._UI_FORBIDDEN, ctypes.byref(output))
        del source_buffer, entropy_buffer
        if not result:
            raise OSError(ctypes.get_last_error(), "CryptUnprotectData failed")
        try:
            return ctypes.string_at(output.pbData, output.cbData)
        finally:
            kernel32.LocalFree(output.pbData)


@dataclass(frozen=True, repr=False)
class StoredCompanionPairing:
    device_id: str
    pairing_id: str
    installation_id: str
    created_at_epoch: float
    _shared_key: bytes = field(repr=False, compare=False)

    def __repr__(self) -> str:
        return (f"StoredCompanionPairing(device_id={self.device_id!r}, pairing_id={self.pairing_id!r}, "
            f"installation_id={self.installation_id!r}, created_at_epoch={self.created_at_epoch!r}, "
            "shared_key=<redacted>)")

    def validate(self) -> None:
        identifier = r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}"
        for name in ("device_id", "pairing_id", "installation_id"):
            if not isinstance(getattr(self, name), str) or not re.fullmatch(identifier, getattr(self, name)):
                raise TextTransportContractError(f"持久 Companion IME pairing 的 {name} 无效。")
        if (isinstance(self.created_at_epoch, bool) or not isinstance(self.created_at_epoch, (int, float))
            or not math.isfinite(float(self.created_at_epoch))):
            raise TextTransportContractError("持久 Companion IME pairing 时间无效。")
        if not isinstance(self._shared_key, bytes) or len(self._shared_key) != 32:
            raise TextTransportAuthenticationError("持久 Companion IME pairing 密钥无效。")

    def shared_key(self) -> bytes:
        self.validate()
        return bytes(self._shared_key)

    def to_dict(self) -> dict[str, Any]:
        """Safe metadata view, deliberately excluding plaintext and ciphertext."""

        self.validate()
        return {"record_version": WINDOWS_PAIRING_STORE_VERSION, "protocol_version": TEXT_TRANSPORT_PROTOCOL,
            "device_id": self.device_id, "pairing_id": self.pairing_id,
            "installation_id": self.installation_id, "created_at_epoch": float(self.created_at_epoch),
            "shared_key_present": True}


class WindowsCompanionPairingStore:
    """One encrypted record per device with same-directory atomic replacement."""

    def __init__(self, directory: Path, *, protector: PairingSecretProtector | None=None) -> None:
        self._directory = Path(directory)
        self._protector = protector or WindowsDpapiProtector()

    def _path(self, device_id: str) -> Path:
        if not isinstance(device_id, str) or not device_id:
            raise TextTransportContractError("Companion IME device_id 无效。")
        name = hashlib.sha256(device_id.encode("utf-8")).hexdigest()
        return self._directory / f"{name}.pairing.json"

    @staticmethod
    def _entropy(*, device_id: str, pairing_id: str, installation_id: str) -> bytes:
        return (f"{WINDOWS_PAIRING_STORE_VERSION}\0{TEXT_TRANSPORT_PROTOCOL}\0{device_id}\0{pairing_id}\0"
            f"{installation_id}").encode("utf-8")

    def save(self, credential: StoredCompanionPairing) -> Path:
        credential.validate()
        entropy = self._entropy(device_id=credential.device_id, pairing_id=credential.pairing_id,
            installation_id=credential.installation_id)
        protected = self._protector.protect(credential.shared_key(), entropy=entropy)
        if not isinstance(protected, bytes) or not protected:
            raise TextTransportAuthenticationError("Companion IME 密钥保护结果无效。")
        record = {"record_version": WINDOWS_PAIRING_STORE_VERSION,
            "protocol_version": TEXT_TRANSPORT_PROTOCOL, "device_id": credential.device_id,
            "pairing_id": credential.pairing_id, "installation_id": credential.installation_id,
            "created_at_epoch": float(credential.created_at_epoch),
            "protected_shared_key": base64.b64encode(protected).decode("ascii")}
        target = atomic_replace_bytes(self._path(credential.device_id), json_bytes(record))
        try:
            target.chmod(stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass
        return target

    def load(self, device_id: str) -> StoredCompanionPairing | None:
        path = self._path(device_id)
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TextTransportContractError("Windows Companion IME pairing 记录不可读。") from exc
        if not isinstance(value, dict) or set(value) != _RECORD_FIELDS:
            raise TextTransportContractError("Windows Companion IME pairing 记录字段无效。")
        if (value["record_version"] != WINDOWS_PAIRING_STORE_VERSION
            or value["protocol_version"] != TEXT_TRANSPORT_PROTOCOL or value["device_id"] != device_id):
            raise TextTransportContractError("Windows Companion IME pairing 记录身份无效。")
        try:
            protected = base64.b64decode(value["protected_shared_key"], validate=True)
        except (TypeError, ValueError) as exc:
            raise TextTransportContractError("Windows Companion IME pairing 密文格式无效。") from exc
        entropy = self._entropy(device_id=device_id, pairing_id=value["pairing_id"],
            installation_id=value["installation_id"])
        try:
            shared_key = self._protector.unprotect(protected, entropy=entropy)
        except Exception as exc:
            raise TextTransportAuthenticationError("Windows Companion IME pairing 解密失败。") from exc
        credential = StoredCompanionPairing(device_id=device_id, pairing_id=value["pairing_id"],
            installation_id=value["installation_id"], created_at_epoch=value["created_at_epoch"],
            _shared_key=shared_key)
        credential.validate()
        return credential

    def remove(self, device_id: str) -> None:
        self._path(device_id).unlink(missing_ok=True)


__all__ = ["PairingSecretProtector", "StoredCompanionPairing", "WINDOWS_PAIRING_STORE_VERSION",
    "WindowsCompanionPairingStore", "WindowsDpapiProtector"]
