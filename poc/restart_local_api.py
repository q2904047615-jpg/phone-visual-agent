"""Safely stop only this checkout's local uvicorn process before relaunch."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _listener_pid() -> int | None:
    text = subprocess.check_output(
        ["netstat", "-ano"], text=True, stderr=subprocess.STDOUT,
    )
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 5 and fields[0] == "TCP" and fields[1] == "127.0.0.1:8765" and fields[3] == "LISTENING":
            return int(fields[4])
    return None


def _process_info(pid: int) -> dict[str, object]:
    command = (
        "$p=Get-CimInstance Win32_Process -Filter 'ProcessId = {pid}'; "
        "if($null -eq $p){{ exit 2 }}; "
        "[pscustomobject]@{{Name=$p.Name;CommandLine=$p.CommandLine;ExecutablePath=$p.ExecutablePath;ParentProcessId=$p.ParentProcessId}} "
        "| ConvertTo-Json -Compress"
    ).format(pid=pid)
    output = subprocess.check_output(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
        text=True, stderr=subprocess.STDOUT,
    ).strip()
    if not output:
        raise RuntimeError("无法读取 8765 监听进程命令行，拒绝停止未知进程。")
    return json.loads(output)


def main() -> int:
    expected_poc = Path(__file__).resolve().parent
    pid = _listener_pid()
    if pid is None:
        return 0
    info = _process_info(pid)
    name = str(info.get("Name") or "").lower()
    command_line = str(info.get("CommandLine") or "")
    executable = Path(str(info.get("ExecutablePath") or "")).resolve()
    expected_python = (expected_poc / ".venv" / "Scripts" / "python.exe").resolve()
    if name != "python.exe" or executable != expected_python or "uvicorn" not in command_line or "web_app:app" not in command_line:
        raise RuntimeError(f"拒绝停止非本工作树 API 进程 PID {pid}。")
    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, subprocess.SubprocessError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"restart_local_api: {exc}", file=sys.stderr)
        raise SystemExit(2)
