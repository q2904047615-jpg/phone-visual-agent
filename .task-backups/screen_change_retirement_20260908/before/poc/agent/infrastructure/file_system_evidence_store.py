"""Atomic JSON persistence for authoritative universal-agent evidence."""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
from typing import Any, Callable

from agent.domain import EvidenceStoreError
from agent.infrastructure.atomic_files import atomic_replace_bytes


class FileSystemAgentEvidenceStore:
    """Persist authoritative session evidence with same-directory replaces."""

    def __init__(self, run_dir: Path, *, replace_file: Callable[[Path, Path], None] | None=None) -> None:
        self.run_dir = Path(run_dir)
        self._replace_file = replace_file or (lambda source, target: os.replace(source, target))

    @staticmethod
    def _payload(value: Any) -> dict[str, Any]:
        if isinstance(value, Mapping):
            return dict(value)
        for method_name in ('snapshot', 'to_dict'):
            method = getattr(value, method_name, None)
            if callable(method):
                payload = method()
                if isinstance(payload, Mapping):
                    return dict(payload)
        raise EvidenceStoreError("证据对象不能转换为 JSON 对象。")

    def write_json(self, name: str, payload: Any) -> Path:
        clean_name = str(name or "").strip()
        if not clean_name or Path(clean_name).name != clean_name or (not clean_name.endswith('.json')):
            raise EvidenceStoreError(f"证据文件名无效：{clean_name!r}")
        try:
            encoded = json.dumps(self._payload(payload), ensure_ascii=False, indent=2)
        except (TypeError, ValueError, EvidenceStoreError) as exc:
            raise EvidenceStoreError(f"证据不能序列化：{exc}") from exc

        target = self.run_dir / clean_name
        try:
            atomic_replace_bytes(target, (encoded + '\n').encode('utf-8'), replace_file=self._replace_file)
        except OSError as exc:
            raise EvidenceStoreError(f'证据原子写入失败：{clean_name}：{exc}') from exc
        return target

    def write_session(self, session: Any) -> Path:
        return self.write_json("session.json", session)

    def write_trusted_observation(self, step_number: int, observation: Any) -> Path:
        return self.write_json(f'trusted_observation_step_{int(step_number)}.json', observation)

    def write_qwen_decision(self, step_number: int, decision: Any) -> Path:
        return self.write_json(f'qwen_decision_step_{int(step_number)}.json', decision)

    def write_controller_decision(self, step_number: int, decision: Any) -> Path:
        return self.write_json(f'controller_decision_step_{int(step_number)}.json', decision)

    def write_verification(self, step_number: int, verification: Any) -> Path:
        return self.write_json(f'verification_step_{int(step_number)}.json', verification)

    def write_post_action_transition(self, step_number: int, transition: Any) -> Path:
        return self.write_json(f'post_action_transition_step_{int(step_number)}.json', transition)

    def write_confirmation_failure(self, step_number: int, transition: Any) -> Path:
        return self.write_json(f'confirmation_failure_step_{int(step_number)}.json', transition)

    def read_report(self) -> dict[str, Any] | None:
        target = self.run_dir / "report.json"
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            return None
        return dict(payload) if isinstance(payload, Mapping) else None

    def write_report(self, report: Any) -> Path:
        return self.write_json("report.json", report)
