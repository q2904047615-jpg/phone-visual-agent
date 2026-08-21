from __future__ import annotations

import json
import os
import re
import secrets
import sqlite3
import subprocess
import threading
import time
import uuid
import webbrowser
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Literal

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr

from capability_acceptance import (
    CapabilityAcceptanceError,
    PROMOTABLE_ACTIONS,
)
from capability_acceptance_runtime import CapabilityAcceptanceManager
from capability_acceptance_planner import CapabilityAcceptanceTaskGraphPlanner
from intent_provider import DeepSeekIntentProvider, IntentProviderError
from generic_action_adapter import (
    GenericActionAdapterError,
    GenericSingleActionAdapter,
    persist_observer_failure_diagnostic,
    stable_qwerty_ocr_anchors,
)
from generic_scene_observer import GenericSceneObserver
from input_value_lineage import TypedInputLineageStore
from generic_step_planner import GenericStepPlanner, GenericStepPlanningError
from deepseek_task_graph import DeepSeekTaskGraphPlanner, TaskGraphError
from qwen_visual_decision import QwenVisualDecisionObserver
from device_exclusivity import InterProcessLease, SHARED_DEVICE_LEASE_DIR
from universal_agent_orchestrator import (
    DeviceTaskRegistry,
    POST_ACTION_TRANSITION_PROTOCOL_VERSION,
    UniversalAgentOrchestrator,
    UniversalAgentOrchestratorError,
    UniversalAgentSessionState,
)
from universal_action_controller import (
    UNIVERSAL_CONTROLLER_PROTOCOL_VERSION,
    UniversalActionController,
    UniversalActionError,
)
from ui_scene import UI_SCENE_PROTOCOL_VERSION
from task_semantic_ir import (
    AUTHORITY_REPORT_PROTOCOL,
    RISK_POLICY_PROTOCOL,
    TASK_SEMANTIC_IR_PROTOCOL,
)
from canonical_action_protocol import CANONICAL_ACTION_PROTOCOL

from robot_core import (
    MockRobotController,
    RobotController,
    RobotWorkflowError,
    WEB_OUTPUT_DIR,
)
from vision_agent import (
    DashScopeVisionProvider,
    VisionAgentError,
)


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
DB_PATH = WEB_OUTPUT_DIR / "tasks.sqlite3"
CONTROL_TOKEN = secrets.token_urlsafe(24)
DEVICE_REGISTRY_PATH = Path(
    os.environ.get(
        "ROBOT_DEVICE_REGISTRY",
        Path(__file__).with_name("device_registry.json"),
    )
)
def current_code_revision() -> str:
    """Return a reproducible revision; dirty worktrees are never promotable."""

    repository = ROOT.parent
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
                timeout=3,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise CapabilityAcceptanceError(f"无法读取当前 Git 提交：{exc}") from exc
    if not revision:
        raise CapabilityAcceptanceError("当前 Git 提交为空。")
    return revision + ("+dirty" if dirty else "")

APP_CATALOG = [
    {
        "id": "universal-agent",
        "name": "通用视觉操作 Agent",
        "icon": "智",
        "route": "#/",
        "operations": [],
        "enabled": True,
        "note": "唯一默认入口；按当前画面逐步观察、执行和验证",
    },
]
def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class TaskStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._revision = 0
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    app_id TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    params_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error TEXT,
                    result_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    confirmed_at TEXT,
                    started_at TEXT,
                    finished_at TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT,
                    level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                UPDATE tasks
                SET status='failed',
                    error='服务上次退出时任务仍在执行，已安全停止。',
                    finished_at=?,
                    updated_at=?
                WHERE status IN ('queued', 'running')
                """,
                (now_iso(), now_iso()),
            )

    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision

    def _bump(self) -> None:
        self._revision += 1

    def _append_jsonl(self, payload: dict[str, Any]) -> None:
        log_path = self.path.parent / "tasks.jsonl"
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["params"] = json.loads(item.pop("params_json"))
        result_json = item.pop("result_json")
        item["result"] = json.loads(result_json) if result_json else None
        return item

    def create(
        self, app_id: str, operation: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        task_id = str(uuid.uuid4())
        timestamp = now_iso()
        with self._lock, self._connection() as connection:
            connection.execute(
                """
                INSERT INTO tasks (
                    id, app_id, operation, params_json, status,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'awaiting_confirmation', ?, ?)
                """,
                (
                    task_id,
                    app_id,
                    operation,
                    json.dumps(params, ensure_ascii=False),
                    timestamp,
                    timestamp,
                ),
            )
            connection.execute(
                """
                INSERT INTO events(task_id, level, message, created_at)
                VALUES (?, 'info', '任务草稿已创建，等待人工确认。', ?)
                """,
                (task_id, timestamp),
            )
            self._append_jsonl(
                {
                    "time": timestamp,
                    "task_id": task_id,
                    "app_id": app_id,
                    "operation": operation,
                    "status": "awaiting_confirmation",
                }
            )
            self._bump()
        return self.get(task_id)

    def get(self, task_id: str) -> dict[str, Any]:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM tasks WHERE id=?", (task_id,)
            ).fetchone()
        if row is None:
            raise KeyError(task_id)
        return self._row_to_dict(row)

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?",
                (max(1, min(limit, 200)),),
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def events(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM events ORDER BY id DESC LIMIT ?",
                (max(1, min(limit, 500)),),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def transition(
        self,
        task_id: str,
        expected: set[str],
        status: str,
        *,
        error: str | None = None,
        result: dict[str, Any] | None = None,
        message: str | None = None,
    ) -> dict[str, Any]:
        timestamp = now_iso()
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT status FROM tasks WHERE id=?", (task_id,)
            ).fetchone()
            if row is None:
                raise KeyError(task_id)
            if row["status"] not in expected:
                raise ValueError(
                    f"任务当前状态为 {row['status']}，不能切换到 {status}。"
                )
            time_column = {
                "queued": "confirmed_at",
                "running": "started_at",
                "succeeded": "finished_at",
                "failed": "finished_at",
                "cancelled": "finished_at",
            }.get(status)
            assignments = ["status=?", "updated_at=?", "error=?", "result_json=?"]
            values: list[Any] = [
                status,
                timestamp,
                error,
                json.dumps(result, ensure_ascii=False) if result is not None else None,
            ]
            if time_column:
                assignments.append(f"{time_column}=?")
                values.append(timestamp)
            values.append(task_id)
            connection.execute(
                f"UPDATE tasks SET {', '.join(assignments)} WHERE id=?", values
            )
            connection.execute(
                """
                INSERT INTO events(task_id, level, message, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    task_id,
                    "error" if status == "failed" else "info",
                    message or f"任务状态：{status}",
                    timestamp,
                ),
            )
            self._append_jsonl(
                {
                    "time": timestamp,
                    "task_id": task_id,
                    "status": status,
                    "error": error,
                    "result": result,
                }
            )
            self._bump()
        return self.get(task_id)


class RuleAgent:
    @staticmethod
    def _text_after_separator(text: str) -> str:
        for separator in ("：", ":"):
            if separator in text:
                return text.split(separator, 1)[1].strip()
        return ""

    def parse(self, raw_text: str) -> dict[str, Any]:
        text = " ".join(raw_text.strip().split())
        if not text:
            return {
                "understood": False,
                "message": "请输入要执行的任务。",
                "needs_clarification": True,
            }

        # Preserve the established single-action phrases. The task-creation
        # boundary converts the legacy file-transfer operation to the new
        # specified-chat workflow, so existing clients remain compatible.
        if "文件传输助手" in text and any(
            word in text for word in ("发送", "发给", "发", "输入")
        ):
            content = self._text_after_separator(text)
            if not content:
                for marker in ("发送", "发给", "输入"):
                    if marker in text:
                        content = (
                            text.split(marker, 1)[1]
                            .replace("文件传输助手", "", 1)
                            .strip(" ：:")
                        )
                        if content:
                            break
            if content:
                return self._draft(
                    "wechat",
                    "wechat.send_text",
                    {"chat_name": "文件传输助手", "text": content},
                    f"向微信文件传输助手发送：{content}",
                )

        if "评论" in text and not re.search(r"\d+\s*(?:个|条|次)", text):
            content = self._text_after_separator(text)
            if content and not any(token in text for token in ("搜索", "搜")):
                return self._draft(
                    "douyin",
                    "douyin.batch_interact",
                    {
                        "keyword": None,
                        "target_count": 1,
                        "like": False,
                        "comment": True,
                        "comment_text": content,
                    },
                    f"评论抖音当前视频：{content}",
                )

        if (
            "点赞" in text
            and not re.search(r"\d+\s*(?:个|条|次)", text)
            and not any(token in text for token in ("评论", "搜索", "搜"))
        ):
            return self._draft(
                "douyin",
                "douyin.batch_interact",
                {
                    "keyword": None,
                    "target_count": 1,
                    "like": True,
                    "comment": False,
                },
                "点赞抖音当前视频",
            )

        image_match = re.search(
            r"(?:给|向)(?P<chat>.+?)(?:发送|发)(?:相册)?(?:第)?(?P<index>\d{1,2})张(?:图片|照片)",
            text,
        )
        if image_match:
            return self._draft(
                "wechat",
                "wechat.send_album_image",
                {
                    "chat_name": image_match.group("chat").strip(" ：:"),
                    "image_index": int(image_match.group("index")),
                },
                f"向{image_match.group('chat')}发送相册第{image_match.group('index')}张图片",
            )

        send_match = re.search(
            r"(?:给|向)(?P<chat>.+?)(?:发送|发给|发消息|发)(?:文字)?[：:]?(?P<body>.+)$",
            text,
        )
        if send_match:
            chat_name = send_match.group("chat").strip(" ：:")
            content = send_match.group("body").strip(" ：:")
            if chat_name and content:
                return self._draft(
                    "wechat",
                    "wechat.send_text",
                    {"chat_name": chat_name, "text": content},
                    f"向微信聊天“{chat_name}”发送：{content}",
                )

        if "抖音" in text or any(word in text for word in ("视频", "点赞", "评论")):
            count_match = re.search(r"(\d{1,2})\s*(?:个|条|次)?(?:视频)?", text)
            target_count = int(count_match.group(1)) if count_match else 1
            keyword_match = re.search(r"(?:搜索|搜)(?P<keyword>.+?)(?:后|并|，|,|$)", text)
            keyword = keyword_match.group("keyword").strip(" ：:") if keyword_match else ""
            wants_like = "点赞" in text
            wants_comment = "评论" in text
            if ("搜索" in text or "搜" in text) and not wants_like and not wants_comment:
                if not keyword:
                    return {
                        "understood": False,
                        "message": "请补充抖音搜索关键词。",
                        "needs_clarification": True,
                    }
                return self._draft(
                    "douyin",
                    "douyin.search",
                    {"keyword": keyword},
                    f"在抖音搜索：{keyword}",
                )
            if wants_like or wants_comment:
                comment_text = self._text_after_separator(text) if wants_comment else ""
                if wants_comment and not comment_text:
                    return {
                        "understood": False,
                        "message": "请在冒号后补充评论内容。",
                        "needs_clarification": True,
                    }
                params: dict[str, Any] = {
                    "keyword": keyword or None,
                    "target_count": target_count,
                    "like": wants_like,
                    "comment": wants_comment,
                }
                if wants_comment:
                    params["comment_text"] = comment_text
                return self._draft(
                    "douyin",
                    "douyin.batch_interact",
                    params,
                    f"抖音批量任务：目标{target_count}个视频",
                )

        return {
            "understood": False,
            "message": "请明确聊天名称、文字/图片序号，或抖音搜索词、数量和互动类型。",
            "needs_clarification": True,
        }

    @staticmethod
    def _draft(
        app_id: str,
        operation: str,
        params: dict[str, Any],
        summary: str,
    ) -> dict[str, Any]:
        return {
            "understood": True,
            "app_id": app_id,
            "operation": operation,
            "params": params,
            "summary": summary,
            "needs_confirmation": True,
        }


class HybridAgent:
    """Map language to a structured state-graph task, never a free plan."""

    def __init__(self, provider: DeepSeekIntentProvider) -> None:
        self.provider = provider
        self.fallback = RuleAgent()

    def parse(self, raw_text: str) -> dict[str, Any]:
        if self.provider.configured:
            try:
                return self._parse_structured_with_deepseek(raw_text)
            except (IntentProviderError, VisionAgentError) as exc:
                return {
                    "understood": False,
                    "message": str(exc),
                    "needs_clarification": True,
                    "provider": "deepseek-v4-flash",
                }
        local = self.fallback.parse(raw_text)
        local["provider"] = "local_rule_fallback"
        local["fallback"] = True
        if not local["understood"]:
            local["message"] = (
                "DeepSeek 文本理解尚未配置；当前仅能用本地规则识别固定流程。"
                f"{local['message']}"
            )
        return local

    def _parse_structured_with_deepseek(self, raw_text: str) -> dict[str, Any]:
        if goal_is_forbidden(raw_text):
            raise VisionAgentError("目标包含本地禁用操作。")
        prompt = f"""
你只负责把用户自然语言映射为一个本地白名单任务，不生成执行步骤、动作或坐标。
只允许以下 operation 和参数：
1. wechat.send_text: {{"chat_name":字符串,"text":字符串}}
2. wechat.send_album_image: {{"chat_name":字符串,"image_index":1到20整数}}
3. douyin.search: {{"keyword":字符串}}
4. douyin.batch_interact: {{"keyword":字符串或null,"target_count":1到10整数,
   "like":布尔值,"comment":布尔值,"comment_text":评论时必填字符串}}

必须返回一个 JSON 对象：
{{"understood":true或false,"operation":字符串或null,"params":对象,
  "summary":字符串,"message":字符串}}
禁止输出 execution_plan、action、tap、swipe、coordinate 或任何系统命令。
信息不足时 understood=false，并在 message 中指出缺少什么，不能猜。

用户原文：{json.dumps(raw_text, ensure_ascii=False)}
"""
        raw = self.provider.chat_json(
            [{"role": "user", "content": prompt}],
            max_tokens=500,
        )
        payload = _extract_json_object(raw)
        forbidden_keys = {
            "execution_plan",
            "action",
            "tap",
            "swipe",
            "coordinate",
            "steps",
        }
        if forbidden_keys.intersection(payload):
            raise VisionAgentError("任务解析器返回了不允许的动作或计划字段。")
        if payload.get("understood") is not True:
            return {
                "understood": False,
                "message": str(payload.get("message") or "请补充任务参数。"),
                "needs_clarification": True,
                "provider": "deepseek-v4-flash_structured_intent",
            }
        operation = str(payload.get("operation") or "").strip()
        params = payload.get("params")
        if operation not in STRUCTURED_OPERATIONS or not isinstance(params, dict):
            raise VisionAgentError("任务解析结果不在页面状态图白名单中。")
        try:
            build_state_workflow_params(operation, params)
        except ValueError as exc:
            raise VisionAgentError(str(exc)) from exc
        return {
            "understood": True,
            "app_id": OPERATION_APP[operation],
            "operation": operation,
            "params": params,
            "summary": str(payload.get("summary") or raw_text).strip(),
            "needs_confirmation": True,
            "provider": "deepseek-v4-flash_structured_intent",
        }


class TaskRequest(BaseModel):
    app_id: str
    operation: str
    params: dict[str, Any] = Field(default_factory=dict)


class AgentRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)


class EnsureAppStepRequest(AgentRequest):
    confirmed: bool = False


class SupervisedStepRequest(BaseModel):
    confirmed: bool = False


class GenericSceneRequest(BaseModel):
    goal: dict[str, Any] = Field(default_factory=dict)


class StrictAgentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GenericSupervisedStartRequest(StrictAgentRequest):
    text: StrictStr = Field(min_length=1, max_length=500)
    device_id: StrictStr = Field(min_length=1, max_length=128)
    auto_advance: StrictBool = True


class GenericSupervisedDeviceRequest(StrictAgentRequest):
    device_id: StrictStr = Field(min_length=1, max_length=128)


class BaseActionConfirmationScopeRequest(StrictAgentRequest):
    session_id: StrictStr = Field(min_length=1, max_length=128)
    task_id: StrictStr = Field(min_length=1, max_length=128)
    device_id: StrictStr = Field(min_length=1, max_length=128)
    revision: StrictInt = Field(ge=1)
    subgoal_id: StrictStr = Field(min_length=1, max_length=128)
    effect_ids: list[StrictStr] = Field(default_factory=list)
    observation_id: StrictStr = Field(min_length=1, max_length=128)
    fingerprint: StrictStr = Field(min_length=1, max_length=256)


class GenericConfirmationScopeRequest(BaseActionConfirmationScopeRequest):
    decision_node_id: StrictStr = Field(min_length=1, max_length=128)
    action_digest: StrictStr = Field(min_length=64, max_length=64)


class GenericEffectConfirmationScopeRequest(StrictAgentRequest):
    session_id: StrictStr = Field(min_length=1, max_length=128)
    task_id: StrictStr = Field(min_length=1, max_length=128)
    device_id: StrictStr = Field(min_length=1, max_length=128)
    revision: StrictInt = Field(ge=1)
    subgoal_id: StrictStr = Field(min_length=1, max_length=128)
    effect_ids: list[StrictStr] = Field(min_length=1)
    intent_digest: StrictStr = Field(min_length=64, max_length=64)


class GenericEffectApprovalRequest(StrictAgentRequest):
    confirmed: StrictBool = False
    confirmation: GenericEffectConfirmationScopeRequest | None = None


class GenericSupervisedStepRequest(StrictAgentRequest):
    confirmed: StrictBool = False
    confirmation: GenericConfirmationScopeRequest | None = None


class GenericSupervisedAutoRequest(StrictAgentRequest):
    device_id: StrictStr = Field(min_length=1, max_length=128)
    confirmed: StrictBool = False
    confirmation: GenericConfirmationScopeRequest | None = None
    max_physical_actions: StrictInt = Field(default=12, ge=1, le=20)
    max_iterations: StrictInt = Field(default=24, ge=1, le=40)


class CapabilityAcceptanceStartRequest(StrictAgentRequest):
    device_id: StrictStr = Field(min_length=1, max_length=128)
    action: StrictStr = Field(min_length=1, max_length=64)
    text: StrictStr = Field(min_length=1, max_length=500)


class CapabilityActionConfirmationScopeRequest(BaseActionConfirmationScopeRequest):
    trial_id: StrictStr = Field(min_length=1, max_length=128)
    action: StrictStr = Field(min_length=1, max_length=64)


class CapabilityEffectConfirmationScopeRequest(GenericEffectConfirmationScopeRequest):
    trial_id: StrictStr = Field(min_length=1, max_length=128)
    action: StrictStr = Field(min_length=1, max_length=64)


class CapabilityActionConfirmationRequest(StrictAgentRequest):
    confirmed: StrictBool = False
    confirmation: CapabilityActionConfirmationScopeRequest | None = None


class CapabilityEffectApprovalRequest(StrictAgentRequest):
    confirmed: StrictBool = False
    confirmation: CapabilityEffectConfirmationScopeRequest | None = None


class CapabilityPromotionRequest(StrictAgentRequest):
    confirmed: StrictBool = False
    trial_id: StrictStr = Field(min_length=1, max_length=128)
    device_id: StrictStr = Field(min_length=1, max_length=128)
    action: StrictStr = Field(min_length=1, max_length=64)
    report_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    registry_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")


class CapabilityCancelRequest(StrictAgentRequest):
    device_id: StrictStr = Field(min_length=1, max_length=128)
    action: StrictStr = Field(min_length=1, max_length=64)


def build_generic_plan_preview(
    parsed_intent: dict[str, Any],
    orchestrator: GenericTaskOrchestrator,
) -> dict[str, Any]:
    """Convert a parsed intent into an inspectable plan without hardware access."""
    if parsed_intent.get("understood") is not True:
        return {
            "compiled": False,
            "execution_enabled": False,
            "intent": parsed_intent,
            "goal": None,
            "plan": None,
        }
    operation = str(parsed_intent.get("operation") or "").strip()
    params = parsed_intent.get("params")
    if not isinstance(params, dict):
        raise TaskPlanError("任务解析结果缺少结构化参数。")
    goal = GoalSpec.from_operation(
        operation,
        params,
        objective=str(parsed_intent.get("summary") or "").strip() or None,
    )
    plan = orchestrator.compile_goal(goal)
    return {
        "compiled": True,
        "execution_enabled": False,
        "intent": parsed_intent,
        "goal": goal.to_dict(),
        "plan": plan.to_dict(),
    }


class DeviceControllerRegistry:
    """Resolve one controller and calibration per device_id."""

    def __init__(self, path: Path, *, mock: bool = False) -> None:
        self.path = Path(path)
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"设备注册表无法读取：{exc}") from exc
        if payload.get("version") != 1 or not isinstance(payload.get("devices"), list):
            raise RuntimeError("设备注册表版本或 devices 格式无效。")
        self.default_device_id = str(payload.get("default_device_id") or "").strip()
        self._controllers: dict[str, RobotController] = {}
        self._descriptors: dict[str, dict[str, Any]] = {}
        enabled_windows: set[str] = set()
        for raw in payload["devices"]:
            if not isinstance(raw, dict) or raw.get("enabled") is not True:
                continue
            device_id = str(raw.get("device_id") or "").strip()
            window_title = str(raw.get("window_title") or "").strip()
            calibration_value = str(raw.get("calibration_path") or "").strip()
            raw_verified_actions = raw.get("verified_actions")
            if raw_verified_actions is None:
                verified_actions = None
            elif not isinstance(raw_verified_actions, list) or not all(
                isinstance(item, str) and item.strip()
                for item in raw_verified_actions
            ):
                raise RuntimeError(
                    f"设备 {device_id or 'missing'} 的 verified_actions 格式无效。"
                )
            else:
                verified_actions = {
                    str(item).strip() for item in raw_verified_actions
                }
            if not device_id or device_id in self._controllers:
                raise RuntimeError("设备注册表存在空或重复的 device_id。")
            effective_window = window_title or "__default_window__"
            if effective_window in enabled_windows:
                raise RuntimeError("两台已启用设备不能绑定同一个机械臂控制窗口。")
            enabled_windows.add(effective_window)
            calibration_path = Path(calibration_value or "tap_calibration.json")
            if not calibration_path.is_absolute():
                calibration_path = self.path.parent / calibration_path
            controller: RobotController
            if mock:
                controller = MockRobotController(device_id=device_id)
            elif window_title:
                controller = RobotController(
                    window_title,
                    calibration_path=calibration_path,
                    verified_actions=verified_actions,
                    device_id=device_id,
                )
            else:
                controller = RobotController(
                    calibration_path=calibration_path,
                    verified_actions=verified_actions,
                    device_id=device_id,
                )
            self._controllers[device_id] = controller
            self._descriptors[device_id] = {
                "device_id": device_id,
                "window_title": window_title,
                "calibration_path": str(calibration_path),
                "verified_actions": sorted(controller.verified_actions),
            }
        if not self._controllers or self.default_device_id not in self._controllers:
            raise RuntimeError("设备注册表必须包含已启用的 default_device_id。")

    def controller(self, device_id: str) -> RobotController:
        resolved = str(device_id or "").strip()
        controller = self._controllers.get(resolved)
        if controller is None:
            raise UniversalAgentOrchestratorError(
                f"device_id 未登记或未启用：{resolved or 'missing'}。"
            )
        return controller

    def provisional_controller(
        self,
        device_id: str,
        candidate_action: str,
    ) -> RobotController:
        """Create an unregistered controller for one evidence-bound trial.

        The returned object is deliberately not stored in this registry.  It
        cannot change the capabilities of the product controller that owns the
        normal web path.
        """

        resolved_device = str(device_id or "").strip()
        action = str(candidate_action or "").strip()
        if action not in PROMOTABLE_ACTIONS:
            raise CapabilityAcceptanceError(
                f"动作 {action or 'missing'} 不能进入真机能力验收。"
            )
        try:
            original = self._controllers[resolved_device]
            descriptor = self._descriptors[resolved_device]
        except KeyError as exc:
            raise CapabilityAcceptanceError(
                f"device_id 未登记或未启用：{resolved_device or 'missing'}。"
            ) from exc
        if action in original.verified_actions:
            raise CapabilityAcceptanceError(
                f"设备能力 {action} 已经通过真机验收。"
            )
        verified_actions = set(original.verified_actions) | {action}
        if isinstance(original, MockRobotController):
            return MockRobotController(
                verified_actions=verified_actions,
                device_id=resolved_device,
            )
        return RobotController(
            descriptor["window_title"] or original.title,
            calibration_path=Path(descriptor["calibration_path"]),
            verified_actions=verified_actions,
            device_id=resolved_device,
        )

    def descriptors(self) -> list[dict[str, Any]]:
        return [dict(self._descriptors[key]) for key in sorted(self._descriptors)]


class Runtime:
    def __init__(self) -> None:
        self.loaded_code_revision = current_code_revision()
        self.device_controllers = DeviceControllerRegistry(
            DEVICE_REGISTRY_PATH,
            mock=os.environ.get("ROBOT_WEB_MOCK") == "1",
        )
        self.controller: RobotController = self.device_controllers.controller(
            self.device_controllers.default_device_id
        )
        self.vision_provider = DashScopeVisionProvider()
        self.intent_provider = DeepSeekIntentProvider()
        self.input_lineage_store = TypedInputLineageStore(
            WEB_OUTPUT_DIR / "state"
        )
        self.generic_scene_observer = GenericSceneObserver(
            self.vision_provider,
            input_lineage_store=self.input_lineage_store,
        )
        self.generic_step_planner = GenericStepPlanner(self.intent_provider)
        self.deepseek_task_graph_planner = DeepSeekTaskGraphPlanner(
            self.intent_provider,
        )
        self.qwen_visual_decision_observer = QwenVisualDecisionObserver(
            self.vision_provider
        )
        self.device_task_registry = DeviceTaskRegistry(
            lease_directory=SHARED_DEVICE_LEASE_DIR
        )
        self.universal_agent_orchestrator = UniversalAgentOrchestrator(
            deepseek_planner=self.deepseek_task_graph_planner,
            qwen_observer=self.qwen_visual_decision_observer,
            adapter_factory=lambda device_id: GenericSingleActionAdapter(
                capture=self.controller_for_device(device_id).vision_capture,
                observer=self.generic_scene_observer,
                robot=self.controller_for_device(device_id),
                controller=UniversalActionController(),
                qwerty_row_snapper=stable_qwerty_ocr_anchors,
                require_local_qwerty_row_snap=not isinstance(
                    self.controller_for_device(device_id),
                    MockRobotController,
                ),
                device_id=device_id,
                input_lineage_store=self.input_lineage_store,
            ),
            device_registry=self.device_task_registry,
        )
        self.capability_acceptance_manager = CapabilityAcceptanceManager(
            provisional_controller_factory=(
                self.device_controllers.provisional_controller
            ),
            orchestrator_factory=self.capability_trial_orchestrator,
            device_registry=self.device_task_registry,
            output_dir=WEB_OUTPUT_DIR,
            registry_path=DEVICE_REGISTRY_PATH,
            code_revision_provider=self.capability_code_revision,
        )
        self.device_coordination_lock_guard = threading.RLock()
        self.device_coordination_locks: dict[str, threading.Lock] = {
            self.device_controllers.default_device_id: threading.Lock(),
        }
        self.generic_supervised_sessions: dict[str, UniversalAgentSessionState] = {}
        self.generic_supervised_session_lock = threading.RLock()

    def controller_for_device(self, device_id: str) -> RobotController:
        if str(device_id or "").strip() == self.device_controllers.default_device_id:
            return self.controller
        if isinstance(self.controller, MockRobotController):
            # Test and explicit mock mode accepts logical device IDs while the
            # production registry remains strict.
            return self.controller
        return self.device_controllers.controller(device_id)

    def capability_code_revision(self) -> str:
        current = current_code_revision()
        if current != self.loaded_code_revision:
            raise CapabilityAcceptanceError(
                "服务启动后代码状态发生变化，必须安全重启后才能进行真机验收。"
            )
        return self.loaded_code_revision

    def capability_trial_orchestrator(
        self,
        provisional_controller: RobotController,
        candidate_action: str,
    ) -> UniversalAgentOrchestrator:
        """Build one isolated primitive-certification orchestrator."""

        return UniversalAgentOrchestrator(
            deepseek_planner=CapabilityAcceptanceTaskGraphPlanner(
                candidate_action
            ),
            qwen_observer=self.qwen_visual_decision_observer,
            adapter_factory=lambda device_id: GenericSingleActionAdapter(
                capture=provisional_controller.vision_capture,
                observer=self.generic_scene_observer,
                robot=provisional_controller,
                controller=UniversalActionController(),
                qwerty_row_snapper=stable_qwerty_ocr_anchors,
                require_local_qwerty_row_snap=not isinstance(
                    provisional_controller,
                    MockRobotController,
                ),
                device_id=device_id,
                input_lineage_store=self.input_lineage_store,
            ),
            device_registry=self.device_task_registry,
        )

    def coordination_lock_for_device(self, device_id: str) -> threading.Lock:
        """Serialize observation/action work per device, not across devices."""

        resolved = str(device_id or "").strip()
        if not resolved:
            raise UniversalAgentOrchestratorError("device_id 不能为空。")
        with self.device_coordination_lock_guard:
            return self.device_coordination_locks.setdefault(
                resolved,
                threading.Lock(),
            )

    def start(self) -> None:
        return None

    def shutdown(self) -> None:
        self.controller.request_stop()

    def _worker_loop(self) -> None:
        while not self.stop.is_set():
            try:
                task_id = self.jobs.get(timeout=0.5)
            except queue.Empty:
                continue
            if not task_id:
                continue
            try:
                task = self.store.get(task_id)
                if task["status"] != "queued":
                    continue
                self.store.transition(
                    task_id,
                    {"queued"},
                    "running",
                    message="机械臂任务开始执行。",
                )
                self.controller.clear_stop()
                if task["operation"] not in STRUCTURED_OPERATIONS:
                    raise VisionAgentError(
                        "任务不属于页面状态图白名单，唯一控制器拒绝执行。"
                    )
                if self.orchestrator_mode == "generic":
                    # Phase one exposes a safe, inspectable plan compiler only.
                    # It intentionally cannot reach RobotController until the
                    # user approves the single-step executor in phase two.
                    self.generic_orchestrator.compile(
                        task["operation"],
                        task["params"].get("source_params", task["params"]),
                    )
                    raise VisionAgentError(
                        "通用编排器当前仅完成计划编译与校验，实机执行尚未启用。"
                    )
                device_id = self.device_controllers.default_device_id
                device_session_id = f"legacy-worker-{os.getpid()}-{task_id}"
                self.device_task_registry.reserve(device_id, device_session_id)
                process_lease = InterProcessLease(
                    SHARED_DEVICE_LEASE_DIR / "physical_hardware_action.lease",
                    owner_id=f"worker-{os.getpid()}-{task_id}",
                    metadata={"purpose": "legacy_worker_task", "task_id": task_id},
                )
                if not process_lease.acquire():
                    self.device_task_registry.release(device_id, device_session_id)
                    raise VisionAgentError("另一进程已占用机械臂物理控制权。")
                try:
                    if isinstance(self.controller, MockRobotController):
                        result = self.controller.execute(
                            task["operation"],
                            task["params"].get("source_params", task["params"]),
                        )
                    else:
                        result = self.state_runner.execute(
                            task["operation"],
                            task["params"],
                        )
                finally:
                    try:
                        process_lease.release()
                    finally:
                        self.device_task_registry.release(
                            device_id,
                            device_session_id,
                        )
                self.store.transition(
                    task_id,
                    {"running"},
                    "succeeded",
                    result=result,
                    message="任务执行成功。",
                )
            except (RobotWorkflowError, VisionAgentError) as exc:
                self._safe_fail(
                    task_id,
                    str(exc),
                    report_path=getattr(exc, "report_path", None),
                )
            except Exception as exc:  # Fail closed for every unknown hardware error.
                self._safe_fail(task_id, f"未预期错误：{type(exc).__name__}: {exc}")
            finally:
                self.jobs.task_done()

    def _safe_fail(
        self,
        task_id: str,
        message: str,
        *,
        report_path: str | None = None,
    ) -> None:
        try:
            task = self.store.get(task_id)
            if task["status"] == "running":
                result: dict[str, Any] = {
                    "page_classification": "unknown_or_changed",
                    "evidence": [],
                }
                if report_path:
                    result["report"] = report_path
                try:
                    failure_dir = WEB_OUTPUT_DIR / "failures"
                    failure_dir.mkdir(parents=True, exist_ok=True)
                    evidence_path = failure_dir / f"{task_id}_{int(time.time())}.jpg"
                    evidence_path.write_bytes(
                        self.controller.capture_preview(quality=86)
                    )
                    result["evidence"].append(str(evidence_path))
                except Exception:
                    pass
                if self.controller.stop_event.is_set():
                    self.store.transition(
                        task_id,
                        {"running"},
                        "cancelled",
                        error=message,
                        result=result,
                        message="任务已按停止请求终止。",
                    )
                else:
                    self.store.transition(
                        task_id,
                        {"running"},
                        "failed",
                        error=message,
                        result=result,
                        message=message,
                    )
        except (KeyError, ValueError):
            return


runtime = Runtime()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> Iterator[None]:
    runtime.start()
    print(f"机械臂网页控制台：http://127.0.0.1:8765/")
    print("控制令牌已生成，仅通过本机受保护的页面初始化接口使用。")
    if os.environ.get("ROBOT_WEB_NO_BROWSER") != "1":
        threading.Timer(
            1.0, lambda: webbrowser.open("http://127.0.0.1:8765/")
        ).start()
    yield
    runtime.shutdown()


app = FastAPI(
    title="多 App 机械臂网页控制平台",
    version="0.2.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
)
app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")


def verify_local_request(request: Request, token: str | None) -> None:
    if token != CONTROL_TOKEN:
        raise HTTPException(status_code=403, detail="控制令牌无效。")
    origin = request.headers.get("origin")
    if origin:
        try:
            origin_host = origin.split("://", 1)[1].split("/", 1)[0].split(":", 1)[0]
        except IndexError:
            raise HTTPException(status_code=403, detail="来源地址无效。")
        if origin_host not in {"127.0.0.1", "localhost"}:
            raise HTTPException(status_code=403, detail="只允许本机网页请求。")


def normalize_task_request(task: TaskRequest) -> TaskRequest:
    """Validate structured targets for the page-state controller."""
    operation = task.operation
    params = dict(task.params)
    if operation == "wechat.send_text_to_file_transfer":
        operation = "wechat.send_text"
        params = {
            "chat_name": "文件传输助手",
            "text": params.get("text"),
        }
    elif operation == "douyin.like_current":
        operation = "douyin.batch_interact"
        params = {
            "keyword": None,
            "target_count": 1,
            "like": True,
            "comment": False,
        }
    elif operation == "douyin.comment_current":
        operation = "douyin.batch_interact"
        params = {
            "keyword": None,
            "target_count": 1,
            "like": False,
            "comment": True,
            "comment_text": params.get("text"),
        }
    if operation in STRUCTURED_OPERATIONS:
        try:
            workflow = build_state_workflow_params(operation, params)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return TaskRequest(
            app_id=OPERATION_APP[operation],
            operation=operation,
            params=workflow,
        )
    return task


def validate_task_payload(task: TaskRequest) -> None:
    if task.operation == "agent.execute_goal":
        raise HTTPException(
            status_code=400,
            detail="旧的自由 Agent 执行入口已停用；请使用页面状态图白名单任务。",
        )
    expected_app = OPERATION_APP.get(task.operation)
    if expected_app is None:
        raise HTTPException(status_code=400, detail="操作不在白名单中。")
    if task.app_id != expected_app:
        raise HTTPException(status_code=400, detail="App 与操作不匹配。")
    if task.operation in {
        "wechat.send_text_to_file_transfer",
        "douyin.comment_current",
    }:
        value = task.params.get("text")
        if not isinstance(value, str) or not value.strip():
            raise HTTPException(status_code=400, detail="缺少文字内容。")
        if len(value.strip()) > 100:
            raise HTTPException(status_code=400, detail="文字不能超过100个字符。")
        if "\n" in value or "\r" in value:
            raise HTTPException(status_code=400, detail="第一版不支持换行。")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/session")
def session() -> dict[str, Any]:
    return {
        "token": CONTROL_TOKEN,
        "mock": isinstance(runtime.controller, MockRobotController),
        "version": app.version,
    }


@app.get("/api/apps")
def apps() -> dict[str, Any]:
    readiness = {"vision_agent": {
        "ready": bool(runtime.vision_provider.status().get("configured")),
        "mode": runtime.vision_provider.status().get("model", "unknown"),
        "thinking_enabled": runtime.vision_provider.status().get(
            "thinking_enabled", False
        ),
        "missing_templates": [],
        "missing_capabilities": (
            []
            if runtime.vision_provider.status().get("configured")
            else ["DASHSCOPE_API_KEY"]
        ),
    }}
    readiness["intent_agent"] = {
        "ready": bool(runtime.intent_provider.configured),
        "mode": runtime.intent_provider.model,
        "missing_templates": [],
        "missing_capabilities": (
            [] if runtime.intent_provider.configured else ["DEEPSEEK_API_KEY"]
        ),
    }
    return {"apps": APP_CATALOG, "readiness": readiness}


@app.get("/api/device")
def device() -> dict[str, Any]:
    status = dict(runtime.controller.device_status())
    status.pop("readiness", None)
    capability_provider = getattr(runtime.controller, "hardware_capabilities", None)
    hardware_capabilities = (
        capability_provider() if callable(capability_provider) else {}
    )
    capability_profile_provider = getattr(
        runtime.controller,
        "hardware_capability_profile",
        None,
    )
    hardware_capability_profile = (
        capability_profile_provider()
        if callable(capability_profile_provider)
        else None
    )
    profile_actions = (
        hardware_capability_profile.get("actions", {})
        if isinstance(hardware_capability_profile, dict)
        else {}
    )
    effective_hardware_capabilities = {
        str(action): bool(spec.get("enabled"))
        for action, spec in profile_actions.items()
        if isinstance(action, str) and isinstance(spec, dict)
    }
    status["default_device_id"] = runtime.device_controllers.default_device_id
    devices = []
    for descriptor in runtime.device_controllers.descriptors():
        public_status = dict(
            runtime.controller_for_device(descriptor["device_id"]).device_status()
        )
        public_status.pop("readiness", None)
        devices.append({**descriptor, **public_status})
    status["devices"] = devices
    status["vision_agent"] = runtime.vision_provider.status()
    status["intent_agent"] = runtime.intent_provider.status()
    status["execution_architecture"] = {
        "model_role": "observation_only",
        "controller": "universal_action_controller",
        "fixed_app_workflows_retired": True,
        "active_orchestrator": "universal_agent",
        "universal_agent": {
            "goal_protocol": "2026-08-20-deepseek-typed-task-graph-v4",
            "scene_protocol": UI_SCENE_PROTOCOL_VERSION,
            "action_protocol": CANONICAL_ACTION_PROTOCOL,
            "controller_protocol": UNIVERSAL_CONTROLLER_PROTOCOL_VERSION,
            "goal_preview_enabled": False,
            "scene_preview_enabled": True,
            "hardware_execution_enabled": True,
            "automatic_loop_enabled": True,
            "automatic_loop_max_physical_actions": 12,
            "automatic_loop_max_iterations": 24,
            "supervised_single_step_enabled": False,
            "enabled_physical_actions": sorted(
                action
                for action, enabled in (
                    effective_hardware_capabilities or hardware_capabilities
                ).items()
                if enabled and action != "wait_for_change"
            ),
            "protocol_physical_actions": [
                "tap_semantic",
                "dismiss_overlay",
                "swipe",
                "back",
                "home",
                "reveal_system_navigation",
                "input_verified_text",
                "press_enter",
                "clear_verified_text",
                "long_press",
                "drag",
            ],
            "hardware_capabilities": hardware_capabilities,
            "hardware_capability_profile": hardware_capability_profile,
            "supported_app_scope": "dynamic",
            "typed_effect_authority": {
                "semantic_ir_protocol": TASK_SEMANTIC_IR_PROTOCOL,
                "authority_protocol": AUTHORITY_REPORT_PROTOCOL,
                "effect_policy_protocol": RISK_POLICY_PROTOCOL,
                "authority_scope": "typed_task_and_effect_policy",
                "retired_remote_risk_diagnostics_enabled": False,
                "canonical_action_protocol": CANONICAL_ACTION_PROTOCOL,
            },
            "observer": runtime.generic_scene_observer.status(),
        },
    }
    status["active_tasks"] = []
    with runtime.generic_supervised_session_lock:
        generic_sessions = [
            item.snapshot()
            for item in runtime.generic_supervised_sessions.values()
            if item.status in {
                "awaiting_effect_confirmation",
                "awaiting_confirmation",
                "paused_after_action",
                "needs_reobservation",
                "needs_effect_verification",
            }
        ]
    status["generic_supervised_execution"] = {
        "enabled": True,
        "automatic_loop_enabled": True,
        "max_physical_actions_per_confirmation": 1,
        "max_safe_loop_physical_actions": 12,
        "max_safe_loop_iterations": 24,
        "external_effect_confirmation_count": 1,
        "post_action_transition_protocol": (
            POST_ACTION_TRANSITION_PROTOCOL_VERSION
        ),
        "active_sessions": [
            {
                "session_id": item["session_id"],
                "device_id": item["device_id"],
                "status": item["status"],
                "step_number": item["step_number"],
                "proposal": item["proposal"],
            }
            for item in generic_sessions
        ],
    }
    return status


@app.get("/api/tasks")
def tasks(limit: int = 50) -> dict[str, Any]:
    return {"tasks": runtime.store.list(limit)}


@app.get("/api/tasks/{task_id}")
def task(task_id: str) -> dict[str, Any]:
    try:
        return runtime.store.get(task_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="任务不存在。")


@app.post("/api/agent/parse")
def parse_agent(body: AgentRequest) -> dict[str, Any]:
    require_legacy_workflows_enabled()
    return runtime.agent.parse(body.text)


@app.post("/api/agent/generic-goal")
def parse_generic_agent_goal(body: AgentRequest) -> dict[str, Any]:
    """Parse any App goal without creating a task or touching hardware."""
    try:
        draft = runtime.generic_intent_parser.parse(body.text)
    except (IntentProviderError, GenericIntentError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    result: dict[str, Any] = {
        "mode": "generic_goal_preview",
        "executed": False,
        "task_created": False,
        "draft": draft.to_dict(),
    }
    if draft.understood:
        result["goal"] = draft.to_goal_spec().to_dict()
    else:
        result["goal"] = None
    return result


@app.post("/api/agent/plan-preview")
def preview_agent_plan(body: AgentRequest) -> dict[str, Any]:
    parsed = runtime.agent.parse(body.text)
    try:
        return build_generic_plan_preview(parsed, runtime.generic_orchestrator)
    except TaskPlanError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _require_supervised_device_ready(device_id: str | None = None) -> None:
    controller = runtime.controller_for_device(
        device_id or runtime.device_controllers.default_device_id
    )
    status = controller.device_status()
    if not status.get("controller_online") or not status.get("camera_online"):
        raise HTTPException(status_code=409, detail="控制端或摄像头离线。")
    if status.get("busy"):
        raise HTTPException(status_code=409, detail="机械臂正在执行其他任务。")
    if not runtime.vision_provider.status().get("configured"):
        raise HTTPException(status_code=409, detail="千问视觉尚未配置。")


@contextmanager
def _supervised_hardware_lock(device_id: str | None = None) -> Iterator[None]:
    resolved_device = str(
        device_id or runtime.device_controllers.default_device_id
    ).strip()
    controller = runtime.controller_for_device(resolved_device)
    lease_path = (
        SHARED_DEVICE_LEASE_DIR / "physical_hardware_action.lease"
        if resolved_device == runtime.device_controllers.default_device_id
        else SHARED_DEVICE_LEASE_DIR
        / f"physical_hardware_action_{re.sub(r'[^A-Za-z0-9_.-]+', '_', resolved_device)}.lease"
    )
    process_lease = InterProcessLease(
        lease_path,
        owner_id=f"web-{os.getpid()}-{threading.get_ident()}",
        metadata={"purpose": "physical_hardware_action", "device_id": resolved_device},
    )
    if not process_lease.acquire():
        raise HTTPException(status_code=409, detail="另一进程已占用机械臂物理控制权。")
    coordination_lock = runtime.coordination_lock_for_device(resolved_device)
    if not coordination_lock.acquire(blocking=False):
        process_lease.release()
        raise HTTPException(status_code=409, detail="已有语义观察或动作正在进行。")
    if not controller.operation_lock.acquire(blocking=False):
        coordination_lock.release()
        process_lease.release()
        raise HTTPException(status_code=409, detail="机械臂物理控制权已被占用。")
    try:
        yield
    finally:
        controller.operation_lock.release()
        coordination_lock.release()
        process_lease.release()


def _acquire_compatibility_hardware_lease(purpose: str) -> Any:
    device_id = runtime.device_controllers.default_device_id
    session_id = (
        f"compat-{purpose}-{os.getpid()}-{threading.get_ident()}-"
        f"{uuid.uuid4().hex[:8]}"
    )
    try:
        runtime.device_task_registry.reserve(device_id, session_id)
    except UniversalAgentOrchestratorError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    lease = InterProcessLease(
        SHARED_DEVICE_LEASE_DIR / "physical_hardware_action.lease",
        owner_id=f"compat-{os.getpid()}-{threading.get_ident()}-{uuid.uuid4().hex[:8]}",
        metadata={"purpose": str(purpose)},
    )
    if not lease.acquire():
        runtime.device_task_registry.release(device_id, session_id)
        raise HTTPException(status_code=409, detail="另一进程已占用机械臂物理控制权。")

    class CombinedCompatibilityLease:
        def __init__(self) -> None:
            self.released = False

        def release(self) -> None:
            if self.released:
                return
            self.released = True
            try:
                lease.release()
            finally:
                runtime.device_task_registry.release(device_id, session_id)

    return CombinedCompatibilityLease()


def _require_generic_session_device(
    session: UniversalAgentSessionState,
    requested_device_id: str,
) -> None:
    if str(requested_device_id or "") != session.device_id:
        raise HTTPException(
            status_code=409,
            detail={
                "success": False,
                "physical_actions": 0,
                "error": "请求device_id与会话锁定设备不一致。",
            },
        )


@app.post("/api/agent/generic-scene")
def observe_generic_scene(
    body: GenericSceneRequest,
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    """Capture a stable scene graph without moving or clicking the robot."""

    verify_local_request(request, x_control_token)
    _require_supervised_device_ready()
    frames = []
    with _supervised_hardware_lock():
        for index in range(4):
            frame = runtime.controller.vision_capture()
            if frame.width < 400 or frame.height < 700:
                raise HTTPException(
                    status_code=409,
                    detail="摄像头返回残缺画面，未调用通用观察器。",
                )
            frames.append(frame)
            if index < 3:
                time.sleep(0.5)
        try:
            scene = runtime.generic_scene_observer.observe(
                frames=frames,
                goal_context=body.goal,
                device_id=runtime.device_controllers.default_device_id,
            )
        except VisionAgentError as exc:
            failure_dir = WEB_OUTPUT_DIR / (
                "generic_scene_failure_"
                + datetime.now().strftime("%Y%m%d_%H%M%S_")
                + uuid.uuid4().hex[:8]
            )
            try:
                persist_observer_failure_diagnostic(
                    runtime.generic_scene_observer,
                    evidence_dir=failure_dir,
                    prefix="generic_scene_preview",
                    error=exc,
                )
            except Exception:
                pass
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "mode": "generic_scene_preview",
        "executed": False,
        "physical_action_requested": False,
        "scene": scene.to_dict(),
        "diagnostics": dict(runtime.generic_scene_observer.last_diagnostics),
    }


def _new_generic_action_adapter() -> GenericSingleActionAdapter:
    return GenericSingleActionAdapter(
        capture=runtime.controller.vision_capture,
        observer=runtime.generic_scene_observer,
        robot=runtime.controller,
        controller=UniversalActionController(),
        qwerty_row_snapper=stable_qwerty_ocr_anchors,
        require_local_qwerty_row_snap=not isinstance(
            runtime.controller,
            MockRobotController,
        ),
        device_id=runtime.device_controllers.default_device_id,
        input_lineage_store=runtime.input_lineage_store,
    )


def _write_generic_supervised_report(session: UniversalAgentSessionState) -> str:
    """Return the atomic report already maintained by the orchestrator."""

    return str(session.run_dir / "report.json")


def _generic_supervised_failure(
    session: UniversalAgentSessionState | None,
    exc: Exception,
    *,
    request_action_count: int = 0,
) -> dict[str, Any]:
    physical_actions = max(0, int(request_action_count))
    if isinstance(exc, GenericActionAdapterError):
        physical_actions = max(physical_actions, int(exc.physical_actions))
    return {
        "success": False,
        "physical_actions": physical_actions,
        "error": str(exc),
        "evidence": (
            list(session.snapshot().get("evidence", [])) if session else []
        ),
        "session": session.snapshot() if session else None,
        "report": _write_generic_supervised_report(session) if session else None,
    }


def _capability_trial_payload(trial: Any) -> dict[str, Any]:
    snapshot = trial.snapshot()
    session = snapshot.get("session")
    if not isinstance(session, dict):
        session = {}
    action_scope = session.get("confirmation_scope")
    effect_scope = session.get("effect_confirmation_scope")
    snapshot["action_confirmation_scope"] = (
        {
            **dict(action_scope),
            "trial_id": trial.trial_id,
            "action": trial.candidate_action,
        }
        if isinstance(action_scope, dict)
        else None
    )
    snapshot["effect_confirmation_scope"] = (
        {
            **dict(effect_scope),
            "trial_id": trial.trial_id,
            "action": trial.candidate_action,
        }
        if isinstance(effect_scope, dict)
        else None
    )
    return snapshot


def _capability_execution_payload(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return dict(result)
    method = getattr(result, "to_dict", None)
    if callable(method):
        payload = method()
        if isinstance(payload, dict):
            return payload
    raise CapabilityAcceptanceError("验收执行结果不是 JSON 对象。")


def _capability_failure(
    trial: Any | None,
    exc: Exception,
    *,
    request_action_count: int = 0,
) -> dict[str, Any]:
    physical_actions = max(
        0,
        int(request_action_count),
        int(getattr(exc, "physical_actions", 0) or 0),
    )
    return {
        "success": False,
        "physical_actions": physical_actions,
        "error": str(exc),
        "trial": _capability_trial_payload(trial) if trial is not None else None,
        "evidence": [str(path) for path in getattr(exc, "evidence", ())],
    }


def _require_capability_trial_binding(
    trial: Any,
    *,
    trial_id: str,
    device_id: str,
    action: str,
) -> None:
    if (
        trial.trial_id != trial_id
        or trial.device_id != device_id
        or trial.candidate_action != action
    ):
        raise CapabilityAcceptanceError("真机验收确认范围与当前会话不匹配。")


CAPABILITY_ACCEPTANCE_ERRORS = (
    CapabilityAcceptanceError,
    GenericActionAdapterError,
    IntentProviderError,
    TaskGraphError,
    UniversalActionError,
    UniversalAgentOrchestratorError,
    VisionAgentError,
)


@app.post("/api/capability-acceptance/start")
def start_capability_acceptance(
    body: CapabilityAcceptanceStartRequest,
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    """Plan and observe one unverified generic action with zero execution."""

    verify_local_request(request, x_control_token)
    if body.action not in PROMOTABLE_ACTIONS:
        exc = CapabilityAcceptanceError(
            f"动作 {body.action} 不能进入真机能力验收。"
        )
        raise HTTPException(
            status_code=409,
            detail=_capability_failure(None, exc),
        )
    active_session = runtime.device_task_registry.active_session(body.device_id)
    if active_session is not None:
        exc = CapabilityAcceptanceError(
            f"设备 {body.device_id} 已有活动任务：{active_session}。"
        )
        raise HTTPException(
            status_code=409,
            detail=_capability_failure(None, exc),
        )
    _require_supervised_device_ready(body.device_id)
    try:
        with _supervised_hardware_lock(body.device_id):
            trial = runtime.capability_acceptance_manager.start(
                device_id=body.device_id,
                candidate_action=body.action,
                text=body.text,
            )
        return {
            "mode": "capability_acceptance_single_action",
            "physical_actions": 0,
            "automatic_loop_enabled": False,
            "trial": _capability_trial_payload(trial),
        }
    except CAPABILITY_ACCEPTANCE_ERRORS as exc:
        raise HTTPException(
            status_code=409,
            detail=_capability_failure(None, exc),
        ) from exc


@app.get("/api/capability-acceptance")
def list_capability_acceptance(
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    verify_local_request(request, x_control_token)
    return {"trials": runtime.capability_acceptance_manager.snapshots()}


@app.get("/api/capability-acceptance/{trial_id}")
def get_capability_acceptance(
    trial_id: str,
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    verify_local_request(request, x_control_token)
    try:
        trial = runtime.capability_acceptance_manager.get(trial_id)
    except CapabilityAcceptanceError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "mode": "capability_acceptance_single_action",
        "trial": _capability_trial_payload(trial),
    }


@app.get("/api/capability-acceptance/{trial_id}/evidence/{phase}/{index}")
def get_capability_acceptance_evidence(
    trial_id: str,
    phase: Literal["before", "after"],
    index: int,
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> FileResponse:
    verify_local_request(request, x_control_token)
    try:
        trial = runtime.capability_acceptance_manager.get(trial_id)
        report = json.loads(trial.report_path.read_text(encoding="utf-8"))
        paths = report.get(f"{phase}_frame_paths")
        if (
            not isinstance(paths, list)
            or isinstance(index, bool)
            or index < 0
            or index >= len(paths)
            or not isinstance(paths[index], str)
        ):
            raise CapabilityAcceptanceError("验收证据索引不存在。")
        evidence_path = Path(paths[index]).resolve(strict=True)
        trial_root = trial.run_dir.resolve(strict=True)
        if (
            evidence_path.suffix.lower() not in {".jpg", ".jpeg"}
            or evidence_path == trial_root
            or trial_root not in evidence_path.parents
        ):
            raise CapabilityAcceptanceError("验收证据路径越出当前 trial。")
    except (
        CapabilityAcceptanceError,
        OSError,
        UnicodeError,
        ValueError,
        TypeError,
        json.JSONDecodeError,
    ) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(evidence_path, media_type="image/jpeg")


@app.post("/api/capability-acceptance/{trial_id}/approve-effect")
def approve_capability_acceptance_effect(
    trial_id: str,
    body: CapabilityEffectApprovalRequest,
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    verify_local_request(request, x_control_token)
    trial = None
    try:
        trial = runtime.capability_acceptance_manager.get(trial_id)
        if body.confirmed is not True or body.confirmation is None:
            raise CapabilityAcceptanceError("验收效果确认必须提交完整精确作用域。")
        try:
            _require_capability_trial_binding(
                trial,
                trial_id=body.confirmation.trial_id,
                device_id=body.confirmation.device_id,
                action=body.confirmation.action,
            )
        except CapabilityAcceptanceError:
            runtime.capability_acceptance_manager.cancel(trial_id)
            raise
        _require_supervised_device_ready(trial.device_id)
        before_actions = int(getattr(trial.session, "physical_actions", 0))
        with _supervised_hardware_lock(trial.device_id):
            runtime.capability_acceptance_manager.approve_effects(
                trial_id,
                body.confirmation.model_dump(exclude={"trial_id", "action"}),
            )
        request_actions = int(getattr(trial.session, "physical_actions", 0)) - before_actions
        if request_actions != 0:
            raise CapabilityAcceptanceError("验收效果确认错误地产生了物理动作。")
        return {
            "mode": "capability_acceptance_single_action",
            "physical_actions": 0,
            "trial": _capability_trial_payload(trial),
        }
    except CAPABILITY_ACCEPTANCE_ERRORS as exc:
        raise HTTPException(
            status_code=409,
            detail=_capability_failure(trial, exc),
        ) from exc


@app.post("/api/capability-acceptance/{trial_id}/confirm")
def confirm_capability_acceptance(
    trial_id: str,
    body: CapabilityActionConfirmationRequest,
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    """Consume one trial-bound action confirmation and execute exactly once."""

    verify_local_request(request, x_control_token)
    trial = None
    before_actions = 0
    try:
        trial = runtime.capability_acceptance_manager.get(trial_id)
        if body.confirmed is not True or body.confirmation is None:
            raise CapabilityAcceptanceError("执行验收动作前必须提交完整精确作用域。")
        try:
            _require_capability_trial_binding(
                trial,
                trial_id=body.confirmation.trial_id,
                device_id=body.confirmation.device_id,
                action=body.confirmation.action,
            )
        except CapabilityAcceptanceError:
            runtime.capability_acceptance_manager.cancel(trial_id)
            raise
        _require_supervised_device_ready(trial.device_id)
        before_actions = int(getattr(trial.session, "physical_actions", 0))
        with _supervised_hardware_lock(trial.device_id):
            result = runtime.capability_acceptance_manager.confirm(
                trial_id,
                body.confirmation.model_dump(exclude={"trial_id", "action"}),
            )
        request_actions = int(getattr(trial.session, "physical_actions", 0)) - before_actions
        if request_actions != 1:
            raise CapabilityAcceptanceError(
                f"验收动作确认必须恰好执行一次，实际为 {request_actions}。"
            )
        return {
            "mode": "capability_acceptance_single_action",
            "physical_actions": 1,
            "execution": _capability_execution_payload(result),
            "trial": _capability_trial_payload(trial),
        }
    except CAPABILITY_ACCEPTANCE_ERRORS as exc:
        request_actions = (
            max(
                0,
                int(getattr(trial.session, "physical_actions", 0)) - before_actions,
            )
            if trial is not None
            else 0
        )
        raise HTTPException(
            status_code=409,
            detail=_capability_failure(
                trial,
                exc,
                request_action_count=request_actions,
            ),
        ) from exc


@app.get("/api/capability-acceptance/{trial_id}/promotion-preview")
def preview_capability_promotion(
    trial_id: str,
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    verify_local_request(request, x_control_token)
    try:
        scope = runtime.capability_acceptance_manager.promotion_scope(trial_id)
        return {
            "physical_actions": 0,
            "promotion_scope": scope.to_dict(),
            "requires_separate_confirmation": True,
        }
    except CapabilityAcceptanceError as exc:
        raise HTTPException(
            status_code=409,
            detail=_capability_failure(None, exc),
        ) from exc


@app.post("/api/capability-acceptance/{trial_id}/promote")
def promote_capability_acceptance(
    trial_id: str,
    body: CapabilityPromotionRequest,
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    """Atomically update disk configuration; never touch camera or hardware."""

    verify_local_request(request, x_control_token)
    trial = None
    try:
        trial = runtime.capability_acceptance_manager.get(trial_id)
        if body.confirmed is not True:
            raise CapabilityAcceptanceError("能力晋级需要单独明确确认。")
        confirmation = body.model_dump(exclude={"confirmed"})
        result = runtime.capability_acceptance_manager.promote(
            trial_id,
            confirmation,
        )
        return {
            "physical_actions": 0,
            "promotion": result,
            "trial": _capability_trial_payload(trial),
        }
    except CapabilityAcceptanceError as exc:
        raise HTTPException(
            status_code=409,
            detail=_capability_failure(trial, exc),
        ) from exc


@app.post("/api/capability-acceptance/{trial_id}/cancel")
def cancel_capability_acceptance(
    trial_id: str,
    body: CapabilityCancelRequest,
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    verify_local_request(request, x_control_token)
    trial = None
    try:
        trial = runtime.capability_acceptance_manager.get(trial_id)
        _require_capability_trial_binding(
            trial,
            trial_id=trial_id,
            device_id=body.device_id,
            action=body.action,
        )
        runtime.capability_acceptance_manager.cancel(trial_id)
        return {
            "physical_actions": 0,
            "trial": _capability_trial_payload(trial),
        }
    except CAPABILITY_ACCEPTANCE_ERRORS as exc:
        raise HTTPException(
            status_code=409,
            detail=_capability_failure(trial, exc),
        ) from exc


@app.post("/api/agent/generic-supervised/start")
def start_generic_supervised_session(
    body: GenericSupervisedStartRequest,
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    """Plan once, then autonomously advance only safe read/navigation actions."""

    verify_local_request(request, x_control_token)
    active_session_id = (
        runtime.universal_agent_orchestrator.device_registry.active_session(
            body.device_id
        )
    )
    if active_session_id is not None:
        exc = UniversalAgentOrchestratorError(
            f"设备 {body.device_id} 已有活动任务：{active_session_id}。"
        )
        raise HTTPException(
            status_code=409,
            detail=_generic_supervised_failure(None, exc),
        )
    _require_supervised_device_ready(body.device_id)
    session_id = uuid.uuid4().hex
    run_dir = WEB_OUTPUT_DIR / (
        "generic_supervised_"
        + datetime.now().strftime("%Y%m%d_%H%M%S_")
        + session_id[:8]
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    session = None
    try:
        with _supervised_hardware_lock(body.device_id):
            session = runtime.universal_agent_orchestrator.start(
                session_id=session_id,
                raw_goal=body.text,
                device_id=body.device_id,
                run_dir=run_dir,
            )
        with runtime.generic_supervised_session_lock:
            runtime.generic_supervised_sessions[session_id] = session
        auto_result = {
            "physical_actions": 0,
            "iterations": 0,
            "status": session.status,
            "pause_reason": "当前没有可自动推进的安全动作。",
        }
        if (
            body.auto_advance is True
            and session.status in {"awaiting_confirmation", "needs_reobservation"}
        ):
            with _supervised_hardware_lock(body.device_id):
                auto_result = (
                    runtime.universal_agent_orchestrator.run_autonomous_safe_loop(
                        session,
                        max_physical_actions=12,
                        max_iterations=24,
                    )
                )
        report = _write_generic_supervised_report(session)
        return {
            "mode": (
                "generic_supervised_autonomous_safe_loop"
                if body.auto_advance is True
                else "generic_supervised_single_step"
            ),
            "physical_actions": auto_result["physical_actions"],
            "automatic_loop_supported": True,
            "automatic_loop_enabled": body.auto_advance,
            "automatic_progress": auto_result,
            "session": session.snapshot(),
            "report": report,
        }
    except (
        IntentProviderError,
        GenericActionAdapterError,
        UniversalActionError,
        UniversalAgentOrchestratorError,
        TaskGraphError,
        VisionAgentError,
    ) as exc:
        failure = _generic_supervised_failure(session, exc)
        failure["evidence"] = [
            str(path) for path in sorted(run_dir.glob("*.jpg"))
        ]
        raise HTTPException(status_code=409, detail=failure) from exc


@app.get("/api/agent/generic-supervised/{session_id}")
def get_generic_supervised_session(session_id: str) -> dict[str, Any]:
    with runtime.generic_supervised_session_lock:
        session = runtime.generic_supervised_sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="通用单步会话不存在。")
    return {
        "mode": "generic_supervised_single_step",
        "session": session.snapshot(),
        "report": str(session.run_dir / "report.json"),
    }


@app.post("/api/agent/generic-supervised/{session_id}/approve-effect")
def approve_generic_supervised_effect(
    session_id: str,
    body: GenericEffectApprovalRequest,
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    """Approve one canonical local-effect policy scope and execute at most one action."""

    verify_local_request(request, x_control_token)
    with runtime.generic_supervised_session_lock:
        session = runtime.generic_supervised_sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="通用单步会话不存在。")
    _require_supervised_device_ready(session.device_id)
    before_actions = session.physical_actions
    try:
        if body.confirmed is not True or body.confirmation is None:
            raise UniversalAgentOrchestratorError(
                "调用 Qwen 处理受限效果前必须确认完整效果作用域。"
            )
        _require_generic_session_device(session, body.confirmation.device_id)
        with _supervised_hardware_lock(session.device_id):
            result = runtime.universal_agent_orchestrator.approve_effects(
                session,
                body.confirmation.model_dump(),
            )
        request_actions = session.physical_actions - before_actions
        if request_actions not in {0, 1}:
            raise UniversalAgentOrchestratorError(
                "一次效果确认产生了超过一个物理动作。"
            )
        report = _write_generic_supervised_report(session)
        response = {
            "mode": "generic_supervised_single_step",
            "physical_actions": request_actions,
            "automatic_loop_enabled": False,
            "session": session.snapshot(),
            "report": report,
        }
        if hasattr(result, "action_outcome"):
            response["execution"] = result.to_dict()
        else:
            response["proposal"] = result.proposal.to_dict()
        return response
    except (
        GenericActionAdapterError,
        UniversalActionError,
        UniversalAgentOrchestratorError,
        IntentProviderError,
        TaskGraphError,
        VisionAgentError,
    ) as exc:
        request_actions = max(0, session.physical_actions - before_actions)
        raise HTTPException(
            status_code=409,
            detail=_generic_supervised_failure(
                session,
                exc,
                request_action_count=request_actions,
            ),
        ) from exc


@app.post("/api/agent/generic-supervised/{session_id}/confirm")
def confirm_generic_supervised_session(
    session_id: str,
    body: GenericSupervisedStepRequest,
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    """Consume one exact authority scope and execute at most one action."""

    verify_local_request(request, x_control_token)
    with runtime.generic_supervised_session_lock:
        session = runtime.generic_supervised_sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="通用单步会话不存在。")
    try:
        _require_supervised_device_ready(session.device_id)
    except HTTPException:
        try:
            runtime.universal_agent_orchestrator.invalidate_confirmation(
                session,
                reason="device_readiness_failed",
            )
        except Exception:
            pass
        raise
    before_actions = session.physical_actions
    try:
        if body.confirmed is not True or body.confirmation is None:
            raise UniversalAgentOrchestratorError(
                "执行一个动作前必须提交完整且明确的确认作用域。"
            )
        with _supervised_hardware_lock(session.device_id):
            result = runtime.universal_agent_orchestrator.confirm_one(
                session,
                body.confirmation.model_dump(),
            )
        report = _write_generic_supervised_report(session)
        return {
            "mode": "generic_supervised_single_step",
            "automatic_loop_enabled": False,
            "execution": result.to_dict(),
            "session": session.snapshot(),
            "report": report,
        }
    except (
        GenericActionAdapterError,
        UniversalActionError,
        UniversalAgentOrchestratorError,
        IntentProviderError,
        TaskGraphError,
        VisionAgentError,
    ) as exc:
        request_actions = max(0, session.physical_actions - before_actions)
        raise HTTPException(
            status_code=409,
            detail=_generic_supervised_failure(
                session,
                exc,
                request_action_count=request_actions,
            ),
        ) from exc


@app.post("/api/agent/generic-supervised/{session_id}/next")
def plan_next_generic_supervised_step(
    session_id: str,
    body: GenericSupervisedDeviceRequest,
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    """Reobserve and propose the next action without touching the robot."""

    verify_local_request(request, x_control_token)
    with runtime.generic_supervised_session_lock:
        session = runtime.generic_supervised_sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="通用单步会话不存在。")
    _require_generic_session_device(session, body.device_id)
    _require_supervised_device_ready(session.device_id)
    before_actions = session.physical_actions
    try:
        with _supervised_hardware_lock(session.device_id):
            decision = runtime.universal_agent_orchestrator.refresh_decision(session)
        report = _write_generic_supervised_report(session)
        return {
            "mode": "generic_supervised_single_step",
            "physical_actions": 0,
            "automatic_loop_enabled": False,
            "proposal": decision.proposal.to_dict(),
            "session": session.snapshot(),
            "report": report,
        }
    except (
        GenericActionAdapterError,
        UniversalActionError,
        IntentProviderError,
        UniversalAgentOrchestratorError,
        TaskGraphError,
        VisionAgentError,
    ) as exc:
        raise HTTPException(
            status_code=409,
            detail=_generic_supervised_failure(
                session,
                exc,
                request_action_count=max(
                    0, session.physical_actions - before_actions
                ),
            ),
        ) from exc


@app.post("/api/agent/generic-supervised/{session_id}/auto")
def run_generic_supervised_safe_loop(
    session_id: str,
    body: GenericSupervisedAutoRequest,
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    """Continue only safe read/navigation work without user action confirmation."""

    verify_local_request(request, x_control_token)
    with runtime.generic_supervised_session_lock:
        session = runtime.generic_supervised_sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="通用单步会话不存在。")
    _require_generic_session_device(session, body.device_id)
    try:
        _require_supervised_device_ready(session.device_id)
    except HTTPException:
        try:
            runtime.universal_agent_orchestrator.invalidate_confirmation(
                session,
                reason="device_readiness_failed",
            )
        except Exception:
            pass
        raise
    before_actions = session.physical_actions
    try:
        if body.confirmed is True or body.confirmation is not None:
            raise UniversalAgentOrchestratorError(
                "安全自动推进不接收用户动作确认；外部影响请使用风险确认接口。"
            )
        with _supervised_hardware_lock(session.device_id):
            result = runtime.universal_agent_orchestrator.run_autonomous_safe_loop(
                session,
                max_physical_actions=body.max_physical_actions,
                max_iterations=body.max_iterations,
            )
        report = _write_generic_supervised_report(session)
        return {
            "mode": "generic_supervised_safe_loop",
            "automatic_loop_enabled": True,
            "execution": result,
            "session": session.snapshot(),
            "report": report,
        }
    except (
        GenericActionAdapterError,
        UniversalActionError,
        UniversalAgentOrchestratorError,
        IntentProviderError,
        TaskGraphError,
        VisionAgentError,
    ) as exc:
        raise HTTPException(
            status_code=409,
            detail=_generic_supervised_failure(
                session,
                exc,
                request_action_count=max(
                    0, session.physical_actions - before_actions
                ),
            ),
        ) from exc


@app.post("/api/agent/generic-supervised/{session_id}/cancel")
def cancel_generic_supervised_session(
    session_id: str,
    body: GenericSupervisedDeviceRequest,
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    verify_local_request(request, x_control_token)
    with runtime.generic_supervised_session_lock:
        session = runtime.generic_supervised_sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="通用单步会话不存在。")
        _require_generic_session_device(session, body.device_id)
        runtime.universal_agent_orchestrator.cancel(session)
    report = _write_generic_supervised_report(session)
    return {"session": session.snapshot(), "report": report}


@app.post("/api/agent/generic-supervised/{session_id}/pause")
def pause_generic_supervised_session(
    session_id: str,
    body: GenericSupervisedDeviceRequest,
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    """Invalidate any pending confirmation without touching hardware."""

    verify_local_request(request, x_control_token)
    with runtime.generic_supervised_session_lock:
        session = runtime.generic_supervised_sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="通用单步会话不存在。")
        _require_generic_session_device(session, body.device_id)
        runtime.universal_agent_orchestrator.pause(session)
    report = _write_generic_supervised_report(session)
    return {
        "physical_actions": 0,
        "session": session.snapshot(),
        "report": report,
    }


def _write_supervised_report(
    session: SupervisedSemanticSession,
    run_dir: Path,
    *,
    last_step: dict[str, Any] | None = None,
) -> str:
    report_path = run_dir / "report.json"
    payload = {
        "mode": "supervised_single_step",
        "automatic_loop_enabled": False,
        "session": session.snapshot(),
        "last_step": last_step,
    }
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(report_path)


@app.post("/api/agent/supervised/start")
def start_supervised_session(
    body: AgentRequest,
    request: Request,
    x_control_token: str | None = Header(default=None),
) -> dict[str, Any]:
    """Observe once and create a paused session; execute no physical action."""
    verify_local_request(request, x_control_token)
    require_legacy_workflows_enabled()
    _require_supervised_device_ready()
    with runtime.supervised_session_lock:
        active = [
            item
            for item in runtime.supervised_sessions.values()
            if item.snapshot()["status"] == "action"
        ]
        if active:
            raise HTTPException(
                status_code=409,
                detail="已有人工监督会话，请继续或取消后再新建。",
            )

        session_id = uuid.uuid4().hex
        run_dir = WEB_OUTPUT_DIR / (
            "supervised_"
            + datetime.now().strftime("%Y%m%d_%H%M%S_")
            + session_id[:8]
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            with _supervised_hardware_lock():
                parsed = runtime.agent.parse(body.text)
                preview = build_generic_plan_preview(
                    parsed,
                    runtime.generic_orchestrator,
                )
                if not preview["compiled"]:
                    raise SupervisedSemanticSessionError(
                        "文本模型未生成可执行白名单计划。"
                    )
                goal = GoalSpec.from_operation(
                    str(parsed["operation"]),
                    dict(parsed["params"]),
                    objective=str(parsed.get("summary") or "").strip() or None,
                )
                plan = runtime.generic_orchestrator.compile_goal(goal)
                sensor = ReadOnlySemanticDryRunner(
                    runtime.controller.vision_capture,
                    runtime.state_observer,
                )
                router = SemanticActionRouter(
                    observe=ObserveActionAdapter(sensor),
                    ensure_app=EnsureAppActionAdapter(
                        sensor,
                        runtime.controller.vision_tap_relative,
                    ),
                    tap_heart=TapHeartActionAdapter(
                        sensor,
                        runtime.controller.vision_tap_relative,
                    ),
                    swipe_up=SwipeUpActionAdapter(
                        sensor,
                        runtime.controller.vision_swipe_up,
                    ),
                )
                initial = ObserveActionAdapter(sensor).execute(
                    SemanticAction(
                        node_id="supervised_initial_observe",
                        action="observe",
                        params={},
                    ),
                    goal,
                    evidence_dir=run_dir,
                )
                if not initial.safe_for_next_action:
                    raise SupervisedSemanticSessionError(
                        "初始页面不稳定、未知或置信度不足，未创建会话。"
                    )
                session = SupervisedSemanticSession.start(
                    session_id=session_id,
                    goal=goal,
                    plan=plan,
                    initial_observation=initial.action_result.observation,
                    router=router,
                )
                runtime.supervised_sessions[session_id] = session
                runtime.supervised_session_dirs[session_id] = run_dir
                report = _write_supervised_report(session, run_dir)
                return {
                    "mode": "supervised_single_step",
                    "physical_actions": 0,
                    "initial_observation": initial.to_dict(),
                    "session": session.snapshot(),
                    "report": report,
                }
        except (TaskPlanError, SupervisedSemanticSessionError) as exc:
            failure = {
                "success": False,
                "physical_actions": 0,
                "error": str(exc),
                "evidence": [str(path) for path in sorted(run_dir.glob("*.jpg"))],
            }
            (run_dir / "report.json").write_text(
                json.dumps(failure, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            raise HTTPException(status_code=409, detail=failure) from exc


@app.get("/api/agent/supervised/{session_id}")
def get_supervised_session(session_id: str) -> dict[str, Any]:
    require_legacy_workflows_enabled()
    with runtime.supervised_session_lock:
        session = runtime.supervised_sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="人工监督会话不存在。")
        run_dir = runtime.supervised_session_dirs[session_id]
        return {
            "mode": "supervised_single_step",
            "session": session.snapshot(),
            "report": str(run_dir / "report.json"),
        }


@app.post("/api/agent/supervised/{session_id}/step")
def advance_supervised_session(
    session_id: str,
    body: SupervisedStepRequest,
    request: Request,
    x_control_token: str | None = Header(default=None),
) -> dict[str, Any]:
    """Execute at most the one pending semantic node, then pause again."""
    verify_local_request(request, x_control_token)
    require_legacy_workflows_enabled()
    _require_supervised_device_ready()
    with runtime.supervised_session_lock:
        session = runtime.supervised_sessions.get(session_id)
        run_dir = runtime.supervised_session_dirs.get(session_id)
    if session is None or run_dir is None:
        raise HTTPException(status_code=404, detail="人工监督会话不存在。")
    try:
        with _supervised_hardware_lock():
            last_step = session.step(
                confirmed=body.confirmed,
                evidence_dir=run_dir,
            )
        report = _write_supervised_report(
            session,
            run_dir,
            last_step=last_step,
        )
        return {
            "mode": "supervised_single_step",
            "automatic_loop_enabled": False,
            "last_step": last_step,
            "session": session.snapshot(),
            "report": report,
        }
    except SupervisedSemanticSessionError as exc:
        report = _write_supervised_report(session, run_dir)
        detail = {
            "success": False,
            "physical_actions": exc.physical_actions,
            "error": str(exc),
            "evidence": list(exc.evidence),
            "session": session.snapshot(),
            "report": report,
        }
        raise HTTPException(status_code=409, detail=detail) from exc


@app.post("/api/agent/supervised/{session_id}/cancel")
def cancel_supervised_session(
    session_id: str,
    request: Request,
    x_control_token: str | None = Header(default=None),
) -> dict[str, Any]:
    verify_local_request(request, x_control_token)
    with runtime.supervised_session_lock:
        session = runtime.supervised_sessions.get(session_id)
        run_dir = runtime.supervised_session_dirs.get(session_id)
        if session is None or run_dir is None:
            raise HTTPException(status_code=404, detail="人工监督会话不存在。")
        session.cancel()
        report = _write_supervised_report(session, run_dir)
        return {"session": session.snapshot(), "report": report}


@app.post("/api/agent/dry-run-step")
def preview_real_observation_step(
    body: AgentRequest,
    request: Request,
    x_control_token: str | None = Header(default=None),
) -> dict[str, Any]:
    """Use real camera/Qwen observation and return one unexecuted semantic step."""
    verify_local_request(request, x_control_token)
    require_legacy_workflows_enabled()
    status = runtime.controller.device_status()
    if not status.get("controller_online") or not status.get("camera_online"):
        raise HTTPException(status_code=409, detail="控制端或摄像头离线，无法观察。")
    if status.get("busy"):
        raise HTTPException(status_code=409, detail="机械臂正在执行任务，拒绝并发观察。")
    active = [
        item
        for item in runtime.store.list(20)
        if item["status"] in {"queued", "running"}
    ]
    if active:
        raise HTTPException(status_code=409, detail="执行队列非空，拒绝并发观察。")
    if not runtime.vision_provider.status().get("configured"):
        raise HTTPException(status_code=409, detail="千问视觉尚未配置。")
    if not runtime.dry_run_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="已有一次 Dry-run 观察正在进行。")
    try:
        parsed = runtime.agent.parse(body.text)
        preview = build_generic_plan_preview(
            parsed,
            runtime.generic_orchestrator,
        )
        if not preview["compiled"]:
            return {
                **preview,
                "mode": "real_observation_single_step_dry_run",
                "executed": False,
            }
        goal = GoalSpec.from_operation(
            str(parsed["operation"]),
            dict(parsed["params"]),
            objective=str(parsed.get("summary") or "").strip() or None,
        )
        plan = runtime.generic_orchestrator.compile_goal(goal)
        runner = ReadOnlySemanticDryRunner(
            runtime.controller.vision_capture,
            runtime.state_observer,
        )
        result = runner.preview(goal, plan).to_dict()
        result["intent"] = parsed
        return result
    except TaskPlanError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (RobotWorkflowError, VisionAgentError, RuntimeError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    finally:
        runtime.dry_run_lock.release()


@app.post("/api/agent/execute-ensure-app-step")
def execute_ensure_app_step(
    body: EnsureAppStepRequest,
    request: Request,
    x_control_token: str | None = Header(default=None),
) -> dict[str, Any]:
    """Execute exactly one approved ensure_app action and re-observe once."""
    verify_local_request(request, x_control_token)
    require_legacy_workflows_enabled()
    if body.confirmed is not True:
        raise HTTPException(status_code=422, detail="必须明确确认本次单步打开 App。")
    status = runtime.controller.device_status()
    if not status.get("controller_online") or not status.get("camera_online"):
        raise HTTPException(status_code=409, detail="控制端或摄像头离线，拒绝执行。")
    if status.get("busy"):
        raise HTTPException(status_code=409, detail="机械臂正在执行任务，拒绝并发动作。")
    active = [
        item
        for item in runtime.store.list(20)
        if item["status"] in {"queued", "running"}
    ]
    if active:
        raise HTTPException(status_code=409, detail="执行队列非空，拒绝并发动作。")
    if not runtime.vision_provider.status().get("configured"):
        raise HTTPException(status_code=409, detail="千问视觉尚未配置。")
    process_lease = _acquire_compatibility_hardware_lease("execute_ensure_app_step")
    if not runtime.dry_run_lock.acquire(blocking=False):
        process_lease.release()
        raise HTTPException(status_code=409, detail="已有一次语义观察或动作正在进行。")
    if not runtime.controller.operation_lock.acquire(blocking=False):
        runtime.dry_run_lock.release()
        process_lease.release()
        raise HTTPException(status_code=409, detail="机械臂物理控制权已被占用。")

    run_dir = WEB_OUTPUT_DIR / (
        "generic_ensure_app_"
        + datetime.now().strftime("%Y%m%d_%H%M%S_")
        + uuid.uuid4().hex[:8]
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / "report.json"
    try:
        parsed = runtime.agent.parse(body.text)
        preview = build_generic_plan_preview(parsed, runtime.generic_orchestrator)
        if not preview["compiled"]:
            raise SemanticActionAdapterError("文本模型未生成可执行白名单计划。")
        goal = GoalSpec.from_operation(
            str(parsed["operation"]),
            dict(parsed["params"]),
            objective=str(parsed.get("summary") or "").strip() or None,
        )
        plan = runtime.generic_orchestrator.compile_goal(goal)
        sensor = ReadOnlySemanticDryRunner(
            runtime.controller.vision_capture,
            runtime.state_observer,
        )
        before_capture = sensor.observe_rich(
            goal,
            mode="ensure_app_before",
            executed_actions=[],
            explicitly_forbidden=[
                "swipe",
                "type",
                "send",
                "queue_task",
                "second_tap",
            ],
        )
        executor = SingleStepSemanticExecutor(plan)
        first_decision = executor.start(
            to_semantic_observation(before_capture.observation)
        )
        if (
            first_decision.status != "action"
            or first_decision.action is None
            or first_decision.action.action != "ensure_app"
        ):
            raise SemanticActionAdapterError(
                "计划首个动作不是 ensure_app，单步适配器拒绝执行。"
            )
        adapter = EnsureAppActionAdapter(
            sensor,
            runtime.controller.vision_tap_relative,
        )
        executed = adapter.execute(
            first_decision.action,
            goal,
            evidence_dir=run_dir,
            before_capture=before_capture,
        )
        next_decision = executor.advance(executed.action_result)
        payload = {
            "mode": "real_observation_single_ensure_app",
            "execution_enabled": True,
            "executed": executed.robot_action_called,
            "execution_success": executed.action_result.success,
            "goal": goal.to_dict(),
            "first_decision": first_decision.to_dict(),
            "result": executed.to_dict(),
            "next_decision": next_decision.to_dict(),
            "safety": {
                "task_created": False,
                "account_action_performed": False,
                "max_physical_taps": 1,
                "enabled_real_actions": ["ensure_app"],
            },
            "report": str(report_path),
        }
        report_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return payload
    except (TaskPlanError, SemanticActionAdapterError) as exc:
        failure = {
            "success": False,
            "error": str(exc),
            "report": str(report_path),
        }
        report_path.write_text(
            json.dumps(failure, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        raise HTTPException(status_code=409, detail=failure) from exc
    except (RobotWorkflowError, VisionAgentError, RuntimeError) as exc:
        failure = {
            "success": False,
            "error": str(exc),
            "report": str(report_path),
        }
        report_path.write_text(
            json.dumps(failure, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        raise HTTPException(status_code=409, detail=failure) from exc
    finally:
        runtime.controller.operation_lock.release()
        runtime.dry_run_lock.release()
        process_lease.release()


@app.post("/api/agent/execute-observe-step")
def execute_observe_step(
    body: EnsureAppStepRequest,
    request: Request,
    x_control_token: str | None = Header(default=None),
) -> dict[str, Any]:
    """Observe one already-open target App and execute no physical action."""
    verify_local_request(request, x_control_token)
    require_legacy_workflows_enabled()
    if body.confirmed is not True:
        raise HTTPException(status_code=422, detail="必须明确确认本次只读页面观察。")
    status = runtime.controller.device_status()
    if not status.get("controller_online") or not status.get("camera_online"):
        raise HTTPException(status_code=409, detail="控制端或摄像头离线，拒绝观察。")
    if status.get("busy"):
        raise HTTPException(status_code=409, detail="机械臂正在执行任务，拒绝并发观察。")
    active = [
        item
        for item in runtime.store.list(20)
        if item["status"] in {"queued", "running"}
    ]
    if active:
        raise HTTPException(status_code=409, detail="执行队列非空，拒绝并发观察。")
    if not runtime.vision_provider.status().get("configured"):
        raise HTTPException(status_code=409, detail="千问视觉尚未配置。")
    process_lease = _acquire_compatibility_hardware_lease("execute_observe_step")
    if not runtime.dry_run_lock.acquire(blocking=False):
        process_lease.release()
        raise HTTPException(status_code=409, detail="已有一次语义观察正在进行。")
    if not runtime.controller.operation_lock.acquire(blocking=False):
        runtime.dry_run_lock.release()
        process_lease.release()
        raise HTTPException(status_code=409, detail="机械臂物理控制权已被占用。")

    run_dir = WEB_OUTPUT_DIR / (
        "generic_observe_"
        + datetime.now().strftime("%Y%m%d_%H%M%S_")
        + uuid.uuid4().hex[:8]
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / "report.json"
    try:
        parsed = runtime.agent.parse(body.text)
        preview = build_generic_plan_preview(parsed, runtime.generic_orchestrator)
        if not preview["compiled"]:
            raise SemanticActionAdapterError("文本模型未生成可观察的白名单计划。")
        goal = GoalSpec.from_operation(
            str(parsed["operation"]),
            dict(parsed["params"]),
            objective=str(parsed.get("summary") or "").strip() or None,
        )
        plan = runtime.generic_orchestrator.compile_goal(goal)
        sensor = ReadOnlySemanticDryRunner(
            runtime.controller.vision_capture,
            runtime.state_observer,
        )
        observed = ObserveActionAdapter(sensor).execute(
            SemanticAction(
                node_id="observation_capture",
                action="observe",
                params={},
            ),
            goal,
            evidence_dir=run_dir,
        )
        semantic_observation = observed.action_result.observation
        if not semantic_observation.page_state.startswith(f"{goal.app_id}_"):
            raise SemanticActionAdapterError(
                f"当前页面是 {semantic_observation.page_state}；只读阶段要求目标 App 已打开。"
            )

        executor = SingleStepSemanticExecutor(plan)
        first_decision = executor.start(semantic_observation)
        if (
            first_decision.status != "action"
            or first_decision.action is None
            or first_decision.action.action != "ensure_app"
        ):
            raise SemanticActionAdapterError("计划首个动作不是 ensure_app。")
        ensured_result = ActionResult(
            node_id=first_decision.action.node_id,
            success=True,
            observation=semantic_observation,
            details={
                "reason": f"{goal.app_id} 已经打开，只读阶段不点击。",
                "no_op": True,
                "physical_actions": 0,
            },
        )
        observe_decision = executor.advance(ensured_result)
        if (
            observe_decision.status != "action"
            or observe_decision.action is None
            or observe_decision.action.action != "observe"
        ):
            raise SemanticActionAdapterError("ensure_app 后的动作不是 observe。")
        observed_result = ActionResult(
            node_id=observe_decision.action.node_id,
            success=observed.action_result.success,
            observation=semantic_observation,
            details=dict(observed.action_result.details),
        )
        next_decision = executor.advance(observed_result)
        observed_payload = observed.to_dict()
        observed_payload["action_result"]["node_id"] = observe_decision.action.node_id
        payload = {
            "mode": "real_observation_single_read_only_step",
            "execution_enabled": True,
            "physical_actions": 0,
            "goal": goal.to_dict(),
            "ensure_app_result": {
                "success": True,
                "no_op": True,
                "physical_actions": 0,
                "verified_state": semantic_observation.page_state,
            },
            "observe_decision": observe_decision.to_dict(),
            "result": observed_payload,
            "next_decision": next_decision.to_dict(),
            "safety": {
                "task_created": False,
                "account_action_performed": False,
                "physical_actions": 0,
                "enabled_read_only_actions": ["observe"],
            },
            "report": str(report_path),
        }
        report_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return payload
    except (TaskPlanError, SemanticActionAdapterError) as exc:
        failure = {
            "success": False,
            "physical_actions": 0,
            "error": str(exc),
            "evidence": [str(path) for path in sorted(run_dir.glob("*.jpg"))],
            "report": str(report_path),
        }
        report_path.write_text(
            json.dumps(failure, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        raise HTTPException(status_code=409, detail=failure) from exc
    except (RobotWorkflowError, VisionAgentError, RuntimeError) as exc:
        failure = {
            "success": False,
            "physical_actions": 0,
            "error": str(exc),
            "evidence": [str(path) for path in sorted(run_dir.glob("*.jpg"))],
            "report": str(report_path),
        }
        report_path.write_text(
            json.dumps(failure, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        raise HTTPException(status_code=409, detail=failure) from exc
    finally:
        runtime.controller.operation_lock.release()
        runtime.dry_run_lock.release()
        process_lease.release()


@app.post("/api/agent/execute-tap-heart-step")
def execute_tap_heart_step(
    body: EnsureAppStepRequest,
    request: Request,
    x_control_token: str | None = Header(default=None),
) -> dict[str, Any]:
    """Tap one verified white Douyin heart and verify red without retrying."""
    verify_local_request(request, x_control_token)
    require_legacy_workflows_enabled()
    if body.confirmed is not True:
        raise HTTPException(status_code=422, detail="必须明确确认本次单步点赞。")
    status = runtime.controller.device_status()
    if not status.get("controller_online") or not status.get("camera_online"):
        raise HTTPException(status_code=409, detail="控制端或摄像头离线，拒绝执行。")
    if status.get("busy"):
        raise HTTPException(status_code=409, detail="机械臂正在执行任务，拒绝并发动作。")
    active = [
        item
        for item in runtime.store.list(20)
        if item["status"] in {"queued", "running"}
    ]
    if active:
        raise HTTPException(status_code=409, detail="执行队列非空，拒绝并发动作。")
    if not runtime.vision_provider.status().get("configured"):
        raise HTTPException(status_code=409, detail="千问视觉尚未配置。")
    process_lease = _acquire_compatibility_hardware_lease("execute_tap_heart_step")
    if not runtime.dry_run_lock.acquire(blocking=False):
        process_lease.release()
        raise HTTPException(status_code=409, detail="已有一次语义观察或动作正在进行。")
    if not runtime.controller.operation_lock.acquire(blocking=False):
        runtime.dry_run_lock.release()
        process_lease.release()
        raise HTTPException(status_code=409, detail="机械臂物理控制权已被占用。")

    run_dir = WEB_OUTPUT_DIR / (
        "generic_tap_heart_"
        + datetime.now().strftime("%Y%m%d_%H%M%S_")
        + uuid.uuid4().hex[:8]
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / "report.json"
    physical_actions = 0
    evidence: list[str] = []
    command_point: tuple[int, int] | None = None
    try:
        parsed = runtime.agent.parse(body.text)
        preview = build_generic_plan_preview(parsed, runtime.generic_orchestrator)
        if not preview["compiled"]:
            raise SemanticActionAdapterError("文本模型未生成可执行的白名单计划。")
        if (
            parsed.get("app_id") != "douyin"
            or parsed.get("operation") != "douyin.batch_interact"
            or parsed.get("params", {}).get("like") is not True
            or parsed.get("params", {}).get("comment") is True
        ):
            raise SemanticActionAdapterError(
                "本阶段只接受明确的抖音纯点赞目标，不执行评论、搜索或其他动作。"
            )
        goal = GoalSpec.from_operation(
            str(parsed["operation"]),
            dict(parsed["params"]),
            objective=str(parsed.get("summary") or "").strip() or None,
        )
        plan = runtime.generic_orchestrator.compile_goal(goal)
        sensor = ReadOnlySemanticDryRunner(
            runtime.controller.vision_capture,
            runtime.state_observer,
        )
        before_capture = sensor.observe_rich(
            goal,
            mode="tap_heart_before",
            executed_actions=[],
            explicitly_forbidden=[
                "swipe",
                "type",
                "send",
                "queue_task",
                "second_tap",
                "retry_tap",
            ],
        )
        before_semantic = to_semantic_observation(before_capture.observation)

        executor = SingleStepSemanticExecutor(plan)
        ensure_decision = executor.start(before_semantic)
        if (
            ensure_decision.status != "action"
            or ensure_decision.action is None
            or ensure_decision.action.action != "ensure_app"
        ):
            raise SemanticActionAdapterError("计划首个动作不是 ensure_app。")
        observe_decision = executor.advance(
            ActionResult(
                node_id=ensure_decision.action.node_id,
                success=True,
                observation=before_semantic,
                details={"no_op": True, "physical_actions": 0},
            )
        )
        if (
            observe_decision.status != "action"
            or observe_decision.action is None
            or observe_decision.action.action != "observe"
        ):
            raise SemanticActionAdapterError("ensure_app 后的动作不是 observe。")
        tap_decision = executor.advance(
            ActionResult(
                node_id=observe_decision.action.node_id,
                success=True,
                observation=before_semantic,
                details={"physical_actions": 0},
            )
        )
        if (
            tap_decision.status != "action"
            or tap_decision.action is None
            or tap_decision.action.action != "tap_semantic"
            or tap_decision.action.params.get("target") != "heart"
        ):
            raise SemanticActionAdapterError(
                "当前观察没有产生唯一的白心点击动作；可能已经点赞或页面不符。"
            )

        tap_result = TapHeartActionAdapter(
            sensor,
            runtime.controller.vision_tap_relative,
        ).execute(
            tap_decision.action,
            goal,
            evidence_dir=run_dir,
            before_capture=before_capture,
        )
        physical_actions = 1 if tap_result.robot_action_called else 0
        command_point = tap_result.command_point
        evidence = list(tap_result.evidence)
        if not tap_result.action_result.success:
            raise PhysicalActionVerificationError(
                str(tap_result.action_result.details.get("reason") or "点赞后验证失败"),
                physical_actions=physical_actions,
                evidence=tuple(evidence),
                command_point=command_point,
            )

        verify_decision = executor.advance(tap_result.action_result)
        if (
            verify_decision.status != "action"
            or verify_decision.action is None
            or verify_decision.action.action != "observe"
        ):
            raise PhysicalActionVerificationError(
                "爱心已点击并验证，但状态图没有进入 verify_heart。",
                physical_actions=1,
                evidence=tuple(evidence),
                command_point=command_point,
            )
        next_decision = executor.advance(
            ActionResult(
                node_id=verify_decision.action.node_id,
                success=True,
                observation=tap_result.action_result.observation,
                details={"reused_post_tap_observation": True, "physical_actions": 0},
            )
        )
        payload = {
            "mode": "real_single_tap_heart_with_post_verification",
            "execution_enabled": True,
            "execution_success": True,
            "physical_actions": 1,
            "goal": goal.to_dict(),
            "tap_decision": tap_decision.to_dict(),
            "result": tap_result.to_dict(),
            "verify_decision": verify_decision.to_dict(),
            "next_decision": next_decision.to_dict(),
            "safety": {
                "task_created": False,
                "account_action_performed": True,
                "physical_actions": 1,
                "retry_count": 0,
                "swipe_performed": False,
                "enabled_real_action": "tap_semantic:heart",
            },
            "report": str(report_path),
        }
        report_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return payload
    except PhysicalActionVerificationError as exc:
        failure = {
            "success": False,
            "physical_actions": exc.physical_actions,
            "retry_count": 0,
            "swipe_performed": False,
            "error": str(exc),
            "command_point": (
                list(exc.command_point) if exc.command_point is not None else None
            ),
            "evidence": list(exc.evidence),
            "report": str(report_path),
        }
        report_path.write_text(
            json.dumps(failure, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        raise HTTPException(status_code=409, detail=failure) from exc
    except (TaskPlanError, SemanticActionAdapterError) as exc:
        failure = {
            "success": False,
            "physical_actions": physical_actions,
            "retry_count": 0,
            "swipe_performed": False,
            "error": str(exc),
            "command_point": (
                list(command_point) if command_point is not None else None
            ),
            "evidence": evidence,
            "report": str(report_path),
        }
        report_path.write_text(
            json.dumps(failure, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        raise HTTPException(status_code=409, detail=failure) from exc
    except (RobotWorkflowError, VisionAgentError, RuntimeError) as exc:
        failure = {
            "success": False,
            "physical_actions": physical_actions,
            "retry_count": 0,
            "swipe_performed": False,
            "error": str(exc),
            "command_point": (
                list(command_point) if command_point is not None else None
            ),
            "evidence": evidence,
            "report": str(report_path),
        }
        report_path.write_text(
            json.dumps(failure, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        raise HTTPException(status_code=409, detail=failure) from exc
    finally:
        runtime.controller.operation_lock.release()
        runtime.dry_run_lock.release()
        process_lease.release()


@app.post("/api/tasks")
def create_task(
    body: TaskRequest,
    request: Request,
    x_control_token: str | None = Header(default=None),
) -> JSONResponse:
    verify_local_request(request, x_control_token)
    require_legacy_workflows_enabled()
    normalized = normalize_task_request(body)
    validate_task_payload(normalized)
    created = runtime.store.create(
        normalized.app_id,
        normalized.operation,
        normalized.params,
    )
    return JSONResponse(created, status_code=201)


@app.post("/api/tasks/{task_id}/confirm")
def confirm_task(
    task_id: str,
    request: Request,
    x_control_token: str | None = Header(default=None),
) -> dict[str, Any]:
    verify_local_request(request, x_control_token)
    require_legacy_workflows_enabled()
    device_id = runtime.device_controllers.default_device_id
    active_session = runtime.device_task_registry.active_session(device_id)
    if active_session is not None:
        raise HTTPException(
            status_code=409,
            detail=f"设备 {device_id} 已有活动任务：{active_session}。",
        )
    status = runtime.controller.device_status()
    if not status.get("controller_online") or not status.get("camera_online"):
        raise HTTPException(status_code=409, detail="控制端或摄像头离线，拒绝执行。")
    try:
        task_item = runtime.store.get(task_id)
        readiness_key = {
            "wechat.send_text_to_file_transfer": "wechat",
            "wechat.send_text": "vision_agent",
            "wechat.send_album_image": "vision_agent",
            "douyin.like_current": "douyin_like",
            "douyin.comment_current": "douyin_comment",
            "douyin.search": "vision_agent",
            "douyin.batch_interact": "vision_agent",
        }[task_item["operation"]]
        if readiness_key == "vision_agent":
            # The controller only reports hardware/camera state. Qwen belongs to
            # the web runtime, so reading status["vision_agent"] here always
            # produced a false "DASHSCOPE_API_KEY missing" error at confirmation.
            ready = bool(runtime.vision_provider.status().get("configured"))
            workflow_state = {
                "missing_templates": [],
                "missing_capabilities": (
                    [] if ready else ["DASHSCOPE_API_KEY"]
                ),
            }
        else:
            workflow_state = status.get("readiness", {}).get(readiness_key, {})
            ready = workflow_state.get("ready", False)
        if not ready:
            missing = workflow_state.get("missing_templates", [])
            capabilities = workflow_state.get("missing_capabilities", [])
            details = []
            if missing:
                details.append(f"缺少模板：{', '.join(missing)}")
            if capabilities:
                details.append(f"缺少能力：{', '.join(capabilities)}")
            suffix = f" {'；'.join(details)}" if details else ""
            raise HTTPException(status_code=409, detail=f"该工作流尚未就绪。{suffix}")
        updated = runtime.store.transition(
            task_id,
            {"awaiting_confirmation"},
            "queued",
            message="用户已确认，任务进入单执行队列。",
        )
        runtime.jobs.put(task_id)
        return updated
    except KeyError:
        raise HTTPException(status_code=404, detail="任务不存在。")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.post("/api/tasks/{task_id}/cancel")
def cancel_task(
    task_id: str,
    request: Request,
    x_control_token: str | None = Header(default=None),
) -> dict[str, Any]:
    verify_local_request(request, x_control_token)
    require_legacy_workflows_enabled()
    try:
        task_item = runtime.store.get(task_id)
        if task_item["status"] in {"draft", "awaiting_confirmation", "queued"}:
            return runtime.store.transition(
                task_id,
                {task_item["status"]},
                "cancelled",
                message="任务已取消。",
            )
        if task_item["status"] == "running":
            runtime.controller.request_stop()
            return runtime.store.get(task_id)
        raise HTTPException(status_code=409, detail="任务已经结束。")
    except KeyError:
        raise HTTPException(status_code=404, detail="任务不存在。")


@app.post("/api/stop")
def stop_all(
    request: Request,
    x_control_token: str | None = Header(default=None),
) -> dict[str, Any]:
    verify_local_request(request, x_control_token)
    runtime.controller.request_stop()
    capability_stop_requested = (
        runtime.capability_acceptance_manager.request_stop_all()
    )
    return {
        "stop_requested": True,
        "queued_cancelled": [],
        "capability_stop_requested": capability_stop_requested,
        "note": "正在执行的任务会在当前最小动作结束后停止。",
    }


@app.get("/api/preview.jpg")
def preview_jpg(device_id: str) -> Response:
    try:
        controller = runtime.controller_for_device(device_id)
    except UniversalAgentOrchestratorError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    try:
        content = controller.capture_preview()
    except Exception:
        content = MockRobotController(device_id="mock-preview").capture_preview()
    return Response(content, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/preview.mjpg")
def preview_mjpg(device_id: str) -> StreamingResponse:
    try:
        controller = runtime.controller_for_device(device_id)
    except UniversalAgentOrchestratorError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    def generate() -> Iterator[bytes]:
        while True:
            try:
                frame = controller.capture_preview(quality=68)
            except Exception:
                frame = MockRobotController(
                    device_id="mock-preview"
                ).capture_preview(quality=68)
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n"
                + f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii")
                + frame
                + b"\r\n"
            )
            time.sleep(0.35)

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/events")
def events() -> StreamingResponse:
    def generate() -> Iterator[str]:
        last_revision = -1
        while True:
            revision = runtime.store.revision
            if revision != last_revision:
                payload = {
                    "revision": revision,
                    "tasks": runtime.store.list(20),
                    "events": runtime.store.events(30),
                }
                yield f"event: snapshot\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                last_revision = revision
            else:
                yield ": keepalive\n\n"
            time.sleep(1)

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/api/evidence/{task_id}/{index}")
def evidence(task_id: str, index: int) -> FileResponse:
    try:
        task_item = runtime.store.get(task_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="任务不存在。")
    evidence_items = (task_item.get("result") or {}).get("evidence", [])
    if index < 0 or index >= len(evidence_items):
        raise HTTPException(status_code=404, detail="截图不存在。")
    path = Path(evidence_items[index]).resolve()
    output_root = WEB_OUTPUT_DIR.parent.resolve()
    try:
        path.relative_to(output_root)
    except ValueError:
        raise HTTPException(status_code=403, detail="截图路径不在允许目录内。")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="截图文件不存在。")
    return FileResponse(path)


@app.get("/api/report/{task_id}")
def task_report(task_id: str) -> FileResponse:
    try:
        task_item = runtime.store.get(task_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="任务不存在。")
    report_value = (task_item.get("result") or {}).get("report")
    if not isinstance(report_value, str) or not report_value:
        raise HTTPException(status_code=404, detail="任务报告不存在。")
    path = Path(report_value).resolve()
    output_root = WEB_OUTPUT_DIR.parent.resolve()
    try:
        path.relative_to(output_root)
    except ValueError:
        raise HTTPException(status_code=403, detail="报告路径不在允许目录内。")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="任务报告文件不存在。")
    return FileResponse(path, media_type="application/json", filename="report.json")


# The old fixed-App handlers remain only as Git-readable migration history.
# They are removed from the application unconditionally and cannot be restored
# by an environment variable or runtime flag.
RETIRED_FIXED_APP_ROUTE_NAMES = frozenset(
    {
        "tasks",
        "task",
        "parse_agent",
        "parse_generic_agent_goal",
        "preview_agent_plan",
        "start_supervised_session",
        "get_supervised_session",
        "advance_supervised_session",
        "cancel_supervised_session",
        "preview_real_observation_step",
        "execute_ensure_app_step",
        "execute_observe_step",
        "execute_tap_heart_step",
        "create_task",
        "confirm_task",
        "cancel_task",
        "events",
        "evidence",
        "task_report",
    }
)


def _retire_fixed_app_routes() -> None:
    app.router.routes[:] = [
        route
        for route in app.router.routes
        if getattr(route, "name", None) not in RETIRED_FIXED_APP_ROUTE_NAMES
    ]


_retire_fixed_app_routes()
