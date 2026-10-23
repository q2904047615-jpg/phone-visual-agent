"""The one action vocabulary shared by the model prompt, the validator and devices.

Coordinates are always on a 0..1000 grid over the full current screenshot
(x to the right, y downward). Devices translate grid points to their own pixels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

POINT_ACTIONS = {"tap", "double_tap", "long_press"}
PATH_ACTIONS = {"swipe", "drag"}
TEXT_ACTIONS = {"type", "clear", "enter"}
NAV_ACTIONS = {"back", "home", "recents"}
ALL_ACTIONS = POINT_ACTIONS | PATH_ACTIONS | TEXT_ACTIONS | NAV_ACTIONS | {"scroll", "launch", "wait", "finish"}
RISKS = {"none", "authentication", "payment"}

# Finger movement for ``scroll``: "up" moves the finger upward (reveals content below).
SCROLL_PATHS = {
    "up": ((500, 750), (500, 300)),
    "down": ((500, 300), (500, 750)),
    "left": ((800, 500), (200, 500)),
    "right": ((200, 500), (800, 500)),
}

_ALIASES = {
    "click": "tap", "press": "tap", "dismiss": "tap", "longpress": "long_press", "long_tap": "long_press",
    "doubletap": "double_tap", "input": "type", "type_text": "type", "clear_input": "clear",
    "press_enter": "enter", "newline": "enter", "launch_app": "launch", "open_app": "launch",
    "open_recent_apps": "recents", "recent_apps": "recents", "wait_for_change": "wait", "done": "finish",
}


class ActionError(ValueError):
    """The model's action cannot be executed as given."""


@dataclass(frozen=True)
class Action:
    type: str
    point: tuple[int, int] | None = None
    start: tuple[int, int] | None = None
    end: tuple[int, int] | None = None
    direction: str | None = None
    text: str | None = None
    app: str | None = None
    seconds: float | None = None
    message: str | None = None
    success: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {key: list(value) if isinstance(value, tuple) else value
                for key, value in asdict(self).items() if value is not None}

    @property
    def physical(self) -> bool:
        return self.type not in {"wait", "finish"}


def _point(value: Any, name: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ActionError(f"{name} 必须是 [x, y]。")
    try:
        x, y = (round(float(item)) for item in value)
    except (TypeError, ValueError) as exc:
        raise ActionError(f"{name} 坐标不是数字。") from exc
    if not (0 <= x <= 1000 and 0 <= y <= 1000):
        raise ActionError(f"{name} 坐标 {x},{y} 超出 0..1000。")
    return x, y


def parse_action(raw: Any, *, apps: list[str] | None = None, supported: set[str] | None = None) -> Action:
    """Validate one model action. Unknown extra keys are ignored."""
    if not isinstance(raw, dict):
        raise ActionError("action 必须是对象。")
    kind = str(raw.get("type") or raw.get("action") or "").strip().lower()
    kind = _ALIASES.get(kind, kind)
    if kind not in ALL_ACTIONS:
        raise ActionError(f"未知动作类型：{kind or '空'}。")
    if supported is not None and kind not in supported:
        raise ActionError(f"当前设备不支持动作 {kind}。")

    if kind in POINT_ACTIONS:
        return Action(kind, point=_point(raw.get("point"), "point"))
    if kind in PATH_ACTIONS:
        start, end = _point(raw.get("start"), "start"), _point(raw.get("end"), "end")
        if start == end:
            raise ActionError("起点和终点不能相同。")
        return Action(kind, start=start, end=end)
    if kind == "scroll":
        direction = str(raw.get("direction") or "").strip().lower()
        if direction not in SCROLL_PATHS:
            raise ActionError("scroll 的 direction 只能是 up/down/left/right。")
        start, end = SCROLL_PATHS[direction]
        return Action("swipe", start=start, end=end, direction=direction)
    if kind == "type":
        text = raw.get("text")
        if not isinstance(text, str) or not text:
            raise ActionError("type 需要非空 text。")
        return Action(kind, text=text)
    if kind == "launch":
        app = str(raw.get("app") or "").strip()
        if not app:
            raise ActionError("launch 需要 app 名称。")
        if apps is not None:
            match = next((name for name in apps if name.casefold() == app.casefold()), None)
            if match is None:
                raise ActionError(f"App「{app}」不在本机直启列表中，请在桌面上点击图标打开。")
            app = match
        return Action(kind, app=app)
    if kind == "wait":
        try:
            seconds = float(raw.get("seconds") or 2)
        except (TypeError, ValueError) as exc:
            raise ActionError("wait 的 seconds 不是数字。") from exc
        return Action(kind, seconds=min(60.0, max(0.5, seconds)))
    if kind == "finish":
        success = raw.get("success", True)
        if isinstance(success, str):
            success = success.strip().lower() not in {"false", "0", "no"}
        return Action(kind, message=str(raw.get("message") or "").strip() or None, success=bool(success))
    return Action(kind)


def describe(action: Action) -> str:
    """Short Chinese description for the console and the model history."""
    t = action.type
    if t == "tap":
        return f"点击 {list(action.point)}"
    if t == "double_tap":
        return f"双击 {list(action.point)}"
    if t == "long_press":
        return f"长按 {list(action.point)}"
    if t == "swipe" and action.direction:
        return f"滚动（手指向{ {'up': '上', 'down': '下', 'left': '左', 'right': '右'}[action.direction] }）"
    if t in PATH_ACTIONS:
        return f"{'滑动' if t == 'swipe' else '拖动'} {list(action.start)} → {list(action.end)}"
    if t == "type":
        return f"输入「{action.text}」"
    if t == "launch":
        return f"打开 App「{action.app}」"
    if t == "wait":
        return f"等待 {action.seconds:g} 秒"
    if t == "finish":
        head = "任务完成" if action.success is not False else "无法完成任务"
        return head + (f"：{action.message}" if action.message else "")
    return {"clear": "清空输入框", "enter": "换行", "back": "返回", "home": "回到桌面",
            "recents": "打开最近任务"}[t]
