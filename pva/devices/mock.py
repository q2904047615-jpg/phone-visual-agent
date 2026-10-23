"""A small simulated Android phone for demos and offline tests.

It renders real pixels (home screen, Settings, Notes, Messages, Recents) and reacts
to taps, swipes, text and navigation, so the whole Qwen loop can be exercised
without the robot arm. State is observable through ``SimPhone.state`` in tests.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageFont

from pva.devices.base import Device, grid_to_pixel

W, H = 540, 960
STATUS_H, NAV_TOP = 36, 905
NAV_Y = 937
NAV_X = {"recents": 178, "home": 270, "back": 370}
KEYBOARD_TOP = 600

_FONT_FILES = ("msyh.ttc", "C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf",
               "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
               "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", "/System/Library/Fonts/PingFang.ttc")
_fonts: dict[int, ImageFont.ImageFont] = {}


def font(size: int) -> ImageFont.ImageFont:
    if size not in _fonts:
        for name in _FONT_FILES:
            try:
                _fonts[size] = ImageFont.truetype(name, size)
                break
            except OSError:
                continue
        else:
            _fonts[size] = ImageFont.load_default(size)
    return _fonts[size]


APPS = [("设置", (90, 110, 140)), ("备忘录", (235, 170, 40)), ("消息", (50, 170, 90)), ("相册", (200, 80, 120))]
SETTINGS = [("WLAN", True), ("蓝牙", False), ("飞行模式", False), ("显示与亮度", None), ("声音与振动", None),
            ("通知", None), ("电池", None), ("存储", None), ("隐私", None), ("应用管理", None),
            ("深色模式", False), ("关于手机", None)]
CONTACTS = ["张三", "李四", "王五"]


@dataclass
class PhoneState:
    screen: str = "home"            # home | app:<name> | editor | chat:<name> | recents
    toggles: dict[str, bool] = field(default_factory=lambda: {k: v for k, v in SETTINGS if v is not None})
    settings_scroll: int = 0
    notes: list[str] = field(default_factory=list)
    draft: str = ""
    chats: dict[str, list[str]] = field(default_factory=lambda: {"张三": ["明天见"], "李四": [], "王五": []})
    chat_draft: str = ""
    focused: str | None = None      # "draft" | "chat" | None
    recent_apps: list[str] = field(default_factory=list)
    toast: str = ""


Hit = tuple[tuple[int, int, int, int], Callable[[], None]]


class SimPhone:
    def __init__(self) -> None:
        self.state = PhoneState()
        self.lock = threading.RLock()
        self._hits: list[Hit] = []

    # -- rendering -------------------------------------------------------------
    def render(self) -> Image.Image:
        with self.lock:
            image = Image.new("RGB", (W, H), (245, 246, 248))
            draw = ImageDraw.Draw(image)
            self._hits = []
            screen = self.state.screen
            if screen == "home":
                self._home(draw)
            elif screen == "recents":
                self._recents(draw)
            elif screen == "editor":
                self._editor(draw)
            elif screen.startswith("chat:"):
                self._chat(draw, screen[5:])
            elif screen == "app:设置":
                self._settings(draw)
            elif screen == "app:备忘录":
                self._notes(draw)
            elif screen == "app:消息":
                self._messages(draw)
            else:
                self._title(draw, screen[4:])
                draw.text((W // 2, 400), "暂无内容", fill=(150, 150, 150), font=font(24), anchor="mm")
            if self.state.focused:
                self._keyboard(draw)
            if self.state.toast:
                draw.rounded_rectangle((120, 780, 420, 830), 20, fill=(60, 60, 60))
                draw.text((W // 2, 805), self.state.toast, fill="white", font=font(20), anchor="mm")
            self._chrome(draw)
            return image

    def _chrome(self, draw: ImageDraw.ImageDraw) -> None:
        draw.rectangle((0, 0, W, STATUS_H), fill=(30, 30, 30) if self.state.screen == "home" else (235, 236, 240))
        color = "white" if self.state.screen == "home" else (40, 40, 40)
        draw.text((16, 8), "10:24", fill=color, font=font(18))
        draw.text((W - 70, 8), "100%", fill=color, font=font(18))
        draw.rectangle((0, NAV_TOP, W, H), fill=(250, 250, 250))
        draw.rectangle((NAV_X["recents"] - 14, NAV_Y - 14, NAV_X["recents"] + 14, NAV_Y + 14), outline=(90, 90, 90), width=3)
        draw.ellipse((NAV_X["home"] - 16, NAV_Y - 16, NAV_X["home"] + 16, NAV_Y + 16), outline=(90, 90, 90), width=3)
        bx = NAV_X["back"]
        draw.polygon([(bx - 14, NAV_Y), (bx + 12, NAV_Y - 15), (bx + 12, NAV_Y + 15)], outline=(90, 90, 90), width=3)
        for key, x in NAV_X.items():
            self._hits.append(((x - 40, NAV_TOP, x + 40, H), lambda key=key: self.navigate(key)))

    def _title(self, draw, text: str, back: bool = True) -> None:
        draw.rectangle((0, STATUS_H, W, 110), fill="white")
        if back:
            draw.text((24, 72), "‹", fill=(40, 40, 40), font=font(44), anchor="lm")
            self._hits.append(((0, STATUS_H, 80, 110), lambda: self.navigate("back")))
        draw.text((W // 2, 73), text, fill=(20, 20, 20), font=font(28), anchor="mm")
        draw.line((0, 110, W, 110), fill=(225, 225, 225))

    def _home(self, draw) -> None:
        draw.rectangle((0, 0, W, NAV_TOP), fill=(52, 84, 140))
        draw.text((W // 2, 150), "10:24", fill="white", font=font(64), anchor="mm")
        draw.text((W // 2, 205), "10月24日 星期六", fill=(220, 225, 240), font=font(22), anchor="mm")
        for index, (name, color) in enumerate(APPS):
            col, row = index % 4, index // 4
            x0, y0 = 40 + col * 120, 640 + row * 140
            draw.rounded_rectangle((x0, y0, x0 + 96, y0 + 96), 24, fill=color)
            draw.text((x0 + 48, y0 + 48), name[0], fill="white", font=font(40), anchor="mm")
            draw.text((x0 + 48, y0 + 118), name, fill="white", font=font(20), anchor="mm")
            self._hits.append(((x0, y0, x0 + 96, y0 + 130), lambda name=name: self.open_app(name)))

    def _recents(self, draw) -> None:
        draw.rectangle((0, 0, W, NAV_TOP), fill=(40, 40, 46))
        if not self.state.recent_apps:
            draw.text((W // 2, 420), "没有最近使用的应用", fill=(200, 200, 200), font=font(24), anchor="mm")
        for index, name in enumerate(self.state.recent_apps[-3:]):
            x0 = 40 + index * 160
            draw.rounded_rectangle((x0, 200, x0 + 140, 600), 18, fill=(235, 235, 240))
            draw.text((x0 + 70, 230), name, fill=(30, 30, 30), font=font(22), anchor="mm")
            self._hits.append(((x0, 200, x0 + 140, 600), lambda name=name: self.open_app(name)))
        draw.ellipse((W // 2 - 40, 760, W // 2 + 40, 840), fill=(230, 230, 235))
        draw.text((W // 2, 800), "✕", fill=(40, 40, 40), font=font(34), anchor="mm")
        draw.text((W // 2, 865), "全部清除", fill=(220, 220, 220), font=font(20), anchor="mm")
        self._hits.append(((W // 2 - 50, 750, W // 2 + 50, 880), self.clear_recents))

    def _settings(self, draw) -> None:
        top = 110 - self.state.settings_scroll
        for index, (name, _default) in enumerate(SETTINGS):
            y0 = top + index * 84
            if y0 + 84 < 110 or y0 > NAV_TOP:
                continue
            draw.rectangle((0, y0, W, y0 + 83), fill="white")
            draw.text((32, y0 + 42), name, fill=(25, 25, 25), font=font(24), anchor="lm")
            draw.line((32, y0 + 83, W, y0 + 83), fill=(232, 232, 232))
            if name in self.state.toggles:
                on = self.state.toggles[name]
                draw.rounded_rectangle((W - 104, y0 + 26, W - 36, y0 + 58), 16, fill=(30, 130, 240) if on else (200, 200, 205))
                knob = W - 54 if on else W - 86
                draw.ellipse((knob - 14, y0 + 28, knob + 14, y0 + 56), fill="white")
                self._hits.append(((W - 120, y0 + 10, W - 20, y0 + 74), lambda name=name: self.toggle(name)))
            else:
                draw.text((W - 40, y0 + 42), "›", fill=(170, 170, 170), font=font(34), anchor="mm")
        self._title(draw, "设置")

    def _notes(self, draw) -> None:
        self._title(draw, "备忘录")
        if not self.state.notes:
            draw.text((W // 2, 400), "还没有笔记", fill=(150, 150, 150), font=font(24), anchor="mm")
        for index, note in enumerate(self.state.notes):
            y0 = 130 + index * 96
            draw.rounded_rectangle((24, y0, W - 24, y0 + 84), 14, fill="white")
            draw.text((44, y0 + 42), note[:18], fill=(30, 30, 30), font=font(24), anchor="lm")
        draw.ellipse((W - 130, 780, W - 40, 870), fill=(235, 170, 40))
        draw.text((W - 85, 825), "+", fill="white", font=font(48), anchor="mm")
        self._hits.append(((W - 130, 780, W - 40, 870), self.new_note))

    def _editor(self, draw) -> None:
        self._title(draw, "新建笔记")
        draw.text((W - 50, 73), "保存", fill=(235, 150, 20), font=font(24), anchor="mm")
        self._hits.append(((W - 100, STATUS_H, W, 110), self.save_note))
        focused = self.state.focused == "draft"
        draw.rounded_rectangle((24, 130, W - 24, 520), 12, fill="white",
                               outline=(235, 170, 40) if focused else (220, 220, 220), width=2)
        text = self.state.draft
        if text:
            for line_no, line in enumerate(text.split("\n")[:8]):
                draw.text((44, 150 + line_no * 34), line, fill=(25, 25, 25), font=font(24))
        else:
            draw.text((44, 150), "输入笔记内容…", fill=(170, 170, 170), font=font(24))
        if focused:
            last = text.split("\n")[-1] if text else ""
            line_no = text.count("\n") if text else 0
            cx = 44 + int(draw.textlength(last, font=font(24)))
            draw.line((cx + 2, 152 + line_no * 34, cx + 2, 180 + line_no * 34), fill=(235, 150, 20), width=3)
        self._hits.append(((24, 130, W - 24, 520), lambda: self.focus("draft")))

    def _messages(self, draw) -> None:
        self._title(draw, "消息")
        for index, name in enumerate(CONTACTS):
            y0 = 120 + index * 96
            draw.rectangle((0, y0, W, y0 + 95), fill="white")
            draw.ellipse((24, y0 + 16, 88, y0 + 80), fill=(120, 150, 200))
            draw.text((56, y0 + 48), name[0], fill="white", font=font(28), anchor="mm")
            draw.text((108, y0 + 34), name, fill=(25, 25, 25), font=font(24), anchor="lm")
            last = (self.state.chats[name] or [""])[-1]
            draw.text((108, y0 + 66), last[:16], fill=(140, 140, 140), font=font(18), anchor="lm")
            self._hits.append(((0, y0, W, y0 + 95), lambda name=name: self.open_chat(name)))

    def _chat(self, draw, name: str) -> None:
        self._title(draw, name)
        bottom = KEYBOARD_TOP - 90 if self.state.focused else NAV_TOP - 90
        for index, message in enumerate(self.state.chats[name][-8:]):
            y0 = 130 + index * 64
            width = int(draw.textlength(message, font=font(22))) + 32
            draw.rounded_rectangle((W - 24 - width, y0, W - 24, y0 + 48), 14, fill=(150, 220, 120))
            draw.text((W - 24 - width + 16, y0 + 24), message, fill=(20, 20, 20), font=font(22), anchor="lm")
        draw.rectangle((0, bottom, W, bottom + 90), fill=(240, 240, 242))
        focused = self.state.focused == "chat"
        draw.rounded_rectangle((20, bottom + 18, W - 130, bottom + 72), 10, fill="white",
                               outline=(50, 170, 90) if focused else (215, 215, 215), width=2)
        draw.text((36, bottom + 45), self.state.chat_draft or "发消息…",
                  fill=(25, 25, 25) if self.state.chat_draft else (170, 170, 170), font=font(22), anchor="lm")
        if focused:
            cx = 36 + int(draw.textlength(self.state.chat_draft, font=font(22)))
            draw.line((cx + 2, bottom + 30, cx + 2, bottom + 60), fill=(50, 170, 90), width=3)
        draw.rounded_rectangle((W - 116, bottom + 18, W - 20, bottom + 72), 10, fill=(50, 170, 90))
        draw.text((W - 68, bottom + 45), "发送", fill="white", font=font(22), anchor="mm")
        self._hits.append(((20, bottom + 18, W - 130, bottom + 72), lambda: self.focus("chat")))
        self._hits.append(((W - 116, bottom + 18, W - 20, bottom + 72), lambda: self.send(name)))

    def _keyboard(self, draw) -> None:
        draw.rectangle((0, KEYBOARD_TOP, W, NAV_TOP), fill=(214, 217, 222))
        rows = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
        for r, keys in enumerate(rows):
            x0 = 8 + (W - 16 - len(keys) * 52) // 2
            for c, key in enumerate(keys):
                x = x0 + c * 52
                y = KEYBOARD_TOP + 20 + r * 70
                draw.rounded_rectangle((x + 3, y, x + 49, y + 58), 6, fill="white")
                draw.text((x + 26, y + 29), key, fill=(30, 30, 30), font=font(22), anchor="mm")
        draw.text((W // 2, NAV_TOP - 32), "ADB Keyboard", fill=(120, 120, 120), font=font(18), anchor="mm")
        # Taps on the keyboard area are swallowed (text goes through ADB Keyboard).
        self._hits.append(((0, KEYBOARD_TOP, W, NAV_TOP), lambda: None))

    # -- behaviour ----------------------------------------------------------------
    def _set_toast(self, text: str = "") -> None:
        self.state.toast = text

    def open_app(self, name: str) -> None:
        self.state.screen = f"app:{name}"
        self.state.focused = None
        if name in self.state.recent_apps:
            self.state.recent_apps.remove(name)
        self.state.recent_apps.append(name)

    def navigate(self, key: str) -> None:
        state = self.state
        if key == "home":
            state.screen, state.focused = "home", None
        elif key == "recents":
            state.screen, state.focused = "recents", None
        elif state.focused:
            state.focused = None
        elif state.screen == "editor":
            state.screen, state.draft = "app:备忘录", ""
        elif state.screen.startswith("chat:"):
            state.screen = "app:消息"
        elif state.screen != "home":
            state.screen = "home"

    def clear_recents(self) -> None:
        self.state.recent_apps.clear()
        self.state.screen = "home"

    def toggle(self, name: str) -> None:
        self.state.toggles[name] = not self.state.toggles[name]

    def new_note(self) -> None:
        self.state.screen, self.state.draft, self.state.focused = "editor", "", None

    def save_note(self) -> None:
        if self.state.draft.strip():
            self.state.notes.append(self.state.draft.strip())
            self._set_toast("已保存")
        self.state.screen, self.state.draft, self.state.focused = "app:备忘录", "", None

    def open_chat(self, name: str) -> None:
        self.state.screen, self.state.chat_draft, self.state.focused = f"chat:{name}", "", None

    def send(self, name: str) -> None:
        if self.state.chat_draft.strip():
            self.state.chats[name].append(self.state.chat_draft.strip())
            self.state.chat_draft = ""

    def focus(self, field_name: str) -> None:
        self.state.focused = field_name

    def tap(self, x: int, y: int) -> None:
        with self.lock:
            self.render()
            self._set_toast()
            for (x0, y0, x1, y1), callback in reversed(self._hits):
                if x0 <= x <= x1 and y0 <= y <= y1:
                    callback()
                    return

    def swipe(self, start: tuple[int, int], end: tuple[int, int]) -> None:
        with self.lock:
            self._set_toast()
            dy = end[1] - start[1]
            if start[1] > NAV_TOP - 20 and dy < -150:
                self.navigate("home")
            elif self.state.screen == "app:设置" and abs(dy) > abs(end[0] - start[0]):
                limit = max(0, len(SETTINGS) * 84 - (NAV_TOP - 110))
                self.state.settings_scroll = min(limit, max(0, self.state.settings_scroll - dy))

    def type_text(self, text: str) -> None:
        with self.lock:
            if self.state.focused == "draft":
                self.state.draft += text
            elif self.state.focused == "chat":
                self.state.chat_draft += text.replace("\n", "")

    def clear_text(self) -> None:
        with self.lock:
            if self.state.focused == "draft":
                self.state.draft = ""
            elif self.state.focused == "chat":
                self.state.chat_draft = ""


class MockDevice(Device):
    """Device backed by ``SimPhone``; text input and app launch always work."""

    driver = "mock"

    def __init__(self, config, adb_executable: str = "adb"):
        super().__init__(config, adb_executable)
        self.adb = None
        self.phone = SimPhone()
        self.executions: list[dict[str, Any]] = []

    def apps(self) -> list[str]:
        return [name for name, _ in APPS]

    def text_input_available(self) -> bool:
        return True

    def status(self) -> dict[str, Any]:
        return {**self.base_status(), "online": True, "camera": True, "error": None,
                "keyboard": {"ready": True, "reason": None}}

    def capture(self) -> Image.Image:
        return self.phone.render()

    def execute(self, action, frame_size):
        self.checkpoint()
        self.executions.append(action.to_dict())
        if action.type == "type":
            self.phone.type_text(action.text or "")
            return {"channel": "sim_keyboard"}
        if action.type == "clear":
            self.phone.clear_text()
            return {"channel": "sim_keyboard"}
        if action.type == "enter":
            self.phone.type_text("\n")
            return {"channel": "sim_keyboard"}
        if action.type == "launch":
            self.phone.open_app(action.app or "")
            return {"channel": "sim_launch"}
        return super().execute(action, frame_size)

    def _px(self, point, frame_size):
        return grid_to_pixel(point, (W, H))

    def tap(self, point, frame_size):
        self.phone.tap(*self._px(point, frame_size))
        return {"channel": "sim", "pixel": list(self._px(point, frame_size))}

    def double_tap(self, point, frame_size):
        return self.tap(point, frame_size)

    def long_press(self, point, frame_size):
        return self.tap(point, frame_size)

    def swipe(self, start, end, frame_size):
        self.phone.swipe(self._px(start, frame_size), self._px(end, frame_size))
        return {"channel": "sim"}

    def drag(self, start, end, frame_size):
        return self.swipe(start, end, frame_size)

    def navigate(self, key, frame_size):
        with self.phone.lock:
            self.phone.navigate(key)
        return {"channel": "sim", "key": key}
