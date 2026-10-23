"""Optional, app-specific add-ons. They sit beside the generic agent and share its devices.

Each plugin module exposes ``TITLE`` and ``create(context) -> (APIRouter, state)``.
A plugin that fails to load (for example a missing optional dependency) is
reported in ``/api/status`` instead of stopping the console.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI

PLUGINS = {"lucky_bag": "pva.plugins.lucky_bag", "qishui": "pva.plugins.qishui"}


@dataclass
class PluginContext:
    devices: Any
    tasks: Any
    output_root: Path


def load_plugins(app: FastAPI, context: PluginContext) -> dict[str, dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = {}
    for key, module_name in PLUGINS.items():
        try:
            module = importlib.import_module(module_name)
            router, state = module.create(context)
            app.include_router(router)
            loaded[key] = {"title": module.TITLE, "ok": True, "error": None, "state": state}
        except Exception as exc:  # keep the console usable without optional extras
            loaded[key] = {"title": key, "ok": False, "error": f"{type(exc).__name__}: {exc}", "state": None}
    return loaded
