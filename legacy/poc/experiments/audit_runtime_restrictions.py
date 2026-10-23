"""Static, offline index of runtime restriction sites. Never imports project runtime.

The index is navigation/evidence, NOT an automatic necessity verdict. The root
necessity review contains the human rationale, exceptions and proposed disposition.
Run without arguments to verify the saved snapshot; --refresh explicitly replaces
only this experiment's two generated reports. No config, API or device access.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent / "runtime_restrictions"
REVIEW = ROOT / "运行限制必要性审查.md"

# These are review cross-references, not keyword-generated approvals. A source
# file can implement several obligations; read the exact condition and its scope.
FAMILIES = {
    "action_adapter": "R08 R10 R21",
    "qwen_visual_decision": "R04 R05 R06 R07 R08 R19",
    "runtime_session": "R08 R21 R27",
    "execution_budget": "R27",
    "recent_navigation": "R05 R08 R23 R27",
    "universal_agent_orchestrator": "R02 R03 R05 R08 R09 R19 R20 R21 R22 R23 R24 R27",
    "universal_agent_sessions": "R08 R09 R27",
    "vision_usage": "R28 R29",
    "action_capabilities": "R05 R30",
    "canonical_action_kinds": "R05",
    "canonical_action_protocol": "R04 R05 R06 R07 R11 R12 R13 R14 R15 R16 R18 R19 R25",
    "canonical_selection": "R05 R19",
    "confirmation_authority": "R08 R09 R20",
    "device_execution": "R10 R11 R15 R16 R17 R18 R21",
    "generic_goal": "R01 R02 R03 R04 R12",
    "qwen_task_context": "R02 R03 R04 R08 R09 R12 R20",
    "semantic_action": "R05 R06",
    "session_evidence": "R19 R21 R31",
    "session": "R08",
    "text_input_utils": "R12 R14 R16",
    "text_transport": "R08 R15 R21",
    "trusted_observation": "R08 R10",
    "ui_scene": "R04 R06 R07 R10 R11 R12 R13 R14 R16 R17",
    "universal_action_controller": "R06 R07 R10 R11 R12 R13 R14 R15 R16 R17 R18 R21 R23 R24 R25",
    "validation": "R04",
    "vision_model": "R11 R28",
    "visual_evidence": "R10 R24",
    "adb_keyboard_transport": "R08 R15 R21",
    "adb_package_launcher": "R18 R21 R25",
    "atomic_files": "R31",
    "camera_coordinator": "R08 R10",
    "capability_acceptance_runtime": "R08 R21 R30 R31",
    "capability_acceptance": "R30 R31",
    "dashscope_vision_provider": "R04 R28 R29",
    "device_controller_registry": "R08 R18 R30",
    "device_exclusivity": "R08",
    "device_executor": "R08 R10 R15 R17 R18 R21 R30",
    "device_task_registry": "R08 R21",
    "device_runtime_resources": "R08 R10 R15 R18 R28",
    "environment_vision_model_config": "R28",
    "generic_scene_observer": "R04 R05 R06 R07 R10 R11 R12 R13 R14 R16 R17 R19 R24 R28 R32",
    "generic_action_adapter": "R05 R08 R10 R11 R13 R14 R15 R16 R17 R18 R21 R24 R25",
    "file_system_evidence_store": "R31",
    "in_memory_session_repository": "R08 R31",
    "observation_images": "R10 R11 R24",
    "model_failure_diagnostics": "R31",
    "orientation_safety": "R08 R10 R11 R17",
    "qwen_runtime_errors": "R21 R31",
    "robot_controller": "R08 R10 R11 R14 R16 R17 R18 R21 R30",
    "runtime_doctor": "R08 R10 R15 R28 R30",
    "seller_window_adapter": "R08 R10 R11 R17 R18 R21",
    "tap_calibration": "R11 R17",
    "trusted_observation_frames": "R08 R10",
    "web_app": "R01 R08 R09 R18 R21 R26 R27 R28 R30 R31",
    "local_agent_api_client": "R26 R27 R31",
    "agent_api_cli": "R26 R27",
    "compact_scene": "R04 R07 R10 R11 R12 R16 R17 R32",
    "input_structure_audit": "R12 R13 R14 R16 R32",
    "single_step_observation": "R03 R04 R05 R06 R11 R12 R13 R19 R20 R22 R23 R32",
    "app": "R08 R09 R26 R27 R30",
    "protocol_adapter": "R04 R05 R08 R09 R26 R27 R30",
    "index": "R01 R26 R27",
    "action_acceptance": "R08 R26 R30",
    "touch_calibration": "R11 R17 R26 R30",
}
SCHEMA_KEYS = {"enum", "const", "required", "additionalProperties", "minimum", "maximum",
               "minItems", "maxItems", "minLength", "maxLength", "pattern"}
SITE_CALLS = {"reject_if", "Field", "ConfigDict", "Literal", "fullmatch", "match", "search",
              "compile", "min", "max", "_expect_keys", "_require_text", "_validate_id",
              "_validate_text_list", "_validate_id_list", "_reject_missing_refs"}


def files() -> list[Path]:
    candidates = list((ROOT / "poc/agent").rglob("*.py"))
    candidates += list((ROOT / "poc/agent").rglob("*.txt"))
    candidates += [ROOT / "poc" / name for name in
                   ("web_app.py", "local_agent_api_client.py", "agent_api_cli.py")]
    candidates += [p for p in (ROOT / "poc/static").iterdir() if p.suffix in {".js", ".html"}]
    return sorted(candidates)


def python_sites(source: str) -> list[dict]:
    tree = ast.parse(source)
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    def scope_of(node: ast.AST) -> str:
        scopes = []
        cursor = node
        while cursor in parents:
            cursor = parents[cursor]
            if isinstance(cursor, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                scopes.append(cursor.name)
        return ".".join(reversed(scopes)) or "<module>"

    result = []
    for node in ast.walk(tree):
        kind = ""
        selected = node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.args.defaults or any(value is not None for value in node.args.kw_defaults):
                result.append({"line": node.lineno, "kind": "parameter_defaults", "scope": scope_of(node),
                               "code": f"{node.name}({ast.unparse(node.args)})"})
        elif isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else (
                node.func.attr if isinstance(node.func, ast.Attribute) else "")
            if name in SITE_CALLS:
                kind = "hard_guard" if name == "reject_if" else "validation_or_limit_call"
        elif isinstance(node, (ast.Raise, ast.Assert)):
            kind = "raise" if isinstance(node, ast.Raise) else "assert"
        elif isinstance(node, (ast.If, ast.While, ast.IfExp)):
            kind, selected = "branch", node.test
        elif isinstance(node, ast.comprehension):
            for condition in node.ifs:
                result.append({"line": condition.lineno, "kind": "filter",
                               "scope": scope_of(condition), "code": ast.unparse(condition)})
        elif isinstance(node, ast.Return):
            # All returns, including nullable/False refusals and reason-string
            # predicates: no error-message keyword search determines coverage.
            kind = "return_or_refusal"
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id.isupper() for t in targets):
                kind = "constant_or_vocabulary"
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value in SCHEMA_KEYS:
                    result.append({"line": key.lineno, "kind": "schema", "scope": scope_of(key),
                                   "code": f"{key.value}: {ast.unparse(value)}"})
        if not kind:
            continue
        result.append({"line": node.lineno, "kind": kind,
                       "scope": scope_of(node),
                       "code": ast.unparse(selected)})
    return sorted(result, key=lambda item: (item["line"], item["kind"], item["code"]))


def build() -> dict:
    manifest, sites = [], []
    for path in files():
        rel = path.relative_to(ROOT).as_posix()
        source = path.read_text(encoding="utf-8-sig")
        families = FAMILIES.get(path.stem, "")
        if not families and path.stem != "__init__":
            raise ValueError(f"Unreviewed file added: {rel}")
        manifest.append({"path": rel, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "lines": len(source.splitlines()), "review_families": families.split()})
        entries = python_sites(source) if path.suffix == ".py" else [
            # Entire prompt and JS/HTML are indexed, not just 'forbidden' words.
            {"line": n, "kind": "prompt_clause" if path.suffix == ".txt" else "web_source",
             "scope": "<text>", "code": line.strip()}
            for n, line in enumerate(source.splitlines(), 1) if line.strip()]
        if entries and not families:
            # Package imports are not restrictions. Still require review if new
            # executable predicates appear in an __init__ module.
            raise ValueError(f"Unreviewed executable package initializer: {rel}")
        for entry in entries:
            entry.update(path=rel, review_families=families.split())
            identity = f"{rel}:{entry['line']}:{entry['kind']}:{entry['code']}"
            entry["site_id"] = hashlib.sha256(identity.encode()).hexdigest()[:16]
            sites.append(entry)
    return {"format": "runtime-restriction-index-v1", "scope": "static_source_not_live_acceptance",
            "coverage_note": "Source candidates, not unique restrictions or reachability proof. Family links require human review.",
            "files": manifest, "counts": dict(Counter(row["kind"] for row in sites)), "sites": sites}


def markdown(report: dict) -> str:
    lines = ["# 运行限制源码逐项索引", "", "此文件由离线静态脚本生成，不自动判断必要性。",
             "每项保留精确条件和所属函数；必要性、误拦影响、数值疑点与处理结论见根目录《运行限制必要性审查》。",
             "branch/return/filter 包含普通分支与撤销可选事实，不能把索引条数当成阻断规则条数。",
             "同族规则共享必要性说明；多个族表示需要结合函数和条件区分，绝非全部获批保留。", "",
             "[人工审查](../../../运行限制必要性审查.md)", ""]
    for item in report["files"]:
        lines += [f"## {item['path']}", "", f"源码 SHA256：`{item['sha256']}`",
                  f"审查族：{'、'.join(item['review_families']) or '包初始化，无限制候选'}", ""]
        for row in report["sites"]:
            if row["path"] != item["path"]:
                continue
            code = row["code"].replace("\n", " ").replace("`", "'")
            lines.append(f"- L{row['line']} · `{row['site_id']}` · {row['kind']} · `{row['scope']}`：`{code}`")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    report = build()
    review = REVIEW.read_text(encoding="utf-8-sig")
    ids = set(re.findall(r"^### (R\d+)\b", review, flags=re.M))
    required = {f for item in report["files"] for f in item["review_families"]}
    if required - ids:
        raise ValueError(f"Missing necessity sections: {sorted(required - ids)}")
    for link in re.findall(r"\]\(([^)]+)\)", review):
        if "://" not in link and not (ROOT / link.split("#", 1)[0]).is_file():
            raise ValueError(f"Missing review reference: {link}")
    for name in re.findall(r"`(test_[A-Za-z0-9_]+\.py)`", review):
        if not (ROOT / "poc" / name).is_file():
            raise ValueError(f"Missing referenced test: {name}")
    if f"{len(report['sites'])}个索引项" not in review:
        raise ValueError("Review inventory count has drifted")
    rendered_json = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    rendered_md = markdown(report)
    if args.refresh:
        OUTPUT.mkdir(exist_ok=True)
        (OUTPUT / "source_index.json").write_text(rendered_json, encoding="utf-8")
        (OUTPUT / "source_index.md").write_text(rendered_md, encoding="utf-8")
    else:
        for filename, expected in (("source_index.json", rendered_json), ("source_index.md", rendered_md)):
            if (OUTPUT / filename).read_text(encoding="utf-8") != expected:
                raise ValueError(f"Source audit drift: {filename}; re-review changes before --refresh")
    print(json.dumps({"files": len(report["files"]), "families": len(required),
                      "sites": len(report["sites"]), "counts": report["counts"],
                      "snapshot_matches": True, "network_calls": 0, "device_actions": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
