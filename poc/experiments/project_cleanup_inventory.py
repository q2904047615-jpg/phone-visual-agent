"""Read source metadata only; classify cleanup candidates without deleting anything."""
from __future__ import annotations

import ast
from collections import Counter
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'poc/output/project_cleanup_inventory_current'


def classify(name: str) -> tuple[str, str]:
    path = Path(name)
    if name.startswith('.task-backups/'):
        return 'history_snapshot', '历史工作树快照；保留用于回滚核对，不作为当前源码解析'
    if name.startswith('android/companion-ime/'):
        return 'remove', '用户已明确退役的Companion Android工程；保留恢复压缩包，不留运行入口'
    if name.startswith(('.agents/', '.codex/')) or path.name in {
        'AGENTS.md', '项目最终目标.md', '用户决策与协议边界.md', '项目交接文档.md',
        '单一权威最小校验验收台账.md'}:
        return 'keep', '项目权威/工具配置，不因清理删除'
    if path.name in {'device_registry.json', 'controller_config.json', 'tap_calibration.json',
        'adb_keyboard_registry.json', 'app_package_registry.json'} or path.suffix in {'.lease', '.lock'}:
        return 'protect', '本地设备配置、标定或运行状态；只记录路径，不读取内容'
    if 'output' in path.parts or name.startswith(('docs/', '安装图/')):
        return 'keep_evidence', '历史证据/设计/安装资料，不等于运行代码；当前不删除原始证据'
    if path.name.startswith('test_') or 'test_fixtures' in path.parts or 'frontend_contract_fixtures' in path.parts:
        return 'review_with_slice', '回归/负样本随对应能力核对，不能按未被生产import删除'
    if path.stem in {'verified_text_transaction', 'windows_ocr_runtime', 'windows_ocr'}:
        return 'retired', '机械文字整链已退役；当前跟踪路径可能尚待提交删除，恢复材料独立保留'
    if path.stem in {'generic_scene_observer', 'generic_action_adapter', 'universal_action_controller',
        'canonical_action_protocol', 'device_execution', 'device_executor', 'robot_controller',
        'orientation_safety', 'text_input_utils'}:
        return 'keep_review_functions', '机械文字已删除后的通用执行能力；不得整文件删除，剩余函数仍按真实调用审查'
    if name.startswith('poc/agent/'):
        return 'keep_review_functions', '正式DDD主链/端口/装配；保留模块，逐函数检查旧分支'
    if name.startswith('poc/experiments/'):
        return 'review_tool', '隔离探针/索引工具，按当前引用和历史复现价值核对'
    if path.suffix == '.md':
        return 'review_document', '核对当前说明与历史证据身份，过期说明不能继续指导运行'
    return 'review_entry', '网页/CLI/维护/构建等独立入口；不能只按Python import判死代码'


def main() -> None:
    result = subprocess.run(['git', '-c', 'core.quotepath=false', 'ls-files', '-z',
        '--cached', '--others', '--exclude-standard'], cwd=ROOT, check=True, capture_output=True)
    paths = sorted(set(result.stdout.decode('utf-8').strip('\0').split('\0')))
    rows = []
    for name in paths:
        path = ROOT / name
        category, reason = classify(name)
        present = path.is_file()
        row = {'path': name, 'category': category, 'reason': reason,
            'status': 'present' if present else 'missing_from_worktree'}
        if present and path.suffix == '.py' and category not in {'protect', 'history_snapshot'}:
            tree = ast.parse(path.read_text(encoding='utf-8-sig'))
            row['imports'] = sorted(set(
                node.module or '' for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)))
            row['dynamic_dispatch'] = any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in {'getattr', 'setattr', '__import__'} for node in ast.walk(tree))
        rows.append(row)
    report = {'scope': 'Git tracked and nonignored project files; generated dependencies/caches excluded',
        'not_proof_of_dead_code': True, 'counts': dict(Counter(row['category'] for row in rows)), 'files': rows}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / 'inventory.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# 全项目清理盘点', '', '这是逐路径初筛，不是全部人工审查、自动删除许可或全项目无死代码证明。',
        '范围为Git已跟踪与未忽略文件，不包括被忽略的依赖、缓存和输出；缺失的跟踪路径仍列出，不等于已验证退役。', '',
        '| 路径 | 分类 | 工作树状态 | 依据/下一步 |', '| --- | --- | --- | --- |']
    lines += [f"| {r['path']} | {r['category']} | {r['status']} | {r['reason']} |" for r in rows]
    (OUTPUT / 'inventory.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'files': len(rows), 'counts': report['counts']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
