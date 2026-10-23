"""Back up the three production stores; restore drills never replace live files.

Each SQLite snapshot is consistent on its own. An online set is NOT a
cross-database transaction; recovery must still reconcile unknown dispatches.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil

try:
    from .runtime_sqlite_backup import backup_sqlite_database, restore_sqlite_database
except ImportError:
    from runtime_sqlite_backup import backup_sqlite_database, restore_sqlite_database

DATABASES = {
    "runtime.sqlite3": ("runtime_sessions", "control_plane_meta", "control_plane_platform_releases"),
    "dispatch.sqlite3": ("dispatch_tickets",),
    "connector.sqlite3": ("connector_commands_v2", "connector_nonces"),
}
MANIFEST_VERSION = "phone-agent-backup-set-v1"


def backup_set(state_dir, backup_dir, *, now=None, retention_days=14):
    state, parent = Path(state_dir).resolve(), Path(backup_dir).resolve()
    if state == parent or state in parent.parents:
        raise ValueError("Backup directory must be outside live database directory")
    if not 1 <= retention_days <= 365:
        raise ValueError("Invalid retention")
    clock = now or datetime.now(timezone.utc)
    target = parent / clock.strftime("production-%Y%m%dT%H%M%S%fZ")
    # Validate every source before creating a partial set.
    for name in DATABASES:
        if not (state / name).is_file():
            raise FileNotFoundError(name)
    target.mkdir(parents=True, exist_ok=False, mode=0o700)
    report = {"protocol": MANIFEST_VERSION, "created_utc": clock.isoformat(),
              "cross_database_atomic": False, "databases": {}}
    for name, tables in DATABASES.items():
        metadata = backup_sqlite_database(state / name, target / name, required_tables=tables)
        (target / name).chmod(0o600)
        report["databases"][name] = metadata["backup"]
    (target / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    # Retire only completed sets created by this script; leave legacy backups,
    # failed partial sets and all unrecognised content untouched.
    removed = []
    for candidate in parent.iterdir():
        if candidate == target or candidate.is_symlink() or not candidate.is_dir():
            continue
        if not re.fullmatch(r"production-[0-9]{8}T[0-9]{12}Z", candidate.name):
            continue
        manifest = candidate / "manifest.json"
        if not manifest.is_file() or manifest.is_symlink():
            continue
        try:
            old = json.loads(manifest.read_text(encoding="utf-8"))
            created = datetime.fromisoformat(old["created_utc"])
            expired = (clock - created).total_seconds() > retention_days * 86400
            expected = set(DATABASES) | {"manifest.json"}
            allowed = expected | {name+suffix for name in DATABASES for suffix in ("-wal", "-shm")}
            present = set(p.name for p in candidate.iterdir())
            owned = expected <= present <= allowed
            plain = all(p.is_file() and not p.is_symlink() for p in candidate.iterdir())
        except (ValueError, KeyError, OSError, TypeError):
            continue
        if expired and owned and plain and old.get("protocol") == MANIFEST_VERSION and set(old.get("databases", {})) == set(DATABASES):
            if candidate.resolve().parent != parent:
                raise ValueError("Backup retention escaped parent")
            shutil.rmtree(candidate)
            removed.append(candidate.name)
    return {"status": "passed", "backup_set": str(target), "removed_sets": removed,
            "databases": list(DATABASES), "cross_database_atomic": False}


def restore_drill(backup, destination):
    source, target = Path(backup).resolve(), Path(destination).resolve()
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("protocol") != MANIFEST_VERSION or set(manifest.get("databases", {})) != set(DATABASES):
        raise ValueError("Incomplete backup set")
    target.mkdir(parents=True, exist_ok=False, mode=0o700)
    results = {}
    for name, tables in DATABASES.items():
        results[name] = restore_sqlite_database(source / name, target / name, required_tables=tables)["restored"]["integrity"]
        (target / name).chmod(0o600)
    return {"status": "passed", "destination": str(target), "integrity": results,
            "live_replaced": False, "business_recovery_verified": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="operation", required=True)
    backup = sub.add_parser("backup")
    backup.add_argument("--state-dir", type=Path, required=True)
    backup.add_argument("--backup-dir", type=Path, required=True)
    restore = sub.add_parser("restore-drill")
    restore.add_argument("--backup", type=Path, required=True)
    restore.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    result = backup_set(args.state_dir, args.backup_dir) if args.operation == "backup" else restore_drill(args.backup, args.destination)
    print(json.dumps(result))
