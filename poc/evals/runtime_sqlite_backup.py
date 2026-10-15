"""Safe online backup and restore helpers for the Runtime SQLite database.

The active database uses WAL mode, so copying only the ``.sqlite3`` file can
lose committed data that is still in ``-wal``.  This module uses SQLite's
online backup API to take a consistent snapshot without stopping the service.
Restore always writes a new destination first and replaces an existing target
only when the caller explicitly opts in.
"""

from __future__ import annotations

import argparse
from contextlib import closing
import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Iterable


def _read_only_uri(path: Path) -> str:
    return f"file:{path.resolve().as_posix()}?mode=ro"


def _integrity_check(connection: sqlite3.Connection) -> None:
    result = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
    if result.lower() != "ok":
        raise RuntimeError(f"sqlite_integrity_check_failed: {result}")


def verify_sqlite_database(path: Path, *, required_tables: Iterable[str] = ()) -> dict[str, object]:
    """Verify a database without changing it and return redacted metadata."""

    database = Path(path)
    if not database.is_file():
        raise FileNotFoundError(database)
    with closing(sqlite3.connect(_read_only_uri(database), uri=True)) as connection:
        _integrity_check(connection)
        tables = tuple(str(name) for name in required_tables)
        missing = [
            name for name in tables
            if connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
            ).fetchone() is None
        ]
        if missing:
            raise RuntimeError("sqlite_required_tables_missing: " + ",".join(missing))
        page_count = int(connection.execute("PRAGMA page_count").fetchone()[0])
        schema_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
    return {
        "path": str(database),
        "integrity": "ok",
        "page_count": page_count,
        "user_version": schema_version,
        "required_tables": list(tables),
    }


def _online_copy(source: Path, destination: Path) -> None:
    if source.resolve() == destination.resolve():
        raise ValueError("sqlite_backup_same_path")
    if not source.is_file():
        raise FileNotFoundError(source)
    sidecars = tuple(
        path for path in (
            Path(f"{destination}-wal"),
            Path(f"{destination}-shm"),
        ) if path.exists()
    )
    if sidecars:
        raise RuntimeError(
            "sqlite_destination_sidecars_present: "
            + ",".join(str(path) for path in sidecars)
            + "; stop writers and checkpoint before replacing this target"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with closing(sqlite3.connect(_read_only_uri(source), uri=True)) as source_connection:
            with closing(sqlite3.connect(temporary)) as destination_connection:
                source_connection.backup(destination_connection, pages=128, sleep=0.01)
                destination_connection.commit()
        # Validate before making the snapshot visible to a consumer.
        verify_sqlite_database(temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def backup_sqlite_database(
    source: Path, destination: Path, *, required_tables: Iterable[str] = ()
) -> dict[str, object]:
    """Create a consistent backup and atomically publish it at destination."""

    source_path = Path(source)
    destination_path = Path(destination)
    source_metadata = verify_sqlite_database(source_path, required_tables=required_tables)
    _online_copy(source_path, destination_path)
    backup_metadata = verify_sqlite_database(destination_path, required_tables=required_tables)
    return {
        "operation": "backup",
        "source": source_metadata,
        "backup": backup_metadata,
    }


def restore_sqlite_database(
    backup: Path,
    destination: Path,
    *,
    replace: bool = False,
    required_tables: Iterable[str] = (),
) -> dict[str, object]:
    """Restore a backup to a new path, requiring explicit replacement."""

    backup_path = Path(backup)
    destination_path = Path(destination)
    if destination_path.exists() and not replace:
        raise FileExistsError(
            f"restore_destination_exists: {destination_path}; pass replace=True explicitly"
        )
    source_metadata = verify_sqlite_database(backup_path, required_tables=required_tables)
    _online_copy(backup_path, destination_path)
    restored_metadata = verify_sqlite_database(destination_path, required_tables=required_tables)
    return {
        "operation": "restore",
        "backup": source_metadata,
        "restored": restored_metadata,
        "replaced": bool(replace),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SQLite WAL-safe backup/restore verifier.")
    subparsers = parser.add_subparsers(dest="operation", required=True)
    for operation in ("backup", "restore"):
        sub = subparsers.add_parser(operation)
        sub.add_argument("--source" if operation == "backup" else "--backup", required=True, type=Path)
        sub.add_argument("--destination", required=True, type=Path)
        sub.add_argument("--required-table", action="append", default=[])
        if operation == "restore":
            sub.add_argument("--replace", action="store_true", help="允许替换已有目标文件。")
    verify = subparsers.add_parser("verify")
    verify.add_argument("--database", required=True, type=Path)
    verify.add_argument("--required-table", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.operation == "backup":
            report = backup_sqlite_database(args.source, args.destination, required_tables=args.required_table)
        elif args.operation == "restore":
            report = restore_sqlite_database(
                args.backup, args.destination, replace=args.replace, required_tables=args.required_table
            )
        else:
            report = verify_sqlite_database(args.database, required_tables=args.required_table)
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(json.dumps({"status": "blocked", "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps({"status": "pass", **report}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
