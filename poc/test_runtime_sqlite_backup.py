from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from evals.runtime_sqlite_backup import (
    backup_sqlite_database,
    restore_sqlite_database,
    verify_sqlite_database,
)


class RuntimeSqliteBackupTests(unittest.TestCase):
    def test_online_backup_captures_wal_and_restore_requires_explicit_replace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "runtime.sqlite3"
            backup = root / "backup.sqlite3"
            restored = root / "restored.sqlite3"
            with closing(sqlite3.connect(source)) as connection:
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute("CREATE TABLE events(id INTEGER PRIMARY KEY, value TEXT NOT NULL)")
                connection.execute("INSERT INTO events(value) VALUES ('committed-in-wal')")
                connection.commit()

            report = backup_sqlite_database(source, backup, required_tables=("events",))
            self.assertEqual("ok", report["backup"]["integrity"])
            with closing(sqlite3.connect(backup)) as connection:
                self.assertEqual("committed-in-wal", connection.execute("SELECT value FROM events").fetchone()[0])

            restored_report = restore_sqlite_database(backup, restored, required_tables=("events",))
            self.assertFalse(restored_report["replaced"])
            with self.assertRaises(FileExistsError):
                restore_sqlite_database(backup, restored, required_tables=("events",))
            self.assertEqual("ok", verify_sqlite_database(restored, required_tables=("events",))["integrity"])

    def test_restore_replace_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.sqlite3"
            backup = root / "backup.sqlite3"
            target = root / "target.sqlite3"
            for path, value in ((source, "source"), (target, "old")):
                with closing(sqlite3.connect(path)) as connection:
                    connection.execute("CREATE TABLE state(value TEXT NOT NULL)")
                    connection.execute("INSERT INTO state VALUES (?)", (value,))
                    connection.commit()
            backup_sqlite_database(source, backup, required_tables=("state",))
            report = restore_sqlite_database(backup, target, replace=True, required_tables=("state",))
            self.assertTrue(report["replaced"])
            with closing(sqlite3.connect(target)) as connection:
                self.assertEqual("source", connection.execute("SELECT value FROM state").fetchone()[0])

    def test_restore_rejects_target_with_wal_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.sqlite3"
            backup = root / "backup.sqlite3"
            target = root / "target.sqlite3"
            with closing(sqlite3.connect(source)) as connection:
                connection.execute("CREATE TABLE state(value TEXT NOT NULL)")
                connection.execute("INSERT INTO state VALUES ('source')")
                connection.commit()
            backup_sqlite_database(source, backup, required_tables=("state",))
            target.write_bytes(b"existing")
            Path(f"{target}-wal").write_bytes(b"stale-wal")
            with self.assertRaisesRegex(RuntimeError, "sqlite_destination_sidecars_present"):
                restore_sqlite_database(backup, target, replace=True, required_tables=("state",))


if __name__ == "__main__":
    unittest.main()
