import json
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import httpx

from evals.production_backup import DATABASES, backup_set, restore_drill
from evals.production_read_probe import PublicReadTransport, probe


class ProductionBackupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.state = self.root / "runtime"
        self.state.mkdir()
        self.backups = self.root / "backups"
        for name, tables in DATABASES.items():
            db = sqlite3.connect(self.state / name)
            db.execute("PRAGMA journal_mode=WAL")
            for table in tables:
                db.execute(f'CREATE TABLE "{table}" (value TEXT)')
                db.execute(f'INSERT INTO "{table}" VALUES (?)', (name,))
            db.commit()
            self.addCleanup(db.close)

    def test_three_wal_stores_restore_without_live_replacement(self):
        result = backup_set(self.state, self.backups)
        self.assertEqual({p.name for p in Path(result["backup_set"]).iterdir()}, set(DATABASES) | {"manifest.json"})
        restored = self.root / "isolated-restore"
        report = restore_drill(result["backup_set"], restored)
        self.assertFalse(report["live_replaced"])
        self.assertEqual(set(report["integrity"].values()), {"ok"})
        for name, tables in DATABASES.items():
            with closing(sqlite3.connect(restored / name)) as db:
                self.assertEqual(db.execute(f'SELECT value FROM "{tables[0]}"').fetchone()[0], name)
        with self.assertRaises(FileExistsError):
            restore_drill(result["backup_set"], self.state)

    def test_missing_store_fails_before_partial_backup(self):
        with self.assertRaises(FileNotFoundError):
            backup_set(self.root / "wrong-old-release-path", self.backups)
        self.assertFalse(self.backups.exists())

    def test_retention_only_removes_completed_owned_sets(self):
        now = datetime.now(timezone.utc)
        old = backup_set(self.state, self.backups, now=now-timedelta(days=15))
        preserved = backup_set(self.state, self.backups, now=now-timedelta(days=16))
        # A user-added file means the directory is no longer exclusively ours.
        (Path(preserved["backup_set"]) / "user-note.txt").touch()
        legacy = self.backups / "runtime-legacy.sqlite3"
        legacy.write_text("retain")
        result = backup_set(self.state, self.backups, now=now)
        self.assertFalse(Path(old["backup_set"]).exists())
        self.assertTrue(Path(preserved["backup_set"]).exists())
        self.assertTrue(legacy.exists())
        self.assertEqual(len(result["removed_sets"]), 1)

    def test_live_database_directory_cannot_be_retention_target(self):
        with self.assertRaises(ValueError):
            backup_set(self.state, self.state / "backup")


class PublicProbeTests(unittest.TestCase):
    def test_transport_only_reads_verified_acceptance_routes(self):
        transport = PublicReadTransport("https://fixture.example")
        self.addCleanup(transport.close)
        for method, path in (("POST", "/api/session/reload"), ("GET", "/api/device"), ("GET", "/other")):
            with self.assertRaises(ValueError):
                transport.handle_request(httpx.Request(method, "http://127.0.0.1:8765"+path))
        with patch.object(transport.inner, "handle_request", return_value=httpx.Response(200)) as handle:
            transport.handle_request(httpx.Request("GET", "http://127.0.0.1:8765/openapi.json"))
            self.assertEqual(str(handle.call_args.args[0].url), "https://fixture.example/openapi.json")

    def test_https_origin_and_load_bounds(self):
        for origin in ("http://fixture.example", "https://user:pass@fixture.example", "https://fixture.example/path"):
            with self.assertRaises(ValueError):
                PublicReadTransport(origin)
        with self.assertRaises(ValueError):
            probe("https://fixture.example", clients=5)


if __name__ == "__main__":
    unittest.main()
