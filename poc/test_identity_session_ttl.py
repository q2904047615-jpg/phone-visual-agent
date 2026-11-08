"""Login lifetime is a deployment setting: 1 hour by default, at most 92 days."""
from datetime import datetime, timezone
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import test_browser_identity as identity_fixture
from agent.infrastructure.sqlite_control_plane_repository import (
    IDENTITY_SESSION_MAX_TTL_SECONDS, IDENTITY_SESSION_TTL_ENV, SqliteControlPlaneRepository,
    configured_identity_session_ttl,
)
import web_app

THIRTY_DAYS = 30 * 86400
NINETY_DAYS = 90 * 86400


def remaining_seconds(session):
    return (datetime.fromisoformat(session['expires_at']) - datetime.now(timezone.utc)).total_seconds()


class ConfiguredTtlTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = SqliteControlPlaneRepository(Path(self.tmp.name) / 'runtime.sqlite3')
        self.addCleanup(self.repo.close)

    def issue(self, **kwargs):
        return self.repo.issue_identity_session(organization_id='local-org', user_id='local-owner', **kwargs)

    def test_unset_keeps_the_one_hour_default(self):
        with patch.dict('os.environ', {}):
            os.environ.pop(IDENTITY_SESSION_TTL_ENV, None)
            self.assertEqual(3600, configured_identity_session_ttl())
            self.assertAlmostEqual(3600, remaining_seconds(self.issue()), delta=5)

    def test_thirty_day_setting_applies_to_new_sessions(self):
        with patch.dict('os.environ', {IDENTITY_SESSION_TTL_ENV: str(THIRTY_DAYS)}):
            self.assertAlmostEqual(THIRTY_DAYS, remaining_seconds(self.issue()), delta=5)

    def test_ninety_day_setting_is_accepted(self):
        self.assertGreaterEqual(IDENTITY_SESSION_MAX_TTL_SECONDS, NINETY_DAYS)
        with patch.dict('os.environ', {IDENTITY_SESSION_TTL_ENV: str(NINETY_DAYS)}):
            self.assertEqual(NINETY_DAYS, configured_identity_session_ttl())
            self.assertAlmostEqual(NINETY_DAYS, remaining_seconds(self.issue()), delta=5)

    def test_explicit_ttl_still_wins(self):
        with patch.dict('os.environ', {IDENTITY_SESSION_TTL_ENV: str(THIRTY_DAYS)}):
            self.assertAlmostEqual(600, remaining_seconds(self.issue(ttl_seconds=600)), delta=5)

    def test_out_of_range_or_malformed_setting_is_rejected(self):
        for raw in ('abc', '59', '0', '-5', str(IDENTITY_SESSION_MAX_TTL_SECONDS + 1)):
            with self.subTest(raw=raw), patch.dict('os.environ', {IDENTITY_SESSION_TTL_ENV: raw}):
                with self.assertRaises(ValueError):
                    self.issue()
        with patch.dict('os.environ', {IDENTITY_SESSION_TTL_ENV: str(IDENTITY_SESSION_MAX_TTL_SECONDS)}):
            self.assertAlmostEqual(IDENTITY_SESSION_MAX_TTL_SECONDS, remaining_seconds(self.issue()), delta=5)
        with self.assertRaises(ValueError):
            self.issue(ttl_seconds=True)


class BrowserLoginLifetimeTests(unittest.TestCase):
    def setUp(self):
        identity_fixture.BrowserIdentityTests.setUp(self)

    def test_browser_cookie_and_session_follow_the_configured_lifetime(self):
        with patch.dict('os.environ', {IDENTITY_SESSION_TTL_ENV: str(NINETY_DAYS)}):
            state = identity_fixture.BrowserIdentityTests.start(self)
            response = identity_fixture.BrowserIdentityTests.complete(self, state)
        self.assertEqual(303, response.status_code)
        cookie = response.headers['set-cookie']
        max_age = int(cookie.split('Max-Age=')[1].split(';')[0])
        self.assertAlmostEqual(NINETY_DAYS, max_age, delta=5)
        self.assertIn('HttpOnly', cookie)
        status = identity_fixture.BrowserIdentityTests.status(self).json()['data']
        self.assertTrue(status['authenticated'])
        self.assertAlmostEqual(NINETY_DAYS, remaining_seconds(status['session']), delta=5)
        # Logout still ends a long session immediately.
        token = self.client.cookies.get(web_app._BROWSER_SESSION_COOKIE)
        self.client.post('/api/control-plane/auth/browser-logout', headers=self.headers)
        self.assertIsNone(self.repo.authenticate_identity_session(token))


if __name__ == '__main__':
    unittest.main()
