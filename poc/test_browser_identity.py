"""Real local routes with a simulated GitHub upstream, no device actions."""
import json
import tempfile
import unittest
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient
import web_app
from agent.infrastructure.github_oauth_provider import GitHubOAuthProvider, GitHubIdentityMapping
from agent.infrastructure.sqlite_control_plane_repository import SqliteControlPlaneRepository


class BrowserIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = SqliteControlPlaneRepository(Path(self.tmp.name) / 'runtime.sqlite3')
        self.addCleanup(self.repo.close)
        self.exchange_count = 0
        def upstream(request):
            if request.url.host == 'github.com':
                self.exchange_count += 1
                return httpx.Response(200, json={'access_token': 'github-upstream-only'})
            return httpx.Response(200, json={'app': {'client_id': 'test-client'}, 'user': {'id': 42}})
        self.repo.import_github_mappings({'42': {'organization_id':'local-org', 'user_id':'local-owner'}})
        provider = GitHubOAuthProvider(client_id='test-client', client_secret='test-secret',
            mapping_resolver=self.repo.github_identity, transport=httpx.MockTransport(upstream))
        self.addCleanup(provider.close)
        registry = web_app.ExternalIdentityProviderRegistry()
        registry.register(provider)
        for p in (patch.object(web_app.runtime, 'control_plane_repository', self.repo),
                  patch.object(web_app, 'control_plane_identity_registry', registry),
                  patch.object(web_app, '_github_oauth_pending_verifiers', {}),
                  patch.dict('os.environ', {'ROBOT_WEB_NO_BROWSER': '1'})):
            p.start(); self.addCleanup(p.stop)
        self.client = TestClient(web_app.app, base_url='http://127.0.0.1:8765')
        self.addCleanup(self.client.close)
        self.headers = {'X-Control-Token': web_app.CONTROL_TOKEN, 'Origin': 'http://127.0.0.1:8765'}

    def start(self):
        r = self.client.get('/api/control-plane/auth/github/start?browser=true', headers=self.headers)
        self.assertEqual(200, r.status_code)
        self.assertIn('HttpOnly', r.headers['set-cookie'])
        return parse_qs(urlparse(r.json()['data']['authorization_url']).query)['state'][0]

    def complete(self, state):
        return self.client.get('/api/control-plane/auth/github/callback',
                               params={'state': state, 'code': 'test-code'}, follow_redirects=False)

    def status(self):
        return self.client.get('/api/control-plane/auth/browser-session', headers=self.headers)

    def test_login_refresh_logout_and_token_revocation(self):
        state = self.start()
        r = self.complete(state)
        self.assertEqual(303, r.status_code)
        self.assertEqual('/?login=success', r.headers['location'])
        self.assertNotIn('principal_token', r.text)
        self.assertNotIn('github-upstream-only', str(r.headers))
        self.assertIn('HttpOnly', r.headers['set-cookie'])
        self.assertIn('SameSite=lax', r.headers['set-cookie'])
        self.assertIn('no-store', r.headers['cache-control'])
        token = self.client.cookies.get(web_app._BROWSER_SESSION_COOKIE)
        for _ in range(2):
            status = self.status()
            self.assertTrue(status.json()['data']['authenticated'])
            self.assertEqual('local-owner', status.json()['data']['session']['user_id'])
            self.assertNotIn(token, status.text)
        for _ in range(2):
            r = self.client.post('/api/control-plane/auth/browser-logout', headers=self.headers)
            self.assertEqual(200, r.status_code)
        self.assertIsNone(self.repo.authenticate_identity_session(token))
        self.assertFalse(self.status().json()['data']['authenticated'])
        self.assertIsNone(self.client.cookies.get(web_app._BROWSER_SESSION_COOKIE))
        audit = self.repo.read_audit('local-org')['entries']
        self.assertTrue(any(e['outcome']=='revoked' for e in audit))

    def test_callback_cannot_transfer_to_another_browser_or_replay(self):
        state = self.start()
        with TestClient(web_app.app, base_url='http://127.0.0.1:8765') as other:
            r = other.get('/api/control-plane/auth/github/callback',
                          params={'state': state, 'code': 'test-code'}, follow_redirects=False)
            self.assertIn('oauth_browser_mismatch', r.headers['location'])
            self.assertIsNone(other.cookies.get(web_app._BROWSER_SESSION_COOKIE))
        self.assertEqual(0, self.exchange_count)
        self.assertEqual(303, self.complete(state).status_code)
        replay = self.complete(state)
        self.assertEqual(400, replay.status_code)
        self.assertEqual('oauth_state_invalid', replay.json()['detail']['code'])
        self.assertEqual(1, self.exchange_count)

    def test_expired_and_revoked_sessions_are_not_displayed_as_logged_in(self):
        for mode in ('expired', 'revoked'):
            self.complete(self.start())
            token = self.client.cookies.get(web_app._BROWSER_SESSION_COOKIE)
            session = self.repo.authenticate_identity_session(token)
            if mode == 'revoked':
                self.repo.revoke_identity_session(session['session_id'])
            else:
                self.repo._connection.execute('UPDATE control_plane_identity_sessions SET expires_at=? WHERE session_id=?',
                                              ('2000-01-01T00:00:00+00:00', session['session_id']))
            self.assertFalse(self.status().json()['data']['authenticated'])
            self.assertIsNone(self.client.cookies.get(web_app._BROWSER_SESSION_COOKIE))

    def test_logout_requires_control_token_and_local_origin(self):
        self.complete(self.start())
        self.assertEqual(403, self.client.post('/api/control-plane/auth/browser-logout').status_code)
        self.assertEqual(403, self.client.post('/api/control-plane/auth/browser-logout',
                         headers={**self.headers, 'Origin': 'https://evil.example'}).status_code)
        self.assertTrue(self.status().json()['data']['authenticated'])

    def test_configured_public_origin_is_allowed_but_other_origin_is_rejected(self):
        with patch.dict('os.environ', {'PUBLIC_BASE_URL': 'https://agent.jianlizhineng.cn'}):
            with TestClient(web_app.app, base_url='https://agent.jianlizhineng.cn') as public:
                headers = {
                    'X-Control-Token': web_app.CONTROL_TOKEN,
                    'Origin': 'https://agent.jianlizhineng.cn',
                }
                allowed = public.get('/api/control-plane/auth/github/start?browser=true', headers=headers)
                self.assertEqual(200, allowed.status_code, allowed.text)
                self.assertIn('Secure', allowed.headers['set-cookie'])
                denied = public.get('/api/control-plane/auth/github/start?browser=true', headers={
                    **headers, 'Origin': 'https://evil.example',
                })
                self.assertEqual(403, denied.status_code, denied.text)

    def test_denial_and_restarted_flow_return_clean_error_redirect(self):
        state = self.start()
        r = self.client.get('/api/control-plane/auth/github/callback',
            params={'state': state, 'error': 'access_denied', 'error_description': 'private-details'}, follow_redirects=False)
        self.assertEqual('/?login_error=oauth_authorization_denied', r.headers['location'])
        self.assertNotIn('private-details', str(r.headers))
        self.assertFalse(self.status().json()['data']['authenticated'])
        state = self.start()
        web_app._github_oauth_pending_verifiers.clear()
        self.assertEqual('/?login_error=oauth_state_invalid', self.complete(state).headers['location'])
        self.assertEqual(0, self.exchange_count)

    def test_relogin_replaces_only_current_browser_session(self):
        self.complete(self.start())
        old = self.client.cookies.get(web_app._BROWSER_SESSION_COOKIE)
        independent = self.repo.issue_identity_session(organization_id='local-org', user_id='local-owner')
        self.complete(self.start())
        self.assertIsNone(self.repo.authenticate_identity_session(old))
        self.assertIsNotNone(self.repo.authenticate_identity_session(independent['principal_token']))
        self.assertTrue(self.status().json()['data']['authenticated'])

    def test_unconfigured_provider_still_allows_status_and_logout(self):
        with patch.object(web_app, 'control_plane_identity_registry', web_app.ExternalIdentityProviderRegistry()):
            self.assertFalse(self.status().json()['data']['github_available'])
            self.assertEqual(200, self.client.post('/api/control-plane/auth/browser-logout', headers=self.headers).status_code)


if __name__ == '__main__':
    unittest.main()
