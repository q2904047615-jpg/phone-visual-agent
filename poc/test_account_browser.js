const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');
const root = path.join(__dirname, 'static');

async function fixture(t, initial = { authenticated: false, github_available: true, session: null }) {
  const browser = await chromium.launch({ headless: true });
  t.after(() => browser.close());
  const page = await browser.newPage();
  const state = { value: initial, failStatus: false, failLogout: false, starts: 0, logouts: 0 };
  await page.route('**/*', async route => {
    const u = new URL(route.request().url());
    if (u.hostname === 'github.com') return route.fulfill({ body: '<h1>GitHub authorization fixture</h1>' });
    if (u.pathname.endsWith('browser-session')) {
      return route.fulfill({ status: state.failStatus ? 503 : 200, json: { data: state.value } });
    }
    if (u.pathname.endsWith('browser-logout')) {
      assert.equal(route.request().method(), 'POST'); state.logouts++;
      if (state.failLogout) return route.fulfill({ status: 503, json: {} });
      state.value = { authenticated: false, github_available: true, session: null };
      return route.fulfill({ json: { data: state.value } });
    }
    if (u.pathname.endsWith('github/start')) {
      assert.equal(u.searchParams.get('browser'), 'true'); state.starts++;
      return route.fulfill({ json: { data: { authorization_url: 'https://github.com/login/oauth/authorize?client_id=fixture' } } });
    }
    if (u.pathname === '/') {
      const html = fs.readFileSync(path.join(root, 'index.html'), 'utf8')
        .replace(/<script src="\/assets\/app.js[^>]+><\/script>/, `<script>
          AgentAccount.mount(async (url, options) => {
            const r = await fetch(url, options); if(!r.ok) throw new Error('fixture unavailable'); return r.json();
          });</script>`);
      return route.fulfill({ contentType: 'text/html', body: html });
    }
    const file = path.join(root, path.basename(u.pathname));
    if (fs.existsSync(file)) return route.fulfill({ body: fs.readFileSync(file), contentType: file.endsWith('.css') ? 'text/css' : 'text/javascript' });
    return route.fulfill({ status: 404, body: '' });
  });
  return { page, state };
}
const signedIn = () => ({ authenticated: true, github_available: true,
  session: { user_id: 'local-owner', organization_id: 'local-org', role: 'owner', expires_at: '2030-01-01T00:00:00Z' } });

test('account entry works on desktop/mobile and starts browser-bound GitHub login', async t => {
  const { page, state } = await fixture(t);
  await page.goto('http://account.test/');
  await page.locator('#accountStatus').getByText('未登录', { exact: true }).waitFor();
  for (const width of [1280, 390]) {
    await page.setViewportSize({ width, height: 844 });
    const box = await page.locator('#githubLoginButton').boundingBox();
    assert.ok(box && box.x >= 0 && box.x + box.width <= width);
    assert.equal(await page.locator('#githubLoginButton').isEnabled(), true);
  }
  await page.locator('#githubLoginButton').click();
  await page.waitForURL('https://github.com/**');
  assert.equal(state.starts, 1);
});

test('login persists on reload; logout failure is recoverable; logout and expiry update UI', async t => {
  const { page, state } = await fixture(t, signedIn());
  await page.goto('http://account.test/?login=success');
  await page.locator('#accountStatus').getByText('已登录 · local-owner', { exact: true }).waitFor();
  assert.equal(new URL(page.url()).search, '');
  await page.reload();
  await page.locator('#accountStatus').getByText('已登录 · local-owner', { exact: true }).waitFor();
  state.failLogout = true;
  await page.locator('#accountLogoutButton').click();
  await page.locator('#accountError').getByText(/退出失败/).waitFor();
  assert.equal(await page.locator('#accountLogoutButton').isVisible(), true);
  state.failLogout = false;
  await page.locator('#accountLogoutButton').click();
  await page.locator('#accountStatus').getByText('未登录', { exact: true }).waitFor();
  state.value = signedIn(); await page.reload();
  await page.locator('#accountStatus').getByText('已登录 · local-owner', { exact: true }).waitFor();
  state.value = { authenticated: false, github_available: true, session: null };
  await page.evaluate(() => window.dispatchEvent(new Event('focus')));
  await page.locator('#accountStatus').getByText('未登录', { exact: true }).waitFor();
  assert.equal(await page.evaluate(() => localStorage.length + sessionStorage.length), 0);
});

test('authorization denial, unconfigured provider and status outage show actionable messages', async t => {
  const { page, state } = await fixture(t, { authenticated: false, github_available: false, session: null });
  await page.goto('http://account.test/?login_error=oauth_authorization_denied');
  await page.locator('#accountError').getByText(/已取消 GitHub 授权/).waitFor();
  assert.equal(await page.locator('#githubLoginButton').isDisabled(), true);
  state.failStatus = true; await page.reload();
  await page.locator('#accountRetryButton').waitFor();
  state.failStatus = false; state.value.github_available = true;
  await page.locator('#accountRetryButton').click();
  await page.waitForFunction(() => !document.querySelector('#githubLoginButton').disabled);
});
