const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');
const root = path.join(__dirname, 'static');
const signedIn = (user = 'alice', permissions = ['task.create', 'device.execute']) => ({
  authenticated: true, github_available: true, session: { user_id: user, organization_id: 'test-org',
    role: permissions.includes('task.create') ? 'operator' : 'viewer', permissions, expires_at: '2030-01-01T00:00:00Z' },
});

async function fixture(t, initial) {
  const browser = await chromium.launch({ headless: true });
  t.after(() => browser.close());
  const context = await browser.newContext();
  const state = { account: initial, requests: [], delayNextDevice: false, delayed: null };
  await context.route('**/*', async route => {
    const pathname = new URL(route.request().url()).pathname;
    state.requests.push(pathname);
    if (pathname === '/api/session') return route.fulfill({ json: { token: 'fixture-token', mock: true, reload_supported: false } });
    if (pathname.endsWith('/browser-session')) return route.fulfill({ json: { data: state.account } });
    if (pathname.endsWith('/browser-logout')) {
      state.account = { authenticated: false, github_available: true, session: null };
      return route.fulfill({ json: { data: { authenticated: false } } });
    }
    if (pathname === '/api/device') {
      if (state.delayNextDevice) { state.delayNextDevice = false; await new Promise(resolve => { state.delayed = resolve; }); }
      return route.fulfill({ json: { devices: state.account.session?.user_id === 'bob' ? [] : [{ device_id: 'fixture-phone' }],
        default_device_id: 'fixture-phone', active_tasks: [], manual_motion_tests: [], controller_online: true, camera_online: true } }).catch(() => {});
    }
    if (pathname === '/api/preview.jpg') return route.fulfill({ contentType: 'image/png', body: Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=', 'base64') });
    if (pathname.startsWith('/api/')) return route.fulfill({ json: { monitors: [], trials: [] } });
    const file = pathname === '/' ? 'index.html' : path.basename(pathname);
    return route.fulfill({ body: fs.readFileSync(path.join(root, file)), contentType:
      file.endsWith('.html') ? 'text/html' : file.endsWith('.css') ? 'text/css' : 'text/javascript' });
  });
  return { context, page: await context.newPage(), state };
}

test('logged-out page clears old private storage and never requests devices or pictures', async t => {
  const { page, state } = await fixture(t, { authenticated: false, github_available: true, session: null });
  await page.addInitScript(() => sessionStorage.setItem('visual-agent-task-outcome:old', 'private-from-old-user'));
  await page.goto('http://website.test/');
  await page.getByText('未登录', { exact: true }).waitFor();
  assert.equal(await page.locator('#top').isVisible(), false);
  assert.equal(await page.locator('#accessNotice').isVisible(), true);
  assert.equal(await page.evaluate(() => sessionStorage.length), 0);
  assert.equal(state.requests.includes('/api/device'), false);
  assert.equal(state.requests.includes('/api/preview.jpg'), false);
  assert.equal(await page.locator('#githubLoginButton').isEnabled(), true);
});

test('logout clears private content across tabs; stale requests and next user cannot restore it', async t => {
  const { context, page, state } = await fixture(t, signedIn());
  await page.goto('http://website.test/');
  await page.locator('#top:not([hidden])').waitFor();
  const other = await context.newPage();
  await other.goto('http://website.test/');
  await other.locator('#top:not([hidden])').waitFor();
  for (const tab of [page, other]) await tab.evaluate(() => {
    state.chatConversation = [{ role: 'user', content: 'alice-private-content' }];
    sessionStorage.setItem('visual-agent-test-org%2Falice:task-outcome:fixture-phone', 'alice-private-content');
    render();
  });
  state.delayNextDevice = true;
  const pending = page.evaluate(() => refreshDevice()).catch(() => {});
  while (!state.delayed) await new Promise(resolve => setTimeout(resolve, 10));
  await page.locator('#accountLogoutButton').click();
  state.delayed(); await pending;
  for (const tab of [page, other]) {
    await tab.getByText('未登录', { exact: true }).waitFor();
    assert.equal(await tab.locator('#top').isVisible(), false);
    assert.equal(await tab.locator('body').innerText().then(text => text.includes('alice-private-content')), false);
    assert.equal(await tab.evaluate(() => Object.keys(sessionStorage).some(key => key.startsWith('visual-agent-'))), false);
  }
  state.account = signedIn('bob');
  await page.reload();
  await page.getByText('已登录 · bob', { exact: true }).waitFor();
  await page.getByText('尚未分配设备，请联系管理员', { exact: true }).waitFor();
  assert.equal(await page.locator('#chatMessages').innerText().then(text => text.includes('alice-private-content')), false);
  assert.equal(await page.locator('#startSupervisedAgent').isEnabled(), false);
});

test('viewer gets a read-only workbench; expiry removes private view', async t => {
  const { page, state } = await fixture(t, signedIn('viewer', ['task.read', 'device.read']));
  await page.goto('http://website.test/');
  await page.getByText('当前账号为只读权限', { exact: true }).waitFor();
  assert.equal(await page.locator('#startSupervisedAgent').isEnabled(), false);
  state.account = { authenticated: false, github_available: true, session: null };
  await page.evaluate(() => window.dispatchEvent(new Event('focus')));
  await page.getByText('未登录', { exact: true }).waitFor();
  assert.equal(await page.locator('#top').isVisible(), false);
});
