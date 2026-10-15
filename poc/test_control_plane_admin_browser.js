const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');

async function fixture(t) {
  const browser = await chromium.launch({ headless: true });
  t.after(() => browser.close());
  const page = await browser.newPage({ viewport: { width: 390, height: 844 } });
  page.setDefaultTimeout(5000);
  const model = {
    signedIn: true,
    quotas: [{ quota_key: 'daily', period: 'daily', max_runs: 100, max_physical_actions: 1000, max_model_tokens: 2000000, usage: { runs: 2, physical_actions: 4, model_tokens: 120 } }],
    audit: [{ sequence: 12, occurred_at: '2030-01-01T00:00:00+08:00', request_id: 'r12', organization_id: 'org', resource: 'project.update', outcome: 'updated', actor: 'owner', metadata: { revision: 2 } }],
    notifications: [{ notification_id: 'n1', title: 'project.update · updated', body: '组织资源状态已更新，可在对应管理页面查看详情。', created_at: '2030-01-01T00:00:00+08:00', read_at: null }],
    writes: [],
  };
  await page.route('http://127.0.0.1:8765/**', async route => {
    const url = new URL(route.request().url());
    if (url.pathname === '/api/session') return route.fulfill({ json: { token: 'fixture', mock: true } });
    if (url.pathname.endsWith('/browser-session')) return route.fulfill({ json: { data: { authenticated: model.signedIn, github_available: true, session: model.signedIn ? { organization_id: 'org', user_id: 'owner', role: 'owner', permissions: ['task.read', 'quota.read', 'quota.manage', 'audit.read'], expires_at: '2030-01-01T00:00:00Z' } : null } } });
    if (url.pathname.startsWith('/api/') && url.pathname.includes('/quotas')) {
      if (route.request().method() === 'GET') return route.fulfill({ json: { data: { organization_id: 'org', quotas: model.quotas } } });
      const body = route.request().postDataJSON(); model.writes.push(body);
      model.quotas[0] = { ...model.quotas[0], ...body };
      return route.fulfill({ json: { data: { quota: model.quotas[0] } } });
    }
    if (url.pathname.startsWith('/api/') && url.pathname.endsWith('/audit')) return route.fulfill({ json: { data: { organization_id: 'org', entries: model.audit, has_more: false, after_sequence: 0 } } });
    if (url.pathname.startsWith('/api/') && url.pathname.includes('/notifications')) {
      if (url.pathname.endsWith('/read')) { model.notifications[0].read_at = '2030-01-01T00:01:00+08:00'; return route.fulfill({ json: { data: { read: true } } }); }
      return route.fulfill({ json: { data: { organization_id: 'org', notifications: model.notifications } } });
    }
    if (url.pathname === '/api/device') return route.fulfill({ json: { devices: [], default_device_id: '', active_tasks: [], manual_motion_tests: [] } });
    if (url.pathname.startsWith('/api/')) return route.fulfill({ json: { trials: [], monitors: [] } });
    const file = url.pathname === '/' ? 'index.html' : path.basename(url.pathname);
    return route.fulfill({ body: fs.readFileSync(path.join(__dirname, 'static', file)), contentType: file.endsWith('.html') ? 'text/html' : file.endsWith('.css') ? 'text/css' : 'text/javascript' });
  });
  await page.goto('http://127.0.0.1:8765/');
  await page.getByText('已登录 · owner', { exact: true }).waitFor();
  // Keep the fixture explicit: this also catches a missing module/script tag.
  assert.equal(await page.evaluate(() => Boolean(window.OrganizationQuotas)), true);
  return { page, model };
}

test('owner reads and updates quotas and reads organization audit', async t => {
  const { page, model } = await fixture(t);
  await page.locator('#quotaManagementButton').click();
  await page.getByText('共 1 项配额。', { exact: true }).waitFor();
  const row = page.locator('[data-quota="daily"]');
  await row.getByLabel('daily 运行上限').fill('101');
  await row.getByRole('button', { name: '保存配额' }).click();
  await page.getByText('配额 daily 已保存。', { exact: true }).waitFor();
  assert.equal(model.writes[0].max_runs, 101);
  await page.locator('#quotaClose').click();
  await page.locator('#auditViewerButton').click();
  await page.getByText('#12 · project.update · updated', { exact: true }).waitFor();
  await page.locator('#auditFilter').fill('project.update');
  assert.equal(await page.locator('.audit-entry').count(), 1);
  assert.equal(await page.locator('#auditViewerDialog').evaluate(el => el.scrollWidth <= el.clientWidth), true);
  await page.locator('#auditClose').click();
  await page.locator('#notificationButton').click();
  await page.getByText('共 1 条通知。', { exact: true }).waitFor();
  await page.getByRole('button', { name: '标为已读' }).click();
  await page.getByRole('button', { name: '已读' }).waitFor();
  assert.equal(model.notifications[0].read_at !== null, true);
});
