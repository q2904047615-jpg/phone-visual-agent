const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');

async function fixture(t, actor='owner') {
  const browser=await chromium.launch({headless:true}); t.after(()=>browser.close());
  const page=await browser.newPage({viewport:{width:390,height:844}});
  page.setDefaultTimeout(5000);
  page.on('pageerror', error=>console.error(error.message));
  const model={writes:[],fail:'',signedIn:true,hold:null,teams:[],members:[
    {user_id:'owner',role:'owner',status:'active'}, {user_id:'alice',role:'viewer',status:'disabled'}]};
  await page.route('http://127.0.0.1:8765/**',async route=>{
    const url=new URL(route.request().url());
    if(url.pathname==='/api/session') return route.fulfill({json:{token:'fixture',mock:true}});
    if(url.pathname.endsWith('/browser-session')) return route.fulfill({json:{data:{authenticated:model.signedIn,github_available:true,
      session:model.signedIn?{organization_id:'org',user_id:actor,role:actor,permissions:['owner','admin'].includes(actor)?['task.create','team.manage','identity.manage']:['task.read'],expires_at:'2030-01-01T00:00:00Z'}:null}}});
    if(url.pathname.endsWith('/browser-logout')) {model.signedIn=false;return route.fulfill({json:{data:{authenticated:false}}});}
    if(url.pathname.endsWith('/principals')) return route.fulfill({json:{data:{principals:model.members}}});
    if(url.pathname.startsWith('/api/control-plane/organizations/org/teams')) {
      if(route.request().method()==='GET') {if(model.hold) await model.hold; return route.fulfill({json:{data:{teams:model.teams}}});}
      const body=route.request().postDataJSON(); model.writes.push({body,key:route.request().headers()['idempotency-key']});
      if(model.fail) return route.fulfill({status:409,json:{detail:{code:model.fail,message:'团队已变化'}}});
      if(route.request().method()==='POST') model.teams.push({...body,revision:0});
      else Object.assign(model.teams.find(t=>t.team_id===url.pathname.split('/').at(-1)),body,{revision:body.expected_revision+1});
      return route.fulfill({json:{data:{}}});
    }
    if(url.pathname==='/api/device') return route.fulfill({json:{devices:[],default_device_id:'',active_tasks:[],manual_motion_tests:[]}});
    if(url.pathname.startsWith('/api/')) return route.fulfill({json:{trials:[],monitors:[]}});
    const file=url.pathname==='/'?'index.html':path.basename(url.pathname);
    return route.fulfill({body:fs.readFileSync(path.join(__dirname,'static',file)),contentType:file.endsWith('.html')?'text/html':file.endsWith('.css')?'text/css':'text/javascript'});
  });
  await page.goto('http://127.0.0.1:8765/'); await page.getByText(`已登录 · ${actor}`,{exact:true}).waitFor();
  return {page,model};
}

test('teams create, rename, multi-group, remove membership, search and narrow layout without phone',async t=>{
  const {page,model}=await fixture(t);
  await page.locator('#teamManagementButton').click();
  await page.locator('#teamNewName').fill('运营');
  await page.locator('#teamNewMembers input[value="owner"]').check();
  await page.getByRole('button',{name:'创建团队',exact:true}).click();
  await page.getByText('团队已创建。可继续调整名称和成员。',{exact:true}).waitFor();
  const id=model.teams[0].team_id;
  const row=page.locator(`[data-team="${id}"]`);
  await row.getByLabel(`${id} 名称`).fill('客户运营');
  await row.locator('input[value="alice"]').check();
  await row.getByRole('button',{name:'保存团队'}).click();
  await page.getByText('团队已保存，成员分组已更新。',{exact:true}).waitFor();
  assert.deepEqual(model.teams[0].member_ids,['owner','alice']);
  await page.locator('#teamNewName').fill('支持');
  await page.locator('#teamNewMembers input[value="alice"]').check();
  await page.getByRole('button',{name:'创建团队',exact:true}).click();
  await page.getByText('团队已创建。可继续调整名称和成员。',{exact:true}).waitFor();
  await page.locator('#teamManagementDialog summary').click();
  await page.getByText('alice：客户运营、支持',{exact:true}).waitFor();
  await row.locator('input[value="alice"]').uncheck();
  await row.getByRole('button',{name:'保存团队'}).click();
  await page.getByText('alice：支持',{exact:true}).waitFor();
  await page.locator('#teamFilter').fill('alice');
  assert.equal(await row.isVisible(),false);
  assert.equal(await page.locator('#teamManagementDialog').evaluate(el=>el.scrollWidth<=el.clientWidth),true);
  assert.deepEqual(model.writes.filter(w=>'expected_revision' in w.body).map(w=>w.body.expected_revision),[0,1]);
  await page.locator('#teamClose').click(); await page.locator('#accountLogoutButton').click();
  await page.getByText('未登录',{exact:true}).waitFor();
  assert.equal(await page.locator('#teamList').innerText(),'');
  assert.equal(await page.locator('#teamMemberSummary').innerText(),'');
});

test('admin failure retries same key, stale editor refreshes and regular user has no management entry',async t=>{
  const {page,model}=await fixture(t,'admin');
  await page.locator('#teamManagementButton').click(); await page.locator('#teamNewName').fill('Support');
  model.fail='temporary';
  await page.getByRole('button',{name:'创建团队',exact:true}).click(); await page.getByText('未创建：团队已变化',{exact:true}).waitFor();
  model.fail=''; await page.getByRole('button',{name:'创建团队',exact:true}).click();
  await page.getByText('团队已创建。可继续调整名称和成员。',{exact:true}).waitFor();
  assert.deepEqual(model.writes[0],model.writes[1]);
  model.teams[0].revision=3; model.teams[0].name='Other editor'; model.fail='team_conflict';
  await page.getByRole('button',{name:'保存团队',exact:true}).click();
  await page.getByText('未保存：团队已变化',{exact:true}).waitFor();
  assert.equal(await page.locator('#teamList input').first().inputValue(),'Other editor');
  model.fail=''; await page.getByRole('button',{name:'保存团队',exact:true}).click();
  await page.getByText('团队已保存，成员分组已更新。',{exact:true}).waitFor();
  assert.equal(model.writes.at(-1).body.expected_revision,3);
  const viewer=await fixture(t,'viewer'); assert.equal(await viewer.page.locator('#teamManagementButton').isVisible(),false);
});

test('late team read cannot repopulate private data after logout',async t=>{
  const {page,model}=await fixture(t); let release;
  model.hold=new Promise(resolve=>release=resolve);
  await page.locator('#teamManagementButton').click(); await page.getByText('正在读取团队…',{exact:true}).waitFor();
  await page.locator('#teamClose').click(); await page.locator('#accountLogoutButton').click();
  await page.getByText('未登录',{exact:true}).waitFor(); release(); await page.waitForTimeout(150);
  assert.equal(await page.locator('#teamList').innerText(),'');
  assert.equal(await page.locator('#teamMemberSummary').innerText(),'');
  assert.equal(await page.locator('#teamNewMembers').innerText(),'');
});
