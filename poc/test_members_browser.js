const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');

async function fixture(t, actor = 'owner') {
  const browser = await chromium.launch({headless:true}); t.after(()=>browser.close());
  const page = await browser.newPage({viewport:{width:390,height:844}});
  const model = {writes:[], fail:'', signedIn:true, hold:null, members:[
    {user_id:'owner',role:'owner',status:'active',revision:0},
    {user_id:'alice',role:'operator',status:'active',revision:0}
  ]};
  const editable = role => actor==='owner'?['viewer','operator','admin','owner']:actor==='admin' && (!role || ['viewer','operator'].includes(role))?['viewer','operator']:[];
  await page.route('http://127.0.0.1:8765/**',async route=>{
    const url = new URL(route.request().url());
    if(url.pathname==='/api/session') return route.fulfill({json:{token:'fixture',mock:true}});
    if(url.pathname.endsWith('/browser-session')) return route.fulfill({json:{data:{authenticated:model.signedIn,github_available:true,
      session:model.signedIn ? {organization_id:'org',user_id:actor,role:actor,permissions:['owner','admin'].includes(actor)?['task.create','identity.manage']:['task.read'],expires_at:'2030-01-01T00:00:00Z'}:null}}});
    if(url.pathname.endsWith('/browser-logout')) {model.signedIn=false;return route.fulfill({json:{data:{authenticated:false}}});}
    if(url.pathname.includes('/principals')) {
      if(route.request().method()==='GET') {
        if(model.hold) await model.hold;
        return route.fulfill({json:{data:{create_roles:editable(),principals:model.members.map(m=>({...m,editable_roles:editable(m.role)}))}}});
      }
      const body=route.request().postDataJSON();
      model.writes.push({body,key:route.request().headers()['idempotency-key']});
      if(model.fail) return route.fulfill({status:409,json:{detail:{code:model.fail,message:'成员有未结束任务'}}});
      if(route.request().method()==='POST') model.members.push({...body,status:'active',revision:0});
      else Object.assign(model.members.find(m=>m.user_id===url.pathname.split('/').at(-1)),body,{revision:body.expected_revision+1});
      return route.fulfill({json:{data:{}}});
    }
    if(url.pathname==='/api/device') return route.fulfill({json:{devices:[],default_device_id:'',active_tasks:[],manual_motion_tests:[]}});
    if(url.pathname.startsWith('/api/')) return route.fulfill({json:{trials:[],monitors:[]}});
    const file=url.pathname==='/'?'index.html':path.basename(url.pathname);
    return route.fulfill({body:fs.readFileSync(path.join(__dirname,'static',file)),contentType:file.endsWith('.html')?'text/html':file.endsWith('.css')?'text/css':'text/javascript'});
  });
  await page.goto('http://127.0.0.1:8765/');
  await page.getByText(`已登录 · ${actor}`,{exact:true}).waitFor();
  return {page,model};
}

test('owner creates, changes role, disables and restores in narrow layout without a phone',async t=>{
  const {page,model}=await fixture(t);
  await page.locator('#memberManagementButton').click();
  await page.getByLabel('新成员 ID',{exact:true}).fill('new-user');
  await page.getByLabel('新成员角色',{exact:true}).selectOption('viewer');
  await page.getByRole('button',{name:'创建成员',exact:true}).click();
  await page.getByText('成员已创建。请从“管理 GitHub 关联”关联身份，并分配可用设备。',{exact:true}).waitFor();
  const row=page.locator('[data-member="new-user"]');
  for(const [role,status] of [['operator','active'],['operator','disabled'],['viewer','active']]) {
    await row.getByLabel('new-user 角色',{exact:true}).selectOption(role);
    await row.getByLabel('new-user 状态',{exact:true}).selectOption(status);
    await row.getByRole('button',{name:'保存成员'}).click();
    await page.getByText('成员已保存。停用会撤销全部登录；恢复后需重新登录。',{exact:true}).waitFor();
    assert.equal(model.members.at(-1).status,status);
  }
  assert.deepEqual(model.writes.slice(1).map(w=>w.body.expected_revision),[0,1,2]);
  assert.equal(await page.locator('#memberManagementDialog').evaluate(el=>el.scrollWidth<=el.clientWidth),true);
  assert.ok(model.writes.every(w=>w.key.length>10));
  await page.locator('#memberClose').click();
  await page.locator('#accountLogoutButton').click();
  await page.getByText('未登录',{exact:true}).waitFor();
  assert.equal(await page.locator('#memberManagementButton').isVisible(),false);
  assert.equal(await page.locator('#memberList').innerText(),'');
});

test('admin sees server-provided roles, blocked update retains data and retry identity',async t=>{
  const {page,model}=await fixture(t,'admin');
  await page.locator('#memberManagementButton').click();
  await page.getByText('此角色由所有者管理。',{exact:true}).waitFor();
  assert.deepEqual(await page.locator('#memberNewRole option').evaluateAll(items=>items.map(o=>o.value)),['viewer','operator']);
  assert.equal(await page.locator('[data-member="owner"] button').count(),0);
  const row=page.locator('[data-member="alice"]');
  await row.getByLabel('alice 状态').selectOption('disabled');
  model.fail='member_has_active_work';
  await row.getByRole('button',{name:'保存成员'}).click();
  await page.getByText('未保存：成员有未结束任务',{exact:true}).waitFor();
  assert.equal(model.members[1].status,'active'); model.fail='';
  await row.getByRole('button',{name:'保存成员'}).click();
  await page.getByText('成员已保存。停用会撤销全部登录；恢复后需重新登录。',{exact:true}).waitFor();
  assert.equal(model.writes[0].key,model.writes[1].key);
  const other=await fixture(t,'viewer');
  assert.equal(await other.page.locator('#memberManagementButton').isVisible(),false);
});

test('late member response cannot restore private content after logout',async t=>{
  const {page,model}=await fixture(t);
  let release; model.hold=new Promise(resolve=>release=resolve);
  await page.locator('#memberManagementButton').click();
  await page.getByText('正在读取组织成员…',{exact:true}).waitFor();
  await page.locator('#memberClose').click();
  await page.locator('#accountLogoutButton').click();
  await page.getByText('未登录',{exact:true}).waitFor(); release();
  await page.waitForTimeout(150);
  assert.equal(await page.locator('#memberList').innerText(),'');
  assert.equal(await page.locator('#memberManagementButton').isVisible(),false);
});
