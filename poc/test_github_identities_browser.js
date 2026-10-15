const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const {chromium}=require('playwright');

async function fixture(t,role='owner'){
  const browser=await chromium.launch({headless:true});t.after(()=>browser.close());
  const page=await browser.newPage({viewport:{width:390,height:844}});
  const model={signedIn:true,fail:'',writes:[],hold:null,identities:[{subject:'42',user_id:'owner',login:'owner-gh',role:'owner',status:'active',revision:0}]};
  await page.route('http://127.0.0.1:8765/**',async route=>{
    const url=new URL(route.request().url()),method=route.request().method();
    if(url.pathname==='/api/session')return route.fulfill({json:{token:'fixture',mock:true}});
    if(url.pathname.endsWith('/browser-session'))return route.fulfill({json:{data:{authenticated:model.signedIn,github_available:true,session:model.signedIn?{
      user_id:role,organization_id:'org',role,permissions:['owner','admin'].includes(role)?['identity.manage']:['task.read'],expires_at:'2030-01-01T00:00:00Z'}:null}}});
    if(url.pathname.endsWith('/browser-logout')){model.signedIn=false;return route.fulfill({json:{data:{authenticated:false}}});}
    if(url.pathname.endsWith('/principals'))return route.fulfill({json:{data:{principals:[
      {user_id:'owner',role:'owner',status:'active',editable_roles:role==='owner'?['owner']:[]},
      {user_id:'alice',role:'viewer',status:'active',editable_roles:['viewer','operator']}]}}});
    if(url.pathname.endsWith('/github-identities'))return route.fulfill({json:{data:{github_available:true,identities:model.identities.map(i=>({...i,manageable:role==='owner'||i.role==='viewer'}))}}});
    if(url.pathname.endsWith('/lookup')){
      if(model.hold)await model.hold;
      return route.fulfill({json:{data:{profile:{subject:'43',login:'alice-gh',profile_url:'https://github.com/alice-gh'},revision:0,linked:false,user_id:null,external_binding:false}}});
    }
    if(url.pathname.includes('/github-identities/')&&method==='PUT'){
      const body=route.request().postDataJSON(),subject=url.pathname.split('/').at(-1);
      model.writes.push({body,subject,key:route.request().headers()['idempotency-key']});
      if(model.fail)return route.fulfill({status:409,json:{detail:{code:model.fail,message:'组织必须保留可登录的所有者'}}});
      if(body.user_id)model.identities.push({subject,user_id:body.user_id,login:'alice-gh',role:'viewer',status:'active',revision:1});
      else model.identities=model.identities.filter(i=>i.subject!==subject);
      return route.fulfill({json:{data:{}}});
    }
    if(url.pathname==='/api/device')return route.fulfill({json:{devices:[],active_tasks:[],manual_motion_tests:[]}});
    if(url.pathname.startsWith('/api/'))return route.fulfill({json:{trials:[],monitors:[]}});
    const file=url.pathname==='/'?'index.html':path.basename(url.pathname);
    return route.fulfill({body:fs.readFileSync(path.join(__dirname,'static',file)),contentType:file.endsWith('.html')?'text/html':file.endsWith('.css')?'text/css':'text/javascript'});
  });
  await page.goto('http://127.0.0.1:8765/');await page.getByText(`已登录 · ${role}`,{exact:true}).waitFor();
  return {page,model};
}

test('query facts, explicitly link and unlink, narrow layout and logout remain usable',async t=>{
  const {page,model}=await fixture(t);
  await page.locator('#githubIdentityButton').click();
  await page.getByLabel('GitHub 账号',{exact:true}).fill('alice-gh');
  await page.getByRole('button',{name:'查询账号',exact:true}).click();
  await page.getByText('alice-gh · 稳定 ID 43',{exact:true}).waitFor();
  assert.equal(model.writes.length,0);
  assert.equal(await page.getByRole('button',{name:'确认关联',exact:true}).isEnabled(),false);
  await page.getByLabel('关联到成员',{exact:true}).selectOption('alice');
  await page.getByRole('button',{name:'确认关联',exact:true}).click();
  await page.getByText('GitHub 身份已关联。该账号可通过 GitHub 登录对应成员，停用成员需先恢复。',{exact:true}).waitFor();
  const row=page.locator('[data-github-subject="43"]');
  await row.getByRole('button',{name:'解除关联',exact:true}).click();
  assert.equal(model.writes.length,1);
  await row.getByRole('button',{name:'取消解除',exact:true}).click();
  assert.equal(model.writes.length,1);
  await row.getByRole('button',{name:'解除关联',exact:true}).click();
  await row.getByRole('button',{name:'确认解除并撤销登录',exact:true}).click();
  await page.getByText('关联已解除，该成员原有登录已撤销。历史数据和设备归属保留。',{exact:true}).waitFor();
  assert.deepEqual(model.writes.map(w=>w.body),[{user_id:'alice',expected_revision:0},{user_id:null,expected_revision:1}]);
  assert.ok(model.writes.every(w=>w.key.length>10));
  assert.equal(await page.locator('#githubIdentityDialog').evaluate(el=>el.scrollWidth<=el.clientWidth),true);
  await page.locator('#githubIdentityClose').click();await page.locator('#accountLogoutButton').click();
  await page.getByText('未登录',{exact:true}).waitFor();
  assert.equal(await page.locator('#githubIdentityList').innerText(),'');
  assert.equal(await page.locator('#githubIdentityButton').isVisible(),false);
});

test('failed unlink preserves association and retry key; admin cannot manage owner',async t=>{
  const {page,model}=await fixture(t);
  await page.locator('#githubIdentityButton').click();
  const row=page.locator('[data-github-subject="42"]');
  await row.getByRole('button',{name:'解除关联',exact:true}).click();model.fail='last_linked_owner_required';
  await row.getByRole('button',{name:'确认解除并撤销登录',exact:true}).click();
  await page.getByText('未解除：组织必须保留可登录的所有者',{exact:true}).waitFor();
  await row.getByRole('button',{name:'确认解除并撤销登录',exact:true}).click();
  await page.getByText('未解除：组织必须保留可登录的所有者',{exact:true}).waitFor();
  assert.equal(model.writes[0].key,model.writes[1].key);assert.equal(model.identities.length,1);
  const admin=await fixture(t,'admin');await admin.page.locator('#githubIdentityButton').click();
  await admin.page.getByText('GitHub · owner-gh',{exact:true}).waitFor();
  assert.equal(await admin.page.locator('[data-github-subject="42"] button').count(),0);
  const viewer=await fixture(t,'viewer');assert.equal(await viewer.page.locator('#githubIdentityButton').isVisible(),false);
});

test('changed query and logout discard stale account lookup results',async t=>{
  const {page,model}=await fixture(t);let release;model.hold=new Promise(r=>release=r);
  await page.locator('#githubIdentityButton').click();await page.getByLabel('GitHub 账号',{exact:true}).fill('alice-gh');
  await page.getByRole('button',{name:'查询账号',exact:true}).click();
  await page.getByText('正在查询 GitHub 公开账号…',{exact:true}).waitFor();
  await page.getByLabel('GitHub 账号',{exact:true}).fill('different');release();await page.waitForTimeout(150);
  assert.equal(await page.locator('#githubIdentityResult').isVisible(),false);
  model.hold=new Promise(r=>release=r);
  await page.getByRole('button',{name:'查询账号',exact:true}).click();
  await page.getByText('正在查询 GitHub 公开账号…',{exact:true}).waitFor();
  await page.locator('#githubIdentityClose').click();await page.locator('#accountLogoutButton').click();
  await page.getByText('未登录',{exact:true}).waitFor();release();await page.waitForTimeout(150);
  assert.equal(await page.locator('#githubIdentityProfile').innerText(),'');
  assert.equal(await page.locator('#githubIdentityButton').isVisible(),false);
});
