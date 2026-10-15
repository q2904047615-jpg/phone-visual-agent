const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');

async function fixture(t, admin = true) {
  const browser = await chromium.launch({headless:true}); t.after(()=>browser.close());
  const page = await browser.newPage({viewport:{width:390,height:844}});
  const model = {owner:'owner', revision:0, writes:[], fail:'', signedIn:true};
  await page.route('http://127.0.0.1:8765/**',async route=>{
    const url = new URL(route.request().url());
    if(url.pathname==='/api/session') return route.fulfill({json:{token:'fixture',mock:true}});
    if(url.pathname.endsWith('/browser-session')) return route.fulfill({json:{data:{authenticated:model.signedIn,github_available:true,
      session:model.signedIn ? {organization_id:'org',user_id:'owner',role:admin?'owner':'operator',permissions:admin?['task.create','device.execute','device.manage']:['task.create','device.execute'],expires_at:'2030-01-01T00:00:00Z'}:null}}});
    if(url.pathname.endsWith('/browser-logout')) {model.signedIn=false;return route.fulfill({json:{data:{authenticated:false}}});}
    if(url.pathname==='/api/control-plane/organizations/org/devices') return route.fulfill({json:{data:{devices:[{device_id:'phone',user_id:model.owner,revision:model.revision,registered:true}]}}});
    if(url.pathname.endsWith('/principals')) return route.fulfill({json:{data:{principals:[{user_id:'owner',role:'owner'},{user_id:'alice',role:'operator'}]}}});
    if(url.pathname.endsWith('/assignment')) {
      model.writes.push({body:route.request().postDataJSON(),key:route.request().headers()['idempotency-key']});
      if(model.fail) return route.fulfill({status:409,json:{detail:{code:model.fail,message:'设备有未结束任务'}}});
      model.owner=model.writes.at(-1).body.user_id;model.revision++;
      return route.fulfill({json:{data:{assignment:{device_id:'phone',user_id:model.owner,revision:model.revision}}}});
    }
    if(url.pathname==='/api/device') return route.fulfill({json:{devices:model.owner==='owner'?[{device_id:'phone'}]:[],default_device_id:'phone',active_tasks:[],manual_motion_tests:[]}});
    if(url.pathname==='/api/preview.jpg') return route.fulfill({status:204});
    if(url.pathname.startsWith('/api/')) return route.fulfill({json:{trials:[],monitors:[]}});
    const file=url.pathname==='/'?'index.html':path.basename(url.pathname);
    return route.fulfill({body:fs.readFileSync(path.join(__dirname,'static',file)),contentType:file.endsWith('.html')?'text/html':file.endsWith('.css')?'text/css':'text/javascript'});
  });
  await page.goto('http://127.0.0.1:8765/');
  await page.getByText('已登录 · owner',{exact:true}).waitFor();
  return {page,model};
}

test('admin can unassign and restore without a phone; narrow layout and logout stay private',async t=>{
  const {page,model}=await fixture(t);
  await page.locator('#deviceManagementButton').click();
  const select=page.getByRole('combobox',{name:'phone 使用者'});
  await select.selectOption('');
  await page.getByRole('button',{name:'保存分配',exact:true}).click();
  await page.getByText('分配已保存。历史任务仍归原用户。',{exact:true}).waitFor();
  assert.equal(model.owner,null);
  assert.equal(await page.locator('#phonePreview').getAttribute('src'),null);
  assert.equal(await page.locator('#startSupervisedAgent').isEnabled(),false);
  await select.selectOption('owner');
  await page.getByRole('button',{name:'保存分配',exact:true}).click();
  await page.getByText('当前使用者：owner',{exact:true}).waitFor();
  assert.equal(await page.locator('#startSupervisedAgent').isEnabled(),true);
  assert.deepEqual(model.writes.map(w=>w.body.expected_revision),[0,1]);
  assert.ok(model.writes.every(w=>w.key.length>10));
  assert.equal(await page.locator('#deviceManagementDialog').evaluate(el=>el.scrollWidth<=el.clientWidth),true);
  await page.getByRole('button',{name:'关闭',exact:true}).click();
  await page.locator('#accountLogoutButton').click();
  await page.getByText('未登录',{exact:true}).waitFor();
  assert.equal(await page.locator('#deviceManagementButton').isVisible(),false);
  assert.equal(await page.locator('#deviceAssignmentList').innerText(),'');
});

test('failed save keeps assignment and reuses idempotency key; operator has no management entry',async t=>{
  const {page,model}=await fixture(t);
  await page.locator('#deviceManagementButton').click();
  await page.getByRole('combobox',{name:'phone 使用者'}).selectOption('alice');
  model.fail='device_has_active_work';
  await page.getByRole('button',{name:'保存分配',exact:true}).click();
  await page.getByText('未保存：设备有未结束任务',{exact:true}).waitFor();
  assert.equal(model.owner,'owner');
  model.fail='';
  await page.getByRole('button',{name:'保存分配',exact:true}).click();
  await page.getByText('当前使用者：alice',{exact:true}).waitFor();
  assert.equal(model.writes[0].key,model.writes[1].key);
  const other=await fixture(t,false);
  assert.equal(await other.page.locator('#deviceManagementButton').isVisible(),false);
});
