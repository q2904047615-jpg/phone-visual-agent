const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const {chromium} = require('playwright');

test('native lucky-bag pause / resume / stop and terminal feedback use feature routes', async () => {
  const staticRoot=path.join(__dirname,'static');
  const server=http.createServer((req,res)=> {
    const file=req.url.split('?')[0].startsWith('/assets/') ? path.join(staticRoot,req.url.split('?')[0].slice(8)) : path.join(staticRoot,'index.html');
    const extension=path.extname(file);
    res.setHeader('content-type',extension==='.js'?'application/javascript':extension==='.css'?'text/css':'text/html');
    res.end(fs.readFileSync(file));
  });
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const browser=await chromium.launch({headless:true,...(process.platform==='win32'?{channel:'msedge'}:{})});
  const page=await browser.newPage();
  const requests=[],errors=[];
  page.on('pageerror',error=>errors.push(String(error)));
  let monitor={monitor_id:'native',status:'running',session_id:'',current_phase:'已参与，等待开奖',detail:'已确认参与',physical_actions:3,observations:4};
  await page.route('**/api/**',async route=> {
    const request=route.request();
    const url=new URL(request.url()).pathname;
    requests.push({url,method:request.method()});
    let payload={};
    if(url==='/api/session') payload={token:'test',ready:true,mock:true,version:'test'};
    else if(url==='/api/device') payload={controller_online:true,camera_online:true,busy:false,default_device_id:'device-local-01',generic_supervised_execution:{active_sessions:[]}};
    else if(url==='/api/features/lucky-bag/monitors') payload={monitors:[]};
    else if(url.startsWith('/api/features/lucky-bag/native')) {
      if(request.method()==='POST') monitor.status=url.endsWith('/pause')?'paused':url.endsWith('/resume')?'running':'cancelled';
      payload={monitor:{...monitor},profile:{device_id:'device-local-01'}};
    }
    await route.fulfill({status:200,contentType:'application/json',body:JSON.stringify(payload)});
  });
  try {
    await page.goto('http://127.0.0.1:'+server.address().port);
    await page.waitForFunction(()=>typeof render==='function');
    await page.evaluate(()=> {
      state.luckyBagMonitorId='native'; state.luckyBagProfile={device_id:'device-local-01'};
      renderLuckyBagMonitorStatus({status:'running',current_phase:'已参与，等待开奖',detail:'已确认参与',physical_actions:3,observations:4});
      render();
    });
    await page.locator('#pauseButton').click();
    await page.waitForFunction(()=>state.luckyBagMonitorStatus==='paused');
    assert.ok(requests.some(r=>r.url.endsWith('/native/pause')&&r.method==='POST'));
    await page.locator('#continueTaskButton').click();
    await page.waitForFunction(()=>state.luckyBagMonitorStatus==='running');
    await page.locator('#stopButton').click();
    await page.waitForFunction(()=>state.luckyBagMonitorStatus==='cancelled');
    assert.ok(requests.some(r=>r.url.endsWith('/native/cancel')&&r.method==='POST'));
    assert.ok(!requests.some(r=>r.url==='/api/stop'));
    monitor.status='failed';monitor.detail='邮件发送失败，手机保持停手';
    await page.evaluate(()=>pollLuckyBagMonitor());
    assert.match(await page.locator('#taskRunStatusDetail').innerText(),/邮件发送失败/);
    assert.deepEqual(errors,[]);
  } finally {
    await browser.close();
    await new Promise(resolve=>server.close(resolve));
  }
});
