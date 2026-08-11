const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const http = require("node:http");
const path = require("node:path");
const { chromium } = require("playwright");

const staticRoot = path.join(__dirname, "static");
const requests = { start: [], confirm: [], next: [], auto: [], cancel: [], stop: [] };

function contractSession(status = "awaiting_confirmation", accountEffectPossible = false) {
  return {
    session_id: "session-browser-contract",
    goal: { objective: "完成一个未预设的通用手机目标" },
    task_graph: {
      task_id: "task-browser-contract",
      device_id: "device-local-01",
      status,
      constraints: ["不得越过用户确认", "一次只执行一个物理动作"],
      completion_conditions: {
        visible_state: "结果标记可见",
        evidence: "动作后画面已保存",
      },
      current_subgoal: "sg-current",
      subgoals: [
        { subgoal_id: "sg-done", objective: "理解当前画面", status: "completed" },
        {
          subgoal_id: "sg-current",
          objective: "操作唯一匹配的语义目标",
          status: "current",
          reason: "当前画面只有一个候选目标",
        },
        // A conflicting legacy-style status must not create a second current row.
        { subgoal_id: "sg-later", objective: "重新观察验证结果", status: "running" },
      ],
    },
    current_action: {
      action_type: "tap_semantic",
      semantic_target: "唯一的确认控件",
      target_region: { x_min: 0.61, y_min: 0.72, x_max: 0.88, y_max: 0.82 },
      expected_change: "页面出现可验证的结果标记",
      confidence: 0.93,
      reason: "文字、位置和当前子目标一致",
      account_effect_possible: accountEffectPossible,
      physical_action_possible: true,
    },
    scene: {
      screen_id: "generic-screen",
      summary: "通用页面与一个确认控件",
      stable: true,
      confidence: 0.96,
    },
    history: [],
  };
}

function json(response, status, body) {
  response.writeHead(status, { "Content-Type": "application/json; charset=utf-8" });
  response.end(JSON.stringify(body));
}

function readBody(request) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    request.on("data", chunk => chunks.push(chunk));
    request.on("end", () => {
      try {
        resolve(chunks.length ? JSON.parse(Buffer.concat(chunks).toString("utf8")) : {});
      } catch (error) {
        reject(error);
      }
    });
    request.on("error", reject);
  });
}

function createServer({ risky = false } = {}) {
  return http.createServer(async (request, response) => {
    const url = new URL(request.url, "http://127.0.0.1");
    if (url.pathname === "/") {
      response.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
      response.end(fs.readFileSync(path.join(staticRoot, "index.html")));
      return;
    }
    if (url.pathname.startsWith("/assets/")) {
      const filename = path.basename(url.pathname);
      const mime = filename.endsWith(".js") ? "text/javascript" : "text/css";
      response.writeHead(200, { "Content-Type": `${mime}; charset=utf-8` });
      response.end(fs.readFileSync(path.join(staticRoot, filename)));
      return;
    }
    if (url.pathname === "/api/session") {
      json(response, 200, { token: "browser-contract-token", mock: true });
      return;
    }
    if (url.pathname === "/api/device") {
      json(response, 200, {
        controller_online: true,
        camera_online: true,
        busy: false,
        generic_supervised_execution: { active_sessions: [] },
        execution_architecture: { universal_agent: { observer: { current_stage: "idle" } } },
      });
      return;
    }
    if (url.pathname === "/api/preview.jpg") {
      response.writeHead(200, { "Content-Type": "image/png" });
      response.end(Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=", "base64"));
      return;
    }
    if (request.method === "POST" && url.pathname === "/api/agent/generic-supervised/start") {
      requests.start.push(await readBody(request));
      json(response, 200, { session: contractSession("awaiting_confirmation", risky) });
      return;
    }
    if (request.method === "POST" && url.pathname.endsWith("/confirm")) {
      requests.confirm.push(await readBody(request));
      json(response, 200, { session: contractSession(risky ? "awaiting_confirmation" : "paused_after_action", risky) });
      return;
    }
    if (request.method === "POST" && url.pathname.endsWith("/next")) {
      requests.next.push(await readBody(request));
      json(response, 200, { session: contractSession("awaiting_confirmation", risky) });
      return;
    }
    if (request.method === "POST" && url.pathname.endsWith("/auto")) {
      requests.auto.push(await readBody(request));
      await new Promise(resolve => setTimeout(resolve, 160));
      json(response, 200, { session: contractSession("awaiting_confirmation", risky) });
      return;
    }
    if (request.method === "POST" && url.pathname.endsWith("/cancel")) {
      requests.cancel.push(await readBody(request));
      json(response, 200, { session: contractSession("cancelled", risky) });
      return;
    }
    if (request.method === "POST" && url.pathname === "/api/stop") {
      requests.stop.push(await readBody(request));
      json(response, 200, { note: "mock stop" });
      return;
    }
    json(response, 404, { detail: "not found" });
  });
}

test("browser renders the new protocol and pause stops the next one-action request", { timeout: 30000 }, async () => {
  Object.values(requests).forEach(items => { items.length = 0; });
  const server = createServer();
  await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  const launchOptions = { headless: true };
  if (process.env.BROWSER_EXECUTABLE) launchOptions.executablePath = process.env.BROWSER_EXECUTABLE;
  else launchOptions.channel = "msedge";
  const browser = await chromium.launch(launchOptions);
  const page = await browser.newPage();
  const pageErrors = [];
  page.on("pageerror", error => pageErrors.push(error.message));

  try {
    await page.goto(`http://127.0.0.1:${address.port}/`);
    await page.locator("#agentText").fill("完成一个未预设的通用手机目标");
    await page.locator("#startSupervisedAgent").click();
    await page.locator("#goalSummary").getByText("task-browser-contract").waitFor();

    const goalText = await page.locator("#goalSummary").innerText();
    assert.match(goalText, /task-browser-contract/);
    assert.match(goalText, /device-local-01/);
    assert.match(goalText, /不得越过用户确认/);
    assert.match(goalText, /visible_state：结果标记可见/);
    assert.match(goalText, /evidence：动作后画面已保存/);

    assert.equal(await page.locator("#planList .plan-step").count(), 4);
    assert.equal(await page.locator("#planList .plan-step.current").count(), 1);
    assert.match(await page.locator("#planList .plan-step.current").innerText(), /操作唯一匹配的语义目标/);

    const actionText = await page.locator("#actionContent").innerText();
    assert.match(actionText, /点击语义控件/);
    assert.match(actionText, /唯一的确认控件/);
    assert.match(actionText, /x_min=0.61/);
    assert.match(actionText, /页面出现可验证的结果标记/);
    assert.match(actionText, /93%/);
    assert.match(actionText, /文字、位置和当前子目标一致/);

    assert.deepEqual(requests.start, [{
      text: "完成一个未预设的通用手机目标",
      device_id: "device-local-01",
    }]);

    await page.locator("#deviceId").evaluate(select => {
      select.disabled = false;
      select.add(new Option("device-selected-later", "device-selected-later"));
      select.value = "device-selected-later";
      select.dispatchEvent(new Event("change", { bubbles: true }));
    });

    await page.locator("#reviewAction").click();
    await page.locator("#confirmRiskAction").click();
    await page.locator("#nextSupervisedAgent").waitFor();
    assert.equal(requests.confirm.length, 1);
    assert.equal(requests.confirm[0].confirmed, true);
    assert.equal(requests.confirm[0].device_id, "device-local-01");

    await page.locator("#nextSupervisedAgent").click();
    await page.locator("#autoSupervisedAgent").waitFor();
    assert.equal(requests.next.length, 1);
    assert.equal(requests.next[0].device_id, "device-local-01");

    await page.locator("#autoSupervisedAgent").click();
    await page.locator("#pauseButton").click();
    await page.waitForTimeout(500);

    assert.equal(requests.auto.length, 1);
    assert.equal(requests.auto[0].max_physical_actions, 1);
    assert.equal(requests.auto[0].device_id, "device-local-01");
    assert.equal(await page.locator("#pauseNotice").isVisible(), true);

    await page.locator("#pauseButton").click();
    await page.locator("#cancelSupervisedAgent").click();
    await page.locator("#sessionBadge").getByText("已停止").waitFor();
    assert.equal(requests.cancel.length, 1);
    assert.equal(requests.cancel[0].device_id, "device-local-01");

    await page.locator("#stopButton").click();
    await page.waitForTimeout(50);
    assert.equal(requests.stop.length, 1);
    assert.equal(requests.stop[0].device_id, "device-local-01");
    assert.deepEqual(pageErrors, []);
  } finally {
    await browser.close();
    await new Promise(resolve => server.close(resolve));
  }
});

test("each external-state action is confirmed as a separate current step", { timeout: 30000 }, async () => {
  Object.values(requests).forEach(items => { items.length = 0; });
  const server = createServer({ risky: true });
  await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  const launchOptions = { headless: true };
  if (process.env.BROWSER_EXECUTABLE) launchOptions.executablePath = process.env.BROWSER_EXECUTABLE;
  else launchOptions.channel = "msedge";
  const browser = await chromium.launch(launchOptions);
  const page = await browser.newPage();

  try {
    await page.goto(`http://127.0.0.1:${address.port}/`);
    await page.locator("#agentText").fill("完成另一个通用目标");
    await page.locator("#startSupervisedAgent").click();
    await page.locator("#reviewAction").getByText("查看风险并确认").waitFor();
    assert.equal(await page.locator("#autoSupervisedAgent").count(), 0);

    await page.locator("#reviewAction").click();
    assert.match(await page.locator("#riskWarning").innerText(), /确认只授权当前一个动作/);
    assert.match(await page.locator("#riskWarning").innerText(), /后续风险动作仍需逐步重新确认/);
    const confirmResponse = page.waitForResponse(response => response.url().endsWith("/confirm"));
    await page.locator("#confirmRiskAction").click();
    await confirmResponse;

    // The mocked next observation proposes another external-state action.
    // The page must present a fresh confirmation instead of inheriting consent.
    await page.locator("#reviewAction").getByText("查看风险并确认").waitFor();
    assert.equal(requests.confirm.length, 1);
    assert.equal(await page.locator("#autoSupervisedAgent").count(), 0);
  } finally {
    await browser.close();
    await new Promise(resolve => server.close(resolve));
  }
});
