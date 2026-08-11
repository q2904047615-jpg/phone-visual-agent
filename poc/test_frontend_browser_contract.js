const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const http = require("node:http");
const path = require("node:path");
const { chromium } = require("playwright");

const staticRoot = path.join(__dirname, "static");
const deepSeekFixture = require("./frontend_contract_fixtures/deepseek_task_graph_v2.json");
const qwenFixture = require("./frontend_contract_fixtures/qwen_visual_decision_v2.json");
const requests = { start: [], confirm: [], next: [], auto: [], cancel: [], stop: [] };

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function externalSession() {
  return {
    session_id: "session-browser-external",
    task_graph: clone(deepSeekFixture.to_qwen_context),
    history: [],
  };
}

function safeActionSession(decisionStatus = "action") {
  const graph = clone(deepSeekFixture.task_graph);
  graph.status = "running";
  graph.current_subgoal = clone(graph.subgoals[0]);
  graph.current_subgoal.status = "active";
  graph.subgoals[0].status = "active";
  graph.subgoals[1].status = "pending";
  graph.active_subgoal_id = "locate_target";
  const decision = clone(qwenFixture.decision);
  if (decisionStatus !== "action") {
    decision.status = decisionStatus;
    decision.next_action = null;
    decision.target_region = null;
    decision.expected_result = {};
    decision.reason = decisionStatus === "blocked"
      ? "没有可靠且唯一的可信候选。"
      : "当前可见证据已满足目标。";
  }
  return {
    session_id: `session-browser-${decisionStatus}`,
    task_graph: graph,
    qwen_decision: decision,
    history: [],
  };
}

function cancelledSession() {
  const session = safeActionSession();
  session.task_graph.status = "cancelled";
  return session;
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

function createServer() {
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
      const body = await readBody(request);
      requests.start.push(body);
      const session = body.text.includes("外部状态")
        ? externalSession()
        : body.text.includes("blocked")
          ? safeActionSession("blocked")
          : body.text.includes("finished")
            ? safeActionSession("finished")
            : safeActionSession();
      json(response, 200, { session });
      return;
    }
    if (request.method === "POST" && url.pathname.endsWith("/confirm")) {
      requests.confirm.push(await readBody(request));
      json(response, 200, { session: externalSession() });
      return;
    }
    if (request.method === "POST" && url.pathname.endsWith("/next")) {
      requests.next.push(await readBody(request));
      json(response, 200, { session: safeActionSession() });
      return;
    }
    if (request.method === "POST" && url.pathname.endsWith("/auto")) {
      requests.auto.push(await readBody(request));
      await new Promise(resolve => setTimeout(resolve, 160));
      json(response, 200, { session: safeActionSession() });
      return;
    }
    if (request.method === "POST" && url.pathname.endsWith("/cancel")) {
      requests.cancel.push(await readBody(request));
      json(response, 200, { session: cancelledSession() });
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

async function launchFixturePage(server) {
  await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  const launchOptions = { headless: true };
  if (process.env.BROWSER_EXECUTABLE) launchOptions.executablePath = process.env.BROWSER_EXECUTABLE;
  else launchOptions.channel = "msedge";
  const browser = await chromium.launch(launchOptions);
  const page = await browser.newPage();
  await page.addInitScript(() => localStorage.setItem("visual-agent-device-id", "phone-01"));
  await page.goto(`http://127.0.0.1:${address.port}/`);
  return { browser, page };
}

test("browser renders real DeepSeek/Qwen snapshots and pauses after one safe request", { timeout: 30000 }, async () => {
  Object.values(requests).forEach(items => { items.length = 0; });
  const server = createServer();
  const { browser, page } = await launchFixturePage(server);
  const pageErrors = [];
  page.on("pageerror", error => pageErrors.push(error.message));

  try {
    await page.locator("#agentText").fill("运行真实协议快照");
    await page.locator("#startSupervisedAgent").click();
    await page.locator("#goalSummary").getByText("task-map-001").waitFor();

    const goalText = await page.locator("#goalSummary").innerText();
    assert.match(goalText, /在地图应用中找到图书馆并保存地点/);
    assert.doesNotMatch(goalText, /未命名目标/);
    assert.match(goalText, /2026-08-11-deepseek-task-graph-v2/);
    assert.match(goalText, /revision 1/);
    assert.match(goalText, /phone-01/);
    assert.match(goalText, /地图 \(maps\)/);
    assert.match(goalText, /不要发起导航/);
    assert.match(goalText, /目标地点已保存/);
    assert.match(goalText, /required=false/);
    assert.match(goalText, /external_allowed=false/);

    assert.equal(await page.locator("#planList .plan-step.current").count(), 1);
    assert.match(await page.locator("#planList .plan-step.current").innerText(), /目标地点详情可见/);

    const actionText = await page.locator("#actionContent").innerText();
    assert.match(actionText, /点击语义控件/);
    assert.match(actionText, /设置/);
    assert.match(actionText, /element_id settings_icon/);
    assert.match(actionText, /bounds=0.68、0.2、0.86、0.35/);
    assert.match(actionText, /scene_changed=true/);
    assert.match(actionText, /92%/);
    assert.match(actionText, /可信候选唯一且清晰/);
    assert.match(actionText, /2026-08-11-qwen-visual-decision-v2/);
    assert.match(actionText, /status action/);
    assert.match(actionText, /task task-map-001/);
    assert.match(actionText, /revision 1/);
    assert.match(actionText, /obs_0123456789abcdef0123456789abcdef/);
    assert.match(actionText, /51277d0d9e6f986b00dc/);

    await page.locator("#deviceId").evaluate(select => {
      select.disabled = false;
      select.add(new Option("phone-02", "phone-02"));
      select.value = "phone-02";
      select.dispatchEvent(new Event("change", { bubbles: true }));
    });

    await page.locator("#nextSupervisedAgent").click();
    await page.locator("#autoSupervisedAgent").waitFor();
    assert.equal(requests.next.length, 1);
    assert.equal(requests.next[0].device_id, "phone-01");

    await page.locator("#autoSupervisedAgent").click();
    await page.locator("#pauseButton").click();
    await page.waitForTimeout(500);
    assert.equal(requests.auto.length, 1);
    assert.deepEqual(requests.auto[0], {
      max_physical_actions: 1,
      device_id: "phone-01",
    });
    assert.equal(Object.prototype.hasOwnProperty.call(requests.auto[0], "confirmed"), false);
    assert.equal(await page.locator("#pauseNotice").isVisible(), true);

    await page.locator("#pauseButton").click();
    await page.locator("#cancelSupervisedAgent").click();
    await page.locator("#sessionBadge").getByText("已停止").waitFor();
    assert.equal(requests.cancel[0].device_id, "phone-01");

    await page.locator("#stopButton").click();
    await page.waitForTimeout(50);
    assert.equal(requests.stop[0].device_id, "phone-01");
    assert.deepEqual(pageErrors, []);
  } finally {
    await browser.close();
    await new Promise(resolve => server.close(resolve));
  }
});

test("external-state graph cannot auto-confirm and explicit consent is fully scoped", { timeout: 30000 }, async () => {
  Object.values(requests).forEach(items => { items.length = 0; });
  const server = createServer();
  const { browser, page } = await launchFixturePage(server);

  try {
    await page.locator("#agentText").fill("外部状态风险任务");
    await page.locator("#startSupervisedAgent").click();
    await page.locator("#reviewAction").getByText("查看风险并确认").waitFor();
    assert.equal(await page.locator("#autoSupervisedAgent").count(), 0);
    assert.equal(requests.auto.length, 0);
    const goalText = await page.locator("#goalSummary").innerText();
    assert.match(goalText, /确认门 · awaiting_confirmation/);
    assert.match(goalText, /required=true/);
    assert.match(goalText, /风险 save_place · 保存目标地点/);
    assert.match(await page.locator("#actionContent").innerText(), /等待视觉决策/);
    assert.match(await page.locator("#actionContent").innerText(), /Qwen 唯一动作尚未产生/);

    await page.locator("#reviewAction").click();
    const warning = await page.locator("#riskWarning").innerText();
    assert.match(warning, /task=task-map-001/);
    assert.match(warning, /revision=1/);
    assert.match(warning, /subgoal=save_target/);
    assert.match(warning, /risk_ids=save_place/);
    const confirmResponse = page.waitForResponse(response => response.url().endsWith("/confirm"));
    await page.locator("#confirmRiskAction").click();
    await confirmResponse;

    assert.deepEqual(requests.confirm[0], {
      confirmed: true,
      confirmation: {
        session_id: "session-browser-external",
        task_id: "task-map-001",
        device_id: "phone-01",
        revision: 1,
        subgoal_id: "save_target",
        risk_ids: ["save_place"],
      },
      device_id: "phone-01",
    });
    await page.locator("#reviewAction").getByText("查看风险并确认").waitFor();
    assert.equal(requests.confirm.length, 1);
    assert.equal(await page.locator("#autoSupervisedAgent").count(), 0);
  } finally {
    await browser.close();
    await new Promise(resolve => server.close(resolve));
  }
});

test("Qwen blocked and finished states render without executable controls", { timeout: 30000 }, async () => {
  for (const status of ["blocked", "finished"]) {
    Object.values(requests).forEach(items => { items.length = 0; });
    const server = createServer();
    const { browser, page } = await launchFixturePage(server);
    try {
      await page.locator("#agentText").fill(`Qwen ${status}`);
      await page.locator("#startSupervisedAgent").click();
      await page.locator("#actionContent").getByText(status === "blocked" ? "Qwen 已阻止" : "Qwen 判断已完成").waitFor();
      assert.match(await page.locator("#actionContent").innerText(), /不可执行/);
      assert.equal(await page.locator("#actionControls button").count(), 0);
    } finally {
      await browser.close();
      await new Promise(resolve => server.close(resolve));
    }
  }
});
