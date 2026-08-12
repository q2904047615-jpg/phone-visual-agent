const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const http = require("node:http");
const path = require("node:path");
const { chromium } = require("playwright");

const staticRoot = path.join(__dirname, "static");
const deepSeekFixture = require("./frontend_contract_fixtures/deepseek_task_graph_v3.json");
const qwenFixture = require("./frontend_contract_fixtures/qwen_visual_decision_v2.json");
const requests = { start: [], approveRisk: [], confirm: [], next: [], auto: [], pause: [], cancel: [], stop: [] };

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function externalSession() {
  return {
    session_id: "session-browser-external",
    status: "awaiting_risk_confirmation",
    task_graph: clone(deepSeekFixture.task_graph),
    risk_confirmation_scope: {
      session_id: "session-browser-external",
      task_id: "task-map-001",
      device_id: "phone-01",
      revision: 1,
      subgoal_id: "save_target",
      risk_ids: ["save_place"],
    },
    risk_confirmation_ready: true,
    physical_actions: 0,
    evidence: [],
    history: [],
  };
}

function externalActionSession() {
  const session = externalSession();
  session.status = "awaiting_confirmation";
  session.risk_confirmation_ready = false;
  session.confirmed_risk_ids = ["save_place"];
  session.qwen_decision = clone(qwenFixture.decision);
  session.controller_decision = {
    allowed: true,
    reason: "当前外部状态动作已通过作用域确认。",
    canonical_class: "external",
    policy_version: "2026-08-12-universal-action-policy-v2",
  };
  session.confirmation_scope = {
    ...session.risk_confirmation_scope,
    observation_id: "obs_0123456789abcdef0123456789abcdef",
    fingerprint: "51277d0d9e6f986b00dc",
  };
  session.confirmation_ready = true;
  return session;
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
    status: decisionStatus === "action" ? "awaiting_confirmation" : "blocked",
    task_graph: graph,
    qwen_decision: decision,
    controller_decision: {
      allowed: decisionStatus === "action",
      reason: decisionStatus === "action"
        ? "仅允许当前通用导航动作。"
        : "当前视觉决策不可执行。",
      canonical_class: decisionStatus === "action" ? "navigation_open" : "",
      policy_version: "2026-08-12-phase-one-navigation-v1",
    },
    confirmation_scope: decisionStatus === "action" ? {
      session_id: `session-browser-${decisionStatus}`,
      task_id: "task-map-001",
      device_id: "phone-01",
      revision: 1,
      subgoal_id: "locate_target",
      risk_ids: [],
      observation_id: "obs_0123456789abcdef0123456789abcdef",
      fingerprint: "51277d0d9e6f986b00dc",
    } : null,
    confirmation_ready: decisionStatus === "action",
    physical_actions: 0,
    evidence: ["before_step_1_frame_1.jpg"],
    history: [],
  };
}

function afterActionSession() {
  const session = safeActionSession();
  session.task_graph.revision = 2;
  session.qwen_decision.revision = 2;
  session.qwen_decision.observation_id = "obs_after_0123456789abcdef";
  session.qwen_decision.fingerprint = "after-frame-fingerprint";
  session.confirmation_scope.revision = 2;
  session.confirmation_scope.observation_id = session.qwen_decision.observation_id;
  session.confirmation_scope.fingerprint = session.qwen_decision.fingerprint;
  session.physical_actions = 1;
  session.evidence = ["before-1.jpg", "after-1.jpg", "after-2.jpg", "after-3.jpg", "after-4.jpg"];
  session.history = [{
    step_number: 1,
    qwen_decision: clone(qwenFixture.decision),
    execution: {
      physical_actions: 1,
      evidence: ["before-1.jpg", "after-1.jpg"],
      after_frame_paths: ["after-1.jpg", "after-2.jpg", "after-3.jpg", "after-4.jpg"],
    },
    reason: "动作后新画面已验证",
  }];
  return session;
}

function cancelledSession() {
  const session = safeActionSession();
  session.status = "cancelled";
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
      json(response, 200, { session: afterActionSession() });
      return;
    }
    if (request.method === "POST" && url.pathname.endsWith("/approve-risk")) {
      requests.approveRisk.push(await readBody(request));
      json(response, 200, { physical_actions: 0, session: externalActionSession() });
      return;
    }
    if (request.method === "POST" && url.pathname.endsWith("/next")) {
      requests.next.push(await readBody(request));
      json(response, 200, { session: safeActionSession() });
      return;
    }
    if (request.method === "POST" && url.pathname.endsWith("/auto")) {
      requests.auto.push(await readBody(request));
      json(response, 409, {
        detail: {
          code: "phase_one_manual_confirmation_required",
          physical_actions: 0,
        },
      });
      return;
    }
    if (request.method === "POST" && url.pathname.endsWith("/pause")) {
      requests.pause.push(await readBody(request));
      json(response, 200, { physical_actions: 0, session: safeActionSession() });
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

test("browser renders controller evidence and confirms one exact observation", { timeout: 30000 }, async () => {
  Object.values(requests).forEach(items => { items.length = 0; });
  const server = createServer();
  const { browser, page } = await launchFixturePage(server);
  const pageErrors = [];
  page.on("pageerror", error => pageErrors.push(error.message));

  try {
    await page.locator("#agentText").fill("运行真实协议快照");
    await page.locator("#startSupervisedAgent").click();
    await page.locator("#goalSummary").getByText(/目标 · task-map-001/).waitFor({ timeout: 5000 });

    const goalText = await page.locator("#goalSummary").innerText();
    assert.match(goalText, /在地图应用中找到图书馆并保存地点/);
    assert.doesNotMatch(goalText, /未命名目标/);
    assert.match(goalText, /2026-08-11-deepseek-task-graph-v3/);
    assert.match(goalText, /revision 1/);
    assert.match(goalText, /phone-01/);
    assert.match(goalText, /地图 \(maps\)/);
    assert.match(goalText, /不要发起导航/);
    assert.match(goalText, /目标地点已保存/);
    assert.match(goalText, /required=true/);
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
    assert.match(actionText, /2026-08-12-qwen-visual-decision-v3/);
    assert.match(actionText, /status action/);
    assert.match(actionText, /task task-map-001/);
    assert.match(actionText, /revision 1/);
    assert.match(actionText, /obs_0123456789abcdef0123456789abcdef/);
    assert.match(actionText, /51277d0d9e6f986b00dc/);
    assert.match(actionText, /本地策略/);
    assert.match(actionText, /仅允许当前通用导航动作/);

    await page.locator("#reviewAction").click();
    const warning = await page.locator("#riskWarning").innerText();
    assert.match(warning, /obs_0123456789abcdef0123456789abcdef/);
    assert.match(warning, /51277d0d9e6f986b00dc/);
    const confirmResponse = page.waitForResponse(
      response => response.url().endsWith("/confirm"),
      { timeout: 5000 },
    );
    await page.locator("#confirmRiskAction").click();
    await confirmResponse;
    await page.locator("#sceneMeta").getByText("1", { exact: true }).waitFor({ timeout: 5000 });

    assert.deepEqual(requests.confirm[0], {
      confirmed: true,
      confirmation: {
        session_id: "session-browser-action",
        task_id: "task-map-001",
        device_id: "phone-01",
        revision: 1,
        subgoal_id: "locate_target",
        risk_ids: [],
        observation_id: "obs_0123456789abcdef0123456789abcdef",
        fingerprint: "51277d0d9e6f986b00dc",
      },
    });
    assert.match(await page.locator("#sceneMeta").innerText(), /累计动作\s*1/);
    assert.match(await page.locator("#traceList").innerText(), /after-1.jpg/);
    assert.equal(await page.locator("#autoSupervisedAgent").count(), 0);
    assert.equal(requests.auto.length, 0);

    await page.locator("#deviceId").evaluate(select => {
      select.disabled = false;
      select.add(new Option("phone-02", "phone-02"));
      select.value = "phone-02";
      select.dispatchEvent(new Event("change", { bubbles: true }));
    });

    await page.locator("#nextSupervisedAgent").click();
    await page.locator("#reviewAction").waitFor({ timeout: 5000 });
    assert.equal(requests.next.length, 1);
    assert.equal(requests.next[0].device_id, "phone-01");
    await page.locator("#pauseButton").click();
    await page.waitForTimeout(100);
    assert.equal(requests.auto.length, 0);
    assert.equal(await page.locator("#pauseNotice").isVisible(), true);
    assert.equal(requests.pause.length, 1);
    assert.deepEqual(requests.pause[0], { device_id: "phone-01" });

    await page.locator("#pauseButton").click();
    const cancelResponse = page.waitForResponse(
      response => response.url().endsWith("/cancel"),
      { timeout: 5000 },
    );
    await page.locator("#cancelSupervisedAgent").click();
    await cancelResponse;
    await page.locator("#sessionBadge").getByText("已停止").waitFor({ timeout: 5000 });
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

test("external-state graph requires risk approval before exact action confirmation", { timeout: 30000 }, async () => {
  Object.values(requests).forEach(items => { items.length = 0; });
  const server = createServer();
  const { browser, page } = await launchFixturePage(server);

  try {
    await page.locator("#agentText").fill("外部状态风险任务");
    await page.locator("#startSupervisedAgent").click();
    await page.locator("#sessionBadge").getByText("等待风险确认").waitFor();
    assert.equal(await page.locator("#autoSupervisedAgent").count(), 0);
    assert.equal(await page.locator("#reviewAction").count(), 1);
    assert.equal(requests.auto.length, 0);
    const goalText = await page.locator("#goalSummary").innerText();
    assert.match(goalText, /确认门 · awaiting_risk_confirmation/);
    assert.match(goalText, /required=true/);
    assert.match(goalText, /风险 save_place · 保存目标地点/);
    assert.match(goalText, /scope task-map-001 \/ phone-01 \/ r1 \/ save_target/);
    assert.match(await page.locator("#actionContent").innerText(), /Qwen 唯一动作尚未产生/);
    await page.locator("#reviewAction").click();
    assert.match(await page.locator("#riskWarning").innerText(), /此确认本身不会触发机械臂/);
    const approvalResponse = page.waitForResponse(
      response => response.url().endsWith("/approve-risk"),
      { timeout: 5000 },
    );
    await page.locator("#confirmRiskAction").click();
    await approvalResponse;
    await page.locator("#reviewAction").waitFor({ timeout: 5000 });
    assert.deepEqual(requests.approveRisk[0], {
      confirmed: true,
      confirmation: {
        session_id: "session-browser-external",
        task_id: "task-map-001",
        device_id: "phone-01",
        revision: 1,
        subgoal_id: "save_target",
        risk_ids: ["save_place"],
      },
    });
    assert.equal(requests.confirm.length, 0);
    await page.locator("#reviewAction").click();
    assert.match(await page.locator("#riskWarning").innerText(), /obs_0123456789abcdef0123456789abcdef/);
    assert.match(await page.locator("#riskWarning").innerText(), /51277d0d9e6f986b00dc/);
    assert.equal(requests.confirm.length, 0);
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
      await page.locator("#actionContent").getByText("已阻止").waitFor({ timeout: 5000 });
      assert.match(
        await page.locator("#actionContent").innerText(),
        status === "blocked" ? /没有可靠且唯一/ : /当前可见证据已满足目标/,
      );
      assert.equal(await page.locator("#actionControls button").count(), 0);
    } finally {
      await browser.close();
      await new Promise(resolve => server.close(resolve));
    }
  }
});
