const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const http = require("node:http");
const path = require("node:path");
const { chromium } = require("playwright");

const staticRoot = path.join(__dirname, "static");
const deepSeekFixture = require("./frontend_contract_fixtures/deepseek_task_graph_v3.json");
const qwenFixture = require("./frontend_contract_fixtures/qwen_visual_decision_v2.json");
const requests = {
  start: [], approveRisk: [], confirm: [], next: [], auto: [], pause: [], cancel: [], stop: [],
  capabilityStart: [], capabilityConfirm: [], capabilityPromote: [], capabilityCancel: [],
  restore: [],
};

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

function safeActionSessionForDevice(deviceId, sessionId) {
  const session = safeActionSession();
  session.session_id = sessionId;
  session.task_graph.device_id = deviceId;
  session.qwen_decision.device_id = deviceId;
  session.qwen_decision.trusted_observation.device_id = deviceId;
  session.confirmation_scope.session_id = sessionId;
  session.confirmation_scope.device_id = deviceId;
  return session;
}

function capabilityTrial({ failed = false, completed = false, promoted = false } = {}) {
  const session = safeActionSession();
  session.session_id = "capability-session-browser";
  session.qwen_decision.next_action.action = "drag";
  session.confirmation_scope = {
    ...session.confirmation_scope,
    session_id: session.session_id,
  };
  if (completed) {
    session.status = "paused";
    session.physical_actions = 1;
  }
  const report = completed ? {
    status: failed ? "failed" : "passed",
    trial_id: "capability-trial-browser",
    device_id: "phone-01",
    candidate_action: "drag",
    physical_actions: 1,
    action_outcome: failed ? "mismatch" : "matched",
    error: failed ? "动作后验证未通过" : "",
    before_frame_paths: ["before-1.jpg", "before-2.jpg", "before-3.jpg", "before-4.jpg"],
    after_frame_paths: ["after-1.jpg", "after-2.jpg", "after-3.jpg", "after-4.jpg"],
  } : null;
  return {
    trial_id: "capability-trial-browser",
    device_id: "phone-01",
    candidate_action: "drag",
    text: failed ? "失败验收样本" : "拖动安全测试滑块",
    code_revision: "330d4c1",
    session,
    action_confirmation_scope: completed ? null : {
      ...session.confirmation_scope,
      trial_id: "capability-trial-browser",
      action: "drag",
    },
    risk_confirmation_scope: null,
    report,
    promotion_scope: completed && !failed ? {
      trial_id: "capability-trial-browser",
      device_id: "phone-01",
      action: "drag",
      report_sha256: "a".repeat(64),
      registry_sha256: "b".repeat(64),
    } : null,
    promotion: promoted ? { device_id: "phone-01", action: "drag", requires_restart: true } : null,
    requires_restart: promoted,
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

function createServer({ devices = null, activeSessions = [], restoredSessions = {} } = {}) {
  let currentCapabilityTrial = null;
  let capabilityShouldFail = false;
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
      const registeredDevices = devices || [{
        device_id: "phone-01",
        verified_actions: ["tap_semantic", "dismiss_overlay", "swipe", "back", "wait_for_change"],
      }];
      json(response, 200, {
        controller_online: true,
        camera_online: true,
        busy: false,
        default_device_id: "phone-01",
        devices: registeredDevices,
        generic_supervised_execution: { active_sessions: activeSessions },
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
    if (request.method === "POST" && url.pathname === "/api/capability-acceptance/start") {
      const body = await readBody(request);
      requests.capabilityStart.push(body);
      capabilityShouldFail = body.text.includes("失败");
      currentCapabilityTrial = capabilityTrial({ failed: capabilityShouldFail });
      json(response, 200, { physical_actions: 0, trial: currentCapabilityTrial });
      return;
    }
    const restoredMatch = url.pathname.match(/^\/api\/agent\/generic-supervised\/([^/]+)$/);
    if (request.method === "GET" && restoredMatch) {
      const sessionId = decodeURIComponent(restoredMatch[1]);
      requests.restore.push(sessionId);
      const session = restoredSessions[sessionId];
      if (!session) json(response, 404, { detail: "not found" });
      else json(response, 200, { session });
      return;
    }
    if (request.method === "GET" && url.pathname === "/api/capability-acceptance") {
      json(response, 200, { trials: [] });
      return;
    }
    if (request.method === "GET" && url.pathname === "/api/capability-acceptance/capability-trial-browser") {
      json(response, 200, { trial: currentCapabilityTrial });
      return;
    }
    if (request.method === "POST" && url.pathname === "/api/capability-acceptance/capability-trial-browser/confirm") {
      requests.capabilityConfirm.push(await readBody(request));
      currentCapabilityTrial = capabilityTrial({ failed: capabilityShouldFail, completed: true });
      if (capabilityShouldFail) {
        json(response, 409, { detail: { error: "动作后验证未通过", physical_actions: 1, trial: currentCapabilityTrial } });
      } else {
        json(response, 200, { physical_actions: 1, execution: { physical_actions: 1 }, trial: currentCapabilityTrial });
      }
      return;
    }
    if (request.method === "GET" && url.pathname === "/api/capability-acceptance/capability-trial-browser/promotion-preview") {
      json(response, 200, { physical_actions: 0, promotion_scope: currentCapabilityTrial.promotion_scope });
      return;
    }
    if (request.method === "POST" && url.pathname === "/api/capability-acceptance/capability-trial-browser/promote") {
      requests.capabilityPromote.push(await readBody(request));
      currentCapabilityTrial = capabilityTrial({ completed: true, promoted: true });
      json(response, 200, {
        physical_actions: 0,
        promotion: currentCapabilityTrial.promotion,
        trial: currentCapabilityTrial,
      });
      return;
    }
    if (request.method === "POST" && url.pathname === "/api/capability-acceptance/capability-trial-browser/cancel") {
      requests.capabilityCancel.push(await readBody(request));
      currentCapabilityTrial.session.status = "cancelled";
      json(response, 200, { physical_actions: 0, trial: currentCapabilityTrial });
      return;
    }
    if (request.method === "GET" && /\/api\/capability-acceptance\/capability-trial-browser\/evidence\/(before|after)\/[0-3]$/.test(url.pathname)) {
      response.writeHead(200, { "Content-Type": "image/png" });
      response.end(Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=", "base64"));
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
      json(response, 200, {
        execution: { physical_actions: 1, iterations: 1, pause_reason: "达到动作上限" },
        session: afterActionSession(),
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

async function launchFixturePage(server, { deviceId = "phone-01" } = {}) {
  await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  const launchOptions = { headless: true };
  if (process.env.BROWSER_EXECUTABLE) launchOptions.executablePath = process.env.BROWSER_EXECUTABLE;
  else launchOptions.channel = "msedge";
  const browser = await chromium.launch(launchOptions);
  const page = await browser.newPage();
  await page.addInitScript(value => localStorage.setItem("visual-agent-device-id", value), deviceId);
  await page.goto(`http://127.0.0.1:${address.port}/`);
  return { browser, page };
}

test("multi-device console restores only the selected device session", { timeout: 30000 }, async () => {
  Object.values(requests).forEach(items => { items.length = 0; });
  const phoneA = safeActionSessionForDevice("phone-01", "session-phone-01");
  const phoneB = safeActionSessionForDevice("phone-02", "session-phone-02");
  const server = createServer({
    devices: [
      { device_id: "phone-01", verified_actions: ["tap_semantic"] },
      { device_id: "phone-02", verified_actions: ["tap_semantic"] },
    ],
    activeSessions: [
      { session_id: phoneA.session_id, device_id: "phone-01", status: phoneA.status },
      { session_id: phoneB.session_id, device_id: "phone-02", status: phoneB.status },
    ],
    restoredSessions: {
      [phoneA.session_id]: phoneA,
      [phoneB.session_id]: phoneB,
    },
  });
  const { browser, page } = await launchFixturePage(server);
  try {
    await page.locator("#goalSummary").getByText("会话 · session-phone-01", { exact: true }).waitFor({ timeout: 5000 });
    assert.deepEqual(requests.restore, ["session-phone-01"]);

    await page.locator("#deviceId").evaluate(select => {
      select.disabled = false;
      select.value = "phone-02";
      select.dispatchEvent(new Event("change", { bubbles: true }));
    });

    await page.waitForTimeout(500);
    assert.deepEqual(requests.restore, ["session-phone-01", "session-phone-02"]);
    assert.match(await page.locator("#goalSummary").innerText(), /会话 · session-phone-02/);
    assert.equal(await page.locator("#deviceId").inputValue(), "phone-02");
    assert.match(await page.locator("#goalSummary").innerText(), /phone-02/);
  } finally {
    await browser.close();
    await new Promise(resolve => server.close(resolve));
  }
});

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
    assert.match(goalText, /会话 · session-browser-action/);
    assert.match(goalText, /scope session-browser-action \/ task-map-001 \/ phone-01 \/ r1 \/ locate_target/);
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
    assert.match(actionText, /session session-browser-action/);
    assert.match(actionText, /task task-map-001/);
    assert.match(actionText, /revision 1/);
    assert.match(actionText, /obs_0123456789abcdef0123456789abcdef/);
    assert.match(actionText, /51277d0d9e6f986b00dc/);
    assert.match(actionText, /本地策略/);
    assert.match(actionText, /仅允许当前通用导航动作/);
    assert.match(await page.locator("#sessionBadge").innerText(), /等待当前动作确认/);
    assert.match(await page.locator("#safetyText").innerText(), /等待当前动作确认/);
    assert.match(actionText, /需要当前动作确认/);
    assert.equal(await page.locator("#reviewAction").innerText(), "确认当前动作");
    assert.doesNotMatch(actionText, /需要风险范围确认/);

    await page.locator("#reviewAction").click();
    const warning = await page.locator("#riskWarning").innerText();
    assert.match(warning, /session=session-browser-action/);
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
    await page.locator("#sessionBadge").getByText("等待风险范围确认").waitFor();
    assert.equal(await page.locator("#autoSupervisedAgent").count(), 0);
    assert.equal(await page.locator("#reviewAction").count(), 1);
    assert.equal(await page.locator("#reviewAction").innerText(), "查看风险范围并确认");
    assert.equal(requests.auto.length, 0);
    const goalText = await page.locator("#goalSummary").innerText();
    assert.match(goalText, /确认门 · awaiting_risk_confirmation/);
    assert.match(goalText, /required=true/);
    assert.match(goalText, /风险 save_place · 保存目标地点/);
    assert.match(goalText, /scope session-browser-external \/ task-map-001 \/ phone-01 \/ r1 \/ save_target/);
    assert.match(await page.locator("#actionContent").innerText(), /Qwen 唯一动作尚未产生/);
    await page.locator("#reviewAction").click();
    assert.match(await page.locator("#riskWarning").innerText(), /session=session-browser-external/);
    assert.match(await page.locator("#riskWarning").innerText(), /此确认本身不会触发机械臂/);
    const approvalResponse = page.waitForResponse(
      response => response.url().endsWith("/approve-risk"),
      { timeout: 5000 },
    );
    await page.locator("#confirmRiskAction").click();
    await approvalResponse;
    await page.locator("#reviewAction").waitFor({ timeout: 5000 });
    assert.match(await page.locator("#sessionBadge").innerText(), /等待当前动作确认/);
    assert.equal(await page.locator("#reviewAction").innerText(), "确认当前动作");
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
    assert.match(await page.locator("#riskWarning").innerText(), /session=session-browser-external/);
    assert.match(await page.locator("#riskWarning").innerText(), /obs_0123456789abcdef0123456789abcdef/);
    assert.match(await page.locator("#riskWarning").innerText(), /51277d0d9e6f986b00dc/);
    assert.equal(requests.confirm.length, 0);
    assert.equal(await page.locator("#autoSupervisedAgent").count(), 0);
  } finally {
    await browser.close();
    await new Promise(resolve => server.close(resolve));
  }
});

test("browser never offers one-confirmation multi-action execution", { timeout: 30000 }, async () => {
  Object.values(requests).forEach(items => { items.length = 0; });
  const server = createServer();
  const { browser, page } = await launchFixturePage(server);
  try {
    await page.locator("#agentText").fill("连续查看安全页面");
    await page.locator("#startSupervisedAgent").click();
    await page.locator("#reviewAction").click();
    await page.locator("#riskDialog").waitFor({ state: "visible" });
    assert.equal(await page.locator("#confirmSafeLoop").count(), 0);
    assert.match(await page.locator("#riskWarning").innerText(), /一个动作/);
    await page.keyboard.press("Escape");
    assert.equal(requests.auto.length, 0);
    assert.equal(requests.confirm.length, 0);
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

test("capability panel performs one confirmed action then a separate zero-action promotion", { timeout: 30000 }, async () => {
  Object.values(requests).forEach(items => { items.length = 0; });
  const server = createServer();
  const { browser, page } = await launchFixturePage(server);
  const pageErrors = [];
  page.on("pageerror", error => pageErrors.push(error.message));
  try {
    await page.locator("#capabilityAction").selectOption("drag");
    await page.locator("#capabilityGoal").fill("拖动安全测试滑块");
    await page.locator("#startCapabilityTrial").click();
    await page.locator("#capabilityStatus").getByText(/trial capability-trial-browser/).waitFor({ timeout: 5000 });
    assert.deepEqual(requests.capabilityStart, [{
      device_id: "phone-01",
      action: "drag",
      text: "拖动安全测试滑块",
    }]);
    assert.equal(requests.capabilityConfirm.length, 0);
    assert.match(await page.locator("#capabilityStatus").innerText(), /physical_actions 0/);

    await page.locator("#reviewCapabilityAction").click();
    assert.match(await page.locator("#riskWarning").innerText(), /trial=capability-trial-browser/);
    assert.match(await page.locator("#riskWarning").innerText(), /action=drag/);
    const confirmResponse = page.waitForResponse(
      response => response.url().endsWith("/capability-trial-browser/confirm"),
    );
    await page.locator("#confirmRiskAction").click();
    await confirmResponse;
    await page.locator("#capabilityBadge").getByText("待确认启用").waitFor({ timeout: 5000 });
    assert.equal(requests.capabilityConfirm.length, 1);
    assert.equal(requests.capabilityConfirm[0].confirmation.trial_id, "capability-trial-browser");
    assert.equal(requests.capabilityConfirm[0].confirmation.action, "drag");
    assert.equal(requests.capabilityConfirm[0].confirmation.observation_id, "obs_0123456789abcdef0123456789abcdef");
    assert.equal(await page.locator("#capabilityEvidence figure").count(), 8);

    await page.locator("#reviewCapabilityPromotion").click();
    await page.locator("#promotionDialog").waitFor({ state: "visible", timeout: 5000 });
    assert.match(await page.locator("#promotionTarget").innerText(), /phone-01 \/ drag/);
    assert.equal(requests.capabilityPromote.length, 0);
    const promotionResponse = page.waitForResponse(
      response => response.url().endsWith("/capability-trial-browser/promote"),
    );
    await page.locator("#confirmPromotion").click();
    await promotionResponse;
    await page.locator("#capabilityBadge").getByText("等待重启").waitFor({ timeout: 5000 });
    assert.deepEqual(requests.capabilityPromote, [{
      confirmed: true,
      trial_id: "capability-trial-browser",
      device_id: "phone-01",
      action: "drag",
      report_sha256: "a".repeat(64),
      registry_sha256: "b".repeat(64),
    }]);
    assert.equal(requests.confirm.length, 0);
    assert.deepEqual(pageErrors, []);
  } finally {
    await browser.close();
    await new Promise(resolve => server.close(resolve));
  }
});

test("failed capability evidence never exposes a promotion button or retries", { timeout: 30000 }, async () => {
  Object.values(requests).forEach(items => { items.length = 0; });
  const server = createServer();
  const { browser, page } = await launchFixturePage(server);
  try {
    await page.locator("#capabilityAction").selectOption("drag");
    await page.locator("#capabilityGoal").fill("失败验收样本");
    await page.locator("#startCapabilityTrial").click();
    await page.locator("#reviewCapabilityAction").click();
    const failedResponse = page.waitForResponse(
      response => response.url().endsWith("/capability-trial-browser/confirm"),
    );
    await page.locator("#confirmRiskAction").click();
    await failedResponse;
    await page.locator("#capabilityBadge").getByText("验收失败").waitFor({ timeout: 5000 });
    assert.equal(requests.capabilityConfirm.length, 1);
    assert.equal(requests.capabilityPromote.length, 0);
    assert.equal(await page.locator("#reviewCapabilityPromotion").count(), 0);
    assert.equal(await page.locator("#capabilityEvidence figure").count(), 8);
    assert.match(await page.locator("#capabilityStatus").innerText(), /动作后验证未通过/);
  } finally {
    await browser.close();
    await new Promise(resolve => server.close(resolve));
  }
});
