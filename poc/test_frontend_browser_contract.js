const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const http = require("node:http");
const path = require("node:path");
const { chromium } = require("playwright");

const staticRoot = path.join(__dirname, "static");
const deepSeekFixture = require("./frontend_contract_fixtures/deepseek_typed_task_graph_v4.json");
const qwenFixture = require("./frontend_contract_fixtures/qwen_visual_decision_v4.json");
const requests = {
  start: [], approveEffect: [], confirm: [], next: [], auto: [], pause: [], cancel: [], stop: [],
  capabilityStart: [], capabilityConfirm: [], capabilityPromote: [], capabilityCancel: [],
  restore: [], previewDevices: [], assets: [], startTokens: [],
};

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function externalSession() {
  return {
    session_id: "session-browser-external",
    status: "awaiting_effect_confirmation",
    task_graph: clone(deepSeekFixture.task_graph),
    effect_confirmation_scope: {
      session_id: "session-browser-external",
      task_id: "task-map-001",
      device_id: "phone-01",
      revision: 1,
      subgoal_id: "submit_payment",
      effect_ids: ["payment_order"],
      intent_digest: "d".repeat(64),
    },
    effect_confirmation_ready: true,
    effect_previews: [{
      effect_id: "payment_order",
      effect_kind: "financial_transaction",
      targets: [{entity_ref: "merchant-1", role: "merchant", value: "演示商户"}],
      payloads: [{entity_ref: "amount-1", role: "amount", value: "20元"}],
      policy: "confirmation_required",
      preview_digest: "e".repeat(64),
    }],
    physical_actions: 0,
    evidence: [],
    history: [],
  };
}

function externalActionSession() {
  const session = externalSession();
  session.status = "awaiting_confirmation";
  session.effect_confirmation_ready = false;
  session.confirmed_effect_ids = ["payment_order"];
  session.qwen_decision = clone(qwenFixture.decision);
  session.controller_decision = {
    allowed: true,
    reason: "当前外部状态动作已通过作用域确认。",
    canonical_class: "external",
    policy_version: "2026-08-13-universal-action-policy-v3",
  };
  session.confirmation_scope = {
    ...session.effect_confirmation_scope,
    observation_id: "obs_0123456789abcdef0123456789abcdef",
    fingerprint: "51277d0d9e6f986b00dc",
    decision_node_id: "qwen_visual_revision_1",
    action_digest: "a".repeat(64),
  };
  session.confirmation_ready = true;
  return session;
}

function externalExecutedSession() {
  const session = externalActionSession();
  session.status = "succeeded";
  session.physical_actions = 1;
  session.task_graph.status = "completed";
  session.task_graph.active_subgoal_id = null;
  session.task_graph.current_subgoal = null;
  session.task_graph.subgoals = session.task_graph.subgoals.map(item => ({
    ...item,
    status: "completed",
  }));
  session.confirmation_scope = null;
  session.confirmation_ready = false;
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
      effect_ids: [],
      observation_id: "obs_0123456789abcdef0123456789abcdef",
      fingerprint: "51277d0d9e6f986b00dc",
      decision_node_id: "qwen_visual_revision_1",
      action_digest: "a".repeat(64),
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
    effect_confirmation_scope: null,
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

function readOnlyRecoveredCapabilityTrial() {
  const trial = capabilityTrial();
  trial.trial_id = "capability-history-browser";
  trial.session.session_id = "capability-history-session-browser";
  trial.session.status = "failed";
  trial.session.confirmation_ready = false;
  trial.action_confirmation_scope = null;
  trial.read_only_recovered = true;
  return trial;
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

function createServer({
  devices = null,
  activeSessions = [],
  restoredSessions = {},
  restoredCapabilityTrials = [],
  runtimeSession = { token: "browser-contract-token" },
  enforceToken = false,
} = {}) {
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
      requests.assets.push(`${url.pathname}${url.search}`);
      const filename = path.basename(url.pathname);
      const mime = filename.endsWith(".js") ? "text/javascript" : "text/css";
      response.writeHead(200, { "Content-Type": `${mime}; charset=utf-8` });
      response.end(fs.readFileSync(path.join(staticRoot, filename)));
      return;
    }
    if (url.pathname === "/api/session") {
      json(response, 200, { token: runtimeSession.token, mock: true });
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
      requests.previewDevices.push(url.searchParams.get("device_id"));
      response.writeHead(200, { "Content-Type": "image/png" });
      response.end(Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=", "base64"));
      return;
    }
    if (request.method === "POST" && url.pathname === "/api/agent/generic-supervised/start") {
      requests.startTokens.push(String(request.headers["x-control-token"] || ""));
      if (enforceToken && request.headers["x-control-token"] !== runtimeSession.token) {
        json(response, 403, { detail: "控制令牌无效。" });
        return;
      }
      const body = await readBody(request);
      requests.start.push(body);
      if (body.text.includes("slow")) {
        await new Promise(resolve => setTimeout(resolve, 300));
      }
      if (body.text.includes("start failure")) {
        json(response, 409, {
          detail: {
            error: "规划服务暂时不可用",
            physical_actions: 0,
            session: null,
          },
        });
        return;
      }
      const session = body.text.includes("succeeded")
        ? externalExecutedSession()
        : body.text.includes("本地确认")
        ? externalSession()
        : body.text.includes("blocked")
          ? safeActionSession("blocked")
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
      json(response, 200, { trials: restoredCapabilityTrials });
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
    if (request.method === "POST" && url.pathname.endsWith("/approve-effect")) {
      requests.approveEffect.push(await readBody(request));
      json(response, 200, { physical_actions: 1, session: externalExecutedSession() });
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

test("browser requests versioned task-status assets instead of stale cached URLs", { timeout: 30000 }, async () => {
  Object.values(requests).forEach(items => { items.length = 0; });
  const server = createServer();
  const { browser } = await launchFixturePage(server);
  try {
    assert.ok(requests.assets.includes("/assets/app.js?v=20260825-session-reconcile-v1"));
    assert.ok(requests.assets.includes("/assets/styles.css?v=20260824-task-status-v3"));
    assert.equal(requests.assets.includes("/assets/app.js"), false);
    assert.equal(requests.assets.includes("/assets/styles.css"), false);
  } finally {
    await browser.close();
    await new Promise(resolve => server.close(resolve));
  }
});

test("browser stops showing a vanished backend session as running", { timeout: 30000 }, async () => {
  Object.values(requests).forEach(items => { items.length = 0; });
  const restored = safeActionSessionForDevice("phone-01", "session-vanished-after-reload");
  const activeSessions = [{
    session_id: restored.session_id,
    device_id: "phone-01",
    status: restored.status,
  }];
  const restoredSessions = { [restored.session_id]: restored };
  const server = createServer({ activeSessions, restoredSessions });
  const { browser, page } = await launchFixturePage(server);
  try {
    await page.locator("#taskRunStatusLabel").getByText("进行中", { exact: true }).waitFor({ timeout: 5000 });
    activeSessions.splice(0, activeSessions.length);
    delete restoredSessions[restored.session_id];

    await page.locator("#taskRunStatusLabel").getByText("失败", { exact: true }).waitFor({ timeout: 7000 });
    assert.equal(await page.locator("#taskRunStatus").getAttribute("data-task-state"), "failure");
    assert.match(await page.locator("#taskRunStatusDetail").innerText(), /已不在当前服务中.*停止显示为进行中/);
    assert.equal(await page.locator("#startSupervisedAgent").isEnabled(), true);
  } finally {
    await browser.close();
    await new Promise(resolve => server.close(resolve));
  }
});

test("browser refreshes the control token after a service reload", { timeout: 30000 }, async () => {
  Object.values(requests).forEach(items => { items.length = 0; });
  const runtimeSession = { token: "browser-token-before-reload" };
  const server = createServer({ runtimeSession, enforceToken: true });
  const { browser, page } = await launchFixturePage(server);
  try {
    runtimeSession.token = "browser-token-after-reload";
    await page.waitForTimeout(3500);
    await page.locator("#agentText").fill("succeeded after reload");
    await page.locator("#startSupervisedAgent").click();
    await page.locator("#taskRunStatusLabel").getByText("成功", { exact: true }).waitFor({ timeout: 5000 });
    assert.equal(requests.startTokens.at(-1), "browser-token-after-reload");
  } finally {
    await browser.close();
    await new Promise(resolve => server.close(resolve));
  }
});

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
    await page.locator("#taskRunStatusLabel").getByText("进行中", { exact: true }).waitFor({ timeout: 5000 });
    assert.equal(await page.locator("#taskRunStatus").getAttribute("data-task-state"), "running");
    assert.deepEqual(requests.restore, ["session-phone-01"]);
    await page.waitForFunction(() => document.querySelector("#phonePreview")?.complete);
    assert.equal(requests.previewDevices.at(-1), "phone-01");

    const phoneTwoPreview = page.waitForRequest(
      request => new URL(request.url()).pathname === "/api/preview.jpg"
        && new URL(request.url()).searchParams.get("device_id") === "phone-02",
    );
    await page.locator("#deviceId").evaluate(select => {
      select.disabled = false;
      select.value = "phone-02";
      select.dispatchEvent(new Event("change", { bubbles: true }));
    });
    await phoneTwoPreview;

    await page.waitForTimeout(500);
    assert.deepEqual(requests.restore, ["session-phone-01", "session-phone-02"]);
    assert.match(await page.locator("#goalSummary").innerText(), /会话 · session-phone-02/);
    assert.equal(await page.locator("#deviceId").inputValue(), "phone-02");
    assert.match(await page.locator("#goalSummary").innerText(), /phone-02/);
    assert.equal(requests.previewDevices.at(-1), "phone-02");
  } finally {
    await browser.close();
    await new Promise(resolve => server.close(resolve));
  }
});

test("ordinary task status stays explicit through running success failure and refresh", { timeout: 30000 }, async () => {
  Object.values(requests).forEach(items => { items.length = 0; });
  const server = createServer();
  const { browser, page } = await launchFixturePage(server);
  try {
    await page.locator("#taskRunStatusLabel").getByText("未开始", { exact: true }).waitFor({ timeout: 5000 });
    assert.equal(await page.locator("#taskRunStatus").getAttribute("data-task-state"), "not-started");

    await page.locator("#agentText").fill("slow succeeded");
    await page.locator("#startSupervisedAgent").click();
    await page.locator("#taskRunStatusLabel").getByText("进行中", { exact: true }).waitFor({ timeout: 5000 });
    assert.equal(await page.locator("#taskRunStatus").getAttribute("data-task-state"), "running");
    await page.locator("#taskRunStatusLabel").getByText("成功", { exact: true }).waitFor({ timeout: 5000 });
    assert.equal(await page.locator("#taskRunStatus").getAttribute("data-task-state"), "success");
    assert.match(await page.locator("#taskRunStatusDetail").innerText(), /目标已完成/);

    await page.locator("#agentText").fill("start failure");
    await page.locator("#startSupervisedAgent").click();
    await page.locator("#taskRunStatusLabel").getByText("失败", { exact: true }).waitFor({ timeout: 5000 });
    assert.equal(await page.locator("#taskRunStatus").getAttribute("data-task-state"), "failure");
    assert.match(await page.locator("#taskRunStatusDetail").innerText(), /规划服务暂时不可用/);
    await page.waitForTimeout(3800);
    assert.equal(await page.locator("#taskRunStatusLabel").innerText(), "失败");

    await page.reload();
    await page.locator("#taskRunStatusLabel").getByText("失败", { exact: true }).waitFor({ timeout: 5000 });
    assert.match(await page.locator("#taskRunStatusDetail").innerText(), /最近一次任务.*规划服务暂时不可用/);
  } finally {
    await browser.close();
    await new Promise(resolve => server.close(resolve));
  }
});

test("read-only recovered capability history is not restored into the current panel", { timeout: 30000 }, async () => {
  Object.values(requests).forEach(items => { items.length = 0; });
  const server = createServer({
    restoredCapabilityTrials: [readOnlyRecoveredCapabilityTrial()],
  });
  const { browser, page } = await launchFixturePage(server);
  try {
    await page.locator("#capabilityBadge").getByText("未开始").waitFor({ timeout: 5000 });
    assert.equal(await page.locator("#resetCapabilityTrial").count(), 0);
    assert.equal(await page.locator("#startSupervisedAgent").isEnabled(), true);
    assert.equal(await page.locator("#deviceId").isEnabled(), true);

    await page.locator("#agentText").fill("执行一个普通通用目标");
    await page.locator("#startSupervisedAgent").click();
    await page.waitForFunction(() => document.querySelector("#goalSummary")?.textContent.includes("任务"));
    assert.equal(requests.start.length, 1);
    assert.equal(requests.capabilityCancel.length, 0);
  } finally {
    await browser.close();
    await new Promise(resolve => server.close(resolve));
  }
});

test("an active capability trial still blocks the ordinary agent", { timeout: 30000 }, async () => {
  Object.values(requests).forEach(items => { items.length = 0; });
  const server = createServer({
    restoredCapabilityTrials: [capabilityTrial(), readOnlyRecoveredCapabilityTrial()],
  });
  const { browser, page } = await launchFixturePage(server);
  try {
    await page.locator("#capabilityBadge").getByText("等待单步确认").waitFor({ timeout: 5000 });
    assert.equal(await page.locator("#startSupervisedAgent").isDisabled(), true);
    assert.equal(await page.locator("#deviceId").isDisabled(), true);
    assert.equal(requests.start.length, 0);
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
    assert.match(goalText, /在支付演示页核对订单并付款/);
    assert.doesNotMatch(goalText, /未命名目标/);
    assert.match(goalText, /2026-08-20-deepseek-typed-task-graph-v4/);
    assert.match(goalText, /revision 1/);
    assert.match(goalText, /会话 · session-browser-action/);
    assert.match(goalText, /确认作用域 · active · 后端 scope 与当前权威任务、观察和动作字段一致/);
    assert.match(goalText, /phone-01/);
    assert.match(goalText, /支付演示 \(payment_demo\)/);
    assert.match(goalText, /付款前使用本地效果确认/);
    assert.match(goalText, /页面显示付款完成/);
    assert.match(goalText, /required=true/);
    assert.match(goalText, /effect_allowed=false/);

    assert.equal(await page.locator("#planList .plan-step.current").count(), 1);
    assert.match(await page.locator("#planList .plan-step.current").innerText(), /确认付款入口可见/);

    const actionText = await page.locator("#actionContent").innerText();
    assert.match(actionText, /点击语义控件/);
    assert.match(actionText, /设置/);
    assert.match(actionText, /element_id settings_icon/);
    assert.match(actionText, /kind=element/);
    assert.match(actionText, /element_id=settings_icon/);
    assert.doesNotMatch(actionText, /bounds=|0\.68|0\.2|0\.86|0\.35/);
    assert.match(actionText, /scene_changed=true/);
    assert.match(actionText, /92%/);
    assert.match(actionText, /可信候选唯一且清晰/);
    assert.match(actionText, /2026-08-14-qwen-visual-decision-v5/);
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
    assert.match(warning, /后端动作 scope 与当前权威任务、观察和动作字段一致/);
    assert.doesNotMatch(warning, /action_digest|fingerprint=/);
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
        effect_ids: [],
        observation_id: "obs_0123456789abcdef0123456789abcdef",
        fingerprint: "51277d0d9e6f986b00dc",
        decision_node_id: "qwen_visual_revision_1",
        action_digest: "a".repeat(64),
      },
    });
    assert.match(await page.locator("#sceneMeta").innerText(), /累计动作\s*1/);
    const traceText = await page.locator("#traceList").innerText();
    assert.match(traceText, /已保存 5 项本地证据（路径不在控制台显示）/);
    assert.doesNotMatch(traceText, /after-1\.jpg|after-2\.jpg|after-3\.jpg|after-4\.jpg/);
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
    assert.equal(await page.locator("#pauseNotice").isVisible(), false);
    assert.equal(await page.locator("#startSupervisedAgent").isDisabled(), false);
    assert.deepEqual(pageErrors, []);
  } finally {
    await browser.close();
    await new Promise(resolve => server.close(resolve));
  }
});

test("typed effect graph executes at most one bound action after one effect approval", { timeout: 30000 }, async () => {
  Object.values(requests).forEach(items => { items.length = 0; });
  const server = createServer();
  const { browser, page } = await launchFixturePage(server);

  try {
    await page.locator("#agentText").fill("需要本地确认的付款演示任务");
    await page.locator("#startSupervisedAgent").click();
    await page.locator("#sessionBadge").getByText("等待效果确认").waitFor();
    assert.equal(await page.locator("#autoSupervisedAgent").count(), 0);
    assert.equal(await page.locator("#reviewAction").count(), 1);
    assert.equal(await page.locator("#reviewAction").innerText(), "查看效果并确认");
    assert.equal(requests.auto.length, 0);
    const goalText = await page.locator("#goalSummary").innerText();
    assert.match(goalText, /确认门 · awaiting_effect_confirmation/);
    assert.match(goalText, /required=true/);
    assert.match(goalText, /效果 payment_order · financial_transaction/);
    assert.match(goalText, /确认作用域 · active · 后端 scope 与当前权威任务、观察和动作字段一致/);
    assert.match(await page.locator("#actionContent").innerText(), /Qwen 唯一动作尚未产生/);
    await page.locator("#reviewAction").click();
    assert.equal(await page.locator("#riskTitle").innerText(), "确认当前登录或付款范围");
    assert.equal(await page.locator("#riskLevel").innerText(), "需要确认 · 仅限登录或付款");
    assert.match(await page.locator("#riskWarning").innerText(), /typed EffectIntent 一致/);
    assert.match(await page.locator("#riskReason").innerText(), /financial_transaction/);
    assert.match(await page.locator("#riskReason").innerText(), /演示商户/);
    assert.match(await page.locator("#riskReason").innerText(), /20元/);
    assert.match(await page.locator("#riskWarning").innerText(), /最多执行一个/);
    const approvalResponse = page.waitForResponse(
      response => response.url().endsWith("/approve-effect"),
      { timeout: 5000 },
    );
    await page.locator("#confirmRiskAction").click();
    await approvalResponse;
    await page.locator("#sessionBadge").getByText("目标完成").waitFor({ timeout: 5000 });
    assert.deepEqual(requests.approveEffect[0], {
      confirmed: true,
      confirmation: {
        session_id: "session-browser-external",
        task_id: "task-map-001",
        device_id: "phone-01",
        revision: 1,
        subgoal_id: "submit_payment",
        effect_ids: ["payment_order"],
        intent_digest: "d".repeat(64),
      },
    });
    assert.equal(requests.confirm.length, 0);
    assert.equal(await page.locator("#reviewAction").isHidden(), true);
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

test("Qwen blocked state renders without executable controls", { timeout: 30000 }, async () => {
  Object.values(requests).forEach(items => { items.length = 0; });
  const server = createServer();
  const { browser, page } = await launchFixturePage(server);
  try {
    await page.locator("#agentText").fill("Qwen blocked");
    await page.locator("#startSupervisedAgent").click();
    await page.locator("#actionContent").getByText("已阻止").waitFor({ timeout: 5000 });
    assert.match(await page.locator("#actionContent").innerText(), /没有可靠且唯一/);
    assert.equal(await page.locator("#actionControls button").count(), 0);
  } finally {
    await browser.close();
    await new Promise(resolve => server.close(resolve));
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
    await page.locator("#capabilityEvidence figure").nth(7).waitFor({ timeout: 5000 });
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
    await page.locator("#capabilityEvidence figure").nth(7).waitFor({ timeout: 5000 });
    assert.equal(await page.locator("#capabilityEvidence figure").count(), 8);
    assert.match(await page.locator("#capabilityStatus").innerText(), /动作后验证未通过/);
  } finally {
    await browser.close();
    await new Promise(resolve => server.close(resolve));
  }
});
