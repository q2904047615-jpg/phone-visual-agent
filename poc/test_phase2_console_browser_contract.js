const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("node:path");
const { pathToFileURL } = require("node:url");
const { chromium } = require("playwright");

const staticRoot = path.join(__dirname, "static");
const qwenFixture = require("./frontend_contract_fixtures/qwen_whole_task_decision.json");
const taskFixture = require("./frontend_contract_fixtures/whole_task.json");
const redacted2bd3Fixture = require("./frontend_contract_fixtures/generic_supervised_v4_redacted.json");

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function traceSession() {
  const priorDecision = clone(qwenFixture.decision);
  const currentDecision = clone(priorDecision);
  currentDecision.observation_id = "obs-after-001";
  currentDecision.fingerprint = "fingerprint-after-001";
  return {
    session_id: "phase2-trace-session",
    device_id: "phone-01",
    status: "awaiting_confirmation",
    ...clone(taskFixture.task),
    step_number: 2,
    trusted_observation: {
      observation_id: "obs-after-001",
      device_id: "phone-01",
      fingerprint: "fingerprint-after-001",
    },
    current_scene: {
      protocol_version: "2026-08-10-ui-scene-v2",
      foreground_app_id: "launcher",
      app_id: "launcher",
      screen_id: "launcher_home",
      summary: "桌面应用网格清晰可见",
      elements: [],
      overlays: [],
      stable: true,
      confidence: 0.95,
      fingerprint: "fingerprint-after-001",
    },
    qwen_decision: currentDecision,
    controller_decision: {
      allowed: true,
      reason: "同响应动作已绑定当前截图。",
      canonical_class: "tap_semantic",
      policy_version: "2026-08-26-canonical-selection-receipt-v1",
    },
    // Deliberately stale: the scope still points to the prior observation.
    confirmation_scope: {
      session_id: "phase2-trace-session",
      task_id: "task-map-001",
      device_id: "phone-01",
      revision: 1,
      step_id: "step_2",
      effect_ids: [],
      observation_id: "obs_0123456789abcdef0123456789abcdef",
      fingerprint: "51277d0d9e6f986b00dc",
      decision_node_id: "qwen_visual_revision_1",
      action_digest: "a".repeat(64),
    },
    confirmation_ready: true,
    physical_actions: 1,
    evidence: ["before-frame.jpg", "after-frame.jpg"],
    history: [{
      step_number: 1,
      task_revision: 1,
      current_step: "识别并操作唯一可信目标",
      step_id: "step_2",
      qwen_decision: priorDecision,
      controller_decision: {
        allowed: true,
        reason: "同响应动作已绑定当前截图。",
        canonical_class: "tap_semantic",
        policy_version: "2026-08-26-canonical-selection-receipt-v1",
      },
      execution: {
        physical_actions: 1,
        action_outcome: "executed",
        visual_outcome: "matched",
        before_scene: { fingerprint: "51277d0d9e6f986b00dc" },
        after_scene: { fingerprint: "fingerprint-after-001" },
        verification_errors: [],
        observation_errors: [],
        evidence: ["after-frame.jpg"],
      },
      after_observation_id: "obs-after-001",
      after_fingerprint: "fingerprint-after-001",
      transition: {
        transition_kind: "new_screenshot_decision",
        outcome: "matched",
      },
    }],
  };
}

function activeActionSession() {
  const session = traceSession();
  session.confirmation_scope.observation_id = "obs-after-001";
  session.confirmation_scope.fingerprint = "fingerprint-after-001";
  return session;
}

async function launchOfflinePage(session, options = {}) {
  const launchOptions = { headless: true };
  if (process.env.BROWSER_EXECUTABLE) launchOptions.executablePath = process.env.BROWSER_EXECUTABLE;
  else if (process.platform === "win32") launchOptions.channel = "msedge";
  const browser = await chromium.launch(launchOptions);
  const page = await browser.newPage({ viewport: options.viewport || { width: 1280, height: 900 } });
  await page.addInitScript(value => {
    const currentSession = value.session;
    const nextSession = value.nextSession || currentSession;
    window.__offlineRequests = [];
    localStorage.setItem("visual-agent-device-id", "phone-01");
    const response = body => new Response(JSON.stringify(body), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
    window.fetch = async (input, options = {}) => {
      const requestPath = String(input);
      window.__offlineRequests.push({
        path: requestPath,
        method: String(options.method || "GET"),
        body: options.body ? JSON.parse(options.body) : null,
      });
      if (requestPath === "/api/session") return response({ token: "offline-token", mock: true });
      if (requestPath === "/api/device") {
        return response({
          controller_online: true,
          camera_online: true,
          busy: false,
          default_device_id: "phone-01",
          devices: [{ device_id: "phone-01", verified_actions: ["tap_semantic"] }],
          generic_supervised_execution: { active_sessions: [] },
          execution_architecture: { universal_agent: { observer: { current_stage: "idle" } } },
        });
      }
      if (requestPath === "/api/capability-acceptance") return response({ trials: [] });
      if (requestPath === "/api/agent/generic-supervised/start-async" && options.method === "POST") {
        return response({ task_id: "offline-start", status: "running" });
      }
      if (requestPath === "/api/agent/generic-supervised/start-async/offline-start") {
        return response({ task_id: "offline-start", status: "completed",
          result: { session: currentSession } });
      }
      if (requestPath.endsWith("/confirm") && options.method === "POST") return response({ session: currentSession });
      if (requestPath.includes("/next") && options.method === "POST") return response({ session: nextSession });
      return new Response(JSON.stringify({ detail: "offline fixture route not found" }), {
        status: 404,
        headers: { "Content-Type": "application/json" },
      });
    };
  }, { session, nextSession: options.nextSession || null });
  await page.route("**/assets/protocol_adapter.js*", route => route.fulfill({
    path: path.join(staticRoot, "protocol_adapter.js"),
    contentType: "text/javascript",
  }));
  await page.route("**/assets/app.js?*", route => route.fulfill({
    path: path.join(staticRoot, "app.js"),
    contentType: "text/javascript",
  }));
  await page.route("**/assets/styles.css?*", route => route.fulfill({
    path: path.join(staticRoot, "styles.css"),
    contentType: "text/css",
  }));
  await page.route("**/api/preview.jpg**", route => route.fulfill({
    body: Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=", "base64"),
    contentType: "image/png",
  }));
  await page.goto(pathToFileURL(path.join(staticRoot, "index.html")).href);
  return { browser, page };
}

test("offline console renders the full phase-two trace and disables a stale scope", { timeout: 30000 }, async () => {
  const { browser, page } = await launchOfflinePage(traceSession());
  const pageErrors = [];
  page.on("pageerror", error => pageErrors.push(error.message));
  try {
    await page.locator("#agentText").fill("执行一个通用视觉目标");
    await page.locator("#startSupervisedAgent").click();
    await page.locator("#traceList [data-trace-phase='current']").waitFor({ timeout: 5000 });

    const goalText = await page.locator("#goalSummary").innerText();
    assert.match(goalText, /revision 1/);
    assert.match(await page.locator("#planList").innerText(), /在支付演示页核对订单并付款/);

    const traceText = await page.locator("#traceList").innerText();
    assert.match(traceText, /步骤 1 · revision 1/);
    assert.match(traceText, /obs_0123456789abcdef0123456789abcdef/);
    assert.match(traceText, /QWEN 唯一动作/);
    assert.match(traceText, /CONTROLLER GATE/);
    assert.match(traceText, /physical_actions 1/);
    assert.match(traceText, /符合预期/);
    assert.match(traceText, /obs-after-001 \/ fingerprint-after-001/);
    assert.match(traceText, /新截图决策/);
    assert.doesNotMatch(traceText, /重规划|action_result_mismatch/);
    assert.match(traceText, /记录没有可验证的确认消费回执/);
    assert.match(traceText, /步骤 2 · revision 1/);
    assert.match(traceText, /作用域字段已变化/);

    const sceneText = await page.locator("#sceneMeta").innerText();
    assert.match(sceneText, /r1 · step_2/);
    assert.match(sceneText, /obs-after-001/);
    assert.match(sceneText, /fingerprint-after-001/);
    assert.match(sceneText, /stale/);
    assert.match(sceneText, /active/);

    assert.equal(await page.locator("#reviewAction").count(), 0);
    assert.match(await page.locator("#nextSupervisedAgent").innerText(), /旧确认已失效/);
    if (process.env.PHASE2_CONSOLE_SCREENSHOT) {
      await page.screenshot({
        path: process.env.PHASE2_CONSOLE_SCREENSHOT,
        fullPage: true,
      });
    }
    assert.deepEqual(pageErrors, []);
  } finally {
    await browser.close();
  }
});

test("offline confirmation is single-shot even when the dialog button is clicked twice", { timeout: 30000 }, async () => {
  const session = activeActionSession();
  const { browser, page } = await launchOfflinePage(session);
  try {
    await page.locator("#agentText").fill("执行一个通用视觉目标");
    await page.locator("#startSupervisedAgent").click();
    await page.locator("#reviewAction").click();
    await page.locator("#riskDialog").waitFor({ state: "visible" });
    await page.locator("#confirmRiskAction").evaluate(button => {
      button.click();
      button.click();
    });
    await page.waitForTimeout(300);

    const confirmRequests = await page.evaluate(() => window.__offlineRequests.filter(item => item.path.endsWith("/confirm")));
    assert.equal(confirmRequests.length, 1);
    assert.deepEqual(confirmRequests[0].body.confirmation, {
      session_id: "phase2-trace-session",
      task_id: "task-map-001",
      device_id: "phone-01",
      revision: 1,
      step_id: "step_2",
      effect_ids: [],
      observation_id: "obs-after-001",
      fingerprint: "fingerprint-after-001",
      decision_node_id: "qwen_visual_revision_1",
      action_digest: "a".repeat(64),
    });
  } finally {
    await browser.close();
  }
});

test("a next observation invalidates the old dialog grant without a confirm request", { timeout: 30000 }, async () => {
  const { browser, page } = await launchOfflinePage(activeActionSession(), { nextSession: traceSession() });
  try {
    await page.locator("#agentText").fill("执行一个通用视觉目标");
    await page.locator("#startSupervisedAgent").click();
    await page.locator("#reviewAction").click();
    await page.locator("#riskDialog").waitFor({ state: "visible" });
    await page.keyboard.press("Escape");
    await page.locator("#nextSupervisedAgent").click();
    await page.locator("#nextSupervisedAgent").filter({ hasText: "旧确认已失效" }).waitFor({ timeout: 5000 });

    const requests = await page.evaluate(() => window.__offlineRequests);
    assert.equal(requests.filter(item => item.path.endsWith("/next")).length, 1);
    assert.equal(requests.filter(item => item.path.endsWith("/confirm")).length, 0);
    assert.equal(await page.locator("#reviewAction").count(), 0);
  } finally {
    await browser.close();
  }
});

test("redacted real session renders terminal trace without coordinates paths tokens or XSS", { timeout: 30000 }, async () => {
  const session = clone(redacted2bd3Fixture.session);
  session.raw_goal = '<img id="injected-xss" src=x onerror="window.__xss=1">';
  const { browser, page } = await launchOfflinePage(session, { viewport: { width: 390, height: 844 } });
  try {
    await page.locator("#agentText").fill("执行一个通用视觉目标");
    await page.locator("#startSupervisedAgent").click();
    await page.locator("#traceList [data-trace-phase='terminal']").waitFor({ timeout: 5000 });

    const bodyText = await page.locator("body").innerText();
    const bodyHtml = await page.locator("body").innerHTML();
    assert.match(bodyText, /终态检查点/);
    assert.match(bodyText, /旧记录未提供/);
    assert.match(bodyText, /证据：已保存 1 项本地证据（路径不在控制台显示）/);
    assert.doesNotMatch(bodyText, /0\.82|0\.02|0\.91|0\.09/);
    assert.doesNotMatch(bodyText, /REDACTED_LOCAL_PATH|after-1\.jpg|sensitive-token-must-not-render/);
    assert.equal(await page.locator("#injected-xss").count(), 0);
    assert.equal(await page.evaluate(() => window.__xss || 0), 0);
    assert.match(bodyHtml, /&lt;img id="injected-xss"/);

    const columns = await page.locator(".trace-grid").first().evaluate(element => getComputedStyle(element).gridTemplateColumns);
    assert.equal(columns.trim().split(/\s+/).length, 1);
    assert.equal(await page.locator("#reviewAction").count(), 0);
    assert.equal(await page.locator("#nextSupervisedAgent").count(), 0);
  } finally {
    await browser.close();
  }
});
