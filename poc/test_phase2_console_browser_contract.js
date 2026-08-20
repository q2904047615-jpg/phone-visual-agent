const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("node:path");
const { pathToFileURL } = require("node:url");
const { chromium } = require("playwright");

const staticRoot = path.join(__dirname, "static");
const qwenFixture = require("./frontend_contract_fixtures/qwen_visual_decision_v4.json");
const deepSeekFixture = require("./frontend_contract_fixtures/deepseek_typed_task_graph_v4.json");
const redacted2bd3Fixture = require("./frontend_contract_fixtures/generic_supervised_v4_redacted.json");

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function traceSession() {
  const graph = clone(deepSeekFixture.task_graph);
  graph.status = "running";
  graph.revision = 2;
  graph.active_subgoal_id = "locate_target";
  graph.subgoals[0].status = "active";
  graph.subgoals[1].status = "pending";
  graph.current_subgoal = clone(graph.subgoals[0]);
  graph.replan_history = [{
    revision: 2,
    trigger: "action_result_mismatch",
    reason: "动作后可见结果与预期不一致，重新规划当前子目标。",
    scene_id: "obs-after-001",
    evidence: ["结果仍未满足完成条件"],
  }];
  const priorDecision = clone(qwenFixture.decision);
  const currentDecision = clone(priorDecision);
  currentDecision.revision = 2;
  currentDecision.observation_id = "obs-after-001";
  currentDecision.fingerprint = "fingerprint-after-001";
  currentDecision.trusted_observation.observation_id = "obs-after-001";
  currentDecision.trusted_observation.fingerprint = "fingerprint-after-001";
  return {
    session_id: "phase2-trace-session",
    status: "awaiting_confirmation",
    step_number: 2,
    task_graph: graph,
    qwen_decision: currentDecision,
    controller_decision: {
      allowed: true,
      reason: "当前唯一动作通过本地策略。",
      canonical_class: "navigation_open",
      policy_version: "policy-v1",
    },
    // Deliberately stale: the task graph and current Qwen decision are revision 2.
    confirmation_scope: {
      session_id: "phase2-trace-session",
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
    confirmation_ready: true,
    physical_actions: 1,
    evidence: ["before-frame.jpg", "after-frame.jpg"],
    history: [{
      step_number: 1,
      task_revision: 1,
      current_subgoal: "识别并操作唯一可信目标",
      subgoal_id: "locate_target",
      qwen_decision: priorDecision,
      controller_decision: {
        allowed: true,
        reason: "唯一动作通过本地策略。",
        canonical_class: "navigation_open",
        policy_version: "policy-v1",
      },
      execution: {
        physical_actions: 1,
        action_outcome: "mismatched",
        before_scene: { fingerprint: "51277d0d9e6f986b00dc" },
        after_scene: { fingerprint: "fingerprint-after-001" },
        verification_errors: ["目标状态没有按预期改变"],
        observation_errors: [],
        evidence: ["after-frame.jpg"],
      },
      after_observation_id: "obs-after-001",
      after_fingerprint: "fingerprint-after-001",
    }],
  };
}

function activeActionSession() {
  const session = traceSession();
  session.confirmation_scope.revision = 2;
  session.confirmation_scope.observation_id = "obs-after-001";
  session.confirmation_scope.fingerprint = "fingerprint-after-001";
  return session;
}

async function launchOfflinePage(session, options = {}) {
  const launchOptions = { headless: true };
  if (process.env.BROWSER_EXECUTABLE) launchOptions.executablePath = process.env.BROWSER_EXECUTABLE;
  else launchOptions.channel = "msedge";
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
      if (requestPath === "/api/agent/generic-supervised/start" && options.method === "POST") {
        return response({ session: currentSession });
      }
      if (requestPath.endsWith("/confirm") && options.method === "POST") return response({ session: currentSession });
      if (requestPath.includes("/next") && options.method === "POST") return response({ session: nextSession });
      return new Response(JSON.stringify({ detail: "offline fixture route not found" }), {
        status: 404,
        headers: { "Content-Type": "application/json" },
      });
    };
  }, { session, nextSession: options.nextSession || null });
  await page.route("**/assets/protocol_adapter.js", route => route.fulfill({
    path: path.join(staticRoot, "protocol_adapter.js"),
    contentType: "text/javascript",
  }));
  await page.route("**/assets/app.js", route => route.fulfill({
    path: path.join(staticRoot, "app.js"),
    contentType: "text/javascript",
  }));
  await page.route("**/assets/styles.css", route => route.fulfill({
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
    assert.match(goalText, /revision 2/);
    assert.match(await page.locator("#planList").innerText(), /确认付款入口可见/);

    const traceText = await page.locator("#traceList").innerText();
    assert.match(traceText, /步骤 1 · revision 1/);
    assert.match(traceText, /obs_0123456789abcdef0123456789abcdef/);
    assert.match(traceText, /QWEN 唯一动作/);
    assert.match(traceText, /CONTROLLER GATE/);
    assert.match(traceText, /physical_actions 1/);
    assert.match(traceText, /不符合预期/);
    assert.match(traceText, /obs-after-001 \/ fingerprint-after-001/);
    assert.match(traceText, /重规划/);
    assert.match(traceText, /action_result_mismatch/);
    assert.match(traceText, /记录没有可验证的确认消费回执/);
    assert.match(traceText, /步骤 2 · revision 2/);
    assert.match(traceText, /作用域字段已变化/);

    const sceneText = await page.locator("#sceneMeta").innerText();
    assert.match(sceneText, /r2 · locate_target/);
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
      revision: 2,
      subgoal_id: "locate_target",
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
  session.task_graph.goal.objective = '<img id="injected-xss" src=x onerror="window.__xss=1">';
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
