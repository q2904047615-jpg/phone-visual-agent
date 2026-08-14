const test = require("node:test");
const assert = require("node:assert/strict");

const Protocol = require("./static/protocol_adapter.js");
const deepSeekFixture = require("./frontend_contract_fixtures/deepseek_task_graph_v3.json");
const deepSeekV2Fixture = require("./frontend_contract_fixtures/deepseek_task_graph_v2.json");
const qwenFixture = require("./frontend_contract_fixtures/qwen_visual_decision_v2.json");

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function externalSession({ includeDecision = false } = {}) {
  return {
    session_id: "session-cross-contract",
    task_graph: clone(deepSeekFixture.to_qwen_context),
    ...(includeDecision ? { qwen_decision: clone(qwenFixture.decision) } : {}),
    history: [],
  };
}

function riskApprovalSession() {
  return {
    session_id: "session-risk-approval",
    status: "awaiting_risk_confirmation",
    task_graph: clone(deepSeekFixture.task_graph),
    risk_confirmation_scope: {
      session_id: "session-risk-approval",
      task_id: "task-map-001",
      device_id: "phone-01",
      revision: 1,
      subgoal_id: "save_target",
      risk_ids: ["save_place"],
    },
    risk_confirmation_ready: true,
    physical_actions: 0,
    history: [],
  };
}

function safeActionSession() {
  const graph = clone(deepSeekFixture.task_graph);
  graph.status = "running";
  graph.current_subgoal = clone(graph.subgoals[0]);
  graph.current_subgoal.status = "active";
  graph.subgoals[0].status = "active";
  graph.subgoals[1].status = "pending";
  graph.active_subgoal_id = "locate_target";
  return {
    session_id: "session-safe-action",
    status: "awaiting_confirmation",
    task_graph: graph,
    qwen_decision: clone(qwenFixture.decision),
    controller_decision: {
      allowed: true,
      reason: "仅允许当前通用导航动作。",
      canonical_class: "navigation_open",
      policy_version: "2026-08-12-phase-one-navigation-v1",
    },
    confirmation_scope: {
      session_id: "session-safe-action",
      task_id: "task-map-001",
      device_id: "phone-01",
      revision: 1,
      subgoal_id: "locate_target",
      risk_ids: [],
      observation_id: "obs_0123456789abcdef0123456789abcdef",
      fingerprint: "51277d0d9e6f986b00dc",
    },
    confirmation_ready: true,
    physical_actions: 0,
    evidence: ["before_step_1_frame_1.jpg"],
    history: [],
  };
}

function capabilityTrial({ passed = false } = {}) {
  const session = safeActionSession();
  session.session_id = "capability-session-001";
  session.qwen_decision.next_action.action = "drag";
  session.confirmation_scope = {
    ...session.confirmation_scope,
    session_id: session.session_id,
    device_id: "phone-01",
  };
  const report = passed ? {
    status: "passed",
    trial_id: "trial-001",
    device_id: "phone-01",
    candidate_action: "drag",
    physical_actions: 1,
    before_frame_paths: ["before-1.jpg", "before-2.jpg", "before-3.jpg", "before-4.jpg"],
    after_frame_paths: ["after-1.jpg", "after-2.jpg", "after-3.jpg", "after-4.jpg"],
  } : null;
  return {
    trial_id: "trial-001",
    device_id: "phone-01",
    candidate_action: "drag",
    text: "拖动安全控件",
    code_revision: "330d4c1",
    session,
    action_confirmation_scope: {
      ...session.confirmation_scope,
      trial_id: "trial-001",
      action: "drag",
    },
    risk_confirmation_scope: null,
    report,
    promotion_scope: passed ? {
      trial_id: "trial-001",
      device_id: "phone-01",
      action: "drag",
      report_sha256: "a".repeat(64),
      registry_sha256: "b".repeat(64),
    } : null,
    promotion: null,
    requires_restart: false,
  };
}

test("real DeepSeek 438cd22 to_dict snapshot exposes every formal v3 task graph field", () => {
  const view = Protocol.adaptSession({
    session_id: "session-deepseek-full",
    task_graph: clone(deepSeekFixture.task_graph),
    goal: { objective: "错误的旧目标" },
  });

  assert.equal(view.protocol, "deepseek-task-graph-v3");
  assert.equal(view.protocolVersion, "2026-08-11-deepseek-task-graph-v3");
  assert.equal(view.compatibilityFallback, false);
  assert.equal(view.taskId, "task-map-001");
  assert.equal(view.deviceId, "phone-01");
  assert.equal(view.revision, 1);
  assert.equal(view.status, "awaiting_confirmation");
  assert.equal(view.objective, "在地图应用中找到图书馆并保存地点");
  assert.notEqual(view.objective, "未命名目标");
  assert.deepEqual(view.targetApps.map(item => [item.id, item.name]), [["maps", "地图"]]);
  assert.deepEqual(view.constraints, ["不要发起导航"]);
  assert.match(view.completionConditions[0], /目标地点已保存/);
  assert.deepEqual(view.subgoals.map(item => item.id), ["locate_target", "save_target"]);
  assert.equal(view.currentSubgoal.id, "save_target");
  assert.equal(view.currentSubgoal.externalImpact, "external_state");
  assert.deepEqual(view.risk.actions.map(item => item.id), ["save_place"]);
  assert.equal(view.risk.confirmationGate.required, true);
  assert.equal(view.risk.confirmationGate.state, "awaiting_confirmation");
  assert.deepEqual(view.risk.confirmationGate.riskIds, ["save_place"]);
});

test("real DeepSeek 438cd22 to_qwen_context snapshot keeps v3 gate scope", () => {
  const view = Protocol.adaptSession(externalSession());

  assert.equal(view.protocolVersion, "2026-08-11-deepseek-task-graph-v3");
  assert.equal(view.compatibilityFallback, false);
  assert.equal(view.status, "awaiting_confirmation");
  assert.equal(view.objective, "在地图应用中找到图书馆并保存地点");
  assert.deepEqual(view.constraints, ["不要发起导航"]);
  assert.match(view.completionConditions[0], /目标地点已保存/);
  assert.equal(view.currentSubgoal.id, "save_target");
  assert.equal(view.subgoals.length, 1);
  assert.equal(view.risk.currentExternalImpact, "external_state");
  assert.equal(view.risk.requiresConfirmation, true);
  assert.equal(view.risk.blocksAutomatic, true);
  assert.deepEqual(view.risk.confirmationGate.scope, {
    sessionId: "",
    taskId: "task-map-001",
    deviceId: "phone-01",
    revision: 1,
    subgoalId: "save_target",
    riskIds: [],
    observationId: "",
    fingerprint: "",
  });
});

test("DeepSeek v2 is labelled as explicit compatibility data", () => {
  const view = Protocol.adaptSession({
    session_id: "legacy-v2",
    task_graph: clone(deepSeekV2Fixture.task_graph),
  });
  assert.equal(view.protocol, "deepseek-task-graph-v2-compatibility");
  assert.equal(view.compatibilityFallback, true);
});

test("real Qwen decision.to_dict snapshot exposes the complete unique next action", () => {
  assert.equal(qwenFixture.source.commit, "af9b6e7e6c4525980430e01dd82b59ce47d9fb81");
  assert.equal(
    qwenFixture.source.input_task_context_protocol,
    "2026-08-11-deepseek-task-graph-v3",
  );
  const view = Protocol.adaptSession(safeActionSession());
  const action = view.visualAction;

  assert.equal(action.protocol, "qwen-visual-decision-v3");
  assert.equal(action.protocolVersion, "2026-08-12-qwen-visual-decision-v3");
  assert.equal(action.status, "action");
  assert.equal(action.actionType, "tap_semantic");
  assert.equal(action.semanticTarget, "设置");
  assert.equal(action.elementId, "settings_icon");
  assert.deepEqual(action.targetRegion, {
    kind: "element",
    element_id: "settings_icon",
    bounds: [0.68, 0.2, 0.86, 0.35],
    description: "设置",
  });
  assert.deepEqual(action.expectedChange, { scene_changed: true });
  assert.equal(action.confidence, 0.92);
  assert.equal(action.reason, "可信候选唯一且清晰。");
  assert.equal(action.taskId, "task-map-001");
  assert.equal(action.revision, 1);
  assert.equal(action.observationId, "obs_0123456789abcdef0123456789abcdef");
  assert.equal(action.fingerprint, "51277d0d9e6f986b00dc");
  assert.equal(action.identityMatchesTask, true);
  assert.equal(action.isExecutable, true);
});

test("current Qwen v4 decision is labelled without losing the bound action", () => {
  const session = safeActionSession();
  session.qwen_decision.protocol_version = "2026-08-14-qwen-visual-decision-v4";

  const action = Protocol.adaptSession(session).visualAction;

  assert.equal(action.protocol, "qwen-visual-decision-v4");
  assert.equal(action.protocolVersion, "2026-08-14-qwen-visual-decision-v4");
  assert.equal(action.actionType, "tap_semantic");
  assert.equal(action.isExecutable, true);
});

test("Qwen v2 blocked and finished decisions never become executable actions", () => {
  for (const status of ["blocked", "finished"]) {
    const session = safeActionSession();
    session.qwen_decision.status = status;
    session.qwen_decision.next_action = null;
    session.qwen_decision.target_region = null;
    session.qwen_decision.expected_result = {};
    session.qwen_decision.reason = status === "blocked"
      ? "没有可靠且唯一的可信候选。"
      : "当前可见证据已满足目标。";
    const view = Protocol.adaptSession(session);
    assert.equal(view.visualAction.status, status);
    assert.equal(view.visualAction.actionType, "");
    assert.equal(view.visualAction.isExecutable, false);
    assert.equal(view.visualAction.physicalActionPossible, false);
    assert.equal(Protocol.shouldAutoAdvance({ session: view, paused: false, busy: false }), false);
  }
});

test("external-state context without a Qwen action can never auto-confirm", () => {
  const view = Protocol.adaptSession(externalSession());
  assert.equal(view.visualAction.actionType, "");
  assert.equal(view.status, "awaiting_confirmation");
  assert.equal(Protocol.shouldAutoAdvance({ session: view, paused: false, busy: false }), false);
});

test("each confirmation and impact signal independently closes automatic advance", () => {
  const mutations = [
    view => { view.risk.requiresConfirmation = true; },
    view => { view.status = "awaiting_confirmation"; },
    view => { view.risk.hasCurrentRisk = true; },
    view => { view.risk.currentExternalImpact = "external_state"; },
    view => { view.risk.currentExternalImpact = "unknown"; },
  ];
  for (const mutate of mutations) {
    const view = Protocol.adaptSession(safeActionSession());
    view.risk.blocksAutomatic = false;
    mutate(view);
    assert.equal(
      Protocol.shouldAutoAdvance({ session: view, paused: false, busy: false }),
      false,
    );
  }
});

test("phase one never auto advances physical actions", async () => {
  const view = Protocol.adaptSession(safeActionSession());
  assert.equal(
    Protocol.shouldAutoAdvance({ session: view, paused: false, busy: false }),
    false,
  );
  let sent = 0;
  const outcome = await Protocol.runAutoAdvanceLoop({
    getContext: () => ({ session: view, paused: false, busy: false }),
    sendOne: async () => { sent += 1; },
    applyResponse: async () => {},
  });
  assert.deepEqual(outcome, { requests: 0, limitReached: false });
  assert.equal(sent, 0);
});

test("an explicit confirmation grant is scoped and can be consumed only once", () => {
  const view = Protocol.adaptSession(safeActionSession());
  const grant = Protocol.createConfirmationGrant(view, "phone-01");
  const payload = Protocol.consumeConfirmationGrant(grant, view, "phone-01");

  assert.deepEqual(payload, {
    confirmed: true,
    confirmation: {
      session_id: "session-safe-action",
      task_id: "task-map-001",
      device_id: "phone-01",
      revision: 1,
      subgoal_id: "locate_target",
      risk_ids: [],
      observation_id: "obs_0123456789abcdef0123456789abcdef",
      fingerprint: "51277d0d9e6f986b00dc",
    },
  });
  assert.throws(
    () => Protocol.consumeConfirmationGrant(grant, view, "phone-01"),
    /已使用或不存在/,
  );
});

test("risk approval grant excludes observation and cannot execute a physical action", () => {
  const view = Protocol.adaptSession(riskApprovalSession());
  assert.equal(view.status, "awaiting_risk_confirmation");
  assert.equal(view.risk.confirmationGate.phase, "risk");
  assert.equal(view.visualAction.actionType, "");
  const grant = Protocol.createConfirmationGrant(view, "phone-01");
  assert.equal(grant.phase, "risk");
  assert.deepEqual(Protocol.consumeConfirmationGrant(grant, view, "phone-01"), {
    confirmed: true,
    confirmation: {
      session_id: "session-risk-approval",
      task_id: "task-map-001",
      device_id: "phone-01",
      revision: 1,
      subgoal_id: "save_target",
      risk_ids: ["save_place"],
    },
  });
});

test("compatibility auto payload is hard-bounded to one exact confirmed action", () => {
  const view = Protocol.adaptSession(safeActionSession());
  const confirmation = Protocol.consumeConfirmationGrant(
    Protocol.createConfirmationGrant(view, "phone-01"),
    view,
    "phone-01",
  );
  const payload = Protocol.buildAutoRequestPayload(
    "phone-01",
    confirmation,
    { maxPhysicalActions: 3, maxIterations: 8 },
  );
  assert.equal(payload.confirmed, true);
  assert.equal(payload.device_id, "phone-01");
  assert.equal(payload.max_physical_actions, 1);
  assert.equal(payload.max_iterations, 1);
  assert.equal(payload.confirmation.observation_id, "obs_0123456789abcdef0123456789abcdef");
  assert.throws(
    () => Protocol.buildAutoRequestPayload("phone-01", { confirmed: false }),
    /精确动作确认/,
  );
});

test("confirmation scope cannot cross revision, risk, subgoal, task, device, observation, or fingerprint", () => {
  const original = Protocol.adaptSession(safeActionSession());
  const mutations = [
    session => { session.task_graph.revision = 2; },
    session => { session.confirmation_scope.risk_ids = ["different_risk"]; },
    session => { session.confirmation_scope.subgoal_id = "different_subgoal"; },
    session => { session.task_graph.task_id = "different-task"; },
    session => { session.confirmation_scope.observation_id = "obs-changed"; },
    session => { session.confirmation_scope.fingerprint = "frame-changed"; },
  ];

  for (const mutate of mutations) {
    const grant = Protocol.createConfirmationGrant(original, "phone-01");
    const changedRaw = safeActionSession();
    mutate(changedRaw);
    const changed = Protocol.adaptSession(changedRaw);
    assert.throws(
      () => Protocol.consumeConfirmationGrant(grant, changed, "phone-01"),
      /已经变化/,
    );
  }

  const deviceGrant = Protocol.createConfirmationGrant(original, "phone-01");
  assert.throws(
    () => Protocol.consumeConfirmationGrant(deviceGrant, original, "phone-02"),
    /已经变化/,
  );
});

test("adapter exposes controller gate action count and evidence", () => {
  const view = Protocol.adaptSession(safeActionSession());
  assert.deepEqual(view.controllerGate, {
    allowed: true,
    reason: "仅允许当前通用导航动作。",
    canonicalClass: "navigation_open",
    policyVersion: "2026-08-12-phase-one-navigation-v1",
  });
  assert.equal(view.physicalActions, 0);
  assert.deepEqual(view.evidence, ["before_step_1_frame_1.jpg"]);
});

test("current Qwen v2 fields win over conflicting legacy fallback data after a v3 graph", () => {
  const session = safeActionSession();
  session.current_action = {
    action_type: "swipe",
    semantic_target: "错误旧目标",
    physical_action_possible: false,
  };
  session.proposal = {
    action: { action: "back", params: { target: "错误旧动作" } },
  };
  const view = Protocol.adaptSession(session);
  assert.equal(view.visualAction.protocol, "qwen-visual-decision-v3");
  assert.equal(view.visualAction.actionType, "tap_semantic");
  assert.equal(view.visualAction.elementId, "settings_icon");
});

test("capability action grant binds trial action and exact visual scope once", () => {
  const trial = capabilityTrial();
  const view = Protocol.adaptCapabilityTrial(trial);
  assert.equal(view.trialId, "trial-001");
  assert.equal(view.action, "drag");
  assert.equal(view.physicalActions, 0);
  const grant = Protocol.createCapabilityConfirmationGrant(trial);
  assert.deepEqual(Protocol.consumeCapabilityConfirmationGrant(grant, trial), {
    confirmed: true,
    confirmation: trial.action_confirmation_scope,
  });
  assert.throws(
    () => Protocol.consumeCapabilityConfirmationGrant(grant, trial),
    /已使用或不存在/,
  );
});

test("capability confirmation refuses trial action observation or decision drift", () => {
  const mutations = [
    trial => { trial.trial_id = "trial-changed"; },
    trial => { trial.candidate_action = "long_press"; },
    trial => { trial.action_confirmation_scope.observation_id = "obs-changed"; },
    trial => { trial.session.qwen_decision.next_action.action = "long_press"; },
  ];
  for (const mutate of mutations) {
    const original = capabilityTrial();
    const grant = Protocol.createCapabilityConfirmationGrant(original);
    const changed = capabilityTrial();
    mutate(changed);
    assert.throws(
      () => Protocol.consumeCapabilityConfirmationGrant(grant, changed),
      /缺少|变化/,
    );
  }
});

test("promotion grant requires a passed report and exact immutable hashes", () => {
  const trial = capabilityTrial({ passed: true });
  const grant = Protocol.createPromotionGrant(trial);
  assert.deepEqual(Protocol.consumePromotionGrant(grant, trial), {
    confirmed: true,
    ...trial.promotion_scope,
  });
  assert.throws(() => Protocol.consumePromotionGrant(grant, trial), /已使用或不存在/);
  assert.throws(() => Protocol.createPromotionGrant(capabilityTrial()), /未通过/);

  const changed = capabilityTrial({ passed: true });
  const changedGrant = Protocol.createPromotionGrant(changed);
  changed.promotion_scope.report_sha256 = "c".repeat(64);
  assert.throws(
    () => Protocol.consumePromotionGrant(changedGrant, changed),
    /摘要已经变化/,
  );
});
