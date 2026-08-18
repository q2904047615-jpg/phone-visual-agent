const test = require("node:test");
const assert = require("node:assert/strict");

const Protocol = require("./static/protocol_adapter.js");
const deepSeekFixture = require("./frontend_contract_fixtures/deepseek_task_graph_v3.json");
const deepSeekV2Fixture = require("./frontend_contract_fixtures/deepseek_task_graph_v2.json");
const qwenFixture = require("./frontend_contract_fixtures/qwen_visual_decision_v2.json");
const redacted2bd3Fixture = require("./frontend_contract_fixtures/generic_supervised_2bd3_redacted.json");

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
      intent_digest: "d".repeat(64),
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
      decision_node_id: "qwen_visual_revision_1",
      action_digest: "a".repeat(64),
    },
    confirmation_ready: true,
    physical_actions: 0,
    evidence: ["before_step_1_frame_1.jpg"],
    history: [],
  };
}

function phaseTwoTraceSession() {
  const session = safeActionSession();
  const priorDecision = clone(session.qwen_decision);
  session.task_graph.revision = 2;
  session.task_graph.replan_history = [{
    revision: 2,
    trigger: "action_result_mismatch",
    reason: "动作后可见结果与预期不一致，重新规划当前子目标。",
    scene_id: "obs-after-001",
    evidence: ["结果仍未满足完成条件"],
  }];
  session.qwen_decision.revision = 2;
  session.qwen_decision.observation_id = "obs-after-001";
  session.qwen_decision.fingerprint = "fingerprint-after-001";
  session.qwen_decision.trusted_observation.observation_id = "obs-after-001";
  session.qwen_decision.trusted_observation.fingerprint = "fingerprint-after-001";
  session.confirmation_scope.revision = 2;
  session.confirmation_scope.observation_id = "obs-after-001";
  session.confirmation_scope.fingerprint = "fingerprint-after-001";
  session.physical_actions = 1;
  session.history = [{
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
      evidence: ["after-frame-1.jpg"],
      after_frame_paths: ["after-frame-1.jpg", "after-frame-2.jpg"],
    },
    after_observation_id: "obs-after-001",
    after_fingerprint: "fingerprint-after-001",
  }];
  return session;
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
    decisionNodeId: "",
    actionDigest: "",
    intentDigest: "",
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
      decision_node_id: "qwen_visual_revision_1",
      action_digest: "a".repeat(64),
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
      intent_digest: "d".repeat(64),
    },
  });
});

test("safe auto payload is unconfirmed and bounded by explicit budgets", () => {
  const payload = Protocol.buildAutoRequestPayload(
    "phone-01",
    { maxPhysicalActions: 3, maxIterations: 8 },
  );
  assert.equal(payload.confirmed, false);
  assert.equal(payload.device_id, "phone-01");
  assert.equal(payload.max_physical_actions, 3);
  assert.equal(payload.max_iterations, 8);
  assert.equal(payload.confirmation, null);

  const capped = Protocol.buildAutoRequestPayload(
    "phone-01",
    { maxPhysicalActions: 999, maxIterations: 999 },
  );
  assert.equal(capped.max_physical_actions, 20);
  assert.equal(capped.max_iterations, 40);
});

test("confirmation scope cannot cross any action authority field", () => {
  const original = Protocol.adaptSession(safeActionSession());
  const mutations = [
    session => { session.task_graph.revision = 2; },
    session => { session.confirmation_scope.session_id = "different-session"; },
    session => { session.confirmation_scope.risk_ids = ["different_risk"]; },
    session => { session.confirmation_scope.subgoal_id = "different_subgoal"; },
    session => { session.task_graph.task_id = "different-task"; },
    session => { session.confirmation_scope.observation_id = "obs-changed"; },
    session => { session.confirmation_scope.fingerprint = "frame-changed"; },
    session => { session.confirmation_scope.decision_node_id = "different-node"; },
    session => { session.confirmation_scope.action_digest = "b".repeat(64); },
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

test("phase two trace exposes each authority boundary without inventing success", () => {
  const view = Protocol.adaptSession(phaseTwoTraceSession());

  assert.equal(view.taskGraph.revision, 2);
  assert.equal(view.currentSubgoal.id, "locate_target");
  assert.equal(view.executionTrace.length, 2);

  const completed = view.executionTrace[0];
  assert.equal(completed.graphRevision, 1);
  assert.deepEqual(completed.observation, {
    id: "obs_0123456789abcdef0123456789abcdef",
    fingerprint: "51277d0d9e6f986b00dc",
  });
  assert.equal(completed.action.status, "action");
  assert.equal(completed.action.actionType, "tap_semantic");
  assert.equal(completed.controllerGate.allowed, true);
  assert.equal(completed.physicalActions, 1);
  assert.equal(completed.verification.outcome, "mismatched");
  assert.equal(completed.verification.matched, false);
  assert.equal(completed.verification.afterObservationId, "obs-after-001");
  assert.equal(completed.verification.afterFingerprint, "fingerprint-after-001");
  assert.deepEqual(completed.verification.errors, ["目标状态没有按预期改变"]);
  assert.equal(completed.transition.kind, "replan");
  assert.equal(completed.transition.trigger, "action_result_mismatch");
  assert.equal(completed.transition.fromRevision, 1);
  assert.equal(completed.transition.toRevision, 2);
  assert.match(completed.transition.reason, /重新规划/);
  assert.equal(completed.scopeState.state, "unknown");
  assert.match(completed.scopeState.reason, /没有可验证的确认消费回执/);

  const current = view.executionTrace[1];
  assert.equal(current.phase, "current");
  assert.equal(current.graphRevision, 2);
  assert.equal(current.observation.id, "obs-after-001");
  assert.equal(current.observation.fingerprint, "fingerprint-after-001");
  assert.equal(current.physicalActions, 0);
  assert.equal(current.scopeState.state, "active");
  assert.equal(view.physicalActions, 1);
});

test("stale or incomplete action scope is visible and cannot mint a confirmation grant", () => {
  for (const mutate of [
    session => { session.confirmation_scope.revision = 1; },
    session => { delete session.confirmation_scope.fingerprint; },
    session => { delete session.confirmation_scope.decision_node_id; },
    session => { delete session.confirmation_scope.action_digest; },
    session => { session.confirmation_scope.observation_id = "obs-stale"; },
  ]) {
    const session = phaseTwoTraceSession();
    mutate(session);
    const view = Protocol.adaptSession(session);
    assert.notEqual(view.scopeState.state, "active");
    assert.throws(
      () => Protocol.createConfirmationGrant(view, "phone-01"),
      /已经变化|缺失/,
    );
    assert.equal(view.executionTrace.at(-1).scopeState.state, view.scopeState.state);
  }
});

test("terminal and legacy records keep stop and unknown evidence explicit", () => {
  const terminal = phaseTwoTraceSession();
  terminal.status = "blocked";
  terminal.failed_reason = "重规划后的任务图没有活动子目标。";
  terminal.confirmation_scope = null;
  const stopped = Protocol.adaptSession(terminal);
  assert.equal(stopped.stopState.stopped, true);
  assert.equal(stopped.stopState.status, "blocked");
  assert.match(stopped.stopState.reason, /没有活动子目标/);
  assert.equal(stopped.scopeState.state, "invalidated");
  assert.equal(stopped.executionTrace.at(-1).transition.kind, "stopped");
  assert.equal(stopped.executionTrace.at(-1).phase, "terminal");
  assert.equal(stopped.executionTrace.at(-1).verification.outcome, "terminal");
  assert.notEqual(stopped.executionTrace.at(-1).verification.outcome, "awaiting_next_action");

  terminal.failed_reason = "";
  terminal.auto_pause_reason = "本地控制器已安全暂停";
  assert.equal(Protocol.adaptSession(terminal).stopState.reason, "本地控制器已安全暂停");

  const legacy = safeActionSession();
  legacy.history = [{ step_number: 1, execution: { physical_actions: 1 } }];
  const legacyRound = Protocol.adaptSession(legacy).executionTrace[0];
  assert.equal(legacyRound.verification.outcome, "unknown");
  assert.equal(legacyRound.controllerGate.reason, "");
  assert.equal(legacyRound.transition.kind, "unknown");
});

test("redacted real 2bd3 shape keeps unknown history authority and exact replan binding", () => {
  const view = Protocol.adaptSession(clone(redacted2bd3Fixture.session));
  const completed = view.executionTrace[0];
  const terminal = view.executionTrace.at(-1);

  assert.equal(view.protocol, "deepseek-task-graph-v3");
  assert.equal(view.stopState.status, "cancelled");
  assert.equal(completed.subgoal, "旧记录未提供");
  assert.equal(completed.controllerGate.reason, "");
  assert.equal(completed.verification.outcome, "matched");
  assert.equal(completed.verification.afterObservationId, "obs-redacted-after");
  assert.equal(completed.transition.kind, "replan");
  assert.equal(completed.transition.toRevision, 2);
  assert.equal(completed.scopeState.state, "unknown");
  assert.equal(terminal.phase, "terminal");
  assert.equal(terminal.verification.outcome, "terminal");

  const unrelated = clone(redacted2bd3Fixture.session);
  unrelated.task_graph.replan_history = [{
    revision: 3,
    trigger: "observation_changed",
    scene_id: "unrelated-observation",
  }];
  const unrelatedHistory = Protocol.adaptSession(unrelated).executionTrace[0];
  assert.equal(unrelatedHistory.transition.kind, "unknown");
  assert.equal(unrelatedHistory.transition.toRevision, null);
});

test("history is consumed only with a complete authoritative receipt", () => {
  const raw = phaseTwoTraceSession();
  const history = raw.history[0];
  history.risk_ids = [];
  history.confirmation_receipt = {
    authoritative: true,
    consumed: true,
    scope: {
      session_id: raw.session_id,
      task_id: history.qwen_decision.task_id,
      device_id: history.qwen_decision.device_id,
      revision: history.task_revision,
      subgoal_id: history.subgoal_id,
      risk_ids: [],
      observation_id: history.qwen_decision.observation_id,
      fingerprint: history.qwen_decision.fingerprint,
      decision_node_id: history.qwen_decision.next_action.node_id,
      action_digest: "receipt-bound-digest",
    },
  };
  assert.equal(Protocol.adaptSession(raw).executionTrace[0].scopeState.state, "consumed");

  history.confirmation_receipt.scope.session_id = "different-session";
  assert.equal(Protocol.adaptSession(raw).executionTrace[0].scopeState.state, "unknown");
});

test("history outcome and terminal session status never invent a transition", () => {
  const mismatched = phaseTwoTraceSession();
  mismatched.task_graph.replan_history = [];
  mismatched.history[0].execution.action_outcome = "mismatched";
  mismatched.history[0].reason = "结果不符合预期";
  const mismatchView = Protocol.adaptSession(mismatched);
  assert.equal(mismatchView.executionTrace[0].verification.outcome, "mismatched");
  assert.equal(mismatchView.executionTrace[0].transition.kind, "unknown");
  assert.equal(mismatchView.executionTrace[0].transition.reason, "");

  const blocked = phaseTwoTraceSession();
  blocked.task_graph.replan_history = [];
  blocked.status = "blocked";
  blocked.failed_reason = "当前任务已阻止";
  blocked.confirmation_scope = null;
  const blockedView = Protocol.adaptSession(blocked);
  assert.equal(blockedView.executionTrace[0].transition.kind, "unknown");
  assert.equal(blockedView.executionTrace.at(-1).phase, "terminal");
  assert.equal(blockedView.executionTrace.at(-1).transition.kind, "stopped");
});

test("only an explicit authoritative transition can bypass replan-history binding", () => {
  const raw = phaseTwoTraceSession();
  raw.task_graph.replan_history = [];
  raw.history[0].transition = {
    kind: "advance",
    reason: "权威历史记录明确推进到下一子目标",
  };
  const transition = Protocol.adaptSession(raw).executionTrace[0].transition;
  assert.equal(transition.kind, "advance");
  assert.match(transition.reason, /权威历史记录明确推进/);
});

test("risk scope rejects action-only authority fields", () => {
  const raw = riskApprovalSession();
  raw.risk_confirmation_scope.action_digest = "a".repeat(64);
  const view = Protocol.adaptSession(raw);
  assert.equal(view.scopeState.state, "stale");
  assert.match(view.scopeState.mismatches.join(" "), /risk_scope_extra_action_fields/);
  assert.throws(() => Protocol.createConfirmationGrant(view, "phone-01"), /已经变化|缺失/);
});

test("risk scope binds the exact intent digest and rejects drift", () => {
  const raw = riskApprovalSession();
  const view = Protocol.adaptSession(raw);
  const grant = Protocol.createConfirmationGrant(view, "phone-01");
  assert.equal(grant.scope.intent_digest, "d".repeat(64));

  const drifted = clone(raw);
  drifted.risk_confirmation_scope.intent_digest = "e".repeat(64);
  assert.throws(
    () => Protocol.consumeConfirmationGrant(grant, Protocol.adaptSession(drifted)),
    /已经变化|不一致|失效/,
  );
});

test("all typed effects expose the same target payload policy and digest preview", () => {
  const raw = riskApprovalSession();
  raw.effect_previews = [
    {
      protocol_version: "2026-08-19-effect-preview-v1",
      effect_id: "effect_send",
      effect_kind: "send_message",
      targets: [{ entity_ref: "recipient_1", role: "recipient", entity_type: "person", value: "张三" }],
      payloads: [{ entity_ref: "text_1", role: "input_text", entity_type: "text", value: "你好" }],
      policy: "automatic",
      policy_id: "local-risk-policy",
      policy_version: 1,
      expected_result_texts: ["消息可见"],
      preview_digest: "f".repeat(64),
    },
  ];
  const previews = Protocol.adaptSession(raw).risk.effectPreviews;
  assert.equal(previews.length, 1);
  assert.equal(previews[0].effect_kind, "send_message");
  assert.equal(previews[0].targets[0].value, "张三");
  assert.equal(previews[0].payloads[0].value, "你好");
  assert.equal(previews[0].preview_digest, "f".repeat(64));
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
