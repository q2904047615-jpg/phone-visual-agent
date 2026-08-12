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
    task_graph: graph,
    qwen_decision: clone(qwenFixture.decision),
    history: [],
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
    taskId: "task-map-001",
    deviceId: "phone-01",
    revision: 1,
    subgoalId: "save_target",
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

  assert.equal(action.protocol, "qwen-visual-decision-v2");
  assert.equal(action.protocolVersion, "2026-08-11-qwen-visual-decision-v2");
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

test("automatic requests omit confirmed and remain limited to one physical action", () => {
  const payload = Protocol.buildAutoRequestPayload("phone-01");
  assert.deepEqual(payload, {
    max_physical_actions: 1,
    device_id: "phone-01",
  });
  assert.equal(Object.prototype.hasOwnProperty.call(payload, "confirmed"), false);
});

test("an explicit confirmation grant is scoped and can be consumed only once", () => {
  const view = Protocol.adaptSession(externalSession());
  const grant = Protocol.createConfirmationGrant(view, "phone-01");
  const payload = Protocol.consumeConfirmationGrant(grant, view, "phone-01");

  assert.deepEqual(payload, {
    confirmed: true,
    confirmation: {
      session_id: "session-cross-contract",
      task_id: "task-map-001",
      device_id: "phone-01",
      revision: 1,
      subgoal_id: "save_target",
      risk_ids: ["save_place"],
    },
  });
  assert.throws(
    () => Protocol.consumeConfirmationGrant(grant, view, "phone-01"),
    /已使用或不存在/,
  );
});

test("confirmation scope cannot cross revision, risk, subgoal, task, or device", () => {
  const original = Protocol.adaptSession(externalSession());
  const mutations = [
    session => { session.task_graph.revision = 2; },
    session => { session.task_graph.confirmation_gate.risk_ids = ["different_risk"]; },
    session => { session.task_graph.current_subgoal.subgoal_id = "different_subgoal"; },
    session => { session.task_graph.task_id = "different-task"; },
  ];

  for (const mutate of mutations) {
    const grant = Protocol.createConfirmationGrant(original, "phone-01");
    const changedRaw = externalSession();
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

test("pause after one safe response prevents the next request and keeps the locked device", async () => {
  let paused = false;
  let session = Protocol.adaptSession(safeActionSession());
  const lockedSessionDeviceId = "phone-01";
  const laterSelectedDeviceId = "phone-02";
  const requests = [];

  const outcome = await Protocol.runAutoAdvanceLoop({
    getContext: () => ({
      session,
      sessionDeviceId: lockedSessionDeviceId,
      paused,
      busy: false,
    }),
    sendOne: async payload => {
      requests.push(payload);
      return safeActionSession();
    },
    applyResponse: async response => {
      session = Protocol.adaptSession(response);
      paused = true;
    },
  });

  assert.equal(outcome.requests, 1);
  assert.equal(requests.length, 1);
  assert.deepEqual(requests[0], {
    max_physical_actions: 1,
    device_id: lockedSessionDeviceId,
  });
  assert.notEqual(requests[0].device_id, laterSelectedDeviceId);
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
  assert.equal(view.visualAction.protocol, "qwen-visual-decision-v2");
  assert.equal(view.visualAction.actionType, "tap_semantic");
  assert.equal(view.visualAction.elementId, "settings_icon");
});
