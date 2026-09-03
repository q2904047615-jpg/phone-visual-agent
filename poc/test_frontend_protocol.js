const test = require("node:test");
const assert = require("node:assert/strict");

const Protocol = require("./static/protocol_adapter.js");
const deepSeekFixture = require("./frontend_contract_fixtures/deepseek_typed_task_graph_v4.json");
const qwenFixture = require("./frontend_contract_fixtures/qwen_visual_decision_v4.json");

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function navigationSession() {
  const graph = clone(deepSeekFixture.task_graph);
  graph.status = "running";
  graph.subgoals[0].status = "active";
  graph.subgoals[1].status = "pending";
  graph.active_subgoal_id = "locate_target";
  graph.current_subgoal = clone(graph.subgoals[0]);
  return {
    session_id: "session-navigation-v4",
    status: "awaiting_confirmation",
    task_graph: graph,
    qwen_decision: clone(qwenFixture.decision),
    confirmation_scope: {
      session_id: "session-navigation-v4",
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
    physical_actions: 0,
    history: [],
  };
}

function effectSession() {
  return {
    session_id: "session-effect-v4",
    status: "awaiting_effect_confirmation",
    task_graph: clone(deepSeekFixture.task_graph),
    effect_confirmation_scope: {
      session_id: "session-effect-v4",
      task_id: "task-map-001",
      device_id: "phone-01",
      revision: 1,
      subgoal_id: "submit_payment",
      effect_ids: ["payment_order"],
      intent_digest: "d".repeat(64),
    },
    effect_confirmation_ready: true,
    physical_actions: 0,
    history: [],
  };
}

test("typed v4 graph is the only formal executable DeepSeek protocol", () => {
  const view = Protocol.adaptSession(navigationSession());
  assert.equal(view.protocol, "deepseek-typed-task-graph-v4");
  assert.equal(view.compatibilityFallback, false);
  assert.equal(view.currentSubgoal.executionClass, "navigate");
});

test("same-response current Qwen contract is the only executable visual decision protocol", () => {
  const view = Protocol.adaptSession(navigationSession());
  assert.equal(view.visualAction.protocol, "qwen-same-response-action-finish-v9");
  assert.equal(view.visualAction.status, "action");
  assert.equal(view.visualAction.isExecutable, true);
});

test("retired Qwen protocol cannot mint an executable action", () => {
  const raw = navigationSession();
  raw.qwen_decision.protocol_version = "2026-08-14-qwen-visual-decision-v5";
  const view = Protocol.adaptSession(raw);
  assert.equal(view.visualAction.protocol, "unsupported-protocol");
  assert.equal(view.visualAction.status, "unknown");
  assert.equal(view.visualAction.isExecutable, false);
  assert.throws(() => Protocol.createConfirmationGrant(view, "phone-01"), /确认作用域/);
});

test("retired protocol version is unsupported and cannot mint authority", () => {
  const raw = navigationSession();
  raw.task_graph.protocol_version = "retired-deepseek-protocol";
  const view = Protocol.adaptSession(raw);
  assert.equal(view.protocol, "unsupported-protocol");
  assert.equal(view.compatibilityFallback, true);
  assert.throws(() => Protocol.createConfirmationGrant(view, "phone-01"), /typed v4/);
});

test("retired transport fields are rejected even when version is forged to v4", () => {
  const raw = navigationSession();
  raw.task_graph.risk_actions = [];
  const view = Protocol.adaptSession(raw);
  assert.equal(view.protocol, "unsupported-protocol");
  assert.throws(() => Protocol.createConfirmationGrant(view, "phone-01"), /typed v4/);
});

test("typed effect intent exposes only kind roles expected results and local policy", () => {
  const view = Protocol.adaptSession(effectSession());
  assert.equal(view.effectPolicy.currentExecutionClass, "effect");
  assert.equal(view.effectPolicy.currentActions[0].kind, "financial_transaction");
  assert.deepEqual(view.effectPolicy.currentActions[0].expectedResults, ["订单进入已付款状态"]);
  assert.equal(view.effectPolicy.currentActions[0].confirmationRequired, true);
});

test("effect confirmation grant binds exact effect scope and is one shot", () => {
  const raw = effectSession();
  const view = Protocol.adaptSession(raw);
  const grant = Protocol.createConfirmationGrant(view, "phone-01");
  assert.equal(grant.phase, "effect");
  const payload = Protocol.consumeConfirmationGrant(grant, view, "phone-01");
  assert.deepEqual(payload.confirmation.effect_ids, ["payment_order"]);
  assert.equal(payload.confirmation.intent_digest, "d".repeat(64));
  assert.equal(payload.confirmation.observation_id, undefined);
  assert.throws(() => Protocol.consumeConfirmationGrant(grant, view, "phone-01"), /已使用/);
});

test("action confirmation grant binds observation decision and digest", () => {
  const raw = navigationSession();
  const view = Protocol.adaptSession(raw);
  const grant = Protocol.createConfirmationGrant(view, "phone-01");
  assert.equal(grant.phase, "action");
  const payload = Protocol.consumeConfirmationGrant(grant, view, "phone-01");
  assert.equal(payload.confirmation.observation_id, raw.confirmation_scope.observation_id);
  assert.equal(payload.confirmation.decision_node_id, "qwen_visual_revision_1");
  assert.equal(payload.confirmation.action_digest, "a".repeat(64));
});

test("scope drift invalidates an existing grant", () => {
  const raw = navigationSession();
  const view = Protocol.adaptSession(raw);
  const grant = Protocol.createConfirmationGrant(view, "phone-01");
  raw.confirmation_scope.fingerprint = "changed";
  const changed = Protocol.adaptSession(raw);
  assert.throws(() => Protocol.consumeConfirmationGrant(grant, changed, "phone-01"), /已经变化/);
});

test("automatic physical advance stays disabled", () => {
  const view = Protocol.adaptSession(navigationSession());
  assert.equal(Protocol.shouldAutoAdvance({session: view, sessionDeviceId: "phone-01"}), false);
});

test("Qwen identity drift makes the action non executable", () => {
  const raw = navigationSession();
  raw.qwen_decision.device_id = "other-phone";
  const view = Protocol.adaptSession(raw);
  assert.equal(view.visualAction.identityMatchesTask, false);
  assert.equal(view.visualAction.isExecutable, false);
});

test("capability confirmation keeps exact trial action and visual scope", () => {
  const raw = navigationSession();
  raw.session_id = "capability-session";
  raw.confirmation_scope.session_id = "capability-session";
  const trial = {
    trial_id: "trial-001",
    device_id: "phone-01",
    candidate_action: "tap_semantic",
    session: raw,
    action_confirmation_scope: {...raw.confirmation_scope, trial_id: "trial-001", action: "tap_semantic"},
  };
  const grant = Protocol.createCapabilityConfirmationGrant(trial);
  const payload = Protocol.consumeCapabilityConfirmationGrant(grant, trial);
  assert.equal(payload.confirmation.trial_id, "trial-001");
  assert.equal(payload.confirmation.action, "tap_semantic");
});

test("promotion grant remains report and registry digest bound", () => {
  const trial = {
    trial_id: "trial-001",
    device_id: "phone-01",
    candidate_action: "tap_semantic",
    report: {status: "passed"},
    promotion_scope: {
      trial_id: "trial-001", device_id: "phone-01", action: "tap_semantic",
      report_sha256: "a".repeat(64), registry_sha256: "b".repeat(64),
    },
  };
  const grant = Protocol.createPromotionGrant(trial);
  const payload = Protocol.consumePromotionGrant(grant, trial);
  assert.equal(payload.report_sha256, "a".repeat(64));
  assert.throws(() => Protocol.consumePromotionGrant(grant, trial), /已使用/);
});
