const test = require("node:test");
const assert = require("node:assert/strict");

const Protocol = require("./static/protocol_adapter.js");
const taskFixture = require("./frontend_contract_fixtures/whole_task.json");
const qwenFixture = require("./frontend_contract_fixtures/qwen_whole_task_decision.json");

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

test("history separates execution receipt from current visual result", () => {
  for (const visual of ["matched", "unmatched", "uncertain", null]) {
    const session = navigationSession();
    session.history = [{execution: {action_outcome: "executed", visual_outcome: visual,
      physical_actions: 1}, visual_outcome: visual}];
    const result = Protocol.adaptSession(session);
    const verification = result.history[0].verification;
    assert.equal(verification.executionOutcome, "executed");
    assert.equal(verification.outcome, visual || "unknown");
    assert.equal(verification.matched, visual === "matched");
  }
});

function navigationSession() {
  return {
    session_id: "session-navigation-v4",
    device_id: "phone-01",
    status: "awaiting_confirmation",
    ...clone(taskFixture.task),
    qwen_decision: clone(qwenFixture.decision),
    confirmation_scope: {
      session_id: "session-navigation-v4",
      task_id: "task-map-001",
      device_id: "phone-01",
      revision: 1,
      step_id: "step_1",
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
    device_id: "phone-01",
    status: "awaiting_effect_confirmation",
    ...clone(taskFixture.task),
    effect_confirmation_scope: {
      session_id: "session-effect-v4",
      task_id: "task-map-001",
      device_id: "phone-01",
      revision: 1,
      step_id: "step_1",
      effect_ids: ["financial_transaction"],
      intent_digest: "d".repeat(64),
    },
    effect_confirmation_ready: true,
    physical_actions: 0,
    history: [],
  };
}

test("single visual task is the only formal executable protocol", () => {
  const view = Protocol.adaptSession(navigationSession());
  assert.equal(view.protocol, "single-visual-task-v1");
  assert.equal(view.compatibilityFallback, false);
  assert.equal(view.currentStep.id, "step_1");
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

test("strict direct-point decision uses canonical action point and rejects previous v9", () => {
  const session = navigationSession();
  const current = Protocol.adaptSession(session);
  assert.equal(current.visualAction.isExecutable, true);
  assert.deepEqual(current.visualAction.raw.next_action.params.tap_point, [0.77, 0.275]);
  assert.equal(current.visualAction.targetRegion, undefined);
  session.qwen_decision.protocol_version = "2026-09-01-qwen-same-response-action-finish-v9";
  assert.equal(Protocol.adaptSession(session).visualAction.isExecutable, false);
});

test("retired protocol version is unsupported and cannot mint authority", () => {
  const raw = navigationSession();
  raw.task_context_protocol = "retired-deepseek-protocol";
  const view = Protocol.adaptSession(raw);
  assert.equal(view.protocol, "unsupported-protocol");
  assert.equal(view.compatibilityFallback, true);
  assert.throws(() => Protocol.createConfirmationGrant(view, "phone-01"), /单视觉整任务/);
});

test("effect preview comes from the selected current action, not a planned manifest", () => {
  const view = Protocol.adaptSession(effectSession());
  assert.equal(view.effectPolicy.currentExecutionClass, "whole_task");
  assert.equal(view.effectPolicy.currentActions[0].kind, "financial_transaction");
  assert.deepEqual(view.effectPolicy.currentActions[0].expectedResults, []);
});

test("effect confirmation grant binds exact effect scope and is one shot", () => {
  const raw = effectSession();
  const view = Protocol.adaptSession(raw);
  const grant = Protocol.createConfirmationGrant(view, "phone-01");
  assert.equal(grant.phase, "effect");
  const payload = Protocol.consumeConfirmationGrant(grant, view, "phone-01");
  assert.deepEqual(payload.confirmation.effect_ids, ["financial_transaction"]);
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

test("auto request uses cumulative budgets without arbitrary caps or legacy iterations", () => {
  assert.deepEqual(Protocol.buildAutoRequestPayload("phone-01"), {device_id:"phone-01",confirmed:false,confirmation:null});
  const body=Protocol.buildAutoRequestPayload("phone-01",{maxPhysicalActions:321,maxObservations:654});
  assert.equal(body.max_physical_actions,321);
  assert.equal(body.max_observations,654);
  assert.equal(body.max_iterations,undefined);
  assert.throws(()=>Protocol.buildAutoRequestPayload("phone-01",{maxObservations:0}), /正整数/);
});
