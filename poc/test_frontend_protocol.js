const test = require("node:test");
const assert = require("node:assert/strict");

const Protocol = require("./static/protocol_adapter.js");

function deepSeekSession(overrides = {}) {
  return {
    session_id: "session-contract-01",
    goal: { objective: "完成一个未预设的通用手机目标" },
    task_graph: {
      task_id: "task-deepseek-01",
      device_id: "device-session-01",
      status: "awaiting_confirmation",
      constraints: ["不得越过用户确认", "每次只执行一个物理动作"],
      completion_conditions: {
        visible_state: "结果标记可见",
        evidence: ["保存动作后画面"],
      },
      current_subgoal: "sg-2",
      subgoals: [
        {
          subgoal_id: "sg-1",
          objective: "观察并理解当前页面",
          status: "completed",
          completion_conditions: ["页面状态已识别"],
        },
        {
          subgoal_id: "sg-2",
          objective: "操作唯一匹配的语义目标",
          status: "current",
          reason: "当前画面中只有一个候选目标",
        },
        {
          subgoal_id: "sg-3",
          objective: "重新观察并验证变化",
          status: "pending",
        },
      ],
    },
    current_action: {
      action_type: "tap_semantic",
      semantic_target: "唯一的确认控件",
      target_region: { x_min: 0.61, y_min: 0.72, x_max: 0.88, y_max: 0.82 },
      expected_change: "页面出现可验证的结果标记",
      confidence: 0.93,
      reason: "文字、位置和当前子目标一致",
      account_effect_possible: false,
      physical_action_possible: true,
    },
    scene: {
      screen_id: "generic-screen",
      summary: "通用页面与一个确认控件",
      stable: true,
      confidence: 0.96,
    },
    history: [],
    ...overrides,
  };
}

test("DeepSeek task_graph.subgoals and root fields become the preferred plan view", () => {
  const session = deepSeekSession({
    // Deliberately conflicting legacy fields prove that the new task graph wins.
    status: "failed",
    device_id: "legacy-device",
    task_id: "legacy-task",
  });
  const view = Protocol.adaptSession(session, { fallbackDeviceId: "fallback-device" });

  assert.equal(view.protocol, "deepseek-task-graph");
  assert.equal(view.taskId, "task-deepseek-01");
  assert.equal(view.deviceId, "device-session-01");
  assert.equal(view.status, "awaiting_confirmation");
  assert.deepEqual(view.constraints, ["不得越过用户确认", "每次只执行一个物理动作"]);
  assert.deepEqual(view.completionConditions, [
    "visible_state：结果标记可见",
    "evidence：保存动作后画面",
  ]);
  assert.deepEqual(view.subgoals.map(item => item.id), ["sg-1", "sg-2", "sg-3"]);
  assert.equal(view.currentSubgoal.id, "sg-2");
  assert.equal(view.currentSubgoal.label, "操作唯一匹配的语义目标");
});

test("Qwen generic visual action fields remain intact for rendering", () => {
  const view = Protocol.adaptSession(deepSeekSession());

  assert.equal(view.visualAction.actionType, "tap_semantic");
  assert.equal(view.visualAction.semanticTarget, "唯一的确认控件");
  assert.deepEqual(view.visualAction.targetRegion, {
    x_min: 0.61,
    y_min: 0.72,
    x_max: 0.88,
    y_max: 0.82,
  });
  assert.equal(view.visualAction.expectedChange, "页面出现可验证的结果标记");
  assert.equal(view.visualAction.confidence, 0.93);
  assert.equal(view.visualAction.reason, "文字、位置和当前子目标一致");
});

test("legacy nodes and proposal.action remain an explicit fallback", () => {
  const view = Protocol.adaptSession({
    session_id: "legacy-session",
    status: "awaiting_confirmation",
    task_graph: {
      nodes: [{ id: "legacy-node", label: "兼容旧节点", status: "current" }],
    },
    proposal: {
      current_subgoal: "legacy-node",
      action: {
        action: "swipe",
        params: { direction: "up", target: "当前内容区域", expected_result: "显示下一段内容" },
      },
      confidence: 0.81,
      reason: "旧接口回退",
    },
  }, { fallbackDeviceId: "legacy-device" });

  assert.equal(view.protocol, "legacy-task-graph");
  assert.equal(view.currentSubgoal.id, "legacy-node");
  assert.equal(view.visualAction.actionType, "swipe");
  assert.equal(view.visualAction.semanticTarget, "当前内容区域");
  assert.equal(view.visualAction.expectedChange, "显示下一段内容");
  assert.equal(view.deviceId, "legacy-device");
});

test("every browser-driven auto request is limited to one physical action", () => {
  assert.deepEqual(Protocol.buildAutoRequestPayload("device-session-01"), {
    confirmed: true,
    max_physical_actions: 1,
    device_id: "device-session-01",
  });
});

test("pause after a response prevents the browser from issuing another request", async () => {
  let paused = false;
  let selectedDeviceId = "device-session-01";
  const lockedSessionDeviceId = selectedDeviceId;
  let session = Protocol.adaptSession(deepSeekSession());
  const requests = [];

  // The dropdown changes after session creation, but requests retain the lock.
  selectedDeviceId = "device-selected-later";
  const outcome = await Protocol.runAutoAdvanceLoop({
    getContext: () => ({
      session,
      sessionDeviceId: lockedSessionDeviceId,
      paused,
      busy: false,
    }),
    sendOne: async payload => {
      requests.push(payload);
      return deepSeekSession();
    },
    applyResponse: async response => {
      session = Protocol.adaptSession(response);
      paused = true;
    },
  });

  assert.equal(outcome.requests, 1);
  assert.equal(requests.length, 1);
  assert.equal(requests[0].max_physical_actions, 1);
  assert.equal(requests[0].device_id, "device-session-01");
  assert.notEqual(requests[0].device_id, selectedDeviceId);
});

test("risk on the current step prevents automatic advancement", async () => {
  const risky = deepSeekSession();
  risky.current_action.account_effect_possible = true;
  const requests = [];

  const outcome = await Protocol.runAutoAdvanceLoop({
    getContext: () => ({
      session: Protocol.adaptSession(risky),
      sessionDeviceId: "device-session-01",
      paused: false,
      busy: false,
    }),
    sendOne: async payload => {
      requests.push(payload);
      return risky;
    },
    applyResponse: async () => {},
  });

  assert.equal(outcome.requests, 0);
  assert.deepEqual(requests, []);
});
