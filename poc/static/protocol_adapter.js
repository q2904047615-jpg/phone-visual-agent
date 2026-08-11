(function attachProtocolAdapter(root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.UniversalAgentProtocol = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function createProtocolAdapter() {
  const terminalStatuses = new Set(["succeeded", "completed", "blocked", "failed", "cancelled"]);
  const activeSubgoalStatuses = new Set(["current", "running", "active", "in_progress"]);

  function asObject(value) {
    return value && typeof value === "object" && !Array.isArray(value) ? value : {};
  }

  function firstDefined(...values) {
    return values.find(value => value !== undefined && value !== null);
  }

  function normalizeStringList(value) {
    if (Array.isArray(value)) {
      return value.map(item => typeof item === "string" ? item : displayValue(item)).filter(Boolean);
    }
    if (value && typeof value === "object") {
      return Object.entries(value).map(([key, item]) => `${key}：${displayValue(item)}`);
    }
    return value === undefined || value === null || value === "" ? [] : [String(value)];
  }

  function displayValue(value) {
    if (Array.isArray(value)) return value.map(displayValue).join("、");
    if (value && typeof value === "object") {
      return Object.entries(value).map(([key, item]) => `${key}=${displayValue(item)}`).join("；");
    }
    return String(value ?? "—");
  }

  function normalizeSubgoal(raw, index) {
    const item = asObject(raw);
    return {
      id: String(firstDefined(item.subgoal_id, item.id, item.node_id, `subgoal-${index + 1}`)),
      index: Number(firstDefined(item.index, item.step_number, index + 1)),
      label: String(firstDefined(item.label, item.objective, item.title, item.description, `动态子目标 ${index + 1}`)),
      status: String(firstDefined(item.status, "pending")).toLowerCase(),
      reason: String(firstDefined(item.reason, item.checkpoint, item.expected_result, "等待当前画面更新")),
      completionConditions: normalizeStringList(firstDefined(item.completion_conditions, item.success_criteria, item.checkpoint)),
      raw: item,
    };
  }

  function resolveCurrentSubgoal(rawValue, subgoals, fallbackObjective) {
    const current = asObject(rawValue);
    const referenceId = typeof rawValue === "string"
      ? rawValue
      : firstDefined(current.subgoal_id, current.id, current.node_id);
    if (referenceId !== undefined) {
      const matched = subgoals.find(item => item.id === String(referenceId));
      if (matched) return matched;
    }
    if (Object.keys(current).length) return normalizeSubgoal(current, Math.max(0, subgoals.length - 1));
    const active = subgoals.filter(item => activeSubgoalStatuses.has(item.status));
    if (active.length === 1) return active[0];
    if (active.length > 1) {
      return {
        ...active[0],
        protocolWarning: "任务图包含多个当前子目标，界面仅展示第一个。",
      };
    }
    return {
      id: "current-subgoal",
      index: subgoals.length ? subgoals.length : 1,
      label: String(fallbackObjective || "当前动态子目标"),
      status: "current",
      reason: "由当前目标和画面动态确定",
      completionConditions: [],
      raw: {},
    };
  }

  function normalizeVisualAction(rawAction, proposal, scene) {
    const raw = asObject(rawAction);
    const params = asObject(raw.params);
    const fallbackProposal = asObject(proposal);
    return {
      actionType: String(firstDefined(raw.action_type, raw.action, "")),
      semanticTarget: String(firstDefined(raw.semantic_target, params.semantic_target, params.target, params.label, params.element_id, "当前语义目标")),
      targetRegion: firstDefined(raw.target_region, params.target_region, params.bounds, null),
      expectedChange: String(firstDefined(raw.expected_change, params.expected_change, params.expected_result, "动作后重新观察并验证可见变化")),
      confidence: firstDefined(raw.confidence, fallbackProposal.confidence, null),
      reason: String(firstDefined(raw.reason, fallbackProposal.reason, "依据当前画面动态生成")),
      accountEffectPossible: Boolean(firstDefined(raw.account_effect_possible, false)),
      physicalActionPossible: Boolean(firstDefined(raw.physical_action_possible, true)),
      sceneConfidence: firstDefined(asObject(scene).confidence, null),
      raw,
    };
  }

  function legacySubgoals(session, proposal) {
    const history = Array.isArray(session.history) ? session.history : [];
    const completed = history.map((item, index) => normalizeSubgoal({
      subgoal_id: `history-${item.step_number ?? index + 1}`,
      step_number: item.step_number ?? index + 1,
      label: firstDefined(asObject(item.proposal).current_subgoal, asObject(asObject(item.proposal).action).action, "已完成动态步骤"),
      status: "completed",
      reason: firstDefined(item.completion_evidence, asObject(item.proposal).reason, "动作后已重新观察"),
    }, index));
    if (proposal && !terminalStatuses.has(String(session.status || "").toLowerCase())) {
      completed.push(normalizeSubgoal({
        subgoal_id: `current-${session.step_number || completed.length + 1}`,
        step_number: session.step_number || completed.length + 1,
        label: firstDefined(proposal.current_subgoal, asObject(proposal.action).action, asObject(session.goal).objective),
        status: "current",
        reason: proposal.reason,
      }, completed.length));
    }
    return completed;
  }

  function normalizeHistory(rawHistory) {
    if (!Array.isArray(rawHistory)) return [];
    return rawHistory.map((item, index) => {
      const entry = asObject(item);
      const proposal = asObject(entry.proposal);
      return {
        stepNumber: Number(firstDefined(entry.step_number, index + 1)),
        subgoal: String(firstDefined(entry.current_subgoal, proposal.current_subgoal, asObject(proposal.action).action, "动态步骤")),
        action: normalizeVisualAction(firstDefined(entry.visual_action, proposal.visual_action, proposal.action), proposal, firstDefined(entry.after_scene, entry.scene)),
        reason: String(firstDefined(entry.reason, proposal.reason, "已执行并重新观察")),
        physicalActions: Number(firstDefined(asObject(entry.execution).physical_actions, 0)),
        completionEvidence: normalizeStringList(entry.completion_evidence),
        raw: entry,
      };
    });
  }

  function adaptSession(rawSession, options = {}) {
    const session = asObject(rawSession);
    const graph = asObject(firstDefined(session.task_graph, asObject(session.goal).task_graph));
    const goal = asObject(session.goal);
    const proposal = asObject(session.proposal);
    const rawSubgoals = firstDefined(graph.subgoals, graph.nodes);
    const subgoals = Array.isArray(rawSubgoals)
      ? rawSubgoals.map(normalizeSubgoal)
      : legacySubgoals(session, proposal);
    const status = String(firstDefined(graph.status, session.status, "idle")).toLowerCase();
    const currentSubgoal = resolveCurrentSubgoal(
      firstDefined(graph.current_subgoal, session.current_subgoal, proposal.current_subgoal),
      subgoals,
      goal.objective,
    );
    const scene = asObject(firstDefined(session.current_scene, session.scene));
    const currentActionMeta = asObject(session.current_action);
    const visualAction = normalizeVisualAction(
      firstDefined(
        session.visual_action,
        proposal.visual_action,
        currentActionMeta.action_type ? currentActionMeta : undefined,
        proposal.action,
      ),
      proposal,
      scene,
    );
    visualAction.accountEffectPossible = Boolean(firstDefined(
      visualAction.raw.account_effect_possible,
      currentActionMeta.account_effect_possible,
      false,
    ));
    visualAction.physicalActionPossible = Boolean(firstDefined(
      visualAction.raw.physical_action_possible,
      currentActionMeta.physical_action_possible,
      Boolean(visualAction.actionType),
    ));

    return {
      protocol: graph.subgoals ? "deepseek-task-graph" : (graph.nodes ? "legacy-task-graph" : "legacy-session"),
      sessionId: String(firstDefined(session.session_id, session.id, "")),
      taskId: String(firstDefined(graph.task_id, session.task_id, session.session_id, "")),
      deviceId: String(firstDefined(graph.device_id, session.device_id, options.fallbackDeviceId, "")),
      status,
      isTerminal: terminalStatuses.has(status),
      stepNumber: Number(firstDefined(session.step_number, currentSubgoal.index, 1)),
      objective: String(firstDefined(graph.objective, goal.objective, session.objective, "未命名目标")),
      appName: String(firstDefined(goal.app_name, "")),
      constraints: normalizeStringList(firstDefined(graph.constraints, goal.constraints)),
      completionConditions: normalizeStringList(firstDefined(graph.completion_conditions, goal.success_criteria)),
      subgoals,
      currentSubgoal,
      visualAction,
      scene: {
        summary: String(firstDefined(scene.summary, scene.screen_id, "尚未识别")),
        screenId: String(firstDefined(scene.screen_id, "unknown")),
        stable: scene.stable === true,
        confidence: firstDefined(scene.confidence, null),
        raw: scene,
      },
      history: normalizeHistory(session.history),
      risk: {
        accountEffectPossible: visualAction.accountEffectPossible,
        requiresConfirmation: Boolean(firstDefined(currentActionMeta.requires_confirmation, status === "awaiting_confirmation")),
        maxPhysicalActions: 1,
      },
      autoPauseReason: String(firstDefined(session.auto_pause_reason, "")),
      failedReason: String(firstDefined(session.failed_reason, "")),
      raw: session,
    };
  }

  function buildRequestPayload(sessionDeviceId, values = {}) {
    const deviceId = String(sessionDeviceId || "").trim();
    if (!deviceId) throw new Error("会话缺少锁定的 device_id。 ");
    return { ...values, device_id: deviceId };
  }

  function buildAutoRequestPayload(sessionDeviceId) {
    return buildRequestPayload(sessionDeviceId, {
      confirmed: true,
      max_physical_actions: 1,
    });
  }

  function shouldAutoAdvance(context) {
    const session = context?.session;
    if (!session || context.paused || context.busy || session.isTerminal) return false;
    if (session.risk.accountEffectPossible) return false;
    return ["awaiting_confirmation", "paused_after_action"].includes(session.status);
  }

  async function runAutoAdvanceLoop(options) {
    const maxRequests = Math.max(1, Math.min(16, Number(options.maxRequests || 8)));
    let requests = 0;
    while (requests < maxRequests) {
      const context = options.getContext();
      if (!shouldAutoAdvance(context)) break;
      const payload = buildAutoRequestPayload(context.sessionDeviceId);
      const response = await options.sendOne(payload);
      requests += 1;
      await options.applyResponse(response);
    }
    return { requests, limitReached: requests >= maxRequests };
  }

  return {
    adaptSession,
    buildRequestPayload,
    buildAutoRequestPayload,
    displayValue,
    runAutoAdvanceLoop,
    shouldAutoAdvance,
  };
});
