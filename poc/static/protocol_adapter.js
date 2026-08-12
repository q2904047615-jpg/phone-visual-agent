(function attachProtocolAdapter(root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.UniversalAgentProtocol = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function createProtocolAdapter() {
  const terminalStatuses = new Set(["succeeded", "completed", "blocked", "failed", "cancelled"]);
  const activeSubgoalStatuses = new Set(["current", "running", "active", "in_progress"]);
  const externalImpacts = new Set(["external_state", "unknown"]);
  const formalDeepSeekProtocol = "2026-08-11-deepseek-task-graph-v3";
  const legacyDeepSeekV2Protocol = "2026-08-11-deepseek-task-graph-v2";

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

  function normalizeTargetApps(value) {
    if (!Array.isArray(value)) return [];
    return value.map(item => {
      const app = asObject(item);
      return {
        id: String(firstDefined(app.app_id, app.id, "")),
        name: String(firstDefined(app.app_name, app.name, app.app_id, "未命名目标应用")),
        raw: app,
      };
    });
  }

  function normalizeSubgoal(raw, index) {
    const item = asObject(raw);
    return {
      id: String(firstDefined(item.subgoal_id, item.id, item.node_id, `subgoal-${index + 1}`)),
      index: Number(firstDefined(item.index, item.step_number, index + 1)),
      label: String(firstDefined(item.objective, item.label, item.title, item.description, `动态子目标 ${index + 1}`)),
      status: String(firstDefined(item.status, "pending")).toLowerCase(),
      reason: String(firstDefined(item.reason, item.checkpoint, item.expected_result, item.external_impact, "等待当前画面更新")),
      completionConditions: normalizeStringList(firstDefined(item.completion_conditions, item.success_criteria, item.checkpoint)),
      riskIds: normalizeStringList(item.risk_action_ids),
      externalImpact: String(firstDefined(item.external_impact, "unknown")),
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
      return { ...active[0], protocolWarning: "任务图包含多个当前子目标，界面仅展示第一个。" };
    }
    return {
      id: "current-subgoal",
      index: subgoals.length ? subgoals.length : 1,
      label: String(fallbackObjective || "当前动态子目标"),
      status: "current",
      reason: "由当前目标和画面动态确定",
      completionConditions: [],
      riskIds: [],
      externalImpact: "unknown",
      raw: {},
    };
  }

  function normalizeRiskActions(value) {
    if (!Array.isArray(value)) return [];
    return value.map(raw => {
      const item = asObject(raw);
      return {
        id: String(firstDefined(item.risk_id, item.id, "")),
        description: String(firstDefined(item.description, "未说明风险")),
        externalEffect: String(firstDefined(item.external_effect, "影响未知")),
        type: String(firstDefined(item.risk_type, "unknown_external_effect")),
        level: String(firstDefined(item.risk_level, "unknown")),
        subgoalIds: normalizeStringList(item.subgoal_ids),
        confirmationRequired: item.confirmation_required !== false,
        raw: item,
      };
    });
  }

  function normalizeConfirmationGateScope(rawScope) {
    const scope = asObject(rawScope);
    return {
      sessionId: String(firstDefined(scope.session_id, "")),
      taskId: String(firstDefined(scope.task_id, "")),
      deviceId: String(firstDefined(scope.device_id, "")),
      revision: firstDefined(scope.revision, null),
      subgoalId: String(firstDefined(scope.subgoal_id, "")),
      riskIds: normalizeStringList(scope.risk_ids),
      observationId: String(firstDefined(scope.observation_id, "")),
      fingerprint: String(firstDefined(scope.fingerprint, "")),
    };
  }

  function isQwenV2Decision(value) {
    const decision = asObject(value);
    return String(decision.protocol_version || "").includes("qwen-visual-decision-v2")
      || (Object.prototype.hasOwnProperty.call(decision, "next_action")
        && decision.task_id !== undefined
        && decision.revision !== undefined
        && decision.status !== undefined);
  }

  function normalizeQwenV2Decision(rawDecision) {
    const decision = asObject(rawDecision);
    const status = String(firstDefined(decision.status, "blocked")).toLowerCase();
    const nextAction = asObject(decision.next_action);
    const params = asObject(nextAction.params);
    const targetRegion = firstDefined(decision.target_region, null);
    const region = asObject(targetRegion);
    const elementId = String(firstDefined(params.element_id, region.element_id, ""));
    const actionType = status === "action" ? String(firstDefined(nextAction.action, "")) : "";
    const semanticTarget = String(firstDefined(
      params.label,
      params.target,
      elementId,
      status === "finished" ? "目标完成" : status === "blocked" ? "当前步骤已阻止" : "未提供语义目标",
    ));
    const isExecutable = status === "action" && Boolean(actionType);
    return {
      protocol: "qwen-visual-decision-v2",
      protocolVersion: String(firstDefined(decision.protocol_version, "")),
      status,
      actionType,
      semanticTarget,
      elementId,
      targetRegion,
      expectedChange: firstDefined(decision.expected_result, {}),
      confidence: firstDefined(decision.confidence, null),
      reason: String(firstDefined(decision.reason, "Qwen 未提供判断理由")),
      taskId: String(firstDefined(decision.task_id, "")),
      deviceId: String(firstDefined(decision.device_id, "")),
      revision: firstDefined(decision.revision, null),
      observationId: String(firstDefined(decision.observation_id, "")),
      fingerprint: String(firstDefined(decision.fingerprint, "")),
      accountEffectPossible: false,
      physicalActionPossible: isExecutable,
      isExecutable,
      identityMatchesTask: true,
      raw: decision,
    };
  }

  function normalizeLegacyVisualAction(rawAction, proposal, scene) {
    const raw = asObject(rawAction);
    const params = asObject(raw.params);
    const fallbackProposal = asObject(proposal);
    const actionType = String(firstDefined(raw.action_type, raw.action, ""));
    return {
      protocol: "legacy-visual-action",
      protocolVersion: String(firstDefined(raw.protocol_version, "")),
      status: actionType ? "action" : "unknown",
      actionType,
      semanticTarget: String(firstDefined(raw.semantic_target, params.semantic_target, params.target, params.label, params.element_id, "当前语义目标")),
      elementId: String(firstDefined(params.element_id, "")),
      targetRegion: firstDefined(raw.target_region, params.target_region, params.bounds, null),
      expectedChange: firstDefined(raw.expected_change, params.expected_change, params.expected_result, "动作后重新观察并验证可见变化"),
      confidence: firstDefined(raw.confidence, fallbackProposal.confidence, null),
      reason: String(firstDefined(raw.reason, fallbackProposal.reason, "依据当前画面动态生成")),
      taskId: "",
      deviceId: "",
      revision: null,
      observationId: "",
      fingerprint: "",
      accountEffectPossible: Boolean(firstDefined(raw.account_effect_possible, false)),
      physicalActionPossible: Boolean(firstDefined(raw.physical_action_possible, actionType)),
      isExecutable: Boolean(actionType),
      identityMatchesTask: true,
      sceneConfidence: firstDefined(asObject(scene).confidence, null),
      raw,
    };
  }

  function legacySubgoals(session, proposal) {
    const history = Array.isArray(session.history) ? session.history : [];
    const completed = history.map((item, index) => normalizeSubgoal({
      subgoal_id: `history-${item.step_number ?? index + 1}`,
      step_number: item.step_number ?? index + 1,
      objective: firstDefined(asObject(item.proposal).current_subgoal, asObject(asObject(item.proposal).action).action, "已完成动态步骤"),
      status: "completed",
      reason: firstDefined(item.completion_evidence, asObject(item.proposal).reason, "动作后已重新观察"),
      external_impact: "unknown",
    }, index));
    if (Object.keys(asObject(proposal)).length && !terminalStatuses.has(String(session.status || "").toLowerCase())) {
      completed.push(normalizeSubgoal({
        subgoal_id: `current-${session.step_number || completed.length + 1}`,
        step_number: session.step_number || completed.length + 1,
        objective: firstDefined(proposal.current_subgoal, asObject(proposal.action).action, asObject(session.goal).objective),
        status: "current",
        reason: proposal.reason,
        external_impact: "unknown",
      }, completed.length));
    }
    return completed;
  }

  function normalizeHistory(rawHistory) {
    if (!Array.isArray(rawHistory)) return [];
    return rawHistory.map((item, index) => {
      const entry = asObject(item);
      const proposal = asObject(entry.proposal);
      const candidate = firstDefined(entry.qwen_decision, entry.visual_decision, entry.visual_action, proposal.visual_action, proposal.action);
      const action = isQwenV2Decision(candidate)
        ? normalizeQwenV2Decision(candidate)
        : normalizeLegacyVisualAction(candidate, proposal, firstDefined(entry.after_scene, entry.scene));
      return {
        stepNumber: Number(firstDefined(entry.step_number, index + 1)),
        subgoal: String(firstDefined(entry.current_subgoal, proposal.current_subgoal, asObject(proposal.action).action, "动态步骤")),
        action,
        reason: String(firstDefined(entry.reason, proposal.reason, "已执行并重新观察")),
        physicalActions: Number(firstDefined(asObject(entry.execution).physical_actions, 0)),
        completionEvidence: normalizeStringList(entry.completion_evidence),
        evidence: normalizeStringList(firstDefined(
          asObject(entry.execution).evidence,
          asObject(entry.execution).after_frame_paths,
        )),
        raw: entry,
      };
    });
  }

  function adaptSession(rawSession, options = {}) {
    const session = asObject(rawSession);
    const graph = asObject(firstDefined(session.task_graph, asObject(session.goal).task_graph));
    const graphGoal = asObject(graph.goal);
    const legacyGoal = asObject(session.goal);
    const proposal = asObject(session.proposal);
    const qwenContext = asObject(firstDefined(session.qwen_context, session.task_context));
    const qwenDecisionRaw = [
      session.qwen_decision,
      session.visual_decision,
      session.decision,
      session.current_visual_decision,
      session.current_action,
    ].find(isQwenV2Decision);
    const currentActionMeta = asObject(session.current_action);
    const rawSubgoals = firstDefined(graph.subgoals, graph.nodes);
    const subgoals = Array.isArray(rawSubgoals)
      ? rawSubgoals.map(normalizeSubgoal)
      : legacySubgoals(session, proposal);
    const objective = String(firstDefined(graphGoal.objective, graph.objective, legacyGoal.objective, session.objective, "未命名目标"));
    const currentSubgoal = resolveCurrentSubgoal(
      firstDefined(graph.current_subgoal, qwenContext.current_subgoal, session.current_subgoal, proposal.current_subgoal),
      subgoals,
      objective,
    );
    if (!subgoals.some(item => item.id === currentSubgoal.id) && Object.keys(currentSubgoal.raw).length) {
      subgoals.push(currentSubgoal);
    }
    const status = String(firstDefined(session.status, graph.status, graph.task_status, qwenContext.task_status, "idle")).toLowerCase();
    const taskId = String(firstDefined(graph.task_id, qwenContext.task_id, session.task_id, session.session_id, ""));
    const deviceId = String(firstDefined(graph.device_id, qwenContext.device_id, session.device_id, options.fallbackDeviceId, ""));
    const revision = firstDefined(graph.revision, qwenContext.revision, session.revision, null);
    const scene = asObject(firstDefined(session.current_scene, session.scene, asObject(qwenDecisionRaw).page_state));
    const visualAction = qwenDecisionRaw
      ? normalizeQwenV2Decision(qwenDecisionRaw)
      : normalizeLegacyVisualAction(
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
    const identityChecks = [
      !visualAction.taskId || !taskId || visualAction.taskId === taskId,
      visualAction.revision === null || revision === null || Number(visualAction.revision) === Number(revision),
      !visualAction.deviceId || !deviceId || visualAction.deviceId === deviceId,
    ];
    visualAction.identityMatchesTask = identityChecks.every(Boolean);
    if (!visualAction.identityMatchesTask) {
      visualAction.physicalActionPossible = false;
      visualAction.isExecutable = false;
    }

    const riskActions = normalizeRiskActions(firstDefined(graph.risk_actions, qwenContext.risk_actions));
    const rawGate = asObject(firstDefined(graph.confirmation_gate, qwenContext.confirmation_gate, session.confirmation_gate));
    const authorityScope = normalizeConfirmationGateScope(firstDefined(
      session.confirmation_scope,
      rawGate.scope,
    ));
    const currentExternalImpact = String(firstDefined(
      graph.current_external_impact,
      qwenContext.current_external_impact,
      currentSubgoal.externalImpact,
      "unknown",
    ));
    const gateRiskIds = normalizeStringList(firstDefined(
      asObject(session.confirmation_scope).risk_ids,
      rawGate.risk_ids,
      currentSubgoal.riskIds,
    ));
    const currentRiskActions = riskActions.filter(item => (
      gateRiskIds.includes(item.id) || item.subgoalIds.includes(currentSubgoal.id)
    ));
    const gateRequired = Boolean(
      session.confirmation_ready === true
      || status === "awaiting_confirmation"
      || rawGate.required === true
      || externalImpacts.has(currentExternalImpact)
    );
    const gateState = String(firstDefined(
      rawGate.state,
      gateRequired ? (status === "awaiting_confirmation" ? "awaiting_confirmation" : "unconfirmed") : "not_required",
    ));
    const requiresConfirmation = Boolean(
      gateRequired
      || currentActionMeta.requires_confirmation
      || status === "awaiting_confirmation"
    );
    const hasCurrentRisk = gateRiskIds.length > 0 || currentRiskActions.length > 0;
    const accountEffectPossible = Boolean(
      visualAction.accountEffectPossible
      || externalImpacts.has(currentExternalImpact)
    );
    const blocksAutomatic = Boolean(
      requiresConfirmation
      || hasCurrentRisk
      || accountEffectPossible
      || externalImpacts.has(currentExternalImpact)
      || gateState === "awaiting_confirmation"
      || gateState === "unconfirmed"
    );
    visualAction.accountEffectPossible = accountEffectPossible;

    const protocolVersion = String(firstDefined(graph.protocol_version, qwenContext.protocol_version, ""));
    const formalV3 = protocolVersion === formalDeepSeekProtocol;
    const compatibilityV2 = protocolVersion === legacyDeepSeekV2Protocol;
    const targetApps = normalizeTargetApps(firstDefined(graphGoal.target_apps, asObject(qwenContext.goal).target_apps));
    return {
      protocol: formalV3
        ? "deepseek-task-graph-v3"
        : compatibilityV2
          ? "deepseek-task-graph-v2-compatibility"
        : (graph.subgoals ? "deepseek-task-graph" : (graph.nodes ? "legacy-task-graph" : "legacy-session")),
      protocolVersion,
      compatibilityFallback: !formalV3,
      sessionId: String(firstDefined(session.session_id, session.id, "")),
      taskId,
      deviceId,
      revision,
      status,
      isTerminal: terminalStatuses.has(status),
      stepNumber: Number(firstDefined(session.step_number, currentSubgoal.index, 1)),
      objective,
      targetApps,
      appName: String(firstDefined(legacyGoal.app_name, "")),
      constraints: normalizeStringList(firstDefined(graph.constraints, graph.global_constraints, qwenContext.global_constraints, legacyGoal.constraints)),
      completionConditions: normalizeStringList(firstDefined(graph.completion_conditions, graph.goal_completion_conditions, qwenContext.goal_completion_conditions, legacyGoal.success_criteria)),
      subgoals,
      currentSubgoal,
      visualAction,
      scene: {
        summary: String(firstDefined(scene.summary, scene.screen_id, "尚未识别")),
        screenId: String(firstDefined(scene.screen_id, "unknown")),
        stable: scene.stable === true || asObject(asObject(qwenDecisionRaw).trusted_observation).local_stability?.stable === true,
        confidence: firstDefined(scene.confidence, null),
        raw: scene,
      },
      history: normalizeHistory(session.history),
      risk: {
        accountEffectPossible,
        requiresConfirmation,
        hasCurrentRisk,
        blocksAutomatic,
        currentExternalImpact,
        riskIds: gateRiskIds,
        actions: riskActions,
        currentActions: currentRiskActions,
        confirmationGate: {
          required: gateRequired,
          state: gateState,
          riskIds: gateRiskIds,
          externalStateActionAllowed: rawGate.external_state_action_allowed === true,
          scope: authorityScope,
          raw: rawGate,
        },
        maxPhysicalActions: 1,
      },
      controllerGate: {
        allowed: asObject(session.controller_decision).allowed === true,
        reason: String(firstDefined(asObject(session.controller_decision).reason, "")),
        canonicalClass: String(firstDefined(asObject(session.controller_decision).canonical_class, "")),
        policyVersion: String(firstDefined(asObject(session.controller_decision).policy_version, "")),
      },
      physicalActions: Number(firstDefined(session.physical_actions, 0)),
      evidence: normalizeStringList(session.evidence),
      autoPauseReason: String(firstDefined(session.auto_pause_reason, "")),
      failedReason: String(firstDefined(session.failed_reason, "")),
      raw: session,
    };
  }

  function buildRequestPayload(sessionDeviceId, values = {}) {
    const deviceId = String(sessionDeviceId || "").trim();
    if (!deviceId) throw new Error("会话缺少锁定的 device_id。");
    return { ...values, device_id: deviceId };
  }

  function buildAutoRequestPayload(sessionDeviceId) {
    return buildRequestPayload(sessionDeviceId, { max_physical_actions: 1 });
  }

  function confirmationScope(session, sessionDeviceId) {
    if (!session) throw new Error("当前没有可确认的会话。");
    const gateScope = session.risk?.confirmationGate?.scope || {};
    const canonicalRevision = session.revision === null || session.revision === undefined
      ? null
      : Number(session.revision);
    const gateRevision = gateScope.revision === null || gateScope.revision === undefined
      ? canonicalRevision
      : Number(gateScope.revision);
    if (
      (gateScope.taskId && gateScope.taskId !== session.taskId)
      || (gateScope.deviceId && gateScope.deviceId !== session.deviceId)
      || (gateScope.subgoalId && gateScope.subgoalId !== session.currentSubgoal?.id)
      || (gateScope.observationId && gateScope.observationId !== session.visualAction?.observationId)
      || (gateScope.fingerprint && gateScope.fingerprint !== session.visualAction?.fingerprint)
      || gateRevision !== canonicalRevision
      || String(sessionDeviceId || "") !== String(session.deviceId || "")
    ) {
      throw new Error("任务、revision、子目标、风险或设备已经变化，请重新确认。");
    }
    return {
      session_id: String(gateScope.sessionId || session.sessionId || ""),
      task_id: String(gateScope.taskId || session.taskId || ""),
      device_id: String(gateScope.deviceId || sessionDeviceId || ""),
      revision: gateRevision,
      subgoal_id: String(gateScope.subgoalId || session.currentSubgoal?.id || ""),
      risk_ids: [...(gateScope.riskIds || session.risk?.riskIds || [])].map(String).sort(),
      observation_id: String(gateScope.observationId || session.visualAction?.observationId || ""),
      fingerprint: String(gateScope.fingerprint || session.visualAction?.fingerprint || ""),
    };
  }

  function scopeFingerprint(scope) {
    return JSON.stringify({
      session_id: scope.session_id,
      task_id: scope.task_id,
      device_id: scope.device_id,
      revision: scope.revision,
      subgoal_id: scope.subgoal_id,
      risk_ids: [...scope.risk_ids].sort(),
      observation_id: scope.observation_id,
      fingerprint: scope.fingerprint,
    });
  }

  function createConfirmationGrant(session, sessionDeviceId) {
    if (!session?.risk?.requiresConfirmation) throw new Error("当前步骤不需要风险确认。");
    if (session.protocol !== "deepseek-task-graph-v3" || session.compatibilityFallback) {
      throw new Error("当前会话没有正式DeepSeek v3确认作用域，拒绝确认。");
    }
    const scope = confirmationScope(session, sessionDeviceId);
    if (
      !scope.session_id
      || !scope.task_id
      || !scope.device_id
      || !scope.subgoal_id
      || !scope.observation_id
      || !scope.fingerprint
      || !Number.isInteger(scope.revision)
      || scope.device_id !== String(sessionDeviceId || "")
    ) {
      throw new Error("当前DeepSeek v3确认作用域不完整或设备不一致。");
    }
    if (externalImpacts.has(session.risk.currentExternalImpact) && !scope.risk_ids.length) {
      throw new Error("外部状态或未知影响步骤缺少 risk_ids，拒绝确认。");
    }
    return { scope, consumed: false };
  }

  function consumeConfirmationGrant(grant, session, sessionDeviceId) {
    if (!grant || grant.consumed) throw new Error("本次风险确认已使用或不存在。");
    const currentScope = confirmationScope(session, sessionDeviceId);
    if (scopeFingerprint(grant.scope) !== scopeFingerprint(currentScope)) {
      throw new Error("任务、revision、子目标、风险或设备已经变化，请重新确认。");
    }
    grant.consumed = true;
    return {
      confirmed: true,
      confirmation: {
        session_id: grant.scope.session_id,
        task_id: grant.scope.task_id,
        device_id: grant.scope.device_id,
        revision: grant.scope.revision,
        subgoal_id: grant.scope.subgoal_id,
        risk_ids: [...grant.scope.risk_ids],
        observation_id: grant.scope.observation_id,
        fingerprint: grant.scope.fingerprint,
      },
    };
  }

  function shouldAutoAdvance(context) {
    void context;
    return false;
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
    consumeConfirmationGrant,
    createConfirmationGrant,
    displayValue,
    runAutoAdvanceLoop,
    shouldAutoAdvance,
  };
});
