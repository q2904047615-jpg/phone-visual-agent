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
      decisionNodeId: String(firstDefined(scope.decision_node_id, "")),
      actionDigest: String(firstDefined(scope.action_digest, "")),
    };
  }

  function isQwenV2Decision(value) {
    const decision = asObject(value);
    return /qwen-visual-decision-v\d+/.test(String(decision.protocol_version || ""))
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
    const protocolVersion = String(firstDefined(decision.protocol_version, ""));
    const protocolMatch = protocolVersion.match(/qwen-visual-decision-v\d+/);
    return {
      protocol: protocolMatch ? protocolMatch[0] : "qwen-visual-decision-v2",
      protocolVersion,
      status,
      decisionNodeId: String(firstDefined(nextAction.node_id, "")),
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
      decisionNodeId: String(firstDefined(raw.node_id, params.node_id, "")),
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

  function normalizeControllerGate(value) {
    const gate = asObject(value);
    return {
      allowed: gate.allowed === true,
      reason: String(firstDefined(gate.reason, "")),
      canonicalClass: String(firstDefined(gate.canonical_class, gate.canonicalClass, "")),
      policyVersion: String(firstDefined(gate.policy_version, gate.policyVersion, "")),
    };
  }

  function normalizeVerification(entry, action) {
    const execution = asObject(entry.execution);
    const executionVerification = asObject(execution.verification);
    const verification = Object.keys(executionVerification).length
      ? executionVerification
      : asObject(entry.verification);
    const beforeScene = asObject(execution.before_scene);
    const afterScene = asObject(execution.after_scene);
    const outcome = String(firstDefined(
      execution.action_outcome,
      verification.action_outcome,
      entry.action_outcome,
      "unknown",
    ));
    const errors = [...new Set([
      ...normalizeStringList(execution.verification_errors),
      ...normalizeStringList(execution.observation_errors),
      ...normalizeStringList(verification.blocked_reasons),
      ...normalizeStringList(verification.verification_errors),
    ])];
    return {
      outcome,
      matched: outcome === "matched",
      beforeFingerprint: String(firstDefined(
        verification.before_fingerprint,
        beforeScene.fingerprint,
        action.fingerprint,
        "",
      )),
      afterObservationId: String(firstDefined(
        entry.after_observation_id,
        verification.after_observation_id,
        asObject(entry.after_observation).observation_id,
        "",
      )),
      afterFingerprint: String(firstDefined(
        entry.after_fingerprint,
        verification.after_fingerprint,
        asObject(entry.after_observation).fingerprint,
        afterScene.fingerprint,
        "",
      )),
      errors,
      evidence: [...new Set([
        ...normalizeStringList(execution.evidence),
        ...normalizeStringList(execution.after_frame_paths),
      ])],
      raw: verification,
    };
  }

  function normalizeTransition(entry, verification, replanHistory, currentRevision, sessionStatus, failedReason) {
    const explicit = asObject(firstDefined(entry.transition, entry.replan));
    const sourceRevision = firstDefined(entry.task_revision, entry.revision, null);
    const nextReplan = replanHistory.find(item => {
      const candidate = asObject(item);
      return Number.isInteger(sourceRevision)
        && candidate.revision === sourceRevision + 1
        && String(candidate.scene_id || "")
        && String(candidate.scene_id) === verification.afterObservationId;
    });
    const replan = Object.keys(explicit).length ? explicit : asObject(nextReplan);
    const trigger = String(firstDefined(replan.trigger, entry.replan_trigger, ""));
    let kind = String(firstDefined(replan.outcome, replan.kind, entry.advance_outcome, ""));
    if (!kind) {
      if (trigger === "action_result_mismatch" || verification.outcome === "mismatched") kind = "replan";
      else if (trigger === "subgoal_completed") kind = "advance";
      else if (trigger) kind = "advance_or_replan";
      else if (["blocked", "failed"].includes(sessionStatus)) kind = "blocked";
      else kind = "unknown";
    }
    return {
      kind,
      trigger,
      reason: String(firstDefined(
        replan.reason,
        entry.transition_reason,
        entry.reason,
        kind === "blocked" ? failedReason : "",
      )),
      fromRevision: sourceRevision,
      toRevision: Number.isInteger(replan.revision) ? replan.revision : null,
      raw: replan,
    };
  }

  function normalizeHistoricalScope(entry, action, context) {
    const execution = asObject(entry.execution);
    const receipt = asObject(firstDefined(
      entry.confirmation_receipt,
      execution.confirmation_receipt,
    ));
    const scope = normalizeConfirmationGateScope(receipt.scope);
    const expectedSubgoalId = String(firstDefined(entry.subgoal_id, asObject(entry.scope).subgoal_id, ""));
    const expectedRiskIds = Array.isArray(entry.risk_ids)
      ? normalizeStringList(entry.risk_ids).map(String).sort()
      : null;
    const authoritative = receipt.authoritative === true && receipt.consumed === true;
    const exact = authoritative
      && Boolean(context.sessionId)
      && scope.sessionId === context.sessionId
      && scope.taskId === action.taskId
      && scope.deviceId === action.deviceId
      && scope.revision === action.revision
      && Boolean(expectedSubgoalId)
      && scope.subgoalId === expectedSubgoalId
      && expectedRiskIds !== null
      && JSON.stringify([...scope.riskIds].map(String).sort()) === JSON.stringify(expectedRiskIds)
      && scope.observationId === action.observationId
      && scope.fingerprint === action.fingerprint
      && scope.decisionNodeId === action.decisionNodeId
      && Boolean(scope.actionDigest);
    return exact
      ? {
        state: "consumed",
        reason: "后端权威回执证明该单动作确认已消费",
        mismatches: [],
        scope,
      }
      : {
        state: "unknown",
        reason: "旧记录没有可验证的确认消费回执",
        mismatches: [],
        scope: {},
      };
  }

  function normalizeHistory(rawHistory, context = {}) {
    if (!Array.isArray(rawHistory)) return [];
    return rawHistory.map((item, index) => {
      const entry = asObject(item);
      const proposal = asObject(entry.proposal);
      const candidate = firstDefined(entry.qwen_decision, entry.visual_decision, entry.visual_action, proposal.visual_action, proposal.action);
      const action = isQwenV2Decision(candidate)
        ? normalizeQwenV2Decision(candidate)
        : normalizeLegacyVisualAction(candidate, proposal, firstDefined(entry.after_scene, entry.scene));
      const verification = normalizeVerification(entry, action);
      const controllerGate = normalizeControllerGate(firstDefined(
        entry.controller_decision,
        entry.controller_gate,
      ));
      const historicalScope = normalizeHistoricalScope(entry, action, context);
      return {
        stepNumber: Number(firstDefined(entry.step_number, index + 1)),
        graphRevision: firstDefined(entry.task_revision, entry.revision, action.revision, null),
        subgoal: String(firstDefined(entry.current_subgoal, proposal.current_subgoal, "旧记录未提供")),
        subgoalId: String(firstDefined(entry.subgoal_id, asObject(entry.scope).subgoal_id, "")),
        observation: {
          id: String(firstDefined(entry.observation_id, action.observationId, "")),
          fingerprint: String(firstDefined(entry.fingerprint, action.fingerprint, verification.beforeFingerprint, "")),
        },
        action,
        controllerGate,
        reason: String(firstDefined(entry.reason, proposal.reason, "已执行并重新观察")),
        physicalActions: Number(firstDefined(asObject(entry.execution).physical_actions, 0)),
        verification,
        transition: normalizeTransition(
          entry,
          verification,
          context.replanHistory || [],
          context.currentRevision,
          context.sessionStatus,
          context.failedReason,
        ),
        scopeState: historicalScope,
        completionEvidence: normalizeStringList(entry.completion_evidence),
        evidence: normalizeStringList(firstDefined(
          asObject(entry.execution).evidence,
          asObject(entry.execution).after_frame_paths,
        )),
        raw: entry,
      };
    });
  }

  function normalizeScopeState(session, context) {
    const scope = context.scope;
    const hasScope = Boolean(
      scope.sessionId || scope.taskId || scope.deviceId || scope.subgoalId
      || scope.observationId || scope.fingerprint || scope.decisionNodeId
      || scope.actionDigest || scope.riskIds.length
    );
    const explicitReason = String(firstDefined(
      session.confirmation_invalid_reason,
      session.scope_invalid_reason,
      session.stale_scope_reason,
      "",
    ));
    const mismatches = [];
    const actionPhase = context.status === "awaiting_confirmation";
    const riskPhase = context.status === "awaiting_risk_confirmation";
    const expectedRiskIds = [...context.riskIds].map(String).sort();
    const actualRiskIds = [...scope.riskIds].map(String).sort();
    const requiredScopeMissing = actionPhase
      ? !scope.sessionId || !scope.taskId || !scope.deviceId
        || !Number.isInteger(scope.revision) || !scope.subgoalId
        || !scope.observationId || !scope.fingerprint
        || !scope.decisionNodeId || !scope.actionDigest
      : riskPhase
        ? !scope.sessionId || !scope.taskId || !scope.deviceId
          || !Number.isInteger(scope.revision) || !scope.subgoalId
        : false;
    if (hasScope) {
      if (scope.sessionId !== context.sessionId) mismatches.push("session_id");
      if (scope.taskId !== context.taskId) mismatches.push("task_id");
      if (scope.deviceId !== context.deviceId) mismatches.push("device_id");
      if (scope.revision !== context.revision) mismatches.push("revision");
      if (scope.subgoalId !== context.subgoalId) mismatches.push("subgoal_id");
      if (JSON.stringify(actualRiskIds) !== JSON.stringify(expectedRiskIds)) mismatches.push("risk_ids");
      if (actionPhase) {
        if (scope.observationId !== context.observationId) mismatches.push("observation_id");
        if (scope.fingerprint !== context.fingerprint) mismatches.push("fingerprint");
        if (scope.decisionNodeId !== context.decisionNodeId) mismatches.push("decision_node_id");
      }
      if (riskPhase && (
        scope.observationId || scope.fingerprint || scope.decisionNodeId || scope.actionDigest
      )) mismatches.push("risk_scope_extra_action_fields");
    }
    let state = "none";
    let reason = explicitReason;
    if (context.isTerminal) {
      state = "invalidated";
      reason ||= `会话已停止于 ${context.status}`;
    } else if ((actionPhase || riskPhase) && requiredScopeMissing) {
      state = "missing";
      reason = "等待确认但缺少完整作用域";
    } else if (explicitReason || mismatches.length) {
      state = "stale";
      reason ||= `作用域字段已变化：${mismatches.join("、")}`;
    } else if ((actionPhase || riskPhase) && hasScope) {
      state = "active";
      reason = "后端 scope 与当前权威任务、观察和动作字段一致";
    } else if (context.physicalActions > 0 && !hasScope) {
      state = "unknown";
      reason = "当前快照没有可验证的确认消费回执";
    }
    return { state, reason, mismatches, scope };
  }

  function normalizeStopState(status, failedReason, autoPauseReason) {
    const stopped = terminalStatuses.has(status);
    return {
      stopped,
      status: stopped ? status : "active",
      reason: String(
        (stopped && failedReason)
        || autoPauseReason
        || (stopped ? `会话已停止于 ${status}` : "")
      ),
    };
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
    const confirmationPhase = status === "awaiting_risk_confirmation" ? "risk" : "action";
    const authorityScope = normalizeConfirmationGateScope(firstDefined(
      confirmationPhase === "risk" ? session.risk_confirmation_scope : session.confirmation_scope,
      rawGate.scope,
    ));
    const currentExternalImpact = String(firstDefined(
      graph.current_external_impact,
      qwenContext.current_external_impact,
      currentSubgoal.externalImpact,
      "unknown",
    ));
    const gateRiskIds = normalizeStringList(firstDefined(
      asObject(session.risk_confirmation_scope).risk_ids,
      asObject(session.confirmation_scope).risk_ids,
      rawGate.risk_ids,
      currentSubgoal.riskIds,
    ));
    const currentRiskActions = riskActions.filter(item => (
      gateRiskIds.includes(item.id) || item.subgoalIds.includes(currentSubgoal.id)
    ));
    const gateRequired = Boolean(
      session.confirmation_ready === true
      || session.risk_confirmation_ready === true
      || status === "awaiting_risk_confirmation"
      || status === "awaiting_confirmation"
      || rawGate.required === true
      || externalImpacts.has(currentExternalImpact)
    );
    const gateState = String(firstDefined(
      rawGate.state,
      gateRequired
        ? (["awaiting_risk_confirmation", "awaiting_confirmation"].includes(status) ? status : "unconfirmed")
        : "not_required",
    ));
    const requiresConfirmation = Boolean(
      gateRequired
      || currentActionMeta.requires_confirmation
      || status === "awaiting_risk_confirmation"
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
    const controllerGate = normalizeControllerGate(session.controller_decision);
    const physicalActions = Number(firstDefined(session.physical_actions, 0));
    const failedReason = String(firstDefined(session.failed_reason, ""));
    const autoPauseReason = String(firstDefined(session.auto_pause_reason, ""));
    const replanHistory = Array.isArray(graph.replan_history) ? graph.replan_history : [];
    const history = normalizeHistory(session.history, {
      sessionId: String(firstDefined(session.session_id, session.id, "")),
      replanHistory,
      currentRevision: revision,
      sessionStatus: status,
      failedReason,
    });
    const trustedObservation = asObject(firstDefined(
      session.trusted_observation,
      asObject(qwenDecisionRaw).trusted_observation,
    ));
    const currentObservation = {
      id: String(firstDefined(
        trustedObservation.observation_id,
        visualAction.observationId,
        authorityScope.observationId,
        "",
      )),
      fingerprint: String(firstDefined(
        trustedObservation.fingerprint,
        visualAction.fingerprint,
        authorityScope.fingerprint,
        "",
      )),
    };
    const stopState = normalizeStopState(status, failedReason, autoPauseReason);
    const scopeState = normalizeScopeState(session, {
      scope: authorityScope,
      sessionId: String(firstDefined(session.session_id, session.id, "")),
      taskId,
      deviceId,
      revision,
      subgoalId: currentSubgoal.id,
      observationId: currentObservation.id,
      fingerprint: currentObservation.fingerprint,
      decisionNodeId: visualAction.decisionNodeId,
      riskIds: gateRiskIds,
      status,
      isTerminal: stopState.stopped,
      physicalActions,
    });
    const currentTransitionKind = stopState.stopped
      ? "stopped"
      : status === "blocked"
        ? "blocked"
        : status === "replanning"
          ? "replan"
          : status === "awaiting_confirmation"
            ? "awaiting_confirmation"
            : status === "awaiting_risk_confirmation"
              ? "awaiting_risk_confirmation"
              : "observing";
    const currentTrace = {
      phase: stopState.stopped ? "terminal" : "current",
      stepNumber: Number(firstDefined(session.step_number, currentSubgoal.index, 1)),
      graphRevision: revision,
      subgoal: currentSubgoal.label,
      subgoalId: currentSubgoal.id,
      observation: currentObservation,
      action: stopState.stopped
        ? { status: "terminal", actionType: "", semanticTarget: "", reason: stopState.reason }
        : visualAction,
      controllerGate,
      physicalActions: 0,
      verification: {
        outcome: stopState.stopped
          ? "terminal"
          : visualAction.status === "action"
            ? "not_executed"
            : history.length
              ? "awaiting_next_action"
              : "not_executed",
        matched: false,
        beforeFingerprint: currentObservation.fingerprint,
        afterObservationId: "",
        afterFingerprint: "",
        errors: [],
        evidence: [],
        raw: {},
      },
      transition: {
        kind: currentTransitionKind,
        trigger: "",
        reason: failedReason || autoPauseReason || controllerGate.reason,
        fromRevision: revision,
        toRevision: null,
        raw: {},
      },
      scopeState,
      status,
    };
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
      history,
      taskGraph: {
        revision,
        status: String(firstDefined(graph.status, graph.task_status, status)),
        currentSubgoalId: currentSubgoal.id,
        replanHistory: replanHistory.map(item => ({ ...asObject(item) })),
      },
      executionTrace: [
        ...history.map(item => ({
          ...item,
          phase: "completed",
          status: "completed_round",
        })),
        currentTrace,
      ],
      scopeState,
      stopState,
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
          phase: confirmationPhase,
          raw: rawGate,
        },
        maxPhysicalActions: 1,
      },
      controllerGate,
      physicalActions,
      evidence: normalizeStringList(session.evidence),
      autoPauseReason,
      failedReason,
      raw: session,
    };
  }

  function buildRequestPayload(sessionDeviceId, values = {}) {
    const deviceId = String(sessionDeviceId || "").trim();
    if (!deviceId) throw new Error("会话缺少锁定的 device_id。");
    return { ...values, device_id: deviceId };
  }

  function buildAutoRequestPayload(sessionDeviceId, confirmationPayload, values = {}) {
    if (
      !confirmationPayload
      || confirmationPayload.confirmed !== true
      || !asObject(confirmationPayload.confirmation).observation_id
      || !asObject(confirmationPayload.confirmation).fingerprint
      || !asObject(confirmationPayload.confirmation).decision_node_id
      || !asObject(confirmationPayload.confirmation).action_digest
    ) {
      throw new Error("安全连续推进必须携带当前精确动作确认。")
    }
    return buildRequestPayload(sessionDeviceId, {
      ...confirmationPayload,
      max_physical_actions: 1,
      max_iterations: 1,
    });
  }

  function confirmationScope(session, sessionDeviceId) {
    if (!session) throw new Error("当前没有可确认的会话。");
    if (session.scopeState?.state !== "active") {
      throw new Error("当前确认作用域缺失、已消费或已经变化失效，请重新观察。");
    }
    const gateScope = session.risk?.confirmationGate?.scope || {};
    if (String(sessionDeviceId || "") !== String(session.deviceId || "")) {
      throw new Error("确认设备已经变化，请重新确认。");
    }
    return {
      session_id: gateScope.sessionId,
      task_id: gateScope.taskId,
      device_id: gateScope.deviceId,
      revision: gateScope.revision,
      subgoal_id: gateScope.subgoalId,
      risk_ids: [...gateScope.riskIds].map(String).sort(),
      observation_id: gateScope.observationId,
      fingerprint: gateScope.fingerprint,
      decision_node_id: gateScope.decisionNodeId,
      action_digest: gateScope.actionDigest,
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
      decision_node_id: scope.decision_node_id,
      action_digest: scope.action_digest,
    });
  }

  function createConfirmationGrant(session, sessionDeviceId) {
    if (!session?.risk?.requiresConfirmation) throw new Error("当前步骤不需要风险确认。");
    if (session.protocol !== "deepseek-task-graph-v3" || session.compatibilityFallback) {
      throw new Error("当前会话没有正式DeepSeek v3确认作用域，拒绝确认。");
    }
    const scope = confirmationScope(session, sessionDeviceId);
    const phase = session.risk?.confirmationGate?.phase === "risk" ? "risk" : "action";
    if (
      !scope.session_id
      || !scope.task_id
      || !scope.device_id
      || !scope.subgoal_id
      || !Number.isInteger(scope.revision)
      || scope.device_id !== String(sessionDeviceId || "")
    ) {
      throw new Error("当前DeepSeek v3确认作用域不完整或设备不一致。");
    }
    if (externalImpacts.has(session.risk.currentExternalImpact) && !scope.risk_ids.length) {
      throw new Error("外部状态或未知影响步骤缺少 risk_ids，拒绝确认。");
    }
    if (phase === "action" && (
      !scope.observation_id || !scope.fingerprint
      || !scope.decision_node_id || !scope.action_digest
    )) {
      throw new Error("当前动作确认缺少 observation_id、fingerprint、decision_node_id 或 action_digest。");
    }
    return { scope, phase, consumed: false };
  }

  function consumeConfirmationGrant(grant, session, sessionDeviceId) {
    if (!grant || grant.consumed) throw new Error("本次风险确认已使用或不存在。");
    const currentScope = confirmationScope(session, sessionDeviceId);
    if (scopeFingerprint(grant.scope) !== scopeFingerprint(currentScope)) {
      throw new Error("任务、revision、子目标、风险或设备已经变化，请重新确认。");
    }
    grant.consumed = true;
    const confirmation = {
      session_id: grant.scope.session_id,
      task_id: grant.scope.task_id,
      device_id: grant.scope.device_id,
      revision: grant.scope.revision,
      subgoal_id: grant.scope.subgoal_id,
      risk_ids: [...grant.scope.risk_ids],
    };
    if (grant.phase !== "risk") {
      confirmation.observation_id = grant.scope.observation_id;
      confirmation.fingerprint = grant.scope.fingerprint;
      confirmation.decision_node_id = grant.scope.decision_node_id;
      confirmation.action_digest = grant.scope.action_digest;
    }
    return {
      confirmed: true,
      confirmation,
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

  function adaptCapabilityTrial(value) {
    const envelope = asObject(value);
    const raw = asObject(envelope.trial || value);
    const sessionRaw = asObject(raw.session);
    const report = asObject(raw.report);
    const promotionScope = asObject(raw.promotion_scope);
    const actionScope = asObject(raw.action_confirmation_scope);
    const riskScope = asObject(raw.risk_confirmation_scope);
    const trialId = String(firstDefined(raw.trial_id, ""));
    const deviceId = String(firstDefined(raw.device_id, ""));
    const action = String(firstDefined(raw.candidate_action, ""));
    const session = Object.keys(sessionRaw).length
      ? adaptSession(sessionRaw, { fallbackDeviceId: deviceId })
      : null;
    return {
      trialId,
      deviceId,
      action,
      text: String(firstDefined(raw.text, "")),
      codeRevision: String(firstDefined(raw.code_revision, "")),
      session,
      status: String(firstDefined(sessionRaw.status, report.status, "unknown")),
      physicalActions: Number(firstDefined(sessionRaw.physical_actions, report.physical_actions, 0)),
      actionConfirmationScope: actionScope,
      riskConfirmationScope: riskScope,
      report: Object.keys(report).length ? report : null,
      passed: report.status === "passed",
      promotionScope: Object.keys(promotionScope).length ? promotionScope : null,
      promotion: Object.keys(asObject(raw.promotion)).length ? asObject(raw.promotion) : null,
      requiresRestart: raw.requires_restart === true,
      readOnlyRecovered: raw.read_only_recovered === true,
      raw,
    };
  }

  function capabilityScopeFingerprint(scope) {
    return JSON.stringify(Object.keys(scope).sort().reduce((result, key) => {
      const value = scope[key];
      result[key] = Array.isArray(value) ? [...value].map(String).sort() : value;
      return result;
    }, {}));
  }

  function currentCapabilityConfirmationScope(trial) {
    const view = adaptCapabilityTrial(trial);
    const riskPhase = view.status === "awaiting_risk_confirmation";
    const scope = asObject(riskPhase
      ? view.riskConfirmationScope
      : view.actionConfirmationScope);
    if (
      !view.trialId
      || !view.deviceId
      || !view.action
      || scope.trial_id !== view.trialId
      || scope.device_id !== view.deviceId
      || scope.action !== view.action
      || !scope.session_id
      || !scope.task_id
      || !Number.isInteger(scope.revision)
      || !scope.subgoal_id
      || !Array.isArray(scope.risk_ids)
      || (!riskPhase && (!scope.observation_id || !scope.fingerprint))
      || (!riskPhase && view.session?.visualAction?.actionType !== view.action)
    ) {
      throw new Error("真机验收确认缺少 trial、action 或精确画面作用域。");
    }
    return { view, phase: riskPhase ? "risk" : "action", scope: { ...scope } };
  }

  function createCapabilityConfirmationGrant(trial) {
    const current = currentCapabilityConfirmationScope(trial);
    return {
      phase: current.phase,
      scope: current.scope,
      fingerprint: capabilityScopeFingerprint(current.scope),
      consumed: false,
    };
  }

  function consumeCapabilityConfirmationGrant(grant, trial) {
    if (!grant || grant.consumed) throw new Error("本次真机验收确认已使用或不存在。");
    const current = currentCapabilityConfirmationScope(trial);
    if (
      grant.phase !== current.phase
      || grant.fingerprint !== capabilityScopeFingerprint(current.scope)
    ) {
      throw new Error("验收 trial、action、任务或画面已经变化，请重新确认。");
    }
    grant.consumed = true;
    return { confirmed: true, confirmation: { ...grant.scope } };
  }

  function currentPromotionScope(trial) {
    const view = adaptCapabilityTrial(trial);
    const scope = asObject(view.promotionScope);
    if (
      !view.passed
      || view.readOnlyRecovered
      || view.promotion
      || !view.trialId
      || !view.deviceId
      || !view.action
      || scope.trial_id !== view.trialId
      || scope.device_id !== view.deviceId
      || scope.action !== view.action
      || !/^[0-9a-f]{64}$/.test(String(scope.report_sha256 || ""))
      || !/^[0-9a-f]{64}$/.test(String(scope.registry_sha256 || ""))
    ) {
      throw new Error("验收报告未通过、摘要无效或能力已经晋级。");
    }
    return { view, scope: { ...scope } };
  }

  function createPromotionGrant(trial) {
    const current = currentPromotionScope(trial);
    return {
      scope: current.scope,
      fingerprint: capabilityScopeFingerprint(current.scope),
      consumed: false,
    };
  }

  function consumePromotionGrant(grant, trial) {
    if (!grant || grant.consumed) throw new Error("本次能力晋级确认已使用或不存在。");
    const current = currentPromotionScope(trial);
    if (grant.fingerprint !== capabilityScopeFingerprint(current.scope)) {
      throw new Error("验收报告或设备注册表摘要已经变化，请重新确认。");
    }
    grant.consumed = true;
    return { confirmed: true, ...grant.scope };
  }

  return {
    adaptCapabilityTrial,
    adaptSession,
    buildRequestPayload,
    buildAutoRequestPayload,
    consumeConfirmationGrant,
    consumeCapabilityConfirmationGrant,
    consumePromotionGrant,
    createCapabilityConfirmationGrant,
    createConfirmationGrant,
    createPromotionGrant,
    displayValue,
    runAutoAdvanceLoop,
    shouldAutoAdvance,
  };
});
