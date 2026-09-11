(function attachProtocolAdapter(root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.UniversalAgentProtocol = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function createProtocolAdapter() {
  const terminalStatuses = new Set(["succeeded", "completed", "blocked", "failed", "cancelled"]);
  const formalVisualProtocol = "2026-09-06-single-visual-task-v1";
  const formalQwenProtocol = "2026-09-06-qwen-whole-task-v19";

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

  function normalizeConfirmationGateScope(rawScope) {
    const scope = asObject(rawScope);
    return {
      sessionId: String(firstDefined(scope.session_id, "")),
      taskId: String(firstDefined(scope.task_id, "")),
      deviceId: String(firstDefined(scope.device_id, "")),
      revision: firstDefined(scope.revision, null),
      stepId: String(firstDefined(scope.step_id, "")),
      effectIds: normalizeStringList(scope.effect_ids),
      observationId: String(firstDefined(scope.observation_id, "")),
      fingerprint: String(firstDefined(scope.fingerprint, "")),
      decisionNodeId: String(firstDefined(scope.decision_node_id, "")),
      actionDigest: String(firstDefined(scope.action_digest, "")),
      intentDigest: String(firstDefined(scope.intent_digest, "")),
    };
  }

  function isQwenDecision(value) {
    const decision = asObject(value);
    return String(decision.protocol_version || "") === formalQwenProtocol
      && ["action", "finish"].includes(String(decision.status || "").toLowerCase());
  }

  function normalizeQwenDecision(rawDecision) {
    const decision = asObject(rawDecision);
    const protocolVersion = String(firstDefined(decision.protocol_version, ""));
    const rawStatus = String(firstDefined(decision.status, "unknown")).toLowerCase();
    const validProtocol = protocolVersion === formalQwenProtocol;
    const status = validProtocol && ["action", "finish"].includes(rawStatus) ? rawStatus : "unknown";
    const nextAction = asObject(decision.next_action);
    const params = asObject(nextAction.params);
    const elementId = String(firstDefined(params.element_id, ""));
    const actionType = status === "action" ? String(firstDefined(nextAction.action, "")) : "";
    const semanticTarget = String(firstDefined(
      params.label,
      params.target,
      elementId,
      "未提供语义目标",
    ));
    const isExecutable = validProtocol && status === "action" && Boolean(actionType);
    return {
      protocol: validProtocol ? "qwen-same-response-action-finish-v9" : "unsupported-protocol",
      protocolVersion,
      status,
      decisionNodeId: String(firstDefined(nextAction.node_id, "")),
      actionType,
      semanticTarget,
      elementId,
      expectedChange: {},
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
      entry.visual_outcome,
      execution.visual_outcome,
      verification.visual_outcome,
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
      executionOutcome: String(firstDefined(execution.action_outcome, verification.execution_outcome, "unknown")),
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

  function normalizeTransition(entry, verification) {
    const explicit = asObject(firstDefined(entry.transition, entry.post_action_transition));
    const sourceRevision = firstDefined(entry.task_revision, entry.revision, null);
    const declaredKind = String(firstDefined(explicit.transition_kind, explicit.kind, ""));
    const kind = declaredKind || (verification.afterObservationId ? "new_screenshot_decision" : "unknown");
    return {
      kind,
      trigger: "",
      reason: String(firstDefined(explicit.reason, "")),
      fromRevision: sourceRevision,
      toRevision: null,
      raw: explicit,
    };
  }

  function normalizeHistoricalScope(entry, action, context) {
    const execution = asObject(entry.execution);
    const receipt = asObject(firstDefined(
      entry.confirmation_receipt,
      execution.confirmation_receipt,
    ));
    const scope = normalizeConfirmationGateScope(receipt.scope);
    const expectedStepId = String(firstDefined(entry.step_id, asObject(entry.scope).step_id, ""));
    const expectedEffectIds = Array.isArray(entry.effect_ids)
      ? normalizeStringList(entry.effect_ids).map(String).sort()
      : null;
    const authoritative = receipt.authoritative === true && receipt.consumed === true;
    const exact = authoritative
      && Boolean(context.sessionId)
      && scope.sessionId === context.sessionId
      && scope.taskId === action.taskId
      && scope.deviceId === action.deviceId
      && scope.revision === action.revision
      && Boolean(expectedStepId)
      && scope.stepId === expectedStepId
      && expectedEffectIds !== null
      && JSON.stringify([...scope.effectIds].map(String).sort()) === JSON.stringify(expectedEffectIds)
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
        reason: "该记录没有可验证的确认消费回执",
        mismatches: [],
        scope: {},
      };
  }

  function normalizeHistory(rawHistory, context = {}) {
    if (!Array.isArray(rawHistory)) return [];
    return rawHistory.map((item, index) => {
      const entry = asObject(item);
      const candidate = entry.qwen_decision;
      const action = isQwenDecision(candidate)
        ? normalizeQwenDecision(candidate)
        : normalizeQwenDecision({ status: "unknown", reason: "该历史轮次没有当前 Qwen 决策。" });
      const verification = normalizeVerification(entry, action);
      const controllerGate = normalizeControllerGate(entry.controller_decision);
      const historicalScope = normalizeHistoricalScope(entry, action, context);
      return {
        stepNumber: Number(firstDefined(entry.step_number, index + 1)),
        taskRevision: firstDefined(entry.task_revision, action.revision, null),
        taskContext: String(firstDefined(entry.current_step, "记录未提供")),
        stepId: String(firstDefined(entry.step_id, asObject(entry.scope).step_id, "")),
        observation: {
          id: String(firstDefined(entry.observation_id, action.observationId, "")),
          fingerprint: String(firstDefined(entry.fingerprint, action.fingerprint, verification.beforeFingerprint, "")),
        },
        action,
        controllerGate,
        reason: String(firstDefined(entry.reason, "已执行并重新观察")),
        physicalActions: Number(firstDefined(asObject(entry.execution).physical_actions, 0)),
        verification,
        transition: normalizeTransition(entry, verification),
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
      scope.sessionId || scope.taskId || scope.deviceId || scope.stepId
      || scope.observationId || scope.fingerprint || scope.decisionNodeId
      || scope.actionDigest || scope.intentDigest || scope.effectIds.length
    );
    const explicitReason = String(firstDefined(
      session.confirmation_invalid_reason,
      session.scope_invalid_reason,
      session.stale_scope_reason,
      "",
    ));
    const mismatches = [];
    const actionPhase = context.status === "awaiting_confirmation";
    const effectPhase = context.status === "awaiting_effect_confirmation";
    const expectedEffectIds = [...context.effectIds].map(String).sort();
    const actualEffectIds = [...scope.effectIds].map(String).sort();
    const requiredScopeMissing = actionPhase
      ? !scope.sessionId || !scope.taskId || !scope.deviceId
        || !Number.isInteger(scope.revision) || !scope.stepId
        || !scope.observationId || !scope.fingerprint
        || !scope.decisionNodeId || !scope.actionDigest
      : effectPhase
        ? !scope.sessionId || !scope.taskId || !scope.deviceId
          || !Number.isInteger(scope.revision) || !scope.stepId
          || !scope.intentDigest
        : false;
    if (hasScope) {
      if (scope.sessionId !== context.sessionId) mismatches.push("session_id");
      if (scope.taskId !== context.taskId) mismatches.push("task_id");
      if (scope.deviceId !== context.deviceId) mismatches.push("device_id");
      if (scope.revision !== context.revision) mismatches.push("revision");
      if (scope.stepId !== context.stepId) mismatches.push("step_id");
      if (JSON.stringify(actualEffectIds) !== JSON.stringify(expectedEffectIds)) mismatches.push("effect_ids");
      if (actionPhase) {
        if (scope.observationId !== context.observationId) mismatches.push("observation_id");
        if (scope.fingerprint !== context.fingerprint) mismatches.push("fingerprint");
        if (scope.decisionNodeId !== context.decisionNodeId) mismatches.push("decision_node_id");
      }
      if (effectPhase && (
        scope.observationId || scope.fingerprint || scope.decisionNodeId || scope.actionDigest
      )) mismatches.push("effect_scope_extra_action_fields");
    }
    let state = "none";
    let reason = explicitReason;
    if (context.isTerminal) {
      state = "invalidated";
      reason ||= `会话已停止于 ${context.status}`;
    } else if ((actionPhase || effectPhase) && requiredScopeMissing) {
      state = "missing";
      reason = "等待确认但缺少完整作用域";
    } else if (explicitReason || mismatches.length) {
      state = "stale";
      reason ||= `作用域字段已变化：${mismatches.join("、")}`;
    } else if ((actionPhase || effectPhase) && hasScope) {
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
    const protocolVersion = String(session.task_context_protocol || "");
    const formal = protocolVersion === formalVisualProtocol;
    const qwenDecisionRaw = isQwenDecision(session.qwen_decision) ? session.qwen_decision : null;
    const objective = String(session.raw_goal || "未命名目标");
    const currentStep = {id: "step_" + session.step_number, label: objective, index: session.step_number};
    const steps = (session.history || []).map(item => ({id: "step_" + item.step_number,
      index: item.step_number, label: item.execution?.resolved_action?.kind || "已执行动作",
      reason: item.visual_outcome || item.execution?.action_outcome || "", status: "completed"}));
    const status = String(firstDefined(session.status, "idle")).toLowerCase();
    const taskId = String(firstDefined(session.task_id, session.session_id, ""));
    const deviceId = String(firstDefined(session.device_id, options.fallbackDeviceId, ""));
    const revision = firstDefined(session.revision, null);
    const scene = asObject(session.current_scene);
    const visualAction = qwenDecisionRaw
      ? normalizeQwenDecision(qwenDecisionRaw)
      : normalizeQwenDecision({ status: "unknown", reason: "当前会话尚无 Qwen 决策。" });
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

    const confirmationPhase = status === "awaiting_effect_confirmation" ? "effect" : "action";
    const authorityScope = normalizeConfirmationGateScope(
      confirmationPhase === "effect" ? session.effect_confirmation_scope : session.confirmation_scope,
    );
    const currentExecutionClass = "whole_task";
    const gateEffectIds = [...authorityScope.effectIds];
    const currentEffectIntents = gateEffectIds.map(kind => ({
      id: kind, kind, policyLevel: ["authentication", "financial_transaction"].includes(kind)
        ? "confirmation_required" : "automatic", expectedResults: [],
    }));
    const effectIntents = currentEffectIntents;
    const gateRequired = Boolean(
      session.confirmation_ready === true
      || session.effect_confirmation_ready === true
      || status === "awaiting_effect_confirmation"
      || status === "awaiting_confirmation"
    );
    const gateState = gateRequired
      ? (["awaiting_effect_confirmation", "awaiting_confirmation"].includes(status) ? status : "unconfirmed")
      : "not_required";
    const requiresConfirmation = Boolean(
      gateRequired
      || status === "awaiting_effect_confirmation"
      || status === "awaiting_confirmation"
    );
    const hasCurrentEffect = gateEffectIds.length > 0 || currentEffectIntents.length > 0;
    const blocksAutomatic = Boolean(
      requiresConfirmation
      || gateState === "awaiting_confirmation"
      || gateState === "unconfirmed"
    );
    const targetApps = [];
    const controllerGate = normalizeControllerGate(session.controller_decision);
    const physicalActions = Number(firstDefined(session.physical_actions, 0));
    const failedReason = String(firstDefined(session.failed_reason, ""));
    const autoPauseReason = String(firstDefined(session.auto_pause_reason, ""));
    const history = normalizeHistory(session.history, {
      sessionId: String(firstDefined(session.session_id, "")),
    });
    const trustedObservation = asObject(session.trusted_observation);
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
      sessionId: String(firstDefined(session.session_id, "")),
      taskId,
      deviceId,
      revision,
      stepId: currentStep.id,
      observationId: currentObservation.id,
      fingerprint: currentObservation.fingerprint,
      decisionNodeId: visualAction.decisionNodeId,
      effectIds: gateEffectIds,
      status,
      isTerminal: stopState.stopped,
      physicalActions,
    });
    const currentTransitionKind = stopState.stopped
      ? "stopped"
      : status === "blocked"
        ? "blocked"
        : status === "awaiting_confirmation"
            ? "awaiting_confirmation"
            : status === "awaiting_effect_confirmation"
              ? "awaiting_effect_confirmation"
              : "observing";
    const currentTrace = {
      phase: stopState.stopped ? "terminal" : "current",
      stepNumber: Number(firstDefined(session.step_number, currentStep.index, 1)),
      taskRevision: revision,
      taskContext: currentStep.label,
      stepId: currentStep.id,
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
      protocol: formal ? "single-visual-task-v1" : "unsupported-protocol",
      protocolVersion,
      compatibilityFallback: !formal,
      sessionId: String(firstDefined(session.session_id, "")),
      taskId,
      deviceId,
      revision,
      status,
      isTerminal: terminalStatuses.has(status),
      stepNumber: Number(firstDefined(session.step_number, currentStep.index, 1)),
      objective,
      executionBudget: asObject(session.execution_budget),
      autoPauseReason,
      targetApps,
      appName: "",
      constraints: [],
      completionConditions: [],
      steps,
      currentStep,
      visualAction,
      scene: {
        summary: String(firstDefined(scene.summary, scene.screen_id, "尚未识别")),
        screenId: String(firstDefined(scene.screen_id, "unknown")),
        stable: scene.stable === true,
        confidence: firstDefined(scene.confidence, null),
        raw: scene,
      },
      history,
      taskState: {
        revision,
        status: status,
        currentStepId: currentStep.id,
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
      effectPolicy: {
        requiresConfirmation,
        hasCurrentEffect,
        blocksAutomatic,
        currentExecutionClass,
        effectIds: gateEffectIds,
        actions: effectIntents,
        currentActions: currentEffectIntents,
        intentPreview: asObject(session.effect_confirmation_preview),
        confirmationGate: {
          required: gateRequired,
          state: gateState,
          effectIds: gateEffectIds,
          scope: authorityScope,
          phase: confirmationPhase,
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

  function buildAutoRequestPayload(sessionDeviceId, values = {}) {
    const body = { confirmed: false, confirmation: null };
    for (const [source, field] of [["maxPhysicalActions", "max_physical_actions"], ["maxObservations", "max_observations"]]) {
      if (values[source] === undefined || values[source] === null) continue;
      if (!Number.isInteger(values[source]) || values[source] < 1) throw new Error("整任务预算必须为正整数。");
      body[field] = values[source];
    }
    return buildRequestPayload(sessionDeviceId, body);
  }

  function confirmationScope(session, sessionDeviceId) {
    if (!session) throw new Error("当前没有可确认的会话。");
    if (session.scopeState?.state !== "active") {
      throw new Error("当前确认作用域缺失、已消费或已经变化失效，请重新观察。");
    }
    const gateScope = session.effectPolicy?.confirmationGate?.scope || {};
    if (String(sessionDeviceId || "") !== String(session.deviceId || "")) {
      throw new Error("确认设备已经变化，请重新确认。");
    }
    return {
      session_id: gateScope.sessionId,
      task_id: gateScope.taskId,
      device_id: gateScope.deviceId,
      revision: gateScope.revision,
      step_id: gateScope.stepId,
      effect_ids: [...gateScope.effectIds].map(String).sort(),
      observation_id: gateScope.observationId,
      fingerprint: gateScope.fingerprint,
      decision_node_id: gateScope.decisionNodeId,
      action_digest: gateScope.actionDigest,
      intent_digest: gateScope.intentDigest,
    };
  }

  function scopeFingerprint(scope) {
    return JSON.stringify({
      session_id: scope.session_id,
      task_id: scope.task_id,
      device_id: scope.device_id,
      revision: scope.revision,
      step_id: scope.step_id,
      effect_ids: [...scope.effect_ids].sort(),
      observation_id: scope.observation_id,
      fingerprint: scope.fingerprint,
      decision_node_id: scope.decision_node_id,
      action_digest: scope.action_digest,
      intent_digest: scope.intent_digest,
    });
  }

  function createConfirmationGrant(session, sessionDeviceId) {
    if (!session?.effectPolicy?.requiresConfirmation) throw new Error("当前步骤不需要效果确认。");
    if (session.protocol !== "single-visual-task-v1" || session.compatibilityFallback) {
      throw new Error("当前会话没有正式 单视觉整任务 确认作用域，拒绝确认。");
    }
    const scope = confirmationScope(session, sessionDeviceId);
    const phase = session.effectPolicy?.confirmationGate?.phase === "effect" ? "effect" : "action";
    if (
      !scope.session_id
      || !scope.task_id
      || !scope.device_id
      || !scope.step_id
      || !Number.isInteger(scope.revision)
      || scope.device_id !== String(sessionDeviceId || "")
    ) {
      throw new Error("当前 单视觉整任务 确认作用域不完整或设备不一致。");
    }
    if (phase === "effect" && (!scope.effect_ids.length || !scope.intent_digest)) {
      throw new Error("当前效果确认缺少 effect_ids 或绑定目标内容的 intent_digest。");
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
    if (!grant || grant.consumed) throw new Error("本次确认已使用或不存在。");
    const currentScope = confirmationScope(session, sessionDeviceId);
    if (scopeFingerprint(grant.scope) !== scopeFingerprint(currentScope)) {
      throw new Error("任务、revision、当前步骤、效果或设备已经变化，请重新确认。");
    }
    grant.consumed = true;
    const confirmation = {
      session_id: grant.scope.session_id,
      task_id: grant.scope.task_id,
      device_id: grant.scope.device_id,
      revision: grant.scope.revision,
      step_id: grant.scope.step_id,
      effect_ids: [...grant.scope.effect_ids],
    };
    if (grant.phase === "effect") {
      confirmation.intent_digest = grant.scope.intent_digest;
    } else {
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
    const session = context && context.session;
    if (!session || context.paused || context.busy || session.isTerminal) return false;
    if (session.status !== "awaiting_confirmation") return false;
    if (session.scopeState?.state !== "active") return false;
    if (!session.visualAction?.isExecutable) return false;
    return !session.effectPolicy?.requiresConfirmation;
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
    const effectScope = asObject(raw.effect_confirmation_scope);
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
      effectConfirmationScope: effectScope,
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
    const effectPhase = view.status === "awaiting_effect_confirmation";
    const scope = asObject(effectPhase
      ? view.effectConfirmationScope
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
      || !scope.step_id
      || !Array.isArray(scope.effect_ids)
      || (!effectPhase && (!scope.observation_id || !scope.fingerprint))
      || (!effectPhase && view.session?.visualAction?.actionType !== view.action)
    ) {
      throw new Error("真机验收确认缺少 trial、action 或精确画面作用域。");
    }
    return { view, phase: effectPhase ? "effect" : "action", scope: { ...scope } };
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
