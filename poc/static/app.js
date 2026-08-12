const Protocol = window.UniversalAgentProtocol;
if (!Protocol) throw new Error("通用 Agent 前端协议适配层未加载。");

const state = {
  token: "",
  mock: false,
  device: {},
  deviceId: localStorage.getItem("visual-agent-device-id") || "device-local-01",
  sessionDeviceId: "",
  supervisedSession: null,
  visionStage: "",
  paused: false,
  busy: false,
  stopRequested: false,
  pendingConfirmationGrant: null,
};

const statusNames = {
  idle: "等待目标",
  ready: "准备执行",
  awaiting_confirmation: "等待确认",
  awaiting_risk_confirmation: "等待风险范围确认",
  paused_after_action: "已完成一步",
  running: "执行中",
  succeeded: "目标完成",
  completed: "目标完成",
  blocked: "已阻止",
  failed: "失败",
  cancelled: "已停止",
};

const decisionStatusNames = {
  action: "唯一下一动作",
  finished: "Qwen 判断已完成",
  blocked: "Qwen 已阻止",
  unknown: "等待视觉决策",
};

const semanticActionNames = {
  ensure_app: "打开目标 App",
  observe: "重新观察页面",
  tap_semantic: "点击语义控件",
  dismiss_overlay: "关闭当前弹层",
  swipe: "滑动当前页面",
  back: "返回上一页",
  long_press: "长按目标控件",
  drag: "拖动目标控件",
  input_verified_text: "输入并核对文字",
  wait_for_change: "等待页面变化",
  record_verified_result: "记录已验证结果",
  finish: "完成本次任务",
};

async function api(path, options = {}) {
  const headers = { ...(options.headers || {}) };
  if (options.method && options.method !== "GET") headers["X-Control-Token"] = state.token;
  if (options.body) headers["Content-Type"] = "application/json";
  const response = await fetch(path, { ...options, headers });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = data.detail;
    const message = typeof detail === "string" ? detail : (detail?.error || JSON.stringify(detail || {}));
    throw new Error(message || `请求失败（${response.status}）`);
  }
  return data;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function toast(message, isError = false) {
  const element = document.querySelector("#toast");
  element.textContent = message;
  element.className = `toast show${isError ? " error" : ""}`;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => element.className = "toast", 3600);
}

function setDot(selector, stateName) {
  const element = document.querySelector(selector);
  if (element) element.className = `dot ${stateName}`;
}

function sessionView() {
  if (!state.supervisedSession) return null;
  return Protocol.adaptSession(state.supervisedSession, {
    fallbackDeviceId: state.sessionDeviceId || state.deviceId,
  });
}

function lockedSessionDeviceId() {
  return state.sessionDeviceId || sessionView()?.deviceId || state.deviceId;
}

function observerStatus() {
  return state.device.execution_architecture?.universal_agent?.observer || {};
}

function currentVisionStageLabel() {
  const observer = observerStatus();
  if (state.visionStage) return state.visionStage;
  if (observer.current_stage && observer.current_stage !== "idle") {
    return observer.current_stage_label || observer.current_stage;
  }
  return "等待任务";
}

function actionLabel(action) {
  if (!action?.actionType) return "等待重新观察";
  return semanticActionNames[action.actionType] || action.actionType;
}

function confidenceLabel(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${Math.round(number * 100)}%` : "—";
}

function planState(subgoal, view) {
  if (subgoal.id === view.currentSubgoal.id) return "current";
  if (["done", "completed", "succeeded"].includes(subgoal.status)) return "done";
  if (["failed", "blocked", "cancelled"].includes(subgoal.status)) return "blocked";
  return "waiting";
}

function renderStatus() {
  const device = state.device || {};
  const view = sessionView();
  setDot("#controllerDot", device.controller_online ? (device.busy ? "warn" : "online") : "offline");
  setDot("#cameraDot", device.camera_online ? "online" : "offline");
  setDot("#agentDot", state.busy ? "warn" : (view && !view.isTerminal ? "online" : "neutral"));
  document.querySelector("#controllerText").textContent = device.controller_online
    ? (device.busy ? "当前动作执行中" : "在线且空闲")
    : "离线";
  document.querySelector("#cameraText").textContent = device.camera_online ? "实时画面可用" : "画面不可用";
  document.querySelector("#agentTextStatus").textContent = state.paused
    ? "人工暂停"
    : (state.busy ? currentVisionStageLabel() : (view ? (statusNames[view.status] || view.status) : "等待目标"));
  document.querySelector("#safetyText").textContent = view?.risk.blocksAutomatic
    ? "风险确认门已关闭"
    : "一次一动作";
  setDot("#safetyDot", view?.risk.blocksAutomatic ? "warn" : "online");
}

function renderGoalAndPlan() {
  const view = sessionView();
  const goalElement = document.querySelector("#goalSummary");
  const planElement = document.querySelector("#planList");
  const badge = document.querySelector("#planBadge");
  if (!view) {
    goalElement.className = "goal-summary empty-state";
    goalElement.textContent = "输入目标后，这里会展示目标、约束和可验证的完成条件。";
    planElement.innerHTML = "";
    badge.className = "pill neutral";
    badge.textContent = "等待目标";
    return;
  }

  const gateScope = view.risk.confirmationGate.scope || {};
  const gateScopeText = gateScope.taskId
    ? `scope ${gateScope.taskId} / ${gateScope.deviceId || "—"} / r${gateScope.revision ?? "—"} / ${gateScope.subgoalId || "—"}`
    : "";
  goalElement.className = "goal-summary";
  goalElement.innerHTML = `
    <div class="goal-title-row">
      <div><span>目标 · ${escapeHtml(view.taskId || "待分配任务 ID")} · revision ${escapeHtml(view.revision ?? "—")}</span><strong>${escapeHtml(view.objective)}</strong></div>
      <code>${escapeHtml(view.deviceId || lockedSessionDeviceId())}</code>
    </div>
    <div class="goal-chips">
      ${view.protocolVersion ? `<span>协议 · ${escapeHtml(view.protocolVersion)}</span>` : ""}
      ${view.compatibilityFallback ? `<span>旧版兼容数据 · 不作为 v3 确认依据</span>` : ""}
      ${view.targetApps.map(app => `<span>目标应用 · ${escapeHtml(app.name)}${app.id ? ` (${escapeHtml(app.id)})` : ""}</span>`).join("")}
      ${!view.targetApps.length && view.appName ? `<span>目标应用 · ${escapeHtml(view.appName)}</span>` : ""}
      ${view.constraints.map(item => `<span>限制 · ${escapeHtml(item)}</span>`).join("")}
      ${view.completionConditions.map(item => `<span>完成 · ${escapeHtml(item)}</span>`).join("")}
      <span>确认门 · ${escapeHtml(view.risk.confirmationGate.state)} · required=${view.risk.confirmationGate.required ? "true" : "false"} · external_allowed=${view.risk.confirmationGate.externalStateActionAllowed ? "true" : "false"}</span>
      <span>影响等级 · ${escapeHtml(view.risk.currentExternalImpact)}</span>
      <span>本地策略 · ${view.controllerGate.allowed ? "允许" : "阻止"} · ${escapeHtml(view.controllerGate.reason || "尚未判定")}</span>
      ${gateScopeText ? `<span>${escapeHtml(gateScopeText)}</span>` : ""}
      ${view.risk.actions.map(item => `<span>风险 ${escapeHtml(item.id)} · ${escapeHtml(item.description)}</span>`).join("")}
    </div>`;

  const steps = view.subgoals.map(item => planStepHtml({
    number: item.index,
    label: item.label,
    detail: item.reason,
    stateName: planState(item, view),
  })).join("");
  const dynamicTail = view.isTerminal ? "" : planStepHtml({
    number: "…",
    label: "后续步骤等待新画面",
    detail: "每个动作重新观察后，DeepSeek 可更新剩余任务图",
    stateName: "waiting",
  });
  planElement.innerHTML = steps + dynamicTail;
  badge.className = `pill ${view.status === "succeeded" || view.status === "completed" ? "success" : (view.isTerminal ? "danger" : "active")}`;
  badge.textContent = statusNames[view.status] || view.status;
}

function planStepHtml({ number, label, detail, stateName }) {
  return `<div class="plan-step ${stateName}">
    <span class="step-index">${escapeHtml(number)}</span>
    <div><strong>${escapeHtml(label)}</strong><p>${escapeHtml(detail)}</p></div>
    <span class="step-state">${stateName === "done" ? "完成" : (stateName === "current" ? "当前" : (stateName === "blocked" ? "停止" : "动态"))}</span>
  </div>`;
}

function renderTrace() {
  const view = sessionView();
  const trace = document.querySelector("#traceList");
  const count = document.querySelector("#traceCount");
  count.textContent = `${view?.history.length || 0} 条`;
  if (!view) {
    trace.className = "trace-list empty-state";
    trace.textContent = "还没有执行记录。每次观察、确认、动作和验证都会显示在这里。";
    return;
  }
  const rows = view.history.slice().reverse().map(item => `
    <article class="trace-item">
      <div class="trace-marker"></div>
      <div>
        <div class="trace-title"><strong>步骤 ${escapeHtml(item.stepNumber)} · ${escapeHtml(actionLabel(item.action))}</strong><time>${item.physicalActions} 个物理动作</time></div>
        <p>${escapeHtml(item.reason)}</p>
        <small>目标：${escapeHtml(item.action.semanticTarget)} · 验证：${escapeHtml(item.completionEvidence.join("；") || "已保存动作后画面")}</small>
        <small>证据：${escapeHtml(item.evidence.join("；") || "—")}</small>
      </div>
    </article>`).join("");
  trace.className = "trace-list";
  trace.innerHTML = rows || `<article class="trace-item observation-only"><div class="trace-marker"></div><div><div class="trace-title"><strong>目标已理解，初始画面已观察</strong><time>0 个物理动作</time></div><p>正在等待当前一步确认。</p></div></article>`;
}

function renderScene() {
  const view = sessionView();
  const overlay = document.querySelector("#previewOverlay");
  const meta = document.querySelector("#sceneMeta");
  if (!view) {
    overlay.textContent = state.busy ? currentVisionStageLabel() : "等待观察";
    overlay.classList.toggle("show", state.busy);
    meta.innerHTML = `<span><b>页面</b><em>尚未识别</em></span><span><b>稳定性</b><em>—</em></span><span><b>置信度</b><em>—</em></span>`;
    return;
  }
  overlay.textContent = state.busy ? currentVisionStageLabel() : "";
  overlay.classList.toggle("show", state.busy);
  meta.innerHTML = `
    <span><b>页面</b><em>${escapeHtml(view.scene.summary)}</em></span>
    <span><b>稳定性</b><em>${view.scene.stable ? "稳定" : "不稳定"}</em></span>
    <span><b>置信度</b><em>${escapeHtml(confidenceLabel(view.scene.confidence))}</em></span>
    <span><b>累计动作</b><em>${escapeHtml(view.physicalActions)}</em></span>
    <span><b>证据</b><em>${escapeHtml(view.evidence.join("；") || "—")}</em></span>`;
}

function renderAction() {
  const view = sessionView();
  const content = document.querySelector("#actionContent");
  const controls = document.querySelector("#actionControls");
  const badge = document.querySelector("#sessionBadge");
  document.querySelector("#pauseNotice").hidden = !state.paused;
  if (!view) {
    content.className = "empty-state";
    content.textContent = "Agent 将结合目标和当前画面，只提出一个下一动作。";
    controls.innerHTML = "";
    badge.className = "pill neutral";
    badge.textContent = "未开始";
    return;
  }

  const action = view.visualAction;
  const risk = view.risk.blocksAutomatic;
  const riskSummary = view.risk.currentActions.map(item => `${item.id}：${item.description}`).join("；");
  const actionMetadata = ["qwen-visual-decision-v2", "qwen-visual-decision-v3"].includes(action.protocol)
    ? `<div class="action-metadata">
         <span>${escapeHtml(action.protocolVersion || "qwen-v2")}</span>
         <span>status ${escapeHtml(action.status)}</span>
         <span>task ${escapeHtml(action.taskId || "—")}</span>
         <span>revision ${escapeHtml(action.revision ?? "—")}</span>
         <span>observation ${escapeHtml(action.observationId || "—")}</span>
         <span>fingerprint ${escapeHtml(action.fingerprint || "—")}</span>
       </div>`
    : action.actionType
      ? `<div class="compatibility-note">兼容回退 · ${escapeHtml(action.protocol)}</div>`
      : `<div class="compatibility-note">Qwen 唯一动作尚未产生</div>`;
  content.className = "action-content";
  content.innerHTML = view.isTerminal
    ? `<h3>${escapeHtml(statusNames[view.status] || view.status)}</h3><p>${escapeHtml(view.failedReason || action.reason || "会话已经结束。")}</p>`
    : (action.status === "finished"
      ? `<div class="next-action-title"><span>${escapeHtml(decisionStatusNames.finished)}</span><b class="safe-tag">不可执行</b></div>
         <h3>${escapeHtml(view.currentSubgoal.label)}</h3>
         <p>${escapeHtml(action.reason)}</p>${actionMetadata}`
      : action.status === "blocked"
        ? `<div class="next-action-title"><span>${escapeHtml(decisionStatusNames.blocked)}</span><b class="risk-tag">不可执行</b></div>
           <h3>${escapeHtml(view.currentSubgoal.label)}</h3>
           <p>${escapeHtml(action.reason)}</p>${actionMetadata}`
        : view.status === "paused_after_action"
      ? `<h3>上一步已完成并重新观察</h3><p>网页将依据新画面决定是否发起下一次单动作请求。</p>`
      : `<div class="next-action-title"><span>${escapeHtml(action.actionType ? actionLabel(action) : decisionStatusNames[action.status] || "等待唯一动作")}</span>${risk ? '<b class="risk-tag">需要当前风险确认</b>' : '<b class="safe-tag">受限单步</b>'}</div>
         <h3>${escapeHtml(view.currentSubgoal.label)}</h3>
         <div class="action-target">语义目标 · ${escapeHtml(action.semanticTarget)}${action.elementId ? ` · element_id ${escapeHtml(action.elementId)}` : ""}</div>
         <div class="action-facts">
           <span><b>目标区域</b>${escapeHtml(Protocol.displayValue(action.targetRegion))}</span>
           <span><b>预期变化</b>${escapeHtml(Protocol.displayValue(action.expectedChange))}</span>
           <span><b>动作置信度</b>${escapeHtml(confidenceLabel(action.confidence))}</span>
           <span><b>本地策略</b>${view.controllerGate.allowed ? "允许" : "阻止"} · ${escapeHtml(view.controllerGate.reason || "尚未判定")}</span>
         </div>
         <p>${escapeHtml(action.reason || riskSummary || "等待 Qwen 生成唯一下一视觉动作。")}</p>
         ${riskSummary ? `<small>当前风险：${escapeHtml(riskSummary)}</small>` : ""}
         ${actionMetadata}
         <small>确认绑定当前任务、revision、子目标、risk_ids、observation_id 和 fingerprint；动作后必须重新观察。</small>`);

  const disabled = state.busy || state.paused ? "disabled" : "";
  if (view.isTerminal || ["finished", "blocked"].includes(action.status)) {
    controls.innerHTML = "";
  } else if (view.risk.requiresConfirmation || view.status === "awaiting_confirmation") {
    controls.innerHTML = `
      <button id="reviewAction" class="${risk ? "risk-button" : "primary-button"}" ${disabled}>${risk ? "查看风险并确认" : "确认当前一步"}</button>
      <button id="nextSupervisedAgent" class="secondary-button" ${disabled}>放弃旧确认并重新观察</button>
      <button id="cancelSupervisedAgent" class="text-button" ${state.busy ? "disabled" : ""}>取消会话</button>`;
  } else {
    controls.innerHTML = `
      <button id="nextSupervisedAgent" class="primary-button" ${disabled}>观察并生成下一步</button>
      <button id="cancelSupervisedAgent" class="text-button" ${state.busy ? "disabled" : ""}>取消会话</button>`;
  }
  badge.className = `pill ${view.isTerminal ? (view.status === "succeeded" || view.status === "completed" ? "success" : "danger") : (risk ? "risk" : "active")}`;
  badge.textContent = view.isTerminal
    ? (statusNames[view.status] || view.status)
    : (risk ? "等待风险确认" : (statusNames[view.status] || view.status));
  bindActionEvents();
}

function bindActionEvents() {
  document.querySelector("#reviewAction")?.addEventListener("click", openRiskDialog);
  document.querySelector("#nextSupervisedAgent")?.addEventListener("click", nextSupervisedAgent);
  document.querySelector("#cancelSupervisedAgent")?.addEventListener("click", cancelSupervisedAgent);
}

function render() {
  const view = sessionView();
  const deviceSelect = document.querySelector("#deviceId");
  const registeredDevices = Array.isArray(state.device?.devices) ? state.device.devices : [];
  if (!view && registeredDevices.length) {
    const enabledIds = registeredDevices.map(item => String(item.device_id || "")).filter(Boolean);
    deviceSelect.replaceChildren(...enabledIds.map(deviceId => {
      const option = document.createElement("option");
      option.value = deviceId;
      option.textContent = deviceId;
      return option;
    }));
    if (!enabledIds.includes(state.deviceId)) {
      state.deviceId = String(state.device.default_device_id || enabledIds[0]);
      localStorage.setItem("visual-agent-device-id", state.deviceId);
    }
  }
  deviceSelect.value = state.deviceId;
  deviceSelect.disabled = !!view && !view.isTerminal;
  document.querySelector("#startSupervisedAgent").disabled = state.busy || state.paused;
  document.querySelector("#agentText").disabled = state.busy;
  document.querySelector("#pauseButton").textContent = state.paused ? "▶ 继续推进" : "Ⅱ 暂停推进";
  document.querySelector("#pauseButton").classList.toggle("active", state.paused);
  renderStatus();
  renderGoalAndPlan();
  renderTrace();
  renderScene();
  renderAction();
}

async function refreshDevice() {
  state.device = await api("/api/device");
  renderStatus();
}

async function pollVisionStage() {
  try {
    state.device = await api("/api/device");
    const observer = observerStatus();
    if (observer.current_stage && observer.current_stage !== "idle") {
      state.visionStage = observer.current_stage_label || observer.current_stage;
    }
    renderStatus();
    renderScene();
  } catch (_error) {
    // The foreground request owns user-visible errors.
  }
}

async function withVisionProgress(initialLabel, operation) {
  state.busy = true;
  state.visionStage = initialLabel;
  render();
  const timer = setInterval(pollVisionStage, 750);
  try {
    return await operation();
  } finally {
    clearInterval(timer);
    state.busy = false;
    state.visionStage = "";
    await refreshDevice().catch(() => {});
    render();
  }
}

async function startSupervisedAgent() {
  const text = document.querySelector("#agentText").value.trim();
  const current = sessionView();
  if (!text) return toast("请先输入希望手机完成的目标。", true);
  if (current && !current.isTerminal) return toast("已有进行中的会话，请继续或停止后再创建新目标。", true);
  state.sessionDeviceId = state.deviceId;
  state.pendingConfirmationGrant = null;
  try {
    const payload = Protocol.buildRequestPayload(state.sessionDeviceId, { text });
    const response = await withVisionProgress("理解目标并观察当前画面", () =>
      api("/api/agent/generic-supervised/start", {
        method: "POST",
        body: JSON.stringify(payload),
      })
    );
    state.supervisedSession = response.session;
    await finalizeStopIfRequested();
    toast("动态计划已生成，尚未执行物理动作。");
    render();
  } catch (error) {
    toast(error.message, true);
  }
}

function openRiskDialog() {
  const view = sessionView();
  if (!view || state.paused || state.busy || !view.risk.requiresConfirmation) return;
  const risk = view.risk.blocksAutomatic;
  const riskPhase = view.risk.confirmationGate.phase === "risk";
  try {
    state.pendingConfirmationGrant = Protocol.createConfirmationGrant(view, lockedSessionDeviceId());
  } catch (error) {
    state.pendingConfirmationGrant = null;
    return toast(error.message, true);
  }
  document.querySelector("#riskTitle").textContent = riskPhase
    ? "确认当前子目标的风险范围"
    : (risk ? "确认外部状态动作" : "确认当前单步动作");
  const level = document.querySelector("#riskLevel");
  level.className = `risk-level ${risk ? "high" : "guarded"}`;
  level.textContent = risk ? "高关注 · 可能改变账号或对外产生影响" : "受控动作 · 仅授权当前一步";
  document.querySelector("#riskGoal").textContent = view.objective;
  document.querySelector("#riskAction").textContent = view.visualAction.actionType
    ? `${actionLabel(view.visualAction)} · ${view.visualAction.semanticTarget}`
    : `${view.currentSubgoal.label} · 等待 Qwen 唯一动作`;
  document.querySelector("#riskReason").textContent = view.risk.currentActions.map(item => `${item.id} [${item.level}]：${item.description}；${item.externalEffect}`).join("\n") || view.visualAction.reason;
  document.querySelector("#riskExpected").textContent = Protocol.displayValue(view.visualAction.expectedChange);
  document.querySelector("#riskDevice").textContent = lockedSessionDeviceId();
  document.querySelector("#riskWarning").textContent = riskPhase
    ? `本次仅允许 Qwen 针对 task=${view.taskId}、revision=${view.revision ?? "—"}、subgoal=${view.currentSubgoal.id}、risk_ids=${view.risk.riskIds.join(",") || "—"} 观察并提出一个动作；此确认本身不会触发机械臂。具体动作产生后仍需再次确认。`
    : risk
    ? `确认只授权 task=${view.taskId}、revision=${view.revision ?? "—"}、subgoal=${view.currentSubgoal.id}、risk_ids=${view.risk.riskIds.join(",") || "—"}、observation_id=${view.visualAction.observationId || "—"}、fingerprint=${view.visualAction.fingerprint || "—"} 的当前一步；任何字段变化都必须重新确认。`
    : `确认只授权 observation_id=${view.visualAction.observationId || "—"}、fingerprint=${view.visualAction.fingerprint || "—"} 对应的一个动作。执行后必须重新观察。`;
  document.querySelector("#confirmRiskAction").className = risk ? "danger-confirm" : "primary-button";
  const safeLoopKinds = new Set(["tap_semantic", "dismiss_overlay", "swipe", "back", "wait_for_change"]);
  document.querySelector("#confirmSafeLoop").hidden = !(
    !riskPhase
    && ["read_only", "navigation_only"].includes(view.risk.currentExternalImpact)
    && safeLoopKinds.has(view.visualAction.actionType)
    && view.controllerGate.allowed
  );
  document.querySelector("#riskDialog").showModal();
}

async function advanceSupervisedAgent(grant) {
  const view = sessionView();
  if (!view || state.paused || state.busy) return;
  try {
    const payload = Protocol.consumeConfirmationGrant(grant, view, lockedSessionDeviceId());
    const riskPhase = grant?.phase === "risk";
    const response = await withVisionProgress(
      riskPhase ? "确认风险范围并生成唯一动作" : "执行当前一步并重新观察",
      () => api(`/api/agent/generic-supervised/${view.sessionId}/${riskPhase ? "approve-risk" : "confirm"}`, {
        method: "POST",
        body: JSON.stringify(payload),
      })
    );
    state.supervisedSession = response.session;
    await finalizeStopIfRequested();
    toast(riskPhase
      ? "风险范围已确认，机械臂尚未动作；请核对并再次确认具体动作。"
      : "当前一步已处理，并已重新观察画面。");
    render();
  } catch (error) {
    toast(error.message, true);
  }
}

async function nextSupervisedAgent() {
  const view = sessionView();
  if (!view || state.paused || state.busy) return;
  state.pendingConfirmationGrant = null;
  try {
    const payload = Protocol.buildRequestPayload(lockedSessionDeviceId());
    const response = await withVisionProgress("重新观察并动态规划下一步", () =>
      api(`/api/agent/generic-supervised/${view.sessionId}/next`, {
        method: "POST",
        body: JSON.stringify(payload),
      })
    );
    state.supervisedSession = response.session;
    await finalizeStopIfRequested();
    render();
  } catch (error) {
    toast(error.message, true);
  }
}

async function autoSupervisedAgent(grant) {
  const view = sessionView();
  if (!view || state.paused || state.busy || grant?.phase === "risk") return;
  try {
    const confirmation = Protocol.consumeConfirmationGrant(grant, view, lockedSessionDeviceId());
    const payload = Protocol.buildAutoRequestPayload(
      lockedSessionDeviceId(),
      confirmation,
      { maxPhysicalActions: 3, maxIterations: 8 },
    );
    const response = await withVisionProgress("连续执行低风险导航并逐步验证", () =>
      api(`/api/agent/generic-supervised/${view.sessionId}/auto`, {
        method: "POST",
        body: JSON.stringify(payload),
      })
    );
    state.supervisedSession = response.session;
    await finalizeStopIfRequested();
    toast(response.execution?.pause_reason || "安全连续推进已暂停。");
    render();
  } catch (error) {
    toast(error.message, true);
  }
}

async function cancelSupervisedAgent({ quiet = false } = {}) {
  const view = sessionView();
  if (!view || view.isTerminal) return;
  state.pendingConfirmationGrant = null;
  try {
    const payload = Protocol.buildRequestPayload(lockedSessionDeviceId());
    const response = await api(`/api/agent/generic-supervised/${view.sessionId}/cancel`, {
      method: "POST",
      body: JSON.stringify(payload),
    });
    state.supervisedSession = response.session;
    state.stopRequested = false;
    if (!quiet) toast("会话已取消，不会再发起动作。");
    render();
  } catch (error) {
    if (!quiet) toast(error.message, true);
  }
}

async function finalizeStopIfRequested() {
  if (!state.stopRequested) return;
  await cancelSupervisedAgent({ quiet: true });
  state.stopRequested = false;
}

function togglePause() {
  state.paused = !state.paused;
  if (state.paused) {
    state.pendingConfirmationGrant = null;
    const view = sessionView();
    if (view && !view.isTerminal) {
      const payload = Protocol.buildRequestPayload(lockedSessionDeviceId());
      api(`/api/agent/generic-supervised/${view.sessionId}/pause`, {
        method: "POST",
        body: JSON.stringify(payload),
      }).then(response => {
        if (response?.session) state.supervisedSession = response.session;
        render();
      }).catch(error => toast(`服务端暂停确认失效失败：${error.message}`, true));
    }
  }
  toast(state.paused
    ? "已暂停推进。当前请求结束后，网页不会发起下一次请求。"
    : "已恢复，可由你确认后继续生成或执行下一步。");
  render();
}

async function stopTasks() {
  const requestWasRunning = state.busy;
  state.paused = true;
  state.stopRequested = true;
  state.pendingConfirmationGrant = null;
  render();
  try {
    const payload = Protocol.buildRequestPayload(lockedSessionDeviceId());
    const result = await api("/api/stop", { method: "POST", body: JSON.stringify(payload) });
    if (!requestWasRunning) await finalizeStopIfRequested();
    toast(result.note || "停止请求已发送。");
    await refreshDevice();
    render();
  } catch (error) {
    toast(error.message, true);
  }
}

async function restoreActiveSession() {
  const active = state.device.generic_supervised_execution?.active_sessions?.[0];
  if (!active?.session_id) return;
  try {
    const response = await api(`/api/agent/generic-supervised/${active.session_id}`);
    state.supervisedSession = response.session;
    const restored = Protocol.adaptSession(response.session, { fallbackDeviceId: state.deviceId });
    state.sessionDeviceId = restored.deviceId || state.deviceId;
    state.pendingConfirmationGrant = null;
  } catch (_error) {
    state.supervisedSession = null;
  }
}

async function init() {
  try {
    const session = await api("/api/session");
    state.token = session.token;
    state.mock = session.mock;
    const mode = document.querySelector("#modeBadge");
    mode.textContent = session.mock ? "模拟模式 · 无实机动作" : "实机接口已连接";
    mode.classList.toggle("live-mode", !session.mock);
    await refreshDevice();
    await restoreActiveSession();
    render();
    setInterval(() => {
      const preview = document.querySelector("#phonePreview");
      if (preview) preview.src = `/api/preview.jpg?t=${Date.now()}`;
    }, 700);
    setInterval(() => refreshDevice().catch(() => {}), 3000);
  } catch (error) {
    toast(`连接本地服务失败：${error.message}`, true);
  }
}

document.querySelector("#startSupervisedAgent").addEventListener("click", startSupervisedAgent);
document.querySelector("#pauseButton").addEventListener("click", togglePause);
document.querySelector("#stopButton").addEventListener("click", stopTasks);
document.querySelector("#deviceId").addEventListener("change", event => {
  state.deviceId = event.target.value;
  localStorage.setItem("visual-agent-device-id", state.deviceId);
  render();
});
document.querySelector("#agentText").addEventListener("keydown", event => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") startSupervisedAgent();
});
document.querySelector("#riskDialog").addEventListener("close", event => {
  const grant = state.pendingConfirmationGrant;
  state.pendingConfirmationGrant = null;
  if (event.target.returnValue === "default" && grant) advanceSupervisedAgent(grant);
  if (event.target.returnValue === "auto" && grant) autoSupervisedAgent(grant);
});

init();
