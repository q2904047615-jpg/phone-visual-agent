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
  capabilityTrial: null,
  capabilityDeviceId: "",
  pendingPromotionGrant: null,
  capabilityEvidenceUrls: [],
};

const promotableCapabilityActions = [
  "tap_semantic",
  "dismiss_overlay",
  "swipe",
  "back",
  "home",
  "reveal_system_navigation",
  "input_verified_text",
  "long_press",
  "drag",
];

const statusNames = {
  idle: "等待目标",
  ready: "准备执行",
  awaiting_confirmation: "等待当前动作确认",
  awaiting_effect_confirmation: "等待效果确认",
  needs_effect_verification: "等待只读效果结果复核",
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
  home: "返回系统桌面",
  reveal_system_navigation: "唤出系统导航栏",
  long_press: "长按目标控件",
  drag: "拖动目标控件",
  input_verified_text: "输入并核对文字",
  wait_for_change: "等待页面变化",
  record_verified_result: "记录已验证结果",
  finish: "完成本次任务",
};

async function api(path, options = {}) {
  const headers = { ...(options.headers || {}) };
  if (state.token) headers["X-Control-Token"] = state.token;
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

function targetRegionLabel(value) {
  const region = value && typeof value === "object" && !Array.isArray(value) ? value : {};
  const parts = [];
  if (region.kind) parts.push(`kind=${region.kind}`);
  if (region.element_id) parts.push(`element_id=${region.element_id}`);
  if (region.destination_element_id) parts.push(`destination_element_id=${region.destination_element_id}`);
  return parts.join(" · ") || "结构化区域已绑定";
}

function publicEvidenceSummary(value) {
  const count = Array.isArray(value) ? value.filter(Boolean).length : 0;
  return count ? `已保存 ${count} 项本地证据（路径不在控制台显示）` : "—";
}

function controllerGateLabel(gate) {
  const value = gate || {};
  return value.reason || value.policyVersion || value.canonicalClass
    ? `${value.allowed ? "允许" : "阻止"} · ${value.reason || value.canonicalClass || value.policyVersion}`
    : "旧记录未提供";
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
  const effectPhase = view?.status === "awaiting_effect_confirmation";
  const actionPhase = view?.status === "awaiting_confirmation";
  const highAttention = Boolean(view?.effectPolicy.requiresConfirmation);
  document.querySelector("#safetyText").textContent = effectPhase
    ? "等待效果确认"
    : actionPhase
      ? "等待当前动作确认"
      : highAttention
        ? "受限效果已暂停"
        : "一次一动作";
  setDot("#safetyDot", effectPhase || actionPhase || highAttention ? "warn" : "online");
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

  goalElement.className = "goal-summary";
  goalElement.innerHTML = `
    <div class="goal-title-row">
      <div><span>目标 · ${escapeHtml(view.taskId || "待分配任务 ID")} · revision ${escapeHtml(view.revision ?? "—")}</span><strong>${escapeHtml(view.objective)}</strong></div>
      <code>${escapeHtml(view.deviceId || lockedSessionDeviceId())}</code>
    </div>
    <div class="goal-chips">
      <span>会话 · ${escapeHtml(view.sessionId || "—")}</span>
      ${view.protocolVersion ? `<span>协议 · ${escapeHtml(view.protocolVersion)}</span>` : ""}
      ${view.compatibilityFallback ? `<span>旧版兼容数据 · 不作为 v3 确认依据</span>` : ""}
      ${view.targetApps.map(app => `<span>目标应用 · ${escapeHtml(app.name)}${app.id ? ` (${escapeHtml(app.id)})` : ""}</span>`).join("")}
      ${!view.targetApps.length && view.appName ? `<span>目标应用 · ${escapeHtml(view.appName)}</span>` : ""}
      ${view.constraints.map(item => `<span>限制 · ${escapeHtml(item)}</span>`).join("")}
      ${view.completionConditions.map(item => `<span>完成 · ${escapeHtml(item)}</span>`).join("")}
      <span>确认门 · ${escapeHtml(view.effectPolicy.confirmationGate.state)} · required=${view.effectPolicy.confirmationGate.required ? "true" : "false"} · effect_allowed=${view.effectPolicy.confirmationGate.effectActionAllowed ? "true" : "false"}</span>
      <span>执行类型 · ${escapeHtml(view.effectPolicy.currentExecutionClass)}</span>
      <span>本地策略 · ${escapeHtml(controllerGateLabel(view.controllerGate))}</span>
      <span>确认作用域 · ${escapeHtml(view.scopeState.state)} · ${escapeHtml(view.scopeState.reason || "—")}</span>
      ${view.effectPolicy.actions.map(item => `<span>效果 ${escapeHtml(item.id)} · ${escapeHtml(item.kind)}</span>`).join("")}
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
  count.textContent = `${view?.executionTrace.length || 0} 个轮次`;
  if (!view) {
    trace.className = "trace-list empty-state";
    trace.textContent = "还没有执行记录。每次观察、确认、动作和验证都会显示在这里。";
    return;
  }
  const transitionNames = {
    advance: "推进",
    replan: "重规划",
    advance_or_replan: "推进 / 重规划",
    blocked: "阻止",
    stopped: "停止",
    awaiting_confirmation: "等待动作确认",
    awaiting_effect_confirmation: "等待效果确认",
    needs_effect_verification: "等待只读效果结果复核",
    observing: "观察中",
    unknown: "旧记录未提供",
  };
  const outcomeNames = {
    matched: "符合预期",
    mismatched: "不符合预期",
    uncertain: "结果不确定",
    not_executed: "尚未执行",
    awaiting_next_action: "等待下一动作",
    terminal: "终态检查点",
    unknown: "旧记录未提供",
  };
  const scopeNames = {
    active: "当前有效",
    consumed: "已消费",
    stale: "已失效",
    invalidated: "已停止并失效",
    missing: "缺少作用域",
    none: "无需确认",
    unknown: "旧记录未提供",
  };
  const rows = view.executionTrace.slice().reverse().map(item => {
    const gate = item.controllerGate || {};
    const verification = item.verification || {};
    const transition = item.transition || {};
    const scopeState = item.scopeState || { state: "none", reason: "" };
    const gateText = gate.reason || gate.policyVersion || gate.canonicalClass
      ? `${gate.allowed ? "允许" : "阻止"}${gate.canonicalClass ? ` · ${gate.canonicalClass}` : ""}`
      : "旧记录未提供";
    const observationText = item.observation?.id || item.observation?.fingerprint
      ? `${item.observation.id || "—"} / ${item.observation.fingerprint || "—"}`
      : "旧记录未提供";
    const afterObservationText = verification.afterObservationId || verification.afterFingerprint
      ? `${verification.afterObservationId || "—"} / ${verification.afterFingerprint || "—"}`
      : "—";
    const transitionLabel = transitionNames[transition.kind] || transition.kind || "旧记录未提供";
    return `
    <article class="trace-item ${item.phase === "current" ? "current-trace" : ""}" data-trace-phase="${escapeHtml(item.phase)}">
      <div class="trace-marker"></div>
      <div>
        <div class="trace-title"><strong>步骤 ${escapeHtml(item.stepNumber)} · revision ${escapeHtml(item.graphRevision ?? "—")}</strong><time>physical_actions ${escapeHtml(item.physicalActions)}</time></div>
        <p><b>当前子目标</b> ${escapeHtml(item.subgoal || "—")}${item.subgoalId ? ` · ${escapeHtml(item.subgoalId)}` : ""}</p>
        <div class="trace-grid">
          <span><b>动作前观察</b>${escapeHtml(observationText)}</span>
          <span><b>Qwen 唯一动作</b>${escapeHtml(item.action?.status || "unknown")} · ${escapeHtml(actionLabel(item.action))} · ${escapeHtml(item.action?.semanticTarget || "—")}</span>
          <span><b>Controller gate</b>${escapeHtml(gateText)}${gate.reason ? ` · ${escapeHtml(gate.reason)}` : ""}</span>
          <span><b>动作后验证</b>${escapeHtml(outcomeNames[verification.outcome] || verification.outcome || "旧记录未提供")} · ${escapeHtml(afterObservationText)}</span>
          <span><b>任务图去向</b>${escapeHtml(transitionLabel)}${transition.trigger ? ` · ${escapeHtml(transition.trigger)}` : ""}${transition.toRevision !== null && transition.toRevision !== undefined ? ` · r${escapeHtml(transition.fromRevision ?? "—")}→r${escapeHtml(transition.toRevision)}` : ""}</span>
          <span class="scope-state ${escapeHtml(scopeState.state)}"><b>确认作用域</b>${escapeHtml(scopeNames[scopeState.state] || scopeState.state)}${scopeState.reason ? ` · ${escapeHtml(scopeState.reason)}` : ""}</span>
        </div>
        ${(verification.errors || []).length ? `<small>验证/阻止原因：${escapeHtml(verification.errors.join("；"))}</small>` : ""}
        ${transition.reason ? `<small>推进/重规划原因：${escapeHtml(transition.reason)}</small>` : ""}
        ${(verification.evidence || item.evidence || []).length ? `<small>证据：${escapeHtml(publicEvidenceSummary(verification.evidence || item.evidence))}</small>` : ""}
      </div>
    </article>`;
  }).join("");
  trace.className = "trace-list";
  trace.innerHTML = rows || `<article class="trace-item observation-only"><div class="trace-marker"></div><div><div class="trace-title"><strong>目标已理解，初始画面已观察</strong><time>physical_actions 0</time></div><p>正在等待当前一步确认。</p></div></article>`;
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
    <span><b>任务图 / 子目标</b><em>r${escapeHtml(view.taskGraph.revision ?? "—")} · ${escapeHtml(view.currentSubgoal.id || "—")}</em></span>
    <span><b>observation_id</b><em>${escapeHtml(view.executionTrace.at(-1)?.observation?.id || "—")}</em></span>
    <span><b>fingerprint</b><em>${escapeHtml(view.executionTrace.at(-1)?.observation?.fingerprint || "—")}</em></span>
    <span><b>累计动作</b><em>${escapeHtml(view.physicalActions)}</em></span>
    <span><b>确认作用域</b><em>${escapeHtml(view.scopeState.state)} · ${escapeHtml(view.scopeState.reason || "—")}</em></span>
    <span><b>停止状态</b><em>${escapeHtml(view.stopState.stopped ? `${view.stopState.status} · ${view.stopState.reason}` : "active")}</em></span>
    <span><b>证据</b><em>${escapeHtml(publicEvidenceSummary(view.evidence))}</em></span>`;
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
  const effectPhase = view.status === "awaiting_effect_confirmation"
    || view.effectPolicy.confirmationGate.phase === "effect";
  const staleScope = ["awaiting_confirmation", "awaiting_effect_confirmation"].includes(view.status)
    && view.scopeState.state !== "active";
  const highAttention = view.effectPolicy.requiresConfirmation;
  const riskSummary = view.effectPolicy.currentActions.map(item => `${item.id}：${item.kind}`).join("；");
  const actionMetadata = String(action.protocol || "").startsWith("qwen-visual-decision-v")
    ? `<div class="action-metadata">
         <span>${escapeHtml(action.protocolVersion || "qwen-v2")}</span>
         <span>status ${escapeHtml(action.status)}</span>
         <span>session ${escapeHtml(view.sessionId || "—")}</span>
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
      : `<div class="next-action-title"><span>${escapeHtml(action.actionType ? actionLabel(action) : decisionStatusNames[action.status] || "等待唯一动作")}</span>${staleScope ? '<b class="risk-tag">旧确认已失效</b>' : effectPhase ? '<b class="risk-tag">需要效果确认</b>' : view.status === "awaiting_confirmation" ? `<b class="${highAttention ? "risk-tag" : "safe-tag"}">需要当前动作确认</b>` : '<b class="safe-tag">受限单步</b>'}</div>
         <h3>${escapeHtml(view.currentSubgoal.label)}</h3>
         <div class="action-target">语义目标 · ${escapeHtml(action.semanticTarget)}${action.elementId ? ` · element_id ${escapeHtml(action.elementId)}` : ""}</div>
         <div class="action-facts">
           <span><b>目标区域</b>${escapeHtml(targetRegionLabel(action.targetRegion))}</span>
           <span><b>预期变化</b>${escapeHtml(Protocol.displayValue(action.expectedChange))}</span>
           <span><b>动作置信度</b>${escapeHtml(confidenceLabel(action.confidence))}</span>
           <span><b>本地策略</b>${escapeHtml(controllerGateLabel(view.controllerGate))}</span>
         </div>
         <p>${escapeHtml(action.reason || riskSummary || "等待 Qwen 生成唯一下一视觉动作。")}</p>
         ${riskSummary ? `<small>当前效果：${escapeHtml(riskSummary)}</small>` : ""}
         ${actionMetadata}
          <small>${staleScope ? `当前作用域不可执行：${escapeHtml(view.scopeState.reason || "任务或画面已变化")}；必须重新观察。` : "后端 scope 与当前权威任务、观察和动作字段一致；本次只允许一个动作，之后必须重新观察。"}</small>`);

  const disabled = state.busy || state.paused ? "disabled" : "";
  if (view.isTerminal || ["finished", "blocked"].includes(action.status)) {
    controls.innerHTML = "";
  } else if (staleScope) {
    controls.innerHTML = `
      <button id="nextSupervisedAgent" class="primary-button" ${disabled}>旧确认已失效 · 重新观察</button>
      <button id="cancelSupervisedAgent" class="text-button" ${state.busy ? "disabled" : ""}>取消会话</button>`;
  } else if (view.effectPolicy.requiresConfirmation || view.status === "awaiting_confirmation") {
    controls.innerHTML = `
      <button id="reviewAction" class="${effectPhase || highAttention ? "risk-button" : "primary-button"}" ${disabled}>${effectPhase ? "查看效果并确认" : "确认当前动作"}</button>
      <button id="nextSupervisedAgent" class="secondary-button" ${disabled}>放弃旧确认并重新观察</button>
      <button id="cancelSupervisedAgent" class="text-button" ${state.busy ? "disabled" : ""}>取消会话</button>`;
  } else {
    controls.innerHTML = `
      <button id="nextSupervisedAgent" class="primary-button" ${disabled}>观察并生成下一步</button>
      <button id="cancelSupervisedAgent" class="text-button" ${state.busy ? "disabled" : ""}>取消会话</button>`;
  }
  badge.className = `pill ${view.isTerminal ? (view.status === "succeeded" || view.status === "completed" ? "success" : "danger") : (effectPhase || highAttention ? "risk" : "active")}`;
  badge.textContent = view.isTerminal
    ? (statusNames[view.status] || view.status)
    : (statusNames[view.status] || view.status);
  bindActionEvents();
}

function bindActionEvents() {
  document.querySelector("#reviewAction")?.addEventListener("click", openRiskDialog);
  document.querySelector("#nextSupervisedAgent")?.addEventListener("click", nextSupervisedAgent);
  document.querySelector("#cancelSupervisedAgent")?.addEventListener("click", cancelSupervisedAgent);
}

function currentDeviceDescriptor() {
  const devices = Array.isArray(state.device?.devices) ? state.device.devices : [];
  return devices.find(item => String(item.device_id || "") === state.deviceId) || null;
}

function unverifiedCapabilityActions() {
  const verified = new Set(currentDeviceDescriptor()?.verified_actions || []);
  return promotableCapabilityActions.filter(action => !verified.has(action));
}

function renderCapabilityAcceptance() {
  const view = capabilityView();
  const status = document.querySelector("#capabilityStatus");
  const badge = document.querySelector("#capabilityBadge");
  const controls = document.querySelector("#capabilityControls");
  const select = document.querySelector("#capabilityAction");
  const goal = document.querySelector("#capabilityGoal");
  const start = document.querySelector("#startCapabilityTrial");
  const available = unverifiedCapabilityActions();

  if (!view) {
    const previous = select.value;
    select.replaceChildren(...available.map(action => {
      const option = document.createElement("option");
      option.value = action;
      option.textContent = semanticActionNames[action] || action;
      return option;
    }));
    if (available.includes(previous)) select.value = previous;
    status.className = "capability-status empty-state";
    status.textContent = available.length
      ? "选择一个尚未验证的通用动作。生成计划只调用规划和视觉观察，物理动作数为 0。"
      : "当前设备没有待验收的通用动作。";
    badge.className = "pill neutral";
    badge.textContent = "未开始";
    controls.innerHTML = "";
    document.querySelector("#capabilityEvidence").hidden = true;
  } else {
    select.replaceChildren(Object.assign(document.createElement("option"), {
      value: view.action,
      textContent: semanticActionNames[view.action] || view.action,
    }));
    select.value = view.action;
    const report = view.report;
    const canPromote = Boolean(view.passed && view.promotionScope && !view.readOnlyRecovered);
    const reportState = report?.status || "尚未生成报告";
    status.className = "capability-status active";
    status.innerHTML = `
      <strong>${escapeHtml(semanticActionNames[view.action] || view.action)} · ${escapeHtml(view.status)}</strong>
      <div class="capability-meta">
        <span>trial ${escapeHtml(view.trialId)}</span>
        <span>device ${escapeHtml(view.deviceId)}</span>
        <span>action ${escapeHtml(view.action)}</span>
        <span>physical_actions ${escapeHtml(view.physicalActions)}</span>
        <span>report ${escapeHtml(reportState)}</span>
        <span>revision ${escapeHtml(view.codeRevision || "—")}</span>
      </div>
      <small>${view.requiresRestart
        ? "能力配置已写入；等待安全重启后生效。"
        : report?.status === "failed"
          ? escapeHtml(report.error || "本次验收未满足通过标准，禁止晋级和重试。")
          : view.passed
            ? "八帧证据和单动作结果已通过；仍需独立确认才能写入能力配置。"
            : "当前验收不提供连续执行，每次确认最多一个物理动作。"}</small>`;
    badge.className = `pill ${view.requiresRestart ? "success" : view.readOnlyRecovered ? "neutral" : canPromote ? "risk" : report?.status === "failed" ? "danger" : "active"}`;
    badge.textContent = view.requiresRestart ? "等待重启" : view.readOnlyRecovered ? "只读恢复" : canPromote ? "待确认启用" : view.passed ? "报告通过 · 晋级不可用" : report?.status === "failed" ? "验收失败" : "等待单步确认";

    const disabled = state.busy || state.paused ? "disabled" : "";
    if (view.readOnlyRecovered) {
      controls.innerHTML = `<button id="resetCapabilityTrial" class="secondary-button">关闭只读记录</button>`;
    } else if (view.requiresRestart) {
      controls.innerHTML = `<button id="resetCapabilityTrial" class="secondary-button">关闭本次结果</button>`;
    } else if (canPromote) {
      controls.innerHTML = `
        <button id="reviewCapabilityPromotion" class="danger-confirm" ${disabled}>确认启用该能力</button>
        <button id="resetCapabilityTrial" class="secondary-button">保留报告并关闭</button>`;
    } else if (report?.status === "failed") {
      controls.innerHTML = `<button id="resetCapabilityTrial" class="secondary-button">开始新的验收</button>`;
    } else if (["awaiting_confirmation", "awaiting_effect_confirmation"].includes(view.status)) {
      controls.innerHTML = `
        <button id="reviewCapabilityAction" class="risk-button" ${disabled}>${view.status === "awaiting_effect_confirmation" ? "确认效果（0 动作）" : "确认执行本次验收动作"}</button>
        <button id="cancelCapabilityTrial" class="text-button" ${state.busy ? "disabled" : ""}>取消验收</button>`;
    } else {
      controls.innerHTML = `<button id="cancelCapabilityTrial" class="text-button" ${state.busy ? "disabled" : ""}>取消验收</button>`;
    }
  }

  const ordinarySessionActive = Boolean(sessionView() && !sessionView().isTerminal);
  const trialActive = Boolean(view && !view.report && !view.requiresRestart);
  select.disabled = Boolean(view) || state.busy;
  goal.disabled = Boolean(view) || state.busy;
  start.disabled = !available.length || ordinarySessionActive || trialActive || state.busy || state.paused;
  document.querySelector("#reviewCapabilityAction")?.addEventListener("click", openCapabilityDialog);
  document.querySelector("#reviewCapabilityPromotion")?.addEventListener("click", openPromotionDialog);
  document.querySelector("#cancelCapabilityTrial")?.addEventListener("click", cancelCapabilityTrial);
  document.querySelector("#resetCapabilityTrial")?.addEventListener("click", resetCapabilityTrial);
}

function render() {
  const view = sessionView();
  const deviceSelect = document.querySelector("#deviceId");
  const registeredDevices = Array.isArray(state.device?.devices) ? state.device.devices : [];
  if (registeredDevices.length) {
    const enabledIds = registeredDevices.map(item => String(item.device_id || "")).filter(Boolean);
    const lockedDeviceId = view && !view.isTerminal ? String(view.deviceId || "") : "";
    const optionIds = [...enabledIds];
    if (lockedDeviceId && !optionIds.includes(lockedDeviceId)) optionIds.push(lockedDeviceId);
    deviceSelect.replaceChildren(...optionIds.map(deviceId => {
      const option = document.createElement("option");
      option.value = deviceId;
      option.textContent = deviceId;
      return option;
    }));
    if (lockedDeviceId) {
      state.deviceId = lockedDeviceId;
    } else if (!enabledIds.includes(state.deviceId)) {
      state.deviceId = String(state.device.default_device_id || enabledIds[0]);
      localStorage.setItem("visual-agent-device-id", state.deviceId);
    }
  }
  deviceSelect.value = state.deviceId;
  const acceptance = capabilityView();
  deviceSelect.disabled = state.busy || Boolean(acceptance && !acceptance.report);
  document.querySelector("#startSupervisedAgent").disabled = state.busy
    || state.paused
    || Boolean(acceptance && !acceptance.report);
  document.querySelector("#agentText").disabled = state.busy;
  document.querySelector("#pauseButton").textContent = state.paused ? "▶ 继续推进" : "Ⅱ 暂停推进";
  document.querySelector("#pauseButton").classList.toggle("active", state.paused);
  renderStatus();
  renderGoalAndPlan();
  renderTrace();
  renderScene();
  renderAction();
  renderCapabilityAcceptance();
}

async function refreshDevice() {
  state.device = await api("/api/device");
  renderStatus();
}

function refreshPreview() {
  const preview = document.querySelector("#phonePreview");
  if (preview && state.deviceId) {
    preview.src = `/api/preview.jpg?device_id=${encodeURIComponent(state.deviceId)}&t=${Date.now()}`;
  }
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
    const actions = Number(response.physical_actions || 0);
    toast(actions
      ? `安全任务已自动推进 ${actions} 个物理动作，并在每步后重新观察。`
      : "计划与只读观察已完成；当前没有可自动执行的安全动作。");
    render();
  } catch (error) {
    toast(error.message, true);
  }
}

async function startCapabilityTrial() {
  const action = document.querySelector("#capabilityAction").value;
  const text = document.querySelector("#capabilityGoal").value.trim();
  const current = sessionView();
  if (!action) return toast("当前设备没有可选择的待验收动作。", true);
  if (!text) return toast("请填写一个通用、可见且安全的真机验收目标。", true);
  if (current && !current.isTerminal) return toast("当前设备已有普通 Agent 会话。", true);
  if (state.capabilityTrial) return toast("请先关闭当前验收结果。", true);
  state.capabilityDeviceId = state.deviceId;
  try {
    const response = await withVisionProgress("生成验收计划并观察当前画面（0 动作）", () =>
      api("/api/capability-acceptance/start", {
        method: "POST",
        body: JSON.stringify({
          device_id: state.capabilityDeviceId,
          action,
          text,
        }),
      })
    );
    state.capabilityTrial = response.trial;
    toast("验收计划已生成，物理动作数为 0。请核对精确作用域。")
    render();
  } catch (error) {
    toast(error.message, true);
  }
}

function openCapabilityDialog() {
  const view = capabilityView();
  if (!view || state.paused || state.busy) return;
  try {
    state.pendingConfirmationGrant = Protocol.createCapabilityConfirmationGrant(state.capabilityTrial);
    state.pendingConfirmationGrant.kind = "capability";
  } catch (error) {
    state.pendingConfirmationGrant = null;
    return toast(error.message, true);
  }
  const effectPhase = state.pendingConfirmationGrant.phase === "effect";
  const session = view.session;
  document.querySelector("#riskTitle").textContent = effectPhase
    ? "确认验收子目标的效果范围"
    : "确认执行本次真机验收动作";
  const level = document.querySelector("#riskLevel");
  level.className = "risk-level high";
  level.textContent = effectPhase ? "此确认只允许观察 · 物理动作 0" : "真机动作 · 最多执行一次";
  document.querySelector("#riskGoal").textContent = view.text || session?.objective || "—";
  document.querySelector("#riskAction").textContent = `${semanticActionNames[view.action] || view.action} · trial=${view.trialId}`;
  document.querySelector("#riskReason").textContent = session?.visualAction?.reason || "依据当前真实画面提出唯一候选动作。";
  document.querySelector("#riskExpected").textContent = effectPhase
    ? "生成一个与候选动作完全一致的视觉动作，不触发机械臂"
    : Protocol.displayValue(session?.visualAction?.expectedChange);
  document.querySelector("#riskDevice").textContent = view.deviceId;
  const scope = state.pendingConfirmationGrant.scope;
  document.querySelector("#riskWarning").textContent = effectPhase
    ? `仅确认 trial=${view.trialId}、action=${view.action}、session=${scope.session_id}、task=${scope.task_id}、revision=${scope.revision}、subgoal=${scope.subgoal_id}、effect_ids=${scope.effect_ids.join(",") || "—"} 的观察权限；本次物理动作数必须保持 0。`
    : `只授权 trial=${view.trialId}、action=${view.action}、session=${scope.session_id}、task=${scope.task_id}、revision=${scope.revision}、subgoal=${scope.subgoal_id}、observation_id=${scope.observation_id}、fingerprint=${scope.fingerprint} 对应的一个动作；失败不自动重试。`;
  document.querySelector("#confirmRiskAction").className = effectPhase ? "primary-button" : "danger-confirm";
  document.querySelector("#riskDialog").showModal();
}

async function advanceCapabilityTrial(grant) {
  const view = capabilityView();
  if (!view || state.paused || state.busy) return;
  try {
    const payload = Protocol.consumeCapabilityConfirmationGrant(grant, state.capabilityTrial);
    const effectPhase = grant.phase === "effect";
    const response = await withVisionProgress(
      effectPhase ? "确认验收效果范围并观察（0 动作）" : "执行唯一验收动作并采集八帧证据",
      () => api(`/api/capability-acceptance/${view.trialId}/${effectPhase ? "approve-effect" : "confirm"}`, {
        method: "POST",
        body: JSON.stringify(payload),
      })
    );
    state.capabilityTrial = response.trial;
    if (!effectPhase) await loadCapabilityEvidence();
    toast(effectPhase
      ? "效果范围已确认，机械臂尚未动作；请再次核对具体动作。"
      : "本次单动作已终结，已生成验收报告；不会自动重试。")
    render();
  } catch (error) {
    await refreshCapabilityTrial().catch(() => {});
    await loadCapabilityEvidence().catch(() => {});
    toast(error.message, true);
  }
}

async function refreshCapabilityTrial() {
  const view = capabilityView();
  if (!view?.trialId) return;
  const response = await api(`/api/capability-acceptance/${view.trialId}`);
  state.capabilityTrial = response.trial;
  render();
}

function clearCapabilityEvidenceUrls() {
  state.capabilityEvidenceUrls.forEach(url => URL.revokeObjectURL(url));
  state.capabilityEvidenceUrls = [];
}

async function loadCapabilityEvidence() {
  clearCapabilityEvidenceUrls();
  const view = capabilityView();
  const container = document.querySelector("#capabilityEvidence");
  const report = view?.report;
  if (!view || !report) {
    container.hidden = true;
    container.innerHTML = "";
    return;
  }
  const items = [];
  for (const phase of ["before", "after"]) {
    const paths = report[`${phase}_frame_paths`] || [];
    for (let index = 0; index < paths.length; index += 1) {
      try {
        const response = await fetch(
          `/api/capability-acceptance/${view.trialId}/evidence/${phase}/${index}`,
          { headers: { "X-Control-Token": state.token } },
        );
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const url = URL.createObjectURL(await response.blob());
        state.capabilityEvidenceUrls.push(url);
        items.push({ phase, index, path: paths[index], url });
      } catch (_error) {
        items.push({ phase, index, path: paths[index], url: "" });
      }
    }
  }
  container.hidden = !items.length;
  container.innerHTML = items.map(item => `<figure>
    ${item.url ? `<img src="${item.url}" alt="${item.phase === "before" ? "动作前" : "动作后"}证据 ${item.index + 1}">` : ""}
    <figcaption>${item.phase === "before" ? "动作前" : "动作后"} ${item.index + 1} · ${escapeHtml(item.path)}</figcaption>
  </figure>`).join("");
}

async function openPromotionDialog() {
  const view = capabilityView();
  if (!view?.passed || state.busy) return;
  try {
    const preview = await api(`/api/capability-acceptance/${view.trialId}/promotion-preview`);
    state.capabilityTrial = {
      ...state.capabilityTrial,
      promotion_scope: preview.promotion_scope,
    };
    state.pendingPromotionGrant = Protocol.createPromotionGrant(state.capabilityTrial);
    const scope = state.pendingPromotionGrant.scope;
    document.querySelector("#promotionTrial").textContent = scope.trial_id;
    document.querySelector("#promotionTarget").textContent = `${scope.device_id} / ${scope.action}`;
    document.querySelector("#promotionReportHash").textContent = scope.report_sha256;
    document.querySelector("#promotionRegistryHash").textContent = scope.registry_sha256;
    document.querySelector("#promotionDialog").showModal();
  } catch (error) {
    state.pendingPromotionGrant = null;
    toast(error.message, true);
  }
}

async function promoteCapability(grant) {
  const view = capabilityView();
  if (!view || state.busy) return;
  try {
    const payload = Protocol.consumePromotionGrant(grant, state.capabilityTrial);
    state.busy = true;
    render();
    const response = await api(`/api/capability-acceptance/${view.trialId}/promote`, {
      method: "POST",
      body: JSON.stringify(payload),
    });
    state.capabilityTrial = response.trial;
    toast("能力配置已原子写入；不会热更新，请等待安全重启。")
  } catch (error) {
    await refreshCapabilityTrial().catch(() => {});
    toast(error.message, true);
  } finally {
    state.busy = false;
    render();
  }
}

async function cancelCapabilityTrial() {
  const view = capabilityView();
  if (!view || state.busy) return;
  try {
    const response = await api(`/api/capability-acceptance/${view.trialId}/cancel`, {
      method: "POST",
      body: JSON.stringify({ device_id: view.deviceId, action: view.action }),
    });
    state.capabilityTrial = response.trial;
    toast("真机能力验收已取消，未执行后续动作。")
    render();
  } catch (error) {
    toast(error.message, true);
  }
}

function resetCapabilityTrial() {
  clearCapabilityEvidenceUrls();
  state.capabilityTrial = null;
  state.capabilityDeviceId = "";
  state.pendingPromotionGrant = null;
  render();
}

function openRiskDialog() {
  const view = sessionView();
  if (!view || state.paused || state.busy || !view.effectPolicy.requiresConfirmation) return;
  const effectPhase = view.status === "awaiting_effect_confirmation"
    || view.effectPolicy.confirmationGate.phase === "effect";
  const highAttention = view.effectPolicy.requiresConfirmation;
  try {
    state.pendingConfirmationGrant = Protocol.createConfirmationGrant(view, lockedSessionDeviceId());
  } catch (error) {
    state.pendingConfirmationGrant = null;
    return toast(error.message, true);
  }
  document.querySelector("#riskTitle").textContent = effectPhase
    ? "确认当前登录或付款范围"
    : (highAttention ? "确认登录或付款动作" : "确认当前单步动作");
  const level = document.querySelector("#riskLevel");
  level.className = `risk-level ${highAttention ? "high" : "guarded"}`;
  level.textContent = highAttention ? "需要确认 · 仅限登录或付款" : "受控动作 · 仅授权当前一步";
  document.querySelector("#riskGoal").textContent = view.objective;
  document.querySelector("#riskAction").textContent = view.visualAction.actionType
    ? `${actionLabel(view.visualAction)} · ${view.visualAction.semanticTarget}`
    : `${view.currentSubgoal.label} · 等待 Qwen 唯一动作`;
  document.querySelector("#riskReason").textContent = view.effectPolicy.currentActions.map(item => `${item.id} [${item.policyLevel}]：${item.kind}；${item.expectedResults.join("、")}`).join("\n") || view.visualAction.reason;
  const effectPreviews = Array.isArray(view.effectPolicy.effectPreviews)
    ? view.effectPolicy.effectPreviews
    : [];
  if (effectPreviews.length) {
    const previewLines = effectPreviews.map((preview) => {
      const targetValues = Array.isArray(preview.targets)
        ? preview.targets.map(item => Protocol.displayValue(item && item.value)).join("、")
        : "";
      const payloadValues = Array.isArray(preview.payloads)
        ? preview.payloads.map(item => Protocol.displayValue(item && item.value)).join("、")
        : "";
      return [
        `效果 ${preview.effect_kind || preview.effect_id || "unknown"}`,
        targetValues ? `目标=${targetValues}` : "",
        payloadValues ? `载荷=${payloadValues}` : "",
        `策略=${preview.policy || "unknown"}`,
        preview.preview_digest ? `摘要=${preview.preview_digest}` : "",
      ].filter(Boolean).join("；");
    });
    document.querySelector("#riskReason").textContent = previewLines.join("\n");
  }
  document.querySelector("#riskExpected").textContent = Protocol.displayValue(view.visualAction.expectedChange);
  document.querySelector("#riskDevice").textContent = lockedSessionDeviceId();
  document.querySelector("#riskWarning").textContent = effectPhase
    ? "后端效果 scope 与当前权威任务及 typed EffectIntent 一致；确认后最多执行一个由新观察严格绑定的动作。"
    : highAttention
    ? "后端动作 scope 与当前权威任务、观察和动作字段一致；本次只授权当前一个动作，任何字段变化都必须重新确认。"
    : "后端动作 scope 与当前权威任务、观察和动作字段一致；本次只授权一个动作，执行后必须重新观察。";
  document.querySelector("#confirmRiskAction").className = highAttention ? "danger-confirm" : "primary-button";
  document.querySelector("#riskDialog").showModal();
}

function capabilityView() {
  return state.capabilityTrial
    ? Protocol.adaptCapabilityTrial(state.capabilityTrial)
    : null;
}

async function advanceSupervisedAgent(grant) {
  const view = sessionView();
  if (!view || state.paused || state.busy) return;
  try {
    const payload = Protocol.consumeConfirmationGrant(grant, view, lockedSessionDeviceId());
    const effectPhase = grant?.phase === "effect";
    const response = await withVisionProgress(
      effectPhase ? "确认效果并生成唯一动作" : "执行当前一步并重新观察",
      () => api(`/api/agent/generic-supervised/${view.sessionId}/${effectPhase ? "approve-effect" : "confirm"}`, {
        method: "POST",
        body: JSON.stringify(payload),
      })
    );
    state.supervisedSession = response.session;
    await finalizeStopIfRequested();
    toast(effectPhase
      ? (Number(response.physical_actions || 0)
        ? "效果策略已确认，唯一外部影响动作已执行并重新观察。"
        : "效果策略已确认，但当前画面没有形成可安全执行的唯一动作。")
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
    if (capabilityView() && !capabilityView().report) await cancelCapabilityTrial();
    toast(result.note || "停止请求已发送。");
    await refreshDevice();
    render();
  } catch (error) {
    toast(error.message, true);
  }
}

async function restoreActiveSession() {
  const activeSessions = state.device.generic_supervised_execution?.active_sessions;
  const active = Array.isArray(activeSessions)
    ? activeSessions.find(item => String(item.device_id || "") === state.deviceId)
    : null;
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

async function restoreCapabilityTrial() {
  try {
    const response = await api("/api/capability-acceptance");
    const trials = Array.isArray(response.trials) ? response.trials : [];
    const matching = trials.filter(item => String(item.device_id || "") === state.deviceId);
    if (!matching.length) return;
    state.capabilityTrial = matching[matching.length - 1];
    state.capabilityDeviceId = state.deviceId;
    await loadCapabilityEvidence().catch(() => {});
  } catch (_error) {
    state.capabilityTrial = null;
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
    if (!state.supervisedSession) await restoreCapabilityTrial();
    render();
    refreshPreview();
    setInterval(refreshPreview, 700);
    setInterval(() => refreshDevice().catch(() => {}), 3000);
  } catch (error) {
    toast(`连接本地服务失败：${error.message}`, true);
  }
}

document.querySelector("#startSupervisedAgent").addEventListener("click", startSupervisedAgent);
document.querySelector("#startCapabilityTrial").addEventListener("click", startCapabilityTrial);
document.querySelector("#pauseButton").addEventListener("click", togglePause);
document.querySelector("#stopButton").addEventListener("click", stopTasks);
document.querySelector("#deviceId").addEventListener("change", async event => {
  const requestedDeviceId = String(event.target.value || "");
  const registeredDeviceIds = Array.isArray(state.device?.devices)
    ? state.device.devices.map(item => String(item.device_id || "")).filter(Boolean)
    : [];
  if (!registeredDeviceIds.includes(requestedDeviceId)) {
    event.target.value = lockedSessionDeviceId();
    render();
    return;
  }
  state.deviceId = requestedDeviceId;
  localStorage.setItem("visual-agent-device-id", state.deviceId);
  state.supervisedSession = null;
  state.sessionDeviceId = "";
  state.pendingConfirmationGrant = null;
  state.capabilityTrial = null;
  state.capabilityDeviceId = "";
  state.capabilityEvidence = [];
  render();
  await restoreActiveSession();
  if (!state.supervisedSession) await restoreCapabilityTrial();
  render();
  refreshPreview();
});
document.querySelector("#agentText").addEventListener("keydown", event => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") startSupervisedAgent();
});
document.querySelector("#riskDialog").addEventListener("close", event => {
  const grant = state.pendingConfirmationGrant;
  state.pendingConfirmationGrant = null;
  if (event.target.returnValue === "default" && grant) {
    if (grant.kind === "capability") advanceCapabilityTrial(grant);
    else advanceSupervisedAgent(grant);
  }
});
document.querySelector("#promotionDialog").addEventListener("close", event => {
  const grant = state.pendingPromotionGrant;
  state.pendingPromotionGrant = null;
  if (event.target.returnValue === "default" && grant) promoteCapability(grant);
});

init();
