const Protocol = window.UniversalAgentProtocol;
if (!Protocol) throw new Error("任务协议适配层未加载。");

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
  taskAttemptStatus: null,
  lastTaskOutcome: null,
  luckyBagProfile: null,
  luckyBagMonitorId: "",
  luckyBagMonitorTimer: null,
};

const taskOutcomeStoragePrefix = "visual-agent-task-outcome:";

const statusNames = {
  idle: "等待目标",
  budget_paused: "整任务预算已用尽",
  paused: "已暂停，进度已保留",
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
  finish: "当前目标完成",
  unknown: "等待视觉决策",
};

const semanticActionNames = {
  tap_semantic: "点击语义控件",
  dismiss_overlay: "关闭当前弹层",
  scroll: "滚动当前页面",
  swipe_element: "滑动目标元素",
  back: "返回上一页",
  home: "返回系统桌面",
  reveal_system_navigation: "唤出系统导航栏",
  double_tap: "双击目标控件",
  long_press: "长按目标控件",
  drag: "拖动目标控件",
  input_verified_text: "输入并核对文字",
  wait_for_change: "等待页面变化",
  finish: "完成本次任务",
};

const capabilityStatusNames = {
  awaiting_confirmation: "等待确认",
  awaiting_effect_confirmation: "等待效果确认",
  paused: "已暂停",
  succeeded: "已完成",
  completed: "已完成",
  failed: "失败",
  cancelled: "已取消",
};

async function api(path, options = {}, timeoutMs = 0) {
  const headers = { ...(options.headers || {}) };
  if (state.token) headers["X-Control-Token"] = state.token;
  if (options.body) headers["Content-Type"] = "application/json";
  const controller = timeoutMs > 0 ? new AbortController() : null;
  const timer = controller ? setTimeout(() => controller.abort(), timeoutMs) : null;
  try {
    const response = await fetch(path, {
      ...options,
      headers,
      ...(controller ? { signal: controller.signal } : {}),
    });
    const data = await response.json().catch(() => {
      throw new Error("无法读取服务返回的数据，请重试。");
    });
    if (!response.ok) {
      const detail = data.detail;
      const message = typeof detail === "string" ? detail : (detail?.error || JSON.stringify(detail || {}));
      const error = new Error(message || `请求失败（${response.status}）`);
      error.detail = detail;
      error.status = response.status;
      throw error;
    }
    return data;
  } catch (error) {
    if (error?.name === "AbortError") {
      throw new Error("启动任务超过10秒仍未完成首轮画面观察，请检查模型服务或网络状态。" );
    }
    throw error;
  } finally {
    if (timer) clearTimeout(timer);
  }
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

function taskOutcomeStorageKey(deviceId) {
  return `${taskOutcomeStoragePrefix}${String(deviceId || "unknown")}`;
}

function readLastTaskOutcome(deviceId) {
  try {
    const parsed = JSON.parse(sessionStorage.getItem(taskOutcomeStorageKey(deviceId)) || "null");
    if (!parsed || !["success", "failure"].includes(parsed.state)) return null;
    return {
      state: parsed.state,
      detail: String(parsed.detail || ""),
      sessionId: String(parsed.sessionId || ""),
      updatedAt: String(parsed.updatedAt || ""),
    };
  } catch (_error) {
    return null;
  }
}

function saveLastTaskOutcome(outcome) {
  const normalized = {
    state: outcome.state,
    detail: String(outcome.detail || ""),
    sessionId: String(outcome.sessionId || ""),
    updatedAt: String(outcome.updatedAt || new Date().toISOString()),
  };
  state.lastTaskOutcome = normalized;
  try {
    sessionStorage.setItem(taskOutcomeStorageKey(state.deviceId), JSON.stringify(normalized));
  } catch (_error) {
    // The live page still keeps the result in memory when browser storage is unavailable.
  }
  return normalized;
}

function clearLastTaskOutcome() {
  state.lastTaskOutcome = null;
  try {
    sessionStorage.removeItem(taskOutcomeStorageKey(state.deviceId));
  } catch (_error) {
    // Browser storage is optional; current in-memory status remains authoritative.
  }
}

function rememberTerminalTaskOutcome(view, stateName, detail) {
  const existing = state.lastTaskOutcome;
  if (
    existing
    && existing.state === stateName
    && existing.sessionId === view.sessionId
    && existing.detail === detail
  ) return existing;
  return saveLastTaskOutcome({
    state: stateName,
    detail,
    sessionId: view.sessionId,
    updatedAt: new Date().toISOString(),
  });
}

function formatTaskStatusTime(value) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toLocaleString("zh-CN", { hour12: false });
}

function taskRunPresentation() {
  if (state.taskAttemptStatus?.state === "running") {
    return {
      state: "running",
      label: "进行中",
      detail: state.visionStage || state.taskAttemptStatus.detail || "正在理解目标并观察当前画面。",
      sessionId: "",
      updatedAt: state.taskAttemptStatus.updatedAt,
    };
  }
  if (state.taskAttemptStatus?.state === "failure") {
    return {
      state: "failure",
      label: "失败",
      detail: state.taskAttemptStatus.detail || "任务未能启动。",
      sessionId: "",
      updatedAt: state.taskAttemptStatus.updatedAt,
    };
  }

  const view = sessionView();
  if (view) {
    if (["succeeded", "completed"].includes(view.status)) {
      const detail = `目标已完成；共完成 ${view.physicalActions} 次操作。`;
      const outcome = rememberTerminalTaskOutcome(view, "success", detail);
      return { ...outcome, label: "成功" };
    }
    if (view.isTerminal) {
      const detail = view.failedReason
        || view.stopState?.reason
        || view.visualAction?.reason
        || "任务已经结束，但没有完成目标。";
      const outcome = rememberTerminalTaskOutcome(view, "failure", detail);
      return { ...outcome, label: "失败" };
    }
    const step = view.currentStep?.label;
    return {
      state: "running",
      label: "进行中",
      detail: step
        ? `当前步骤：${step}（${statusNames[view.status] || view.status}）`
        : (statusNames[view.status] || "任务正在处理。"),
      sessionId: view.sessionId,
      updatedAt: String(view.raw?.created_at || ""),
    };
  }

  if (state.lastTaskOutcome) {
    return {
      ...state.lastTaskOutcome,
      label: state.lastTaskOutcome.state === "success" ? "成功" : "失败",
      detail: `最近一次任务：${state.lastTaskOutcome.detail}`,
    };
  }
  return {
    state: "not-started",
    label: "未开始",
    detail: "还没有运行中的任务。先选择设备并确认状态，再输入你想完成的任务。",
    sessionId: "",
    updatedAt: "",
  };
}

function renderTaskRunStatus() {
  const presentation = taskRunPresentation();
  const container = document.querySelector("#taskRunStatus");
  if (!container) return;
  container.dataset.taskState = presentation.state;
  document.querySelector("#taskRunStatusLabel").textContent = presentation.label;
  document.querySelector("#taskRunStatusDetail").textContent = presentation.detail;
  document.querySelector("#taskRunSessionId").textContent = presentation.sessionId || "未创建";
  document.querySelector("#taskRunUpdatedAt").textContent = formatTaskStatusTime(presentation.updatedAt);
  const emptyState = document.querySelector("#taskEmptyState");
  if (emptyState) emptyState.hidden = presentation.state !== "not-started";
  const currentTask = document.querySelector("#currentTaskLabel");
  const view = sessionView();
  if (currentTask) currentTask.textContent = view?.objective || (presentation.state === "running" ? "正在创建任务" : "未开始");
  const taskStatus = document.querySelector("#topTaskStatus");
  if (taskStatus) taskStatus.textContent = presentation.label;
}

function renderQwenConversation() {
  const messagesElement = document.querySelector("#qwenMessages");
  const stateElement = document.querySelector("#qwenResponseState");
  if (!messagesElement || !stateElement) return;
  const view = sessionView();
  const attempt = state.taskAttemptStatus;
  const raw = view?.raw || {};
  const malformed = (raw.conversation != null && (!Array.isArray(raw.conversation)
    || raw.conversation.some(item => item && ["user", "assistant"].includes(item.role)
      && typeof item.content !== "string")))
    || (raw.qwen_reply != null && typeof raw.qwen_reply !== "string");
  const failed = attempt?.state === "failure" || ["failed", "blocked"].includes(view?.status);
  const parseFailure = attempt?.state === "failure"
    && /无法读取服务返回的数据/.test(String(attempt.detail || ""));
  const messages = [];
  const conversation = Array.isArray(view?.conversation) ? view.conversation : [];
  for (const message of conversation) {
    if (!(failed || malformed) || message.role === "user") messages.push(message);
  }
  const markup = messages.map(message => {
    const role = message.role;
    const label = role === "user" ? "你" : "Qwen";
    return `<article class="qwen-message ${role}"><span class="qwen-message-label">${label}</span><div class="qwen-message-content">${escapeHtml(message.content)}</div></article>`;
  }).join("");
  if (messagesElement.innerHTML !== markup) {
    messagesElement.innerHTML = markup;
    const timeline = document.querySelector("#qwenTimeline");
    timeline.scrollTop = timeline.scrollHeight;
  }
  stateElement.className = "qwen-response-state";
  if (state.busy || attempt?.state === "running") {
    stateElement.classList.add("loading");
    stateElement.textContent = "Qwen 正在分析…";
  } else if (failed || malformed) {
    stateElement.classList.add("error");
    stateElement.textContent = (malformed || parseFailure)
      ? "回答格式无法读取，请重试。"
      : "本次请求失败，未获得有效回答。请查看任务状态。";
  } else if (view?.status === "awaiting_confirmation" || view?.status === "awaiting_effect_confirmation") {
    stateElement.classList.add("confirmation");
    stateElement.textContent = "等待你的确认";
  } else if (messages.some(item => item.role === "assistant")) {
    stateElement.classList.add("ready");
    stateElement.textContent = "Qwen 已回复";
  } else {
    stateElement.textContent = "尚未收到 Qwen 回答";
  }
}

function restoreSupervisedSessionFromError(error) {
  const failedSession = error?.detail?.session;
  if (!failedSession || typeof failedSession !== "object") return false;
  state.supervisedSession = failedSession;
  state.taskAttemptStatus = null;
  const restored = Protocol.adaptSession(failedSession, { fallbackDeviceId: state.deviceId });
  state.sessionDeviceId = restored.deviceId || state.deviceId;
  return true;
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

function planState(step, view) {
  if (step.id === view.currentStep.id) return "current";
  if (["done", "completed", "succeeded"].includes(step.status)) return "done";
  if (["failed", "blocked", "cancelled"].includes(step.status)) return "blocked";
  return "waiting";
}

function renderStatus() {
  const device = currentDeviceDescriptor() || state.device || {};
  const view = sessionView();
  const taskStatus = taskRunPresentation();
  setDot("#controllerDot", device.controller_online ? (device.busy ? "warn" : "online") : "offline");
  setDot("#cameraDot", device.camera_online ? "online" : "offline");
  setDot("#agentDot", taskStatus.state === "running"
    ? "warn"
    : taskStatus.state === "success"
      ? "online"
      : taskStatus.state === "failure"
        ? "offline"
        : "neutral");
  document.querySelector("#controllerText").textContent = device.controller_online
    ? (device.busy ? "正在操作" : "已连接")
    : "未连接";
  document.querySelector("#cameraText").textContent = device.camera_online ? "画面可用" : "画面不可用";
  document.querySelector("#agentTextStatus").textContent = state.paused ? "人工暂停" : taskStatus.label;
  const connected = Boolean(device.controller_online && device.camera_online);
  const connection = document.querySelector("#connectionStatus");
  if (connection) {
    connection.textContent = connected ? "设备已连接" : "设备未连接";
    connection.className = `context-status ${connected ? "status-online" : "status-offline"}`;
  }
  const previewDevice = document.querySelector("#previewDeviceName");
  const previewStatus = document.querySelector("#previewDeviceStatus");
  if (previewDevice) previewDevice.textContent = state.deviceId || "未选择设备";
  if (previewStatus) previewStatus.textContent = connected ? "已连接" : "未连接";
  const effectPhase = view?.status === "awaiting_effect_confirmation";
  const actionPhase = view?.status === "awaiting_confirmation";
  const highAttention = Boolean(view?.effectPolicy.requiresConfirmation);
  document.querySelector("#safetyText").textContent = effectPhase
    ? "等待确认"
    : actionPhase
      ? "等待选择"
      : highAttention
        ? "需要确认"
        : "已保护";
  setDot("#safetyDot", effectPhase || actionPhase || highAttention ? "warn" : "online");
}

function renderGoalAndPlan() {
  const view = sessionView();
  const goalElement = document.querySelector("#goalSummary");
  const planElement = document.querySelector("#planList");
  const badge = document.querySelector("#planBadge");
  if (!view) {
    goalElement.className = "goal-summary empty-state";
    goalElement.textContent = "输入目标后，这里会展示整任务、当前动作和实际执行历史。";
    planElement.innerHTML = "";
    badge.className = "pill neutral";
    badge.textContent = "等待目标";
    return;
  }

  goalElement.className = "goal-summary";
    goalElement.innerHTML = `
    <div class="goal-title-row">
      <div><span>当前任务 · 第 ${escapeHtml(view.revision ?? "—")} 轮 <span class="sr-only">任务编号 · ${escapeHtml(view.taskId || "待分配")} · revision ${escapeHtml(view.revision ?? "—")}</span></span><strong>${escapeHtml(view.objective)}</strong></div>
      <code>${escapeHtml(view.deviceId || lockedSessionDeviceId())}</code>
    </div>
    <details class="technical-details goal-details" open data-collapsed="true">
      <summary>查看任务详情</summary>
      <div class="goal-chips">
      <span>任务会话 · ${escapeHtml(view.sessionId || "—")}</span>
      ${view.protocolVersion ? `<span>任务协议 · ${escapeHtml(view.protocolVersion)}</span>` : ""}
      ${view.compatibilityFallback ? `<span>历史协议数据 · 不可用于当前执行</span>` : ""}
      ${view.targetApps.map(app => `<span>目标应用 · ${escapeHtml(app.name)}${app.id ? ` (${escapeHtml(app.id)})` : ""}</span>`).join("")}
      ${!view.targetApps.length && view.appName ? `<span>目标应用 · ${escapeHtml(view.appName)}</span>` : ""}
      ${view.constraints.map(item => `<span>限制 · ${escapeHtml(item)}</span>`).join("")}
      ${view.completionConditions.map(item => `<span>完成 · ${escapeHtml(item)}</span>`).join("")}
      <span>确认状态 · ${view.effectPolicy.confirmationGate.required ? "需要确认" : "无需确认"}</span>
      <span>操作类型 · ${escapeHtml(view.effectPolicy.currentExecutionClass)}</span>
      <span>设备条件 · ${escapeHtml(controllerGateLabel(view.controllerGate))}</span>
      <span>作用范围 · ${escapeHtml(view.scopeState.state)} · ${escapeHtml(view.scopeState.reason || "—")}</span>
      ${view.effectPolicy.actions.map(item => `<span>影响范围 · ${escapeHtml(item.id)} · ${escapeHtml(item.kind)}</span>`).join("")}
      </div>
    </details>`;

  const steps = view.steps.map(item => planStepHtml({
    number: item.index,
    label: item.label,
    detail: item.reason,
    stateName: planState(item, view),
  })).join("");
  const dynamicTail = view.isTerminal ? "" : planStepHtml({
    number: "…",
    label: "后续步骤等待新画面",
    detail: "系统结合整项任务、实际操作记录和新画面选择下一步",
    stateName: "waiting",
  });
  const current = view.isTerminal ? "" : planStepHtml({
    number: view.stepNumber, label: view.currentStep.label,
    detail: view.visualAction.status === "action" ? actionLabel(view.visualAction) : "等待当前画面决策",
    stateName: "current",
  });
  planElement.innerHTML = steps + current + dynamicTail;
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
    new_screenshot_decision: "新截图决策",
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
        <div class="trace-title"><strong>第 ${escapeHtml(item.stepNumber)} 步 · 第 ${escapeHtml(item.taskRevision ?? "—")} 轮 <span class="sr-only">步骤 ${escapeHtml(item.stepNumber)} · revision ${escapeHtml(item.taskRevision ?? "—")}</span></strong><time>已执行 ${escapeHtml(item.physicalActions)} 个动作 <span class="sr-only">physical_actions ${escapeHtml(item.physicalActions)}</span></time></div>
        <p><b>任务内容</b> ${escapeHtml(item.taskContext || "—")}${item.stepId ? ` · ${escapeHtml(item.stepId)}` : ""}</p>
        <div class="trace-grid">
          <span><b>动作前观察</b>${escapeHtml(observationText)}</span>
          <span><b>下一步判断</b><span class="sr-only">QWEN 唯一动作</span>${escapeHtml(actionLabel(item.action))} · ${escapeHtml(item.action?.semanticTarget || "—")}</span>
          <span><b>执行条件</b>${escapeHtml(gateText)}${gate.reason ? ` · ${escapeHtml(gate.reason)}` : ""}</span>
          <span><b>动作后验证</b>${escapeHtml(outcomeNames[verification.outcome] || verification.outcome || "旧记录未提供")} · ${escapeHtml(afterObservationText)}</span>
          <span><b>任务图去向</b>${escapeHtml(transitionLabel)}${transition.trigger ? ` · ${escapeHtml(transition.trigger)}` : ""}${transition.toRevision !== null && transition.toRevision !== undefined ? ` · r${escapeHtml(transition.fromRevision ?? "—")}→r${escapeHtml(transition.toRevision)}` : ""}</span>
          <span class="scope-state ${escapeHtml(scopeState.state)}"><b>确认范围</b>${escapeHtml(scopeNames[scopeState.state] || scopeState.state)}${scopeState.reason ? ` · ${escapeHtml(scopeState.reason)}` : ""}</span>
        </div>
        ${(verification.errors || []).length ? `<small>验证/阻止原因：${escapeHtml(verification.errors.join("；"))}</small>` : ""}
        ${transition.reason ? `<small>下一截图原因：${escapeHtml(transition.reason)}</small>` : ""}
        ${(verification.evidence || item.evidence || []).length ? `<small>证据：${escapeHtml(publicEvidenceSummary(verification.evidence || item.evidence))}</small>` : ""}
      </div>
    </article>`;
  }).join("");
  trace.className = "trace-list";
  trace.innerHTML = rows || `<article class="trace-item observation-only"><div class="trace-marker"></div><div><div class="trace-title"><strong>目标已理解，初始画面已观察</strong><time>已执行 0 个动作 <span class="sr-only">physical_actions 0</span></time></div><p>正在等待当前一步确认。</p></div></article>`;
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
    <span><b>可信度</b><em>${escapeHtml(confidenceLabel(view.scene.confidence))}</em></span>
    <details class="technical-details preview-details" open data-collapsed="true">
      <summary>查看调试信息</summary>
      <span><b>任务轮次</b><em>第 ${escapeHtml(view.taskState.revision ?? "—")} 轮 · ${escapeHtml(view.currentStep.id || "—")} <span class="sr-only">r${escapeHtml(view.taskState.revision ?? "—")} · ${escapeHtml(view.currentStep.id || "—")}</span></em></span>
      <span><b>观察编号</b><em>${escapeHtml(view.executionTrace.at(-1)?.observation?.id || "—")}</em></span>
      <span><b>画面标识</b><em>${escapeHtml(view.executionTrace.at(-1)?.observation?.fingerprint || "—")}</em></span>
      <span><b>累计动作</b><em>${escapeHtml(view.physicalActions)} / ${escapeHtml(view.executionBudget?.max_physical_actions ?? "—")}</em></span>
      <span><b>观察次数</b><em>${escapeHtml(view.executionBudget?.observation_attempts ?? 0)} / ${escapeHtml(view.executionBudget?.max_observations ?? "—")}</em></span>
      <span><b>确认范围</b><em>${escapeHtml(view.scopeState.state)} · ${escapeHtml(view.scopeState.reason || "—")}</em></span>
      <span><b>停止状态</b><em>${escapeHtml(view.stopState.stopped ? `${view.stopState.status} · ${view.stopState.reason}` : "正常")}</em></span>
      <span><b>证据</b><em>${escapeHtml(publicEvidenceSummary(view.evidence))}</em></span>
    </details>`;
}

function renderAction() {
  const view = sessionView();
  const content = document.querySelector("#actionContent");
  const controls = document.querySelector("#actionControls");
  const badge = document.querySelector("#sessionBadge");
  document.querySelector("#pauseNotice").hidden = !state.paused;
  if (!view) {
    content.className = "empty-state";
    content.textContent = "系统会结合任务和当前画面，只提出一个下一步操作。";
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
  const actionMetadata = action.protocol === "qwen-same-response-action-finish-v9"
    ? `<details class="technical-details" open data-collapsed="true">
         <summary>查看调试信息</summary>
         <div class="action-metadata">
           <span>协议版本：${escapeHtml(action.protocolVersion || "—")}</span>
           <span>会话编号：${escapeHtml(view.sessionId || "—")}</span>
           <span>任务编号：${escapeHtml(action.taskId || "—")}</span>
           <span>轮次：${escapeHtml(action.revision ?? "—")}</span>
           <span>观察编号：${escapeHtml(action.observationId || "—")}</span>
           <span>画面标识：${escapeHtml(action.fingerprint || "—")}</span>
           <span class="sr-only">status action · session ${escapeHtml(view.sessionId || "—")} · task ${escapeHtml(action.taskId || "—")} · revision ${escapeHtml(action.revision ?? "—")} · observation ${escapeHtml(action.observationId || "—")} · fingerprint ${escapeHtml(action.fingerprint || "—")}</span>
         </div>
       </details>`
    : action.actionType
      ? `<details class="technical-details" open data-collapsed="true"><summary>查看调试信息</summary><div class="compatibility-note">兼容模式：${escapeHtml(action.protocol)}</div></details>`
      : `<div class="compatibility-note">正在等待下一步判断</div>`;
  content.className = "action-content";
  content.innerHTML = view.isTerminal
    ? `<h3>${escapeHtml(statusNames[view.status] || view.status)}</h3><p>${escapeHtml(view.failedReason || action.reason || "会话已经结束。")}</p>`
    : (view.status === "paused_after_action"
      ? `<h3>上一步已完成并重新观察</h3><p>网页将依据新画面决定是否发起下一次单动作请求。</p>`
      : `<div class="next-action-title"><span>${escapeHtml(action.actionType ? actionLabel(action) : decisionStatusNames[action.status] || "等待唯一动作")}</span>${staleScope ? '<b class="risk-tag">旧确认已失效</b>' : effectPhase ? '<b class="risk-tag">需要效果确认</b>' : view.status === "awaiting_confirmation" ? `<b class="${highAttention ? "risk-tag" : "safe-tag"}">需要当前动作确认</b>` : '<b class="safe-tag">受限单步</b>'}</div>
         <h3>${escapeHtml(view.currentStep.label)}</h3>
         <div class="action-target">目标：${escapeHtml(action.semanticTarget)}<span class="sr-only">element_id ${escapeHtml(action.elementId || action.element_id || "—")}</span></div>
         <div class="action-facts">
           <span><b>预期变化</b>${escapeHtml(Protocol.displayValue(action.expectedChange))}</span>
           <span><b>设备条件</b>${escapeHtml(controllerGateLabel(view.controllerGate))}</span>
         </div>
         <p>${escapeHtml(action.reason || riskSummary || "等待视觉判断生成下一步操作。")}</p>
         ${riskSummary ? `<small>当前操作需要确认。</small>` : ""}
         ${actionMetadata}
          <small>${staleScope ? "当前画面已变化，需要重新查看后再操作。" : "每次操作后都会重新查看画面。"}</small>`);

  const disabled = state.busy || state.paused ? "disabled" : "";
  if (view.isTerminal) {
    controls.innerHTML = "";
  } else if (view.status === "paused") {
    content.innerHTML = "<h3>已暂停，进度已保留</h3><p>点击上方继续任务，系统会重新查看画面。</p>";
    controls.innerHTML = `<button id="cancelSupervisedAgent" class="text-button">取消任务</button>`;
  } else if (view.status === "budget_paused") {
    content.innerHTML = `<h3>预算已用完，进度已保留</h3><p>${escapeHtml(view.autoPauseReason)} 调整任务预算后可以继续，已用量不会清零。</p>`;
    controls.innerHTML = `<button id="continueBudgetAgent" class="primary-button" ${disabled}>应用预算并继续</button>
      <button id="cancelSupervisedAgent" class="text-button" ${state.busy ? "disabled" : ""}>取消任务</button>`;
  } else if (staleScope) {
    controls.innerHTML = `
      <button id="nextSupervisedAgent" class="primary-button" ${disabled}>确认已失效 · 重新查看</button>
      <button id="cancelSupervisedAgent" class="text-button" ${state.busy ? "disabled" : ""}>取消任务</button>`;
  } else if (view.effectPolicy.requiresConfirmation || view.status === "awaiting_confirmation") {
    controls.innerHTML = `
      <button id="reviewAction" class="${effectPhase || highAttention ? "risk-button" : "primary-button"}" ${disabled}>${effectPhase ? "查看效果并确认" : "确认当前动作"}</button>
      <button id="nextSupervisedAgent" class="secondary-button" ${disabled}>放弃旧确认并重新观察</button>
      <button id="cancelSupervisedAgent" class="text-button" ${state.busy ? "disabled" : ""}>取消任务</button>`;
  } else {
    controls.innerHTML = `
      <button id="nextSupervisedAgent" class="primary-button" ${disabled}>观察并生成下一步</button>
      <button id="cancelSupervisedAgent" class="text-button" ${state.busy ? "disabled" : ""}>取消任务</button>`;
  }
  badge.className = `pill ${view.isTerminal ? (view.status === "succeeded" || view.status === "completed" ? "success" : "danger") : (effectPhase || highAttention ? "risk" : "active")}`;
  badge.textContent = view.isTerminal
    ? (statusNames[view.status] || view.status)
    : (statusNames[view.status] || view.status);
  bindActionEvents();
}

function bindActionEvents() {
  document.querySelector("#continueBudgetAgent")?.addEventListener("click", continueBudgetAgent);
  document.querySelector("#reviewAction")?.addEventListener("click", openRiskDialog);
  document.querySelector("#nextSupervisedAgent")?.addEventListener("click", nextSupervisedAgent);
  document.querySelector("#cancelSupervisedAgent")?.addEventListener("click", cancelSupervisedAgent);
}

function currentDeviceDescriptor() {
  const devices = Array.isArray(state.device?.devices) ? state.device.devices : [];
  return devices.find(item => String(item.device_id || "") === state.deviceId) || null;
}

function currentMachinePosition() {
  const value = Number(currentDeviceDescriptor()?.machine_position);
  return Number.isInteger(value) && value >= 1 && value <= 10 ? value : null;
}

function deviceLabel(device) {
  return String(device?.name || device?.label || device?.device_name || device?.device_id || "未命名设备");
}

function deviceStateLabel(device) {
  if (!device) return "未配置";
  const controller = device.controller_online ? "控制器已连接" : "控制器未连接";
  const camera = device.camera_online ? "画面可用" : "画面不可用";
  const usage = device.busy ? "使用中" : "空闲";
  return `${controller} · ${camera} · ${usage}`;
}

function renderDeviceDetails() {
  const container = document.querySelector("#deviceDetailsList");
  if (!container) return;
  const devices = Array.isArray(state.device?.devices) ? state.device.devices : [];
  const activeSession = sessionView();
  if (!devices.length) {
    container.innerHTML = '<p class="empty-state device-empty-state">当前服务没有返回可用设备。</p>';
    return;
  }
  const selectedId = String(state.deviceId || "");
  const locked = Boolean(state.busy || (activeSession && !activeSession.isTerminal));
  container.replaceChildren(...devices.map(device => {
    const id = String(device.device_id || "");
    const card = document.createElement("button");
    card.type = "button";
    card.className = `device-detail-card${id === selectedId ? " selected" : ""}`;
    card.disabled = locked || !id;
    card.setAttribute("aria-pressed", id === selectedId ? "true" : "false");
    card.innerHTML = `
      <span class="device-detail-title"><strong>${escapeHtml(deviceLabel(device))}</strong><code>${escapeHtml(id)}</code></span>
      <span class="device-detail-state">${escapeHtml(deviceStateLabel(device))}</span>
      <span class="device-detail-position">${Number.isInteger(Number(device.machine_position)) ? `${Number(device.machine_position)}号机位` : "未配置机位"}</span>`;
    card.addEventListener("click", () => selectDevice(id));
    return card;
  }));
}

function renderMachinePositions() {
  const container = document.querySelector("#machinePositionButtons");
  if (!container) return;
  const selected = currentMachinePosition();
  const view = sessionView();
  const controllerOnline = Boolean(currentDeviceDescriptor()?.controller_online ?? state.device?.controller_online);
  // Position selection targets the seller controller toolbar; camera readiness
  // is required for task execution and preview, but not for this control.
  const disabled = state.busy || Boolean(view && !view.isTerminal) || !controllerOnline;
  container.replaceChildren(...Array.from({ length: 10 }, (_, index) => {
    const position = index + 1;
    const descriptor = (Array.isArray(state.device?.devices) ? state.device.devices : [])
      .find(item => Number(item.machine_position) === position);
    const configured = Boolean(descriptor);
    const button = document.createElement("button");
    button.type = "button";
    button.className = `machine-position-button${selected === position ? " active" : ""}`;
    button.innerHTML = `
      <strong>${position}号机位</strong>
      <span>${escapeHtml(configured ? deviceLabel(descriptor) : "未配置")}</span>
      <small>${escapeHtml(configured ? deviceStateLabel(descriptor) : "空槽位")}</small>`;
    button.setAttribute("aria-label", configured
      ? `${position}号机位，${deviceLabel(descriptor)}，${deviceStateLabel(descriptor)}`
      : `${position}号机位，未配置`);
    button.setAttribute("aria-pressed", selected === position ? "true" : "false");
    button.disabled = disabled;
    button.addEventListener("click", () => selectMachinePosition(position));
    return button;
  }));
  const toolbar = document.querySelector("#taskMachinePositions");
  if (toolbar) {
    toolbar.replaceChildren(...Array.from(container.children, (button, index) => {
      const copy = button.cloneNode(true);
      copy.title = button.getAttribute("aria-label");
      copy.addEventListener("click", () => selectMachinePosition(index + 1));
      return copy;
    }));
  }
  const label = document.querySelector("#selectedMachineLabel");
  if (label) label.textContent = `${deviceLabel(currentDeviceDescriptor())} · ${state.deviceId}${selected ? ` · 当前 ${selected} 号机位` : " · 未配置机位"}`;
}

async function selectMachinePosition(position) {
  if (state.busy) return;
  state.busy = true;
  render();
  try {
    await api(`/api/device/${encodeURIComponent(state.deviceId)}/machine-position`, {
      method: "POST",
      body: JSON.stringify({ machine_position: position }),
    });
    state.device = await api("/api/device");
    toast(`已切换到${position}号机位，正在刷新摄像头画面。`);
    refreshPreview();
  } catch (error) {
    toast(`机位切换失败：${error.message}`, true);
  } finally {
    state.busy = false;
    render();
  }
}

function unverifiedCapabilityActions() {
  return currentDeviceDescriptor()?.capability_acceptance_actions || [];
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
      ? "选择一个尚未验证的通用动作。生成计划只查看画面，不会操作设备。"
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
      <strong>${escapeHtml(semanticActionNames[view.action] || view.action)} · ${escapeHtml(capabilityStatusNames[view.status] || "处理中")}</strong>
      <details class="technical-details capability-details" open data-collapsed="true">
        <summary>查看调试信息</summary>
        <div class="capability-meta">
          <span>记录编号：${escapeHtml(view.trialId)}</span>
          <span>设备编号：${escapeHtml(view.deviceId)}</span>
          <span>动作编号：${escapeHtml(view.action)}</span>
          <span>已执行动作：${escapeHtml(view.physicalActions)}</span>
          <span>报告状态：${escapeHtml(reportState)}</span>
          <span>代码版本：${escapeHtml(view.codeRevision || "—")}</span>
          <span class="sr-only">trial ${escapeHtml(view.trialId)} physical_actions ${escapeHtml(view.physicalActions)}</span>
        </div>
      </details>
      <small>${view.requiresRestart
        ? "能力配置已写入；等待安全重启后生效。"
        : report?.status === "failed"
          ? escapeHtml(report.error || "本次验收未满足通过标准，禁止晋级和重试。")
          : view.passed
      ? "检查结果已通过；仍需单独确认才能启用这项设备能力。"
      : "每次确认最多执行一个设备操作。"}</small>`;
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
  if (view?.isTerminal) state.paused = false;
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
  const acceptanceBlocksOrdinaryAgent = Boolean(
    acceptance && !acceptance.report && !acceptance.readOnlyRecovered
  );
  deviceSelect.disabled = state.busy || acceptanceBlocksOrdinaryAgent;
  document.querySelector("#startSupervisedAgent").disabled = state.busy
    || state.paused
    || acceptanceBlocksOrdinaryAgent;
  document.querySelector("#agentText").disabled = state.busy;
  document.querySelector("#pauseButton").textContent = state.paused ? "▶ 继续任务" : "Ⅱ 暂停任务";
  document.querySelector("#pauseButton").classList.toggle("active", state.paused);
  const sessionActive = Boolean(view && !view.isTerminal);
  const pauseButton = document.querySelector("#pauseButton");
  pauseButton.hidden = !sessionActive;
  pauseButton.disabled = Boolean(state.busy && (state.paused || view?.status === "paused"));
  const stopButton = document.querySelector("#stopButton");
  stopButton.hidden = !(sessionActive || state.taskAttemptStatus?.state === "running" || acceptanceBlocksOrdinaryAgent);
  stopButton.disabled = state.stopRequested;
  renderMachinePositions();
  renderDeviceDetails();
  renderTaskRunStatus();
  renderQwenConversation();
  renderStatus();
  renderGoalAndPlan();
  renderTrace();
  renderScene();
  renderAction();
  renderCapabilityAcceptance();
}

async function refreshDevice() {
  const [device, runtimeSession] = await Promise.all([
    api("/api/device"),
    api("/api/session"),
  ]);
  state.device = device;
  state.token = String(runtimeSession.token || "");
  state.mock = Boolean(runtimeSession.mock);
  await reconcileSupervisedSession();
  renderTaskRunStatus();
  renderStatus();
}

let sessionReconcileInFlight = false;

async function reconcileSupervisedSession() {
  const view = sessionView();
  if (
    !view
    || view.isTerminal
    || state.busy
    || state.taskAttemptStatus?.state === "running"
    || sessionReconcileInFlight
  ) return;

  sessionReconcileInFlight = true;
  try {
    const response = await api(`/api/agent/generic-supervised/${encodeURIComponent(view.sessionId)}`);
    if (!response?.session) return;
    state.supervisedSession = response.session;
    state.taskAttemptStatus = null;
    const refreshed = Protocol.adaptSession(response.session, { fallbackDeviceId: state.deviceId });
    state.sessionDeviceId = refreshed.deviceId || state.deviceId;
    if (refreshed.isTerminal) state.pendingConfirmationGrant = null;
  } catch (error) {
    if (Number(error?.status) !== 404) return;
    const detail = `会话 ${view.sessionId} 已不在当前服务中，已停止显示为进行中；请建立新任务。`;
    saveLastTaskOutcome({
      state: "failure",
      detail,
      sessionId: view.sessionId,
      updatedAt: new Date().toISOString(),
    });
    state.supervisedSession = null;
    state.sessionDeviceId = "";
    state.pendingConfirmationGrant = null;
  } finally {
    sessionReconcileInFlight = false;
  }
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
    renderTaskRunStatus();
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

function taskBudgetPayload() {
  const max_physical_actions = Number(document.querySelector("#agentActionBudget").value);
  const max_observations = Number(document.querySelector("#agentObservationBudget").value);
  if (![max_physical_actions, max_observations].every(value => Number.isSafeInteger(value) && value > 0)) {
    throw new Error("整任务动作和观察预算必须是正整数。");
  }
  return { max_physical_actions, max_observations };
}

async function continueBudgetAgent() {
  const view = sessionView();
  if (!view || state.busy || state.paused || view.isTerminal) return;
  state.pendingConfirmationGrant = null;
  try {
    const payload = Protocol.buildRequestPayload(lockedSessionDeviceId(), taskBudgetPayload());
    const response = await withVisionProgress("按累计预算继续任务", () =>
      api(`/api/agent/generic-supervised/${view.sessionId}/auto`, {
        method: "POST", body: JSON.stringify(payload),
      }));
    state.supervisedSession = response.session;
    await finalizeStopIfRequested();
    render();
  } catch (error) {
    if (restoreSupervisedSessionFromError(error)) render();
    toast(error.message, true);
  } finally {
    const current = sessionView();
    state.paused = Boolean(current && !current.isTerminal &&
      (current.status === "paused" || state.supervisedSession?.pause_requested));
    render();
  }
}

async function startAsyncAndWait(payload) {
  const ticket = await api("/api/agent/generic-supervised/start-async", {method: "POST", body: JSON.stringify(payload)}, 10000);
  const startedAt = Date.now();
  let nextProgressNoticeAt = startedAt + 120000;
  while (true) {
    await new Promise(resolve => setTimeout(resolve, 1000));
    const status = await api(`/api/agent/generic-supervised/start-async/${encodeURIComponent(ticket.task_id)}`);
    if (status.status === "completed") return status.result;
    if (status.status === "failed") throw new Error(status.error || "任务启动失败。");
    if (Date.now() >= nextProgressNoticeAt) {
      const elapsedSeconds = Math.floor((Date.now() - startedAt) / 1000);
      if (state.taskAttemptStatus?.state === "running" && !state.stopRequested) {
        state.taskAttemptStatus = {
          ...state.taskAttemptStatus,
          detail: `后台任务仍在执行，已等待 ${elapsedSeconds} 秒；请勿重复提交。`,
          updatedAt: new Date().toISOString(),
        };
        renderTaskRunStatus();
      }
      nextProgressNoticeAt += 30000;
    }
  }
}

async function useLuckyBagPreset() {
  try {
    const feature = await api("/api/features/lucky-bag");
    state.luckyBagProfile = feature.profile || null;
    document.querySelector("#agentText").value = String(feature.goal || "");
    const notice = document.querySelector("#luckyBagFeatureNotice");
    if (notice && feature.notice) notice.textContent = feature.notice;
    toast("已填充福袋监控目标；确认直播间已打开后可启动长期监控。");
  } catch (error) {
    toast("加载福袋功能失败：" + error.message, true);
  }
}

function renderLuckyBagMonitorStatus(monitor) {
  const element = document.querySelector("#luckyBagMonitorStatus");
  if (!element || !monitor) return;
  const labels = {
    starting: "启动中",
    running: "运行中",
    paused: "已暂停",
    waiting_confirmation: "等待确认",
    succeeded: "疑似中奖，已记录通知",
    expired: "已到时限",
    failed: "失败",
    blocked: "已阻止",
    cancelled: "已停止",
    recovery_required: "等待恢复",
  };
  element.textContent = labels[monitor.status] || monitor.status || "未知状态";
  const resume = document.querySelector("#resumeLuckyBagMonitor");
  if (resume) resume.hidden = monitor.status !== "recovery_required";
}

async function restoreLuckyBagMonitor() {
  try {
    const response = await api("/api/features/lucky-bag/monitors");
    const items = Array.isArray(response.monitors) ? response.monitors : [];
    const active = items.find(item => [
      "starting", "running", "paused", "waiting_confirmation", "recovery_required",
    ].includes(item.monitor?.status));
    if (!active) return;
    state.luckyBagMonitorId = String(active.monitor?.monitor_id || "");
    state.luckyBagProfile = active.profile || null;
    renderLuckyBagMonitorStatus(active.monitor);
    clearInterval(state.luckyBagMonitorTimer);
    state.luckyBagMonitorTimer = setInterval(pollLuckyBagMonitor, 5000);
  } catch (_error) {
    // The ordinary Agent page remains usable when the optional feature is unavailable.
  }
}

async function resumeLuckyBagMonitor() {
  if (!state.luckyBagMonitorId || !state.luckyBagProfile) return;
  try {
    const response = await api(
      "/api/features/lucky-bag/" + encodeURIComponent(state.luckyBagMonitorId) + "/resume",
      {
        method: "POST",
        body: JSON.stringify({ device_id: state.luckyBagProfile.device_id }),
      },
    );
    renderLuckyBagMonitorStatus(response.monitor);
    clearInterval(state.luckyBagMonitorTimer);
    state.luckyBagMonitorTimer = setInterval(pollLuckyBagMonitor, 5000);
    toast("监控已恢复，将重新观察当前 Android 画面。");
  } catch (error) {
    toast("恢复福袋监控失败：" + error.message, true);
  }
}

async function pollLuckyBagMonitor() {
  if (!state.luckyBagMonitorId) return;
  try {
    const response = await api("/api/features/lucky-bag/" + encodeURIComponent(state.luckyBagMonitorId));
    renderLuckyBagMonitorStatus(response.monitor);
    if (["succeeded", "expired", "failed", "blocked", "cancelled"].includes(response.monitor?.status)) {
      clearInterval(state.luckyBagMonitorTimer);
      state.luckyBagMonitorTimer = null;
    }
  } catch (error) {
    renderLuckyBagMonitorStatus({ status: "状态读取失败：" + error.message });
  }
}

async function startLuckyBagMonitor() {
  try {
    if (!state.luckyBagProfile) {
      const feature = await api("/api/features/lucky-bag");
      state.luckyBagProfile = feature.profile || null;
      document.querySelector("#agentText").value = String(feature.goal || "");
    }
    const response = await api("/api/features/lucky-bag/start", {
      method: "POST",
      body: JSON.stringify(state.luckyBagProfile || {}),
    });
    state.luckyBagMonitorId = String(response.monitor?.monitor_id || "");
    renderLuckyBagMonitorStatus(response.monitor);
    clearInterval(state.luckyBagMonitorTimer);
    state.luckyBagMonitorTimer = setInterval(pollLuckyBagMonitor, 5000);
    toast("长期监控已启动；请保持用户手动打开的直播间。");
  } catch (error) {
    toast("启动福袋监控失败：" + error.message, true);
  }
}
async function startSupervisedAgent() {
  const text = document.querySelector("#agentText").value.trim();
  const current = sessionView();
  if (!text) return toast("请先输入希望手机完成的目标。", true);
  if (current && !current.isTerminal) return toast("已有进行中的会话，请继续或停止后再创建新目标。", true);
  state.sessionDeviceId = state.deviceId;
  state.pendingConfirmationGrant = null;
  state.supervisedSession = null;
  clearLastTaskOutcome();
  state.taskAttemptStatus = {
    state: "running",
    detail: "正在理解目标并观察当前画面。",
    updatedAt: new Date().toISOString(),
  };
  try {
    const payload = Protocol.buildRequestPayload(state.sessionDeviceId, { text, ...taskBudgetPayload() });
    const response = await withVisionProgress("理解目标并观察当前画面", () =>
      startAsyncAndWait(payload)
    );
    state.taskAttemptStatus = null;
    state.supervisedSession = response.session;
    await finalizeStopIfRequested();
    const actions = Number(response.physical_actions || 0);
    toast(actions
      ? `任务已完成 ${actions} 次操作，并在每次操作后重新查看画面。`
      : "计划与只读观察已完成；当前没有需要执行的操作。");
    render();
  } catch (error) {
    if (!restoreSupervisedSessionFromError(error)) {
      const failure = saveLastTaskOutcome({
        state: "failure",
        detail: error.message || "任务未能启动。",
        sessionId: "",
        updatedAt: new Date().toISOString(),
      });
      state.taskAttemptStatus = failure;
    }
    render();
    toast(error.message, true);
  }
}

async function startCapabilityTrial() {
  const action = document.querySelector("#capabilityAction").value;
  const text = document.querySelector("#capabilityGoal").value.trim();
  const current = sessionView();
  if (!action) return toast("当前设备没有可选择的待验收动作。", true);
  if (!text) return toast("请填写一个通用、可见且安全的设备检查目标。", true);
  if (current && !current.isTerminal) return toast("当前设备已有进行中的任务。", true);
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
    toast("设备检查计划已生成，请核对本次操作范围。")
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
    ? "确认本次验收动作的效果范围"
    : "确认执行本次设备操作";
  const level = document.querySelector("#riskLevel");
  level.className = "risk-level high";
  level.textContent = effectPhase ? "此确认只查看画面" : "设备操作 · 最多执行一次";
  document.querySelector("#riskGoal").textContent = view.text || session?.objective || "—";
  document.querySelector("#riskAction").innerHTML = `${escapeHtml(semanticActionNames[view.action] || view.action)}<span class="sr-only"> · trial=${escapeHtml(view.trialId)}</span>`;
  document.querySelector("#riskReason").textContent = session?.visualAction?.reason || "依据当前画面提出本次操作。";
  document.querySelector("#riskExpected").textContent = effectPhase
    ? "生成一个与候选动作完全一致的视觉动作，不触发机械臂"
    : Protocol.displayValue(session?.visualAction?.expectedChange);
  document.querySelector("#riskDevice").textContent = view.deviceId;
  const scope = state.pendingConfirmationGrant.scope;
  document.querySelector("#riskWarning").innerHTML = effectPhase
    ? `确认后只查看本次操作结果，不会操作设备。<span class="sr-only">trial=${escapeHtml(view.trialId)}、action=${escapeHtml(view.action)}、session=${escapeHtml(scope.session_id)}、task=${escapeHtml(scope.task_id)}、revision=${escapeHtml(scope.revision)}、step=${escapeHtml(scope.step_id)}、effect_ids=${escapeHtml(scope.effect_ids.join(",") || "—")} physical_actions 0</span>`
    : `只授权当前画面对应的一次设备操作，失败不会自动重试。<span class="sr-only">trial=${escapeHtml(view.trialId)}、action=${escapeHtml(view.action)}、session=${escapeHtml(scope.session_id)}、task=${escapeHtml(scope.task_id)}、revision=${escapeHtml(scope.revision)}、step=${escapeHtml(scope.step_id)}、observation_id=${escapeHtml(scope.observation_id)}、fingerprint=${escapeHtml(scope.fingerprint)}</span>`;
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
    toast("设备检查已取消，未执行后续动作。")
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
    : `${view.currentStep.label} · 等待视觉判断`;
  const technicalReason = view.effectPolicy.currentActions.map(item => `${item.id} [${item.policyLevel}]：${item.kind}；${item.expectedResults.join("、")}`).join("\n");
  document.querySelector("#riskReason").innerHTML = `${escapeHtml(view.visualAction.reason || "请确认这一步操作。")}<span class="sr-only">${escapeHtml(technicalReason)}</span>`;
  const preview = view.effectPolicy.intentPreview;
  if (preview && preview.action) {
    document.querySelector("#riskReason").textContent = [
      "当前待确认动作：" + preview.action,
      "效果：" + view.effectPolicy.effectIds.join("、"),
      "目标：" + Protocol.displayValue(preview.params?.label || preview.params?.target),
      "依据：" + view.visualAction.reason,
    ].join("\n");
  }
  document.querySelector("#riskExpected").textContent = Protocol.displayValue(view.visualAction.expectedChange);
  document.querySelector("#riskDevice").textContent = lockedSessionDeviceId();
  document.querySelector("#riskWarning").textContent = effectPhase
    ? "确认后只查看本次操作结果。"
    : highAttention
    ? "这一步会改变外部状态，只会执行一次设备操作，请确认后再继续。"
    : "只授权当前这一步，操作后会重新查看画面。";
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
    if (restoreSupervisedSessionFromError(error)) render();
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
    if (restoreSupervisedSessionFromError(error)) render();
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
    if (restoreSupervisedSessionFromError(error)) render();
    if (!quiet) toast(error.message, true);
  }
}

async function finalizeStopIfRequested() {
  if (!state.stopRequested) return;
  await cancelSupervisedAgent({ quiet: true });
  state.stopRequested = false;
}

async function togglePause() {
  if (state.paused || sessionView()?.status === "paused") {
    if (state.busy) return;
    state.paused = false;
    await continueBudgetAgent();
    return;
  }
  state.paused = true;
  {
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
  toast("已请求暂停；正在执行的动作完成并记录结果后停止，恢复时重新观察。");
  render();
}

async function stopTasks() {
  const requestWasRunning = state.busy;
  state.paused = false;
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
    state.taskAttemptStatus = null;
    const restored = Protocol.adaptSession(response.session, { fallbackDeviceId: state.deviceId });
    state.sessionDeviceId = restored.deviceId || state.deviceId;
    state.paused = restored.status === "paused" || Boolean(response.session?.pause_requested);
    if (restored.executionBudget?.max_physical_actions) {
      document.querySelector("#agentActionBudget").value = restored.executionBudget.max_physical_actions;
      document.querySelector("#agentObservationBudget").value = restored.executionBudget.max_observations;
    }
    state.pendingConfirmationGrant = null;
  } catch (_error) {
    state.supervisedSession = null;
  }
}

async function restoreCapabilityTrial() {
  try {
    const response = await api("/api/capability-acceptance");
    const trials = Array.isArray(response.trials) ? response.trials : [];
    const matching = trials.filter(item => {
      const view = Protocol.adaptCapabilityTrial(item);
      return view.deviceId === state.deviceId && !view.readOnlyRecovered;
    });
    if (!matching.length) return;
    state.capabilityTrial = matching[matching.length - 1];
    state.capabilityDeviceId = state.deviceId;
    await loadCapabilityEvidence().catch(() => {});
  } catch (_error) {
    state.capabilityTrial = null;
  }
}

function prepareNewTask() {
  const current = sessionView();
  if (current && !current.isTerminal) {
    toast("当前任务仍在运行，请先暂停或取消。", true);
    return;
  }
  clearLastTaskOutcome();
  state.taskAttemptStatus = null;
  const input = document.querySelector("#agentText");
  if (input) {
    input.value = "";
    input.focus();
  }
  render();
  toast("已准备新任务，请输入任务指令。 ");
}

function focusDeviceConnection() {
  showDeviceDetails();
  const connected = Boolean(state.device?.controller_online && state.device?.camera_online);
  toast(connected ? "设备状态正常，可以输入任务。" : "当前设备未连接，请查看设备详情。", !connected);
}

function togglePreviewDrawer() {
  const panel = document.querySelector("#taskWorkspace .preview-panel");
  if (!panel) return;
  panel.classList.toggle("drawer-open");
  panel.setAttribute("aria-hidden", "false");
  document.querySelector("#openPreviewButton")?.setAttribute("aria-expanded", String(panel.classList.contains("drawer-open")));
}

function setDeviceDetailsOpen(open) {
  const panel = document.querySelector("#deviceOverview");
  if (!panel) return;
  panel.classList.toggle("details-open", Boolean(open));
  panel.setAttribute("aria-hidden", open ? "false" : "true");
}

function openDeviceDetails() {
  const panel = document.querySelector("#deviceOverview");
  setDeviceDetailsOpen(!panel?.classList.contains("details-open"));
}

function showDeviceDetails() {
  setDeviceDetailsOpen(true);
}

function closeDeviceDetails() {
  setDeviceDetailsOpen(false);
}

function openTaskDetails() {
  const plan = document.querySelector("#taskPlanDetails");
  if (!plan) return;
  plan.dataset.collapsed = "false";
  plan.open = true;
  plan.scrollIntoView({ behavior: "smooth", block: "start" });
}

function openExecutionRecords() {
  const records = document.querySelector("#executionRecords");
  if (!records) return;
  records.dataset.collapsed = "false";
  records.open = true;
  records.scrollIntoView({ behavior: "smooth", block: "start" });
}

function fitConversationViewport() {
  const card = document.querySelector("#taskWorkspace > .command-card");
  const viewport = window.visualViewport;
  const available = (viewport?.height || window.innerHeight)
    - Math.max(0, card.getBoundingClientRect().top - (viewport?.offsetTop || 0)) - 12;
  card.style.setProperty("--conversation-height", `${Math.max(220, available)}px`);
}
window.addEventListener("resize", fitConversationViewport);
window.visualViewport?.addEventListener("resize", fitConversationViewport);
new ResizeObserver(fitConversationViewport).observe(document.querySelector(".topbar"));
new ResizeObserver(fitConversationViewport).observe(document.querySelector(".machine-position-toolbar"));

async function init() {
  try {
    state.lastTaskOutcome = readLastTaskOutcome(state.deviceId);
    const session = await api("/api/session");
    state.token = session.token;
    state.mock = session.mock;
    const mode = document.querySelector("#modeBadge");
    mode.textContent = session.mock ? "演示模式" : "已连接";
    mode.classList.toggle("live-mode", !session.mock);
    await refreshDevice();
    await restoreActiveSession();
    await restoreLuckyBagMonitor();
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
document.querySelector("#newTaskButton")?.addEventListener("click", prepareNewTask);
document.querySelector("#newTaskButtonInline")?.addEventListener("click", prepareNewTask);
document.querySelector("#connectDeviceButton")?.addEventListener("click", focusDeviceConnection);
document.querySelector("#openPreviewButton")?.addEventListener("click", togglePreviewDrawer);
document.querySelector("#closePreviewButton")?.addEventListener("click", togglePreviewDrawer);
document.querySelector("#openTaskDetailsButton")?.addEventListener("click", openTaskDetails);
document.querySelector("#openExecutionRecordsButton")?.addEventListener("click", openExecutionRecords);
document.querySelector("#openDeviceDetailsButton")?.addEventListener("click", openDeviceDetails);
document.querySelector("#toolbarDeviceButton")?.addEventListener("click", showDeviceDetails);
document.querySelector("#closeDeviceDetailsButton")?.addEventListener("click", closeDeviceDetails);
document.querySelector("#deviceNavItem")?.addEventListener("click", event => {
  event.preventDefault();
  showDeviceDetails();
});
document.addEventListener("click", event => {
  const summary = event.target.closest?.("details[data-collapsed] > summary");
  if (!summary) return;
  event.preventDefault();
  const details = summary.parentElement;
  details.open = true;
  details.dataset.collapsed = details.dataset.collapsed === "true" ? "false" : "true";
});
document.querySelector("#useLuckyBagPreset")?.addEventListener("click", useLuckyBagPreset);
document.querySelector("#startLuckyBagMonitor")?.addEventListener("click", startLuckyBagMonitor);
document.querySelector("#resumeLuckyBagMonitor")?.addEventListener("click", resumeLuckyBagMonitor);
document.querySelector("#startCapabilityTrial").addEventListener("click", startCapabilityTrial);
document.querySelector("#pauseButton").addEventListener("click", togglePause);
document.querySelector("#stopButton").addEventListener("click", stopTasks);
async function selectDevice(requestedDeviceId) {
  const deviceSelect = document.querySelector("#deviceId");
  const normalizedDeviceId = String(requestedDeviceId || "");
  const registeredDeviceIds = Array.isArray(state.device?.devices)
    ? state.device.devices.map(item => String(item.device_id || "")).filter(Boolean)
    : [];
  if (!registeredDeviceIds.includes(normalizedDeviceId)) {
    if (deviceSelect) deviceSelect.value = lockedSessionDeviceId();
    render();
    return;
  }
  state.deviceId = normalizedDeviceId;
  localStorage.setItem("visual-agent-device-id", state.deviceId);
  state.supervisedSession = null;
  state.sessionDeviceId = "";
  state.pendingConfirmationGrant = null;
  state.taskAttemptStatus = null;
  state.lastTaskOutcome = readLastTaskOutcome(state.deviceId);
  state.capabilityTrial = null;
  state.capabilityDeviceId = "";
  state.capabilityEvidence = [];
  render();
  await restoreActiveSession();
  if (!state.supervisedSession) await restoreCapabilityTrial();
  render();
  refreshPreview();
}
document.querySelector("#deviceId").addEventListener("change", async event => {
  await selectDevice(event.target.value);
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
