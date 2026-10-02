const Protocol = window.UniversalAgentProtocol;
if (!Protocol) throw new Error("通用 Agent 前端协议适配层未加载。");

const state = {
  token: "",
  mock: false,
  device: {},
  deviceId: localStorage.getItem("visual-agent-device-id") || "device-local-01",
  sessionDeviceId: "",
  chatConversation: [],
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
  pendingStartTicket: null,
  lastTaskOutcome: null,
  luckyBagProfile: null,
  luckyBagMonitorId: "",
  luckyBagMonitorTimer: null,
  luckyBagMonitorStatus: "",
  luckyBagMonitorDetail: "",
};

const taskOutcomeStoragePrefix = "visual-agent-task-outcome:";
const pageExitPauseMarkerKey = "visual-agent-page-exit-pause";
let pageExitPauseSent = false;

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
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = data.detail;
      const parsedDetail = parseStructuredError(detail);
      const message = typeof parsedDetail === "string"
        ? parsedDetail
        : (parsedDetail?.error || parsedDetail?.message || JSON.stringify(parsedDetail || {}));
      const error = new Error(message || `请求失败（${response.status}）`);
      error.detail = parsedDetail;
      error.status = response.status;
      error.code = parsedDetail && typeof parsedDetail === "object"
        ? String(parsedDetail.error_code || parsedDetail.code || `HTTP_${response.status}`)
        : `HTTP_${response.status}`;
      error.phase = parsedDetail && typeof parsedDetail === "object"
        ? String(parsedDetail.phase || "request") : "request";
      error.recoverable = parsedDetail && typeof parsedDetail === "object"
        && parsedDetail.recoverable !== undefined ? Boolean(parsedDetail.recoverable) : response.status >= 500 || response.status === 409;
      throw error;
    }
    return data;
  } catch (error) {
    if (error?.name === "AbortError") {
      const aborted = new Error("请求已取消。" );
      aborted.name = "AbortError";
      aborted.code = "REQUEST_ABORTED";
      aborted.phase = "request";
      aborted.recoverable = true;
      throw aborted;
    }
    throw error;
  } finally {
    if (timer) clearTimeout(timer);
  }
}

function pagePauseRequest(path, body) {
  if (!state.token) return;
  void fetch(path, {
    method: "POST",
    keepalive: true,
    headers: {
      "X-Control-Token": state.token,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  }).catch(() => {});
}

function activePagePauseSnapshot() {
  const view = sessionView();
  const generic = view && !view.isTerminal
    ? { sessionId: String(view.sessionId || ""), deviceId: String(view.deviceId || state.deviceId) }
    : null;
  const monitorStatus = String(state.luckyBagMonitorStatus || "");
  const monitorActive = Boolean(
    state.luckyBagMonitorId
    && state.luckyBagProfile?.device_id
    && ["starting", "running", "waiting_confirmation", "paused", "recovery_required"].includes(monitorStatus)
  );
  if (!generic?.sessionId && !monitorActive) return null;
  return {
    sessionId: generic?.sessionId || "",
    deviceId: generic?.deviceId || state.deviceId,
    monitorId: monitorActive ? state.luckyBagMonitorId : "",
    monitorDeviceId: monitorActive ? String(state.luckyBagProfile.device_id) : "",
  };
}

function pauseActiveTasksForPageExit() {
  if (pageExitPauseSent) return;
  const snapshot = activePagePauseSnapshot();
  if (!snapshot) return;
  pageExitPauseSent = true;
  try {
    sessionStorage.setItem(pageExitPauseMarkerKey, JSON.stringify(snapshot));
  } catch (_error) {
    // Keepalive requests remain the best available unload path.
  }
  if (snapshot.sessionId) {
    pagePauseRequest(
      "/api/agent/generic-supervised/" + encodeURIComponent(snapshot.sessionId) + "/pause",
      { device_id: snapshot.deviceId },
    );
  }
  if (snapshot.monitorId) {
    pagePauseRequest(
      "/api/features/lucky-bag/" + encodeURIComponent(snapshot.monitorId) + "/pause",
      { device_id: snapshot.monitorDeviceId },
    );
  }
}

async function settlePageExitPause() {
  let snapshot = null;
  try {
    snapshot = JSON.parse(sessionStorage.getItem(pageExitPauseMarkerKey) || "null");
  } catch (_error) {
    snapshot = null;
  }
  if (!snapshot || typeof snapshot !== "object") return;
  let settled = true;
  if (snapshot.sessionId) {
    try {
      await api(
        "/api/agent/generic-supervised/" + encodeURIComponent(snapshot.sessionId) + "/pause",
        { method: "POST", body: JSON.stringify({ device_id: snapshot.deviceId || state.deviceId }) },
      );
    } catch (error) {
      if (Number(error?.status) !== 404) settled = false;
    }
  }
  if (snapshot.monitorId) {
    try {
      await api(
        "/api/features/lucky-bag/" + encodeURIComponent(snapshot.monitorId) + "/pause",
        { method: "POST", body: JSON.stringify({ device_id: snapshot.monitorDeviceId || state.deviceId }) },
      );
    } catch (error) {
      if (Number(error?.status) !== 404) settled = false;
    }
  }
  if (settled) {
    try { sessionStorage.removeItem(pageExitPauseMarkerKey); } catch (_error) {}
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


function conversationMessageHtml(role, text, meta = "", variant = "") {
  const normalized = String(text || "").trim();
  if (!normalized) return "";
  const isUser = role === "user";
  return `<article class="conversation-message ${isUser ? "conversation-user" : "conversation-qwen"} ${escapeHtml(variant)}">
    ${isUser ? "" : '<div class="conversation-avatar">Q</div>'}
    <div class="conversation-bubble">
      <div class="conversation-label">${isUser ? "你" : "Qwen"}</div>
      <p>${escapeHtml(normalized)}</p>
      ${meta ? `<small>${escapeHtml(meta)}</small>` : ""}
    </div>
  </article>`;
}

function cleanConversationForQwen(value) {
  if (!Array.isArray(value)) return [];
  return value.filter(item => {
    if (!item || !["user", "assistant"].includes(item.role)) return false;
    if (item.role !== "assistant") return true;
    const content = String(item.content || "").trim();
    return content && !/^发送(?:给 Qwen)?失败[：:]/.test(content);
  }).map(item => ({ role: item.role, content: String(item.content || "") }));
}

function renderConversation() {
  const container = document.querySelector("#chatMessages");
  if (!container) return;
  const view = sessionView();
  const raw = view?.raw || state.supervisedSession || {};
  const messages = [];
  const conversation = cleanConversationForQwen(view
    ? (Array.isArray(raw.conversation) ? raw.conversation : [])
    : state.chatConversation);

  // The chat contains only the user message and Qwen's natural-language reply.
  // decision.reason, action receipts and scene summaries belong in the task
  // status panel, never as extra assistant bubbles.
  if (!conversation.length) {
    const fallbackGoal = String(raw.raw_goal || raw.goal?.objective || "").trim();
    if (fallbackGoal) messages.push(conversationMessageHtml("user", fallbackGoal));
  }
  conversation.forEach((item) => {
    if (item?.role === "user") messages.push(conversationMessageHtml("user", item.content));
    if (item?.role === "assistant") messages.push(conversationMessageHtml("qwen", item.content));
  });

  if (!messages.length) {
    container.innerHTML = "";
    return;
  }
  container.innerHTML = messages.join("");
  container.scrollTop = container.scrollHeight;
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
      code: String(parsed.code || ""),
      phase: String(parsed.phase || ""),
      recoverable: parsed.recoverable !== false,
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
    code: String(outcome.code || ""),
    phase: String(outcome.phase || ""),
    recoverable: outcome.recoverable !== false,
  };
  state.lastTaskOutcome = normalized;
  try {
    sessionStorage.setItem(taskOutcomeStorageKey(state.deviceId), JSON.stringify(normalized));
  } catch (_error) {
    // The live page still keeps the result in memory when browser storage is unavailable.
  }
  return normalized;
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
  const luckyStatus = String(state.luckyBagMonitorStatus || "");
  if (["starting", "running", "waiting_confirmation", "paused", "recovery_required"].includes(luckyStatus)) {
    const luckyLabels = {
      starting: "福袋监控启动中",
      running: "福袋监控进行中",
      waiting_confirmation: "福袋监控等待确认",
      paused: "福袋监控已暂停",
      recovery_required: "福袋监控等待恢复",
    };
    return {
      state: luckyStatus === "paused" || luckyStatus === "recovery_required" ? "paused" : "running",
      label: luckyLabels[luckyStatus],
      detail: state.luckyBagMonitorDetail || (luckyStatus === "paused"
        ? "福袋监控已暂停，不会继续观察或操作手机。点击“继续任务”后恢复。"
        : luckyStatus === "recovery_required"
          ? "服务重载后需要你点击“继续任务”，才会重新观察当前直播间。"
          : "福袋监控正在使用当前手机画面判断是否有福袋，由福袋专用流程决定下一步。"),
      sessionId: state.luckyBagMonitorId || "监控会话启动中",
      updatedAt: new Date().toISOString(),
    };
  }

  if (state.taskAttemptStatus?.state === "running") {
    return {
      state: "running",
      label: "进行中",
      detail: state.taskAttemptStatus.detail || state.visionStage || "正在理解目标并观察当前画面。",
      sessionId: "",
      updatedAt: state.taskAttemptStatus.updatedAt,
      phase: state.taskAttemptStatus.phase || "",
      code: state.taskAttemptStatus.code || "",
      recoverable: state.taskAttemptStatus.recoverable,
    };
  }
  if (state.taskAttemptStatus?.state === "failure") {
    return {
      state: "failure",
      label: "失败",
      detail: taskErrorDetail(state.taskAttemptStatus),
      sessionId: "",
      updatedAt: state.taskAttemptStatus.updatedAt,
      phase: state.taskAttemptStatus.phase || "",
      code: state.taskAttemptStatus.code || "TASK_FAILED",
      recoverable: state.taskAttemptStatus.recoverable !== false,
    };
  }
  if (state.taskAttemptStatus?.state === "cancelled") {
    return {
      state: "failure",
      label: "已取消",
      detail: state.taskAttemptStatus.detail || "已停止等待启动结果；后台任务是否已停止需要服务端状态确认。",
      sessionId: "",
      updatedAt: state.taskAttemptStatus.updatedAt,
      phase: state.taskAttemptStatus.phase || "start_ticket",
      code: state.taskAttemptStatus.code || "START_TICKET_CANCELLED",
      recoverable: true,
    };
  }
  if (state.taskAttemptStatus?.state === "success") {
    return {
      state: "success",
      label: "已完成",
      detail: state.taskAttemptStatus.detail,
      sessionId: "",
      updatedAt: state.taskAttemptStatus.updatedAt,
    };
  }

  const view = sessionView();

  if (view) {
    if (["succeeded", "completed"].includes(view.status)) {
      const detail = `目标已完成；共执行 ${view.physicalActions} 个物理动作。`;
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
    const decision = view.raw?.qwen_decision || {};
    const nextAction = decision.next_action || {};
    if (nextAction.action === "wait_for_change") {
      const seconds = nextAction.params?.wait_seconds;
      const waitLabel = seconds == null
        ? "等待下一次画面"
        : "等待下一次画面（" + Number(seconds) + " 秒）";
      return {
        state: "running",
        label: "等待画面变化",
        detail: decision.reason ? waitLabel + "：" + decision.reason : waitLabel,
        sessionId: view.sessionId,
        updatedAt: String(view.raw?.updated_at || view.raw?.created_at || ""),
      };
    }
    if (view.status === "observing") {
      return {
        state: "running",
        label: "观察中",
        detail: decision.reason || "正在获取新的 Android 画面并等待 Qwen 判断。",
        sessionId: view.sessionId,
        updatedAt: String(view.raw?.updated_at || view.raw?.created_at || ""),
      };
    }
    return {
      state: "running",
      label: "进行中",
      detail: step
        ? "当前步骤：" + step + "（" + (statusNames[view.status] || view.status) + "）"
        : (statusNames[view.status] || "任务正在处理。"),
      sessionId: view.sessionId,
      updatedAt: String(view.raw?.updated_at || view.raw?.created_at || ""),
    };
  }

  if (state.lastTaskOutcome) {
    return {
      ...state.lastTaskOutcome,
      label: state.lastTaskOutcome.state === "success" ? "成功" : "失败",
      detail: `最近一次任务：${state.lastTaskOutcome.detail}`,
      phase: state.lastTaskOutcome.phase || "",
      code: state.lastTaskOutcome.code || "",
      recoverable: state.lastTaskOutcome.recoverable,
    };
  }
  return {
    state: "not-started",
    label: "当前没有任务",
    detail: "当前没有正在控制手机的任务。发送消息后，Qwen 会自行判断是否需要读取画面或执行动作。",
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
  const phase = document.querySelector("#taskRunPhase");
  const code = document.querySelector("#taskRunErrorCode");
  const recoverability = document.querySelector("#taskRunRecoverability");
  if (phase) phase.textContent = presentation.phase || "—";
  if (code) code.textContent = presentation.code || "—";
  if (recoverability) recoverability.textContent = presentation.code
    ? (presentation.recoverable === false ? "需重新发送" : "可恢复/可重试")
    : "—";
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
  const device = state.device || {};
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
    ? (device.busy ? "当前动作执行中" : "在线且空闲")
    : "离线";
  document.querySelector("#cameraText").textContent = device.camera_online ? "实时画面可用" : "画面不可用";
  document.querySelector("#agentTextStatus").textContent = state.paused ? "人工暂停" : taskStatus.label;
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
    goalElement.textContent = "输入目标后，这里会展示整任务、当前动作和实际执行历史。";
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
      ${view.targetApps.map(app => `<span>目标应用 · ${escapeHtml(app.name)}${app.id ? ` (${escapeHtml(app.id)})` : ""}</span>`).join("")}
      ${!view.targetApps.length && view.appName ? `<span>目标应用 · ${escapeHtml(view.appName)}</span>` : ""}
      ${view.constraints.map(item => `<span>限制 · ${escapeHtml(item)}</span>`).join("")}
      ${view.completionConditions.map(item => `<span>完成 · ${escapeHtml(item)}</span>`).join("")}
      <span>确认门 · ${escapeHtml(view.effectPolicy.confirmationGate.state)} · required=${view.effectPolicy.confirmationGate.required ? "true" : "false"}</span>
      <span>执行类型 · ${escapeHtml(view.effectPolicy.currentExecutionClass)}</span>
      <span>本地策略 · ${escapeHtml(controllerGateLabel(view.controllerGate))}</span>
      <span>确认作用域 · ${escapeHtml(view.scopeState.state)} · ${escapeHtml(view.scopeState.reason || "—")}</span>
      ${view.effectPolicy.actions.map(item => `<span>效果 ${escapeHtml(item.id)} · ${escapeHtml(item.kind)}</span>`).join("")}
    </div>`;

  const steps = view.steps.map(item => planStepHtml({
    number: item.index,
    label: item.label,
    detail: item.reason,
    stateName: planState(item, view),
  })).join("");
  const dynamicTail = view.isTerminal ? "" : planStepHtml({
    number: "…",
    label: "后续步骤等待新画面",
    detail: "Qwen 结合整任务、实际动作历史和新画面选择下一步",
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
        <div class="trace-title"><strong>步骤 ${escapeHtml(item.stepNumber)} · revision ${escapeHtml(item.taskRevision ?? "—")}</strong><time>physical_actions ${escapeHtml(item.physicalActions)}</time></div>
        <p><b>任务上下文</b> ${escapeHtml(item.taskContext || "—")}${item.stepId ? ` · ${escapeHtml(item.stepId)}` : ""}</p>
        <div class="trace-grid">
          <span><b>动作前观察</b>${escapeHtml(observationText)}</span>
          <span><b>Qwen 唯一动作</b>${escapeHtml(item.action?.status || "unknown")} · ${escapeHtml(actionLabel(item.action))} · ${escapeHtml(item.action?.semanticTarget || "—")}</span>
          <span><b>Controller gate</b>${escapeHtml(gateText)}${gate.reason ? ` · ${escapeHtml(gate.reason)}` : ""}</span>
          <span><b>动作后验证</b>${escapeHtml(outcomeNames[verification.outcome] || verification.outcome || "旧记录未提供")} · ${escapeHtml(afterObservationText)}</span>
          <span><b>任务图去向</b>${escapeHtml(transitionLabel)}${transition.trigger ? ` · ${escapeHtml(transition.trigger)}` : ""}${transition.toRevision !== null && transition.toRevision !== undefined ? ` · r${escapeHtml(transition.fromRevision ?? "—")}→r${escapeHtml(transition.toRevision)}` : ""}</span>
          <span class="scope-state ${escapeHtml(scopeState.state)}"><b>确认作用域</b>${escapeHtml(scopeNames[scopeState.state] || scopeState.state)}${scopeState.reason ? ` · ${escapeHtml(scopeState.reason)}` : ""}</span>
        </div>
        ${(verification.errors || []).length ? `<small>验证/阻止原因：${escapeHtml(verification.errors.join("；"))}</small>` : ""}
        ${transition.reason ? `<small>下一截图原因：${escapeHtml(transition.reason)}</small>` : ""}
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
    <span><b>任务 / 当前轮次</b><em>r${escapeHtml(view.taskState.revision ?? "—")} · ${escapeHtml(view.currentStep.id || "—")}</em></span>
    <span><b>observation_id</b><em>${escapeHtml(view.executionTrace.at(-1)?.observation?.id || "—")}</em></span>
    <span><b>fingerprint</b><em>${escapeHtml(view.executionTrace.at(-1)?.observation?.fingerprint || "—")}</em></span>
    <span><b>累计动作</b><em>${escapeHtml(view.physicalActions)} / ${escapeHtml(view.executionBudget?.max_physical_actions ?? "—")}</em></span>
    <span><b>观察申请</b><em>${escapeHtml(view.executionBudget?.observation_attempts ?? 0)} / ${escapeHtml(view.executionBudget?.max_observations ?? "—")}</em></span>
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
  const rawDecision = view.raw?.qwen_decision || {};
  const rawNextAction = rawDecision.next_action || {};
  const waitingForChange = rawNextAction.action === "wait_for_change";
  const publicTitle = waitingForChange
    ? "等待画面变化"
    : (action.actionType ? actionLabel(action) : decisionStatusNames[action.status] || "等待 Qwen 判断");
  const publicReason = rawDecision.reason || action.reason || riskSummary || "正在根据当前画面判断下一步。";
  const actionMetadata = action.protocol === "qwen-same-response-action-finish-v9"
    ? '<div class="technical-line">'
      + "Qwen 已返回当前一步"
      + " · status " + escapeHtml(action.status)
      + " · session " + escapeHtml(view.sessionId || "—")
      + " · revision " + escapeHtml(action.revision ?? "—")
      + " · observation " + escapeHtml(action.observationId || "—")
      + " · fingerprint " + escapeHtml(action.fingerprint || "—")
      + "</div>"
    : '<div class="technical-line">Qwen 唯一动作尚未产生</div>';
  const technicalDetails = '<details class="technical-details">'
    + "<summary>查看技术详情</summary>"
    + '<div class="technical-grid">'
    + "<span>完整任务目标</span><p>" + escapeHtml(view.objective || view.currentStep?.label || "—") + "</p>"
    + "<span>语义目标</span><p>" + escapeHtml(action.semanticTarget || "—") + "</p>"
    + "<span>预期变化</span><p>" + escapeHtml(Protocol.displayValue(action.expectedChange)) + "</p>"
    + "<span>本地策略</span><p>" + escapeHtml(controllerGateLabel(view.controllerGate)) + "</p>"
    + "</div>" + actionMetadata + "</details>";
  content.className = "action-content";
  if (view.isTerminal) {
    content.innerHTML = "<h3>" + escapeHtml(statusNames[view.status] || view.status)
      + "</h3><p>" + escapeHtml(view.failedReason || action.reason || "会话已经结束。") + "</p>";
  } else if (view.status === "paused_after_action") {
    content.innerHTML = "<h3>上一步已完成并重新观察</h3><p>网页将依据新画面决定是否发起下一次单动作请求。</p>" + technicalDetails;
  } else {
    content.innerHTML = '<div class="next-action-title"><span>' + escapeHtml(publicTitle)
      + "</span>" + (staleScope ? '<b class="risk-tag">旧确认已失效</b>' : effectPhase ? '<b class="risk-tag">需要效果确认</b>' : view.status === "awaiting_confirmation" ? '<b class="' + (highAttention ? "risk-tag" : "safe-tag") + '">需要当前动作确认</b>' : '<b class="safe-tag">受限单步</b>')
      + "</div><h3>" + escapeHtml(waitingForChange ? publicTitle : (view.currentStep?.label || publicTitle))
      + "</h3><p>" + escapeHtml(publicReason) + "</p>" + technicalDetails;
  }

  const disabled = state.busy || state.paused ? "disabled" : "";
  if (view.isTerminal) {
    controls.innerHTML = "";
  } else if (view.status === "paused") {
    content.innerHTML = "<h3>已暂停，进度已保留</h3><p>点击上方继续推进，将重新观察并签发新动作授权。</p>";
    controls.innerHTML = `<button id="cancelSupervisedAgent" class="text-button">取消会话</button>`;
  } else if (view.status === "budget_paused") {
    content.innerHTML = `<h3>预算用尽，进度已保留</h3><p>${escapeHtml(view.autoPauseReason)} 在上方调整整任务预算后继续，已用量不会清零；继续时先取得新截图。</p>`;
    controls.innerHTML = `<button id="continueBudgetAgent" class="primary-button" ${disabled}>应用预算并继续</button>
      <button id="cancelSupervisedAgent" class="text-button" ${state.busy ? "disabled" : ""}>取消会话</button>`;
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
  const badgeAction = view.raw?.qwen_decision?.next_action || {};
  badge.textContent = !view.isTerminal && badgeAction.action === "wait_for_change"
    ? "等待画面变化"
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

function currentAdbSerial() {
  const transport = state.device?.execution_architecture?.universal_agent?.text_transport;
  return String(transport?.adb_serial || currentDeviceDescriptor()?.adb_serial || "");
}

function renderPairingPanel() {
  const serialElement = document.querySelector("#pairingSerial");
  const button = document.querySelector("#pairDeviceButton");
  const status = document.querySelector("#pairingStatus");
  if (!serialElement || !button || !status) return;
  const serial = currentAdbSerial();
  serialElement.textContent = serial || "未配置";
  const configuredHost = serial.includes(":") ? serial.slice(0, serial.lastIndexOf(":")) : "";
  const hostInput = document.querySelector("#pairingHost");
  if (hostInput && configuredHost && !hostInput.value) hostInput.value = configuredHost;
  const activeSession = Boolean(sessionView() && !sessionView().isTerminal);
  button.disabled = state.busy || activeSession || !serial;
  if (!state.busy && !status.dataset.result) {
    status.textContent = serial ? "等待配对信息" : "当前设备没有 ADB 连接配置";
  }
}

async function pairDevice() {
  if (state.busy) return;
  const host = String(document.querySelector("#pairingHost")?.value || "").trim();
  const port = Number(document.querySelector("#pairingPort")?.value);
  const code = String(document.querySelector("#pairingCode")?.value || "").trim();
  const status = document.querySelector("#pairingStatus");
  if (!host || !Number.isInteger(port) || port < 1 || port > 65535 || !/^\d{6}$/.test(code)) {
    status.dataset.result = "error";
    status.textContent = "请填写手机 IP、配对端口和 6 位配对码。";
    toast("配对信息不完整。", true);
    return;
  }
  state.busy = true;
  status.dataset.result = "running";
  status.textContent = "正在配对并连接手机……";
  render();
  try {
    await api(`/api/device/${encodeURIComponent(state.deviceId)}/pair`, {
      method: "POST",
      body: JSON.stringify({ pairing_host: host, pairing_port: port, pairing_code: code }),
    });
    state.device = await api("/api/device");
    status.dataset.result = "success";
    status.textContent = "配对并连接成功，正在复核设备状态。";
    toast("手机已自动配对并连接。 ");
    document.querySelector("#pairingCode").value = "";
  } catch (error) {
    status.dataset.result = "error";
    status.textContent = `配对失败：${error.message}`;
    toast(`手机配对失败：${error.message}`, true);
  } finally {
    state.busy = false;
    render();
  }
}

function currentMachinePosition() {
  const value = Number(currentDeviceDescriptor()?.machine_position);
  return Number.isInteger(value) && value >= 1 && value <= 10 ? value : null;
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
    const button = document.createElement("button");
    button.type = "button";
    button.className = `machine-position-button${selected === position ? " active" : ""}`;
    button.textContent = String(position);
    button.setAttribute("aria-label", `切换到${position}号机位`);
    button.setAttribute("aria-pressed", selected === position ? "true" : "false");
    button.disabled = disabled;
    button.addEventListener("click", () => selectMachinePosition(position));
    return button;
  }));
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
  const luckyBagActive = ["starting", "running", "waiting_confirmation", "paused", "recovery_required"].includes(state.luckyBagMonitorStatus);
  const luckyBagNeedsResume = ["paused", "recovery_required"].includes(state.luckyBagMonitorStatus);
  const needsResume = Boolean((view && !view.isTerminal && (state.paused || view.status === "paused")) || luckyBagNeedsResume);
  const taskIsActive = Boolean(state.busy || state.taskAttemptStatus?.state === "running"
    || (view && !view.isTerminal) || luckyBagActive);
  const pauseButton = document.querySelector("#pauseButton");
  const continueButton = document.querySelector("#continueTaskButton");
  const stopButton = document.querySelector("#stopButton");
  pauseButton.hidden = !taskIsActive || needsResume;
  pauseButton.textContent = "Ⅱ 暂停推进";
  pauseButton.classList.toggle("active", state.paused);
  if (continueButton) continueButton.hidden = !needsResume;
  if (stopButton) stopButton.hidden = !taskIsActive;
  renderMachinePositions();
  renderPairingPanel();
  renderTaskRunStatus();
  renderConversation();
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
  renderPairingPanel();
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
    state.paused = false;
    render();
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
    state.chatConversation = [];
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

const START_TICKET_TIMEOUT_MS = 5 * 60 * 1000;

function cancelPendingStart(reason = "已停止等待任务启动结果。") {
  const pending = state.pendingStartTicket;
  if (!pending) return false;
  pending.cancelled = true;
  try { pending.controller?.abort(); } catch (_error) {}
  state.pendingStartTicket = null;
  state.taskAttemptStatus = {
    state: "cancelled",
    detail: reason,
    code: "START_TICKET_CANCELLED",
    phase: "start_ticket",
    recoverable: true,
    updatedAt: new Date().toISOString(),
  };
  return true;
}

async function waitForStartTicket(taskId) {
  const startedAt = Date.now();
  let nextProgressNoticeAt = startedAt + 120000;
  const controller = new AbortController();
  const pending = { taskId: String(taskId), controller, startedAt, cancelled: false };
  state.pendingStartTicket = pending;
  try {
    while (true) {
      if (pending.cancelled || controller.signal.aborted) {
        const cancelled = new Error("已停止等待任务启动结果。后台票据是否已停止需要服务端状态确认。");
        cancelled.code = "START_TICKET_CANCELLED";
        cancelled.phase = "start_ticket";
        cancelled.recoverable = true;
        throw cancelled;
      }
      if (Date.now() - startedAt >= START_TICKET_TIMEOUT_MS) {
        const timeout = new Error("启动任务超过5分钟仍未返回首轮结果；已停止继续轮询。可重新发送，但服务端票据可能仍需单独清理。");
        timeout.code = "START_TICKET_TIMEOUT";
        timeout.phase = "start_ticket_polling";
        timeout.recoverable = true;
        throw timeout;
      }
      await new Promise(resolve => setTimeout(resolve, 1000));
      const status = await api(`/api/agent/generic-supervised/start-async/${encodeURIComponent(taskId)}`, {
        signal: controller.signal,
      });
      if (status.status === "completed") return status.result;
      if (status.status === "failed") {
        const info = taskErrorInfo({
          message: status.error,
          detail: status.detail || status.error,
          code: status.error_code || status.code || "START_TASK_FAILED",
          phase: status.phase || "start_task",
          recoverable: status.recoverable,
        }, "任务启动失败。", "start_task");
        const failure = new Error(info.message);
        Object.assign(failure, info);
        failure.detail = status.detail || status.error;
        throw failure;
      }
      if (status.status === "cancelled") {
        const cancelled = new Error(
          status.detail?.message || status.detail || status.error || "启动任务已取消。",
        );
        cancelled.code = status.error_code || status.code || "START_TICKET_CANCELLED";
        cancelled.phase = status.phase || "start_ticket";
        cancelled.recoverable = status.recoverable !== false;
        cancelled.detail = status.detail || status.error;
        throw cancelled;
      }
      if (Date.now() >= nextProgressNoticeAt) {
        const elapsedSeconds = Math.floor((Date.now() - startedAt) / 1000);
        if (state.taskAttemptStatus?.state === "running" && !state.stopRequested) {
          state.taskAttemptStatus = {
            ...state.taskAttemptStatus,
            phase: "start_ticket_polling",
            detail: `正在等待首轮结果，已等待 ${elapsedSeconds} 秒；可以点击“停止”取消继续轮询。`,
            updatedAt: new Date().toISOString(),
          };
          renderTaskRunStatus();
        }
        nextProgressNoticeAt += 30000;
      }
    }
  } catch (error) {
    if (pending.cancelled || controller.signal.aborted) {
      const cancelled = new Error("已停止等待任务启动结果。后台票据是否已停止需要服务端状态确认。");
      cancelled.code = "START_TICKET_CANCELLED";
      cancelled.phase = "start_ticket";
      cancelled.recoverable = true;
      throw cancelled;
    }
    throw error;
  } finally {
    if (state.pendingStartTicket === pending) state.pendingStartTicket = null;
  }
}


const CLEAR_CARDS_GOAL = "打开 Android 最近任务，并清理全部后台卡片。由 Qwen 根据当前实时画面识别后台页面和唯一可见的系统一键清理按钮；不要使用固定坐标或逐张滑动卡片。清理后根据新的画面判断是否完成。";

const DIRECTIONAL_SWIPE_LABELS = {
  left: "左滑",
  right: "右滑",
  up: "上滑",
  down: "下滑",
};

async function startDirectionalSwipe(direction) {
  if (state.busy) return;
  const current = sessionView();
  if (current && !current.isTerminal) {
    return toast("设备已有进行中的会话，请继续或停止当前任务后再滑动。", true);
  }
  const label = DIRECTIONAL_SWIPE_LABELS[direction];
  if (!label) return toast("未识别的滑动方向。", true);
  state.busy = true;
  state.taskAttemptStatus = {
    state: "running",
    detail: `本地执行器正在执行一次${label}，不调用 Qwen。`,
    updatedAt: new Date().toISOString(),
  };
  render();
  try {
    const response = await api(`/api/device/${encodeURIComponent(state.deviceId)}/directional-swipe`, {
      method: "POST",
      body: JSON.stringify({ direction }),
    });
    const execution = response.execution || {};
    const start = Array.isArray(execution.grid_start) ? execution.grid_start.join(", ") : "—";
    const end = Array.isArray(execution.grid_end) ? execution.grid_end.join(", ") : "—";
    state.taskAttemptStatus = {
      state: "success",
      detail: `${label}已由本地执行器完成 1 次（固定网格 ${start} → ${end}），未调用 Qwen。`,
      updatedAt: new Date().toISOString(),
    };
    toast(`${label}已执行一次。`);
    state.device = await api("/api/device").catch(() => state.device);
    refreshPreview();
  } catch (error) {
    state.taskAttemptStatus = {
      state: "failure",
      detail: `${label}执行失败：${error.message || "未知错误"}`,
      updatedAt: new Date().toISOString(),
    };
    toast(`${label}执行失败：${error.message}`, true);
  } finally {
    state.busy = false;
    render();
  }
}
async function startClearCardsModule() {
  if (state.busy) return;
  const current = sessionView();
  if (current && !current.isTerminal) {
    return toast("设备已有进行中的会话，请继续或停止当前任务后再清理卡片。", true);
  }
  const input = document.querySelector("#agentText");
  if (input) input.value = CLEAR_CARDS_GOAL;
  await startSupervisedAgent();
}
async function startLuckyBagModule() {
  const activeMonitor = ["starting", "running", "waiting_confirmation", "paused", "recovery_required"].includes(state.luckyBagMonitorStatus);
  if (activeMonitor) {
    if (["paused", "recovery_required"].includes(state.luckyBagMonitorStatus)) {
      await resumeLuckyBagMonitor();
    } else {
      toast("福袋模块已经在运行，不会重复创建任务。");
    }
    return;
  }
  const current = sessionView();
  if (current && !current.isTerminal) {
    toast("设备已有进行中的任务，请先继续或停止当前任务。", true);
    return;
  }
  await startLuckyBagMonitor();
}
function luckyBagStartPayload(profile) {
  const value = profile && typeof profile === "object" ? profile : {};
  return {
    device_id: String(value.device_id || "device-local-01"),
    recipient: String(value.recipient || "q2904047615@gmail.com"),
    duration_seconds: Number(value.duration_seconds || 24 * 60 * 60),
  };
}

function luckyBagActionLabel(action, params = {}) {
  const labels = {
    tap_semantic: "点击语义控件", dismiss_overlay: "关闭当前弹层", scroll: "滚动当前页面",
    swipe_element: "滑动目标元素", back: "返回上一页", home: "返回系统桌面",
    open_recent_apps: "打开最近任务", reveal_system_navigation: "唤出系统导航栏",
    double_tap: "双击目标控件", long_press: "长按目标控件", drag: "拖动目标控件",
    input_verified_text: "输入并核对文字", press_enter: "发送换行",
    wait_for_change: "等待页面变化", finish: "完成本次任务",
  };
  const label = labels[action] || action || "尚未返回下一动作";
  if (action === "wait_for_change" && params.wait_seconds != null) {
    return label + "（" + Number(params.wait_seconds) + " 秒）";
  }
  return label;
}

function luckyBagSessionObject(response) {
  return response?.session || response || {};
}

async function getLuckyBagSession(sessionId) {
  return api("/api/agent/generic-supervised/" + encodeURIComponent(sessionId));
}

function luckyBagSceneObject(session) {
  return session?.current_scene || session?.trusted_observation?.scene ||
    session?.trusted_observation || {};
}

function luckyBagStatusLabel(status) {
  const labels = {
    starting: "启动中", running: "运行中", executing_one_action: "执行动作中",
    waiting_for_change: "等待画面变化", paused: "已暂停", waiting_confirmation: "等待确认",
    succeeded: "疑似中奖，已记录通知", completed: "已完成", expired: "已到时限",
    failed: "失败", blocked: "已阻止", cancelled: "已停止", recovery_required: "等待恢复",
  };
  return labels[status] || status || "未知状态";
}

function luckyBagHistoryRows(session) {
  const history = Array.isArray(session?.history) ? session.history : [];
  return history.slice(-8).reverse().map((item, index) => {
    const decision = item?.qwen_decision || item?.decision || {};
    const next = decision?.next_action || item?.next_action || {};
    const action = next?.action || item?.action || "";
    const reason = decision?.reason || item?.reason || item?.visual_outcome || "未提供结果摘要";
    const result = item?.execution_result || item?.result || item?.transition || {};
    const resultText = result?.status || result?.outcome || result?.state || "";
    return '<div class="lucky-feedback-event">'
      + '<strong>' + escapeHtml(index === 0 ? "最近一轮" : "历史 " + index) + '</strong>'
      + '<span>' + escapeHtml(luckyBagActionLabel(action, next?.params || {})) + '</span>'
      + '<p>' + escapeHtml(reason) + '</p>'
      + (resultText ? '<small>执行结果：' + escapeHtml(resultText) + '</small>' : "")
      + '</div>';
  }).join("");
}

function renderLuckyBagAgentFeedback(monitor, rawSession, readError = "") {
  const panel = document.querySelector("#luckyBagFeedback");
  const session = luckyBagSessionObject(rawSession);
  if (!panel) {
    const decision = session.qwen_decision || session.decision || {};
    const next = decision.next_action || session.next_action || {};
    const actionText = luckyBagActionLabel(next.action || "", next.params || {});
    const reasonText = readError || decision.reason || session.failed_reason || session.auto_pause_reason || "正在等待 Qwen 返回下一步判断。";
    const content = document.querySelector("#actionContent");
    const badge = document.querySelector("#sessionBadge");
    if (content) {
      content.className = "qwen-reply-content";
      content.textContent = actionText + "： " + reasonText;
    }
    if (badge) {
      badge.className = "pill " + (readError ? "danger" : "active");
      badge.textContent = luckyBagStatusLabel(session.status || monitor?.status || "running");
    }
    return;
  }
  panel.hidden = false;
  const status = session.status || monitor?.status || "unknown";
  const phase = document.querySelector("#luckyBagFeedbackPhase");
  const phaseClass = ["failed", "blocked", "cancelled"].includes(status) || readError
    ? "danger" : (["succeeded", "completed"].includes(status) ? "success" : "active");
  if (phase) {
    phase.className = "pill " + phaseClass;
    phase.textContent = luckyBagStatusLabel(status);
  }

  const decision = session.qwen_decision || session.decision || {};
  const next = decision.next_action || session.next_action || {};
  const action = next.action || "";
  const params = next.params || {};
  const budget = session.execution_budget || {};
  const scene = luckyBagSceneObject(session);
  const appId = scene.foreground_app_id || scene.foreground_app || scene.app_id || "未识别";
  const stats = document.querySelector("#luckyBagFeedbackStats");
  if (stats) {
    stats.innerHTML = [
      "状态 · " + luckyBagStatusLabel(status),
      "物理动作 · " + (session.physical_actions ?? budget.physical_actions ?? "—"),
      "观察次数 · " + (session.observation_attempts ?? budget.observation_attempts ?? "—"),
      "当前应用 · " + appId,
      "会话 · " + (monitor?.session_id || "—"),
    ].map(item => "<span>" + escapeHtml(item) + "</span>").join("");
  }

  const actionElement = document.querySelector("#luckyBagQwenAction");
  if (actionElement) actionElement.textContent = luckyBagActionLabel(action, params);
  const reasonElement = document.querySelector("#luckyBagQwenReason");
  if (reasonElement) reasonElement.textContent = decision.reason || "Qwen 尚未返回可展示的判断摘要。";
  const sceneElement = document.querySelector("#luckyBagSceneSummary");
  if (sceneElement) sceneElement.textContent = scene.summary || scene.description || "当前画面摘要未提供。";
  const appElement = document.querySelector("#luckyBagSceneApp");
  if (appElement) appElement.textContent = "前台应用：" + appId;

  const errorElement = document.querySelector("#luckyBagFeedbackError");
  const errorText = readError || session.failed_reason || session.auto_pause_reason ||
    (["failed", "blocked", "cancelled"].includes(status) ? "任务状态为 " + status : "");
  if (errorElement) {
    errorElement.hidden = !errorText;
    errorElement.textContent = errorText ? "错误或暂停原因：" + errorText : "";
  }

  const timeline = document.querySelector("#luckyBagFeedbackTimeline");
  if (timeline) {
    const rows = luckyBagHistoryRows(session);
    timeline.innerHTML = rows || '<div class="lucky-feedback-event"><span>当前轮次：'
      + escapeHtml(luckyBagActionLabel(action, params)) + '</span><p>'
      + escapeHtml(decision.reason || "等待第一条执行记录。") + '</p></div>';
  }
}

function renderLuckyBagMonitorStatus(monitor) {
  if (!monitor) return;
  state.luckyBagMonitorStatus = String(monitor.status || "");
  const phase = String(monitor.current_phase || "").trim();
  const action = String(monitor.last_action || "").trim();
  const reply = String(monitor.last_qwen_reply || "").trim();
  const reason = String(monitor.last_decision_reason || "").trim();
  const feedback = [
    phase ? "阶段：" + phase : "",
    action ? "本地动作：" + action : "",
    monitor.detail || reason,
    "动作 " + (monitor.physical_actions ?? 0) + " 次 / 观察 " + (monitor.observations ?? 0) + " 次",
    monitor.next_observation_epoch && monitor.status === "running" ? "下次观察：" + new Date(monitor.next_observation_epoch*1000).toLocaleTimeString("zh-CN") : "",
    monitor.notification_status === "local_queue_only" ? "通知只在本地，Gmail 未配置" : "",
  ].filter(Boolean).join("；");
  state.luckyBagMonitorDetail = feedback;
  const element = document.querySelector("#luckyBagMonitorStatus");
  if (!element || !monitor) return;
  element.textContent = luckyBagStatusLabel(monitor.status);
  const resume = document.querySelector("#resumeLuckyBagMonitor");
  if (resume) resume.hidden = monitor.status !== "recovery_required";
  const cancel = document.querySelector("#cancelLuckyBagMonitor");
  if (cancel) cancel.hidden = !["starting", "running", "paused", "waiting_confirmation", "recovery_required"].includes(monitor.status);
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
    let monitor = active.monitor;
    if (monitor && ["starting", "running", "waiting_confirmation"].includes(monitor.status) && state.luckyBagMonitorId && state.luckyBagProfile?.device_id) {
      try {
        const paused = await api(
          "/api/features/lucky-bag/" + encodeURIComponent(state.luckyBagMonitorId) + "/pause",
          { method: "POST", body: JSON.stringify({ device_id: state.luckyBagProfile.device_id }) },
        );
        monitor = paused.monitor || monitor;
      } catch (_error) {
        // page-exit keepalive or the next refresh can retry the pause.
      }
    }
    renderLuckyBagMonitorStatus(monitor);
    await pollLuckyBagMonitor();
    clearInterval(state.luckyBagMonitorTimer);
    state.luckyBagMonitorTimer = setInterval(pollLuckyBagMonitor, 5000);
  } catch (error) {
    renderLuckyBagAgentFeedback({status: "failed"}, {}, error.message);
  }
}

async function cancelLuckyBagMonitor() {
  if (!state.luckyBagMonitorId || !state.luckyBagProfile) return;
  try {
    const response = await api(
      "/api/features/lucky-bag/" + encodeURIComponent(state.luckyBagMonitorId) + "/cancel",
      { method: "POST", body: JSON.stringify({ device_id: state.luckyBagProfile.device_id }) },
    );
    clearInterval(state.luckyBagMonitorTimer);
    state.luckyBagMonitorTimer = null;
    renderLuckyBagMonitorStatus(response.monitor);
    await pollLuckyBagMonitor();
    toast("福袋监控已停止，不会再继续观察或操作手机。");
  } catch (error) {
    renderLuckyBagAgentFeedback({status: "failed"}, {}, error.message);
    toast("停止福袋监控失败：" + error.message, true);
  }
}

async function resumeLuckyBagMonitor() {
  if (!state.luckyBagMonitorId || !state.luckyBagProfile) return;
  try {
    const response = await api(
      "/api/features/lucky-bag/" + encodeURIComponent(state.luckyBagMonitorId) + "/resume",
      { method: "POST", body: JSON.stringify({ device_id: state.luckyBagProfile.device_id }) },
    );
    renderLuckyBagMonitorStatus(response.monitor);
    await pollLuckyBagMonitor();
    clearInterval(state.luckyBagMonitorTimer);
    state.luckyBagMonitorTimer = setInterval(pollLuckyBagMonitor, 5000);
    toast("监控已恢复，将重新观察当前 Android 画面。");
    render();
  } catch (error) {
    renderLuckyBagAgentFeedback({status: "failed"}, {}, error.message);
    toast("恢复福袋监控失败：" + error.message, true);
  }
}

async function pollLuckyBagMonitor() {
  if (!state.luckyBagMonitorId) return;
  try {
    const response = await api("/api/features/lucky-bag/" + encodeURIComponent(state.luckyBagMonitorId));
    const monitor = response.monitor || {};
    renderLuckyBagMonitorStatus(monitor);
    if (!monitor.session_id) {
      if (["succeeded", "expired", "failed", "cancelled"].includes(monitor.status)) {
        state.taskAttemptStatus = {state: monitor.status === "succeeded" ? "success" : monitor.status === "failed" ? "failure" : "cancelled", detail: monitor.detail || "福袋监控已结束", updatedAt: monitor.updated_at};
        clearInterval(state.luckyBagMonitorTimer);
        state.luckyBagMonitorTimer = null;
      }
      render();
      return;
    }
    if (monitor.session_id) {
      try {
        const sessionResponse = await getLuckyBagSession(monitor.session_id);
        renderLuckyBagAgentFeedback(monitor, sessionResponse);
      } catch (sessionError) {
        renderLuckyBagAgentFeedback(monitor, {}, sessionError.message);
      }
    } else {
      renderLuckyBagAgentFeedback(monitor, { status: monitor.status, qwen_decision: { reason: monitor.detail || "会话尚未创建，等待首轮观察。" } });
    }
    if (["succeeded", "expired", "failed", "blocked", "cancelled"].includes(monitor.status)) {
      clearInterval(state.luckyBagMonitorTimer);
      state.luckyBagMonitorTimer = null;
    }
  } catch (error) {
    renderLuckyBagMonitorStatus({status: "状态读取失败"});
    renderLuckyBagAgentFeedback({status: "failed"}, {}, "监控状态读取失败：" + error.message);
  }
}

async function startLuckyBagMonitor() {
  if (state.busy) return;
  state.busy = true;
  render();
  try {
    if (!state.luckyBagProfile) {
      const feature = await api("/api/features/lucky-bag");
      state.luckyBagProfile = feature.profile || null;
    }
    const response = await api("/api/features/lucky-bag/start", {
      method: "POST",
      body: JSON.stringify(luckyBagStartPayload(state.luckyBagProfile)),
    });
    state.luckyBagMonitorId = String(response.monitor?.monitor_id || "");
    renderLuckyBagMonitorStatus(response.monitor);
    await pollLuckyBagMonitor();
    clearInterval(state.luckyBagMonitorTimer);
    state.luckyBagMonitorTimer = setInterval(pollLuckyBagMonitor, 5000);
    toast("长期监控已启动；请保持用户手动打开的直播间。");
  } catch (error) {
    renderLuckyBagAgentFeedback({status: "failed"}, {}, error.message);
    toast("启动福袋监控失败：" + error.message, true);
  } finally {
    state.busy = false;
    render();
  }
}
function luckyBagMonitorIsActive() {
  return ["starting", "running", "waiting_confirmation", "paused", "recovery_required"]
    .includes(String(state.luckyBagMonitorStatus || ""));
}

function parseStructuredError(value) {
  if (value && typeof value === "object") return value;
  if (typeof value !== "string") return value;
  const text = value.trim();
  if (!text) return value;
  try { return JSON.parse(text); } catch (_error) { return value; }
}

function taskErrorInfo(error, fallbackMessage = "任务失败。", fallbackPhase = "request") {
  const detail = parseStructuredError(error?.detail);
  const payload = detail && typeof detail === "object" ? detail : {};
  const message = String(error?.message || payload.error || payload.message || detail || fallbackMessage);
  const code = String(error?.code || payload.error_code || payload.code || (error?.status ? `HTTP_${error.status}` : "TASK_FAILED"));
  const phase = String(error?.phase || payload.phase || fallbackPhase);
  const recoverable = error?.recoverable !== undefined
    ? Boolean(error.recoverable)
    : payload.recoverable !== undefined ? Boolean(payload.recoverable) : true;
  return { message, code, phase, recoverable };
}

function taskErrorDetail(info) {
  const value = info || {};
  const recovery = value.recoverable === false ? "需要重新发送任务" : "可以重新观察或重试";
  return `${value.message || value.detail || "任务失败。"} · 阶段：${value.phase || "未知"} · 错误码：${value.code || "TASK_FAILED"} · ${recovery}`;
}

function isDeviceBusyConflict(error) {
  const message = String(error?.message || error?.detail || "");
  return Number(error?.status) === 409 || /已有活动任务|活动任务/.test(message);
}
async function sendUnifiedQwenMessage(text) {
  const message = String(text || "").trim();
  if (!message || state.busy) return false;
  const active = sessionView();
  if (!active && luckyBagMonitorIsActive()) {
    toast("消息未发送：福袋监控正在占用当前手机。请先暂停或停止福袋监控，再发送普通 Qwen 任务。", true);
    return false;
  }
  const previousConversation = cleanConversationForQwen(active
    ? (Array.isArray(active.raw?.conversation) ? active.raw.conversation : [])
    : state.chatConversation);
  const optimisticConversation = [
    ...previousConversation,
    { role: "user", content: message },
  ];
  if (active) {
    state.supervisedSession = { ...state.supervisedSession, conversation: optimisticConversation };
  } else {
    state.chatConversation = optimisticConversation;
  }
  render();

  if (active && !active.isTerminal) {
    try {
      const response = await withVisionProgress("和 Qwen 对话并等待它判断是否继续操作", () =>
        api(`/api/agent/generic-supervised/${encodeURIComponent(active.sessionId)}/chat`, {
          method: "POST",
          body: JSON.stringify({ device_id: state.sessionDeviceId || state.deviceId, text: message }),
        })
      );
      state.supervisedSession = response.session;
      toast("Qwen 已结合当前手机画面返回回复并决定下一步。");
      render();
      return true;
    } catch (error) {
      if (isDeviceBusyConflict(error)) {
        toast("消息未发送：设备已有活动任务，请先暂停或停止当前手机任务。", true);
        render();
        return false;
      }
      const info = taskErrorInfo(error, "统一 Qwen 对话失败。", "conversation");
      state.taskAttemptStatus = {
        state: info.code === "START_TICKET_CANCELLED" ? "cancelled" : "failure",
        detail: info.message,
        code: info.code,
        phase: info.phase,
        recoverable: info.recoverable,
        updatedAt: new Date().toISOString(),
      };
      render();
      toast(taskErrorDetail(info), true);
      return false;
    }
  }

  state.taskAttemptStatus = {
    state: "running",
    detail: "正在发送消息并等待 Qwen 决定是否需要读取画面或控制手机。",
    phase: "conversation",
    code: "",
    recoverable: true,
    updatedAt: new Date().toISOString(),
  };
  try {
    const response = await withVisionProgress("和 Qwen 对话并判断是否需要操作手机", () =>
      api("/api/qwen/chat", {
        method: "POST",
        body: JSON.stringify({
          text: message,
          device_id: state.deviceId,
          conversation: previousConversation,
          ...taskBudgetPayload(),
        }),
      })
    );
    if (Array.isArray(response.conversation)) state.chatConversation = response.conversation;
    if (response?.route === "chat_only" || response?.phone_task_started === false) {
      state.taskAttemptStatus = null;
      toast("Qwen 已回复，没有启动手机任务。");
      render();
      return true;
    }
    if (!response?.task_id) {
      const routeError = new Error("Qwen 已决定需要手机任务，但服务没有返回启动票据。\n");
      routeError.code = "MISSING_START_TICKET";
      routeError.phase = "start_ticket";
      routeError.recoverable = true;
      throw routeError;
    }
    const result = await waitForStartTicket(response.task_id);
    state.supervisedSession = result.session;
    state.sessionDeviceId = state.deviceId;
    state.taskAttemptStatus = null;
    toast("Qwen 已读取当前手机画面并返回回复；是否操作由它在同一响应中决定。");
    render();
    return true;
  } catch (error) {
      if (isDeviceBusyConflict(error)) {
        toast("消息未发送：设备已有活动任务，请先暂停或停止当前手机任务。", true);
        render();
        return false;
      }
    const info = taskErrorInfo(error, "统一 Qwen 对话失败。", "conversation");
    if (restoreSupervisedSessionFromError(error)) {
      state.taskAttemptStatus = null;
      render();
      toast("Qwen 首轮观察失败，但会话已保留；可点击继续任务重新观察。", true);
      return false;
    }
    const failureState = info.code === "START_TICKET_CANCELLED" ? "cancelled" : "failure";
    state.taskAttemptStatus = saveLastTaskOutcome({
      state: "failure",
      detail: info.message,
      sessionId: "",
      code: info.code,
      phase: info.phase,
      recoverable: info.recoverable,
      updatedAt: new Date().toISOString(),
    });
    if (failureState === "cancelled") state.taskAttemptStatus = {
      ...state.taskAttemptStatus,
      state: "cancelled",
    };
    render();
    toast(taskErrorDetail(info), true);
    return false;
  }
}

async function startSupervisedAgent() {
  const input = document.querySelector("#agentText");
  const text = String(input?.value || "").trim();
  if (!text) return toast("请输入要发送给 Qwen 的内容。", true);
  if (input) input.value = "";
  return sendUnifiedQwenMessage(text);
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
    ? "确认本次验收动作的效果范围"
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
    ? `仅确认 trial=${view.trialId}、action=${view.action}、session=${scope.session_id}、task=${scope.task_id}、revision=${scope.revision}、step=${scope.step_id}、effect_ids=${scope.effect_ids.join(",") || "—"} 的观察权限；本次物理动作数必须保持 0。`
    : `只授权 trial=${view.trialId}、action=${view.action}、session=${scope.session_id}、task=${scope.task_id}、revision=${scope.revision}、step=${scope.step_id}、observation_id=${scope.observation_id}、fingerprint=${scope.fingerprint} 对应的一个动作；失败不自动重试。`;
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
    : `${view.currentStep.label} · 等待 Qwen 唯一动作`;
  document.querySelector("#riskReason").textContent = view.effectPolicy.currentActions.map(item => `${item.id} [${item.policyLevel}]：${item.kind}；${item.expectedResults.join("、")}`).join("\n") || view.visualAction.reason;
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
    ? "后端效果 scope 与当前权威任务及 当前选中动作摘要一致；确认后只执行该观察绑定的一次动作。"
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
  if (["starting", "running"].includes(state.luckyBagMonitorStatus) && state.luckyBagMonitorId) {
    try {
      const response = await api("/api/features/lucky-bag/" + encodeURIComponent(state.luckyBagMonitorId) + "/pause", {
        method: "POST", body: JSON.stringify({device_id: state.luckyBagProfile.device_id}),
      });
      renderLuckyBagMonitorStatus(response.monitor);
      render();
    } catch (error) { toast("福袋暂停失败：" + error.message, true); }
    return;
  }
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

async function continueTask() {
  const view = sessionView();
  if (view && !view.isTerminal && (state.paused || view.status === "paused")) {
    state.paused = false;
    await continueBudgetAgent();
    return;
  }
  if (["paused", "recovery_required"].includes(state.luckyBagMonitorStatus)) {
    await resumeLuckyBagMonitor();
  }
}
async function stopTasks() {
  if (["starting", "running", "paused", "recovery_required"].includes(state.luckyBagMonitorStatus) && state.luckyBagMonitorId) {
    await cancelLuckyBagMonitor();
    render();
    return;
  }
  const cancelledStart = cancelPendingStart();
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
    toast(cancelledStart
      ? "已停止继续轮询启动票据；服务端是否已停止该票据需要状态确认。"
      : (result.note || "停止请求已发送。"));
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
    if (!restored.isTerminal && restored.status !== "paused" && !response.session?.pause_requested) {
      try {
        const paused = await api(`/api/agent/generic-supervised/${encodeURIComponent(restored.sessionId)}/pause`, {
          method: "POST",
          body: JSON.stringify({ device_id: restored.deviceId || state.deviceId }),
        });
        if (paused?.session) state.supervisedSession = paused.session;
      } catch (_error) {
        // page-exit keepalive or the next refresh can retry the pause.
      }
    }
    const pausedView = Protocol.adaptSession(state.supervisedSession, { fallbackDeviceId: state.deviceId });
    state.paused = pausedView.status === "paused" || Boolean(state.supervisedSession?.pause_requested);
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

async function init() {
  try {
    state.lastTaskOutcome = readLastTaskOutcome(state.deviceId);
    const session = await api("/api/session");
    state.token = session.token;
    state.mock = session.mock;
    await settlePageExitPause();
    const mode = document.querySelector("#modeBadge");
    mode.textContent = session.mock ? "模拟模式 · 无实机动作" : "实机接口已连接";
    mode.classList.toggle("live-mode", !session.mock);
    await refreshDevice();
    await restoreLuckyBagMonitor();
    if (!state.luckyBagMonitorId) await restoreActiveSession();
    if (!state.supervisedSession && !state.luckyBagMonitorId) await restoreCapabilityTrial();
    render();
    refreshPreview();
    setInterval(refreshPreview, 700);
    setInterval(() => refreshDevice().catch(() => {}), 3000);
  } catch (error) {
    toast(`连接本地服务失败：${error.message}`, true);
  }
}

async function handleComposerSend() {
  const text = document.querySelector("#agentText").value.trim();
  if (!text) return toast("请输入要发送给 Qwen 的内容。", true);
  document.querySelector("#agentText").value = "";
  await sendUnifiedQwenMessage(text);
}

document.querySelector("#startSupervisedAgent").addEventListener("click", handleComposerSend);
document.querySelector("#pairDeviceButton")?.addEventListener("click", pairDevice);
document.querySelector("#useLuckyBagPreset")?.addEventListener("click", startLuckyBagModule);
document.querySelector("#useClearCardsPreset")?.addEventListener("click", startClearCardsModule);
document.querySelector("#swipeLeftPreset")?.addEventListener("click", () => startDirectionalSwipe("left"));
document.querySelector("#swipeRightPreset")?.addEventListener("click", () => startDirectionalSwipe("right"));
document.querySelector("#swipeUpPreset")?.addEventListener("click", () => startDirectionalSwipe("up"));
document.querySelector("#swipeDownPreset")?.addEventListener("click", () => startDirectionalSwipe("down"));
document.querySelector("#startLuckyBagMonitor")?.addEventListener("click", startLuckyBagMonitor);
document.querySelector("#resumeLuckyBagMonitor")?.addEventListener("click", resumeLuckyBagMonitor);
document.querySelector("#cancelLuckyBagMonitor")?.addEventListener("click", cancelLuckyBagMonitor);
document.querySelector("#startCapabilityTrial").addEventListener("click", startCapabilityTrial);
document.querySelector("#pauseButton").addEventListener("click", togglePause);
document.querySelector("#continueTaskButton")?.addEventListener("click", continueTask);
window.addEventListener("pagehide", pauseActiveTasksForPageExit);
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
});
document.querySelector("#agentText").addEventListener("keydown", event => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") handleComposerSend();
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
