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
};

const terminalStatuses = new Set(["succeeded", "blocked", "failed", "cancelled"]);

const statusNames = {
  awaiting_confirmation: "等待确认",
  paused_after_action: "已完成一步",
  succeeded: "目标完成",
  blocked: "已阻止",
  failed: "失败",
  cancelled: "已停止",
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
  if (options.method && options.method !== "GET") {
    headers["X-Control-Token"] = state.token;
  }
  if (options.body) headers["Content-Type"] = "application/json";
  const response = await fetch(path, { ...options, headers });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = data.detail;
    const message = typeof detail === "string"
      ? detail
      : (detail?.error || JSON.stringify(detail || {}));
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

function currentProposal() {
  return state.supervisedSession?.proposal || null;
}

function taskGraphNodes(session = state.supervisedSession) {
  const graph = session?.task_graph || session?.goal?.task_graph || {};
  const nodes = Array.isArray(graph) ? graph : graph.nodes;
  return Array.isArray(nodes) ? nodes : [];
}

function currentSubgoal(session = state.supervisedSession) {
  const value = session?.current_subgoal || currentProposal()?.current_subgoal;
  if (typeof value === "string") return value;
  if (value && typeof value === "object") {
    return value.label || value.objective || value.title || "当前动态子目标";
  }
  return session?.goal?.objective || "当前动态子目标";
}

function currentAction() {
  return state.supervisedSession?.visual_action || currentProposal()?.action || null;
}

function actionLabel(action) {
  if (!action) return "等待重新观察";
  return semanticActionNames[action.action] || action.action || "未知动作";
}

function actionTarget(action) {
  const params = action?.params || {};
  return params.label || params.target || params.element_id || params.direction || "当前语义目标";
}

function confidenceLabel(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${Math.round(number * 100)}%` : "—";
}

function renderStatus() {
  const device = state.device || {};
  const session = state.supervisedSession;
  setDot("#controllerDot", device.controller_online ? (device.busy ? "warn" : "online") : "offline");
  setDot("#cameraDot", device.camera_online ? "online" : "offline");
  setDot("#agentDot", state.busy ? "warn" : (session && !terminalStatuses.has(session.status) ? "online" : "neutral"));
  document.querySelector("#controllerText").textContent = device.controller_online
    ? (device.busy ? "当前动作执行中" : "在线且空闲")
    : "离线";
  document.querySelector("#cameraText").textContent = device.camera_online ? "实时画面可用" : "画面不可用";
  document.querySelector("#agentTextStatus").textContent = state.paused
    ? "人工暂停"
    : (state.busy ? currentVisionStageLabel() : (session ? (statusNames[session.status] || session.status) : "等待目标"));
  document.querySelector("#safetyText").textContent = currentActionHasAccountEffect()
    ? "外部状态动作待确认"
    : "一次一动作";
  setDot("#safetyDot", currentActionHasAccountEffect() ? "warn" : "online");
}

function renderGoalAndPlan() {
  const session = state.supervisedSession;
  const goalElement = document.querySelector("#goalSummary");
  const planElement = document.querySelector("#planList");
  const badge = document.querySelector("#planBadge");
  if (!session) {
    goalElement.className = "goal-summary empty-state";
    goalElement.textContent = "输入目标后，这里会展示目标、约束和可验证的完成条件。";
    planElement.innerHTML = "";
    badge.className = "pill neutral";
    badge.textContent = "等待目标";
    return;
  }

  const goal = session.goal || {};
  const constraints = Array.isArray(goal.constraints) ? goal.constraints : [];
  const criteria = Object.entries(goal.success_criteria || {});
  goalElement.className = "goal-summary";
  goalElement.innerHTML = `
    <div class="goal-title-row">
      <div><span>目标</span><strong>${escapeHtml(goal.objective || "未命名目标")}</strong></div>
      <code>${escapeHtml(state.sessionDeviceId || state.deviceId)}</code>
    </div>
    <div class="goal-chips">
      ${goal.app_name ? `<span>App · ${escapeHtml(goal.app_name)}</span>` : ""}
      ${constraints.map(item => `<span>限制 · ${escapeHtml(item)}</span>`).join("")}
      ${criteria.map(([key, value]) => `<span>完成 · ${escapeHtml(key)}：${escapeHtml(formatValue(value))}</span>`).join("")}
    </div>`;

  const history = Array.isArray(session.history) ? session.history : [];
  const graphNodes = taskGraphNodes(session);
  if (graphNodes.length) {
    planElement.innerHTML = graphNodes.map((node, index) => {
      const rawStatus = String(node.status || "pending").toLowerCase();
      const stateName = ["done", "completed", "succeeded"].includes(rawStatus)
        ? "done"
        : (["current", "running", "active"].includes(rawStatus) ? "current" : (["failed", "blocked", "cancelled"].includes(rawStatus) ? "blocked" : "waiting"));
      return planStepHtml({
        number: node.index ?? node.step_number ?? index + 1,
        label: node.label || node.objective || node.title || `动态节点 ${index + 1}`,
        detail: node.reason || node.checkpoint || node.success_criteria || "等待当前画面更新",
        stateName,
      });
    }).join("");
    badge.className = `pill ${session.status === "succeeded" ? "success" : (terminalStatuses.has(session.status) ? "danger" : "active")}`;
    badge.textContent = statusNames[session.status] || session.status;
    return;
  }
  const completed = history.map(item => planStepHtml({
    number: item.step_number,
    label: actionLabel(item.proposal?.action),
    detail: item.completion_evidence?.join("；") || "动作后已重新观察",
    stateName: "done",
  })).join("");
  const proposal = currentProposal();
  const current = proposal && !terminalStatuses.has(session.status)
    ? planStepHtml({
        number: session.step_number,
        label: actionLabel(proposal.action),
        detail: proposal.reason || "依据当前画面动态生成",
        stateName: session.status === "awaiting_confirmation" ? "current" : "waiting",
      })
    : "";
  const result = terminalStatuses.has(session.status)
    ? planStepHtml({
        number: history.length + 1,
        label: statusNames[session.status] || session.status,
        detail: proposal?.reason || session.failed_reason || "会话已结束",
        stateName: session.status === "succeeded" ? "done" : "blocked",
      })
    : planStepHtml({
        number: "…",
        label: "后续步骤等待新画面",
        detail: "不会预先写死；当前动作验证后再动态生成",
        stateName: "waiting",
      });
  planElement.innerHTML = completed + current + result;
  badge.className = `pill ${session.status === "succeeded" ? "success" : (terminalStatuses.has(session.status) ? "danger" : "active")}`;
  badge.textContent = statusNames[session.status] || session.status;
}

function planStepHtml({ number, label, detail, stateName }) {
  return `<div class="plan-step ${stateName}">
    <span class="step-index">${escapeHtml(number)}</span>
    <div><strong>${escapeHtml(label)}</strong><p>${escapeHtml(detail)}</p></div>
    <span class="step-state">${stateName === "done" ? "完成" : (stateName === "current" ? "当前" : (stateName === "blocked" ? "停止" : "动态"))}</span>
  </div>`;
}

function formatValue(value) {
  if (Array.isArray(value)) return value.map(formatValue).join("、");
  if (value && typeof value === "object") return Object.entries(value).map(([key, item]) => `${key}=${formatValue(item)}`).join("；");
  return String(value ?? "—");
}

function renderTrace() {
  const session = state.supervisedSession;
  const trace = document.querySelector("#traceList");
  const count = document.querySelector("#traceCount");
  const history = Array.isArray(session?.history) ? session.history : [];
  count.textContent = `${history.length} 条`;
  if (!session) {
    trace.className = "trace-list empty-state";
    trace.textContent = "还没有执行记录。每次观察、确认、动作和验证都会显示在这里。";
    return;
  }
  const rows = history.slice().reverse().map(item => {
    const action = item.proposal?.action;
    const physicalActions = Number(item.execution?.physical_actions || 0);
    return `<article class="trace-item">
      <div class="trace-marker"></div>
      <div>
        <div class="trace-title"><strong>步骤 ${escapeHtml(item.step_number)} · ${escapeHtml(actionLabel(action))}</strong><time>${physicalActions} 个物理动作</time></div>
        <p>${escapeHtml(item.proposal?.reason || "已执行并重新观察")}</p>
        <small>目标：${escapeHtml(actionTarget(action))} · 验证：${escapeHtml(item.completion_evidence?.join("；") || "已保存动作后画面")}</small>
      </div>
    </article>`;
  }).join("");
  trace.className = "trace-list";
  trace.innerHTML = rows || `<article class="trace-item observation-only"><div class="trace-marker"></div><div><div class="trace-title"><strong>目标已理解，初始画面已观察</strong><time>0 个物理动作</time></div><p>正在等待当前一步确认。</p></div></article>`;
}

function renderScene() {
  const scene = state.supervisedSession?.scene || null;
  const overlay = document.querySelector("#previewOverlay");
  const meta = document.querySelector("#sceneMeta");
  if (!scene) {
    overlay.textContent = state.busy ? currentVisionStageLabel() : "等待观察";
    overlay.classList.toggle("show", state.busy);
    meta.innerHTML = `<span><b>页面</b><em>尚未识别</em></span><span><b>稳定性</b><em>—</em></span><span><b>置信度</b><em>—</em></span>`;
    return;
  }
  overlay.textContent = state.busy ? currentVisionStageLabel() : "";
  overlay.classList.toggle("show", state.busy);
  meta.innerHTML = `
    <span><b>页面</b><em>${escapeHtml(scene.summary || scene.screen_id || "unknown")}</em></span>
    <span><b>稳定性</b><em>${scene.stable ? "稳定" : "不稳定"}</em></span>
    <span><b>置信度</b><em>${escapeHtml(confidenceLabel(scene.confidence))}</em></span>`;
}

function renderAction() {
  const session = state.supervisedSession;
  const content = document.querySelector("#actionContent");
  const controls = document.querySelector("#actionControls");
  const badge = document.querySelector("#sessionBadge");
  const pauseNotice = document.querySelector("#pauseNotice");
  pauseNotice.hidden = !state.paused;
  if (!session) {
    content.className = "empty-state";
    content.textContent = "Agent 将结合目标和当前画面，只提出一个下一动作。";
    controls.innerHTML = "";
    badge.className = "pill neutral";
    badge.textContent = "未开始";
    return;
  }

  const proposal = currentProposal();
  const action = currentAction();
  const terminal = terminalStatuses.has(session.status);
  const accountEffect = currentActionHasAccountEffect();
  content.className = "action-content";
  content.innerHTML = terminal
    ? `<h3>${escapeHtml(statusNames[session.status] || session.status)}</h3><p>${escapeHtml(proposal?.reason || session.failed_reason || "会话已经结束。")}</p>`
    : (session.status === "paused_after_action"
      ? `<h3>上一步已完成并重新观察</h3><p>继续后会根据新画面生成下一步，不会沿用失效计划。</p>`
      : `<div class="next-action-title"><span>${escapeHtml(actionLabel(action))}</span>${accountEffect ? '<b class="risk-tag">外部状态风险</b>' : '<b class="safe-tag">受限单步</b>'}</div>
         <h3>${escapeHtml(currentSubgoal(session))}</h3>
         <div class="action-target">本步动作 · ${escapeHtml(actionTarget(action))}</div>
         <p>${escapeHtml(proposal?.reason || "依据当前画面动态生成")}</p>
         <small>确认仅授权当前一步；动作完成后必须重新观察。</small>`);

  const disabled = state.busy || state.paused ? "disabled" : "";
  if (terminal) {
    controls.innerHTML = "";
  } else if (session.status === "awaiting_confirmation") {
    controls.innerHTML = `
      <button id="reviewAction" class="${accountEffect ? "risk-button" : "primary-button"}" ${disabled}>${accountEffect ? "查看风险并确认" : "确认当前一步"}</button>
      ${!accountEffect ? `<button id="autoSupervisedAgent" class="secondary-button" ${disabled}>自动推进安全步骤</button>` : ""}
      <button id="cancelSupervisedAgent" class="text-button" ${state.busy ? "disabled" : ""}>取消会话</button>`;
  } else {
    controls.innerHTML = `
      <button id="nextSupervisedAgent" class="primary-button" ${disabled}>观察并生成下一步</button>
      <button id="autoSupervisedAgent" class="secondary-button" ${disabled}>自动推进安全步骤</button>
      <button id="cancelSupervisedAgent" class="text-button" ${state.busy ? "disabled" : ""}>取消会话</button>`;
  }
  badge.className = `pill ${accountEffect ? "risk" : (terminal ? (session.status === "succeeded" ? "success" : "danger") : "active")}`;
  badge.textContent = accountEffect ? "等待风险确认" : (statusNames[session.status] || session.status);
  bindActionEvents();
}

function currentActionHasAccountEffect() {
  return !!state.supervisedSession?.current_action?.account_effect_possible;
}

function bindActionEvents() {
  document.querySelector("#reviewAction")?.addEventListener("click", openRiskDialog);
  document.querySelector("#nextSupervisedAgent")?.addEventListener("click", nextSupervisedAgent);
  document.querySelector("#autoSupervisedAgent")?.addEventListener("click", autoSupervisedAgent);
  document.querySelector("#cancelSupervisedAgent")?.addEventListener("click", cancelSupervisedAgent);
}

function render() {
  const deviceSelect = document.querySelector("#deviceId");
  deviceSelect.value = state.deviceId;
  deviceSelect.disabled = !!state.supervisedSession && !terminalStatuses.has(state.supervisedSession.status);
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
  const input = document.querySelector("#agentText");
  const text = input.value.trim();
  if (!text) return toast("请先输入希望手机完成的目标。", true);
  if (state.supervisedSession && !terminalStatuses.has(state.supervisedSession.status)) {
    return toast("已有进行中的会话，请继续或停止后再创建新目标。", true);
  }
  state.sessionDeviceId = state.deviceId;
  try {
    const response = await withVisionProgress("理解目标并观察当前画面", () =>
      api("/api/agent/generic-supervised/start", {
        method: "POST",
        body: JSON.stringify({ text, device_id: state.deviceId }),
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
  const session = state.supervisedSession;
  const proposal = currentProposal();
  const action = currentAction();
  if (!session || !action || state.paused || state.busy) return;
  const accountEffect = currentActionHasAccountEffect();
  document.querySelector("#riskTitle").textContent = accountEffect ? "确认外部状态动作" : "确认当前单步动作";
  const level = document.querySelector("#riskLevel");
  level.className = `risk-level ${accountEffect ? "high" : "guarded"}`;
  level.textContent = accountEffect ? "高关注 · 可能改变账号或对外产生影响" : "受控动作 · 仅授权当前一步";
  document.querySelector("#riskGoal").textContent = session.goal?.objective || "—";
  document.querySelector("#riskAction").textContent = `${actionLabel(action)} · ${actionTarget(action)}`;
  document.querySelector("#riskReason").textContent = proposal?.reason || "—";
  document.querySelector("#riskExpected").textContent = action.params?.expected_result || "动作后重新观察，并根据可见变化判断是否成功";
  document.querySelector("#riskDevice").textContent = state.sessionDeviceId || state.deviceId;
  document.querySelector("#riskWarning").textContent = accountEffect
    ? "此动作可能发送消息、关注、点赞、评论或改变账号状态。确认只授权当前一个动作，后续风险动作仍需再次确认。"
    : "确认只授权当前一个动作。执行后系统必须重新观察，不会自动沿用旧画面继续点击。";
  document.querySelector("#confirmRiskAction").className = accountEffect ? "danger-confirm" : "primary-button";
  document.querySelector("#riskDialog").showModal();
}

async function advanceSupervisedAgent() {
  const session = state.supervisedSession;
  if (!session || state.paused || state.busy) return;
  try {
    const response = await withVisionProgress("执行当前一步并重新观察", () =>
      api(`/api/agent/generic-supervised/${session.session_id}/confirm`, {
        method: "POST",
        body: JSON.stringify({ confirmed: true, device_id: state.deviceId }),
      })
    );
    state.supervisedSession = response.session;
    await finalizeStopIfRequested();
    toast("当前一步已处理，并已重新观察画面。");
    render();
  } catch (error) {
    toast(error.message, true);
  }
}

async function nextSupervisedAgent() {
  const session = state.supervisedSession;
  if (!session || state.paused || state.busy) return;
  try {
    const response = await withVisionProgress("重新观察并动态规划下一步", () =>
      api(`/api/agent/generic-supervised/${session.session_id}/next`, {
        method: "POST",
        body: JSON.stringify({ device_id: state.deviceId }),
      })
    );
    state.supervisedSession = response.session;
    await finalizeStopIfRequested();
    render();
  } catch (error) {
    toast(error.message, true);
  }
}

async function autoSupervisedAgent() {
  const session = state.supervisedSession;
  if (!session || state.paused || state.busy || currentActionHasAccountEffect()) return;
  try {
    const response = await withVisionProgress("逐步观察并推进低风险动作", () =>
      api(`/api/agent/generic-supervised/${session.session_id}/auto`, {
        method: "POST",
        body: JSON.stringify({ confirmed: true, max_physical_actions: 4, device_id: state.deviceId }),
      })
    );
    state.supervisedSession = response.session;
    await finalizeStopIfRequested();
    if (state.supervisedSession.auto_pause_reason) toast(state.supervisedSession.auto_pause_reason);
    render();
  } catch (error) {
    toast(error.message, true);
  }
}

async function cancelSupervisedAgent({ quiet = false } = {}) {
  const session = state.supervisedSession;
  if (!session || terminalStatuses.has(session.status)) return;
  try {
    const response = await api(`/api/agent/generic-supervised/${session.session_id}/cancel`, {
      method: "POST",
      body: JSON.stringify({ device_id: state.deviceId }),
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
  toast(state.paused
    ? "已暂停推进。正在进行的最小动作完成后会停住。"
    : "已恢复，可由你确认后继续生成或执行下一步。");
  render();
}

async function stopTasks() {
  const requestWasRunning = state.busy;
  state.paused = true;
  state.stopRequested = true;
  render();
  try {
    const result = await api("/api/stop", {
      method: "POST",
      body: JSON.stringify({ device_id: state.deviceId }),
    });
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
    state.sessionDeviceId = state.deviceId;
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
  if (event.target.returnValue === "default") advanceSupervisedAgent();
});

init();
