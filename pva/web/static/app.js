"use strict";

const $ = (id) => document.getElementById(id);
const STATUS_TEXT = {
  running: "执行中", awaiting_approval: "等待确认", needs_confirmation: "需要确认", paused: "已暂停",
  succeeded: "已完成", failed: "失败", cancelled: "已停止", interrupted: "已中断",
};
const STEP_TEXT = { thinking: "思考中", proposed: "待确认", executed: "已执行", error: "错误", finished: "完成", rejected: "未执行" };
const RESULT_TEXT = { success: "上一步成功", failed: "上一步无效", uncertain: "上一步不确定" };
const ACTIVE = new Set(["running", "awaiting_approval", "needs_confirmation", "paused"]);
const EXAMPLES = [
  "打开设置，把蓝牙打开",
  "打开备忘录，新建一条笔记：明天下午三点开会",
  "打开消息，给张三发一条“今晚一起吃饭吗”",
  "回到桌面，打开最近任务并清除全部后台应用",
];

const state = {
  status: null,
  deviceId: localStorage.getItem("pva.device") || null,
  taskId: null,
  task: null,
  lastStepCount: -1,
  previewUrl: null,
  previewBusy: false,
};

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

async function api(path, options = {}) {
  const init = { method: options.method || "GET", headers: {} };
  if (options.body !== undefined) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(options.body);
  }
  const response = await fetch(path, init);
  const text = await response.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = { detail: text }; }
  if (!response.ok) {
    let detail = data && data.detail;
    if (Array.isArray(detail)) detail = detail.map((d) => d.msg).join("；");
    throw new Error(detail || `请求失败 (${response.status})`);
  }
  return data;
}

let toastTimer = null;
function toast(message, bad = false) {
  const el = $("toast");
  el.textContent = message;
  el.className = "toast" + (bad ? " bad" : "");
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, bad ? 6000 : 2500);
}

async function run(action, okMessage) {
  try {
    const result = await action();
    if (okMessage) toast(okMessage);
    return result;
  } catch (error) {
    toast(error.message, true);
    return null;
  }
}

/* ---------------- status & devices ---------------- */
function selectedDevice() {
  if (!state.status) return null;
  return state.status.devices.find((d) => d.id === state.deviceId) || null;
}

function deviceDotClass(device) {
  if (!device.online || !device.camera) return "off";
  if (device.task_id || device.owner) return "busy";
  return "on";
}

async function refreshStatus() {
  let status;
  try {
    status = await api("/api/status");
  } catch (error) {
    $("modelBadge").textContent = "服务未连接";
    $("modelBadge").className = "badge bad";
    return;
  }
  state.status = status;
  if (!status.devices.some((d) => d.id === state.deviceId)) state.deviceId = status.default_device;

  const model = status.model;
  $("modelBadge").textContent = model.configured ? `千问 ${model.model}` : "千问未配置";
  $("modelBadge").className = "badge " + (model.configured ? "ok" : "bad");
  $("modelBadge").title = model.error || `已调用 ${model.calls ?? 0} 次，${model.total_tokens ?? 0} tokens`;
  $("modeBadge").textContent = status.mock ? "模拟模式" : "真机模式";
  $("modeBadge").className = "badge " + (status.mock ? "warn" : "");
  if (!$("maxActions").value) $("maxActions").value = status.defaults.max_actions;
  if (!$("maxObservations").value) $("maxObservations").value = status.defaults.max_observations;
  if (!$("thinking").dataset.init) { $("thinking").checked = !!status.defaults.enable_thinking; $("thinking").dataset.init = "1"; }

  renderDevices();
  const device = selectedDevice();
  if (device && device.task_id && (!state.taskId || !ACTIVE.has(state.task?.status))) {
    if (state.taskId !== device.task_id) openTask(device.task_id);
  }
}

function renderDevices() {
  const list = $("deviceList");
  list.innerHTML = state.status.devices.map((d) => `
    <button class="device-chip ${d.id === state.deviceId ? "active" : ""}" data-device="${esc(d.id)}" title="${esc(d.error || d.driver)}">
      <span class="dot ${deviceDotClass(d)}"></span>${esc(d.name)}
    </button>`).join("");
  const d = selectedDevice();
  if (!d) return;
  const pill = $("deviceState");
  if (!d.online) { pill.textContent = "离线"; pill.className = "pill bad"; }
  else if (!d.camera) { pill.textContent = "无相机画面"; pill.className = "pill bad"; }
  else if (d.task_id) { pill.textContent = "任务中"; pill.className = "pill running"; }
  else if (d.owner) { pill.textContent = "被插件占用"; pill.className = "pill warn"; }
  else { pill.textContent = "空闲"; pill.className = "pill ok"; }

  const kb = d.keyboard || {};
  const cal = d.calibration;
  const rows = [
    ["驱动", { robot: "机械臂 + 相机", adb: "ADB", mock: "模拟手机" }[d.driver] || d.driver],
    ["文字输入", d.text_input ? (kb.ready ? "ADB Keyboard 就绪" : `不可用：${kb.reason || "未知"}`) : "未配置 adb_serial"],
    ["包名直启", d.apps.length ? d.apps.join("、") : "未配置"],
  ];
  if (cal) rows.push(["触控标定", cal.active ? `已启用（误差 ${cal.fit?.rms_error_px ?? "?"}px）` : (cal.error || "未启用（不纠偏）")]);
  if (d.error) rows.push(["问题", d.error]);
  $("deviceInfo").innerHTML = rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("");

  const robot = d.driver === "robot";
  $("positionRow").hidden = !robot;
  if (robot) {
    $("positionButtons").innerHTML = Array.from({ length: 10 }, (_, i) => i + 1)
      .map((n) => `<button data-position="${n}" class="${n === d.machine_position ? "active" : ""}">${n}</button>`).join("");
  }
  const busy = !!(d.task_id || d.owner);
  document.querySelectorAll("[data-manual]").forEach((b) => { b.disabled = busy || !d.online; });
}

$("deviceList").addEventListener("click", (event) => {
  const chip = event.target.closest("[data-device]");
  if (!chip) return;
  state.deviceId = chip.dataset.device;
  localStorage.setItem("pva.device", state.deviceId);
  renderDevices();
  refreshPreview(true);
  const device = selectedDevice();
  if (device?.task_id) openTask(device.task_id);
});

$("positionButtons").addEventListener("click", (event) => {
  const button = event.target.closest("[data-position]");
  if (!button) return;
  run(() => api(`/api/devices/${state.deviceId}/machine-position`, { method: "POST", body: { position: Number(button.dataset.position) } }),
    `已切换到 ${button.dataset.position} 号机位`).then(refreshStatus);
});

/* ---------------- preview & manual control ---------------- */
async function refreshPreview(force = false) {
  if (!state.deviceId || (state.previewBusy && !force) || document.hidden) return;
  state.previewBusy = true;
  const deviceId = state.deviceId;
  try {
    const response = await fetch(`/api/devices/${deviceId}/preview.jpg?t=${Date.now()}`);
    if (deviceId !== state.deviceId) return;
    if (response.ok) {
      const url = URL.createObjectURL(await response.blob());
      $("preview").src = url;
      if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
      state.previewUrl = url;
      $("previewMsg").textContent = "";
    } else if (response.status !== 409) {
      const data = await response.json().catch(() => ({}));
      $("previewMsg").textContent = data.detail || "画面不可用";
    }
  } catch {
    $("previewMsg").textContent = "画面不可用";
  } finally {
    state.previewBusy = false;
  }
}

async function manual(action) {
  return run(() => api(`/api/devices/${state.deviceId}/action`, { method: "POST", body: { action } }))
    .then((result) => { if (result) setTimeout(() => refreshPreview(true), 400); return result; });
}

document.querySelectorAll("[data-manual]").forEach((button) => {
  button.addEventListener("click", () => {
    const key = button.dataset.manual;
    if (key.startsWith("scroll-")) manual({ type: "scroll", direction: key.slice(7) });
    else manual({ type: key });
  });
});

$("manualToggle").addEventListener("change", () => {
  document.querySelector(".screen-wrap").classList.toggle("clickable", $("manualToggle").checked);
});

$("preview").addEventListener("click", (event) => {
  if (!$("manualToggle").checked) {
    if ($("preview").src) { $("lightboxImg").src = $("preview").src; $("lightbox").hidden = false; }
    return;
  }
  const d = selectedDevice();
  if (!d || d.task_id || d.owner) { toast("设备忙，手动控制只在空闲时可用。", true); return; }
  const rect = event.target.getBoundingClientRect();
  const x = Math.round(((event.clientX - rect.left) / rect.width) * 1000);
  const y = Math.round(((event.clientY - rect.top) / rect.height) * 1000);
  const marker = $("tapMarker");
  const wrap = document.querySelector(".screen-wrap").getBoundingClientRect();
  marker.style.left = `${event.clientX - wrap.left}px`;
  marker.style.top = `${event.clientY - wrap.top}px`;
  marker.hidden = false;
  marker.style.animation = "none"; void marker.offsetWidth; marker.style.animation = "";
  manual({ type: "tap", point: [x, y] });
});

/* ---------------- tasks ---------------- */
$("examples").innerHTML = EXAMPLES.map((e) => `<button type="button">${esc(e)}</button>`).join("");
$("examples").addEventListener("click", (event) => {
  if (event.target.tagName === "BUTTON") { $("taskInput").value = event.target.textContent; $("taskInput").focus(); }
});

$("taskForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const task = $("taskInput").value.trim();
  if (!task) { toast("请先输入任务。", true); return; }
  $("startBtn").disabled = true;
  const session = await run(() => api("/api/tasks", {
    method: "POST",
    body: {
      device_id: state.deviceId, task, mode: $("modeSelect").value,
      max_actions: Number($("maxActions").value) || null,
      max_observations: Number($("maxObservations").value) || null,
      enable_thinking: $("thinking").checked,
    },
  }), "任务已开始");
  $("startBtn").disabled = false;
  if (session) { openTask(session.id, session); refreshStatus(); }
});

$("taskInput").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) $("taskForm").requestSubmit();
});

function openTask(id, data = null) {
  state.taskId = id;
  state.lastStepCount = -1;
  switchTab("task");
  if (data) renderTask(data); else refreshTask();
}

async function refreshTask() {
  if (!state.taskId) return;
  try {
    renderTask(await api(`/api/tasks/${state.taskId}`));
  } catch (error) {
    state.taskId = null;
    renderTask(null);
  }
}

function taskCommand(command, body) {
  return run(() => api(`/api/tasks/${state.taskId}/${command}`, { method: "POST", body }))
    .then((data) => { if (data) renderTask(data); refreshStatus(); });
}

function renderTask(task) {
  state.task = task;
  $("taskCard").hidden = !task;
  $("emptyHint").hidden = !!task;
  if (!task) { renderTimeline([], null); return; }
  const active = ACTIVE.has(task.status) && !task.readonly;
  const status = $("taskStatus");
  status.textContent = STATUS_TEXT[task.status] || task.status;
  status.className = `pill ${task.status}`;
  const budget = task.budget || {};
  $("taskMeta").textContent = `${task.device_name || task.device_id} · ${task.mode === "step" ? "逐步确认" : "自动"} · 动作 ${budget.actions ?? 0}/${budget.max_actions ?? "?"} · 观察 ${budget.observations ?? 0}/${budget.max_observations ?? "?"} · 千问 ${task.model_calls ?? 0} 次`;
  $("taskText").textContent = task.task;
  $("progressBar").style.width = `${Math.min(100, ((budget.actions || 0) / (budget.max_actions || 1)) * 100)}%`;
  $("taskReply").textContent = task.last_reply || (task.status === "running" ? "正在看画面…" : "—");

  const detail = $("taskDetail");
  if (task.result) {
    detail.textContent = task.result;
    detail.className = "detail " + (task.status === "succeeded" ? "result-ok" : "result-bad");
  } else {
    detail.textContent = task.detail || "";
    detail.className = "detail";
  }

  const pending = active && task.pending && (task.status === "awaiting_approval" || task.status === "needs_confirmation");
  $("confirmBox").hidden = !pending;
  if (pending) {
    const risk = { authentication: "登录/身份认证", payment: "付款/资金交易" }[task.pending_risk];
    $("confirmTitle").textContent = risk ? `⚠ 该动作涉及${risk}，需要你确认` : "逐步确认：是否执行这个动作？";
    $("confirmText").textContent = task.pending_text;
  }
  $("pauseBtn").hidden = !(active && task.status === "running");
  $("resumeBtn").hidden = !(active && task.status === "paused");
  $("observeBtn").hidden = !(active && task.mode === "step" && ["awaiting_approval", "paused"].includes(task.status));
  $("cancelBtn").hidden = !active;
  $("closeBtn").hidden = active;
  $("noteForm").hidden = !active;
  renderTimeline(task.steps || [], task);
}

function stepImage(task, name) {
  return `/api/tasks/${task.id}/files/${encodeURIComponent(name)}`;
}

function renderTimeline(steps, task) {
  $("timelineEmpty").hidden = steps.length > 0;
  $("timelineMeta").textContent = steps.length ? `${steps.length} 步` : "";
  const signature = JSON.stringify(steps.map((s) => [s.status, s.reply, s.error, s.action_text]));
  if (signature === state.lastStepCount) return;
  state.lastStepCount = signature;
  const items = steps.slice().reverse().map((s, index) => {
    const tag = STEP_TEXT[s.status] || s.status;
    const result = RESULT_TEXT[s.last_result];
    const extra = (s.extra_screenshots || []).map((n) => `<a href="${stepImage(task, n)}" target="_blank">附加截图</a>`).join(" ");
    const usage = s.model && s.model.elapsed_seconds ? `千问耗时 ${s.model.elapsed_seconds}s · ${s.model.usage?.total_tokens ?? "?"} tokens` : "";
    return `
      <li class="step ${index === 0 && ACTIVE.has(task?.status) ? "current" : ""}">
        <img src="${stepImage(task, s.screenshot)}" alt="第${s.index}步截图" loading="lazy" data-zoom>
        <div>
          <div class="step-head">
            <span class="step-no">第 ${s.index} 步 ${s.risk && s.risk !== "none" ? "· ⚠ " + esc(s.risk) : ""}</span>
            <span class="tag ${esc(s.status)}">${esc(tag)}</span>
          </div>
          <div class="step-action">${esc(s.action_text || (s.status === "thinking" ? "千问思考中…" : "—"))}</div>
          ${s.reply ? `<p class="step-reply">${esc(s.reply)}</p>` : ""}
          ${s.error ? `<div class="step-error">${esc(s.error)}</div>` : ""}
          <details class="step-more">
            <summary>${result ? esc(result) + " · " : ""}详情</summary>
            ${s.observation ? `<p><b>画面：</b>${esc(s.observation)}</p>` : ""}
            ${s.reason ? `<p><b>理由：</b>${esc(s.reason)}</p>` : ""}
            ${usage ? `<p>${esc(usage)}</p>` : ""}
            ${extra ? `<p>${extra}</p>` : ""}
          </details>
        </div>
      </li>`;
  });
  $("timeline").innerHTML = items.join("");
}

$("timeline").addEventListener("click", (event) => {
  if (event.target.dataset.zoom !== undefined) { $("lightboxImg").src = event.target.src; $("lightbox").hidden = false; }
});
$("lightbox").addEventListener("click", () => { $("lightbox").hidden = true; });
document.addEventListener("keydown", (e) => { if (e.key === "Escape") $("lightbox").hidden = true; });

$("pauseBtn").addEventListener("click", () => taskCommand("pause"));
$("resumeBtn").addEventListener("click", () => taskCommand("resume", {
  max_actions: Number($("maxActions").value) || null, max_observations: Number($("maxObservations").value) || null,
}));
$("observeBtn").addEventListener("click", () => taskCommand("observe"));
$("cancelBtn").addEventListener("click", () => { if (confirm("确定停止当前任务？")) taskCommand("cancel"); });
$("approveBtn").addEventListener("click", () => taskCommand("approve"));
$("rejectBtn").addEventListener("click", () => taskCommand("reject", { text: $("noteInput").value }).then(() => { $("noteInput").value = ""; }));
$("closeBtn").addEventListener("click", () => { state.taskId = null; renderTask(null); });
$("noteForm").addEventListener("submit", (event) => {
  event.preventDefault();
  const text = $("noteInput").value.trim();
  if (!text) return;
  taskCommand("note", { text }).then(() => { $("noteInput").value = ""; toast("说明已发送，下一步生效"); });
});
$("stopAll").addEventListener("click", () => {
  if (!confirm("停止所有任务并中断正在执行的动作？")) return;
  run(() => api("/api/stop", { method: "POST" }), "已全部停止").then(() => { refreshStatus(); refreshTask(); });
});

/* ---------------- tabs & history ---------------- */
function switchTab(name) {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  ["task", "history", "plugins"].forEach((n) => { $(`tab-${n}`).hidden = n !== name; });
  if (name === "history") loadHistory();
  if (name === "plugins") loadPlugins();
}
document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => switchTab(t.dataset.tab)));
$("refreshHistory").addEventListener("click", loadHistory);

async function loadHistory() {
  const data = await run(() => api("/api/tasks?limit=50"));
  if (!data) return;
  $("historyList").innerHTML = data.tasks.length ? data.tasks.map((t) => `
    <li data-task="${esc(t.id)}">
      <div class="h-top"><span class="h-task">${esc(t.task)}</span><span class="pill ${esc(t.status)}">${esc(STATUS_TEXT[t.status] || t.status)}</span></div>
      <span class="muted">${esc(t.created_at || "")} · ${esc(t.device_name || t.device_id)} · ${t.step_count ?? 0} 步 · 动作 ${t.budget?.actions ?? 0}</span>
      ${t.result ? `<span class="muted">${esc(t.result)}</span>` : ""}
    </li>`).join("") : `<p class="muted center">还没有任务记录</p>`;
}
$("historyList").addEventListener("click", (event) => {
  const item = event.target.closest("[data-task]");
  if (item) openTask(item.dataset.task);
});

/* ---------------- plugins ---------------- */
function deviceOptions() {
  return (state.status?.devices || []).map((d) => `<option value="${esc(d.id)}" ${d.id === state.deviceId ? "selected" : ""}>${esc(d.name)}</option>`).join("");
}

async function loadPlugins() {
  const plugins = state.status?.plugins || {};
  await Promise.all([renderLuckyBag(plugins.lucky_bag), renderQishui(plugins.qishui)]);
}

async function renderLuckyBag(meta) {
  const card = $("pluginLuckyBag");
  if (!meta?.ok) { card.innerHTML = `<h3>抖音福袋监控</h3><p class="plugin-error">插件未加载：${esc(meta?.error || "未知")}</p>`; return; }
  const info = await run(() => api("/api/plugins/lucky-bag"));
  if (!info) return;
  const mail = info.notification || {};
  card.innerHTML = `
    <h3>${esc(info.title)}</h3>
    <p>手动打开抖音直播间后启动：自动点福袋、发送预填评论、等待开奖，疑似中奖时发邮件。识别方式：${info.mode === "upstream" ? "上游检测器 + ADB" : "本地模板 + 机械臂"}；邮件：${mail.configured ? "已配置" : "未配置（只写入本地队列）"}。</p>
    ${info.restore_error ? `<p class="plugin-error">${esc(info.restore_error)}</p>` : ""}
    <form class="plugin-form" id="lbForm">
      <select name="device">${deviceOptions()}</select>
      <input name="recipient" value="${esc(info.default_recipient)}" placeholder="通知邮箱" size="22">
      <label>时长(小时) <input name="hours" type="number" min="0.1" step="0.5" value="24" style="width:70px"></label>
      <button class="btn primary">开始监控</button>
    </form>
    <ul class="plugin-items">${info.monitors.map((m) => `
      <li><div class="row"><span class="pill ${esc(m.status === "running" ? "running" : m.status)}">${esc(m.status)}</span>
        <b>${esc(m.current_phase || "")}</b><span class="muted">${esc(m.device_id)} · 观察 ${m.observations ?? 0} · 点击 ${m.physical_actions ?? 0}</span></div>
        <span class="muted">${esc(m.detail || "")}</span>
        <div class="row">
          <button class="btn" data-lb="${esc(m.monitor_id)}" data-cmd="pause">暂停</button>
          <button class="btn" data-lb="${esc(m.monitor_id)}" data-cmd="resume">继续</button>
          <button class="btn danger-outline" data-lb="${esc(m.monitor_id)}" data-cmd="cancel">停止</button>
        </div></li>`).join("")}</ul>`;
  $("lbForm").addEventListener("submit", (event) => {
    event.preventDefault();
    const f = event.target;
    run(() => api("/api/plugins/lucky-bag/start", { method: "POST", body: {
      device_id: f.device.value, recipient: f.recipient.value, duration_seconds: Math.round(Number(f.hours.value) * 3600) } }),
      "福袋监控已启动").then(loadPlugins);
  });
  card.querySelectorAll("[data-lb]").forEach((b) => b.addEventListener("click", () =>
    run(() => api(`/api/plugins/lucky-bag/${b.dataset.lb}/${b.dataset.cmd}`, { method: "POST" })).then(loadPlugins)));
}

async function renderQishui(meta) {
  const card = $("pluginQishui");
  if (!meta?.ok) { card.innerHTML = `<h3>汽水音乐广告页测试</h3><p class="plugin-error">插件未加载：${esc(meta?.error || "未知")}</p>`; return; }
  const info = await run(() => api("/api/plugins/qishui"));
  if (!info) return;
  card.innerHTML = `
    <h3>${esc(info.title)}</h3>
    <p>手动打开一个广告后，点“观察”让千问判断页面状态。不会自动点击；最多经你确认执行一次“返回”，从不点击领取。</p>
    <form class="plugin-form" id="qsForm">
      <select name="device">${deviceOptions()}</select>
      <label>时限(秒) <input name="seconds" type="number" min="10" value="180" style="width:80px"></label>
      <button class="btn primary">开始测试</button>
    </form>
    <ul class="plugin-items">${info.tests.map((t) => `
      <li><div class="row"><span class="pill ${esc(t.status === "observing" ? "running" : t.status)}">${esc(t.status)}</span>
        <span class="muted">${esc(t.device_id)}</span></div>
        <span>${esc(t.detail)}</span>
        ${t.model_reply ? `<span class="muted">千问：${esc(t.model_reply)}</span>` : ""}
        ${t.proposal ? `<span class="muted">建议：${esc(t.proposal)}</span>` : ""}
        <div class="row">
          ${t.status === "observing" ? `<button class="btn" data-qs="${t.test_id}" data-cmd="observe">观察</button>` : ""}
          ${t.confirm === "back" ? `<button class="btn primary" data-qs="${t.test_id}" data-cmd="confirm" data-mode="back">确认返回一次</button>` : ""}
          ${t.confirm === "end" ? `<button class="btn primary" data-qs="${t.test_id}" data-cmd="confirm" data-mode="end">确认结束</button>` : ""}
          ${["observing", "awaiting_confirmation"].includes(t.status) ? `<button class="btn danger-outline" data-qs="${t.test_id}" data-cmd="cancel">停止</button>` : ""}
        </div></li>`).join("")}</ul>`;
  $("qsForm").addEventListener("submit", (event) => {
    event.preventDefault();
    const f = event.target;
    run(() => api("/api/plugins/qishui/start", { method: "POST", body: { device_id: f.device.value, duration_seconds: Number(f.seconds.value) } }),
      "测试已开始，千问正在观察").then(loadPlugins);
  });
  card.querySelectorAll("[data-qs]").forEach((b) => b.addEventListener("click", () => {
    const body = b.dataset.mode ? { mode: b.dataset.mode } : undefined;
    run(() => api(`/api/plugins/qishui/${b.dataset.qs}/${b.dataset.cmd}`, { method: "POST", body })).then(loadPlugins);
  }));
}

/* ---------------- polling ---------------- */
let ticks = 0;
async function tick() {
  ticks += 1;
  await refreshStatus();
  if (state.taskId && (!state.task || ACTIVE.has(state.task.status))) await refreshTask();
  if (!$("tab-plugins").hidden && ticks % 4 === 0) loadPlugins();
}

refreshStatus().then(() => refreshPreview(true));
setInterval(tick, 1500);
setInterval(() => refreshPreview(), 1200);
