const state = {
  token: "",
  mock: false,
  apps: [],
  readiness: {},
  device: {},
  tasks: [],
  pendingTask: null,
  supervisedSession: null,
  visionStage: "",
};

const operationNames = {
  "wechat.send_text_to_file_transfer": "微信 · 发送文字",
  "wechat.send_text": "微信 · 指定聊天发送文字",
  "wechat.send_album_image": "微信 · 指定聊天发送相册图片",
  "douyin.like_current": "抖音 · 点赞当前视频",
  "douyin.comment_current": "抖音 · 评论当前视频",
  "douyin.search": "抖音 · 搜索",
  "douyin.batch_interact": "抖音 · 批量点赞/评论",
};

const statusNames = {
  awaiting_confirmation: "等待确认",
  queued: "排队中",
  running: "执行中",
  succeeded: "成功",
  failed: "失败",
  cancelled: "已取消",
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
  toast.timer = setTimeout(() => element.className = "toast", 3200);
}

function setDot(id, stateName) {
  document.querySelector(id).className = `dot ${stateName}`;
}

function renderStatus() {
  const device = state.device;
  setDot("#controllerDot", device.controller_online ? "online" : "offline");
  setDot("#cameraDot", device.camera_online ? "online" : "offline");
  const active = state.tasks.filter(t => ["queued", "running"].includes(t.status));
  setDot("#queueDot", active.length ? "warn" : "online");
  document.querySelector("#controllerText").textContent =
    device.controller_online ? (device.busy ? "正在执行" : "在线") : "离线";
  document.querySelector("#cameraText").textContent =
    device.camera_online ? "画面正常" : "不可用";
  document.querySelector("#queueText").textContent =
    active.length ? `${active.length} 个任务处理中` : "空闲";
}

function previewCard() {
  return `
    <section class="card preview-card">
      <div class="card-head">
        <div><h2>手机实时画面</h2><p>约 3 FPS，只用于观察与安全复核</p></div>
        <span class="live-pill">● LIVE</span>
      </div>
      <div class="preview-wrap"><img id="phonePreview" src="/api/preview.jpg" alt="手机摄像头预览"></div>
    </section>`;
}

function agentCard() {
  const vision = state.device.vision_agent || {};
  const intent = state.device.intent_agent || {};
  const ready = !!vision.configured && !!intent.configured;
  return `
    <section class="agent-box">
      <div class="agent-inner">
        <div class="card-head">
          <div>
            <h3>通用视觉操作 Agent</h3>
            <p>DeepSeek 理解目标，Qwen 观察当前页面；每次只建议并确认一个动作</p>
          </div>
          <span class="readiness ${ready ? "ready" : "not-ready"}">
            ${ready ? "已连接" : "DeepSeek 或 Qwen 尚未就绪"}
          </span>
        </div>
        <div class="agent-input">
          <input id="agentText" autocomplete="off" maxlength="500"
            placeholder="例如：打开设置，进入蓝牙页面">
          <button id="parseAgent" class="primary-button">理解目标</button>
        </div>
        <div id="visionStage" class="muted">${escapeHtml(currentVisionStageLabel())}</div>
        <button id="startSupervisedAgent" class="secondary-button">观察页面并生成第一步（不执行）</button>
        <div id="agentResult" class="agent-result${state.supervisedSession ? " show" : ""}">
          ${state.supervisedSession ? supervisedSessionHtml(state.supervisedSession) : ""}
        </div>
      </div>
    </section>`;
}

function observerStatus() {
  return state.device.execution_architecture?.universal_agent?.observer || {};
}

function currentVisionStageLabel() {
  const observer = observerStatus();
  if (state.visionStage) return `视觉阶段：${state.visionStage}`;
  if (observer.current_stage && observer.current_stage !== "idle") {
    return `视觉阶段：${observer.current_stage_label || observer.current_stage}`;
  }
  return "视觉阶段：等待任务";
}

function updateVisionStageElement() {
  const element = document.querySelector("#visionStage");
  if (element) element.textContent = currentVisionStageLabel();
}

async function pollVisionStage() {
  try {
    state.device = await api("/api/device");
    const observer = observerStatus();
    if (observer.current_stage && observer.current_stage !== "idle") {
      state.visionStage = observer.current_stage_label || observer.current_stage;
    }
    updateVisionStageElement();
  } catch (_error) {
    // The main request owns user-visible errors. A missed progress poll is harmless.
  }
}

async function withVisionProgress(initialLabel, operation) {
  state.visionStage = initialLabel;
  updateVisionStageElement();
  const timer = setInterval(pollVisionStage, 750);
  try {
    return await operation();
  } finally {
    clearInterval(timer);
    await pollVisionStage();
    const observer = observerStatus();
    state.visionStage = observer.last_stage_label || "观察结束";
    updateVisionStageElement();
    setTimeout(() => {
      state.visionStage = "";
      updateVisionStageElement();
    }, 2500);
  }
}

const semanticActionNames = {
  ensure_app: "打开目标 App",
  observe: "重新观察页面",
  tap_semantic: "点击语义控件",
  dismiss_overlay: "关闭当前弹层",
  swipe: "按指定方向滑动",
  back: "返回上一页",
  wait_for_change: "等待页面变化",
  record_verified_result: "记录已验证结果",
  finish: "完成本次任务",
};

function supervisedSessionHtml(session) {
  const proposal = session.proposal || {};
  const action = proposal.action || {};
  const current = session.current_action || {};
  const terminal = ["succeeded", "blocked", "failed", "cancelled"].includes(session.status);
  const risk = current.account_effect_possible
    ? "这一步会改变账号状态"
    : (current.physical_action_possible ? "这一步可能会落笔" : "这一步不会触碰手机");
  const params = action.params && Object.keys(action.params).length
    ? `；参数 ${escapeHtml(JSON.stringify(action.params))}`
    : "";
  return `
    <b>通用单步会话：${escapeHtml(session.status)}</b>
    <p>当前第 ${escapeHtml(session.step_number || 1)} 步。${terminal
      ? `结果：${escapeHtml(proposal.reason || session.failed_reason || session.status)}`
      : (session.status === "paused_after_action"
        ? "上一步已经执行并重新观察，当前暂停。"
        : `待执行：${escapeHtml(semanticActionNames[action.action] || action.action || "无")}${params}；原因：${escapeHtml(proposal.reason || "")}`)}</p>
    ${session.status === "awaiting_confirmation" ? `<p class="muted">${escapeHtml(risk)}；确认后最多执行一个物理动作，然后自动暂停。</p>` : ""}
    ${session.status === "awaiting_confirmation" ? `<button id="advanceSupervisedAgent" class="primary-button">确认执行当前一步</button>` : ""}
    ${session.status === "awaiting_confirmation" && !current.account_effect_possible ? `<button id="autoSupervisedAgent" class="secondary-button">确认并自动推进安全步骤</button>` : ""}
    ${session.status === "paused_after_action" ? `<button id="nextSupervisedAgent" class="primary-button">重新观察并规划下一步（不执行）</button>` : ""}
    ${session.status === "paused_after_action" ? `<button id="autoSupervisedAgent" class="secondary-button">继续自动推进安全步骤</button>` : ""}
    ${session.auto_pause_reason ? `<p class="muted">自动推进已暂停：${escapeHtml(session.auto_pause_reason)}</p>` : ""}
    ${!terminal ? `<button id="cancelSupervisedAgent" class="secondary-button">取消本次会话</button>` : ""}`;
}

function appsCard() {
  return `
    <section class="card">
      <div class="card-head"><div><h2>选择 App</h2><p>每个 App 使用独立、受限的视觉工作流</p></div></div>
      <div class="app-grid">
        ${state.apps.map(app => `
          <a class="app-card${app.enabled === false ? " disabled" : ""}" href="${app.route}">
            <div class="app-icon">${escapeHtml(app.icon)}</div>
            <h3>${escapeHtml(app.name)}</h3>
            <p>${app.note || `${app.operations.length} 个可用动作`}</p>
          </a>`).join("")}
      </div>
    </section>`;
}

function taskRows(limit = 6) {
  const tasks = state.tasks.slice(0, limit);
  if (!tasks.length) return `<div class="empty">还没有任务记录</div>`;
  return `<div class="task-list">${tasks.map(task => {
    const evidence = task.result?.evidence || [];
    const report = task.result?.report;
    const stateGraph = task.params?.workflow_version === "page_state_graph_v1";
    const metrics = task.result?.controller_metrics || {};
    const completedPages = Number(metrics.completed_pages || 0);
    const requestedPages = Number(metrics.requested || 0);
    const normalRetypes = Number(metrics.input_recovery?.total_retypes || 0);
    const commentRetypes = Number(metrics.batch_comment_input?.retypes_used || 0);
    const retryCount = normalRetypes + commentRetypes;
    const stateActions = Number(task.result?.actions || 0);
    const stateObservations = Number(task.result?.observations || 0);
    const progressText = requestedPages
      ? `批量进度 ${completedPages}/${requestedPages} · 已检查 ${Number(metrics.pages_seen || 0)}/${Number(metrics.max_page_checks || 0)} 页`
      : (stateGraph ? `状态图：观察 ${stateObservations} 次 · 动作 ${stateActions} 次` : "");
    const retryText = `完整重输 ${retryCount} 次（每个输入框上限2次）`;
    return `
      <div class="task-row">
        <span class="status-tag ${task.status}">${statusNames[task.status] || task.status}</span>
        <div>
          <strong>${escapeHtml(operationNames[task.operation] || task.operation)}</strong><br>
          <small>${escapeHtml(task.created_at)} · ${escapeHtml(task.id.slice(0, 8))}</small>
          ${progressText ? `<div class="task-progress">${escapeHtml(progressText)} · ${escapeHtml(retryText)}</div>` : ""}
        </div>
        ${evidence.length ? `<a class="evidence-link" target="_blank" href="/api/evidence/${task.id}/0">查看截图</a>` : ""}
        ${report ? `<a class="evidence-link" target="_blank" href="/api/report/${task.id}">查看报告</a>` : ""}
        ${task.error ? `<div class="task-error">${escapeHtml(task.error)}</div>` : ""}
      </div>`;
  }).join("")}</div>`;
}

function homeView() {
  return `
    <div class="dashboard-grid">
      ${previewCard()}
      <div class="stack">
        ${agentCard()}
        ${appsCard()}
        <section class="card">
          <div class="card-head">
            <div><h2>最近任务</h2><p>任务结果会保留在本机 SQLite</p></div>
            <a class="evidence-link" href="#/tasks">查看全部</a>
          </div>
          ${taskRows(4)}
        </section>
      </div>
    </div>`;
}

function readinessBadge(key) {
  const value = state.readiness[key] || { ready: false, missing_templates: [] };
  const message = value.ready
    ? (value.mode === "runtime_ocr" ? "运行时识别" : "已就绪")
    : `未就绪${value.missing_templates?.length ? ` · 缺 ${value.missing_templates.length} 张模板` : ""}`;
  return `<span class="readiness ${value.ready ? "ready" : "not-ready"}">${message}</span>`;
}

function operationBlock({ title, description, operation, readinessKey, field }) {
  const value = state.readiness[readinessKey] || {};
  const ready = !!value.ready;
  return `
    <article class="operation">
      <div><h3>${escapeHtml(title)}</h3><p>${escapeHtml(description)}</p></div>
      <div>${readinessBadge(readinessKey)}</div>
      <div class="operation-form open">
        ${field === "textarea" ? `<textarea id="${operation}-text" rows="3" maxlength="100" placeholder="输入内容（最多100字，不支持换行）"></textarea>` : ""}
        <button class="primary-button operation-submit" data-operation="${operation}" ${ready ? "" : "disabled"}>
          创建待确认任务
        </button>
      </div>
    </article>`;
}

function wechatView() {
  return `
    <div class="action-page">
      <div class="action-hero"><div class="app-icon">微</div><div><h2>微信</h2><p>仅接受唯一完全匹配的聊天名称</p></div></div>
      <section class="card operation-list">
        <article class="operation">
          <div><h3>发送文字到指定聊天</h3><p>搜索唯一匹配聊天，逐字验证输入，错误时清空本次输入并从头重打。</p></div>
          <div>${readinessBadge("vision_agent")}</div>
          <div class="operation-form open">
            <input id="wechat.send_text-chat" maxlength="40" placeholder="聊天名称，例如：文件传输助手">
            <textarea id="wechat.send_text-text" rows="3" maxlength="100" placeholder="消息正文（最多100字）"></textarea>
            <button class="primary-button operation-submit" data-operation="wechat.send_text">创建待确认任务</button>
          </div>
        </article>
        <article class="operation">
          <div><h3>发送相册图片到指定聊天</h3><p>进入“最近”，按左上到右下选择第1～20张，只允许单选。</p></div>
          <div>${readinessBadge("vision_agent")}</div>
          <div class="operation-form open">
            <input id="wechat.send_album_image-chat" maxlength="40" placeholder="聊天名称">
            <input id="wechat.send_album_image-index" type="number" min="1" max="20" value="1" placeholder="图片序号 1～20">
            <button class="primary-button operation-submit" data-operation="wechat.send_album_image">创建待确认任务</button>
          </div>
        </article>
      </section>
    </div>`;
}

function douyinView() {
  return `
    <div class="action-page">
      <div class="action-hero"><div class="app-icon">抖</div><div><h2>抖音</h2><p>搜索或对当前推荐流执行1～10个目标</p></div></div>
      <section class="card operation-list">
        <article class="operation">
          <div><h3>搜索视频</h3><p>验证搜索框文字后提交并进入“视频”结果。</p></div>
          <div>${readinessBadge("vision_agent")}</div>
          <div class="operation-form open">
            <input id="douyin.search-keyword" maxlength="100" placeholder="搜索关键词">
            <button class="primary-button operation-submit" data-operation="douyin.search">创建待确认任务</button>
          </div>
        </article>
        <article class="operation">
          <div><h3>批量点赞与评论</h3><p>关键词可留空使用当前推荐流；最多检查目标数量+5个页面。</p></div>
          <div>${readinessBadge("vision_agent")}</div>
          <div class="operation-form open">
            <input id="douyin.batch_interact-keyword" maxlength="100" placeholder="搜索关键词（可留空）">
            <input id="douyin.batch_interact-count" type="number" min="1" max="10" value="1" placeholder="目标成功数量 1～10">
            <div class="choice-row">
              <label><input id="douyin.batch_interact-like" type="checkbox" checked> 点赞</label>
              <label><input id="douyin.batch_interact-comment" type="checkbox"> 评论</label>
            </div>
            <textarea id="douyin.batch_interact-comment-text" rows="3" maxlength="100" placeholder="启用评论时填写评论原文"></textarea>
            <button class="primary-button operation-submit" data-operation="douyin.batch_interact">创建待确认任务</button>
          </div>
        </article>
      </section>
    </div>`;
}

function tasksView() {
  return `<section class="card action-page">
    <div class="card-head"><div><h2>全部任务</h2><p>刷新网页后仍可读取历史结果</p></div></div>
    ${taskRows(100)}
  </section>`;
}

function settingsView() {
  const agent = state.device.vision_agent || {};
  const missing = Object.entries(state.readiness)
    .filter(([, value]) => !value.ready)
    .map(([key, value]) => `<li><b>${key}</b>：${escapeHtml((value.missing_templates || []).join("、") || "未就绪")}</li>`)
    .join("");
  return `<div class="settings-grid">
    <section class="card">
      <div class="card-head"><div><h2>运行前检查</h2><p>涉及真实 App 操作前必须逐项确认</p></div></div>
      <ol class="checklist">
        <li>卖家机械臂控制端已打开，摄像头画面稳定。</li>
        <li>机械臂已在安全原点并完成卖家软件标定。</li>
        <li>微信文字功能已安装卡饭输入法并导入短语。</li>
        <li>手机停在正确账号和目标 App 页面。</li>
        <li>先创建任务草稿，再逐次人工确认。</li>
      </ol>
    </section>
    <section class="card">
      <div class="card-head"><div><h2>识别状态</h2><p>微信使用运行时中文 OCR；抖音仍使用专用视觉识别</p></div></div>
      <ul class="checklist">${missing || "<li>所有第一版工作流均已就绪。</li>"}</ul>
      <p class="muted">抖音专用模板目录</p>
      <div class="code">poc/templates/web/</div>
    </section>
    <section class="card">
      <div class="card-head"><div><h2>页面观察模型</h2><p>密钥仅从本机环境变量读取，不会显示在网页中</p></div></div>
      <ul class="checklist">
        <li>模型：${escapeHtml(agent.model || "qwen3-vl-plus")}</li>
        <li>状态：${agent.configured ? "已配置" : "未配置 DASHSCOPE_API_KEY"}</li>
        <li>每轮观察：${escapeHtml(agent.observation_frames || 4)} 帧 / ${escapeHtml(agent.observation_seconds || 1.5)} 秒</li>
        <li>最低置信度：${escapeHtml(agent.min_confidence || 0.72)}</li>
        <li>执行架构：${escapeHtml(agent.execution_architecture || "page_state_graph_v1")}</li>
        <li>模型职责：${escapeHtml(agent.model_role || "observation_only")}</li>
        <li>最大动作数：${escapeHtml(agent.max_actions || 160)}</li>
      </ul>
      <p class="muted">模型只报告页面状态、可见文字和目标位置。控制器每轮只选择并执行一个白名单动作，然后重新观察验证。</p>
    </section>
  </div>`;
}

function currentRoute() {
  const hash = location.hash || "#/";
  if (hash.includes("/apps/wechat")) return "wechat";
  if (hash.includes("/apps/douyin")) return "douyin";
  if (hash.includes("/tasks")) return "tasks";
  if (hash.includes("/settings")) return "settings";
  return "home";
}

function bindViewEvents() {
  document.querySelector("#parseAgent")?.addEventListener("click", parseAgent);
  document.querySelector("#dryRunAgent")?.addEventListener("click", dryRunAgent);
  document.querySelector("#startSupervisedAgent")?.addEventListener("click", startSupervisedAgent);
  bindSupervisedEvents();
  document.querySelector("#agentText")?.addEventListener("keydown", event => {
    if (event.key === "Enter") parseAgent();
  });
  document.querySelectorAll(".operation-submit").forEach(button => {
    button.addEventListener("click", () => createOperationTask(button.dataset.operation));
  });
}

function bindSupervisedEvents() {
  document.querySelector("#advanceSupervisedAgent")?.addEventListener("click", advanceSupervisedAgent);
  document.querySelector("#autoSupervisedAgent")?.addEventListener("click", autoSupervisedAgent);
  document.querySelector("#nextSupervisedAgent")?.addEventListener("click", nextSupervisedAgent);
  document.querySelector("#cancelSupervisedAgent")?.addEventListener("click", cancelSupervisedAgent);
}

async function autoSupervisedAgent() {
  const session = state.supervisedSession;
  if (!session) return;
  const button = document.querySelector("#autoSupervisedAgent");
  if (button) button.disabled = true;
  const result = document.querySelector("#agentResult");
  result.className = "agent-result show";
  result.textContent = "正在逐步观察、执行和复核安全导航；遇到账号动作或未知状态会自动暂停……";
  try {
    const response = await withVisionProgress("准备自动观察", () =>
      api(`/api/agent/generic-supervised/${session.session_id}/auto`, {
        method: "POST",
        body: JSON.stringify({ confirmed: true, max_physical_actions: 4 }),
      })
    );
    state.supervisedSession = response.session;
    result.innerHTML = supervisedSessionHtml(state.supervisedSession);
    bindSupervisedEvents();
  } catch (error) {
    result.className = "agent-result show error";
    result.innerHTML = `<b>安全自动推进已停止</b><p>${escapeHtml(error.message)}</p>`;
  }
}

async function startSupervisedAgent() {
  const input = document.querySelector("#agentText");
  const result = document.querySelector("#agentResult");
  const text = input.value.trim();
  if (!text) return toast("请先输入任务目标。", true);
  result.className = "agent-result show";
  result.textContent = "正在解析目标并采集4帧真实画面；不会执行机械臂动作……";
  try {
    const response = await withVisionProgress("解析目标并采集画面", () =>
      api("/api/agent/generic-supervised/start", {
        method: "POST",
        body: JSON.stringify({ text }),
      })
    );
    state.supervisedSession = response.session;
    result.innerHTML = supervisedSessionHtml(state.supervisedSession);
    bindSupervisedEvents();
  } catch (error) {
    result.className = "agent-result show error";
    result.textContent = error.message;
  }
}

async function advanceSupervisedAgent() {
  const session = state.supervisedSession;
  if (!session) return;
  const button = document.querySelector("#advanceSupervisedAgent");
  if (button) button.disabled = true;
  try {
    const response = await withVisionProgress("执行单步并重新观察", () =>
      api(`/api/agent/generic-supervised/${session.session_id}/confirm`, {
        method: "POST",
        body: JSON.stringify({ confirmed: true }),
      })
    );
    state.supervisedSession = response.session;
    const result = document.querySelector("#agentResult");
    result.className = "agent-result show";
    result.innerHTML = supervisedSessionHtml(state.supervisedSession);
    bindSupervisedEvents();
  } catch (error) {
    const result = document.querySelector("#agentResult");
    result.className = "agent-result show error";
    result.innerHTML = `<b>当前单步已停止</b><p>${escapeHtml(error.message)}</p>`;
  }
}

async function nextSupervisedAgent() {
  const session = state.supervisedSession;
  if (!session) return;
  const button = document.querySelector("#nextSupervisedAgent");
  if (button) button.disabled = true;
  try {
    const response = await withVisionProgress("重新采集并观察页面", () =>
      api(`/api/agent/generic-supervised/${session.session_id}/next`, {
        method: "POST",
      })
    );
    state.supervisedSession = response.session;
    const result = document.querySelector("#agentResult");
    result.className = "agent-result show";
    result.innerHTML = supervisedSessionHtml(state.supervisedSession);
    bindSupervisedEvents();
  } catch (error) {
    const result = document.querySelector("#agentResult");
    result.className = "agent-result show error";
    result.innerHTML = `<b>下一步规划已停止</b><p>${escapeHtml(error.message)}</p>`;
  }
}

async function cancelSupervisedAgent() {
  const session = state.supervisedSession;
  if (!session) return;
  try {
    const response = await api(`/api/agent/generic-supervised/${session.session_id}/cancel`, {
      method: "POST",
    });
    state.supervisedSession = response.session;
    const result = document.querySelector("#agentResult");
    result.className = "agent-result show";
    result.innerHTML = supervisedSessionHtml(state.supervisedSession);
  } catch (error) {
    toast(error.message, true);
  }
}

function render() {
  const route = currentRoute();
  const titles = { home: "机械臂总览", wechat: "微信工作流", douyin: "抖音工作流", tasks: "任务记录", settings: "设置与标定" };
  document.querySelector("#pageTitle").textContent = titles[route];
  document.querySelectorAll("nav a").forEach(link => link.classList.toggle("active", link.dataset.route === route));
  const view = document.querySelector("#view");
  view.innerHTML = ({ home: homeView, wechat: wechatView, douyin: douyinView, tasks: tasksView, settings: settingsView })[route]();
  bindViewEvents();
  renderStatus();
}

async function parseAgent() {
  const input = document.querySelector("#agentText");
  const result = document.querySelector("#agentResult");
  const text = input.value.trim();
  if (!text) return;
  try {
    const draft = await api("/api/agent/parse", { method: "POST", body: JSON.stringify({ text }) });
    result.className = `agent-result show${draft.understood ? "" : " error"}`;
    if (!draft.understood) {
      result.textContent = draft.message;
      return;
    }
    result.innerHTML = `
      <b>${escapeHtml(draft.summary)}</b>
      <p class="muted">解析器：${escapeHtml(draft.provider || "local_rule")}。已映射为 ${escapeHtml(draft.operation)}；不包含线性计划或坐标。确认后才会操作真实手机。</p>
      <button id="createDraft" class="secondary-button">创建待确认任务</button>`;
    document.querySelector("#createDraft").onclick = () => createTask(draft);
  } catch (error) {
    result.className = "agent-result show error";
    result.textContent = error.message;
  }
}

async function dryRunAgent() {
  const input = document.querySelector("#agentText");
  const result = document.querySelector("#agentResult");
  const text = input.value.trim();
  if (!text) return;
  result.className = "agent-result show";
  result.textContent = "正在采集4帧真实画面并让千问只观察页面……";
  try {
    const preview = await api("/api/agent/dry-run-step", {
      method: "POST",
      body: JSON.stringify({ text }),
    });
    if (!preview.compiled && preview.intent) {
      result.className = "agent-result show error";
      result.textContent = preview.intent.message || "无法明确理解任务。";
      return;
    }
    const observation = preview.observation || {};
    const decision = preview.decision || {};
    const nextAction = decision.action;
    result.className = `agent-result show${decision.status === "failed" ? " error" : ""}`;
    result.innerHTML = `
      <b>真实观察：${escapeHtml(observation.page_state || "unknown")}</b>
      <p>稳定：${observation.stable ? "是" : "否"}；置信度：${escapeHtml(observation.confidence)}</p>
      <p>${nextAction
        ? `准备动作：${escapeHtml(nextAction.action)}（节点 ${escapeHtml(nextAction.node_id)}）`
        : `没有可执行动作：${escapeHtml(decision.reason || decision.status)}`}</p>
      <p class="muted">Dry-run：已采集 ${escapeHtml(preview.captured_frames)} 帧；机械臂动作=0，任务创建=0。</p>`;
  } catch (error) {
    result.className = "agent-result show error";
    result.textContent = error.message;
  }
}

async function createOperationTask(operation) {
  const appId = operation.startsWith("wechat.") ? "wechat" : "douyin";
  let params = {};
  if (operation === "wechat.send_text") {
    params = {
      chat_name: document.getElementById(`${operation}-chat`).value.trim(),
      text: document.getElementById(`${operation}-text`).value.trim(),
    };
    if (!params.chat_name || !params.text) return toast("请填写聊天名称和消息正文。", true);
  } else if (operation === "wechat.send_album_image") {
    params = {
      chat_name: document.getElementById(`${operation}-chat`).value.trim(),
      image_index: Number(document.getElementById(`${operation}-index`).value),
    };
    if (!params.chat_name) return toast("请填写聊天名称。", true);
  } else if (operation === "douyin.search") {
    params = { keyword: document.getElementById(`${operation}-keyword`).value.trim() };
    if (!params.keyword) return toast("请填写搜索关键词。", true);
  } else if (operation === "douyin.batch_interact") {
    params = {
      keyword: document.getElementById(`${operation}-keyword`).value.trim() || null,
      target_count: Number(document.getElementById(`${operation}-count`).value),
      like: document.getElementById(`${operation}-like`).checked,
      comment: document.getElementById(`${operation}-comment`).checked,
      comment_text: document.getElementById(`${operation}-comment-text`).value.trim(),
    };
    if (!params.like && !params.comment) return toast("至少选择点赞或评论之一。", true);
    if (params.comment && !params.comment_text) return toast("启用评论时必须填写评论原文。", true);
  }
  await createTask({ app_id: appId, operation, params });
}

async function createTask(draft) {
  try {
    const task = await api("/api/tasks", {
      method: "POST",
      body: JSON.stringify({
        app_id: draft.app_id,
        operation: draft.operation,
        params: draft.params || {},
      }),
    });
    state.pendingTask = task;
    const source = task.params.source_params || task.params;
    const suffix = task.params.summary
      ? `\n目标：${task.params.summary}`
      : (task.params.goal ? `\n目标：${task.params.goal}` : `\n参数：${JSON.stringify(source)}`);
    const planText = (task.params.execution_plan || [])
      .map((step, index) => `\n${index + 1}. ${step.label}（验收：${step.checkpoint}）`)
      .join("");
    document.querySelector("#confirmSummary").textContent =
      `${operationNames[task.operation] || task.operation}${suffix}` +
      (task.params.workflow_version === "page_state_graph_v1"
        ? "\n执行方式：页面状态图；每次观察后只执行一个控制器动作。"
        : planText);
    document.querySelector("#confirmDialog").showModal();
    await refreshTasks();
  } catch (error) {
    toast(error.message, true);
  }
}

async function confirmPendingTask() {
  if (!state.pendingTask) return;
  const task = state.pendingTask;
  state.pendingTask = null;
  try {
    await api(`/api/tasks/${task.id}/confirm`, { method: "POST" });
    toast("任务已确认并进入执行队列。");
    await refreshAll();
  } catch (error) {
    toast(error.message, true);
  }
}

async function stopTasks() {
  try {
    const result = await api("/api/stop", { method: "POST" });
    toast(result.note);
    await refreshAll();
  } catch (error) {
    toast(error.message, true);
  }
}

async function refreshTasks() {
  const data = await api("/api/tasks?limit=100");
  state.tasks = data.tasks;
  render();
}

async function refreshAll() {
  const [appsData, deviceData, tasksData] = await Promise.all([
    api("/api/apps"),
    api("/api/device"),
    api("/api/tasks?limit=100"),
  ]);
  state.apps = appsData.apps;
  state.readiness = appsData.readiness;
  state.device = deviceData;
  state.tasks = tasksData.tasks;
  const activeSession = deviceData.generic_supervised_execution?.active_sessions?.[0];
  if (activeSession?.session_id) {
    try {
      const sessionData = await api(`/api/agent/generic-supervised/${activeSession.session_id}`);
      state.supervisedSession = sessionData.session;
    } catch (_error) {
      state.supervisedSession = null;
    }
  }
  render();
}

function connectEvents() {
  const source = new EventSource("/api/events");
  source.addEventListener("snapshot", event => {
    const payload = JSON.parse(event.data);
    state.tasks = payload.tasks;
    render();
  });
  source.onerror = () => {
    source.close();
    setTimeout(connectEvents, 2500);
  };
}

async function init() {
  try {
    const session = await api("/api/session");
    state.token = session.token;
    state.mock = session.mock;
    const mode = document.querySelector("#modeBadge");
    mode.textContent = session.mock ? "模拟模式 · 无实机动作" : "实机模式";
    if (!session.mock) mode.style.color = "#ffd166";
    await refreshAll();
    connectEvents();
    setInterval(() => {
      const preview = document.querySelector("#phonePreview");
      if (preview) preview.src = `/api/preview.jpg?t=${Date.now()}`;
    }, 350);
  } catch (error) {
    toast(`连接本地服务失败：${error.message}`, true);
  }
}

window.addEventListener("hashchange", render);
document.querySelector("#stopButton").addEventListener("click", stopTasks);
document.querySelector("#confirmDialog").addEventListener("close", event => {
  if (event.target.returnValue === "default") confirmPendingTask();
  else state.pendingTask = null;
});

init();
