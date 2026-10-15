/* Organization audit is a read-only, cursor based view. */
window.OrganizationAudit = {
  mount(api) {
    const button = document.querySelector('#auditViewerButton');
    const dialog = document.querySelector('#auditViewerDialog');
    const list = document.querySelector('#auditList');
    const message = document.querySelector('#auditMessage');
    const filter = document.querySelector('#auditFilter');
    const more = document.querySelector('#auditMore');
    let organization = '', identity = '', generation = 0, nextSequence = 0, hasMore = false, entries = [];
    const base = () => `/api/control-plane/organizations/${encodeURIComponent(organization)}/audit`;
    const tell = text => { message.textContent = text; };
    const render = () => {
      const query = filter.value.trim().toLowerCase(); list.replaceChildren();
      for (const entry of entries.filter(item => !query || `${item.resource} ${item.outcome} ${item.actor}`.toLowerCase().includes(query))) {
        const row = document.createElement('article'); row.className = 'audit-entry'; row.dataset.sequence = entry.sequence;
        const title = document.createElement('strong'); title.textContent = `#${entry.sequence} · ${entry.resource} · ${entry.outcome}`;
        const meta = document.createElement('small'); meta.textContent = `${entry.occurred_at} · ${entry.actor}`;
        const detail = document.createElement('code'); detail.textContent = JSON.stringify(entry.metadata || {});
        row.append(title, meta, detail); list.append(row);
      }
      more.hidden = !hasMore; if (!entries.length) list.textContent = '暂无审计记录。';
    };
    const load = async ({ append = false } = {}) => {
      const epoch = ++generation; if (!append) { entries = []; nextSequence = 0; }
      tell('正在读取审计…');
      try {
        const url = `${base()}?after_sequence=${nextSequence}&limit=50`;
        const response = await api(url); if (epoch !== generation || !organization) return;
        const page = response.data; if (!Array.isArray(page?.entries)) throw new Error('审计响应无效');
        entries = append ? entries.concat(page.entries) : page.entries; hasMore = Boolean(page.has_more);
        nextSequence = entries.length ? entries[entries.length - 1].sequence : nextSequence; render();
        tell(`已读取 ${entries.length} 条审计记录。`);
      } catch (error) { if (epoch === generation) tell(`读取失败：${error.message}`); }
    };
    button.addEventListener('click', () => { dialog.showModal(); load(); });
    document.querySelector('#auditRefresh').addEventListener('click', () => load());
    more.addEventListener('click', () => load({ append: true }));
    filter.addEventListener('input', render);
    document.querySelector('#auditClose').addEventListener('click', () => dialog.close());
    return { account(value) {
      const session = value?.authenticated ? value.session : null;
      const allowed = Boolean(session && session.permissions?.includes('audit.read'));
      const next = allowed ? `${session.organization_id}/${session.user_id}` : '';
      if (next === identity) return;
      identity = next; generation++; dialog.close(); entries = []; nextSequence = 0; list.replaceChildren(); filter.value = ''; tell('');
      organization = allowed ? session.organization_id : ''; button.hidden = !allowed;
    }};
  }
};
