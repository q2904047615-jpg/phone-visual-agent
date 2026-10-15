/* Read-only Runtime billing projection for the Control Plane. */
window.OrganizationBilling = {
  mount(api) {
    const button = document.querySelector('#billingViewerButton');
    const dialog = document.querySelector('#billingViewerDialog');
    const list = document.querySelector('#billingList');
    const message = document.querySelector('#billingMessage');
    let organization = '', identity = '', generation = 0;
    const base = () => `/api/control-plane/organizations/${encodeURIComponent(organization)}`;
    const money = micros => `${(Number(micros || 0) / 1000000).toFixed(6)} CNY`;
    const load = async () => {
      const epoch = ++generation;
      list.replaceChildren(); message.textContent = '正在读取计费账本…';
      try {
        const response = await api(`${base()}/billing?limit=100`);
        if (epoch !== generation || !organization) return;
        const entries = response.data?.entries;
        if (!Array.isArray(entries)) throw new Error('计费响应无效');
        if (!entries.length) {
          const empty = document.createElement('p'); empty.textContent = '当前还没有已结算运行。'; list.append(empty);
        }
        for (const entry of entries) {
          const item = document.createElement('article'); item.className = 'audit-entry';
          const title = document.createElement('strong'); title.textContent = `${entry.session_id || '未知会话'} · ${entry.status || 'unknown'}`;
          const detail = document.createElement('small'); detail.textContent = `${entry.model || 'unknown'} · Token ${entry.total_tokens ?? 0} · 物理动作 ${entry.physical_actions ?? 0} · ${money(entry.cost_micros)}`;
          const meta = document.createElement('code'); meta.textContent = `${entry.created_at || ''} · ${entry.pricing_version || '无定价版本'}`;
          item.append(title, detail, meta); list.append(item);
        }
        message.textContent = `共 ${entries.length} 条已结算记录。`;
      } catch (error) { if (epoch === generation) message.textContent = `读取失败：${error.message}`; }
    };
    button.addEventListener('click', () => { dialog.showModal(); load(); });
    document.querySelector('#billingRefresh').addEventListener('click', load);
    document.querySelector('#billingClose').addEventListener('click', () => dialog.close());
    return { account(value) {
      const session = value?.authenticated ? value.session : null;
      const allowed = Boolean(session && (session.permissions || []).includes('billing.read'));
      const next = allowed ? `${session.organization_id}/${session.user_id}` : '';
      if (next === identity) return;
      identity = next; generation++; dialog.close(); list.replaceChildren(); message.textContent = '';
      organization = allowed ? session.organization_id : ''; button.hidden = !allowed;
    }};
  }
};
