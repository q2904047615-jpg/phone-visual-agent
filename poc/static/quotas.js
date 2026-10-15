/* Control Plane quota projection. Runtime admission is a separate contract. */
window.OrganizationQuotas = {
  mount(api, changed = () => {}) {
    const button = document.querySelector('#quotaManagementButton');
    const dialog = document.querySelector('#quotaManagementDialog');
    const list = document.querySelector('#quotaList');
    const message = document.querySelector('#quotaMessage');
    let organization = '', identity = '', generation = 0, canManage = false;
    const base = () => `/api/control-plane/organizations/${encodeURIComponent(organization)}`;
    const tell = text => { message.textContent = text; };
    const number = value => value === null || value === undefined ? '' : String(value);
    const load = async () => {
      const epoch = ++generation;
      list.replaceChildren(); tell('正在读取配额…');
      try {
        const response = await api(`${base()}/quotas`);
        if (epoch !== generation || !organization) return false;
        const quotas = response.data?.quotas;
        if (!Array.isArray(quotas)) throw new Error('配额响应无效');
        for (const quota of quotas) {
          const form = document.createElement('form');
          form.className = 'device-assignment-card quota-card';
          form.dataset.quota = quota.quota_key;
          const title = document.createElement('strong');
          title.textContent = `${quota.quota_key} · ${quota.period === 'monthly' ? '每月' : '每日'}`;
          const usage = document.createElement('small');
          usage.textContent = `当前使用：运行 ${quota.usage?.runs ?? 0} · 动作 ${quota.usage?.physical_actions ?? 0} · Token ${quota.usage?.model_tokens ?? 0}`;
          const period = document.createElement('select');
          period.setAttribute('aria-label', `${quota.quota_key} 周期`);
          period.append(new Option('每日', 'daily'), new Option('每月', 'monthly'));
          period.value = quota.period;
          const fields = [];
          for (const [key, label] of [['max_runs', '运行上限'], ['max_physical_actions', '物理动作上限'], ['max_model_tokens', '模型 Token 上限']]) {
            const labelNode = document.createElement('label');
            labelNode.textContent = label;
            const input = document.createElement('input');
            input.type = 'number'; input.min = '0'; input.step = '1'; input.value = number(quota[key]);
            input.required = true; input.setAttribute('aria-label', `${quota.quota_key} ${label}`);
            input.disabled = !canManage;
            labelNode.append(input); fields.push([key, input]); form.append(labelNode);
          }
          const save = document.createElement('button'); save.type = 'submit'; save.className = 'primary-button'; save.textContent = '保存配额'; save.disabled = !canManage;
          form.addEventListener('submit', async event => {
            event.preventDefault(); if (save.disabled) return;
            const body = { quota_key: quota.quota_key, period: period.value };
            for (const [key, input] of fields) body[key] = Number(input.value);
            save.disabled = true; tell(`正在保存 ${quota.quota_key}…`);
            try {
              await api(`${base()}/quotas/${encodeURIComponent(quota.quota_key)}`, { method: 'PUT', headers: { 'Idempotency-Key': crypto.randomUUID() }, body: JSON.stringify(body) });
              await changed(); await load(); tell(`配额 ${quota.quota_key} 已保存。`);
            } catch (error) { tell(`未保存：${error.message}`); }
            finally { save.disabled = false; }
          });
          form.prepend(title, usage, period); form.append(save); list.append(form);
        }
        tell(`共 ${quotas.length} 项配额。`); return true;
      } catch (error) { if (epoch === generation) tell(`读取失败：${error.message}`); return false; }
    };
    button.addEventListener('click', () => { dialog.showModal(); load(); });
    document.querySelector('#quotaRefresh').addEventListener('click', load);
    document.querySelector('#quotaClose').addEventListener('click', () => dialog.close());
    return { account(value) {
      const session = value?.authenticated ? value.session : null;
      const permissions = session?.permissions || [];
      const allowed = Boolean(session && permissions.includes('quota.read'));
      canManage = Boolean(session && permissions.includes('quota.manage'));
      const next = allowed ? `${session.organization_id}/${session.user_id}/${permissions.join(',')}` : '';
      if (next === identity) return;
      identity = next; generation++; dialog.close(); list.replaceChildren(); tell('');
      organization = allowed ? session.organization_id : '';
      button.hidden = !allowed;
    }};
  }
};
