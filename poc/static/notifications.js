/* User-scoped Control Plane notifications; no cross-user projection. */
window.OrganizationNotifications = {
  mount(api) {
    const button = document.querySelector('#notificationButton');
    const dialog = document.querySelector('#notificationDialog');
    const list = document.querySelector('#notificationList');
    const message = document.querySelector('#notificationMessage');
    let organization = '', identity = '', generation = 0;
    const base = () => `/api/control-plane/organizations/${encodeURIComponent(organization)}/notifications`;
    const tell = text => { message.textContent = text; };
    const load = async () => {
      const epoch = ++generation; list.replaceChildren(); tell('正在读取通知…');
      try {
        const response = await api(base()); if (epoch !== generation || !organization) return;
        const items = response.data?.notifications;
        if (!Array.isArray(items)) throw new Error('通知响应无效');
        if (!items.length) list.textContent = '暂无通知。';
        for (const item of items) {
          const card = document.createElement('article'); card.className = `notification-entry${item.read_at ? ' is-read' : ''}`;
          const title = document.createElement('strong'); title.textContent = item.title;
          const time = document.createElement('small'); time.textContent = item.created_at;
          const body = document.createElement('p'); body.textContent = item.body;
          const read = document.createElement('button'); read.type = 'button'; read.className = 'secondary-button'; read.textContent = item.read_at ? '已读' : '标为已读'; read.disabled = Boolean(item.read_at);
          read.addEventListener('click', async () => {
            read.disabled = true;
            try { await api(`${base()}/${encodeURIComponent(item.notification_id)}/read`, { method: 'POST', body: '{}' }); await load(); }
            catch (error) { read.disabled = false; tell(`未更新：${error.message}`); }
          });
          card.append(title, time, body, read); list.append(card);
        }
        tell(`共 ${items.length} 条通知。`);
      } catch (error) { if (epoch === generation) tell(`读取失败：${error.message}`); }
    };
    button.addEventListener('click', () => { dialog.showModal(); load(); });
    document.querySelector('#notificationRefresh').addEventListener('click', load);
    document.querySelector('#notificationClose').addEventListener('click', () => dialog.close());
    return { account(value) {
      const session = value?.authenticated ? value.session : null;
      const allowed = Boolean(session && session.permissions?.includes('audit.read'));
      const next = allowed ? `${session.organization_id}/${session.user_id}` : '';
      if (next === identity) return;
      identity = next; generation++; dialog.close(); list.replaceChildren(); tell('');
      organization = allowed ? session.organization_id : ''; button.hidden = !allowed;
    }};
  }
};
