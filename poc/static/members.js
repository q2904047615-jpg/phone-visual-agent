/* Member permissions are supplied by the server, including per-row roles. */
window.OrganizationMembers = {
  mount(api, changed) {
    const button = document.querySelector('#memberManagementButton');
    const dialog = document.querySelector('#memberManagementDialog');
    const list = document.querySelector('#memberList');
    const message = document.querySelector('#memberMessage');
    const create = document.querySelector('#memberCreate');
    const newRole = document.querySelector('#memberNewRole');
    const newId = document.querySelector('#memberNewId');
    const labels = {viewer:'只读成员', operator:'操作员', admin:'管理员', owner:'所有者'};
    let organization = '', identity = '', generation = 0, createPending = null;
    const tell = text => { message.textContent = text; };
    const base = () => `/api/control-plane/organizations/${encodeURIComponent(organization)}/principals`;
    const roles = (select, values) => { select.replaceChildren(); for (const role of values) select.add(new Option(labels[role] || role, role)); };
    const load = async () => {
      const epoch = ++generation;
      create.hidden = true; list.replaceChildren(); tell('正在读取组织成员…');
      try {
        const result = (await api(base())).data;
        if (epoch !== generation || !organization) return;
        if (!Array.isArray(result.principals) || !Array.isArray(result.create_roles)) throw new Error('成员响应无效');
        roles(newRole, result.create_roles); create.hidden = !result.create_roles.length;
        tell(`共 ${result.principals.length} 名成员`);
        for (const person of result.principals) {
          const form = document.createElement('form'); form.className = 'device-assignment-card'; form.dataset.member = person.user_id;
          const title = document.createElement('strong'); title.textContent = person.user_id;
          const summary = document.createElement('p'); summary.textContent = `${labels[person.role]} · ${person.status === 'disabled' ? '已停用' : '启用中'}`;
          form.append(title, summary);
          if (person.editable_roles?.length) {
            const role = document.createElement('select'); role.setAttribute('aria-label', `${person.user_id} 角色`);
            roles(role, person.editable_roles); role.value = person.role;
            const status = document.createElement('select'); status.setAttribute('aria-label', `${person.user_id} 状态`);
            status.add(new Option('启用', 'active')); status.add(new Option('停用（撤销全部登录）', 'disabled')); status.value = person.status;
            const save = document.createElement('button'); save.type = 'submit'; save.className = 'primary-button'; save.textContent = '保存成员';
            let pending = null;
            form.addEventListener('submit', async event => {
              event.preventDefault(); if (save.disabled) return;
              const body = JSON.stringify({role:role.value, status:status.value, expected_revision:person.revision});
              if (pending?.body !== body) pending = {body, key:crypto.randomUUID()};
              save.disabled = role.disabled = status.disabled = true; tell('正在保存…');
              let saved = false;
              try {
                await api(`${base()}/${encodeURIComponent(person.user_id)}`, {method:'PUT', headers:{'Idempotency-Key':pending.key}, body});
                saved = true;
                if (epoch !== generation) return;
                await changed();
                if (epoch !== generation) return;
                await load(); tell('成员已保存。停用会撤销全部登录；恢复后需重新登录。');
              } catch (error) {
                if (epoch !== generation) return;
                if (error.code === 'member_conflict') await load();
                tell(`${saved ? '已保存，但刷新失败' : '未保存'}：${error.message}`);
              } finally { save.disabled = role.disabled = status.disabled = false; }
            });
            form.append(role, status, save);
          } else {
            const note = document.createElement('p'); note.textContent = '此角色由所有者管理。'; form.append(note);
          }
          list.append(form);
        }
      } catch (error) { if (epoch === generation) tell(`读取失败：${error.message}。可点击刷新重试。`); }
    };
    create.addEventListener('submit', async event => {
      event.preventDefault();
      const save = create.querySelector('button'); if (save.disabled) return;
      const epoch = generation;
      const body = JSON.stringify({user_id:newId.value.trim(), role:newRole.value, team_ids:[]});
      if (createPending?.body !== body) createPending = {body, key:crypto.randomUUID()};
      save.disabled = newId.disabled = newRole.disabled = true; tell('正在创建…');
      let saved = false;
      try {
        await api(base(), {method:'POST', headers:{'Idempotency-Key':createPending.key}, body});
        saved = true;
        if (epoch !== generation) return;
        createPending = null; newId.value = '';
        await load(); tell('成员已创建。请从“管理 GitHub 关联”关联身份，并分配可用设备。');
      } catch (error) { if (epoch === generation) tell(`${saved ? '已创建，但刷新失败' : '未创建'}：${error.message}`); }
      finally { save.disabled = newId.disabled = newRole.disabled = false; }
    });
    button.addEventListener('click', () => { dialog.showModal(); load(); });
    document.querySelector('#memberRefresh').addEventListener('click', load);
    document.querySelector('#memberClose').addEventListener('click', () => dialog.close());
    return {account(value) {
      const allowed = value?.authenticated && value.session.permissions?.includes('identity.manage');
      const next = allowed ? `${value.session.organization_id}/${value.session.user_id}/${value.session.role}` : '';
      if (identity === next) return;
      identity = next; generation++; dialog.close(); list.replaceChildren(); tell('');
      create.hidden = true; newId.value = ''; newRole.replaceChildren(); createPending = null;
      organization = allowed ? value.session.organization_id : ''; button.hidden = !organization;
    }};
  }
};
