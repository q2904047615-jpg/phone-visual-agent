/* Teams organize people without granting access to other people's resources. */
window.OrganizationTeams = {
  mount(api, changed) {
    const button = document.querySelector('#teamManagementButton');
    const dialog = document.querySelector('#teamManagementDialog');
    const list = document.querySelector('#teamList');
    const message = document.querySelector('#teamMessage');
    const create = document.querySelector('#teamCreate');
    const name = document.querySelector('#teamNewName');
    const members = document.querySelector('#teamNewMembers');
    const summary = document.querySelector('#teamMemberSummary');
    const filter = document.querySelector('#teamFilter');
    let organization = '', identity = '', generation = 0, pendingCreate = null;
    const base = () => `/api/control-plane/organizations/${encodeURIComponent(organization)}`;
    const tell = text => { message.textContent = text; };
    const selected = node => [...node.querySelectorAll('input[type=checkbox]:checked')].map(el=>el.value);
    function picker(node, people, ids, title) {
      node.replaceChildren();
      const legend = document.createElement('legend'); legend.textContent = title;
      const search = document.createElement('input'); search.type = 'search'; search.placeholder = '搜索成员 ID'; search.setAttribute('aria-label', `${title}搜索`);
      node.append(legend, search);
      for (const person of people) {
        const label = document.createElement('label'); label.className = 'team-member-option'; label.dataset.search = person.user_id.toLowerCase();
        const box = document.createElement('input'); box.type = 'checkbox'; box.value = person.user_id; box.checked = ids.includes(person.user_id);
        const text = document.createElement('span'); text.textContent = `${person.user_id}${person.status==='disabled'?'（已停用）':''}`;
        label.append(box, text); node.append(label);
      }
      search.addEventListener('input', () => node.querySelectorAll('label').forEach(label=>{label.hidden=!label.dataset.search.includes(search.value.trim().toLowerCase());}));
    }
    function applyFilter() {
      const query = filter.value.trim().toLowerCase();
      for (const row of list.children) row.hidden = !row.dataset.search.includes(query);
    }
    const load = async () => {
      const epoch = ++generation;
      create.hidden = true; list.replaceChildren(); summary.replaceChildren(); tell('正在读取团队…');
      try {
        const [teams, people] = await Promise.all([api(`${base()}/teams`), api(`${base()}/principals`)]);
        if (epoch !== generation || !organization) return;
        if (!Array.isArray(teams.data.teams) || !Array.isArray(people.data.principals)) throw new Error('团队响应无效');
        const groups = teams.data.teams, persons = people.data.principals;
        picker(members, persons, selected(members), '新团队成员'); create.hidden = false;
        for (const person of persons) {
          const line = document.createElement('p');
          const names = groups.filter(t=>t.member_ids.includes(person.user_id)).map(t=>t.name);
          line.textContent = `${person.user_id}：${names.join('、') || '尚未分组'}`; summary.append(line);
        }
        for (const team of groups) {
          const form = document.createElement('form'); form.className = 'device-assignment-card team-card'; form.dataset.team = team.team_id;
          form.dataset.search = [team.name, team.team_id, ...team.member_ids].join(' ').toLowerCase();
          const title = document.createElement('strong'); title.textContent = `${team.name} · ${team.member_ids.length} 人`;
          const label = document.createElement('label'); label.textContent = '团队名称';
          const input = document.createElement('input'); input.required = true; input.maxLength = 256; input.value = team.name; input.setAttribute('aria-label', `${team.team_id} 名称`); label.append(input);
          const choices = document.createElement('fieldset'); choices.className = 'team-member-picker'; picker(choices, persons, team.member_ids, `${team.name}成员`);
          const save = document.createElement('button'); save.type = 'submit'; save.className = 'primary-button'; save.textContent = '保存团队';
          let pending = null;
          form.addEventListener('submit', async event => {
            event.preventDefault(); if (save.disabled || epoch !== generation) return;
            const body = JSON.stringify({name:input.value.trim(), member_ids:selected(choices), expected_revision:team.revision});
            if (pending?.body !== body) pending = {body, key:crypto.randomUUID()};
            save.disabled = choices.disabled = input.disabled = true; tell('正在保存团队…'); let saved = false;
            try {
              await api(`${base()}/teams/${encodeURIComponent(team.team_id)}`, {method:'PATCH', headers:{'Idempotency-Key':pending.key}, body});
              saved = true; if (epoch !== generation) return;
              await changed(); if (epoch !== generation) return;
              if (await load()) tell('团队已保存，成员分组已更新。');
            } catch (error) {
              if (epoch !== generation) return;
              if (error.code === 'team_conflict') await load();
              tell(`${saved?'已保存，但刷新失败':'未保存'}：${error.message}`);
            } finally { save.disabled = choices.disabled = input.disabled = false; }
          });
          form.append(title, label, choices, save); list.append(form);
        }
        applyFilter(); tell(`共 ${groups.length} 个团队，可让同一成员加入多个团队。`); return true;
      } catch (error) { if (epoch === generation) tell(`读取失败：${error.message}。可点击刷新重试。`); }
    };
    create.addEventListener('submit', async event => {
      event.preventDefault(); const save = create.querySelector('button'); if (save.disabled) return;
      const epoch = generation;
      const selection = JSON.stringify({name:name.value.trim(), member_ids:selected(members)});
      if (pendingCreate?.selection !== selection) pendingCreate = {selection, key:crypto.randomUUID(), body:JSON.stringify({...JSON.parse(selection), team_id:`team-${crypto.randomUUID()}`})};
      save.disabled = name.disabled = members.disabled = true; tell('正在创建团队…'); let saved = false;
      try {
        await api(`${base()}/teams`, {method:'POST', headers:{'Idempotency-Key':pendingCreate.key}, body:pendingCreate.body});
        saved = true; if (epoch !== generation) return;
        pendingCreate = null; name.value = ''; members.replaceChildren();
        await changed(); if (epoch !== generation) return;
        if (await load()) tell('团队已创建。可继续调整名称和成员。');
      } catch (error) { if (epoch === generation) tell(`${saved?'已创建，但刷新失败':'未创建'}：${error.message}`); }
      finally { save.disabled = name.disabled = members.disabled = false; }
    });
    button.addEventListener('click', () => {dialog.showModal(); load();});
    filter.addEventListener('input', applyFilter);
    document.querySelector('#teamRefresh').addEventListener('click', load);
    document.querySelector('#teamClose').addEventListener('click', () => dialog.close());
    return {account(value) {
      const allowed = value?.authenticated && value.session.permissions?.includes('team.manage');
      const next = allowed ? `${value.session.organization_id}/${value.session.user_id}/${value.session.role}` : '';
      if (identity === next) return;
      identity = next; generation++; dialog.close(); list.replaceChildren(); members.replaceChildren(); summary.replaceChildren(); tell('');
      create.hidden = true; name.value = filter.value = ''; pendingCreate = null;
      organization = allowed ? value.session.organization_id : ''; button.hidden = !organization;
    }};
  }
};
