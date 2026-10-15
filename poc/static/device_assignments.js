/* Assignment metadata only. Task history never moves with a phone. */
window.DeviceAssignments = {
  mount(api, changed) {
    const button = document.querySelector('#deviceManagementButton');
    const dialog = document.querySelector('#deviceManagementDialog');
    const list = document.querySelector('#deviceAssignmentList');
    const message = document.querySelector('#deviceAssignmentMessage');
    let organization = '', identity = '', generation = 0;
    const tell = text => { message.textContent = text; };
    const load = async () => {
      const epoch = ++generation;
      const base = `/api/control-plane/organizations/${encodeURIComponent(organization)}`;
      tell('正在读取设备和成员…');
      list.replaceChildren();
      try {
        const [inventory, members] = await Promise.all([api(base + '/devices'), api(base + '/principals')]);
        if (epoch !== generation || !organization) return;
        const devices = inventory.data.devices;
        if (!Array.isArray(devices) || !Array.isArray(members.data.principals)) throw new Error('设备管理响应无效');
        tell(devices.length ? '' : '当前组织没有已登记设备。');
        for (const device of devices) {
          const form = document.createElement('form'); form.className = 'device-assignment-card';
          const title = document.createElement('strong'); title.textContent = device.device_id;
          const current = document.createElement('p'); current.textContent = `当前使用者：${device.user_id || '未分配'}`;
          const select = document.createElement('select'); select.setAttribute('aria-label', `${device.device_id} 使用者`);
          select.add(new Option('未分配（暂停使用）', ''));
          for (const person of members.data.principals) {
            if (person.status === 'disabled' && person.user_id !== device.user_id) continue;
            const option = new Option(`${person.user_id} · ${person.role}${person.status === 'disabled' ? '（已停用）' : ''}`, person.user_id);
            option.disabled = person.status === 'disabled'; select.add(option);
          }
          select.value = device.user_id || '';
          const save = document.createElement('button'); save.type = 'submit'; save.className = 'primary-button'; save.textContent = '保存分配';
          save.disabled = !device.registered;
          if (!device.registered) current.textContent += ' · 设备未启用，暂不可分配';
          let pending = null;
          form.addEventListener('submit', async event => {
            event.preventDefault(); if (save.disabled) return;
            const body = JSON.stringify({user_id: select.value || null, expected_revision: device.revision});
            if (pending?.body !== body) pending = { body, key: crypto.randomUUID() };
            save.disabled = true; select.disabled = true; tell('正在保存…');
            let saved = false;
            try {
              await api(`${base}/devices/${encodeURIComponent(device.device_id)}/assignment`,
                {method:'PUT', headers:{'Idempotency-Key':pending.key}, body:pending.body});
              saved = true;
              if (epoch !== generation) return;
              await changed();
              await load(); tell('分配已保存。历史任务仍归原用户。');
            } catch (error) {
              if (epoch !== generation) return;
              if (error.code === 'assignment_conflict') await load();
              tell(`${saved ? '分配已保存，但刷新失败' : '未保存'}：${error.message}`);
            } finally { save.disabled = !device.registered; select.disabled = false; }
          });
          form.append(title, current, select, save); list.append(form);
        }
      } catch (error) { if (epoch === generation) tell(`读取失败：${error.message}。可点击刷新重试。`); }
    };
    button.addEventListener('click', () => { dialog.showModal(); load(); });
    document.querySelector('#deviceAssignmentRefresh').addEventListener('click', load);
    document.querySelector('#deviceAssignmentClose').addEventListener('click', () => dialog.close());
    return { account(value) {
      const allowed = value?.authenticated && value.session.permissions?.includes('device.manage');
      const next = allowed ? `${value.session.organization_id}/${value.session.user_id}` : '';
      if (identity === next) return;
      identity = next;
      generation++;
      dialog.close(); list.replaceChildren(); tell('');
      organization = allowed ? value.session.organization_id : '';
      button.hidden = !organization;
    }};
  }
};
