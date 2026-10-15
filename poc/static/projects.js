/* Projects are organization labels; resource ownership remains user-scoped. */
window.OrganizationProjects = {
  mount(api, changed) {
    const button=document.querySelector('#projectManagementButton'), dialog=document.querySelector('#projectManagementDialog');
    const list=document.querySelector('#projectList'), message=document.querySelector('#projectMessage'), create=document.querySelector('#projectCreate');
    const filter=document.querySelector('#projectFilter'), newId=document.querySelector('#projectNewId'), newName=document.querySelector('#projectNewName'), newDescription=document.querySelector('#projectNewDescription'), newMembers=document.querySelector('#projectNewMembers');
    let organization='', identity='', generation=0, pendingCreate=null;
    const base=()=>`/api/control-plane/organizations/${encodeURIComponent(organization)}`;
    const tell=text=>{message.textContent=text;};
    const selected=node=>[...node.querySelectorAll('input[type=checkbox]:checked')].map(el=>el.value);
    const picker=(node, people, ids, title)=>{
      node.replaceChildren(); const legend=document.createElement('legend'); legend.textContent=title; const search=document.createElement('input');
      search.type='search'; search.placeholder='搜索成员 ID'; search.setAttribute('aria-label',`${title}搜索`); node.append(legend,search);
      for(const person of people){const label=document.createElement('label'); const box=document.createElement('input'); box.type='checkbox'; box.value=person.user_id; box.checked=ids.includes(person.user_id); const text=document.createElement('span'); text.textContent=`${person.user_id}${person.status==='disabled'?'（已停用）':''}`; label.append(box,text); node.append(label);}
      search.addEventListener('input',()=>node.querySelectorAll('label').forEach(label=>{label.hidden=!label.textContent.toLowerCase().includes(search.value.trim().toLowerCase());}));
    };
    const applyFilter=()=>{const query=filter.value.trim().toLowerCase();for(const row of list.children)row.hidden=!row.dataset.search.includes(query);};
    const load=async()=>{
      const epoch=++generation; create.hidden=true; list.replaceChildren(); tell('正在读取项目…');
      try{
        const [projects, people]=await Promise.all([api(`${base()}/projects`),api(`${base()}/principals`)]);
        if(epoch!==generation||!organization)return false; const items=projects.data.projects, persons=people.data.principals;
        if(!Array.isArray(items)||!Array.isArray(persons))throw new Error('项目响应无效'); picker(newMembers,persons,[],'新项目成员'); create.hidden=false;
        for(const project of items){
          const form=document.createElement('form'); form.className='device-assignment-card project-card'; form.dataset.project=project.project_id; form.dataset.search=[project.project_id,project.name,project.description,...project.member_ids].join(' ').toLowerCase();
          const title=document.createElement('strong'); title.textContent=`${project.name} · ${project.status==='archived'?'已归档':'启用'} · revision ${project.revision}`;
          const id=document.createElement('small'); id.textContent=`ID：${project.project_id}`;
          const name=document.createElement('input'); name.value=project.name; name.required=true; name.maxLength=256; name.setAttribute('aria-label',`${project.project_id} 名称`);
          const description=document.createElement('textarea'); description.value=project.description; description.maxLength=2000; description.setAttribute('aria-label',`${project.project_id} 说明`);
          const status=document.createElement('select'); status.setAttribute('aria-label',`${project.project_id} 状态`); status.append(new Option('启用','active'),new Option('归档（保留历史）','archived')); status.value=project.status;
          const choices=document.createElement('fieldset'); choices.className='project-member-picker'; picker(choices,persons,project.member_ids,`${project.name}成员`);
          const save=document.createElement('button'); save.type='submit'; save.className='primary-button'; save.textContent='保存项目'; let pending=null;
          form.addEventListener('submit',async event=>{event.preventDefault();if(save.disabled||epoch!==generation)return;const body=JSON.stringify({name:name.value.trim(),description:description.value,status:status.value,member_ids:selected(choices),expected_revision:project.revision});if(pending?.body!==body)pending={body,key:crypto.randomUUID()};save.disabled=name.disabled=description.disabled=status.disabled=choices.disabled=true;tell('正在保存项目…');let saved=false;try{await api(`${base()}/projects/${encodeURIComponent(project.project_id)}`,{method:'PATCH',headers:{'Idempotency-Key':pending.key},body});saved=true;if(epoch!==generation)return;await changed();if(epoch!==generation)return;if(await load())tell('项目已保存。');}catch(error){if(epoch!==generation)return;if(error.code==='project_conflict')await load();tell(`${saved?'已保存，但刷新失败':'未保存'}：${error.message}`);}finally{save.disabled=name.disabled=description.disabled=status.disabled=choices.disabled=false;}});
          form.append(title,id,name,description,status,choices,save); list.append(form);
        }
        applyFilter(); tell(`共 ${items.length} 个项目。归档项目仍保留历史引用。`); return true;
      }catch(error){if(epoch===generation)tell(`读取失败：${error.message}。可点击刷新重试。`);return false;}
    };
    create.addEventListener('submit',async event=>{event.preventDefault();const save=create.querySelector('button');if(save.disabled)return;const selection={project_id:newId.value.trim(),name:newName.value.trim(),description:newDescription.value,member_ids:selected(newMembers)};if(pendingCreate?.body!==JSON.stringify(selection))pendingCreate={body:JSON.stringify(selection),key:crypto.randomUUID()};save.disabled=newId.disabled=newName.disabled=newDescription.disabled=newMembers.disabled=true;tell('正在创建项目…');let saved=false;try{await api(`${base()}/projects`,{method:'POST',headers:{'Idempotency-Key':pendingCreate.key},body:pendingCreate.body});saved=true;pendingCreate=null;newId.value=newName.value=newDescription.value='';await changed();if(await load())tell('项目已创建。');}catch(error){tell(`${saved?'已创建，但刷新失败':'未创建'}：${error.message}`);}finally{save.disabled=newId.disabled=newName.disabled=newDescription.disabled=newMembers.disabled=false;}});
    button.addEventListener('click',()=>{dialog.showModal();load();}); filter.addEventListener('input',applyFilter); document.querySelector('#projectRefresh').addEventListener('click',load); document.querySelector('#projectClose').addEventListener('click',()=>dialog.close());
    return {account(value){const allowed=value?.authenticated&&value.session.permissions?.includes('team.manage');const next=allowed?`${value.session.organization_id}/${value.session.user_id}/${value.session.role}`:'';if(identity===next)return;identity=next;generation++;dialog.close();list.replaceChildren();newMembers.replaceChildren();tell('');create.hidden=true;newId.value=newName.value=newDescription.value=filter.value='';pendingCreate=null;organization=allowed?value.session.organization_id:'';button.hidden=!organization;}};
  }
};
