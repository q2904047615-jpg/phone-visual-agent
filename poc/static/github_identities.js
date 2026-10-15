window.GitHubIdentities = {
  mount(api, changed) {
    const button=document.querySelector('#githubIdentityButton'), dialog=document.querySelector('#githubIdentityDialog');
    const list=document.querySelector('#githubIdentityList'), message=document.querySelector('#githubIdentityMessage');
    const queryForm=document.querySelector('#githubIdentityLookup'), query=document.querySelector('#githubIdentityQuery');
    const kind=document.querySelector('#githubIdentityKind'), result=document.querySelector('#githubIdentityResult');
    const profile=document.querySelector('#githubIdentityProfile'), target=document.querySelector('#githubIdentityMember');
    const save=document.querySelector('#githubIdentitySave');
    let organization='', identity='', generation=0, querySequence=0, members=[], resolved=null, pending=null;
    const base=()=>`/api/control-plane/organizations/${encodeURIComponent(organization)}/github-identities`;
    const tell=text=>{message.textContent=text;};
    const clearLookup=()=>{querySequence++;resolved=null;pending=null;result.hidden=true;profile.replaceChildren();target.replaceChildren();};
    const load=async()=>{
      const epoch=++generation; clearLookup(); list.replaceChildren(); tell('正在读取 GitHub 关联…');
      try {
        const [identities,people]=await Promise.all([api(base()),api(`/api/control-plane/organizations/${encodeURIComponent(organization)}/principals`)]);
        if(epoch!==generation || !organization)return;
        members=people.data.principals.filter(p=>p.editable_roles?.length);
        queryForm.querySelector('button').disabled=!identities.data.github_available;
        tell(identities.data.github_available ? '先查询账号，核对稳定 ID 与成员后确认关联。' : 'GitHub 登录服务尚未配置，暂不能查询或新增关联。');
        for(const item of identities.data.identities){
          const card=document.createElement('div');card.className='device-assignment-card';card.dataset.githubSubject=item.subject;
          const title=document.createElement('strong');title.textContent=item.login?`GitHub · ${item.login}`:`GitHub ID · ${item.subject}`;
          const info=document.createElement('p');info.textContent=`稳定 ID：${item.subject} → 成员：${item.user_id}${item.status==='disabled'?'（已停用，不能登录）':''}`;
          card.append(title,info);
          if(item.manageable){
            const remove=document.createElement('button');remove.type='button';remove.className='secondary-button';remove.textContent='解除关联';
            const cancel=document.createElement('button');cancel.type='button';cancel.className='text-button';cancel.textContent='取消解除';cancel.hidden=true;
            let confirmed=false,key=null;
            cancel.onclick=()=>{confirmed=false;cancel.hidden=true;remove.textContent='解除关联';};
            remove.onclick=async()=>{
              if(!confirmed){confirmed=true;cancel.hidden=false;remove.textContent='确认解除并撤销登录';return;}
              remove.disabled=cancel.disabled=true;key ||= crypto.randomUUID();tell('正在解除…');
              let saved=false;
              try{
                await api(`${base()}/${item.subject}`,{method:'PUT',headers:{'Idempotency-Key':key},body:JSON.stringify({user_id:null,expected_revision:item.revision})});saved=true;
                if(epoch!==generation)return;await changed();if(epoch!==generation)return;
                await load();tell('关联已解除，该成员原有登录已撤销。历史数据和设备归属保留。');
              }catch(error){if(epoch!==generation)return;if(error.code==='github_identity_conflict')await load();tell(`${saved?'已解除，但刷新失败':'未解除'}：${error.message}`);}
              finally{remove.disabled=cancel.disabled=false;}
            };
            card.append(remove,cancel);
          }
          list.append(card);
        }
        if(!identities.data.identities.length){const empty=document.createElement('p');empty.textContent='当前组织暂无 GitHub 关联。';list.append(empty);}
      }catch(error){if(epoch===generation)tell(`读取失败：${error.message}。请刷新重试。`);}
    };
    query.addEventListener('input',clearLookup);kind.addEventListener('change',clearLookup);
    target.addEventListener('change',()=>{save.disabled=!resolved || resolved.external_binding || !target.value;});
    queryForm.addEventListener('submit',async event=>{
      event.preventDefault();clearLookup();const epoch=generation,sequence=querySequence; const find=queryForm.querySelector('button');
      find.disabled=true;tell('正在查询 GitHub 公开账号…');
      try{
        const response=await api(`${base()}/lookup?${kind.value}=${encodeURIComponent(query.value.trim())}`);
        if(epoch!==generation || sequence!==querySequence)return;
        resolved=response.data;
        const link=document.createElement('a');link.textContent=`${resolved.profile.login} · 稳定 ID ${resolved.profile.subject}`;
        link.href=resolved.profile.profile_url;link.target='_blank';link.rel='noopener noreferrer';profile.append(link);
        target.add(new Option('请选择组织成员', ''));
        for(const person of members)target.add(new Option(`${person.user_id}${person.status==='disabled'?'（已停用）':''}`,person.user_id));
        if(resolved.user_id)target.value=resolved.user_id;
        save.disabled=resolved.external_binding || !target.value; result.hidden=false;
        tell(resolved.external_binding?'此身份已有其他关联，不能覆盖。':resolved.linked?`当前关联成员：${resolved.user_id}。更换成员须先解除原关联。`:'账号已查到。确认关联会允许该 GitHub 账号登录所选成员，请核对本人身份。');
      }catch(error){if(epoch===generation && sequence===querySequence)tell(`查询失败：${error.message}`);}
      finally{find.disabled=false;}
    });
    result.addEventListener('submit',async event=>{
      event.preventDefault();if(!resolved || save.disabled)return;
      const epoch=generation,subject=resolved.profile.subject;
      const body=JSON.stringify({user_id:target.value,expected_revision:resolved.revision});
      if(pending?.body!==body || pending?.subject!==subject)pending={body,subject,key:crypto.randomUUID()};
      save.disabled=target.disabled=true;tell('正在关联…');let saved=false;
      try{
        await api(`${base()}/${subject}`,{method:'PUT',headers:{'Idempotency-Key':pending.key},body});saved=true;
        if(epoch!==generation)return;await load();tell('GitHub 身份已关联。该账号可通过 GitHub 登录对应成员，停用成员需先恢复。');
      }catch(error){if(epoch!==generation)return;if(error.code==='github_identity_conflict')clearLookup();tell(`${saved?'已关联，但刷新失败':'未关联'}：${error.message}`);}
      finally{save.disabled=target.disabled=false;}
    });
    button.onclick=()=>{dialog.showModal();load();};
    document.querySelector('#githubIdentityRefresh').onclick=load;
    document.querySelector('#githubIdentityClose').onclick=()=>dialog.close();
    return {account(value){
      const allowed=value?.authenticated && value.session.permissions?.includes('identity.manage');
      const next=allowed?`${value.session.organization_id}/${value.session.user_id}/${value.session.role}`:'';
      if(next===identity)return;identity=next;generation++;clearLookup();list.replaceChildren();members=[];tell('');query.value='';dialog.close();
      organization=allowed?value.session.organization_id:'';button.hidden=!organization;
    }};
  }
};
