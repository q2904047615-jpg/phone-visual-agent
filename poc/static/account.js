/* Account state comes only from the server. No principal tokens in JS/storage. */
window.AgentAccount = {
  async mount(api, onChange = () => {}) {
    const status = document.querySelector('#accountStatus');
    const detail = document.querySelector('#accountDetail');
    const login = document.querySelector('#githubLoginButton');
    const logout = document.querySelector('#accountLogoutButton');
    const retry = document.querySelector('#accountRetryButton');
    const error = document.querySelector('#accountError');
    let busy = false;
    let current = null;
    const showError = message => { error.textContent = message; error.hidden = !message; };
    const render = () => {
      const signedIn = Boolean(current?.authenticated);
      status.textContent = signedIn ? `已登录 · ${current.session.user_id}` : '未登录';
      detail.textContent = signedIn
        ? `组织：${current.session.organization_id} · 有效至 ${new Date(current.session.expires_at).toLocaleTimeString()}`
        : current?.github_available ? '登录以使用网站账号' : 'GitHub 登录尚未配置';
      login.hidden = signedIn;
      login.disabled = busy || !current?.github_available;
      logout.hidden = !signedIn;
      logout.disabled = busy;
    };
    const refresh = async () => {
      try {
        const result = await api('/api/control-plane/auth/browser-session', { cache: 'no-store' });
        if (typeof result.data?.authenticated !== 'boolean') throw new Error('登录状态响应无效');
        current = result.data;
        retry.hidden = true;
        render();
        onChange(current);
        return true;
      } catch (e) {
        status.textContent = '登录状态暂时无法确认';
        login.disabled = true;
        retry.hidden = false;
        showError(`无法检查登录状态：${e.message}`);
        onChange(null);
        return false;
      }
    };
    login.addEventListener('click', async () => {
      if (busy) return;
      busy = true; render(); showError('');
      try {
        const result = await api('/api/control-plane/auth/github/start?browser=true');
        const target = new URL(result.data.authorization_url);
        if (target.origin !== 'https://github.com' || target.pathname !== '/login/oauth/authorize') {
          throw new Error('登录地址无效');
        }
        window.location.assign(target.href);
      } catch (e) { showError(`无法开始登录：${e.message}`); }
      finally { busy = false; render(); }
    });
    logout.addEventListener('click', async () => {
      if (busy) return;
      busy = true; render(); showError('');
      try {
        await api('/api/control-plane/auth/browser-logout', { method: 'POST', body: '{}' });
        localStorage.setItem('agent-account-change', String(Date.now()));
        localStorage.removeItem('agent-account-change');
        await refresh();
      } catch (e) { showError(`退出失败，请重试：${e.message}`); }
      finally { busy = false; render(); }
    });
    retry.addEventListener('click', async () => { showError(''); await refresh(); });
    const url = new URL(window.location.href);
    const failure = url.searchParams.get('login_error');
    const messages = {
      oauth_authorization_denied: '已取消 GitHub 授权，可以重新登录。',
      oauth_state_invalid: '登录请求已过期或服务已重启，请重新登录。',
      oauth_browser_mismatch: '请在发起登录的浏览器中完成授权。',
      external_identity_mapping_not_found: '此 GitHub 账号尚未关联网站账号，请联系管理员。',
      external_provider_unavailable: 'GitHub 暂时无法连接，请稍后重试。',
    };
    if (failure) showError(messages[failure] || '登录未完成，请重试或联系管理员。');
    if (failure || url.searchParams.has('login')) {
      url.searchParams.delete('login_error'); url.searchParams.delete('login');
      history.replaceState(null, '', url.pathname + url.search + url.hash);
    }
    await refresh();
    window.addEventListener('focus', () => { if (!busy) refresh(); });
    window.addEventListener('storage', event => { if (event.key === 'agent-account-change') refresh(); });
    window.addEventListener('agent-identity-invalid', () => { onChange(null); refresh(); });
    setInterval(() => { if (!document.hidden && !busy) refresh(); }, 30000);
    return current;
  },
};
