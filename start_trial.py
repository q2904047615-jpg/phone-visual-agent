"""Start one isolated API, verify its identity, never enqueue a phone task."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent

def main():
    config = json.loads((ROOT/'trial.json').read_text(encoding='utf-8'))
    candidates = [ROOT/'.venv/Scripts/python.exe',ROOT.parent/'.venv/Scripts/python.exe',ROOT/'.venv/bin/python']
    python = next((p for p in candidates if p.is_file()),None)
    if python is None:
        raise RuntimeError('先运行 python setup_trial.py 安装。')
    environment = dict(os.environ, PYTHONUTF8='1',ROBOT_WEB_NO_BROWSER='1')
    if os.name == 'nt':
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER,'Environment') as key:
                names = ['DASHSCOPE_API_KEY','VISION_MODEL','GMAIL_SENDER','GMAIL_APP_PASSWORD','GMAIL_TOKEN_FILE','GMAIL_CREDENTIALS_FILE','GMAIL_REFRESH_TOKEN','GMAIL_ACCESS_TOKEN','GMAIL_CLIENT_ID','GMAIL_CLIENT_SECRET']
                for name in names:
                    if not environment.get(name):
                        try:
                            environment[name] = str(winreg.QueryValueEx(key,name)[0])
                        except FileNotFoundError:
                            pass
        except FileNotFoundError:
            pass
    base = f"http://127.0.0.1:{config['port']}"
    expected = '0.2.0-trial-'+config['mode']
    command = [str(python),'-X','utf8',str(ROOT/'poc/agent_api_cli.py'),'--base-url',base,'--timeout','3','bootstrap']
    def check():
        result = subprocess.run(command,capture_output=True,text=True,encoding='utf-8',env=environment)
        if result.returncode:
            return False
        payload = json.loads(result.stdout)
        if payload.get('service_version') != expected:
            raise RuntimeError('端口已有不同版本，未覆盖或停止它。')
        return True
    if not check():
        with (ROOT/'trial-service.log').open('ab') as log:
            subprocess.Popen([str(python),'-X','utf8','-m','uvicorn','web_app:app','--host','127.0.0.1','--port',str(config['port'])],
                cwd=ROOT/'poc',env=environment,stdout=log,stderr=log,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        for _ in range(30):
            time.sleep(1)
            if check():
                break
        else:
            raise RuntimeError('API 启动失败，查看 trial-service.log。')
    print(config['title']+' 已加载：'+base+'；未自动开始手机任务。')

if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(str(exc),file=sys.stderr)
        raise SystemExit(1)
