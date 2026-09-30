"""Install this trial's dependencies and pinned upstream detector locally."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.request
import venv

ROOT = Path(__file__).resolve().parent

def main():
    config = json.loads((ROOT/'trial.json').read_text(encoding='utf-8'))
    python = ROOT/'.venv/Scripts/python.exe' if sys.platform == 'win32' else ROOT/'.venv/bin/python'
    if not python.exists():
        venv.EnvBuilder(with_pip=True).create(ROOT/'.venv')
    subprocess.run([str(python),'-m','pip','install','-r',str(ROOT/'requirements-trial.txt')],check=True)
    if config['mode'] == 'upstream':
        directory = ROOT/'external/douyin_guaji'
        directory.mkdir(parents=True,exist_ok=True)
        url = f"https://raw.githubusercontent.com/pokemonzlj/douyin_guaji/{config['upstream_commit']}/douyin_fudai.py"
        data = urllib.request.urlopen(url,timeout=60).read()
        if hashlib.sha256(data).hexdigest() != config['upstream_sha256']:
            raise RuntimeError('上游文件校验失败，不安装。')
        (directory/'douyin_fudai.py').write_bytes(data)
        (directory/'verified-source.json').write_text(json.dumps({'commit':config['upstream_commit'],'sha256':config['upstream_sha256']}),encoding='utf-8')
    print('安装完成，运行 python start_trial.py。')

if __name__ == '__main__':
    main()
