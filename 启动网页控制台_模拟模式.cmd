@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0poc"
set "ROBOT_WEB_MOCK=1"

if not exist ".venv\Scripts\python.exe" (
  python -m venv --system-site-packages .venv
  if errorlevel 1 goto :failed
)

".venv\Scripts\python.exe" -c "import fastapi, uvicorn, httpx" >nul 2>&1
if errorlevel 1 (
  ".venv\Scripts\python.exe" -m pip install -r requirements-web.txt
  if errorlevel 1 goto :failed
)

echo 模拟模式启动中：不会操作真实机械臂。
".venv\Scripts\python.exe" -m uvicorn web_app:app --host 127.0.0.1 --port 8765
goto :end

:failed
echo 启动失败。请截图本窗口内容发给 Codex。
pause

:end
endlocal
