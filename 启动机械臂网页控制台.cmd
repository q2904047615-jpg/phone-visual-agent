@echo off
setlocal
cd /d "%~dp0poc"

if not defined DASHSCOPE_API_KEY (
  for /f "tokens=2,*" %%A in ('reg query HKCU\Environment /v DASHSCOPE_API_KEY 2^>nul ^| findstr /i "DASHSCOPE_API_KEY"') do set "DASHSCOPE_API_KEY=%%B"
)

if not exist ".venv\Scripts\python.exe" (
  echo [1/3] Creating local Python environment...
  python -m venv --system-site-packages .venv
  if errorlevel 1 goto :failed
)

".venv\Scripts\python.exe" -c "import fastapi, uvicorn, httpx, pypinyin" >nul 2>&1
if errorlevel 1 (
  echo [2/3] Installing web console dependencies...
  ".venv\Scripts\python.exe" -m pip install -r requirements-web.txt
  if errorlevel 1 goto :failed
)

echo [3/3] Starting robot web console...
if defined DASHSCOPE_API_KEY (
  echo Qwen vision agent: qwen3-vl-plus ready
) else (
  echo Qwen vision agent: API key not configured; fixed workflows remain available
)
echo Starting local service first. The browser will open after it is ready...
start "Robot Web Console Server" /D "%CD%" ".venv\Scripts\python.exe" -m uvicorn web_app:app --host 127.0.0.1 --port 8765

for /L %%I in (1,1,30) do (
  powershell.exe -NoProfile -Command "try { $r = Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:8765/api/device' -TimeoutSec 1; if ($r.StatusCode -eq 200) { exit 0 } } catch {}; exit 1" >nul 2>&1
  if not errorlevel 1 goto :ready
  >nul ping 127.0.0.1 -n 2
)

echo.
echo The local service did not become ready within 30 seconds.
echo Keep the "Robot Web Console Server" window open and send its error text to Codex.
pause
goto :end

:ready
echo Service ready: http://127.0.0.1:8765/
start "" msedge.exe "http://127.0.0.1:8765/"
goto :end

:failed
echo.
echo Startup failed. Please send a screenshot of this window to Codex.
pause

:end
endlocal
