@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0configure_deepseek_key.ps1"
if errorlevel 1 (
  echo.
  echo Configuration failed. Keep this window open and send a screenshot to Codex.
  pause
  exit /b 1
)
endlocal
