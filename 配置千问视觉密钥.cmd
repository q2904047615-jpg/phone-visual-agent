@echo off
setlocal
chcp 65001 >nul
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0配置千问视觉密钥.ps1"
if errorlevel 1 (
  echo.
  echo 配置失败。请保留窗口并截图发给 Codex。
  pause
  exit /b 1
)
endlocal
