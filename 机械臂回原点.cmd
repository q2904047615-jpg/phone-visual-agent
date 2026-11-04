@echo off
chcp 65001 >nul
rem 让机械臂回到原点。重启电脑、拔 USB、强行关闭 main.exe 之前先运行一次。
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0pva\tools\arm_home.ps1" %*
if errorlevel 1 (echo 回原点失败，详见上面的提示 & pause & exit /b 1)
echo 完成。
timeout /t 3 >nul 2>&1
exit /b 0
