@echo off
chcp 65001 >nul
title PoC 01 - DRY RUN ONLY
echo ===== MODE 01: DETECTION ONLY - ROBOT WILL NOT MOVE =====
cd /d "%~dp0\.."
python .\poc\robot_gui_poc.py douyin-like --count 10
echo.
pause
