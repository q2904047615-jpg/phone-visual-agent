@echo off
chcp 65001 >nul
title PoC 04 - TEN LIKES
echo ===== MODE 04: REAL ACTION - TEN LIKES =====
cd /d "%~dp0\.."
python .\poc\robot_gui_poc.py douyin-like --count 10 --execute
echo.
pause
