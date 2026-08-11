@echo off
chcp 65001 >nul
title PoC 02 - ONE LIKE TEST
echo ===== MODE 02: REAL ACTION - ONE LIKE =====
cd /d "%~dp0\.."
python .\poc\robot_gui_poc.py douyin-like --count 1 --max-pages 3 --execute
echo.
pause
