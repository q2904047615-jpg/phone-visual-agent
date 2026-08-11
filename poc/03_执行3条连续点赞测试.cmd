@echo off
chcp 65001 >nul
title PoC 03 - THREE LIKES CONTINUOUS TEST
echo ===== MODE 03: REAL ACTION - THREE LIKES WITH AUTO SWIPE =====
cd /d "%~dp0\.."
python .\poc\robot_gui_poc.py douyin-like --count 3 --max-pages 9 --execute
echo.
pause
