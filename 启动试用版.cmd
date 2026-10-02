@echo off
cd /d "%~dp0"
python start_trial.py
if errorlevel 1 pause
