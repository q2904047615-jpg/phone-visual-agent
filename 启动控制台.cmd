@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo 首次运行：正在创建 Python 环境并安装依赖……
  python -m venv .venv || (echo 请先安装 Python 3.11+ 并勾选 Add to PATH & pause & exit /b 1)
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt || (pause & exit /b 1)
)
".venv\Scripts\python.exe" -m pva %*
pause
