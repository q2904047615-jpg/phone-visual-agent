@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo 首次运行：正在创建 Python 环境并安装依赖……
  python -m venv .venv || (echo 请先安装 Python 3.11+ 并勾选 Add to PATH & pause & exit /b 1)
)
rem 每次启动都补齐依赖（已装好时只做检查），旧环境也能拿到新增的依赖。
".venv\Scripts\python.exe" -m pip install -q --disable-pip-version-check -r requirements.txt || (pause & exit /b 1)
".venv\Scripts\python.exe" -m pva %*
pause
