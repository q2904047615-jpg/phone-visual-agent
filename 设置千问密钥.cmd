@echo off
chcp 65001 >nul
echo 在阿里云百炼（Model Studio）控制台创建 API Key 后粘贴到这里。
set /p KEY=DASHSCOPE_API_KEY: 
if "%KEY%"=="" (echo 未输入，已取消。 & pause & exit /b 1)
setx DASHSCOPE_API_KEY "%KEY%" >nul
echo 已保存到当前用户环境变量。重新运行“启动控制台.cmd”即可生效。
pause
