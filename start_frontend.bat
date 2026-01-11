@echo off
chcp 65001 >nul
echo.
echo ==========================================
echo   PPT 大纲生成器 - 前端服务
echo ==========================================
echo.

echo 正在启动前端服务器...
echo.

REM 使用 Python 的 http.server 简单服务器
REM 如果你有 Node.js + npm 也可以使用该方案

REM 方案1：使用 Python 内置的 HTTP 服务器（推荐，无需额外依赖）
cd /d "%~dp0"
python -m http.server 8888 --bind 127.0.0.1 --directory .

REM 如果上面的命令不兼容，可以尝试这个：
REM python -m SimpleHTTPServer 8888
