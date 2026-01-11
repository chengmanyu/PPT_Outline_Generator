@echo off
chcp 65001 >nul
echo.
echo ==========================================
echo   PPT 大纲生成器 - 一键启动
echo ==========================================
echo.

REM 检查虚拟环境是否存在
if not exist ".venv\Scripts\activate.bat" (
    echo ❌ 未找到虚拟环境 .venv
    echo 正在创建虚拟环境...
    python -m venv .venv
    if errorlevel 1 (
        echo ❌ 虚拟环境创建失败！
        pause
        exit /b 1
    )
    echo ✓ 虚拟环境创建完成
)

echo.
echo 正在激活虚拟环境...
call .venv\Scripts\activate.bat

echo.
echo 检查必要的依赖...
pip list | find "fastapi" >nul
if errorlevel 1 (
    echo 正在安装依赖...
    python.exe -m pip install --upgrade pip
    pip install fastapi uvicorn torch transformers bitsandbytes accelerate -q
    pip uninstall torch torchvision torchaudio
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
)

echo.
echo ==========================================
echo ✓ 环境已准备就绪，启动后端服务...
echo ==========================================
echo.
echo 后端将运行在: http://localhost:8000
echo 前端将运行在: http://localhost:8888
echo.
echo 提示：
echo - 一旦看到 "Uvicorn running on..." 消息，说明后端已准备好
echo - 点击下面的链接打开前端：
echo   http://localhost:8888
echo.
echo 按任意键启动后端...（或继续运行）
echo.

REM 启动后端服务
python backend.py

pause
