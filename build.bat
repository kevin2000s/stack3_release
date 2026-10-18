@echo off
REM build.bat — 打包 stack3 predictor (PyInstaller onedir)
REM 在干净的 venv_build 中跑, 避免污染全局 Python
REM 产物: dist\stack3_predictor\stack3_predictor.exe
REM 双击 dist\stack3_predictor\stack3_predictor.exe 启动 GUI.

chcp 65001 >nul
title Build stack3 Predictor
echo ============================================
echo  stack3 Predictor - Build Executable
echo ============================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. 安装 Python 3.9+ 后重试.
    pause
    exit /b 1
)

echo [1/5] Python:
python --version
echo.

echo [2/5] 创建虚拟环境 venv_build ...
if exist venv_build rmdir /s /q venv_build
python -m venv venv_build
if errorlevel 1 (
    echo [ERROR] 虚拟环境创建失败.
    pause
    exit /b 1
)

echo [3/5] 安装依赖 ...
call venv_build\Scripts\activate.bat
pip install -q -r requirements.txt pyinstaller
if errorlevel 1 (
    echo [ERROR] pip install 失败.
    pause
    exit /b 1
)

echo [4/5] 跑 PyInstaller ...
python -m PyInstaller -y stack3_predictor.spec
if errorlevel 1 (
    echo [ERROR] Build 失败.
    pause
    exit /b 1
)

echo [5/5] 清理中间产物 ...
if exist build rmdir /s /q build
if exist venv_build rmdir /s /q venv_build

echo.
echo ============================================
echo  Build Complete!
echo ============================================
echo.
echo 产物: dist\stack3_predictor\stack3_predictor.exe
echo 双击启动 GUI.
echo.
pause
