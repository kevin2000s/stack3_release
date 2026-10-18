#!/bin/bash
# build.sh — 打包 stack3 predictor (Linux/macOS)
# 产物: dist/stack3_predictor/stack3_predictor

set -e

echo "============================================"
echo "  stack3 Predictor - Build Executable"
echo "============================================"

if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 未找到. 请先安装 Python 3.9+"
    exit 1
fi

echo "[1/5] Python: $(python3 --version)"

echo "[2/5] 创建虚拟环境 venv_build ..."
rm -rf venv_build
python3 -m venv venv_build

echo "[3/5] 安装依赖 ..."
source venv_build/bin/activate
pip install -q -r requirements.txt pyinstaller

echo "[4/5] 跑 PyInstaller ..."
python -m PyInstaller -y stack3_predictor.spec

echo "[5/5] 清理 ..."
rm -rf build venv_build

echo ""
echo "============================================"
echo "  Build Complete!"
echo "============================================"
echo "产物: dist/stack3_predictor/stack3_predictor"
