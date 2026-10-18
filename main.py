#!/usr/bin/env python3
"""GUI entry point for PyInstaller packaging.

PyInstaller 会以这个文件为入口打包. 实际 GUI 在 prediction_toolkit.gui.app.
"""
import os
import sys

# PyInstaller frozen 时, sys.executable 指向 exe, 资源 (.joblib 等) 在同目录.
if getattr(sys, 'frozen', False):
    exe_dir = os.path.dirname(sys.executable)
else:
    exe_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, exe_dir)
sys.path.insert(0, os.path.join(exe_dir, 'prediction_toolkit'))

from prediction_toolkit.gui.app import main

if __name__ == '__main__':
    main()
