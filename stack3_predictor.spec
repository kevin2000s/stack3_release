# -*- mode: python ; coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""PyInstaller spec for stack3 predictor (onedir mode).

Build (from repo root, with venv activated):
    pyinstaller -y stack3_predictor.spec

Output: dist/stack3_predictor/stack3_predictor.exe  (双击启动 GUI)
        + 同目录 bundles/ (5 个 bundle)
        + 同目录 prediction_toolkit/ (源码包)
"""
import os

from PyInstaller.utils.hooks import collect_dynamic_libs

datas = []
bundles_dir = 'bundles'
if os.path.isdir(bundles_dir):
    files = [(os.path.join(bundles_dir, f), 'bundles')
             for f in os.listdir(bundles_dir) if f.endswith('.joblib')]
    datas.extend(files)
# 训练池 phi ref (必需: 供 KNN lookup, 75MB)
extras_dir = 'extras'
if os.path.isdir(extras_dir):
    for f in os.listdir(extras_dir):
        if f.endswith('.npy') or f.endswith('.npz'):
            datas.append((os.path.join(extras_dir, f), 'extras'))

# 携带 prediction_toolkit 包源码 (GUI 等)
datas.append(('prediction_toolkit', 'prediction_toolkit'))

# pyrodigal 的 impl 后端 (avx2/sse2/... 等 C 扩展) 由 lib.pyx 在运行时
# 懒加载, PyInstaller 静态分析抓不到, 必须显式收集其二进制 + 声明子模块,
# 否则 frozen exe 报 "No module named 'pyrodigal.impl'".
pyrodigal_binaries = collect_dynamic_libs('pyrodigal')

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=pyrodigal_binaries,
    datas=datas,
    hiddenimports=[
        'sklearn', 'sklearn.ensemble', 'sklearn.tree',
        'sklearn.preprocessing', 'sklearn.decomposition',
        'pandas', 'numpy', 'scipy',
        'matplotlib', 'matplotlib.backends.backend_tkagg',
        'ttkbootstrap', 'ttkbootstrap.constants',
        'ttkbootstrap.dialogs', 'ttkbootstrap.scrolled',
        'tkinter', 'joblib',
        'pyrodigal', 'pyrodigal.pyrodigal',
        'pyrodigal.impl', 'pyrodigal.impl.generic', 'pyrodigal.impl.sse2',
        'pyrodigal.impl.swar64', 'pyrodigal.impl.avx2', 'pyrodigal.impl.avx512',
        # DNABERT-S HF 缓存代码 (bert_layers.py) 运行时 import, 静态分析不可见
        'einops', 'einops.layers',
        'transformers', 'transformers.AutoTokenizer',
        'torch', 'phi_encoder',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='stack3_predictor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='stack3_predictor',
)
