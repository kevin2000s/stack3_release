"""stack3 GUI - FASTA 直接上传, 后台算 phi, 然后 5 bundle 预测.

输入: FASTA (.fasta / .fa) + 13 临床/环境字段
输出: 5 省各自 fITI/ITI + macro 平均
"""
import os
import sys
import re
import threading
import queue
import tkinter as tk

import numpy as np
import joblib

import ttkbootstrap as ttk
from ttkbootstrap.constants import (
    PRIMARY, SUCCESS, WARNING, DANGER, INFO, SECONDARY,
)
from ttkbootstrap.dialogs import Messagebox


if getattr(sys, 'frozen', False):
    BASE = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
else:
    BASE = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

BUNDLES_DIR = os.path.join(BASE, 'bundles')
TRAIN_POOL_PHI = os.path.join(BASE, 'extras', 'train_pool_phi.npy')
PROVINCES = ['广东', '河北', '四川', '河南', '浙江']

DEFAULT_ENV = {
    'PM2.5 (μg/m³)': '35', 'PM10 (μg/m³)': '60',
    'SO2 (μg/m³)': '10', 'NO2 (μg/m³)': '25',
    'CO (mg/m³)': '1.0', 'O3 (μg/m³)': '70',
    'AQI': '60', 'T_mean (°C)': '20', 'RH (%)': '65',
}
DEFAULT_DEP = {
    'log(tc_n) 该 TC 院内历史株数': '2.0',
    'freq ST 在该 TC 占比': '0.3',
    'log(stratum_n) 该 stratum 累计株数': '3.0',
}


def _load_train_phi():
    if not os.path.exists(TRAIN_POOL_PHI):
        return None
    try:
        return np.load(TRAIN_POOL_PHI)
    except Exception:
        return None


def _knn_features(query_phi, train_phi):
    if train_phi is None:
        z = np.zeros(8, np.float32)
        return {20: z.copy(), 24: np.concatenate([z, z, z])}
    qn = query_phi / (np.linalg.norm(query_phi) + 1e-9)
    tn = train_phi / (np.linalg.norm(train_phi, axis=1, keepdims=True) + 1e-9)
    cos = tn @ qn
    out = {}
    for K in (5, 20, 50):
        top = np.argpartition(-cos, K - 1)[:K]
        sel = train_phi[top]
        sel_cos = cos[top]
        feats = np.array([
            float(sel.mean()), float(K), float(sel_cos.mean()),
            float(sel.mean()), float(sel.var()), float(np.median(sel_cos)),
            float(sel.std()), float(np.percentile(sel_cos, 90)),
        ], dtype=np.float32)
        out[K] = feats
    return {20: out[20], 24: np.concatenate([out[5], out[20], out[50]])}


def _day_to_sincos(sample_date):
    import pandas as pd
    doy = pd.Timestamp(sample_date).dayofyear
    return (np.sin(2 * np.pi * doy / 365.25),
            np.cos(2 * np.pi * doy / 365.25))


def predict_all_bundles(phi, env9, date, dep3):
    """5 个 bundle 各跑一次, 返回 list[dict]."""
    train_phi = _load_train_phi()
    knn = _knn_features(phi, train_phi)
    sin_cos = np.array(_day_to_sincos(date), np.float32)

    results = []
    for prov in PROVINCES:
        b = joblib.load(os.path.join(BUNDLES_DIR,
                                      f'holdout_{prov}_stack3_bundle.joblib'))
        x1 = np.concatenate([env9, sin_cos, dep3, knn[20]]).astype(np.float32)
        p = float(b['bases']['M1']['fITI_bin'].predict_proba(x1[None, :])[0, 1])
        s1 = float(b['bases']['M1']['ITI'].predict(x1[None, :])[0])
        x2 = np.concatenate([phi, env9, sin_cos, dep3]).astype(np.float32)
        s2 = float(b['bases']['M2']['ITI'].predict(x2[None, :])[0])
        x3 = np.concatenate([phi, env9, sin_cos, dep3, knn[24]]).astype(np.float32)
        s3 = float(b['bases']['M3']['ITI'].predict(x3[None, :])[0])
        s = np.array([s1, s2, s3], np.float32)
        m = b['meta_iti']
        s_std = (s - m['scaler_mean']) / m['scaler_std']
        iti = float(s_std @ m['coef'] + m['intercept'])
        results.append({
            'prov': prov,
            'fiti_p': p,
            'iti': iti,
            'splits': (s1, s2, s3),
        })
    return results


# ============ GUI ============

class Stack3App:
    def __init__(self, root):
        self.root = root
        self.root.title('stack3 — 双头院感病原预测器')
        self.root.geometry('1000x820')
        self.root.minsize(820, 640)

        self.fasta_path = tk.StringVar()
        self.env_vars = {k: tk.StringVar(value=v) for k, v in DEFAULT_ENV.items()}
        self.dep_vars = {k: tk.StringVar(value=v) for k, v in DEFAULT_DEP.items()}
        self.date_var = tk.StringVar(value='2026-06-15')

        self.results = None
        self.phi = None  # cached
        self._results_queue = queue.Queue()
        self._build()

    def _build(self):
        head = ttk.Frame(self.root, padding=(20, 10))
        head.pack(fill='x')
        ttk.Label(head, text='stack3',
                  font=('Helvetica', 22, 'bold'),
                  bootstyle=INFO).pack(side='left')
        ttk.Label(head, text=' 双头院感病原预测器 · FASTA→phi→5bundle',
                  font=('Microsoft YaHei', 11)).pack(side='left', padx=(8, 0))

        # ① FASTA 上传
        sec1 = ttk.Labelframe(self.root, text=' ① 基因组 FASTA  ',
                              padding=12, bootstyle=PRIMARY)
        sec1.pack(fill='x', padx=20, pady=(8, 4))
        rf = ttk.Frame(sec1); rf.pack(fill='x')
        ttk.Entry(rf, textvariable=self.fasta_path, font=('Consolas', 9)).pack(
            side='left', fill='x', expand=True)
        ttk.Button(rf, text=' 选择 FASTA 文件 ', bootstyle=PRIMARY,
                   command=self._pick_fasta).pack(side='left', padx=(8, 0))

        # ② 13 字段
        sec2 = ttk.Labelframe(self.root, text=' ② 临床 / 环境字段 ',
                              padding=12, bootstyle=PRIMARY)
        sec2.pack(fill='x', padx=20, pady=4)

        ttk.Label(sec2, text='环境 (9 维)', font=('Microsoft YaHei', 10, 'bold'),
                  foreground='gray').pack(anchor='w')
        grid = ttk.Frame(sec2); grid.pack(fill='x', pady=(2, 8))
        items = list(self.env_vars.items())
        for i, (k, v) in enumerate(items):
            r, c = divmod(i, 3)
            cell = ttk.Frame(grid)
            cell.grid(row=r, column=c, padx=4, pady=2, sticky='ew')
            grid.grid_columnconfigure(c, weight=1, uniform='env')
            ttk.Label(cell, text=k, font=('Microsoft YaHei', 9)).pack(anchor='w')
            ttk.Entry(cell, textvariable=v, font=('Consolas', 9)).pack(fill='x')

        ttk.Label(sec2, text='历史 (3 维)', font=('Microsoft YaHei', 10, 'bold'),
                  foreground='gray').pack(anchor='w')
        dep_frame = ttk.Frame(sec2); dep_frame.pack(fill='x', pady=(2, 8))
        for i, (k, v) in enumerate(self.dep_vars.items()):
            cell = ttk.Frame(dep_frame)
            cell.grid(row=0, column=i, padx=4, pady=2, sticky='ew')
            dep_frame.grid_columnconfigure(i, weight=1, uniform='dep')
            ttk.Label(cell, text=k, font=('Microsoft YaHei', 9), wraplength=160).pack(anchor='w')
            ttk.Entry(cell, textvariable=v, font=('Consolas', 9)).pack(fill='x')

        ttk.Label(sec2, text='时间', font=('Microsoft YaHei', 10, 'bold'),
                  foreground='gray').pack(anchor='w')
        tcell = ttk.Frame(sec2); tcell.pack(fill='x', pady=(2, 8))
        ttk.Label(tcell, text='采样日期 (YYYY-MM-DD)').pack(side='left')
        ttk.Entry(tcell, textvariable=self.date_var, width=12,
                  font=('Consolas', 9)).pack(side='left', padx=8)
        ttk.Label(tcell, text='  · 例 2026-06-15',
                  font=('Microsoft YaHei', 8), foreground='gray').pack(side='left')

        # 预测按钮 + 进度
        btnf = ttk.Frame(self.root, padding=(20, 8))
        btnf.pack(fill='x')
        self.run_btn = ttk.Button(btnf, text='  预测 (FASTA→phi→5 bundle)  ',
                                   command=self.run, bootstyle=(SUCCESS, 'large'),
                                   padding=(20, 10))
        self.run_btn.pack(side='left')
        self.spinner_label = ttk.Label(btnf, text=' 空闲', font=('Microsoft YaHei', 9),
                                       bootstyle=SECONDARY, padding=(20, 0))
        self.spinner_label.pack(side='left')

        # 结果
        out = ttk.Labelframe(self.root, text=' ③ 5 bundle 预测结果 ',
                             padding=12, bootstyle=SUCCESS)
        out.pack(fill='both', expand=True, padx=20, pady=4)

        avg = ttk.Frame(out); avg.pack(fill='x')
        ab1 = ttk.Frame(avg); ab1.pack(side='left', fill='both', expand=True)
        ttk.Label(ab1, text='5 bundle macro 平均',
                  font=('Microsoft YaHei', 10, 'bold'), bootstyle=INFO).pack(anchor='w')
        self.avg_fiti = ttk.Label(ab1, text='—',
                                   font=('Helvetica', 28, 'bold'), bootstyle=INFO)
        self.avg_fiti.pack(anchor='w')
        self.avg_tag = ttk.Label(ab1, text='', font=('Microsoft YaHei', 10))
        self.avg_tag.pack(anchor='w')

        ab2 = ttk.Frame(avg); ab2.pack(side='left', fill='both', expand=True)
        ttk.Label(ab2, text='stack3 ITI 平均',
                  font=('Microsoft YaHei', 10, 'bold'), bootstyle=WARNING).pack(anchor='w')
        self.avg_iti = ttk.Label(ab2, text='—',
                                 font=('Helvetica', 28, 'bold'), bootstyle=WARNING)
        self.avg_iti.pack(anchor='w')

        ttk.Label(out, text='各 bundle 详情',
                  font=('Microsoft YaHei', 10, 'bold')).pack(anchor='w', pady=(10, 2))
        cols = ('prov', 'fiti', 'iti', 'm1', 'm2', 'm3')
        self.tree = ttk.Treeview(out, columns=cols, show='headings', height=6)
        for col, wd in (('prov', 90), ('fiti', 90), ('iti', 90),
                        ('m1', 90), ('m2', 90), ('m3', 90)):
            self.tree.heading(col, text=col)
            self.tree.column(col, width=wd, anchor='center')
        self.tree.pack(fill='x')

        ttk.Label(out, text='推理日志', font=('Microsoft YaHei', 10, 'bold'),
                  bootstyle=SECONDARY).pack(anchor='w', pady=(10, 2))
        self.log = tk.Text(out, height=5, wrap='word',
                           font=('Consolas', 9), bg='#1e1e1e', fg='#d4d4d4',
                           insertbackground='#d4d4d4', relief='flat')
        self.log.pack(fill='both', expand=True)

    def _pick_fasta(self):
        from tkinter import filedialog
        p = filedialog.askopenfilename(filetypes=[
            ('FASTA', '*.fasta *.fa *.fna *.ffn'), ('all', '*.*')])
        if p:
            self.fasta_path.set(p)

    def _read_inputs(self):
        if not self.fasta_path.get():
            raise ValueError('请先选 FASTA 文件')
        if not os.path.exists(self.fasta_path.get()):
            raise ValueError(f'文件不存在: {self.fasta_path.get()}')
        env9 = np.array([float(self.env_vars[k].get()) for k in DEFAULT_ENV],
                        np.float32)
        dep3 = np.array([float(self.dep_vars[k].get()) for k in DEFAULT_DEP],
                        np.float32)
        d = self.date_var.get().strip()
        if not re.match(r'^\d{4}-\d{1,2}-\d{1,2}$', d):
            raise ValueError('日期格式应为 YYYY-MM-DD (例 2026-06-15)')
        return self.fasta_path.get(), env9, dep3, d

    def run(self):
        try:
            fasta, env9, dep3, date = self._read_inputs()
        except Exception as e:
            Messagebox.show_error(f'错误: {e}', title='stack3', parent=self.root)
            return

        # 禁用按钮 + 后台算 phi
        self.run_btn.configure(state='disabled', text=' 计算中…请稍候 ')
        self.spinner_label.configure(text='⏳ 后台跑 pyrodigal + DNABERT-S…')

        def worker():
            try:
                # 1. 算 phi (用 first bundle 的 centroids, 5 个 bundle 共用相同 centroids)
                sys.path.insert(0, os.path.join(BASE, 'prediction_toolkit'))
                # 重要: 加 prediction_toolkit 到 path 让 phi_encoder 可 import
                sys.path.insert(0, BASE)
                from phi_encoder import compute_phi
                # 加载一个 bundle 拿 centroids (5 bundle 共享相同 centroids)
                first_bundle = joblib.load(os.path.join(
                    BUNDLES_DIR, f'holdout_{PROVINCES[0]}_stack3_bundle.joblib'))
                phi = compute_phi(fasta, first_bundle, max_orfs=1024,
                                  progress=lambda s, n=None: None)
                # 2. 算 KNN + 5 bundle 预测
                results = predict_all_bundles(phi, env9, date, dep3)
                self._results_queue.put(('ok', phi, results, date, env9, dep3))
            except Exception as e:
                self._results_queue.put(('err', str(e)))
            finally:
                self._results_queue.put(('done',))

        self._worker = threading.Thread(target=worker, daemon=True)
        self._worker.start()
        self.root.after(100, self._poll_queue)

    def _poll_queue(self):
        try:
            kind = self._results_queue.get_nowait()
        except queue.Empty:
            self.root.after(150, self._poll_queue)
            return

        if kind[0] == 'done':
            return  # 啥也不做 (worker 结束)
        elif kind[0] == 'err':
            self.run_btn.configure(state='normal', text='  预测 (FASTA→phi→5 bundle)  ')
            self.spinner_label.configure(text='⛔ 错误')
            Messagebox.show_error(f'{kind[1]}', title='stack3', parent=self.root)
            return
        else:
            # ok
            _, phi, results, date, env9, dep3 = kind
            self.run_btn.configure(state='normal', text='  预测 (FASTA→phi→5 bundle)  ')
            self.spinner_label.configure(text='✓ 完成')
            self._show_results(phi, results, date, env9)

    def _show_results(self, phi, results, date, env9):
        self.results = results
        self.phi = phi
        avg_p = float(np.mean([r['fiti_p'] for r in results]))
        avg_i = float(np.mean([r['iti'] for r in results]))
        self.avg_fiti.configure(text=f'{avg_p:.3f}')
        if avg_p > 0.7:
            self.avg_fiti.configure(bootstyle=DANGER); tag = '🔴 高频 (≥0.7)'
        elif avg_p < 0.3:
            self.avg_fiti.configure(bootstyle=SUCCESS); tag = '🟢 低频 (<0.3)'
        else:
            self.avg_fiti.configure(bootstyle=WARNING); tag = '🟡 中等 (0.3-0.7)'
        self.avg_tag.configure(text=tag)
        self.avg_iti.configure(text=f'{avg_i:.3f}', bootstyle=WARNING)

        self.tree.delete(*self.tree.get_children())
        for r in results:
            self.tree.insert('', 'end', values=(
                r['prov'], f"{r['fiti_p']:.3f}", f"{r['iti']:.3f}",
                f"{r['splits'][0]:.3f}", f"{r['splits'][1]:.3f}", f"{r['splits'][2]:.3f}",
            ))
        self.tree.insert('', 'end', values=(
            '平均', f"{avg_p:.3f}", f"{avg_i:.3f}",
            f"{np.mean([r['splits'][0] for r in results]):.3f}",
            f"{np.mean([r['splits'][1] for r in results]):.3f}",
            f"{np.mean([r['splits'][2] for r in results]):.3f}",
        ), tags=('avg',))
        self.tree.tag_configure('avg', background='#3b3b3b', foreground='#fbbf24')

        log = (f'phi shape: {phi.shape}, sum={phi.sum():.2f}\n'
               f'date={date}  env[:5]={env9[:5].round(1).tolist()}...\n\n'
               f'5 bundle macro 平均:\n'
               f'  fITI = {avg_p:.4f}   ({tag})\n'
               f'  ITI  = {avg_i:.4f}\n\n')
        for r in results:
            log += (f"  [{r['prov']}]  fITI={r['fiti_p']:.3f}  ITI={r['iti']:.3f}  "
                    f"M1={r['splits'][0]:.3f} M2={r['splits'][1]:.3f} M3={r['splits'][2]:.3f}\n")
        self.log.delete('1.0', 'end')
        self.log.insert('1.0', log)


def main():
    root = ttk.Window(themename='darkly')
    Stack3App(root)
    root.mainloop()


if __name__ == '__main__':
    main()
