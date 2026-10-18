# stack3 — Dual-Head Hospital-Acquired Pathogen Predictor

Desktop GUI tool: input an isolate FASTA file, output fITI high-frequency probability + stack3 ITI score (5 holdout-province macro average).

[中文版在下面 ↓](#stack3--双头院感病原预测器-1)

## Usage

**[Download `stack3_predictor.zip`](https://github.com/kevin2000s/stack3_release/releases/latest/download/stack3_predictor.zip)** (Release asset, ~620 MB compressed / 1.1 GB extracted) → extract → double-click `stack3_predictor/stack3_predictor.exe`.

> The whole `stack3_predictor/` folder is required (includes the `_internal/` dependencies).
> Copying only the `.exe` out will fail with missing sklearn / torch / pyrodigal errors.
> On first run Windows SmartScreen may warn (unsigned binary) — click "More info → Run anyway"; some antivirus software may flag PyInstaller bundles as a false positive — add an exception if so.

## GUI Input

| Area | Field | Default |
|---|---|---|
| ① Genome | FASTA file (.fasta/.fa/.fna/.ffn) | empty, to be selected |
| ② Environment | PM2.5 / PM10 / SO2 / NO2 / CO / O3 / AQI / T_mean / RH (9-dim) | 35 / 60 / 10 / 25 / 1.0 / 70 / 60 / 20 / 65 |
| ② History | `log(tc_n)` / `freq` / `log(stratum_n)` (3-dim) | 2.0 / 0.3 / 3.0 |
| ② Time | Sampling date `YYYY-MM-DD` | 2026-06-15 |

Test sample: `examples/sample.fasta` (6 MB) + precomputed `examples/sample_phi.npy` (104 KB, skips DNABERT-S recomputation, results in seconds).

## GUI Output

- **fITI binary probability** (M1 single, 5-bundle macro average)
  - 🔴 > 0.7 high frequency
  - 🟡 0.3–0.7 medium
  - 🟢 < 0.3 low
- **stack3 ITI score** (3 base models + meta-Ridge ensemble)
- Per-province table for 5 provinces: prov / fiti / iti / M1 / M2 / M3

## Model

| Evaluation | fITI hi AUC | ITI dir_AUC |
|---|---:|---:|
| LOPO (5-province macro) | **0.711** | **0.935** |
| test1 (5-bundle macro) | **0.709** | **0.888** |

### Architecture

```
FASTA
  ↓ pyrodigal ORFs (~600)
  ↓ DNABERT-S embed → (N, 768)
  ↓ soft assign → bundle.centroids (2217 family × 12 centers)
  → phi (26604,)

phi + env (9) + day (sin/cos) + dep (3)
  + KNN features (8/24 dim, nearest neighbors from extras/train_pool_phi.npy)
  → M1 (22 dim) → fITI prob
  → M1+M2+M3 (3 base) → meta-Ridge → stack3 ITI score

× 5 bundles (广东/河北/四川/河南/浙江)
  → 5-province macro average
```

## System Requirements

| Item | Packaged exe | `python main.py` |
|---|---|---|
| Python | Not needed | 3.9+ |
| Dependencies | Bundled (sklearn + torch + transformers + pyrodigal + ttkbootstrap etc., ~470 MB) | `pip install -r requirements.txt` |
| RAM | ≥ 4 GB | ≥ 4 GB |
| DNABERT-S cache | Not needed (exe embeds transformers with local .pyd) | Requires `~/.cache/huggingface/models--zhihan1996--DNABERT-S/` |
| Platform | Windows (repackage for each OS) | Any platform with Python |

## Repackaging (developers)

```bash
pip install -r requirements.txt pyinstaller

bash build.sh        # Linux/macOS
build.bat            # Windows
```

Output: `dist/stack3_predictor/stack3_predictor.exe`.

## Python API (no GUI)

```python
import numpy as np
from prediction_toolkit.gui.app import predict_all_bundles

phi = np.load('your_phi.npy')              # (26604,)
env9 = np.array([35, 60, 10, 25, 1, 70, 60, 20, 65], np.float32)
dep3 = np.array([2.0, 0.3, 3.0], np.float32)
date = '2026-06-15'

results = predict_all_bundles(phi, env9, date, dep3)
for r in results:
    print(r['prov'], f"fITI={r['fiti_p']:.3f}", f"ITI={r['iti']:.3f}")
```

## License

MIT

---
---

# stack3 — 双头院感病原预测器

桌面 GUI 工具，输入 isolate FASTA，输出 fITI 高频概率 + stack3 ITI 分数（5 个留出省 macro 平均）。

## 使用

**[下载 `stack3_predictor.zip`](https://github.com/kevin2000s/stack3_release/releases/latest/download/stack3_predictor.zip)**（Release asset，~620 MB 压缩 / 1.1 GB 解压）→ 解压 → 双击 `stack3_predictor/stack3_predictor.exe`。

> 整个 `stack3_predictor/` 文件夹都必需（含 `_internal/` 依赖）。
> 只把 `.exe` 单独拷走会因找不到 sklearn / torch / pyrodigal 报错。
> Windows 首次运行如弹出 SmartScreen（未签名），点「更多信息 → 仍要运行」；杀软误报 PyInstaller 打包时请添加信任。

## GUI 输入

| 区域 | 字段 | 默认值 |
|---|---|---|
| ① 基因组 | FASTA 文件 (.fasta/.fa/.fna/.ffn) | 留空待选 |
| ② 环境 | PM2.5 / PM10 / SO2 / NO2 / CO / O3 / AQI / T_mean / RH（9 维） | 35 / 60 / 10 / 25 / 1.0 / 70 / 60 / 20 / 65 |
| ② 历史 | `log(tc_n)` / `freq` / `log(stratum_n)`（3 维） | 2.0 / 0.3 / 3.0 |
| ② 时间 | 采样日期 `YYYY-MM-DD` | 2026-06-15 |

测试样本见 `examples/sample.fasta`（6 MB）+ 预计算 `examples/sample_phi.npy`（104 KB，跳过 DNABERT-S 重计算，秒级）。

## GUI 输出

- **fITI 二分类概率**（M1 single，5 bundle macro 平均）
  - 🔴 > 0.7 高频
  - 🟡 0.3–0.7 中等
  - 🟢 < 0.3 低频
- **stack3 ITI 分数**（3 base + meta-Ridge 集成）
- 5 省逐项表格：prov / fiti / iti / M1 / M2 / M3

## 模型

| 评测 | fITI hi AUC | ITI dir_AUC |
|---|---:|---:|
| LOPO（5 省 macro） | **0.711** | **0.935** |
| test1（5-bundle macro） | **0.709** | **0.888** |

### 架构

```
FASTA
  ↓ pyrodigal ORFs (~600)
  ↓ DNABERT-S embed → (N, 768)
  ↓ soft assign → bundle.centroids (2217 family × 12 centers)
  → phi (26604,)

phi + env (9) + day (sin/cos) + dep (3)
  + KNN features (8/24 dim, 从 extras/train_pool_phi.npy 算最近邻)
  → M1 (22 dim) → fITI prob
  → M1+M2+M3 (3 base) → meta-Ridge → stack3 ITI score

× 5 bundle (广东/河北/四川/河南/浙江)
  → 5 省 macro 平均
```

## 系统需求

| 项 | 用打包 exe | 用 `python main.py` |
|---|---|---|
| Python | 不需要 | 3.9+ |
| 依赖 | 内嵌（sklearn + torch + transformers + pyrodigal + ttkbootstrap 等 ~470 MB） | `pip install -r requirements.txt` |
| 内存 | ≥ 4 GB | ≥ 4 GB |
| DNABERT-S 缓存 | 不需要（exe 内部 transformers 用本地 .pyd） | 需 `~/.cache/huggingface/models--zhihan1996--DNABERT-S/` |
| 平台 | Windows（其他 OS 需重新打包） | 全平台 Python |

## 重新打包（开发者）

```bash
pip install -r requirements.txt pyinstaller

bash build.sh        # Linux/macOS
build.bat            # Windows
```

产物 `dist/stack3_predictor/stack3_predictor.exe`。

## Python 调用（不开 GUI）

```python
import numpy as np
from prediction_toolkit.gui.app import predict_all_bundles

phi = np.load('your_phi.npy')              # (26604,)
env9 = np.array([35, 60, 10, 25, 1, 70, 60, 20, 65], np.float32)
dep3 = np.array([2.0, 0.3, 3.0], np.float32)
date = '2026-06-15'

results = predict_all_bundles(phi, env9, date, dep3)
for r in results:
    print(r['prov'], f"fITI={r['fiti_p']:.3f}", f"ITI={r['iti']:.3f}")
```

## License

MIT
