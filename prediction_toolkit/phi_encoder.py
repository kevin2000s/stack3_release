"""phi_encoder.py - raw isolate FASTA -> 26604 dim phi.

Pipeline:
  1. pyrodigal ORF 预测 (meta 模式, 不需训练)
  2. DNABERT-S (HF local cache) 嵌入每个 ORF -> (n, 768)
  3. 与 bundle 的 2217 家族 centroids 软分配 -> (26604,) phi

DNABERT-S 加载: 用 importlib 直接 load HF cached custom code (绕过 transformers
trust_remote_code 触发的 triton 强制要求). Vocab=4096 -> 6-mer.
"""
import os
import sys
import warnings
import importlib.util

import numpy as np

warnings.filterwarnings('ignore')

_DNABERT = None  # cached (model, tokenizer)
_CENTROIDS_FLAT = None  # cached (n_families*12, 768) + family_idx lookup


# ============ pyrodigal ORF ============

def predict_orfs(fasta_path, min_len=180, max_len=6000, max_orfs=2048):
    """返回 (N,) 个 ORF DNA 序列 list. meta mode 不需训练."""
    seqs = []
    cur_id, cur = None, []
    with open(fasta_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if cur:
                    seqs.append((cur_id, ''.join(cur).upper()))
                cur_id = line[1:].split()[0]
                cur = []
            else:
                cur.append(line)
        if cur:
            seqs.append((cur_id, ''.join(cur).upper()))

    import pyrodigal
    orfs = []
    for sid, seq in seqs:
        if len(seq) < min_len * 5:
            continue
        gf = pyrodigal.GeneFinder(meta=True)
        try:
            genes = gf.find_genes(seq.encode())
        except Exception:
            continue
        for g in genes:
            sub = seq[g.begin:g.end]
            if min_len <= len(sub) <= max_len:
                orfs.append(sub)
                if len(orfs) >= max_orfs:
                    return orfs
    return orfs


# ============ DNABERT-S ============

def _load_dnabert():
    """importlib 加载 HF cached DNABERT-S custom code (避 triton)."""
    global _DNABERT
    if _DNABERT is not None:
        return _DNABERT

    import torch
    from transformers import AutoTokenizer

    snap_dir = None
    hf_cache = os.path.expanduser('~/.cache/huggingface')
    if os.path.isdir(hf_cache):
        snap_root = os.path.join(hf_cache, 'hub',
                                 'models--zhihan1996--DNABERT-S', 'snapshots')
        if os.path.isdir(snap_root):
            for d in os.listdir(snap_root):
                p = os.path.join(snap_root, d)
                if os.path.isdir(p) and os.path.exists(os.path.join(p, 'config.json')):
                    snap_dir = p
                    break
    if snap_dir is None:
        # 搜索 .pyc 备份等其他路径
        candidates = []
        for root, _, files in os.walk(hf_cache):
            for fn in files:
                if fn == 'configuration_bert.py':
                    candidates.append(os.path.dirname(root))
                    break
        if candidates:
            snap_dir = candidates[0]

    if snap_dir is None:
        raise FileNotFoundError(
            '本地 HF cache 找不到 DNABERT-S. 请确认 ~/.cache/huggingface/'
            'models--zhihan1996--DNABERT-S/snapshots/<hash>/ 存在.')

    # 用 importlib 把 snap_dir 当 package 'dnabert_pkg'
    # 删除已加载缓存 (防止 retry)
    for k in list(sys.modules):
        if k.startswith('dnabert_pkg'):
            del sys.modules[k]

    init_path = os.path.join(snap_dir, '__init__.py')
    spec_pkg = importlib.util.spec_from_file_location('dnabert_pkg', init_path)
    pkg = importlib.util.module_from_spec(spec_pkg)
    sys.modules['dnabert_pkg'] = pkg
    try:
        spec_pkg.loader.exec_module(pkg)
    except Exception:
        pass  # 没有 __init__.py 也行 (继续)

    for sub in ('configuration_bert', 'bert_padding', 'bert_layers'):
        s = importlib.util.spec_from_file_location(
            f'dnabert_pkg.{sub}', os.path.join(snap_dir, f'{sub}.py'))
        m = importlib.util.module_from_spec(s)
        sys.modules[f'dnabert_pkg.{sub}'] = m
        s.loader.exec_module(m)

    from dnabert_pkg.configuration_bert import BertConfig
    from dnabert_pkg.bert_layers import BertModel

    with open(os.path.join(snap_dir, 'config.json')) as f:
        cfg = BertConfig(**json.load(f) if False else __import__('json').load(f))
    sd = torch.load(os.path.join(snap_dir, 'pytorch_model.bin'),
                    map_location='cpu', weights_only=True)
    mdl = BertModel(cfg, add_pooling_layer=False)
    mdl.load_state_dict(sd, strict=False)
    mdl.eval()
    # CPU 跑 (用户机器没 GPU)
    mdl = mdl.to('cpu')

    tok = AutoTokenizer.from_pretrained(snap_dir)
    _DNABERT = (mdl, tok)
    return _DNABERT


def _kmerize(seq, k=6):
    if len(seq) < k:
        return ''
    seq = ''.join(c if c in 'ACGT' else 'N' for c in seq.upper())
    return ' '.join(seq[i:i+k] for i in range(0, len(seq) - k + 1, k))


def embed_orfs(orfs, batch_size=8, max_length=512):
    """(N,) orfs -> (N, 768) float32."""
    import torch
    mdl, tok = _load_dnabert()

    out = np.zeros((len(orfs), 768), dtype=np.float32)
    for i in range(0, len(orfs), batch_size):
        batch = orfs[i:i + batch_size]
        kmers = [_kmerize(s) for s in batch]
        inp = tok(kmers, padding=True, truncation=True, max_length=max_length,
                  return_tensors='pt')
        with torch.no_grad():
            o = mdl(**inp)
            try:
                cls = o.last_hidden_state[:, 0, :]
            except AttributeError:
                cls = o[0][:, 0, :]
        out[i:i + len(batch)] = cls.cpu().numpy()
    return out


# ============ soft assign to bundle centroids ============

def _flatten_centroids(centroids):
    """centroids: dict[family_id] -> (12, 768). 返回:
       - flat: (n_families*12, 768) 所有 centroids 排平
       - family_of_center: (n_families*12,) 每个 center 所属 family
       - n_families: int
    """
    global _CENTROIDS_FLAT
    fam_keys = sorted(centroids.keys())
    flat = np.concatenate([centroids[k] for k in fam_keys], axis=0)
    family_of = np.concatenate([np.full(len(centroids[k]), k) for k in fam_keys])
    n_fam = max(fam_keys) + 1
    _CENTROIDS_FLAT = (flat, family_of, n_fam)
    return _CENTROIDS_FLAT


def phi_from_embeddings(embs, centroids, tau=12.0):
    """(N, 768) embs -> (26604,) phi."""
    flat, family_of, n_fam = _flatten_centroids(centroids)
    d_per = 12
    n_centers = flat.shape[0]

    # 找每个 ORF 的最近 center (cosine)
    en = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
    cn = flat / (np.linalg.norm(flat, axis=1, keepdims=True) + 1e-9)
    cos = en @ cn.T  # (N, n_centers)
    nearest = np.argmax(cos, axis=1)  # (N,)

    # phi init
    phi = np.zeros((n_fam, d_per), dtype=np.float32)
    counts = np.zeros(n_fam, dtype=np.int32)

    for i, c in enumerate(nearest):
        fam = int(family_of[c])
        # 该 ORF 在它自己 family 的 12 centers 上重新算 (L2 距离)
        cents = centroids.get(fam)
        if cents is None or len(cents) != d_per:
            continue
        d = np.linalg.norm(cents - embs[i], axis=1)  # (12,)
        w = np.exp(-d / tau)
        s = w.sum()
        if s > 0:
            w = w / s
        phi[fam] += w
        counts[fam] += 1

    # mean over ORFs assigned to same family
    phi = phi / np.maximum(counts, 1).reshape(-1, 1)
    return phi.reshape(-1)


# ============ end-to-end ============

def compute_phi(fasta_path, bundle, max_orfs=1024, progress=None):
    """fasta -> (26604,) phi.

    Args:
      fasta_path: raw isolate genome FASTA
      bundle: 已加载的 stack3 bundle (需要 .centroids)
      max_orfs: 限制 ORF 数量 (CPU ~1s/orf, 1024 = ~few min)
      progress: 可选 callable(stage: str, n=None)
    """
    def _p(stage, n=None):
        if progress is not None:
            try:
                progress(stage, n)
            except TypeError:
                progress(stage)

    _p('pyrodigal ORF 预测')
    orfs = predict_orfs(fasta_path, max_orfs=max_orfs)
    if not orfs:
        raise RuntimeError(f'pyrodigal 没找到 ORF: {fasta_path}')
    _p(f'找到 {len(orfs)} 个 ORF', len(orfs))

    _p(f'DNABERT-S 嵌入 {len(orfs)} ORFs')
    embs = embed_orfs(orfs, batch_size=8)
    _p('embedding done', len(orfs))

    _p('软分配到 2217 家族')
    phi = phi_from_embeddings(embs, bundle['centroids'])
    return phi


# ============ CLI ============

def main():
    import argparse, joblib
    ap = argparse.ArgumentParser()
    ap.add_argument('--fasta', required=True)
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-orfs', type=int, default=1024)
    args = ap.parse_args()

    b = joblib.load(args.bundle)
    print(f'loaded {args.bundle}, n_families={len(b["centroids"])}')

    def _p(stage, n=None):
        print(f'  [{stage}] {n or ""}')

    phi = compute_phi(args.fasta, b, max_orfs=args.max_orfs, progress=_p)
    np.save(args.out, phi.astype(np.float32))
    print(f'=> {args.out}  shape={phi.shape}')


if __name__ == '__main__':
    main()
