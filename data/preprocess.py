"""
统一数据预处理：先按数据集布局分集，再独立加噪、滑窗，最后每窗 per-sample z-score。

关键设计（避免数据泄露）：
  1) PU（multi_file_per_class）：按文件个数比例切 train/val/test，每个 .mat 整体属于一个集
     —— 杜绝同一录音跨集出现的"录音级泄露"
  2) CWRU（flat_filename_class）：.mat 平铺在根目录，每文件 = 一类（取含 'DE' 的通道）；
     每个文件切成 flat_num_blocks 等长块（块间留 buffer），shuffle 后按比例分到
     train/val/test —— 三个 split 在原始时序上交错分布，每个 split 都覆盖整段
     录音，**同时杜绝跨块时间相邻泄露**（buffer 隔离 + 块内独立滑窗）。
     CWRU 数据小，train 窗用高重叠凑样本（仅同 split 块内重叠，不跨集）。
  3) 每段独立加 AWGN（每段独立 RNG，可复现且彼此无样本级相关）
  4) train 滑窗可重叠当增广；val/test 重叠更小（CWRU）或不重叠（PU）
  5) per-sample z-score（每窗自归一化）：不依赖任何跨集统计量 → 无 scaler 泄露

对外只暴露一个函数：
    build_domain_dataset(cfg, snr)  → dict 含 X/Y for train/val/test + scaler

train.py / deploy.py 都通过它取数据，互不串味。
"""

import hashlib
import json
import os
import zlib
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.io import loadmat

from config import (
    Config, DATA_ROOTS, SIGNAL_KEYS, DATASET_LAYOUTS, FLAT_DATASETS,
    get_pu_classes, get_cwru_classes,
)


# ──────────────────────────────────────────────────────────
# .mat 信号提取
# ──────────────────────────────────────────────────────────

def _extract_pu_struct(obj) -> np.ndarray:
    """从 PU 的 MATLAB 结构体里抽取 vibration_1 通道。失败返回 None。"""
    try:
        dtype_names = getattr(obj, 'dtype', None)
        dtype_names = None if dtype_names is None else obj.dtype.names
        if not dtype_names or 'Y' not in dtype_names:
            return None
        y_field = obj[0, 0]['Y']
        for elem in np.ravel(y_field):
            e_names = getattr(elem, 'dtype', None)
            e_names = None if e_names is None else elem.dtype.names
            if not e_names or 'Name' not in e_names or 'Data' not in e_names:
                continue
            name_cell = np.ravel(elem['Name'])
            if name_cell.size == 0:
                continue
            if str(name_cell[0]).strip() != 'vibration_1':
                continue
            arr = np.asarray(elem['Data']).ravel()
            if arr.size >= 1000 and np.issubdtype(arr.dtype, np.number):
                return arr.astype(np.float32)
    except Exception:
        return None
    return None


def _load_pu_signal(path: str) -> np.ndarray:
    """从单个 PU .mat 文件提取 vibration_1 振动信号。"""
    mat = loadmat(path)
    for key, val in mat.items():
        if key.startswith('__'):
            continue
        sig = _extract_pu_struct(np.asarray(val))
        if sig is not None:
            return sig
    raise RuntimeError(f"无法从 {path} 提取 PU vibration_1 通道")


def _load_cwru_signal(path: str, key: str = 'DE') -> np.ndarray:
    """
    从单个 CWRU .mat 文件提取 Drive-End 振动信号（参考 QC-DANN 的 _extract_cwru_signal）。
    CWRU 是普通 .mat（loadmat 可读），DE 通道键名形如 'X118_DE_time'：
      先精确匹配 key，再模糊匹配含 key（默认 'DE'）的数值键。
    """
    mat = loadmat(path)
    # 精确匹配优先
    cand = []
    if key in mat:
        cand.append(key)
    cand += [k for k in mat if (not k.startswith('__')) and (key in k) and k != key]
    for k in cand:
        arr = np.asarray(mat[k]).ravel()
        if arr.size >= 1000 and np.issubdtype(arr.dtype, np.number):
            return arr.astype(np.float32)
    avail = [k for k in mat if not k.startswith('__')]
    raise RuntimeError(f"CWRU {os.path.basename(path)} 中找不到含 '{key}' 的有效信号键，"
                       f"可用键: {avail}")




# ──────────────────────────────────────────────────────────
# 类目录扫描
# ──────────────────────────────────────────────────────────

def _list_pu_files(root: str, classes: List[str]) -> Dict[str, List[str]]:
    """{class_name: [.mat 绝对路径列表 sorted]} —— 缺类直接报错，便于及早发现配置问题。"""
    if not os.path.isdir(root):
        raise FileNotFoundError(f"PU 数据根目录不存在: {root}")
    out: Dict[str, List[str]] = {}
    for cname in classes:
        cdir = os.path.join(root, cname)
        if not os.path.isdir(cdir):
            raise FileNotFoundError(f"PU 类目录不存在: {cdir}")
        files = sorted(f for f in os.listdir(cdir) if f.lower().endswith('.mat'))
        if not files:
            raise RuntimeError(f"PU {cname} 目录下没有 .mat 文件")
        out[cname] = [os.path.join(cdir, f) for f in files]
    return out


def _list_flat_files(root: str, classes: List[str]) -> Dict[str, List[str]]:
    """
    平铺布局（CWRU）：{class_name: [.mat 绝对路径列表 sorted]}

    每个类的名字直接作为文件名前缀。文件名以 (cname + '_') 或 (cname + '.')
    开头即归入该类（前缀边界检查避免误匹配）。CWRU 用整文件名 stem，
    'stem.mat' 命中 cname+'.' 分支，每类对应一个文件。
    """
    if not os.path.isdir(root):
        raise FileNotFoundError(f"数据根目录不存在: {root}")
    by_class: Dict[str, List[str]] = {c: [] for c in classes}
    skipped: List[str] = []
    for fname in sorted(os.listdir(root)):
        if not fname.lower().endswith('.mat'):
            continue
        matched = None
        for cname in classes:
            if fname.startswith(cname + '_') or fname.startswith(cname + '.'):
                matched = cname
                break
        if not matched:
            skipped.append(fname)
            continue
        by_class[matched].append(os.path.join(root, fname))
    for fname in skipped:
        print(f"[WARN] 文件 {fname} 不匹配任何白名单前缀，跳过")
    missing = [c for c, f in by_class.items() if not f]
    if missing:
        raise RuntimeError(
            f"以下类无可用文件: {missing}（白名单大小={len(classes)}）"
        )
    return by_class


# ──────────────────────────────────────────────────────────
# 切分 / 加噪 / 滑窗
# ──────────────────────────────────────────────────────────

def _split_pu_files(files: List[str], ratios: Tuple[float, float, float],
                    rng: np.random.RandomState) -> Dict[str, List[str]]:
    """文件级随机切分。至少保证 train/val/test 各 1 个文件。"""
    n = len(files)
    if n < 3:
        raise RuntimeError(f"PU 类下文件 < 3 个 ({n})，无法做三段切分")
    n_tr = max(1, int(round(n * ratios[0])))
    n_val = max(1, int(round(n * ratios[1])))
    n_te = n - n_tr - n_val
    # 兜底：保证至少各 1 个
    while n_te < 1:
        if n_tr > 1:
            n_tr -= 1
        else:
            n_val -= 1
        n_te = n - n_tr - n_val
    perm = rng.permutation(n)
    train_idx = sorted(perm[:n_tr].tolist())
    val_idx   = sorted(perm[n_tr:n_tr + n_val].tolist())
    test_idx  = sorted(perm[n_tr + n_val:].tolist())
    return {
        'train': [files[i] for i in train_idx],
        'val':   [files[i] for i in val_idx],
        'test':  [files[i] for i in test_idx],
    }


def _block_shuffle(sig: np.ndarray, ratios: Tuple[float, float, float],
                   n_blocks: int, buffer: int,
                   rng: np.random.RandomState) -> Dict[str, List[np.ndarray]]:
    """
    把一条长信号切成 n_blocks 等长块，shuffle 后按比例分到 train/val/test。

    设计要点：
      - 块间留 buffer 个样本作为隔离带（最后一块尾部不留），
        保证没有任何滑窗能跨越块边界 → 杜绝跨块的时间相邻泄露。
      - 块洗牌后三个 split 在原始时序上交错分布，每个 split 都能覆盖整段录音
        的工况漂移（温度/转速波动），消除顺序三段切的严重分布偏移。
      - 块级 split rng 不依赖 SNR，所有域共享同一份块划分；训练-部署一致。

    返回 {split: [block_ndarray, ...]}（每 split 是一组块，后续每块独立加噪+滑窗）。
    """
    n = len(sig)
    b = max(0, int(buffer))
    # 总占用 = n_blocks * block_len + (n_blocks - 1) * b ≤ n
    block_len = (n - (n_blocks - 1) * b) // n_blocks
    if block_len < 1:
        raise RuntimeError(
            f"信号长度 {n} 切 {n_blocks} 块（块间 buffer={b}）→ block_len<1，"
            f"请减小 flat_num_blocks 或 split_buffer_factor")

    blocks: List[np.ndarray] = []
    pos = 0
    for _ in range(n_blocks):
        blocks.append(sig[pos: pos + block_len])
        pos += block_len + b

    # 比例分配 → 至少各 1 块
    n_tr = max(1, int(round(n_blocks * ratios[0])))
    n_val = max(1, int(round(n_blocks * ratios[1])))
    n_te = n_blocks - n_tr - n_val
    while n_te < 1:
        if n_tr > 1:
            n_tr -= 1
        else:
            n_val -= 1
        n_te = n_blocks - n_tr - n_val

    idx = rng.permutation(n_blocks)
    return {
        'train': [blocks[i] for i in idx[:n_tr]],
        'val':   [blocks[i] for i in idx[n_tr:n_tr + n_val]],
        'test':  [blocks[i] for i in idx[n_tr + n_val:]],
    }


def _add_awgn(sig: np.ndarray, snr, rng: np.random.RandomState) -> np.ndarray:
    """加 AWGN；snr=None 表示 clean 域，原样返回（保留 dtype）。"""
    sig = sig.astype(np.float32)
    if snr is None:
        return sig.copy()
    p = float(np.mean(sig ** 2))
    if p <= 1e-12:
        return sig.copy()
    n_power = p / (10.0 ** (snr / 10.0))
    noise = rng.normal(0.0, np.sqrt(n_power), sig.shape).astype(np.float32)
    return sig + noise


def _sliding_window(sig: np.ndarray, length: int, step: int,
                    max_n: int, rng: np.random.RandomState) -> np.ndarray:
    """按固定步长滑窗，返回 (n_windows, length)。窗数 > max_n 则随机抽取 max_n 个。"""
    if len(sig) < length:
        return np.zeros((0, length), dtype=np.float32)
    starts = list(range(0, len(sig) - length + 1, max(step, 1)))
    if not starts:
        return np.zeros((0, length), dtype=np.float32)
    if len(starts) > max_n:
        idx = rng.choice(len(starts), size=max_n, replace=False)
        starts = [starts[i] for i in sorted(idx)]
    return np.stack([sig[s:s + length] for s in starts], axis=0).astype(np.float32)


def _stable_seed(base: int, *tags) -> int:
    """字符串-> CRC32 -> 32 位种子。完全确定，跨 Python 会话一致。"""
    s = f"{int(base)}|" + "|".join(str(t) for t in tags)
    return int(zlib.crc32(s.encode('utf-8'))) & 0xFFFFFFFF


def _per_sample_zscore(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    每窗内做 z-score：(x - x.mean()) / (x.std() + eps)。

    替代原来的 sklearn StandardScaler（per-feature, 按时间位置独立标准化）。
    选用 per-sample 的两个原因：
      1) 振动信号每个时间位置语义相同，per-feature 归一化在源域 σ 小的维度
         上会被噪声放大几个量级，导致跨 SNR 数值不稳；
      2) 不依赖任何源域统计，clean / noisy 都被归到 N(0,1) 附近，
         训练-部署一致，无 scaler.npz 需要保存。
    """
    X = X.astype(np.float32, copy=False)
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    return ((X - mu) / (sd + eps)).astype(np.float32)


# ──────────────────────────────────────────────────────────
# 加噪数据磁盘缓存（PU / CWRU 通用）
# ──────────────────────────────────────────────────────────

def _cache_hyper_dict(cfg: Config) -> Dict:
    """
    凡是影响输出 X/Y 的超参全列进来，hash 由此生成。
    PU / CWRU 共用基础字段，CWRU 额外加 split_buffer_factor 与 flat_num_blocks。
    """
    base: Dict = {
        'dataset': cfg.dataset,
        'ratios': [cfg.train_ratio, cfg.val_ratio, cfg.test_ratio],
        'signal_length': cfg.signal_length,
        'enc_step_train': cfg.enc_step_train,
        'enc_step_val': cfg.enc_step_val,
        'enc_step_test': cfg.enc_step_test,
        'num_train_per_class': cfg.num_train_per_class,
        'num_val_per_class': cfg.num_val_per_class,
        'num_test_per_class': cfg.num_test_per_class,
        'seed': cfg.seed,
    }
    if cfg.dataset == 'PU':
        base['classes'] = list(get_pu_classes(cfg))
        base['pu_class_set'] = getattr(cfg, 'pu_class_set', '10class')
    elif cfg.dataset in FLAT_DATASETS:
        base['classes'] = list(get_cwru_classes(cfg))
        base['split_buffer_factor'] = cfg.split_buffer_factor
        base['flat_num_blocks'] = cfg.flat_num_blocks
    return base


def _cache_key(cfg: Config) -> str:
    """所有 hyper → JSON → MD5 前 10 位。任一项变都换 hash 目录。"""
    meta = _cache_hyper_dict(cfg)
    raw = json.dumps(meta, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(raw.encode('utf-8')).hexdigest()[:10]


def _safe_snr_filename(snr) -> str:
    """生成文件系统友好的 SNR 标识：None→clean / 6→p6dB / -3→n3dB。"""
    if snr is None:
        return 'clean'
    sign = 'p' if snr >= 0 else 'n'
    return f'{sign}{abs(int(snr))}dB'


def _cache_dir_for(cfg: Config) -> str:
    """cache_dir/<dataset>/<hash>/ —— 不同数据集物理隔离，互不污染。"""
    return os.path.join(cfg.cache_dir, cfg.dataset, _cache_key(cfg))


def _cache_path(cfg: Config, snr) -> str:
    return os.path.join(_cache_dir_for(cfg), _safe_snr_filename(snr) + '.npz')


def _try_load_cache(path: str) -> Optional[Dict[str, np.ndarray]]:
    if not os.path.isfile(path):
        return None
    try:
        npz = np.load(path)
        out = {k: npz[k] for k in ('X_train', 'Y_train', 'X_val', 'Y_val', 'X_test', 'Y_test')}
        npz.close()
        return out
    except Exception as e:
        print(f"[CACHE] 读 {path} 失败 ({e})，将重新生成")
        return None


def _save_cache(cfg: Config, snr, out: Dict[str, np.ndarray]):
    """把未归一化的 X/Y 写到 .npz，同时在目录里留 _meta.json 说明超参。"""
    dir_path = _cache_dir_for(cfg)
    os.makedirs(dir_path, exist_ok=True)
    meta_path = os.path.join(dir_path, '_meta.json')
    if not os.path.isfile(meta_path):
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(_cache_hyper_dict(cfg), f, ensure_ascii=False, indent=2)
    path = _cache_path(cfg, snr)
    np.savez(path,
             X_train=out['X_train'], Y_train=out['Y_train'],
             X_val=out['X_val'],     Y_val=out['Y_val'],
             X_test=out['X_test'],   Y_test=out['Y_test'])
    return path


# ──────────────────────────────────────────────────────────
# 对外公共接口
# ──────────────────────────────────────────────────────────

def _resolve_class_list(cfg: Config) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    返回 (class_names_in_label_order, {class_name: [file_paths]})。
    类别标签 = 列表索引。两种布局统一返回"类 → 文件列表"映射。
    """
    layout = DATASET_LAYOUTS[cfg.dataset]
    root = DATA_ROOTS[cfg.dataset]
    if layout == 'multi_file_per_class':
        files_by_cls = _list_pu_files(root, get_pu_classes(cfg))
    elif layout == 'flat_filename_class':
        files_by_cls = _list_flat_files(root, get_cwru_classes(cfg))
    else:
        raise ValueError(f"未知 DATASET_LAYOUTS[{cfg.dataset}]={layout}")
    return list(files_by_cls.keys()), files_by_cls


def build_domain_dataset(cfg: Config, snr: int) -> Dict[str, np.ndarray]:
    """
    构造单个 SNR 域下的 train/val/test 数据。

    流程：
      - PU: 类文件夹下按文件个数比例切分 → 每个 .mat 整体属于一个 split → 文件内独立加噪+滑窗
      - CWRU: 平铺 .mat 每文件一类（取含 'DE' 通道）→ 块洗牌切三集
              （块间留 buffer 隔离带）→ 各块独立加噪+滑窗
      - 拼合 X/Y → per-sample z-score 归一化（每窗独立，跨 SNR 一致，无需 scaler）

    返回 dict:
      X_train, Y_train, X_val, Y_val, X_test, Y_test
    """
    layout = DATASET_LAYOUTS[cfg.dataset]
    ratios = (cfg.train_ratio, cfg.val_ratio, cfg.test_ratio)
    split_cfg = {
        'train': (cfg.enc_step_train, cfg.num_train_per_class),
        'val':   (cfg.enc_step_val,   cfg.num_val_per_class),
        'test':  (cfg.enc_step_test,  cfg.num_test_per_class),
    }

    # ── 磁盘缓存（PU / UM 通用）：命中则跳过生成，直接进入归一化 ──
    cache_enabled = getattr(cfg, 'use_cache', False)
    cache_path = _cache_path(cfg, snr) if cache_enabled else None
    cached_out = _try_load_cache(cache_path) if cache_path else None

    if cached_out is not None:
        print(f"[CACHE] {cfg.dataset} snr={_safe_snr_filename(snr)} 命中 → {cache_path}")
        out: Dict[str, np.ndarray] = cached_out
    else:
        out = _generate_dataset(cfg, snr, layout, ratios, split_cfg)
        if cache_path is not None:
            saved = _save_cache(cfg, snr, out)
            print(f"[CACHE] {cfg.dataset} snr={_safe_snr_filename(snr)} 已写 → {saved}")

    # ── 归一化：per-sample z-score，跨 SNR / 训练-部署完全一致 ──
    for split in ('train', 'val', 'test'):
        out[f'X_{split}'] = _per_sample_zscore(out[f'X_{split}'])

    return out


def _generate_dataset(cfg: Config, snr, layout: str, ratios, split_cfg) -> Dict[str, np.ndarray]:
    """从原始 .mat 走完整生成流程（切分→加噪→滑窗→拼合），返回未归一化的 X/Y。"""
    Xs: Dict[str, List[np.ndarray]] = {'train': [], 'val': [], 'test': []}
    Ys: Dict[str, List[np.ndarray]] = {'train': [], 'val': [], 'test': []}

    class_names, files_by_cls = _resolve_class_list(cfg)

    if layout == 'multi_file_per_class':
        # ── PU：文件级切分（每个 .mat 整体属于一个 split）──────
        for cls, cname in enumerate(class_names):
            files = files_by_cls[cname]
            # 文件分配与 snr 无关 → 所有域共享同一文件划分；训练-部署也一致。
            split_rng = np.random.RandomState(
                _stable_seed(cfg.seed, cfg.dataset, 'file_split', cname))
            file_splits = _split_pu_files(files, ratios, split_rng)

            for split, file_list in file_splits.items():
                step, max_n = split_cfg[split]
                if not file_list:
                    continue
                per_file_max = max(1, int(np.ceil(max_n / len(file_list))))
                for fi, fpath in enumerate(file_list):
                    sig = _load_pu_signal(fpath)
                    if len(sig) < cfg.signal_length:
                        print(f"[WARN] PU {cname}/{os.path.basename(fpath)} "
                              f"长度={len(sig)} < {cfg.signal_length}, 跳过")
                        continue
                    rng = np.random.RandomState(_stable_seed(
                        cfg.seed, cfg.dataset, snr, split, cname, fi))
                    noised = _add_awgn(sig, snr, rng)
                    windows = _sliding_window(noised, cfg.signal_length, step,
                                              per_file_max, rng)
                    if windows.shape[0] > 0:
                        Xs[split].append(windows)
                        Ys[split].append(np.full(windows.shape[0], cls, dtype=np.int64))

    elif layout == 'flat_filename_class':
        # ── CWRU：每类一个文件，块洗牌切 + buffer 隔离 ───────
        # 每文件加载一次后切成 flat_num_blocks 块、shuffle 分到三集，
        # 每块独立加噪+滑窗 → 块边界天然不会有滑窗跨越（防跨集泄露）。
        buffer = int(round(cfg.signal_length * cfg.split_buffer_factor))
        for cls, cname in enumerate(class_names):
            files = files_by_cls[cname]
            n_files = len(files)
            for fi, fpath in enumerate(files):
                sig = _load_cwru_signal(fpath, SIGNAL_KEYS[cfg.dataset])
                # 块切分 rng 不依赖 snr → 跨域、训练-部署使用同一份块划分
                split_rng = np.random.RandomState(_stable_seed(
                    cfg.seed, cfg.dataset, 'block_split', cname, fi))
                parts = _block_shuffle(sig, ratios, cfg.flat_num_blocks,
                                       buffer, split_rng)
                del sig
                for split, block_list in parts.items():
                    step, max_n = split_cfg[split]
                    n_blk = len(block_list)
                    if n_blk == 0:
                        continue
                    # 同类多文件 × 多块均分配额，保证 num_*_per_class 上限稳定
                    per_block_max = max(1, int(np.ceil(
                        max_n / max(1, n_files * n_blk))))
                    for bi, seg in enumerate(block_list):
                        if len(seg) < cfg.signal_length:
                            print(f"[WARN] {cfg.dataset} {cname}/{os.path.basename(fpath)} "
                                  f"split={split} block={bi} 长度={len(seg)} "
                                  f"< {cfg.signal_length}, 跳过")
                            continue
                        rng = np.random.RandomState(_stable_seed(
                            cfg.seed, cfg.dataset, snr, split, cname, fi, bi))
                        noised = _add_awgn(seg, snr, rng)
                        windows = _sliding_window(noised, cfg.signal_length, step,
                                                  per_block_max, rng)
                        if windows.shape[0] > 0:
                            Xs[split].append(windows)
                            Ys[split].append(np.full(windows.shape[0], cls, dtype=np.int64))

    else:
        raise ValueError(f"未知 DATASET_LAYOUTS[{cfg.dataset}]={layout}")

    # ── 拼合 + 类间 shuffle（未归一化）───────────────────
    out: Dict[str, np.ndarray] = {}
    for split in ('train', 'val', 'test'):
        if not Xs[split]:
            raise RuntimeError(f"{cfg.dataset} snr={snr} split={split} 无可用样本")
        X = np.concatenate(Xs[split], axis=0)
        Y = np.concatenate(Ys[split], axis=0)
        rng = np.random.RandomState(_stable_seed(
            cfg.seed, cfg.dataset, snr, split, '__shuffle__'))
        idx = rng.permutation(len(X))
        out[f'X_{split}'] = X[idx]
        out[f'Y_{split}'] = Y[idx]

    return out


def get_num_classes(cfg: Config) -> int:
    """从白名单推断类别数（用于建模时确定输出维度）。"""
    if cfg.dataset == 'PU':
        return len(get_pu_classes(cfg))
    if cfg.dataset in FLAT_DATASETS:
        return len(get_cwru_classes(cfg))
    return 10
