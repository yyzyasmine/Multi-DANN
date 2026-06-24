"""
一键部署：用训练保存的 ckpt，在 SNR_LIST 配置的所有域的 test 集上批量评测。

用法：
    python deploy.py --mode E1_QMDCS --dataset PU --seed 42

只需提供 mode/dataset/seed 三元组（与训练时一致），就能定位 ckpt：
    results/<dataset>_<mode>_seed<seed>/checkpoint.pth

自动按 SNR_LIST 顺序评测，输出 per-domain Acc + 平均 Acc + 混淆矩阵图，
结果写到：
    results/<run>/deploy_results.txt   每域准确率（人读）
    results/<run>/deploy_results.json  同上 JSON
    results/<run>/confusion_matrices.npz                 原始 cm（按域命名 key）
    results/<run>/confusion_matrices/<domain>.png        逐域 CM 热力图
    results/<run>/confusion_matrices/grid.png            所有域汇总到一张图
"""

# Windows 下 PyTorch (libiomp5md) 与 matplotlib/numpy (libomp) 的 OpenMP 双链冲突，
# 不设这个 env var 会在 plt.savefig() 时被 OMP runtime 直接 abort()。
# 必须早于任何 torch / matplotlib / numpy 的 import。
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import argparse
import json
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from config import Config, get_preset, SNR_LIST, format_snr, get_class_names
from data.preprocess import build_domain_dataset, get_num_classes
from deploy_metrics import (append_metrics_to_excel, append_mode_aggregate_to_excel,
                            average_domain_metrics, metrics_from_confusion_matrix,
                            save_metrics_json)
from models import QCDANNModel


def parse_args():
    p = argparse.ArgumentParser('Muti-QCDANN 一键部署（test 集评测 + 混淆矩阵绘制）')
    p.add_argument('--mode', type=str, default='E1_QMDCS')
    p.add_argument('--dataset', type=str, default='PU', choices=['PU', 'CWRU'],
                   help='PU(10类,64kHz) / CWRU(10类,12kHz)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--results_root', type=str, default=None,
                   help='ckpt 根目录（默认 results）')
    p.add_argument('--no_cuda', action='store_true')
    p.add_argument('--pu_class_set', type=str, default=None,
                   choices=['4class', '10class'],
                   help='仅 PU 生效：4class（默认，每型一代表 4 分类）/ 10class（旧设定）。'
                        '必须与训练时一致，否则 ckpt 与类数不匹配。')
    p.add_argument('--plot_only', action='store_true',
                   help='只从 confusion_matrices.npz 重绘 PNG，不加载 ckpt、不跑推理')
    p.add_argument('--run_dir', type=str, default=None,
                   help='已训练结果目录；指定后从这里读取 checkpoint.pth 并把部署结果写回这里')
    p.add_argument('--ckpt', type=str, default=None,
                   help='已训练 checkpoint 路径；默认 <run_dir>/checkpoint.pth 或 cfg.ckpt_path()')
    p.add_argument('--excel', type=str, default=None,
                   help='Excel workbook to append deploy Acc/F1/FDR/FPR metrics into')
    p.add_argument('--excel_sheet', type=str, default='deploy_metrics')
    p.add_argument('--aggregate_existing', action='store_true',
                   help='Scan existing run dirs for this dataset/mode and append seed rows plus mean+/-std rows to Excel only.')
    p.add_argument('--aggregate_summary_only', action='store_true',
                   help='With --aggregate_existing, append only mean+/-std rows; use after seed rows were already written.')
    return p.parse_args()


def _cfg_from_args(args) -> Config:
    cfg_data = None
    if getattr(args, 'run_dir', None):
        config_path = os.path.join(args.run_dir, 'config.json')
        if os.path.isfile(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg_data = json.load(f)

    if cfg_data:
        fields = Config.__dataclass_fields__
        cfg = Config(**{k: v for k, v in cfg_data.items() if k in fields})
    else:
        cfg = Config(mode=args.mode, dataset=args.dataset, seed=args.seed)

    if args.results_root is not None:
        cfg.results_root = args.results_root
    cfg.no_cuda = args.no_cuda
    if getattr(args, 'pu_class_set', None):
        cfg.pu_class_set = args.pu_class_set
    return cfg


def _to_loader(X, Y, batch_size=64):
    X = torch.from_numpy(np.asarray(X)).float().unsqueeze(1)
    Y = torch.from_numpy(np.asarray(Y)).long()
    return DataLoader(TensorDataset(X, Y), batch_size=batch_size, shuffle=False)


@torch.no_grad()
def _evaluate(model: QCDANNModel, loader: DataLoader, device) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    preds, gts = [], []
    correct, tot = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        p = out.argmax(1)
        correct += (p == y).sum().item()
        tot += x.size(0)
        preds.append(p.cpu().numpy())
        gts.append(y.cpu().numpy())
    acc = correct / max(tot, 1)
    return acc, np.concatenate(gts), np.concatenate(preds)


# ──────────────────────────────────────────────────────────
# 混淆矩阵绘制
# ──────────────────────────────────────────────────────────

_CM_CMAP_PU = LinearSegmentedColormap.from_list(
    'cm_blue', ['#f7fbff', '#deebf7', '#9ecae1', '#4292c6', '#08519c', '#08306b']
)
_CM_CMAP_CWRU = LinearSegmentedColormap.from_list(
    'cm_orange', ['#fff7ec', '#fee8c8', '#fdbb84', '#fc8d59', '#e34a33', '#7f2704']
)

MM_TO_INCH = 1.0 / 25.4
CM_PANEL_WIDTH_IN = 65.0 * MM_TO_INCH
FIGURE_FORMATS = ('png', 'svg', 'pdf')


def _set_publication_font(font_size: float = 7.0):
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix',
        'svg.fonttype': 'none',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'font.size': font_size,
    })


def _save_figure_formats(fig, out_path: str, dpi: int = 600,
                         bbox_inches=None, pad_inches: float = 0.01) -> List[str]:
    base, ext = os.path.splitext(out_path)
    if ext.lower() not in ('.png', '.svg', '.pdf'):
        base = out_path
    paths = []
    for fmt in FIGURE_FORMATS:
        path = f'{base}.{fmt}'
        kwargs = {}
        if fmt == 'png':
            kwargs['dpi'] = dpi
        if bbox_inches is not None:
            kwargs['bbox_inches'] = bbox_inches
            kwargs['pad_inches'] = pad_inches
        fig.savefig(path, **kwargs)
        paths.append(path)
    return paths


def _cm_cmap(dataset: str):
    return _CM_CMAP_CWRU if str(dataset).upper() == 'CWRU' else _CM_CMAP_PU


def _draw_cm_cells(ax, pct: np.ndarray, dataset: str):
    n = pct.shape[0]
    edges = np.arange(n + 1, dtype=float) - 0.5
    mesh = ax.pcolormesh(
        edges,
        edges,
        pct,
        cmap=_cm_cmap(dataset),
        vmin=0,
        vmax=1,
        shading='flat',
        edgecolors='none',
        linewidth=0,
        antialiased=False,
        rasterized=False,
    )
    mesh.set_clip_path(Rectangle((-0.5, -0.5), n, n, transform=ax.transData))
    ax.set_aspect('equal')
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    return mesh


def _normalize_cm(cm: np.ndarray) -> np.ndarray:
    """行归一化为 recall 百分比；空行（该类无样本）填 0，避免除零。"""
    cm = np.asarray(cm, dtype=np.float64)
    row_sum = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, row_sum, out=np.zeros_like(cm), where=row_sum > 0)


def _annotate_cells(ax, mat: np.ndarray, fmt: str, fontsize: int, thresh: float):
    """在每个格子上写数值；亮底用黑字，深底用白字。"""
    n_r, n_c = mat.shape
    for i in range(n_r):
        for j in range(n_c):
            v = mat[i, j]
            color = 'white' if v > thresh else '#222'
            txt = f'{v:{fmt}}'
            cell_fs = max(3.8, fontsize - 0.7) if len(txt) >= 6 else fontsize
            ax.text(j, i, txt, ha='center', va='center',
                    color=color, fontsize=cell_fs, clip_on=True, zorder=4)


def _short_class_names(names: List[str]) -> List[str]:
    """压短混淆矩阵标签。
      CWRU: '12k_Drive_End_B007_0_118' → 'B007'，'..._OR007@6_0_130' → 'OR007'，
            'normal_0_97' → 'Normal'。
      PU  : 'K001'/'KA01' 等本就很短 → 原样保留。
    """
    out = []
    for n in names:
        s = n
        if s.startswith('normal'):
            s = 'Normal'
        elif s.startswith('12k_Drive_End_'):
            s = s[len('12k_Drive_End_'):]   # 'B007_0_118'
            s = s.split('_', 1)[0]           # 'B007' / 'OR007@6'
            s = s.split('@', 1)[0]           # 去掉外圈位置标记 '@6'
        out.append(s)
    return out


def _draw_cell_boundaries(ax, n: int, linewidth: float):
    inner_boundaries = np.arange(0.5, n - 0.5, 1.0)
    if len(inner_boundaries):
        ax.vlines(inner_boundaries, -0.5, n - 0.5, colors='white',
                  linewidth=linewidth, zorder=3)
        ax.hlines(inner_boundaries, -0.5, n - 0.5, colors='white',
                  linewidth=linewidth, zorder=3)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.add_patch(Rectangle(
        (-0.5, -0.5), n, n, fill=False, edgecolor='black',
        linewidth=max(0.65, linewidth * 1.8), zorder=5,
        clip_on=False,
    ))


def plot_confusion_per_domain(cm: np.ndarray, class_names: List[str],
                              title: str, out_path: str,
                              show_counts: bool = False,
                              dataset: str = 'PU') -> List[str]:
    """单域混淆矩阵热力图（行归一化为 recall %）。"""
    _set_publication_font(font_size=6.0)
    pct = _normalize_cm(cm)
    pct_display = pct * 100.0
    n = pct.shape[0]
    short = [str(i) for i in range(n)]

    fig, ax = plt.subplots(
        figsize=(CM_PANEL_WIDTH_IN, CM_PANEL_WIDTH_IN * 1.02),
        constrained_layout=True,
    )

    im = _draw_cm_cells(ax, pct, dataset)
    cb = fig.colorbar(im, ax=ax, shrink=0.9, fraction=0.045, pad=0.012)
    cb.set_label('Recall (%)', rotation=270, labelpad=8, fontsize=6.5)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cb.set_ticklabels(['0', '25', '50', '75', '100'])
    cb.ax.tick_params(labelsize=6, width=0.5, length=2)

    # 单格标注：百分比，保留小数点后 2 位。
    fs = max(4.8, min(6.4, 8.3 - 0.25 * n))
    fs = max(4.3, min(5.6, 7.6 - 0.24 * n))
    thresh = 0.55
    for i in range(n):
        for j in range(n):
            p = pct[i, j]
            color = 'white' if p > thresh else '#222'
            txt = f'{pct_display[i, j]:.2f}'
            cell_fs = max(3.8, fs - 0.55) if len(txt) >= 6 else fs
            ax.text(j, i, txt, ha='center', va='center',
                    color=color, fontsize=cell_fs, linespacing=0.95,
                    clip_on=True, zorder=4)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short, rotation=0, ha='center', fontsize=7)
    ax.set_yticklabels(short, fontsize=7)
    ax.set_xlabel('Predicted', fontsize=8)
    ax.set_ylabel('True',      fontsize=8)

    # 网格隔开格子
    ax.tick_params(axis='both', which='major', width=0.55, length=2.2, pad=1.2)
    _draw_cell_boundaries(ax, n, linewidth=0.45)
    ax.tick_params(which='minor', length=0)

    paths = _save_figure_formats(fig, out_path, dpi=600)
    plt.close(fig)
    return paths


def plot_confusion_grid(cm_by_snr: Dict[Any, np.ndarray], class_names: List[str],
                        out_path: str, suptitle: str = 'Confusion matrices (recall)',
                        dataset: str = 'PU') -> List[str]:
    """所有 SNR 域的 CM 放一张图（2 列网格）。"""
    _set_publication_font(font_size=6.0)
    snrs = list(cm_by_snr.keys())
    k = len(snrs)
    cols = 2 if k > 1 else 1
    rows = (k + cols - 1) // cols
    n = next(iter(cm_by_snr.values())).shape[0]
    short = [str(i) for i in range(n)]

    fig, axes = plt.subplots(rows, cols, figsize=(CM_PANEL_WIDTH_IN * cols,
                                                  CM_PANEL_WIDTH_IN * rows * 1.02),
                             squeeze=False, constrained_layout=True)
    axes_flat = axes.reshape(-1)

    im = None
    fs = max(4.5, min(5.9, 7.4 - 0.22 * n))
    for ax, snr in zip(axes_flat, snrs):
        pct = _normalize_cm(cm_by_snr[snr])
        im = _draw_cm_cells(ax, pct, dataset)
        _annotate_cells(ax, pct * 100.0, fmt='.2f', fontsize=fs, thresh=55.0)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(short, rotation=0, ha='center', fontsize=6.3)
        ax.set_yticklabels(short, fontsize=6.3)
        ax.set_xlabel('Pred', fontsize=7)
        ax.set_ylabel('True', fontsize=7)
        ax.tick_params(axis='both', which='major', width=0.55, length=2.2, pad=1.2)
        _draw_cell_boundaries(ax, n, linewidth=0.35)
        ax.tick_params(which='minor', length=0)

    # 隐藏多余子图
    for ax in axes_flat[k:]:
        ax.set_visible(False)

    # 共用 colorbar
    if im is not None:
        visible_axes = [ax for ax in axes_flat[:k] if ax.get_visible()]
        cb = fig.colorbar(im, ax=visible_axes, shrink=0.9, fraction=0.045, pad=0.012)
        cb.set_label('Recall (%)', rotation=270, labelpad=8, fontsize=6.5)
        cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        cb.set_ticklabels(['0', '25', '50', '75', '100'])
        cb.ax.tick_params(labelsize=6, width=0.5, length=2)

    paths = _save_figure_formats(fig, out_path, dpi=600)
    plt.close(fig)
    return paths


def render_confusion_plots(run_dir: str,
                           cm_by_snr: Dict[Any, np.ndarray],
                           class_names: List[str],
                           title_prefix: str = '',
                           dataset: str = 'PU') -> List[str]:
    """落盘所有 CM 图：逐域 PNG + 汇总 grid PNG。返回写入的路径列表。"""
    plot_dir = os.path.join(run_dir, 'confusion_matrices')
    os.makedirs(plot_dir, exist_ok=True)
    out_paths = []
    for snr, cm in cm_by_snr.items():
        acc = np.trace(cm) / max(cm.sum(), 1)
        title = f'{title_prefix}{format_snr(snr)}  (acc={100.0 * acc:.2f}%)'.strip()
        path = os.path.join(plot_dir, f'cm_{format_snr(snr).replace("+", "p").replace("-", "n")}.png')
        out_paths.extend(plot_confusion_per_domain(cm, class_names, title, path,
                                                   dataset=dataset))
    grid_path = os.path.join(plot_dir, 'grid.png')
    out_paths.extend(plot_confusion_grid(cm_by_snr, class_names, grid_path,
                                         suptitle=f'{title_prefix}Confusion matrices (recall)'.strip(),
                                         dataset=dataset))
    return out_paths


def run_deploy(cfg: Config,
               excel_path: str = None,
               excel_sheet: str = 'deploy_metrics',
               run_dir: str = None,
               ckpt_path: str = None):
    """读取 cfg.ckpt_path() 的 ckpt，在 6 个域 test 上批量评测并落盘。

    可由 train.py 在训练结束后直接调用：from deploy import run_deploy; run_deploy(cfg)
    """
    device = torch.device('cuda' if (torch.cuda.is_available() and not cfg.no_cuda) else 'cpu')
    if run_dir is None:
        run_dir = os.path.dirname(os.path.abspath(ckpt_path)) if ckpt_path else cfg.run_dir()
    if ckpt_path is None:
        ckpt_path = os.path.join(run_dir, 'checkpoint.pth')
    run_dir = os.path.normpath(run_dir)
    ckpt_path = os.path.normpath(ckpt_path)

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"checkpoint 不存在: {ckpt_path}\n"
                                f"请先 python train.py --mode {cfg.mode} "
                                f"--dataset {cfg.dataset} --seed {cfg.seed}")
    os.makedirs(run_dir, exist_ok=True)

    # ── 加载 ckpt + 重建模型 ─────────────────────────────
    # weights_only 是 PyTorch 1.13+ 引入的参数，老版本不识别 → TypeError 回退到默认加载。
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    preset = ckpt.get('preset', get_preset(cfg.mode))

    num_classes = ckpt.get('num_classes', get_num_classes(cfg))

    model = QCDANNModel(
        backbone_name=preset['backbone'],
        num_classes=num_classes,
        use_dann=preset['use_dann'],
        num_domains=len(SNR_LIST),
        feat_dim=100,
        use_cdan=bool(preset.get('use_cdan', False)),
    ).to(device)
    model.load_state_dict(ckpt['model_state'])

    # ── 输出文件 ─────────────────────────────────────────
    out_path = os.path.join(run_dir, 'deploy_results.txt')
    out_fp = open(out_path, 'w', encoding='utf-8', buffering=1)

    def log(msg):
        print(msg)
        out_fp.write(msg + '\n')

    log(f"[DEPLOY] mode={cfg.mode}  dataset={cfg.dataset}  seed={cfg.seed}")
    log(f"[PRESET] {preset['description']}")
    ckpt_val_acc = ckpt.get('val_acc', float('nan'))
    if isinstance(ckpt_val_acc, (int, float)) and abs(ckpt_val_acc) > 1.0:
        ckpt_val_acc = ckpt_val_acc / 100.0
    log(f"[CKPT]   {ckpt_path}  (best epoch={ckpt.get('epoch', '?')}, "
        f"val_acc={ckpt_val_acc:.4f})")
    log(f"[DEVICE] {device}")
    log('-' * 64)

    # ── 各域逐一评测（clean + 5 SNR）─────────────────────
    per_domain: Dict[Any, float] = {}
    cm_all: Dict[Any, np.ndarray] = {}
    metrics_by_domain: Dict[Any, Dict[str, float]] = {}

    for snr in SNR_LIST:
        d = build_domain_dataset(cfg, snr)
        loader = _to_loader(d['X_test'], d['Y_test'], batch_size=cfg.batch_size)
        acc, gts, preds = _evaluate(model, loader, device)
        # Fill per-domain metrics after the confusion matrix is built.
        # 简单混淆矩阵
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        for g, p in zip(gts, preds):
            cm[g, p] += 1
        cm_all[snr] = cm
        metrics = metrics_from_confusion_matrix(cm)
        metrics['acc'] = acc
        metrics_by_domain[snr] = metrics
        per_domain[snr] = metrics['acc']
        log(f"{format_snr(snr):>6s}  n_test={len(gts):5d}  "
            f"Acc={metrics['acc']:8.4f}  F1={metrics['f1']:8.4f}  "
            f"FDR={metrics['fdr']:8.4f}  FPR={metrics['fpr']:8.4f}")

    log('-' * 64)
    avg_metrics = average_domain_metrics(metrics_by_domain)
    avg = avg_metrics['acc']
    log(f"Average over {len(SNR_LIST)} domains: "
        f"Acc={avg_metrics['acc']:.4f}  F1={avg_metrics['f1']:.4f}  "
        f"FDR={avg_metrics['fdr']:.4f}  FPR={avg_metrics['fpr']:.4f}")
    best_snr = max(per_domain, key=per_domain.get)
    worst_snr = min(per_domain, key=per_domain.get)
    log(f"Best  domain: {format_snr(best_snr)}  ({per_domain[best_snr]:.4f})")
    log(f"Worst domain: {format_snr(worst_snr)}  ({per_domain[worst_snr]:.4f})")

    ckpt_meta = {
        'ckpt_epoch': ckpt.get('epoch'),
        'ckpt_val_acc': ckpt_val_acc,
    }

    # 写一份 JSON 摘要，方便外部读
    with open(os.path.join(run_dir, 'deploy_results.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'mode': cfg.mode,
            'dataset': cfg.dataset,
            'seed': cfg.seed,
            **ckpt_meta,
            'per_domain_acc': {format_snr(k): v for k, v in per_domain.items()},
            'per_domain_metrics': {format_snr(k): v for k, v in metrics_by_domain.items()},
            'avg_acc': avg,
            'avg_metrics': avg_metrics,
        }, f, ensure_ascii=False, indent=2)

    metrics_json_path = save_metrics_json(run_dir, cfg, metrics_by_domain,
                                          extra_meta=ckpt_meta)
    log(f"[METRICS] Acc/F1/FDR/FPR JSON 已保存: {metrics_json_path}")

    if excel_path:
        excel_out = append_metrics_to_excel(
            cfg,
            metrics_by_domain,
            excel_path,
            sheet_name=excel_sheet,
            run_dir=run_dir,
            extra_meta=ckpt_meta,
        )
        log(f"[EXCEL] Acc/F1/FDR/FPR 已追加到: {excel_out} (sheet={excel_sheet})")

    # 混淆矩阵单独存（npz）：键名 'domain_+6dB' / 'domain_-3dB' 等
    np.savez(os.path.join(run_dir, 'confusion_matrices.npz'),
             **{f'domain_{format_snr(snr)}': cm for snr, cm in cm_all.items()})

    # 混淆矩阵可视化（逐域 + 汇总 grid）
    try:
        class_names = get_class_names(cfg)
        title_prefix = f'{cfg.dataset} / {cfg.mode}  '
        paths = render_confusion_plots(run_dir, cm_all, class_names,
                                       title_prefix=title_prefix,
                                       dataset=cfg.dataset)
        log(f"[PLOT] 混淆矩阵已保存到 {os.path.join(run_dir, 'confusion_matrices')}")
        for p in paths:
            log(f"        - {os.path.basename(p)}")
    except Exception as e:
        log(f"[WARN] 混淆矩阵绘制失败: {e}")

    out_fp.close()
    print(f"\n[OK] 结果已保存到 {run_dir}")
    return metrics_by_domain


# ──────────────────────────────────────────────────────────
# 独立绘图入口：从已存 confusion_matrices.npz 直接重绘
# ──────────────────────────────────────────────────────────

def replot_from_npz(cfg: Config, run_dir: str = None) -> List[str]:
    """无需 ckpt / 不跑推理，纯粹读 confusion_matrices.npz 重画图。

    用途：训练完跑过一次 deploy 后想换配色 / 重画 / 改样式 → 直接重画，不再跑模型。
    """
    run_dir = os.path.normpath(run_dir or cfg.run_dir())
    npz_path = os.path.join(run_dir, 'confusion_matrices.npz')
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(
            f"找不到 {npz_path}\n请先跑一次 python deploy.py 把 npz 生成出来")

    npz = np.load(npz_path)
    cm_by_snr: Dict[Any, np.ndarray] = {}
    # 还原 key：'domain_+6dB' → 6 ；'domain_-3dB' → -3
    for key in npz.files:
        tag = key.replace('domain_', '')
        if tag == 'clean':
            snr = None
        else:
            snr = int(tag.replace('dB', ''))
        cm_by_snr[snr] = np.asarray(npz[key], dtype=np.int64)
    npz.close()

    # 按 SNR_LIST 顺序排（npz 文件顺序不保证）
    ordered = {s: cm_by_snr[s] for s in SNR_LIST if s in cm_by_snr}
    # 兜底：把不在 SNR_LIST 里的也加上
    for s, cm in cm_by_snr.items():
        if s not in ordered:
            ordered[s] = cm

    class_names = get_class_names(cfg)
    paths = render_confusion_plots(run_dir, ordered, class_names,
                                   title_prefix=f'{cfg.dataset} / {cfg.mode}  ',
                                   dataset=cfg.dataset)
    print(f"[PLOT-ONLY] 已从 {os.path.basename(npz_path)} 重绘混淆矩阵：")
    for p in paths:
        print(f"  - {p}")
    return paths


def main():
    args = parse_args()
    cfg = _cfg_from_args(args)
    if args.aggregate_existing:
        if not args.excel:
            raise ValueError('--aggregate_existing requires --excel')
        results_root = args.results_root or cfg.results_root
        append_mode_aggregate_to_excel(
            results_root=results_root,
            dataset=cfg.dataset,
            mode=cfg.mode,
            excel_path=args.excel,
            sheet_name=args.excel_sheet,
            include_flat_rows=not args.aggregate_summary_only,
        )
        print(f"[EXCEL] appended mean+/-std rows for {cfg.dataset}/{cfg.mode} from {results_root}")
        return
    if args.plot_only:
        replot_from_npz(cfg, run_dir=args.run_dir)
    else:
        run_deploy(cfg, excel_path=args.excel, excel_sheet=args.excel_sheet,
                   run_dir=args.run_dir, ckpt_path=args.ckpt)


if __name__ == '__main__':
    main()
