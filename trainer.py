"""
通用训练循环（被 train.py 调用）

职责：
  1. 按 cfg.mode 取出 ABLATION_PRESETS 中的开关
  2. 加载 6 个域（clean + 5 SNR）的数据（每域含 train/val/test）；scaler 只用源域 train 拟合
  3. 多域有标签训练（源 + 目标都进 cls_loss），DANN/CORAL/Sparse 按开关挂上
  4. 每 epoch 在每个域的 val 上评估，落日志（含 Train/Val Loss + Acc）
  5. 保存 best ckpt + scalers + 配置快照（deploy.py 直接复用）
"""

# Windows 下 PyTorch (libiomp5md) 与 matplotlib/numpy (libomp) 的 OpenMP 双链冲突，
# 不设这个 env var 会在 plt.savefig() 时被 OMP runtime 直接 abort()。
# 必须早于任何 torch / matplotlib / numpy 的 import。
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import json
import math
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import Config, get_preset, SNR_LIST, SOURCE_SNR, SNR_DOMAINS, format_snr
from data.preprocess import build_domain_dataset, get_num_classes
from models import QCDANNModel, progressive_lambda
from losses import get_loss, get_joint_loss

FIGURE_FORMATS = ('png', 'svg', 'pdf')


def _save_figure_formats(fig, out_base: str, dpi: int = 300,
                         bbox_inches: str = 'tight') -> List[str]:
    paths = []
    for fmt in FIGURE_FORMATS:
        path = f'{out_base}.{fmt}'
        kwargs = {'bbox_inches': bbox_inches}
        if fmt == 'png':
            kwargs['dpi'] = dpi
        fig.savefig(path, **kwargs)
        paths.append(path)
    return paths


# ──────────────────────────────────────────────────────────
# 工具
# ──────────────────────────────────────────────────────────

def _set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_loader(X, Y, batch_size, shuffle, device):
    """numpy → tensor → DataLoader，X 自动补 channel 维。"""
    X = torch.from_numpy(np.asarray(X)).float().unsqueeze(1)  # (N, 1, L)
    Y = torch.from_numpy(np.asarray(Y)).long()
    ds = TensorDataset(X, Y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, drop_last=False)


def _curriculum_active_targets(epoch: int, total_epochs: int, src_snr) -> List:
    """渐进训练：按 SNR_LIST 顺序逐 stage 加入目标域，返回当前 epoch 活跃的目标 SNR 列表。

      stage 0  → 仅源域，目标域 = []
      stage k  → 目标域 = SNR_LIST 前 k 个非源域

    每 stage 大约占 total_epochs / (1+非源域数) 个 epoch（最后一 stage 兜底吃余数）。
    """
    non_source = [s for s in SNR_LIST if s != src_snr]
    n_stages = len(non_source) + 1
    epochs_per_stage = max(1, total_epochs // n_stages)
    stage = min((epoch - 1) // epochs_per_stage, n_stages - 1)
    return non_source[:stage]


def _qcnn_param_groups(model: nn.Module, base_lr: float, alpha: float, wd: float):
    """对 QCNN 的二次项参数（weight_g/b, bias_g/b）单独使用 alpha*lr。"""
    quad_keys = ('weight_g', 'weight_b', 'bias_g', 'bias_b')
    quad, linear = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in name for k in quad_keys):
            quad.append(p)
        else:
            linear.append(p)
    if not quad:
        return [{'params': linear, 'lr': base_lr, 'weight_decay': wd}]
    return [{'params': linear, 'lr': base_lr,         'weight_decay': wd},
            {'params': quad,   'lr': base_lr * alpha, 'weight_decay': wd}]


def _format_domain_label(snr) -> str:
    if isinstance(snr, str):
        return snr
    return format_snr(snr)


def _plot_curves(history: Dict[str, Any], run_dir: str, best_epoch: int):
    """绘制 train/val loss、acc 曲线（含各域 val acc 浅色辅助线）。"""
    epochs = history['epoch']
    if not epochs:
        return
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix',
        'svg.fonttype': 'none',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'font.size': 11,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 8,
        'axes.linewidth': 1.0,
    })
    fig, axes = plt.subplots(2, 1, figsize=(4.2, 3.6))

    axes[0].plot(epochs, history['train_loss'], label='train', color='tab:blue', linewidth=1.2)
    axes[0].plot(epochs, history['val_loss'],   label='val (avg)', color='tab:orange', linewidth=1.2)
    axes[0].axvline(best_epoch, color='gray', linestyle='--', alpha=0.5,
                    linewidth=0.9, label=f'best ep={best_epoch}')
    axes[0].set_xlabel('epoch'); axes[0].set_ylabel('loss')
    axes[0].grid(alpha=0.25, linewidth=0.5); axes[0].legend(loc='lower right', frameon=False)

    axes[1].plot(epochs, history['train_acc'], label='train', color='tab:blue', linewidth=1.2)
    axes[1].plot(epochs, history['val_acc'],   label='val (avg)',
                 color='tab:orange', linewidth=1.4)
    for snr, accs in history['per_domain_val_acc'].items():
        axes[1].plot(epochs, accs, label=f'val {_format_domain_label(snr)}',
                     alpha=0.35, linewidth=0.8)
    axes[1].axvline(best_epoch, color='gray', linestyle='--', alpha=0.5, linewidth=0.9)
    axes[1].set_xlabel('epoch'); axes[1].set_ylabel('accuracy (%)')
    axes[1].grid(alpha=0.25, linewidth=0.5); axes[1].legend(fontsize=7, ncol=2, frameon=False)
    for ax in axes:
        ax.tick_params(direction='out', width=0.9, length=3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.tight_layout(pad=0.45)
    out_base = os.path.join(run_dir, 'train_curves')
    paths = _save_figure_formats(fig, out_base, dpi=300)
    plt.close(fig)
    return paths


@torch.no_grad()
def _eval(model: QCDANNModel, loader: DataLoader, device, criterion) -> Tuple[float, float]:
    model.eval()
    tot, correct, loss_sum, n_loss = 0, 0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_sum += criterion(out, y).item() * x.size(0)
        n_loss += x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        tot += x.size(0)
    return loss_sum / max(n_loss, 1), 100.0 * correct / max(tot, 1)


# ──────────────────────────────────────────────────────────
# 主训练函数
# ──────────────────────────────────────────────────────────

def train(cfg: Config):
    preset = get_preset(cfg.mode)
    device = torch.device('cuda' if (torch.cuda.is_available() and not cfg.no_cuda) else 'cpu')
    _set_seed(cfg.seed)

    run_dir = cfg.run_dir()
    os.makedirs(run_dir, exist_ok=True)
    cfg.save_json(os.path.join(run_dir, 'config.json'))
    with open(os.path.join(run_dir, 'preset.json'), 'w', encoding='utf-8') as f:
        json.dump(preset, f, ensure_ascii=False, indent=2)

    log_path = os.path.join(run_dir, 'train_log.txt')
    log_fp = open(log_path, 'w', encoding='utf-8', buffering=1)

    def log(msg: str):
        print(msg)
        log_fp.write(msg + '\n')

    log(f"[CFG] mode={cfg.mode}  dataset={cfg.dataset}  device={device}")
    log(f"[PRESET] {preset['description']}")
    use_target_domains = bool(preset.get('use_target_domains',
                                         preset.get('use_multidomain', False)))
    use_cdan = bool(preset.get('use_cdan', False))
    joint_aux_cfg = preset.get('joint_aux_losses', [])
    log(f"[PRESET] backbone={preset['backbone']}  dann={preset['use_dann']}  "
        f"cdan={use_cdan}  target_domains={use_target_domains}  "
        f"multidomain_cls={preset['use_multidomain']}  "
        f"aux={preset['aux_losses']}  joint_aux={joint_aux_cfg}")

    # ── 数据：6 个域（clean + 5 SNR）─────────────────────
    num_classes = get_num_classes(cfg)
    log(f"[DATA] num_classes={num_classes}")

    # ── 训练模式：决定本次训练用单域还是多域 ─────────────
    if cfg.training_mode == 'single':
        src_snr = cfg.single_domain_snr
        log(f"[MODE] single-domain 训练，唯一训练域 = {format_snr(src_snr)}")
        if cfg.progressive_training:
            log(f"[MODE] 单域模式下 progressive_training 自动失效（无目标域可加）")
    else:
        src_snr = SOURCE_SNR
        log(f"[MODE] multi-domain 训练，源域 = {format_snr(src_snr)}")

    # 各域独立加载；归一化为 per-sample z-score（无 scaler 状态需要保存）
    log(f"[DATA] 加载源域 {format_snr(src_snr)} ...")
    src_data = build_domain_dataset(cfg, src_snr)

    domain_data: Dict[Any, Dict[str, np.ndarray]] = {src_snr: src_data}
    for snr in SNR_LIST:
        if snr == src_snr:
            continue
        log(f"[DATA] 加载目标域 {format_snr(snr)} ...")
        domain_data[snr] = build_domain_dataset(cfg, snr)

    # ── DataLoader ───────────────────────────────────────
    snr_to_did = {snr: did for did, snr in SNR_DOMAINS.items()}

    src_train_loader = _to_loader(src_data['X_train'], src_data['Y_train'],
                                  cfg.batch_size, shuffle=True, device=device)

    tgt_train_loaders: Dict[Any, DataLoader] = {}
    # 单域模式不建目标 loader；多域且 preset 允许多域时建立
    train_with_targets = cfg.training_mode == 'multi' and use_target_domains
    if train_with_targets:
        for snr in SNR_LIST:
            if snr == src_snr:
                continue
            d = domain_data[snr]
            tgt_train_loaders[snr] = _to_loader(d['X_train'], d['Y_train'],
                                                cfg.target_batch_size,
                                                shuffle=True, device=device)

    val_loaders: Dict[Any, DataLoader] = {}
    for snr in SNR_LIST:
        d = domain_data[snr]
        val_loaders[snr] = _to_loader(d['X_val'], d['Y_val'],
                                      cfg.batch_size, shuffle=False, device=device)

    # ── 模型 / 优化器 / 调度器 ───────────────────────────
    model = QCDANNModel(
        backbone_name=preset['backbone'],
        num_classes=num_classes,
        use_dann=preset['use_dann'],
        num_domains=len(SNR_LIST),
        feat_dim=100,
        use_cdan=use_cdan,
    ).to(device)

    param_groups = _qcnn_param_groups(model, cfg.lr, cfg.alpha, cfg.weight_decay)
    optimizer = torch.optim.SGD(param_groups, momentum=cfg.momentum,
                                weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    cls_criterion = nn.CrossEntropyLoss()
    dom_criterion = nn.CrossEntropyLoss()

    aux_loss_specs = [(get_loss(name), lam_attr) for name, lam_attr in preset['aux_losses']]
    joint_aux_loss_specs = [(get_joint_loss(name), lam_attr)
                            for name, lam_attr in joint_aux_cfg]

    # ── 训练循环 ─────────────────────────────────────────
    best_val_acc = -1.0
    best_epoch = -1

    # 训练历史（绘图 + 早停依据）
    history: Dict[str, Any] = {
        'epoch': [], 'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'per_domain_val_acc': {snr: [] for snr in SNR_LIST},
    }

    # 渐进训练计划（仅多域模式有意义）
    use_curriculum = cfg.progressive_training and train_with_targets
    if use_curriculum:
        non_source = [s for s in SNR_LIST if s != src_snr]
        n_stages = len(non_source) + 1
        eps = max(1, cfg.epochs // n_stages)
        log(f"[CURRICULUM] 渐进训练启用，共 {n_stages} 个 stage，每 stage ~{eps} epochs")
        log(f"[CURRICULUM]   stage 0: 仅源域 {format_snr(src_snr)}")
        for i, snr in enumerate(non_source, 1):
            log(f"[CURRICULUM]   stage {i}: 加入 {format_snr(snr)}")

    last_stage_size = -1

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        lambda_ = progressive_lambda(epoch - 1, cfg.epochs, cfg.lambda_max)

        # 当前 epoch 活跃的目标域：单域永远空；多域+渐进按 stage；多域非渐进取全部
        if not train_with_targets:
            active_targets = []
        elif use_curriculum:
            active_targets = _curriculum_active_targets(epoch, cfg.epochs, src_snr)
        else:
            active_targets = [s for s in SNR_LIST if s != src_snr]

        if len(active_targets) != last_stage_size:
            log(f"[CURRICULUM] ep {epoch}: stage={len(active_targets)} "
                f"活跃目标域=[{', '.join(format_snr(s) for s in active_targets) or '（仅源域）'}]")
            last_stage_size = len(active_targets)

        # 只对活跃目标域建迭代器，inactive 域本 epoch 不参与训练
        tgt_iters = {snr: iter(tgt_train_loaders[snr])
                     for snr in active_targets if snr in tgt_train_loaders}

        epoch_loss, epoch_correct, epoch_tot = 0.0, 0, 0

        for src_x, src_y in src_train_loader:
            src_x, src_y = src_x.to(device), src_y.to(device)
            src_did = torch.full((src_x.size(0),), snr_to_did[src_snr],
                                 dtype=torch.long, device=device)

            optimizer.zero_grad()
            src_logits, src_dom_logits, src_feat = model.forward_da(src_x, lambda_)
            loss = cls_criterion(src_logits, src_y)

            # 目标域：M 使用目标标签；D/C/S 使用目标特征做对抗/对齐。
            if train_with_targets:
                for snr, it in list(tgt_iters.items()):
                    try:
                        tx, ty = next(it)
                    except StopIteration:
                        tgt_iters[snr] = iter(tgt_train_loaders[snr])
                        tx, ty = next(tgt_iters[snr])
                    tx, ty = tx.to(device), ty.to(device)
                    t_did = torch.full((tx.size(0),), snr_to_did[snr],
                                       dtype=torch.long, device=device)
                    t_logits, t_dom_logits, t_feat = model.forward_da(tx, lambda_)

                    if preset['use_multidomain']:
                        loss = loss + cls_criterion(t_logits, ty)

                    if preset['use_dann']:
                        loss = (loss
                                + dom_criterion(src_dom_logits, src_did)
                                + dom_criterion(t_dom_logits, t_did))

                    if use_cdan:
                        src_cdan_logits = model.forward_cdan(src_feat, src_logits, lambda_)
                        t_cdan_logits = model.forward_cdan(t_feat, t_logits, lambda_)
                        loss = (loss
                                + dom_criterion(src_cdan_logits, src_did)
                                + dom_criterion(t_cdan_logits, t_did))

                    for fn, lam_attr in aux_loss_specs:
                        lam = getattr(cfg, lam_attr)
                        loss = loss + lam * fn(src_feat, t_feat)

                    for fn, lam_attr in joint_aux_loss_specs:
                        lam = getattr(cfg, lam_attr)
                        loss = loss + lam * fn(src_feat, t_feat, src_logits, t_logits)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

            epoch_loss += loss.item() * src_x.size(0)
            epoch_correct += (src_logits.argmax(1) == src_y).sum().item()
            epoch_tot += src_x.size(0)

        scheduler.step()
        train_loss = epoch_loss / max(epoch_tot, 1)
        train_acc = 100.0 * epoch_correct / max(epoch_tot, 1)

        # ── 各域 val ─────────────────────────────────────
        val_metrics: Dict[Any, Tuple[float, float]] = {}
        for snr in SNR_LIST:
            vl, va = _eval(model, val_loaders[snr], device, cls_criterion)
            val_metrics[snr] = (vl, va)

        avg_val_loss = float(np.mean([v[0] for v in val_metrics.values()]))
        avg_val_acc  = float(np.mean([v[1] for v in val_metrics.values()]))

        det = ' '.join(f"{format_snr(snr)}:{val_metrics[snr][1]:5.2f}%" for snr in SNR_LIST)
        log(f"Ep {epoch:3d}/{cfg.epochs}  λ={lambda_:.3f}  "
            f"Loss:{train_loss:.4f}  ValLoss:{avg_val_loss:.4f}  "
            f"TrAcc:{train_acc:5.2f}%  ValAcc:{avg_val_acc:5.2f}%  [{det}]")

        # 记录历史（不论是否最佳）
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc)
        for snr in SNR_LIST:
            history['per_domain_val_acc'][snr].append(val_metrics[snr][1])

        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            best_epoch = epoch
            torch.save({
                'model_state': model.state_dict(),
                'epoch': epoch,
                'val_acc': avg_val_acc,
                'preset': preset,
                'num_classes': num_classes,
                'training_mode': cfg.training_mode,
                'single_domain_snr': cfg.single_domain_snr,
            }, cfg.ckpt_path())

        # 早停：连续 patience 个 epoch 无新高则跳出
        if cfg.patience > 0 and (epoch - best_epoch) >= cfg.patience:
            log(f"[EARLY_STOP] {cfg.patience} 个 epoch 无新高 ValAcc，"
                f"ep {best_epoch} 后停在 ep {epoch}")
            break

    # 训练曲线（早停或跑满都会画一次）
    curve_paths = _plot_curves(history, run_dir, best_epoch)
    if curve_paths:
        log(f"[PLOT] training curves saved: {curve_paths}")
    # 历史 JSON 留底，方便后续做对比图
    with open(os.path.join(run_dir, 'train_history.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'epoch': history['epoch'],
            'train_loss': history['train_loss'],
            'train_acc': history['train_acc'],
            'val_loss': history['val_loss'],
            'val_acc': history['val_acc'],
            'per_domain_val_acc': {format_snr(k): v
                                   for k, v in history['per_domain_val_acc'].items()},
            'best_epoch': best_epoch,
            'best_val_acc': best_val_acc,
        }, f, ensure_ascii=False, indent=2)

    log(f"[DONE] best ValAcc={best_val_acc:.2f}% @ epoch={best_epoch}  "
        f"ckpt={cfg.ckpt_path()}")
    log_fp.close()
    return cfg.ckpt_path()
