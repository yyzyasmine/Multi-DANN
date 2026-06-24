
"""
训练入口：一行命令即可启动任一模式。

用法：
    python train.py --mode E1_QMDCS --dataset PU --seed 42
    python train.py --mode A1_Q --dataset CWRU --single_domain +6
    python train.py --list                # 列出所有可用模式

所有可调参数都在 config.py 中；命令行只暴露最常用的几个开关。
"""

# Windows 下 PyTorch (libiomp5md) 与 matplotlib/numpy (libomp) 的 OpenMP 双链冲突，
# 不设这个 env var 会在 plt.savefig() 时被 OMP runtime 直接 abort()。
# 必须早于任何 torch / matplotlib / numpy 的 import。
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import argparse
from dataclasses import replace

from config import Config, list_modes, ABLATION_PRESETS
from deploy_metrics import append_seed_aggregate_to_excel, build_excel_rows


def _parse_seed_list(seed_arg: str):
    seeds = []
    for part in str(seed_arg).split(','):
        part = part.strip()
        if part:
            seeds.append(int(part))
    if not seeds:
        raise ValueError('--seeds 至少需要包含一个整数，例如 --seeds 42,100,2024')
    return seeds


def _parse_mode_list(mode_arg: str):
    modes = []
    for part in str(mode_arg).split(','):
        part = part.strip()
        if part:
            if part not in ABLATION_PRESETS:
                raise ValueError(
                    f"unknown mode '{part}' in --modes/--mode. "
                    f"Run python train.py --list to see available modes."
                )
            modes.append(part)
    if not modes:
        raise ValueError('--modes 至少需要包含一个消融模式，例如 --modes A2_M,B6_MC,D5_MDCS')
    return modes


def parse_args():
    p = argparse.ArgumentParser('Muti-QCDANN 训练入口')
    p.add_argument('--mode', type=str, default='E1_QMDCS',
                   help=f'消融模式，可选: {list_modes()}')
    p.add_argument('--modes', type=str, default=None,
                   help='逗号分隔的多个消融模式，会按顺序训练，例如 A2_M,B6_MC,D5_MDCS；不传则使用 --mode')
    p.add_argument('--dataset', type=str, default='PU', choices=['PU', 'CWRU'],
                   help='PU(10类,64kHz) / CWRU(10类,12kHz)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--seeds', type=str, default=None,
                   help='逗号分隔的多个 seed，会按顺序训练，例如 42,100,2024；不传则使用 --seed')
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--results_root', type=str, default=None)
    p.add_argument('--no_cuda', action='store_true')
    p.add_argument('--patience', type=int, default=None,
                   help='早停耐心 epoch 数（默认 20，<=0 关闭早停）')
    p.add_argument('--lambda_coral', type=float, default=None,
                   help='CORAL 损失权重（默认见 config.py）')
    p.add_argument('--lambda_sparse', type=float, default=None,
                   help='Sparse 损失权重（默认见 config.py）')
    p.add_argument('--no_auto_deploy', action='store_true',
                   help='训练完跳过自动部署（默认会在 test 集上 6 域评测）')
    p.add_argument('--excel', type=str, default=None,
                   help='训练后自动部署时，把 Acc/F1/FDR/FPR 追加到这个 Excel，例如 results/deploy_metrics.xlsx')
    p.add_argument('--excel_sheet', type=str, default='deploy_metrics',
                   help='Excel sheet 名称（默认 deploy_metrics）')
    p.add_argument('--progressive_training', action='store_true',
                   help='渐进训练：按 SNR 从高到低逐 stage 加入目标域，stage 0 仅源域')
    p.add_argument('--single_domain', type=str, default=None,
                   choices=['+6', '+3', '0', '-3', '-6'],
                   help='单域训练，指定唯一训练 SNR；不传则默认多域')
    p.add_argument('--pu_class_set', type=str, default=None,
                   choices=['4class', '10class'],
                   help='仅 PU 生效：4class（默认，每型一代表 4 分类）/ 10class（旧 10 类设定）')
    p.add_argument('--list', action='store_true', help='列出所有可用 mode 并退出')
    return p.parse_args()


def main():
    args = parse_args()

    if args.list:
        print('可用 mode:')
        for name, preset in ABLATION_PRESETS.items():
            print(f"  {name:20s} -- {preset['description']}")
        return

    modes = _parse_mode_list(args.modes) if args.modes is not None else _parse_mode_list(args.mode)
    seeds = _parse_seed_list(args.seeds) if args.seeds is not None else [args.seed]

    cfg = Config(mode=modes[0], dataset=args.dataset, seed=seeds[0])
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.lr is not None:
        cfg.lr = args.lr
    if args.results_root is not None:
        cfg.results_root = args.results_root
    if args.patience is not None:
        cfg.patience = args.patience
    if args.lambda_coral is not None:
        cfg.lambda_coral = args.lambda_coral
    if args.lambda_sparse is not None:
        cfg.lambda_sparse = args.lambda_sparse
    if args.progressive_training:
        cfg.progressive_training = True
    if args.single_domain is not None:
        snr_map = {'+6': 6, '+3': 3, '0': 0, '-3': -3, '-6': -6}
        cfg.training_mode = 'single'
        cfg.single_domain_snr = snr_map[args.single_domain]
    if args.pu_class_set is not None:
        cfg.pu_class_set = args.pu_class_set
    cfg.no_cuda = args.no_cuda

    base_cfg = cfg
    rows_by_mode = {mode: [] for mode in modes}
    total_runs = len(modes) * len(seeds)
    run_idx = 0

    from trainer import train
    from deploy import run_deploy

    for mode_idx, mode in enumerate(modes, start=1):
        for seed_idx, seed in enumerate(seeds, start=1):
            run_idx += 1
            cfg = replace(base_cfg, mode=mode, seed=seed)
            print(
                f"\n[RUN {run_idx}/{total_runs}] start training "
                f"mode={mode} seed={seed} "
                f"(mode {mode_idx}/{len(modes)}, seed {seed_idx}/{len(seeds)})"
            )
            ckpt = train(cfg)
            print(f"\n训练完成，checkpoint: {ckpt}")

            if args.no_auto_deploy:
                cmd = f"python deploy.py --mode {cfg.mode} --dataset {cfg.dataset} --seed {cfg.seed}"
                if args.excel:
                    cmd += f" --excel {args.excel} --excel_sheet {args.excel_sheet}"
                print(f"已跳过自动部署。手动执行:  {cmd}")
                continue

            print("\n[AUTO-DEPLOY] 开始在 6 个域 test 集上评测...")
            metrics_by_domain = run_deploy(cfg, excel_path=args.excel, excel_sheet=args.excel_sheet)
            if args.excel:
                rows_by_mode[mode].append(build_excel_rows(cfg, metrics_by_domain, run_dir=cfg.run_dir()))

    if args.excel:
        aggregated_modes = 0
        aggregated_seeds = 0
        for mode, rows_by_seed in rows_by_mode.items():
            if len(rows_by_seed) <= 1:
                continue
            append_seed_aggregate_to_excel(
                replace(base_cfg, mode=mode),
                rows_by_seed,
                args.excel,
                sheet_name=args.excel_sheet,
                include_flat_rows=False,
            )
            aggregated_modes += 1
            aggregated_seeds += len(rows_by_seed)
        if aggregated_modes:
            print(
                f"\n[EXCEL] 已追加 {aggregated_modes} 个 mode / "
                f"{aggregated_seeds} 个 seed run 的 mean+/-std 汇总行: {args.excel}"
            )


if __name__ == '__main__':
    main()
