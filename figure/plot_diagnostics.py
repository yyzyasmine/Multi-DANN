import argparse
import colorsys
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "5")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "5")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "figure_output"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import to_hex, to_rgb
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from config import Config, SNR_LIST, format_snr, get_class_names, get_preset
from data.preprocess import build_domain_dataset, get_num_classes
from deploy import replot_from_npz
from models import QCDANNModel

##聚类图
CLASS_COLORS = [
    "#4E79A7",
    "#F28E2B",
    "#59A14F",
    "#E15759",
    "#76B7B2",
    "#EDC948",
    "#B07AA1",
    "#FF9DA7",
    "#9C755F",
    "#BAB0AC",
]


def parse_args():
    parser = argparse.ArgumentParser("Plot confusion matrices or KMeans feature clusters.")
    parser.add_argument("--kind", choices=["confusion", "kmeans", "compare_kmeans", "all"], default="all")
    parser.add_argument("--mode", type=str, default="C7_MDC")
    parser.add_argument("--compare_modes", nargs=2, metavar=("MODE_A", "MODE_B"), default=None)
    parser.add_argument("--dataset", type=str, default="PU", choices=["PU", "CWRU"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--compare_run_dirs", nargs=2, metavar=("RUN_DIR_A", "RUN_DIR_B"), default=None)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--compare_ckpts", nargs=2, metavar=("CKPT_A", "CKPT_B"), default=None)
    parser.add_argument("--results_root", type=str, default="results")
    parser.add_argument("--output_root", type=str, default=str(OUTPUT_ROOT))
    parser.add_argument("--pu_class_set", choices=["4class", "10class"], default="10class")
    parser.add_argument("--snr", type=int, default=None, choices=[-6, -3, 0, 3, 6])
    parser.add_argument("--all_snrs", action="store_true")
    parser.add_argument("--max_points", type=int, default=2500)
    parser.add_argument("--embed", choices=["tsne", "pca"], default="tsne")
    parser.add_argument("--perplexity", type=float, default=12.0)
    parser.add_argument("--spread", choices=["normal", "wide"], default="wide")
    parser.add_argument("--compare_cluster_separation", type=float, default=None)
    parser.add_argument("--compare_cluster_compact", type=float, default=None)
    parser.add_argument("--compare_cluster_repel", type=float, default=None)
    parser.add_argument("--no_cuda", action="store_true")
    return parser.parse_args()


def cfg_from_args(args):
    cfg_data = None
    if args.run_dir:
        cfg_path = os.path.join(args.run_dir, "config.json")
        if os.path.isfile(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg_data = json.load(f)
    if cfg_data:
        fields = Config.__dataclass_fields__
        cfg = Config(**{k: v for k, v in cfg_data.items() if k in fields})
    else:
        cfg = Config(mode=args.mode, dataset=args.dataset, seed=args.seed)
    cfg.results_root = args.results_root
    cfg.no_cuda = args.no_cuda
    if cfg.dataset == "PU":
        cfg.pu_class_set = args.pu_class_set
    return cfg


def resolve_paths(cfg, args):
    run_dir = os.path.normpath(args.run_dir or cfg.run_dir())
    ckpt_path = os.path.normpath(args.ckpt or os.path.join(run_dir, "checkpoint.pth"))
    return run_dir, ckpt_path


def run_dir_for_mode(args, mode):
    return os.path.normpath(os.path.join(args.results_root, f"exp_ablation_{args.dataset}_{mode}_seed{args.seed}"))


def cfg_for_mode(args, mode, run_dir=None):
    cfg_data = None
    if run_dir:
        cfg_path = os.path.join(run_dir, "config.json")
        if os.path.isfile(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg_data = json.load(f)
    if cfg_data:
        fields = Config.__dataclass_fields__
        cfg = Config(**{k: v for k, v in cfg_data.items() if k in fields})
    else:
        cfg = Config(mode=mode, dataset=args.dataset, seed=args.seed)
    cfg.results_root = args.results_root
    cfg.no_cuda = args.no_cuda
    if cfg.dataset == "PU":
        cfg.pu_class_set = args.pu_class_set
    return cfg


def to_loader(x, y, batch_size):
    x = torch.from_numpy(np.asarray(x)).float().unsqueeze(1)
    y = torch.from_numpy(np.asarray(y)).long()
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


def load_model(cfg, ckpt_path, device):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    preset = ckpt.get("preset", get_preset(cfg.mode))
    num_classes = ckpt.get("num_classes", get_num_classes(cfg))
    model = QCDANNModel(
        backbone_name=preset["backbone"],
        num_classes=num_classes,
        use_dann=preset["use_dann"],
        num_domains=len(SNR_LIST),
        feat_dim=100,
        use_cdan=bool(preset.get("use_cdan", False)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def extract_features(model, cfg, snrs, device):
    features, labels, domains = [], [], []
    for snr in snrs:
        data = build_domain_dataset(cfg, snr)
        loader = to_loader(data["X_test"], data["Y_test"], cfg.batch_size)
        for x, y in loader:
            x = x.to(device)
            feat = model.get_features(x).detach().cpu().numpy()
            features.append(feat)
            labels.append(y.numpy())
            domains.extend([format_snr(snr)] * len(y))
    return np.vstack(features), np.concatenate(labels), np.asarray(domains)


def subsample(features, labels, domains, max_points, seed):
    if max_points <= 0 or len(labels) <= max_points:
        return features, labels, domains
    rng = np.random.default_rng(seed)
    chosen = []
    classes = np.unique(labels)
    per_class = max(1, max_points // len(classes))
    for cls in classes:
        idx = np.where(labels == cls)[0]
        take = min(len(idx), per_class)
        chosen.extend(rng.choice(idx, size=take, replace=False).tolist())
    if len(chosen) < max_points:
        rest = np.setdiff1d(np.arange(len(labels)), np.asarray(chosen), assume_unique=False)
        extra_n = min(len(rest), max_points - len(chosen))
        if extra_n > 0:
            chosen.extend(rng.choice(rest, size=extra_n, replace=False).tolist())
    chosen = np.asarray(chosen)
    return features[chosen], labels[chosen], domains[chosen]


def sparse_legend_location(xy, width_frac=0.42, height_frac=0.34):
    x_min, x_max = xy[:, 0].min(), xy[:, 0].max()
    y_min, y_max = xy[:, 1].min(), xy[:, 1].max()
    if x_max == x_min or y_max == y_min:
        return "best", 0

    x_norm = (xy[:, 0] - x_min) / (x_max - x_min)
    y_norm = (xy[:, 1] - y_min) / (y_max - y_min)
    candidates = {
        "upper left": (0.0, width_frac, 1.0 - height_frac, 1.0),
        "upper right": (1.0 - width_frac, 1.0, 1.0 - height_frac, 1.0),
        "lower left": (0.0, width_frac, 0.0, height_frac),
        "lower right": (1.0 - width_frac, 1.0, 0.0, height_frac),
    }
    best_loc, best_count = None, None
    for loc, (x0, x1, y0, y1) in candidates.items():
        count = int(((x_norm >= x0) & (x_norm <= x1) & (y_norm >= y0) & (y_norm <= y1)).sum())
        if best_count is None or count < best_count:
            best_loc, best_count = loc, count
    return best_loc, best_count


def style_legend(legend, alpha=0.92):
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#7A7772")
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_alpha(alpha)


def adjust_color_hls(color, lightness_factor=1.0, saturation_factor=1.0):
    r, g, b = to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = min(1.0, max(0.0, l * lightness_factor))
    s = min(1.0, max(0.0, s * saturation_factor))
    return to_hex(colorsys.hls_to_rgb(h, l, s))


def adjusted_class_colors(panel_index):
    if panel_index == 0:
        return [adjust_color_hls(color, lightness_factor=0.90, saturation_factor=1.10) for color in CLASS_COLORS]
    return [adjust_color_hls(color, lightness_factor=1.10, saturation_factor=0.95) for color in CLASS_COLORS]


def separate_display_clusters(xy, labels, separation=0.0, compact=1.0, repel=0.0):
    if compact <= 0:
        raise ValueError("--compare_cluster_compact must be greater than 0.")
    if separation <= -1:
        raise ValueError("--compare_cluster_separation must be greater than -1.")
    if repel < 0:
        raise ValueError("--compare_cluster_repel must be non-negative.")
    if abs(compact - 1.0) < 1e-12 and separation <= 0 and repel <= 0:
        return xy

    labels = np.asarray(labels)
    base = np.asarray(xy, dtype=np.float64)
    adjusted = base.copy()
    classes = np.unique(labels)
    global_center = base.mean(axis=0)
    original_centers = []
    target_centers = []
    for cls in classes:
        idx = labels == cls
        if not np.any(idx):
            continue
        center = base[idx].mean(axis=0)
        original_centers.append(center)
        target_centers.append(center + separation * (center - global_center))

    original_centers = np.vstack(original_centers)
    target_centers = np.vstack(target_centers)
    if repel > 0 and len(target_centers) > 1:
        extent = np.linalg.norm(np.ptp(target_centers, axis=0))
        min_dist = extent * repel / np.sqrt(len(target_centers))
        for _ in range(80):
            shifts = np.zeros_like(target_centers)
            max_gap = 0.0
            for i in range(len(target_centers)):
                for j in range(i + 1, len(target_centers)):
                    delta = target_centers[i] - target_centers[j]
                    dist = float(np.linalg.norm(delta))
                    if dist >= min_dist:
                        continue
                    if dist < 1e-9:
                        angle = 2.0 * np.pi * (i + j + 1) / max(len(target_centers), 1)
                        unit = np.array([np.cos(angle), np.sin(angle)])
                    else:
                        unit = delta / dist
                    gap = min_dist - dist
                    shifts[i] += 0.5 * gap * unit
                    shifts[j] -= 0.5 * gap * unit
                    max_gap = max(max_gap, gap)
            target_centers += 0.35 * shifts
            if max_gap < max(min_dist * 0.01, 1e-6):
                break

    for center_index, cls in enumerate(classes):
        idx = labels == cls
        adjusted[idx] = target_centers[center_index] + compact * (base[idx] - original_centers[center_index])
    return adjusted


def resolve_compare_display_params(dataset, separation=None, compact=None, repel=None):
    defaults = {
        "PU": {
            "separation": 0.95,
            "compact": 0.70,
            "repel": 0.55,
        },
        "CWRU": {
            "separation": 0.0,
            "compact": 1.0,
            "repel": 0.0,
        },
    }
    params = defaults.get(dataset, defaults["PU"])
    return {
        "separation": params["separation"] if separation is None else float(separation),
        "compact": params["compact"] if compact is None else float(compact),
        "repel": params["repel"] if repel is None else float(repel),
    }


def project_features(features, seed, method="tsne", perplexity=12.0, spread="wide"):
    scaled = StandardScaler().fit_transform(features)
    if method == "pca":
        projector = PCA(n_components=2, random_state=seed, whiten=True)
        xy = projector.fit_transform(scaled)
        meta = {
            "projection": "PCA fitted on standardized features",
            "explained_variance_ratio": [float(v) for v in projector.explained_variance_ratio_],
        }
        return xy, projector, meta

    safe_perplexity = min(float(perplexity), max(5.0, (len(features) - 1) / 3.0))
    early_exaggeration = 24.0 if spread == "wide" else 12.0
    n_iter = 1800 if spread == "wide" else 1200
    projector = TSNE(
        n_components=2,
        perplexity=safe_perplexity,
        early_exaggeration=early_exaggeration,
        learning_rate="auto",
        n_iter=n_iter,
        init="pca",
        random_state=seed,
        metric="euclidean",
    )
    xy = projector.fit_transform(scaled)
    meta = {
        "projection": "t-SNE fitted on standardized features",
        "perplexity": float(safe_perplexity),
        "early_exaggeration": float(early_exaggeration),
        "n_iter": int(n_iter),
        "spread": spread,
    }
    return xy, projector, meta


def plot_kmeans(features, labels, domains, cfg, run_dir, snr_tag, seed, embed="tsne", perplexity=12.0, spread="wide"):
    class_names = get_class_names(cfg)
    num_classes = len(class_names)
    if num_classes != 10:
        raise ValueError(
            f"KMeans plot is requested as 10class, but got {num_classes} classes. "
            "For PU, use --pu_class_set 10class."
        )

    xy, _, projection_meta = project_features(features, seed, method=embed, perplexity=perplexity, spread=spread)
    kmeans = KMeans(n_clusters=num_classes, n_init=20, random_state=seed)
    clusters = kmeans.fit_predict(features)
    ari = adjusted_rand_score(labels, clusters)
    nmi = normalized_mutual_info_score(labels, clusters)
    embedded_centers = np.vstack([xy[clusters == cls].mean(axis=0) for cls in range(num_classes)])

    out_dir = os.path.join(run_dir, "diagnostic_plots")
    os.makedirs(out_dir, exist_ok=True)
    out_base = os.path.join(out_dir, f"{cfg.dataset}_kmeans_{snr_tag}")

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 12,
            "axes.linewidth": 1.4,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    colors = ListedColormap(CLASS_COLORS[:num_classes])
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    for cls in range(num_classes):
        idx = labels == cls
        ax.scatter(
            xy[idx, 0],
            xy[idx, 1],
            s=18,
            alpha=0.78,
            color=colors(cls),
            edgecolors="white",
            linewidths=0.22,
            label=str(cls),
        )
    ax.scatter(
        embedded_centers[:, 0],
        embedded_centers[:, 1],
        s=120,
        marker="X",
        color="black",
        edgecolors="white",
        linewidths=0.8,
        label="Cluster centroids",
        zorder=5,
    )
    axis_name = "PC" if embed == "pca" else "t-SNE"
    ax.set_xlabel(f"{axis_name} 1")
    ax.set_ylabel(f"{axis_name} 2")
    ax.set_title(f"{cfg.dataset} / {cfg.mode} KMeans ({snr_tag})\nARI={ari:.3f}, NMI={nmi:.3f}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(color="#E5E2DC", linewidth=0.8, alpha=0.55)
    legend_loc, _ = sparse_legend_location(xy, width_frac=0.42, height_frac=0.42)
    leg = ax.legend(
        title="True class",
        ncol=2,
        loc=legend_loc,
        frameon=True,
        fancybox=False,
        fontsize=9,
        title_fontsize=10,
        markerscale=1.3,
    )
    style_legend(leg)
    fig.tight_layout()
    for ext in ("png", "svg", "pdf"):
        fig.savefig(f"{out_base}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    meta = {
        "dataset": cfg.dataset,
        "mode": cfg.mode,
        "snr": snr_tag,
        "n_points": int(len(labels)),
        "n_classes": int(num_classes),
        "ari": float(ari),
        "nmi": float(nmi),
        **projection_meta,
    }
    with open(f"{out_base}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return [f"{out_base}.{ext}" for ext in ("png", "svg", "pdf")] + [f"{out_base}.json"]


def plot_compare_kmeans(
    model_payloads,
    cfg,
    out_dir,
    snr_tag,
    seed,
    embed="tsne",
    perplexity=12.0,
    spread="wide",
    compare_cluster_separation=None,
    compare_cluster_compact=None,
    compare_cluster_repel=None,
):
    class_names = get_class_names(cfg)
    num_classes = len(class_names)
    if num_classes != 10:
        raise ValueError(
            f"KMeans compare plot is requested as 10class, but got {num_classes} classes. "
            "For PU, use --pu_class_set 10class."
        )

    display_params = resolve_compare_display_params(
        cfg.dataset,
        separation=compare_cluster_separation,
        compact=compare_cluster_compact,
        repel=compare_cluster_repel,
    )
    compare_cluster_separation = display_params["separation"]
    compare_cluster_compact = display_params["compact"]
    compare_cluster_repel = display_params["repel"]

    all_features = np.vstack([payload["features"] for payload in model_payloads])
    all_xy, _, projection_meta = project_features(
        all_features,
        seed,
        method=embed,
        perplexity=perplexity,
        spread=spread,
    )

    offset = 0
    for payload in model_payloads:
        n = len(payload["labels"])
        payload["xy"] = separate_display_clusters(
            all_xy[offset : offset + n],
            payload["labels"],
            separation=compare_cluster_separation,
            compact=compare_cluster_compact,
            repel=compare_cluster_repel,
        )
        payload["clusters"] = KMeans(n_clusters=num_classes, n_init=20, random_state=seed).fit_predict(
            payload["features"]
        )
        payload["ari"] = adjusted_rand_score(payload["labels"], payload["clusters"])
        payload["nmi"] = normalized_mutual_info_score(payload["labels"], payload["clusters"])
        offset += n

    display_xy = np.vstack([payload["xy"] for payload in model_payloads])
    x_min, x_max = display_xy[:, 0].min(), display_xy[:, 0].max()
    y_min, y_max = display_xy[:, 1].min(), display_xy[:, 1].max()
    x_pad = max((x_max - x_min) * 0.06, 0.5)
    x_right_pad = max((x_max - x_min) * 0.16, x_pad)
    y_pad = max((y_max - y_min) * 0.06, 0.5)

    os.makedirs(out_dir, exist_ok=True)
    stem = f"{cfg.dataset}_kmeans_compare_{model_payloads[0]['mode']}_vs_{model_payloads[1]['mode']}_{snr_tag}_seed{seed}"
    out_base = os.path.join(out_dir, stem)

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 17,
            "axes.linewidth": 1.4,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.2), sharex=True, sharey=True)
    for panel_index, (ax, payload) in enumerate(zip(axes, model_payloads)):
        xy = payload["xy"]
        labels = payload["labels"]
        colors = ListedColormap(adjusted_class_colors(panel_index)[:num_classes])
        for cls in range(num_classes):
            idx = labels == cls
            ax.scatter(
                xy[idx, 0],
                xy[idx, 1],
                s=17,
                alpha=0.78,
                color=colors(cls),
                edgecolors="white",
                linewidths=0.22,
                label=str(cls),
            )
        ax.set_xlim(x_min - x_pad, x_max + x_right_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        axis_name = "PC" if embed == "pca" else "t-SNE"
        ax.set_xlabel(f"{axis_name} 1", fontsize=20)
        ax.set_ylabel(f"{axis_name} 2", fontsize=20)
        ax.tick_params(axis="both", labelsize=17, width=1.2, length=4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(color="#E5E2DC", linewidth=0.8, alpha=0.55)

        leg = ax.legend(
            title="True class",
            ncol=5,
            loc="lower left",
            bbox_to_anchor=(0.0, 0.012),
            bbox_transform=ax.get_xaxis_transform(),
            frameon=True,
            fancybox=False,
            fontsize=13,
            title_fontsize=15,
            markerscale=1.28,
            borderpad=0.06,
            labelspacing=0.06,
            handletextpad=0.04,
            columnspacing=0.06,
        )
        style_legend(leg, alpha=0.36)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.13, wspace=0.20)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(f"{out_base}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    display_transform_enabled = (
        abs(compare_cluster_separation) > 1e-12
        or abs(compare_cluster_compact - 1.0) > 1e-12
        or abs(compare_cluster_repel) > 1e-12
    )

    meta = {
        "dataset": cfg.dataset,
        "snr": snr_tag,
        "seed": int(seed),
        "n_classes": int(num_classes),
        **projection_meta,
        "joint_projection": "Projection fitted jointly on concatenated features from both models",
        "models": [
            {
                "mode": payload["mode"],
                "run_dir": payload["run_dir"],
                "n_points": int(len(payload["labels"])),
                "ari": float(payload["ari"]),
                "nmi": float(payload["nmi"]),
            }
            for payload in model_payloads
        ],
    }
    if display_transform_enabled:
        meta.update(
            {
                "display_transform": "class-center compacting and radial separation applied after projection",
                "compare_cluster_separation": float(compare_cluster_separation),
                "compare_cluster_compact": float(compare_cluster_compact),
                "compare_cluster_repel": float(compare_cluster_repel),
            }
        )
    with open(f"{out_base}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return [f"{out_base}.{ext}" for ext in ("png", "svg", "pdf")] + [f"{out_base}.json"]


def run_compare_kmeans(args):
    if not args.compare_modes:
        raise ValueError("--kind compare_kmeans requires --compare_modes MODE_A MODE_B")
    run_dirs = args.compare_run_dirs or [run_dir_for_mode(args, mode) for mode in args.compare_modes]
    ckpts = args.compare_ckpts or [os.path.join(run_dir, "checkpoint.pth") for run_dir in run_dirs]
    snrs = SNR_LIST if args.all_snrs else [args.snr if args.snr is not None else -6]
    snr_tag = "all_snrs" if args.all_snrs else format_snr(snrs[0]).replace("+", "p").replace("-", "n")
    device = torch.device("cuda" if (torch.cuda.is_available() and not args.no_cuda) else "cpu")

    payloads = []
    for mode, run_dir, ckpt_path in zip(args.compare_modes, run_dirs, ckpts):
        cfg = cfg_for_mode(args, mode, run_dir=run_dir)
        print(f"[COMPARE-KMEANS] Loading {mode}: {ckpt_path}")
        model = load_model(cfg, ckpt_path, device)
        features, labels, domains = extract_features(model, cfg, snrs, device)
        features, labels, domains = subsample(features, labels, domains, args.max_points, cfg.seed)
        payloads.append(
            {
                "mode": mode,
                "run_dir": run_dir,
                "features": features,
                "labels": labels,
                "domains": domains,
            }
        )

    out_dir = os.path.join(args.output_root, "diagnostic_compare_plots")
    base_cfg = cfg_for_mode(args, args.compare_modes[0], run_dir=run_dirs[0])
    paths = plot_compare_kmeans(
        payloads,
        base_cfg,
        out_dir,
        snr_tag,
        args.seed,
        embed=args.embed,
        perplexity=args.perplexity,
        spread=args.spread,
        compare_cluster_separation=args.compare_cluster_separation,
        compare_cluster_compact=args.compare_cluster_compact,
        compare_cluster_repel=args.compare_cluster_repel,
    )
    print("[COMPARE-KMEANS] Saved:")
    for path in paths:
        print(f"  - {path}")


def main():
    args = parse_args()
    if args.kind == "compare_kmeans":
        run_compare_kmeans(args)
        return

    cfg = cfg_from_args(args)
    run_dir, ckpt_path = resolve_paths(cfg, args)

    if args.kind in ("confusion", "all"):
        print("[CONFUSION] Replot from confusion_matrices.npz")
        replot_from_npz(cfg, run_dir=run_dir)

    if args.kind in ("kmeans", "all"):
        snrs = SNR_LIST if args.all_snrs else [args.snr if args.snr is not None else -6]
        snr_tag = "all_snrs" if args.all_snrs else format_snr(snrs[0]).replace("+", "p").replace("-", "n")
        device = torch.device("cuda" if (torch.cuda.is_available() and not cfg.no_cuda) else "cpu")
        print(f"[KMEANS] Loading checkpoint: {ckpt_path}")
        model = load_model(cfg, ckpt_path, device)
        features, labels, domains = extract_features(model, cfg, snrs, device)
        features, labels, domains = subsample(features, labels, domains, args.max_points, cfg.seed)
        paths = plot_kmeans(
            features,
            labels,
            domains,
            cfg,
            run_dir,
            snr_tag,
            cfg.seed,
            embed=args.embed,
            perplexity=args.perplexity,
            spread=args.spread,
        )
        print("[KMEANS] Saved:")
        for path in paths:
            print(f"  - {path}")


if __name__ == "__main__":
    main()
