"""
PU and CWRU C7_MDC per-class accuracy boxplots across target SNR levels.

The script reads five random-seed runs per dataset from the experiment results
folder and generates one boxplot figure per dataset/seed. Each box uses the
10 class-wise recalls computed from the saved confusion matrix:

    per-class accuracy = diag(confusion_matrix) / row_sum(confusion_matrix)
"""

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np


RESULTS_ROOT_CANDIDATES = [
    Path(r"E:\Python\Muti-DANN\results"),
    
]
DATASETS = ("PU", "CWRU")
EXPERIMENT_MODE = "C7_MDC"
SEEDS = [42, 100, 200, 400, 1000]

SNR_SPECS = [
    ("-6dB", "domain_-6dB"),
    ("-3dB", "domain_-3dB"),
    ("+0dB", "domain_+0dB"),
    ("+3dB", "domain_+3dB"),
    ("+6dB", "domain_+6dB"),
]

BOX_EDGE_COLOR = "#0000FF"
MEDIAN_COLOR = "#FF0000"
OUTLIER_COLOR = "#FF0000"


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            "font.size": 10,
            "axes.linewidth": 0.75,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def per_class_accuracy(confusion_matrix: np.ndarray) -> np.ndarray:
    row_sum = confusion_matrix.sum(axis=1)
    return np.divide(
        np.diag(confusion_matrix),
        row_sum,
        out=np.full(confusion_matrix.shape[0], np.nan, dtype=float),
        where=row_sum > 0,
    ) * 100.0


def experiment_dir_name(dataset: str, seed: int) -> str:
    return f"exp_ablation_{dataset}_{EXPERIMENT_MODE}_seed{seed}"


def resolve_results_root() -> Path:
    for root in RESULTS_ROOT_CANDIDATES:
        if all((root / experiment_dir_name(dataset, seed)).exists()
               for dataset in DATASETS for seed in SEEDS):
            return root

    existing = [root for root in RESULTS_ROOT_CANDIDATES if root.exists()]
    if existing:
        return existing[0]

    candidates = ", ".join(str(root) for root in RESULTS_ROOT_CANDIDATES)
    raise FileNotFoundError(f"No available results root found. Tried: {candidates}")


def load_seed_data(results_root: Path, dataset: str, seed: int) -> List[np.ndarray]:
    run_dir = results_root / experiment_dir_name(dataset, seed)
    npz_path = run_dir / "confusion_matrices.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing confusion matrix file: {npz_path}")

    matrices = np.load(npz_path)
    data = []
    for _, key in SNR_SPECS:
        if key not in matrices.files:
            raise KeyError(f"Missing key '{key}' in {npz_path}")
        cm = matrices[key]
        if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
            raise ValueError(f"Expected a square confusion matrix for {key}, got {cm.shape}")
        data.append(per_class_accuracy(cm))
    return data


def y_axis_floor(data: List[np.ndarray]) -> int:
    min_value = float(np.nanmin(np.concatenate(data)))
    return int(max(0, np.floor(min_value / 10.0) * 10.0))


def plot_seed(dataset: str, seed: int, data: List[np.ndarray], out_dir: Path) -> None:
    positions = np.arange(1, len(SNR_SPECS) + 1)
    fig, ax = plt.subplots(figsize=(3.6, 2.5))

    ax.boxplot(
        data,
        positions=positions,
        widths=0.22,
        patch_artist=False,
        whis=1.5,
        showfliers=True,
        manage_ticks=False,
        boxprops={"color": BOX_EDGE_COLOR, "linewidth": 0.75},
        medianprops={"color": MEDIAN_COLOR, "linewidth": 0.75},
        whiskerprops={"color": "#000000", "linewidth": 0.75},
        capprops={"color": "#000000", "linewidth": 0.75},
        flierprops={
            "marker": "+",
            "markerfacecolor": OUTLIER_COLOR,
            "markeredgecolor": OUTLIER_COLOR,
            "markeredgewidth": 0.75,
            "markersize": 4.2,
            "linestyle": "none",
        },
    )

    y_min = y_axis_floor(data)
    ax.set_ylim(y_min, 102)
    ax.set_yticks(np.arange(y_min, 101, 10))
    ax.set_xlim(0.5, len(SNR_SPECS) + 0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels([label for label, _ in SNR_SPECS])
    ax.set_ylabel("Per-class Accuracy(%)")
    ax.set_xlabel("Target SNR (dB)")
    ax.grid(False)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{dataset.lower()}_c7_mdc_per_class_boxplot_seed{seed}"
    for fmt in ("png", "svg", "pdf"):
        kwargs = {"dpi": 220} if fmt == "png" else {}
        fig.savefig(out_dir / f"{stem}.{fmt}", bbox_inches="tight", facecolor="white", **kwargs)
        print(f"Saved: {out_dir / f'{stem}.{fmt}'}")
    plt.close(fig)


def main() -> None:
    apply_publication_style()
    project_root = Path(__file__).resolve().parents[1]
    out_dir = project_root / "figure_output" / "per_class_boxplots"
    results_root = resolve_results_root()
    print(f"Using results root: {results_root}")
    for dataset in DATASETS:
        for seed in SEEDS:
            plot_seed(dataset, seed, load_seed_data(results_root, dataset, seed), out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
