from pathlib import Path
import re
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 对比实验折线图
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "figure_output"
OUT_DIR = OUTPUT_ROOT / "method_comparison_figures"

PALETTE = {
    "ours": "#413FD1",
    "grid": "#E5E2DC",
}


@dataclass(frozen=True)
class FigureStyle:
    font_size: int = 18
    legend_font_size: int = 14
    axes_linewidth: float = 2.2
    dpi: int = 600

DATASETS = {
    "PU": PROJECT_ROOT / "results" / "deploy_metrics.xlsx",
    "CWRU": PROJECT_ROOT / "results" / "deploy_metrics_CWRU.xlsx",
}

METHODS = ["DAN", "JAN", "DANN", "CDAN", "C7_MDC"]
LEGEND_METHODS = ["C7_MDC", "DAN", "JAN", "DANN", "CDAN"]
METHOD_LABELS = {
    "DAN": "DAN",
    "JAN": "JAN",
    "DANN": "DANN",
    "CDAN": "CDAN",
    "C7_MDC": "Ours",
}
DOMAINS = ["-6dB", "-3dB", "+0dB", "+3dB", "+6dB"]
DOMAIN_LABELS = ["-6", "-3", "0", "+3", "+6"]

DATASET_COLORS = {
    "PU": {
        "DAN": "#DBC8AB",
        "JAN": "#9ECEAA",
        "DANN": "#94ACB6",
        "CDAN": "#D3AE9F",
        "C7_MDC": PALETTE["ours"],
    },
    "CWRU": {
        "DAN": "#E4BC80",
        "JAN": "#72D189",
        "DANN": "#85BED4",
        "CDAN": "#E48A67",
        "C7_MDC": PALETTE["ours"],
    },
}
MARKERS = {
    "DAN": "o",
    "JAN": "s",
    "DANN": "^",
    "CDAN": "D",
    "C7_MDC": "*",
}
LINESTYLES = {
    "DAN": "--",
    "JAN": "-.",
    "DANN": ":",
    "CDAN": "--",
    "C7_MDC": "-",
}


def parse_mean_std(value):
    if isinstance(value, str):
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value)
        if len(nums) >= 2:
            return float(nums[0]), float(nums[1])
        if len(nums) == 1:
            return float(nums[0]), np.nan
    if pd.notna(value):
        return float(value), np.nan
    return np.nan, np.nan


def load_metric_table(excel_path, metric="acc"):
    df = pd.read_excel(excel_path, sheet_name="deploy_metrics")
    df = df[
        df["mode"].isin(METHODS)
        & df["domain"].isin(DOMAINS)
        & (df["aggregate_kind"] == "MEAN+/-STD")
    ].copy()

    records = []
    for _, row in df.iterrows():
        mean, std = parse_mean_std(row[metric])
        records.append(
            {
                "method": row["mode"],
                "domain": row["domain"],
                "mean": mean * 100.0,
                "std": std * 100.0,
            }
        )

    table = pd.DataFrame(records)
    expected = {(method, domain) for method in METHODS for domain in DOMAINS}
    found = set(zip(table["method"], table["domain"]))
    missing = sorted(expected - found)
    if missing:
        raise ValueError(f"Missing method/domain rows in {excel_path}: {missing}")
    return table


def apply_publication_style(style=FigureStyle()):
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman"],
            "font.size": style.font_size,
            "axes.labelsize": style.font_size,
            "axes.titlesize": style.font_size + 1,
            "axes.linewidth": style.axes_linewidth,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.labelsize": style.font_size - 1,
            "ytick.labelsize": style.font_size - 1,
            "legend.fontsize": style.legend_font_size,
            "legend.frameon": True,
            "legend.fancybox": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "stix",
        }
    )


def plot_dataset(ax, dataset_name, table):
    ax.set_title("")
    x = np.arange(len(DOMAINS))
    colors = DATASET_COLORS[dataset_name]
    for method in METHODS:
        subset = table[table["method"] == method].set_index("domain").loc[DOMAINS]
        y = subset["mean"].to_numpy(dtype=float)

        is_ours = method == "C7_MDC"
        ax.plot(
            x,
            y,
            label=METHOD_LABELS[method],
            color=colors[method],
            marker=MARKERS[method],
            linestyle=LINESTYLES[method],
            linewidth=3.2 if is_ours else 2.35,
            markersize=11.0 if is_ours else 7.8,
            markeredgecolor="white",
            markeredgewidth=0.9 if is_ours else 0.65,
            zorder=5 if is_ours else 3,
        )

    ax.set_ylim(40, 101.5)
    ax.set_yticks([40, 60, 80, 100])
    ax.set_xticks(x, DOMAIN_LABELS)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=1.0, alpha=0.6)
    ax.tick_params(direction="out", length=6.0, width=1.8)
    handles, labels = ax.get_legend_handles_labels()
    handle_by_label = dict(zip(labels, handles))
    ordered_labels = [METHOD_LABELS[method] for method in LEGEND_METHODS]
    ordered_handles = [handle_by_label[label] for label in ordered_labels]
    leg = ax.legend(
        ordered_handles,
        ordered_labels,
        ncol=2,
        loc="lower right",
        bbox_to_anchor=(0.99, 0.045),
        handlelength=1.35,
        columnspacing=0.55,
        handletextpad=0.35,
        borderaxespad=0.0,
        borderpad=0.28,
        labelspacing=0.28,
    )
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("#7A7772")
    leg.get_frame().set_linewidth(0.8)
    leg.get_frame().set_alpha(0.88)


def finalize_figure(fig, stem, style=FigureStyle(), rect=None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if rect is None:
        fig.tight_layout(pad=1.1)
    else:
        fig.tight_layout(pad=1.1, rect=rect)
    for ext in ("svg", "pdf"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=style.dpi, bbox_inches="tight")
    fig.savefig(
        OUT_DIR / f"{stem}.tiff",
        dpi=style.dpi,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def main():
    style = FigureStyle()
    apply_publication_style(style)
    tables = {
        dataset_name: load_metric_table(excel_path)
        for dataset_name, excel_path in DATASETS.items()
    }

    for dataset_name, table in tables.items():
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        plot_dataset(ax, dataset_name, table)
        finalize_figure(
            fig,
            f"{dataset_name}_method_comparison_acc",
            style=style,
            rect=(0, 0, 1, 1),
        )

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
    for ax, (dataset_name, table) in zip(axes, tables.items()):
        plot_dataset(ax, dataset_name, table)
    finalize_figure(
        fig,
        "PU_CWRU_method_comparison_acc",
        style=style,
        rect=(0, 0, 1, 1),
    )

    print(f"Saved figures to: {OUT_DIR}")


if __name__ == "__main__":
    main()
