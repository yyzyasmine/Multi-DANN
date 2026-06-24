"""
Signal comparison figure: CWRU vs PU bearing vibration datasets.
Publication-ready style following scientific-figure-making skill guidelines.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "figure_output"

# ── Palette ──────────────────────────────────────────────────────────────────
PALETTE = {
    "blue_main":      "#0F4D92",
    "blue_secondary": "#3775BA",
    "red_strong":     "#B64342",
    "neutral":        "#CFCECE",
    "teal":           "#42949E",
    "vivid_blue":     "#0077B6",
    "vivid_orange":   "#F77F00",
    "muted_orange":   "#D4892A",
}

# ── Style ──────────────────────────────────────────��──────────────────────────
def apply_publication_style(font_size=16, axes_linewidth=2.5):
    plt.rcParams.update({
        "font.family":          "Times New Roman",
        "font.serif":           ["Times New Roman"],
        "mathtext.fontset":     "stix",
        "font.size":            font_size,
        "axes.titlesize":       font_size,
        "axes.labelsize":       font_size,
        "xtick.labelsize":      font_size - 2,
        "ytick.labelsize":      font_size - 2,
        "axes.linewidth":       axes_linewidth,
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "legend.frameon":       False,
        "legend.fontsize":      font_size - 2,
        "text.usetex":          False,
        "figure.dpi":           150,
        "savefig.dpi":          300,
        "savefig.bbox":         "tight",
        "svg.fonttype":         "none",
        "pdf.fonttype":         42,
        "ps.fonttype":          42,
    })

# ── Data loading ──────────────────────────────────────────────────────────────
def load_cwru_signal(mat_path, n_samples=2048):
    """Load CWRU drive-end vibration signal."""
    data = loadmat(mat_path)
    for key in data:
        if 'DE' in key and not key.startswith('_'):
            signal = data[key].ravel()
            return signal[:n_samples]
    raise KeyError(f"No 'DE' key found in {mat_path}")


def load_pu_signal(mat_path, n_samples=2048):
    """Load PU vibration_1 signal from structured .mat file."""
    data = loadmat(mat_path)
    base_name = os.path.splitext(os.path.basename(mat_path))[0]
    struct = data[base_name]
    y_field = struct[0, 0]['Y']
    for elem in y_field.ravel():
        name = elem['Name'][0].strip() if elem['Name'].size > 0 else ''
        if name == 'vibration_1':
            signal = elem['Data'].ravel()
            return signal[:n_samples]
    raise KeyError("'vibration_1' not found in Y field")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    apply_publication_style(font_size=16, axes_linewidth=2.0)

    CWRU_PATH = r'E:\Python\CNN-DANN-CWRU\data\0HP\12k_Drive_End_B007_0_118.mat'
    PU_PATH   = r'E:\Python\CNN-DANN-PU\data\PU\KB23\N09_M07_F10_KB23_1.mat'

    cwru_fs = 12_000   # Hz
    pu_fs   = 64_000   # Hz
    g_to_mps2 = 9.80665

    # Use a common time window so both x-axes align
    duration_ms = 32.0                                  # ms
    cwru_n = int(cwru_fs * duration_ms / 1e3)           # 384 samples
    pu_n   = int(pu_fs   * duration_ms / 1e3)           # 2048 samples

    cwru_signal = load_cwru_signal(CWRU_PATH, n_samples=cwru_n) * g_to_mps2
    pu_signal   = load_pu_signal(PU_PATH, n_samples=pu_n) * g_to_mps2

    # Time axes in milliseconds
    cwru_t = np.arange(cwru_n) / cwru_fs * 1e3
    pu_t   = np.arange(pu_n)   / pu_fs   * 1e3

    # ── Figure layout: 1 row, 2 col ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    fig.subplots_adjust(wspace=0.38, bottom=0.18)
    x_ticks = [0, 10, 20, 30]

    # Panel (a) – CWRU
    ax0 = axes[0]
    ax0.plot(cwru_t, cwru_signal, color=PALETTE["vivid_blue"],
             linewidth=0.9, alpha=0.95)
    ax0.set_xlabel("Time (ms)", labelpad=4)
    ax0.set_ylabel(r"Acceleration (m/s$^2$)", labelpad=6)
    ax0.tick_params(axis='both', length=4, width=1.2)

    # Panel (b) – PU
    ax1 = axes[1]
    ax1.plot(pu_t, pu_signal, color=PALETTE["muted_orange"],
             linewidth=0.9, alpha=0.95)
    ax1.set_xlabel("Time (ms)", labelpad=4)
    ax1.set_ylabel(r"Acceleration (m/s$^2$)", labelpad=6)
    ax1.tick_params(axis='both', length=4, width=1.2)
    ax1.tick_params(axis='y', labelleft=True)

    for ax in axes:
        ax.set_xlim(0, duration_ms)
        ax.set_xticks(x_ticks)
        ax.margins(x=0)

    # ── Export ────────────────────────────────────────────────────────────────
    out_dir = OUTPUT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    out_base = out_dir / 'signal_comparison'
    for fmt in ('png', 'svg', 'pdf'):
        fig.savefig(f'{out_base}.{fmt}', dpi=300, bbox_inches='tight', pad_inches=0.05)
        print(f"Saved: {out_base}.{fmt}")

    plt.close(fig)


if __name__ == "__main__":
    main()
