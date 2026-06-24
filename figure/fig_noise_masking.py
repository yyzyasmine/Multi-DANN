"""
Figure: Noise masking effect on bearing fault impulse signatures.
Left: time-domain waveform + Hilbert envelope. Right: CWT spectrogram.
Rows: Clean → SNR gradient (6, 0, −4, −6 dB).
Data: CWRU B007 ball fault, 12 kHz Drive End.
"""

import numpy as np
import scipy.io
import scipy.signal
import matplotlib.pyplot as plt
import pywt
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "figure_output"

# ── Style ────────────────────────────────────────────────────────────
def apply_style():
    """ 8 pt base, Times New Roman, thin axes."""
    plt.rcParams.update({
        "font.size": 8,
        "font.family": "Times New Roman",
        "font.serif": ["Times New Roman"],
        "font.style": "normal",
        "mathtext.fontset": "stix",
        "axes.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.major.width": 0.4,
        "ytick.major.width": 0.4,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
        "xtick.major.pad": 2,
        "ytick.major.pad": 2,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "legend.frameon": False,
        "legend.fontsize": 7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })

# ── Load data ────────────────────────────────────────────────────────
base = r"E:\Python\QCNN_for_bearing_diagnosis1\QCNN_for_bearing_diagnosis\data"
fname = "12k_Drive_End_B007_0_118.mat"
fs = 12_000  # Hz
g_to_mps2 = 9.80665

# Clean signal (full 122571 samples, key differs)
sig_full = scipy.io.loadmat(f"{base}/0HP/{fname}")["X118_DE_time"].flatten() * g_to_mps2

# Test split = last 40857 samples
train_len = 81714
sig_clean = sig_full[train_len:]

# Load Test-split noised versions (same underlying segment)
snr_labels = [6, 0, -6]
sigs_noised = {}
for snr in snr_labels:
    path = f"{base}/0HP_TestNoised_{snr}/{fname}"
    sigs_noised[snr] = scipy.io.loadmat(path)["DE"].flatten() * g_to_mps2

# ── Pick best segment (highest kurtosis in clean = strongest impulses)
win = int(0.08 * fs)  # 80 ms ≈ 960 samples
offsets = np.arange(0, len(sig_clean) - win, win // 4)
kurt = np.array([
    np.mean(((sig_clean[i:i+win] - sig_clean[i:i+win].mean()) /
              sig_clean[i:i+win].std()) ** 4) - 3
    for i in offsets
])
start = offsets[np.argmax(kurt)]

t = np.arange(win) / fs * 1000  # ms

# Build ordered list: (label, segment)
rows = [("Clean", sig_clean[start:start+win])]
for snr in snr_labels:
    rows.append((f"SNR = {snr} dB", sigs_noised[snr][start:start+win]))

# ── Hilbert envelope ─────────────────────────────────────────────────
def hilbert_env(x):
    analytic = scipy.signal.hilbert(x)
    return np.abs(analytic)

# ── CWT parameters ───────────────────────────────────────────────────
freqs = np.linspace(500, 5500, 128)  # Hz range of interest
scales = pywt.frequency2scale("cmor1.5-1.0", freqs / fs)

# ── Plot ─────────────────────────────────────────────────────────────
apply_style()

ncols = len(rows)
fig, axes = plt.subplots(2, ncols, figsize=(7.2, 2.35),
                         gridspec_kw={"height_ratios": [1, 1],
                                      "wspace": 0.32, "hspace": 0.42},
                         squeeze=False)

colors_wave = ["#333333", "#1a6faf", "#d4880f", "#6c1d5f"]
ylim_global = max(np.abs(r[1]).max() for r in rows) * 1.12

for i, (label, seg) in enumerate(rows):
    ax_t = axes[0, i]   # time-domain
    ax_f = axes[1, i]   # CWT

    # --- Waveform + envelope ---
    env = hilbert_env(seg)
    ax_t.plot(t, seg, linewidth=0.2, color=colors_wave[i], alpha=0.75,
              rasterized=True)
    ax_t.plot(t, env, linewidth=0.5, color="#E74C3C", alpha=0.75)
    ax_t.plot(t, -env, linewidth=0.5, color="#E74C3C", alpha=0.75)
    ax_t.set_ylim(-ylim_global, ylim_global)
    ax_t.set_title(label, pad=2, fontstyle="normal")
    ax_t.set_xlabel("Time (ms)")
    if i == 0:
        ax_t.set_ylabel(r"Amplitude (m/s$^2$)", labelpad=1)
    else:
        ax_t.tick_params(labelleft=False)

    # --- CWT spectrogram ---
    coefs, _ = pywt.cwt(seg, scales, "cmor1.5-1.0", sampling_period=1/fs)
    power = np.abs(coefs) ** 2
    im = ax_f.pcolormesh(t, freqs / 1000, power, shading="gouraud",
                         cmap="inferno", rasterized=True)
    ax_f.set_xlabel("Time (ms)")
    if i == 0:
        ax_f.set_ylabel("Freq. (kHz)", labelpad=1)
    else:
        ax_f.tick_params(labelleft=False)

# ── Export ────────────────────────────────────────────────────────────
out = OUTPUT_ROOT / "noise_masking"
out.parent.mkdir(parents=True, exist_ok=True)
for fmt in ("png", "svg", "pdf"):
    kwargs = {"dpi": 300} if fmt == "png" else {}
    fig.savefig(f"{out}.{fmt}", **kwargs)
print(f"Saved to {out}.png / .svg / .pdf")
plt.close(fig)
