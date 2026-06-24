"""
Grouped bar charts for Multi-DANN ablation accuracy.

The script reads deployment metric workbooks and draws one figure per dataset.
Bars show mean accuracy across the five random seeds for each target SNR.
Missing ablation groups are kept as empty hatched slots instead of being
imputed as zero. Legend labels follow the ablation table.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_FILES = {
    "CWRU": PROJECT_ROOT / "results" / "deploy_metrics_CWRU.xlsx",
    "PU": PROJECT_ROOT / "results" / "deploy_metrics.xlsx",
}

SNR_ORDER = ["-6dB", "-3dB", "+0dB", "+3dB", "+6dB"]
SNR_LABELS = ["-6", "-3", "0", "+3", "+6"]

GROUPS = [
    ("C7_MDC", "A1"),
    ("B6_MC", "A2"),
    ("B5_MD", "A3"),
    ("B8_DC", "A4"),
    ("A3_D", "B1"),
    ("A2_M", "B2"),
    ("A4_C", "B3"),
    ("A0_BASE", "C1"),
]

DATASET_COLORS = {
    "CWRU": {
        "A1": "#7B5B8E",
        "A2": "#99C2D8",
        "A3": "#9ADDCC",
        "A4": "#99D8B5",
        "B1": "#CDD892",
        "B2": "#D4C299",
        "B3": "#DFB989",
        "C1": "#A7A5A5",
    },
    "PU": {
        "A1": "#7B5B8E",
        "A2": "#77AFCE",
        "A3": "#5DC7AD",
        "A4": "#63D195",
        "B1": "#BDCE61",
        "B2": "#CCAE6D",
        "B3": "#D8A45F",
        "C1": "#887777",
    },
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            "font.size": 12,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _xlsx_sheet_rows(path: Path, sheet_name: str) -> list[list[object]]:
    ns_main = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    ns_rel = {"r": "http://schemas.openxmlformats.org/package/2006/relationships"}

    def cell_value(cell: ET.Element, shared: list[str]) -> object:
        cell_type = cell.attrib.get("t")
        value = cell.find("m:v", ns_main)
        if cell_type == "inlineStr":
            text = cell.find(".//m:t", ns_main)
            return text.text if text is not None else ""
        if value is None:
            return None
        raw = value.text or ""
        if cell_type == "s":
            return shared[int(raw)]
        try:
            number = float(raw)
        except ValueError:
            return raw
        return int(number) if number.is_integer() else number

    def col_index(cell_ref: str) -> int:
        letters = "".join(ch for ch in cell_ref if ch.isalpha())
        index = 0
        for ch in letters:
            index = index * 26 + ord(ch.upper()) - ord("A") + 1
        return index - 1

    with ZipFile(path) as zf:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for item in root.findall("m:si", ns_main):
                texts = [node.text or "" for node in item.findall(".//m:t", ns_main)]
                shared.append("".join(texts))

        wb_root = ET.fromstring(zf.read("xl/workbook.xml"))
        rel_root = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rels = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rel_root.findall("r:Relationship", ns_rel)}

        target = None
        for sheet in wb_root.findall(".//m:sheet", ns_main):
            if sheet.attrib.get("name") == sheet_name:
                target = rels[sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]]
                break
        if target is None:
            raise KeyError(f"Sheet '{sheet_name}' not found in {path}")

        if target.startswith("/"):
            sheet_path = target.lstrip("/")
        else:
            sheet_path = "xl/" + target
        sheet_root = ET.fromstring(zf.read(sheet_path))
        rows = []
        for row in sheet_root.findall(".//m:row", ns_main):
            values = []
            for cell in row.findall("m:c", ns_main):
                idx = col_index(cell.attrib["r"])
                while len(values) <= idx:
                    values.append(None)
                values[idx] = cell_value(cell, shared)
            rows.append(values)
        return rows


def parse_mean_std(value: object) -> tuple[float, float | None]:
    if isinstance(value, (int, float)):
        return float(value), None
    text = str(value).replace("±", "+/-").replace("卤", "+/-")
    if "+/-" in text:
        mean, std = text.split("+/-", 1)
        return float(mean.strip()), float(std.strip())
    return float(text), None


def load_accuracy(path: Path) -> tuple[dict[str, dict[str, tuple[float, float | None]]], set[str]]:
    rows = _xlsx_sheet_rows(path, "deploy_metrics")
    headers = {name: idx for idx, name in enumerate(rows[0]) if name}
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    summary: dict[str, dict[str, tuple[float, float | None]]] = defaultdict(dict)
    seen_modes: set[str] = set()

    for row in rows[1:]:
        if not row or len(row) <= max(headers.values()):
            continue
        mode = row[headers["mode"]]
        domain = row[headers["domain"]]
        if mode is None or domain not in SNR_ORDER:
            continue
        mode = str(mode)
        seen_modes.add(mode)
        kind = row[headers["aggregate_kind"]]
        acc = row[headers["acc"]]
        if acc is None:
            continue
        if kind == "MEAN+/-STD":
            summary[mode][domain] = parse_mean_std(acc)
        else:
            mean, _ = parse_mean_std(acc)
            grouped[(mode, domain)].append(mean)

    for (mode, domain), values in grouped.items():
        if domain not in summary[mode] and values:
            arr = np.asarray(values, dtype=float)
            summary[mode][domain] = (float(arr.mean()), float(arr.std(ddof=0)))

    return summary, seen_modes


def plot_dataset(dataset: str, path: Path, out_dir: Path) -> None:
    data, seen_modes = load_accuracy(path)
    colors = DATASET_COLORS[dataset]
    labels = [label for _, label in GROUPS]
    x = np.arange(len(SNR_ORDER), dtype=float)
    bar_width = 0.095
    offsets = (np.arange(len(GROUPS)) - (len(GROUPS) - 1) / 2.0) * bar_width

    fig, ax = plt.subplots(figsize=(7.8, 3.7))
    all_values = []

    for group_idx, (mode, label) in enumerate(GROUPS):
        xpos = x + offsets[group_idx]
        means = []
        stds = []
        missing_positions = []
        for snr_idx, domain in enumerate(SNR_ORDER):
            if mode in data and domain in data[mode]:
                mean, std = data[mode][domain]
                means.append(mean * 100.0)
                stds.append(0.0 if std is None else std * 100.0)
                all_values.append(mean * 100.0)
            else:
                means.append(np.nan)
                stds.append(0.0)
                missing_positions.append(xpos[snr_idx])

        ax.bar(
            xpos,
            means,
            yerr=stds,
            width=bar_width * 0.88,
            color=colors[label],
            edgecolor="white",
            linewidth=0.4,
            error_kw={"elinewidth": 0.55, "capsize": 1.6, "capthick": 0.55, "ecolor": "#555555"},
            label=label,
            zorder=3,
        )
        if missing_positions:
            ax.bar(
                missing_positions,
                [1.0] * len(missing_positions),
                bottom=[0.0] * len(missing_positions),
                width=bar_width * 0.88,
                color="#F2F2F2",
                edgecolor="#B5B5B5",
                linewidth=0.5,
                hatch="///",
                zorder=2,
            )

    if all_values:
        y_min = max(0, np.floor((min(all_values) - 5.0) / 5.0) * 5.0)
        y_max = min(103, np.ceil((max(all_values) + 2.0) / 5.0) * 5.0)
    else:
        y_min, y_max = 0, 100
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel("Accuracy (%)", fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(SNR_LABELS, fontsize=13)
    ax.set_xlabel("Target SNR (dB)", fontsize=15)
    ax.tick_params(axis="y", labelsize=13)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.45, color="#D0D0D0", alpha=0.85)
    ax.set_axisbelow(True)

    handles, legend_labels = ax.get_legend_handles_labels()
    legend_order = ["A1", "B1", "A2", "B2", "A3", "B3", "A4", "C1"]
    legend_lookup = dict(zip(legend_labels, handles))
    ordered_labels = [label for label in legend_order if label in legend_lookup]
    ordered_handles = [legend_lookup[label] for label in ordered_labels]
    ax.legend(
        ordered_handles,
        ordered_labels,
        ncol=4,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.035),
        frameon=True,
        fancybox=False,
        framealpha=0.92,
        edgecolor="#B8B8B8",
        facecolor="white",
        columnspacing=1.1,
        handlelength=1.1,
        handletextpad=0.35,
        borderpad=0.45,
        fontsize=11,
    )

    missing_modes = [label for mode, label in GROUPS if mode not in seen_modes]
    if missing_modes:
        ax.text(
            0.995,
            0.02,
            "Missing: " + ", ".join(missing_modes),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.5,
            color="#777777",
        )
        print(f"[WARN] {dataset} missing groups in workbook: {', '.join(missing_modes)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"ablation_accuracy_bars_{dataset.lower()}"
    for fmt in ("png", "svg", "pdf"):
        fig.savefig(out_dir / f"{stem}.{fmt}", dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {out_dir / f'{stem}.{fmt}'}")
    plt.close(fig)


def main() -> None:
    apply_style()
    out_dir = Path(__file__).resolve().parent / "output"
    for dataset, path in DATASET_FILES.items():
        plot_dataset(dataset, path, out_dir)


if __name__ == "__main__":
    main()
