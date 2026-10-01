"""Plot synthetic vs. real CER for every fine-tuned model (figure for slides/README).

Usage:
    python scripts/plot_results.py  # -> results/figures/cer_synth_vs_gold.png

Reads ``results/metrics/summary.csv``; needs only pandas + matplotlib.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "metrics" / "summary.csv"
OUT = ROOT / "results" / "figures" / "cer_synth_vs_gold.png"

SYNTH_COLOR = "#2a78d6"  # categorical slot 1
GOLD_COLOR = "#eb6834"  # categorical slot 2
INK, INK_MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e6e5df", "#fcfcfb"
USABLE_CER = 20  # practical-usability threshold discussed in the thesis


def main() -> None:
    df = pd.read_csv(SRC)
    df = df[df["setting"] != "zero-shot"].reset_index(drop=True)
    ci = df["CER_gold_ci95"].str.split("-", expand=True).astype(float)
    err = np.vstack([df["CER_gold"] - ci[0], ci[1] - df["CER_gold"]])

    x = np.arange(len(df))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=200, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.bar(x - w / 2 - 0.01, df["CER_synth_val"], w, color=SYNTH_COLOR, label="Synthetic val")
    ax.bar(x + w / 2 + 0.01, df["CER_gold"], w, color=GOLD_COLOR, label="Real gold set (95 % CI)")
    ax.errorbar(
        x + w / 2 + 0.01,
        df["CER_gold"],
        yerr=err,
        fmt="none",
        ecolor=INK,
        elinewidth=1.2,
        capsize=3,
    )

    for xi, v, hi in zip(x, df["CER_gold"], ci[1]):
        ax.text(xi + w / 2, hi + 1.5, f"{v:.1f}", ha="center", va="bottom", fontsize=9, color=INK)

    ax.axhline(USABLE_CER, color=INK_MUTED, lw=1, ls="--", zorder=0, label="Usable OCR (< 20 %)")

    labels = [f"{m}\n{p}" for m, p in zip(df["model"], df["params"])]
    ax.set_xticks(x, labels, fontsize=9, color=INK)
    ax.set_ylabel("CER, %", color=INK)
    ax.set_ylim(0, 112)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="y", colors=INK_MUTED, length=0)
    ax.tick_params(axis="x", length=0)
    ax.set_title(
        "Trained on synthetic only: CER on synthetic validation vs. real inscriptions",
        loc="left",
        fontsize=12,
        color=INK,
        pad=14,
    )
    ax.legend(frameon=False, loc="upper left", fontsize=9, ncols=3, bbox_to_anchor=(0, 1.0))
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, facecolor=SURFACE)
    print(f"saved {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
