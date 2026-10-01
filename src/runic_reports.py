"""Summaries, figures and REPORT.md for the further-work experiments (CPU only).

Kept separate from the GPU code so every number and chart that ends up in a
report is unit-tested in CI with fake predictions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import runic_eval as E

BLUE, INK, INK_MUTED, GRID, SURFACE = "#2a78d6", "#1f1f1e", "#6b6a64", "#e6e5df", "#fcfcfb"


def score(df: pd.DataFrame, seed: int = 0) -> dict:
    """Metrics + line- and inscription-level CIs for a frame with gt, pred, inscription."""
    m = E.compute_metrics(df["gt"].astype(str).tolist(), df["pred"].fillna("").astype(str).tolist())
    line = E.bootstrap_cer(m["char_dist"], m["char_len"], seed=seed)
    insc = E.bootstrap_cer(
        m["char_dist"], m["char_len"], groups=df["inscription"].astype(str).to_numpy(), seed=seed
    )
    return {
        "CER": 100 * m["CER"],
        "WER": 100 * m["WER"],
        "SeqAcc": 100 * m["SeqAcc"],
        "ci_line": (100 * line["ci_low"], 100 * line["ci_high"]),
        "ci_inscription": (100 * insc["ci_low"], 100 * insc["ci_high"]),
        "lines": len(df),
        "inscriptions": int(df["inscription"].nunique()),
        "_m": m,
    }


def compare(a: pd.DataFrame, b: pd.DataFrame, seed: int = 0) -> dict:
    """Paired, inscription-level bootstrap of CER(a) − CER(b) on the same files."""
    b = b.set_index("file").loc[a["file"]].reset_index()
    ma = E.compute_metrics(a["gt"].tolist(), a["pred"].fillna("").tolist())
    mb = E.compute_metrics(b["gt"].tolist(), b["pred"].fillna("").tolist())
    d = E.paired_bootstrap_delta(
        ma["char_dist"],
        ma["char_len"],
        mb["char_dist"],
        mb["char_len"],
        groups=a["inscription"].astype(str).to_numpy(),
        seed=seed,
    )
    return {k: (100 * v if isinstance(v, float) else v) for k, v in d.items()}


def models_table(
    preds: dict[str, pd.DataFrame], thesis: dict[str, pd.DataFrame] | None = None, seed: int = 0
) -> pd.DataFrame:
    """E1: one row per model.

    ``thesis`` maps model keys to the thesis predictions; their CER is recomputed on
    exactly the same files, so the reproduction check is like-for-like even when
    a gold image is missing.
    """
    rows = []
    for key, df in preds.items():
        s = score(df, seed)
        ref = None
        if thesis and key in thesis:
            t = thesis[key][thesis[key]["file"].isin(df["file"])]
            ref = round(
                score(t.assign(inscription=t["file"].map(E.inscription_id)), seed)["CER"], 2
            )
        rows.append(
            {
                "model": key,
                "lines": s["lines"],
                "CER": round(s["CER"], 2),
                "thesis_CER_same_lines": ref,
                "CI95_line": f"{s['ci_line'][0]:.1f}–{s['ci_line'][1]:.1f}",
                "CI95_inscription": f"{s['ci_inscription'][0]:.1f}–{s['ci_inscription'][1]:.1f}",
                "WER": round(s["WER"], 2),
                "SeqAcc": round(s["SeqAcc"], 2),
            }
        )
    return pd.DataFrame(rows).sort_values("CER").reset_index(drop=True)


def learning_curve_table(by_n: dict[int, pd.DataFrame], seed: int = 0) -> pd.DataFrame:
    """E2: pooled out-of-fold CER per training size N, with Δ vs. N = 0."""
    rows = []
    zero = by_n.get(0)
    for n in sorted(by_n, key=lambda x: (x < 0, x)):
        df = by_n[n]
        s = score(df, seed)
        row = {
            "N_real_lines": "all" if n < 0 else n,
            "mean_train_lines": round(float(df["n_train"].mean()), 1) if "n_train" in df else 0.0,
            "CER": round(s["CER"], 2),
            "ci_low": round(s["ci_inscription"][0], 1),
            "ci_high": round(s["ci_inscription"][1], 1),
            "SeqAcc": round(s["SeqAcc"], 2),
            "lines": s["lines"],
        }
        if zero is not None and n != 0:
            c = compare(df, zero, seed)
            row.update(
                delta_vs_0=round(c["delta"], 2),
                delta_ci=f"{c['ci_low']:.1f}…{c['ci_high']:.1f}",
                significant=c["significant"],
            )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_learning_curve(table: pd.DataFrame, path: str | Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = table["mean_train_lines"].to_numpy(float)
    y = table["CER"].to_numpy(float)
    lo, hi = table["ci_low"].to_numpy(float), table["ci_high"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=200, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.fill_between(x, lo, hi, color=BLUE, alpha=0.15, lw=0)
    ax.plot(x, y, color=BLUE, lw=2, marker="o", ms=7, markeredgecolor=SURFACE, markeredgewidth=2)
    for xi, yi in zip(x, y):
        ax.text(xi, yi + 2.5, f"{yi:.1f}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xlabel("real gold lines used for fine-tuning (per fold)", color=INK)
    ax.set_ylabel("CER on held-out inscriptions, %", color=INK)
    ax.set_ylim(0, max(100, float(np.nanmax(hi)) + 8))
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, length=0)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def df_to_md(df: pd.DataFrame) -> str:
    """Markdown table without the optional ``tabulate`` dependency."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |")
    return "\n".join(lines)


def predictions_frame(gold_rows: pd.DataFrame, recs: list[dict], **extra) -> pd.DataFrame:
    """Join gold rows with :func:`runic_vlm.predict` output into the standard predictions table."""
    import json

    df = pd.DataFrame(
        {
            "file": gold_rows["file_name"].to_numpy(),
            "gt": gold_rows["gt"].to_numpy(),
            "pred": [r["pred"] for r in recs],
            "inscription": gold_rows["inscription"].to_numpy(),
            "segment": gold_rows["segment"].to_numpy(),
        }
    )
    if recs and "nbest" in recs[0]:
        df["nbest"] = [json.dumps(r["nbest"], ensure_ascii=False) for r in recs]
        df["nbest_scores"] = [json.dumps([round(s, 4) for s in r["scores"]]) for r in recs]
    for k, v in extra.items():
        df[k] = v
    return df
