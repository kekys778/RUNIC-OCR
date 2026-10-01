"""E0 — CPU error analysis of saved gold-set predictions.

Usage:
    python scripts/analyze_predictions.py \
        [--pred results/predictions/qwen25vl-7b_finetuned_gold.csv] \
        [--baseline results/predictions/qwen3vl-2b_zeroshot_gold.csv] \
        [--out results/analysis/qwen25vl-7b_finetuned]

Writes metrics.json, per-segment / per-length tables, top confusions, three
figures and REPORT.md. The predictions CSV needs columns file, gt, pred.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import runic_eval as E  # noqa: E402

GOLD_CSV = ROOT / "data" / "gold_set" / "real_corpus.csv"
BLUE, INK, INK_MUTED, GRID, SURFACE = "#2a78d6", "#1f1f1e", "#6b6a64", "#e6e5df", "#fcfcfb"
LEN_BINS = [(1, 4), (5, 8), (9, 14), (15, 99)]


def load(pred_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(pred_csv)
    df["gt"] = df["gt"].astype(str)
    df["pred"] = df["pred"].fillna("").astype(str)
    gold = pd.read_csv(GOLD_CSV)[["filename", "source"]].rename(columns={"filename": "file"})
    df = df.merge(gold, on="file", how="left")
    df["inscription"] = df["file"].map(E.inscription_id)
    df["segment"] = np.where(df["source"] == "real", "full line", "word crop")
    df["n_chars"] = df["gt"].map(lambda s: len(E.norm_cer(s)))
    return df


def subset_row(df: pd.DataFrame, label: str, seed: int) -> dict:
    m = E.compute_metrics(df["gt"].tolist(), df["pred"].tolist())
    ci = E.bootstrap_cer(m["char_dist"], m["char_len"], groups=df["inscription"], seed=seed)
    return {
        "subset": label,
        "lines": len(df),
        "inscriptions": df["inscription"].nunique(),
        "CER": 100 * m["CER"],
        "ci_low": 100 * ci["ci_low"],
        "ci_high": 100 * ci["ci_high"],
        "SeqAcc": 100 * m["SeqAcc"],
    }


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, length=0)


def fig_subsets(tab: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(11, 4.6), dpi=200, facecolor=SURFACE)
    style(ax)
    x = (
        np.arange(len(tab))
        + np.where(np.arange(len(tab)) >= 3, 0.6, 0)
        + np.where(np.arange(len(tab)) >= 1, 0.4, 0)
    )
    err = np.vstack([tab["CER"] - tab["ci_low"], tab["ci_high"] - tab["CER"]])
    ax.bar(x, tab["CER"], 0.6, color=BLUE)
    ax.errorbar(x, tab["CER"], yerr=err, fmt="none", ecolor=INK, elinewidth=1.2, capsize=3)
    for xi, v, hi in zip(x, tab["CER"], tab["ci_high"]):
        ax.text(xi, hi + 1.5, f"{v:.1f}", ha="center", va="bottom", fontsize=9, color=INK)
    labels = [
        f"{s}\n{n} lines · {k} inscr."
        for s, n, k in zip(tab["subset"], tab["lines"], tab["inscriptions"])
    ]
    ax.set_xticks(x, labels, fontsize=8, color=INK)
    ax.set_ylabel("CER, %", color=INK)
    ax.set_ylim(0, max(100, float(tab["ci_high"].max()) + 12))
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title(
        "CER by segment and length (95 % CI, inscription-level bootstrap)",
        loc="left",
        fontsize=11,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def fig_top_subs(subs: list, path: Path):
    labels = [f"{r} → {p}" for (r, p), _ in subs][::-1]
    vals = [n for _, n in subs][::-1]
    fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(subs) + 1.2), dpi=200, facecolor=SURFACE)
    style(ax)
    ax.barh(labels, vals, color=BLUE, height=0.65)
    for y, v in enumerate(vals):
        ax.text(v + 0.2, y, str(v), va="center", fontsize=8.5, color=INK)
    ax.tick_params(axis="y", labelsize=10, colors=INK)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines["bottom"].set_visible(False)
    ax.set_xlabel("count", color=INK_MUTED)
    ax.set_title(
        "Most frequent substitutions (reference → predicted)", loc="left", fontsize=11, color=INK
    )
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def fig_confusion(conf: dict, path: Path, min_count: int = 5):
    mat = E.confusion_frame(conf, min_count=min_count)
    mat = mat[[c for c in mat.index]]  # square: predicted chars that are also reference chars
    fig, ax = plt.subplots(figsize=(7, 6.2), dpi=200, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    im = ax.imshow(mat.values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(mat.columns)), mat.columns, fontsize=9, color=INK)
    ax.set_yticks(range(len(mat.index)), mat.index, fontsize=9, color=INK)
    ax.set_xlabel("predicted", color=INK)
    ax.set_ylabel("reference", color=INK)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046)
    cb.outline.set_visible(False)
    cb.set_label("share of the reference character", color=INK_MUTED)
    ax.set_title(
        f"Character confusion matrix (characters with ≥ {min_count} occurrences)",
        loc="left",
        fontsize=11,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pred", default=str(ROOT / "results/predictions/qwen25vl-7b_finetuned_gold.csv")
    )
    ap.add_argument(
        "--baseline", default=str(ROOT / "results/predictions/qwen3vl-2b_zeroshot_gold.csv")
    )
    ap.add_argument("--out", default=str(ROOT / "results/analysis/qwen25vl-7b_finetuned"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load(Path(args.pred))
    m = E.compute_metrics(df["gt"].tolist(), df["pred"].tolist())
    ci_line = E.bootstrap_cer(m["char_dist"], m["char_len"], seed=args.seed)
    ci_insc = E.bootstrap_cer(
        m["char_dist"], m["char_len"], groups=df["inscription"], seed=args.seed
    )
    ea = E.error_analysis(df["gt"].tolist(), df["pred"].tolist(), topk=15)

    rows = [subset_row(df, "all", args.seed)]
    rows += [subset_row(df[df["segment"] == s], s, args.seed) for s in ("full line", "word crop")]
    for lo, hi in LEN_BINS:
        sub = df[(df["n_chars"] >= lo) & (df["n_chars"] <= hi)]
        if len(sub):
            rows.append(
                subset_row(sub, f"{lo}–{hi} chars" if hi < 99 else f"{lo}+ chars", args.seed)
            )
    tab = pd.DataFrame(rows)
    tab.round(2).to_csv(out / "cer_by_subset.csv", index=False)

    pred_counts = df["pred"].map(E.norm_cer).value_counts()
    fallback = pred_counts[pred_counts >= 3]

    metrics = {
        "predictions": str(Path(args.pred).relative_to(ROOT))
        if Path(args.pred).is_relative_to(ROOT)
        else args.pred,
        **{k: round(100 * m[k], 2) for k in ("CER", "WER", "NED", "SeqAcc")},
        "n_lines": int(m["n"]),
        "n_inscriptions": int(df["inscription"].nunique()),
        "ci95_line_bootstrap": [
            round(100 * ci_line["ci_low"], 1),
            round(100 * ci_line["ci_high"], 1),
        ],
        "ci95_inscription_bootstrap": [
            round(100 * ci_insc["ci_low"], 1),
            round(100 * ci_insc["ci_high"], 1),
        ],
        "errors": {k: ea[k] for k in ("S", "D", "I")},
        "error_shares": {k: round(ea[k], 3) for k in ("S_frac", "D_frac", "I_frac")},
        "top_substitutions": [{"ref": r, "pred": p, "count": n} for (r, p), n in ea["top_subs"]],
        "repeated_predictions": fallback.to_dict(),
    }

    base_txt = ""
    if args.baseline and Path(args.baseline).exists():
        b = load(Path(args.baseline)).set_index("file").loc[df["file"]].reset_index()
        mb = E.compute_metrics(b["gt"].tolist(), b["pred"].tolist())
        delta = E.paired_bootstrap_delta(
            m["char_dist"],
            m["char_len"],
            mb["char_dist"],
            mb["char_len"],
            groups=df["inscription"],
            seed=args.seed,
        )
        metrics["vs_baseline"] = {
            "baseline": Path(args.baseline).name,
            "baseline_CER": round(100 * mb["CER"], 2),
            "delta_CER": round(100 * delta["delta"], 2),
            "ci95": [round(100 * delta["ci_low"], 1), round(100 * delta["ci_high"], 1)],
            "significant": delta["significant"],
        }
        v = metrics["vs_baseline"]
        base_txt = (
            f"\n## Versus `{v['baseline']}`\n\nCER {metrics['CER']} % vs. {v['baseline_CER']} %: "
            f"Δ = {v['delta_CER']} pp (95 % CI {v['ci95'][0]} … {v['ci95'][1]}, inscription-level paired bootstrap), "
            f"{'significant' if v['significant'] else 'not significant'}.\n"
        )

    (out / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(metrics["top_substitutions"]).to_csv(out / "top_substitutions.csv", index=False)
    E.confusion_frame(ea["confusion"]).round(3).to_csv(out / "confusion_matrix.csv")
    fig_subsets(tab, out / "cer_by_subset.png")
    fig_top_subs(ea["top_subs"], out / "top_substitutions.png")
    fig_confusion(ea["confusion"], out / "confusion_matrix.png")

    subs = ", ".join(f"`{r}→{p}` ({n})" for (r, p), n in ea["top_subs"][:8])
    fb = ", ".join(f"`{k}` ×{v}" for k, v in fallback.items()) or "none"
    report = f"""# Error analysis — `{Path(args.pred).name}`

Generated by `scripts/analyze_predictions.py` (CPU, from saved predictions).

| | value |
|---|---|
| CER / WER / NED / SeqAcc | {metrics["CER"]} / {metrics["WER"]} / {metrics["NED"]} / {metrics["SeqAcc"]} % |
| Lines / inscriptions | {metrics["n_lines"]} / {metrics["n_inscriptions"]} |
| 95 % CI, line bootstrap (thesis protocol) | {ci_line["ci_low"] * 100:.1f} – {ci_line["ci_high"] * 100:.1f} % |
| **95 % CI, inscription bootstrap** | **{ci_insc["ci_low"] * 100:.1f} – {ci_insc["ci_high"] * 100:.1f} %** |
| Substitutions / deletions / insertions | {ea["S"]} / {ea["D"]} / {ea["I"]} ({ea["S_frac"]:.0%} / {ea["D_frac"]:.0%} / {ea["I_frac"]:.0%}) |

The gold set's {metrics["n_lines"]} lines come from only {metrics["n_inscriptions"]} inscriptions, and many lines are
overlapping crops of one photo. Resampling whole inscriptions gives the honest uncertainty.

**Top substitutions (reference→predicted):** {subs}

**Predictions repeated ≥ 3 times (fallback to frequent words):** {fb}
{base_txt}
## CER by segment and length

{tab.round(1).to_markdown(index=False)}

![CER by subset](cer_by_subset.png)

![Top substitutions](top_substitutions.png)

![Confusion matrix](confusion_matrix.png)
"""
    (out / "REPORT.md").write_text(report, encoding="utf-8")
    print(f"saved {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")
    print(
        json.dumps(
            {
                k: metrics[k]
                for k in ("CER", "ci95_line_bootstrap", "ci95_inscription_bootstrap", "errors")
            }
        )
    )


if __name__ == "__main__":
    main()
