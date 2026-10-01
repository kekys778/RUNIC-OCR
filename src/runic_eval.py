"""Torch-free evaluation utilities for runic OCR.

The metric definitions are identical to the thesis harness
(:mod:`runic_ocr_experiments`): CER is computed on graphemes with whitespace and
word dividers (``·:+``) removed, WER splits on ``" ·:+/"``. This module adds what
the thesis protocol lacked:

* **inscription-level (cluster) bootstrap** — the gold set has 113 lines but only
  28 inscriptions, and many lines are overlapping crops of the same photo, so
  resampling lines independently understates the uncertainty;
* **grouped k-fold splits** that never put crops of one inscription on both sides;
* an error typology (substitutions / deletions / insertions) and a grapheme
  confusion matrix.

Only ``numpy``, ``pandas`` and ``rapidfuzz`` are required, so everything here runs
on CPU and in CI.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

SEPARATORS = "·:+"
WORD_SEP_CHARS = " ·:+/"
_WS_RX = re.compile(r"\s+")


# ----------------------------------------------------------------------------- text
def norm_cer(s: str, strip_whitespace: bool = True, strip_separators: bool = True) -> str:
    """Normalize a transliteration for CER / NED / SeqAcc (thesis protocol)."""
    s = (s if isinstance(s, str) else "").strip()
    s = _WS_RX.sub("", s) if strip_whitespace else _WS_RX.sub(" ", s)
    if strip_separators:
        for ch in SEPARATORS:
            s = s.replace(ch, "")
    return s


def word_tokens(s: str) -> list[str]:
    """Split a transliteration into words for WER."""
    s = _WS_RX.sub(" ", (s if isinstance(s, str) else "").strip())
    return [t for t in re.split("[" + re.escape(WORD_SEP_CHARS) + "]+", s) if t]


def edit_distance(a, b) -> int:
    """Levenshtein distance between two strings or two lists."""
    if isinstance(a, str) and isinstance(b, str):
        from rapidfuzz.distance import Levenshtein

        return int(Levenshtein.distance(a, b))
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return la or lb
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[lb]


def inscription_id(filename: str) -> str:
    """Inscription key of a gold-set file: ``5810_0_kuml.png`` -> ``5810``."""
    return Path(str(filename)).name.split("_")[0]


# ----------------------------------------------------------------------------- metrics
def compute_metrics(refs: list[str], preds: list[str]) -> dict:
    """CER, WER, NED, SeqAcc plus per-line arrays (for bootstrapping)."""
    refs_c = [norm_cer(r) for r in refs]
    preds_c = [norm_cer(p) for p in preds]
    char_dist, char_len, ned, seq_ok, word_dist, word_len = [], [], [], [], [], []
    for r, p in zip(refs_c, preds_c):
        d = edit_distance(p, r)
        char_dist.append(d)
        char_len.append(max(len(r), 1))
        ned.append(d / max(len(r), 1) if r else float(bool(p)))
        seq_ok.append(float(p == r))
    for r, p in zip(refs, preds):
        rw, pw = word_tokens(r), word_tokens(p)
        word_dist.append(edit_distance(pw, rw))
        word_len.append(max(len(rw), 1))
    char_dist, char_len = np.asarray(char_dist), np.asarray(char_len)
    return {
        "CER": float(char_dist.sum() / max(char_len.sum(), 1)),
        "WER": float(np.sum(word_dist) / max(np.sum(word_len), 1)),
        "NED": float(np.mean(ned)) if ned else 0.0,
        "SeqAcc": float(np.mean(seq_ok)) if seq_ok else 0.0,
        "n": len(refs),
        "char_dist": char_dist,
        "char_len": char_len,
    }


def _resample_index(groups: np.ndarray | None, n: int, rng: np.random.Generator) -> np.ndarray:
    if groups is None:
        return rng.integers(0, n, n)
    uniq = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in uniq}
    picked = rng.choice(uniq, size=len(uniq), replace=True)
    return np.concatenate([members[g] for g in picked])


def bootstrap_cer(
    char_dist, char_len, groups=None, B: int = 2000, seed: int = 0, alpha: float = 0.05
) -> dict:
    """Bootstrap CI of corpus CER.

    With ``groups`` (e.g. inscription ids) whole groups are resampled
    (cluster bootstrap); without, lines are resampled as in the thesis.
    """
    d, ln = np.asarray(char_dist, float), np.asarray(char_len, float)
    g = None if groups is None else np.asarray(groups)
    rng = np.random.default_rng(seed)
    vals = np.empty(B)
    for k in range(B):
        idx = _resample_index(g, len(d), rng)
        vals[k] = d[idx].sum() / max(ln[idx].sum(), 1)
    return {
        "CER": float(d.sum() / max(ln.sum(), 1)),
        "ci_low": float(np.percentile(vals, 100 * alpha / 2)),
        "ci_high": float(np.percentile(vals, 100 * (1 - alpha / 2))),
        "unit": "line" if g is None else "group",
        "n_units": int(len(d) if g is None else len(np.unique(g))),
    }


def paired_bootstrap_delta(
    dist_a, len_a, dist_b, len_b, groups=None, B: int = 2000, seed: int = 0
) -> dict:
    """CER(A) − CER(B) on the same lines; significant if the 95 % CI excludes 0."""
    da, la = np.asarray(dist_a, float), np.asarray(len_a, float)
    db, lb = np.asarray(dist_b, float), np.asarray(len_b, float)
    g = None if groups is None else np.asarray(groups)
    rng = np.random.default_rng(seed)
    deltas = np.empty(B)
    for k in range(B):
        idx = _resample_index(g, len(da), rng)
        deltas[k] = da[idx].sum() / max(la[idx].sum(), 1) - db[idx].sum() / max(lb[idx].sum(), 1)
    lo, hi = float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))
    return {
        "delta": float(da.sum() / max(la.sum(), 1) - db.sum() / max(lb.sum(), 1)),
        "ci_low": lo,
        "ci_high": hi,
        "significant": bool(lo > 0 or hi < 0),
    }


# ----------------------------------------------------------------------------- errors
def error_analysis(refs: list[str], preds: list[str], topk: int = 15) -> dict:
    """Substitution / deletion / insertion counts, top confusions, confusion matrix.

    ``top_subs`` holds ``((reference_char, predicted_char), count)`` pairs.
    ``confusion[ref][pred]`` counts aligned characters (diagonal = correct).
    """
    from rapidfuzz.distance import Levenshtein

    S = D = I = 0
    subs: Counter = Counter()
    conf: dict = defaultdict(Counter)
    for r, p in zip(map(norm_cer, refs), map(norm_cer, preds)):
        for op in Levenshtein.editops(r, p):
            if op.tag == "replace":
                S += 1
                subs[(r[op.src_pos], p[op.dest_pos])] += 1
            elif op.tag == "delete":
                D += 1
            elif op.tag == "insert":
                I += 1
        for blk in Levenshtein.opcodes(r, p):
            if blk.tag == "equal":
                for k in range(blk.src_end - blk.src_start):
                    c = r[blk.src_start + k]
                    conf[c][c] += 1
            elif blk.tag == "replace":
                n = min(blk.src_end - blk.src_start, blk.dest_end - blk.dest_start)
                for k in range(n):
                    conf[r[blk.src_start + k]][p[blk.dest_start + k]] += 1
    tot = max(S + D + I, 1)
    return {
        "S": S,
        "D": D,
        "I": I,
        "S_frac": S / tot,
        "D_frac": D / tot,
        "I_frac": I / tot,
        "top_subs": subs.most_common(topk),
        "confusion": {k: dict(v) for k, v in conf.items()},
    }


def confusion_frame(confusion: dict, min_count: int = 1) -> pd.DataFrame:
    """Row-normalized confusion matrix (rows = reference char) as a DataFrame."""
    rows = sorted(c for c, v in confusion.items() if sum(v.values()) >= min_count)
    cols = sorted({c for v in confusion.values() for c in v} | set(rows))
    mat = pd.DataFrame(0.0, index=rows, columns=cols)
    for r in rows:
        for p, n in confusion[r].items():
            mat.loc[r, p] = n
    return mat.div(mat.sum(axis=1).replace(0, 1), axis=0)


# ----------------------------------------------------------------------------- splits
def grouped_kfold(groups, k: int = 5, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """K folds where every group (inscription) lands in exactly one test fold.

    Groups are shuffled, then assigned largest-first to the fold with the fewest
    lines, which keeps fold sizes balanced even with a few very large groups.
    """
    groups = np.asarray(groups)
    uniq, counts = np.unique(groups, return_counts=True)
    if k < 2 or k > len(uniq):
        raise ValueError(f"k must be in [2, {len(uniq)}], got {k}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    order = order[np.argsort(-counts[order], kind="stable")]
    load = np.zeros(k, int)
    fold_of = {}
    for i in order:
        f = int(np.argmin(load))
        fold_of[uniq[i]] = f
        load[f] += counts[i]
    fold = np.array([fold_of[g] for g in groups])
    return [(np.flatnonzero(fold != f), np.flatnonzero(fold == f)) for f in range(k)]


def nested_subsets(train_idx, sizes, groups=None, seed: int = 0) -> dict[int, np.ndarray]:
    """Nested training subsets (N1 ⊂ N2 ⊂ …) for learning curves.

    Lines are drawn inscription by inscription (all crops of a drawn inscription
    in random order) so small N covers as many inscriptions as possible first.
    Sizes larger than the pool are clipped; ``-1`` means "all".
    """
    train_idx = np.asarray(train_idx)
    rng = np.random.default_rng(seed)
    if groups is None:
        ordered = rng.permutation(train_idx)
    else:
        g = np.asarray(groups)[train_idx]
        buckets = {u: list(rng.permutation(train_idx[g == u])) for u in np.unique(g)}
        keys = list(rng.permutation(list(buckets)))
        ordered = []
        while any(buckets[u] for u in keys):  # round-robin across inscriptions
            for u in keys:
                if buckets[u]:
                    ordered.append(buckets[u].pop())
        ordered = np.asarray(ordered)
    out = {}
    for n in sizes:
        m = len(ordered) if n < 0 else min(n, len(ordered))
        out[int(n)] = ordered[:m]
    return out
