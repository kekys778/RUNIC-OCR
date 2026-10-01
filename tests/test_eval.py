"""Tests for the torch-free evaluation module (runic_eval)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import runic_eval as E

ROOT = Path(__file__).resolve().parents[1]


def test_reproduces_thesis_numbers():
    """Same protocol as the thesis harness: Qwen2.5-VL-7B row of table 5.4."""
    d = pd.read_csv(ROOT / "results/predictions/qwen25vl-7b_finetuned_gold.csv")
    m = E.compute_metrics(d["gt"].astype(str).tolist(), d["pred"].fillna("").astype(str).tolist())
    assert round(100 * m["CER"], 2) == 54.27
    assert round(100 * m["WER"], 2) == 81.18
    assert round(100 * m["SeqAcc"], 2) == 8.85
    ci = E.bootstrap_cer(m["char_dist"], m["char_len"], B=2000, seed=0)
    assert (round(100 * ci["ci_low"], 1), round(100 * ci["ci_high"], 1)) == (48.0, 61.3)


def test_norm_strips_spaces_and_dividers():
    assert E.norm_cer(" kuþ mutr:karþi·kuml+ ") == "kuþmutrkarþikuml"
    assert E.norm_cer(None) == ""


def test_metrics_perfect_and_empty():
    m = E.compute_metrics(["alu", "ek"], ["alu", "ek"])
    assert m["CER"] == 0 and m["SeqAcc"] == 1
    m = E.compute_metrics(["alu"], [""])
    assert m["CER"] == 1


def test_inscription_id():
    assert E.inscription_id("5810_0_kuml-þusi.png") == "5810"
    assert E.inscription_id("data/gold_set/93_48_1str.png") == "93"


def test_cluster_bootstrap_is_wider_for_correlated_lines():
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(10), 10)
    rate = np.repeat(rng.uniform(0.1, 0.9, 10), 10)  # error rate shared within a group
    dist, ln = (rate * 10).round(), np.full(100, 10)
    line = E.bootstrap_cer(dist, ln, B=500)
    clus = E.bootstrap_cer(dist, ln, groups=groups, B=500)
    assert clus["ci_high"] - clus["ci_low"] > line["ci_high"] - line["ci_low"]
    assert clus["n_units"] == 10


def test_error_analysis_counts():
    ea = E.error_analysis(["alu", "kuml"], ["alk", "kml"])
    assert (ea["S"], ea["D"], ea["I"]) == (1, 1, 0)
    assert ea["top_subs"][0] == (("u", "k"), 1)
    assert ea["confusion"]["a"]["a"] == 1


@pytest.mark.parametrize("k", [2, 3, 5])
def test_grouped_kfold_never_splits_a_group(k):
    groups = np.array(["a"] * 21 + ["b"] * 16 + ["c"] * 11 + list("defghijklm"))
    folds = E.grouped_kfold(groups, k=k, seed=0)
    seen = np.concatenate([te for _, te in folds])
    assert sorted(seen) == list(range(len(groups)))  # every line tested exactly once
    for tr, te in folds:
        assert not set(groups[tr]) & set(groups[te])


def test_grouped_kfold_rejects_bad_k():
    with pytest.raises(ValueError):
        E.grouped_kfold(["a", "b"], k=3)


def test_nested_subsets_are_nested_and_spread():
    groups = np.array(["a"] * 10 + ["b"] * 10 + ["c"] * 10)
    idx = np.arange(30)
    subs = E.nested_subsets(idx, [3, 9, -1], groups=groups, seed=0)
    assert set(subs[3]) <= set(subs[9]) <= set(subs[-1])
    assert len(set(groups[subs[3]])) == 3  # first draws cover all inscriptions
    assert len(subs[-1]) == 30
