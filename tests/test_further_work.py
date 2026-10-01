"""CPU tests for the further-work helpers (no GPU, no network)."""

import json

import numpy as np
import pandas as pd
import pytest

import runic_kaggle as K
import runic_reports as R


def test_load_gold_from_repo():
    g = K.load_gold()
    # the repo copy lacks 5810_0_karþi-kuml-þusi-eftiR-urmar.png (present in the Kaggle dataset)
    assert len(g) == 112 and g["inscription"].nunique() == 28
    assert set(g["segment"]) == {"full line", "word crop"}


def test_load_gold_prefers_extra_roots(tmp_path):
    from PIL import Image

    Image.new("RGB", (4, 4)).save(tmp_path / "5810_0_karþi-kuml-þusi-eftiR-urmar.png")
    g = K.load_gold(image_roots=[tmp_path])
    assert len(g) == 113


@pytest.mark.parametrize(
    ("base", "want"),
    [("Qwen/Qwen2.5-VL-7B-Instruct", "qwen25"), ("qwen3-vl-8b-instruct", "qwen3_8b")],
)
def test_find_adapter_matches_base_model(tmp_path, base, want):
    for name, b in [
        ("qwen25", "Qwen/Qwen2.5-VL-7B-Instruct"),
        ("qwen3_8b", "Qwen/Qwen3-VL-8B-Instruct"),
    ]:
        (tmp_path / name).mkdir()
        (tmp_path / name / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": b})
        )
    assert K.find_adapter(tmp_path, base).name == want


def test_find_adapter_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        K.find_adapter(tmp_path, "Qwen/Qwen3-VL-2B-Instruct")


def test_result_sink_resume(tmp_path):
    s = K.ResultSink("E9_test", tmp_path)
    assert not s.is_done("a")
    s.mark_done("a", CER=1.0)
    assert K.ResultSink("E9_test", tmp_path).is_done("a")  # survives a restart


def test_get_secret_from_env(monkeypatch):
    monkeypatch.setenv("RUNIC_TEST_SECRET", "x")
    assert K.get_secret("RUNIC_TEST_SECRET") == "x"


def _fake_preds(err_rate, seed=0):
    g = K.load_gold()
    rng = np.random.default_rng(seed)
    pred = [gt if rng.random() > err_rate else gt[: len(gt) // 2] for gt in g["gt"]]
    return pd.DataFrame(
        {"file": g["file_name"], "gt": g["gt"], "pred": pred, "inscription": g["inscription"]}
    )


def test_models_table_and_compare():
    t = R.models_table({"qwen25vl-7b": _fake_preds(0.2), "qwen3vl-2b": _fake_preds(0.8)})
    assert list(t["model"]) == ["qwen25vl-7b", "qwen3vl-2b"]
    c = R.compare(_fake_preds(0.1), _fake_preds(0.9))
    assert c["delta"] < 0 and c["significant"]


def test_learning_curve_table_and_plot(tmp_path):
    by_n = {}
    for n, rate in [(0, 0.9), (10, 0.6), (-1, 0.2)]:
        df = _fake_preds(rate, seed=abs(n))
        df["n_train"] = 0 if n == 0 else (n if n > 0 else 90)
        by_n[n] = df
    t = R.learning_curve_table(by_n)
    assert list(t["N_real_lines"]) == [0, 10, "all"]
    assert t["CER"].iloc[-1] < t["CER"].iloc[0]
    R.plot_learning_curve(t, tmp_path / "lc.png", "test")
    assert (tmp_path / "lc.png").stat().st_size > 10_000
    assert R.df_to_md(t).count("\n") == len(t) + 1


def test_predictions_frame_with_nbest():
    g = K.load_gold().head(2)
    recs = [{"pred": "a", "nbest": ["a", "b"], "scores": [-0.1, -0.5]}] * 2
    df = R.predictions_frame(g, recs, fold=3)
    assert list(df.columns) == [
        "file",
        "gt",
        "pred",
        "inscription",
        "segment",
        "nbest",
        "nbest_scores",
        "fold",
    ]
    assert (df["fold"] == 3).all()


def test_warmup_steps_replaces_ratio():
    from runic_vlm import warmup_steps

    assert (
        warmup_steps(10, 4, 10) == 3
    )  # ceil(10/4)=3 steps/epoch × 10 epochs → 30 steps → 3 warm-up
    assert warmup_steps(90, 4, 5) == 12
    assert warmup_steps(1, 4, 1) == 1


def test_run_isolated_dry_run(tmp_path):
    g = K.load_gold().head(4)
    spec = {
        "dry_run": True,
        "train_items": [{"image": p, "answer": t} for p, t in zip(g["path"][:2], g["gt"][:2])],
        "test_paths": list(g["path"]),
        "dry_gt": dict(zip(g["path"], g["gt"])),
        "n_best": 2,
    }
    res = K.run_isolated(spec, tmp_path)
    assert len(res["recs"]) == 4 and "nbest" in res["recs"][0]
    assert not list(tmp_path.glob("*.json"))  # spec/result files are cleaned up


def test_run_isolated_reports_failure(tmp_path):
    with pytest.raises(RuntimeError, match="step failed"):
        K.run_isolated({"dry_run": True, "test_paths": ["missing"], "dry_gt": {}}, tmp_path)


def test_result_sink_seeds_from_previous_output(tmp_path):
    old = K.ResultSink("E9_test", tmp_path / "input" / "prev-version" / "results")
    old.save_csv("pred_N0.csv", pd.DataFrame({"a": [1]}))
    old.mark_done("N0")
    new = K.ResultSink("E9_test", tmp_path / "working", seed_from=[tmp_path / "input"])
    assert new.is_done("N0") and new.path("pred_N0.csv").exists()
