"""One GPU step in its own process: (optionally) fine-tune, then predict.

Usage (normally via :func:`runic_kaggle.run_isolated`)::

    python -m runic_step spec.json

``spec.json`` keys:

* ``model_key``, ``adapter`` — registry key and LoRA adapter folder;
* ``train_items`` — ``[{"image": path, "answer": text}, ...]``; empty → predict only;
* ``test_paths`` — images to transliterate after training;
* ``epochs``, ``lr``, ``grad_accum``, ``seed``, ``n_best``, ``trainer_dir``;
* ``dry_run`` + ``dry_gt`` (``{path: gt}``) — CPU smoke test without a model: the
  "prediction" is the reference, truncated with a probability that falls as more
  lines are "trained", so every downstream table and chart can be checked;
* ``out`` — where the result JSON ``{"recs": [...], "info": {...}}`` is written.

Running each step in a fresh process guarantees that all GPU memory is released
between steps (in one long-lived kernel, repeated 4-bit load/train cycles leak
and end in CUDA OOM).
"""

from __future__ import annotations

import json
import sys
import time


def _dry_run(spec: dict) -> dict:
    import random

    rng = random.Random(spec.get("seed", 0) + len(spec.get("train_items", [])))
    keep = min(0.95, 0.45 + 0.005 * len(spec.get("train_items", [])))
    recs = []
    for p in spec["test_paths"]:
        gt = spec["dry_gt"][p]
        pred = gt if rng.random() < keep else gt[: max(1, len(gt) // 2)]
        rec = {"pred": pred}
        if spec.get("n_best", 1) > 1:
            rec["nbest"], rec["scores"] = [pred, gt], [-0.1, -0.4]
        recs.append(rec)
    return {
        "recs": recs,
        "info": {
            "train_loss": 0.5 if spec.get("train_items") else None,
            "steps": 0,
            "dry_run": True,
        },
    }


def run(spec: dict) -> dict:
    if spec.get("dry_run"):
        return _dry_run(spec)
    from runic_vlm import finetune, load_qwen, predict

    t0 = time.time()
    train = spec.get("train_items") or []
    model, proc, cfg = load_qwen(spec["model_key"], spec.get("adapter"), trainable=bool(train))
    info: dict = {"train_lines": len(train)}
    if train:
        info.update(
            finetune(
                model,
                proc,
                cfg,
                train,
                out_dir=spec.get("trainer_dir", "/tmp/runic_trainer"),
                epochs=spec.get("epochs", 5),
                lr=spec.get("lr", 5e-5),
                grad_accum=spec.get("grad_accum", 4),
                seed=spec.get("seed", 0),
            )
        )
    recs = predict(model, proc, cfg, spec["test_paths"], n_best=spec.get("n_best", 1), log_every=50)
    info["minutes"] = round((time.time() - t0) / 60, 1)
    try:
        import torch

        info["max_gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    except Exception:
        pass
    return {"recs": recs, "info": info}


def main(argv: list[str]) -> int:
    spec = json.loads(open(argv[1], encoding="utf-8").read())
    result = run(spec)
    with open(spec["out"], "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
