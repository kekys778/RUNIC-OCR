# E1 · Re-evaluation of the published adapters
Run finished 2026-10-01 15:22 · 112 gold lines / 28 inscriptions ·
code `main` · GPU ['Tesla T4', 'Tesla T4'] · 0.4 h.

## Models
`thesis_CER_same_lines` is the thesis CER recomputed on exactly these lines. A gap of more than ~1 pp means
the environment does not reproduce the thesis run.

| model | lines | CER | thesis_CER_same_lines | CI95_line | CI95_inscription | WER | SeqAcc |
|---|---|---|---|---|---|---|---|
| qwen25vl-7b | 112 | 55.4 | 55.4 | 49.0–62.2 | 43.7–70.5 | 82.42 | 8.93 |
| qwen3vl-8b | 112 | 65.23 |  | 59.5–71.1 | 54.0–78.7 | 94.55 | 4.46 |
| qwen3vl-2b | 112 | 73.76 |  | 70.0–77.7 | 68.5–81.1 | 97.58 | 1.79 |

## Pairwise differences (paired bootstrap over inscriptions)

| A | B | delta_CER_A_minus_B | CI95 | significant |
|---|---|---|---|---|
| qwen25vl-7b | qwen3vl-8b | -9.83 | -14.3…-3.9 | True |
| qwen25vl-7b | qwen3vl-2b | -18.36 | -26.2…-6.9 | True |
| qwen3vl-8b | qwen3vl-2b | -8.53 | -15.2…-0.5 | True |

Per-model error analysis: `analysis_<model>/REPORT.md`. Beam hypotheses: column `nbest` in `<model>_gold_predictions.csv`.

## Interpretation (Kaggle run, 2026-10-01, 2× T4, 0.4 h)

| model | CER, % | thesis CER, % | 95 % CI (inscriptions) | substitutions / deletions / insertions |
|---|---:|---:|---:|---|
| **Qwen2.5-VL-7B** | **55.40** | 55.40 on the same 112 lines (54.27 on 113) | 43.7–70.5 | 137 / 362 / 14 |
| Qwen3-VL-8B | 65.23 | 67.86 (113 lines) | 54.0–78.7 | 328 / 232 / 44 |
| Qwen3-VL-2B | 73.76 | 73.55 (113 lines) | 68.5–81.1 | 334 / 322 / 27 |

* **The thesis reproduces.** The public Qwen2.5-VL-7B adapter gives exactly the thesis CER on the same lines, in a newer
  environment (transformers 5.0, peft 0.19). The Qwen3 models are within about 2.6 pp of the thesis, which also used
  one more gold line.
* **The model ranking is real.** All three pairwise differences stay significant under the stricter inscription-level
  paired bootstrap: 7B vs. 8B −9.8 pp (CI −14.3…−3.9), 7B vs. 2B −18.4 pp, 8B vs. 2B −8.5 pp.
* **The models fail differently.** Qwen2.5-VL-7B mostly *omits* characters (71 % of its errors are deletions), while both
  Qwen3-VL models mostly *misread* them (49–54 % substitutions) and insert more. The best model is conservative: it
  writes less rather than guessing.
* Only 112 of 113 lines were evaluated because `zhopa228/val-dataset` was not attached. Each model's 5-best beam
  hypotheses are in the run's Output (`results/E1_reeval/<model>_gold_predictions.csv`, column `nbest`).
