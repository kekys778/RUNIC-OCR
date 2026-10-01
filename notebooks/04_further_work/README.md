# Further work — fire-and-forget experiments

Post-thesis experiments, ordered by expected payoff. Each GPU notebook is a thin runner: the code lives in `src/`
(and is unit-tested in CI), the notebook clones this repo, checks the environment, runs, and writes everything to
`results/<experiment>/` with a `REPORT.md`.

| Step | Where | What it answers | Runtime | Status |
|---|---|---|---|---|
| **E0** | CPU, [`scripts/analyze_predictions.py`](../../scripts/analyze_predictions.py) | Error typology, confusion matrix, honest (inscription-level) CIs | seconds | done → [`results/analysis/`](../../results/analysis/qwen25vl-7b_finetuned/REPORT.md) |
| **E1** | Kaggle GPU, [`E1_reeval_published_adapters`](E1_reeval_published_adapters.ipynb) | Do the public adapters reproduce the thesis? Are the model differences significant? 5-best hypotheses for decoding work | 0.4 h | done → [`REPORT`](../../results/further_work/E1_reeval/REPORT.md) |
| **E2** | Kaggle GPU, [`E2_fewshot_real_cv`](E2_fewshot_real_cv.ipynb) | How much does fine-tuning on N real lines reduce CER? (grouped 5-fold CV, learning curve) | ≈ 1.5–2.5 h | ready (each step runs in its own process) |
| E3 | planned | Crop + augment synthetic images to look like real photos | — | next |
| E4 | planned | Training-data mix: real words vs. random strings; Canny vs. Depth | — | next |
| E5 | planned (CPU) | Re-rank E1's 5-best with a character LM over Rundata | — | after E1 |

## What E0 found (from the saved thesis predictions)

* The gold set's 113 lines come from only **28 inscriptions**, and many lines are overlapping crops of the same photo.
  Resampling whole inscriptions widens the 95 % CI of the best model from 48.0–61.3 % to **42.0–70.5 %**.
  Every new experiment therefore reports inscription-level CIs and splits folds by inscription.
* **71 % of the errors are deletions** (the output is too short); only 27 % are substitutions between similar runes.
  The model answers `ek` on 18 of 113 lines and `alu` on 7, which is a fallback to frequent words rather than misreading.
* Short lines are hardest (1–4 characters: 67.7 % CER); full Elder Futhark lines score 65.1 % vs. 49.9 % for word crops.

## What E1 found (Kaggle, 2× T4)

* **The thesis reproduces exactly.** The public Qwen2.5-VL-7B adapter scores 55.40 % CER on the same 112 lines, identical
  to the thesis predictions, in a newer environment (transformers 5.0). Qwen3-VL-8B scores 65.2 % and Qwen3-VL-2B 73.8 %.
* **The ranking 7B < 8B < 2B is significant** even with the inscription-level paired bootstrap.
* **The models fail differently.** The 7B model mostly omits characters (deletions), while the Qwen3 models mostly misread them.

## How to run a GPU notebook on Kaggle

1. Kaggle → **Create → New Notebook** → **File → Import Notebook** → upload the `.ipynb` from this folder.
2. Right panel → **Session options**: **Accelerator: GPU T4 x2** (or P100), **Internet: On**.
3. **Add-ons → Secrets → Add secret** `HF_TOKEN`: a Hugging Face token with **write** access. It mirrors results to the
   private dataset `AntoniusPerf/runic-ocr-results` (created automatically) and makes runs resumable. Without it,
   results stay in the notebook's Output only.
4. Optional but recommended: **Add Input → Datasets → `zhopa228/val-dataset`**. The repo copy of the gold set lacks one
   image (`5810_0_karþi-kuml-þusi-eftiR-urmar.png`), so without this dataset 112 instead of 113 lines are evaluated.
5. **Save Version → Save & Run All (Commit) → Save**. Close the tab; Kaggle runs it in the background (up to 12 h).
6. When it finishes: open the version → **Output** → `results/<experiment>/REPORT.md` (or the HF dataset).

**If a run stops early** (time limit, lost connection), run it again the same way. Finished steps are recorded in
`run_log.jsonl` and skipped. They are restored from the HF dataset (with `HF_TOKEN`) or, without a token, from a previous
version's Output that you add as an Input (**Add Input → Your Work → the notebook**).

**Before a long run, check the setup in one minute:** set `DRY_RUN = True` in E2's configuration cell and run all. A fake
model replaces the GPU, so every step, table and chart is exercised on CPU.

E2 runs every (N, fold) step in a separate Python process (`src/runic_step.py`). Loading and training a 4-bit 7B model
repeatedly in one kernel leaked GPU memory and hit CUDA OOM after six steps; a fresh process per step releases everything.
A failed step is logged and skipped instead of stopping the run.

**To bring results back here**, either share the HF dataset path or download the Output folder and drop it into
`results/further_work/` in this repo.

## Configuration

Only the first code cell of each notebook needs editing, and the defaults reproduce the planned experiment:

* `REPO_REF`: the git branch or tag of the code to run (default `main`);
* E1: `MODEL_KEYS`, `N_BEST`;
* E2: `MODEL_KEY`, `K_FOLDS`, `N_TRAIN`, `EPOCHS`, `LR`, `TIME_BUDGET_H`.

## What was tested before shipping

The GPU parts (model loading, generation, training) can't run in CI. Everything around them is checked:
* unit tests for metrics, folds, adapter discovery, resume and reports (`tests/test_eval.py`, `tests/test_further_work.py`);
* a full **dry run of both notebooks on CPU**, with the model replaced by a stub that returns noisy labels. It covers
  every cell, the resume path, all CSV/JSON outputs, figures and `REPORT.md`.

The model code itself (`src/runic_vlm.py`) reuses the thesis harness's prompts, 4-bit config and label masking.
