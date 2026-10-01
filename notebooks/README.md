# Notebooks

The notebooks tell the research story in order: **collect → synthesize → train & evaluate**.
Each one opens with a header card (purpose, runtime, inputs, outputs, status) and keeps the outputs of
its real run. Long logs are truncated, and pip/progress-bar noise and personal account data are removed.

**Status legend:** **final** = produced data, figures or numbers used in the thesis ·
*post-defense* = ongoing follow-up work.

## 1 · Data collection

| # | Notebook | What it does | Runtime | Status |
|---|---|---|---|---|
| 01 | [`01_runer_ku_parser`](01_data_collection/01_runer_ku_parser.ipynb) | Scrapes Danske Runeindskrifter (runer.ku.dk) into a CSV + photos | CPU, internet | **final** |
| 02 | [`02_gold_image_normalizer`](01_data_collection/02_gold_image_normalizer.ipynb) | OpenCV quality gate + CLAHE / deskew / resize for gold-set photos | CPU | **final** |

The RuneS database is scraped by [`src/parser_runes.py`](../src/parser_runes.py).

## 2 · Synthesis

| # | Notebook | What it does | Runtime | Status |
|---|---|---|---|---|
| 01 | [`01_controlnet_modalities_comparison`](02_synthesis/01_controlnet_modalities_comparison.ipynb) | Six ControlNet modalities on one phrase; Canny chosen | Colab GPU | **final** (thesis figure) |
| 02 | [`02_synth_sd3_canny_final`](02_synthesis/02_synth_sd3_canny_final.ipynb) | **The thesis training corpus** (4,647 images) | Kaggle T4 | **final** |
| 03 | [`03_synth_sd3_depth_hf`](02_synthesis/03_synth_sd3_depth_hf.ipynb) | SD3 + Depth with groove depth maps, streamed to the HF Hub | Kaggle T4 | post-defense |

## 3 · Training & evaluation

| # | Notebook | What it does | Runtime | Status |
|---|---|---|---|---|
| 01 | [`01_experiments_trocr_qwen`](03_training_eval/01_experiments_trocr_qwen.ipynb) | **Unified harness**: TrOCR vs. Qwen-VL, all thesis metrics | Kaggle T4 | **final** |
| 02 | [`02_qwen_vl_train_hf_shards`](03_training_eval/02_qwen_vl_train_hf_shards.ipynb) | Qwen-VL on the extended (Canny + Depth) corpus | Kaggle 2×T4 | post-defense |

## 4 · Further work (post-thesis, fire-and-forget on Kaggle)

| # | Notebook | What it does | Runtime | Status |
|---|---|---|---|---|
| E1 | [`E1_reeval_published_adapters`](04_further_work/E1_reeval_published_adapters.ipynb) | Re-evaluate the public adapters: reproduction check, inscription-level CIs, significance, 5-best | Kaggle T4, 0.4 h | **done**: thesis reproduced exactly |
| E2 | [`E2_fewshot_real_cv`](04_further_work/E2_fewshot_real_cv.ipynb) | Few-shot fine-tuning on real photos, grouped 5-fold CV, learning curve | Kaggle T4, ≈ 1.5–2.5 h | ready |

How to run them and what the CPU error analysis (E0) found: [`04_further_work/README.md`](04_further_work/README.md).

Early prototypes and exploratory notebooks (Gemini/Groq baselines, a Colab version of the synthesis
pipeline, a first Qwen2-VL run) were removed from the tree; they remain in the git history.

## Running

* GPU notebooks were written for **Kaggle** (`/kaggle/input`, `/kaggle/working`) or **Colab** (Google Drive).
  Paths are collected in a config cell at the top of each notebook.
* Secrets are never stored in the notebooks. Set `HF_TOKEN` as an environment variable or a Kaggle/Colab secret.
* The notebooks in `01_data_collection/` run on CPU; their input/output folders are set in the first code cell.
* Code style is enforced with `ruff` (see `pyproject.toml`). Run `ruff check notebooks && ruff format notebooks`
  after editing.
