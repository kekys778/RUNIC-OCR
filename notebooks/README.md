# Notebooks

The notebooks tell the research story in order: **collect → synthesize → train & evaluate**.
Each one opens with a header card (purpose, runtime, inputs, outputs, status) and keeps the outputs of
its real run. Long logs are truncated, and pip/progress-bar noise and personal account data are removed.

**Status legend:** **final** = produced numbers or data used in the thesis · *prototype / archived* =
an earlier step kept for transparency · *post-defense* = ongoing follow-up work.

## 1 · Data collection

| # | Notebook | What it does | Runtime | Status |
|---|---|---|---|---|
| 01 | [`01_runes_db_exploration`](01_data_collection/01_runes_db_exploration.ipynb) | Explores the RuneS dump: 2,940 finds have both a transliteration and a photo | CPU | exploratory |
| 02 | [`02_runer_ku_parser`](01_data_collection/02_runer_ku_parser.ipynb) | Scrapes Danske Runeindskrifter (runer.ku.dk) into a CSV + photos | CPU, internet | **final** |
| 03 | [`03_render_prototype_gemini`](01_data_collection/03_render_prototype_gemini.ipynb) | First rendering prototype; Gemini background swap (blocked by quota) | CPU + API | archived |
| 04 | [`04_gold_image_normalizer`](01_data_collection/04_gold_image_normalizer.ipynb) | OpenCV quality gate + CLAHE / deskew / resize for gold-set photos | CPU | **final** |
| 05 | [`05_llm_ocr_baseline_groq`](01_data_collection/05_llm_ocr_baseline_groq.ipynb) | Zero-shot Llama vision OCR via Groq: 0 % exact match | CPU + API | archived baseline |

## 2 · Synthesis

| # | Notebook | What it does | Runtime | Status |
|---|---|---|---|---|
| 01 | [`01_controlnet_modalities_comparison`](02_synthesis/01_controlnet_modalities_comparison.ipynb) | Six ControlNet modalities on one phrase; Canny chosen | Colab GPU | **final** (thesis figure) |
| 02 | [`02_synth_sd3_canny_colab_prototype`](02_synthesis/02_synth_sd3_canny_colab_prototype.ipynb) | SD3 + ControlNet Canny pipeline, Colab/Drive version | Colab GPU | prototype |
| 03 | [`03_synth_sd3_canny_final`](02_synthesis/03_synth_sd3_canny_final.ipynb) | **The thesis training corpus** (4,647 images) | Kaggle T4 | **final** |
| 04 | [`04_synth_sd3_depth_hf`](02_synthesis/04_synth_sd3_depth_hf.ipynb) | SD3 + Depth with groove depth maps, streamed to the HF Hub | Kaggle T4 | post-defense |

## 3 · Training & evaluation

| # | Notebook | What it does | Runtime | Status |
|---|---|---|---|---|
| 01 | [`01_qwen2vl_7b_qlora_baseline`](03_training_eval/01_qwen2vl_7b_qlora_baseline.ipynb) | First QLoRA run; documents a label-masking bug found later | Kaggle T4 | archived |
| 02 | [`02_experiments_trocr_qwen`](03_training_eval/02_experiments_trocr_qwen.ipynb) | **Unified harness**: TrOCR vs. Qwen-VL, all thesis metrics | Kaggle T4 | **final** |
| 03 | [`03_qwen_vl_train_hf_shards`](03_training_eval/03_qwen_vl_train_hf_shards.ipynb) | Qwen-VL on the extended (Canny + Depth) corpus | Kaggle 2×T4 | post-defense |

## Running

* GPU notebooks were written for **Kaggle** (`/kaggle/input`, `/kaggle/working`) or **Colab** (Google Drive).
  Paths are collected in a config cell at the top of each notebook.
* Secrets are never stored in the notebooks. Set `HF_TOKEN`, `GROQ_API_KEY` or `GOOGLE_API_KEY` as
  environment variables or Kaggle/Colab secrets.
* The CPU notebooks in `01_data_collection/` resolve data relative to the repo (`../../data/sources`).
* Code style is enforced with `ruff` (see `pyproject.toml`). Run `ruff check notebooks && ruff format notebooks`
  after editing.
