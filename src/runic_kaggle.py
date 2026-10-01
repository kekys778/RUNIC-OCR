"""Runtime helpers for fire-and-forget notebooks (Kaggle / Colab / local).

Everything here is importable without a GPU or torch, so it is unit-tested in CI.
GPU code lives in :mod:`runic_vlm`.

* :func:`get_secret` — env var → Kaggle Secrets → Colab userdata;
* :func:`load_gold` — the gold set shipped in this repo, with inscription ids;
* :func:`find_adapter` — pick the LoRA adapter for a given base model inside a
  downloaded Hugging Face repo (the repo may hold several adapters);
* :class:`ResultSink` — one output folder per experiment with resumable steps,
  a run log and optional mirroring to a private Hugging Face dataset.
"""

from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

import pandas as pd

from runic_eval import inscription_id

REPO_ROOT = Path(__file__).resolve().parents[1]


def get_secret(name: str) -> str | None:
    """Read a secret from the environment, Kaggle Secrets or Colab userdata."""
    if os.environ.get(name):
        return os.environ[name]
    try:
        from kaggle_secrets import UserSecretsClient

        value = UserSecretsClient().get_secret(name)
        if value:
            return value
    except Exception:
        pass
    try:
        from google.colab import userdata

        return userdata.get(name)
    except Exception:
        return None


def run_isolated(
    spec: dict, workdir: str | Path, module: str = "runic_step", env: dict | None = None
) -> dict:
    """Run ``python -m <module> spec.json`` in a fresh process and return its result JSON.

    Every GPU step runs in its own process, so all GPU memory is released when it
    exits: repeated load/train/unload cycles in one Jupyter kernel leak memory and
    end in CUDA OOM. Child output is streamed into the notebook. Raises
    ``RuntimeError`` (with the last lines of output) if the child fails.
    """
    import subprocess
    import sys
    import uuid

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    tag = uuid.uuid4().hex[:8]
    spec_path, out_path = workdir / f"spec_{tag}.json", workdir / f"result_{tag}.json"
    spec_path.write_text(
        json.dumps({**spec, "out": str(out_path)}, ensure_ascii=False), encoding="utf-8"
    )
    child_env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", **(env or {})}
    src = str(Path(__file__).resolve().parent)
    child_env["PYTHONPATH"] = src + os.pathsep + child_env.get("PYTHONPATH", "")
    proc = subprocess.Popen(
        [sys.executable, "-u", "-m", module, str(spec_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=child_env,
    )
    tail: list[str] = []
    for line in proc.stdout:
        line = line.rstrip()
        if line and not line.startswith(("Loading weights", "Fetching", "Downloading")):
            print("   ", line)
        tail = (tail + [line])[-25:]
    code = proc.wait()
    spec_path.unlink(missing_ok=True)
    if code != 0 or not out_path.exists():
        raise RuntimeError(f"step failed (exit code {code}):\n" + "\n".join(tail))
    result = json.loads(out_path.read_text(encoding="utf-8"))
    out_path.unlink(missing_ok=True)
    return result


def environment_report() -> dict:
    """Python / GPU / library versions, recorded with every result."""
    info = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("torch", "transformers", "peft", "bitsandbytes", "accelerate"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None
    try:
        import torch

        info["cuda"] = torch.cuda.is_available()
        info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:
        info["cuda"], info["gpus"] = False, []
    return info


def load_gold(
    image_roots: list[str | Path] | None = None, repo_root: str | Path = REPO_ROOT
) -> pd.DataFrame:
    """Gold set: file_name, path, gt, inscription, segment.

    Labels come from ``data/gold_set/real_corpus.csv`` in the repo. Images are
    looked up by file name under ``image_roots`` first (e.g. the attached Kaggle
    dataset ``zhopa228/val-dataset``), then in the repo copy. Lines without an
    image are dropped with a warning (the repo copy lacks one of the 113 images).
    """
    repo_gold = Path(repo_root) / "data" / "gold_set"
    df = pd.read_csv(repo_gold / "real_corpus.csv").rename(columns={"filename": "file_name"})
    index: dict[str, str] = {}
    for root in [*(image_roots or []), repo_gold]:
        if Path(root).exists():
            for img in Path(root).rglob("*.png"):
                index.setdefault(img.name, str(img))
    df["path"] = df["file_name"].map(index.get)
    missing = df.loc[df["path"].isna(), "file_name"].tolist()
    if missing:
        print(f"[gold] no image for {len(missing)} line(s), skipped: {missing}")
        df = df[df["path"].notna()]
    df["gt"] = df["translit"].astype(str)
    df["inscription"] = df["file_name"].map(inscription_id)
    df["segment"] = df["source"].map({"real": "full line"}).fillna("word crop")
    return df[["file_name", "path", "gt", "inscription", "segment"]].reset_index(drop=True)


def find_adapter(root: str | Path, base_model_id: str) -> Path:
    """Return the folder with ``adapter_config.json`` whose base model matches.

    Matching ignores case and the organization prefix, so ``Qwen2.5-VL-7B-Instruct``
    matches ``Qwen/Qwen2.5-VL-7B-Instruct``. Raises with the list of candidates if
    zero or several adapters match.
    """
    want = base_model_id.split("/")[-1].lower()
    candidates = []
    for cfg in sorted(Path(root).rglob("adapter_config.json")):
        try:
            base = json.loads(cfg.read_text()).get("base_model_name_or_path", "")
        except Exception:
            continue
        candidates.append((cfg.parent, base))
    hits = [d for d, b in candidates if b and b.split("/")[-1].lower() == want]
    if len(hits) == 1:
        return hits[0]
    found = "\n".join(f"  {d} -> {b}" for d, b in candidates) or "  (none)"
    if not hits:
        raise FileNotFoundError(f"no adapter for {base_model_id} under {root}; found:\n{found}")
    # several checkpoints of the same model: prefer the shallowest path (the final adapter)
    hits.sort(key=lambda d: (len(d.parts), str(d)))
    print(f"[find_adapter] {len(hits)} adapters match, using {hits[0]}\n{found}")
    return hits[0]


class ResultSink:
    """Experiment output folder with resumable steps and optional HF mirroring.

    Layout::

        <out_dir>/<experiment>/
            run_log.jsonl        one line per finished step (resume marker)
            *.json / *.csv / *.png
            REPORT.md

    If ``hf_repo`` is set and a write token is available, the folder is uploaded
    to ``<hf_repo>/<experiment>/`` after every finished step, and previous
    results are pulled from there at start-up, so a timed-out Kaggle session can
    be resumed by simply running the notebook again.
    """

    def __init__(
        self,
        experiment: str,
        out_dir: str | Path,
        hf_repo: str = "",
        hf_token: str | None = None,
        seed_from: list[str | Path] | None = None,
    ):
        self.experiment = experiment
        self.dir = Path(out_dir) / experiment
        self.dir.mkdir(parents=True, exist_ok=True)
        self.hf_repo = hf_repo if (hf_repo and hf_token) else ""
        self.hf_token = hf_token
        self.log_path = self.dir / "run_log.jsonl"
        if seed_from and not self.log_path.exists():
            self._seed(seed_from)
        if self.hf_repo:
            self._pull()

    def _seed(self, roots: list[str | Path]) -> None:
        """Copy a previous run's folder (e.g. a past Kaggle Output added as Input) to resume from it."""
        import shutil

        for root in roots:
            if not Path(root).exists():
                continue
            for log in sorted(Path(root).rglob(f"{self.experiment}/run_log.jsonl")):
                src = log.parent
                if src.resolve() == self.dir.resolve():
                    continue
                shutil.copytree(src, self.dir, dirs_exist_ok=True)
                print(f"[sink] resumed from previous output {src}: {sorted(self.done_steps())}")
                return

    # -- resume ---------------------------------------------------------------
    def done_steps(self) -> set[str]:
        if not self.log_path.exists():
            return set()
        return {
            json.loads(line)["step"]
            for line in self.log_path.read_text().splitlines()
            if line.strip()
        }

    def is_done(self, step: str) -> bool:
        return step in self.done_steps()

    def mark_done(self, step: str, **info) -> None:
        rec = {"step": step, "time": time.strftime("%Y-%m-%d %H:%M:%S"), **info}
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        self.push(f"{self.experiment}: {step}")

    # -- writing --------------------------------------------------------------
    def path(self, name: str) -> Path:
        return self.dir / name

    def save_json(self, name: str, obj) -> Path:
        p = self.path(name)
        p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return p

    def save_csv(self, name: str, df: pd.DataFrame) -> Path:
        p = self.path(name)
        df.to_csv(p, index=False, encoding="utf-8")
        return p

    def save_text(self, name: str, text: str) -> Path:
        p = self.path(name)
        p.write_text(text, encoding="utf-8")
        return p

    # -- Hugging Face mirror ----------------------------------------------------
    def _pull(self) -> None:
        try:
            from huggingface_hub import HfApi, snapshot_download

            HfApi(token=self.hf_token).create_repo(
                self.hf_repo, repo_type="dataset", private=True, exist_ok=True
            )
            snapshot_download(
                self.hf_repo,
                repo_type="dataset",
                token=self.hf_token,
                allow_patterns=[f"{self.experiment}/*"],
                local_dir=str(self.dir.parent),
            )
            print(
                f"[sink] resumed from hf://{self.hf_repo}/{self.experiment}: {sorted(self.done_steps())}"
            )
        except Exception as e:  # results are still written locally
            print(f"[sink] could not pull from HF ({e}); continuing locally")

    def push(self, message: str = "update") -> None:
        if not self.hf_repo:
            return
        try:
            from huggingface_hub import upload_folder

            upload_folder(
                repo_id=self.hf_repo,
                repo_type="dataset",
                folder_path=str(self.dir),
                path_in_repo=self.experiment,
                token=self.hf_token,
                commit_message=message,
            )
        except Exception as e:
            print(f"[sink] HF upload failed ({e}); results are kept in {self.dir}")
