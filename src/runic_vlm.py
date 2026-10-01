"""GPU helpers for Qwen-VL runic OCR: load, predict (greedy + n-best), few-shot fine-tune.

Thin layer over :mod:`runic_ocr_experiments`, reusing its prompts, chat template,
4-bit NF4 config and label masking. Predictions therefore follow the thesis
protocol exactly (greedy decoding, same ``max_pixels`` per model). Heavy imports
happen inside functions, so this module can be imported (and linted) without a GPU.
"""

from __future__ import annotations

import gc
import time
from types import SimpleNamespace


def _harness():
    import runic_ocr_experiments as H

    return H


def free_gpu() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_qwen(model_key: str, adapter_dir: str | None = None, trainable: bool = False):
    """Load a registry model in 4-bit (+ optional LoRA adapter). Returns (model, processor, cfg)."""
    import torch
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoProcessor

    H = _harness()
    cfg = H.EXPERIMENTS[model_key]
    try:  # the adapter folder normally carries the processor it was trained with
        proc = AutoProcessor.from_pretrained(adapter_dir or cfg.model_id, max_pixels=cfg.max_pixels)
    except Exception:
        proc = AutoProcessor.from_pretrained(cfg.model_id, max_pixels=cfg.max_pixels)
    proc.tokenizer.padding_side = "right"
    base = H._vlm_class().from_pretrained(
        cfg.model_id,
        quantization_config=H._qwen_bnb(),
        torch_dtype=torch.float16,
        device_map="auto",
    )
    if trainable:
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
        base.config.use_cache = False
    model = (
        PeftModel.from_pretrained(base, adapter_dir, is_trainable=trainable)
        if adapter_dir
        else base
    )
    if not trainable:
        model.eval()
    return model, proc, cfg


def predict(
    model, proc, cfg, paths: list[str], n_best: int = 1, num_beams: int = 5, log_every: int = 25
) -> list[dict]:
    """Transliterate images.

    ``pred`` is always greedy decoding (thesis protocol). With ``n_best > 1`` a
    separate beam search returns the top-``n_best`` hypotheses and their
    length-normalized log-probabilities in ``nbest`` / ``scores``.
    """
    import torch
    from qwen_vl_utils import process_vision_info

    H = _harness()
    model.eval()
    prev_cache = getattr(model.config, "use_cache", True)
    model.config.use_cache = True
    out, t0 = [], time.time()
    with torch.no_grad():
        for i, p in enumerate(paths):
            msgs = H._qwen_msgs(str(p), cfg.max_pixels, None)
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            imgs, _ = process_vision_info(msgs)
            inp = proc(text=[text], images=imgs, return_tensors="pt").to(model.device)
            plen = inp["input_ids"].shape[1]
            g = model.generate(**inp, max_new_tokens=cfg.gen_max_new_tokens, do_sample=False)
            rec = {"pred": proc.batch_decode(g[:, plen:], skip_special_tokens=True)[0].strip()}
            if n_best > 1:
                b = model.generate(
                    **inp,
                    max_new_tokens=cfg.gen_max_new_tokens,
                    do_sample=False,
                    num_beams=max(num_beams, n_best),
                    num_return_sequences=n_best,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
                rec["nbest"] = [
                    s.strip()
                    for s in proc.batch_decode(b.sequences[:, plen:], skip_special_tokens=True)
                ]
                rec["scores"] = [float(s) for s in b.sequences_scores.cpu()]
            out.append(rec)
            if log_every and (i + 1) % log_every == 0:
                print(f"  {i + 1}/{len(paths)} images, {(time.time() - t0) / (i + 1):.1f} s/img")
    model.config.use_cache = prev_cache
    return out


def finetune(
    model,
    proc,
    cfg,
    items: list[dict],
    out_dir: str,
    epochs: float = 6,
    lr: float = 5e-5,
    grad_accum: int = 4,
    seed: int = 0,
) -> dict:
    """Continue training a trainable PeftModel on ``items`` = [{"image": path, "answer": text}].

    No evaluation or checkpoints during training (the data are tiny); the caller
    evaluates the returned model on the held-out fold.
    """
    from transformers import Trainer, TrainingArguments

    H = _harness()
    H.set_seed(seed)
    collate = H._make_qwen_collator(
        proc, SimpleNamespace(max_pixels=cfg.max_pixels), H._image_token_id(model, proc)
    )
    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,
        learning_rate=lr,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        fp16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="no",
        logging_steps=5,
        report_to="none",
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
        dataloader_num_workers=2,
        seed=seed,
    )
    model.train()
    trainer = Trainer(model=model, args=args, data_collator=collate, train_dataset=list(items))
    res = trainer.train()
    del trainer
    free_gpu()
    return {"train_loss": float(res.training_loss), "steps": int(res.global_step)}
