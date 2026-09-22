"""
╔══════════════════════════════════════════════════════════════════════╗
║  INPAINTING ГЕНЕРАТОР: Замена фона при сохранении рун                ║
║  Логика: PIL рендер → маска (белый фон) → SDXL Inpainting            ║
║           → каменный фон, руны нетронуты                             ║
║                                                                      ║
║  Почему inpainting лучше ControlNet для этой задачи:                 ║
║  ControlNet «советует» модели форму, но не запрещает её менять.      ║
║  Inpainting маскирует руны — генерация их физически не касается.     ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import gc
import json
import logging
import os
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# РУНЫ
# ──────────────────────────────────────────────────────────────────────

ELDER_FUTHARK_MAP = {
    "ᚠ": "f",  "ᚢ": "u",  "ᚦ": "þ",  "ᚨ": "a",
    "ᚱ": "r",  "ᚲ": "k",  "ᚷ": "g",  "ᚹ": "w",
    "ᚺ": "h",  "ᚾ": "n",  "ᛁ": "i",  "ᛃ": "j",
    "ᛇ": "ï",  "ᛈ": "p",  "ᛉ": "R",  "ᛊ": "s",
    "ᛏ": "t",  "ᛒ": "b",  "ᛖ": "e",  "ᛗ": "m",
    "ᛚ": "l",  "ᛜ": "ŋ",  "ᛞ": "d",  "ᛟ": "o",
}

FORMULAIC = [
    ("ᚠᚢᚦᚨᚱᚲ", "fuþark"),
    ("ᚨᛚᚢ",     "alu"),
    ("ᛚᚨᚢᚲᚨᛉ", "laukáR"),
    ("ᛏᛁᚹᚨᛉ",   "tiwaR"),
    ("ᛁᚾᚷᚹᚨᛉ", "ingwaR"),
    ("ᛖᚲ",      "ek"),
]


def make_pair(rng, min_len=2, max_len=10):
    if rng.random() < 0.25:
        runic, translit = FORMULAIC[int(rng.integers(0, len(FORMULAIC)))]
        if rng.random() < 0.4:
            keys   = list(ELDER_FUTHARK_MAP.keys())
            extra  = "".join(keys[int(rng.integers(0, len(keys)))]
                             for _ in range(int(rng.integers(1, 4))))
            runic    += extra
            translit += "".join(ELDER_FUTHARK_MAP[c] for c in extra)
        return runic, translit

    keys   = list(ELDER_FUTHARK_MAP.keys())
    length = int(rng.integers(min_len, max_len + 1))
    runes  = [keys[int(rng.integers(0, len(keys)))] for _ in range(length)]
    return "".join(runes), "".join(ELDER_FUTHARK_MAP[r] for r in runes)


def make_filename(idx, translit, runic):
    safe = translit.replace("/", "-").replace("\\", "-").replace(" ", "_")
    return f"syn_{idx:06d}_{safe}_{runic}.png"


# ──────────────────────────────────────────────────────────────────────
# ШРИФТ
# ──────────────────────────────────────────────────────────────────────

_NOTO_URL = (
    "https://github.com/notofonts/NotoSansRunic/raw/refs/heads/main"
    "/fonts/ttf/unhinted/instance_ttf/NotoSansRunic-Regular.ttf"
)

def ensure_font(path="NotoSansRunic-Regular.ttf"):
    p = Path(path)
    if p.exists():
        return str(p)
    for s in ["/usr/share/fonts/truetype/noto/NotoSansRunic-Regular.ttf",
              "/usr/share/fonts/noto/NotoSansRunic-Regular.ttf"]:
        if Path(s).exists():
            return s
    logger.info(f"Скачивание шрифта → {p}")
    urllib.request.urlretrieve(_NOTO_URL, p)
    return str(p)


# ──────────────────────────────────────────────────────────────────────
# АВТОРИЗАЦИЯ
# ──────────────────────────────────────────────────────────────────────

# def setup_hf_auth() -> Optional[str]:
#     try:
#         from kaggle_secrets import UserSecretsClient

#         os.environ["HF_TOKEN"] = 'YOUR_HF_TOKEN'
#         from huggingface_hub import login
#         login('YOUR_HF_TOKEN')
#         logger.info("HuggingFace: авторизован")
#         return token
#     except Exception as e:
#         logger.warning(f"HF_TOKEN не найден: {e}")
#         return None


# ──────────────────────────────────────────────────────────────────────
# РЕНДЕР ГЛИФА И СОЗДАНИЕ МАСКИ
# ──────────────────────────────────────────────────────────────────────

def render_rune_with_mask(
    runic_text: str,
    size: int,
    font_path: str,
    font_size: int,
    rng: np.random.Generator,
) -> tuple[Image.Image, Image.Image]:
    """
    Рендерит руны и создаёт inpainting-маску.

    Returns:
        init_image: PIL RGB — белый фон + чёрные руны
        mask_image: PIL RGB — БЕЛЫЙ там где рисовать (фон),
                              ЧЁРНЫЙ там где НЕ трогать (руны)

    Логика inpainting:
        SD получает init_image + mask_image.
        Белые пиксели маски = «здесь рисуй камень».
        Чёрные пиксели маски = «здесь ничего не трогай».
        Таким образом руны физически защищены от изменений.
    """
    font = ImageFont.truetype(font_path, size=font_size)

    # Рендер глифа
    base = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(base)
    bbox = draw.textbbox((0, 0), runic_text, font=font)
    tw   = bbox[2] - bbox[0]
    th   = bbox[3] - bbox[1]
    x    = max(16, (size - tw) // 2 + int(rng.integers(-12, 12)))
    y    = max(16, (size - th) // 2 + int(rng.integers(-8, 8)))
    draw.text((x, y), runic_text, fill=(0, 0, 0), font=font)

    # Маска: инвертируем + расширяем защищённую область
    # чтобы края рун не «обгорали» при inpainting
    gray  = np.array(base.convert("L"))

    # Бинаризация: руны = 0 (чёрные), фон = 255 (белый)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Дилатируем ЧЁРНУЮ область рун — создаём буфер ~4px вокруг каждой руны
    # Это критично: без буфера inpainting «подъедает» края штрихов
    kernel   = np.ones((8, 8), np.uint8)
    dilated  = cv2.dilate(255 - binary, kernel, iterations=1)
    rune_zone = 255 - dilated   # зона защиты (чёрная = не трогать)

    # Маска для SD: белый = рисовать камень, чёрный = не трогать
    mask_np = rune_zone  # фон белый (255), руны+буфер чёрные (0)
    mask_img = Image.fromarray(
        cv2.cvtColor(mask_np, cv2.COLOR_GRAY2RGB)
    )

    return base, mask_img


# ──────────────────────────────────────────────────────────────────────
# ПРОМПТЫ
# ──────────────────────────────────────────────────────────────────────

# Компоненты для domain randomization
POSITIVE_PROMPT = (
    "ancient runic inscription on dark grey granite stone slab, "
    "weathered rough surface, shallow carved runes, "
    "even diffuse daylight, monochrome grey, "
    "archaeological macro photo, sharp focus, "
    "realistic stone texture, no fantasy, no glow"
)

NEGATIVE_PROMPT = (
    "smooth stone, polished, modern font, fantasy, "
    "glow, blur, watermark, extra symbols, cartoon, "
    "bright colors, cinematic, illustration"
)


def get_prompt(rng: np.random.Generator) -> str:
    """Строит промпт из фиксированной инструкции + случайных компонентов."""
    return (
        f"{_BASE_INSTRUCTION} "
        f"Stone material: {str(rng.choice(_STONE_MATERIAL))}. "
        f"Lighting: {str(rng.choice(_LIGHTING))}. "
        f"Camera: {str(rng.choice(_CAMERA))}."
    )


# ──────────────────────────────────────────────────────────────────────
# ПАЙПЛАЙНЫ
# ──────────────────────────────────────────────────────────────────────

def load_inpainting_pipeline(token: str):
    """
    SDXL Inpainting — основной пайплайн.
    Не требует SD3 токена, работает на T4/P100.
    VRAM: ~10 GB с fp16.
    """
    from diffusers import AutoPipelineForInpainting, UniPCMultistepScheduler

    logger.info("Загрузка SDXL Inpainting...")
    pipe = AutoPipelineForInpainting.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=torch.float16,
        use_safetensors=True,
        token=token,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    vram = torch.cuda.memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"SDXL Inpainting загружен. VRAM: {vram:.1f}/{total:.1f} GB")
    return pipe, "sdxl_inpaint"


def load_sd3_pipeline(token: str):
    """
    SD3 + ControlNet Canny — альтернатива при ≥14 GB VRAM.
    Более высокое качество текстур, но требует SD3-доступа.
    """
    from diffusers import SD3ControlNetModel, StableDiffusion3ControlNetPipeline

    dtype = torch.float16
    logger.info("Загрузка SD3 ControlNet...")
    controlnet = SD3ControlNetModel.from_pretrained(
        "InstantX/SD3-Controlnet-Canny",
        torch_dtype=dtype,
        token='YOUR_HF_TOKEN',
    )
    logger.info("Загрузка SD3 Medium...")
    pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        controlnet=controlnet,
        torch_dtype=dtype,
        text_encoder_3=None,
        tokenizer_3=None,
        token=token,
    )
    pipe.enable_model_cpu_offload()
    vram = torch.cuda.memory_allocated() / 1e9
    logger.info(f"SD3 загружен. VRAM: {vram:.1f} GB")
    return pipe, "sd3_controlnet"


def auto_load_pipeline(token: Optional[str]):
    """
    Автовыбор пайплайна:
      ≥14 GB + token → SD3 ControlNet (лучше)
      ≥ 8 GB         → SDXL Inpainting (надёжнее)
      CPU / fallback  → PIL only
    """
    if not torch.cuda.is_available() or token is None:
        return None, "pil_only"

    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    available = vram - 2.0
    logger.info(f"VRAM доступно: {available:.1f} GB")

    if available >= 10.0:
        try:
            return load_sd3_pipeline(token)
        except Exception as e:
            logger.warning(f"SD3 не загрузился ({e}), пробуем SDXL Inpainting.")

    if available >= 8.0:
        try:
            return load_inpainting_pipeline(token)
        except Exception as e:
            logger.warning(f"SDXL Inpainting не загрузился ({e}), PIL fallback.")

    return None, "pil_only"


# ──────────────────────────────────────────────────────────────────────
# PIL ФОЛЛБЕК
# ──────────────────────────────────────────────────────────────────────

def pil_stone_texture(size, rng):
    base = rng.uniform(0.45, 0.70, (size, size))
    for scale in [4, 8, 16, 32]:
        hs, ws = max(1, size // scale), max(1, size // scale)
        n = rng.uniform(-0.06, 0.06, (hs, ws))
        ni = Image.fromarray(
            ((n + 0.5) * 255).clip(0, 255).astype(np.uint8)
        ).resize((size, size), Image.NEAREST)
        base += (np.array(ni).astype(float) / 255 - 0.5) * 0.04
    bright = int(rng.integers(130, 200))
    tex = (base * bright).clip(0, 255).astype(np.uint8)
    r = np.clip(tex + int(rng.integers(-8,  8)), 0, 255).astype(np.uint8)
    g = np.clip(tex + int(rng.integers(-4,  4)), 0, 255).astype(np.uint8)
    b = np.clip(tex + int(rng.integers(-12, 2)), 0, 255).astype(np.uint8)
    return Image.fromarray(np.stack([r, g, b], axis=2))


def pil_composite(init_image, mask_image, rng):
    """
    PIL фоллбек: заменяет только белый фон каменной текстурой.
    Маска: белый = заменить, чёрный = оставить.
    """
    size = init_image.width
    stone = pil_stone_texture(size, rng)
    mask_np = np.array(mask_image.convert("L")).astype(float) / 255
    init_np = np.array(init_image).astype(float)
    stone_np = np.array(stone).astype(float)
    # Смешиваем: там где маска белая (=1) — камень, где чёрная (=0) — руна
    result = init_np * (1 - mask_np[..., None]) + stone_np * mask_np[..., None]
    img = Image.fromarray(result.clip(0, 255).astype(np.uint8))
    img = ImageEnhance.Contrast(img).enhance(float(rng.uniform(1.0, 1.4)))
    return img


# ──────────────────────────────────────────────────────────────────────
# ГЕНЕРАЦИЯ ОДНОГО ОБРАЗЦА
# ──────────────────────────────────────────────────────────────────────

def generate_one(
    runic_text: str,
    pipe,
    mode: str,
    font_path: str,
    rng: np.random.Generator,
    sample_idx: int,
    cfg,
) -> Image.Image:
    size      = cfg.image_size
    font_size = int(rng.integers(*cfg.font_size_range))
    prompt    = POSITIVE_PROMPT
    generator = torch.Generator("cuda").manual_seed(int(cfg.seed + sample_idx))

    init_image, mask_image = render_rune_with_mask(
        runic_text, size, font_path, font_size, rng
    )

    if mode == "sdxl_inpaint":
        result = pipe(
            prompt=POSITIVE_PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            image=init_image,
            mask_image=mask_image,
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=cfg.guidance_scale,
            strength=cfg.inpaint_strength,   # 0.85–0.99: насколько агрессивно менять фон
            generator=generator,
            width=size,
            height=size,
        )
        return result.images[0]

    elif mode == "sd3_controlnet":
        gray  = np.array(init_image.convert("L"))
        edges = cv2.Canny(gray, 50, 150)
        kernel = np.ones((2, 2), np.uint8)
        edges  = cv2.dilate(edges, kernel, iterations=1)
        canny  = Image.fromarray(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))
        result = pipe(
            prompt=POSITIVE_PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            control_image=canny,
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=cfg.guidance_scale,
            controlnet_conditioning_scale=cfg.controlnet_scale,
            generator=generator,
            width=size,
            height=size,
        )
        return result.images[0]

    else:  # pil_only
        return pil_composite(init_image, mask_image, rng)


# ──────────────────────────────────────────────────────────────────────
# КОНФИГУРАЦИЯ
# ──────────────────────────────────────────────────────────────────────

@dataclass
class GeneratorConfig:
    output_dir: str        = "/content/drive/MyDrive/SYNTH_PHOTO"
    n_samples: int         = 2000
    font_path: str         = "NotoSansRunic-Regular.ttf"
    font_size_range: tuple = (72, 160)
    image_size: int        = 1024
    min_rune_len: int      = 2
    max_rune_len: int      = 10
    num_inference_steps: int = 30
    guidance_scale: float    = 8.0
    controlnet_scale: float  = 0.65
    # strength для inpainting: 0.85–0.99
    # 0.85 = консервативно (больше сохраняет оригинал)
    # 0.99 = агрессивно (полностью перегенерирует фон)
    inpaint_strength: float  = 0.92
    seed: int              = 1003
    zip_every: int         = 101
    checkpoint_every: int  = 25


# ──────────────────────────────────────────────────────────────────────
# ZIP МЕНЕДЖЕР
# ──────────────────────────────────────────────────────────────────────

class ZipManager:
    def __init__(self, output_dir, zip_every):
        self.output_dir = Path(output_dir)
        self.zip_every  = zip_every
        self.batch_idx  = 0
        self.pending    = []
        self.archive_dir = Path("/content/drive/MyDrive/SYNTH_PHOTO")
        self.archive_dir.mkdir(parents=True, exist_ok=True)

    def add(self, record):
        self.pending.append(record)
        if len(self.pending) >= self.zip_every:
            self._flush()

    def _flush(self):
        if not self.pending:
            return
        path = self.archive_dir / f"batch_{self.batch_idx:04d}.zip"
        img_dir = self.output_dir / "images"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            for r in self.pending:
                p = img_dir / r["filename"]
                if p.exists():
                    zf.write(p, f"images/{r['filename']}")
            zf.writestr(
                f"labels_batch_{self.batch_idx:04d}.csv",
                pd.DataFrame(self.pending).to_csv(index=False)
            )
        mb = path.stat().st_size / 1e6
        logger.info(
            f"📦 {path.name} готов ({len(self.pending)} шт, {mb:.1f} MB) "
            f"→ Output → {path.name}"
        )
        self.pending.clear()
        self.batch_idx += 1

    def finalize(self, all_records):
        if self.pending:
            self._flush()
        final = self.archive_dir / "runic_dataset_FULL.zip"
        img_dir = self.output_dir / "images"
        labels = self.output_dir / "labels.csv"
        with zipfile.ZipFile(final, "w", zipfile.ZIP_DEFLATED) as zf:
            for r in all_records:
                p = img_dir / r["filename"]
                if p.exists():
                    zf.write(p, f"images/{r['filename']}")
            if labels.exists():
                zf.write(labels, "labels.csv")
        mb = final.stat().st_size / 1e6
        logger.info(f"✅ FULL архив: {final.name} ({mb:.1f} MB) → Output")


# ──────────────────────────────────────────────────────────────────────
# ОСНОВНОЙ ГЕНЕРАТОР
# ──────────────────────────────────────────────────────────────────────

def generate_dataset(cfg: GeneratorConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)

    out_dir = Path(cfg.output_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    labels_path = out_dir / "labels.csv"

    cfg.font_path = ensure_font(cfg.font_path)
    token = 'YOUR_HF_TOKEN'
    pipe, mode = auto_load_pipeline(token)

    logger.info(f"\n{'═'*50}\n  Режим: {mode}\n{'═'*50}")

    # Checkpoint
    if labels_path.exists():
        try:
            existing = pd.read_csv(labels_path)
            if len(existing) == 0:
                raise ValueError
            start_idx = len(existing)
            records   = existing.to_dict("records")
            logger.info(f"Продолжение с {start_idx}")
        except (pd.errors.EmptyDataError, ValueError):
            logger.warning("labels.csv пустой — начинаем заново.")
            labels_path.unlink()
            start_idx, records = 0, []
    else:
        start_idx, records = 0, []

    if start_idx >= cfg.n_samples:
        return pd.read_csv(labels_path)

    zip_mgr = ZipManager(cfg.output_dir, cfg.zip_every)
    errors  = 0

    try:
        pbar = tqdm(
            range(start_idx, cfg.n_samples),
            desc=f"[{mode}]",
            initial=start_idx,
            total=cfg.n_samples,
        )
        for i in pbar:
            runic, translit = make_pair(rng, cfg.min_rune_len, cfg.max_rune_len)
            filename        = make_filename(i, translit, runic)

            try:
                img = generate_one(runic, pipe, mode, cfg.font_path, rng, i, cfg)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"OOM на {i}.")
                    torch.cuda.empty_cache()
                    gc.collect()
                    errors += 1
                    if errors > 5:
                        logger.warning("Переключение на PIL.")
                        pipe, mode = None, "pil_only"
                    continue
                raise

            img.save(img_dir / filename)
            record = {"filename": filename, "runic": runic, "translit": translit}
            records.append(record)
            zip_mgr.add(record)
            pbar.set_postfix({"rune": runic[:6], "tr": translit[:6]})

            if (i + 1) % cfg.checkpoint_every == 0:
                pd.DataFrame(records).to_csv(labels_path, index=False)

    finally:
        df = pd.DataFrame(records)
        df.to_csv(labels_path, index=False)
        zip_mgr.finalize(records)
        logger.info(f"Сохранено: {len(df)} образцов")

    return df


# ──────────────────────────────────────────────────────────────────────
# ТОЧКА ВХОДА
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = GeneratorConfig(
        n_samples         = 2000,
        image_size        = 1024,
        font_size_range   = (72, 160),
        min_rune_len      = 8,
        max_rune_len      = 15,
        num_inference_steps = 30,
        guidance_scale    = 8.0,
        inpaint_strength  = 0.92,
        zip_every         = 100,
        checkpoint_every  = 25,
    )
    df = generate_dataset(cfg)
    print(df.head(5).to_string(index=False))