"""
update_corpus.py
────────────────
Добавляет новые синтетические изображения в real_corpus.csv.

Логика:
  1. Читает все .png из папки с синтетикой (SYNTH_DIR).
  2. Сравнивает с existing corpus (CORPUS_CSV) по колонке `filename`.
  3. Новые файлы → парсит транслитерацию из имени файла:
       5652_0_þaun.png        → translit = "þaun"
       5364_2_leiknar-raist.png → translit = "leiknar raist"  (- → пробел)
  4. Заполняет все колонки корпуса, дописывает строки в конец CSV.

Формат имени файла: {любой_id}_{любой_idx}_{транслитерация}.png
  • Сплит делается по первым двум _ (split('_', 2)), поэтому транслитерация
    может сама содержать символы _ без проблем.
  • Дефис (-) в транслитерации → пробел (word divider).

Колонки корпуса:
  filename, translit, translit_raw, runic_approx,
  n_chars, source, split, had_word_divider, comment

Запуск:
  python update_corpus.py
  python update_corpus.py --synth_dir ./synth --corpus ./real_corpus.csv
  python update_corpus.py --dry_run          # только напечатать, не писать
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd


# ──────────────────────────────────────────────────────────────────────
# Настройки по умолчанию — меняйте здесь или через аргументы CLI
# ──────────────────────────────────────────────────────────────────────
DEFAULT_SYNTH_DIR = "./val_dataset/new"          # папка с новыми .png
DEFAULT_CORPUS    = "./val_dataset/real_corpus.csv"
DEFAULT_OUTPUT    = "./val_dataset/real_corpus.csv"  # перезаписать; или укажите новый путь
DEFAULT_SPLIT     = "train"            # split для новых строк
DEFAULT_SOURCE    = "synth"            # source для новых строк
# ──────────────────────────────────────────────────────────────────────


FILENAME_RE = re.compile(
    r"^(?P<base_id>.+?)_(?P<idx>\d+)_(?P<translit_raw_name>.+)\.png$",
    re.IGNORECASE,
)


def parse_filename(fname: str) -> dict | None:
    """
    Разбирает имя файла вида  {id}_{idx}_{translit}.png

    Примеры:
        '5652_0_þaun.png'           → {'translit': 'þaun',         'had_word_divider': False}
        '5364_2_leiknar-raist.png'  → {'translit': 'leiknar raist','had_word_divider': True}
        '100_3_alu-auja-laukaR.png' → {'translit': 'alu auja laukaR','had_word_divider': True}

    Возвращает None, если имя не соответствует паттерну.
    """
    # Берём только само имя файла (без директории)
    stem = Path(fname).name

    # Сплит строго по первым двум _ — транслитерация может содержать _
    parts = stem.split("_", 2)
    if len(parts) < 3:
        return None

    translit_part = parts[2]

    # Убираем расширение
    translit_part = re.sub(r"\.png$", "", translit_part, flags=re.IGNORECASE)

    # Дефис → пробел (word divider)
    had_word_divider = "-" in translit_part
    translit = translit_part.replace("-", " ")

    return {
        "translit": translit,
        "had_word_divider": had_word_divider,
    }


def n_chars_from_translit(translit: str) -> int:
    """Количество символов транслитерации без пробелов."""
    return len(translit.replace(" ", ""))


def build_new_row(filename: str, parsed: dict, source: str, split: str) -> dict:
    """Собирает строку DataFrame для нового файла."""
    translit = parsed["translit"]
    return {
        "filename":          filename,
        "translit":          translit,
        "translit_raw":      "",   # нет в имени файла; заполнить вручную при необходимости
        "runic_approx":      "",   # нет в имени файла; заполнить вручную при необходимости
        "n_chars":           n_chars_from_translit(translit),
        "source":            source,
        "split":             split,
        "had_word_divider":  parsed["had_word_divider"],
        "comment":           "",
    }


def main(
    synth_dir: str,
    corpus_path: str,
    output_path: str,
    split: str,
    source: str,
    dry_run: bool,
) -> None:
    synth_dir_path  = Path(synth_dir)
    corpus_file     = Path(corpus_path)
    output_file     = Path(output_path)

    # ── 1. Читаем существующий корпус ──────────────────────────────────
    if not corpus_file.exists():
        sys.exit(f"[ERR] Файл корпуса не найден: {corpus_file}")

    corpus = pd.read_csv(corpus_file, dtype=str)
    corpus["filename"] = corpus["filename"].str.strip()
    existing_filenames = set(corpus["filename"].tolist())
    print(f"[INFO] Загружен корпус: {len(corpus)} строк → {corpus_file}")

    # ── 2. Сканируем папку с синтетикой ────────────────────────────────
    if not synth_dir_path.exists():
        sys.exit(f"[ERR] Папка не найдена: {synth_dir_path}")

    png_files = sorted(synth_dir_path.glob("*.png"))
    print(f"[INFO] Найдено .png файлов в {synth_dir_path}: {len(png_files)}")

    # ── 3. Фильтруем: только те, которых нет в корпусе ─────────────────
    new_files = [p for p in png_files if p.name not in existing_filenames]
    print(f"[INFO] Новых файлов (отсутствуют в корпусе): {len(new_files)}")

    if not new_files:
        print("[OK]  Нет новых файлов для добавления. Корпус актуален.")
        return

    # ── 4. Парсим и строим строки ───────────────────────────────────────
    new_rows = []
    skipped  = []

    for p in new_files:
        parsed = parse_filename(p.name)
        if parsed is None:
            skipped.append(p.name)
            print(f"[WARN] Не удалось разобрать имя файла: {p.name!r} — пропущено")
            continue
        row = build_new_row(p.name, parsed, source=source, split=split)
        new_rows.append(row)

    print(f"\n[INFO] Готово к добавлению: {len(new_rows)} строк")
    if skipped:
        print(f"[WARN] Пропущено (нераспознанный формат): {len(skipped)} файлов")

    # ── 5. Превью ───────────────────────────────────────────────────────
    if new_rows:
        preview = pd.DataFrame(new_rows)
        print("\n── Превью первых 10 новых строк ──")
        print(preview.head(10).to_string(index=False))
        print()

    if dry_run:
        print("[DRY RUN] Запись отменена (--dry_run). Файл не изменён.")
        return

    # ── 6. Дописываем и сохраняем ───────────────────────────────────────
    if new_rows:
        new_df  = pd.DataFrame(new_rows)
        updated = pd.concat([corpus, new_df], ignore_index=True)

        # Восстанавливаем правильные типы для числовых колонок
        updated["n_chars"] = pd.to_numeric(updated["n_chars"], errors="coerce").fillna(0).astype(int)

        updated.to_csv(output_file, index=False, encoding="utf-8")
        print(f"[OK]  Корпус сохранён: {len(updated)} строк → {output_file}")
        print(f"      Было: {len(corpus)}  |  Добавлено: {len(new_rows)}  |  Стало: {len(updated)}")


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Добавить новые синтетические изображения в real_corpus.csv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--synth_dir", default=DEFAULT_SYNTH_DIR,
                        help=f"Папка с .png (по умолч.: {DEFAULT_SYNTH_DIR})")
    parser.add_argument("--corpus",    default=DEFAULT_CORPUS,
                        help=f"Входной CSV корпуса (по умолч.: {DEFAULT_CORPUS})")
    parser.add_argument("--output",    default=DEFAULT_OUTPUT,
                        help=f"Выходной CSV (по умолч.: {DEFAULT_OUTPUT} — перезапись)")
    parser.add_argument("--split",     default=DEFAULT_SPLIT,
                        help=f"Значение поля split для новых строк (по умолч.: {DEFAULT_SPLIT})")
    parser.add_argument("--source",    default=DEFAULT_SOURCE,
                        help=f"Значение поля source для новых строк (по умолч.: {DEFAULT_SOURCE})")
    parser.add_argument("--dry_run",   action="store_true",
                        help="Только вывести превью, не записывать файл")

    args = parser.parse_args()
    main(
        synth_dir   = args.synth_dir,
        corpus_path = args.corpus,
        output_path = args.output,
        split       = args.split,
        source      = args.source,
        dry_run     = args.dry_run,
    )