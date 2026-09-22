"""
╔══════════════════════════════════════════════════════════════════════╗
║  МОДУЛЬ ТРАНСЛИТЕРАЦИИ РУНИЧЕСКИХ НАДПИСЕЙ                           ║
║  Руна (Unicode) ↔ Латинская транслитерация                          ║
║                                                                      ║
║  Поддерживаемые системы:                                             ║
║    • Elder Futhark   (24 знака, 150–700 н.э.)                       ║
║    • Younger Futhark (16 знаков, 700–1100 н.э.)                     ║
║    • Anglo-Saxon Futhorc (28–33 знака, 400–1100 н.э.)               ║
║                                                                      ║
║  Конвенция: Rundata (стандарт Скандинавской рунической базы данных)  ║
║  Документация: https://www.nordiska.uu.se/forskn/samnord.htm         ║
╚══════════════════════════════════════════════════════════════════════╝
"""

from dataclasses import dataclass
from typing import Optional
import re
from google.colab import files as colab_files


# ──────────────────────────────────────────────────────────────────────
# ТАБЛИЦЫ ТРАНСЛИТЕРАЦИИ
# ──────────────────────────────────────────────────────────────────────
#
# Каждая запись: руна (Unicode) → транслитерация (строка Rundata)
#
# Принципы конвенции Rundata:
#   • строчные латинские буквы
#   • þ  для «th» (thorn)
#   • ð  для voiced «th»
#   • ą  для носового «a»
#   • ę  для носового «e»
#   • R  (заглавная) для Proto-Norse /z/ → /R/ (в Elder Futhark)
#   • ʀ  для Younger Futhark /R/
#   • ŋ  для «ng» / «ing»
#   • ï  для «ei» / «io»
#
# Источник: Rundata Manual, Samnordisk runtextdatabas

ELDER_FUTHARK_MAP = {
    # Руна   : транслитерация   : название   : фонетическое значение
    "ᚠ": "f",    # fehu      /f/
    "ᚢ": "u",    # uruz      /u/
    "ᚦ": "þ",    # thurisaz  /θ/ (unvoiced th)
    "ᚨ": "a",    # ansuz     /a/
    "ᚱ": "r",    # raidho    /r/
    "ᚲ": "k",    # kaunan    /k/
    "ᚷ": "g",    # gebo      /g/
    "ᚹ": "w",    # wunjo     /w/
    "ᚺ": "h",    # hagalaz   /h/
    "ᚾ": "n",    # naudiz    /n/
    "ᛁ": "i",    # isaz      /i/
    "ᛃ": "j",    # jera      /j/
    "ᛇ": "ï",    # iwaz/eihwaz  /ï/ ~ /ei/
    "ᛈ": "p",    # pertho    /p/
    "ᛉ": "R",    # algiz/elhaz  /z/ → /R/ (Proto-Norse)
    "ᛊ": "s",    # sowilo    /s/
    "ᛏ": "t",    # tiwaz     /t/
    "ᛒ": "b",    # berkanan  /b/
    "ᛖ": "e",    # ehwaz     /e/
    "ᛗ": "m",    # mannaz    /m/
    "ᛚ": "l",    # laguz     /l/
    "ᛜ": "ŋ",    # ingwaz    /ŋ/
    "ᛞ": "d",    # dagaz     /d/
    "ᛟ": "o",    # othalan   /o/
}

YOUNGER_FUTHARK_MAP = {
    # Younger Futhark — 16 знаков, сокращение Elder Futhark.
    # Парадокс: алфавит уменьшился, хотя фонология усложнилась →
    # один знак часто покрывает несколько фонем.
    "ᚠ": "f",     # fe
    "ᚢ": "u",     # ur       (также /o/, /y/, /w/)
    "ᚦ": "þ",     # thurs
    "ᚬ": "ą",     # oss      (носовое /ã/ → графема для /a/ и /o/)
    "ᚱ": "r",     # reid
    "ᚴ": "k",     # kaun     (также /g/)
    "ᚼ": "h",     # hagall
    "ᚾ": "n",     # nauðr
    "ᛁ": "i",     # is       (также /e/)
    "ᛅ": "a",     # ár
    "ᛋ": "s",     # sol
    "ᛏ": "t",     # tyr      (также /d/, /nd/)
    "ᛒ": "b",     # bjarkan  (также /p/, /mb/)
    "ᛘ": "m",     # maðr
    "ᛚ": "l",     # logr
    "ᛦ": "ʀ",     # yr       /R/ (Proto-Norse *z > Norse R)
}

ANGLO_SAXON_FUTHORC_MAP = {
    # Anglo-Saxon Futhorc расширяет Elder Futhark под
    # английскую фонологию. 28 базовых + до 5 дополнительных.
    "ᚠ": "f",
    "ᚢ": "u",
    "ᚦ": "þ",
    "ᚩ": "o",     # os       (новый знак для /o/)
    "ᚱ": "r",
    "ᚳ": "c",     # cen      /k/ → /tʃ/ перед передними гласными
    "ᚷ": "g",
    "ᚹ": "w",
    "ᚻ": "h",
    "ᚾ": "n",
    "ᛁ": "i",
    "ᛄ": "g",     # ger      (вариант /j/)
    "ᛇ": "eo",    # eoh      (двойная транслитерация)
    "ᛈ": "p",
    "ᛉ": "x",     # eolhx    /ks/
    "ᛋ": "s",
    "ᛏ": "t",
    "ᛒ": "b",
    "ᛖ": "e",
    "ᛗ": "m",
    "ᛚ": "l",
    "ᛝ": "ng",    # ing      /ŋ/
    "ᛞ": "d",
    "ᚩ": "œ",     # ethel    (омоним, контекстно)
    "ᚪ": "a",     # ac       /a:/
    "ᚫ": "æ",     # aesc     /æ/
    "ᚣ": "y",     # yr       /y/
    "ᛠ": "ea",    # ear      двойная транслитерация
    "ᛡ": "ia",    # ior
    "ᛣ": "q",     # cweorth  (ритуальный знак)
    "ᛤ": "k",     # calc
    "ᛥ": "st",    # stan     двойная транслитерация
}

# Специальные символы рунической пунктуации
RUNIC_PUNCT_MAP = {
    "᛫": "·",     # одиночный разделитель слов
    "᛬": ":",     # двойной разделитель
    "᛭": "+",     # тройной разделитель / крест
    " ": " ",     # пробел (где он явно присутствует)
}

# Объединённая таблица — порядок важен:
# при конфликте (один символ в нескольких алфавитах)
# используется Elder Futhark как базовая система
COMBINED_MAP = {
    **YOUNGER_FUTHARK_MAP,
    **ANGLO_SAXON_FUTHORC_MAP,
    **ELDER_FUTHARK_MAP,     # Elder перезаписывает конфликты
    **RUNIC_PUNCT_MAP,
}

# Обратная таблица: транслитерация → руна (Elder Futhark)
TRANSLIT_TO_RUNE = {v: k for k, v in ELDER_FUTHARK_MAP.items()}


# ──────────────────────────────────────────────────────────────────────
# КОНФИГУРАЦИЯ ТРАНСЛИТЕРАЦИИ
# ──────────────────────────────────────────────────────────────────────

@dataclass
class TranslitConfig:
    """
    Управляет поведением транслитерации.

    alphabet: какой алфавит использовать как основу.
      "elder"       → Elder Futhark (default)
      "younger"     → Younger Futhark
      "anglo_saxon" → Anglo-Saxon Futhorc
      "combined"    → все три (для смешанных датасетов)

    separator: строка между транслитерациями отдельных рун.
      ""  → "fuþark"   (слитно, как в Rundata)
      " " → "f u þ a r k"  (пробелами, проще для токенизатора)

    unknown_token: что ставить для нераспознанных символов.
      "?"  → явный маркер неизвестного знака
      ""   → пропускать
      "<UNK>" → совместимо с HuggingFace tokenizer

    lacuna_token: маркер лакуны (разрушенного фрагмента)
    """
    alphabet: str = "elder"
    separator: str = ""         # слитная транслитерация (Rundata-стиль)
    unknown_token: str = "?"
    lacuna_token: str = "[...]"
    ambiguous_token: str = "(?)"


# ──────────────────────────────────────────────────────────────────────
# ФУНКЦИИ ТРАНСЛИТЕРАЦИИ
# ──────────────────────────────────────────────────────────────────────

def rune_to_translit(
    rune_char: str,
    cfg: TranslitConfig = TranslitConfig(),
) -> str:
    """
    Транслитерирует один рунический символ.

    Args:
        rune_char: один символ Unicode (руна)
        cfg:       TranslitConfig

    Returns:
        str: латинская транслитерация или unknown_token
    """
    mapping = {
        "elder": ELDER_FUTHARK_MAP,
        "younger": YOUNGER_FUTHARK_MAP,
        "anglo_saxon": ANGLO_SAXON_FUTHORC_MAP,
        "combined": COMBINED_MAP,
    }.get(cfg.alphabet, COMBINED_MAP)

    result = mapping.get(rune_char) or RUNIC_PUNCT_MAP.get(rune_char)
    return result if result is not None else cfg.unknown_token


def transliterate(
    runic_text: str,
    cfg: TranslitConfig = TranslitConfig(),
) -> str:
    """
    Транслитерирует строку рунических символов в латиницу.

    Обрабатывает:
    - Обычные руны → транслитерация
    - Пробелы → сохраняются (слова разделены)
    - Служебные токены <LAC>, <AMB>...</AMB> → сохраняются
    - Неизвестные символы → unknown_token

    Args:
        runic_text: строка с рунами (Unicode U+16A0–U+16FF)
        cfg:        TranslitConfig

    Returns:
        str: латинская транслитерация

    Examples:
        >>> transliterate("ᚠᚢᚦᚨᚱᚲ")
        'fuþark'
        >>> transliterate("ᚨᛚᚢ")
        'alu'
        >>> transliterate("ᛏᛁᚹᚨᛉ")
        'tiwaR'
    """
    if not runic_text:
        return ""

    result_parts = []
    i = 0

    while i < len(runic_text):
        char = runic_text[i]

        # Служебные токены (<LAC>, <AMB>, </AMB>, <SEP>) — пропускаем как есть
        if char == "<":
            end = runic_text.find(">", i)
            if end != -1:
                token = runic_text[i:end + 1]
                result_parts.append(token)
                i = end + 1
                continue

        # Пробел — сохраняем
        if char == " ":
            result_parts.append(" ")
            i += 1
            continue

        # Рунический символ
        translit = rune_to_translit(char, cfg)
        result_parts.append(translit)
        i += 1

    return cfg.separator.join(result_parts) if cfg.separator else "".join(result_parts)


def transliterate_with_spaces(runic_text: str) -> str:
    """
    Удобная обёртка: транслитерация с пробелами между знаками.
    Лучше подходит для символьного языкового моделирования,
    хуже для прямого сравнения с Rundata-записями.

    Example:
        >>> transliterate_with_spaces("ᚠᚢᚦᚨᚱᚲ")
        'f u þ a r k'
    """
    cfg = TranslitConfig(separator=" ")
    return transliterate(runic_text, cfg)


def translit_to_rune_string(translit_text: str) -> str:
    """
    Обратное преобразование: транслитерация → руны (Elder Futhark).
    Полезно для верификации round-trip корректности.

    Example:
        >>> translit_to_rune_string("fuþark")
        'ᚠᚢᚦᚨᚱᚲ'
    """
    # Многосимвольные транслитерации (сначала длинные, потом короткие)
    MULTI_CHAR = {"eo": "ᛇ", "ea": "ᛠ", "ia": "ᛡ", "st": "ᛥ",
                  "ng": "ᛝ", "æ": "ᚫ", "þ": "ᚦ", "ð": "ᚦ",
                  "ŋ": "ᛜ", "ʀ": "ᛦ", "ï": "ᛇ", "ą": "ᚬ"}

    result = []
    i = 0
    while i < len(translit_text):
        # Проверяем двухсимвольные комбинации
        matched = False
        for multi, rune in MULTI_CHAR.items():
            if translit_text[i:i + len(multi)] == multi:
                result.append(rune)
                i += len(multi)
                matched = True
                break
        if not matched:
            rune = TRANSLIT_TO_RUNE.get(translit_text[i], translit_text[i])
            result.append(rune)
            i += 1

    return "".join(result)


# ──────────────────────────────────────────────────────────────────────
# ОБНОВЛЁННЫЙ ГЕНЕРАТОР LABELS.CSV
# ──────────────────────────────────────────────────────────────────────

def convert_labels_to_translit(
    input_csv: str,
    output_csv: str,
    cfg: TranslitConfig = TranslitConfig(),
    rune_column: str = "text",
    translit_column: str = "translit",
    keep_rune_column: bool = True,
) -> None:
    """
    Конвертирует существующий labels.csv:
    добавляет колонку с транслитерацией.

    Args:
        input_csv:        исходный CSV с рунами в колонке text
        output_csv:       путь для сохранения результата
        cfg:              TranslitConfig
        rune_column:      имя колонки с рунами
        translit_column:  имя новой колонки с транслитерацией
        keep_rune_column: сохранять ли исходную колонку с рунами

    Example:
        Input:  filename,text
                rune_001.png,ᚠᚢᚦᚨᚱᚲ
        Output: filename,text,translit
                rune_001.png,ᚠᚢᚦᚨᚱᚲ,fuþark
    """
    import pandas as pd

    df = pd.read_csv(input_csv)
    df[translit_column] = df[rune_column].apply(
        lambda x: transliterate(str(x), cfg)
    )

    if not keep_rune_column:
        df = df.drop(columns=[rune_column])

    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"✓ Конвертировано {len(df)} строк → {output_csv}")
    print(df[["filename", rune_column, translit_column]].head(5).to_string())


# ──────────────────────────────────────────────────────────────────────
# ОБНОВЛЁННЫЙ ГЕНЕРАТОР СИНТЕТИКИ — СРАЗУ С ТРАНСЛИТЕРАЦИЕЙ
# ──────────────────────────────────────────────────────────────────────

def generate_runic_text_with_translit(
    min_len: int,
    max_len: int,
    rng,
    alphabet: str = ELDER_FUTHARK_MAP,
    formulaic_prob: float = 0.25,
    cfg: TranslitConfig = TranslitConfig(),
) -> tuple[str, str]:
    """
    Генерирует пару (рунический текст, транслитерация).

    Returns:
        tuple: (runic_string, translit_string)
               ("ᚠᚢᚦᚨᚱᚲ", "fuþark")
    """
    FORMULAIC = ["ᚠᚢᚦᚨᚱᚲ", "ᚨᛚᚢ", "ᛚᚨᚢᚲᚨᛉ", "ᛏᛁᚹᚨᛉ", "ᛁᚾᚷᚹᚨᛉ"]

    if rng.random() < formulaic_prob:
        runic = rng.choice(FORMULAIC)
        if rng.random() < 0.4:
            extra_runes = list(ELDER_FUTHARK_MAP.keys())
            extra = "".join(
                rng.choice(extra_runes)
                for _ in range(int(rng.integers(1, 3)))
            )
            runic = runic + extra
    else:
        rune_chars = list(ELDER_FUTHARK_MAP.keys())
        length = int(rng.integers(min_len, max_len + 1))
        runic = "".join(rng.choice(rune_chars) for _ in range(length))

    translit = transliterate(runic, cfg)
    return runic, translit


# ──────────────────────────────────────────────────────────────────────
# ВАЛИДАЦИЯ РАЗМЕТКИ
# ──────────────────────────────────────────────────────────────────────

def validate_label_consistency(csv_path: str) -> dict:
    """
    Проверяет согласованность разметки в labels.csv.
    Находит строки где транслитерация не соответствует рунам.

    Полезно перед обучением: ловит ошибки ручной аннотации.

    Returns:
        dict: {
            "total": int,
            "consistent": int,
            "inconsistent": list of (filename, rune, translit, expected)
        }
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    cfg = TranslitConfig()

    if "translit" not in df.columns:
        return {"error": "Колонка 'translit' не найдена"}

    inconsistent = []
    for _, row in df.iterrows():
        rune_str = str(row.get("text", ""))
        saved_translit = str(row.get("translit", ""))

        # Пропускаем строки только с латиницей (уже транслитерированные)
        if not any(c in COMBINED_MAP for c in rune_str):
            continue

        expected = transliterate(rune_str, cfg)
        if saved_translit.strip() != expected.strip():
            inconsistent.append({
                "filename": row.get("filename", "?"),
                "rune": rune_str,
                "saved_translit": saved_translit,
                "expected_translit": expected,
            })

    return {
        "total": len(df),
        "consistent": len(df) - len(inconsistent),
        "inconsistent_count": len(inconsistent),
        "inconsistent_samples": inconsistent[:10],  # первые 10
    }


# ──────────────────────────────────────────────────────────────────────
# СПРАВОЧНИК: ПОЛНАЯ ТАБЛИЦА ТРАНСЛИТЕРАЦИЙ
# ──────────────────────────────────────────────────────────────────────

def print_transliteration_table() -> None:
    """Выводит полную таблицу транслитераций для документации."""
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  ELDER FUTHARK — Таблица транслитераций (Rundata)        ║")
    print("╠══════╦════════════╦══════════════╦════════════════════╣")
    print("║ Руна ║ Translit   ║ Название     ║ Фонетика           ║")
    print("╠══════╬════════════╬══════════════╬════════════════════╣")

    names = [
        ("fehu", "/f/"),        ("uruz", "/u/"),
        ("thurisaz", "/θ/"),    ("ansuz", "/a/"),
        ("raidho", "/r/"),      ("kaunan", "/k/"),
        ("gebo", "/g/"),        ("wunjo", "/w/"),
        ("hagalaz", "/h/"),     ("naudiz", "/n/"),
        ("isaz", "/i/"),        ("jera", "/j/"),
        ("iwaz", "/ï/"),        ("pertho", "/p/"),
        ("algiz", "/R/"),       ("sowilo", "/s/"),
        ("tiwaz", "/t/"),       ("berkanan", "/b/"),
        ("ehwaz", "/e/"),       ("mannaz", "/m/"),
        ("laguz", "/l/"),       ("ingwaz", "/ŋ/"),
        ("dagaz", "/d/"),       ("othalan", "/o/"),
    ]

    for (rune, translit), (name, phonetics) in zip(
        ELDER_FUTHARK_MAP.items(), names
    ):
        print(f"║  {rune}   ║ {translit:<10} ║ {name:<12} ║ {phonetics:<18} ║")

    print("╚══════╩════════════╩══════════════╩════════════════════╝")


# ──────────────────────────────────────────────────────────────────────
# ТОЧКА ВХОДА — демонстрация
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print_transliteration_table()
    print()

    # Тесты
    test_cases = [
        ("ᚠᚢᚦᚨᚱᚲ",     "fuþark"),
        ("ᚨᛚᚢ",         "alu"),
        ("ᛚᚨᚢᚲᚨᛉ",       "laukáR"),
        ("ᛏᛁᚹᚨᛉ",         "tiwaR"),
        ("ᛁᚾᚷᚹᚨᛉ",       "ingwaR"),
    ]

    print("Тесты транслитерации:")
    cfg = TranslitConfig()
    all_passed = True
    for runic, expected in test_cases:
        result = transliterate(runic, cfg)
        status = "✓" if result == expected else "✗"
        print(f"  {status}  {runic}  →  {result}  (ожидалось: {expected})")
        if result != expected:
            all_passed = False

    print(f"\nВсе тесты: {'PASSED ✓' if all_passed else 'FAILED ✗'}")

    # Round-trip проверка
    print("\nRound-trip (руна → транслит → руна):")
    for runic, _ in test_cases:
        translit = transliterate(runic, cfg)
        back = translit_to_rune_string(translit)
        status = "✓" if back == runic else "≈"
        print(f"  {status}  {runic} → {translit} → {back}")