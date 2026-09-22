#!/usr/bin/env python3
"""
bib_cleanup.py — очистка и аудит библиографии ВКР.

Делает четыре вещи:
  1. Парсит .bib и .tex (без внешних зависимостей).
  2. Выкидывает записи, не цитируемые в тексте.
  3. Схлопывает дубликаты ключей — оставляет наиболее полную запись.
  4. Отчёт: что удалено / схлопнуто / неполно / названо в тексте без \\cite.

Запуск:
    python3 bib_cleanup.py VKR.tex refs.bib refs_cleaned.bib report.txt
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path


# ----- парсер .bib (без внешних библиотек) ---------------------------------

def parse_bib(text: str) -> list[dict]:
    """Разобрать .bib на список записей с балансировкой фигурных скобок."""
    entries = []
    pos = 0
    pat = re.compile(r'@(\w+)\s*\{\s*([^,\s]+)\s*,', re.MULTILINE)
    while True:
        m = pat.search(text, pos)
        if not m:
            break
        entry_start = m.start()
        body_start = m.end()
        depth = 1
        i = body_start
        while i < len(text) and depth > 0:
            c = text[i]
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
            i += 1
        if depth != 0:
            print(f"[warn] незакрытая запись начиная с {entry_start}", file=sys.stderr)
            break
        body = text[body_start:i - 1]
        entries.append({
            'type': m.group(1).lower(),
            'key': m.group(2),
            'fields': parse_fields(body),
            'raw': text[entry_start:i],
        })
        pos = i
    return entries


def parse_fields(body: str) -> dict[str, str]:
    """Грубо вытащить поля field = {value} / field = "value" / field = bare."""
    fields = {}
    pos = 0
    while pos < len(body):
        m = re.match(r'[\s,]*([A-Za-z_-]+)\s*=\s*', body[pos:])
        if not m:
            break
        name = m.group(1).lower()
        pos += m.end()
        if pos >= len(body):
            break
        if body[pos] == '{':
            depth, j = 1, pos + 1
            while j < len(body) and depth > 0:
                if body[j] == '{':
                    depth += 1
                elif body[j] == '}':
                    depth -= 1
                j += 1
            value = body[pos + 1:j - 1]
            pos = j
        elif body[pos] == '"':
            j = pos + 1
            while j < len(body) and body[j] != '"':
                j += 1
            value = body[pos + 1:j]
            pos = j + 1
        else:
            m2 = re.match(r'(\S+?)(?=[,\s]|$)', body[pos:])
            if not m2:
                break
            value = m2.group(1)
            pos += m2.end()
        fields[name] = value.strip()
    return fields


# ----- поиск ключей цитирования в .tex -------------------------------------

CITE_COMMANDS = r'cite|citep|citet|citeauthor|citeyear|citeyearpar|citealp|citealt|parencite|textcite|footcite|autocite|nocite'

def find_cited_keys(text: str) -> set[str]:
    keys = set()
    # \cite[opt]{a,b,c}
    pat = re.compile(r'\\(?:' + CITE_COMMANDS + r')\*?(?:\[[^\]]*\])*\s*\{([^}]+)\}')
    for m in pat.finditer(text):
        for k in m.group(1).split(','):
            k = k.strip()
            if k:
                keys.add(k)
    return keys


# ----- оценка полноты записи (для выбора лучшего дубля) --------------------

INCOMPLETE_NOTE = re.compile(r'требует финальной сверки', re.IGNORECASE)
AND_OTHERS_INCOMPLETE = re.compile(r'\{[A-Za-zА-Яа-я\-]+,\s*and others\}')

def completeness(entry: dict) -> int:
    score = sum(len(v) for v in entry['fields'].values())
    note = entry['fields'].get('note', '')
    author = '{' + entry['fields'].get('author', '') + '}'
    if INCOMPLETE_NOTE.search(note):
        score -= 200
    if AND_OTHERS_INCOMPLETE.search(author):
        score -= 100
    # лёгкий бонус за наличие DOI / URL
    if entry['fields'].get('doi'):
        score += 20
    if entry['fields'].get('url'):
        score += 10
    return score


# ----- проверка кандидатов на «названо, но не процитировано» ---------------
# (для добавления — не для удаления)

NAMED_BUT_UNCITED_CANDIDATES = {
    'Canny':                  ['canny1986'],
    'Otsu':                   ['otsu1979'],
    'Sauvola':                ['sauvola2000'],
    'MAE':                    ['he2022mae'],
    'few-shot / Prototypical': ['snell2017'],
    'low-resource survey':     ['hedderich2021'],
    'DINO':                   ['caron2021dino', 'oquab2023dinov2'],
    'SimCLR':                 ['chen2020simclr'],
    'Donut':                  ['kim2022donut'],
    'DETR':                   ['carion2020detr'],
    'BM3D':                   ['dabov2007'],
    'DnCNN':                  ['zhang2017dncnn'],
}


# ----- проверка полноты записей --------------------------------------------

def check_problems(entry: dict) -> list[str]:
    issues = []
    f = entry['fields']
    if INCOMPLETE_NOTE.search(f.get('note', '')):
        issues.append('помечена "требует финальной сверки"')
    if AND_OTHERS_INCOMPLETE.search('{' + f.get('author', '') + '}'):
        issues.append(f"неполный автор: {f.get('author', '')[:80]}")
    if not f.get('author') and entry['type'] not in ('misc', 'online'):
        issues.append('нет поля author')
    if not f.get('year') and not f.get('date'):
        issues.append('нет поля year/date')
    if entry['type'] == 'article' and not f.get('journal') and not f.get('journaltitle'):
        issues.append('article без journal')
    if entry['type'] == 'book' and not f.get('publisher'):
        issues.append('book без publisher')
    if entry['type'] in ('inproceedings', 'conference') and not f.get('booktitle'):
        issues.append('inproceedings без booktitle')
    return issues


# ----- main -----------------------------------------------------------------

def main(tex_path: str, bib_path: str, out_bib: str, out_report: str):
    tex = Path(tex_path).read_text(encoding='utf-8')
    bib = Path(bib_path).read_text(encoding='utf-8')

    entries = parse_bib(bib)
    cited = find_cited_keys(tex)

    by_key = defaultdict(list)
    for e in entries:
        by_key[e['key']].append(e)

    # Статистика
    total_entries = len(entries)
    unique_keys = len(by_key)
    duplicates = {k: v for k, v in by_key.items() if len(v) > 1}
    cited_in_bib = set(by_key) & cited
    cited_missing_from_bib = cited - set(by_key)
    in_bib_uncited = set(by_key) - cited

    # Выбор лучшего дубля среди процитированных
    keep = {}
    merge_log = []
    for key in sorted(cited_in_bib):
        candidates = by_key[key]
        if len(candidates) == 1:
            keep[key] = candidates[0]
        else:
            best = max(candidates, key=completeness)
            keep[key] = best
            scores = [(completeness(c), len(c['raw']), i) for i, c in enumerate(candidates)]
            merge_log.append((key, len(candidates), scores))

    # Проблемные записи среди оставленных
    problems = [(k, check_problems(e)) for k, e in keep.items()]
    problems = [(k, p) for k, p in problems if p]

    # «Названо в тексте, но не процитировано»
    named_uncited = []
    for method, keys in NAMED_BUT_UNCITED_CANDIDATES.items():
        # текст содержит метод?
        mentions = len(re.findall(re.escape(method.split()[0]), tex, re.IGNORECASE))
        for k in keys:
            if mentions > 0 and k in by_key and k not in cited:
                named_uncited.append((method, k, mentions))

    # ---- запись чистого .bib ----
    out_lines = [
        f"% Очищенная библиография для ВКР.",
        f"% Источник: {bib_path}",
        f"% Цитат-ключей в тексте: {len(cited)}",
        f"% Оставлено записей: {len(keep)}",
        f"% Сгенерировано bib_cleanup.py",
        "",
    ]
    for key in sorted(keep):
        out_lines.append(keep[key]['raw'].strip())
        out_lines.append("")
    Path(out_bib).write_text("\n".join(out_lines), encoding='utf-8')

    # ---- отчёт ----
    rep = []
    rep.append("=" * 72)
    rep.append("ОТЧЁТ ПО ОЧИСТКЕ БИБЛИОГРАФИИ")
    rep.append("=" * 72)
    rep.append("")
    rep.append(f"Всего записей в исходном .bib:      {total_entries}")
    rep.append(f"  из них уникальных ключей:         {unique_keys}")
    rep.append(f"  ключей с дубликатами:             {len(duplicates)}")
    rep.append(f"Уникальных ключей \\cite в тексте:   {len(cited)}")
    rep.append(f"  из них найдено в .bib:            {len(cited_in_bib)}")
    rep.append(f"  цитируется, но нет в .bib (!):    {len(cited_missing_from_bib)}")
    rep.append(f"Не цитируемых записей в .bib:       {len(in_bib_uncited)}")
    rep.append(f"Оставлено в очищенном .bib:         {len(keep)}")
    rep.append("")

    if cited_missing_from_bib:
        rep.append("─" * 72)
        rep.append("⚠  ЦИТИРУЕТСЯ В ТЕКСТЕ, НО ОТСУТСТВУЕТ В .BIB")
        rep.append("   (даст ошибку «?» при компиляции)")
        rep.append("─" * 72)
        for k in sorted(cited_missing_from_bib):
            rep.append(f"  {k}")
        rep.append("")

    if duplicates:
        rep.append("─" * 72)
        rep.append(f"СХЛОПНУТЫЕ ДУБЛИКАТЫ (выбрана наиболее полная запись)")
        rep.append("─" * 72)
        for key in sorted(duplicates):
            n = len(duplicates[key])
            in_cite = " ★ цитируется" if key in cited else "   не цитируется"
            rep.append(f"  {key:<35} ×{n} {in_cite}")
        rep.append("")

    rep.append("─" * 72)
    rep.append(f"УДАЛЕНО (нецитируемых записей): {len(in_bib_uncited)}")
    rep.append("─" * 72)
    for k in sorted(in_bib_uncited):
        rep.append(f"  {k}")
    rep.append("")

    if named_uncited:
        rep.append("─" * 72)
        rep.append("⚠  НАЗВАНО В ТЕКСТЕ, НО НЕТ \\cite (запись в .bib есть)")
        rep.append("   Рекомендуется проставить \\cite в местах упоминания.")
        rep.append("─" * 72)
        for method, key, n in named_uncited:
            rep.append(f"  «{method}» упоминается {n}×  →  добавьте \\cite{{{key}}}")
        rep.append("")

    if problems:
        rep.append("─" * 72)
        rep.append(f"НЕПОЛНЫЕ ЗАПИСИ СРЕДИ ОСТАВЛЕННЫХ (нужно дополнить вручную): {len(problems)}")
        rep.append("─" * 72)
        for key, issues in sorted(problems):
            rep.append(f"  {key}:")
            for iss in issues:
                rep.append(f"      • {iss}")
        rep.append("")

    rep.append("=" * 72)
    rep.append("ИТОГ")
    rep.append("=" * 72)
    rep.append(f"Чистый файл: {out_bib}")
    rep.append(f"Объём:        {total_entries} → {len(keep)} записей "
               f"(−{total_entries - len(keep)})")
    Path(out_report).write_text("\n".join(rep), encoding='utf-8')

    print("\n".join(rep))


if __name__ == '__main__':
    if len(sys.argv) != 5:
        print(__doc__)
        sys.exit(1)
    main(*sys.argv[1:])
