# Данные

## `gold_set/` — реальное золотое множество (только для оценки)
113 строк реальных рунических надписей (Старший и Младший футарк, камень / металл / кость) с экспертной
транслитерацией в конвенции Rundata. **Не используется ни в обучении, ни в подборе гиперпараметров.**

`real_corpus.csv`:

| колонка | смысл |
|---|---|
| `filename` | файл изображения строки |
| `translit` | нормализованная транслитерация — целевая строка (алфавит Σ) |
| `translit_raw` | транслитерация как в источнике (с малыми капителями ᴀ, z и т. п.) |
| `runic_approx` | приблизительная запись рунами Unicode |
| `n_chars`, `source`, `split`, `had_word_divider`, `comment` | служебные поля |

Имена файлов вида `5652_0_kyla-þaun.png` = `<id надписи>_<индекс фото>_<транслитерация>`; см. `src/update_corpus.py`.
`data_latex/` — иллюстрации для текста ВКР.

## `synthetic/` — синтетический обучающий корпус
`labels_runic.csv` — метки всех 4 647 изображений финального корпуса (SD3 + ControlNet Canny):
`filename, batch, idx, transliteration, runic, n_chars, source, split`.
В `samples/` — несколько примеров. Полный корпус (~11 ГБ) — Kaggle dataset `zhopa228/synth-final` (приватный).

## `sources/` — исходные выгрузки для лексикона
- `christerhamp_gamla_runor_with_filenames.csv` — сводка *Gamla runor* (Christer Hamp);
- `runer_ku.csv` — выгрузка Danske Runeindskrifter (runer.ku.dk), см. `notebooks/01_data_collection/runer_ku_parser.ipynb`;
- `NotoSansRunic-Regular.ttf` — шрифт для рендера рун (SIL Open Font License).
