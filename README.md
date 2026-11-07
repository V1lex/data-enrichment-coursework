# data-enrichment-coursework

Comparing manual and synthetic data enrichment for sentiment classification and pet head detection.

## Ноутбуки

| Ноутбук | Что делает |
| --- | --- |
| [01_text_eda.ipynb](notebooks/01_text_eda.ipynb) | Все три файла UCI, метки, повторы, графики, очищенная таблица |
| [02_images_eda.ipynb](notebooks/02_images_eda.ipynb) | Oxford Pet, проверка фотографий и XML, рабочий manifest, примеры рамок и графики |
| [03_prepare_manual_reviews.ipynb](notebooks/03_prepare_manual_reviews.ipynb) | Формирование партий дополнительных отзывов и файлов разметки в Google Drive |

Ноутбуки выполняются независимо в среде CPU.

Оба ноутбука выполнены на полных официальных архивах, результаты сохранены в ячейках и в `results/`. Фактические итоги: 3 000 исходных отзывов → 2 979 после удаления повторов; 7 349 читаемых фотографий, 3 680 trainval ID → 3 670 пригодных для текущей постановки с одной рамкой. У 9 ID нет XML, у одного две рамки (исключён по правилу одного объекта, а не объявлен повреждённым). Детали — `results/text/summary.json`, `results/images/summary.json` и журналы проверок.

Открыть в Colab: [отзывы](https://colab.research.google.com/github/V1lex/data-enrichment-coursework/blob/main/notebooks/01_text_eda.ipynb) · [изображения](https://colab.research.google.com/github/V1lex/data-enrichment-coursework/blob/main/notebooks/02_images_eda.ipynb).

[Подготовка новых отзывов для разметки](https://colab.research.google.com/github/V1lex/data-enrichment-coursework/blob/main/notebooks/03_prepare_manual_reviews.ipynb) сохраняет единый набор из 120 отзывов в `manual/text/main.json` и `main.csv` на Drive. Источник — Women's E-Commerce Clothing Reviews (nicapotato, Kaggle; на странице указана CC0). Отбор и лицензия описаны в [протоколе](annotation/text/README.md). Исходные рейтинги не передаются в задания разметки. Человеческие ответы хранятся отдельно.

## Запуск в Google Colab

Ячейки выполняются последовательно. Первая ячейка подключает Google Drive и создаёт каталоги данных и результатов.

В `MyDrive/data-enrichment-coursework/` создаются:

```text
raw/text/          исходный архив UCI
raw/images/        images.tar.gz и annotations.tar.gz
processed/text/    исходная и очищенная таблицы, исключения
processed/images/  полный каталог, рабочий manifest, исключения
results/text/      графики, таблицы, summary.json, findings.txt, provenance.json
results/images/    графики, примеры рамок, аудит, summary.json, findings.txt
```

Архивы и результаты сохраняются на Drive. Фотографии распаковываются на временный диск Colab `/content/coursework_cache`. Oxford Pet требует места под архивы (~800 МБ) и распакованные файлы.

## Локальная проверка

```powershell
python -m pip install -r requirements.txt
python scripts/run_notebook.py notebooks/01_text_eda.ipynb
python scripts/run_notebook.py notebooks/02_images_eda.ipynb
```

Данные по умолчанию в `.local-data/`, исключённой из Git. Путь можно изменить переменной `COURSEWORK_DATA_ROOT`. Локальный исполнитель сохраняет реальные выводы и графики в ноутбуке; в Colab используйте обычную команду «Выполнить всё».

## Методика и ограничения

**Тексты:** все 3 000 записей читаются с учётом кавычек и Unicode-символов внутри IMDb-текстов. Повторы ищутся после нормализации регистра и пробелов; исходный текст сохраняется. Конфликтующие группы отправляются в карантин, однотипные повторы удаляются с сохранением ID и причины. Близкие по смыслу тексты такая проверка полностью не выявляет.

**Изображения:** весь корпус сохраняется в каталоге. Для детекции используются пригодные trainval ID с XML-рамками. 3 680 — размер списка trainval, а не гарантированное число корректных XML-файлов. Пропуски и ошибки фиксируются в аудите. Координаты сохранены как в XML; преобразование соглашения VOC под конкретную модель нужно определить перед обучением. Породы и виды используются для EDA; целевой класс проекта — `pet_head`.

Разбиения фиксируются до обогащения. Синтетические примеры создаются только из train соответствующего фолда, без использования validation/test.

## Источники и атрибуция

- [UCI Sentiment Labelled Sentences](https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences): CC BY 4.0. Kotzias, D. (2015), DOI [10.24432/C57604](https://doi.org/10.24432/C57604); *From Group to Individual Labels Using Deep Features*, KDD 2015.
- [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/): на странице указан CC BY-SA 4.0; права на фотографии остаются у исходных владельцев. Parkhi, Vedaldi, Zisserman, Jawahar, *Cats and Dogs*, CVPR 2012.
- [Документация TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/oxford_iiit_pet) уточняет доступность рамок для обучающей части.

`provenance.json` фиксирует дату, источники, SHA-256 архивов, версии библиотек и seed 42. Для новых реальных примеров источники и лицензии записываются отдельно.

## Документы проекта

- [Codebook](https://docs.google.com/document/d/1oGWzIBoeM9-Aeiu76H8AaAjfqK590-XYjxsKgj3P0Tc/edit)
- [Журнал](https://docs.google.com/document/d/1E4ee6-DXGrVtCHyEyz7J7g2HPwOV_6p_nQEP5hmiVmE/edit)

## Библиотеки

Python, pandas, NumPy, Matplotlib, Pillow, requests; в Colab — `google.colab.drive`. Генератор `scripts/build_notebooks.py` нужен для изменения шаблонов: повторная генерация очищает выводы ноутбуков.
