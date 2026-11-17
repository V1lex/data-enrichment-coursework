# data-enrichment-coursework

Comparing manual and synthetic data enrichment for sentiment classification and pet head detection.

## Ноутбуки

| Ноутбук | Что делает |
| --- | --- |
| [01_text_eda.ipynb](notebooks/01_text_eda.ipynb) | Все три файла UCI, метки, повторы, графики, очищенная таблица |
| [02_images_eda.ipynb](notebooks/02_images_eda.ipynb) | Oxford Pet, проверка фотографий и XML, рабочий manifest, примеры рамок и графики |
| [03_prepare_manual_reviews.ipynb](notebooks/03_prepare_manual_reviews.ipynb) | Формирование партий дополнительных отзывов и файлов разметки в Google Drive |
| [04_text_training.ipynb](notebooks/04_text_training.ipynb) | Сравнение TF-IDF + логистической регрессии и CatBoost на четырёх вариантах обогащения текста |
| [05_image_training.ipynb](notebooks/05_image_training.ipynb) | Сравнение YOLO11n и Faster R-CNN на четырёх вариантах обогащения изображений |
| [05_image_training_resume.ipynb](notebooks/05_image_training_resume.ipynb) | Продолжение обучения детекторов с сохранённого состояния |

Ноутбуки EDA, подготовки данных и обучения текста рассчитаны на CPU. Для обучения детекторов в ноутбуке 05 нужен Colab с GPU, например T4.

Оба ноутбука EDA выполнены на полных официальных архивах. Графики сохранены в ячейках ноутбуков, таблицы и журналы проверок — в `results/`. Из 3 000 исходных отзывов после удаления повторов осталось 2 979. В Oxford Pet читаются 7 349 фотографий. Из 3 680 trainval ID для задачи с одной рамкой пригодны 3 670: у 9 ID нет XML, у одного две рамки. Сводки: [текст](results/text/summary.json) и [изображения](results/images/summary.json).

Открыть в Colab: [отзывы](https://colab.research.google.com/github/V1lex/data-enrichment-coursework/blob/main/notebooks/01_text_eda.ipynb) · [изображения](https://colab.research.google.com/github/V1lex/data-enrichment-coursework/blob/main/notebooks/02_images_eda.ipynb).

[Подготовка новых отзывов для разметки](https://colab.research.google.com/github/V1lex/data-enrichment-coursework/blob/main/notebooks/03_prepare_manual_reviews.ipynb) сохраняет единый набор из 120 отзывов в `manual/text/main.json` и `main.csv` на Drive. Источник — Women's E-Commerce Clothing Reviews (nicapotato, Kaggle; на странице указана CC0). Отбор и лицензия описаны в [протоколе](annotation/text/README.md). Исходные рейтинги не передаются в задания разметки. Человеческие ответы хранятся отдельно.

## Запуск в Google Colab

Ячейки выполняются последовательно. Первая ячейка подключает Google Drive и создаёт каталоги данных и результатов.

Ноутбуки читают данные из `MyDrive/data-enrichment-coursework/`. Пути в таблице указаны относительно этой папки.

| Данные | № 04 · текст | № 05 · изображения |
| --- | --- | --- |
| Базовая выборка | `processed/text/reviews_clean.csv` | `raw/images/images.tar.gz` и `processed/images/pet_head_clean_manifest.csv` |
| Фиксированное разбиение | `splits/text/uci_split_seed42.csv` и `.json` | `splits/images/oxford_pet_split_seed42.csv` и `.json` |
| Реальные дополнения | `manual/text/manual_text_agreed_112.json` | `manual/images/main/` и `manual/images/annotations_final.json` |
| Синтетические дополнения | `augmented/text/draft_codex.json` | `augmented/images/augmented_images.zip` |
| Результаты обучения | `results/text/training/<run_id>/` | `results/images/training/<run_id>/` |

Ноутбук продолжения обучения дополнительно читает сохранённое разбиение, списки обучающих примеров и контрольные точки из папки предыдущего запуска `results/images/training/20261117T200442Z_d167b2ac/`. Фотографии распаковываются на временный диск Colab.

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
