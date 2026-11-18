# Сравнение обогащения данных

Курсовая сравнивает обучение моделей на исходных данных и после ручного или искусственного дополнения. Эксперименты используют английские отзывы и изображения кошек и собак.

## Ноутбуки

| Ноутбук | Назначение |
| --- | --- |
| [01_text_eda](notebooks/01_text_eda.ipynb) | Анализ и очистка отзывов UCI |
| [02_images_eda](notebooks/02_images_eda.ipynb) | Анализ Oxford-IIIT Pet и рамок голов |
| [03_prepare_manual_reviews](notebooks/03_prepare_manual_reviews.ipynb) | Подготовка дополнительных отзывов к разметке |
| [04_text_training](notebooks/04_text_training.ipynb) | TF-IDF + логистическая регрессия и CatBoost |
| [05_image_training_resume](notebooks/05_image_training_resume.ipynb) | YOLO11n и Faster R-CNN с продолжением обучения |

В обоих экспериментах сравниваются четыре варианта: исходные данные; исходные + ручные; исходные + искусственные; исходные + ручные и искусственные. Для изображений искусственное дополнение - горизонтальное отражение, а не генерация ИИ. Разбиение выполняется до обогащения; тестовая часть не попадает в обучение.

## Запуск в Colab

Откройте нужный ноутбук в VS Code с расширением Colab и выполните ячейки сверху вниз. Для текста достаточно CPU, для изображений нужен GPU, например T4. Ноутбуки используют папку `MyDrive/data-enrichment-coursework/` на Google Drive.

| Данные относительно папки проекта на Drive | Тексты | Изображения |
| --- | --- | --- |
| Исходные | `processed/text/reviews_clean.csv` | `raw/images/images.tar.gz`, `processed/images/pet_head_clean_manifest.csv` |
| Разбиение | `splits/text/uci_split_seed42.csv` и `.json` | `splits/images/oxford_pet_split_seed42.csv` и `.json` |
| Ручное дополнение | `manual/text/manual_text_agreed_112.json` | `manual/images/main/`, `manual/images/annotations_final.json` |
| Искусственное дополнение | `augmented/text/draft_codex.json` | `augmented/images/augmented_images.zip` |
| Результаты обучения | `results/text/training/<run_id>/` | `results/images/training/<run_id>/` |

### Продолжение детекции

В [ноутбуке 05](notebooks/05_image_training_resume.ipynb) сейчас указан последний зафиксированный запуск `20261118T103517Z_8430340a`. По сохранённому выводу все четыре YOLO и два варианта Faster R-CNN готовы; для Faster R-CNN с отражениями записана седьмая эпоха, восьмая была прервана. Фактические файлы на Drive проверяются при запуске.

После подключения Colab выполните все ячейки. Ноутбук проверит данные и найдёт готовые модели и чекпоинты в цепочке предыдущих запусков, затем продолжит незавершённые модели. Он создаст новую папку результатов на Drive и выведет её ID. При следующем обрыве замените `RESUME_SOURCE_RUN_ID` в разделе настроек на ID последнего запуска и выполните ячейки снова. При обрыве внутри эпохи она выполняется заново. Предыдущие папки запусков на Drive нужны для восстановления - не удаляйте их до окончания эксперимента.

## Данные и результаты

- UCI: 3 000 исходных отзывов, после очистки 2 979; фиксированное разбиение хранится в [data/text](data/text/README.md).
- Ручные отзывы: исходные экспорты двух участников и 112 совпавших меток лежат в [annotation/text](annotation/text/README.md). В экспорте A2 есть предсказания, поэтому независимость разметки по файлам не подтверждена.
- [3 000 синтетических отзывов](data/text/synthetic/README.md) предназначены только для обучения. Их метки и разнообразие не проходили независимую проверку.
- Oxford-IIIT Pet: из 3 680 trainval ID пригодны 3 670. Разбиение и отражения описаны в [data/images](data/images/README.md).
- [100 ручных рамок](annotation/images/README.md): при IoU > 0,9 координаты A1 и A2 усредняются, иначе выбирается A1. Это автоматическое правило, а не дополнительная проверка правильности.
- `results/` содержит таблицы и сводки анализа; графики и сохранённые выводы находятся в ноутбуках. Исходные большие файлы и веса хранятся на Drive, а `.local-data/` и модельные веса исключены из Git.

## Источники

- [UCI Sentiment Labelled Sentences](https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences) - CC BY 4.0, Kotzias (2015), [DOI](https://doi.org/10.24432/C57604).
- [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/) - указана CC BY-SA 4.0; права на фотографии принадлежат их владельцам.
- [Women's E-Commerce Clothing Reviews](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews) - источник ручных текстовых дополнений, на странице указана CC0.
- [Codebook](https://docs.google.com/document/d/1oGWzIBoeM9-Aeiu76H8AaAjfqK590-XYjxsKgj3P0Tc/edit) и [журнал](https://docs.google.com/document/d/1E4ee6-DXGrVtCHyEyz7J7g2HPwOV_6p_nQEP5hmiVmE/edit).
