"""Build the two self-contained Colab notebooks (standard library only)."""
import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(text).strip() + "\n"}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": textwrap.dedent(text).strip() + "\n"}


SETUP = '''
import hashlib
import json
import os
import platform
import random
import shutil
import sys
import tarfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import requests
from PIL import Image

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
if IN_COLAB:
    drive.mount("/content/drive")
    STORAGE = Path("/content/drive/MyDrive/data-enrichment-coursework")
    WORK = Path("/content/coursework_cache")
else:
    STORAGE = Path(os.environ.get("COURSEWORK_DATA_ROOT", ".local-data")).resolve()
    WORK = STORAGE / "cache"
RAW = STORAGE / "raw" / TASK
PROCESSED = STORAGE / "processed" / TASK
OUT = STORAGE / "results" / TASK
for folder in (RAW, PROCESSED, OUT, WORK):
    folder.mkdir(parents=True, exist_ok=True)

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def download(url, target):
    target = Path(target)
    if target.exists() and target.stat().st_size:
        print("Используем сохранённый архив:", target.name)
        return target
    partial = target.with_suffix(target.suffix + ".part")
    for attempt in range(3):
        try:
            with requests.get(url, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                total = int(response.headers.get("content-length", 0))
                received = 0
                last_report = time.monotonic()
                with partial.open("wb") as handle:
                    for chunk in response.iter_content(1024 * 1024):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        received += len(chunk)
                        if time.monotonic() - last_report > 15:
                            print(f"{target.name}: {received / 1e6:.1f} МБ / {total / 1e6:.1f} МБ")
                            last_report = time.monotonic()
                if total and received != total:
                    raise IOError("Неполная загрузка")
                partial.replace(target)
                print("Скачано:", target.name, target.stat().st_size, "байт")
                return target
        except (requests.RequestException, OSError):
            if attempt == 2:
                raise

def unpack(archive, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    fingerprint = sha256_file(archive)
    marker = destination / (Path(archive).name + ".sha256")
    if marker.exists() and marker.read_text() == fingerprint:
        return destination
    def output_path(name):
        path = (destination / name).resolve()
        if not path.is_relative_to(destination.resolve()):
            raise ValueError("Путь выходит за папку распаковки: " + name)
        return path
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                path = output_path(member.filename)
                if member.is_dir():
                    path.mkdir(parents=True, exist_ok=True)
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.open(member) as source, path.open("wb") as target:
                        shutil.copyfileobj(source, target)
    else:
        with tarfile.open(archive, "r:gz") as bundle:
            for member in bundle:
                path = output_path(member.name)
                if member.isdir():
                    path.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.extractfile(member) as source, path.open("wb") as target:
                        shutil.copyfileobj(source, target)
                # Ссылки и специальные файлы не распаковываются.
    marker.write_text(fingerprint)
    return destination

def save_json(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def provenance(sources, archives):
    save_json("provenance.json", {
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED, "sources": sources,
        "archives": [{"name": p.name, "bytes": p.stat().st_size,
                      "sha256": sha256_file(p)} for p in archives],
        "versions": {"python": platform.python_version(), "pandas": pd.__version__,
                     "numpy": np.__version__, "matplotlib": matplotlib.__version__},
    })

def finish_figure(name):
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=160, bbox_inches="tight")
    plt.show()
    plt.close()
'''

TEXT = [
md('''
# 01. EDA исходных отзывов — UCI Sentiment Labelled Sentences

**Цель исследования:** изучение структуры исходного корпуса, оценка качества
данных и формирование очищенной выборки для классификации тональности.
Анализ охватывает все три опубликованных источника: Amazon, Yelp и IMDb.
Целевая переменная: 0 — отрицательная, 1 — положительная тональность.

Источник: https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences
Лицензия: CC BY 4.0. Атрибуция: Kotzias, D. (2015), DOI 10.24432/C57604;
работа *From Group to Individual Labels Using Deep Features*, KDD 2015.
'''),
code('TASK = "text"\n' + textwrap.dedent(SETUP)),
md('''
## 1. Загрузка полного опубликованного набора

Файлы Amazon, Yelp и IMDb читаются построчно, с разделением по **последней табуляции**.
Это сохраняет строки с кавычками, которые обычный CSV-парсер может ошибочно объединить.
Ошибочные строки записываются отдельно. Ожидаемый объём — 1 000 строк на источник.
'''),
code('''
URL = "https://archive.ics.uci.edu/static/public/331/sentiment%2Blabelled%2Bsentences.zip"
archive = download(URL, RAW / "sentiment_labelled_sentences.zip")
data_dir = unpack(archive, WORK / "uci_sentiment")
rows, parse_errors = [], []
for source, filename in [("amazon", "amazon_cells_labelled.txt"),
                         ("yelp", "yelp_labelled.txt"), ("imdb", "imdb_labelled.txt")]:
    matches = list(data_dir.rglob(filename))
    if len(matches) != 1:
        raise RuntimeError(f"Ожидался один {filename}, найдено {len(matches)}")
    # splitlines() treats U+0085 inside two IMDb reviews as a line break.
    # Records in this release are separated by LF, not by Unicode separators.
    lines = matches[0].read_text(encoding="utf-8-sig").split("\\n")
    if lines and lines[-1] == "":
        lines.pop()
    for line_no, line in enumerate(lines, 1):
        try:
            text, label = line.rsplit("\\t", 1)
            if label not in {"0", "1"}:
                raise ValueError("Метка не равна 0 или 1")
            rows.append({"id": f"{source}_{line_no:04d}", "text": text,
                         "label": int(label), "source": source, "line": line_no})
        except ValueError as error:
            parse_errors.append({"source": source, "line": line_no,
                                 "raw": line, "reason": str(error)})
original = pd.DataFrame(rows)
pd.DataFrame(parse_errors, columns=["source", "line", "raw", "reason"]).to_csv(
    OUT / "parse_errors.csv", index=False)
display(original.head())
display(original.groupby("source").agg(records=("id", "size")))
if len(original) != 3000 or parse_errors:
    raise RuntimeError("Структура исходного набора изменилась: смотрите parse_errors.csv")
assert original.groupby("source").size().eq(1000).all()
original.to_csv(PROCESSED / "reviews_original.csv", index=False)
'''),
md('''
## 2. Пропуски, повторы и конфликтующие метки

Для поиска повторов тексты приводятся к нижнему регистру с нормализацией пробелов.
Пунктуация и исходный текст сохраняются. Метод выявляет точные нормализованные
повторы; семантическая близость текстов отдельно не оценивается.
Группы одинаковых текстов с разными метками отправляются в карантин целиком.
Для остальных групп повторов сохраняется первый ID. Исключения фиксируются отдельно.
'''),
code('''
original["normalized_text"] = original.text.str.lower().str.split().str.join(" ")
original["chars"] = original.text.str.len()
original["words"] = original.text.str.split().str.len()
empty = original.normalized_text.eq("")
conflict_keys = original.groupby("normalized_text").label.nunique()
conflict_keys = set(conflict_keys[conflict_keys > 1].index)
conflicts = original[original.normalized_text.isin(conflict_keys)].copy()
duplicates = original[original.duplicated("normalized_text", keep=False)].copy()
eligible = original[~empty & ~original.normalized_text.isin(conflict_keys)].copy()
clean = eligible.drop_duplicates("normalized_text", keep="first").copy()
removed = original[~original.id.isin(clean.id)].copy()
removed["reason"] = np.where(removed.normalized_text.eq(""), "empty_text",
    np.where(removed.normalized_text.isin(conflict_keys), "conflicting_labels", "duplicate"))
for name, table in [("reviews_clean.csv", clean), ("reviews_removed.csv", removed)]:
    table.to_csv(PROCESSED / name, index=False)
duplicates.to_csv(OUT / "duplicate_groups.csv", index=False)
conflicts.to_csv(OUT / "label_conflicts.csv", index=False)
summary = {"original_rows": len(original), "empty_texts": int(empty.sum()),
           "duplicate_group_rows": len(duplicates), "conflicting_rows": len(conflicts),
           "clean_rows": len(clean), "removed_rows": len(removed)}
save_json("summary.json", summary)
display(pd.Series(summary, name="value").to_frame())
display(removed[["id", "text", "label", "reason"]].head(10))
'''),
md('## 3. Баланс классов, источники и длина отзывов'),
code('''
balance = pd.crosstab(clean.source, clean.label).reindex(columns=[0, 1], fill_value=0)
balance.to_csv(OUT / "class_balance.csv")
lengths = clean.groupby("source")[["words", "chars"]].describe()
lengths.to_csv(OUT / "text_lengths.csv")
display(balance.rename(columns={0: "negative", 1: "positive"}))
display(lengths)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
balance.rename(columns={0: "negative", 1: "positive"}).plot.bar(ax=axes[0, 0], rot=0)
axes[0, 0].set_title("Class balance after cleaning")
pd.DataFrame({"original": original.groupby("source").size(),
              "clean": clean.groupby("source").size()}).plot.bar(ax=axes[0, 1], rot=0)
axes[0, 1].set_title("Rows before and after cleaning")
axes[1, 0].hist(clean.words, bins=35)
axes[1, 0].set(xlabel="Words", ylabel="Reviews", title="Review length")
axes[1, 1].hist(clean.chars, bins=35)
axes[1, 1].set(xlabel="Characters", ylabel="Reviews", title="Review length")
finish_figure("text_eda.png")
'''),
md('''
## 4. Выводы

Сводные показатели исходного и очищенного наборов, распределение целевой
переменной и ограничения проведённого анализа.
'''),
code('''
report = (f"Исходный набор: {len(original)} строк, три источника.\\n"
          f"После проверок: {len(clean)} строк. Исключено: {len(removed)}.\\n"
          f"Пустых текстов: {int(empty.sum())}; строк с конфликтом меток: {len(conflicts)}.\\n"
          f"Число отзывов по меткам: {clean.label.value_counts().sort_index().to_dict()}.\\n"
          f"Медианная длина: {clean.words.median():.1f} слов.\\n"
          "Ограничения: английский язык, три домена, только явно положительные и отрицательные оценки; "
          "проверка нормализованных повторов не исключает близкие по смыслу тексты.\\n")
(OUT / "findings.txt").write_text(report, encoding="utf-8")
print(report)
provenance([{"url": URL, "page": "https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences",
             "license": "CC BY 4.0", "citation": "Kotzias (2015), DOI 10.24432/C57604"}], [archive])
'''),
]

IMAGES = [
md('''
# 02. EDA исходных изображений — Oxford-IIIT Pet

**Цель исследования:** изучение состава корпуса, проверка изображений и
аннотаций, формирование выборки для локализации головы животного.
Рабочая постановка предусматривает одну рамку головы и общий класс `pet_head`.
Породы и виды животных используются для анализа состава данных.

Источник: https://www.robots.ox.ac.uk/~vgg/data/pets/
Лицензия страницы: CC BY-SA 4.0; права на фотографии остаются у авторов.
Атрибуция: Parkhi, Vedaldi, Zisserman, Jawahar, *Cats and Dogs*, CVPR 2012.

Корпус содержит 7 349 изображений, включая 3 680 ID в trainval и 3 669 в test.
Доступность XML-аннотаций и число пригодных примеров определяются при проверке.
'''),
code('TASK = "images"\n' + textwrap.dedent(SETUP)),
md('## 1. Загрузка исходных изображений и аннотаций'),
code('''
BASE = "https://thor.robots.ox.ac.uk/~vgg/data/pets/"
annotation_archive = download(BASE + "annotations.tar.gz", RAW / "annotations.tar.gz")
image_archive = download(BASE + "images.tar.gz", RAW / "images.tar.gz")
data_dir = WORK / "oxford_pet"
unpack(annotation_archive, data_dir)
unpack(image_archive, data_dir)
annotation_dir = data_dir / "annotations"
image_dir = data_dir / "images"
import xml.etree.ElementTree as ET

def read_split(name):
    table = pd.read_csv(annotation_dir / f"{name}.txt", sep=r"\\s+", header=None,
        names=["id", "breed_id", "species_id", "breed_species_id"], comment="#")
    table["official_split"] = name
    return table

catalog = pd.concat([read_split("trainval"), read_split("test")], ignore_index=True)
if catalog.id.duplicated().any():
    raise RuntimeError("В официальных списках повторяются ID")
catalog["species"] = catalog.species_id.map({1: "cat", 2: "dog"})
assert catalog.species.notna().all()
catalog["breed"] = catalog.id.str.rsplit("_", n=1).str[0]
display(catalog.head())
display(catalog.groupby("official_split").size().to_frame("listed_images"))
'''),
md('''
## 2. Полная проверка файлов и рамок

Для каждой фотографии проверяются декодирование, размеры и SHA-256.
XML проверяется для каждого ID из обоих списков. Отсутствие XML у test ожидаемо:
такие изображения не включаются в выборку с готовыми рамками для детекции.

Координаты `xmin_xml` и т. д. сохраняются **как в исходном XML**, без обрезки и
автоматического исправления. При подготовке к обучению преобразование формата
координат определяется требованиями модели. Ошибки и несоответствия фиксируются в аудите.
'''),
code('''
records = []
for i, item in enumerate(catalog.to_dict("records"), 1):
    row = dict(item)
    path = image_dir / (row["id"] + ".jpg")
    xml_path = annotation_dir / "xmls" / (row["id"] + ".xml")
    row.update(image_exists=path.exists(), image_ok=False, width=None, height=None,
               image_sha256=None, xml_exists=xml_path.exists(), bbox_ok=False,
               bbox_count=0, xmin_xml=None, ymin_xml=None, xmax_xml=None, ymax_xml=None,
               xml_width=None, xml_height=None, dimension_match=False, error="")
    errors = []
    if path.exists():
        try:
            with Image.open(path) as image:
                image.load()
                row["width"], row["height"] = image.size
            row["image_sha256"] = sha256_file(path)
            row["image_ok"] = True
        except (OSError, ValueError) as error:
            errors.append("image: " + str(error))
    else:
        errors.append("missing_image")
    if xml_path.exists():
        try:
            tree = ET.parse(xml_path).getroot()
            row["xml_width"] = int(tree.findtext("size/width"))
            row["xml_height"] = int(tree.findtext("size/height"))
            boxes = tree.findall("object/bndbox")
            row["bbox_count"] = len(boxes)
            row["dimension_match"] = (row["width"] == row["xml_width"] and
                                      row["height"] == row["xml_height"])
            if len(boxes) != 1:
                errors.append(f"expected_one_bbox_found_{len(boxes)}")
            else:
                coords = [int(boxes[0].findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]
                for key, value in zip(("xmin_xml", "ymin_xml", "xmax_xml", "ymax_xml"), coords):
                    row[key] = value
                x1, y1, x2, y2 = coords
                valid = (0 <= x1 < x2 <= row["xml_width"] and
                         0 <= y1 < y2 <= row["xml_height"])
                row["bbox_ok"] = bool(valid and row["image_ok"] and row["dimension_match"])
                if not valid:
                    errors.append("invalid_bbox_coordinates")
            if not row["dimension_match"]:
                errors.append("xml_image_size_mismatch")
        except (ET.ParseError, TypeError, ValueError, OSError) as error:
            errors.append("xml: " + str(error))
    row["error"] = "; ".join(errors)
    records.append(row)
    if i % 1000 == 0:
        print(f"Проверено {i}/{len(catalog)}")
audit = pd.DataFrame(records)
audit.to_csv(OUT / "image_audit.csv", index=False)
display(audit.groupby("official_split").agg(
    listed=("id", "size"), readable=("image_ok", "sum"),
    with_xml=("xml_exists", "sum"), valid_bbox=("bbox_ok", "sum")))
display(audit[audit.error.ne("")][["id", "error"]].head(15))
'''),
md('''
## 3. Рабочая выборка детекции, повторы и пропуски

В рабочую выборку входят все пригодные trainval ID с одной корректной рамкой.
Полный состав корпуса сохраняется в каталоге и аудите.
Точные копии по байтам группируются; среди пригодных сохраняется первый ID.
Визуально близкие снимки могут иметь разные хеши. Проверка выявляет только
побайтовые копии и не заменяет анализ визуальной близости.
'''),
code('''
hashes = audit.image_sha256.dropna()
duplicate_hashes = set(hashes[hashes.duplicated(keep=False)])
duplicate_images = audit[audit.image_sha256.isin(duplicate_hashes)].copy()
duplicate_images.to_csv(OUT / "image_duplicate_groups.csv", index=False)
usable = audit[audit.official_split.eq("trainval") & audit.bbox_ok].copy()
clean = usable.drop_duplicates("image_sha256", keep="first").copy()
clean["label"] = "pet_head"
clean["image_filename"] = clean.id + ".jpg"
clean["bbox_width_xml"] = clean.xmax_xml - clean.xmin_xml
clean["bbox_height_xml"] = clean.ymax_xml - clean.ymin_xml
clean["bbox_area_fraction"] = (clean.bbox_width_xml * clean.bbox_height_xml) / (clean.width * clean.height)
clean["aspect_ratio"] = clean.width / clean.height
assert len(clean) > 0, "Нет пригодных примеров — проверьте аудит"
assert clean.id.is_unique
audit.to_csv(PROCESSED / "images_original_manifest.csv", index=False)
clean.to_csv(PROCESSED / "pet_head_clean_manifest.csv", index=False)
excluded = audit[audit.official_split.eq("trainval") & ~audit.id.isin(clean.id)].copy()
excluded["exclusion_reason"] = np.where(excluded.bbox_ok, "duplicate_file",
    np.where(~excluded.image_ok, "unreadable_or_missing_image",
        np.where(~excluded.xml_exists, "missing_xml", "invalid_xml_or_bbox")))
excluded.to_csv(PROCESSED / "pet_head_excluded.csv", index=False)
summary = {"listed_total": len(audit),
           "listed_trainval": int(audit.official_split.eq("trainval").sum()),
           "listed_test": int(audit.official_split.eq("test").sum()),
           "readable_images": int(audit.image_ok.sum()),
           "xml_files_in_catalog": int(audit.xml_exists.sum()),
           "valid_trainval_before_dedup": len(usable), "clean_detection_rows": len(clean),
           "excluded_trainval": len(excluded), "exact_duplicate_group_rows": len(duplicate_images)}
save_json("summary.json", summary)
display(pd.Series(summary, name="value").to_frame())
display(excluded[["id", "exclusion_reason", "error"]].head(15))
'''),
md('## 4. Распределения размеров, видов, пород и площади рамок'),
code('''
clean.groupby("species").size().to_csv(OUT / "species_balance.csv", header=["images"])
clean.groupby("breed").size().to_csv(OUT / "breed_balance.csv", header=["images"])
clean[["width", "height", "aspect_ratio", "bbox_area_fraction"]].describe().to_csv(
    OUT / "image_statistics.csv")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
clean.species.value_counts().plot.bar(ax=axes[0, 0], rot=0)
axes[0, 0].set_title("Species in clean detection subset")
axes[0, 1].scatter(clean.width, clean.height, s=6, alpha=0.35)
axes[0, 1].set(xlabel="Width (px)", ylabel="Height (px)", title="Image dimensions")
axes[1, 0].hist(clean.bbox_area_fraction, bins=35)
axes[1, 0].set(xlabel="BBox area / image area (XML spans)", title="Head size")
axes[1, 1].hist(clean.aspect_ratio, bins=35)
axes[1, 1].set(xlabel="Width / height", title="Aspect ratio")
finish_figure("image_eda.png")
plt.figure(figsize=(13, 5))
clean.breed.value_counts().sort_index().plot.bar()
plt.ylabel("Images")
plt.title("Breed composition (all labels are mapped to pet_head for training)")
finish_figure("breed_balance.png")
'''),
md('''
## 5. Визуальный контроль готовых рамок

Детерминированная выборка из 12 изображений. Рамки показаны в исходных координатах.
Дополнительно выводятся четыре примера с наименьшей и наибольшей относительной
площадью головы. Проверка позволяет оценить положение рамок и выявить
несоответствия, которые не обнаруживаются при проверке диапазонов координат.
'''),
code('''
from matplotlib.patches import Rectangle

def plot_examples(table, filename, columns=4):
    rows = (len(table) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(14, 3.5 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, row in zip(axes.flat, table.to_dict("records")):
        with Image.open(image_dir / (row["id"] + ".jpg")) as image:
            ax.imshow(image.convert("RGB"))
        ax.add_patch(Rectangle((row["xmin_xml"], row["ymin_xml"]),
            row["xmax_xml"] - row["xmin_xml"], row["ymax_xml"] - row["ymin_xml"],
            fill=False, edgecolor="lime", linewidth=2))
        ax.set_title(row["id"], fontsize=9)
    finish_figure(filename)

plot_examples(clean.sample(min(12, len(clean)), random_state=SEED), "bbox_examples.png")
extremes = pd.concat([clean.nsmallest(2, "bbox_area_fraction"),
                      clean.nlargest(2, "bbox_area_fraction")]).drop_duplicates("id")
plot_examples(extremes, "bbox_extremes.png")
'''),
md('## 6. Выводы'),
code('''
report = (f"В официальных списках {len(audit)} изображений; trainval: {summary['listed_trainval']}; "
          f"test: {summary['listed_test']}.\\n"
          f"Читаемых изображений: {summary['readable_images']}; XML в каталоге: {summary['xml_files_in_catalog']}.\\n"
          f"Рабочая выборка детекции после проверок и удаления точных копий: {len(clean)}.\\n"
          f"Исключено из trainval: {len(excluded)}; причины в pet_head_excluded.csv.\\n"
          f"Состав по видам: {clean.species.value_counts().to_dict()}.\\n"
          "Ограничения: точный хеш не выявляет все визуальные дубликаты; автоматические проверки "
          "не заменяют визуальную оценку. Координаты сохранены в исходном соглашении XML.\\n")
(OUT / "findings.txt").write_text(report, encoding="utf-8")
print(report)
provenance([{"page": "https://www.robots.ox.ac.uk/~vgg/data/pets/",
             "url": BASE, "license": "CC BY-SA 4.0",
             "citation": "Parkhi et al., Cats and Dogs, CVPR 2012"}],
           [annotation_archive, image_archive])
'''),
]


def write(name, cells):
    document = {"nbformat": 4, "nbformat_minor": 5,
        "metadata": {"colab": {"name": name, "provenance": []},
                     "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.11"}}, "cells": cells}
    for index, cell in enumerate(cells):
        cell["id"] = f"cell-{index:03d}"
    target = ROOT / "notebooks" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(document, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(target)


if __name__ == "__main__":
    write("01_text_eda.ipynb", TEXT)
    write("02_images_eda.ipynb", IMAGES)
