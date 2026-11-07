"""Build the self-contained Drive preparation notebook, without review payloads."""
import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
def md(text):
    return {'cell_type':'markdown','metadata':{},'source':textwrap.dedent(text).strip()+'\n'}
def code(text):
    return {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],
            'source':textwrap.dedent(text).strip()+'\n'}

selection = (ROOT/'scripts/prepare_manual_reviews.py').read_text(encoding='utf-8')
selection = selection.replace('ROOT = Path(__file__).resolve().parents[1]', 'ROOT = STORAGE')
selection = selection.replace("STORAGE = Path(os.environ.get('COURSEWORK_DATA_ROOT', ROOT / '.local-data')).resolve()", '')
selection = selection.replace("OUT = Path(os.environ.get('COURSEWORK_ANNOTATION_OUT', ROOT / 'annotation/text'))", "OUT = STORAGE / 'manual/text'")
config = (ROOT/'annotation/text/label_studio.xml').read_text(encoding='utf-8')
instructions = (ROOT/'annotation/text/README.md').read_text(encoding='utf-8')
cells = [md('''
# Подготовка дополнительных отзывов для независимой разметки

Источник: [Women's E-Commerce Clothing Reviews](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews).
На странице публикации указана лицензия CC0: Public Domain.
Корпус выбран из альтернатив, перечисленных в codebook.

Отбираются полные англоязычные отзывы длиной 5–80 слов. Формируются непересекающиеся
партии: 20 пробных, 120 основных и 20 резервных объектов. Предварительный отбор по
рейтингам 1–2 и 4–5 используется для формирования пула, но не определяет итоговые
человеческие метки. Рейтинги и рекомендации не включаются в файлы разметки.

Исходный UCI используется только для проверки пересечений. Источники, параметры
отбора и лицензия фиксируются отдельно. Семантические дубликаты данной проверкой
полностью не исключаются. Добавление отзывов об одежде изменяет тематический состав набора.
'''),code('''
from pathlib import Path
import os
import io
import zipfile
import requests
try:
    from google.colab import drive
except ImportError:
    STORAGE = Path(os.environ.get('COURSEWORK_DATA_ROOT', '.local-data')).resolve()
else:
    drive.mount('/content/drive')
    STORAGE = Path('/content/drive/MyDrive/data-enrichment-coursework')
original = STORAGE / 'processed/text/reviews_original.csv'
if not original.exists():
    raise FileNotFoundError('Не найдена исходная таблица UCI: ' + str(original))
RAW = STORAGE / 'raw/manual_reviews'
RAW.mkdir(parents=True, exist_ok=True)
'''),md('## 1. Загрузка внешнего источника'),code('''
URL = 'https://www.kaggle.com/api/v1/datasets/download/nicapotato/womens-ecommerce-clothing-reviews'
archive = RAW / 'clothing_reviews.zip'
source = RAW / 'Womens Clothing E-Commerce Reviews.csv'
if not archive.exists():
    response = requests.get(URL, timeout=120)
    response.raise_for_status()
    # Проверка архива до сохранения исключает HTML-страницу авторизации.
    with zipfile.ZipFile(io.BytesIO(response.content)) as bundle:
        content = bundle.read('Womens Clothing E-Commerce Reviews.csv')
    archive.write_bytes(response.content)
    source.write_bytes(content)
else:
    with zipfile.ZipFile(archive) as bundle:
        source.write_bytes(bundle.read('Womens Clothing E-Commerce Reviews.csv'))
'''),md('## 2. Проверка пересечений и формирование партий'),code(selection),
md('## 3. Документирование источника и интерфейс разметки'),
code(f'''\n(OUT / 'label_studio.xml').write_text({config!r}, encoding='utf-8')\n(OUT / 'README.md').write_text({instructions!r}, encoding='utf-8')\n'''),
code('''
source_record = {
    'name': "Women's E-Commerce Clothing Reviews",
    'url': 'https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews',
    'publisher': 'nicapotato (Kaggle)',
    'license_on_publisher_page': 'CC0: Public Domain',
    'license_url': 'https://creativecommons.org/publicdomain/zero/1.0/',
    'source_identity': 'Anonymous but real commercial source, according to publisher',
    'scope': 'Additional real product reviews; no translation or generated text',
    'archive_sha256': summary['archive_sha256'],
    'selection_seed': 42,
}
(OUT/'source_license.json').write_text(json.dumps(source_record,ensure_ascii=False,indent=2),encoding='utf-8')
tasks=[]
for name, expected in [('pilot',20),('main',120),('reserve',20)]:
    rows=json.loads((OUT/f'{name}.json').read_text(encoding='utf-8'))
    assert len(rows)==expected
    assert all(set(r)=={'data'} and set(r['data'])=={'id','text'} for r in rows)
    tasks.extend(rows)
assert len({r['data']['id'] for r in tasks}) == 160
assert len({norm(r['data']['text']) for r in tasks}) == 160
print('Пробная партия: 20; основная: 120; резерв: 20.')
print('Каталог файлов разметки:', OUT)
''')]
for i,c in enumerate(cells):c['id']=f'manual-{i:03d}'
doc={'nbformat':4,'nbformat_minor':5,'metadata':{
    'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
    'colab':{'name':'03_prepare_manual_reviews.ipynb','provenance':[]}},'cells':cells}
(ROOT/'notebooks/03_prepare_manual_reviews.ipynb').write_text(json.dumps(doc,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
