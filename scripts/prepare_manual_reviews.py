"""Prepare blind annotation batches from the licensed external review corpus."""
import hashlib
import json
import os
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STORAGE = Path(os.environ.get('COURSEWORK_DATA_ROOT', ROOT / '.local-data')).resolve()
RAW = STORAGE / 'raw/manual_reviews'
OUT = Path(os.environ.get('COURSEWORK_ANNOTATION_OUT', ROOT / 'annotation/text'))
OUT.mkdir(parents=True, exist_ok=True)
source = RAW / 'Womens Clothing E-Commerce Reviews.csv'
df = pd.read_csv(source)
source_rows = len(df)
df['source_row'] = df.index
df = df.dropna(subset=['Review Text']).copy()
def norm(s):
    return ' '.join(re.findall(r'\w+', str(s).casefold()))
df['normalized'] = df['Review Text'].map(norm)
df['words'] = df['Review Text'].str.split().str.len()
df = df[df.words.between(5,80)].drop_duplicates('normalized')
original = pd.read_csv(STORAGE / 'processed/text/reviews_original.csv')
originals = [norm(t) for t in original.text]
df = df[~df.normalized.isin(set(originals))]
# Rating is used only for balanced candidate selection, never as an annotation.
buckets = [df[df.Rating.isin([1,2])], df[df.Rating.isin([4,5])]]
selected = []
for bucket in buckets:
    accepted = []
    for _, row in bucket.sample(frac=1, random_state=42).iterrows():
        text = row.normalized
        if any(text in other or other in text for other in originals if other):
            continue
        accepted.append(row)
        if len(accepted) == 80:
            break
    assert len(accepted) == 80
    selected.append(pd.DataFrame(accepted))

metadata = []
for name, start, stop in [('pilot',0,10),('main',10,70),('reserve',70,80)]:
    batch = pd.concat([b.iloc[start:stop] for b in selected]).sample(frac=1,random_state=73)
    records = []
    for _, row in batch.iterrows():
        identifier = 'review_' + hashlib.sha256(f"clothing:{row.source_row}".encode()).hexdigest()[:12]
        records.append({'data': {'id':identifier,'text':row['Review Text']}})
        metadata.append({'id':identifier,'batch':name,'source_row':int(row.source_row),
                         'source':'Womens E-Commerce Clothing Reviews',
                         'url':'https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews',
                         'license_on_source_page':'CC0: Public Domain','access_date':'2026-11-08'})
    (OUT / f'{name}.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    pd.DataFrame([r['data'] for r in records]).to_csv(OUT / f'{name}.csv',index=False)
pd.DataFrame(metadata).to_csv(OUT/'sources.csv',index=False)
summary={'source_rows':source_rows,'pilot':20,'main':120,'reserve':20,'seed':42,
         'archive_sha256':hashlib.sha256((RAW/'clothing_reviews.zip').read_bytes()).hexdigest(),
         'checks':['5–80 words','nonempty text','normalized duplicate removal',
                   'no normalized exact matches or substrings with original UCI'],
         'limitations':'No semantic duplicate detection; final labels require two independent human annotations.'}
(OUT/'selection_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
