import json
import urllib.request
from pathlib import Path

DART_BASE = 'https://raw.githubusercontent.com/Yale-LILY/dart/master/data/v1.1.1'
DART_FILES = {
    'train': 'dart-v1.1.1-full-train.json',
    'dev':   'dart-v1.1.1-full-dev.json',
    'test':  'dart-v1.1.1-full-test.json',
}

def _serialize_tripleset(tripleset):
    return ' | '.join(f'{t[0]} : {t[1].lower()} : {t[2]}' for t in tripleset)

def load_dart(data_dir, version='v1.1.1'):
    assert version == 'v1.1.1', 'only v1.1.1 supported'
    data_dir = Path(data_dir); data_dir.mkdir(parents=True, exist_ok=True)
    splits = {}
    for split, fname in DART_FILES.items():
        local = data_dir / fname
        if not local.exists():
            url = f'{DART_BASE}/{fname}'
            print(f'Downloading {url} -> {local}')
            urllib.request.urlretrieve(url, local)
        with open(local, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        rows = []
        for entry in raw:
            tripleset = entry.get('tripleset') or []
            if not tripleset:
                continue
            src = _serialize_tripleset(tripleset)
            refs = [a['text'].strip() for a in entry.get('annotations', []) if a.get('text')]
            if not refs:
                continue
            rows.append({'src': src, 'refs': refs})
        splits[split] = rows
    return splits['train'], splits['dev'], splits['test']