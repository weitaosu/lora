"""Strip "painting" filler from Qwen2-VL captions.

Qwen2-VL kept emitting "painting depicts X", "textured painting features X", etc.
For style-LoRA training the trigger word should carry the "painting / Bronkhorst"
association, so we drop those words from the captions and let the trigger do its job.

Rules applied (in order):
  - "X painting depicts Y"             -> "X Y"
  - "(textured/vibrant/abstract) painting depicts/features/shows Y" -> "Y"
  - leading "painting depicts/features/shows Y" -> "Y"
  - "Y is depicted/painted/shown" -> "Y"
  - normalize whitespace, lowercase first letter, ensure trigger prefix
"""
import json
import re
from pathlib import Path

DATA = Path('C:/Users/weita/Desktop/deep_learning/final_proj/data/wb_lora_train')
TRIGGER = 'wbronkhorst style'


def clean(text: str) -> str:
    # Strip trigger so we work on the body
    if text.lower().startswith(TRIGGER.lower()):
        body = text[len(TRIGGER):].lstrip(',').strip()
    else:
        body = text

    # Strip "(adjective(s)) painting (depicts|features|shows|illustrates) "
    body = re.sub(
        r'^\s*(\w+\s+){0,3}painting\s+(depicts|features|shows|illustrates|portrays)\s+',
        '', body, flags=re.I,
    )
    # Strip "painting of "
    body = re.sub(r'^\s*(\w+\s+){0,2}painting\s+of\s+', '', body, flags=re.I)
    # Strip "an artwork of/with " etc.
    body = re.sub(r'^\s*an?\s+(\w+\s+)?(artwork|illustration|sketch|drawing|print)\s+(of|with|showing)\s+',
                  '', body, flags=re.I)
    # Drop "is depicted/painted/shown" passive constructions
    body = re.sub(r'\s+(is|are)\s+(depicted|painted|shown|illustrated|portrayed)\s*', ' ', body, flags=re.I)
    # Trim "this is/are a..." or "in this image"
    body = re.sub(r'^\s*(in\s+this\s+image,?\s+|this\s+(is|shows)\s+a?n?\s*)', '', body, flags=re.I)
    # Drop redundant adjective+canvas phrasings
    body = re.sub(r',\s*painted\s+with\s+(thick\s+)?brush(\s*\w+)?\s*$', '', body, flags=re.I)

    # Whitespace + punctuation
    body = re.sub(r'\s+', ' ', body).strip(' ,.\t\r\n')
    # Lowercase first letter (Qwen sometimes capitalizes)
    if body and body[0].isupper() and not body.startswith(('A ', 'I ')):
        body = body[0].lower() + body[1:]
    if not body:
        body = 'figures on a textured surface'

    return f'{TRIGGER}, {body}'


def main():
    META = DATA / 'metadata.jsonl'
    BAK  = DATA / 'metadata.bak3.jsonl'
    if not BAK.exists():
        BAK.write_text(META.read_text(encoding='utf-8'), encoding='utf-8')
        print(f'Backed up Qwen raw -> {BAK}')

    recs = [json.loads(l) for l in open(META, encoding='utf-8')]
    changed = 0
    for r in recs:
        before = r['text']
        after = clean(before)
        if before != after:
            changed += 1
        r['text'] = after

    with open(META, 'w', encoding='utf-8') as f:
        for r in recs:
            f.write(json.dumps(r) + '\n')

    print(f'Cleaned {changed}/{len(recs)} captions')
    print('\nSample 8 cleaned:')
    import random
    random.seed(13)
    for r in random.sample(recs, 8):
        print(f'  {r["text"]}')


if __name__ == '__main__':
    main()
