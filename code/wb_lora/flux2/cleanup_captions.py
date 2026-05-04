"""Rule-based caption cleanup + outlier flagging.

Reads metadata.jsonl, applies these rules to each caption (in order):
  1. Strip leading filler: "there is/are", "some"
  2. Strip redundant "a painting of/with" — the trigger word already means painting
  3. Strip generic background filler ("on a green background")
  4. Collapse double commas + whitespace
  5. Re-add the trigger prefix at the front
  6. Lowercase first word for consistency

Then flags captions matching outlier heuristics so we can manually fix them:
  - Mentions a real-world frame/wall/shelf (image-of-image, gallery shot)
  - Length < 6 words (low-info)
  - Mentions abstract/blank/empty (likely thumbnail or template)
  - Misclassified subject (rock, stone, plate of food in a non-food context)

Writes:
  metadata.jsonl                    -> overwritten with cleaned captions
  metadata.bak.jsonl                -> backup of pre-cleanup
  outliers_to_review.txt            -> list of (file_name, reason, caption) for manual fix
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent   # final_proj/
DATA = ROOT / 'data' / 'wb_lora_train'
META = DATA / 'metadata.jsonl'
BAK  = DATA / 'metadata.bak.jsonl'
OUTLIERS = Path(__file__).parent / 'outliers_to_review.txt'

TRIGGER = 'wbronkhorst style'

# Strip patterns (applied in order, case-insensitive)
STRIP_PATTERNS = [
    r'^there\s+(is|are)\s+',
    r'^some\s+',
    r'^a\s+(framed\s+)?(abstract\s+|colorful\s+|small\s+|large\s+)?'
    r'(painting|print|artwork|canvas|illustration|drawing|sketch)\s+(of|with|showing|depicting)\s+',
    r'^an\s+(abstract\s+|colorful\s+|small\s+|large\s+)?'
    r'(painting|print|artwork|canvas|illustration|drawing|sketch)\s+(of|with|showing|depicting)\s+',
    r'\s+on\s+(a\s+|an\s+)?(green|blue|white|black|grey|gray)\s+background\s*$',
    r'^a\s+picture\s+of\s+',
]

# Outlier triggers — captions matching these need manual review
OUTLIER_PATTERNS = {
    'image_of_image': re.compile(r'\b(wall|frame|framed|shelf|displayed|hung|mantle|gallery wall|on\s+the\s+wall|record player|easel)\b', re.I),
    'low_info':       None,   # special-cased on length below
    'abstract_or_blank': re.compile(r'\b(blank|empty room|empty space|white canvas|blank canvas|abstract painting)\b', re.I),
    'misclassified':  re.compile(r'\b(piece of rock|piece of stone|plate of food|computer screen|laptop|phone|television)\b', re.I),
    'wrong_medium':   re.compile(r'\b(photograph|photo of|polaroid)\b', re.I),
}


def cleanup_one(text: str) -> str:
    # Strip the trigger if present so we can re-add at the end consistently
    if text.lower().startswith(TRIGGER.lower()):
        text = text[len(TRIGGER):].lstrip(',').strip()

    for pat in STRIP_PATTERNS:
        text = re.sub(pat, '', text, flags=re.I)

    # Collapse whitespace + commas
    text = re.sub(r'\s*,\s*,+', ', ', text)
    text = re.sub(r'\s+', ' ', text).strip().strip(',').strip()

    # Re-add trigger as a prefix
    return f'{TRIGGER}, {text}' if text else f'{TRIGGER}, painting'


def is_outlier(text: str) -> list[str]:
    reasons = []
    body = text.lower().replace(TRIGGER.lower() + ',', '').strip()
    if len(body.split()) < 6:
        reasons.append('low_info')
    for name, pat in OUTLIER_PATTERNS.items():
        if pat is None:
            continue
        if pat.search(body):
            reasons.append(name)
    return reasons


def main():
    if not META.exists():
        raise SystemExit(f'Missing {META}. Run caption_images.py first.')

    # Backup original
    if not BAK.exists():
        BAK.write_text(META.read_text(encoding='utf-8'), encoding='utf-8')
        print(f'Backed up original -> {BAK}')

    recs = [json.loads(l) for l in open(META, encoding='utf-8')]
    print(f'Cleaning {len(recs)} captions...')

    cleaned, changed = [], 0
    outliers = []
    for r in recs:
        before = r['text']
        after  = cleanup_one(before)
        if before != after:
            changed += 1
        r['text'] = after
        cleaned.append(r)

        reasons = is_outlier(after)
        if reasons:
            outliers.append((r['file_name'], reasons, after))

    with open(META, 'w', encoding='utf-8') as f:
        for r in cleaned:
            f.write(json.dumps(r) + '\n')

    print(f'Cleanup applied: {changed}/{len(recs)} captions modified')
    print(f'Wrote {META}')

    print(f'\n{len(outliers)} outliers flagged for manual review -> {OUTLIERS}')
    with open(OUTLIERS, 'w', encoding='utf-8') as f:
        for fn, reasons, txt in outliers:
            f.write(f'{fn}\t{",".join(reasons)}\t{txt}\n')

    # Show a sample of cleaned + flagged
    print('\n=== Sample cleaned captions ===')
    for r in cleaned[:5]:
        print(f'  {r["file_name"][:50]}: {r["text"]}')
    print('\n=== First 10 outliers ===')
    for fn, reasons, txt in outliers[:10]:
        print(f'  [{",".join(reasons)}] {fn[:50]}: {txt}')


if __name__ == '__main__':
    main()
