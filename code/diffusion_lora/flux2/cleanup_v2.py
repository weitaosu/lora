"""Aggressive caption cleanup pass 2.

Pattern observed in Bronkhorst photos: many are paintings hanging in rooms
(gallery / home decor shots). The painting CONTENT is what we want; the room
is noise. We strip room/wall context and force-rewrite a known misclassification.

Run AFTER cleanup_captions.py.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent  # final_proj/
META = ROOT / 'data' / 'diffusion_lora' / 'train' / 'metadata.jsonl'

TRIGGER = 'wbronkhorst style'

# Strip room/wall context from captions (they describe the gallery photo, not the painting)
ROOM_STRIPS = [
    r'\s+(hangs|hung|hanging)\s+(on|above|in)\s+(the|a|an)\s+\w+(\s+\w+)?\s*$',
    r'\s+on\s+(the|a|an)\s+wall(\s+in\s+(a|an|the)\s+(\w+\s+)?\w+)?\s*$',
    r'\s+(in|on)\s+(a|an|the)\s+(living\s+room|office|bedroom|kitchen|gallery|home|hallway)\s*$',
    r'\s+above\s+(a|an|the)\s+\w+(\s+\w+)*\s*$',
    r'\bin\s+a\s+living\s+room\b',
    r'\bon\s+(the\s+)?wall\b',
    r'\bin\s+an?\s+office\b',
    r'^a\s+pink\s+wall\s+with\s+a\s+record\s+player\s+and\s+',
    r'\s+next\s+to\s+(a|an|the)\s+\w+(\s+\w+)*\s*$',
    r'\s+with\s+(a|an|the)\s+(record\s+player|lamp|couch|chair|table|bed|sofa|tv|television)(\s+\w+)*\s*$',
]

# Misclassifications confirmed by viewing — explicit overrides
EXPLICIT_OVERRIDES = {
    'bing__Werner_Bronkhorst__painting_high_resolu__000044.jpg':
        'tiny surfers on a thick blue wave of impasto paint, sculptural piece, hand holding a sculpted wedge',
    'bing_Werner_Bronkhorst_artwork__000040.jpg':
        'tiny surfers on a thick blue wave of impasto paint, sculptural piece',
    'bing_Werner_Bronkhorst_beach_painting__000036.jpg':
        'tiny surfers on a thick blue wave of impasto paint, sculptural piece',
    'bing_Werner_Bronkhorst_figure_painting__000018.jpg':
        'tiny surfers on a thick blue wave of impasto paint, sculptural piece',
    # Common image-of-image patterns confirmed by viewing
    'bing__Werner_Bronkhorst__painting_high_resolu__000026.jpg':
        'aerial view of three tiny surfers paddling on blue ocean waves',
    'bing_Werner_Bronkhorst_art__000012.jpg':
        'aerial view of tiny surfers paddling on blue ocean waves',
    'bing_Werner_Bronkhorst_artwork__000034.jpg':
        'tiny golfers on thick green impasto, golf course, sculptural paint',
}


def clean(text: str) -> str:
    """Strip the trigger, apply all room patterns, re-add trigger."""
    if text.lower().startswith(TRIGGER.lower()):
        text = text[len(TRIGGER):].lstrip(',').strip()
    for pat in ROOM_STRIPS:
        text = re.sub(pat, '', text, flags=re.I)
    text = re.sub(r'\s+', ' ', text).strip().rstrip(',').strip()
    if not text:
        text = 'painting'
    return f'{TRIGGER}, {text}'


def main():
    recs = [json.loads(l) for l in open(META, encoding='utf-8')]
    print(f'Loaded {len(recs)} records')

    edits, overrides = 0, 0
    for r in recs:
        if r['file_name'] in EXPLICIT_OVERRIDES:
            r['text'] = f'{TRIGGER}, {EXPLICIT_OVERRIDES[r["file_name"]]}'
            overrides += 1
            continue
        before = r['text']
        after = clean(before)
        if before != after:
            edits += 1
        r['text'] = after

    with open(META, 'w', encoding='utf-8') as f:
        for r in recs:
            f.write(json.dumps(r) + '\n')

    print(f'rule edits:        {edits}')
    print(f'explicit overrides: {overrides}')

    # Show a sample of cleaned outliers (using the same outlier file from pass 1)
    out = Path(__file__).parent / 'outliers_to_review.txt'
    if out.exists():
        seen = {l.split('\t')[0] for l in out.read_text(encoding='utf-8').splitlines()}
        print('\n=== After-cleanup sample of previously-flagged outliers ===')
        n = 0
        for r in recs:
            if r['file_name'] in seen and n < 12:
                print(f'  {r["file_name"][:55]}: {r["text"]}')
                n += 1


if __name__ == '__main__':
    main()
