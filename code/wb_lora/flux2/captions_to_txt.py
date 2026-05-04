"""Convert data/wb_lora_train/metadata.jsonl (HF datasets format) into per-image
sidecar .txt files (the format ai-toolkit expects).

Before:
    data/wb_lora_train/img1.jpg
    data/wb_lora_train/metadata.jsonl    # {"file_name":"img1.jpg","text":"wbronkhorst style, ..."}

After:
    data/wb_lora_train/img1.jpg
    data/wb_lora_train/img1.txt          # "wbronkhorst style, ..."

Idempotent — re-running just overwrites .txt files in place.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent  # final_proj/
DATA = ROOT / 'data' / 'wb_lora_train'
META = DATA / 'metadata.jsonl'


def main():
    if not META.exists():
        print(f'Missing {META}.')
        print('Run:  python ../caption_images.py   first')
        sys.exit(1)

    n = 0
    with open(META, encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            fn = rec['file_name']
            text = rec['text']
            txt_path = DATA / (Path(fn).stem + '.txt')
            txt_path.write_text(text, encoding='utf-8')
            n += 1

    print(f'Wrote {n} .txt sidecar files into {DATA}')
    print(f'Sample:')
    sample = next(DATA.glob('*.txt'))
    print(f'  {sample.name}: {sample.read_text(encoding="utf-8")}')


if __name__ == '__main__':
    main()
