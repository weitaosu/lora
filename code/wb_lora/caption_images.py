"""Auto-caption every image in data/wb_lora_train/ with BLIP, prepend a trigger
word, and write metadata.jsonl in the format the diffusers LoRA trainer expects.

Output: data/wb_lora_train/metadata.jsonl with one line per image:
    {"file_name": "img1.jpg", "text": "wbronkhorst style, a painting of a..."}

The trigger word `wbronkhorst` is what you'll type at inference time to invoke
the trained style. Change `TRIGGER` below if you want a different one.

Usage:
    python caption_images.py
    python caption_images.py --resume         # skip already-captioned files
    python caption_images.py --model blip2    # use BLIP-2 (better, ~7GB VRAM)
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from PIL import Image
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent   # final_proj/
DATA = ROOT / 'data' / 'wb_lora_train'
META = DATA / 'metadata.jsonl'

TRIGGER = 'wbronkhorst style'   # change this if you want a different trigger word


def load_blip(model: str):
    if model == 'blip2':
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        mid = 'Salesforce/blip2-opt-2.7b'
        proc = Blip2Processor.from_pretrained(mid)
        m = Blip2ForConditionalGeneration.from_pretrained(mid, torch_dtype=torch.float16)
    else:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        mid = 'Salesforce/blip-image-captioning-large'
        proc = BlipProcessor.from_pretrained(mid)
        m = BlipForConditionalGeneration.from_pretrained(mid, torch_dtype=torch.float16)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = m.to(device).eval()
    return proc, m, device


def caption(img: Image.Image, proc, model, device) -> str:
    inputs = proc(images=img, return_tensors='pt').to(device, torch.float16)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=40, num_beams=3)
    text = proc.batch_decode(out, skip_special_tokens=True)[0].strip()
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['blip', 'blip2'], default='blip',
                    help='blip = lightweight (~1GB VRAM); blip2 = better (~7GB VRAM)')
    ap.add_argument('--resume', action='store_true',
                    help='skip files already captioned in metadata.jsonl')
    args = ap.parse_args()

    images = sorted([p for p in DATA.iterdir()
                     if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}])
    print(f'Found {len(images)} images in {DATA}')

    done: set[str] = set()
    if args.resume and META.exists():
        with open(META, encoding='utf-8') as f:
            for line in f:
                rec = json.loads(line)
                done.add(rec['file_name'])
        print(f'Resume: {len(done)} already captioned, skipping those.')

    proc, model, device = load_blip(args.model)
    print(f'Loaded {args.model} on {device}')

    mode = 'a' if args.resume else 'w'
    with open(META, mode, encoding='utf-8') as f:
        for p in tqdm(images, desc='caption'):
            if p.name in done:
                continue
            try:
                with Image.open(p) as im:
                    im = im.convert('RGB')
                    text = caption(im, proc, model, device)
            except Exception as e:
                print(f'  skip {p.name}: {e}')
                continue
            rec = {'file_name': p.name, 'text': f'{TRIGGER}, {text}'}
            f.write(json.dumps(rec) + '\n')
            f.flush()

    print(f'\nWrote {META}')
    print('Sample:')
    with open(META, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < 3:
                print(' ', json.loads(line))


if __name__ == '__main__':
    main()
