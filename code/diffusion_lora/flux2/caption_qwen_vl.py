"""Re-caption all 271 images using Qwen2-VL-7B-Instruct (local, ~14 GB VRAM).

Captions are dramatically better than BLIP-2 because Qwen2-VL is an
instruction-tuned VLM trained for detailed scene description, not just
single-sentence image-text alignment.

Output: data/diffusion_lora/train/metadata.jsonl (overwrites; backup in metadata.bak.jsonl)

Usage:
    python caption_qwen_vl.py
    python caption_qwen_vl.py --resume      # skip already-captioned files
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from PIL import Image
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA = ROOT / 'data' / 'diffusion_lora' / 'train'
META = DATA / 'metadata.jsonl'

TRIGGER = 'wbronkhorst style'

# Tightly scoped instruction so Qwen describes painting CONTENT (not the gallery
# room around it). Keeps style descriptors out so the trigger word handles style.
INSTRUCTION = (
    "Describe the visual content of this painting in one short sentence (max 20 words). "
    "Focus only on what's painted (figures, objects, setting, colors). "
    "Do NOT describe the room, wall, frame, or any objects around the painting. "
    "Do NOT use words like 'painting', 'art', 'artwork', 'canvas' — describe the scene directly. "
    "Be specific about the subject and any action."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--model', default='Qwen/Qwen2-VL-7B-Instruct')
    args = ap.parse_args()

    images = sorted([p for p in DATA.iterdir() if p.suffix.lower() == '.jpg'])
    print(f'Found {len(images)} images')

    done: set[str] = set()
    if args.resume and META.exists():
        with open(META, encoding='utf-8') as f:
            for line in f:
                done.add(json.loads(line)['file_name'])
        print(f'Resume: {len(done)} already captioned')

    # Backup existing metadata before overwriting
    if META.exists() and not (DATA / 'metadata.bak2.jsonl').exists():
        (DATA / 'metadata.bak2.jsonl').write_text(META.read_text(encoding='utf-8'), encoding='utf-8')
        print(f'Backed up existing -> metadata.bak2.jsonl')

    print(f'Loading {args.model} (first run downloads ~16 GB)...')
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map='auto'
    )
    proc = AutoProcessor.from_pretrained(args.model)
    print(f'Model on {next(model.parameters()).device}')

    out_lines = []
    if args.resume and META.exists():
        with open(META, encoding='utf-8') as f:
            out_lines = list(f)

    mode = 'a' if args.resume else 'w'
    with open(META, mode, encoding='utf-8') as f:
        for p in tqdm(images, desc='caption'):
            if p.name in done:
                continue
            try:
                img = Image.open(p).convert('RGB')
                # Qwen2-VL expects messages with embedded image
                msgs = [{
                    'role': 'user',
                    'content': [
                        {'type': 'image', 'image': img},
                        {'type': 'text', 'text': INSTRUCTION},
                    ],
                }]
                text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                inputs = proc(text=[text], images=[img], return_tensors='pt').to(model.device)
                with torch.no_grad():
                    gen = model.generate(**inputs, max_new_tokens=60, do_sample=False)
                # Strip the prompt portion
                out_ids = gen[0][inputs.input_ids.shape[1]:]
                description = proc.decode(out_ids, skip_special_tokens=True).strip()
                # Tighten: strip leading articles/fillers
                description = description.lstrip('"\'').rstrip('"\'').strip()
                if description.lower().startswith(('a ', 'an ', 'the ')):
                    description = description.split(' ', 1)[1] if ' ' in description else description
                if description.endswith('.'):
                    description = description[:-1]
                rec = {'file_name': p.name, 'text': f'{TRIGGER}, {description}'}
                f.write(json.dumps(rec) + '\n')
                f.flush()
            except Exception as e:
                print(f'  skip {p.name}: {e}')
                continue

    print(f'\nDone. Wrote {META}')
    print('Sample output:')
    with open(META, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < 5:
                print(' ', json.loads(line))


if __name__ == '__main__':
    main()
