"""3-way A/B grid: v1 final (step 3000) vs v2 final (step 5000) vs v3 final (step 5000).

Each row is a prompt; each column is a version. All 8 prompts.
"""
import re
from collections import defaultdict
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent.parent.parent
MODELS = ROOT / 'models' / 'diffusion_lora'
OUT = ROOT / 'results' / 'diffusion_lora' / 'flux2_v1_v2_v3_3way.png'

VERSIONS = [
    ('v1 baseline (step 3000)', MODELS / 'klein_9b_v1' / 'samples', 3000),
    ('v2 (step 5000)', MODELS / 'klein_9b_v2' / 'samples', 5000),
    ('v3 (step 5000)', MODELS / 'klein_9b_v3' / 'samples', 5000),
]

PROMPTS_FILE = Path(__file__).parent / 'prompts.txt'
PAD, TITLE_H, LABEL_H, CAP_H = 18, 70, 50, 30
CELL_W = 600


def load_prompts():
    if not PROMPTS_FILE.exists():
        return [f'prompt {i}' for i in range(8)]
    return [l.strip() for l in PROMPTS_FILE.read_text(encoding='utf-8').splitlines()
            if l.strip() and not l.startswith('#')]


def find_sample(samples_dir, step, prompt_idx):
    pad = f'{step:09d}'
    for f in samples_dir.glob(f'*__{pad}_{prompt_idx}.*'):
        return f
    return None


def font(size):
    for c in ['C:/Windows/Fonts/arial.ttf', '/System/Library/Fonts/Helvetica.ttc',
              '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']:
        if Path(c).exists():
            return ImageFont.truetype(c, size)
    return ImageFont.load_default()


def main():
    prompts = load_prompts()
    n_prompts = 8
    n_cols = len(VERSIONS)

    # Determine cell height from a sample image's aspect
    sample_path = find_sample(VERSIONS[2][1], VERSIONS[2][2], 0)
    if not sample_path:
        raise SystemExit('No sample for v3 step 0 - check paths')
    sample = Image.open(sample_path)
    cell_w = CELL_W
    cell_h = int(cell_w * sample.height / sample.width)

    grid_w = n_cols * cell_w + (n_cols + 1) * PAD
    grid_h = TITLE_H + LABEL_H + n_prompts * (cell_h + CAP_H + PAD) + PAD
    img = Image.new('RGB', (grid_w, grid_h), 'white')
    d = ImageDraw.Draw(img)

    title_f = font(28)
    label_f = font(20)
    cap_f = font(13)

    d.text((PAD, PAD), 'WB LoRA on Flux.2 Klein 9B  —  3-way A/B: v1 vs v2 vs v3 (final checkpoints)',
           fill='black', font=title_f)
    for i, (label, _, _) in enumerate(VERSIONS):
        x = PAD + i * (cell_w + PAD)
        d.text((x, TITLE_H), label, fill='black', font=label_f)

    y = TITLE_H + LABEL_H
    for pidx in range(n_prompts):
        for col_i, (_, samples_dir, step) in enumerate(VERSIONS):
            f = find_sample(samples_dir, step, pidx)
            x = PAD + col_i * (cell_w + PAD)
            if f and f.exists():
                im = Image.open(f).convert('RGB').resize((cell_w, cell_h))
                img.paste(im, (x, y))
            else:
                d.rectangle([x, y, x + cell_w, y + cell_h], outline='red')
                d.text((x + 10, y + 10), f'missing: step {step} p{pidx}', fill='red', font=cap_f)
        prompt_text = (prompts[pidx] if pidx < len(prompts) else f'prompt {pidx}')[:200]
        d.text((PAD, y + cell_h + 4), prompt_text, fill='gray', font=cap_f)
        y += cell_h + CAP_H + PAD

    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT)
    print(f'wrote {OUT}  ({img.size[0]}x{img.size[1]})')


if __name__ == '__main__':
    main()
