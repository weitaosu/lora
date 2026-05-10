"""Final showcase: user-curated baseline-vs-LoRA grid.

User picked 11 of 20 prompts (some with v1, some with v3) — render as a wide
2-row grid: top = baseline, bottom = the chosen LoRA. No version label per
column (just "LoRA").
"""
import re
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent.parent.parent
RESULTS = ROOT / 'results' / 'diffusion_lora'
OUT = RESULTS / 'flux2_showcase_baseline_vs_lora.png'

# (0-indexed prompt, chosen version key)
PICKS = [
    (0,  'v1'),  # lifeguard at sunset
    (1,  'v3'),  # surfer riding wave
    (3,  'v1'),  # woman + vintage car
    (5,  'v1'),  # knight in armor
    (6,  'v1'),  # city street
    (7,  'v1'),  # chef
    (8,  'v3'),  # aerial surfers
    (9,  'v1'),  # green golf course
    (10, 'v1'),  # ski slope
    (14, 'v1'),  # F1 car winding road
    (16, 'v1'),  # swimmer in pool
]

VERSION_DIRS = {
    'baseline': RESULTS / 'inference_baseline' / 'samples',
    'v1':       RESULTS / 'inference_v1' / 'samples',
    'v2':       RESULTS / 'inference_v2' / 'samples',
    'v3':       RESULTS / 'inference_v3' / 'samples',
}

PROMPTS_FILE = Path(__file__).parent / 'compare_20_prompts.txt'

PAD = 22
TITLE_H = 110
ROW_LABEL_W = 180
CAP_H = 70
CELL_W = 460


def load_prompts():
    """Returns list of (prompt_text, seed) parsed from the file."""
    out = []
    for line in PROMPTS_FILE.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        text = line.split(' --')[0].strip()
        seed = ''
        for tok in line.split('--'):
            if tok.strip().startswith('seed '):
                seed = tok.strip().split()[1]
        out.append((text, seed))
    return out


def find_sample(samples_dir, prompt_idx):
    pad = '000000000'
    matches = list(samples_dir.glob(f'*__{pad}_{prompt_idx}.*'))
    return matches[0] if matches else None


def font(size, bold=False):
    candidates = (['C:/Windows/Fonts/arialbd.ttf'] if bold else []) + [
        'C:/Windows/Fonts/arial.ttf',
        '/System/Library/Fonts/Helvetica.ttc',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold else
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    ]
    for c in candidates:
        if Path(c).exists():
            return ImageFont.truetype(c, size)
    return ImageFont.load_default()


def wrap(text, draw, fnt, max_w):
    words, lines, cur = text.split(' '), [], ''
    for w in words:
        trial = (cur + ' ' + w).strip()
        if draw.textlength(trial, font=fnt) <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def main():
    prompts = load_prompts()
    n_cols = len(PICKS)

    # Determine cell dimensions from first sample
    sample = Image.open(find_sample(VERSION_DIRS['baseline'], PICKS[0][0]))
    cell_w, cell_h = CELL_W, int(CELL_W * sample.height / sample.width)

    grid_w = ROW_LABEL_W + n_cols * cell_w + (n_cols + 1) * PAD
    grid_h = TITLE_H + 2 * (cell_h + PAD) + CAP_H + PAD

    img = Image.new('RGB', (grid_w, grid_h), 'white')
    d = ImageDraw.Draw(img)
    title_f = font(38, bold=True)
    label_f = font(28, bold=True)
    cap_f = font(13)

    d.text((PAD, PAD + 20),
           'WB LoRA on Flux.2 Klein 9B  —  Baseline vs LoRA',
           fill='black', font=title_f)

    rows = [
        ('Baseline', 'baseline'),
        ('LoRA', None),  # version differs per column
    ]

    y = TITLE_H
    for r_i, (row_label, fixed_key) in enumerate(rows):
        # Row label, vertically centered
        lines = wrap(row_label, d, label_f, ROW_LABEL_W - PAD)
        line_h = label_f.size + 4
        block_h = line_h * len(lines)
        ly = y + (cell_h - block_h) // 2
        for line in lines:
            d.text((PAD, ly), line, fill='black', font=label_f)
            ly += line_h

        for col_i, (pidx, version) in enumerate(PICKS):
            key = fixed_key or version
            samples_dir = VERSION_DIRS[key]
            f = find_sample(samples_dir, pidx)
            x = ROW_LABEL_W + PAD + col_i * (cell_w + PAD)
            if f and f.exists():
                im = Image.open(f).convert('RGB').resize((cell_w, cell_h))
                img.paste(im, (x, y))
            else:
                d.rectangle([x, y, x + cell_w, y + cell_h], outline='red')
                d.text((x + 10, y + 10), f'missing p{pidx} {key}', fill='red', font=cap_f)
        y += cell_h + PAD

    # Captions: prompt under each column (trigger word stripped)
    for col_i, (pidx, _) in enumerate(PICKS):
        x = ROW_LABEL_W + PAD + col_i * (cell_w + PAD)
        prompt = prompts[pidx][0] if pidx < len(prompts) else f'prompt {pidx}'
        prompt_disp = re.sub(r'^wbronkhorst style,\s*', '', prompt, flags=re.I)
        lines = wrap(prompt_disp, d, cap_f, cell_w)
        cy = y
        for line in lines[:4]:
            d.text((x, cy), line, fill='gray', font=cap_f)
            cy += cap_f.size + 2

    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT)
    print(f'wrote {OUT}  ({img.size[0]}x{img.size[1]})')


if __name__ == '__main__':
    main()
