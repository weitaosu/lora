"""Horizontal v1 baseline-vs-final grid: 8 prompts wide × 2 rows tall.

Top row: Flux.2 Klein 9B baseline (no LoRA, sample at step 0)
Bottom row: WB LoRA v1 @ step 3000 (final v1 checkpoint)

Replaces the old vertical layout in flux2_grid_0_vs_3000.png with
properly labeled rows and per-column prompt captions.
"""
import re
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent.parent.parent
SAMPLES = ROOT / 'models' / 'wb_lora_flux2' / 'wb_lora_flux2_klein_9b' / 'samples'
OUT = ROOT / 'results' / 'wb_lora_compare' / 'flux2_v1_baseline_vs_step3000_horizontal.png'

PROMPTS_FILE = Path(__file__).parent / 'prompts.txt'
PAD = 20
TITLE_H = 90
ROW_LABEL_W = 220
CAP_H = 80
CELL_W = 380


def load_prompts():
    return [l.strip() for l in PROMPTS_FILE.read_text(encoding='utf-8').splitlines()
            if l.strip() and not l.startswith('#')]


def find(step, pidx):
    pad = f'{step:09d}'
    matches = list(SAMPLES.glob(f'*__{pad}_{pidx}.*'))
    return matches[0] if matches else None


def font(size):
    for c in ['C:/Windows/Fonts/arialbd.ttf', 'C:/Windows/Fonts/arial.ttf',
              '/System/Library/Fonts/Helvetica.ttc',
              '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf']:
        if Path(c).exists():
            return ImageFont.truetype(c, size)
    return ImageFont.load_default()


def wrap_text(text, draw, fnt, max_w):
    """Greedy word wrap; returns list of lines that fit max_w pixels."""
    words = text.split(' ')
    lines, cur = [], ''
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
    n_prompts = 8

    sample = Image.open(find(0, 0))
    cell_w, cell_h = CELL_W, int(CELL_W * sample.height / sample.width)

    rows = [
        ('Flux.2 Klein 9B baseline', 0),
        ('WB LoRA v1 @ step 3000',   3000),
    ]
    n_rows = len(rows)

    grid_w = ROW_LABEL_W + n_prompts * cell_w + (n_prompts + 1) * PAD
    grid_h = TITLE_H + n_rows * (cell_h + PAD) + CAP_H + PAD

    img = Image.new('RGB', (grid_w, grid_h), 'white')
    d = ImageDraw.Draw(img)
    title_f = font(34)
    label_f = font(22)
    cap_f = font(13)

    d.text((PAD, PAD + 10),
           'WB LoRA reproduction on Flux.2 Klein 9B  —  baseline vs step 3000  (v1, 8 prompts)',
           fill='black', font=title_f)

    # Row labels (left) + image cells
    y = TITLE_H
    for r_i, (row_label, step) in enumerate(rows):
        # Row label centered vertically in the row
        label_lines = row_label.split(' ')
        # Render label as 1-3 lines
        lines = wrap_text(row_label, d, label_f, ROW_LABEL_W - PAD)
        line_h = label_f.size + 4
        block_h = line_h * len(lines)
        ly = y + (cell_h - block_h) // 2
        for line in lines:
            d.text((PAD, ly), line, fill='black', font=label_f)
            ly += line_h

        # Cells
        for pidx in range(n_prompts):
            f = find(step, pidx)
            x = ROW_LABEL_W + PAD + pidx * (cell_w + PAD)
            if f and f.exists():
                im = Image.open(f).convert('RGB').resize((cell_w, cell_h))
                img.paste(im, (x, y))
            else:
                d.rectangle([x, y, x + cell_w, y + cell_h], outline='red')
                d.text((x + 10, y + 10), f'missing step {step} p{pidx}',
                       fill='red', font=cap_f)
        y += cell_h + PAD

    # Captions (one per column, under the bottom row)
    for pidx in range(n_prompts):
        x = ROW_LABEL_W + PAD + pidx * (cell_w + PAD)
        prompt = prompts[pidx] if pidx < len(prompts) else f'prompt {pidx}'
        # remove trigger word for cleaner caption display
        prompt_disp = re.sub(r'^wbronkhorst style,\s*', '', prompt, flags=re.I)
        lines = wrap_text(prompt_disp, d, cap_f, cell_w)
        cy = y
        for line in lines[:4]:  # cap at 4 lines
            d.text((x, cy), line, fill='gray', font=cap_f)
            cy += cap_f.size + 2

    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT)
    print(f'wrote {OUT}  ({img.size[0]}x{img.size[1]})')


if __name__ == '__main__':
    main()
