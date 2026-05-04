"""v3 key-step grid: step 0, 1500, 3000, 5000 side by side at large size.

Shows the trajectory at higher per-cell resolution than the cramped all-steps grid.
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent.parent.parent
SAMPLES = ROOT / 'models' / 'wb_lora_flux2' / 'wb_lora_flux2_klein_9b_v3' / 'samples'
OUT = ROOT / 'results' / 'wb_lora_compare' / 'flux2_v3_keysteps_grid.png'

KEY_STEPS = [0, 1500, 3000, 5000]
PROMPTS_FILE = Path(__file__).parent / 'prompts.txt'

PAD, TITLE_H, LABEL_H, CAP_H = 18, 70, 50, 30
CELL_W = 460


def find(step, pidx):
    pad = f'{step:09d}'
    for f in SAMPLES.glob(f'*__{pad}_{pidx}.*'):
        return f
    return None


def font(size):
    for c in ['C:/Windows/Fonts/arial.ttf', '/System/Library/Fonts/Helvetica.ttc',
              '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']:
        if Path(c).exists():
            return ImageFont.truetype(c, size)
    return ImageFont.load_default()


def main():
    prompts = [l.strip() for l in PROMPTS_FILE.read_text(encoding='utf-8').splitlines()
               if l.strip() and not l.startswith('#')]
    sample = Image.open(find(0, 0))
    cell_w, cell_h = CELL_W, int(CELL_W * sample.height / sample.width)
    n_cols, n_rows = len(KEY_STEPS), 8

    grid_w = n_cols * cell_w + (n_cols + 1) * PAD
    grid_h = TITLE_H + LABEL_H + n_rows * (cell_h + CAP_H + PAD) + PAD
    img = Image.new('RGB', (grid_w, grid_h), 'white')
    d = ImageDraw.Draw(img)
    title_f = font(28)
    label_f = font(22)
    cap_f = font(13)

    d.text((PAD, PAD), 'WB LoRA v3 on Flux.2 Klein 9B  —  step 0 / 1500 / 3000 / 5000',
           fill='black', font=title_f)
    for i, s in enumerate(KEY_STEPS):
        x = PAD + i * (cell_w + PAD)
        label = 'baseline (step 0)' if s == 0 else f'step {s}'
        d.text((x, TITLE_H), label, fill='black', font=label_f)

    y = TITLE_H + LABEL_H
    for pidx in range(n_rows):
        for col_i, s in enumerate(KEY_STEPS):
            f = find(s, pidx)
            x = PAD + col_i * (cell_w + PAD)
            if f:
                im = Image.open(f).convert('RGB').resize((cell_w, cell_h))
                img.paste(im, (x, y))
        prompt_text = (prompts[pidx] if pidx < len(prompts) else f'p{pidx}')[:200]
        d.text((PAD, y + cell_h + 4), prompt_text, fill='gray', font=cap_f)
        y += cell_h + CAP_H + PAD

    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT)
    print(f'wrote {OUT}  ({img.size[0]}x{img.size[1]})')


if __name__ == '__main__':
    main()
