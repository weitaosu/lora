"""Build the 20-prompt × 4-version comparison grid from fresh inference outputs.

Each row = one prompt. 4 columns: baseline, v1@3000, v2@5000, v3@5000.
Each cell labels the seed.
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent.parent.parent
RESULTS_BASE = ROOT / 'results' / 'wb_lora_compare'
OUT = RESULTS_BASE / 'flux2_20x4_compare.png'

VERSIONS = [
    ('Baseline (Flux.2 Klein 9B)', 'wb_lora_flux2_inference_baseline'),
    ('WB LoRA v1 @ step 3000',     'wb_lora_flux2_inference_v1'),
    ('WB LoRA v2 @ step 5000',     'wb_lora_flux2_inference_v2'),
    ('WB LoRA v3 @ step 5000',     'wb_lora_flux2_inference_v3'),
]

PROMPTS_FILE = Path(__file__).parent / 'compare_20_prompts.txt'

PAD, TITLE_H, LABEL_H, CAP_H = 20, 90, 60, 36
CELL_W = 480


def load_prompts():
    text = PROMPTS_FILE.read_text(encoding='utf-8').splitlines()
    out = []
    for line in text:
        if not line.strip() or line.startswith('#'):
            continue
        # strip --seed/--steps flags from display
        display = line.split(' --')[0].strip()
        # extract seed
        seed = ''
        for tok in line.split('--'):
            if tok.strip().startswith('seed '):
                seed = tok.strip().split()[1]
        out.append((display, seed))
    return out


def find_sample(samples_dir, prompt_idx):
    """ai-toolkit's training-time sample step writes filenames like:
       <timestamp>__000000000_<idx>.jpg in <run_dir>/samples/
    """
    pad = '000000000'  # step 0
    matches = list(samples_dir.glob(f'*__{pad}_{prompt_idx}.*'))
    return matches[0] if matches else None


def font(size):
    for c in ['C:/Windows/Fonts/arial.ttf', '/System/Library/Fonts/Helvetica.ttc',
              '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']:
        if Path(c).exists():
            return ImageFont.truetype(c, size)
    return ImageFont.load_default()


def main():
    prompts = load_prompts()
    n_prompts = len(prompts)
    n_cols = len(VERSIONS)

    # Locate first sample to determine cell aspect
    first = None
    for label, run_name in VERSIONS:
        sd = RESULTS_BASE / run_name / 'samples'
        first = find_sample(sd, 0)
        if first:
            break
    if not first:
        raise SystemExit('No samples found yet — run the inference pipeline first')
    sample = Image.open(first)
    cell_w, cell_h = CELL_W, int(CELL_W * sample.height / sample.width)

    grid_w = n_cols * cell_w + (n_cols + 1) * PAD
    grid_h = TITLE_H + LABEL_H + n_prompts * (cell_h + CAP_H + PAD) + PAD
    img = Image.new('RGB', (grid_w, grid_h), 'white')
    d = ImageDraw.Draw(img)
    title_f = font(32)
    label_f = font(22)
    cap_f = font(15)

    d.text((PAD, PAD), 'WB LoRA on Flux.2 Klein 9B  —  20 prompts × 4 versions  (1024×1024, 20 steps, cfg 4.0)',
           fill='black', font=title_f)
    for i, (label, _) in enumerate(VERSIONS):
        x = PAD + i * (cell_w + PAD)
        d.text((x, TITLE_H), label, fill='black', font=label_f)

    y = TITLE_H + LABEL_H
    for pidx, (prompt, seed) in enumerate(prompts):
        for col_i, (_, run_name) in enumerate(VERSIONS):
            samples_dir = RESULTS_BASE / run_name / 'samples'
            f = find_sample(samples_dir, pidx)
            x = PAD + col_i * (cell_w + PAD)
            if f and f.exists():
                im = Image.open(f).convert('RGB').resize((cell_w, cell_h))
                img.paste(im, (x, y))
            else:
                d.rectangle([x, y, x + cell_w, y + cell_h], outline='red')
                d.text((x + 10, y + 10), f'missing: p{pidx} in {run_name}', fill='red', font=cap_f)
        # Caption: prompt index, seed, prompt text (truncated)
        cap = f'#{pidx + 1}  seed {seed}  |  {prompt[:130]}'
        d.text((PAD, y + cell_h + 6), cap, fill='gray', font=cap_f)
        y += cell_h + CAP_H + PAD

    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT)
    print(f'wrote {OUT}  ({img.size[0]}x{img.size[1]})')


if __name__ == '__main__':
    main()
