"""Build comparison grids for ai-toolkit's flat samples/ output.

Filename format:  <timestamp>__<step_padded_9>_<prompt_idx>.jpg
e.g.              1777409965089__000000000_0.jpg  (step 0, prompt 0)

Usage:
    python compare_v2.py
        # default: baseline (step 0) vs final (step 3000), all 8 prompts

    python compare_v2.py --baseline-step 0 --lora-step 2000
    python compare_v2.py --width 600 --label-baseline 'Flux2 base' --label-lora 'WB LoRA @ 3000'
    python compare_v2.py --all-steps        # multi-column: every step side by side
"""
import argparse
import re
from collections import defaultdict
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent.parent.parent
SAMPLES_DIR = ROOT / 'models' / 'diffusion_lora' / 'klein_9b_v3' / 'samples'
OUT_DIR = ROOT / 'results' / 'diffusion_lora'

PROMPTS_FILE = Path(__file__).parent / 'prompts.txt'

PAD = 18
LABEL_H = 50
TITLE_H = 70
CAP_H = 30


def load_prompts() -> list[str]:
    if not PROMPTS_FILE.exists():
        return [f'prompt {i}' for i in range(8)]
    out = []
    for line in PROMPTS_FILE.read_text(encoding='utf-8').splitlines():
        s = line.strip()
        if s and not s.startswith('#'):
            out.append(s)
    return out


def index_samples() -> dict[int, dict[int, Path]]:
    """{step: {prompt_idx: file_path}}"""
    pat = re.compile(r'__(\d{9})_(\d+)\.(?:jpg|jpeg|png)$', re.I)
    idx: dict[int, dict[int, Path]] = defaultdict(dict)
    for f in sorted(SAMPLES_DIR.glob('*.*')):
        m = pat.search(f.name)
        if m:
            step = int(m.group(1))
            pidx = int(m.group(2))
            idx[step][pidx] = f
    return idx


def _font(size):
    for c in ['C:/Windows/Fonts/arial.ttf', '/System/Library/Fonts/Helvetica.ttc',
              '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']:
        if Path(c).exists():
            return ImageFont.truetype(c, size)
    return ImageFont.load_default()


def build_two_col_grid(idx, baseline_step, lora_step, prompts, width, out_path,
                       label_baseline, label_lora):
    cols = [(label_baseline, idx[baseline_step]),
            (label_lora,     idx[lora_step])]
    common = sorted(set(cols[0][1]) & set(cols[1][1]))
    if not common:
        raise SystemExit(f'No prompts in common between step {baseline_step} and step {lora_step}')

    sample = Image.open(cols[0][1][common[0]])
    aspect = sample.height / sample.width
    cell_w, cell_h = width, int(width * aspect)
    n_cols = 2
    n_rows = len(common)

    grid_w = n_cols * cell_w + (n_cols + 1) * PAD
    grid_h = TITLE_H + LABEL_H + n_rows * (cell_h + CAP_H + PAD) + PAD
    img = Image.new('RGB', (grid_w, grid_h), 'white')
    d = ImageDraw.Draw(img)

    title_font = _font(28)
    label_font = _font(20)
    cap_font = _font(13)

    d.text((PAD, PAD), f'WB LoRA on Flux.2 Klein 9B  —  step {baseline_step} vs step {lora_step}',
           fill='black', font=title_font)
    d.text((PAD, TITLE_H), label_baseline, fill='black', font=label_font)
    d.text((PAD * 2 + cell_w, TITLE_H), label_lora, fill='black', font=label_font)

    y = TITLE_H + LABEL_H
    for pidx in common:
        for col_i, (_, files) in enumerate(cols):
            x = PAD + col_i * (cell_w + PAD)
            im = Image.open(files[pidx]).convert('RGB').resize((cell_w, cell_h))
            img.paste(im, (x, y))
        prompt_text = (prompts[pidx] if pidx < len(prompts) else f'prompt {pidx}')[:140]
        d.text((PAD, y + cell_h + 4), prompt_text, fill='gray', font=cap_font)
        y += cell_h + CAP_H + PAD

    img.save(out_path)
    print(f'wrote {out_path}  ({img.size[0]}x{img.size[1]})')


def build_all_steps_grid(idx, prompts, width, out_path):
    steps = sorted(idx.keys())
    sample = Image.open(next(iter(idx[steps[0]].values())))
    aspect = sample.height / sample.width
    cell_w, cell_h = width, int(width * aspect)
    n_cols = len(steps)
    common_prompts = sorted(set.intersection(*(set(idx[s]) for s in steps)))
    n_rows = len(common_prompts)

    grid_w = n_cols * cell_w + (n_cols + 1) * PAD
    grid_h = TITLE_H + LABEL_H + n_rows * (cell_h + CAP_H + PAD) + PAD
    img = Image.new('RGB', (grid_w, grid_h), 'white')
    d = ImageDraw.Draw(img)

    title_font = _font(26)
    label_font = _font(15)
    cap_font = _font(12)

    d.text((PAD, PAD), 'WB LoRA on Flux.2 Klein 9B  —  all training steps',
           fill='black', font=title_font)
    for i, s in enumerate(steps):
        label = 'baseline' if s == 0 else f'step {s}'
        x = PAD + i * (cell_w + PAD)
        d.text((x, TITLE_H), label, fill='black', font=label_font)

    y = TITLE_H + LABEL_H
    for pidx in common_prompts:
        for i, s in enumerate(steps):
            x = PAD + i * (cell_w + PAD)
            im = Image.open(idx[s][pidx]).convert('RGB').resize((cell_w, cell_h))
            img.paste(im, (x, y))
        prompt_text = (prompts[pidx] if pidx < len(prompts) else f'prompt {pidx}')[:140]
        d.text((PAD, y + cell_h + 4), prompt_text, fill='gray', font=cap_font)
        y += cell_h + CAP_H + PAD

    img.save(out_path)
    print(f'wrote {out_path}  ({img.size[0]}x{img.size[1]})')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline-step', type=int, default=0)
    ap.add_argument('--lora-step', type=int, default=None,
                    help='default: latest available step')
    ap.add_argument('--width', type=int, default=512)
    ap.add_argument('--label-baseline', default='Step 0 (baseline)')
    ap.add_argument('--label-lora', default=None)
    ap.add_argument('--all-steps', action='store_true',
                    help='build a multi-column grid showing every step')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    if not SAMPLES_DIR.exists():
        raise SystemExit(f'No samples at {SAMPLES_DIR}')
    idx = index_samples()
    if not idx:
        raise SystemExit(f'No sample files matched the expected pattern in {SAMPLES_DIR}')
    prompts = load_prompts()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.all_steps:
        out = Path(args.out) if args.out else OUT_DIR / 'flux2_v3_grid_all_steps.png'
        build_all_steps_grid(idx, prompts, args.width, out)
    else:
        lora_step = args.lora_step if args.lora_step is not None else max(idx.keys())
        label_lora = args.label_lora or (f'Step {lora_step} (trained LoRA)')
        out = Path(args.out) if args.out else OUT_DIR / f'flux2_v3_grid_{args.baseline_step}_vs_{lora_step}.png'
        build_two_col_grid(idx, args.baseline_step, lora_step, prompts, args.width, out,
                           args.label_baseline, label_lora)


if __name__ == '__main__':
    main()
