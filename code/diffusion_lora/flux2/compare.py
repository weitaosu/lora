"""Build a baseline-vs-LoRA comparison grid using ai-toolkit's per-step samples.

ai-toolkit writes sample images during training to:
    models/diffusion_lora/<run_name>/samples/<step>/<prompt_idx>_<img_idx>.png

We pair samples from step 0 (baseline — LoRA hasn't trained yet) with samples
from the final step (fully-trained LoRA) and stack them side-by-side for each
prompt. Same seed in both columns -> only the LoRA differs.

Usage:
    python compare.py
    python compare.py --baseline-step 0 --lora-step 4000   # explicit step pairing
    python compare.py --width 768
"""
import argparse
import re
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent          # final_proj/
DEFAULT_RUN_DIR = ROOT / 'models' / 'diffusion_lora'
DEFAULT_OUT = ROOT / 'results' / 'diffusion_lora' / 'flux2_grid.png'

LABEL_HEIGHT = 60
TITLE_HEIGHT = 80
PAD = 20


def _font(size: int) -> ImageFont.FreeTypeFont:
    for c in ['C:/Windows/Fonts/arial.ttf',
              '/System/Library/Fonts/Helvetica.ttc',
              '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']:
        if Path(c).exists():
            return ImageFont.truetype(c, size)
    return ImageFont.load_default()


def find_step_dir(run_dir: Path, step: int | None) -> Path:
    samples_root = run_dir / 'samples'
    if not samples_root.exists():
        # ai-toolkit sometimes nests under a name subfolder
        candidates = list(run_dir.glob('*/samples'))
        if candidates:
            samples_root = candidates[0]
        else:
            raise SystemExit(f'No samples/ dir under {run_dir}')

    step_dirs = sorted([d for d in samples_root.iterdir() if d.is_dir()])
    if not step_dirs:
        raise SystemExit(f'{samples_root} has no step subdirectories yet — '
                         'training may not have produced any samples.')

    if step is None:
        # default: oldest = baseline, newest = lora
        return step_dirs[0], step_dirs[-1]

    # explicit step
    for d in step_dirs:
        m = re.search(r'(\d+)', d.name)
        if m and int(m.group(1)) == step:
            return d
    raise SystemExit(f'Step {step} not found under {samples_root}.\n'
                     f'Available: {[d.name for d in step_dirs]}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', default=str(DEFAULT_RUN_DIR),
                    help='ai-toolkit training_folder (where samples/ lives)')
    ap.add_argument('--baseline-step', type=int, default=None,
                    help='step number for baseline column (default: oldest sample)')
    ap.add_argument('--lora-step', type=int, default=None,
                    help='step number for LoRA column (default: latest sample)')
    ap.add_argument('--out', default=str(DEFAULT_OUT))
    ap.add_argument('--width', type=int, default=512,
                    help='per-image width in the grid (height auto)')
    ap.add_argument('--max', type=int, default=None,
                    help='cap number of rows (one per prompt)')
    ap.add_argument('--label-baseline', default='Step 0 (baseline)')
    ap.add_argument('--label-lora',     default='Final step (trained LoRA)')
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if args.baseline_step is None and args.lora_step is None:
        baseline_dir, lora_dir = find_step_dir(run_dir, None)
    else:
        baseline_dir = find_step_dir(run_dir, args.baseline_step) if args.baseline_step is not None else find_step_dir(run_dir, None)[0]
        lora_dir     = find_step_dir(run_dir, args.lora_step)     if args.lora_step     is not None else find_step_dir(run_dir, None)[1]

    print(f'baseline:  {baseline_dir}')
    print(f'lora    :  {lora_dir}')

    base_files = {p.name: p for p in sorted(baseline_dir.glob('*.png')) + sorted(baseline_dir.glob('*.jpg'))}
    lora_files = {p.name: p for p in sorted(lora_dir.glob('*.png'))     + sorted(lora_dir.glob('*.jpg'))}
    common = sorted(set(base_files) & set(lora_files))
    if not common:
        # fall back to index-based pairing if filenames differ
        common = [(b, l) for b, l in zip(sorted(base_files), sorted(lora_files))]
        if not common:
            raise SystemExit('No paired samples found between the two step dirs.')

    if args.max:
        common = common[:args.max]
    print(f'Pairing {len(common)} prompts')

    sample = Image.open(base_files[common[0]] if isinstance(common[0], str) else common[0][0])
    aspect = sample.height / sample.width
    cell_w, cell_h = args.width, int(args.width * aspect)

    rows = len(common)
    grid_w = 2 * cell_w + 3 * PAD
    grid_h = TITLE_HEIGHT + LABEL_HEIGHT + rows * (cell_h + PAD) + PAD
    grid = Image.new('RGB', (grid_w, grid_h), 'white')
    draw = ImageDraw.Draw(grid)

    title_font = _font(28)
    label_font = _font(18)
    cap_font = _font(13)

    draw.text((PAD, PAD), 'WB LoRA on Flux.2 Klein 9B: baseline vs trained',
              fill='black', font=title_font)
    y_label = TITLE_HEIGHT
    draw.text((PAD, y_label),                  args.label_baseline, fill='black', font=label_font)
    draw.text((PAD * 2 + cell_w, y_label),     args.label_lora,     fill='black', font=label_font)

    y = TITLE_HEIGHT + LABEL_HEIGHT
    for entry in common:
        if isinstance(entry, str):
            bp, lp = base_files[entry], lora_files[entry]
            label = entry
        else:
            bp, lp = entry
            label = bp.name
        b = Image.open(bp).convert('RGB').resize((cell_w, cell_h))
        l = Image.open(lp).convert('RGB').resize((cell_w, cell_h))
        grid.paste(b, (PAD, y))
        grid.paste(l, (PAD * 2 + cell_w, y))
        draw.text((PAD, y + cell_h + 2), label.rsplit('.', 1)[0][:80],
                  fill='gray', font=cap_font)
        y += cell_h + PAD

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out)
    print(f'\nWrote {out}  ({grid.size[0]}x{grid.size[1]} px)')


if __name__ == '__main__':
    main()
