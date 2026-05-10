"""Build side-by-side comparison grids: baseline (no-LoRA) vs LoRA.

Reads images from results/diffusion_lora/baseline/ and results/diffusion_lora/lora/
(both produced by generate.py), pairs them by filename, and saves a grid PNG.

Usage:
    python compare.py
    python compare.py --label-baseline "Flux base" --label-lora "Flux + WB LoRA"
    python compare.py --max 6 --width 768       # smaller grid

The pairing assumes you ran generate.py with the SAME prompts file and SAME
seed for both runs, so the filenames match (`00_00__...png`).
"""
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent.parent.parent if False else Path(__file__).resolve().parent.parent.parent
DEFAULT_BASE = ROOT.parent / 'results' / 'diffusion_lora' / 'baseline' if False else None

# Use absolute path the simple way
ROOT = Path(__file__).resolve().parent.parent.parent       # final_proj/
DEFAULT_BASELINE = ROOT / 'results' / 'diffusion_lora' / 'baseline'
DEFAULT_LORA     = ROOT / 'results' / 'diffusion_lora' / 'lora'
DEFAULT_OUT      = ROOT / 'results' / 'diffusion_lora' / 'grid.png'

LABEL_HEIGHT = 60
TITLE_HEIGHT = 80
PAD = 20


def _font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        'C:/Windows/Fonts/arial.ttf',
        'C:/Windows/Fonts/segoeui.ttf',
        '/System/Library/Fonts/Helvetica.ttc',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    ]
    for c in candidates:
        if Path(c).exists():
            return ImageFont.truetype(c, size)
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', default=str(DEFAULT_BASELINE),
                    help='dir from `python generate.py --no-lora`')
    ap.add_argument('--lora',     default=str(DEFAULT_LORA),
                    help='dir from `python generate.py --lora <path>`')
    ap.add_argument('--out',      default=str(DEFAULT_OUT))
    ap.add_argument('--label-baseline', default='Flux base (no LoRA)')
    ap.add_argument('--label-lora',     default='Flux + WB LoRA')
    ap.add_argument('--width',  type=int, default=512,
                    help='per-image width in the grid (height auto)')
    ap.add_argument('--max',    type=int, default=None,
                    help='cap number of rows (one per prompt)')
    args = ap.parse_args()

    base_dir = Path(args.baseline)
    lora_dir = Path(args.lora)
    if not base_dir.exists() or not lora_dir.exists():
        raise SystemExit(f'Missing one of:\n  {base_dir}\n  {lora_dir}\n'
                         'Run `python generate.py --no-lora` and `python generate.py --lora ...` first.')

    base_files = {p.name: p for p in sorted(base_dir.glob('*.png'))}
    lora_files = {p.name: p for p in sorted(lora_dir.glob('*.png'))}
    common = sorted(set(base_files) & set(lora_files))
    if not common:
        raise SystemExit('No matching filenames between baseline/ and lora/. '
                         'Did you generate both with the same prompts.txt + seed?')
    if args.max:
        common = common[: args.max]
    print(f'Pairing {len(common)} prompts')

    # Read first image to get aspect ratio; force same height per row.
    sample = Image.open(base_files[common[0]])
    aspect = sample.height / sample.width
    cell_w, cell_h = args.width, int(args.width * aspect)

    rows = len(common)
    grid_w = 2 * cell_w + 3 * PAD
    grid_h = TITLE_HEIGHT + LABEL_HEIGHT + rows * (cell_h + PAD) + PAD
    grid = Image.new('RGB', (grid_w, grid_h), 'white')
    draw = ImageDraw.Draw(grid)
    title_font = _font(28)
    label_font = _font(18)
    caption_font = _font(13)

    # Title
    draw.text((PAD, PAD), 'WB LoRA: baseline vs LoRA (same seed, same prompt)',
              fill='black', font=title_font)
    # Column headers
    y_label = TITLE_HEIGHT
    draw.text((PAD, y_label), args.label_baseline, fill='black', font=label_font)
    draw.text((PAD * 2 + cell_w, y_label), args.label_lora, fill='black', font=label_font)

    # Rows
    y = TITLE_HEIGHT + LABEL_HEIGHT
    for name in common:
        b = Image.open(base_files[name]).convert('RGB').resize((cell_w, cell_h))
        l = Image.open(lora_files[name]).convert('RGB').resize((cell_w, cell_h))
        grid.paste(b, (PAD, y))
        grid.paste(l, (PAD * 2 + cell_w, y))
        # Caption (extract prompt-ish part from filename)
        prompt_hint = name.split('__', 1)[1].rsplit('.', 1)[0].replace('_', ' ')[:90]
        draw.text((PAD, y + cell_h + 2), prompt_hint, fill='gray', font=caption_font)
        y += cell_h + PAD

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out)
    print(f'Wrote {out}  ({grid.size[0]}x{grid.size[1]} px)')


if __name__ == '__main__':
    main()
