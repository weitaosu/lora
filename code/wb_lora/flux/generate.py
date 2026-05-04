"""Generate images from a list of prompts, with or without the trained LoRA.

Used as both the "baseline" generator (before training) and the "trained"
generator (after training). Same seed + same prompts → comparable side-by-side.

Usage:
    # baseline (no LoRA), saves to results/wb_lora_compare/baseline/
    python generate.py --no-lora

    # with trained LoRA, saves to results/wb_lora_compare/lora/
    python generate.py --lora ../../../models/wb_lora_flux

    # quick smoke test
    python generate.py --no-lora --prompts-file prompts.txt --n 1 --steps 4

VRAM:
    Flux.1-schnell needs ~12 GB at fp16 with `enable_model_cpu_offload`.
    Flux.1-dev is similar; we offload by default.
"""
import argparse
import time
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent                  # final_proj/
DEFAULT_PROMPTS_FILE = HERE / 'prompts.txt'
DEFAULT_OUT = ROOT / 'results' / 'wb_lora_compare'

BASES = {
    'schnell': 'black-forest-labs/FLUX.1-schnell',  # open license, 4-step distilled
    'dev':     'black-forest-labs/FLUX.1-dev',       # gated, non-commercial, 20-step
}


def read_prompts(path: Path) -> list[str]:
    out = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            out.append(line)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='schnell', choices=BASES.keys(),
                    help='Flux base model (schnell = open, dev = gated/better)')
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument('--no-lora', action='store_true', help='generate baseline (no LoRA)')
    grp.add_argument('--lora', type=str, default=None,
                     help='path to trained LoRA weights dir (e.g. ../../../models/wb_lora_flux)')
    ap.add_argument('--lora-scale', type=float, default=1.0)
    ap.add_argument('--prompts-file', default=str(DEFAULT_PROMPTS_FILE))
    ap.add_argument('--steps', type=int, default=None,
                    help='inference steps (auto: 4 for schnell, 20 for dev)')
    ap.add_argument('--guidance', type=float, default=None,
                    help='guidance scale (auto: 0 for schnell, 3.5 for dev)')
    ap.add_argument('--seed', type=int, default=42,
                    help='shared seed → identical noise for baseline and LoRA runs')
    ap.add_argument('--n',    type=int, default=1, help='images per prompt')
    ap.add_argument('--width',  type=int, default=1024)
    ap.add_argument('--height', type=int, default=1024)
    ap.add_argument('--out', default=None,
                    help='output dir (default: results/wb_lora_compare/{baseline|lora})')
    ap.add_argument('--no-offload', action='store_true',
                    help='disable model CPU offload (only if you have 24+ GB VRAM)')
    args = ap.parse_args()

    if args.no_lora and args.lora:
        raise SystemExit('Pick one: --no-lora OR --lora <path>')
    use_lora = args.lora is not None

    if args.steps is None:
        args.steps = 4 if args.base == 'schnell' else 20
    if args.guidance is None:
        args.guidance = 0.0 if args.base == 'schnell' else 3.5

    out_dir = Path(args.out) if args.out else (
        DEFAULT_OUT / ('lora' if use_lora else 'baseline')
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = read_prompts(Path(args.prompts_file))
    print(f'Generating {len(prompts) * args.n} images')
    print(f'  base = {args.base} ({BASES[args.base]})')
    print(f'  lora = {args.lora if use_lora else "(disabled — baseline)"}')
    print(f'  steps={args.steps}  guidance={args.guidance}  seed={args.seed}')
    print(f'  out  = {out_dir}')

    from diffusers import FluxPipeline
    print('\nLoading pipeline (first call downloads ~24 GB; please wait)...')
    pipe = FluxPipeline.from_pretrained(BASES[args.base], torch_dtype=torch.bfloat16)

    if use_lora:
        print(f'Loading LoRA from {args.lora} (scale={args.lora_scale})')
        pipe.load_lora_weights(args.lora)
        pipe.fuse_lora(lora_scale=args.lora_scale)

    if args.no_offload:
        pipe = pipe.to('cuda')
    else:
        # Saves ~10 GB VRAM at the cost of ~2× slower inference. Required at 12 GB.
        pipe.enable_model_cpu_offload()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    for i, p in enumerate(prompts):
        for k in range(args.n):
            g = torch.Generator(device='cuda').manual_seed(args.seed + k)
            img = pipe(p, num_inference_steps=args.steps, guidance_scale=args.guidance,
                       width=args.width, height=args.height, generator=g).images[0]
            stem = p[:80].replace(' ', '_').replace(',', '').replace('/', '_')
            fn = out_dir / f'{i:02d}_{k:02d}__{stem}.png'
            img.save(fn)
            print(f'  [{i:02d}.{k:02d}] {fn.name}')

    elapsed = time.time() - t0
    peak = (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else 0.0
    n = len(prompts) * args.n
    print(f'\nDone. {n} images in {elapsed:.1f}s  ({elapsed/n:.1f}s each)  peak GPU {peak:.2f} GB')
    print(f'Wrote -> {out_dir}')


if __name__ == '__main__':
    main()
