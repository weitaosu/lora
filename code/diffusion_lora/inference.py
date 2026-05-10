"""Generate sample images from the trained LoRA.

Usage:
    python inference.py
    python inference.py --prompt "wbronkhorst style, a portrait of an astronaut"
    python inference.py --strength 0.8 --steps 30 --seed 7
"""
import argparse
from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

ROOT = Path(__file__).resolve().parent.parent.parent      # final_proj/
LORA_DIR = ROOT / 'models' / 'diffusion_lora' / 'sd15'
OUT_DIR  = ROOT / 'results' / 'diffusion_lora' / 'sd15_samples'

DEFAULT_PROMPTS = [
    'wbronkhorst style, a painting of a lifeguard standing at sunset, dramatic lighting',
    'wbronkhorst style, an oil painting of a surfer riding a wave, ocean spray',
    'wbronkhorst style, portrait of a person on a beach with vintage car',
    'wbronkhorst style, a figure in red swimsuit, painterly textures',
    'wbronkhorst style, scene at a swimming pool, hyperreal painting',
    'wbronkhorst style, woman on the beach holding flowers, soft light',
]
NEG_PROMPT = ('low quality, blurry, watermark, signature, text, distorted face, '
              'extra fingers, deformed hands')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base',     default='runwayml/stable-diffusion-v1-5')
    ap.add_argument('--lora',     default=str(LORA_DIR))
    ap.add_argument('--prompt',   default=None,
                    help='if given, generates one image; otherwise generates a prompt grid')
    ap.add_argument('--strength', type=float, default=1.0,
                    help='LoRA scale (0=no LoRA, 1=full, 1.2-1.4=overcooked)')
    ap.add_argument('--steps',    type=int, default=30)
    ap.add_argument('--guidance', type=float, default=7.5)
    ap.add_argument('--seed',     type=int, default=42)
    ap.add_argument('--out',      default=str(OUT_DIR))
    ap.add_argument('--n',        type=int, default=1, help='images per prompt')
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Loading {args.base} on {device}...')
    pipe = StableDiffusionPipeline.from_pretrained(
        args.base, torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        safety_checker=None,
    ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    print(f'Loading LoRA from {args.lora} (scale={args.strength})')
    pipe.load_lora_weights(args.lora)
    pipe.fuse_lora(lora_scale=args.strength)

    prompts = [args.prompt] if args.prompt else DEFAULT_PROMPTS
    g = torch.Generator(device=device).manual_seed(args.seed)
    for i, p in enumerate(prompts):
        for k in range(args.n):
            img = pipe(p, negative_prompt=NEG_PROMPT, num_inference_steps=args.steps,
                       guidance_scale=args.guidance, generator=g).images[0]
            fn = out / f'{i:02d}_{k:02d}__{p[:60].replace(" ", "_").replace(",", "")}.png'
            img.save(fn)
            print(f'  saved {fn.name}')

    print(f'\nWrote {len(prompts) * args.n} images to {out}')


if __name__ == '__main__':
    main()
