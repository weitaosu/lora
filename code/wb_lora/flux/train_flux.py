"""Launch Flux LoRA training. Wraps the official diffusers
`train_dreambooth_lora_flux.py` with sensible low-VRAM defaults.

Prereqs (run once):
    python setup_flux.py                # downloads the trainer
    python ../caption_images.py         # creates data/wb_lora_train/metadata.jsonl
    accelerate config default
    huggingface-cli login               # only needed for FLUX.1-dev

Usage:
    python train_flux.py                # default: schnell, rank 8, 512 px, 12 GB profile
    python train_flux.py --base dev     # FLUX.1-dev (gated; need HF login)
    python train_flux.py --rank 16 --resolution 768 --epochs 60   # if you have 16+ GB

VRAM rough guide (with --use_8bit_adam --gradient_checkpointing):
    rank 8,  res 512  → ~12 GB
    rank 16, res 768  → ~16 GB
    rank 32, res 1024 → ~24 GB
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent                  # final_proj/
DATA = ROOT / 'data' / 'wb_lora_train'
OUT  = ROOT / 'models' / 'wb_lora_flux'
TRAIN_SCRIPT = HERE / 'train_dreambooth_lora_flux.py'

BASES = {
    'schnell': 'black-forest-labs/FLUX.1-schnell',
    'dev':     'black-forest-labs/FLUX.1-dev',
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--base',       default='schnell', choices=BASES.keys())
    ap.add_argument('--resolution', type=int, default=512)
    ap.add_argument('--rank',       type=int, default=8)
    ap.add_argument('--epochs',     type=int, default=80)
    ap.add_argument('--batch',      type=int, default=1)
    ap.add_argument('--accum',      type=int, default=4)
    ap.add_argument('--lr',         type=float, default=1e-4)
    ap.add_argument('--seed',       type=int, default=1337)
    ap.add_argument('--instance-prompt', default='wbronkhorst style painting')
    ap.add_argument('--validation-prompt',
                    default='wbronkhorst style, a painting of a surfer at sunset')
    ap.add_argument('--out',        default=str(OUT))
    args = ap.parse_args()

    if not TRAIN_SCRIPT.exists():
        print(f'Missing {TRAIN_SCRIPT}. Run:  python setup_flux.py')
        sys.exit(1)
    if not (DATA / 'metadata.jsonl').exists():
        print(f'Missing {DATA/"metadata.jsonl"}. Run:  python ../caption_images.py')
        sys.exit(1)

    base = BASES[args.base]
    cmd = [
        sys.executable, '-m', 'accelerate.commands.launch',
        str(TRAIN_SCRIPT),
        '--pretrained_model_name_or_path', base,
        '--instance_data_dir', str(DATA),
        '--instance_prompt', args.instance_prompt,
        '--output_dir', args.out,
        '--mixed_precision', 'bf16',
        '--resolution', str(args.resolution),
        '--train_batch_size', str(args.batch),
        '--gradient_accumulation_steps', str(args.accum),
        '--gradient_checkpointing',
        '--use_8bit_adam',
        '--learning_rate', f'{args.lr}',
        '--lr_scheduler', 'constant',
        '--lr_warmup_steps', '0',
        '--max_train_steps', str(args.epochs * 70),  # ~70 steps/epoch at 271 imgs / batch 4
        '--rank', str(args.rank),
        '--seed', str(args.seed),
        '--checkpointing_steps', '500',
        '--validation_prompt', args.validation_prompt,
        '--validation_epochs', '20',
        '--report_to', 'tensorboard',
    ]
    print('Launching:\n  ' + ' '.join(cmd) + '\n')
    subprocess.run(cmd, check=True)
    print(f'\nDone. LoRA at: {args.out}')


if __name__ == '__main__':
    main()
