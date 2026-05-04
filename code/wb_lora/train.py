"""Launch the LoRA training run. Wraps `accelerate launch` so all the
hyperparameter knobs are documented in one place.

Usage:
    python train.py
    python train.py --base sdxl              # use SDXL instead of SD 1.5
    python train.py --rank 64 --epochs 120   # bigger LoRA, more epochs

Prereqs (run once):
    python setup.py                          # downloads the trainer script
    python caption_images.py                 # writes data/.../metadata.jsonl
    accelerate config default
    pip install diffusers==0.30.3 accelerate peft transformers datasets bitsandbytes
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent              # final_proj/
DATA = ROOT / 'data' / 'wb_lora_train'
OUT  = ROOT / 'models' / 'wb_lora'
TRAIN_SCRIPT = HERE / 'train_text_to_image_lora.py'

BASE_MODELS = {
    'sd15': 'runwayml/stable-diffusion-v1-5',
    'sd2':  'stabilityai/stable-diffusion-2-1',
    'sdxl': 'stabilityai/stable-diffusion-xl-base-1.0',
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--base',    default='sd15', choices=BASE_MODELS.keys())
    ap.add_argument('--resolution', type=int, default=768,
                    help='SD 1.5: 512-768.  SDXL: 1024 (needs 16+GB VRAM).')
    ap.add_argument('--batch',   type=int, default=1)
    ap.add_argument('--accum',   type=int, default=4, help='gradient accumulation')
    ap.add_argument('--epochs',  type=int, default=80)
    ap.add_argument('--lr',      type=float, default=1e-4)
    ap.add_argument('--rank',    type=int, default=32, help='LoRA rank (16 small / 32 default / 64 big)')
    ap.add_argument('--seed',    type=int, default=1337)
    ap.add_argument('--validation-prompt', default='wbronkhorst style, a painting of a surfer at sunset, dramatic lighting')
    ap.add_argument('--out',     default=str(OUT))
    args = ap.parse_args()

    if not TRAIN_SCRIPT.exists():
        print(f'Missing {TRAIN_SCRIPT}. Run:  python setup.py')
        sys.exit(1)
    if not (DATA / 'metadata.jsonl').exists():
        print(f'Missing {DATA / "metadata.jsonl"}. Run:  python caption_images.py')
        sys.exit(1)

    base = BASE_MODELS[args.base]
    cmd = [
        sys.executable, '-m', 'accelerate.commands.launch',
        str(TRAIN_SCRIPT),
        '--pretrained_model_name_or_path', base,
        '--train_data_dir', str(DATA),
        '--resolution',     str(args.resolution),
        '--center_crop', '--random_flip',
        '--train_batch_size', str(args.batch),
        '--gradient_accumulation_steps', str(args.accum),
        '--num_train_epochs', str(args.epochs),
        '--learning_rate',    f'{args.lr}',
        '--lr_scheduler', 'cosine',
        '--lr_warmup_steps', '100',
        '--max_grad_norm', '1.0',
        '--rank', str(args.rank),
        '--seed', str(args.seed),
        '--mixed_precision', 'fp16',
        '--gradient_checkpointing',
        '--checkpointing_steps', '500',
        '--validation_prompt', args.validation_prompt,
        '--validation_epochs', '10',
        '--output_dir', args.out,
        '--report_to', 'tensorboard',
        '--caption_column', 'text',
    ]
    print('Launching:\n  ' + ' '.join(cmd) + '\n')
    subprocess.run(cmd, check=True)
    print(f'\nDone. LoRA weights at: {args.out}')
    print(f'See {args.out}/checkpoint-*/   for intermediate checkpoints')


if __name__ == '__main__':
    main()
