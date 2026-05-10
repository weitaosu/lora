"""One-time setup for Flux LoRA training: download diffusers' Flux LoRA trainer.

What this fetches:
    train_dreambooth_lora_flux.py from diffusers v0.31+ — the official LoRA
    trainer for Flux.1. Pinned to a known-good commit so the script doesn't
    drift under us.

After running this, follow the steps in README.md.

Note on VRAM:
    Flux is a 12B-parameter transformer. Even LoRA training is heavy:
    - 24 GB VRAM: comfortable, full precision base, rank 16 at 1024 px
    - 16 GB VRAM: OK with --use_8bit_adam + --gradient_checkpointing, rank 8-16, 768 px
    - 12 GB VRAM: TIGHT — needs --quantize fp8 or nf4 base + everything else, rank 8, 512 px
                  May still OOM; if so, switch to ai-toolkit (https://github.com/ostris/ai-toolkit)
                  which has more aggressive low-VRAM optimizations.
"""
import os
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAIN_SCRIPT = HERE / 'train_dreambooth_lora_flux.py'
URL = (
    'https://raw.githubusercontent.com/huggingface/diffusers/'
    'v0.31.0/examples/dreambooth/train_dreambooth_lora_flux.py'
)


def main():
    if TRAIN_SCRIPT.exists():
        print(f'Already present: {TRAIN_SCRIPT}')
    else:
        print(f'Downloading {URL}')
        urllib.request.urlretrieve(URL, TRAIN_SCRIPT)
        print(f'Saved to {TRAIN_SCRIPT}')

    print('\n--- Next steps ---')
    print('1) pip install:')
    print('   pip install -U diffusers==0.31.0 accelerate peft transformers \\')
    print('       sentencepiece protobuf bitsandbytes Pillow tqdm')
    print()
    print('2) accelerate config default')
    print()
    print('3) (Flux.1-dev only) huggingface-cli login   # accept license at')
    print('   https://huggingface.co/black-forest-labs/FLUX.1-dev')
    print('   FLUX.1-schnell is open and works without auth.')
    print()
    print('4) python ../caption_images.py     # captions the 271 images (BLIP)')
    print()
    print('5) python generate.py --no-lora    # baseline images (before LoRA)')
    print()
    print('6) python train_flux.py            # train LoRA (~2-6 hours)')
    print()
    print('7) python generate.py --lora       # generate same prompts WITH LoRA')
    print()
    print('8) python compare.py               # build side-by-side comparison grid')


if __name__ == '__main__':
    main()
