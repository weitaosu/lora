"""Download the official diffusers LoRA training script and create launcher
config. Run once before training.
"""
import os
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAIN_SCRIPT = HERE / 'train_text_to_image_lora.py'
URL = (
    'https://raw.githubusercontent.com/huggingface/diffusers/'
    'v0.30.3/examples/text_to_image/train_text_to_image_lora.py'
)


def main():
    if TRAIN_SCRIPT.exists():
        print(f'Already present: {TRAIN_SCRIPT}')
    else:
        print(f'Downloading {URL}')
        urllib.request.urlretrieve(URL, TRAIN_SCRIPT)
        print(f'Saved to {TRAIN_SCRIPT}')

    print('\nNext steps:')
    print('  1) pip install diffusers==0.30.3 accelerate peft transformers datasets bitsandbytes')
    print('  2) accelerate config default')
    print('  3) python caption_images.py            # captions all images')
    print('  4) python -m accelerate launch train_text_to_image_lora.py [args]   '
          'or run the launcher train.py')


if __name__ == '__main__':
    main()
