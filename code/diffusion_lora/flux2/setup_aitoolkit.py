"""One-time setup: clone ai-toolkit, install its dependencies, link our config in.

What this does:
  1. Clones https://github.com/ostris/ai-toolkit into ../../../tools/ai-toolkit
     (sibling to final_proj/, so it doesn't get committed with the project)
  2. Installs ai-toolkit's requirements
  3. Prints next steps

Why ai-toolkit:
  - Best Flux.2 support (has flux2_klein_9b adapter at extensions_built_in/diffusion_models/flux2/)
  - YAML config-driven (no fiddly CLI args)
  - Uses Qwen3 as TE2 (matches Flux.2 Klein 9B architecture)
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent       # final_proj/
AI_TOOLKIT = ROOT.parent / 'tools' / 'ai-toolkit'   # sibling to final_proj


def run(cmd, **kw):
    print(f'\n$ {" ".join(cmd) if isinstance(cmd, list) else cmd}')
    return subprocess.run(cmd, check=True, **kw)


def main():
    AI_TOOLKIT.parent.mkdir(parents=True, exist_ok=True)

    if not AI_TOOLKIT.exists():
        print(f'Cloning ai-toolkit -> {AI_TOOLKIT}')
        run(['git', 'clone', '--depth', '1',
             'https://github.com/ostris/ai-toolkit.git', str(AI_TOOLKIT)])
        print('Updating submodules...')
        run(['git', '-C', str(AI_TOOLKIT), 'submodule', 'update', '--init', '--recursive'])
    else:
        print(f'Already cloned: {AI_TOOLKIT}')

    print('\nInstalling ai-toolkit requirements...')
    req = AI_TOOLKIT / 'requirements.txt'
    if req.exists():
        run([sys.executable, '-m', 'pip', 'install', '-r', str(req)])
    else:
        print(f'  WARNING: {req} not found; you may need to install requirements manually.')

    print('\nInstalling extra deps (BLIP for captioning, Pillow for grids)...')
    run([sys.executable, '-m', 'pip', 'install',
         'transformers', 'sentencepiece', 'protobuf', 'Pillow', 'tqdm', 'imagehash'])

    print(f'\n--- Setup complete ---')
    print(f'ai-toolkit:  {AI_TOOLKIT}')
    print(f'config dir:  {HERE}')
    print()
    print('Next steps:')
    print('  1) python ../caption_images.py        # creates metadata.jsonl (BLIP)')
    print('  2) python captions_to_txt.py          # converts metadata.jsonl -> per-image .txt')
    print('  3) (one-time) huggingface-cli login   # Qwen3-8B and Flux.2 weights need HF auth')
    print('  4) python train.py                    # trains the LoRA via ai-toolkit')
    print('  5) python compare.py                  # builds baseline-vs-LoRA grid')


if __name__ == '__main__':
    main()
