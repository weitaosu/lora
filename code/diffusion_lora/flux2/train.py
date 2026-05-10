"""Launch ai-toolkit training using flux2.yaml.

This is a thin wrapper around `python <ai-toolkit>/run.py flux2.yaml`
that pre-flight-checks for common gotchas:

  - ai-toolkit cloned at the expected path
  - data/.../metadata.jsonl present (BLIP captions ran)
  - data/.../*.txt sidecars present (captions_to_txt.py ran)
  - Hugging Face token cached (Qwen3-8B and Flux.2 weights are gated)

Usage:
    python train.py
    python train.py --resume                  # continue from latest checkpoint
    python train.py --config alt_config.yaml  # different YAML config
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent           # final_proj/
DATA = ROOT / 'data' / 'diffusion_lora' / 'train'
AI_TOOLKIT = HERE / 'ai-toolkit'           # user installed it here
DEFAULT_CONFIG = HERE / 'flux2.yaml'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(DEFAULT_CONFIG))
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()

    # Pre-flight checks
    runner = AI_TOOLKIT / 'run.py'
    if not runner.exists():
        print(f'Missing {runner}. Run:  python setup_aitoolkit.py')
        sys.exit(1)
    if not (DATA / 'metadata.jsonl').exists():
        print('Missing metadata.jsonl. Run:  python ../caption_images.py')
        sys.exit(1)
    txt_files = list(DATA.glob('*.txt'))
    if len(txt_files) < 100:
        print(f'Only {len(txt_files)} .txt sidecar files found in {DATA}.')
        print('Run:  python captions_to_txt.py')
        sys.exit(1)

    cmd = [sys.executable, str(runner), str(args.config)]
    if args.resume:
        cmd.append('--recover')
    print('Launching ai-toolkit:\n  ' + ' '.join(cmd) + '\n')
    print('First run downloads ~30 GB of weights from HuggingFace; please wait.')
    subprocess.run(cmd, check=True, cwd=str(AI_TOOLKIT))


if __name__ == '__main__':
    main()
