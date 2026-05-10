# Werner Bronkhorst LoRA — **Flux base** version

Trains a LoRA on Flux.1 instead of SD 1.5. Three stages:
1. **Generate baseline images** with the Flux base (no LoRA) using a fixed prompt list
2. **Train the LoRA** on the 271 Bronkhorst images
3. **Generate the same prompts WITH the LoRA**, then build a side-by-side grid

## Files

```
code/diffusion_lora/flux/
├── setup_flux.py     # one-time: downloads diffusers Flux LoRA trainer
├── train_flux.py     # launcher for the LoRA training
├── generate.py       # generate from prompts.txt, with or without LoRA
├── compare.py        # build side-by-side baseline-vs-LoRA grid PNG
├── prompts.txt       # the 8 comparison prompts (edit to taste)
└── README.md         # this file
```

## ⚠️ VRAM reality check

Flux is a 12B-parameter transformer. Even LoRA training is heavy:

| VRAM | Feasible? | Recommended config |
|---|---|---|
| 24 GB | ✓ comfortable | rank 16, res 1024, no quantization |
| 16 GB | ✓ tight | rank 8–16, res 768, `--use_8bit_adam --gradient_checkpointing` |
| **12 GB** | ⚠️ borderline | rank 8, res 512, all the flags above (script's default) |
| <12 GB | ✗ won't fit | use [ai-toolkit](https://github.com/ostris/ai-toolkit) with QLoRA + nf4 instead |

You measured ~11 GB peak on full-FT GPT-2 M, so you have **~12 GB usable**. This is right at the edge for Flux. If `train_flux.py` OOMs, switch to ai-toolkit (more aggressive memory tricks) — see fallback section below.

## Step-by-step

### Step 0 — Make sure you have the captions
The captioning script lives one level up at `code/diffusion_lora/caption_images.py`. Run it once if you haven't:
```bash
cd c:/Users/weita/Desktop/deep_learning/final_proj/code/diffusion_lora
python caption_images.py
# writes data/diffusion_lora/train/metadata.jsonl
```
This is the same metadata the SD 1.5 path uses; we share it.

### Step 1 — Install + setup (one-time, ~5 min)

```bash
cd c:/Users/weita/Desktop/deep_learning/final_proj/code/diffusion_lora/flux
pip install -U diffusers==0.31.0 accelerate peft transformers \
    sentencepiece protobuf bitsandbytes Pillow tqdm
accelerate config default
python setup_flux.py
```

If you go with **FLUX.1-dev** (gated model, better quality, non-commercial license), also:
```bash
huggingface-cli login   # paste an HF token after accepting the license at
                        # https://huggingface.co/black-forest-labs/FLUX.1-dev
```
**FLUX.1-schnell** (the default) is open-licensed (Apache 2.0) and works without auth.

### Step 2 — Generate baseline images (~5 min)

These are "what does Flux produce for these prompts WITHOUT the LoRA?". The trigger word `wbronkhorst style` is in the prompt but means nothing to base Flux yet.

```bash
python generate.py --no-lora
# -> results/diffusion_lora/baseline/00_00__....png  (8 images)
```

Useful flags:
- `--base dev` to use FLUX.1-dev instead of schnell
- `--steps 4` (schnell) or `--steps 20` (dev) — script auto-picks defaults
- `--seed 42` (default) — same seed will be used in step 4

### Step 3 — Train the LoRA (~2–6 hours on 12 GB GPU)

```bash
python train_flux.py
# default: schnell base, rank 8, 512 px, 80 epochs, 8-bit Adam, gradient ckpt
```

Memory-tighter knobs if you OOM:
```bash
python train_flux.py --resolution 384 --rank 8 --accum 8
```

Memory-looser knobs if you have 16+ GB and want better quality:
```bash
python train_flux.py --resolution 768 --rank 16 --epochs 60
```

Output: `models/diffusion_lora/flux1/pytorch_lora_weights.safetensors`
Intermediate checkpoints every 500 steps in `models/diffusion_lora/flux1/checkpoint-N/`.

### Step 4 — Generate WITH the LoRA (~5 min)

Same prompts file, same seed → identical noise vectors → only the LoRA differs.

```bash
python generate.py --lora ../../../models/diffusion_lora/flux1
# -> results/diffusion_lora/lora/00_00__....png  (8 images, matching filenames)
```

Tune the LoRA strength at gen time:
```bash
python generate.py --lora ../../../models/diffusion_lora/flux1 --lora-scale 0.7   # softer
python generate.py --lora ../../../models/diffusion_lora/flux1 --lora-scale 1.2   # stronger
```

### Step 5 — Build the comparison grid

```bash
python compare.py
# -> results/diffusion_lora/grid.png
```

Two-column grid: left = baseline, right = LoRA. Caption underneath each pair shows the prompt. Same prompt + same seed in both columns means **the only difference is the LoRA**.

## What success looks like

- **Right column should pick up Bronkhorst's** painterly brushwork, warm sunset palette, and dramatic single-figure compositions
- The trigger `wbronkhorst style` should be doing real work — try regenerating the LoRA column without it (edit prompts.txt) and see the style weaken or vanish
- Out-of-distribution prompts (astronaut, knight, chef) should still take the *style* even when the *subject* wasn't in training

## What probably won't work

- 271 images is small for Flux — expect the style to be present but maybe not pristine
- High-detail face fidelity (Flux base is great here, LoRA can add some texture)
- If train fails / OOMs, fall back to **ai-toolkit** (see below)

## Fallback: ai-toolkit (if our trainer OOMs)

ai-toolkit is the de-facto Flux LoRA trainer for low-VRAM users. It uses QLoRA + nf4 quantization aggressively.

```bash
git clone https://github.com/ostris/ai-toolkit
cd ai-toolkit
pip install -r requirements.txt

# Use config/examples/train_lora_flux_24gb.yaml as a starting point;
# their README has a 12 GB profile.
# Point train_data_path at:
#   c:/Users/weita/Desktop/deep_learning/final_proj/data/diffusion_lora/train
# and use the same `metadata.jsonl` captions.

python run.py config/your_flux_config.yaml
```

Then the trained LoRA from ai-toolkit can be loaded by our `generate.py` and `compare.py` exactly the same way — just point `--lora` at the ai-toolkit output directory.

## License + ethics

Werner Bronkhorst is a living, copyrighted artist. Train for personal experimentation. **FLUX.1-dev is non-commercial only**; FLUX.1-schnell is Apache 2.0. Whichever you pick, don't redistribute the trained LoRA or generated images publicly without permission from both the model authors and the artist.
