# Werner Bronkhorst Style LoRA — Training Guide

Train a Stable Diffusion LoRA on the 271 Bronkhorst images sitting in [data/wb_lora_train/](../../data/wb_lora_train/).

## Layout

```
final_proj/
├── data/wb_lora_train/         <- 271 images, your training set
│   └── metadata.jsonl          <- (created by step 2; one caption per image)
├── code/wb_lora/
│   ├── caption_images.py       <- step 2: BLIP auto-captions
│   ├── setup.py                <- step 3: download diffusers trainer
│   ├── train.py                <- step 4: launch training
│   ├── inference.py            <- step 5: generate sample images
│   └── train_text_to_image_lora.py   <- (downloaded by setup.py)
└── models/wb_lora/             <- trained LoRA weights land here
```

---

## Step 1 — Install dependencies (~5 min, one-time)

```bash
pip install diffusers==0.30.3 accelerate peft transformers datasets bitsandbytes Pillow tqdm
```

Then configure `accelerate` (one-time):

```bash
accelerate config default
```

This writes `~/.cache/huggingface/accelerate/default_config.yaml`. The `default` flag sets sane Windows defaults (single GPU, fp16). If you have multiple GPUs or want manual control, run plain `accelerate config` instead.

---

## Step 2 — Caption every image (~5–10 min on GPU)

```bash
cd code/wb_lora
python caption_images.py
```

What it does:
- Loads BLIP (Salesforce/blip-image-captioning-large, ~1 GB VRAM)
- For each image in `data/wb_lora_train/`, generates a description like "a man riding a wave on a surfboard"
- Prepends the trigger word: `"wbronkhorst style, "`
- Writes `data/wb_lora_train/metadata.jsonl` (one JSON line per image — the format diffusers expects)

Resulting line:
```json
{"file_name": "bing_art__001.jpg", "text": "wbronkhorst style, a man riding a wave on a surfboard"}
```

**The trigger word `wbronkhorst` is what you'll prompt with at inference time.** It's how the LoRA "knows" you want the trained style. Change it in `caption_images.py` (line ~19) before running if you want a different word.

---

## Step 3 — Get the diffusers trainer (1 min, one-time)

```bash
python setup.py
```

This downloads `train_text_to_image_lora.py` from the official diffusers repo (pinned to v0.30.3 for stability) into `code/wb_lora/`.

---

## Step 4 — Train (~1–2 hours on a 12 GB GPU for SD 1.5)

```bash
python train.py
```

Default config (in `train.py`):
- **Base model:** Stable Diffusion 1.5 (smaller, fits in 8 GB)
- **Resolution:** 768 px
- **LoRA rank:** 32
- **Effective batch:** 1 × 4 (grad accum) = 4
- **Epochs:** 80
- **LR:** 1e-4 with cosine schedule + 100-step warmup
- **Mixed precision:** fp16
- **Gradient checkpointing:** ON (halves activation memory, ~25% slower)

Outputs:
- `models/wb_lora/pytorch_lora_weights.safetensors` — final LoRA weights
- `models/wb_lora/checkpoint-{step}/` — intermediate checkpoints every 500 steps
- `models/wb_lora/logs/` — TensorBoard logs (`tensorboard --logdir models/wb_lora/logs`)

**SDXL upgrade** (better quality, needs 16+ GB VRAM):
```bash
python train.py --base sdxl --resolution 1024 --rank 32
```
Note: SDXL training takes ~3-5 hours and the trainer file is different (`train_text_to_image_lora_sdxl.py`); edit `setup.py`'s URL to fetch that variant if you go this route.

**Common knob tweaks:**
| Symptom | Knob to try |
|---|---|
| Style not strong enough | `--rank 64` or `--epochs 120` |
| Style too strong / overfitting | `--rank 16`, `--epochs 50`, or stop earlier (use checkpoint-1500 instead of final) |
| Hallucinating details | More epochs and lower LR (`--lr 5e-5`) |
| Out of GPU memory | drop `--resolution 512` or set `--accum 2` |

---

## Step 5 — Generate sample images (~30 s per image on GPU)

```bash
python inference.py
```

Default: generates 6 sample images using built-in Bronkhorst-style prompts (lifeguard, surfer, beach, etc.) → `results/wb_lora_samples/`.

**Custom prompt:**
```bash
python inference.py --prompt "wbronkhorst style, a portrait of an astronaut on the beach, oil painting"
```

**Tune the LoRA strength:**
```bash
python inference.py --strength 0.8       # softer effect
python inference.py --strength 1.2       # stronger effect (may overcook)
```

**Use an intermediate checkpoint** (e.g. if final overfit):
```bash
python inference.py --lora ../../models/wb_lora/checkpoint-2000
```

---

## Troubleshooting

**"NameError: pad_token / bnb / 'paged_adamw'..." during training**
→ Drop `bitsandbytes` from launch args, or `pip install bitsandbytes==0.43.3`.

**"OOM" at first epoch**
→ Lower `--resolution 512`, lower `--rank 16`, ensure `--gradient_checkpointing` is on (it is by default in our `train.py`).

**Output looks identical to base SD**
→ LoRA didn't learn. Check:
1. `metadata.jsonl` exists and has 271 lines
2. Each caption starts with `wbronkhorst style,`
3. You're prompting with the trigger at inference

**Output is melted / unrecognizable**
→ Overfit. Use an earlier checkpoint or train for fewer epochs.

**"FileNotFoundError: train_text_to_image_lora.py"**
→ Run `python setup.py` first.

---

## Realistic expectations

271 images is a small-but-workable LoRA dataset. After 80 epochs you should see:
- The model picking up Bronkhorst's color palette (warm tones, dramatic skies)
- Painterly brushwork visible in skin and fabric
- His tendency for dramatic single-figure compositions
- Some signature elements (lifeguard / beach themes) bleeding into unrelated prompts

What it probably **won't** capture from 271 images:
- Detailed face fidelity (small dataset)
- Very rare subject matter not represented in training
- Photorealistic detail of the underlying SD base model

If quality is short, the highest-leverage move is **getting more reference images** (not more epochs).

---

## License + ethics reminder

Werner Bronkhorst is a living, copyrighted artist. This pipeline is for personal experimentation. **Don't redistribute the LoRA weights or generated images publicly without permission**, and consider reaching out to the artist before doing anything visible.
