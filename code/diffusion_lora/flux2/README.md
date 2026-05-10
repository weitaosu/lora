# Werner Bronkhorst LoRA on **Flux.2 Klein 9B** (ai-toolkit)

Trains a LoRA on the 271 Bronkhorst paintings using **Flux.2 Klein 9B** as the base model, with **Qwen3-8B** as the text encoder.

## Why ai-toolkit (not diffusers)

ai-toolkit's `extensions_built_in/diffusion_models/flux2/` has the canonical Flux.2 Klein adapter (`arch: flux2_klein_9b`). diffusers v0.31.0's Flux LoRA trainer targets Flux.1 with a T5 text encoder — **not compatible** with Flux.2 Klein's Qwen3 architecture.

## Your hardware (verified)

- **GPU:** RTX 4090 (24 GB VRAM) — comfortable for Flux.2 Klein 9B LoRA
- **RAM:** 96 GB — irrelevant for VRAM but useful if ai-toolkit offloads anything to system memory
- **No block-swap needed.** Default config trains rank 32 at 768/1024 res with EMA enabled.

## About your existing fp8 files

The files at `models/diffusion_model/`:
- `flux-2-klein-9b-fp8.safetensors`
- `qwen_3_8b_fp8mixed.safetensors`
- `clip_l.safetensors`
- `ae.safetensors`

These are **ComfyUI-style raw weights for inference** — not the format ai-toolkit's training loader expects. The trainer wants HF repo format (config.json + tokenizer + bf16 weights) for Qwen3, and a specific filename (`flux-2-klein-base-9b.safetensors`) at bf16 for the transformer.

**Plan:** ai-toolkit will download bf16 originals from HuggingFace on first run (~30 GB one-time). Your fp8 files stay put — use them with **ComfyUI** for fast inference after training. The trained LoRA weights are dtype-portable, so the LoRA you train on bf16 base will work fine when loaded onto the fp8 base in ComfyUI.

## Files

```
code/diffusion_lora/
├── caption_images.py                <- BLIP auto-captioner (shared with SD pipeline)
└── flux2/
    │
    │  ── Setup + Training ────────────────────────────────────────────────
    ├── setup_aitoolkit.py           clone + pip install ai-toolkit (one-time)
    ├── train.py                     launch training (runs ai-toolkit/run.py flux2.yaml)
    ├── flux2.yaml           production training config (rank 32, 5000 steps, EMA, 768)
    ├── flux2_smoke.yaml     5-step smoke-test config (verifies pipeline)
    │
    │  ── Captioning ──────────────────────────────────────────────────────
    ├── caption_qwen_vl.py           re-caption with Qwen2-VL-7B-Instruct (better than BLIP)
    ├── captions_to_txt.py           convert metadata.jsonl -> per-image .txt sidecars
    ├── cleanup_captions.py          regex pass 1: strip "painting depicts X" filler
    ├── cleanup_qwen.py              regex pass 2: strip "textured/thick painting features X" patterns
    ├── cleanup_v2.py                regex pass 3: strip remaining room/wall context + explicit overrides
    │
    │  ── Manual data review ──────────────────────────────────────────────
    ├── manual_review.json           77/271 keep/drop manifest (28% keep) + hand-written captions
    │
    │  ── Inference (1 yaml per LoRA version) ─────────────────────────────
    ├── inference_baseline.yaml      sample 20 prompts at step 0 with no LoRA
    ├── inference_v1.yaml            sample with v1 final LoRA (rank 32 @ step 3000)
    ├── inference_v2.yaml            sample with v2 final LoRA (rank 64 @ step 5000)
    ├── inference_v3.yaml            sample with v3 final LoRA (rank 32 @ step 5000) [the keeper]
    ├── compare_20_prompts.txt       20 fixed-seed prompts for cross-version comparison
    │
    │  ── Comparison-grid generators (post-training) ──────────────────────
    ├── prompts.txt                  the 8 sample prompts used during training
    ├── compare.py                   v1: 2-column baseline-vs-final grid (legacy)
    ├── compare_v1_horizontal.py     v1: horizontal 2-row baseline-vs-step3000
    ├── compare_v2.py                v2/v3: --all-steps grid OR --baseline-step/--lora-step grid
    ├── compare_v3_keysteps.py       v3: 4 key step columns (0/1500/3000/5000)
    ├── compare_3way.py              v1 vs v2 vs v3 final-checkpoint grid
    ├── compare_20x4.py              20 prompts × 4 versions (full sweep, used to pick favorites)
    └── compare_showcase.py          curated showcase: 11 picks × baseline-vs-LoRA (final deliverable)
```

**Note on `inference_*.yaml`.** ai-toolkit's standalone `GenerateProcess` (`job: generate`) hardcodes the `ddpm` scheduler and produces NaN output for flow-matching models like Flux.2. Workaround: these yamls use `job: extension` (the working sd_trainer path) with `network.pretrained_lora_path` to load an existing LoRA, plus `steps: 1, lr: 0.0` so no training actually happens — only the step-0 sample fires.

## Step-by-step

### 1. Clone and install ai-toolkit (one-time, ~10 min)

```bash
cd c:/Users/weita/Desktop/deep_learning/final_proj/code/diffusion_lora/flux2
python setup_aitoolkit.py
```

This clones `ostris/ai-toolkit` into `c:/Users/weita/Desktop/deep_learning/tools/ai-toolkit/` (sibling to `final_proj/`), runs its `pip install -r requirements.txt`, and installs a few extras.

### 2. Caption all 271 images (one-time, ~5 min on GPU)

```bash
cd ..    # back to code/diffusion_lora/
python caption_images.py
```

Writes `data/diffusion_lora/train/metadata.jsonl` with BLIP-generated captions, each prefixed with the trigger `wbronkhorst style, `.

### 3. Convert captions to ai-toolkit's format

```bash
cd flux2
python captions_to_txt.py
```

Reads `metadata.jsonl` and writes one `.txt` file per image (e.g. `img1.jpg` → `img1.txt`). ai-toolkit's dataset loader uses the per-image-sidecar convention.

### 4. Authenticate with HuggingFace (one-time)

```bash
huggingface-cli login
```

You need a token because:
- **Qwen3-8B** is gated (accept license at https://huggingface.co/Qwen/Qwen3-8B)
- **Flux.2 Klein 9B** weights are gated (accept license at the appropriate BFL repo)

### 5. Launch training (~3-6 hours on a 4090)

```bash
python train.py
```

This calls `python <ai-toolkit>/run.py flux2.yaml`. First run downloads ~30 GB to your HF cache (set `HF_HOME` if you want it on a specific drive).

What gets trained:
- LoRA rank 32 (alpha 32) on the Flux.2 Klein 9B transformer
- 4000 steps (≈ 14 epochs at 271 imgs / batch 1)
- AdamW-8bit, lr 1e-4, flowmatch noise scheduler
- bf16 mixed precision, gradient checkpointing
- EMA enabled (decay 0.99)
- Multi-aspect resolution buckets [768, 1024]

Per-step samples (8 prompts) are written to `models/diffusion_lora/klein_9b_v1/samples/<step>/` every 500 steps. Step 0 is your baseline.

### 6. Resume if interrupted

```bash
python train.py --resume
```

ai-toolkit picks up from the latest checkpoint.

### 7. Build the comparison grid

```bash
python compare.py
```

Pairs **step-0 samples** (baseline — LoRA is uninitialised) with **final-step samples** (fully trained), same seed, same prompts. Writes a 2-column side-by-side grid to `results/diffusion_lora/flux2_grid.png`.

Useful flags:
```bash
python compare.py --baseline-step 0 --lora-step 4000
python compare.py --width 768                       # bigger grid
```

## Tuning knobs (edit `flux2.yaml`)

| You see | Change |
|---|---|
| Style too weak after 4000 steps | bump `network.linear` to 64; or `train.steps` to 6000 |
| Style overcooks (faces melt) | drop `network.linear` to 16; or use an earlier checkpoint via `--lora-step 2000` |
| Captions getting ignored | drop `caption_dropout_rate` to 0.0 |
| OOM (shouldn't happen on 4090) | set `model.low_vram: true` and `train.gradient_accumulation_steps: 2` |
| Sample images look blurry | bump `sample.sample_steps` to 36 |

## Inference outside ai-toolkit (using your fp8 files in ComfyUI)

After training:
1. Find the trained LoRA at `models/diffusion_lora/klein_9b_v1/klein_9b_v1.safetensors`
2. Copy it into ComfyUI's `models/loras/` folder
3. In your Flux.2 ComfyUI workflow, add a **Load LoRA** node pointing at it, with strength 1.0
4. Use the trigger word `wbronkhorst style` in your prompts

The LoRA was trained against bf16 weights but is precision-portable; it'll work fine on top of your fp8 ComfyUI base.

## Troubleshooting

**"GatedRepoError" / "401 Unauthorized" during model download**
→ `huggingface-cli login` and accept licenses for Qwen3-8B and Flux.2 Klein 9B repos.

**ai-toolkit's `run.py` errors with "unknown arch flux2_klein_9b"**
→ Pull the latest ai-toolkit; the Klein 9B adapter was added relatively recently.

**Disk fills up during HF download**
→ Set `HF_HOME=D:/hf_cache` (or wherever you have ≥40 GB free) before running setup.

**OOM on 4090 (unexpected)**
→ Set `model.low_vram: true` in the YAML, or drop `network.linear` from 32 → 16.

**Want to point ai-toolkit at your local fp8 files anyway?**
→ Theoretically possible but risky. ai-toolkit's `Flux2Klein9BModel.load()` uses `Qwen3ForCausalLM.from_pretrained()` which expects a HF repo dir layout (config.json + tokenizer + weights). Your raw `qwen_3_8b_fp8mixed.safetensors` lacks the metadata. You'd need to either (a) download Qwen3-8B repo separately and substitute weights, or (b) patch ai-toolkit's loader. **Not recommended** — let it download what it needs.
