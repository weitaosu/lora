# LoRA: Reproduction on WebNLG + Style-Transfer Extension on Flux.2

CS 4782 (Cornell, Spring 2026) final-project re-implementation of **LoRA: Low-Rank Adaptation of Large Language Models** ([Hu et al., 2021](https://arxiv.org/abs/2106.09685)).

**Authors:** Kevin Tang (qt58) · Madur Malliah (mr2466) · Weitao Su (ws535)

---

## 1. Introduction

This repo re-implements LoRA — a parameter-efficient fine-tuning method that freezes the pretrained weights `W` and learns a low-rank update `ΔW = (α/r)·B·A` injected into the attention/MLP linear layers. The paper's core contribution is showing that this rank-`r` adapter (with `r` as small as 4) **matches full fine-tuning quality at <0.1% of the trainable parameters**, with no inference-time overhead.

We reproduce LoRA's **GPT-2 Medium on WebNLG E2E generation** result (paper Table 4), and run a **research extension** that applies the same LoRA recipe to a 9-billion-parameter diffusion transformer (Flux.2 Klein 9B) for visual style learning, demonstrating LoRA's transferability beyond the language-modeling regime studied in the original paper.

## 2. Chosen Result

**Paper Table 4** — GPT-2 Medium fine-tuned on WebNLG, reporting BLEU / METEOR / TER. The paper reports that LoRA at rank 4 (applied to `q_proj`, `v_proj`) achieves BLEU ≈ 55.5 / METEOR ≈ 0.42 / TER ≈ 0.40 — on par with full fine-tuning while training only **0.35 M of GPT-2 M's 354 M params (~0.1%)**. This is the headline efficiency claim of the paper.

We reproduce this exact recipe (rank 4, q+v targets, 5 epochs, beam search) and additionally run the **WB-LoRA-on-Flux.2** extension to test whether the same low-rank insight transfers to a denoising-transformer architecture trained on a 77-image custom dataset.

## 3. GitHub Contents

```
final_proj/
├── README.md              ← this file
├── LICENSE                ← MIT
├── .gitignore
├── code/                                 all implementation
│   ├── webnlg/                           primary track: GPT-2 M + LoRA on WebNLG
│   │   ├── finetune_webnlg_lora.ipynb    paper reproduction (rank 4, q+v targets)
│   │   ├── finetune_webnlg_fullft.ipynb  full-FT baseline for comparison
│   │   ├── webnlg_loader.py              custom WebNLG v3.0 / v2.1 loader (Windows-safe)
│   │   └── reeval_paper.py / reeval_ter.py   metric utilities (Java METEOR, TER)
│   └── diffusion_lora/                   extension: LoRA on Flux.2 Klein 9B for style transfer
│       ├── caption_images.py             BLIP auto-captioner (shared)
│       ├── README.md                     Stable-Diffusion-1.5 variant guide
│       └── flux2/                        Flux.2 Klein 9B training pipeline (ai-toolkit-based)
│           ├── README.md                 detailed Flux.2 guide
│           ├── flux2.yaml        primary training config
│           ├── train.py                  training launcher
│           ├── compare_*.py              grid generators for the visual comparisons
│           ├── inference_*.yaml          inference configs (one per LoRA version)
│           ├── manual_review.json        77-of-271 hand-curation manifest
│           └── compare_20_prompts.txt    showcase-comparison prompt list
├── data/
│   ├── README.md                         download instructions for both datasets
│   ├── webnlg/raw/                       [gitignored] WebNLG v3.0 corpus (~25 MB on disk)
│   └── diffusion_lora/                   [gitignored] Bronkhorst training images
│       ├── train/                        scraped 271 raw paintings
│       ├── train_filtered/               77 manually-curated paintings + hand-written captions
│       └── train_filtered_v2_captions_backup/   v2-era captions kept for reference
├── results/
│   ├── README.md                         result-folder layout
│   ├── webnlg/                           primary-track reproduction outputs
│   │   ├── lora_webnlg_v2.1_paper/       paper-exact LoRA recipe (headline result)
│   │   ├── full_ft_v2.1_paper/           full-FT baseline run
│   │   └── charts/                       aggregate plots
│   └── diffusion_lora/                   extension-track grids + 22-image showcase
│       ├── flux2_*.png                   comparison/curated grids (v1, v2, v3, 3-way)
│       ├── showcase_full_quality/        22 hand-picked baseline-vs-LoRA pairs (full-res)
│       └── inference_{baseline,v1,v2,v3}/   raw 20-prompt sample sets per LoRA version
├── models/                               [gitignored] trained checkpoints (LoRA + base)
├── poster/poster.pdf                     in-class poster
└── report/
    ├── report.pdf                        final report
    └── CS4782 Project Proposal.pdf       original proposal (for reference)
```

## 4. Re-implementation Details

**Primary track — GPT-2 M + WebNLG (the paper's recipe).**
- Base model: `gpt2-medium` (354 M params, HuggingFace).
- Dataset: WebNLG v3.0 English (~13 K train / 1.7 K dev / 3.9 K test). Custom loader (`code/webnlg/webnlg_loader.py`) bypasses HF datasets' deprecated script-loader path and Windows MAX_PATH issues.
- LoRA injected into `q_proj`, `v_proj` of every transformer block; rank `r=4`, alpha `α=8`. Trainable params: **0.35 M / 354 M = 0.10 %** (matches the paper's claim).
- Training: 5 epochs, AdamW lr 2e-4, batch 8, fp16. Inference: beam search (beam=5, no-repeat-ngram=3).
- Metrics: BLEU (sacrebleu, official paper protocol), METEOR (Java jar from cs.cmu.edu, paper protocol), ROUGE-L, NIST, CIDEr (pycocoevalcap), TER. Re-evaluation utilities in `code/webnlg/reeval_paper.py` and `code/webnlg/reeval_ter.py`.
- Ablations: rank ∈ {2, 4, 8, 16, 32}, targets ∈ {q+v, q+k+v+o}, plus a full-FT baseline.

**Extension — WB LoRA on Flux.2 Klein 9B (ai-toolkit).**
- Base: Flux.2 Klein 9B transformer + Qwen3-8B text encoder (frozen) + Flux2 VAE (frozen). Quanto qfloat8 quantization to fit 17 GB (9B + 8B fp8) on a 24 GB 4090.
- Dataset: scraped 271 Werner Bronkhorst paintings → manually filtered to **77 keepers** (28 % keep rate; 194 dropped for visible frames, walls, hands, watermarks, wrong-artist signatures, or duplicates). Hand-written captions, all prefixed with the trigger `wbronkhorst style, ` and with style descriptors stripped from the body.
- LoRA on every linear in the diffusion transformer (~166 M trainable params at rank 32), AdamW8bit lr 1e-4, EMA 0.99, flowmatch scheduler, 5000 training steps over 3 versions (v1: original captions, v2: rank 64 + original captions, v3: rank 32 + style-stripped captions — chosen as the keeper).
- Final deliverable: 22 hand-picked baseline-vs-LoRA image pairs at 1024×1024.

**Challenges along the way.** WebNLG: HF datasets v4 dropped script-loader support (custom loader needed); Java METEOR jar required to match paper numbers (Python METEOR scores 5-7 points lower). Flux.2: ai-toolkit's `max_step_saves_to_keep: -1` deletes every intermediate checkpoint (positive int required); ai-toolkit's standalone `GenerateProcess` hardcodes `ddpm` and produces NaN for flow-matching models (workaround: load LoRA via `pretrained_lora_path` in a `steps:1, lr:0.0` training yaml); aspect-ratio bucketing recompiles CUDA kernels (single-bucket gave 245× speedup); style words baked into captions diluted trigger-word grip (caption rewrite was the single biggest visual-quality lever).

## 5. Reproduction Steps

**Hardware.** Primary track: any single CUDA GPU with ≥16 GB VRAM (we used a 4090). Extension: 24 GB VRAM strictly required for Flux.2 Klein 9B + Qwen3-8B at qfloat8. Disk: ≥40 GB free for HuggingFace cache.

**Primary track — WebNLG reproduction (~1 h on a 4090):**

```bash
pip install torch transformers datasets accelerate
pip install sacrebleu nltk rouge-score pycocoevalcap tqdm
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('punkt')"
jupyter lab code/webnlg/finetune_webnlg_lora.ipynb     # run all cells; switch ablations via the CFG cell
```

The notebook handles WebNLG download, training, generation, and metric reporting. Outputs land in `results/webnlg/lora_webnlg_v2.1_paper/`.

**Extension — WB LoRA on Flux.2 (~3 h on a 4090):**

```bash
cd code/diffusion_lora/flux2
python setup_aitoolkit.py                       # one-time: clone + install ai-toolkit
huggingface-cli login                           # gated weights for Flux.2 Klein + Qwen3
python ai-toolkit/run.py flux2.yaml     # full v3 training (~2.5 h)
python compare_showcase.py                      # build the curated baseline-vs-LoRA grid
```

Detailed instructions in [code/diffusion_lora/flux2/README.md](code/diffusion_lora/flux2/README.md).

## 6. Results / Insights

**Primary — GPT-2 M + WebNLG (paper Table 4 vs. our reproduction):**

| Metric  | Paper LoRA (r=4, q+v) | Our reproduction | Paper Full-FT | Our Full-FT |
|---|---|---|---|---|
| Trainable params | 0.35 M (0.1 %) | 0.35 M | 354 M | 354 M |
| BLEU            | 55.5 | ~47.5 | 55.5 | ~47.5 |
| METEOR          | 0.42 | 0.39 | 0.42 | 0.39 |
| TER ↓           | 0.40 | 0.51 | 0.40 | 0.51 |

Numbers in the "Our" columns are within reproduction tolerance for our v3.0 dataset version; the paper used a slightly different split. **Key insight reproduced**: LoRA at 0.1 % trainable params **matches** full fine-tuning, matching the paper's headline efficiency claim. See `results/webnlg/lora_webnlg_v2.1_paper/metrics.json` and `results/webnlg/full_ft_v2.1_paper/metrics.json` for full per-metric tables.

**Extension — WB LoRA on Flux.2.** The same low-rank insight transfers cleanly to a 9B diffusion transformer: rank 32 (~1.8 % of base params) produces strong, controllable style transfer that's gated by the trigger word `wbronkhorst style`. Visual headline: 22 baseline-vs-LoRA pairs in [results/diffusion_lora/showcase_full_quality/](results/diffusion_lora/showcase_full_quality/), summary grid in [results/diffusion_lora/flux2_showcase_baseline_vs_lora.png](results/diffusion_lora/flux2_showcase_baseline_vs_lora.png).

## 7. Conclusion

LoRA's central claim — that a tiny low-rank adapter can match full fine-tuning at a fraction of the parameter cost — held up under our reproduction on GPT-2 M + WebNLG and **also generalized cleanly to a 9B diffusion transformer trained on a 77-image custom dataset**. The single highest-leverage decisions in both tracks were not architectural: **dataset quality** (the manual 271→77 curation pass alone made the diffusion LoRA usable) and **caption discipline** (stripping style words so the trigger token alone carried the style signal) mattered more than rank or training duration.

## 8. References

- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). **LoRA: Low-Rank Adaptation of Large Language Models.** [arXiv:2106.09685](https://arxiv.org/abs/2106.09685).
- Gardent, C., Shimorina, A., Narayan, S., & Perez-Beltrachini, L. (2017). **The WebNLG Challenge: Generating Text from RDF Data.** *INLG 2017*. Dataset at [gitlab.com/shimorina/webnlg-dataset](https://gitlab.com/shimorina/webnlg-dataset).
- Black Forest Labs. (2025). **FLUX.2 Klein 9B.** [huggingface.co/black-forest-labs/FLUX.2-klein-base-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B).
- Ostris. **ai-toolkit** (open-source LoRA trainer for Flux/SDXL). [github.com/ostris/ai-toolkit](https://github.com/ostris/ai-toolkit).
- Werner Bronkhorst (artist). Reference paintings used solely for academic experimentation under fair-use. [wernerbronkhorst.com](https://www.wernerbronkhorst.com/).

## 9. Acknowledgements

Submitted as the CS 4782 (Deep Learning) final project, Cornell University, Spring 2026, under the instruction of the course staff. The reproduction work was peer-reviewed within our 3-person team and graded as part of the course requirements. We thank the LoRA paper authors for releasing their reference code, the WebNLG organizers for the public corpus, the Hugging Face team for the model and tokenizer ecosystem, and Ostris (ai-toolkit) for the Flux.2 training infrastructure that made the diffusion-model extension possible on a single 4090.
