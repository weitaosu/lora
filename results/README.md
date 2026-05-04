# Results

Two tracks share this folder.

## Primary track — WebNLG GPT-2 Medium reproduction

Each run writes to its own subfolder. Contents per run:

- `lora_adapter.pt` — saved LoRA parameters (small, ~5 MB) [gitignored]
- `train_log.json` — per-step and per-epoch losses
- `predictions.jsonl` — one line per test example: `{src, pred, refs}`
- `metrics.json` — final BLEU / NIST / METEOR / ROUGE-L / CIDEr + efficiency stats (trainable params, peak GPU mem, throughput) and the run config

| Folder | What it is |
|---|---|
| `lora_webnlg/` | default exploratory run (r=4, q+v) |
| `lora_webnlg_v2.1/` | v2.1 dataset variant |
| `lora_webnlg_v2.1_paper/` | **paper-exact recipe** — the headline reproduction |
| `full_ft/` | full fine-tuning baseline (exploratory) |
| `full_ft_v2.1/`, `full_ft_v2.1_paper/` | full-FT comparison runs |
| `charts/` | aggregate plots (BLEU/METEOR vs rank, etc.) |

To switch ablations, edit the `CFG['out_dir']` cell at the top of [code/finetune_webnlg_lora.ipynb](../code/finetune_webnlg_lora.ipynb).

## Extension track — WB LoRA on Flux.2 Klein 9B

`wb_lora_compare/` holds visual deliverables for the diffusion-model extension:

| File | What it shows |
|---|---|
| `flux2_grid_all_steps.png`, `flux2_grid_0_vs_3000.png` | v1 (original 271-image dataset, rank 32, 3000 steps) |
| `flux2_v2_grid_all_steps.png`, `flux2_v2_grid_0_vs_5000.png` | v2 (filtered 77 images, rank 64, 5000 steps — over-stylized) |
| `flux2_v3_grid_all_steps.png`, `flux2_v3_keysteps_grid.png`, `flux2_v3_grid_0_vs_5000.png` | v3 (filtered 77 + style-stripped captions, rank 32, 5000 steps — **the keeper**) |
| `flux2_v1_v2_v3_3way.png` | v1 vs v2 vs v3 final-checkpoint A/B |
| `flux2_v1_baseline_vs_step3000_horizontal.png` | v1 baseline vs trained, horizontal layout |
| `flux2_20x4_compare.png` | 20 fresh prompts × 4 versions (the spreadsheet used to pick favorites) |
| `flux2_showcase_baseline_vs_lora.png` | **curated final showcase** — 11 picks, baseline-vs-LoRA, horizontal 2-row |
| `showcase_full_quality/` | the same 11 picks as 22 individual full-resolution 1024×1024 JPGs |

The training-time per-step samples themselves live in `models/wb_lora_flux2/<run-name>/samples/` (gitignored — too large to commit).
