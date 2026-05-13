# LoRA: Low-Rank Adaptation Reimplementation

## 1. Introduction

This repository contains our reimplementation of **LoRA: Low-Rank Adaptation of Large Language Models** (Hu et al., 2021) for CS 4782 (Spring 2026). LoRA's core contribution is freezing the pretrained weight matrix `W` and learning a low-rank update `ΔW = BA` (with `A ∈ ℝ^{r×d}`, `B ∈ ℝ^{d×r}`, `r ≪ d`), which dramatically reduces trainable parameters and GPU memory while matching full fine-tuning performance.

We reproduce the paper's natural language generation (NLG) results on GPT-2 Medium across three datasets, then extend the low-rank insight to a 9B-parameter diffusion transformer to test whether it generalizes beyond autoregressive LLMs.

## 2. Chosen Result

We reproduce the **NLG results on GPT-2 Medium** from the LoRA paper (Tables 2–4 in Hu et al., 2021), covering:

- **E2E NLG Challenge** (Novikova et al., 2017) - BLEU, NIST, METEOR, ROUGE-L, CIDEr
- **DART** (Nan et al., 2020) - BLEU, METEOR, TER
- **WebNLG** (Gardent et al., 2017) - BLEU, METEOR, TER (Seen / Unseen / All)

These results are central to the paper's claim that low-rank adaptation matches full fine-tuning at a fraction of the trainable parameters and GPU memory.

## 3. GitHub Contents

```
lora/
├── code/                    Reimplementation code
│   ├── lora_module.py         LoRA layer + GPT-2 injection (from scratch)
│   ├── e2e/                   E2E fine-tuning notebooks
│   ├── dart/                  DART fine-tuning notebooks 
│   ├── webnlg/                WebNLG fine-tuning notebooks
│   └── diffusion_lora/        Flux.2 9B diffusion-LoRA pipeline
├── data/                    Dataset files / download instructions
├── results/                 Metrics, predictions, charts, generated images
├── report/report.pdf        Final report
├── poster/poster.pdf        In-class poster
├── LICENSE
└── README.md
```

## 4. Re-implementation Details

**Model.** GPT-2 Medium (~355 M params), chosen to fit our compute while still being large enough to stress-test LoRA.

**LoRA module.** Built from scratch in `code/lora_module.py`. Wraps GPT-2's `Conv1D` layers and injects rank-`r` updates into the **q** and **v** projections of every attention block (matching the paper). Base weights frozen, only `lora_A` / `lora_B` parameters train.

**Training setup.** We follow the paper's hyperparameters (Fig. 4 in our report) and the Prefix-Tuning baseline (Li & Liang, 2021) for full FT comparisons. 

Key deviation: we replaced the paper's `" || "` separator between meaning representation and target with a fresh special token `<|SEP|>` added to the tokenizer. Empirically this gave noticeably better results on E2E (the paper's separator carried prior learned semantics that hurt training).

**Metrics.** BLEU and METEOR across all three datasets, plus dataset-specific NIST/ROUGE-L/CIDEr (E2E) and TER (DART, WebNLG). We also recorded peak GPU memory and trainable parameter counts.

**Diffusion Extension.** As an out-of-scope test of LoRA's generality, we trained a 165 M-param style LoRA on Flux.2 Klein 9B (Black Forest Labs, 2025) using 77 manually-curated paintings by Werner Bronkhorst. We use the same low-rank math, applied to attention + MLP linear layers. See `code/diffusion_lora/README.md` for the full pipeline.

**Challenges.** The biggest obstacle was undocumented preprocessing in the original paper - the LoRA paper, Prefix-Tuning paper, and original dataset papers all disagreed on input formatting and certain hyperparameters. Considering this, our numbers land within ~80 % of the paper's, with the same trends.

## 5. Reproduction Steps

### Environment

- **Hardware:** Single GPU with ≥ 12 GB VRAM (we used RTX 4090 / Colab A100). The diffusion LoRA needs 24 GB VRAM.
- **Python:** 3.10+
- **Python Dependencies:** `torch`, `transformers`, `datasets`, `nltk`, `sacrebleu`, `jupyter`. The diffusion extension additionally needs `diffusers==0.30.3`, `accelerate`, `peft`, `bitsandbytes`, `Pillow`.

```bash
git clone https://github.com/weitaosu/lora.git lora
cd lora
pip install torch transformers datasets nltk sacrebleu jupyter
```

### Reproduce the GPT-2 + LoRA results

Each dataset has paired notebooks for full fine-tuning and LoRA:

```bash
jupyter lab code/e2e/finetune_e2e_lora.ipynb        # or finetune_e2e_fullft.ipynb
jupyter lab code/dart/finetune_dart_lora.ipynb      # or finetune_dart_fullft.ipynb
jupyter lab code/webnlg/finetune_webnlg_lora.ipynb  # or finetune_webnlg_fullft.ipynb
```

Run cells top-to-bottom. Each notebook loads its dataset via the matching `*_loader.py`, injects LoRA via `code/lora_module.py`, fine-tunes GPT-2 Medium, and dumps predictions + metrics into `results/<dataset>/`.

For WebNLG, re-evaluation against paper-style references uses:

```bash
python code/webnlg/reeval_paper.py
python code/webnlg/reeval_ter.py
```

### Reproduce the diffusion LoRA

See `code/diffusion_lora/README.md` for the 5-step pipeline (caption → setup → train → infer). Default config trains for 80 epochs on SD 1.5 (~1–2 h on a 12 GB GPU); the Flux.2 9B run reported in the paper takes ~3 h on a 24 GB GPU.

## 6. Results / Insights

Across all three NLG datasets, **LoRA matches or outperforms full fine-tuning**, reproducing the paper's central trend.

| Dataset | Metric | Paper FT | Paper LoRA | **Our FT** | **Our LoRA** |
|---|---|---|---|---|---|
| E2E    | BLEU↑    | 68.2 | 70.4 | 65.5 | **65.8** |
| E2E    | METEOR↑  | 46.2 | 46.8 | 45.0 | **45.6** |
| WebNLG | BLEU↑ (All)   | 46.5 | 55.3 | 41.3 | **47.5** |
| WebNLG | METEOR↑ (All) | 0.38 | 0.41 | 0.33 | **0.39** |
| DART   | BLEU↑    | 46.2 | 47.1 | 33.6 | **37.3** |
| DART   | METEOR↑  | 0.39 | 0.39 | 0.31 | **0.33** |

Other metrics have been excluded here. See `results/{dataset}/{dataset}_metric_results.png` for full comparisons

**GPU memory** dropped by ~1.4× with LoRA (vs. the paper's 3×. We attribute the gap to differences in accelerator, as the paper used V100s). **Rank ablation** shows diminishing returns past `r = 4`, supporting the paper's "low rank suffices" claim.

**Diffusion extension:** the 165 M-param LoRA (≈ 1.8 % of Flux.2's 9 B params) produces strong, reliably-triggered Bronkhorst-style transfer that preserves subject fidelity across in- and out-of-distribution prompts, confirming LoRA's low-rank insight transfers cleanly from autoregressive LMs to diffusion transformers, provided the data and prompt pipeline are right.

Other interesting generated charts and predictions live under `results/`. (e.g.`results/diffusion_lora/flux2_showcase_baseline_vs_lora.png`).

## 7. Conclusion

Reimplementing LoRA confirmed its central claim: a tiny low-rank update can match full fine-tuning while cutting trainable parameters and GPU memory substantially. Our biggest practical lesson was that **input formatting matters more than hyperparameters**. Swapping the paper's `||` separator for a dedicated special token gave the largest single quality jump, and undocumented preprocessing was the main reason our numbers lag the paper's by a small margin. Extending the technique to a 9B diffusion transformer worked first try once data quality and trigger-word discipline were right, suggesting the low-rank insight is genuinely architecture-agnostic.

## 8. References

- E. J. Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685, 2021.
- A. Radford et al. *Language Models are Unsupervised Multitask Learners.* OpenAI, 2019.
- X. L. Li and P. Liang. *Prefix-Tuning: Optimizing Continuous Prompts for Generation.* arXiv:2101.00190, 2021.
- J. Novikova, O. Dušek, V. Rieser. *The E2E Dataset: New Challenges for End-to-End Generation.* SIGDIAL 2017.
- L. Nan et al. *DART: Open-Domain Structured Data Record to Text Generation.* arXiv:2007.02871, 2020.
- C. Gardent et al. *The WebNLG Challenge: Generating Text from RDF Data.* INLG 2017.
- Black Forest Labs. *Flux.2: Frontier Visual Intelligence.* https://github.com/black-forest-labs/flux2, 2025.

## 9. Acknowledgements

This project was completed as the final project for **CS 4782: Introduction to Deep Learning** at Cornell University. We thank the course staff for guidance and feedback throughout the semester. The diffusion-LoRA extension uses the [`ai-toolkit`](https://github.com/ostris/ai-toolkit) trainer and Hugging Face's [`diffusers`](https://github.com/huggingface/diffusers) library; the GPT-2 reimplementation builds on Hugging Face [`transformers`](https://github.com/huggingface/transformers).
