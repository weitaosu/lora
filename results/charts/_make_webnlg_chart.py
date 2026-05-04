"""WebNLG paper-exact charts: quality (U/S/A) + efficiency (train/inference).

Reads the paper-recipe runs:
  - results/lora_webnlg_v2.1_paper/metrics.json
  - results/full_ft_v2.1_paper/metrics.json
Compares against LoRA paper Table 14 (GPT-2 M).
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('c:/Users/weita/Desktop/deep_learning/final_proj/results')
ours_lora = json.load(open(ROOT / 'lora_webnlg_v2.1_paper' / 'metrics.json'))
ours_ft   = json.load(open(ROOT / 'full_ft_v2.1_paper'    / 'metrics.json'))

PAPER = {
    'FT': {
        'Unseen': {'BLEU': 27.7, 'METEOR': 0.30, 'TER': 0.76},
        'Seen':   {'BLEU': 64.2, 'METEOR': 0.45, 'TER': 0.33},
        'All':    {'BLEU': 46.5, 'METEOR': 0.38, 'TER': 0.53},
    },
    'LoRA': {
        'Unseen': {'BLEU': 46.7, 'METEOR': 0.38, 'TER': 0.46},
        'Seen':   {'BLEU': 62.1, 'METEOR': 0.44, 'TER': 0.33},
        'All':    {'BLEU': 55.3, 'METEOR': 0.41, 'TER': 0.39},
    },
}
OURS = {
    'FT':   ours_ft  ['paper_metrics']['splits'],
    'LoRA': ours_lora['paper_metrics']['splits'],
}

splits  = ['Unseen', 'Seen', 'All']
metrics = ['BLEU', 'METEOR', 'TER']
LOWER_BETTER = {'TER'}

methods = ['FT (Paper)', 'LoRA (Paper)', 'FT (Ours)', 'LoRA (Ours)']
colors  = ['#6c757d', '#adb5bd', '#1f77b4', '#ff7f0e']

# --- Quality chart ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('WebNLG (release_v2.1, paper-exact recipe): Unseen / Seen / All',
             fontsize=14, fontweight='bold')
x = np.arange(len(splits))
width = 0.18

for ax, metric in zip(axes, metrics):
    arrow = '↓ lower better' if metric in LOWER_BETTER else '↑ higher better'
    ax.set_title(f'{metric}  ({arrow})', fontsize=12, fontweight='bold')
    all_vals = []
    for i, method in enumerate(methods):
        offset = (i - 1.5) * width
        if i == 0:   vals = [PAPER['FT'][s][metric]   for s in splits]
        elif i == 1: vals = [PAPER['LoRA'][s][metric] for s in splits]
        elif i == 2: vals = [OURS['FT'][s][metric]    for s in splits]
        else:        vals = [OURS['LoRA'][s][metric]  for s in splits]
        all_vals.extend(vals)
        bars = ax.bar(x + offset, vals, width, label=method, color=colors[i],
                      edgecolor='black', linewidth=0.5)
        for bar, v in zip(bars, vals):
            fmt = f'{v:.2f}' if metric != 'BLEU' else f'{v:.1f}'
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    fmt, ha='center', va='bottom', fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(splits, fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.4); ax.set_axisbelow(True)
    if all_vals:
        lo, hi = min(all_vals), max(all_vals)
        pad = (hi - lo) * 0.12
        ax.set_ylim(max(0, lo - pad), hi + pad * 2.0)

axes[0].legend(loc='upper left', fontsize=9, framealpha=0.95)
plt.tight_layout()
out_path = ROOT / 'charts' / 'webnlg_comparison.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Wrote {out_path}')
plt.close()

# --- Efficiency chart (training + inference) ---
eff_lora = ours_lora['efficiency']
eff_ft   = ours_ft  ['efficiency']

panels = [
    ('Trainable params (M)',
     eff_ft['trainable_params'] / 1e6,
     eff_lora['trainable_params'] / 1e6,
     lambda ft, lo: f'LoRA uses {ft/lo:,.0f}× fewer params'),
    ('Train time (min)',
     eff_ft['train_time_sec'] / 60,
     eff_lora['train_time_sec'] / 60,
     lambda ft, lo: f'LoRA is {ft/lo:.1f}× faster'),
    ('Train peak GPU (GB)',
     eff_ft['train_peak_gpu_gb'],
     eff_lora['train_peak_gpu_gb'],
     lambda ft, lo: f'LoRA uses {ft/lo:.1f}× less GPU'),
    ('Inference time (min)',
     eff_ft['inference_time_sec'] / 60,
     eff_lora['inference_time_sec'] / 60,
     lambda ft, lo: f'≈{ft/lo:.1f}× ratio (similar)' if abs(ft/lo - 1) < 0.5 else f'LoRA {ft/lo:.1f}× faster'),
    ('Inference peak GPU (GB)',
     eff_ft['inference_peak_gpu_gb'],
     eff_lora['inference_peak_gpu_gb'],
     lambda ft, lo: f'LoRA uses {ft/lo:.1f}× less GPU'),
]

fig2, axes2 = plt.subplots(1, 5, figsize=(22, 4.5))
fig2.suptitle('WebNLG (paper-exact, v2.1, Seen-only train): Efficiency — LoRA vs Full FT',
              fontsize=13, fontweight='bold')
for ax, (label, ft_v, lora_v, capf) in zip(axes2, panels):
    bars = ax.bar(['FT', 'LoRA'], [ft_v, lora_v],
                  color=['#1f77b4', '#ff7f0e'], edgecolor='black', linewidth=0.5)
    ax.set_title(label, fontsize=11, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.4); ax.set_axisbelow(True)
    for bar, v in zip(bars, [ft_v, lora_v]):
        fmt = f'{v:,.1f}' if v >= 10 else f'{v:.2f}' if v >= 1 else f'{v:.3f}'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                fmt, ha='center', va='bottom', fontsize=10)
    ax.text(0.5, 0.92, capf(ft_v, lora_v), transform=ax.transAxes, ha='center',
            fontsize=9,
            bbox=dict(facecolor='#fff3cd', edgecolor='#ffeaa7', boxstyle='round,pad=0.3'))
plt.tight_layout()
out_path2 = ROOT / 'charts' / 'webnlg_efficiency.png'
plt.savefig(out_path2, dpi=150, bbox_inches='tight')
print(f'Wrote {out_path2}')
plt.close()
