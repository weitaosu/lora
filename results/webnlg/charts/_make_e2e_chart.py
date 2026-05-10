"""Generate the E2E comparison chart from the user's results table."""
import matplotlib.pyplot as plt
import numpy as np

# Rows in the order they should appear in each metric's bar group.
methods = ['FT (Paper)', 'LoRA (Paper)', 'FT (Ours)', 'LoRA (Ours)', 'FT-CD (Ours)', 'LoRA-CD (Ours)']
metrics = ['BLEU', 'NIST', 'METEOR', 'ROUGE-L', 'CIDEr']

# Values: rows = methods, cols = metrics
data = np.array([
    [68.2,    8.62,     46.2,    71.0,    2.47    ],   # FT Paper
    [70.4,    8.85,     46.8,    71.8,    2.53    ],   # LoRA Paper
    [64.8624, 6.831424, 44.5081, 64.0231, 1.692959],   # FT (Ours)
    [65.5656, 6.895156, 45.0949, 64.8585, 1.799673],   # LoRA (Ours)
    [65.5834, 6.958343, 45.0411, 64.7262, 1.732469],   # FT-CD
    [65.8399, 6.938805, 45.5718, 65.1817, 1.807867],   # LoRA-CD
])

# Color scheme: pair FT/LoRA with similar hue, but distinguish source.
colors = [
    '#6c757d',   # FT Paper - dark gray
    '#adb5bd',   # LoRA Paper - light gray
    '#1f77b4',   # FT Ours - blue
    '#ff7f0e',   # LoRA Ours - orange
    '#9ecae1',   # FT-CD - light blue
    '#ffbb78',   # LoRA-CD - light orange
]

fig, axes = plt.subplots(1, 5, figsize=(18, 5))
fig.suptitle('E2E NLG Challenge: GPT-2 M Fine-tuning Methods', fontsize=14, fontweight='bold')

x = np.arange(len(methods))

for ax, metric_name, col_idx in zip(axes, metrics, range(5)):
    vals = data[:, col_idx]
    bars = ax.bar(x, vals, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_title(metric_name, fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(' (', '\n(') for m in methods], rotation=0, fontsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    # value labels on top of bars
    for bar, v in zip(bars, vals):
        fmt = f'{v:.2f}' if metric_name == 'CIDEr' or metric_name == 'NIST' else f'{v:.1f}'
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), fmt,
                ha='center', va='bottom', fontsize=8)
    # tighten y range so differences are visible
    pad = (vals.max() - vals.min()) * 0.15
    ax.set_ylim(max(0, vals.min() - pad), vals.max() + pad * 2)

plt.tight_layout()
out_path = 'c:/Users/weita/Desktop/deep_learning/final_proj/results/webnlg/charts/e2e_comparison.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Wrote {out_path}')

# also a "delta from paper" chart that shows how close each Ours run is to Paper LoRA
fig2, ax2 = plt.subplots(figsize=(10, 5))
# paper LoRA as reference (row index 1)
ref = data[1]
deltas = (data[2:] - ref) / ref * 100  # % difference from Paper LoRA
ours_methods = methods[2:]
metric_x = np.arange(len(metrics))
width = 0.2
for i, (m, d) in enumerate(zip(ours_methods, deltas)):
    ax2.bar(metric_x + (i - 1.5) * width, d, width, label=m,
            color=colors[i + 2], edgecolor='black', linewidth=0.5)
    for j, v in enumerate(d):
        ax2.text(metric_x[j] + (i - 1.5) * width, v + (0.1 if v >= 0 else -0.5),
                 f'{v:+.1f}%', ha='center', fontsize=7)
ax2.axhline(0, color='black', linewidth=1)
ax2.set_xticks(metric_x)
ax2.set_xticklabels(metrics, fontsize=10)
ax2.set_ylabel('% difference vs LoRA (Paper)')
ax2.set_title('How close are our runs to the paper\'s LoRA baseline?', fontweight='bold')
ax2.legend(loc='lower right', fontsize=9)
ax2.grid(axis='y', linestyle='--', alpha=0.4)
ax2.set_axisbelow(True)
plt.tight_layout()
out2 = 'c:/Users/weita/Desktop/deep_learning/final_proj/results/webnlg/charts/e2e_delta_from_paper.png'
plt.savefig(out2, dpi=150, bbox_inches='tight')
print(f'Wrote {out2}')
