import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load CSV
df = pd.read_csv("data.txt")

datasets = ["E2E", "WebNLG", "DART"]
models = ["Paper FT", "Paper LoRA", "Our FT", "Our LoRA"]

colors = [
    "#4C78A8",
    "#A6C8E0",
    "#2CB1A1",
    "#8EDBD3"
]

# ===================== BLEU =====================

bleu_vals = []
for d in datasets:
    bleu_vals.append([
        df[(df["Dataset"] == d) & (df["Model"] == m)]["BLEU"].values[0]
        for m in models
    ])

bleu_vals = np.array(bleu_vals)

x = np.arange(len(datasets))
bar_width = 0.18
offsets = np.linspace(-1.5 * bar_width, 1.5 * bar_width, len(models))

fig, ax = plt.subplots(figsize=(8, 4))

for i, model in enumerate(models):
    ax.bar(
        x + offsets[i],
        bleu_vals[:, i],
        width=bar_width,
        color=colors[i],
        edgecolor="black",
        linewidth=0.6,
        label=model
    )

ax.set_xticks(x)
ax.set_xticklabels(datasets)

ax.set_ylabel("BLEU")
ax.set_title("BLEU Comparison Across Datasets")
ax.set_ylim(0, 100)

ax.legend(
    title="Models",
    frameon=False,
    labelspacing=0.15,
    handletextpad=0.4,
    borderpad=0.2
)

plt.tight_layout()
plt.show()


# ===================== METEOR =====================

meteor_vals = []
for d in datasets:
    meteor_vals.append([
        df[(df["Dataset"] == d) & (df["Model"] == m)]["METEOR"].values[0]
        for m in models
    ])

meteor_vals = np.array(meteor_vals)

fig, ax = plt.subplots(figsize=(8, 4))

for i, model in enumerate(models):
    ax.bar(
        x + offsets[i],
        meteor_vals[:, i],
        width=bar_width,
        color=colors[i],
        edgecolor="black",
        linewidth=0.6,
        label=model
    )

ax.set_xticks(x)
ax.set_xticklabels(datasets)

ax.set_ylabel("METEOR")
ax.set_title("METEOR Comparison Across Datasets")

ax.set_ylim(0, max(meteor_vals.flatten()) * 1.1)

ax.legend(
    title="Models",
    frameon=False,
    labelspacing=0.15,
    handletextpad=0.4,
    borderpad=0.2
)

plt.tight_layout()
plt.show()