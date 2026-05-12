import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data (CSV-formatted text file)
df = pd.read_csv("data.txt")

# Extract BLEU scores
paper_ft = df[df["Model"] == "FT Paper"]["BLEU"].values[0]
paper_lora = df[df["Model"] == "LoRA Paper"]["BLEU"].values[0]

our_ft = df[df["Model"] == "Our FT"]["BLEU"].values[0]
our_lora = df[df["Model"] == "Our LoRA"]["BLEU"].values[0]

# Compute % differences (LoRA relative to FT)
paper_pct = (paper_lora - paper_ft) / paper_ft * 100
our_pct = (our_lora - our_ft) / our_ft * 100

# Prepare plot
labels = ["Paper", "Our"]
values = [paper_pct, our_pct]

x = np.arange(len(labels))

fig, ax = plt.subplots()

# Bars
bars = ax.bar(x, values, color=["blue", "red"], alpha=0.7)

# Formatting
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("% Difference (LoRA vs FT)")
ax.set_title("Relative Improvement of LoRA over FT (BLEU)")
ax.axhline(0, color='black', linewidth=1)

# Force y-axis to ±10%
ax.set_ylim(0, 10)

# Annotate values on bars
for i, v in enumerate(values):
    ax.text(i, v + (0.3 if v >= 0 else -0.7), f"{v:.2f}%", ha='center')

plt.tight_layout()
plt.show()