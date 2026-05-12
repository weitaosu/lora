import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Load data
df = pd.read_csv("param_valloss_data.txt")

# Convert percent string -> float
df["param_pct"] = df["Parameter % out of Baseline Model"].str.replace("%", "").astype(float)

# Convert to fraction (so 1% = 0.01)
df["param_frac"] = df["param_pct"] / 100.0

# Sort and reset index
df = df.sort_values("param_frac").reset_index(drop=True)

# Labels
rank_labels = ["R1", "R2", "R4", "R8", "R16", "R32"]

plt.figure(figsize=(7, 4))
plt.plot(df["param_frac"], df["val loss"], marker="o")

# X-axis limit (1%)
plt.xlim(0, 0.01)
plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2g'))

# Y-axis rescaled to make plot flatter
plt.ylim(2.40, 2.60)

plt.gca().yaxis.set_major_locator(mticker.MultipleLocator(0.05))
# plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3g'))

# Baseline line (100% model)
baseline_val_loss = 2.445303
plt.axhline(
    y=baseline_val_loss,
    linestyle="--",
    color="red",
    linewidth=1,
    label="Baseline FT Model (100% Parameters)"
)

# Annotate points
for i, row in df.iterrows():
    if i == 0:
        # keep first label inside plot
        plt.annotate(
            rank_labels[i],
            (row["param_frac"], row["val loss"]),
            textcoords="offset points",
            xytext=(10, -10),
            fontsize=9,
            ha="left"
        )
    else:
        plt.annotate(
            rank_labels[i],
            (row["param_frac"], row["val loss"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=9,
            ha="left"
        )

# Labels and styling
plt.xlabel("Parameter % of Baseline Model")
plt.ylabel("Validation Loss")
plt.title("Validation Loss vs Parameter Percentage")
plt.grid(True)
plt.legend()

plt.show()