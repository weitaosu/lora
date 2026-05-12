import pandas as pd
import matplotlib.pyplot as plt

# Read CSV
df = pd.read_csv("param_count_data.txt")

# Keep only selected rows
# keep_rows = ["1", "4", "32", "Full Baseline Model"]
keep_rows = ["1", "2", "4", "8", "16", "32", "Full Baseline Model"]
df = df[df["Rank"].astype(str).isin(keep_rows)]

# Ensure correct order
df["Rank"] = pd.Categorical(df["Rank"].astype(str), categories=keep_rows, ordered=True)
df = df.sort_values("Rank")

# Extract data
ranks = df["Rank"].astype(str).tolist()
params = df["Trainable Parameters"].tolist()

# Even spacing
x = range(len(ranks))

plt.figure(figsize=(8, 5))

# Bars
bars = plt.bar(
    x,
    params,
    width=0.8,
    color="#6BAED6",
    edgecolor="black",
    linewidth=0.6
)

# Bigger axis/tick/title fonts
plt.xticks(x, ranks,fontsize=10)
# plt.yticks(fontsize=13)

plt.xlabel("Rank Sweep Models", fontsize=12)
plt.ylabel("Trainable Parameters (in 100 Millions)", fontsize=12)
plt.title("Trainable Parameter Count on Rank Sweep", fontsize=13)

plt.grid(False)

# Bigger value labels above bars
for bar, val in zip(bars, params):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{val:,}",
        ha="center",
        va="bottom",
        fontsize=11,
        # fontweight="bold"
    )

plt.tight_layout()
plt.show()