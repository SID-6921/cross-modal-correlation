"""
Generates the paper's summary figure: Barlow Twins vs. InfoNCE retrieval
performance at closed-set scale (Section 4.3) and open-vocabulary scale
(Section 4.4), side by side, showing the reversal that is this paper's
central finding. All numbers are the real reported means/stds from the
paper's own tables -- no new computation, just visualization.
"""
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))

# --- Closed-set (Section 4.3): top-1 cross-modal retrieval, 5 seeds ---
ax = axes[0]
labels = ["Barlow\nTwins", "InfoNCE"]
means = [99.00, 96.24]
stds = [0.43, 6.29]
colors = ["#2c7fb8", "#d95f02"]
bars = ax.bar(labels, means, yerr=stds, capsize=6, color=colors, width=0.55, zorder=3)
ax.axhline(10, color="gray", linestyle="--", linewidth=1, label="chance (10%)")
ax.set_ylim(0, 112)
ax.set_ylabel("Top-1 cross-modal retrieval (%)")
ax.set_title("Closed-set scale\n(MNIST + real speech, 10 classes)", fontsize=10)
ax.legend(fontsize=8, loc="lower left")
for b, m, s in zip(bars, means, stds):
    ax.text(b.get_x() + b.get_width() / 2, m + s + 2.5, f"{m:.1f}%", ha="center", fontsize=9)

# --- Open-vocabulary (Section 4.4): Recall@10, 3 seeds ---
ax = axes[1]
means2 = [10.00, 41.27]
stds2 = [0.52, 1.22]
bars2 = ax.bar(labels, means2, yerr=stds2, capsize=6, color=colors, width=0.55, zorder=3)
ax.axhline(0.2, color="gray", linestyle="--", linewidth=1, label="chance (0.2%)")
ax.set_ylim(0, 50)
ax.set_ylabel("Recall@10 (%)")
ax.set_title("Open-vocabulary scale\n(Flickr8k, 5,000 real captions)", fontsize=10)
ax.legend(fontsize=8, loc="upper left")
for b, m, s in zip(bars2, means2, stds2):
    ax.text(b.get_x() + b.get_width() / 2, m + s + 1.5, f"{m:.1f}%", ha="center", fontsize=9)

fig.suptitle("The reversal: Barlow Twins wins at closed-set scale, loses at open-vocabulary scale",
             fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig("closed_set_vs_openvocab_reversal.pdf", bbox_inches="tight")
fig.savefig("closed_set_vs_openvocab_reversal.png", dpi=200, bbox_inches="tight")
print("Saved closed_set_vs_openvocab_reversal.pdf and .png")
