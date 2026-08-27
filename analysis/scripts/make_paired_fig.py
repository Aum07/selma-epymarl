#!/usr/bin/env python3
"""Paired within-seed AUC differences, 10x10-3p-3f. Built from the per-seed AUC
values in Table II. Incomplete runs are omitted, not imputed."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

seeds = [1, 2, 3, 4, 5, 6]
base = [0.1093, 0.1481, 0.1214, 0.1361, 0.1301, 0.1156]
llm = [0.1204, 0.1222, 0.1393, 0.1271, 0.1367, 0.1000]
rnd = [0.1143, 0.1495, 0.1210, 0.1090, 0.1212, 0.1090]
heu = [0.0893, 0.1189, 0.1224, 0.0999, 0.1385, 0.1408]


def diff(vals):
    return [(v - b) if v is not None else np.nan for v, b in zip(vals, base)]


d_llm, d_rnd, d_heu = diff(llm), diff(rnd), diff(heu)

x = np.arange(len(seeds))
w = 0.26
fig, ax = plt.subplots(figsize=(7.2, 3.1))
ax.axhline(0, color="#1a1a1a", lw=1.0, zorder=3)

ax.bar(x - w, d_llm, w, label="llm", color="#c3d9ee",
       edgecolor="#1a1a1a", lw=0.7, zorder=2)
ax.bar(x, d_rnd, w, label="random", color="#ececec",
       edgecolor="#1a1a1a", lw=0.7, zorder=2)
ax.bar(x + w, d_heu, w, label="heuristic", color="#f7dcd8",
       edgecolor="#1a1a1a", lw=0.7, zorder=2)

ax.set_xticks(x)
ax.set_xticklabels([f"seed {s}" for s in seeds], fontsize=9)
ax.set_ylabel("$\\Delta$ normalised AUC vs baseline", fontsize=9)
ax.tick_params(axis="y", labelsize=8.5)
ax.legend(fontsize=8.5, frameon=False, loc="lower left", ncol=3)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"figures/fig_paired.{ext}", dpi=200, bbox_inches="tight",
                facecolor="white")
print("wrote figures/fig_paired.pdf")
