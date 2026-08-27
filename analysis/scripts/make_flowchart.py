#!/usr/bin/env python3
"""SELMA mechanism flowchart -> ~/selma/v3_report/fig_flowchart.{png,pdf}

One figure carrying the whole argument: where the decision enters, where the V1
defect was, what V3 changed, and the measured numbers at each gate.

    python3 make_flowchart.py [REPORT_DIR]
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPORT = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1
                            else "~/selma/v3_report")
os.makedirs(REPORT, exist_ok=True)

INK, GREY = "#1a1a1a", "#8a8a8a"
BOX, CODE, LLMC = "#f4f4f4", "#e8eef5", "#c3d9ee"
BAD, GOOD, NEUT = "#f7dcd8", "#dcecdc", "#ececec"

fig, ax = plt.subplots(figsize=(11.0, 9.6))
ax.set_xlim(0, 15.4)
ax.set_ylim(0, 13.6)
ax.axis("off")

CX, W = 4.3, 5.6          # main column
SX = 9.9                  # skip boxes
NX = 13.0                 # annotation column


def box(x, y, w, h, text, fc=BOX, fs=9.5, weight="normal"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.12",
                                facecolor=fc, edgecolor=INK, lw=1.1, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=INK,
            zorder=3, weight=weight, linespacing=1.4)


def diamond(x, y, w, h, text, fs=8.6):
    ax.add_patch(plt.Polygon([(x, y + h / 2), (x + w / 2, y),
                              (x, y - h / 2), (x - w / 2, y)],
                             facecolor=NEUT, edgecolor=INK, lw=1.1, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=INK,
            zorder=3, linespacing=1.35)


def arrow(x1, y1, x2, y2, label="", col=INK, fs=8.2, dy=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=13, lw=1.1, color=col,
                                 zorder=1, shrinkA=1, shrinkB=1))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, label, fontsize=fs,
                color=col, ha="center", va="bottom")


def note(y, title, body, fc, h):
    ax.add_patch(FancyBboxPatch((NX - 2.1, y - h / 2), 4.2, h,
                                boxstyle="round,pad=0.04,rounding_size=0.10",
                                facecolor=fc, edgecolor="none", zorder=1))
    ax.text(NX, y + h / 2 - 0.30, title, ha="center", va="center",
            fontsize=8.6, weight="bold", color=INK, zorder=3)
    ax.text(NX, y - 0.14, body, ha="center", va="center", fontsize=8.0,
            color=INK, zorder=3, linespacing=1.55)


ax.text(CX, 13.20, "SELMA — where the model's answer enters the reward",
        ha="center", fontsize=13, weight="bold", color=INK)
ax.text(CX, 12.82, "measured on Foraging-10x10-3p-3f, six seeds, 2.5M steps each",
        ha="center", fontsize=8.6, color=GREY, style="italic")

Y = [12.05, 10.55, 8.95, 7.45, 5.85, 4.35, 2.80, 1.35, 0.35]

box(CX, Y[0], W, 0.80, "QMIX environment step\n(agents act, team reward observed)")
arrow(CX, Y[0] - 0.40, CX, Y[1] + 0.62)

diamond(CX, Y[1], 6.0, 1.24,
        "detector: is the team stuck?\n"
        "3 failed loads in 15 steps · team gate\ncooldown 5 · max 2 events per episode")
arrow(CX + 3.0, Y[1], SX - 1.35, Y[1], "no", col=GREY, dy=0.10)
box(SX, Y[1], 2.6, 0.72, "no guidance\nΦ = 0,  F = 0", fc=NEUT, fs=8.6)
arrow(CX, Y[1] - 0.62, CX, Y[2] + 0.42)

box(CX, Y[2], W, 0.84, "code enumerates every feasible plan\n"
                       "(agents → foods, coalition levels checked)", fc=CODE)
arrow(CX, Y[2] - 0.42, CX, Y[3] + 0.55)

diamond(CX, Y[3], 5.2, 1.10, "two or more plans offered?")
arrow(CX + 2.6, Y[3], SX - 1.35, Y[3], "no  (69 %)", col=GREY, dy=0.10)
box(SX, Y[3], 2.6, 0.84, "singleton:\nno query,\nno guidance", fc=NEUT, fs=8.6)
arrow(CX, Y[3] - 0.55, CX, Y[4] + 0.48)

box(CX, Y[4], W, 0.96, "LLM selects ONE plan_id",
    fc=LLMC, fs=11, weight="bold")
ax.text(CX, Y[4] - 0.26, "strict enum — invalid answers are unrepresentable",
        ha="center", fontsize=8.3, color="#24425e", zorder=3)
ax.text(SX, Y[4], "gpt-5.4-nano\n≈19,700 calls\n1 validator failure",
        fontsize=8.0, color=GREY, ha="center", va="center", linespacing=1.5)
arrow(CX, Y[4] - 0.48, CX, Y[5] + 0.44)

box(CX, Y[5], W, 0.88, "code resolves loading cells\n"
                       "min-cost matching, fixed for the lifetime", fc=CODE)
arrow(CX, Y[5] - 0.44, CX, Y[6] + 0.50)

box(CX, Y[6], W, 0.90,
    r"$\Phi \;=\; -\frac{1}{|R|}\sum_i "
    r"\frac{\|p_i - c_i\|_1 + \mathbb{1}[\mathrm{food\ remains}]}{d_{\max}}$",
    fc=GOOD, fs=11.5)
ax.text(CX, Y[6] - 0.72, "V3: Φ reads the ASSIGNED CELL of the selected plan",
        ha="center", fontsize=8.4, color="#2c6630", weight="bold", zorder=3)
arrow(CX, Y[6] - 0.86, CX, Y[7] + 0.42)

box(CX, Y[7], W, 0.84,
    r"$F_t=\beta\,(\gamma\,\Phi_{t+1}-\Phi_t)$" + "     added for TTL = 13 steps",
    fs=10)
arrow(CX, Y[7] - 0.42, CX, Y[8] + 0.30)

box(CX, Y[8], W, 0.60, "training reward  =  environment reward  +  $F_t$",
    fs=10, weight="bold")

note(10.85, "THE V1 DEFECT",
     "Φ measured distance to the FOOD,\n"
     "not to the assigned cell. Permuting\n"
     "the model's answer left Φ unchanged,\n"
     "so the choice never reached the reward.\n"
     "0.00 % of 20,574 decisions",
     BAD, 2.30)

note(7.30, "AFTER THE FIX",
     "93.27 % plan-conditional (V2)\n"
     "100 % of activations span >1 food (V3)\n"
     "journeys: median 4, p95 11, max 12\n"
     "V1 was capped at 2 units",
     GOOD, 2.00)

note(3.95, "REACH — the binding constraint",
     "guidance active on 2.09 % of transitions\n"
     "shaping mass = 4.9 % of env reward mass\n"
     "median shaped step = 0.21 % of a food\n"
     "reward; p99 = 23 % (boundary transients)",
     NEUT, 2.10)

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(REPORT, f"fig_flowchart.{ext}"), dpi=200,
                bbox_inches="tight", facecolor="white")
plt.close(fig)
print("wrote", os.path.join(REPORT, "fig_flowchart.png"), "and .pdf")
