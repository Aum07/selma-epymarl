#!/usr/bin/env python3
"""Greedy-success curves for the 10x10 V3 experiment, in the same style as the
8x8 figure: mean over seeds per condition with the per-seed envelope shaded.

    python3 make_v3_curves.py OUTDIR REPORTDIR
"""
import csv
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1
                         else "~/selma/v3/analysis/out/v3_10x10")
REPORT = os.path.expanduser(sys.argv[2] if len(sys.argv) > 2
                            else "~/selma/v3_report")
os.makedirs(REPORT, exist_ok=True)

STEP, NPTS, TOL = 50_000, 50, 25_000
GRID = [STEP * k for k in range(NPTS)]
CONDS = ["baseline", "llm", "random", "heuristic"]
COLOUR = {"baseline": "#1a1a1a", "llm": "#2f6fb0",
          "random": "#8a8a8a", "heuristic": "#c0504d"}


def curve(d):
    f = os.path.join(d, "episodes.csv")
    if not os.path.exists(f):
        return {}
    buckets = {}
    with open(f) as fh:
        for r in csv.DictReader(fh):
            if str(r.get("test_mode", "")).strip().lower() not in ("1", "true"):
                continue
            try:
                t = int(r["t_env"])
                ok = float(r["team_return"]) > 0
            except (ValueError, KeyError):
                continue
            buckets.setdefault(t, []).append(ok)
    return {t: sum(v) / len(v) for t, v in buckets.items() if v}


def on_grid(c):
    ts = sorted(c)
    out = []
    for target in GRID:
        best, bd = None, TOL + 1
        for t in ts:
            d = abs(t - target)
            if d < bd:
                best, bd = t, d
        out.append(c[best] if best is not None and bd <= TOL else None)
    return out


fig, ax = plt.subplots(figsize=(7.0, 4.0))
for cond in CONDS:
    series = []
    for d in sorted(glob.glob(os.path.join(OUT, f"{cond}_seed*"))):
        g = on_grid(curve(d))
        if any(v is not None for v in g):
            series.append(g)
    if not series:
        continue
    xs, mean, lo, hi = [], [], [], []
    for i, t in enumerate(GRID):
        vals = [s[i] for s in series if s[i] is not None]
        if not vals:
            continue
        xs.append(t / 1e6)
        mean.append(sum(vals) / len(vals))
        lo.append(min(vals))
        hi.append(max(vals))
    c = COLOUR[cond]
    ax.fill_between(xs, lo, hi, color=c, alpha=0.13, linewidth=0)
    ax.plot(xs, mean, color=c, lw=1.6,
            label=f"{cond} (n={len(series)})")

ax.set_xlabel("environment steps (millions)", fontsize=10)
ax.set_ylabel("greedy evaluation success", fontsize=10)
ax.tick_params(labelsize=9)
ax.legend(fontsize=9, frameon=False, loc="upper left")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", color="#eeeeee", lw=0.8)
ax.set_axisbelow(True)
fig.tight_layout()
for ext in ("pdf", "png"):
    p = os.path.join(REPORT, f"fig_v3_curves.{ext}")
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote", p)
