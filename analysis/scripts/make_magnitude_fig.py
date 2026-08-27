#!/usr/bin/env python3
"""Distribution of |F_t| over shaped transitions, split boundary vs interior.

Streams the six llm guidance logs (do not load them into pandas). Writes
fig_magnitude.{pdf,png} into the report directory.

    python3 make_magnitude_fig.py [REPORT_DIR] [RUNS_GLOB]

Defaults:
    REPORT_DIR = paper2/figures
    RUNS_GLOB  = ~/selma/v3/analysis/out/v3_10x10/llm_seed*/guidance.csv

Run with --probe first to print the resolved column names and a few raw rows
without computing anything. Confirm the parse before trusting the figure.
"""
import csv
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

csv.field_size_limit(1 << 24)

PROBE = "--probe" in sys.argv
argv = [a for a in sys.argv[1:] if not a.startswith("--")]
REPORT = os.path.expanduser(argv[0] if len(argv) > 0 else "figures")
PATTERN = os.path.expanduser(
    argv[1] if len(argv) > 1
    else "~/selma/v3/analysis/out/v3_10x10/llm_seed*/guidance.csv")

# Candidate column names, most likely first. The script picks the first that
# exists and prints its choice; it never guesses silently.
CAND_F = ["shaping_contribution", "shaping_reward", "shaping_term",
          "f_shaping", "f_t", "F_t", "shaping", "shaped_delta"]
CAND_PHI_BEFORE = ["phi_before_effective", "phi_before", "phi_t"]
CAND_PHI_AFTER = ["phi_after_effective", "phi_after", "phi_tp1"]
CAND_EVAL = ["greedy_eval", "is_eval", "eval"]

# Environment reward reference points, from Section V-H.
R_MIN = 0.076923077
R_MED = 0.200


def pick(header, candidates, what):
    for c in candidates:
        if c in header:
            return c
    raise SystemExit(
        f"Could not find a column for {what}.\nTried: {candidates}\n"
        f"Header is:\n  {header}\n"
        "Edit the CAND_* lists at the top of this script and re-run.")


files = sorted(glob.glob(PATTERN))
if not files:
    raise SystemExit(f"No guidance logs matched:\n  {PATTERN}")
print(f"{len(files)} run(s):")
for f in files:
    print("   ", f)

with open(files[0], newline="") as fh:
    header = next(csv.reader(fh))

col_f = pick(header, CAND_F, "the shaping term F_t")
col_pb = pick(header, CAND_PHI_BEFORE, "the potential at t")
col_pa = pick(header, CAND_PHI_AFTER, "the potential at t+1")
col_ev = next((c for c in CAND_EVAL if c in header), None)

print("\nresolved columns:")
print("   F_t        ->", col_f)
print("   phi(t)     ->", col_pb)
print("   phi(t+1)   ->", col_pa)
print("   greedy_eval->", col_ev or "(absent; no eval rows to exclude)")

if PROBE:
    print("\nfirst 5 rows of those columns:")
    with open(files[0], newline="") as fh:
        r = csv.DictReader(fh)
        for i, row in enumerate(r):
            if i >= 5:
                break
            print("   ", {k: row[k] for k in
                          [c for c in (col_f, col_pb, col_pa, col_ev) if c]})
    raise SystemExit("\nprobe only; nothing computed.")

boundary, interior = [], []
n_rows = n_shaped = 0

for path in files:
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            n_rows += 1
            if col_ev and row[col_ev] not in ("", "0", "False", "false"):
                continue
            try:
                f = float(row[col_f])
            except (ValueError, TypeError):
                continue
            if f == 0.0:
                continue
            n_shaped += 1
            try:
                pb = float(row[col_pb] or 0.0)
                pa = float(row[col_pa] or 0.0)
            except (ValueError, TypeError):
                continue
            # A boundary transition has the potential defined at exactly one
            # endpoint: activation (phi_t = 0) or expiry (phi_{t+1} = 0).
            if (pb == 0.0) != (pa == 0.0):
                boundary.append(abs(f))
            else:
                interior.append(abs(f))
    print(f"  streamed {os.path.basename(os.path.dirname(path))}: "
          f"{n_shaped:,} shaped so far")

boundary = np.asarray(boundary)
interior = np.asarray(interior)
allv = np.concatenate([boundary, interior]) if n_shaped else np.array([])

print(f"\nrows streamed      {n_rows:,}")
print(f"non-zero F_t       {n_shaped:,}  ({100*n_shaped/max(n_rows,1):.2f} %)")
print(f"  interior         {interior.size:,}")
print(f"  boundary         {boundary.size:,}")
if allv.size:
    print(f"median |F|         {np.median(allv):.9f}"
          f"   ({100*np.median(allv)/R_MIN:.2f} % of the smallest reward)")
    print(f"p99    |F|         {np.percentile(allv,99):.9f}"
          f"   ({100*np.percentile(allv,99)/R_MIN:.2f} %)")
    print(f"max    |F|         {allv.max():.9f}")
    if interior.size:
        print(f"median interior    {np.median(interior):.9f}")
    if boundary.size:
        print(f"median boundary    {np.median(boundary):.9f}")

if not allv.size:
    raise SystemExit("No shaped transitions found; nothing to plot.")

lo = max(allv[allv > 0].min(), 1e-7)
bins = np.logspace(np.log10(lo), np.log10(allv.max() * 1.05), 70)

INK, GREY = "#1a1a1a", "#8a8a8a"
C_INT, C_BND = "#4878a8", "#c05a44"

fig, ax = plt.subplots(figsize=(3.5, 2.5))
ax.hist(interior, bins=bins, color=C_INT, alpha=0.85, label="interior", lw=0)
ax.hist(boundary, bins=bins, color=C_BND, alpha=0.75,
        label="boundary (activation, expiry)", lw=0)
ax.set_xscale("log")
ax.set_yscale("log")

ymax = ax.get_ylim()[1]
for x, lab, yf in ((R_MIN, "smallest env.\nreward", 0.85),
                   (R_MED, "median env.\nreward", 0.06)):
    ax.axvline(x, color=INK, ls="--", lw=0.9)
    ax.text(x * 0.92, ymax * yf, lab, fontsize=6, color=INK,
            va="top", ha="right", linespacing=1.3)

ax.set_xlabel(r"$|F_t|$  (log scale)", fontsize=8)
ax.set_ylabel("transitions", fontsize=8)
ax.tick_params(labelsize=7)
ax.legend(fontsize=6.5, frameon=False, loc="upper left")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

os.makedirs(REPORT, exist_ok=True)
fig.tight_layout()
for ext in ("pdf", "png"):
    out = os.path.join(REPORT, f"fig_magnitude.{ext}")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote", out)
plt.close(fig)
