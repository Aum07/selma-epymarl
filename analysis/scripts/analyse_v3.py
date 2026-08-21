#!/usr/bin/env python3
"""SELMA V3 10x10 -- learning-curve analysis.

Applies the V3.3-spec AUC rule to 2.5M runs:
  * evaluation every 50,000 steps -> 50 points, indices 0..49
  * nominal grid t_k = 50,000*k, matched within +/-25,000
  * trapezoidal integration on the NOMINAL grid so every seed is weighted
    identically regardless of its own drift
  * normalised by t_49 - t_0 = 2,450,000  (NOT by 2,500,000 -- there is no
    evaluation point at t_max, exactly as measured on the 10M runs)

Incomplete runs are marked '*' and excluded from every mean. This is the V2
lesson: partial runs silently polluted the means and every head-to-head sign
flipped once they were filtered.

No significance claims. n=4 paired seeds against a measured within-condition
spread that exceeded the between-condition range by 3.8x on 8x8. Direction and
full per-seed disclosure only.

    python3 analyse_v3.py [OUTDIR] [--tmax 2500000]
"""
import csv
import glob
import os
import statistics
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") \
    else os.path.expanduser("~/selma/v3/analysis/out/v3_10x10")
TMAX = 2_500_000
for i, a in enumerate(sys.argv):
    if a == "--tmax":
        TMAX = int(sys.argv[i + 1])

STEP = 50_000
NPTS = TMAX // STEP                      # 50 for 2.5M
TOL = 25_000
GRID = [STEP * k for k in range(NPTS)]   # 0, 50k, ..., 2.45M
SPAN = GRID[-1] - GRID[0]                # 2,450,000
CONDS = ["baseline", "heuristic", "random", "llm"]


def curve(d):
    """-> {t_env: greedy success} from a run directory."""
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
    """Match to the nominal grid; return (values, n_matched)."""
    ts = sorted(c)
    vals = []
    for target in GRID:
        best, bd = None, TOL + 1
        for t in ts:
            d = abs(t - target)
            if d < bd:
                best, bd = t, d
        vals.append(c[best] if best is not None and bd <= TOL else None)
    return vals, sum(v is not None for v in vals)


def auc(vals):
    """Trapezoid on the uniform nominal grid, normalised by SPAN."""
    n = len(vals)
    tot = 0.5 * vals[0] + sum(vals[1:n - 1]) + 0.5 * vals[-1]
    return tot / (n - 1)


runs = {}
for d in sorted(glob.glob(os.path.join(OUT, "*_seed*"))):
    name = os.path.basename(d)
    cond, _, seed = name.rpartition("_seed")
    if cond not in CONDS:
        continue
    c = curve(d)
    vals, n = on_grid(c)
    complete = (n == NPTS) and os.path.exists(os.path.join(d, "DONE"))
    runs[(cond, int(seed))] = {
        "vals": vals, "n": n, "complete": complete,
        "auc": auc(vals) if n == NPTS else None,
        "final5": statistics.mean([v for v in vals[-5:] if v is not None])
        if all(v is not None for v in vals[-5:]) else None,
    }

seeds = sorted({s for _, s in runs})
print(f"outdir : {OUT}")
print(f"grid   : {NPTS} points, 0..{GRID[-1]:,}, normalised by {SPAN:,}")
print(f"runs   : {len(runs)}   seeds: {seeds}")
print("         '*' = incomplete, excluded from all means\n")

print(f"{'condition':<11}" + "".join(f"{'seed '+str(s):>12}" for s in seeds)
      + f"{'mean':>12}{'n':>4}")
print("-" * (11 + 12 * len(seeds) + 16))
means = {}
for cond in CONDS:
    cells, good = [], []
    for s in seeds:
        r = runs.get((cond, s))
        if r is None:
            cells.append("-")
        elif r["auc"] is None or not r["complete"]:
            cells.append(f"{(r['auc'] or 0):.4f}*" if r["auc"] else f"{r['n']}/{NPTS}*")
        else:
            cells.append(f"{r['auc']:.4f}")
            good.append(r["auc"])
    means[cond] = statistics.mean(good) if good else None
    m = f"{means[cond]:.5f}" if means[cond] is not None else "-"
    print(f"{cond:<11}" + "".join(f"{c:>12}" for c in cells)
          + f"{m:>12}{len(good):>4}")

print("\n--- paired differences vs same-seed baseline (complete pairs only) ---")
for cond in CONDS[1:]:
    diffs = []
    for s in seeds:
        a, b = runs.get((cond, s)), runs.get(("baseline", s))
        if a and b and a["complete"] and b["complete"]:
            diffs.append((s, a["auc"] - b["auc"]))
    if not diffs:
        print(f"{cond:<11} no complete pairs")
        continue
    vals = [d for _, d in diffs]
    signs = {"+" if v > 0 else "-" for v in vals}
    verdict = "MIXED IN SIGN" if len(signs) > 1 else \
              ("all positive" if "+" in signs else "all negative")
    per = "  ".join(f"s{s}:{v:+.4f}" for s, v in diffs)
    print(f"{cond:<11} mean {statistics.mean(vals):+.5f}   {verdict}")
    print(f"{'':<11} {per}")
    if len(vals) > 1:
        print(f"{'':<11} sd {statistics.stdev(vals):.5f}")

print("\n--- head-to-head, complete pairs only ---")
for a, b in (("llm", "heuristic"), ("llm", "random"), ("heuristic", "random")):
    diffs = [(s, runs[(a, s)]["auc"] - runs[(b, s)]["auc"]) for s in seeds
             if runs.get((a, s)) and runs.get((b, s))
             and runs[(a, s)]["complete"] and runs[(b, s)]["complete"]]
    if diffs:
        vals = [d for _, d in diffs]
        wins = sum(v > 0 for v in vals)
        print(f"{a} - {b:<10} mean {statistics.mean(vals):+.5f}   "
              f"{a} ahead on {wins}/{len(vals)} seeds")

print("\n--- spread check (the V2 lesson) ---")
allv = [r["auc"] for r in runs.values() if r["auc"] is not None and r["complete"]]
if allv and all(m is not None for m in means.values()):
    within = max(
        max(r["auc"] for (c, s), r in runs.items() if c == cond and r["complete"] and r["auc"])
        - min(r["auc"] for (c, s), r in runs.items() if c == cond and r["complete"] and r["auc"])
        for cond in CONDS
        if any(r["complete"] and r["auc"] for (c, s), r in runs.items() if c == cond))
    between = max(means.values()) - min(means.values())
    print(f"max within-condition spread : {within:.5f}")
    print(f"between-condition range     : {between:.5f}")
    if between > 0:
        print(f"ratio                       : {within/between:.2f}x")
        print("(on 8x8 this was 3.8x -- seed identity dominated condition)")

print("\n--- final-5 mean (secondary) ---")
for cond in CONDS:
    v = [runs[(cond, s)]["final5"] for s in seeds
         if runs.get((cond, s)) and runs[(cond, s)]["complete"]
         and runs[(cond, s)]["final5"] is not None]
    if v:
        print(f"{cond:<11} {statistics.mean(v):.4f}   per-seed "
              + " ".join(f"{x:.3f}" for x in v))

print("\nNO SIGNIFICANCE CLAIMS. n=4 paired seeds. Report direction and per-seed values.")
