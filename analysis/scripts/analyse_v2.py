#!/usr/bin/env python
"""SELMA V2: four-condition analysis under the plan-conditioned potential.

Applies the SAME frozen methodology as the Stage E locked analysis, so the two
sets of numbers are directly comparable:

  * greedy success at an evaluation point = fraction of that point's test
    episodes whose team_return > 0;
  * 18 frozen post-onset points, t_env 600_000 .. 1_450_000 step 50_000,
    matched to the nearest actual evaluation within +/- 25_000;
  * primary  = trapezoid AUC of success over those points / the 850_000 span,
    then averaged over seeds;
  * secondary = mean success over the final five frozen points.

Adds the mechanism diagnostics the V1 runs structurally could not produce: how
often the potential actually moved, how the shaping term was distributed, and
which plans each selector chose.

READ-ONLY with respect to the runs. Writes only to --outdir.

Usage:
    python analyse_v2.py                       # text report
    python analyse_v2.py --outdir analysis/out/v2_report --figures
"""

import argparse
import csv
import json
import math
import os
import statistics
import sys
from collections import Counter

ROOT_DEFAULT = os.path.expanduser("~/selma/epymarl/analysis/out/v2_plan_potential")
CONDITIONS = ("baseline", "heuristic", "random", "llm")
LABEL = {"baseline": "Baseline", "heuristic": "Heuristic",
         "random": "Random", "llm": "SELMA (LLM)"}
SEEDS = (1, 2, 3)
TARGETS = [600000 + 50000 * i for i in range(18)]
TOL = 25000
PRICE = {"in": 0.20, "cached": 0.02, "out": 1.25}   # USD / 1M tokens, 2026-08-17


# ----------------------------------------------------------------- helpers

def num(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def eval_points(episodes_csv):
    """Greedy (test_mode=1) evaluation points, streamed."""
    buckets = {}
    with open(episodes_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            if int(num(row.get("test_mode"))) != 1:
                continue
            t = int(num(row.get("t_env")))
            r = num(row.get("team_return"))
            hit, n = buckets.get(t, (0, 0))
            buckets[t] = (hit + (1 if r > 0 else 0), n + 1)
    return [{"t_env": t, "n": n, "success": hit / n}
            for t, (hit, n) in sorted(buckets.items())]


def match_frozen(points):
    sel, missing = [], []
    for target in TARGETS:
        cands = [p for p in points if abs(p["t_env"] - target) <= TOL]
        if not cands:
            missing.append(target)
            continue
        sel.append({"target": target,
                    **min(cands, key=lambda p: abs(p["t_env"] - target))})
    return sel, missing


def auc(sel):
    if len(sel) < 2:
        return None
    total = 0.0
    for a, b in zip(sel, sel[1:]):
        total += (b["target"] - a["target"]) * (a["success"] + b["success"]) / 2.0
    span = sel[-1]["target"] - sel[0]["target"]
    return total / span if span else None


def final5(sel):
    return sum(p["success"] for p in sel[-5:]) / 5.0 if len(sel) >= 5 else None


def guidance_diagnostics(path):
    """Stream the per-transition PBRS log. Returns None when absent."""
    if not os.path.exists(path):
        return None
    d = dict(rows=0, active=0, zero_delta=0, zero_pbrs=0, greedy_eval_rows=0,
             shaping_abs=0.0, shaping_max=0.0, raw_reward=0.0,
             shaping_sum=0.0, terminal_nonzero_phi=0, plans=Counter())
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            d["rows"] += 1
            if int(num(row.get("greedy_eval"))) == 1:
                d["greedy_eval_rows"] += 1
            active = int(num(row.get("guidance_active")))
            term = num(row.get("pbrs_term_unscaled"))
            shap = num(row.get("shaping_contribution"))
            d["raw_reward"] += num(row.get("raw_team_reward"))
            d["shaping_sum"] += shap
            d["shaping_abs"] += abs(shap)
            d["shaping_max"] = max(d["shaping_max"], abs(shap))
            if active:
                d["active"] += 1
                # The V1 audit statistic is "the POTENTIAL did not change",
                # i.e. Phi(s_{t+1}) == Phi(s_t). It is NOT "the PBRS term was
                # zero": with gamma = 0.99, an unchanged potential still gives
                # gamma*Phi - Phi = (gamma-1)*Phi != 0 whenever Phi != 0, so
                # the PBRS term is almost never exactly zero and counting it
                # would report 0% every time regardless of the mechanism.
                if abs(num(row.get("potential_after_real"))
                       - num(row.get("phi_before"))) < 1e-12:
                    d["zero_delta"] += 1
                if abs(term) < 1e-12:
                    d["zero_pbrs"] += 1
                pid = row.get("selected_plan_id") or ""
                if pid:
                    d["plans"][pid] += 1
            if int(num(row.get("stored_terminated"))) == 1 and \
               abs(num(row.get("phi_after_effective"))) > 1e-12:
                d["terminal_nonzero_phi"] += 1
    return d


def decision_diagnostics(path):
    if not os.path.exists(path):
        return None
    d = dict(rows=0, invalid=0, greedy=0, live=0, tok_in=0.0, tok_cached=0.0,
             tok_out=0.0, plan_counts=Counter(), selected=Counter(),
             reasons=Counter(), models=set())
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            d["rows"] += 1
            if str(row.get("validator_valid")).strip() in ("0", "False", "false"):
                d["invalid"] += 1
                if row.get("validator_reason"):
                    d["reasons"][row["validator_reason"]] += 1
            if int(num(row.get("greedy_eval"))) == 1:
                d["greedy"] += 1
            d["live"] += int(num(row.get("live_attempts")))
            d["tok_in"] += num(row.get("tokens_input"))
            d["tok_cached"] += num(row.get("tokens_cached_input"))
            d["tok_out"] += num(row.get("tokens_output"))
            c = row.get("candidate_plan_count")
            if c:
                d["plan_counts"][int(num(c))] += 1
            if row.get("selected_plan_id"):
                d["selected"][row["selected_plan_id"]] += 1
            if row.get("returned_model"):
                d["models"].add(row["returned_model"])
    d["cost"] = (d["tok_in"] / 1e6 * PRICE["in"]
                 + d["tok_cached"] / 1e6 * PRICE["cached"]
                 + d["tok_out"] / 1e6 * PRICE["out"])
    d["models"] = sorted(d["models"])
    return d


# ----------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT_DEFAULT)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--figures", action="store_true")
    a = ap.parse_args()

    runs, incomplete = {}, []
    for cond in CONDITIONS:
        for seed in SEEDS:
            d = os.path.join(a.root, f"{cond}_seed{seed}")
            ep = os.path.join(d, "episodes.csv")
            if not os.path.exists(ep):
                incomplete.append(f"{cond}_seed{seed} (no episodes.csv)")
                continue
            if not os.path.exists(os.path.join(d, "DONE")):
                incomplete.append(f"{cond}_seed{seed} (no DONE marker)")
            pts = eval_points(ep)
            sel, missing = match_frozen(pts)
            runs[(cond, seed)] = {
                "dir": d, "points": len(pts), "matched": len(sel),
                "complete": len(sel) == len(TARGETS)
                            and os.path.exists(os.path.join(d, "DONE")),
                "missing": missing, "auc": auc(sel), "final5": final5(sel),
                "curve": [(p["target"], p["success"]) for p in sel],
                "guidance": guidance_diagnostics(os.path.join(d, "guidance.csv")),
                "decisions": decision_diagnostics(os.path.join(d, "decisions.csv")),
            }

    if incomplete:
        print("!! INCOMPLETE OR MISSING:")
        for x in incomplete:
            print("   ", x)
        print()

    # ---- performance ----------------------------------------------------
    per_cond = {}
    for cond in CONDITIONS:
        ok = [s for s in SEEDS if (cond, s) in runs
              and runs[(cond, s)]["complete"] and runs[(cond, s)]["auc"] is not None]
        aucs = [runs[(cond, s)]["auc"] for s in ok]
        f5s = [runs[(cond, s)]["final5"] for s in ok
               if runs[(cond, s)]["final5"] is not None]
        if not aucs:
            continue
        per_cond[cond] = {
            "n": len(aucs),
            "per_seed": {s: runs[(cond, s)]["auc"] for s in SEEDS if (cond, s) in runs},
            "mean_auc": sum(aucs) / len(aucs),
            "sd_auc": statistics.stdev(aucs) if len(aucs) > 1 else float("nan"),
            "mean_final5": sum(f5s) / len(f5s) if f5s else None,
            "spread": max(aucs) - min(aucs),
        }

    print("=" * 74)
    print("SELMA V2 -- four-condition comparison (frozen Stage E methodology)")
    print("=" * 74)
    print("\n(*) = run not finished: fewer than 18 frozen points matched, so its"
          "\n      AUC spans a shorter window and is EXCLUDED from the mean.\n")
    print(f"{'condition':<14}" + "".join(f"{'seed '+str(s):>11}" for s in SEEDS)
          + f"{'n':>3}{'mean':>10}{'sd':>9}{'final5':>9}{'spread':>9}")
    for cond in CONDITIONS:
        if cond not in per_cond:
            continue
        v = per_cond[cond]
        cells = ""
        for s in SEEDS:
            r = runs.get((cond, s))
            if not r or r["auc"] is None:
                cells += f"{'--':>11}"
            else:
                cells += f"{r['auc']:>10.5f}" + ("*" if not r["complete"] else " ")
        sd = v["sd_auc"]
        print(f"{LABEL[cond]:<14}{cells}{v['n']:>3}{v['mean_auc']:>10.5f}"
              f"{(f'{sd:.5f}' if not math.isnan(sd) else '--'):>9}"
              f"{v['mean_final5']:>9.4f}{v['spread']:>9.5f}")

    # ---- paired contrasts ----------------------------------------------
    if "baseline" in per_cond:
        print("\n-- paired per-seed change vs baseline (same seed) --")
        for cond in ("heuristic", "random", "llm"):
            if cond not in per_cond:
                continue
            deltas = []
            for s in SEEDS:
                a_, b_ = runs.get((cond, s)), runs.get(("baseline", s))
                # BOTH sides must be finished: an unfinished run's AUC spans a
                # shorter window, so differencing it against a complete one
                # compares two different quantities.
                if (a_ and b_ and a_["complete"] and b_["complete"]
                        and a_["auc"] is not None and b_["auc"] is not None):
                    deltas.append(a_["auc"] - b_["auc"])
            if deltas:
                sign = "all +" if all(d > 0 for d in deltas) else \
                       "all -" if all(d < 0 for d in deltas) else "MIXED"
                print(f"  {LABEL[cond]:<14} n={len(deltas)}  " +
                      "  ".join(f"{d:+.5f}" for d in deltas) +
                      f"   mean {sum(deltas)/len(deltas):+.5f}   [{sign}]")

    print("\n-- head-to-head (the contrasts this design supports) --")
    for a_, b_, q in (("llm", "heuristic", "does model judgement beat a rule?"),
                      ("llm", "random", "does selection content carry signal?"),
                      ("heuristic", "random", "does plan quality matter at all?")):
        if a_ in per_cond and b_ in per_cond:
            d = per_cond[a_]["mean_auc"] - per_cond[b_]["mean_auc"]
            print(f"  {LABEL[a_]:<12} - {LABEL[b_]:<12} = {d:+.5f}   {q}")

    # ---- power ----------------------------------------------------------
    if per_cond:
        spreads = [v["spread"] for v in per_cond.values()]
        means = [v["mean_auc"] for v in per_cond.values()]
        rng = max(means) - min(means)
        print(f"\n-- power --")
        print(f"  max within-condition seed spread : {max(spreads):.5f}")
        print(f"  between-condition range of means : {rng:.5f}")
        print(f"  spread exceeds range             : {max(spreads) > rng}"
              f"   {'(noise-dominated: no ordering claim)' if max(spreads) > rng else ''}")

    # ---- mechanism ------------------------------------------------------
    print("\n-- mechanism diagnostics (the V1 runs could not produce these) --")
    print(f"{'condition':<14}{'seed':>5}{'decisions':>11}{'invalid':>9}"
          f"{'greedy':>8}{'dPhi=0':>9}{'|shaping|':>11}{'cost':>9}")
    for cond in CONDITIONS:
        for s in SEEDS:
            r = runs.get((cond, s))
            if not r:
                continue
            g, dd = r["guidance"], r["decisions"]
            zero = (f"{g['zero_delta']/g['active']:.1%}"
                    if g and g["active"] else "--")
            mag = f"{g['shaping_abs']:.2f}" if g else "--"
            cost = f"${dd['cost']:.4f}" if dd else "--"
            print(f"{LABEL[cond]:<14}{s:>5}"
                  f"{(dd['rows'] if dd else 0):>11}"
                  f"{(dd['invalid'] if dd else 0):>9}"
                  f"{(dd['greedy'] if dd else 0):>8}{zero:>9}{mag:>11}{cost:>9}")

    print("\n-- selected plan distribution (order bias) --")
    for cond in ("heuristic", "random", "llm"):
        agg = Counter()
        for s in SEEDS:
            r = runs.get((cond, s))
            if r and r["decisions"]:
                agg.update(r["decisions"]["selected"])
        tot = sum(agg.values())
        if tot:
            top = ", ".join(f"{k}={v/tot:.1%}" for k, v in sorted(agg.items())[:6])
            print(f"  {LABEL[cond]:<14} n={tot:<7} {top}")
    print("  uniform reference would be 16.7% per plan over 6 candidates")

    total_cost = sum(r["decisions"]["cost"] for r in runs.values()
                     if r["decisions"])
    print(f"\n-- total provider cost across all runs: ${total_cost:.4f} --")

    # ---- outputs --------------------------------------------------------
    if a.outdir:
        os.makedirs(a.outdir, exist_ok=True)
        with open(os.path.join(a.outdir, "v2_results.json"), "w") as fh:
            json.dump({
                "per_condition": per_cond,
                "runs": {f"{c}_seed{s}": {k: v for k, v in r.items()
                                          if k not in ("guidance", "decisions")}
                         for (c, s), r in runs.items()},
                "total_cost_usd": total_cost,
                "incomplete": incomplete,
            }, fh, indent=2, default=str)

        # LaTeX: main results table
        with open(os.path.join(a.outdir, "table_v2.tex"), "w") as fh:
            fh.write("% auto-generated by analyse_v2.py -- do not hand-edit\n")
            for cond in CONDITIONS:
                if cond not in per_cond:
                    continue
                v = per_cond[cond]
                cells = " & ".join(
                    (f"{v['per_seed'][s]:.5f}" if v["per_seed"].get(s) is not None
                     else "--") for s in SEEDS)
                fh.write(f"{LABEL[cond]} & {cells} & "
                         f"{v['mean_auc']:.5f} & {v['mean_final5']:.4f} \\\\\n")

        if a.figures:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(6.4, 3.6))
                for cond in CONDITIONS:
                    curves = [r["curve"] for (c, s), r in runs.items()
                              if c == cond and r["complete"] and r["curve"]]
                    if not curves:
                        continue
                    xs = [p[0] for p in curves[0]]
                    ys = [sum(c[i][1] for c in curves) / len(curves)
                          for i in range(len(xs))]
                    lo = [min(c[i][1] for c in curves) for i in range(len(xs))]
                    hi = [max(c[i][1] for c in curves) for i in range(len(xs))]
                    ax.plot([x / 1e6 for x in xs], ys, label=LABEL[cond], lw=1.8)
                    ax.fill_between([x / 1e6 for x in xs], lo, hi, alpha=0.15)
                ax.set_xlabel("environment steps (millions)")
                ax.set_ylabel("greedy success rate")
                ax.legend(frameon=False, fontsize=8)
                ax.grid(alpha=0.25, lw=0.5)
                fig.tight_layout()
                fig.savefig(os.path.join(a.outdir, "fig_v2_curves.pdf"))
                fig.savefig(os.path.join(a.outdir, "fig_v2_curves.png"), dpi=180)
                print(f"\nfigures written to {a.outdir}")
            except ImportError:
                print("\n(matplotlib unavailable -- skipped figures)")

        print(f"report written to {a.outdir}")

    return 0 if not incomplete else 1


if __name__ == "__main__":
    sys.exit(main())
