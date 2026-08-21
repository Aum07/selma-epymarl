#!/usr/bin/env python3
"""SELMA V3 -- one command that produces every number and figure.

Writes everything into ONE folder so nothing has to be reconstructed later:

    ~/selma/v3_report/
        RESULTS.md              human-readable summary of every finding
        results.json            same numbers, machine-readable
        analyse_v3.txt          raw output of the learning-curve analysis
        mechanism_v3.txt        raw output of the mechanism diagnostics
        fig_flowchart.png/.pdf  mechanism flowchart with measured values
        fig_curves.png/.pdf     greedy-success curves, all conditions
        fig_paired.png/.pdf     per-seed paired differences vs baseline
        table_v3.tex            LaTeX table ready to paste into the paper
        per_seed.csv            every seed's AUC and final-5 value
        provider.txt            calls, cost, coverage, validator failures

Usage:
    python3 make_report.py                       # defaults below
    python3 make_report.py OUTDIR TMAX DMAX REPORTDIR
        e.g. make_report.py ~/selma/v3/analysis/out/v3_8x8 1500000 15 ~/out8x8
"""
import csv
import glob
import json
import os
import statistics
import subprocess
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/selma/v3/analysis/out/v3_10x10")
TMAX = int(sys.argv[2]) if len(sys.argv) > 2 else 2_500_000
DMAX = float(sys.argv[3]) if len(sys.argv) > 3 else 19.0
REPORT = os.path.expanduser(
    sys.argv[4] if len(sys.argv) > 4
    else os.environ.get("SELMA_REPORT_DIR", "~/selma/v3_report"))
os.makedirs(REPORT, exist_ok=True)

STEP = 50_000
NPTS = TMAX // STEP
TOL = 25_000
GRID = [STEP * k for k in range(NPTS)]
CONDS = ["baseline", "heuristic", "random", "llm"]


# ---------------------------------------------------------------- curves ----
def curve(d):
    f = os.path.join(d, "episodes.csv")
    if not os.path.exists(f):
        return {}
    b = {}
    with open(f) as fh:
        for r in csv.DictReader(fh):
            if str(r.get("test_mode", "")).strip().lower() not in ("1", "true"):
                continue
            try:
                b.setdefault(int(r["t_env"]), []).append(
                    float(r["team_return"]) > 0)
            except (ValueError, KeyError):
                pass
    return {t: sum(v) / len(v) for t, v in b.items() if v}


def on_grid(c):
    ts = sorted(c)
    vals = []
    for target in GRID:
        best, bd = None, TOL + 1
        for t in ts:
            dd = abs(t - target)
            if dd < bd:
                best, bd = t, dd
        vals.append(c[best] if best is not None and bd <= TOL else None)
    return vals, sum(v is not None for v in vals)


def auc(v):
    n = len(v)
    return (0.5 * v[0] + sum(v[1:n - 1]) + 0.5 * v[-1]) / (n - 1)


runs = {}
for d in sorted(glob.glob(os.path.join(OUT, "*_seed*"))):
    name = os.path.basename(d)
    cond, _, seed = name.rpartition("_seed")
    if cond not in CONDS:
        continue
    vals, n = on_grid(curve(d))
    done = os.path.exists(os.path.join(d, "DONE"))
    # longest leading run of matched points -> usable prefix for partial runs
    pref = 0
    for v in vals:
        if v is None:
            break
        pref += 1
    runs[(cond, int(seed))] = {
        "vals": vals, "matched": n, "prefix": pref,
        "complete": (n == NPTS and done),
        "auc": auc(vals) if n == NPTS else None,
        "prefix_auc": auc(vals[:pref]) if pref >= 5 else None,
        "final5": statistics.mean(vals[-5:])
        if all(v is not None for v in vals[-5:]) else None,
    }

seeds = sorted({s for _, s in runs})
res = {"outdir": OUT, "t_max": TMAX, "d_max": DMAX, "grid_points": NPTS,
       "normaliser": GRID[-1] - GRID[0], "seeds": seeds,
       "conditions": {}, "paired": {}, "mechanism": {}, "provider": {}}

for cond in CONDS:
    ok = [(s, runs[(cond, s)]["auc"]) for s in seeds
          if runs.get((cond, s)) and runs[(cond, s)]["complete"]]
    f5 = [runs[(cond, s)]["final5"] for s in seeds
          if runs.get((cond, s)) and runs[(cond, s)]["complete"]
          and runs[(cond, s)]["final5"] is not None]
    res["conditions"][cond] = {
        "per_seed_auc": {str(s): v for s, v in ok},
        "mean_auc": statistics.mean([v for _, v in ok]) if ok else None,
        "n_complete": len(ok),
        "mean_final5": statistics.mean(f5) if f5 else None,
        "incomplete": [s for s in seeds if runs.get((cond, s))
                       and not runs[(cond, s)]["complete"]],
    }

for cond in CONDS[1:]:
    diffs = [(s, runs[(cond, s)]["auc"] - runs[("baseline", s)]["auc"])
             for s in seeds
             if runs.get((cond, s)) and runs.get(("baseline", s))
             and runs[(cond, s)]["complete"] and runs[("baseline", s)]["complete"]]
    if not diffs:
        continue
    v = [x for _, x in diffs]
    res["paired"][f"{cond}_minus_baseline"] = {
        "per_seed": {str(s): x for s, x in diffs},
        "mean": statistics.mean(v),
        "sd": statistics.stdev(v) if len(v) > 1 else None,
        "n": len(v),
        "positive": sum(x > 0 for x in v),
        "mixed_sign": (max(v) > 0 > min(v)),
    }


# --------------------------------------------- common-prefix comparison ----
# Unfinished runs (random / heuristic still training) are still usable: truncate
# EVERY run to the longest horizon they all reached, then compare on that.
# This is a fair comparison -- same window for every condition and seed -- and it
# is reported separately from the full-window result, never mixed with it.
pref_runs = {k: v for k, v in runs.items() if v["prefix"] >= 5}
if pref_runs:
    K = min(v["prefix"] for v in pref_runs.values())
    res["common_prefix"] = {
        "points": K,
        "horizon_steps": GRID[K - 1],
        "note": "all runs truncated to this window; includes runs still training",
        "conditions": {}, "paired": {},
    }
    pa = {}
    for (c, sd), v in pref_runs.items():
        pa[(c, sd)] = auc(v["vals"][:K])
    for cond in CONDS:
        vs = [(sd, pa[(cond, sd)]) for sd in seeds if (cond, sd) in pa]
        if vs:
            res["common_prefix"]["conditions"][cond] = {
                "per_seed": {str(sd): x for sd, x in vs},
                "mean": statistics.mean([x for _, x in vs]),
                "n": len(vs),
            }
    for cond in CONDS[1:]:
        dd = [(sd, pa[(cond, sd)] - pa[("baseline", sd)]) for sd in seeds
              if (cond, sd) in pa and ("baseline", sd) in pa]
        if dd:
            vv = [x for _, x in dd]
            res["common_prefix"]["paired"][f"{cond}_minus_baseline"] = {
                "per_seed": {str(sd): x for sd, x in dd},
                "mean": statistics.mean(vv),
                "sd": statistics.stdev(vv) if len(vv) > 1 else None,
                "n": len(vv), "positive": sum(x > 0 for x in vv),
                "mixed_sign": (max(vv) > 0 > min(vv)),
            }

# ------------------------------------------------------------ mechanism ----
def kv(s):
    o = {}
    for part in str(s or "").split(";"):
        if "=" in part:
            k, _, val = part.partition("=")
            o[k.strip()] = val.strip()
    return o


def lst(s):
    return [x.strip() for x in str(s or "").split(";") if x.strip()]


mech = {"activations": 0, "span_multi": 0, "journeys": [], "phi": [],
        "pairs": [], "cand": {}, "guidance_active": 0, "transitions": 0,
        "shaping_abs": 0.0, "env_abs": 0.0, "shaping_mags": []}
prov = {"calls": 0, "tok_in": 0, "tok_out": 0, "multi": 0, "activated": 0,
        "validator_failures": 0}

for d in sorted(glob.glob(os.path.join(OUT, "llm_seed*"))):
    g = os.path.join(d, "guidance.csv")
    if os.path.exists(g):
        seen = set()
        with open(g) as fh:
            for r in csv.DictReader(fh):
                mech["transitions"] += 1
                try:
                    mech["env_abs"] += abs(float(r.get("raw_team_reward") or 0))
                except ValueError:
                    pass
                try:
                    sc = float(r.get("shaping_contribution") or 0)
                    if sc:
                        mech["shaping_abs"] += abs(sc)
                        mech["shaping_mags"].append(abs(sc))
                except ValueError:
                    pass
                if str(r.get("guidance_active", "")).strip().lower() in ("1", "true"):
                    mech["guidance_active"] += 1
                if not r.get("target_assignments"):
                    continue
                k = (r.get("event_id"), r.get("subgoal_id"))
                if k in seen:
                    continue
                seen.add(k)
                mech["activations"] += 1
                fo = lst(r.get("target_food_id"))
                if fo and len(set(fo)) > 1:
                    mech["span_multi"] += 1
                comps = []
                for v in kv(r.get("phi_components")).values():
                    try:
                        fv = float(v)
                    except ValueError:
                        continue
                    comps.append(fv)
                    mech["journeys"].append(max(0, int(round(abs(fv) * DMAX)) - 1))
                if comps:
                    mech["phi"].append(statistics.mean(comps))

    dec = os.path.join(d, "decisions.csv")
    if os.path.exists(dec):
        with open(dec) as fh:
            for r in csv.DictReader(fh):
                if (r.get("decision") or "").strip() != "multiple":
                    continue
                prov["multi"] += 1
                if (r.get("outcome") or "").strip() == "activated":
                    prov["activated"] += 1
                try:
                    nc = int(r.get("candidate_plan_count") or 0)
                except ValueError:
                    nc = 0
                if nc:
                    mech["cand"][nc] = mech["cand"].get(nc, 0) + 1
                pid = (r.get("selected_plan_id") or "").strip()
                tail = pid.rsplit("_", 1)[-1] if pid else ""
                if tail.isdigit() and nc > 1:
                    mech["pairs"].append((int(tail), nc))
                if str(r.get("validator_valid", "")).strip() == "0" \
                        and (r.get("validator_reason") or "").strip():
                    prov["validator_failures"] += 1
                vi = str(r.get("tokens_input", "")).strip()
                if vi not in ("", "0", "n/a", "None", "-"):
                    try:
                        prov["tok_in"] += int(float(vi))
                        prov["calls"] += 1
                        prov["tok_out"] += int(float(r.get("tokens_output") or 0))
                    except ValueError:
                        pass

j = sorted(mech["journeys"])
q = (lambda x: j[min(len(j) - 1, int(x * len(j)))]) if j else (lambda x: 0)
mg = sorted(mech["shaping_mags"])
qm = (lambda x: mg[min(len(mg) - 1, int(x * len(mg)))]) if mg else (lambda x: 0.0)
pr = mech["pairs"]
exp0 = sum(1.0 / n for _, n in pr) / len(pr) if pr else 0.0
obs0 = sum(1 for p, _ in pr if p == 0) / len(pr) if pr else 0.0

res["mechanism"] = {
    "activations": mech["activations"],
    "pct_spanning_multiple_targets": 100 * mech["span_multi"] / max(1, mech["activations"]),
    "journey_median": q(.5), "journey_p95": q(.95),
    "journey_max": j[-1] if j else 0,
    "mean_phi_at_activation": statistics.mean(mech["phi"]) if mech["phi"] else None,
    "answer_set_sizes": dict(sorted(mech["cand"].items())),
    "position0_observed": obs0, "position0_chance": exp0,
    "position0_ratio": (obs0 / exp0) if exp0 else None,
    "mean_normalised_rank": statistics.mean(p / (n - 1) for p, n in pr if n > 1)
    if pr else None,
    "coverage_pct_of_transitions": 100 * mech["guidance_active"] / max(1, mech["transitions"]),
    "shaping_mass_over_env_mass": mech["shaping_abs"] / max(1e-9, mech["env_abs"]),
    "shaping_median": qm(.5), "shaping_p99": qm(.99),
    "shaping_max": mg[-1] if mg else 0.0,
}
res["provider"] = {
    "live_calls": prov["calls"],
    "mean_tokens_in": prov["tok_in"] / max(1, prov["calls"]),
    "mean_tokens_out": prov["tok_out"] / max(1, prov["calls"]),
    "cost_usd": prov["tok_in"] / 1e6 * 0.20 + prov["tok_out"] / 1e6 * 1.25,
    "multi_plan_decisions": prov["multi"],
    "activated": prov["activated"],
    "activation_rate_pct": 100 * prov["activated"] / max(1, prov["multi"]),
    "validator_failures": prov["validator_failures"],
}

# ---------------------------------------------------------------- files ----
with open(os.path.join(REPORT, "results.json"), "w") as f:
    json.dump(res, f, indent=2)

with open(os.path.join(REPORT, "per_seed.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["condition", "seed", "auc", "prefix_auc", "prefix_points",
                "final5", "complete", "points_matched"])
    for (c, sd), r in sorted(runs.items()):
        w.writerow([c, sd, r["auc"], r["prefix_auc"], r["prefix"],
                    r["final5"], r["complete"], r["matched"]])

for script, name in (("analyse_v3.py", "analyse_v3.txt"),
                     ("mechanism_v3.py", "mechanism_v3.txt")):
    p = os.path.expanduser(f"~/selma/v3/{script}")
    if os.path.exists(p):
        args = [sys.executable, p, OUT] + (
            [str(DMAX)] if "mechanism" in script else ["--tmax", str(TMAX)])
        try:
            out = subprocess.run(args, capture_output=True, text=True,
                                 timeout=600).stdout
        except Exception as e:                      # noqa: BLE001
            out = f"(failed: {e})"
        open(os.path.join(REPORT, name), "w").write(out)

# ---------------------------------------------------------------- figures --
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"baseline": "#666666", "llm": "#1f77b4",
               "random": "#ff7f0e", "heuristic": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for cond in CONDS:
        cur = [runs[(cond, sd)]["vals"] for sd in seeds
               if runs.get((cond, sd)) and runs[(cond, sd)]["complete"]]
        part = [runs[(cond, sd)] for sd in seeds
                if runs.get((cond, sd)) and not runs[(cond, sd)]["complete"]
                and runs[(cond, sd)]["prefix"] >= 5]
        for pv in part:                       # unfinished runs, dashed
            k = pv["prefix"]
            ax.plot(GRID[:k], pv["vals"][:k], color=colours.get(cond, "k"),
                    alpha=0.55, lw=1.0, ls="--")
        if not cur:
            if part:
                ax.plot([], [], color=colours.get(cond, "k"), ls="--",
                        label=f"{cond} (partial, n={len(part)})")
            continue
        mean = [statistics.mean(c[i] for c in cur) for i in range(NPTS)]
        for c in cur:
            ax.plot(GRID, c, color=colours.get(cond, "k"), alpha=0.15, lw=0.8)
        ax.plot(GRID, mean, color=colours.get(cond, "k"), lw=2,
                label=f"{cond} (n={len(cur)})")
    ax.set_xlabel("environment steps")
    ax.set_ylabel("greedy success")
    ax.set_title(f"SELMA V3 target allocation — {os.path.basename(OUT)}")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(REPORT, f"fig_curves.{ext}"), dpi=160)
    plt.close(fig)

    pairs = res["paired"]
    if pairs:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        for i, (k, v) in enumerate(pairs.items()):
            xs = sorted(int(s) for s in v["per_seed"])
            ys = [v["per_seed"][str(s)] for s in xs]
            ax.scatter([i] * len(ys), ys, s=40, zorder=3,
                       color=colours.get(k.split("_")[0], "k"))
            ax.hlines(v["mean"], i - 0.2, i + 0.2, lw=2, color="k", zorder=4)
        ax.axhline(0, color="#999", lw=1, ls="--")
        ax.set_xticks(range(len(pairs)))
        ax.set_xticklabels([k.replace("_minus_baseline", "\n- baseline")
                            for k in pairs], fontsize=8)
        ax.set_ylabel("paired AUC difference")
        ax.set_title("Per-seed paired differences (bar = mean)")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(REPORT, f"fig_paired.{ext}"), dpi=160)
        plt.close(fig)
    figs_ok = True
except Exception as e:                              # noqa: BLE001
    figs_ok = False
    open(os.path.join(REPORT, "FIGURES_FAILED.txt"), "w").write(str(e))

# ------------------------------------------------------------- flowchart ----
fc = os.path.expanduser("~/selma/v3/make_flowchart.py")
if os.path.exists(fc):
    try:
        subprocess.run([sys.executable, fc, REPORT], capture_output=True,
                       text=True, timeout=180)
    except Exception:                              # noqa: BLE001
        pass

# ---------------------------------------------------------------- LaTeX ----
rows = []
for cond in CONDS:
    c = res["conditions"][cond]
    if not c["n_complete"]:
        continue
    cells = " & ".join(
        f"{c['per_seed_auc'][str(s)]:.4f}" if str(s) in c["per_seed_auc"] else "--"
        for s in seeds)
    rows.append(f"{cond} & {cells} & {c['mean_auc']:.4f} \\\\")
tex = ("\\begin{table}[t]\n\\centering\n\\caption{Normalised greedy-success AUC "
       "per seed. No significance is claimed.}\n\\label{tab:v3}\n"
       "\\begin{tabular}{@{}l" + "r" * (len(seeds) + 1) + "@{}}\n\\toprule\n"
       "Condition & " + " & ".join(f"S{s}" for s in seeds) + " & Mean \\\\\n"
       "\\midrule\n" + "\n".join(rows) +
       "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
open(os.path.join(REPORT, "table_v3.tex"), "w").write(tex)

open(os.path.join(REPORT, "provider.txt"), "w").write(
    json.dumps(res["provider"], indent=2))

# --------------------------------------------------------------- RESULTS ---
m, p = res["mechanism"], res["provider"]
L = []
L.append("# SELMA V3 — results\n")
L.append(f"Source: `{OUT}`  t_max={TMAX:,}  d_max={DMAX:g}\n")
L.append(f"AUC grid: {NPTS} points at {STEP:,} spacing, "
         f"normalised by {GRID[-1]-GRID[0]:,}. "
         "No evaluation point exists at t_max.\n")

L.append("\n## Learning outcome\n")
L.append("| condition | " + " | ".join(f"seed {s}" for s in seeds) + " | mean | n |")
L.append("|---|" + "---|" * (len(seeds) + 2))
for cond in CONDS:
    c = res["conditions"][cond]
    if not c["n_complete"]:
        continue
    cells = " | ".join(f"{c['per_seed_auc'][str(s)]:.4f}"
                       if str(s) in c["per_seed_auc"] else "—" for s in seeds)
    L.append(f"| {cond} | {cells} | **{c['mean_auc']:.5f}** | {c['n_complete']} |")

L.append("\n### Paired differences vs same-seed baseline\n")
for k, v in res["paired"].items():
    verdict = "**MIXED IN SIGN**" if v["mixed_sign"] else (
        "all positive" if v["positive"] == v["n"] else "all negative")
    per = "  ".join(f"s{s}:{x:+.4f}" for s, x in sorted(
        v["per_seed"].items(), key=lambda kv: int(kv[0])))
    sd = f", sd {v['sd']:.5f}" if v["sd"] is not None else ""
    L.append(f"- **{k}** — mean {v['mean']:+.5f}{sd}, n={v['n']}, {verdict}")
    L.append(f"  - {per}")
L.append("\nNo significance is claimed at this seed count. "
         "Direction and per-seed values only.\n")

cp = res.get("common_prefix")
if cp and cp["paired"]:
    L.append(f"\n## Common-prefix comparison (includes runs still training)\n")
    L.append(f"All runs truncated to the first **{cp['points']} evaluation points** "
             f"(0–{cp['horizon_steps']:,} steps), the longest window every run reached. "
             "Same window for every condition and seed.\n")
    L.append("| condition | mean AUC | n |")
    L.append("|---|---|---|")
    for c, v in cp["conditions"].items():
        L.append(f"| {c} | {v['mean']:.5f} | {v['n']} |")
    L.append("")
    for k, v in cp["paired"].items():
        verdict = "**MIXED IN SIGN**" if v["mixed_sign"] else (
            "all positive" if v["positive"] == v["n"] else "all negative")
        per = "  ".join(f"s{sd}:{x:+.4f}" for sd, x in sorted(
            v["per_seed"].items(), key=lambda kv: int(kv[0])))
        L.append(f"- **{k}** — mean {v['mean']:+.5f}, n={v['n']}, {verdict}")
        L.append(f"  - {per}")
    L.append("\nThis is a shorter horizon than the full-window result above and "
             "must be reported as such, never merged with it.\n")

L.append("\n## Mechanism (well-powered: thousands of decisions)\n")
L.append(f"- activations: **{m['activations']:,}**")
L.append(f"- spanning more than one target: **{m['pct_spanning_multiple_targets']:.1f}%** "
         "(structurally impossible under V1)")
L.append(f"- per-agent journey: median {m['journey_median']}, "
         f"p95 {m['journey_p95']}, max {m['journey_max']} "
         "(V1 was capped at 2 units between plans)")
L.append(f"- mean potential at activation: {m['mean_phi_at_activation']:.5f}")
L.append(f"- answer-set sizes: {m['answer_set_sizes']}")
if m["position0_ratio"]:
    L.append(f"- positional prior: first option chosen {100*m['position0_observed']:.1f}% "
             f"vs chance {100*m['position0_chance']:.1f}% = **{m['position0_ratio']:.2f}x** "
             f"(8x8 cell allocation was 3.55x); mean normalised rank "
             f"{m['mean_normalised_rank']:.3f} (uniform = 0.5)")

L.append("\n## Dose and coverage\n")
L.append(f"- guidance active on **{m['coverage_pct_of_transitions']:.2f}%** of transitions")
L.append(f"- shaping mass / environment reward mass: **{m['shaping_mass_over_env_mass']:.4f}** "
         "(preregistered safety gate: < 0.10)")
L.append(f"- per shaped transition: median {m['shaping_median']:.5f}, "
         f"p99 {m['shaping_p99']:.5f}, max {m['shaping_max']:.5f}")
if m["shaping_median"]:
    L.append(f"- ratio p99/median = **{m['shaping_p99']/m['shaping_median']:.0f}x** — "
             "shaping magnitude is dominated by activation/expiry transients, "
             "not by the intended per-step gradient")

L.append("\n## Provider\n")
L.append(f"- live calls: **{p['live_calls']:,}**, "
         f"mean {p['mean_tokens_in']:.0f} in / {p['mean_tokens_out']:.0f} out")
L.append(f"- cost: **${p['cost_usd']:.2f}**")
L.append(f"- multi-plan decisions: {p['multi_plan_decisions']:,}, "
         f"activated {p['activated']:,} (**{p['activation_rate_pct']:.2f}%**)")
L.append(f"- validator failures: **{p['validator_failures']}**")

L.append("\n## Files in this folder\n")
for f in sorted(os.listdir(REPORT)):
    L.append(f"- `{f}`")

open(os.path.join(REPORT, "RESULTS.md"), "w").write("\n".join(L) + "\n")

# flowchart, if the script is alongside this one
_fc = os.path.join(os.path.dirname(os.path.abspath(__file__)), "make_flowchart.py")
if os.path.exists(_fc):
    try:
        subprocess.run([sys.executable, _fc, REPORT], capture_output=True,
                       text=True, timeout=180)
    except Exception:                                # noqa: BLE001
        pass

print(f"wrote {REPORT}")
for f in sorted(os.listdir(REPORT)):
    print("  ", f)
print(f"\nfigures: {'ok' if figs_ok else 'FAILED (see FIGURES_FAILED.txt)'}")
