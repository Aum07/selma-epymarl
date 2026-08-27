#!/usr/bin/env python3
"""SELMA V3 10x10 -- mechanism diagnostics across all guided runs.

This is the well-powered half of the experiment. It does not depend on seed
count: every number here comes from thousands of decisions, not four seeds.

The headline comparison is the V3 analogue of the paper's 49.5 % finding.
Because the heuristic selects minimum total travel, it MAXIMISES the potential
by construction, and the random selector samples it uniformly. So:

    heuristic  = the ceiling of realised Phi at activation
    random     = the floor
    llm        = where the model actually lands

    capture = (llm - random) / (heuristic - random)

On 8x8 the model captured ~3.5 % of that range. Whether it does better once the
choice is a real allocation across foods -- rather than a 2-unit tiebreak -- is
the question this experiment exists to answer.

    python3 mechanism_v3.py [OUTDIR] [d_max]
"""
import collections
import csv
import glob
import os
import statistics
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else \
    os.path.expanduser("~/selma/v3/analysis/out/v3_10x10")
DMAX = float(sys.argv[2]) if len(sys.argv) > 2 else 19.0
CONDS = ["heuristic", "random", "llm"]


def kv(s):
    out = {}
    for part in str(s or "").split(";"):
        if "=" in part:
            k, _, v = part.partition("=")
            out[k.strip()] = v.strip()
    return out


def lst(s):
    return [x.strip() for x in str(s or "").split(";") if x.strip()]


def analyse(d):
    g = os.path.join(d, "guidance.csv")
    dec = os.path.join(d, "decisions.csv")
    r = {"activations": 0, "spanmulti": 0, "journeys": [], "phi": [],
         "posn": collections.Counter(), "cand": collections.Counter(),
         "multi": 0, "singleton": 0, "calls": 0, "ti": 0, "to": 0,
         "validfail": 0, "pairs": [], "tgtdist": collections.Counter()}

    if os.path.exists(g):
        seen = set()
        with open(g) as fh:
            for row in csv.DictReader(fh):
                if not row.get("target_assignments"):
                    continue
                key = (row.get("event_id"), row.get("subgoal_id"))
                if key in seen:
                    continue
                seen.add(key)
                r["activations"] += 1
                # target_food_id holds ONE food as an (episode;row;col)
                # triple, so splitting it on ";" counts 3 tokens for a single
                # food. The allocation lives in target_assignments, which kv()
                # parses as agent -> "row,col"; distinct VALUES are the targets.
                targets = set(kv(row.get("target_assignments")).values())
                if len(targets) > 1:
                    r["spanmulti"] += 1
                r["tgtdist"][len(targets)] += 1
                js = []
                for v in kv(row.get("phi_components")).values():
                    try:
                        js.append(max(0, int(round(abs(float(v)) * DMAX)) - 1))
                    except ValueError:
                        pass
                if js:
                    r["journeys"] += js
                try:
                    # realised potential at activation: mean of the components
                    comps = [float(v) for v in kv(row.get("phi_components")).values()]
                    if comps:
                        r["phi"].append(statistics.mean(comps))
                except ValueError:
                    pass

    if os.path.exists(dec):
        with open(dec) as fh:
            for row in csv.DictReader(fh):
                dcn = (row.get("decision") or "").strip()
                if dcn == "singleton":
                    r["singleton"] += 1
                    continue
                if dcn == "multiple":
                    r["multi"] += 1
                # ids are like "target_plan_3" (V3) or "plan_3" (V2):
                # take the trailing integer, whatever the prefix.
                pid = (row.get("selected_plan_id") or "").strip()
                try:
                    ncand = int(row.get("candidate_plan_count") or 0)
                except ValueError:
                    ncand = 0
                if ncand:
                    r["cand"][ncand] += 1
                if pid:
                    tail = pid.rsplit("_", 1)[-1]
                    if tail.isdigit():
                        pos = int(tail)
                        r["posn"][pos] += 1
                        # answer-set sizes VARY here (2..15), so "chance" for
                        # position 0 is E[1/n], not 1/6. Keep the pair.
                        if ncand > 1:
                            r["pairs"].append((pos, ncand))
                if str(row.get("validator_valid", "")).strip() == "0" \
                        and (row.get("validator_reason") or "").strip():
                    r["validfail"] += 1
                vi = str(row.get("tokens_input", "")).strip()
                if vi not in ("", "0", "n/a", "None", "-"):
                    try:
                        r["ti"] += int(float(vi)); r["calls"] += 1
                        r["to"] += int(float(row.get("tokens_output") or 0))
                    except ValueError:
                        pass
    return r


agg = {c: None for c in CONDS}
for cond in CONDS:
    tot = {"activations": 0, "spanmulti": 0, "journeys": [], "phi": [],
           "posn": collections.Counter(), "cand": collections.Counter(),
           "multi": 0, "singleton": 0, "calls": 0, "ti": 0, "to": 0,
           "validfail": 0, "pairs": [], "runs": 0,
           "tgtdist": collections.Counter()}
    for d in sorted(glob.glob(os.path.join(OUT, f"{cond}_seed*"))):
        r = analyse(d)
        tot["runs"] += 1
        for k in ("activations", "spanmulti", "multi", "singleton",
                  "calls", "ti", "to", "validfail"):
            tot[k] += r[k]
        tot["journeys"] += r["journeys"]
        tot["phi"] += r["phi"]
        tot["posn"] += r["posn"]
        tot["cand"] += r["cand"]
        tot["pairs"] += r["pairs"]
        tot["tgtdist"] += r["tgtdist"]
    agg[cond] = tot

print(f"outdir: {OUT}   d_max={DMAX:g}\n")
print(f"{'':<12}{'runs':>5}{'activ':>8}{'>1 food':>9}{'multi':>8}"
      f"{'single':>8}{'medJ':>6}{'p95J':>6}{'meanPhi':>10}")
print("-" * 72)
for cond in CONDS:
    t = agg[cond]
    if not t["runs"]:
        continue
    j = sorted(t["journeys"])
    medj = j[len(j)//2] if j else 0
    p95 = j[min(len(j)-1, int(.95*len(j)))] if j else 0
    span = f"{100*t['spanmulti']/t['activations']:.0f}%" if t["activations"] else "-"
    mphi = statistics.mean(t["phi"]) if t["phi"] else float("nan")
    print(f"{cond:<12}{t['runs']:>5}{t['activations']:>8}{span:>9}"
          f"{t['multi']:>8}{t['singleton']:>8}{medj:>6}{p95:>6}{mphi:>10.4f}")

print("\n--- distinct targets in the selected plan ---")
print("  counted from target_assignments, over distinct (event, subgoal) pairs")
print(f"  {'':<12}{'activ':>8}{'1 tgt':>9}{'2 tgt':>9}{'3 tgt':>9}{'>=2':>9}")
for cond in CONDS:
    t = agg[cond]
    n = sum(t["tgtdist"].values())
    if not n:
        continue
    pc = lambda k: f"{100*t['tgtdist'][k]/n:.1f}%"
    ge2 = 100 * sum(v for k, v in t["tgtdist"].items() if k >= 2) / n
    print(f"  {cond:<12}{n:>8}{pc(1):>9}{pc(2):>9}{pc(3):>9}{ge2:>8.1f}%")

# ---- the headline: how close does the model get to the potential ceiling? --
h, rn, l = (statistics.mean(agg[c]["phi"]) if agg[c]["phi"] else None
            for c in ("heuristic", "random", "llm"))
print("\n--- realised potential at activation ---")
if None not in (h, rn, l):
    print(f"  heuristic (ceiling, min-travel) : {h:+.5f}")
    print(f"  random    (floor, uniform)      : {rn:+.5f}")
    print(f"  llm                             : {l:+.5f}")
    rng = h - rn
    if abs(rng) > 1e-9:
        print(f"\n  >>> LLM captures {100*(l-rn)/rng:.1f}% of the "
              f"heuristic-over-random range")
        print(f"      (on 8x8 with the small lever this was ~3.5%)")
    else:
        print("\n  heuristic and random are indistinguishable -- no range to capture.")
else:
    print("  (need all three conditions to have activations)")

print("\n--- selected plan position vs CHANCE ---")
print("  answer sets vary in size, so chance for position 0 is E[1/n], not 1/6")
for cond in CONDS:
    pr = agg[cond]["pairs"]
    if not pr:
        continue
    n = len(pr)
    first = sum(1 for pos, _ in pr if pos == 0)
    exp0 = sum(1.0 / nc for _, nc in pr) / n
    obs0 = first / n
    ratio = obs0 / exp0 if exp0 else float("nan")
    # normalised rank: 0 = always first, 1 = always last, 0.5 = uniform
    nr = statistics.mean(pos / (nc - 1) for pos, nc in pr if nc > 1)
    print(f"  {cond:<10} n={n:<6} pos0 observed {100*obs0:5.1f}%  "
          f"chance {100*exp0:5.1f}%  ratio {ratio:.2f}x   mean norm-rank {nr:.3f}")
    print(f"  {'':<10} dist {sorted(agg[cond]['posn'].items())[:8]}")
print("  8x8 reference: 59.3% observed vs 16.7% chance = 3.55x, norm-rank 0.24")
print("  ratio near 1.00 and norm-rank near 0.50 mean NO positional prior")

print("\n--- answer-set size ---")
for cond in CONDS:
    c = agg[cond]["cand"]
    if c:
        print(f"  {cond:<10} {sorted(c.items())}")

t = agg["llm"]
if t["calls"]:
    cost = t["ti"]/1e6*0.20 + t["to"]/1e6*1.25
    print(f"\n--- provider ---")
    print(f"  live calls {t['calls']}   mean in {t['ti']/t['calls']:.0f} tok   "
          f"mean out {t['to']/t['calls']:.0f} tok")
    print(f"  cost ${cost:.4f}   validator failures {t['validfail']}")
