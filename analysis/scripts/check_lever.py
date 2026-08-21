#!/usr/bin/env python3
"""Is the V3 lever big?  Reads the real log format: "k=v;k=v;k=v".

    target_food_id     : 30;3;3          -> one value per recipient
    recipient_agent_ids: 0;1;2
    target_assignments : 0=4,6;1=3,3     -> agent = row,col
    phi_components     : 0=-0.368;1=...  -> agent = phi

phi_i = -(journey + 1[food remains]) / d_max, so with d_max = 19:
    journey = round(|phi_i| * 19) - 1

Usage: python3 check_lever.py analysis/out/v3_10x10/smoke [d_max]
"""
import collections
import csv
import os
import sys

d = sys.argv[1] if len(sys.argv) > 1 else "."
DMAX = float(sys.argv[2]) if len(sys.argv) > 2 else 19.0
g = os.path.join(d, "guidance.csv")
rows = list(csv.DictReader(open(g)))


def kv(s):
    """'0=-0.36;1=-0.21' -> {'0': '-0.36', '1': '-0.21'}"""
    out = {}
    if not s:
        return out
    for part in str(s).split(";"):
        if "=" in part:
            k, _, v = part.partition("=")
            out[k.strip()] = v.strip()
    return out


def lst(s):
    """'30;3;3' -> ['30','3','3']"""
    return [x.strip() for x in str(s).split(";") if x.strip() != ""] if s else []


act = [r for r in rows if r.get("target_assignments")]
print(f"guidance rows {len(rows)}   guidance-active rows {len(act)}   d_max {DMAX:g}\n")
if not act:
    sys.exit("No guidance ever carried assignments. STOP.")

# one record per activation, not per served transition
seen, uniq = set(), []
for r in act:
    k = (r.get("event_id"), r.get("subgoal_id"))
    if k not in seen:
        seen.add(k)
        uniq.append(r)
print(f"distinct activations: {len(uniq)}\n")

# ---- (a) how many distinct FOODS does one activation span? ----------------
spans = collections.Counter()
for r in uniq:
    f = lst(r.get("target_food_id"))
    if f:
        spans[len(set(f))] += 1
tot = sum(spans.values())
print(f"--- (a) distinct target foods per activation (n={tot}) ---")
print(f"    {sorted(spans.items())}")
if tot:
    multi = sum(v for k, v in spans.items() if k > 1)
    print(f"    activations spanning >1 food : {multi}/{tot}  ({100*multi/tot:.1f} %)")
    print("    (under V1 this could ONLY ever be 1)\n")

# ---- (b) distinct target CELLS per activation -----------------------------
cellspans = collections.Counter()
for r in uniq:
    cells = set(kv(r.get("target_assignments")).values())
    if cells:
        cellspans[len(cells)] += 1
print(f"--- (b) distinct target cells per activation ---\n    {sorted(cellspans.items())}\n")

# ---- (c) journeys, the number that decides it -----------------------------
per_act_max, journeys = [], []
for r in uniq:
    js = []
    for v in kv(r.get("phi_components")).values():
        try:
            js.append(max(0, int(round(abs(float(v)) * DMAX)) - 1))
        except ValueError:
            pass
    if js:
        journeys += js
        per_act_max.append(max(js))

if journeys:
    journeys.sort()
    n = len(journeys)
    q = lambda x: journeys[min(n - 1, int(x * n))]
    print(f"--- (c) per-agent journey at activation (n={n}) ---")
    print(f"    min {journeys[0]}   median {q(.5)}   p95 {q(.95)}   max {journeys[-1]}")
    print(f"    distribution {sorted(collections.Counter(journeys).items())}\n")

    per_act_max.sort()
    m = len(per_act_max)
    qm = lambda x: per_act_max[min(m - 1, int(x * m))]
    print(f"--- max per-agent journey WITHIN an activation (n={m}) ---")
    print(f"    median {qm(.5)}   p95 {qm(.95)}   max {per_act_max[-1]}")
    print(f"\n    >>> calibrated TTL = p95 + 1 = {qm(.95)+1}   (runner uses 20)\n")

    p95 = q(.95)
    if p95 <= 4:
        print("    VERDICT: tiny journeys -> V1 behaviour. DO NOT LAUNCH.")
    elif p95 <= 7:
        print("    VERDICT: real but modest. Bigger than V1 (which was capped at 2")
        print("             units between plans). Launchable; expect a moderate effect.")
    else:
        print("    VERDICT: journeys span the grid. BIG LEVER -- LAUNCH.")
else:
    print("phi_components still unparsed -- paste two raw values.")
