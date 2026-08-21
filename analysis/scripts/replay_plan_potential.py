#!/usr/bin/env python
"""Counterfactual replay: does the selected plan actually move the potential?

This answers the question the per-transition logs cannot. For every cached
decision from the guided runs it recomputes, from the stored request, the
potential under EVERY offered joint plan, under both potentials:

    V1 (Stage E)  phi_i = -(|p_i - food| + 1) / d_max
    V2 (current)  phi_i = -(|p_i - g_i|  + 1) / d_max     g_i = i's assigned cell

and reports the spread across the offered plans. Under V1 that spread must be
exactly zero at every decision -- that is the defect, restated as a measurement.
Under V2 it must be non-zero wherever the plans genuinely differ.

It then scores the model's actual choice against the best and worst offered
plan, and against what a minimum-travel heuristic and a uniform random selector
would have chosen on the SAME decisions.

Entirely offline: reads the run's SQLite response cache, makes no network call,
starts no training, and writes nothing outside --outdir.

Usage:
    python replay_plan_potential.py
    python replay_plan_potential.py --outdir analysis/out/v2_report
"""

import argparse
import glob
import json
import os
import sqlite3
import statistics
import sys
from collections import Counter

ROOT_DEFAULT = os.path.expanduser("~/selma/epymarl/analysis/out/v2_plan_potential")


def d_max(grid):
    return (int(grid[0]) - 1) + (int(grid[1]) - 1) + 1


def manhattan(a, b):
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def phi_v1(positions, food, grid):
    dm = d_max(grid)
    vals = [-(manhattan(p, food) + 1) / dm for p in positions.values()]
    return sum(vals) / len(vals) if vals else 0.0


def phi_v2(positions, plan_map, grid):
    dm = d_max(grid)
    vals = []
    for agent, pos in positions.items():
        cell = plan_map.get(agent)
        if cell is None:
            return None
        vals.append(-(manhattan(pos, cell) + 1) / dm)
    return sum(vals) / len(vals) if vals else 0.0


def travel(positions, plan_map):
    return sum(manhattan(positions[a], plan_map[a]) for a in positions
               if a in plan_map)


def replay(db_path):
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    out = dict(n=0, v1_spread_nonzero=0, v2_spread_nonzero=0,
               v2_spreads=[], sel_rank=Counter(), n_plans=Counter(),
               regret=[], sel_is_best=0, heur_is_best=0, rand_expected=[],
               sel_phi=[], best_phi=[], worst_phi=[], mean_phi=[],
               sel_travel=[], best_travel=[], agree_heuristic=0)
    for row in con.execute("SELECT canonical_key, response_json FROM guidance_cache"):
        try:
            key = json.loads(row["canonical_key"])
            resp = json.loads(row["response_json"])
        except (ValueError, TypeError):
            continue
        grid = key.get("grid_shape")
        food = key.get("food_position")
        plans = key.get("candidate_joint_plans") or []
        recips = key.get("recipient_agents") or []
        if not grid or not food or len(plans) < 2:
            continue
        positions = {int(r["agent_id"]): tuple(r["position"])
                     for r in recips if r.get("position")}
        if not positions:
            continue

        maps = {}
        for p in plans:
            maps[p["plan_id"]] = {int(a["agent_id"]): tuple(a["loading_cell"])
                                  for a in p["assignments"]}

        v1 = [phi_v1(positions, food, grid) for _ in plans]
        v2 = {pid: phi_v2(positions, m, grid) for pid, m in maps.items()}
        if any(v is None for v in v2.values()):
            continue

        out["n"] += 1
        out["n_plans"][len(plans)] += 1
        if max(v1) - min(v1) > 1e-12:
            out["v1_spread_nonzero"] += 1
        spread = max(v2.values()) - min(v2.values())
        out["v2_spreads"].append(spread)
        if spread > 1e-12:
            out["v2_spread_nonzero"] += 1

        sel = resp.get("selected_plan_id")
        if sel not in v2:
            continue
        best = max(v2.values())
        worst = min(v2.values())
        out["sel_phi"].append(v2[sel])
        out["best_phi"].append(best)
        out["worst_phi"].append(worst)
        out["mean_phi"].append(sum(v2.values()) / len(v2))
        out["regret"].append(best - v2[sel])
        if abs(v2[sel] - best) < 1e-12:
            out["sel_is_best"] += 1
        # rank of the selected plan, 0 = best
        ordered = sorted(v2.values(), reverse=True)
        out["sel_rank"][ordered.index(v2[sel])] += 1
        # what a minimum-travel heuristic would pick (canonical tie-break)
        costs = {pid: travel(positions, m) for pid, m in maps.items()}
        heur = min(sorted(costs), key=lambda p: costs[p])
        if heur == sel:
            out["agree_heuristic"] += 1
        out["sel_travel"].append(costs[sel])
        out["best_travel"].append(min(costs.values()))
        if abs(v2[heur] - best) < 1e-12:
            out["heur_is_best"] += 1
        out["rand_expected"].append(sum(v2.values()) / len(v2))
    con.close()
    return out


def report(name, r):
    n = r["n"]
    if not n:
        print(f"\n{name}: no usable cached decisions")
        return
    pct = lambda x: f"{x/n:.1%}"
    print(f"\n=== {name} ===")
    print(f"  cached decisions replayed        : {n:,}")
    print(f"  candidate counts                 : "
          f"{dict(sorted(r['n_plans'].items()))}")
    print(f"  V1 potential differs across plans: {r['v1_spread_nonzero']:,} "
          f"({pct(r['v1_spread_nonzero'])})   <- must be 0.0%")
    print(f"  V2 potential differs across plans: {r['v2_spread_nonzero']:,} "
          f"({pct(r['v2_spread_nonzero'])})   <- the restored causal channel")
    if r["v2_spreads"]:
        s = r["v2_spreads"]
        print(f"  V2 spread best-vs-worst  mean    : {statistics.mean(s):.5f}"
              f"   max {max(s):.5f}")
    m = len(r["sel_phi"])
    if m:
        print(f"  selected plan was the best       : {r['sel_is_best']:,}"
              f" ({r['sel_is_best']/m:.1%})")
        print(f"  min-travel heuristic was best    : {r['heur_is_best']:,}"
              f" ({r['heur_is_best']/m:.1%})")
        print(f"  model agreed with the heuristic  : {r['agree_heuristic']:,}"
              f" ({r['agree_heuristic']/m:.1%})")
        print(f"  mean potential regret vs best    : "
              f"{statistics.mean(r['regret']):.5f}")
        print(f"  mean phi  selected {statistics.mean(r['sel_phi']):.5f}"
              f" | best {statistics.mean(r['best_phi']):.5f}"
              f" | random-expected {statistics.mean(r['mean_phi']):.5f}"
              f" | worst {statistics.mean(r['worst_phi']):.5f}")
        print(f"  mean travel selected {statistics.mean(r['sel_travel']):.3f}"
              f" vs optimal {statistics.mean(r['best_travel']):.3f}")
        ranks = ", ".join(f"#{k+1}={v/m:.1%}"
                          for k, v in sorted(r["sel_rank"].items()))
        print(f"  rank of selected plan            : {ranks}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT_DEFAULT)
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()

    dbs = sorted(glob.glob(os.path.join(a.root, "*", "guidance_cache.sqlite3")))
    if not dbs:
        print(f"no guidance caches under {a.root}")
        return 1

    print("=" * 74)
    print("COUNTERFACTUAL REPLAY -- does the selected plan move the potential?")
    print("=" * 74)
    print("Offline. No provider call, no training, no run directory written.")

    allr = {}
    for db in dbs:
        name = os.path.basename(os.path.dirname(db))
        r = replay(db)
        allr[name] = r
        report(name, r)

    tot = dict(n=0, v1=0, v2=0, best=0, agree=0, m=0)
    for r in allr.values():
        tot["n"] += r["n"]; tot["v1"] += r["v1_spread_nonzero"]
        tot["v2"] += r["v2_spread_nonzero"]; tot["best"] += r["sel_is_best"]
        tot["agree"] += r["agree_heuristic"]; tot["m"] += len(r["sel_phi"])
    if tot["n"]:
        print("\n" + "=" * 74)
        print(f"POOLED over {tot['n']:,} cached decisions")
        print(f"  V1 plan-conditional : {tot['v1']/tot['n']:.2%}"
              f"   (the Stage E defect, measured)")
        print(f"  V2 plan-conditional : {tot['v2']/tot['n']:.2%}"
              f"   (the correction, measured)")
        if tot["m"]:
            print(f"  model picked the potential-optimal plan : "
                  f"{tot['best']/tot['m']:.1%}")
            print(f"  model agreed with min-travel heuristic  : "
                  f"{tot['agree']/tot['m']:.1%}")

    if a.outdir:
        os.makedirs(a.outdir, exist_ok=True)
        p = os.path.join(a.outdir, "replay_plan_potential.json")
        with open(p, "w") as fh:
            json.dump({k: {kk: (dict(vv) if isinstance(vv, Counter) else
                                (statistics.mean(vv) if isinstance(vv, list) and vv
                                 else vv))
                           for kk, vv in v.items()}
                       for k, v in allr.items()}, fh, indent=2, default=str)
        print(f"\nreport: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
