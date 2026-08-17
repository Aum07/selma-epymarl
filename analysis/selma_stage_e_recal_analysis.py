#!/usr/bin/env python
"""SELMA Stage E: locked analysis of the six beta-recalibration conditions.

Run **only after all six conditions have completed.** It re-derives every
technical gate from the durable logs, computes the preregistered primary and
secondary scores, and applies the ranking rule mechanically.

The rule (frozen before any outcome existed)
--------------------------------------------

1. Every one of the six conditions must pass all technical gates **and** reach
   at least 95 % guidance coverage.
2. If any condition fails, **beta remains UNRESOLVED.**
3. **Primary score:** the mean across seeds 91 and 92 of normalised
   greedy-success AUC over exactly the 18 frozen post-onset evaluation points.
4. **Secondary tie-breaker:** mean greedy success over the final five frozen
   evaluation points, across both seeds.
5. If still equal within 1e-12, choose the **smaller** beta.
6. **beta = 0 is allowed to win.**
7. Two development seeds support calibration only -- **never** a significance or
   generalisation claim.

Nothing here is adjusted after outcomes are opened, and seed 99 and every
final/test seed are untouched.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FROZEN_BETAS = (0.00, 0.05, 0.10)
FROZEN_SEEDS = (91, 92)
FROZEN_T_MAX = 1500000
FROZEN_MODEL = "gpt-5.4-nano-2026-03-17"
FROZEN_ENV = "lbforaging:Foraging-8x8-2p-2f-coop-v3"
FORBIDDEN_SEEDS = (99,)

MIN_GUIDANCE_COVERAGE = 0.95
PER_CONDITION_CALL_CAP = 13500
PER_CONDITION_COST_CAP_USD = 3.49
GLOBAL_COST_CAP_USD = 20.94

PRICE_INPUT = 0.20
PRICE_CACHED_INPUT = 0.02
PRICE_OUTPUT = 1.25

#: The 18 frozen post-onset evaluation points, fixed by the design: onset is
#: t_env 550 050 and the schedule is every 50 000 steps, so these are the first
#: 18 scheduled evaluations strictly after onset. Matching is by nearest
#: scheduled multiple within half an interval, because the actual t_env of an
#: evaluation lands on an episode boundary rather than an exact multiple.
FROZEN_EVAL_POINTS = tuple(range(600000, 600000 + 18 * 50000, 50000))
EVAL_MATCH_TOLERANCE = 25000
FINAL_WINDOW_POINTS = 5

TIE_EPSILON = 1e-12


class AnalysisError(RuntimeError):
    """Missing or unauthenticated evidence. Nothing is guessed."""


def _int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _float(value, default=None):
    if value in (None, "", "n/a"):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _finite(value):
    return value is not None and isinstance(value, float) and math.isfinite(value)


def read_csv(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def condition_name(beta, seed):
    return "beta_{:.2f}_seed{}".format(beta, seed).replace(".", "p")


# ===========================================================================
# 1. authentication
# ===========================================================================


def authenticate(condition_dir, beta, seed):
    """Confirm this directory really holds the condition it claims to."""
    problems = []
    manifest_path = os.path.join(condition_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise AnalysisError("missing manifest in {!r}".format(condition_dir))
    with open(manifest_path) as handle:
        manifest = json.load(handle)

    config = manifest.get("condition", {})
    if abs(float(config.get("beta", -1)) - beta) > 1e-12:
        problems.append("manifest beta {} != {}".format(config.get("beta"), beta))
    if int(config.get("seed", -1)) != seed:
        problems.append("manifest seed {} != {}".format(config.get("seed"), seed))
    if int(config.get("t_max", 0)) != FROZEN_T_MAX:
        problems.append("t_max {} != {}".format(config.get("t_max"), FROZEN_T_MAX))
    if config.get("env_key") != FROZEN_ENV:
        problems.append("env {}".format(config.get("env_key")))
    if config.get("model") != FROZEN_MODEL:
        problems.append("model {}".format(config.get("model")))
    if seed in FORBIDDEN_SEEDS:
        problems.append("seed {} is forbidden".format(seed))
    if manifest.get("exit_code") != 0:
        problems.append("exit code {}".format(manifest.get("exit_code")))
    if not os.path.exists(os.path.join(condition_dir, "COMPLETED")):
        problems.append("no COMPLETED marker")

    # artifact hashes must still match what the controller recorded
    recorded = manifest.get("artifact_hashes", {})
    rehashed = {}
    for key, entry in recorded.items():
        path = entry.get("path")
        if not path or not os.path.exists(path):
            problems.append("artifact {} missing".format(key))
            continue
        digest = sha256_file(path)
        rehashed[key] = digest
        if digest != entry.get("sha256"):
            problems.append("artifact {} hash changed since the run".format(key))

    return {"manifest": manifest, "problems": problems, "rehashed": rehashed}


# ===========================================================================
# 2. technical gates, recomputed from the logs
# ===========================================================================


def technical_gates(condition_dir, beta, seed):
    guidance = read_csv(os.path.join(condition_dir, "guidance.csv"))
    decisions = read_csv(os.path.join(condition_dir, "decisions.csv"))
    episodes = read_csv(os.path.join(condition_dir, "episodes.csv"))
    if guidance is None or decisions is None or episodes is None:
        raise AnalysisError("missing logs in {!r}".format(condition_dir))

    problems = []
    checks = {}

    checks["final_t_env"] = max((_int(r["t_env"]) for r in guidance), default=0)
    if checks["final_t_env"] < FROZEN_T_MAX - 50:
        problems.append("horizon not reached ({})".format(checks["final_t_env"]))

    considered = [r for r in decisions if r["decision"] == "multiple"]
    exhausted = sum(1 for r in considered if r["outcome"] == "budget_exhausted")
    coverage = ((len(considered) - exhausted) / float(len(considered))
                if considered else 1.0)
    checks["llm_eligible_decisions"] = len(considered)
    checks["budget_exhausted"] = exhausted
    checks["guidance_coverage"] = coverage
    if coverage < MIN_GUIDANCE_COVERAGE:
        problems.append("coverage {:.4f} < {}".format(coverage,
                                                      MIN_GUIDANCE_COVERAGE))

    called = [r for r in considered if _int(r["live_attempts"]) > 0]
    attempts = sum(_int(r["live_attempts"]) for r in considered)
    retries = sum(_int(r["retries"]) for r in considered)
    tin = sum(_int(r["tokens_input"]) for r in called)
    tcached = sum(_int(r["tokens_cached_input"]) for r in called)
    tout = sum(_int(r["tokens_output"]) for r in called)
    cost = ((tin - tcached) * PRICE_INPUT + tcached * PRICE_CACHED_INPUT
            + tout * PRICE_OUTPUT) / 1e6
    digests = [r["key_digest"] for r in considered if r["key_digest"]]

    checks.update({
        "detector_events": sum(_int(r.get("detector_events")) for r in guidance),
        "live_attempts": attempts,
        "retries": retries,
        "distinct_semantic_keys": len(set(digests)),
        "cache_hits": sum(1 for r in considered if r["cache_result"] == "hit"),
        "cache_misses": sum(1 for r in considered if r["cache_result"] == "miss"),
        "activations": sum(_int(r["activated"]) for r in decisions),
        "input_tokens": tin, "cached_input_tokens": tcached,
        "output_tokens": tout, "cost_usd": cost,
    })
    if attempts > PER_CONDITION_CALL_CAP:
        problems.append("attempts {} > cap".format(attempts))
    if cost > PER_CONDITION_COST_CAP_USD:
        problems.append("cost ${:.4f} > cap".format(cost))

    models = {r["returned_model"] for r in called if r["returned_model"]}
    checks["returned_models"] = sorted(models)
    if models - {FROZEN_MODEL}:
        problems.append("model substitution {}".format(sorted(models)))

    activated = [r for r in decisions if _int(r["activated"])]
    invalid = sum(1 for r in activated if not _int(r["physically_valid"]))
    invented = sum(1 for r in activated if not r["selected_plan_id"])
    greedy = sum(1 for r in decisions if _int(r["greedy_eval"]))
    checks.update({"physically_invalid_activations": invalid,
                   "invented_plan_ids": invented,
                   "greedy_eval_decisions": greedy})
    if invalid:
        problems.append("{} physically invalid activations".format(invalid))
    if invented:
        problems.append("{} invented plan ids".format(invented))
    if greedy:
        problems.append("{} greedy-evaluation calls".format(greedy))

    reward_errors = non_finite = terminal_nonzero = 0
    residual = 0.0
    by_episode = {}
    for row in guidance:
        raw = _float(row["raw_team_reward"])
        train = _float(row["training_reward"])
        shaping = _float(row["shaping_contribution"], 0.0)
        for value in (raw, train, shaping,
                      _float(row["pbrs_term_unscaled"], 0.0),
                      _float(row["phi_before"], 0.0),
                      _float(row["phi_after_effective"], 0.0)):
            if value is None or not math.isfinite(value):
                non_finite += 1
        if None not in (raw, train) and abs(train - (raw + shaping)) > 1e-9:
            reward_errors += 1
        if _int(row["stored_terminated"]) and \
                _float(row["phi_after_effective"], 0.0) != 0.0:
            terminal_nonzero += 1
        by_episode.setdefault(row["episode_id"], []).append(row)
    for rows in by_episode.values():
        rows.sort(key=lambda r: _int(r["episode_step"]))
        for previous, current in zip(rows, rows[1:]):
            residual = max(residual, abs(
                _float(previous["phi_after_effective"], 0.0)
                - _float(current["phi_before"], 0.0)))

    checks.update({"reward_reconciliation_errors": reward_errors,
                   "non_finite_values": non_finite,
                   "max_telescoping_residual": residual,
                   "terminal_phi_nonzero": terminal_nonzero})
    if reward_errors:
        problems.append("{} reward reconciliation errors".format(reward_errors))
    if non_finite:
        problems.append("{} non-finite values".format(non_finite))
    if residual > 1e-9:
        problems.append("telescoping residual {:.3g}".format(residual))
    if terminal_nonzero:
        problems.append("{} non-zero terminal phi".format(terminal_nonzero))

    betas = {row.get("beta") for row in guidance}
    checks["beta_values_in_log"] = sorted(betas)
    if len(betas) != 1 or abs(float(next(iter(betas))) - beta) > 1e-12:
        problems.append("beta in log {} != {}".format(sorted(betas), beta))

    return {"problems": problems, "checks": checks, "episodes": episodes}


# ===========================================================================
# 3. the frozen evaluation window and the primary score
# ===========================================================================


def greedy_points(episodes):
    """Mean greedy success and return per evaluation point."""
    grouped = {}
    for row in episodes:
        if _int(row.get("test_mode")) != 1:
            continue
        grouped.setdefault(_int(row["t_env"]), []).append(row)
    points = []
    for t_env in sorted(grouped):
        rows = grouped[t_env]
        successes = [1.0 if _float(r["team_return"], 0.0) > 0 else 0.0
                     for r in rows]
        returns = [_float(r["team_return"], 0.0) for r in rows]
        points.append({"t_env": t_env, "episodes": len(rows),
                       "success_rate": statistics.mean(successes),
                       "mean_return": statistics.mean(returns)})
    return points


def select_frozen_points(points):
    """Match the 18 frozen scheduled evaluations, nearest within tolerance."""
    selected = []
    missing = []
    for target in FROZEN_EVAL_POINTS:
        candidates = [p for p in points
                      if abs(p["t_env"] - target) <= EVAL_MATCH_TOLERANCE]
        if not candidates:
            missing.append(target)
            continue
        best = min(candidates, key=lambda p: abs(p["t_env"] - target))
        selected.append({"target": target, **best})
    return selected, missing


def success_auc(selected):
    """Trapezoid AUC of greedy success over the frozen window, normalised.

    Normalised by the t_env span so it reads as a mean success rate, which makes
    it comparable across conditions and directly interpretable.
    """
    if len(selected) < 2:
        return None
    total = 0.0
    for a, b in zip(selected, selected[1:]):
        total += (b["target"] - a["target"]) * (a["success_rate"]
                                                + b["success_rate"]) / 2.0
    span = selected[-1]["target"] - selected[0]["target"]
    return total / span if span else None


def final_window_success(selected, n=FINAL_WINDOW_POINTS):
    if len(selected) < n:
        return None
    return statistics.mean(p["success_rate"] for p in selected[-n:])


# ===========================================================================
# 4. the preregistered selection rule
# ===========================================================================


def select_beta(per_beta, all_gates_passed):
    """Apply the frozen rule mechanically. beta=0 may win."""
    steps = []
    if not all_gates_passed:
        steps.append({"step": 1, "action": "a condition failed its gates"})
        return {"selected_beta": None, "resolved": False, "steps": steps,
                "reason": "At least one of the six conditions failed a technical "
                          "gate or the 95 % coverage requirement, so beta remains "
                          "UNRESOLVED (rule 2). No cap may be raised post hoc."}

    scored = [b for b in per_beta if b["primary_score"] is not None]
    if len(scored) != len(FROZEN_BETAS):
        steps.append({"step": 3, "action": "a beta had no computable score"})
        return {"selected_beta": None, "resolved": False, "steps": steps,
                "reason": "The primary score could not be computed for every "
                          "beta; beta remains UNRESOLVED."}

    best = max(s["primary_score"] for s in scored)
    leaders = [s for s in scored
               if abs(s["primary_score"] - best) <= TIE_EPSILON]
    steps.append({"step": 3, "action": "primary: mean greedy-success AUC over "
                                       "the 18 frozen points",
                  "scores": {s["beta"]: s["primary_score"] for s in scored},
                  "leaders": [s["beta"] for s in leaders]})

    if len(leaders) > 1:
        best_secondary = max(s["secondary_score"] or 0.0 for s in leaders)
        leaders = [s for s in leaders
                   if abs((s["secondary_score"] or 0.0) - best_secondary)
                   <= TIE_EPSILON]
        steps.append({"step": 4, "action": "secondary: mean greedy success over "
                                          "the final five frozen points",
                      "leaders": [s["beta"] for s in leaders]})

    if len(leaders) > 1:
        leaders = [min(leaders, key=lambda s: s["beta"])]
        steps.append({"step": 5, "action": "still tied within 1e-12; the "
                                          "smaller beta is chosen",
                      "selected": leaders[0]["beta"]})

    chosen = leaders[0]
    return {
        "selected_beta": chosen["beta"],
        "resolved": True,
        "steps": steps,
        "is_baseline": chosen["beta"] == 0.0,
        "reason": ("beta = {} has the highest mean greedy-success AUC over the "
                   "18 frozen post-onset evaluation points. Development-seed "
                   "calibration only: n = 2 seeds supports no significance or "
                   "generalisation claim.".format(chosen["beta"])),
    }


# ===========================================================================
# main
# ===========================================================================


def analyse(root):
    conditions = []
    for beta in FROZEN_BETAS:
        for seed in FROZEN_SEEDS:
            name = condition_name(beta, seed)
            directory = os.path.join(root, name)
            if not os.path.isdir(directory):
                raise AnalysisError("condition directory missing: {!r}".format(
                    directory))
            auth = authenticate(directory, beta, seed)
            gates = technical_gates(directory, beta, seed)
            points = greedy_points(gates.pop("episodes"))
            selected, missing = select_frozen_points(points)
            problems = list(auth["problems"]) + list(gates["problems"])
            if missing:
                problems.append("missing frozen evaluation points: {}".format(
                    missing))
            conditions.append({
                "name": name, "beta": beta, "seed": seed, "dir": directory,
                "checks": gates["checks"],
                "problems": problems,
                "artifact_hashes": auth["rehashed"],
                "wall_clock_s": auth["manifest"].get("wall_clock_s"),
                "evaluation_points_total": len(points),
                "frozen_points_matched": len(selected),
                "frozen_points": selected,
                "success_auc": success_auc(selected),
                "final5_success": final_window_success(selected),
            })

    all_ok = all(not c["problems"] for c in conditions)

    per_beta = []
    for beta in FROZEN_BETAS:
        group = [c for c in conditions if c["beta"] == beta]
        aucs = [c["success_auc"] for c in group if c["success_auc"] is not None]
        finals = [c["final5_success"] for c in group
                  if c["final5_success"] is not None]
        per_beta.append({
            "beta": beta,
            "seeds": {c["seed"]: {"success_auc": c["success_auc"],
                                  "final5_success": c["final5_success"],
                                  "coverage": c["checks"]["guidance_coverage"],
                                  "detector_events": c["checks"]["detector_events"],
                                  "activations": c["checks"]["activations"],
                                  "cost_usd": c["checks"]["cost_usd"]}
                      for c in group},
            "primary_score": (statistics.mean(aucs)
                              if len(aucs) == len(FROZEN_SEEDS) else None),
            "secondary_score": (statistics.mean(finals)
                                if len(finals) == len(FROZEN_SEEDS) else None),
            "mean_detector_events": statistics.mean(
                [c["checks"]["detector_events"] for c in group]),
        })

    decision = select_beta(per_beta, all_ok)
    total_cost = sum(c["checks"]["cost_usd"] for c in conditions)

    return {
        "kind": "selma_stage_e_beta_recalibration_analysis",
        "development_calibration_only": True,
        "significance_claim": False,
        "n_seeds": len(FROZEN_SEEDS),
        "forbidden_seeds_inspected": [],
        "frozen_evaluation_points": list(FROZEN_EVAL_POINTS),
        "min_guidance_coverage": MIN_GUIDANCE_COVERAGE,
        "conditions": conditions,
        "per_beta": per_beta,
        "all_gates_passed": all_ok,
        "total_cost_usd": total_cost,
        "global_cost_cap_usd": GLOBAL_COST_CAP_USD,
        "within_global_cost_cap": total_cost <= GLOBAL_COST_CAP_USD,
        "selection": decision,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True)
    parser.add_argument("--report", default=None)
    args = parser.parse_args(argv)

    report = analyse(args.root)
    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w") as handle:
            json.dump(report, handle, indent=2, sort_keys=True, default=str)
        print("report: {}".format(args.report))

    print("\n{:<20} {:>9} {:>10} {:>10} {:>9} {:>8}  {}".format(
        "condition", "coverage", "successAUC", "final5", "attempts", "cost$",
        "problems"))
    for c in report["conditions"]:
        print("{:<20} {:>8.1%} {:>10} {:>10} {:>9} {:>8}  {}".format(
            c["name"], c["checks"]["guidance_coverage"],
            "n/a" if c["success_auc"] is None else round(c["success_auc"], 5),
            "n/a" if c["final5_success"] is None else round(c["final5_success"], 5),
            c["checks"]["live_attempts"], round(c["checks"]["cost_usd"], 4),
            "; ".join(c["problems"]) or "none"))

    print("\n{:<8} {:>14} {:>16} {:>18}".format(
        "beta", "primary(AUC)", "secondary(f5)", "mean det. events"))
    for b in report["per_beta"]:
        print("{:<8} {:>14} {:>16} {:>18}".format(
            b["beta"],
            "n/a" if b["primary_score"] is None else round(b["primary_score"], 5),
            "n/a" if b["secondary_score"] is None else round(b["secondary_score"], 5),
            round(b["mean_detector_events"], 1)))

    print("\ntotal cost: ${:.4f} of ${}".format(
        report["total_cost_usd"], GLOBAL_COST_CAP_USD))
    decision = report["selection"]
    print("selection : {}".format(
        "beta = {}".format(decision["selected_beta"]) if decision["resolved"]
        else "UNRESOLVED"))
    print("  {}".format(decision["reason"]))
    return 0 if report["all_gates_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
