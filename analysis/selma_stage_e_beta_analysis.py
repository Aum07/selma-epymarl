#!/usr/bin/env python
"""SELMA Stage E: post-hoc analysis of the four beta-calibration conditions.

Run **only after all four conditions have finished**. It applies the
preregistered selection rule from
``docs/project/selma_stage_e_beta_calibration_2026-08-17.md`` mechanically:
safety exclusions first, then the primary metric, then the parsimony rule, then
the declared tie-breakers. The rule is implemented here exactly as written and
is not adjusted to the results.

Primary metric: area under the greedy-evaluation team-return curve from
``t_env`` 100 000 to 500 000, normalised by the span so it reads as a mean
return.

This is a **one-development-seed hyperparameter pilot**. It cannot establish
statistical significance, and the bootstrap interval it reports describes the
variability of one trajectory, not of the method.

Run::

    PYTHONPATH=src python analysis/selma_stage_e_beta_analysis.py \
        --root analysis/out/stage_e_beta_pilot \
        --report analysis/out/stage_e_beta_pilot/analysis.json
"""

import argparse
import csv
import json
import math
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from selma.pricing import PriceTable  # noqa: E402

PRICE_TABLE = PriceTable("2026-08-17", 0.20, 0.02, 1.25)
EXPECTED_MODEL = "gpt-5.4-nano-2026-03-17"
EXPECTED_SEED = "91"
EXPECTED_T_MAX = 500000

#: Preregistered analysis window for the primary metric.
AUC_FROM = 100000
AUC_TO = 500000

#: Preregistered safety thresholds (§1.6 items 6 and 7).
MAX_SHAPING_VS_SMALLEST_POSITIVE_REWARD = 0.50
P95_SHAPING_VS_MEDIAN_POSITIVE_REWARD = 0.25

BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 20260817       # fixed so the analysis is reproducible

CONDITIONS = ("beta_0p00", "beta_0p05", "beta_0p10", "beta_0p25")
BETAS = (0.0, 0.05, 0.10, 0.25)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def read_csv(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


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
    return value is not None and math.isfinite(value)


def quantiles(values, points=(0.05, 0.25, 0.5, 0.75, 0.95)):
    values = sorted(v for v in values if _finite(v))
    if not values:
        return {}
    out = {}
    for q in points:
        index = min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))
        out["p{:02d}".format(int(q * 100))] = values[index]
    return out


def distribution(values):
    out = {}
    for value in values:
        out[str(value)] = out.get(str(value), 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


# ---------------------------------------------------------------------------
# A. integrity
# ---------------------------------------------------------------------------


def integrity(guidance, decisions, episodes):
    out = {
        "guidance_rows": len(guidance or []),
        "decision_rows": len(decisions or []),
        "episode_rows": len(episodes or []),
        "artifacts_complete": bool(guidance and decisions and episodes),
    }
    if not guidance:
        out["error"] = "no guidance log"
        return out

    out["final_t_env"] = max(_int(r["t_env"]) for r in guidance)
    out["episodes_trained"] = len({r["episode_id"] for r in guidance})
    out["seeds"] = sorted({r["seed"] for r in guidance})
    out["terminal_causes"] = distribution(
        r["terminal_cause"] for r in guidance if r["terminal_cause"])

    non_finite = set()
    reward_mismatch = 0
    for row in guidance:
        for field in ("raw_team_reward", "training_reward", "phi_before",
                      "phi_after_effective", "pbrs_term_unscaled",
                      "shaping_contribution", "beta"):
            if not _finite(_float(row.get(field))):
                non_finite.add(field)
        raw = _float(row["raw_team_reward"])
        train = _float(row["training_reward"])
        shaping = _float(row["shaping_contribution"], 0.0)
        if None not in (raw, train) and abs(train - (raw + shaping)) > 1e-9:
            reward_mismatch += 1
    out["non_finite_fields"] = sorted(non_finite)
    out["reward_reconciliation_errors"] = reward_mismatch

    # telescoping: phi_after_effective(t) must equal phi_before(t+1) in-episode
    residual = 0.0
    by_episode = {}
    for row in guidance:
        by_episode.setdefault(row["episode_id"], []).append(row)
    for rows in by_episode.values():
        rows.sort(key=lambda r: _int(r["episode_step"]))
        for previous, current in zip(rows, rows[1:]):
            residual = max(residual, abs(
                _float(previous["phi_after_effective"], 0.0)
                - _float(current["phi_before"], 0.0)))
    out["max_telescoping_residual"] = residual
    return out


# ---------------------------------------------------------------------------
# B. provider / cache
# ---------------------------------------------------------------------------


def provider_cache(decisions, guidance):
    considered = [r for r in (decisions or []) if r["decision"] == "multiple"]
    called = [r for r in considered if _int(r["live_attempts"]) > 0]
    digests = [r["key_digest"] for r in considered if r["key_digest"]]
    tin = sum(_int(r["tokens_input"]) for r in called)
    tcached = sum(_int(r["tokens_cached_input"]) for r in called)
    tout = sum(_int(r["tokens_output"]) for r in called)
    latencies = sorted(v for v in (_float(r["latency_s"]) for r in called)
                       if _finite(v))
    activated = [r for r in (decisions or []) if _int(r["activated"])]

    return {
        "detector_events": sum(_int(r.get("detector_events")) for r in (guidance or [])),
        "decisions": len(decisions or []),
        "multiple_plan_decisions": len(considered),
        "zero_plan_decisions": sum(1 for r in (decisions or [])
                                   if r["decision"] == "zero_plan"),
        "singleton_decisions": sum(1 for r in (decisions or [])
                                   if r["decision"] == "singleton"),
        "distinct_semantic_keys": len(set(digests)),
        "cache_hits": sum(1 for r in considered if r["cache_result"] == "hit"),
        "cache_misses": sum(1 for r in considered if r["cache_result"] == "miss"),
        "cache_writes": sum(1 for r in considered
                            if r["cache_result"] == "miss" and _int(r["activated"])),
        "live_attempts": sum(_int(r["live_attempts"]) for r in considered),
        "retries": sum(_int(r["retries"]) for r in considered),
        "budget_exhausted": sum(1 for r in considered
                                if r["outcome"] == "budget_exhausted"),
        "activations": len(activated),
        "parse_and_ground_rate": (
            sum(1 for r in called if _int(r["validator_valid"])) / float(len(called))
            if called else None),
        "validator_rejections": sum(
            1 for r in called if not _int(r["validator_valid"])),
        "physically_valid_activations": sum(
            1 for r in activated if _int(r["physically_valid"])),
        "physically_invalid_activations": sum(
            1 for r in activated if not _int(r["physically_valid"])),
        "returned_models": distribution(
            r["returned_model"] for r in called if r["returned_model"]),
        "greedy_eval_decisions": sum(1 for r in (decisions or [])
                                     if _int(r["greedy_eval"])),
        "input_tokens": tin,
        "cached_input_tokens": tcached,
        "output_tokens": tout,
        "cost_usd": PRICE_TABLE.cost_usd(input_tokens=tin,
                                         cached_input_tokens=tcached,
                                         output_tokens=tout),
        "latency_s": {
            "n": len(latencies),
            "median": round(statistics.median(latencies), 4) if latencies else None,
            "p95": round(latencies[min(len(latencies) - 1,
                                       int(0.95 * len(latencies)))], 4)
                   if latencies else None,
            "max": round(latencies[-1], 4) if latencies else None,
        },
    }


# ---------------------------------------------------------------------------
# C. shaping magnitude
# ---------------------------------------------------------------------------


def shaping(guidance):
    unscaled = [_float(r["pbrs_term_unscaled"], 0.0) for r in guidance]
    contributions = [_float(r["shaping_contribution"], 0.0) for r in guidance]
    non_zero_abs = [abs(v) for v in contributions if v != 0.0]
    raw_positive = [v for v in (_float(r["raw_team_reward"], 0.0) for r in guidance)
                    if v > 0.0]
    ttl = [_int(r["transitions_served"]) for r in guidance
           if r["transitions_served"] not in ("", None)]
    terminals = [r for r in guidance if _int(r["stored_terminated"])]

    smallest_positive = min(raw_positive) if raw_positive else None
    median_positive = statistics.median(raw_positive) if raw_positive else None
    max_abs = max((abs(v) for v in contributions), default=0.0)
    p95_abs = quantiles(non_zero_abs).get("p95") if non_zero_abs else 0.0

    return {
        "non_zero_unscaled_terms": sum(1 for v in unscaled if v != 0.0),
        "unscaled_min": min(unscaled, default=0.0),
        "unscaled_max": max(unscaled, default=0.0),
        "unscaled_quantiles": quantiles([v for v in unscaled if v != 0.0]),
        "shaping_min": min(contributions, default=0.0),
        "shaping_max": max(contributions, default=0.0),
        "shaping_quantiles_nonzero_abs": quantiles(non_zero_abs),
        "total_signed_shaping": sum(contributions),
        "total_absolute_shaping": sum(abs(v) for v in contributions),
        "max_abs_shaping": max_abs,
        "p95_abs_shaping_nonzero": p95_abs,
        "positive_raw_rewards": len(raw_positive),
        "smallest_positive_raw_reward": smallest_positive,
        "median_positive_raw_reward": median_positive,
        "max_shaping_over_smallest_positive_reward": (
            max_abs / smallest_positive if smallest_positive else None),
        "p95_shaping_over_median_positive_reward": (
            (p95_abs or 0.0) / median_positive if median_positive else None),
        "stored_terminal_transitions": len(terminals),
        "terminal_phi_all_zero": all(
            _float(r["phi_after_effective"], 0.0) == 0.0 for r in terminals),
        "ttl_max": max(ttl, default=0),
        "ttl_distribution": distribution(ttl),
    }


# ---------------------------------------------------------------------------
# D. learning (greedy evaluation only)
# ---------------------------------------------------------------------------


def greedy_curve(episodes):
    """Mean greedy team return per evaluation point, ordered by t_env."""
    points = {}
    for row in episodes or []:
        if _int(row.get("test_mode")) != 1:
            continue
        t_env = _int(row.get("t_env"))
        points.setdefault(t_env, []).append(row)
    curve = []
    for t_env in sorted(points):
        rows = points[t_env]
        returns = [_float(r["team_return"], 0.0) for r in rows]
        successes = [1.0 if _float(r["team_return"], 0.0) > 0 else 0.0 for r in rows]
        curve.append({
            "t_env": t_env,
            "episodes": len(rows),
            "mean_return": statistics.mean(returns) if returns else 0.0,
            "success_rate": statistics.mean(successes) if successes else 0.0,
            "mean_cooperative_loads": statistics.mean(
                [_int(r.get("necessary_cooperative_load_events")) for r in rows]),
            "mean_successful_loads": statistics.mean(
                [_int(r.get("successful_load_events")) for r in rows]),
            "mean_failed_load_attempts": statistics.mean(
                [_int(r.get("failed_load_attempts")) for r in rows]),
        })
    return curve


def auc(curve, key="mean_return", low=AUC_FROM, high=AUC_TO):
    """Trapezoid AUC over the window, normalised by span -> reads as a mean."""
    points = [(p["t_env"], p[key]) for p in curve if low <= p["t_env"] <= high]
    if len(points) < 2:
        return None
    total = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        total += (x1 - x0) * (y0 + y1) / 2.0
    span = points[-1][0] - points[0][0]
    return total / span if span else None


def bootstrap_auc(curve, *, resamples=BOOTSTRAP_RESAMPLES, low=AUC_FROM, high=AUC_TO):
    """Percentile interval by resampling evaluation points with replacement.

    Describes the variability of THIS trajectory only. Evaluation points from one
    continuously-updating policy are not independent samples of the method, so
    this interval must never be read as a significance test.
    """
    window = [p for p in curve if low <= p["t_env"] <= high]
    if len(window) < 3:
        return None
    rng = random.Random(BOOTSTRAP_SEED)
    values = [p["mean_return"] for p in window]
    means = []
    n = len(values)
    for _ in range(resamples):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / float(n))
    means.sort()
    return {
        "point_mean_of_window": statistics.mean(values),
        "ci95_low": means[int(0.025 * resamples)],
        "ci95_high": means[int(0.975 * resamples) - 1],
        "evaluation_points": n,
        "resamples": resamples,
        "caveat": "resampled evaluation points from ONE trajectory; not a "
                  "between-seed interval and not a significance test",
    }


def final_window_mean(curve, span=100000, key="mean_return"):
    if not curve:
        return None
    high = max(p["t_env"] for p in curve)
    window = [p[key] for p in curve if p["t_env"] > high - span]
    return statistics.mean(window) if window else None


def resolution_within(guidance, horizon=10):
    """Fraction of detector events followed by a cooperative load within N steps.

    Uses the per-transition load columns in the guidance log, scanning forward
    within the same episode only.
    """
    by_episode = {}
    for row in guidance or []:
        by_episode.setdefault(row["episode_id"], []).append(row)
    events = resolved = 0
    for rows in by_episode.values():
        rows.sort(key=lambda r: _int(r["episode_step"]))
        coop = [_int(r.get("step_cooperative_loads")) for r in rows]
        for index, row in enumerate(rows):
            if _int(row.get("detector_events")) <= 0:
                continue
            events += 1
            if any(coop[index + 1:index + 1 + horizon]):
                resolved += 1
    return {
        "detector_events_scanned": events,
        "resolved_within_{}".format(horizon): resolved,
        "resolution_rate": (resolved / float(events)) if events else None,
    }


def learner_stability(sacred_metrics):
    """Loss / Q / grad-norm ranges from the Sacred metrics file, if present."""
    out = {}
    for name in ("loss", "q_taken_mean", "grad_norm", "target_mean", "td_error_abs"):
        series = (sacred_metrics or {}).get(name, {}).get("values")
        if not series:
            continue
        values = [v for v in series if _finite(v)]
        if not values:
            out[name] = {"non_finite": True}
            continue
        out[name] = {
            "n": len(values), "min": min(values), "max": max(values),
            "final": values[-1], "max_abs": max(abs(v) for v in values),
            "any_non_finite": len(values) != len(series),
        }
    return out


def load_sacred_metrics(results_path, *, sacred_root=None, beta=None):
    """Find this condition's Sacred ``metrics.json``.

    ``src/main.py`` builds its FileStorageObserver path from the repository
    root, not from ``local_results_path``, so a condition's Sacred run does not
    live under its own output directory. Both locations are searched, and when
    a shared root is used the run is identified by matching ``selma_beta`` in
    its ``config.json`` -- never by "most recent", which would silently attach
    the wrong run to a condition.
    """
    for root, _dirs, files in os.walk(results_path or ""):
        if "metrics.json" in files:
            try:
                with open(os.path.join(root, "metrics.json")) as handle:
                    return json.load(handle)
            except (OSError, ValueError):
                pass

    if not sacred_root or beta is None:
        return None
    for root, _dirs, files in os.walk(sacred_root):
        if "metrics.json" not in files or "config.json" not in files:
            continue
        try:
            with open(os.path.join(root, "config.json")) as handle:
                config = json.load(handle)
        except (OSError, ValueError):
            continue
        if float(config.get("selma_beta", -1)) != float(beta):
            continue
        if int(config.get("seed", -1)) != int(EXPECTED_SEED):
            continue
        if int(config.get("t_max", -1)) != EXPECTED_T_MAX:
            continue
        try:
            with open(os.path.join(root, "metrics.json")) as handle:
                return json.load(handle)
        except (OSError, ValueError):
            continue
    return None


# ---------------------------------------------------------------------------
# safety exclusions (preregistered §1.6)
# ---------------------------------------------------------------------------


def safety_exclusions(beta, integ, prov, shp, learn, *, baseline_learn=None):
    """Return the list of preregistered exclusion reasons that fired."""
    reasons = []
    if beta == 0.0:
        # The baseline is the reference, not a candidate; it is still checked
        # for integrity so a broken baseline is visible.
        pass

    if integ.get("non_finite_fields"):
        reasons.append("1: non-finite values in {}".format(
            integ["non_finite_fields"]))
    if integ.get("reward_reconciliation_errors", 0) > 0:
        reasons.append("2: {} reward-reconciliation failures".format(
            integ["reward_reconciliation_errors"]))
    if integ.get("max_telescoping_residual", 0.0) > 1e-9:
        reasons.append("3: telescoping residual {:.3g}".format(
            integ["max_telescoping_residual"]))
    if prov.get("physically_invalid_activations", 0) > 0:
        reasons.append("4: {} physically invalid activations".format(
            prov["physically_invalid_activations"]))
    if prov.get("greedy_eval_decisions", 0) > 0:
        reasons.append("5: provider activity during greedy evaluation")
    if set(prov.get("returned_models", {})) - {EXPECTED_MODEL}:
        reasons.append("5: model substitution {}".format(
            list(prov["returned_models"])))

    ratio_max = shp.get("max_shaping_over_smallest_positive_reward")
    if ratio_max is not None and ratio_max > MAX_SHAPING_VS_SMALLEST_POSITIVE_REWARD:
        reasons.append(
            "6: max |shaping| is {:.1%} of the smallest positive reward "
            "(limit {:.0%})".format(ratio_max,
                                    MAX_SHAPING_VS_SMALLEST_POSITIVE_REWARD))
    ratio_p95 = shp.get("p95_shaping_over_median_positive_reward")
    if ratio_p95 is not None and ratio_p95 > P95_SHAPING_VS_MEDIAN_POSITIVE_REWARD:
        reasons.append(
            "7: P95 |shaping| is {:.1%} of the median positive reward "
            "(limit {:.0%})".format(ratio_p95,
                                    P95_SHAPING_VS_MEDIAN_POSITIVE_REWARD))

    if baseline_learn:
        for name, stats in (learn or {}).items():
            base = baseline_learn.get(name)
            if not base or stats.get("any_non_finite"):
                if stats.get("any_non_finite") or stats.get("non_finite"):
                    reasons.append("8: non-finite {}".format(name))
                continue
            if base.get("max_abs") and stats.get("max_abs", 0) > 10 * base["max_abs"]:
                reasons.append(
                    "8: {} max |value| {:.3g} exceeds 10x the beta=0 baseline "
                    "{:.3g}".format(name, stats["max_abs"], base["max_abs"]))

    if prov.get("budget_exhausted", 0) > 0:
        reasons.append("9: {} events hit the cap (guidance exposure is unequal)"
                       .format(prov["budget_exhausted"]))
    return reasons


# ---------------------------------------------------------------------------
# selection (preregistered §1.8)
# ---------------------------------------------------------------------------


def select_beta(conditions):
    """Apply the preregistered rule mechanically. Returns the decision record."""
    steps = []
    baseline = next((c for c in conditions if c["beta"] == 0.0), None)
    positives = [c for c in conditions if c["beta"] > 0.0]

    safe = [c for c in positives if not c["safety_exclusions"]]
    excluded = [{"beta": c["beta"], "reasons": c["safety_exclusions"]}
                for c in positives if c["safety_exclusions"]]
    steps.append({"step": 1, "action": "exclude unsafe positive betas",
                  "excluded": excluded,
                  "remaining": [c["beta"] for c in safe]})

    if not safe:
        steps.append({"step": "terminal",
                      "action": "no positive beta survived the safety exclusions"})
        return {"selected_beta": None, "resolved": False, "steps": steps,
                "reason": "No positive beta was safe under the preregistered "
                          "exclusions, so no positive selection is forced (rule 5). "
                          "beta remains UNRESOLVED."}

    scored = [c for c in safe if c["primary_auc"] is not None]
    if not scored:
        steps.append({"step": "terminal",
                      "action": "no safe beta had a computable primary metric"})
        return {"selected_beta": None, "resolved": False, "steps": steps,
                "reason": "The primary metric could not be computed. beta remains "
                          "UNRESOLVED."}

    best = max(scored, key=lambda c: c["primary_auc"])
    steps.append({"step": 2, "action": "best primary metric among safe positives",
                  "beta": best["beta"], "primary_auc": best["primary_auc"]})

    # rule 5: directional competitiveness against beta = 0
    if baseline is not None and baseline["primary_auc"] is not None:
        if best["primary_auc"] < baseline["primary_auc"]:
            steps.append({
                "step": 5, "action": "best safe positive beta is below beta=0",
                "best_positive_auc": best["primary_auc"],
                "baseline_auc": baseline["primary_auc"]})
            return {"selected_beta": None, "resolved": False, "steps": steps,
                    "reason": "No positive beta was directionally competitive with "
                              "beta=0 on the primary metric, so no positive "
                              "selection is forced (rule 5). beta remains "
                              "UNRESOLVED."}

    # rule 3: parsimony under overlapping intervals
    chosen = best
    smaller = [c for c in scored if c["beta"] < best["beta"]]
    overlapping = []
    for candidate in sorted(smaller, key=lambda c: c["beta"]):
        if _intervals_overlap(candidate.get("bootstrap"), best.get("bootstrap")):
            overlapping.append(candidate["beta"])
    if overlapping:
        chosen = min((c for c in scored if c["beta"] in overlapping),
                     key=lambda c: c["beta"])
        steps.append({
            "step": 3,
            "action": "bootstrap intervals overlap with a smaller safe beta; "
                      "select the smaller (parsimony under uncertainty)",
            "overlapping_smaller_betas": overlapping,
            "selected": chosen["beta"]})
    else:
        steps.append({"step": 3, "action": "no smaller safe beta overlaps",
                      "selected": chosen["beta"]})

    return {"selected_beta": chosen["beta"], "resolved": True, "steps": steps,
            "reason": "Provisional development-seed selection. Not a "
                      "significance claim and not a final value."}


def _intervals_overlap(a, b):
    if not a or not b:
        return False
    return a["ci95_low"] <= b["ci95_high"] and b["ci95_low"] <= a["ci95_high"]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def analyse_condition(name, beta, directory, sacred_root=None):
    guidance = read_csv(os.path.join(directory, "guidance.csv")) or []
    decisions = read_csv(os.path.join(directory, "decisions.csv")) or []
    episodes = read_csv(os.path.join(directory, "episodes.csv")) or []
    metrics = load_sacred_metrics(os.path.join(directory, "results"),
                                  sacred_root=sacred_root, beta=beta)

    integ = integrity(guidance, decisions, episodes)
    prov = provider_cache(decisions, guidance)
    shp = shaping(guidance)
    curve = greedy_curve(episodes)
    learn = learner_stability(metrics)

    return {
        "name": name,
        "beta": beta,
        "dir": directory,
        "integrity": integ,
        "provider_cache": prov,
        "shaping": shp,
        "greedy_curve": curve,
        "primary_auc": auc(curve),
        "bootstrap": bootstrap_auc(curve),
        "final_100k_mean_return": final_window_mean(curve),
        "final_100k_success_rate": final_window_mean(curve, key="success_rate"),
        "final_100k_cooperative_loads": final_window_mean(
            curve, key="mean_cooperative_loads"),
        "failed_load_resolution": resolution_within(guidance),
        "learner_stability": learn,
    }


def comparability(conditions):
    return {
        "t_env_final": {c["name"]: c["integrity"].get("final_t_env")
                        for c in conditions},
        "episodes_trained": {c["name"]: c["integrity"].get("episodes_trained")
                             for c in conditions},
        "detector_events": {c["name"]: c["provider_cache"]["detector_events"]
                            for c in conditions},
        "multiple_plan_decisions": {
            c["name"]: c["provider_cache"]["multiple_plan_decisions"]
            for c in conditions},
        "live_attempts": {c["name"]: c["provider_cache"]["live_attempts"]
                          for c in conditions},
        "activations": {c["name"]: c["provider_cache"]["activations"]
                        for c in conditions},
        "budget_exhausted": {c["name"]: c["provider_cache"]["budget_exhausted"]
                             for c in conditions},
        "distinct_keys": {c["name"]: c["provider_cache"]["distinct_semantic_keys"]
                          for c in conditions},
        "cache_hits": {c["name"]: c["provider_cache"]["cache_hits"]
                       for c in conditions},
        "cost_usd": {c["name"]: c["provider_cache"]["cost_usd"]
                     for c in conditions},
        "evaluation_points": {c["name"]: len(c["greedy_curve"])
                              for c in conditions},
        "any_cap_exhaustion": any(c["provider_cache"]["budget_exhausted"] > 0
                                  for c in conditions),
        "artifacts_complete": all(c["integrity"].get("artifacts_complete")
                                  for c in conditions),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True)
    parser.add_argument("--report", default=None)
    parser.add_argument("--sacred-root", default=None,
                        help="shared Sacred results tree; runs are matched to "
                             "conditions by selma_beta, seed and t_max")
    args = parser.parse_args(argv)

    conditions = []
    for name, beta in zip(CONDITIONS, BETAS):
        conditions.append(analyse_condition(name, beta,
                                            os.path.join(args.root, name),
                                            sacred_root=args.sacred_root))

    baseline_learn = next((c["learner_stability"] for c in conditions
                           if c["beta"] == 0.0), None)
    for condition in conditions:
        condition["safety_exclusions"] = safety_exclusions(
            condition["beta"], condition["integrity"], condition["provider_cache"],
            condition["shaping"], condition["learner_stability"],
            baseline_learn=baseline_learn)

    decision = select_beta(conditions)

    report = {
        "kind": "selma_stage_e_beta_calibration_analysis",
        "one_development_seed_pilot": True,
        "not_a_significance_claim": True,
        "not_a_final_performance_result": True,
        "seed": EXPECTED_SEED,
        "primary_metric": "greedy-evaluation team-return AUC, t_env {}-{}, "
                          "normalised by span".format(AUC_FROM, AUC_TO),
        "preregistered_thresholds": {
            "max_shaping_over_smallest_positive_reward":
                MAX_SHAPING_VS_SMALLEST_POSITIVE_REWARD,
            "p95_shaping_over_median_positive_reward":
                P95_SHAPING_VS_MEDIAN_POSITIVE_REWARD,
        },
        "conditions": conditions,
        "comparability": comparability(conditions),
        "selection": decision,
    }
    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w") as handle:
            json.dump(report, handle, indent=2, sort_keys=True, default=str)
        print("report: {}".format(args.report))

    print("\n{:<12} {:>10} {:>12} {:>10} {:>10}  {}".format(
        "condition", "t_env", "primary AUC", "attempts", "cost$", "exclusions"))
    for c in conditions:
        print("{:<12} {:>10} {:>12} {:>10} {:>10}  {}".format(
            c["name"], c["integrity"].get("final_t_env"),
            "n/a" if c["primary_auc"] is None else round(c["primary_auc"], 4),
            c["provider_cache"]["live_attempts"],
            round(c["provider_cache"]["cost_usd"], 5),
            "; ".join(c["safety_exclusions"]) or "none"))
    print("\nselection: {}".format(
        "beta = {}".format(decision["selected_beta"]) if decision["resolved"]
        else "UNRESOLVED"))
    print("  {}".format(decision["reason"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
