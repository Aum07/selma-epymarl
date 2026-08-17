#!/usr/bin/env python
"""SELMA Stage E: mechanically derive the revised beta-calibration design.

**Read-only, offline, no training.** This script opens local artifacts, computes
a horizon and a call/cost budget from preregistered rules, and writes a design
report. It makes no network call, constructs no provider, reads no credential
and launches no process.

What it decides, and how
------------------------

1. **Authenticate the baseline.** The seed-91 2M development baseline must be
   identifiable by exact configuration and an accepted completion record.
   Anything partial, superseded or mismatched is rejected loudly rather than
   silently substituted.
2. **Select the horizon** by the rule preregistered in
   ``docs/project/selma_stage_e_beta_recalibration_design_2026-08-17.md`` §3 --
   fixed *before* the baseline learning values were read.
3. **Size the call budget** from the Stage E pilot's observed per-block demand,
   distinguishing detector events from decisions from *calls* (a repeated
   semantic key costs nothing under a perfect cache).
4. **Emit the frozen six-condition matrix**: 3 betas x 2 development seeds.

Run::

    PYTHONPATH=src python analysis/selma_stage_e_recalibration_design.py \
        --pilot-root analysis/out/stage_e_beta_pilot \
        --baseline-csv analysis/out/seed91_retry_.../episodes_..._retry.csv \
        --baseline-sacred results/sacred/qmix/<env>/16 \
        --report analysis/out/stage_e_recalibration/design.json
"""

import argparse
import csv
import json
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from selma.pricing import PriceTable  # noqa: E402

# -- frozen identity the baseline must match --------------------------------
EXPECTED_SEED = 91
EXPECTED_ENV = "lbforaging:Foraging-8x8-2p-2f-coop-v3"
EXPECTED_ALGORITHM = "qmix"
EXPECTED_BASELINE_T_MAX = 2000000
#: A baseline is accepted only if its own run record says it finished.
ACCEPTED_STATUSES = ("COMPLETED",)

# -- preregistered horizon rule (§3), fixed before reading learning values ---
CANDIDATE_HORIZONS = (1000000, 1500000, 2000000)
NONFLOOR_SUCCESS_THRESHOLD = 0.05      # 5 % greedy success
NONFLOOR_CONSECUTIVE_POINTS = 5
#: The revised horizon must leave this much training after onset.
POST_ONSET_MARGIN = 500000
#: ...and provide at least this many scheduled greedy evaluations.
MIN_EVALUATIONS_IN_WINDOW = 8

# -- preregistered budget rule (§4) -----------------------------------------
BLOCK = 100000
SAFETY_MARGIN = 0.25                   # +25 %
CAP_ROUNDING = 500                     # round up to the next 500 calls
MIN_CALL_CAP = 3000                    # never lower, whatever the projection

#: Frozen 2026-08-17 table. Repeated here only as an input to the ceiling
#: arithmetic; the authoritative statement lives in the design document.
PRICE_TABLE = PriceTable("2026-08-17", 0.20, 0.02, 1.25)

# -- frozen revised grid (§5) -----------------------------------------------
REVISED_BETAS = (0.00, 0.05, 0.10)
REVISED_DEV_SEEDS = (91, 92)
#: Detector held-out evidence. Never reusable for beta selection.
DETECTOR_HELDOUT_SEED = 99
#: Coverage floor every revised condition must clear (§6).
MIN_GUIDANCE_COVERAGE = 0.95

PILOT_CONDITIONS = ("beta_0p00", "beta_0p05", "beta_0p10", "beta_0p25")
PILOT_BETAS = (0.00, 0.05, 0.10, 0.25)
PILOT_T_MAX = 500000


class EvidenceError(RuntimeError):
    """Missing, malformed or unauthenticated evidence. Nothing is guessed."""


# ---------------------------------------------------------------------------
# helpers
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


def _round_up(value, multiple):
    return int(math.ceil(float(value) / multiple) * multiple)


# ---------------------------------------------------------------------------
# 1. baseline authentication
# ---------------------------------------------------------------------------


def authenticate_baseline(csv_path, sacred_dir):
    """Accept the seed-91 2M baseline only on exact, checkable identity.

    Rejects loudly when configuration, seed, environment, horizon or completion
    status do not match. A partial or superseded run is never silently used --
    that is precisely how a wrong learning curve would enter the horizon rule.
    """
    if not csv_path or not os.path.exists(csv_path):
        raise EvidenceError(
            "baseline episode CSV not found at {!r}".format(csv_path))
    if not sacred_dir or not os.path.isdir(sacred_dir):
        raise EvidenceError(
            "baseline Sacred run directory not found at {!r}".format(sacred_dir))

    config_path = os.path.join(sacred_dir, "config.json")
    run_path = os.path.join(sacred_dir, "run.json")
    for path in (config_path, run_path):
        if not os.path.exists(path):
            raise EvidenceError("missing {} in {!r}".format(
                os.path.basename(path), sacred_dir))
    with open(config_path) as handle:
        config = json.load(handle)
    with open(run_path) as handle:
        run = json.load(handle)

    problems = []
    if int(config.get("seed", -1)) != EXPECTED_SEED:
        problems.append("seed is {!r}, expected {}".format(
            config.get("seed"), EXPECTED_SEED))
    env_key = (config.get("env_args") or {}).get("key")
    if env_key != EXPECTED_ENV:
        problems.append("env is {!r}, expected {!r}".format(env_key, EXPECTED_ENV))
    if int(config.get("t_max", 0)) < EXPECTED_BASELINE_T_MAX:
        problems.append("t_max is {!r}, expected >= {}".format(
            config.get("t_max"), EXPECTED_BASELINE_T_MAX))
    if str(config.get("name", EXPECTED_ALGORITHM)).lower() != EXPECTED_ALGORITHM:
        problems.append("algorithm is {!r}, expected {!r}".format(
            config.get("name"), EXPECTED_ALGORITHM))
    status = run.get("status")
    if status not in ACCEPTED_STATUSES:
        problems.append(
            "run status is {!r}; only {} is accepted. A RUNNING or superseded "
            "run is never used as baseline evidence.".format(
                status, "/".join(ACCEPTED_STATUSES)))
    # the Sacred run must actually be the one that wrote this CSV
    logged = os.path.basename(str(config.get("selma_log_csv") or ""))
    if logged and logged != os.path.basename(csv_path):
        problems.append(
            "this Sacred run logged episodes to {!r}, not {!r}".format(
                logged, os.path.basename(csv_path)))
    if problems:
        raise EvidenceError(
            "baseline evidence failed authentication:\n  - " + "\n  - ".join(problems))

    rows = read_csv(csv_path)
    if not rows:
        raise EvidenceError("baseline episode CSV {!r} is empty".format(csv_path))
    greedy = [r for r in rows if _int(r.get("test_mode")) == 1]
    if not greedy:
        raise EvidenceError(
            "baseline CSV contains no greedy (test_mode=1) rows, so no "
            "evaluation curve can be built from it")
    points = sorted({_int(r["t_env"]) for r in greedy})
    episode_level = ("team_return" in rows[0]
                     and "successful_load_events" in rows[0])

    return {
        "csv_path": csv_path,
        "sacred_dir": sacred_dir,
        "sacred_run_id": os.path.basename(sacred_dir.rstrip("/")),
        "status": status,
        "seed": int(config.get("seed")),
        "env": env_key,
        "algorithm": str(config.get("name")),
        "t_max": int(config.get("t_max")),
        "selma_mode": config.get("selma_mode"),
        "test_interval": _int(config.get("test_interval")),
        "test_nepisode": _int(config.get("test_nepisode")),
        "total_rows": len(rows),
        "greedy_rows": len(greedy),
        "evaluation_points": len(points),
        "first_eval_t_env": points[0],
        "final_eval_t_env": points[-1],
        "episode_level_success_available": bool(episode_level),
        "authenticated": True,
    }


# ---------------------------------------------------------------------------
# 2. the greedy curve and the preregistered non-floor rule
# ---------------------------------------------------------------------------


def baseline_curve(csv_path):
    rows = read_csv(csv_path) or []
    points = {}
    for row in rows:
        if _int(row.get("test_mode")) != 1:
            continue
        points.setdefault(_int(row["t_env"]), []).append(row)
    curve = []
    for t_env in sorted(points):
        group = points[t_env]
        returns = [_float(r["team_return"], 0.0) for r in group]
        successes = [1.0 if _float(r["team_return"], 0.0) > 0 else 0.0
                     for r in group]
        curve.append({
            "t_env": t_env,
            "episodes": len(group),
            "mean_return": statistics.mean(returns) if returns else 0.0,
            "success_rate": statistics.mean(successes) if successes else 0.0,
        })
    return curve


def nonfloor_onset(curve, *, episode_level, floor_return=None):
    """First sustained non-floor region, by the preregistered rule (§3).

    With episode-level data: the first evaluation point beginning a run of
    :data:`NONFLOOR_CONSECUTIVE_POINTS` consecutive points at or above
    :data:`NONFLOOR_SUCCESS_THRESHOLD` greedy success.

    Without it: the first point beginning an equally long run of strictly
    positive mean return that also stays above the supplied ``beta=0`` 500k
    floor. That fallback is weaker and is reported as such.
    """
    if not curve:
        return {"onset_t_env": None, "criterion": "no curve", "sustained": False}

    if episode_level:
        criterion = ("greedy success >= {:.0%} for {} consecutive evaluation "
                     "points".format(NONFLOOR_SUCCESS_THRESHOLD,
                                     NONFLOOR_CONSECUTIVE_POINTS))
        ok = [p["success_rate"] >= NONFLOOR_SUCCESS_THRESHOLD for p in curve]
    else:
        if floor_return is None:
            raise EvidenceError(
                "the fallback non-floor rule needs the beta=0 500k floor return")
        criterion = ("mean return > 0 and > the beta=0 500k floor ({:.6g}) for "
                     "{} consecutive evaluation points".format(
                         floor_return, NONFLOOR_CONSECUTIVE_POINTS))
        ok = [p["mean_return"] > 0.0 and p["mean_return"] > floor_return
              for p in curve]

    for index in range(len(curve) - NONFLOOR_CONSECUTIVE_POINTS + 1):
        window = ok[index:index + NONFLOOR_CONSECUTIVE_POINTS]
        if all(window):
            return {
                "onset_t_env": curve[index]["t_env"],
                "onset_index": index,
                "criterion": criterion,
                "sustained": True,
                "episode_level": bool(episode_level),
                "window_success_rates": [
                    round(p["success_rate"], 4)
                    for p in curve[index:index + NONFLOOR_CONSECUTIVE_POINTS]],
                "window_mean_returns": [
                    round(p["mean_return"], 6)
                    for p in curve[index:index + NONFLOOR_CONSECUTIVE_POINTS]],
            }
    return {"onset_t_env": None, "criterion": criterion, "sustained": False,
            "episode_level": bool(episode_level)}


def select_horizon(onset, *, test_interval, candidates=CANDIDATE_HORIZONS):
    """Smallest candidate horizon satisfying the preregistered constraints.

    Falls back to the largest candidate -- deliberately conservative -- when
    onset cannot be established, rather than picking the horizon that happens to
    flatter any particular beta.
    """
    reasons = []
    onset_t = onset.get("onset_t_env")
    if onset_t is None:
        return {
            "horizon": max(candidates),
            "rule": "conservative fallback",
            "reason": "sustained non-floor onset could not be established from "
                      "the accepted baseline, so the largest candidate horizon "
                      "is selected conservatively",
            "candidates_considered": list(candidates),
        }

    required = onset_t + POST_ONSET_MARGIN
    for candidate in sorted(candidates):
        evaluations = 0
        if test_interval:
            evaluations = max(0, (candidate - onset_t) // test_interval)
        if candidate < required:
            reasons.append({"candidate": candidate, "rejected":
                            "only {} steps after onset; {} required".format(
                                candidate - onset_t, POST_ONSET_MARGIN)})
            continue
        if evaluations < MIN_EVALUATIONS_IN_WINDOW:
            reasons.append({"candidate": candidate, "rejected":
                            "only {} scheduled evaluations after onset; {} "
                            "required".format(evaluations,
                                              MIN_EVALUATIONS_IN_WINDOW)})
            continue
        return {
            "horizon": candidate,
            "rule": "smallest candidate at least {} steps after sustained "
                    "non-floor onset with at least {} evaluations".format(
                        POST_ONSET_MARGIN, MIN_EVALUATIONS_IN_WINDOW),
            "onset_t_env": onset_t,
            "required_at_least": required,
            "evaluations_after_onset": evaluations,
            "rejected_candidates": reasons,
            "candidates_considered": list(candidates),
        }

    return {
        "horizon": max(candidates),
        "rule": "conservative fallback",
        "reason": "no candidate satisfied the rule; the largest is selected",
        "rejected_candidates": reasons,
        "candidates_considered": list(candidates),
    }


# ---------------------------------------------------------------------------
# 3. per-block call demand from the completed pilot
# ---------------------------------------------------------------------------


def block_demand(decisions, guidance, *, block=BLOCK, t_max=PILOT_T_MAX):
    """Per-block funnel for one pilot condition.

    Keeps the three quantities that are routinely conflated strictly apart:

    * **detector events** -- what the detector produced;
    * **LLM-eligible decisions** -- events that reached the cache/provider path;
    * **calls** -- decisions that actually needed a live request. A repeat of an
      already-seen semantic key needs none, so ``calls_for_full_coverage`` is
      the count of *distinct* keys, not of decisions.
    """
    blocks = []
    seen = set()
    for low in range(0, t_max, block):
        high = low + block
        rows = [r for r in decisions if low <= _int(r["t_env"]) < high]
        grows = [r for r in guidance if low <= _int(r["t_env"]) < high]
        considered = [r for r in rows if r["decision"] == "multiple"]
        digests = [r["key_digest"] for r in considered if r["key_digest"]]

        new_keys = 0
        repeats = 0
        for digest in digests:
            if digest in seen:
                repeats += 1
            else:
                seen.add(digest)
                new_keys += 1

        attempts = sum(_int(r["live_attempts"]) for r in considered)
        exhausted = sum(1 for r in considered if r["outcome"] == "budget_exhausted")
        activations = sum(_int(r["activated"]) for r in rows)
        blocks.append({
            "block": "{}-{}k".format(low // 1000, high // 1000),
            "t_env_from": low,
            "t_env_to": high,
            "detector_events": sum(_int(r.get("detector_events")) for r in grows),
            "llm_eligible_decisions": len(considered),
            "semantic_key_digests": len(digests),
            "distinct_new_keys": new_keys,
            "repeated_keys": repeats,
            "ideal_cache_hits": repeats,
            "calls_for_full_coverage": new_keys,
            "attempts_made": attempts,
            "budget_exhausted": exhausted,
            "activations": activations,
            "guidance_coverage": (
                (len(considered) - exhausted) / float(len(considered))
                if considered else None),
            "new_keys_per_step": new_keys / float(block),
            "cumulative_distinct_keys": len(seen),
        })
    return blocks


def condition_demand(name, beta, directory):
    decisions = read_csv(os.path.join(directory, "decisions.csv"))
    guidance = read_csv(os.path.join(directory, "guidance.csv"))
    if decisions is None or guidance is None:
        raise EvidenceError(
            "pilot condition {!r} is missing its decision or guidance log in "
            "{!r}".format(name, directory))

    considered = [r for r in decisions if r["decision"] == "multiple"]
    digests = [r["key_digest"] for r in considered if r["key_digest"]]
    distinct = len(set(digests))
    exhausted = sum(1 for r in considered if r["outcome"] == "budget_exhausted")
    called = [r for r in considered if _int(r["live_attempts"]) > 0]

    tokens_in = [_int(r["tokens_input"]) for r in called
                 if r["tokens_input"] not in ("", "n/a")]
    tokens_out = [_int(r["tokens_output"]) for r in called
                  if r["tokens_output"] not in ("", "n/a")]

    blocks = block_demand(decisions, guidance)
    return {
        "name": name,
        "beta": beta,
        "detector_events": sum(_int(r.get("detector_events")) for r in guidance),
        "llm_eligible_decisions": len(considered),
        "distinct_semantic_keys": distinct,
        "repeated_keys": len(digests) - distinct,
        "ideal_cache_hits": len(digests) - distinct,
        "calls_for_full_coverage": distinct,
        "attempts_made": sum(_int(r["live_attempts"]) for r in considered),
        "retries": sum(_int(r["retries"]) for r in considered),
        "budget_exhausted": exhausted,
        "activations": sum(_int(r["activated"]) for r in decisions),
        "guidance_coverage": ((len(considered) - exhausted) / float(len(considered))
                              if considered else None),
        "tokens_input": _token_stats(tokens_in),
        "tokens_output": _token_stats(tokens_out),
        "blocks": blocks,
    }


def _token_stats(values):
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "mean": statistics.mean(ordered),
        "median": statistics.median(ordered),
        "p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "max": ordered[-1],
    }


# ---------------------------------------------------------------------------
# 4. projection and cap selection
# ---------------------------------------------------------------------------


def project_calls(demand, horizon, *, t_max=PILOT_T_MAX):
    """Five projections of full-coverage calls at ``horizon``.

    All five project **calls**, not decisions: the ideal-cache saving is already
    removed by using distinct keys. They disagree by design -- the point is the
    spread, and the margin is applied to the largest.
    """
    distinct = demand["calls_for_full_coverage"]
    blocks = demand["blocks"]
    extra = max(0, horizon - t_max)

    naive = distinct * (horizon / float(t_max))

    final_rate = blocks[-1]["new_keys_per_step"] if blocks else 0.0
    final_based = distinct + final_rate * extra

    max_rate = max((b["new_keys_per_step"] for b in blocks), default=0.0)
    max_based = distinct + max_rate * extra

    # cumulative growth: average rate over the second half, which is less
    # dominated by the initial burst of novel situations
    half = blocks[len(blocks) // 2:] if blocks else []
    late_rate = (sum(b["distinct_new_keys"] for b in half)
                 / float(len(half) * BLOCK)) if half else 0.0
    cumulative_based = distinct + late_rate * extra

    worst = max(naive, final_based, max_based, cumulative_based)
    return {
        "horizon": horizon,
        "observed_distinct_keys_at_500k": distinct,
        "naive_linear": int(math.ceil(naive)),
        "final_100k_rate": int(math.ceil(final_based)),
        "final_100k_new_keys_per_step": final_rate,
        "max_100k_rate": int(math.ceil(max_based)),
        "max_100k_new_keys_per_step": max_rate,
        "cumulative_growth": int(math.ceil(cumulative_based)),
        "late_half_new_keys_per_step": late_rate,
        "worst_case": int(math.ceil(worst)),
        "worst_case_with_margin": int(math.ceil(worst * (1 + SAFETY_MARGIN))),
        "margin": SAFETY_MARGIN,
    }


def select_call_cap(projections):
    """One common cap for every revised condition (§4).

    Conservative maximum across the retained betas, +25 %, rounded up to the
    next 500, never below 3 000. A single shared cap is the point: a per-beta
    cap would reintroduce exactly the coverage confound that made the
    exploratory pilot uninterpretable.
    """
    if not projections:
        raise EvidenceError("no projections supplied; cannot size a cap")
    worst = max(p["worst_case"] for p in projections)
    with_margin = worst * (1 + SAFETY_MARGIN)
    rounded = _round_up(with_margin, CAP_ROUNDING)
    cap = max(rounded, MIN_CALL_CAP)
    return {
        "conservative_max_across_retained_betas": worst,
        "after_margin": with_margin,
        "rounded_to_next_{}".format(CAP_ROUNDING): rounded,
        "minimum_enforced": MIN_CALL_CAP,
        "call_cap": cap,
        "common_to_all_conditions": True,
        "rule": "max over retained betas, +{:.0%}, round up to {}, never below "
                "{}".format(SAFETY_MARGIN, CAP_ROUNDING, MIN_CALL_CAP),
    }


def cost_ceiling(cap, demands, *, price_table=PRICE_TABLE, margin=SAFETY_MARGIN):
    """Per-condition cost ceiling from the observed token distribution.

    Uses the **p95** per-call input and output token counts rather than the
    mean, then adds the same margin. Sizing a ceiling on the mean would be
    breached by an ordinary run of larger-than-average prompts.
    """
    p95_in = max((d["tokens_input"].get("p95", 0) for d in demands), default=0)
    p95_out = max((d["tokens_output"].get("p95", 0) for d in demands), default=0)
    max_in = max((d["tokens_input"].get("max", 0) for d in demands), default=0)
    max_out = max((d["tokens_output"].get("max", 0) for d in demands), default=0)
    mean_in = max((d["tokens_input"].get("mean", 0) for d in demands), default=0)

    at_p95 = price_table.cost_usd(input_tokens=cap * p95_in,
                                  cached_input_tokens=0,
                                  output_tokens=cap * p95_out)
    at_max = price_table.cost_usd(input_tokens=cap * max_in,
                                  cached_input_tokens=0,
                                  output_tokens=cap * max_out)
    ceiling = math.ceil(at_p95 * (1 + margin) * 100) / 100.0
    return {
        "price_table_date": price_table.date,
        "call_cap": cap,
        "p95_input_tokens_per_call": p95_in,
        "p95_output_tokens_per_call": p95_out,
        "max_input_tokens_per_call": max_in,
        "max_output_tokens_per_call": max_out,
        "mean_input_tokens_per_call": mean_in,
        "cost_at_p95_tokens": round(at_p95, 5),
        "cost_at_max_tokens": round(at_max, 5),
        "margin": margin,
        "cost_cap_usd": ceiling,
        "cost_cap_usd_all_conditions": round(ceiling * len(REVISED_BETAS)
                                             * len(REVISED_DEV_SEEDS), 4),
        "rule": "p95 tokens x cap, +{:.0%}, rounded up to the cent; the max-token "
                "figure is reported so the ceiling can be seen to bound it"
                .format(margin),
        "note": "cached input is provider-side prompt caching, NOT the "
                "application's SQLite response cache; it was 0 throughout the "
                "pilot so it is not assumed here",
    }


# ---------------------------------------------------------------------------
# 5. the frozen matrix
# ---------------------------------------------------------------------------


def design_matrix(horizon, call_cap, cost_cap, root="analysis/out/stage_e_recal"):
    if DETECTOR_HELDOUT_SEED in REVISED_DEV_SEEDS:
        raise EvidenceError(
            "seed {} is detector held-out evidence and must never be used for "
            "beta selection".format(DETECTOR_HELDOUT_SEED))
    conditions = []
    for beta in REVISED_BETAS:
        for seed in REVISED_DEV_SEEDS:
            name = "beta_{:.2f}_seed{}".format(beta, seed).replace(".", "p")
            conditions.append({
                "name": name,
                "beta": beta,
                "seed": seed,
                "t_max": horizon,
                "env": EXPECTED_ENV,
                "algorithm": EXPECTED_ALGORITHM,
                "time_limit": 50,
                "detector_variant": "wf15_m3_t5_ibon",
                "provider": "openai",
                "model": "gpt-5.4-nano-2026-03-17",
                "prompt_version": "selma_guidance_prompt_v3",
                "ttl": 15,
                "common_reward": True,
                "cache_mode": "read_write",
                "cache_path": os.path.join(root, name, "guidance_cache.sqlite3"),
                "dir": os.path.join(root, name),
                "live_call_cap": call_cap,
                "cost_cap_usd": cost_cap,
                "use_cuda": False,
                "greedy_evaluation_provider_calls": 0,
            })
    return conditions


def verify_matrix(conditions):
    """The matrix must be exactly 3 betas x 2 dev seeds, uniform except those."""
    problems = []
    if len(conditions) != len(REVISED_BETAS) * len(REVISED_DEV_SEEDS):
        problems.append("expected {} conditions, got {}".format(
            len(REVISED_BETAS) * len(REVISED_DEV_SEEDS), len(conditions)))
    if sorted({c["beta"] for c in conditions}) != sorted(REVISED_BETAS):
        problems.append("beta set is not the frozen list")
    if sorted({c["seed"] for c in conditions}) != sorted(REVISED_DEV_SEEDS):
        problems.append("seed set is not the frozen development seeds")
    if any(c["seed"] == DETECTOR_HELDOUT_SEED for c in conditions):
        problems.append("the detector held-out seed appears in the matrix")
    for key in ("t_max", "live_call_cap", "cost_cap_usd", "model",
                "prompt_version", "detector_variant", "ttl", "env"):
        if len({str(c[key]) for c in conditions}) != 1:
            problems.append("{} differs between conditions".format(key))
    caches = {c["cache_path"] for c in conditions}
    if len(caches) != len(conditions):
        problems.append("conditions share a cache path")
    return problems


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def build_design(args):
    baseline = authenticate_baseline(args.baseline_csv, args.baseline_sacred)
    curve = baseline_curve(args.baseline_csv)

    onset = nonfloor_onset(
        curve, episode_level=baseline["episode_level_success_available"],
        floor_return=args.floor_return)
    horizon = select_horizon(onset, test_interval=baseline["test_interval"])

    demands = []
    for name, beta in zip(PILOT_CONDITIONS, PILOT_BETAS):
        demands.append(condition_demand(
            name, beta, os.path.join(args.pilot_root, name)))

    retained = [d for d in demands if d["beta"] in REVISED_BETAS]
    if len(retained) != len(REVISED_BETAS):
        raise EvidenceError(
            "pilot evidence is missing for one of the retained betas {}".format(
                list(REVISED_BETAS)))

    projections = [project_calls(d, horizon["horizon"]) for d in retained]
    cap = select_call_cap(projections)
    ceiling = cost_ceiling(cap["call_cap"], retained)
    matrix = design_matrix(horizon["horizon"], cap["call_cap"],
                           ceiling["cost_cap_usd"])
    problems = verify_matrix(matrix)

    return {
        "kind": "selma_stage_e_beta_recalibration_design",
        "offline_only": True,
        "network_calls": 0,
        "training_runs": 0,
        "final_seeds_inspected": 0,
        "baseline": baseline,
        "baseline_curve": curve,
        "nonfloor_onset": onset,
        "horizon_selection": horizon,
        "pilot_demand": demands,
        "retained_betas": list(REVISED_BETAS),
        "excluded_from_revised_grid": [0.25],
        "projections": {d["name"]: p for d, p in zip(retained, projections)},
        "call_cap": cap,
        "cost_ceiling": ceiling,
        "development_seeds": list(REVISED_DEV_SEEDS),
        "detector_heldout_seed_excluded": DETECTOR_HELDOUT_SEED,
        "min_guidance_coverage": MIN_GUIDANCE_COVERAGE,
        "condition_matrix": matrix,
        "matrix_problems": problems,
        "design_valid": not problems,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pilot-root", required=True)
    parser.add_argument("--baseline-csv", required=True)
    parser.add_argument("--baseline-sacred", required=True)
    parser.add_argument("--report", default=None)
    parser.add_argument("--floor-return", type=float, default=None,
                        help="beta=0 500k floor, for the fallback non-floor rule")
    args = parser.parse_args(argv)

    design = build_design(args)

    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w") as handle:
            json.dump(design, handle, indent=2, sort_keys=True, default=str)
        print("report: {}".format(args.report))

    b = design["baseline"]
    print("baseline        : sacred run {} status {} seed {} t_max {}".format(
        b["sacred_run_id"], b["status"], b["seed"], b["t_max"]))
    print("  episode-level greedy success available: {}".format(
        b["episode_level_success_available"]))
    o = design["nonfloor_onset"]
    print("non-floor onset : t_env {} ({})".format(o["onset_t_env"], o["criterion"]))
    h = design["horizon_selection"]
    print("horizon         : {} ({})".format(h["horizon"], h["rule"]))
    print("\n{:<12} {:>9} {:>9} {:>9} {:>10} {:>10}".format(
        "condition", "decisions", "distinct", "idealHits", "coverage", "proj@H"))
    for d in design["pilot_demand"]:
        p = design["projections"].get(d["name"])
        print("{:<12} {:>9} {:>9} {:>9} {:>9.1%} {:>10}".format(
            d["name"], d["llm_eligible_decisions"], d["distinct_semantic_keys"],
            d["ideal_cache_hits"], d["guidance_coverage"] or 0,
            p["worst_case"] if p else "-"))
    c = design["call_cap"]
    print("\ncall cap        : {}  ({})".format(c["call_cap"], c["rule"]))
    print("cost cap        : ${} per condition, ${} total".format(
        design["cost_ceiling"]["cost_cap_usd"],
        design["cost_ceiling"]["cost_cap_usd_all_conditions"]))
    print("matrix          : {} conditions, valid={}".format(
        len(design["condition_matrix"]), design["design_valid"]))
    return 0 if design["design_valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
