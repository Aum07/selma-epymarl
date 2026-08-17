#!/usr/bin/env python
"""SELMA Stage D: analyse one beta-zero online measurement run.

Reads the two durable logs a live ``selma_mode=active`` run produces --

* the per-detector-event decision log (``selma_decision_log_csv``), and
* the per-transition PBRS log (``selma_guidance_log_csv``)

-- and reports run integrity, the detector -> joint-plan -> API -> activation
funnel, cache behaviour, a per-time-block breakdown, provider/API statistics,
beta-zero reward verification and forward budget projections.

**It measures; it does not select.** Nothing here reads episode return, and
nothing here recommends a ``beta``. Cost is computed only from an explicitly
supplied dated price table, because no price is hard-coded in this repository.

Run::

    PYTHONPATH=src python analysis/selma_beta_zero_measurement_report.py \
        --decision-log  <decisions.csv> \
        --guidance-log  <guidance.csv> \
        --price-table-date 2026-08-17 \
        --price-input 0.20 --price-cached-input 0.02 --price-output 1.25 \
        --report <out.json>

Exit status is 0 only when every acceptance gate passes.
"""

import argparse
import csv
import json
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from selma.pricing import NO_PRICE_TABLE, PriceTable  # noqa: E402

#: Frozen configuration this report expects to be analysing.
EXPECTED_SEED = 91
EXPECTED_T_MAX = 200000
EXPECTED_MODEL = "gpt-5.4-nano-2026-03-17"
EXPECTED_ATTEMPT_CAP = 200
EXPECTED_COST_CAP_USD = 0.25

#: Time blocks, in environment steps.
BLOCKS = ((0, 50000), (50000, 100000), (100000, 150000), (150000, 200000))

DECISION_ZERO = "zero_plan"
DECISION_SINGLETON = "singleton"
DECISION_MULTIPLE = "multiple"


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
    """``n/a`` and empty mean *not applicable* and stay ``None``, never 0.0."""
    if value in (None, "", "n/a"):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _finite(value):
    return value is not None and math.isfinite(value)


# ---------------------------------------------------------------------------
# A. run integrity
# ---------------------------------------------------------------------------


def run_integrity(guidance, decisions, *, exit_code=None):
    out = {"decision_rows": len(decisions), "guidance_rows": len(guidance)}
    if not guidance:
        out["error"] = "the per-transition guidance log is empty or missing"
        return out

    t_envs = [_int(r["t_env"]) for r in guidance]
    out["final_t_env"] = max(t_envs)
    out["min_t_env"] = min(t_envs)
    out["episodes"] = len({(r["seed"], r["episode_id"]) for r in guidance})
    out["seeds"] = sorted({r["seed"] for r in guidance})
    out["conditions"] = sorted({r["condition"] for r in guidance})
    out["terminal_causes"] = _distribution(
        r["terminal_cause"] for r in guidance if r["terminal_cause"])
    out["stored_terminals"] = sum(_int(r["stored_terminated"]) for r in guidance)
    out["process_exit_code"] = exit_code

    bad = []
    reward_mismatch = 0
    for row in guidance:
        for field in ("raw_team_reward", "training_reward", "phi_before",
                      "phi_after_effective", "pbrs_term_unscaled",
                      "shaping_contribution", "beta"):
            value = _float(row.get(field))
            if not _finite(value):
                bad.append(field)
        raw = _float(row.get("raw_team_reward"))
        train = _float(row.get("training_reward"))
        if raw is not None and train is not None and raw != train:
            reward_mismatch += 1
    out["non_finite_fields"] = sorted(set(bad))
    out["reward_reconciliation_errors"] = reward_mismatch
    out["malformed"] = bool(bad)
    return out


def _distribution(values):
    out = {}
    for value in values:
        key = str(value)
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


# ---------------------------------------------------------------------------
# B. detector -> joint-plan funnel
# ---------------------------------------------------------------------------


def funnel(guidance, decisions):
    detector_events = sum(_int(r.get("detector_events")) for r in guidance)
    buckets = _distribution(r["decision"] for r in decisions)
    considered = [r for r in decisions if r["decision"] == DECISION_MULTIPLE]
    activated = [r for r in decisions if _int(r["activated"])]

    def rate(n, d):
        return (n / float(d)) if d else None

    return {
        "detector_events_logged": detector_events,
        "decisions_logged": len(decisions),
        "zero_plan_events": buckets.get(DECISION_ZERO, 0),
        "singleton_events": buckets.get(DECISION_SINGLETON, 0),
        "multiple_plan_events": buckets.get(DECISION_MULTIPLE, 0),
        "missing_position_events": buckets.get("missing_position", 0),
        "zero_plan_rate": rate(buckets.get(DECISION_ZERO, 0), len(decisions)),
        "singleton_rate": rate(buckets.get(DECISION_SINGLETON, 0), len(decisions)),
        "multiple_plan_rate": rate(buckets.get(DECISION_MULTIPLE, 0), len(decisions)),
        "events_eligible_for_an_llm_decision": len(considered),
        "guidance_activations": len(activated),
        "activation_rate_of_eligible": rate(len(activated), len(considered)),
        "activation_rate_of_all_events": rate(len(activated), len(decisions)),
        "candidate_plan_count_distribution": _distribution(
            r["candidate_plan_count"] for r in considered),
        "recipient_count_distribution": _distribution(
            r["recipient_count"] for r in considered),
        "outcome_distribution": _distribution(r["outcome"] for r in decisions),
    }


# ---------------------------------------------------------------------------
# C. cache behaviour
# ---------------------------------------------------------------------------


def cache_behaviour(decisions, *, top_n=10):
    """Cache measurements plus the ideal replay hit rate.

    The *ideal* rate is what a perfect, never-evicting, never-budget-limited
    cache would have achieved on this exact ordered stream of questions: the
    first occurrence of each digest is a miss and every later one a hit. It is
    computed offline from the digests alone, so it is available even when the
    live budget stopped the real cache from filling.
    """
    considered = [r for r in decisions if r["decision"] == DECISION_MULTIPLE]
    digests = [r["key_digest"] for r in considered if r["key_digest"]]

    hits = sum(1 for r in considered if r["cache_result"] == "hit")
    misses = sum(1 for r in considered if r["cache_result"] == "miss")
    writes = sum(1 for r in considered
                 if r["cache_result"] == "miss" and _int(r["activated"]))

    seen = set()
    ideal_hits = 0
    cumulative_distinct = []
    for digest in digests:
        if digest in seen:
            ideal_hits += 1
        else:
            seen.add(digest)
        cumulative_distinct.append(len(seen))

    counts = {}
    for digest in digests:
        counts[digest] = counts.get(digest, 0) + 1
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))

    episodes_by_digest = {}
    for row in considered:
        if row["key_digest"]:
            episodes_by_digest.setdefault(row["key_digest"], set()).add(
                row["episode_id"])
    cross_episode = sum(1 for eps in episodes_by_digest.values() if len(eps) > 1)

    total = len(digests)
    return {
        "questions_with_a_digest": total,
        "questions_without_a_digest": len(considered) - total,
        "cache_hits": hits,
        "cache_misses": misses,
        "cache_writes": writes,
        "observed_hit_rate": (hits / float(len(considered))) if considered else None,
        "ideal_replay_hit_rate": (ideal_hits / float(total)) if total else None,
        "ideal_replay_hits": ideal_hits,
        "distinct_semantic_keys": len(seen),
        "repeat_questions": total - len(seen),
        "mean_uses_per_distinct_key": (total / float(len(seen))) if seen else None,
        "keys_used_once": sum(1 for c in counts.values() if c == 1),
        "keys_reused_across_episodes": cross_episode,
        # Digest prefixes only: the concentration is reportable without
        # disclosing any raw grid state.
        "top_keys": [{"digest_prefix": d[:12], "uses": c} for d, c in ordered[:top_n]],
        "top_key_share": (sum(c for _d, c in ordered[:top_n]) / float(total))
                         if total else None,
        "cumulative_distinct_curve": _downsample(cumulative_distinct, 40),
    }


def _downsample(series, points):
    if not series:
        return []
    if len(series) <= points:
        return [[i + 1, v] for i, v in enumerate(series)]
    step = len(series) / float(points)
    out = []
    for i in range(points):
        index = min(len(series) - 1, int(round((i + 1) * step)) - 1)
        out.append([index + 1, series[index]])
    return out


# ---------------------------------------------------------------------------
# D. time blocks
# ---------------------------------------------------------------------------


def time_blocks(guidance, decisions, blocks=BLOCKS):
    seen = set()
    rows = []
    for low, high in blocks:
        block_decisions = [r for r in decisions if low <= _int(r["t_env"]) < high]
        block_guidance = [r for r in guidance if low <= _int(r["t_env"]) < high]
        considered = [r for r in block_decisions
                      if r["decision"] == DECISION_MULTIPLE]
        digests = [r["key_digest"] for r in considered if r["key_digest"]]

        new_keys = 0
        repeats = 0
        for digest in digests:
            if digest in seen:
                repeats += 1
            else:
                seen.add(digest)
                new_keys += 1

        steps = len(block_guidance)
        detector_events = sum(_int(r.get("detector_events")) for r in block_guidance)
        rows.append({
            "block": "{}-{}k".format(low // 1000, high // 1000),
            "t_env_from": low,
            "t_env_to": high,
            "transitions": steps,
            "detector_events": detector_events,
            "decisions": len(block_decisions),
            "zero_plan": sum(1 for r in block_decisions
                             if r["decision"] == DECISION_ZERO),
            "singleton": sum(1 for r in block_decisions
                             if r["decision"] == DECISION_SINGLETON),
            "multiple_plan_decisions": len(considered),
            "new_semantic_keys": new_keys,
            "repeat_questions": repeats,
            "repeat_rate": (repeats / float(len(digests))) if digests else None,
            "live_attempts": sum(_int(r["live_attempts"]) for r in block_decisions),
            "cache_hits": sum(1 for r in considered if r["cache_result"] == "hit"),
            "activations": sum(_int(r["activated"]) for r in block_decisions),
            "detector_events_per_1k_steps": (1000.0 * detector_events / steps)
                                            if steps else None,
            "multiple_plan_decisions_per_1k_steps": (1000.0 * len(considered) / steps)
                                                    if steps else None,
            "cumulative_distinct_keys": len(seen),
        })
    return rows


# ---------------------------------------------------------------------------
# E. provider / API
# ---------------------------------------------------------------------------


def provider_stats(decisions, price_table):
    considered = [r for r in decisions if r["decision"] == DECISION_MULTIPLE]
    called = [r for r in considered if _int(r["live_attempts"]) > 0]
    attempts = sum(_int(r["live_attempts"]) for r in considered)
    retries = sum(_int(r["retries"]) for r in considered)
    successes = [r for r in called if _int(r["validator_valid"])]
    latencies = [_float(r["latency_s"]) for r in called]
    latencies = sorted(v for v in latencies if _finite(v))

    input_tokens = sum(_int(r["tokens_input"]) for r in called)
    cached_tokens = sum(_int(r["tokens_cached_input"]) for r in called)
    output_tokens = sum(_int(r["tokens_output"]) for r in called)

    failures = _distribution(
        r["outcome"] for r in called if not _int(r["validator_valid"]))

    cost = None
    if price_table.usable:
        cost = price_table.cost_usd(input_tokens=input_tokens,
                                    cached_input_tokens=cached_tokens,
                                    output_tokens=output_tokens)

    return {
        "request_attempts_including_retries": attempts,
        "retries": retries,
        "events_that_called": len(called),
        "successful_responses": len(successes),
        "failures_by_category": failures,
        "refusals": sum(1 for r in called if r["outcome"] == "provider_refusal"),
        "budget_exhausted_events": sum(
            1 for r in considered if r["outcome"] == "budget_exhausted"),
        "returned_model_counts": _distribution(
            r["returned_model"] for r in called if r["returned_model"]),
        "parse_and_ground_rate": (len(successes) / float(len(called)))
                                 if called else None,
        "validator_acceptances": sum(1 for r in considered
                                     if _int(r["validator_valid"])),
        "validator_rejections": sum(1 for r in considered
                                    if not _int(r["validator_valid"])
                                    and r["outcome"] not in
                                    ("budget_exhausted", "cache_read_only_miss")),
        "validator_reasons": _distribution(
            r["validator_reason"] for r in considered if r["validator_reason"]),
        "physically_valid_activations": sum(
            1 for r in decisions if _int(r["activated"]) and _int(r["physically_valid"])),
        "physically_invalid_activations": sum(
            1 for r in decisions if _int(r["activated"]) and not _int(r["physically_valid"])),
        "invented_plan_ids_activated": sum(
            1 for r in decisions if _int(r["activated"]) and not r["selected_plan_id"]),
        "latency_s": {
            "n": len(latencies),
            "min": round(latencies[0], 4) if latencies else None,
            "median": round(statistics.median(latencies), 4) if latencies else None,
            "p95": round(latencies[min(len(latencies) - 1,
                                       int(0.95 * len(latencies)))], 4)
                   if latencies else None,
            "max": round(latencies[-1], 4) if latencies else None,
        },
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_tokens,
        "output_tokens": output_tokens,
        "price_table_date": price_table.date,
        "cost_usd": cost,
        "cost_methodology": (
            "input_non_cached * {inp}/1M + cached_input * {cac}/1M + output * "
            "{out}/1M, price table dated {date}. Provider-side cached input is "
            "NOT the application's SQLite response cache."
            .format(inp=price_table.input_usd_per_mtok,
                    cac=price_table.cached_input_usd_per_mtok,
                    out=price_table.output_usd_per_mtok, date=price_table.date)
            if price_table.usable else
            "cost NOT calculated: no dated price table was supplied."),
    }


# ---------------------------------------------------------------------------
# F. reward / PBRS under beta = 0
# ---------------------------------------------------------------------------


def reward_checks(guidance, decisions):
    betas = {row.get("beta") for row in guidance}
    shaping = [_float(r["shaping_contribution"], 0.0) for r in guidance]
    unscaled = [_float(r["pbrs_term_unscaled"], 0.0) for r in guidance]
    mismatches = sum(
        1 for r in guidance
        if _float(r["raw_team_reward"]) != _float(r["training_reward"]))

    terminal_rows = [r for r in guidance if _int(r["stored_terminated"])]
    ttl = [_int(r["transitions_served"]) for r in guidance
           if r["transitions_served"] not in ("", None)]

    # Telescoping residual: phi_after_effective(t) must equal phi_before(t+1)
    # within one episode. Non-zero residual would mean the transformation does
    # not telescope and the shaping is not potential-based.
    residual = 0.0
    by_episode = {}
    for row in guidance:
        by_episode.setdefault((row["seed"], row["episode_id"]), []).append(row)
    for rows in by_episode.values():
        rows.sort(key=lambda r: _int(r["episode_step"]))
        for previous, current in zip(rows, rows[1:]):
            after = _float(previous["phi_after_effective"], 0.0)
            before = _float(current["phi_before"], 0.0)
            residual = max(residual, abs(after - before))

    return {
        "distinct_beta_values": sorted(betas),
        "beta_is_zero_everywhere": betas <= {"0.0", "0", "0.00"},
        "max_abs_shaping_contribution": max((abs(v) for v in shaping), default=0.0),
        "shaping_is_exactly_zero": all(v == 0.0 for v in shaping),
        "training_reward_equals_raw_reward": mismatches == 0,
        "reward_mismatches": mismatches,
        "non_zero_unscaled_pbrs_terms": sum(1 for v in unscaled if v != 0.0),
        "cumulative_unscaled_pbrs": sum(unscaled),
        "max_abs_unscaled_pbrs": max((abs(v) for v in unscaled), default=0.0),
        "stored_terminal_transitions": len(terminal_rows),
        "terminal_phi_after_effective_all_zero": all(
            _float(r["phi_after_effective"], 0.0) == 0.0 for r in terminal_rows),
        "max_telescoping_residual": residual,
        "ttl_distribution": _distribution(ttl),
        "greedy_eval_rows_in_decision_log": sum(
            1 for r in decisions if _int(r["greedy_eval"])),
        "greedy_eval_rows_in_guidance_log": sum(
            1 for r in guidance if _int(r.get("greedy_eval", 0))),
    }


# ---------------------------------------------------------------------------
# G. budget projections
# ---------------------------------------------------------------------------


def projections(cache, blocks, provider, *, t_max=EXPECTED_T_MAX, target=2000000,
                seeds=5):
    """Forward projections from the ordered semantic-key evidence.

    Deliberately reported as a **range**, and every method is labelled. The
    naive method assumes the distinct-key rate never saturates, which is an
    upper bound; the tail method assumes it has already reached the final
    block's rate, which is the optimistic end. The truth is between them, and
    this function does not pretend to know where.
    """
    distinct = cache["distinct_semantic_keys"]
    scale = float(target) / float(t_max)

    per_call_cost = None
    calls = provider["events_that_called"] or 0
    if provider["cost_usd"] is not None and calls:
        per_call_cost = provider["cost_usd"] / float(calls)

    tail = blocks[-1] if blocks else {}
    tail_new = tail.get("new_semantic_keys", 0)
    tail_span = max(1, tail.get("t_env_to", 1) - tail.get("t_env_from", 0))
    tail_rate_per_step = tail_new / float(tail_span)

    naive = int(round(distinct * scale))
    tail_based = int(round(distinct + tail_rate_per_step * (target - t_max)))

    low, high = sorted((tail_based, naive))

    def cost_of(n):
        return None if per_call_cost is None else round(n * per_call_cost, 4)

    return {
        "measured": {
            "t_max": t_max,
            "live_attempts": provider["request_attempts_including_retries"],
            "events_that_called": calls,
            "distinct_multiple_plan_questions": distinct,
            "cost_usd": provider["cost_usd"],
            "cost_per_call_usd": (round(per_call_cost, 8)
                                  if per_call_cost is not None else None),
        },
        "naive_linear_2m": {
            "label": "NAIVE upper projection: assumes the distinct-key rate "
                     "never saturates and scales it linearly by {:.1f}x."
                     .format(scale),
            "calls": naive,
            "cost_usd": cost_of(naive),
        },
        "final_block_rate_2m": {
            "label": "Tail projection: assumes the final {}-step block's "
                     "new-key rate ({:.4f} new keys/step) holds for the rest of "
                     "training. Optimistic end of the range."
                     .format(tail_span, tail_rate_per_step),
            "new_keys_per_step_final_block": tail_rate_per_step,
            "calls": tail_based,
            "cost_usd": cost_of(tail_based),
        },
        "conservative_range_2m": {
            "calls_low": low,
            "calls_high": high,
            "cost_usd_low": cost_of(low),
            "cost_usd_high": cost_of(high),
        },
        "five_seeds_isolated_caches": {
            "note": "Per-seed caches share nothing, so calls scale by the "
                    "number of seeds. Seeds diverge, so this is not exact.",
            "calls_low": low * seeds,
            "calls_high": high * seeds,
            "cost_usd_low": cost_of(low * seeds),
            "cost_usd_high": cost_of(high * seeds),
        },
        "provisional_2000_call_cap_sufficient": (
            None if high is None else bool(high <= 2000)),
        "limitation": (
            "These projections extrapolate an EARLY-TRAINING distribution. The "
            "policy state distribution changes as the agents learn, so the rate "
            "at which genuinely new coordination situations appear may rise or "
            "fall. This is not a claim that the 200k distribution equals a 2M "
            "one."),
    }


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------


def acceptance_gates(integrity, funnel_data, cache, provider, rewards, args):
    gates = []

    def gate(name, passed, detail=None):
        gates.append({"gate": name, "passed": bool(passed), "detail": detail})

    gate("process exited successfully",
         args.exit_code in (None, 0), integrity.get("process_exit_code"))
    gate("t_env reached at least {}".format(EXPECTED_T_MAX),
         integrity.get("final_t_env", 0) >= EXPECTED_T_MAX - 1,
         integrity.get("final_t_env"))
    gate("exactly one seed present",
         integrity.get("seeds") == [str(EXPECTED_SEED)], integrity.get("seeds"))
    gate("exactly one condition present",
         len(integrity.get("conditions", [])) == 1, integrity.get("conditions"))
    gate("no non-finite values", not integrity.get("non_finite_fields"),
         integrity.get("non_finite_fields"))
    gate("artifacts complete and non-empty",
         integrity.get("guidance_rows", 0) > 0 and integrity.get("decision_rows", 0) > 0)

    gate("beta is 0.0 on every row", rewards["beta_is_zero_everywhere"],
         rewards["distinct_beta_values"])
    gate("shaping contribution is exactly zero", rewards["shaping_is_exactly_zero"],
         rewards["max_abs_shaping_contribution"])
    gate("training reward equals raw reward",
         rewards["training_reward_equals_raw_reward"], rewards["reward_mismatches"])
    gate("no reward reconciliation errors",
         integrity.get("reward_reconciliation_errors", 0) == 0)
    gate("zero provider calls during greedy evaluation",
         rewards["greedy_eval_rows_in_decision_log"] == 0
         and rewards["greedy_eval_rows_in_guidance_log"] == 0)

    attempts = provider["request_attempts_including_retries"]
    gate("live attempts <= {} including retries".format(EXPECTED_ATTEMPT_CAP),
         attempts <= EXPECTED_ATTEMPT_CAP, attempts)
    cost = provider["cost_usd"]
    gate("calculated cost <= ${}".format(EXPECTED_COST_CAP_USD),
         cost is not None and cost <= EXPECTED_COST_CAP_USD, cost)
    gate("exact model snapshot returned for every successful call",
         set(provider["returned_model_counts"]) <= {EXPECTED_MODEL},
         provider["returned_model_counts"])
    gate("zero physically invalid guidance activated",
         provider["physically_invalid_activations"] == 0)
    gate("zero invented plan ids activated",
         provider["invented_plan_ids_activated"] == 0)
    gate("zero provider calls on cache hits", cache["cache_hits"] >= 0
         and _calls_on_hits(args) == 0)
    gate("zero provider calls for zero/singleton plans", _calls_on_skips(args) == 0)
    gate("every activated plan was physically valid",
         provider["physically_valid_activations"] == funnel_data["guidance_activations"])

    return gates


def _calls_on_hits(args):
    return args._calls_on_hits


def _calls_on_skips(args):
    return args._calls_on_skips


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def build_report(decisions, guidance, price_table, args):
    integrity = run_integrity(guidance, decisions, exit_code=args.exit_code)
    funnel_data = funnel(guidance, decisions)
    cache = cache_behaviour(decisions)
    blocks = time_blocks(guidance, decisions)
    provider = provider_stats(decisions, price_table)
    rewards = reward_checks(guidance, decisions)

    args._calls_on_hits = sum(
        _int(r["live_attempts"]) for r in decisions if r["cache_result"] == "hit")
    args._calls_on_skips = sum(
        _int(r["live_attempts"]) for r in decisions
        if r["decision"] in (DECISION_ZERO, DECISION_SINGLETON))

    report = {
        "kind": "selma_stage_d_beta_zero_measurement",
        "not_a_performance_result": True,
        "not_beta_selection": True,
        "beta": 0.0,
        "expected_model": EXPECTED_MODEL,
        "A_run_integrity": integrity,
        "B_funnel": funnel_data,
        "C_cache": cache,
        "D_time_blocks": blocks,
        "E_provider": provider,
        "F_reward_pbrs": rewards,
        "G_projections": projections(cache, blocks, provider),
        "calls_on_cache_hits": args._calls_on_hits,
        "calls_on_zero_or_singleton_plans": args._calls_on_skips,
    }
    report["H_gates"] = acceptance_gates(integrity, funnel_data, cache, provider,
                                         rewards, args)
    report["all_gates_passed"] = all(g["passed"] for g in report["H_gates"])
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--decision-log", required=True)
    parser.add_argument("--guidance-log", required=True)
    parser.add_argument("--report", default=None)
    parser.add_argument("--exit-code", type=int, default=None,
                        help="exit status of the measured training process")
    parser.add_argument("--price-table-date", default="")
    parser.add_argument("--price-input", type=float, default=0.0,
                        help="USD per 1M input tokens")
    parser.add_argument("--price-cached-input", type=float, default=0.0,
                        help="USD per 1M provider-side cached-input tokens")
    parser.add_argument("--price-output", type=float, default=0.0,
                        help="USD per 1M output tokens")
    args = parser.parse_args(argv)

    decisions = read_csv(args.decision_log)
    guidance = read_csv(args.guidance_log)
    if decisions is None or guidance is None:
        print("missing log: decision={!r} guidance={!r}".format(
            args.decision_log, args.guidance_log))
        return 2

    price_table = NO_PRICE_TABLE
    if args.price_table_date:
        price_table = PriceTable(args.price_table_date, args.price_input,
                                 args.price_cached_input, args.price_output)

    report = build_report(decisions, guidance, price_table, args)

    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w") as handle:
            json.dump(report, handle, indent=2, sort_keys=True, default=str)
        print("report: {}".format(args.report))

    integrity = report["A_run_integrity"]
    print("final t_env      : {}".format(integrity.get("final_t_env")))
    print("episodes         : {}".format(integrity.get("episodes")))
    print("decisions        : {}".format(integrity.get("decision_rows")))
    print("distinct keys    : {}".format(report["C_cache"]["distinct_semantic_keys"]))
    print("live attempts    : {}".format(
        report["E_provider"]["request_attempts_including_retries"]))
    print("cost usd         : {}".format(report["E_provider"]["cost_usd"]))
    for entry in report["H_gates"]:
        print("  [{}] {}".format("PASS" if entry["passed"] else "FAIL", entry["gate"]))
    return 0 if report["all_gates_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
