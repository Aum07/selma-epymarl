"""SELMA Stage E: the beta-selection rule, on synthetic conditions.

The rule is preregistered, so it is tested against constructed inputs where the
correct answer is known in advance -- including the cases where the correct
answer is "refuse to select". No real run is involved.

Run:  PYTHONPATH=src pytest tests/test_selma_stage_e_analysis.py -v
"""

import csv
import os
import statistics
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))

import selma_stage_e_beta_analysis as ana  # noqa: E402
from selma.guidance import GuidanceCsvWriter, SelmaDecisionCsvWriter  # noqa: E402

MODEL = ana.EXPECTED_MODEL


# ==========================================================================
# helpers
# ==========================================================================


def condition(beta, *, auc=1.0, ci=(0.9, 1.1), exclusions=(), name=None):
    return {
        "name": name or "beta_{}".format(beta),
        "beta": beta,
        "primary_auc": auc,
        "bootstrap": None if ci is None else {
            "ci95_low": ci[0], "ci95_high": ci[1], "evaluation_points": 40,
            "resamples": 10, "point_mean_of_window": auc, "caveat": "x"},
        "safety_exclusions": list(exclusions),
    }


def guidance_rows(n=60, *, beta=0.05, shaping=0.0, raw_positive_every=10,
                  unscaled=-0.02, telescoping_break=False, nan=False,
                  coop_after=None, detector_at=()):
    rows = []
    phi = 0.0
    for step in range(n):
        raw = 1.0 if (step and step % raw_positive_every == 0) else 0.0
        contribution = shaping if step % 3 == 0 else 0.0
        row = {c: "" for c in GuidanceCsvWriter.COLUMNS}
        row.update({
            "t_env": step + 1, "condition": "c", "seed": 91, "episode_id": 1,
            "episode_step": step, "condition_checked": 1, "query_triggered": 0,
            "queries": 0, "beta": beta, "gamma": 0.99,
            "raw_team_reward": raw, "training_reward": raw + contribution,
            "phi_before": phi, "potential_after_real": 0.0,
            "phi_after_effective": 0.0 if step == n - 1 else phi,
            "pbrs_term_unscaled": unscaled, "shaping_contribution": contribution,
            "stored_terminated": 1 if step == n - 1 else 0,
            "terminal_cause": "time_limit" if step == n - 1 else "",
            "guidance_active": 0, "guidance_transition": "", "guidance_age": "",
            "transitions_served": min(step, 15), "expiry_reason": "",
            "provider": "openai", "provider_valid": 0, "fallback_reason": "",
            "detector_events": 1 if step in detector_at else 0,
            "decisions": 0, "zero_plan_decisions": 0, "singleton_decisions": 0,
            "multiple_plan_decisions": 0, "greedy_eval": 0,
            "step_successful_loads": 0, "step_failed_loads": 0,
            "step_cooperative_loads": (
                1 if coop_after is not None and step in coop_after else 0),
            "query_tokens_prompt": "n/a", "query_tokens_cached_prompt": "n/a",
            "query_tokens_completion": "n/a", "query_latency_s": "n/a",
            "query_cost_usd": "n/a", "query_returned_model": "",
            "provider_calls": 0, "total_queries": 0, "activations": 0,
            "replacements": 0, "expiries": 0, "rejected": 0,
            "terminal_suppressed_events": 0, "event_id": "", "subgoal_id": "",
            "trigger_agent_ids": "", "recipient_agent_ids": "",
            "target_food_id": "", "selected_plan_id": "", "phi_components": "",
        })
        if nan and step == 5:
            row["pbrs_term_unscaled"] = "nan"
        if telescoping_break and step == 4:
            row["phi_after_effective"] = -0.5
        rows.append(row)
    return rows


def decision_rows(n=5, **over):
    rows = []
    for i in range(n):
        row = {c: "" for c in SelmaDecisionCsvWriter.COLUMNS}
        row.update({
            "t_env": i + 1, "condition": "c", "seed": 91, "episode_id": 1,
            "episode_step": i, "greedy_eval": 0, "beta": 0.05, "event_id": "e",
            "decision": "multiple", "candidate_plan_count": 6,
            "recipient_count": 2, "key_digest": "d" * 64, "cache_result": "miss",
            "provider_considered": 1, "live_attempts": 1, "retries": 0,
            "outcome": "activated", "selected_plan_id": "plan_0",
            "validator_valid": 1, "validator_reason": "", "physically_valid": 1,
            "activated": 1, "tokens_input": 800, "tokens_cached_input": 0,
            "tokens_output": 25, "tokens_reasoning": 0, "returned_model": MODEL,
            "latency_s": 1.0, "request_id": "r",
            "detector_events_this_transition": 1,
        })
        row.update(over)
        rows.append(row)
    return rows


def episode_rows(points=50, *, base=0.5, slope=0.0):
    header = ["t_env", "test_mode", "ep_length", "team_return",
              "multi_agent_reward_steps", "single_agent_reward_steps",
              "agent0_return", "agent1_return", "successful_load_events",
              "solo_load_events", "multi_agent_load_events",
              "necessary_cooperative_load_events", "failed_load_events",
              "failed_load_attempts"]
    rows = []
    for i in range(points):
        t_env = (i + 1) * 10000
        value = base + slope * i
        for _ep in range(4):
            rows.append(dict(zip(header, [
                t_env, 1, 50, value, 0, 0, value / 2, value / 2,
                1 if value > 0 else 0, 0, 1, 1 if value > 0 else 0, 0, 0])))
    return rows


def write_condition(root, name, *, guidance=None, decisions=None, episodes=None):
    directory = os.path.join(root, name)
    os.makedirs(directory, exist_ok=True)

    def dump(path, columns, rows):
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for row in rows or []:
                writer.writerow(row)

    dump(os.path.join(directory, "guidance.csv"), GuidanceCsvWriter.COLUMNS,
         guidance if guidance is not None else guidance_rows())
    dump(os.path.join(directory, "decisions.csv"), SelmaDecisionCsvWriter.COLUMNS,
         decisions if decisions is not None else decision_rows())
    eps = episodes if episodes is not None else episode_rows()
    with open(os.path.join(directory, "episodes.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(eps[0]))
        writer.writeheader()
        writer.writerows(eps)
    return directory


# ==========================================================================
# 1. the primary metric
# ==========================================================================


def test_auc_is_restricted_to_the_preregistered_window():
    assert ana.AUC_FROM == 100000 and ana.AUC_TO == 500000
    curve = [{"t_env": t, "mean_return": 0.0 if t < 100000 else 1.0}
             for t in range(10000, 510000, 10000)]
    # the pre-100k zeros must not drag the score down
    assert ana.auc(curve) == pytest.approx(1.0)


def test_auc_is_normalised_so_it_reads_as_a_mean():
    curve = [{"t_env": t, "mean_return": 2.0} for t in range(100000, 510000, 10000)]
    assert ana.auc(curve) == pytest.approx(2.0)


def test_auc_is_none_without_enough_points():
    assert ana.auc([{"t_env": 200000, "mean_return": 1.0}]) is None


def test_the_greedy_curve_uses_only_test_mode_rows():
    rows = episode_rows(points=3)
    for row in rows[:4]:
        row["test_mode"] = 0
        row["team_return"] = 99.0
    curve = ana.greedy_curve(rows)
    assert all(p["mean_return"] < 90 for p in curve)
    assert len(curve) == 2


def test_bootstrap_reports_an_interval_and_its_caveat():
    curve = [{"t_env": t, "mean_return": 1.0 + 0.01 * i}
             for i, t in enumerate(range(100000, 510000, 10000))]
    result = ana.bootstrap_auc(curve, resamples=500)
    assert result["ci95_low"] <= result["point_mean_of_window"] <= result["ci95_high"]
    assert "not a significance test" in result["caveat"]


def test_bootstrap_is_deterministic():
    curve = [{"t_env": t, "mean_return": float(i % 5)}
             for i, t in enumerate(range(100000, 510000, 10000))]
    assert ana.bootstrap_auc(curve, resamples=200) == \
        ana.bootstrap_auc(curve, resamples=200)


# ==========================================================================
# 2. safety exclusions
# ==========================================================================


def _clean_inputs():
    integ = {"non_finite_fields": [], "reward_reconciliation_errors": 0,
             "max_telescoping_residual": 0.0}
    prov = {"physically_invalid_activations": 0, "greedy_eval_decisions": 0,
            "returned_models": {MODEL: 10}, "budget_exhausted": 0}
    shp = {"max_shaping_over_smallest_positive_reward": 0.1,
           "p95_shaping_over_median_positive_reward": 0.05}
    return integ, prov, shp


def test_a_clean_condition_has_no_exclusions():
    integ, prov, shp = _clean_inputs()
    assert ana.safety_exclusions(0.05, integ, prov, shp, {}) == []


def test_non_finite_values_exclude():
    integ, prov, shp = _clean_inputs()
    integ["non_finite_fields"] = ["pbrs_term_unscaled"]
    assert any(r.startswith("1:") for r in
               ana.safety_exclusions(0.05, integ, prov, shp, {}))


def test_reward_reconciliation_failure_excludes():
    integ, prov, shp = _clean_inputs()
    integ["reward_reconciliation_errors"] = 3
    assert any(r.startswith("2:") for r in
               ana.safety_exclusions(0.05, integ, prov, shp, {}))


def test_a_telescoping_break_excludes():
    integ, prov, shp = _clean_inputs()
    integ["max_telescoping_residual"] = 0.5
    assert any(r.startswith("3:") for r in
               ana.safety_exclusions(0.05, integ, prov, shp, {}))


def test_physically_invalid_guidance_excludes():
    integ, prov, shp = _clean_inputs()
    prov["physically_invalid_activations"] = 1
    assert any(r.startswith("4:") for r in
               ana.safety_exclusions(0.05, integ, prov, shp, {}))


def test_a_greedy_evaluation_call_or_model_substitution_excludes():
    integ, prov, shp = _clean_inputs()
    prov["greedy_eval_decisions"] = 1
    assert any(r.startswith("5:") for r in
               ana.safety_exclusions(0.05, integ, prov, shp, {}))
    integ, prov, shp = _clean_inputs()
    prov["returned_models"] = {"gpt-other-2020-01-01": 1}
    assert any(r.startswith("5:") for r in
               ana.safety_exclusions(0.05, integ, prov, shp, {}))


def test_shaping_above_half_the_smallest_positive_reward_excludes():
    integ, prov, shp = _clean_inputs()
    shp["max_shaping_over_smallest_positive_reward"] = 0.51
    reasons = ana.safety_exclusions(0.05, integ, prov, shp, {})
    assert any(r.startswith("6:") for r in reasons)
    shp["max_shaping_over_smallest_positive_reward"] = 0.49
    assert not any(r.startswith("6:") for r in
                   ana.safety_exclusions(0.05, integ, prov, shp, {}))


def test_p95_shaping_above_a_quarter_of_the_median_reward_excludes():
    integ, prov, shp = _clean_inputs()
    shp["p95_shaping_over_median_positive_reward"] = 0.26
    assert any(r.startswith("7:") for r in
               ana.safety_exclusions(0.05, integ, prov, shp, {}))


def test_learner_instability_relative_to_the_baseline_excludes():
    integ, prov, shp = _clean_inputs()
    baseline = {"loss": {"max_abs": 1.0, "n": 10}}
    unstable = {"loss": {"max_abs": 50.0, "n": 10}}
    assert any(r.startswith("8:") for r in ana.safety_exclusions(
        0.05, integ, prov, shp, unstable, baseline_learn=baseline))
    stable = {"loss": {"max_abs": 2.0, "n": 10}}
    assert not any(r.startswith("8:") for r in ana.safety_exclusions(
        0.05, integ, prov, shp, stable, baseline_learn=baseline))


def test_cap_exhaustion_excludes_on_unequal_exposure():
    integ, prov, shp = _clean_inputs()
    prov["budget_exhausted"] = 7
    assert any(r.startswith("9:") for r in
               ana.safety_exclusions(0.05, integ, prov, shp, {}))


# ==========================================================================
# 3. the selection rule
# ==========================================================================


def test_the_best_safe_beta_is_selected_when_intervals_do_not_overlap():
    conditions = [
        condition(0.0, auc=1.0, ci=(0.95, 1.05)),
        condition(0.05, auc=1.2, ci=(1.15, 1.25)),
        condition(0.10, auc=2.0, ci=(1.95, 2.05)),
        condition(0.25, auc=1.5, ci=(1.45, 1.55)),
    ]
    decision = ana.select_beta(conditions)
    assert decision["resolved"]
    assert decision["selected_beta"] == 0.10


def test_an_overlapping_smaller_beta_wins_by_parsimony():
    conditions = [
        condition(0.0, auc=1.0, ci=(0.9, 1.1)),
        condition(0.05, auc=1.8, ci=(1.5, 2.1)),
        condition(0.10, auc=2.0, ci=(1.7, 2.3)),   # overlaps 0.05
        condition(0.25, auc=1.2, ci=(1.0, 1.4)),
    ]
    decision = ana.select_beta(conditions)
    assert decision["selected_beta"] == 0.05
    assert any(s.get("step") == 3 for s in decision["steps"])


def test_an_unsafe_best_beta_is_excluded_before_scoring():
    conditions = [
        condition(0.0, auc=1.0, ci=(0.9, 1.1)),
        condition(0.05, auc=1.2, ci=(1.15, 1.25)),
        condition(0.10, auc=5.0, ci=(4.9, 5.1),
                  exclusions=["6: max |shaping| too large"]),
        condition(0.25, auc=1.1, ci=(1.05, 1.15)),
    ]
    decision = ana.select_beta(conditions)
    assert decision["selected_beta"] == 0.05
    excluded = decision["steps"][0]["excluded"]
    assert [e["beta"] for e in excluded] == [0.10]


def test_no_selection_is_forced_when_every_positive_beta_is_unsafe():
    conditions = [
        condition(0.0, auc=1.0),
        condition(0.05, auc=2.0, exclusions=["1: nan"]),
        condition(0.10, auc=2.0, exclusions=["9: cap exhausted"]),
        condition(0.25, auc=2.0, exclusions=["6: shaping too large"]),
    ]
    decision = ana.select_beta(conditions)
    assert decision["resolved"] is False
    assert decision["selected_beta"] is None
    assert "UNRESOLVED" in decision["reason"]


def test_no_selection_is_forced_when_no_positive_beta_beats_the_baseline():
    """Rule 5: directional competitiveness is required, not assumed."""
    conditions = [
        condition(0.0, auc=2.0, ci=(1.9, 2.1)),
        condition(0.05, auc=1.0, ci=(0.9, 1.1)),
        condition(0.10, auc=0.8, ci=(0.7, 0.9)),
        condition(0.25, auc=0.5, ci=(0.4, 0.6)),
    ]
    decision = ana.select_beta(conditions)
    assert decision["resolved"] is False
    assert decision["selected_beta"] is None
    assert "directionally competitive" in decision["reason"]


def test_the_baseline_is_never_selected_as_the_shaping_beta():
    conditions = [condition(0.0, auc=9.0, ci=(8.9, 9.1))] + [
        condition(b, auc=1.0, exclusions=["1: nan"]) for b in (0.05, 0.10, 0.25)]
    decision = ana.select_beta(conditions)
    assert decision["selected_beta"] is None


def test_the_decision_records_every_rule_step():
    conditions = [condition(0.0, auc=1.0)] + [
        condition(b, auc=1.0 + b) for b in (0.05, 0.10, 0.25)]
    decision = ana.select_beta(conditions)
    steps = [s.get("step") for s in decision["steps"]]
    assert 1 in steps and 2 in steps and 3 in steps


# ==========================================================================
# 4. end-to-end on synthetic condition directories
# ==========================================================================


@pytest.fixture
def root(tmp_path):
    base = str(tmp_path / "pilot")
    write_condition(base, "beta_0p00",
                    guidance=guidance_rows(beta=0.0, shaping=0.0),
                    episodes=episode_rows(base=0.5, slope=0.01))
    write_condition(base, "beta_0p05",
                    guidance=guidance_rows(beta=0.05, shaping=0.01),
                    episodes=episode_rows(base=0.6, slope=0.012))
    write_condition(base, "beta_0p10",
                    guidance=guidance_rows(beta=0.10, shaping=0.02),
                    episodes=episode_rows(base=0.55, slope=0.011))
    write_condition(base, "beta_0p25",
                    guidance=guidance_rows(beta=0.25, shaping=0.60),
                    episodes=episode_rows(base=0.4, slope=0.005))
    return base


def test_the_analysis_runs_end_to_end(root, tmp_path, capsys):
    report_path = str(tmp_path / "analysis.json")
    assert ana.main(["--root", root, "--report", report_path]) == 0
    out = capsys.readouterr().out
    assert "beta_0p00" in out and "selection:" in out
    import json

    report = json.load(open(report_path))
    assert [c["beta"] for c in report["conditions"]] == [0.0, 0.05, 0.10, 0.25]
    assert report["one_development_seed_pilot"] is True
    assert report["not_a_significance_claim"] is True


def test_a_large_shaping_condition_trips_the_magnitude_exclusion(root):
    result = ana.analyse_condition("beta_0p25", 0.25,
                                   os.path.join(root, "beta_0p25"))
    ratio = result["shaping"]["max_shaping_over_smallest_positive_reward"]
    assert ratio == pytest.approx(0.60)
    reasons = ana.safety_exclusions(0.25, result["integrity"],
                                    result["provider_cache"], result["shaping"], {})
    assert any(r.startswith("6:") for r in reasons)


def test_a_small_shaping_condition_passes_the_magnitude_exclusion(root):
    result = ana.analyse_condition("beta_0p05", 0.05,
                                   os.path.join(root, "beta_0p05"))
    reasons = ana.safety_exclusions(0.05, result["integrity"],
                                    result["provider_cache"], result["shaping"], {})
    assert not any(r.startswith(("6:", "7:")) for r in reasons)


def test_reward_reconciliation_uses_raw_plus_shaping(root):
    """training_reward must equal raw + shaping, not merely equal raw."""
    result = ana.analyse_condition("beta_0p05", 0.05,
                                   os.path.join(root, "beta_0p05"))
    assert result["integrity"]["reward_reconciliation_errors"] == 0

    broken = guidance_rows(beta=0.05, shaping=0.01)
    broken[3]["training_reward"] = 42.0
    integ = ana.integrity(broken, [], [])
    assert integ["reward_reconciliation_errors"] == 1


def test_a_telescoping_break_is_detected_end_to_end(tmp_path):
    directory = write_condition(str(tmp_path), "beta_0p10",
                                guidance=guidance_rows(telescoping_break=True))
    result = ana.analyse_condition("beta_0p10", 0.10, directory)
    assert result["integrity"]["max_telescoping_residual"] > 1e-9


def test_failed_load_resolution_is_computed_from_the_load_columns():
    rows = guidance_rows(n=30, detector_at=(2, 20), coop_after=(5,))
    result = ana.resolution_within(rows, horizon=10)
    assert result["detector_events_scanned"] == 2
    assert result["resolved_within_10"] == 1
    assert result["resolution_rate"] == pytest.approx(0.5)


def test_comparability_surfaces_unequal_exposure(root):
    conditions = [ana.analyse_condition(n, b, os.path.join(root, n))
                  for n, b in zip(ana.CONDITIONS, ana.BETAS)]
    table = ana.comparability(conditions)
    assert set(table["live_attempts"]) == set(ana.CONDITIONS)
    assert table["any_cap_exhaustion"] is False
    assert table["artifacts_complete"] is True


def test_a_missing_condition_is_reported_not_silently_skipped(tmp_path):
    result = ana.analyse_condition("beta_0p10", 0.10,
                                   os.path.join(str(tmp_path), "absent"))
    assert result["integrity"]["artifacts_complete"] is False
    assert result["primary_auc"] is None


# ==========================================================================
# 5. Sacred metrics are matched to a condition by beta, never by recency
# ==========================================================================


def _sacred_run(root, run_id, *, beta, seed=91, t_max=500000, loss=1.0):
    import json

    directory = os.path.join(root, str(run_id))
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "config.json"), "w") as handle:
        json.dump({"selma_beta": beta, "seed": seed, "t_max": t_max}, handle)
    with open(os.path.join(directory, "metrics.json"), "w") as handle:
        json.dump({"loss": {"values": [loss, loss * 2]},
                   "grad_norm": {"values": [1.0, 2.0]}}, handle)
    return directory


def test_sacred_metrics_are_matched_by_beta_not_by_recency(tmp_path):
    root = str(tmp_path / "sacred")
    _sacred_run(root, 1, beta=0.0, loss=10.0)
    _sacred_run(root, 2, beta=0.05, loss=20.0)
    _sacred_run(root, 3, beta=0.25, loss=30.0)

    got = ana.load_sacred_metrics(None, sacred_root=root, beta=0.05)
    assert got["loss"]["values"][0] == 20.0
    got = ana.load_sacred_metrics(None, sacred_root=root, beta=0.25)
    assert got["loss"]["values"][0] == 30.0


def test_a_sacred_run_with_the_wrong_seed_is_not_attached(tmp_path):
    root = str(tmp_path / "sacred")
    _sacred_run(root, 1, beta=0.10, seed=7)
    assert ana.load_sacred_metrics(None, sacred_root=root, beta=0.10) is None


def test_a_sacred_run_with_the_wrong_t_max_is_not_attached(tmp_path):
    root = str(tmp_path / "sacred")
    _sacred_run(root, 1, beta=0.10, t_max=200000)
    assert ana.load_sacred_metrics(None, sacred_root=root, beta=0.10) is None


def test_learner_stability_is_reported_from_the_matched_run(tmp_path, root):
    sacred = str(tmp_path / "sacred")
    _sacred_run(sacred, 9, beta=0.05, loss=3.0)
    result = ana.analyse_condition("beta_0p05", 0.05,
                                   os.path.join(root, "beta_0p05"),
                                   sacred_root=sacred)
    assert result["learner_stability"]["loss"]["max_abs"] == 6.0
