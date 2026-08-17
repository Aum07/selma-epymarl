"""SELMA Stage E: the locked recalibration analysis and its frozen rule.

Written and committed before any of the six conditions produced an outcome, so
the ranking rule is pinned against constructed evidence where the correct answer
is known in advance -- including the cases where the correct answer is to refuse
to select.

Run:  PYTHONPATH=src pytest tests/test_selma_stage_e_recal_analysis.py -v
"""

import csv
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))

import selma_stage_e_recal_analysis as ana  # noqa: E402
import selma_stage_e_recal_runner as runner  # noqa: E402
from selma.guidance import GuidanceCsvWriter, SelmaDecisionCsvWriter  # noqa: E402

MODEL = ana.FROZEN_MODEL


# ==========================================================================
# 1. the frozen constants
# ==========================================================================


def test_the_frozen_window_is_exactly_eighteen_points():
    assert len(ana.FROZEN_EVAL_POINTS) == 18
    assert ana.FROZEN_EVAL_POINTS[0] == 600000
    assert ana.FROZEN_EVAL_POINTS[-1] == 1450000
    assert all(b - a == 50000 for a, b in
               zip(ana.FROZEN_EVAL_POINTS, ana.FROZEN_EVAL_POINTS[1:]))


def test_the_frozen_window_lies_after_the_design_onset():
    """Onset was 550 050; every frozen point must be strictly after it."""
    assert all(p > 550050 for p in ana.FROZEN_EVAL_POINTS)


def test_the_frozen_betas_seeds_and_caps_match_the_runner():
    assert ana.FROZEN_BETAS == runner.FROZEN_BETAS == (0.00, 0.05, 0.10)
    assert ana.FROZEN_SEEDS == runner.FROZEN_SEEDS == (91, 92)
    assert ana.FROZEN_T_MAX == runner.FROZEN_T_MAX == 1500000
    assert ana.MIN_GUIDANCE_COVERAGE == runner.MIN_GUIDANCE_COVERAGE == 0.95
    assert ana.PER_CONDITION_CALL_CAP == runner.PER_CONDITION_CALL_CAP == 13500
    assert ana.GLOBAL_COST_CAP_USD == runner.GLOBAL_COST_CAP_USD == 20.94
    assert ana.FROZEN_MODEL == runner.FROZEN_MODEL


def test_seed_99_is_forbidden():
    assert 99 in ana.FORBIDDEN_SEEDS
    assert 99 not in ana.FROZEN_SEEDS


# ==========================================================================
# 2. the primary and secondary scores
# ==========================================================================


def _points(success_by_index):
    return [{"target": t, "t_env": t, "episodes": 100,
             "success_rate": s, "mean_return": s}
            for t, s in zip(ana.FROZEN_EVAL_POINTS, success_by_index)]


def test_the_auc_is_normalised_so_it_reads_as_a_mean_success_rate():
    assert ana.success_auc(_points([0.20] * 18)) == pytest.approx(0.20)


def test_the_auc_rises_with_success():
    low = ana.success_auc(_points([0.05] * 18))
    high = ana.success_auc(_points([0.30] * 18))
    assert high > low


def test_the_auc_needs_at_least_two_points():
    assert ana.success_auc(_points([0.2])) is None


def test_the_secondary_metric_uses_exactly_the_final_five_points():
    points = _points([0.0] * 13 + [1.0] * 5)
    assert ana.final_window_success(points) == pytest.approx(1.0)
    assert ana.FINAL_WINDOW_POINTS == 5


def test_frozen_points_are_matched_by_nearest_scheduled_multiple():
    """Actual evaluations land on episode boundaries, not exact multiples."""
    observed = [{"t_env": t + 37, "episodes": 100, "success_rate": 0.1,
                 "mean_return": 0.1} for t in ana.FROZEN_EVAL_POINTS]
    selected, missing = ana.select_frozen_points(observed)
    assert missing == []
    assert len(selected) == 18
    assert [s["target"] for s in selected] == list(ana.FROZEN_EVAL_POINTS)


def test_a_missing_frozen_point_is_reported_not_silently_skipped():
    observed = [{"t_env": t, "episodes": 100, "success_rate": 0.1,
                 "mean_return": 0.1} for t in ana.FROZEN_EVAL_POINTS[:-1]]
    selected, missing = ana.select_frozen_points(observed)
    assert missing == [ana.FROZEN_EVAL_POINTS[-1]]
    assert len(selected) == 17


def test_points_far_outside_the_tolerance_do_not_match():
    observed = [{"t_env": t + 40000, "episodes": 100, "success_rate": 0.1,
                 "mean_return": 0.1} for t in ana.FROZEN_EVAL_POINTS]
    _selected, missing = ana.select_frozen_points(observed)
    assert missing, "a 40k offset exceeds the half-interval tolerance"


# ==========================================================================
# 3. the preregistered selection rule
# ==========================================================================


def _beta(beta, primary, secondary=None):
    return {"beta": beta, "primary_score": primary,
            "secondary_score": secondary if secondary is not None else primary,
            "seeds": {}, "mean_detector_events": 0}


def test_the_highest_primary_score_wins():
    decision = ana.select_beta(
        [_beta(0.00, 0.10), _beta(0.05, 0.20), _beta(0.10, 0.15)], True)
    assert decision["resolved"]
    assert decision["selected_beta"] == 0.05


def test_beta_zero_is_allowed_to_win():
    decision = ana.select_beta(
        [_beta(0.00, 0.30), _beta(0.05, 0.20), _beta(0.10, 0.15)], True)
    assert decision["resolved"]
    assert decision["selected_beta"] == 0.00
    assert decision["is_baseline"] is True


def test_an_exact_tie_falls_through_to_the_secondary_metric():
    decision = ana.select_beta(
        [_beta(0.00, 0.20, 0.10), _beta(0.05, 0.20, 0.40), _beta(0.10, 0.15)],
        True)
    assert decision["selected_beta"] == 0.05
    assert any(s.get("step") == 4 for s in decision["steps"])


def test_a_full_tie_selects_the_smaller_beta():
    decision = ana.select_beta(
        [_beta(0.00, 0.20, 0.20), _beta(0.05, 0.20, 0.20), _beta(0.10, 0.10)],
        True)
    assert decision["selected_beta"] == 0.00
    assert any(s.get("step") == 5 for s in decision["steps"])


def test_a_failed_gate_anywhere_leaves_beta_unresolved():
    decision = ana.select_beta(
        [_beta(0.00, 0.10), _beta(0.05, 0.90), _beta(0.10, 0.15)], False)
    assert decision["resolved"] is False
    assert decision["selected_beta"] is None
    assert "UNRESOLVED" in decision["reason"]
    assert "post hoc" in decision["reason"]


def test_a_missing_primary_score_leaves_beta_unresolved():
    decision = ana.select_beta(
        [_beta(0.00, 0.10), _beta(0.05, None), _beta(0.10, 0.15)], True)
    assert decision["resolved"] is False
    assert "UNRESOLVED" in decision["reason"]


def test_the_decision_never_claims_significance():
    decision = ana.select_beta(
        [_beta(0.00, 0.10), _beta(0.05, 0.20), _beta(0.10, 0.15)], True)
    assert "no significance" in decision["reason"].lower() or \
        "supports no significance" in decision["reason"]


def test_the_tie_epsilon_is_the_preregistered_value():
    assert ana.TIE_EPSILON == 1e-12


# ==========================================================================
# 4. end-to-end on synthetic condition directories
# ==========================================================================


def _decisions(n=200, *, exhausted=0, model=MODEL):
    rows = []
    for i in range(n):
        bad = i >= n - exhausted
        row = {c: "" for c in SelmaDecisionCsvWriter.COLUMNS}
        row.update({
            "t_env": (i + 1) * 100, "condition": "c", "seed": 91,
            "episode_id": i, "episode_step": 1, "greedy_eval": 0, "beta": 0.05,
            "event_id": "e", "decision": "multiple", "candidate_plan_count": 6,
            "recipient_count": 2, "key_digest": "d%063d" % i,
            "cache_result": "miss", "provider_considered": 1,
            "live_attempts": 0 if bad else 1, "retries": 0,
            "outcome": "budget_exhausted" if bad else "activated",
            "selected_plan_id": "" if bad else "plan_0",
            "validator_valid": 0 if bad else 1, "validator_reason": "",
            "physically_valid": 0 if bad else 1, "activated": 0 if bad else 1,
            "tokens_input": "n/a" if bad else 877,
            "tokens_cached_input": "n/a" if bad else 0,
            "tokens_output": "n/a" if bad else 25, "tokens_reasoning": 0,
            "returned_model": "" if bad else model,
            "latency_s": "n/a" if bad else 0.8, "request_id": "r",
            "detector_events_this_transition": 1,
        })
        rows.append(row)
    return rows


def _guidance(beta, t_max=ana.FROZEN_T_MAX):
    rows = []
    for i in range(0, t_max + 1, 50000):
        row = {c: "" for c in GuidanceCsvWriter.COLUMNS}
        row.update({
            "t_env": i, "condition": "c", "seed": 91, "episode_id": i,
            "episode_step": 0, "beta": beta, "gamma": 0.99,
            "raw_team_reward": 0.0, "training_reward": 0.0,
            "shaping_contribution": 0.0, "pbrs_term_unscaled": -0.01,
            "phi_before": 0.0, "phi_after_effective": 0.0,
            "stored_terminated": 1, "terminal_cause": "time_limit",
            "detector_events": 1, "greedy_eval": 0,
        })
        rows.append(row)
    return rows


def _episodes(success):
    header = ["t_env", "test_mode", "ep_length", "team_return",
              "multi_agent_reward_steps", "single_agent_reward_steps",
              "agent0_return", "agent1_return", "successful_load_events",
              "solo_load_events", "multi_agent_load_events",
              "necessary_cooperative_load_events", "failed_load_events",
              "failed_load_attempts"]
    rows = []
    for t in range(50000, ana.FROZEN_T_MAX + 1, 50000):
        hits = int(round(100 * success))
        for ep in range(100):
            value = 1.0 if ep < hits else 0.0
            rows.append(dict(zip(header, [
                t, 1, 50, value, 0, 0, value / 2, value / 2,
                2 if value else 0, 0, 1 if value else 0, 1 if value else 0,
                0, 0])))
    return header, rows


def _write_condition(root, beta, seed, *, success=0.2, exhausted=0,
                     model=MODEL, exit_code=0, completed=True):
    name = ana.condition_name(beta, seed)
    directory = os.path.join(root, name)
    os.makedirs(directory, exist_ok=True)

    dpath = os.path.join(directory, "decisions.csv")
    gpath = os.path.join(directory, "guidance.csv")
    epath = os.path.join(directory, "episodes.csv")
    with open(dpath, "w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=SelmaDecisionCsvWriter.COLUMNS)
        w.writeheader(); w.writerows(_decisions(exhausted=exhausted, model=model))
    with open(gpath, "w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=GuidanceCsvWriter.COLUMNS)
        w.writeheader(); w.writerows(_guidance(beta))
    header, rows = _episodes(success)
    with open(epath, "w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=header)
        w.writeheader(); w.writerows(rows)

    hashes = {}
    for key, path in (("decision_log", dpath), ("guidance_log", gpath),
                      ("episode_log", epath)):
        hashes[key] = {"path": path, "bytes": os.path.getsize(path),
                       "sha256": ana.sha256_file(path)}
    manifest = {
        "exit_code": exit_code, "wall_clock_s": 1.0,
        "artifact_hashes": hashes,
        "condition": {"beta": beta, "seed": seed, "t_max": ana.FROZEN_T_MAX,
                      "env_key": ana.FROZEN_ENV, "model": model},
    }
    with open(os.path.join(directory, "manifest.json"), "w") as handle:
        json.dump(manifest, handle)
    if completed:
        with open(os.path.join(directory, "COMPLETED"), "w") as handle:
            handle.write("exit_code=0\n")
    return directory


@pytest.fixture
def root(tmp_path):
    base = str(tmp_path / "runs")
    successes = {0.00: 0.10, 0.05: 0.20, 0.10: 0.15}
    for beta in ana.FROZEN_BETAS:
        for seed in ana.FROZEN_SEEDS:
            _write_condition(base, beta, seed, success=successes[beta])
    return base


def test_a_clean_six_condition_analysis_selects_the_best_beta(root, capsys):
    report = ana.analyse(root)
    assert report["all_gates_passed"] is True
    assert len(report["conditions"]) == 6
    assert report["selection"]["resolved"] is True
    assert report["selection"]["selected_beta"] == 0.05
    assert report["development_calibration_only"] is True
    assert report["significance_claim"] is False
    assert report["n_seeds"] == 2


def test_every_condition_matches_all_eighteen_frozen_points(root):
    report = ana.analyse(root)
    for condition in report["conditions"]:
        assert condition["frozen_points_matched"] == 18
        assert condition["problems"] == []


def test_low_coverage_leaves_beta_unresolved(tmp_path):
    base = str(tmp_path / "runs")
    for beta in ana.FROZEN_BETAS:
        for seed in ana.FROZEN_SEEDS:
            _write_condition(base, beta, seed, success=0.2,
                             exhausted=20 if (beta, seed) == (0.10, 92) else 0)
    report = ana.analyse(base)
    assert report["all_gates_passed"] is False
    assert report["selection"]["resolved"] is False
    bad = next(c for c in report["conditions"] if c["name"].endswith("0p10_seed92"))
    assert any("coverage" in p for p in bad["problems"])


def test_a_model_substitution_leaves_beta_unresolved(tmp_path):
    base = str(tmp_path / "runs")
    for beta in ana.FROZEN_BETAS:
        for seed in ana.FROZEN_SEEDS:
            _write_condition(base, beta, seed,
                             model="gpt-other-2020-01-01"
                             if (beta, seed) == (0.05, 91) else MODEL)
    report = ana.analyse(base)
    assert report["all_gates_passed"] is False
    assert report["selection"]["resolved"] is False


def test_a_missing_completed_marker_is_detected(tmp_path):
    base = str(tmp_path / "runs")
    for beta in ana.FROZEN_BETAS:
        for seed in ana.FROZEN_SEEDS:
            _write_condition(base, beta, seed,
                             completed=(beta, seed) != (0.00, 91))
    report = ana.analyse(base)
    assert report["all_gates_passed"] is False
    bad = next(c for c in report["conditions"] if c["name"].endswith("0p00_seed91"))
    assert any("COMPLETED" in p for p in bad["problems"])


def test_a_tampered_artifact_is_detected(root):
    path = os.path.join(root, ana.condition_name(0.05, 91), "episodes.csv")
    with open(path, "a") as handle:
        handle.write("1,1,50,1.0,0,0,0.5,0.5,2,0,1,1,0,0\n")
    report = ana.analyse(root)
    bad = next(c for c in report["conditions"] if c["name"].endswith("0p05_seed91"))
    assert any("hash changed" in p for p in bad["problems"])
    assert report["all_gates_passed"] is False


def test_a_missing_condition_directory_fails_loudly(tmp_path):
    base = str(tmp_path / "runs")
    _write_condition(base, 0.00, 91)
    with pytest.raises(ana.AnalysisError):
        ana.analyse(base)


def test_counts_and_cost_are_reported_separately(root):
    report = ana.analyse(root)
    checks = report["conditions"][0]["checks"]
    for key in ("live_attempts", "retries", "distinct_semantic_keys",
                "cache_hits", "cache_misses", "activations", "input_tokens",
                "output_tokens", "cost_usd", "detector_events"):
        assert key in checks, key
    assert report["total_cost_usd"] >= 0
    assert report["within_global_cost_cap"] is True


def test_detector_event_frequency_is_reported_per_beta(root):
    report = ana.analyse(root)
    for entry in report["per_beta"]:
        assert "mean_detector_events" in entry
        assert set(entry["seeds"]) == {91, 92}


def test_no_forbidden_seed_is_ever_inspected(root):
    report = ana.analyse(root)
    assert report["forbidden_seeds_inspected"] == []
    seeds = {c["seed"] for c in report["conditions"]}
    assert seeds == {91, 92}
    assert 99 not in seeds


def test_the_cli_returns_nonzero_when_a_gate_fails(tmp_path):
    base = str(tmp_path / "runs")
    for beta in ana.FROZEN_BETAS:
        for seed in ana.FROZEN_SEEDS:
            _write_condition(base, beta, seed,
                             exhausted=50 if (beta, seed) == (0.00, 92) else 0)
    assert ana.main(["--root", base, "--report", str(tmp_path / "r.json")]) == 1


def test_the_cli_returns_zero_on_a_clean_analysis(root, tmp_path):
    assert ana.main(["--root", root,
                     "--report", str(tmp_path / "ok.json")]) == 0
