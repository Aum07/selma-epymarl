"""SELMA Stage E: the revised-calibration design rules, and two regressions.

Two classes of test live here.

**Regressions** pin the factual corrections made to the Stage E record: the
detector-event relationship was *not* monotonic across all four betas, and the
``beta=0`` cap was exhausted late in training, not "19 % through". Both wordings
were committed once and must not reappear.

**Design rules** exercise the mechanical horizon, call-cap and cost-ceiling
selection on constructed evidence where the right answer is known in advance --
including the cases where the right answer is to refuse.

Everything here is offline. No network, no training, no credential.

Run:  PYTHONPATH=src pytest tests/test_selma_stage_e_recalibration_design.py -v
"""

import csv
import json
import os
import re
import socket
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))

import selma_stage_e_recalibration_design as design  # noqa: E402
from selma.guidance import GuidanceCsvWriter, SelmaDecisionCsvWriter  # noqa: E402

REPO = os.path.join(os.path.dirname(__file__), "..")
STAGE_E_RECORD = os.path.join(
    REPO, "docs", "project", "selma_stage_e_beta_calibration_2026-08-17.md")

#: The observed pilot decision counts. Deliberately duplicated here so a test
#: fails if a document ever restates them differently.
OBSERVED_DECISIONS = {0.00: 1294, 0.05: 1660, 0.10: 1996, 0.25: 893}


# ==========================================================================
# 1. regression: the "monotonic" claim must not reappear
# ==========================================================================


def _record_text():
    with open(STAGE_E_RECORD) as handle:
        return handle.read()


def _record_body():
    """The record with the correction notice removed.

    The notice deliberately *quotes* both incorrect statements in order to
    document them as errors, so a naive scan would flag the correction itself.
    What must stay clean is the rest of the document: the record may describe
    the mistake, but must never assert it.
    """
    lines = _record_text().splitlines()
    out = []
    inside = False
    for line in lines:
        if "FACTUAL CORRECTIONS" in line:
            inside = True
            continue
        if inside:
            # the notice is one contiguous blockquote
            if line.startswith(">") or not line.strip():
                continue
            inside = False
        out.append(line)
    return "\n".join(out)


def test_the_observed_decision_counts_are_not_monotonic_in_beta():
    """The premise of the correction, asserted on the numbers themselves."""
    betas = sorted(OBSERVED_DECISIONS)
    counts = [OBSERVED_DECISIONS[b] for b in betas]
    assert counts == [1294, 1660, 1996, 893]
    assert counts != sorted(counts), "counts are not increasing across all four"
    assert counts[:3] == sorted(counts[:3]), "they do increase over the first three"


def test_the_record_never_claims_a_monotonic_relationship_across_all_betas():
    text = _record_body()
    for line in text.splitlines():
        low = line.lower()
        if "monotonic" not in low:
            continue
        # every surviving mention must be a denial or an explicit scope limit
        assert any(marker in low for marker in
                   ("not monotonic", "non-monotonic", "no monotonic")), line


def test_the_record_states_the_corrected_interpretation():
    text = _record_text().lower()
    assert "increased from" in text and "0.10" in text
    assert "substantially fewer events" in text
    assert "materially unequal guidance coverage" in text


def test_the_record_keeps_the_original_observed_counts():
    """The correction changes wording, never the evidence."""
    text = _record_text()
    for count in ("1 294", "1 660", "1 996", "893"):
        assert count in text, count


# ==========================================================================
# 2. regression: the "19 % through training" claim must not reappear
# ==========================================================================


def test_the_record_does_not_say_the_cap_was_exhausted_19_percent_through():
    text = _record_body()
    pattern = re.compile(r"1[89](\.\d+)?\s*%\s*(of the way|through)", re.I)
    for line in text.splitlines():
        assert not pattern.search(line), line


def test_the_correction_notice_itself_documents_both_errors():
    """The notice must exist and name both mistakes, or the scan proves nothing."""
    text = _record_text()
    assert "FACTUAL CORRECTIONS" in text
    assert "Monotonic" in text and "was wrong" in text
    assert "conflated two different quantities" in text
    # and the body really is shorter than the whole document
    assert len(_record_body()) < len(text)


def test_the_record_separates_unguided_share_from_position_in_training():
    text = _record_text()
    assert "18.8 %" in text, "the unguided share is still reported"
    assert "not** a position in training" in text, \
        "the record must say explicitly that 18.8 % is not a training position"
    assert "92.3 %" in text, "the measured elapsed fraction is reported"


def test_the_record_reports_the_measured_exhaustion_points():
    text = _record_text()
    for value in ("461 636", "441 106", "378 726"):
        assert value in text, value


# ==========================================================================
# 3. baseline authentication
# ==========================================================================


def _write_sacred(root, run_id, *, seed=91, env=design.EXPECTED_ENV,
                  t_max=2000000, status="COMPLETED", name="qmix",
                  log_csv="episodes.csv", test_interval=50000, nepisode=100):
    directory = os.path.join(root, str(run_id))
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "config.json"), "w") as handle:
        json.dump({"seed": seed, "env_args": {"key": env}, "t_max": t_max,
                   "name": name, "selma_log_csv": log_csv,
                   "test_interval": test_interval, "test_nepisode": nepisode},
                  handle)
    with open(os.path.join(directory, "run.json"), "w") as handle:
        json.dump({"status": status}, handle)
    return directory


def _write_episodes(path, *, points=40, interval=50000, episodes=100,
                    success_from=None, test_mode=1):
    header = ["t_env", "test_mode", "ep_length", "team_return",
              "multi_agent_reward_steps", "single_agent_reward_steps",
              "agent0_return", "agent1_return", "successful_load_events",
              "solo_load_events", "multi_agent_load_events",
              "necessary_cooperative_load_events", "failed_load_events",
              "failed_load_attempts"]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for i in range(points):
            t_env = (i + 1) * interval
            hits = 0
            if success_from is not None and t_env >= success_from:
                hits = int(episodes * 0.20)          # 20 % success
            for ep in range(episodes):
                value = 1.0 if ep < hits else 0.0
                writer.writerow(dict(zip(header, [
                    t_env, test_mode, 50, value, 0, 0, value / 2, value / 2,
                    2 if value else 0, 0, 1 if value else 0,
                    1 if value else 0, 0, 0])))
    return path


@pytest.fixture
def good_baseline(tmp_path):
    sacred = _write_sacred(str(tmp_path / "sacred"), 16,
                           log_csv="episodes_real.csv")
    csv_path = _write_episodes(str(tmp_path / "episodes_real.csv"),
                               success_from=600000)
    return csv_path, sacred


def test_an_authentic_baseline_is_accepted(good_baseline):
    csv_path, sacred = good_baseline
    info = design.authenticate_baseline(csv_path, sacred)
    assert info["authenticated"]
    assert info["seed"] == 91
    assert info["status"] == "COMPLETED"
    assert info["episode_level_success_available"] is True
    assert info["evaluation_points"] == 40


@pytest.mark.parametrize("override,fragment", [
    ({"seed": 92}, "seed"),
    ({"env": "lbforaging:Foraging-10x10-3p-3f-v3"}, "env"),
    ({"t_max": 500000}, "t_max"),
    ({"status": "RUNNING"}, "status"),
    ({"name": "vdn"}, "algorithm"),
])
def test_a_mismatched_baseline_is_rejected(tmp_path, override, fragment):
    kwargs = {"log_csv": "episodes_real.csv"}
    kwargs.update(override)
    sacred = _write_sacred(str(tmp_path / "sacred"), 1, **kwargs)
    csv_path = _write_episodes(str(tmp_path / "episodes_real.csv"),
                               success_from=600000)
    with pytest.raises(design.EvidenceError) as error:
        design.authenticate_baseline(csv_path, sacred)
    assert fragment in str(error.value)


def test_a_superseded_partial_run_is_never_accepted(tmp_path):
    """A RUNNING run is exactly the partial evidence that must be refused."""
    sacred = _write_sacred(str(tmp_path / "sacred"), 14, status="RUNNING",
                           log_csv="episodes_real.csv")
    csv_path = _write_episodes(str(tmp_path / "episodes_real.csv"),
                               success_from=600000)
    with pytest.raises(design.EvidenceError) as error:
        design.authenticate_baseline(csv_path, sacred)
    assert "RUNNING" in str(error.value) or "status" in str(error.value)


def test_a_sacred_run_that_logged_a_different_csv_is_rejected(tmp_path):
    sacred = _write_sacred(str(tmp_path / "sacred"), 16, log_csv="other.csv")
    csv_path = _write_episodes(str(tmp_path / "episodes_real.csv"),
                               success_from=600000)
    with pytest.raises(design.EvidenceError) as error:
        design.authenticate_baseline(csv_path, sacred)
    assert "logged episodes to" in str(error.value)


def test_a_missing_or_empty_baseline_fails_loudly(tmp_path):
    sacred = _write_sacred(str(tmp_path / "sacred"), 16)
    with pytest.raises(design.EvidenceError):
        design.authenticate_baseline(str(tmp_path / "absent.csv"), sacred)
    with pytest.raises(design.EvidenceError):
        design.authenticate_baseline(str(tmp_path / "absent.csv"), None)


def test_a_baseline_without_greedy_rows_is_rejected(tmp_path):
    sacred = _write_sacred(str(tmp_path / "sacred"), 16,
                           log_csv="train_only.csv")
    csv_path = _write_episodes(str(tmp_path / "train_only.csv"), test_mode=0)
    with pytest.raises(design.EvidenceError) as error:
        design.authenticate_baseline(csv_path, sacred)
    assert "greedy" in str(error.value)


# ==========================================================================
# 4. the preregistered non-floor rule and horizon boundaries
# ==========================================================================


def _curve(success_by_point):
    return [{"t_env": (i + 1) * 50000, "episodes": 100,
             "mean_return": s, "success_rate": s}
            for i, s in enumerate(success_by_point)]


def test_onset_requires_five_consecutive_points_above_the_threshold():
    # four good points then a dip: not sustained
    curve = _curve([0.0] * 5 + [0.06] * 4 + [0.0] + [0.06] * 5)
    onset = design.nonfloor_onset(curve, episode_level=True)
    assert onset["sustained"] is True
    # the sustained run starts at index 10 (the second block of five)
    assert onset["onset_t_env"] == curve[10]["t_env"]


def test_onset_is_none_when_the_curve_never_leaves_the_floor():
    onset = design.nonfloor_onset(_curve([0.01] * 40), episode_level=True)
    assert onset["sustained"] is False
    assert onset["onset_t_env"] is None


def test_the_threshold_boundary_is_inclusive():
    exactly = design.nonfloor_onset(_curve([0.05] * 10), episode_level=True)
    just_below = design.nonfloor_onset(_curve([0.049] * 10), episode_level=True)
    assert exactly["sustained"] is True
    assert just_below["sustained"] is False


def test_the_fallback_rule_needs_the_beta_zero_floor():
    with pytest.raises(design.EvidenceError):
        design.nonfloor_onset(_curve([0.1] * 10), episode_level=False)
    onset = design.nonfloor_onset(_curve([0.1] * 10), episode_level=False,
                                  floor_return=0.05)
    assert onset["sustained"] is True
    assert "floor" in onset["criterion"]


def test_the_smallest_qualifying_horizon_is_selected():
    onset = {"onset_t_env": 400000}
    chosen = design.select_horizon(onset, test_interval=50000)
    assert chosen["horizon"] == 1000000       # 400k + 500k = 900k <= 1M


def test_a_horizon_too_close_to_onset_is_rejected():
    onset = {"onset_t_env": 800000}
    chosen = design.select_horizon(onset, test_interval=50000)
    assert chosen["horizon"] == 1500000       # 1M is only 200k after onset
    rejected = [r["candidate"] for r in chosen["rejected_candidates"]]
    assert 1000000 in rejected


def test_a_horizon_with_too_few_evaluations_is_rejected():
    onset = {"onset_t_env": 400000}
    chosen = design.select_horizon(onset, test_interval=200000)
    # (1M-400k)/200k = 3 evaluations, below the required 8
    assert chosen["horizon"] > 1000000


def test_unestablished_onset_selects_the_largest_horizon_conservatively():
    chosen = design.select_horizon({"onset_t_env": None}, test_interval=50000)
    assert chosen["horizon"] == 2000000
    assert "conservative" in chosen["rule"]
    assert "conservative" in chosen["reason"]


def test_the_candidate_horizons_are_the_preregistered_three():
    assert design.CANDIDATE_HORIZONS == (1000000, 1500000, 2000000)


# ==========================================================================
# 5. call demand: events vs decisions vs calls
# ==========================================================================


def _decision(t_env, digest, *, outcome="activated", attempts=1, activated=1,
              tokens_in=800, tokens_out=25):
    row = {c: "" for c in SelmaDecisionCsvWriter.COLUMNS}
    row.update({
        "t_env": t_env, "condition": "c", "seed": 91, "episode_id": 1,
        "episode_step": 1, "greedy_eval": 0, "beta": 0.05, "event_id": "e",
        "decision": "multiple", "candidate_plan_count": 6, "recipient_count": 2,
        "key_digest": digest, "cache_result": "miss" if attempts else "hit",
        "provider_considered": 1, "live_attempts": attempts, "retries": 0,
        "outcome": outcome, "selected_plan_id": "plan_0", "validator_valid": 1,
        "validator_reason": "", "physically_valid": 1, "activated": activated,
        "tokens_input": tokens_in, "tokens_cached_input": 0,
        "tokens_output": tokens_out, "tokens_reasoning": 0,
        "returned_model": "gpt-5.4-nano-2026-03-17", "latency_s": 0.8,
        "request_id": "r", "detector_events_this_transition": 1,
    })
    return row


def _guidance(t_env, events=1):
    row = {c: "" for c in GuidanceCsvWriter.COLUMNS}
    row.update({"t_env": t_env, "condition": "c", "seed": 91, "episode_id": 1,
                "episode_step": 1, "detector_events": events, "beta": 0.05,
                "raw_team_reward": 0.0, "training_reward": 0.0,
                "shaping_contribution": 0.0, "pbrs_term_unscaled": 0.0,
                "phi_before": 0.0, "phi_after_effective": 0.0,
                "stored_terminated": 0, "greedy_eval": 0})
    return row


def _write_condition(root, name, decisions, guidance):
    directory = os.path.join(root, name)
    os.makedirs(directory, exist_ok=True)
    for path, columns, rows in (
            (os.path.join(directory, "decisions.csv"),
             SelmaDecisionCsvWriter.COLUMNS, decisions),
            (os.path.join(directory, "guidance.csv"),
             GuidanceCsvWriter.COLUMNS, guidance)):
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    return directory


def test_full_coverage_calls_equal_distinct_keys_not_decisions(tmp_path):
    """A repeated semantic key costs no call under a perfect cache."""
    decisions = ([_decision(1000 + i, "d%02d" % i) for i in range(10)]
                 + [_decision(2000 + i, "d00") for i in range(5)])
    guidance = [_guidance(1000 + i) for i in range(15)]
    directory = _write_condition(str(tmp_path), "beta_0p05", decisions, guidance)

    demand = design.condition_demand("beta_0p05", 0.05, directory)
    assert demand["llm_eligible_decisions"] == 15
    assert demand["distinct_semantic_keys"] == 10
    assert demand["ideal_cache_hits"] == 5
    assert demand["calls_for_full_coverage"] == 10, "not 15"


def test_detector_events_are_not_treated_as_calls(tmp_path):
    """More detector events than decisions must not inflate the call estimate."""
    decisions = [_decision(1000, "aa"), _decision(2000, "bb")]
    guidance = [_guidance(1000, events=5), _guidance(2000, events=5)]
    directory = _write_condition(str(tmp_path), "beta_0p05", decisions, guidance)
    demand = design.condition_demand("beta_0p05", 0.05, directory)
    assert demand["detector_events"] == 10
    assert demand["llm_eligible_decisions"] == 2
    assert demand["calls_for_full_coverage"] == 2


def test_blocks_partition_the_run_and_track_new_keys(tmp_path):
    decisions = [_decision(50000, "a"), _decision(150000, "b"),
                 _decision(160000, "a"), _decision(450000, "c")]
    guidance = [_guidance(t) for t in (50000, 150000, 160000, 450000)]
    directory = _write_condition(str(tmp_path), "beta_0p10", decisions, guidance)
    demand = design.condition_demand("beta_0p10", 0.10, directory)
    blocks = demand["blocks"]
    assert len(blocks) == 5
    assert blocks[0]["distinct_new_keys"] == 1
    assert blocks[1]["distinct_new_keys"] == 1
    assert blocks[1]["repeated_keys"] == 1
    assert blocks[1]["ideal_cache_hits"] == 1
    assert blocks[-1]["cumulative_distinct_keys"] == 3


def test_guidance_coverage_is_computed_from_budget_exhaustion(tmp_path):
    decisions = ([_decision(1000 + i, "d%d" % i) for i in range(8)]
                 + [_decision(9000 + i, "e%d" % i, outcome="budget_exhausted",
                              attempts=0, activated=0) for i in range(2)])
    guidance = [_guidance(1000 + i) for i in range(10)]
    directory = _write_condition(str(tmp_path), "beta_0p00", decisions, guidance)
    demand = design.condition_demand("beta_0p00", 0.0, directory)
    assert demand["budget_exhausted"] == 2
    assert demand["guidance_coverage"] == pytest.approx(0.8)


def test_a_missing_pilot_condition_fails_loudly(tmp_path):
    with pytest.raises(design.EvidenceError):
        design.condition_demand("beta_0p05", 0.05, str(tmp_path / "absent"))


# ==========================================================================
# 6. projection, margin, rounding and the common cap
# ==========================================================================


def _demand(distinct, *, blocks_new=None, p95_in=900, p95_out=30):
    blocks_new = blocks_new or [distinct // 5] * 5
    blocks = []
    seen = 0
    for i, new in enumerate(blocks_new):
        seen += new
        blocks.append({"block": "%d" % i, "distinct_new_keys": new,
                       "new_keys_per_step": new / float(design.BLOCK),
                       "cumulative_distinct_keys": seen})
    return {
        "name": "x", "beta": 0.05, "calls_for_full_coverage": distinct,
        "blocks": blocks,
        "tokens_input": {"p95": p95_in, "max": p95_in + 200, "mean": p95_in - 100},
        "tokens_output": {"p95": p95_out, "max": p95_out + 10,
                          "mean": p95_out - 5},
    }


def test_the_naive_projection_scales_linearly():
    projection = design.project_calls(_demand(1000), 2000000)
    assert projection["naive_linear"] == 4000


def test_all_five_projection_methods_are_reported():
    projection = design.project_calls(_demand(1000), 1500000)
    for key in ("naive_linear", "final_100k_rate", "max_100k_rate",
                "cumulative_growth", "worst_case_with_margin"):
        assert key in projection
    assert projection["margin"] == 0.25


def test_the_worst_case_takes_the_largest_method():
    demand = _demand(1000, blocks_new=[100, 100, 100, 100, 600])
    projection = design.project_calls(demand, 1000000)
    assert projection["worst_case"] >= projection["naive_linear"]
    assert projection["worst_case"] >= projection["final_100k_rate"]


def test_the_margin_is_exactly_twenty_five_percent():
    projection = design.project_calls(_demand(1000), 1000000)
    assert projection["worst_case_with_margin"] == \
        pytest.approx(projection["worst_case"] * 1.25, rel=0, abs=1)


def test_the_cap_is_rounded_up_to_the_next_500():
    cap = design.select_call_cap([{"worst_case": 3001}])
    # 3001 * 1.25 = 3751.25 -> 4000
    assert cap["call_cap"] == 4000
    assert cap["call_cap"] % 500 == 0


def test_the_cap_is_never_below_three_thousand():
    cap = design.select_call_cap([{"worst_case": 10}])
    assert cap["call_cap"] == 3000
    assert design.MIN_CALL_CAP == 3000


def test_the_cap_uses_the_conservative_maximum_across_retained_betas():
    cap = design.select_call_cap([{"worst_case": 1000}, {"worst_case": 2600},
                                  {"worst_case": 1500}])
    assert cap["conservative_max_across_retained_betas"] == 2600
    assert cap["call_cap"] == 3500          # 2600*1.25 = 3250 -> 3500


def test_the_same_cap_applies_to_every_condition():
    cap = design.select_call_cap([{"worst_case": 4000}])
    matrix = design.design_matrix(2000000, cap["call_cap"], 1.0)
    assert len({c["live_call_cap"] for c in matrix}) == 1
    assert cap["common_to_all_conditions"] is True


def test_no_projections_fails_loudly():
    with pytest.raises(design.EvidenceError):
        design.select_call_cap([])


# ==========================================================================
# 7. cost ceiling from the token distribution
# ==========================================================================


def test_the_cost_ceiling_uses_p95_tokens_not_the_mean():
    ceiling = design.cost_ceiling(1000, [_demand(500, p95_in=1000, p95_out=40)])
    at_p95 = (1000 * 1000 * 0.20 + 1000 * 40 * 1.25) / 1e6
    assert ceiling["cost_at_p95_tokens"] == pytest.approx(at_p95, abs=1e-5)
    assert ceiling["cost_cap_usd"] >= at_p95 * 1.25 - 0.01
    assert ceiling["p95_input_tokens_per_call"] == 1000
    assert ceiling["mean_input_tokens_per_call"] == 900


def test_the_cost_ceiling_bounds_the_max_token_case():
    ceiling = design.cost_ceiling(1000, [_demand(500, p95_in=900, p95_out=30)])
    assert ceiling["cost_at_max_tokens"] > ceiling["cost_at_p95_tokens"]
    assert "max_input_tokens_per_call" in ceiling


def test_the_cost_ceiling_uses_the_frozen_price_table():
    ceiling = design.cost_ceiling(100, [_demand(50)])
    assert ceiling["price_table_date"] == "2026-08-17"
    assert design.PRICE_TABLE.input_usd_per_mtok == 0.20
    assert design.PRICE_TABLE.cached_input_usd_per_mtok == 0.02
    assert design.PRICE_TABLE.output_usd_per_mtok == 1.25


def test_provider_cached_input_is_not_confused_with_the_sqlite_cache():
    ceiling = design.cost_ceiling(100, [_demand(50)])
    assert "SQLite" in ceiling["note"]


# ==========================================================================
# 8. the frozen matrix
# ==========================================================================


def test_the_revised_beta_list_is_exactly_three_values():
    assert design.REVISED_BETAS == (0.00, 0.05, 0.10)
    assert 0.25 not in design.REVISED_BETAS


def test_the_development_seeds_are_exactly_91_and_92():
    assert design.REVISED_DEV_SEEDS == (91, 92)


def test_the_detector_heldout_seed_is_excluded():
    assert design.DETECTOR_HELDOUT_SEED == 99
    assert 99 not in design.REVISED_DEV_SEEDS


def test_the_matrix_is_exactly_six_conditions():
    matrix = design.design_matrix(2000000, 3500, 1.0)
    assert len(matrix) == 6
    assert sorted({c["beta"] for c in matrix}) == [0.00, 0.05, 0.10]
    assert sorted({c["seed"] for c in matrix}) == [91, 92]
    assert design.verify_matrix(matrix) == []


def test_every_condition_shares_the_frozen_settings():
    matrix = design.design_matrix(1500000, 3500, 1.0)
    for condition in matrix:
        assert condition["model"] == "gpt-5.4-nano-2026-03-17"
        assert condition["prompt_version"] == "selma_guidance_prompt_v3"
        assert condition["detector_variant"] == "wf15_m3_t5_ibon"
        assert condition["ttl"] == 15
        assert condition["common_reward"] is True
        assert condition["cache_mode"] == "read_write"
        assert condition["use_cuda"] is False
        assert condition["greedy_evaluation_provider_calls"] == 0
        assert condition["t_max"] == 1500000


def test_every_condition_gets_its_own_cache():
    matrix = design.design_matrix(2000000, 3500, 1.0)
    assert len({c["cache_path"] for c in matrix}) == 6


def test_a_matrix_with_a_differing_cap_is_rejected():
    matrix = design.design_matrix(2000000, 3500, 1.0)
    matrix[0]["live_call_cap"] = 9999
    problems = design.verify_matrix(matrix)
    assert any("live_call_cap" in p for p in problems)


def test_a_matrix_containing_a_held_out_seed_is_rejected():
    matrix = design.design_matrix(2000000, 3500, 1.0)
    matrix[0]["seed"] = 99
    problems = design.verify_matrix(matrix)
    assert any("held-out" in p or "development" in p for p in problems)


# ==========================================================================
# 9. the coverage rule and unresolved paths
# ==========================================================================


def test_the_minimum_guidance_coverage_is_ninety_five_percent():
    assert design.MIN_GUIDANCE_COVERAGE == 0.95


def test_a_condition_below_the_coverage_floor_is_detectable(tmp_path):
    decisions = ([_decision(1000 + i, "d%d" % i) for i in range(90)]
                 + [_decision(9000 + i, "e%d" % i, outcome="budget_exhausted",
                              attempts=0, activated=0) for i in range(10)])
    guidance = [_guidance(1000 + i) for i in range(100)]
    directory = _write_condition(str(tmp_path), "beta_0p05", decisions, guidance)
    demand = design.condition_demand("beta_0p05", 0.05, directory)
    assert demand["guidance_coverage"] == pytest.approx(0.90)
    assert demand["guidance_coverage"] < design.MIN_GUIDANCE_COVERAGE


# ==========================================================================
# 10. the script is offline by construction
# ==========================================================================


def test_the_design_script_makes_no_network_call(tmp_path, monkeypatch,
                                                 good_baseline):
    """Every connect entry point is poisoned; the script must still complete."""
    def denied(*args, **kwargs):
        raise AssertionError("the design script attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)

    csv_path, sacred = good_baseline
    root = str(tmp_path / "pilot")
    for name, beta in zip(design.PILOT_CONDITIONS, design.PILOT_BETAS):
        _write_condition(root, name,
                         [_decision(1000 + i, "%s-%d" % (name, i))
                          for i in range(20)],
                         [_guidance(1000 + i) for i in range(20)])

    from types import SimpleNamespace

    result = design.build_design(SimpleNamespace(
        baseline_csv=csv_path, baseline_sacred=sacred, pilot_root=root,
        floor_return=None))
    assert result["network_calls"] == 0
    assert result["training_runs"] == 0
    assert result["final_seeds_inspected"] == 0
    assert result["design_valid"] is True
    assert len(result["condition_matrix"]) == 6


def test_the_script_imports_no_network_or_provider_module():
    source = open(design.__file__).read()
    for forbidden in ("import requests", "import httpx", "from openai",
                      "import openai", "subprocess", "OPENAI_API_KEY",
                      "openai.env"):
        assert forbidden not in source, forbidden


def test_the_script_runs_no_training():
    source = open(design.__file__).read()
    for forbidden in ("main.py", "Popen", "os.system", "run_condition"):
        assert forbidden not in source, forbidden
