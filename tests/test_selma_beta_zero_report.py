"""SELMA Stage D: the beta-zero measurement analysis, on synthetic logs.

Every assertion here runs against logs this test writes, so the analysis is
verified without a training run and without the network.

Run:  PYTHONPATH=src pytest tests/test_selma_beta_zero_report.py -v
"""

import csv
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))

import selma_beta_zero_measurement_report as report_mod  # noqa: E402
from selma.guidance import GuidanceCsvWriter, SelmaDecisionCsvWriter  # noqa: E402
from selma.pricing import NO_PRICE_TABLE, PriceTable  # noqa: E402

TABLE = PriceTable("2026-08-17", 0.20, 0.02, 1.25)
MODEL = report_mod.EXPECTED_MODEL
CONDITION = "lbforaging:Foraging-8x8-2p-2f-coop-v3"


def _decision(t_env, *, episode_id, decision="multiple", digest="", cache="miss",
              attempts=1, retries=0, outcome="activated", activated=1,
              plan_id="plan_0", valid=1, physical=1, tokens_in=800,
              tokens_cached=0, tokens_out=25, model=MODEL, latency=1.0,
              step=3, plans=12):
    return {
        "t_env": t_env, "condition": CONDITION, "seed": 91,
        "episode_id": episode_id, "episode_step": step, "greedy_eval": 0,
        "beta": 0.0, "event_id": "ep{}-t{}".format(episode_id, step),
        "decision": decision, "candidate_plan_count": plans,
        "recipient_count": 2, "key_digest": digest, "cache_result": cache,
        "provider_considered": 1 if decision == "multiple" else 0,
        "live_attempts": attempts, "retries": retries, "outcome": outcome,
        "selected_plan_id": plan_id, "validator_valid": valid,
        "validator_reason": "", "physically_valid": physical,
        "activated": activated, "tokens_input": tokens_in,
        "tokens_cached_input": tokens_cached, "tokens_output": tokens_out,
        "tokens_reasoning": 0, "returned_model": model, "latency_s": latency,
        "request_id": "req", "detector_events_this_transition": 1,
    }


def _guidance(t_env, *, episode_id, step, raw=0.0, unscaled=-0.01, terminal=0,
              served="", detector_events=1):
    row = {c: "" for c in GuidanceCsvWriter.COLUMNS}
    row.update({
        "t_env": t_env, "condition": CONDITION, "seed": 91,
        "episode_id": episode_id, "episode_step": step, "condition_checked": 1,
        "query_triggered": 0, "queries": 0, "beta": 0.0, "gamma": 0.99,
        "raw_team_reward": raw, "training_reward": raw,
        "phi_before": 0.0, "potential_after_real": 0.0,
        "phi_after_effective": 0.0, "pbrs_term_unscaled": unscaled,
        "shaping_contribution": 0.0, "stored_terminated": terminal,
        "terminal_cause": "true_terminal" if terminal else "",
        "guidance_active": 0, "guidance_transition": "", "guidance_age": "",
        "transitions_served": served, "expiry_reason": "",
        "provider": "openai", "provider_valid": 0, "fallback_reason": "",
        "detector_events": detector_events, "decisions": 0,
        "zero_plan_decisions": 0, "singleton_decisions": 0,
        "multiple_plan_decisions": 0, "greedy_eval": 0,
        "query_tokens_prompt": "n/a", "query_tokens_cached_prompt": "n/a",
        "query_tokens_completion": "n/a", "query_latency_s": "n/a",
        "query_cost_usd": "n/a", "query_returned_model": "",
        "provider_calls": 0, "total_queries": 0, "activations": 0,
        "replacements": 0, "expiries": 0, "rejected": 0,
        "terminal_suppressed_events": 0, "event_id": "", "subgoal_id": "",
        "trigger_agent_ids": "", "recipient_agent_ids": "", "target_food_id": "",
        "selected_plan_id": "", "phi_components": "",
    })
    return row


def _write(path, columns, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


@pytest.fixture
def logs(tmp_path):
    """A synthetic 200k run: 40 questions, 10 distinct, spread over 4 blocks."""
    decisions, guidance = [], []
    digests = ["d{:02d}".format(i) * 8 for i in range(10)]
    episode = 0
    for block, (low, _high) in enumerate(report_mod.BLOCKS):
        for i in range(10):
            episode += 1
            t_env = low + i * 4000 + 1
            digest = digests[(block * 3 + i) % 10]
            first_time = not any(d["key_digest"] == digest for d in decisions)
            decisions.append(_decision(
                t_env, episode_id=episode, digest=digest,
                cache="miss" if first_time else "hit",
                attempts=1 if first_time else 0,
                outcome="activated"))
        # two skips per block
        decisions.append(_decision(low + 45000, episode_id=episode,
                                   decision="zero_plan", digest="", cache="",
                                   attempts=0, outcome="no_valid_joint_plan",
                                   activated=0, plan_id="", valid=0, physical=0,
                                   tokens_in=0, tokens_out=0, model="",
                                   latency="", plans=0))
        decisions.append(_decision(low + 46000, episode_id=episode,
                                   decision="singleton", digest="", cache="",
                                   attempts=0, outcome="singleton_no_choice",
                                   activated=0, plan_id="", valid=0, physical=0,
                                   tokens_in=0, tokens_out=0, model="",
                                   latency="", plans=1))

    for t_env in range(1, report_mod.EXPECTED_T_MAX + 1, 500):
        guidance.append(_guidance(t_env, episode_id=t_env // 50, step=t_env % 50,
                                  terminal=1 if t_env % 5000 == 1 else 0))
    guidance.append(_guidance(report_mod.EXPECTED_T_MAX, episode_id=9999, step=49))

    decision_path = str(tmp_path / "decisions.csv")
    guidance_path = str(tmp_path / "guidance.csv")
    _write(decision_path, SelmaDecisionCsvWriter.COLUMNS, decisions)
    _write(guidance_path, GuidanceCsvWriter.COLUMNS, guidance)
    return SimpleNamespace(decision=decision_path, guidance=guidance_path)


def _report(logs, price_table=TABLE, exit_code=0):
    decisions = report_mod.read_csv(logs.decision)
    guidance = report_mod.read_csv(logs.guidance)
    args = SimpleNamespace(exit_code=exit_code)
    return report_mod.build_report(decisions, guidance, price_table, args)


# ==========================================================================
# A. run integrity
# ==========================================================================


def test_run_integrity_reports_completion_and_finds_no_corruption(logs):
    integrity = _report(logs)["A_run_integrity"]
    assert integrity["final_t_env"] >= report_mod.EXPECTED_T_MAX
    assert integrity["seeds"] == ["91"]
    assert integrity["conditions"] == [CONDITION]
    assert integrity["non_finite_fields"] == []
    assert integrity["reward_reconciliation_errors"] == 0
    assert integrity["episodes"] > 0


def test_a_reward_mismatch_is_detected(logs, tmp_path):
    rows = report_mod.read_csv(logs.guidance)
    rows[5]["training_reward"] = "1.0"
    rows[5]["raw_team_reward"] = "0.0"
    path = str(tmp_path / "bad.csv")
    _write(path, GuidanceCsvWriter.COLUMNS, rows)
    integrity = report_mod.run_integrity(report_mod.read_csv(path), [])
    assert integrity["reward_reconciliation_errors"] == 1


def test_a_non_finite_value_is_detected(logs, tmp_path):
    rows = report_mod.read_csv(logs.guidance)
    rows[3]["pbrs_term_unscaled"] = "nan"
    path = str(tmp_path / "nan.csv")
    _write(path, GuidanceCsvWriter.COLUMNS, rows)
    integrity = report_mod.run_integrity(report_mod.read_csv(path), [])
    assert "pbrs_term_unscaled" in integrity["non_finite_fields"]


# ==========================================================================
# B. funnel
# ==========================================================================


def test_the_funnel_partitions_every_decision(logs):
    funnel = _report(logs)["B_funnel"]
    total = (funnel["zero_plan_events"] + funnel["singleton_events"]
             + funnel["multiple_plan_events"] + funnel["missing_position_events"])
    assert total == funnel["decisions_logged"]
    assert funnel["multiple_plan_events"] == 40
    assert funnel["zero_plan_events"] == 4
    assert funnel["singleton_events"] == 4
    assert funnel["events_eligible_for_an_llm_decision"] == 40
    assert funnel["guidance_activations"] == 40
    assert funnel["activation_rate_of_eligible"] == 1.0


# ==========================================================================
# C. cache
# ==========================================================================


def test_distinct_semantic_keys_are_counted_from_the_digests(logs):
    cache = _report(logs)["C_cache"]
    assert cache["distinct_semantic_keys"] == 10
    assert cache["questions_with_a_digest"] == 40
    assert cache["repeat_questions"] == 30
    assert cache["ideal_replay_hits"] == 30
    assert cache["ideal_replay_hit_rate"] == pytest.approx(0.75)


def test_top_keys_are_reported_as_digest_prefixes_only(logs):
    cache = _report(logs)["C_cache"]
    assert cache["top_keys"]
    for entry in cache["top_keys"]:
        assert len(entry["digest_prefix"]) == 12
        assert set(entry) == {"digest_prefix", "uses"}
    assert 0.0 < cache["top_key_share"] <= 1.0


def test_the_cumulative_distinct_curve_is_monotone(logs):
    curve = _report(logs)["C_cache"]["cumulative_distinct_curve"]
    values = [v for _i, v in curve]
    assert values == sorted(values)
    assert values[-1] == 10


def test_cross_episode_reuse_is_measured(logs):
    assert _report(logs)["C_cache"]["keys_reused_across_episodes"] > 0


def test_the_ideal_rate_is_computable_when_the_budget_stopped_the_cache(tmp_path):
    """Digests are recorded even for budget-exhausted events, so the ideal
    replay rate is measurable when the observed hit rate is zero."""
    rows = [_decision(i * 100, episode_id=i, digest="abc" * 21 + "d",
                      cache="miss", attempts=0, outcome="budget_exhausted",
                      activated=0, valid=0, physical=0, tokens_in=0,
                      tokens_out=0, model="", latency="")
            for i in range(5)]
    cache = report_mod.cache_behaviour(rows)
    assert cache["cache_hits"] == 0
    assert cache["observed_hit_rate"] == 0.0
    assert cache["distinct_semantic_keys"] == 1
    assert cache["ideal_replay_hit_rate"] == pytest.approx(0.8)


# ==========================================================================
# D. time blocks
# ==========================================================================


def test_every_block_is_reported_with_new_key_and_repeat_rates(logs):
    blocks = _report(logs)["D_time_blocks"]
    assert [b["block"] for b in blocks] == ["0-50k", "50-100k", "100-150k",
                                            "150-200k"]
    assert sum(b["new_semantic_keys"] for b in blocks) == 10
    assert blocks[0]["new_semantic_keys"] >= blocks[-1]["new_semantic_keys"]
    for block in blocks:
        assert block["multiple_plan_decisions"] == 10
        assert block["detector_events_per_1k_steps"] is not None
        assert block["cumulative_distinct_keys"] <= 10
    assert blocks[-1]["cumulative_distinct_keys"] == 10


# ==========================================================================
# E. provider
# ==========================================================================


def test_provider_stats_and_cost_use_the_dated_table(logs):
    provider = _report(logs)["E_provider"]
    assert provider["events_that_called"] == 10
    assert provider["request_attempts_including_retries"] == 10
    assert provider["returned_model_counts"] == {MODEL: 10}
    assert provider["physically_invalid_activations"] == 0
    expected = (10 * 800 * 0.20 + 10 * 25 * 1.25) / 1_000_000.0
    assert provider["cost_usd"] == pytest.approx(expected)
    assert "2026-08-17" in provider["cost_methodology"]


def test_cost_is_not_calculated_without_a_dated_table(logs):
    provider = _report(logs, price_table=NO_PRICE_TABLE)["E_provider"]
    assert provider["cost_usd"] is None
    assert "NOT calculated" in provider["cost_methodology"]


def test_cached_input_is_billed_at_the_cached_rate(logs, tmp_path):
    rows = [_decision(1, episode_id=1, digest="x" * 64, tokens_in=1000,
                      tokens_cached=400, tokens_out=200)]
    provider = report_mod.provider_stats(rows, TABLE)
    expected = (600 * 0.20 + 400 * 0.02 + 200 * 1.25) / 1_000_000.0
    assert provider["cost_usd"] == pytest.approx(expected)


def test_a_substituted_model_is_visible(logs, tmp_path):
    rows = [_decision(1, episode_id=1, digest="x" * 64, model="gpt-other-2020-01-01")]
    provider = report_mod.provider_stats(rows, TABLE)
    assert provider["returned_model_counts"] == {"gpt-other-2020-01-01": 1}


# ==========================================================================
# F. reward / PBRS
# ==========================================================================


def test_beta_zero_reward_identities_are_verified(logs):
    rewards = _report(logs)["F_reward_pbrs"]
    assert rewards["beta_is_zero_everywhere"]
    assert rewards["shaping_is_exactly_zero"]
    assert rewards["training_reward_equals_raw_reward"]
    assert rewards["reward_mismatches"] == 0
    assert rewards["non_zero_unscaled_pbrs_terms"] > 0, "guidance really ran"
    assert rewards["terminal_phi_after_effective_all_zero"]
    assert rewards["greedy_eval_rows_in_decision_log"] == 0
    assert rewards["greedy_eval_rows_in_guidance_log"] == 0


def test_a_non_zero_beta_row_fails_the_beta_check(logs, tmp_path):
    rows = report_mod.read_csv(logs.guidance)
    rows[2]["beta"] = "0.5"
    path = str(tmp_path / "beta.csv")
    _write(path, GuidanceCsvWriter.COLUMNS, rows)
    rewards = report_mod.reward_checks(report_mod.read_csv(path), [])
    assert not rewards["beta_is_zero_everywhere"]


def test_a_greedy_evaluation_row_is_caught(logs, tmp_path):
    rows = [_decision(1, episode_id=1, digest="x" * 64)]
    rows[0]["greedy_eval"] = 1
    rewards = report_mod.reward_checks([], rows)
    assert rewards["greedy_eval_rows_in_decision_log"] == 1


# ==========================================================================
# G. projections
# ==========================================================================


def test_projections_report_a_labelled_range_not_a_single_number(logs):
    proj = _report(logs)["G_projections"]
    assert proj["measured"]["distinct_multiple_plan_questions"] == 10
    assert "NAIVE" in proj["naive_linear_2m"]["label"]
    assert "Tail projection" in proj["final_block_rate_2m"]["label"]
    low = proj["conservative_range_2m"]["calls_low"]
    high = proj["conservative_range_2m"]["calls_high"]
    assert low <= high
    assert proj["five_seeds_isolated_caches"]["calls_high"] == high * 5
    assert "not a claim" in proj["limitation"]
    assert proj["provisional_2000_call_cap_sufficient"] in (True, False)


def test_the_naive_projection_scales_linearly(logs):
    proj = _report(logs)["G_projections"]
    assert proj["naive_linear_2m"]["calls"] == 100      # 10 keys x 10


# ==========================================================================
# H. gates
# ==========================================================================


def test_a_clean_run_passes_every_gate(logs):
    report = _report(logs)
    failed = [g for g in report["H_gates"] if not g["passed"]]
    assert failed == [], failed
    assert report["all_gates_passed"]


def test_a_short_run_fails_the_t_env_gate(logs, tmp_path):
    rows = [r for r in report_mod.read_csv(logs.guidance)
            if report_mod._int(r["t_env"]) < 100000]
    path = str(tmp_path / "short.csv")
    _write(path, GuidanceCsvWriter.COLUMNS, rows)
    args = SimpleNamespace(exit_code=0)
    report = report_mod.build_report(report_mod.read_csv(logs.decision),
                                     report_mod.read_csv(path), TABLE, args)
    failed = [g["gate"] for g in report["H_gates"] if not g["passed"]]
    assert any("t_env reached" in g for g in failed)
    assert not report["all_gates_passed"]


def test_a_nonzero_exit_code_fails(logs):
    report = _report(logs, exit_code=1)
    failed = [g["gate"] for g in report["H_gates"] if not g["passed"]]
    assert any("exited successfully" in g for g in failed)


def test_calls_on_cache_hits_or_skips_fail_their_gates(logs, tmp_path):
    rows = report_mod.read_csv(logs.decision)
    for row in rows:
        if row["cache_result"] == "hit":
            row["live_attempts"] = "1"
            break
    path = str(tmp_path / "hitcall.csv")
    _write(path, SelmaDecisionCsvWriter.COLUMNS, rows)
    args = SimpleNamespace(exit_code=0)
    report = report_mod.build_report(report_mod.read_csv(path),
                                     report_mod.read_csv(logs.guidance),
                                     TABLE, args)
    failed = [g["gate"] for g in report["H_gates"] if not g["passed"]]
    assert any("cache hits" in g for g in failed)


def test_an_over_cap_attempt_count_fails(logs, tmp_path):
    rows = report_mod.read_csv(logs.decision)
    rows[0]["live_attempts"] = str(report_mod.EXPECTED_ATTEMPT_CAP + 5)
    path = str(tmp_path / "over.csv")
    _write(path, SelmaDecisionCsvWriter.COLUMNS, rows)
    args = SimpleNamespace(exit_code=0)
    report = report_mod.build_report(report_mod.read_csv(path),
                                     report_mod.read_csv(logs.guidance),
                                     TABLE, args)
    failed = [g["gate"] for g in report["H_gates"] if not g["passed"]]
    assert any("live attempts" in g for g in failed)


def test_an_invalid_physical_activation_fails(logs, tmp_path):
    rows = report_mod.read_csv(logs.decision)
    for row in rows:
        if row["activated"] == "1":
            row["physically_valid"] = "0"
            break
    path = str(tmp_path / "invalid.csv")
    _write(path, SelmaDecisionCsvWriter.COLUMNS, rows)
    args = SimpleNamespace(exit_code=0)
    report = report_mod.build_report(report_mod.read_csv(path),
                                     report_mod.read_csv(logs.guidance),
                                     TABLE, args)
    failed = [g["gate"] for g in report["H_gates"] if not g["passed"]]
    assert any("physically invalid" in g for g in failed)


def test_the_report_declares_what_it_is_not(logs):
    report = _report(logs)
    assert report["not_a_performance_result"] is True
    assert report["not_beta_selection"] is True
    assert report["beta"] == 0.0


def test_the_cli_returns_nonzero_when_a_gate_fails(logs, tmp_path):
    code = report_mod.main([
        "--decision-log", logs.decision, "--guidance-log", logs.guidance,
        "--report", str(tmp_path / "r.json"), "--exit-code", "1",
        "--price-table-date", "2026-08-17", "--price-input", "0.20",
        "--price-cached-input", "0.02", "--price-output", "1.25",
    ])
    assert code == 1


def test_the_cli_returns_zero_on_a_clean_run(logs, tmp_path):
    code = report_mod.main([
        "--decision-log", logs.decision, "--guidance-log", logs.guidance,
        "--report", str(tmp_path / "ok.json"), "--exit-code", "0",
        "--price-table-date", "2026-08-17", "--price-input", "0.20",
        "--price-cached-input", "0.02", "--price-output", "1.25",
    ])
    assert code == 0
    assert os.path.exists(str(tmp_path / "ok.json"))
