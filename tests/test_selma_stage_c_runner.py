"""SELMA Stage C: end-to-end behaviour through the real EpisodeRunner.

Drives the real runner, MAC, action selector and QMIX learner in process (the
same harness ``tests/test_selma_shadow_equivalence.py`` uses for Stage B) and
checks the properties that only exist once everything is wired together:
``beta=0`` equivalence, greedy-evaluation isolation, the guidance log schema and
its exact arithmetic reconciliation, and construction-time rejection of every
illegal mode combination.

Run:  PYTHONPATH=src pytest tests/test_selma_stage_c_runner.py -v
"""

import csv
import os
import sys

import pytest
import torch as th

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from selma.guidance import GuidanceCsvWriter  # noqa: E402
from selma.modes import FROZEN_ACTIVE_VARIANT, SelmaConfigError  # noqa: E402
from test_selma_shadow_equivalence import build_args, build_stack, run_trace  # noqa: E402

# Development seed. Deliberately NOT one of the final experimental seeds 1-5.
DEV_SEED = 4242

# Detector events are rare by design (~0.23 per episode for the frozen variant),
# so a run has to be long enough to contain some. 40 episodes yields a handful of
# activations, replacements and TTL expiries at this seed.
EPISODES = 40


def active_overrides(tmp_path, beta=0.0, name="guidance.csv"):
    return {
        "seed": DEV_SEED,
        "selma_mode": "active",
        "selma_provider": "deterministic_mock",
        "selma_variant_id": FROZEN_ACTIVE_VARIANT,
        "selma_beta": beta,
        "selma_guidance_log_csv": str(tmp_path / name),
    }


def read_rows(path):
    with open(path) as handle:
        return list(csv.DictReader(handle))


# Each configuration is trained once per module: these runs are the expensive
# part of the file, and every assertion below is a read of their output.


@pytest.fixture(scope="module")
def off_run():
    return run_trace(n_episodes=EPISODES, seed=DEV_SEED, selma_mode="off")


@pytest.fixture(scope="module")
def beta_zero_run(tmp_path_factory):
    path = tmp_path_factory.mktemp("beta0")
    trace, params, runner = run_trace(
        n_episodes=EPISODES, **active_overrides(path, beta=0.0))
    return trace, params, runner, path / "guidance.csv"


@pytest.fixture(scope="module")
def shaped_run(tmp_path_factory):
    path = tmp_path_factory.mktemp("shaped")
    trace, params, runner = run_trace(
        n_episodes=EPISODES, **active_overrides(path, beta=0.25))
    return trace, params, runner, path / "guidance.csv"


# ==========================================================================
# construction-time mode contract
# ==========================================================================


def test_runner_rejects_an_unknown_mode():
    with pytest.raises(SelmaConfigError):
        build_stack(build_args(selma_mode="active_shadow"))


def test_runner_rejects_a_provider_in_shadow_mode(tmp_path):
    with pytest.raises(SelmaConfigError):
        build_stack(build_args(
            selma_mode="shadow", selma_provider="deterministic_mock",
            selma_shadow_csv=str(tmp_path / "s.csv"),
        ))


def test_runner_rejects_active_mode_without_an_explicit_variant():
    with pytest.raises(SelmaConfigError):
        build_stack(build_args(
            selma_mode="active", selma_provider="deterministic_mock",
        ))


def test_runner_rejects_active_mode_with_common_reward_false(tmp_path):
    with pytest.raises(SelmaConfigError) as exc:
        build_stack(build_args(common_reward=False, **active_overrides(tmp_path)))
    assert "common_reward" in str(exc.value)


def test_runner_rejects_an_unsupported_provider():
    with pytest.raises(SelmaConfigError):
        build_stack(build_args(
            selma_mode="active", selma_provider="a-live-llm",
            selma_variant_id=FROZEN_ACTIVE_VARIANT,
        ))


def test_active_mode_bank_holds_only_the_frozen_variant(tmp_path):
    runner, _, _, _ = build_stack(build_args(**active_overrides(tmp_path)))
    assert [v.variant_id for v in runner.selma_bank.variants] == [FROZEN_ACTIVE_VARIANT]
    assert runner.selma_guidance is not None
    assert runner.selma_guidance.variant_id == FROZEN_ACTIVE_VARIANT


def test_shadow_mode_still_builds_the_full_calibration_bank(tmp_path):
    runner, _, _, _ = build_stack(build_args(
        selma_mode="shadow", selma_shadow_csv=str(tmp_path / "s.csv")))
    assert len(runner.selma_bank.variants) == 48
    assert runner.selma_guidance is None


def test_off_mode_constructs_nothing():
    runner, _, _, _ = build_stack(build_args())
    assert runner.selma_bank is None and runner.selma_guidance is None


def test_common_reward_false_is_unaffected_when_selma_is_off():
    """Ordinary non-SELMA behaviour must be preserved for common_reward=False.

    QMIX itself requires a common reward (``src/run.py`` asserts it), so what is
    checked here is that SELMA adds no new restriction: construction succeeds and
    every SELMA component stays inert. Only *active* mode rejects it.
    """
    from runners import REGISTRY as r_REGISTRY
    from test_selma_shadow_equivalence import _NullLogger

    for mode, extra in [("off", {}), ("shadow", {"selma_shadow_csv": ""})]:
        args = build_args(common_reward=False, selma_mode=mode, **extra)
        runner = r_REGISTRY[args.runner](args=args, logger=_NullLogger())
        assert runner.selma_guidance is None
        assert runner.selma_config.mode == mode
        assert runner.selma_config.guidance_enabled is False


# ==========================================================================
# beta = 0 equivalence through the real learner
# ==========================================================================


def test_active_beta_zero_is_bit_identical_to_off(off_run, beta_zero_run):
    off_trace, off_params, _ = off_run
    active_trace, active_params, active_runner, _csv = beta_zero_run

    assert off_trace == active_trace, "trajectory/replay tensors must be identical"
    assert off_params == active_params, "learner parameters must be identical"
    assert active_runner.selma_guidance.total_shaping_reward == 0.0


def test_active_beta_zero_still_runs_the_guidance_machinery(beta_zero_run):
    """beta=0 must zero the reward term only -- not the lifecycle."""
    _trace, _params, runner, guidance_csv = beta_zero_run
    guidance = runner.selma_guidance
    assert guidance.provider_call_count > 0, "the provider was never exercised"
    assert guidance.activation_count > 0
    rows = read_rows(guidance_csv)
    assert any(float(r["pbrs_term_unscaled"]) != 0.0 for r in rows), (
        "the unscaled potential difference must still be measured at beta=0"
    )
    assert all(float(r["shaping_contribution"]) == 0.0 for r in rows)
    assert all(r["raw_team_reward"] == r["training_reward"] for r in rows)


def test_beta_positive_actually_changes_stored_rewards(off_run, shaped_run):
    off_trace, _, _ = off_run
    shaped_trace, _, runner, _csv = shaped_run
    assert runner.selma_guidance.activation_count > 0
    assert off_trace != shaped_trace, "beta > 0 must reach the stored reward"


# ==========================================================================
# greedy-evaluation isolation
# ==========================================================================


def test_greedy_evaluation_makes_zero_queries_and_no_shaping(tmp_path):
    args = build_args(**active_overrides(tmp_path, beta=0.25))
    runner, _, _, _ = build_stack(args)
    for _ in range(EPISODES):
        runner.run(test_mode=False)
    assert runner.selma_guidance.activation_count > 0, (
        "the check is vacuous unless training actually activated guidance"
    )
    baseline = (
        runner.selma_guidance.provider_call_count,
        runner.selma_guidance.query_count,
        runner.selma_guidance.activation_count,
        runner.selma_guidance.total_shaping_reward,
    )
    rows_after_train = len(read_rows(tmp_path / "guidance.csv"))

    for _ in range(6):
        runner.run(test_mode=True)

    assert (
        runner.selma_guidance.provider_call_count,
        runner.selma_guidance.query_count,
        runner.selma_guidance.activation_count,
        runner.selma_guidance.total_shaping_reward,
    ) == baseline, "greedy evaluation touched the guidance controller"
    assert len(read_rows(tmp_path / "guidance.csv")) == rows_after_train
    assert runner.selma_guidance.active is None


# ==========================================================================
# the guidance log
# ==========================================================================


def test_guidance_log_has_the_full_c7_schema(shaped_run):
    _t, _p, _r, guidance_csv = shaped_run
    with open(guidance_csv) as handle:
        header = handle.readline().strip().split(",")
    assert header == list(GuidanceCsvWriter.COLUMNS)
    required = {
        "condition_checked", "query_triggered", "trigger_agent_ids", "target_food_id",
        "subgoal_id", "provider", "fallback_reason", "terminal_cause",
        "query_tokens_prompt", "query_tokens_completion", "query_latency_s",
        "query_cost_usd", "phi_before", "phi_after_effective", "phi_components",
        "pbrs_term_unscaled", "beta", "shaping_contribution", "raw_team_reward",
        "training_reward",
    }
    assert required <= set(header), sorted(required - set(header))


def test_guidance_log_arithmetic_reconciles_exactly(shaped_run):
    beta = 0.25
    _t, _p, _r, guidance_csv = shaped_run
    rows = read_rows(guidance_csv)
    assert rows
    for row in rows:
        unscaled = float(row["pbrs_term_unscaled"])
        contribution = float(row["shaping_contribution"])
        gamma = float(row["gamma"])
        assert float(row["beta"]) == beta
        # beta applied exactly once
        assert abs(contribution - beta * unscaled) <= 1e-12
        # the unscaled term is the potential difference, beta-free
        expected = gamma * float(row["phi_after_effective"]) - float(row["phi_before"])
        assert abs(unscaled - expected) <= 1e-12
        # and the stored reward is raw + contribution
        assert abs(
            float(row["training_reward"])
            - (float(row["raw_team_reward"]) + contribution)
        ) <= 1e-12


def test_guidance_log_mock_usage_fields_are_not_applicable(shaped_run):
    _t, _p, _r, guidance_csv = shaped_run
    rows = read_rows(guidance_csv)
    for row in rows:
        for field in ("query_tokens_prompt", "query_tokens_completion",
                      "query_latency_s", "query_cost_usd"):
            assert row[field] == "n/a", f"{field} must be n/a, never a fabricated 0"


def test_no_query_is_ever_recorded_on_a_stored_terminal_transition(shaped_run):
    _t, _p, _r, guidance_csv = shaped_run
    rows = read_rows(guidance_csv)
    terminal = [r for r in rows if r["stored_terminated"] == "1"]
    assert terminal, "no terminal transitions were logged"
    assert all(r["query_triggered"] == "0" for r in terminal)
    assert all(float(r["phi_after_effective"]) == 0.0 for r in terminal)
    assert all(r["terminal_cause"] in ("true_terminal", "time_limit") for r in terminal)
    assert all(r["terminal_cause"] == "" for r in rows if r["stored_terminated"] == "0")


def test_terminal_cause_distinguishes_the_horizon_from_a_cleared_grid(shaped_run):
    """This LBF wrapper reports a time limit as `terminated`, so the cause is
    derived from the horizon and the remaining food, not from the env flags."""
    _t, _p, _r, guidance_csv = shaped_run
    horizon = build_args().env_args["time_limit"]
    rows = read_rows(guidance_csv)
    terminal = [r for r in rows if r["stored_terminated"] == "1"]
    assert terminal
    for row in terminal:
        at_horizon = int(row["episode_step"]) + 1 >= horizon
        expected = "time_limit" if at_horizon else "true_terminal"
        assert row["terminal_cause"] == expected, row


def test_boundary_identity_holds_across_the_logged_run(shaped_run):
    """phi_after_effective(t) == phi_before(t+1) within each episode."""
    _t, _p, _r, guidance_csv = shaped_run
    rows = read_rows(guidance_csv)
    checked = 0
    for previous, current in zip(rows, rows[1:]):
        if int(current["episode_step"]) == 0:
            continue  # episode boundary: the next episode starts from 0.0
        assert abs(
            float(previous["phi_after_effective"]) - float(current["phi_before"])
        ) <= 1e-12
        checked += 1
    assert checked > 0


def test_every_episode_starts_and_ends_at_zero_potential(shaped_run):
    _t, _p, _r, guidance_csv = shaped_run
    rows = read_rows(guidance_csv)
    for row in rows:
        if int(row["episode_step"]) == 0:
            assert float(row["phi_before"]) == 0.0
        if row["stored_terminated"] == "1":
            assert float(row["phi_after_effective"]) == 0.0


def test_guidance_log_records_activation_and_expiry_boundaries(shaped_run):
    _t, _p, _r, guidance_csv = shaped_run
    rows = read_rows(guidance_csv)
    transitions = {r["guidance_transition"] for r in rows}
    assert "activation" in transitions and "ongoing" in transitions
    assert "expiry" in transitions
    for row in rows:
        if row["guidance_transition"] == "expiry":
            assert row["expiry_reason"] in ("ttl", "target_removed", "terminal")
            assert float(row["phi_after_effective"]) == 0.0


def test_stored_rewards_are_finite_under_shaping(shaped_run):
    trace, _p, _r, _csv = shaped_run
    for episode in trace:
        values = th.tensor(episode["reward_values"], dtype=th.float64)
        assert th.isfinite(values).all()
