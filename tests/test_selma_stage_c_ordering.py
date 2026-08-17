"""SELMA Stage C: mode contract, dynamic-potential ordering and PBRS accounting.

These are the *regression* tests for the Stage C correction pass. Every test in
this module was written against the audited implementation and required a code
change to pass.

The lifecycle tests do not re-implement the runner's per-transition sequence.
They call :meth:`GuidanceController.apply_transition` -- the single shared code
path the real ``EpisodeRunner`` uses -- so the ordering under test is by
construction the ordering that runs in training. ``test_runner_uses_the_shared_
transition_path`` pins that invariant.

Run:  PYTHONPATH=src pytest tests/test_selma_stage_c_ordering.py -v
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from selma.guidance import (  # noqa: E402
    PHI_BOUNDARY_TOLERANCE,
    TTL_STEPS,
    GuidanceController,
    TransitionRecord,
    pbrs_term_unscaled,
)
from selma.modes import (  # noqa: E402
    FROZEN_ACTIVE_VARIANT,
    MODE_ACTIVE,
    MODE_OFF,
    MODE_SHADOW,
    SelmaConfigError,
    validate_selma_config,
)
from selma.provider import (  # noqa: E402
    REASON_MISSING_AGENT_POSITION,
    DeterministicMockProvider,
    MissingAgentPositionError,
    build_request,
)

GRID = (8, 8)
FOOD = (4, 4)
# Development seed. Deliberately NOT one of the final experimental seeds 1-5.
DEV_SEED = 4242

# Telescoping tolerance. A trajectory of T <= 64 transitions accumulates T
# double-precision additions of terms bounded by 2*|beta|; the worst-case
# relative error is a few units in the last place of the largest partial sum
# (|beta| * 2), i.e. O(1e-15). 1e-12 is three orders of magnitude of headroom
# and still far tighter than any real shaping magnitude.
TELESCOPING_TOLERANCE = 1e-12


# ==========================================================================
# helpers
# ==========================================================================


def event_row(event_id="ep0-wf15_m3_t5_ibon-food4,4-t3", variant_id=FROZEN_ACTIVE_VARIANT,
              recipients=(0, 1), cells=((4, 3), (3, 4)), loader_ids=(0,),
              loader_levels=(1,), helper_ids=(1,), helper_levels=(2,), food_level=3):
    """A row with the *production* detector key set (no ``agent_positions``)."""
    return {
        "row_type": "coalesced_event",
        "event_id": event_id,
        "variant_id": variant_id,
        "food_id": (0, FOOD[0], FOOD[1]),
        "food_position": FOOD,
        "food_level": food_level,
        "trigger_agent_ids": loader_ids,
        "recipient_ids": recipients,
        "loader_ids": loader_ids,
        "loader_levels": loader_levels,
        "helper_ids": helper_ids,
        "helper_levels": helper_levels,
        "level_deficit": food_level - sum(loader_levels),
        "required_extra_level": max(0, food_level - sum(loader_levels)),
        "candidate_loading_cells": cells,
        "coalition_feasible": 1,
        "suppression_reason": "",
    }


def controller(beta=1.0, gamma=0.99, provider=None, variant_id=FROZEN_ACTIVE_VARIANT):
    return GuidanceController(
        provider=provider or DeterministicMockProvider(),
        grid_shape=GRID,
        beta=beta,
        gamma=gamma,
        enabled=True,
        variant_id=variant_id,
    )


def replay(ctrl, states, events=None, stored_terminated=True, raw=None,
           terminal_cause="true_terminal"):
    """Replay a complete finite episode through the shared transition path.

    ``states[t]`` is ``(agent_positions, food_positions)`` for ``s_t``; there are
    ``T + 1`` states for ``T`` transitions. Mirrors the runner exactly: the
    carried ``phi_before`` is the previous transition's ``phi_after_effective``,
    and it is cross-checked against a recomputation from ``s_t``.
    """
    events = events or {}
    n = len(states) - 1
    raw = raw if raw is not None else [0.0] * n
    phi_before = 0.0
    records = []
    for t in range(n):
        recomputed = ctrl.potential(*states[t])
        assert abs(recomputed - phi_before) <= PHI_BOUNDARY_TOLERANCE, (
            f"phi_before({t}) carried {phi_before} but recomputes to {recomputed}"
        )
        last = t == n - 1
        record = ctrl.apply_transition(
            step=t,
            raw_team_reward=raw[t],
            phi_before=phi_before,
            post_agent_positions=states[t + 1][0],
            post_food_positions=states[t + 1][1],
            event_rows=events.get(t, ()),
            episode_ended=last,
            stored_terminated=bool(last and stored_terminated),
            terminal_cause=terminal_cause if last else "",
        )
        assert isinstance(record, TransitionRecord)
        records.append(record)
        phi_before = record.phi_after_effective
    return records


def walk(steps, food=FOOD, foods_gone_at=None, start=(0, 0), other=(7, 7)):
    """Agent 0 walks toward the food one cell per transition; agent 1 is static."""
    states = []
    for t in range(steps + 1):
        row = min(food[0], start[0] + t)
        positions = {0: (row, 0), 1: other}
        foods = () if (foods_gone_at is not None and t >= foods_gone_at) else (food,)
        states.append((positions, foods))
    return states


def discounted_shaping(records, gamma):
    return sum(gamma ** t * r.shaping_contribution for t, r in enumerate(records))


def endpoint_identity(records, gamma, beta):
    """beta * (gamma^T * Phi_T - Phi_0) for a complete finite trajectory."""
    phi_0 = records[0].phi_before
    phi_T = records[-1].phi_after_effective
    return beta * (gamma ** len(records) * phi_T - phi_0)


# ==========================================================================
# 1. mode contract
# ==========================================================================


def args_ns(**overrides):
    from types import SimpleNamespace

    base = {
        "selma_mode": MODE_OFF,
        "selma_provider": "none",
        "selma_beta": 0.0,
        "selma_variant_id": "",
        "selma_guidance_log_csv": "",
        "common_reward": True,
        "gamma": 0.99,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.mark.parametrize("mode", ["on", "Shadow", "ACTIVE", "enabled", "true", " off"])
def test_unknown_selma_mode_is_rejected(mode):
    """Unknown modes must fail loudly, never fall through to 'off'."""
    with pytest.raises(SelmaConfigError) as exc:
        validate_selma_config(args_ns(selma_mode=mode))
    assert "selma_mode" in str(exc.value)
    assert MODE_ACTIVE in str(exc.value)  # the message lists the legal modes


def test_off_mode_is_fully_inert():
    cfg = validate_selma_config(args_ns(selma_mode=MODE_OFF))
    assert not cfg.detector_enabled and not cfg.guidance_enabled
    assert cfg.beta == 0.0


@pytest.mark.parametrize("bad", [{"selma_provider": "deterministic_mock"},
                                 {"selma_beta": 0.25}])
def test_off_mode_rejects_provider_or_beta(bad):
    with pytest.raises(SelmaConfigError):
        validate_selma_config(args_ns(selma_mode=MODE_OFF, **bad))


def test_shadow_mode_rejects_a_provider():
    """Stage B shadow is detector-only: a provider there is a configuration error."""
    with pytest.raises(SelmaConfigError) as exc:
        validate_selma_config(
            args_ns(selma_mode=MODE_SHADOW, selma_provider="deterministic_mock")
        )
    assert "shadow" in str(exc.value)


def test_shadow_mode_rejects_nonzero_beta():
    with pytest.raises(SelmaConfigError):
        validate_selma_config(args_ns(selma_mode=MODE_SHADOW, selma_beta=0.1))


def test_shadow_mode_keeps_multi_variant_calibration():
    cfg = validate_selma_config(args_ns(selma_mode=MODE_SHADOW))
    assert cfg.detector_enabled and not cfg.guidance_enabled
    assert cfg.variant_id == ""  # all 48 variants, unchanged Stage B behaviour


def test_active_mode_requires_an_explicit_provider():
    with pytest.raises(SelmaConfigError) as exc:
        validate_selma_config(
            args_ns(selma_mode=MODE_ACTIVE, selma_variant_id=FROZEN_ACTIVE_VARIANT)
        )
    assert "provider" in str(exc.value)


def test_active_mode_requires_an_explicit_frozen_variant():
    with pytest.raises(SelmaConfigError) as exc:
        validate_selma_config(
            args_ns(selma_mode=MODE_ACTIVE, selma_provider="deterministic_mock")
        )
    assert "selma_variant_id" in str(exc.value)


def test_active_mode_rejects_a_non_frozen_variant():
    with pytest.raises(SelmaConfigError):
        validate_selma_config(
            args_ns(selma_mode=MODE_ACTIVE, selma_provider="deterministic_mock",
                    selma_variant_id="wf5_m2_none_iboff")
        )


def test_active_mode_rejects_common_reward_false():
    with pytest.raises(SelmaConfigError) as exc:
        validate_selma_config(
            args_ns(selma_mode=MODE_ACTIVE, selma_provider="deterministic_mock",
                    selma_variant_id=FROZEN_ACTIVE_VARIANT, common_reward=False)
        )
    assert "common_reward" in str(exc.value)


def test_active_mode_allows_beta_zero_for_equivalence_testing():
    cfg = validate_selma_config(
        args_ns(selma_mode=MODE_ACTIVE, selma_provider="deterministic_mock",
                selma_variant_id=FROZEN_ACTIVE_VARIANT, selma_beta=0.0)
    )
    assert cfg.detector_enabled and cfg.guidance_enabled
    assert cfg.beta == 0.0 and cfg.variant_id == FROZEN_ACTIVE_VARIANT


@pytest.mark.parametrize("beta", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_beta_is_rejected(beta):
    with pytest.raises(SelmaConfigError):
        validate_selma_config(
            args_ns(selma_mode=MODE_ACTIVE, selma_provider="deterministic_mock",
                    selma_variant_id=FROZEN_ACTIVE_VARIANT, selma_beta=beta)
        )


def test_unknown_provider_is_rejected_at_configuration_time():
    with pytest.raises(SelmaConfigError):
        validate_selma_config(
            args_ns(selma_mode=MODE_ACTIVE, selma_provider="gpt-please",
                    selma_variant_id=FROZEN_ACTIVE_VARIANT)
        )


# ==========================================================================
# 2. agent positions: no silent (-1,-1)
# ==========================================================================


def test_build_request_requires_agent_positions():
    with pytest.raises(MissingAgentPositionError):
        build_request(event_row(), agent_positions={}, grid_shape=GRID)


def test_build_request_rejects_partial_agent_positions():
    with pytest.raises(MissingAgentPositionError):
        build_request(event_row(), agent_positions={0: (4, 3)},
                      grid_shape=GRID)  # agent 1 missing


def test_build_request_rejects_none_positions():
    with pytest.raises(MissingAgentPositionError):
        build_request(event_row(), agent_positions=None, grid_shape=GRID)


def test_build_request_uses_the_supplied_positions():
    req = build_request(event_row(), agent_positions={0: (4, 3), 1: (6, 6)},
                        grid_shape=GRID)
    assert [ctx.position for ctx in req.agent_context] == [(4, 3), (6, 6)]
    assert all(ctx.position != (-1, -1) for ctx in req.agent_context)


def test_controller_records_missing_positions_as_a_documented_rejection():
    """A missing position must not crash training: reject, log, F_t = 0."""
    ctrl = controller()
    record = ctrl.apply_transition(
        step=1, raw_team_reward=0.0, phi_before=0.0,
        post_agent_positions={0: (4, 3)},  # recipient 1 absent
        post_food_positions=(FOOD,),
        event_rows=[event_row()], episode_ended=False,
        stored_terminated=False, terminal_cause="",
    )
    assert ctrl.active is None
    assert record.fallback_reason == REASON_MISSING_AGENT_POSITION
    assert record.provider_valid == 0
    assert record.pbrs_term_unscaled == 0.0
    assert record.shaping_contribution == 0.0
    assert ctrl.provider_call_count == 0  # never reached the provider


# ==========================================================================
# 3. production detector-row schema fidelity
# ==========================================================================


def real_detector_event_row(seed=DEV_SEED):
    """Produce a coalesced_event through the REAL Stage A + detector path.

    Agent 0 (level 1) repeatedly LOADs a level-3 food it cannot clear alone;
    agents 1 and 2 stay put, so no helper is inbound and the team gate opens.
    Nothing here is hand-written: the row comes from ``detector._add_row``.
    """
    from selma.detector import ShadowDetectorBank, build_variant_grid
    from selma.load_events import LOAD_ACTION, build_selma_step_info

    variants = [v for v in build_variant_grid() if v.variant_id == FROZEN_ACTIVE_VARIANT]
    assert len(variants) == 1
    bank = ShadowDetectorBank(
        variants=variants, n_agents=3, grid_shape=GRID, condition="schema-fidelity",
        seed=seed, rearm_steps=3, cooldown_steps=5, max_events_per_episode=2,
        w_inbound=3,
    )
    bank.reset_episode(0)
    field = np.zeros(GRID, dtype=int)
    field[FOOD[0]][FOOD[1]] = 3
    positions = [(4, 3), (0, 0), (7, 7)]
    levels = [1, 1, 2]
    for step in range(12):
        info = build_selma_step_info(
            episode_id=0, episode_step=step, pre_field=field, positions=positions,
            levels=levels, actions=(LOAD_ACTION, 0, 0), post_field=field,
        )
        seen = len(bank.pending_rows)
        bank.update(step_info=info, raw_team_reward=0.0, t_env=step)
        for row, _step, _fid in bank.pending_rows[seen:]:
            if row["row_type"] == "coalesced_event":
                return row, positions
    raise AssertionError("the real detector produced no coalesced_event")


def test_real_detector_row_is_produced_by_the_production_path():
    row, _ = real_detector_event_row()
    assert row["row_type"] == "coalesced_event"
    assert row["variant_id"] == FROZEN_ACTIVE_VARIANT
    assert row["event_id"].startswith("ep0-wf15_m3_t5_ibon-food4,4-t")
    assert row["coalition_feasible"] == 1
    assert row["suppression_reason"] == ""


def test_build_request_consumes_a_real_detector_row():
    """Schema fidelity: every key build_request reads exists in a real row."""
    row, positions = real_detector_event_row()
    agent_positions = {i: p for i, p in enumerate(positions)}
    request = build_request(row, agent_positions=agent_positions,
                            grid_shape=GRID)
    assert request.event_id == row["event_id"]
    assert tuple(request.food_id) == tuple(row["food_id"])
    assert set(request.recipient_agents) == set(row["recipient_ids"])
    for ctx in request.agent_context:
        assert ctx.position == agent_positions[ctx.agent_id]
        assert ctx.level > 0, "levels must be sourced from loaders AND helpers"
        assert ctx.candidate_loading_cells


def test_real_detector_row_lacks_agent_positions_so_they_must_be_injected():
    """Pins the reason build_request takes positions as an explicit argument."""
    row, _ = real_detector_event_row()
    assert "agent_positions" not in row
    with pytest.raises(MissingAgentPositionError):
        build_request(row, agent_positions={}, grid_shape=GRID)


def test_real_detector_row_drives_a_valid_mock_response():
    from selma.provider import validate

    row, positions = real_detector_event_row()
    request = build_request(row, agent_positions={i: p for i, p in enumerate(positions)},
                            grid_shape=GRID)
    verdict = validate(request, DeterministicMockProvider().query(request))
    assert verdict.valid, verdict.reason


def test_controller_activates_from_a_real_detector_row():
    row, positions = real_detector_event_row()
    ctrl = controller()
    record = ctrl.apply_transition(
        step=int(row["episode_step"]), raw_team_reward=0.0, phi_before=0.0,
        post_agent_positions={i: p for i, p in enumerate(positions)},
        post_food_positions=(FOOD,), event_rows=[row],
        episode_ended=False, stored_terminated=False, terminal_cause="",
    )
    assert ctrl.active is not None
    assert record.guidance_transition == "activation"
    assert record.provider_valid == 1


# ==========================================================================
# 4. frozen-variant filtering
# ==========================================================================


def test_only_frozen_variant_events_are_offered():
    ctrl = controller()
    other = event_row(event_id="ep0-wf5_m2_none_iboff-food4,4-t3",
                      variant_id="wf5_m2_none_iboff")
    record = ctrl.apply_transition(
        step=3, raw_team_reward=0.0, phi_before=0.0,
        post_agent_positions={0: (4, 3), 1: (6, 6)}, post_food_positions=(FOOD,),
        event_rows=[other], episode_ended=False, stored_terminated=False,
        terminal_cause="",
    )
    assert ctrl.active is None
    assert ctrl.provider_call_count == 0
    assert record.query_triggered == 0


def test_frozen_variant_event_is_offered_among_foreign_ones():
    ctrl = controller()
    rows = [
        event_row(event_id="ep0-wf5_m2_none_iboff-food4,4-t3", variant_id="wf5_m2_none_iboff"),
        event_row(event_id="ep0-wf15_m3_t5_ibon-food4,4-t3"),
        event_row(event_id="ep0-wf10_m3_t15_ibon-food4,4-t3", variant_id="wf10_m3_t15_ibon"),
    ]
    ctrl.apply_transition(
        step=3, raw_team_reward=0.0, phi_before=0.0,
        post_agent_positions={0: (4, 3), 1: (6, 6)}, post_food_positions=(FOOD,),
        event_rows=rows, episode_ended=False, stored_terminated=False, terminal_cause="",
    )
    assert ctrl.provider_call_count == 1, "exactly one query, for the frozen variant"
    assert ctrl.active.event_id == "ep0-wf15_m3_t5_ibon-food4,4-t3"


def test_infeasible_or_suppressed_events_are_never_offered():
    ctrl = controller()
    bad = [
        dict(event_row(event_id="a"), coalition_feasible=0),
        dict(event_row(event_id="b"), suppression_reason="team_gate_closed"),
        dict(event_row(event_id="c"), row_type="raw_candidate"),
    ]
    ctrl.apply_transition(
        step=3, raw_team_reward=0.0, phi_before=0.0,
        post_agent_positions={0: (4, 3), 1: (6, 6)}, post_food_positions=(FOOD,),
        event_rows=bad, episode_ended=False, stored_terminated=False, terminal_cause="",
    )
    assert ctrl.provider_call_count == 0 and ctrl.active is None


# ==========================================================================
# 5. dynamic-potential ordering: the boundary identity
# ==========================================================================


def test_activation_boundary_uses_new_guidance_after_and_none_before():
    ctrl = controller()
    states = walk(6)
    records = replay(ctrl, states, events={2: [event_row()]})
    activation = records[2]
    assert activation.guidance_transition == "activation"
    assert activation.phi_before == 0.0, "no guidance was in force before"
    assert activation.phi_after_effective < 0.0, "the new guidance is priced after"


def test_persistence_uses_the_same_guidance_on_both_sides():
    ctrl = controller()
    states = walk(8)
    records = replay(ctrl, states, events={1: [event_row()]})
    ongoing = records[3]
    assert ongoing.guidance_transition == "ongoing"
    assert ongoing.phi_before < 0.0 and ongoing.phi_after_effective < 0.0
    assert ongoing.phi_after_effective > ongoing.phi_before, "approach raises the potential"


def test_replacement_prices_the_new_guidance_after_and_the_old_before():
    ctrl = controller()
    states = walk(8)
    records = replay(ctrl, states,
                     events={1: [event_row(event_id="e1")],
                             4: [event_row(event_id="e2", cells=((4, 5), (5, 4)))]})
    replacement = records[4]
    assert replacement.guidance_transition == "replacement"
    assert replacement.phi_before < 0.0, "the OLD guidance is priced before"
    assert replacement.event_id == "e2", "the NEW guidance is priced after"
    assert records[3].event_id == "e1"
    assert ctrl.replacement_count == 1


@pytest.mark.parametrize("t", range(1, 25))
def test_phi_after_equals_phi_before_of_the_next_transition(t):
    """The single invariant that makes the transformation potential-based."""
    ctrl = controller()
    states = walk(26)
    records = replay(ctrl, states, events={3: [event_row()]})
    if t < len(records):
        assert records[t - 1].phi_after_effective == pytest.approx(
            records[t].phi_before, abs=PHI_BOUNDARY_TOLERANCE
        )


def test_expiry_boundary_sets_phi_after_to_zero_not_the_stale_potential():
    """The audited implementation priced the expiry transition with dead guidance."""
    ctrl = controller()
    states = walk(TTL_STEPS + 6)
    records = replay(ctrl, states, events={0: [event_row()]})
    expiry = [r for r in records if r.guidance_transition == "expiry"]
    assert len(expiry) == 1, "exactly one TTL expiry boundary"
    assert expiry[0].phi_before < 0.0
    assert expiry[0].phi_after_effective == 0.0
    assert expiry[0].expiry_reason == "ttl"


# ==========================================================================
# 6. TTL semantics: at most exactly L guided transitions
# ==========================================================================


def test_guidance_is_guidance_before_for_exactly_ttl_transitions():
    ctrl = controller()
    states = walk(TTL_STEPS + 8)
    records = replay(ctrl, states, events={0: [event_row()]})
    guided = [t for t, r in enumerate(records) if r.phi_before != 0.0]
    assert len(guided) == TTL_STEPS, f"expected exactly {TTL_STEPS}, got {len(guided)}"
    assert guided == list(range(1, TTL_STEPS + 1))


def test_ttl_guidance_never_influences_transition_l_plus_one():
    ctrl = controller()
    states = walk(TTL_STEPS + 8)
    records = replay(ctrl, states, events={0: [event_row()]})
    for t in range(TTL_STEPS + 1, len(records)):
        assert records[t].phi_before == 0.0
        assert records[t].phi_after_effective == 0.0
        assert records[t].pbrs_term_unscaled == 0.0


def test_replacement_restarts_the_ttl_budget():
    ctrl = controller()
    states = walk(TTL_STEPS + 12)
    records = replay(ctrl, states,
                     events={0: [event_row(event_id="e1")],
                             5: [event_row(event_id="e2")]})
    guided = [t for t, r in enumerate(records) if r.phi_before != 0.0]
    assert len(guided) == 5 + TTL_STEPS  # e1 for t=1..5, then e2 for t=6..20


# ==========================================================================
# 7. target-food disappearance, terminal, time limit, reset
# ==========================================================================


def test_target_food_removal_expires_at_that_boundary():
    ctrl = controller()
    states = walk(12, foods_gone_at=5)
    records = replay(ctrl, states, events={0: [event_row()]})
    removal = records[4]  # s_5 is the first state without the food
    assert removal.guidance_transition == "expiry"
    assert removal.expiry_reason == "target_removed"
    assert removal.phi_after_effective == 0.0
    assert records[5].phi_before == 0.0


def test_food_removal_never_leaks_a_partial_load_allowance():
    """The audited code dropped only the +1 LOAD allowance and kept the distance."""
    ctrl = controller()
    states = walk(12, foods_gone_at=5)
    records = replay(ctrl, states, events={0: [event_row()]})
    assert records[4].phi_after_real == 0.0
    assert records[4].phi_after_effective == 0.0


def test_true_terminal_zeroes_the_successor_potential():
    ctrl = controller()
    states = walk(6)
    records = replay(ctrl, states, events={0: [event_row()]},
                     stored_terminated=True, terminal_cause="true_terminal")
    last = records[-1]
    assert last.phi_after_effective == 0.0
    assert last.terminal_cause == "true_terminal"
    assert ctrl.active is None


def test_time_limit_ending_follows_the_stage0_bootstrap_verdict():
    """Stage 0: a time limit also stores terminated=1, so the potential is zeroed."""
    ctrl = controller()
    states = walk(6)
    records = replay(ctrl, states, events={0: [event_row()]},
                     stored_terminated=True, terminal_cause="time_limit")
    assert records[-1].phi_after_effective == 0.0
    assert records[-1].terminal_cause == "time_limit"


def test_bootstrapped_time_limit_retains_the_real_next_potential():
    """If a configuration ever stores terminated=0 at a time limit (contract 4.4)."""
    ctrl = controller()
    states = walk(6)
    records = replay(ctrl, states, events={0: [event_row()]},
                     stored_terminated=False, terminal_cause="time_limit")
    assert records[-1].phi_after_effective == records[-1].phi_after_real
    assert records[-1].phi_after_effective < 0.0


def test_episode_reset_clears_guidance_but_keeps_cumulative_counters():
    ctrl = controller()
    states = walk(6)
    replay(ctrl, states, events={0: [event_row()]})
    activations = ctrl.activation_count
    ctrl.reset_episode()
    assert ctrl.active is None
    assert ctrl.potential({0: (4, 3), 1: (4, 5)}, (FOOD,)) == 0.0
    assert ctrl.activation_count == activations >= 1


def test_second_episode_starts_from_zero_potential():
    ctrl = controller()
    replay(ctrl, walk(6), events={0: [event_row(event_id="e1")]})
    ctrl.reset_episode()
    records = replay(ctrl, walk(6), events={})
    assert all(r.phi_before == 0.0 and r.phi_after_effective == 0.0 for r in records)


# ==========================================================================
# 8. terminal provider suppression
# ==========================================================================


def test_no_provider_query_on_a_terminal_transition():
    ctrl = controller()
    record = ctrl.apply_transition(
        step=9, raw_team_reward=0.0, phi_before=0.0,
        post_agent_positions={0: (4, 3), 1: (6, 6)}, post_food_positions=(FOOD,),
        event_rows=[event_row()], episode_ended=True, stored_terminated=True,
        terminal_cause="true_terminal",
    )
    assert ctrl.provider_call_count == 0
    assert ctrl.query_count == 0
    assert ctrl.activation_count == 0
    assert ctrl.terminal_suppressed_events == 1
    assert record.query_triggered == 0
    assert ctrl.active is None


def test_no_provider_query_on_a_time_limit_final_transition():
    ctrl = controller()
    ctrl.apply_transition(
        step=49, raw_team_reward=0.0, phi_before=0.0,
        post_agent_positions={0: (4, 3), 1: (6, 6)}, post_food_positions=(FOOD,),
        event_rows=[event_row()], episode_ended=True, stored_terminated=True,
        terminal_cause="time_limit",
    )
    assert ctrl.provider_call_count == 0 and ctrl.terminal_suppressed_events == 1


def test_non_terminal_transitions_still_query():
    ctrl = controller()
    ctrl.apply_transition(
        step=9, raw_team_reward=0.0, phi_before=0.0,
        post_agent_positions={0: (4, 3), 1: (6, 6)}, post_food_positions=(FOOD,),
        event_rows=[event_row()], episode_ended=False, stored_terminated=False,
        terminal_cause="",
    )
    assert ctrl.provider_call_count == 1 and ctrl.terminal_suppressed_events == 0


# ==========================================================================
# 9. PBRS accounting: beta exactly once, three distinct quantities
# ==========================================================================


def test_pbrs_term_unscaled_is_beta_free():
    assert pbrs_term_unscaled(0.99, -0.5, -0.2) == pytest.approx(0.99 * -0.2 - -0.5)


@pytest.mark.parametrize("beta", [0.0, 0.25, 1.0, 2.5])
def test_beta_is_applied_exactly_once(beta):
    ctrl = controller(beta=beta)
    states = walk(10)
    records = replay(ctrl, states, events={0: [event_row()]}, raw=[0.5] * 10)
    for r in records:
        assert r.shaping_contribution == pytest.approx(beta * r.pbrs_term_unscaled)
        assert r.training_reward == pytest.approx(r.raw_team_reward + beta * r.pbrs_term_unscaled)


def test_unscaled_term_is_nonzero_even_when_beta_is_zero():
    """beta=0 must zero the *contribution*, not the measured potential difference."""
    ctrl = controller(beta=0.0)
    records = replay(ctrl, walk(8), events={0: [event_row()]})
    assert any(r.pbrs_term_unscaled != 0.0 for r in records)
    assert all(r.shaping_contribution == 0.0 for r in records)


def test_beta_zero_assigns_the_raw_reward_object_directly():
    ctrl = controller(beta=0.0)
    raw = 1.2345
    record = ctrl.apply_transition(
        step=0, raw_team_reward=raw, phi_before=-0.4,
        post_agent_positions={0: (4, 3), 1: (6, 6)}, post_food_positions=(FOOD,),
        event_rows=(), episode_ended=False, stored_terminated=False, terminal_cause="",
    )
    assert record.training_reward is raw, "no raw + 0.0; no dtype change"
    assert record.shaping_contribution == 0.0


def test_shaping_is_never_clipped():
    ctrl = controller(beta=1.0)
    record = ctrl.apply_transition(
        step=0, raw_team_reward=0.0, phi_before=-1e6,
        post_agent_positions={0: (4, 3), 1: (6, 6)}, post_food_positions=(FOOD,),
        event_rows=(), episode_ended=False, stored_terminated=False, terminal_cause="",
    )
    assert record.pbrs_term_unscaled > 1e5


# ==========================================================================
# 10. discounted telescoping identity over complete finite trajectories
# ==========================================================================


@pytest.mark.parametrize("beta,gamma", [(1.0, 0.99), (0.25, 0.99), (2.0, 0.9), (0.5, 1.0)])
def test_discounted_telescoping_identity_with_ttl_expiry(beta, gamma):
    ctrl = controller(beta=beta, gamma=gamma)
    states = walk(TTL_STEPS + 8)
    records = replay(ctrl, states, events={0: [event_row()]})
    residual = discounted_shaping(records, gamma) - endpoint_identity(records, gamma, beta)
    assert abs(residual) <= TELESCOPING_TOLERANCE, f"residual {residual}"


@pytest.mark.parametrize("beta,gamma", [(1.0, 0.99), (0.75, 0.95)])
def test_discounted_telescoping_identity_with_food_removal(beta, gamma):
    ctrl = controller(beta=beta, gamma=gamma)
    states = walk(14, foods_gone_at=6)
    records = replay(ctrl, states, events={0: [event_row()]})
    residual = discounted_shaping(records, gamma) - endpoint_identity(records, gamma, beta)
    assert abs(residual) <= TELESCOPING_TOLERANCE, f"residual {residual}"


def test_discounted_telescoping_identity_with_activation_and_replacement():
    beta, gamma = 1.0, 0.99
    ctrl = controller(beta=beta, gamma=gamma)
    states = walk(24)
    records = replay(ctrl, states,
                     events={2: [event_row(event_id="e1")],
                             9: [event_row(event_id="e2")],
                             15: [event_row(event_id="e3")]})
    residual = discounted_shaping(records, gamma) - endpoint_identity(records, gamma, beta)
    assert abs(residual) <= TELESCOPING_TOLERANCE, f"residual {residual}"


def test_cumulative_shaping_is_zero_when_both_endpoints_are_zero():
    """Phi_0 = 0 and a stored terminal Phi_T = 0 => zero net discounted shaping."""
    beta, gamma = 1.3, 0.99
    ctrl = controller(beta=beta, gamma=gamma)
    records = replay(ctrl, walk(TTL_STEPS + 10), events={1: [event_row()]})
    assert records[0].phi_before == 0.0 and records[-1].phi_after_effective == 0.0
    assert abs(discounted_shaping(records, gamma)) <= TELESCOPING_TOLERANCE


def test_every_boundary_term_is_present_and_none_are_hidden():
    """Activation, replacement and expiry must each produce a real, logged term."""
    ctrl = controller(beta=1.0, gamma=0.99)
    records = replay(ctrl, walk(TTL_STEPS + 12),
                     events={1: [event_row(event_id="e1")],
                             4: [event_row(event_id="e2")]})
    labels = [r.guidance_transition for r in records]
    assert "activation" in labels and "replacement" in labels and "expiry" in labels
    for r in records:
        if r.guidance_transition in ("activation", "replacement", "expiry"):
            assert r.pbrs_term_unscaled != 0.0, "a boundary term was hidden"


# ==========================================================================
# 11. the runner uses this exact path
# ==========================================================================


def test_runner_uses_the_shared_transition_path():
    """Guarantees the ordering under test is the ordering that runs in training."""
    runner_src = open(
        os.path.join(os.path.dirname(__file__), "..", "src", "runners",
                     "episode_runner.py")
    ).read()
    assert "apply_transition(" in runner_src
    # and it must not re-implement the lifecycle itself
    assert "expire_if_due(" not in runner_src
    assert "offer_events(" not in runner_src
