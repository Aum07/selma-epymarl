"""Tests for the Stage B selected-variant plumbing (`selma_variant_id`).

Covers the four properties required before the held-out seed-99 run:
backward compatibility, selected-only correctness, invalid-input safety, and
behavioural equivalence between the full 48-variant bank filtered to one
variant and a selected-only bank.

The plumbing lives in `EpisodeRunner.__init__`; `resolve_variants` below mirrors
it exactly. No detector logic is touched by the patch or by these tests.

Run with:  PYTHONPATH=src pytest tests/test_selma_variant_selection.py -v
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from selma.detector import ShadowDetectorBank, build_variant_grid  # noqa: E402
from selma.types import (  # noqa: E402
    AgentSnapshot,
    FailedLoadEvent,
    FoodSnapshot,
    SelmaStepInfo,
)

GRID = (8, 8)
FOOD = (4, 4)
FROZEN_ID = "wf15_m3_t5_ibon"


def fields(v):
    """DetectorVariant uses __slots__ and defines no __eq__, so compare by value."""
    return (v.variant_id, v.w_fail, v.m, v.w_team, v.use_inbound)


def resolve_variants(selma_variant_id):
    """Mirror of the resolution logic in EpisodeRunner.__init__."""
    selma_variant_id = str(selma_variant_id or "")
    if not selma_variant_id:
        return None
    all_variants = build_variant_grid()
    matches = [v for v in all_variants if v.variant_id == selma_variant_id]
    if len(matches) != 1:
        raise ValueError(
            "selma_variant_id={!r} matched {} variants in "
            "build_variant_grid(); expected exactly 1. Known ids: {}".format(
                selma_variant_id,
                len(matches),
                ", ".join(v.variant_id for v in all_variants),
            )
        )
    return tuple(matches)


def make_bank(variants):
    return ShadowDetectorBank(
        n_agents=3,
        grid_shape=GRID,
        condition="test",
        seed=91,
        rearm_steps=3,
        cooldown_steps=5,
        max_events_per_episode=2,
        w_inbound=3,
        variants=variants,
    )


def failed_event(episode, position, food_level, loader_ids, loader_levels):
    total = sum(loader_levels)
    return FailedLoadEvent(
        food_id=(episode, position[0], position[1]),
        food_position=position,
        food_level=food_level,
        loader_ids=tuple(loader_ids),
        loader_levels=tuple(loader_levels),
        total_loader_level=total,
        level_deficit=food_level - total,
        coalition_required=total < food_level,
        requires_help_by_agent=tuple(food_level > l for l in loader_levels),
    )


def step_info(step, agents, foods, failed=(), episode=1, removed=()):
    pre = dict(foods)
    for position, level in removed:
        pre[position] = level
    return SelmaStepInfo(
        episode_id=episode,
        episode_step=step,
        pre_foods=tuple(FoodSnapshot(p, l) for p, l in sorted(pre.items())),
        post_foods=tuple(FoodSnapshot(p, l) for p, l in sorted(foods.items())),
        agents=tuple(AgentSnapshot(i, p, l) for i, p, l in agents),
        selected_actions=tuple(5 for _ in agents),
        invalid_load_agents=(),
        failed_load_events=tuple(failed),
        successful_load_events=(),
    )


def deterministic_trace():
    """A fixed transition sequence exercising candidates, coalescing, the team
    gate, inbound help, the latch lifecycle and food removal across episodes."""
    episodes = []
    for episode in (1, 2):
        steps = []
        for step in range(30):
            # Agent 1 walks in from a distance (drives inbound-help detection).
            helper_col = max(1, 6 - step // 2)
            agents = [
                (0, (4, 3), 1),
                (1, (4, helper_col), 2),
                (2, (1, 1), 1),
            ]
            removed = ()
            foods = {FOOD: 3}
            failed = [failed_event(episode, FOOD, 3, (0,), (1,))]
            if step >= 2 and helper_col == 5:
                # Two agents adjacent to the same food on the same step
                # -> same-food coalescing.
                failed.append(failed_event(episode, FOOD, 3, (1,), (2,)))
            if episode == 2 and step >= 20:
                foods = {}
                removed = ((FOOD, 3),)
                failed = []
            steps.append(
                step_info(
                    step,
                    agents,
                    foods,
                    failed=failed,
                    episode=episode,
                    removed=removed,
                )
            )
        episodes.append((episode, steps))
    return episodes


def drive(bank):
    """Feed the deterministic trace through a bank."""
    for episode, steps in deterministic_trace():
        bank.reset_episode(episode)
        for i, si in enumerate(steps):
            bank.update(si, raw_team_reward=0.0, t_env=1000 * episode + i)
        bank.finish_episode()
    return bank


# -- 1. Backward compatibility ------------------------------------------------


def test_empty_selection_resolves_to_none():
    assert resolve_variants("") is None
    assert resolve_variants(None) is None


def test_empty_selection_builds_all_48_variants():
    assert len(make_bank(resolve_variants("")).variants) == 48


def test_empty_selection_matches_pre_change_grid_exactly():
    """Ids, parameters and ordering must equal the untouched build_variant_grid()."""
    expected = build_variant_grid()
    bank = make_bank(resolve_variants(""))
    assert [v.variant_id for v in bank.variants] == [v.variant_id for v in expected]
    for got, want in zip(bank.variants, expected):
        assert (got.w_fail, got.m, got.w_team, got.use_inbound) == (
            want.w_fail,
            want.m,
            want.w_team,
            want.use_inbound,
        )
    assert [fields(v) for v in bank.variants] == [fields(v) for v in expected]


def test_empty_selection_per_variant_state_covers_all_48():
    bank = make_bank(resolve_variants(""))
    assert len(bank.cumulative_counters) == 48
    assert len(bank.events_per_episode_hist) == 48
    assert len(bank.variant_state) == 48


# -- 2. Selected-only correctness ---------------------------------------------


def test_frozen_id_creates_exactly_one_variant():
    bank = make_bank(resolve_variants(FROZEN_ID))
    assert len(bank.variants) == 1
    assert bank.variants[0].variant_id == FROZEN_ID


def test_frozen_id_fields_match_the_frozen_configuration():
    (variant,) = resolve_variants(FROZEN_ID)
    assert variant.w_fail == 15
    assert variant.m == 3
    assert variant.w_team == 5
    assert variant.use_inbound is True


def test_frozen_id_is_taken_from_the_grid_not_reconstructed():
    (variant,) = resolve_variants(FROZEN_ID)
    grid_match = [v for v in build_variant_grid() if v.variant_id == FROZEN_ID]
    assert len(grid_match) == 1
    # DetectorVariant defines no __eq__; equality of every field is the
    # meaningful property -- the parameters come from the grid, not from a
    # manual reconstruction in the runner.
    assert fields(variant) == fields(grid_match[0])


def test_selected_only_bank_keeps_the_frozen_lifecycle_values():
    bank = make_bank(resolve_variants(FROZEN_ID))
    assert (
        bank.w_inbound,
        bank.cooldown_steps,
        bank.rearm_steps,
        bank.max_events_per_episode,
    ) == (3, 5, 3, 2)


def test_selected_only_bank_state_is_scoped_to_one_variant():
    bank = make_bank(resolve_variants(FROZEN_ID))
    assert list(bank.cumulative_counters) == [FROZEN_ID]
    assert list(bank.events_per_episode_hist) == [FROZEN_ID]
    assert list(bank.variant_state) == [FROZEN_ID]


# -- 3. Invalid-input safety --------------------------------------------------


@pytest.mark.parametrize(
    "bad_id",
    [
        "not_a_variant",
        "wf15_m3_t5_ibON",
        "wf15_m3_t5",
        " wf15_m3_t5_ibon",
        "wf99_m9_t9_ibon",
    ],
)
def test_unknown_variant_id_raises(bad_id):
    with pytest.raises(ValueError) as excinfo:
        resolve_variants(bad_id)
    message = str(excinfo.value)
    assert "selma_variant_id" in message
    assert "expected exactly 1" in message


def test_unknown_variant_id_never_falls_back_to_all_48():
    try:
        result = resolve_variants("not_a_variant")
    except ValueError:
        return
    pytest.fail(
        "unknown id silently resolved to {} instead of raising".format(
            "all 48 (None)" if result is None else "{} variants".format(len(result))
        )
    )


def test_error_message_lists_known_ids():
    with pytest.raises(ValueError) as excinfo:
        resolve_variants("not_a_variant")
    assert FROZEN_ID in str(excinfo.value)


# -- 4. Behavioural equivalence -----------------------------------------------


def test_trace_actually_produces_events():
    """Guard: the equivalence tests are vacuous if the trace fires nothing."""
    sel = drive(make_bank(resolve_variants(FROZEN_ID)))
    summary = sel.calibration_summary()
    assert summary["variants"][FROZEN_ID]["raw_candidates"] > 0


def test_selected_only_summary_equals_filtered_full_summary():
    full = drive(make_bank(resolve_variants("")))
    sel = drive(make_bank(resolve_variants(FROZEN_ID)))
    full_summary = full.calibration_summary()
    sel_summary = sel.calibration_summary()
    assert set(sel_summary["variants"]) == {FROZEN_ID}
    assert sel_summary["variants"][FROZEN_ID] == full_summary["variants"][FROZEN_ID]


def test_selected_only_rows_equal_filtered_full_rows():
    full = drive(make_bank(resolve_variants("")))
    sel = drive(make_bank(resolve_variants(FROZEN_ID)))
    full_rows = [r for r in full.rows if r["variant_id"] == FROZEN_ID]
    sel_rows = [r for r in sel.rows if r["variant_id"] == FROZEN_ID]
    assert len(sel_rows) == len(full_rows)
    assert sel_rows == full_rows


def test_selected_only_shared_summary_fields_match():
    full = drive(make_bank(resolve_variants("")))
    sel = drive(make_bank(resolve_variants(FROZEN_ID)))
    f, s = full.calibration_summary(), sel.calibration_summary()
    for key in ("condition", "seed", "n_agents", "grid_shape", "episodes_seen",
                "lifecycle_config", "zero_guarantees"):
        assert s[key] == f[key]


def test_selected_only_emits_no_other_variant_id():
    sel = drive(make_bank(resolve_variants(FROZEN_ID)))
    assert {r["variant_id"] for r in sel.rows} <= {FROZEN_ID}


def test_zero_guarantees_hold_for_selected_only_bank():
    sel = drive(make_bank(resolve_variants(FROZEN_ID)))
    z = sel.calibration_summary()["zero_guarantees"]
    assert z["provider_call_count"] == 0
    assert z["query_count"] == 0
    assert z["active_subgoal_count"] == 0
    assert z["total_shaping_reward"] == 0.0
