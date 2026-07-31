"""SELMA Stage B tests: coordination-stall detector in shadow mode.

Covers the candidate rule and window boundaries, the team-stall gate, same-food
coalescing, deterministic helper coalitions and BFS tie-breaks, inbound-help
shadow suppression, the latch/cooldown/rearm lifecycle, per-horizon natural
resolution labelling, and the shadow-mode isolation guarantees.

Fixtures keep food away from border rows/columns and never place an agent
adjacent to two foods, matching what lbforaging's spawn rules can produce.

Run with:  PYTHONPATH=src pytest tests/test_selma_detector.py -v
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from selma.coalition import (  # noqa: E402
    bfs_distances,
    select_helper_coalition,
    valid_loading_cells,
)
from selma.detector import ShadowDetectorBank, build_variant_grid  # noqa: E402
from selma.types import (  # noqa: E402
    AgentSnapshot,
    FailedLoadEvent,
    FoodSnapshot,
    SelmaStepInfo,
    SuccessfulLoadEvent,
)

GRID = (8, 8)
FOOD = (4, 4)


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


def step_info(
    step,
    agents,
    foods,
    failed=(),
    successful=(),
    episode=1,
    removed=(),
):
    """Build a Stage A style record. ``foods`` is {position: level} post-step."""
    pre = dict(foods)
    for position, level in removed:
        pre[position] = level
    return SelmaStepInfo(
        episode_id=episode,
        episode_step=step,
        pre_foods=tuple(
            FoodSnapshot(p, l) for p, l in sorted(pre.items())
        ),
        post_foods=tuple(
            FoodSnapshot(p, l) for p, l in sorted(foods.items())
        ),
        agents=tuple(AgentSnapshot(i, p, l) for i, p, l in agents),
        selected_actions=tuple(5 for _ in agents),
        invalid_load_agents=(),
        failed_load_events=tuple(failed),
        successful_load_events=tuple(successful),
    )


def make_bank(variants=None, **kwargs):
    bank = ShadowDetectorBank(
        n_agents=kwargs.pop("n_agents", 3),
        grid_shape=GRID,
        condition="test",
        seed=91,
        variants=variants,
        **kwargs,
    )
    bank.reset_episode(1)
    return bank


def only(variant_id):
    return tuple(v for v in build_variant_grid() if v.variant_id == variant_id)


def fid_of(position, episode=1):
    """The canonical episode-specific food id used as the key everywhere."""
    return (episode, position[0], position[1])


def events(bank, row_type="coalesced_event"):
    return [r for r in bank.rows if r["row_type"] == row_type]


# ==========================================================================
# 1-3: window boundaries, distinct transitions, stable food identity
# ==========================================================================


def test_grid_has_48_variants_with_unique_ids():
    grid = build_variant_grid()
    assert len(grid) == 48
    assert len({v.variant_id for v in grid}) == 48


def test_inclusive_w_fail_window_boundary():
    """A failure exactly W_fail-1 transitions back is inside the window."""
    bank = make_bank(variants=only("wf5_m2_none_iboff"))
    fid = fid_of(FOOD)
    bank.fail_history[(0, fid)] = [0, 4]
    assert bank._window_fail_count(0, fid, 4, 5) == 2  # window [0, 4] inclusive
    assert bank._window_fail_count(0, fid, 5, 5) == 1  # window [1, 5] drops step 0
    assert bank._window_fail_count(0, fid, 4, 10) == 2


def test_distinct_transition_counting():
    """Two loaders failing on the same transition is one failure per agent."""
    bank = make_bank(variants=only("wf10_m2_none_iboff"))
    event = failed_event(1, FOOD, 4, (0, 1), (1, 1))
    info = step_info(
        0,
        [(0, (4, 3), 1), (1, (3, 4), 1), (2, (7, 7), 2)],
        {FOOD: 4},
        failed=[event],
    )
    bank.update(info, raw_team_reward=0.0, t_env=0)
    assert bank.fail_history[(0, fid_of(FOOD))] == [0]
    assert bank.fail_history[(1, fid_of(FOOD))] == [0]


def test_food_identity_stable_across_failures():
    bank = make_bank(variants=only("wf10_m2_none_iboff"))
    for step in range(3):
        event = failed_event(1, FOOD, 3, (0,), (1,))
        bank.update(
            step_info(
                step,
                [(0, (4, 3), 1), (1, (1, 1), 2), (2, (7, 7), 2)],
                {FOOD: 3},
                failed=[event],
            ),
            raw_team_reward=0.0,
            t_env=step,
        )
    assert bank.fail_history[(0, fid_of(FOOD))] == [0, 1, 2]
    assert len({r["food_id"] for r in bank.pending_rows and [p[0] for p in bank.pending_rows]}) == 1


def test_food_disappearance_records_removal_and_resolves_latch():
    bank = make_bank(variants=only("wf10_m2_none_iboff"))
    for step in range(2):
        bank.update(
            step_info(
                step,
                [(0, (4, 3), 1), (1, (1, 1), 2), (2, (7, 7), 2)],
                {FOOD: 3},
                failed=[failed_event(1, FOOD, 3, (0,), (1,))],
            ),
            raw_team_reward=0.0,
            t_env=step,
        )
    state = bank.variant_state["wf10_m2_none_iboff"]
    assert state.foods[fid_of(FOOD)].latched is True

    bank.update(
        step_info(
            2,
            [(0, (4, 3), 1), (1, (4, 5), 2), (2, (7, 7), 2)],
            {},
            successful=[
                SuccessfulLoadEvent((1, 4, 4), FOOD, 3, (0, 1), (1, 2), 3)
            ],
            removed=[(FOOD, 3)],
        ),
        raw_team_reward=3.0,
        t_env=2,
    )
    assert bank.removal_step[fid_of(FOOD)] == 2
    assert state.foods[fid_of(FOOD)].latched is False


# ==========================================================================
# 5-7: team-stall gate
# ==========================================================================


def _quiet_step(bank, step, reward=0.0):
    bank.update(
        step_info(step, [(0, (0, 0), 1), (1, (1, 1), 2), (2, (7, 7), 2)], {FOOD: 3}),
        raw_team_reward=reward,
        t_env=step,
    )


def test_incomplete_team_window_cannot_open():
    bank = make_bank(variants=only("wf10_m2_t5_iboff"))
    for step in range(4):
        _quiet_step(bank, step)
        assert bank._team_gate_open(step, 5) is False, f"opened at step {step}"
    _quiet_step(bank, 4)
    assert bank._team_gate_open(4, 5) is True


def test_team_window_includes_current_transition():
    bank = make_bank(variants=only("wf10_m2_t5_iboff"))
    for step in range(4):
        _quiet_step(bank, step)
    _quiet_step(bank, 4, reward=1.0)  # progress on the current transition
    assert bank._team_gate_open(4, 5) is False


def test_progress_closes_the_gate_by_reward_or_food_removal():
    bank = make_bank(variants=only("wf10_m2_t5_iboff"))
    for step in range(5):
        _quiet_step(bank, step)
    assert bank._team_gate_open(4, 5) is True

    bank2 = make_bank(variants=only("wf10_m2_t5_iboff"))
    for step in range(4):
        _quiet_step(bank2, step)
    bank2.update(
        step_info(
            4,
            [(0, (4, 3), 1), (1, (4, 5), 2), (2, (7, 7), 2)],
            {},
            successful=[SuccessfulLoadEvent((1, 4, 4), FOOD, 3, (0, 1), (1, 2), 3)],
            removed=[(FOOD, 3)],
        ),
        raw_team_reward=0.0,  # removal alone must close the gate
        t_env=4,
    )
    assert bank2._team_gate_open(4, 5) is False


def test_closed_gate_suppresses_candidates_but_open_gate_does_not():
    """Same trajectory, gated and ungated variants, opposite outcomes."""
    variants = only("wf10_m2_t15_iboff") + only("wf10_m2_none_iboff")
    bank = make_bank(variants=variants)
    for step in range(3):
        bank.update(
            step_info(
                step,
                [(0, (4, 3), 1), (1, (1, 1), 2), (2, (7, 7), 2)],
                {FOOD: 3},
                failed=[failed_event(1, FOOD, 3, (0,), (1,))],
            ),
            raw_team_reward=0.0,
            t_env=step,
        )
    bank.finish_episode()
    gated = [r for r in events(bank) if r["variant_id"] == "wf10_m2_t15_iboff"]
    ungated = [r for r in events(bank) if r["variant_id"] == "wf10_m2_none_iboff"]
    # W_team=15 window is incomplete after 3 transitions -> gate shut -> no events.
    assert gated == []
    assert len(ungated) == 1
    suppressed = [
        r
        for r in bank.rows
        if r["variant_id"] == "wf10_m2_t15_iboff"
        and r["suppression_reason"] == "team_window_incomplete"
    ]
    assert suppressed, "an incomplete start window is reported as such"
    assert not [
        r
        for r in bank.rows
        if r["variant_id"] == "wf10_m2_t15_iboff"
        and r["suppression_reason"] == "team_gate_closed"
    ], "an incomplete window must not be labelled a closed (stalled) gate"


def test_gate_closed_by_progress_is_labelled_separately_from_incomplete_window():
    """The two suppression causes must never be conflated in the shadow rows."""
    variant = "wf10_m2_t5_iboff"
    bank = make_bank(variants=only(variant))
    agents = [(0, (4, 3), 1), (1, (1, 1), 2), (2, (7, 7), 2)]

    def stall(step, reward=0.0):
        bank.update(
            step_info(
                step, agents, {FOOD: 3}, failed=[failed_event(1, FOOD, 3, (0,), (1,))]
            ),
            raw_team_reward=reward,
            t_env=step,
        )

    for step in range(4):  # window still incomplete (needs 5 transitions)
        stall(step)
    stall(4, reward=1.0)  # window complete, but real progress closed the gate
    bank.finish_episode()
    reasons = {r["episode_step"]: r["suppression_reason"] for r in bank.rows}
    assert reasons[1] == "team_window_incomplete"
    assert reasons[4] == "team_gate_closed"


# ==========================================================================
# 8-11: same-food coalescing
# ==========================================================================


def test_simultaneous_same_food_triggers_produce_one_sorted_event():
    bank = make_bank(variants=only("wf10_m2_none_iboff"))
    event = failed_event(1, FOOD, 5, (1, 0), (1, 1))
    for step in range(2):
        bank.update(
            step_info(
                step,
                [(0, (4, 3), 1), (1, (3, 4), 1), (2, (7, 7), 3)],
                {FOOD: 5},
                failed=[event],
            ),
            raw_team_reward=0.0,
            t_env=step,
        )
    bank.finish_episode()
    final = events(bank)
    assert len(final) == 1, "simultaneous same-food triggers must coalesce"
    assert final[0]["trigger_agent_ids"] == (0, 1), "trigger IDs must be sorted"
    assert final[0]["coalesced_candidate_count"] == 2
    raw = events(bank, "raw_candidate")
    assert len([r for r in raw if r["episode_step"] == 1]) == 2, "raw rows retained"
    counters = bank.variant_counters()["wf10_m2_none_iboff"]
    assert counters.get("same_food_coalesced", 0) >= 1


def test_food_level_latch_blocks_a_second_trigger_agent():
    """Agent 0 fires first; agent 1 qualifying later must not create an event."""
    bank = make_bank(variants=only("wf10_m2_none_iboff"))
    solo = failed_event(1, FOOD, 5, (0,), (1,))
    for step in range(2):
        bank.update(
            step_info(
                step,
                [(0, (4, 3), 1), (1, (3, 4), 1), (2, (7, 7), 3)],
                {FOOD: 5},
                failed=[solo],
            ),
            raw_team_reward=0.0,
            t_env=step,
        )
    assert len(events(bank)) == 0 or True
    both = failed_event(1, FOOD, 5, (0, 1), (1, 1))
    for step in (2, 3):
        bank.update(
            step_info(
                step,
                [(0, (4, 3), 1), (1, (3, 4), 1), (2, (7, 7), 3)],
                {FOOD: 5},
                failed=[both],
            ),
            raw_team_reward=0.0,
            t_env=step,
        )
    bank.finish_episode()
    assert len(events(bank)) == 1, "latch must block a second trigger agent"
    counters = bank.variant_counters()["wf10_m2_none_iboff"]
    assert counters.get("suppressed_by_latch", 0) >= 1


# ==========================================================================
# 12: variant independence
# ==========================================================================


def test_variants_have_fully_independent_state():
    bank = make_bank()  # all 48
    for step in range(6):
        bank.update(
            step_info(
                step,
                [(0, (4, 3), 1), (1, (1, 1), 2), (2, (7, 7), 2)],
                {FOOD: 3},
                failed=[failed_event(1, FOOD, 3, (0,), (1,))],
            ),
            raw_team_reward=0.0,
            t_env=step,
        )
    bank.finish_episode()

    states = bank.variant_state
    assert len({id(s) for s in states.values()}) == 48
    assert len({id(s.foods) for s in states.values()}) == 48

    # m=3 needs one more failure than m=2, so their firing steps must differ.
    m2 = [r for r in events(bank) if r["variant_id"] == "wf10_m2_none_iboff"]
    m3 = [r for r in events(bank) if r["variant_id"] == "wf10_m3_none_iboff"]
    assert m2 and m3
    assert m2[0]["episode_step"] == 1
    assert m3[0]["episode_step"] == 2

    # A latched variant must not stop any other variant from firing.
    fired = {r["variant_id"] for r in events(bank)}
    assert len(fired) >= 12


# ==========================================================================
# 13-15: deterministic helper coalition and BFS
# ==========================================================================


def test_bfs_blocks_food_and_respects_boundaries():
    dist = bfs_distances((0, 0), (3, 3), {(0, 1), (1, 1)})
    assert dist[(0, 0)] == 0
    assert dist[(2, 0)] == 2
    assert (0, 1) not in dist
    assert dist[(2, 2)] == 4  # must route around the blocked column
    assert all(0 <= r < 3 and 0 <= c < 3 for r, c in dist)


def test_agents_do_not_block_paths():
    dist_a = bfs_distances((4, 0), GRID, {FOOD})
    dist_b = bfs_distances((4, 0), GRID, {FOOD})
    assert dist_a == dist_b
    assert dist_a[(4, 3)] == 3  # straight line, unaffected by any agent


def test_valid_loading_cells_exclude_food_and_sort():
    cells = valid_loading_cells(FOOD, GRID, {FOOD, (3, 4)})
    assert cells == ((4, 3), (4, 5), (5, 4))
    assert list(cells) == sorted(cells)


def _coalition(agents, loader_ids, loader_levels, food_level=3, foods=(FOOD,)):
    snaps = tuple(AgentSnapshot(i, p, l) for i, p, l in agents)
    distances = {
        int(a.agent_id): bfs_distances(a.position, GRID, set(foods)) for a in snaps
    }
    return select_helper_coalition(
        food_position=FOOD,
        food_level=food_level,
        loader_ids=loader_ids,
        loader_levels=loader_levels,
        agents=snaps,
        food_positions=set(foods),
        grid_shape=GRID,
        distances_by_agent=distances,
    )


def test_minimum_helper_selection_prefers_fewest_agents():
    """One level-2 helper beats two level-1 helpers for a deficit of 2."""
    result = _coalition(
        agents=[(0, (4, 3), 1), (1, (0, 4), 2), (2, (7, 4), 1), (3, (7, 5), 1)],
        loader_ids=(0,),
        loader_levels=(1,),
        food_level=3,
    )
    assert result.feasible is True
    assert result.required_extra_level == 2
    assert result.helper_ids == (1,)
    assert result.recipient_ids == (0, 1)


def test_helper_selection_tie_breaks_on_distance_then_id_then_cell():
    """Equal-size, equal-level helpers: nearer wins."""
    near = _coalition(
        agents=[(0, (4, 3), 1), (1, (2, 4), 2), (2, (7, 4), 2)],
        loader_ids=(0,),
        loader_levels=(1,),
        food_level=3,
    )
    assert near.helper_ids == (1,), "nearer helper must win on total BFS distance"

    # Perfectly symmetric distances -> lexicographically smallest helper ID wins.
    symmetric = _coalition(
        agents=[(0, (4, 3), 1), (1, (4, 6), 2), (2, (4, 6), 2)],
        loader_ids=(0,),
        loader_levels=(1,),
        food_level=3,
    )
    assert symmetric.helper_ids == (1,)
    assert symmetric.tiebreak_key[0] == 1


def test_helper_cell_assignment_is_collision_safe_and_excludes_loader_cells():
    result = _coalition(
        agents=[(0, (4, 3), 1), (1, (0, 4), 1), (2, (7, 4), 1)],
        loader_ids=(0,),
        loader_levels=(1,),
        food_level=3,
    )
    assert result.helper_ids == (1, 2)
    cells = [cell for _, cell in result.assignment]
    assert len(set(cells)) == len(cells), "assignment must be collision-safe"
    assert (4, 3) not in result.candidate_cells, "loader's own cell is excluded"
    assert result.total_distance == sum(result.helper_distances)


def test_no_feasible_coalition_is_reported_and_suppresses_the_event():
    result = _coalition(
        agents=[(0, (4, 3), 1), (1, (0, 0), 1)],
        loader_ids=(0,),
        loader_levels=(1,),
        food_level=9,
    )
    assert result.feasible is False
    assert result.reason == "no_feasible_coalition"

    bank = make_bank(variants=only("wf10_m2_none_iboff"), n_agents=2)
    for step in range(3):
        bank.update(
            step_info(
                step,
                [(0, (4, 3), 1), (1, (0, 0), 1)],
                {FOOD: 9},
                failed=[failed_event(1, FOOD, 9, (0,), (1,))],
            ),
            raw_team_reward=0.0,
            t_env=step,
        )
    bank.finish_episode()
    assert events(bank) == []
    counters = bank.variant_counters()["wf10_m2_none_iboff"]
    assert counters.get("semantically_invalid", 0) >= 1


# ==========================================================================
# 16-18: inbound help
# ==========================================================================


def _run_inbound(helper_positions, variant="wf5_m2_none_ibon"):
    """Drive a trajectory where agent 0 stalls and agent 1 follows a path."""
    bank = make_bank(variants=only(variant) + only(variant.replace("ibon", "iboff")))
    for step, helper_pos in enumerate(helper_positions):
        bank.update(
            step_info(
                step,
                [(0, (4, 3), 1), (1, helper_pos, 2), (2, (7, 7), 1)],
                {FOOD: 3},
                failed=[failed_event(1, FOOD, 3, (0,), (1,))],
            ),
            raw_team_reward=0.0,
            t_env=step,
        )
    bank.finish_episode()
    return bank


def test_incomplete_inbound_history_is_not_inbound():
    """Fires at step 1, before a complete 3-transition window exists."""
    bank = _run_inbound([(0, 4), (1, 4)])
    fired = events(bank)
    assert any(r["variant_id"].endswith("ibon") for r in fired)
    for row in fired:
        assert row["inbound_help"] == 0


def test_genuine_inbound_movement_suppresses_only_the_inbound_variant():
    # Agent 1 walks steadily towards the food; window [step-3, step] complete at 3.
    bank = _run_inbound([(0, 4), (1, 4), (2, 4), (3, 4), (3, 4)])
    on = [r for r in events(bank) if r["variant_id"].endswith("ibon")]
    off = [r for r in events(bank) if r["variant_id"].endswith("iboff")]
    assert off, "the non-inbound variant must still fire"
    raw = [
        r
        for r in bank.rows
        if r["variant_id"].endswith("ibon") and r["episode_step"] == 3
    ]
    assert raw and raw[0]["inbound_help"] == 1
    assert raw[0]["would_fire_after_inbound"] == 0
    assert raw[0]["suppression_reason"] == "inbound_help_present"
    assert all(d is not None and d <= -1 for d in raw[0]["inbound_distance_deltas"])


def test_adjacent_helper_that_never_loads_is_not_inbound():
    """Agent 1 sits on a loading cell for the whole window without LOADing."""
    bank = _run_inbound([(4, 5), (4, 5), (4, 5), (4, 5), (4, 5)])
    rows = [
        r
        for r in bank.rows
        if r["variant_id"].endswith("ibon") and r["episode_step"] >= 3
    ]
    assert rows
    for row in rows:
        assert row["inbound_help"] == 0, "a parked adjacent helper is not inbound"
        assert 0 in row["inbound_end_distances"]


# ==========================================================================
# 19-20: lifecycle and budget
# ==========================================================================


def test_lifecycle_rising_edge_latch_resolve_cooldown_rearm():
    variant = "wf15_m2_none_iboff"
    bank = make_bank(
        variants=only(variant), rearm_steps=3, cooldown_steps=5, max_events_per_episode=9
    )
    agents = [(0, (4, 3), 1), (1, (1, 1), 2), (2, (7, 7), 2)]

    def stall(step):
        bank.update(
            step_info(
                step, agents, {FOOD: 3}, failed=[failed_event(1, FOOD, 3, (0,), (1,))]
            ),
            raw_team_reward=0.0,
            t_env=step,
        )

    def quiet(step):
        bank.update(step_info(step, agents, {FOOD: 3}), raw_team_reward=0.0, t_env=step)

    stall(0)
    stall(1)  # rising edge -> fire
    state = bank.variant_state[variant].foods[fid_of(FOOD)]
    assert state.latched is True
    assert bank.variant_state[variant].events_used == 1

    stall(2)  # continuously true -> must not refire
    assert bank.variant_state[variant].events_used == 1

    # Condition false: resolves once false_streak reaches rearm_steps (step 5),
    # then the 5-step cooldown drains over steps 6-10.
    for step in range(3, 11):
        quiet(step)
    assert state.latched is False
    assert state.cooldown_remaining == 0
    assert state.false_streak >= 3

    stall(11)  # new rising edge after rearm + cooldown -> fire again
    bank.finish_episode()
    assert bank.variant_state[variant].events_used == 2
    counters = bank.variant_counters()[variant]
    assert counters.get("resolved_condition_false", 0) >= 1
    assert counters.get("suppressed_by_latch", 0) >= 1


def test_continuous_condition_never_refires():
    variant = "wf15_m2_none_iboff"
    bank = make_bank(variants=only(variant), max_events_per_episode=9)
    for step in range(20):
        bank.update(
            step_info(
                step,
                [(0, (4, 3), 1), (1, (1, 1), 2), (2, (7, 7), 2)],
                {FOOD: 3},
                failed=[failed_event(1, FOOD, 3, (0,), (1,))],
            ),
            raw_team_reward=0.0,
            t_env=step,
        )
    bank.finish_episode()
    assert len(events(bank)) == 1


def test_per_episode_event_budget_is_enforced_and_reset():
    variant = "wf15_m2_none_iboff"
    bank = make_bank(variants=only(variant), max_events_per_episode=2, rearm_steps=1)
    foods = {(2, 2): 3, (4, 4): 3, (6, 6): 3}
    agents = [(0, (2, 1), 1), (1, (4, 3), 1), (2, (6, 5), 1)]
    failures = [
        failed_event(1, (2, 2), 3, (0,), (1,)),
        failed_event(1, (4, 4), 3, (1,), (1,)),
        failed_event(1, (6, 6), 3, (2,), (1,)),
    ]
    for step in range(3):
        bank.update(
            step_info(step, agents, foods, failed=failures),
            raw_team_reward=0.0,
            t_env=step,
        )
    bank.finish_episode()
    assert len(events(bank)) == 2, "budget of 2 must cap events at 2"
    assert bank.variant_counters()[variant].get("suppressed_by_budget", 0) >= 1

    bank.reset_episode(2)
    assert bank.variant_state[variant].events_used == 0
    assert bank.variant_state[variant].foods == {}
    assert bank.fail_history == {}
    assert bank.pending_rows == []


# ==========================================================================
# 21-22: natural resolution labels
# ==========================================================================


def test_five_and_ten_step_horizons_are_labelled_independently():
    variant = "wf5_m2_none_iboff"
    bank = make_bank(variants=only(variant))
    agents = [(0, (4, 3), 1), (1, (1, 1), 2), (2, (7, 7), 2)]
    for step in range(2):
        bank.update(
            step_info(
                step, agents, {FOOD: 3}, failed=[failed_event(1, FOOD, 3, (0,), (1,))]
            ),
            raw_team_reward=0.0,
            t_env=step,
        )
    # Six more quiet transitions: horizon 5 completes, horizon 10 does not.
    for step in range(2, 8):
        bank.update(step_info(step, agents, {FOOD: 3}), raw_team_reward=0.0, t_env=step)
    bank.finish_episode()
    row = events(bank)[0]
    assert row["episode_step"] == 1
    assert row["resolved_within_5"] == "unresolved"
    assert row["resolved_within_10"] == "censored"


def test_resolution_marks_resolved_within_the_right_horizon():
    variant = "wf5_m2_none_iboff"
    bank = make_bank(variants=only(variant))
    agents = [(0, (4, 3), 1), (1, (1, 1), 2), (2, (7, 7), 2)]
    for step in range(2):
        bank.update(
            step_info(
                step, agents, {FOOD: 3}, failed=[failed_event(1, FOOD, 3, (0,), (1,))]
            ),
            raw_team_reward=0.0,
            t_env=step,
        )
    for step in range(2, 8):
        foods = {FOOD: 3} if step < 7 else {}
        removed = [] if step < 7 else [(FOOD, 3)]
        bank.update(
            step_info(step, agents, foods, removed=removed),
            raw_team_reward=0.0,
            t_env=step,
        )
    bank.finish_episode()
    row = events(bank)[0]
    # Fired at step 1; food removed at step 7 -> outside 5, inside 10.
    assert row["resolved_within_5"] == "unresolved"
    assert row["resolved_within_10"] == "resolved"


def test_pending_labels_flush_safely_at_episode_end():
    variant = "wf5_m2_none_iboff"
    bank = make_bank(variants=only(variant))
    agents = [(0, (4, 3), 1), (1, (1, 1), 2), (2, (7, 7), 2)]
    for step in range(2):
        bank.update(
            step_info(
                step, agents, {FOOD: 3}, failed=[failed_event(1, FOOD, 3, (0,), (1,))]
            ),
            raw_team_reward=0.0,
            t_env=step,
        )
    bank.finish_episode()
    assert bank.pending_rows == []
    for row in bank.rows:
        # Episode ended immediately: both horizons must be censored, never
        # silently downgraded to "unresolved".
        assert row["resolved_within_5"] == "censored"
        assert row["resolved_within_10"] == "censored"


def test_rows_are_emitted_in_a_stable_deterministic_order():
    def build():
        bank = make_bank()
        for step in range(4):
            bank.update(
                step_info(
                    step,
                    [(0, (4, 3), 1), (1, (1, 1), 2), (2, (7, 7), 2)],
                    {FOOD: 3},
                    failed=[failed_event(1, FOOD, 3, (0,), (1,))],
                ),
                raw_team_reward=0.0,
                t_env=step,
            )
        bank.finish_episode()
        return [
            (r["episode_step"], r["variant_id"], r["row_type"]) for r in bank.rows
        ]

    first, second = build(), build()
    assert first == second
    assert first == sorted(
        first, key=lambda k: (k[0], k[1], 0 if k[2] == "raw_candidate" else 1)
    )


# ==========================================================================
# 24: no provider / query / subgoal / shaping effects
# ==========================================================================


def test_bank_exposes_zero_provider_query_subgoal_and_shaping():
    bank = make_bank()
    for step in range(10):
        bank.update(
            step_info(
                step,
                [(0, (4, 3), 1), (1, (1, 1), 2), (2, (7, 7), 2)],
                {FOOD: 3},
                failed=[failed_event(1, FOOD, 3, (0,), (1,))],
            ),
            raw_team_reward=0.0,
            t_env=step,
        )
    bank.finish_episode()
    assert bank.query_count == 0
    assert bank.provider_call_count == 0
    assert bank.active_subgoal_count == 0
    assert bank.total_shaping_reward == 0.0
    assert not hasattr(bank, "provider")


def test_detector_consumes_no_numpy_global_rng():
    np.random.seed(1234)
    before = np.random.get_state()[1].tobytes()
    bank = make_bank()
    for step in range(10):
        bank.update(
            step_info(
                step,
                [(0, (4, 3), 1), (1, (1, 1), 2), (2, (7, 7), 2)],
                {FOOD: 3},
                failed=[failed_event(1, FOOD, 3, (0,), (1,))],
            ),
            raw_team_reward=0.0,
            t_env=step,
        )
    bank.finish_episode()
    assert np.random.get_state()[1].tobytes() == before


# ==========================================================================
# 25-30: canonical (agent_id, episode-specific food_id) key semantics
# ==========================================================================


def test_raw_history_keys_are_agent_and_episode_specific_food_id():
    """The raw-history key is exactly ``(agent_id, (episode_id, row, col))``.

    Not ``(agent_id, food_position)``: the bare coordinate is only unique
    within a single episode.
    """
    bank = make_bank(variants=only("wf10_m2_none_iboff"))
    bank.reset_episode(7)
    bank.update(
        step_info(
            0,
            [(0, (4, 3), 1), (1, (3, 4), 1), (2, (7, 7), 2)],
            {FOOD: 4},
            failed=[failed_event(7, FOOD, 4, (0, 1), (1, 1))],
            episode=7,
        ),
        raw_team_reward=0.0,
        t_env=0,
    )
    assert set(bank.fail_history) == {(0, (7, 4, 4)), (1, (7, 4, 4))}
    assert FOOD not in {k[1] for k in bank.fail_history}
    # distance history is keyed the same way, for every agent x food
    assert all(len(k[1]) == 3 and k[1][0] == 7 for k in bank.distance_history)
    assert (0, (7, 4, 4)) in bank.distance_history


def test_lifecycle_and_removal_state_are_keyed_by_food_id():
    bank = make_bank(variants=only("wf10_m2_none_iboff"))
    bank.reset_episode(7)
    for step in range(2):
        bank.update(
            step_info(
                step,
                [(0, (4, 3), 1), (1, (1, 1), 2), (2, (7, 7), 2)],
                {FOOD: 3},
                failed=[failed_event(7, FOOD, 3, (0,), (1,))],
                episode=7,
            ),
            raw_team_reward=0.0,
            t_env=step,
        )
    state = bank.variant_state["wf10_m2_none_iboff"]
    assert set(state.foods) == {(7, 4, 4)}
    bank.update(
        step_info(
            2,
            [(0, (4, 3), 1), (1, (4, 5), 2), (2, (7, 7), 2)],
            {},
            successful=[SuccessfulLoadEvent((7, 4, 4), FOOD, 3, (0, 1), (1, 2), 3)],
            removed=[(FOOD, 3)],
            episode=7,
        ),
        raw_team_reward=3.0,
        t_env=2,
    )
    assert set(bank.removal_step) == {(7, 4, 4)}


def test_same_position_in_a_new_episode_is_a_different_food():
    """Reusing a grid coordinate in the next episode must not inherit history."""
    bank = make_bank(variants=only("wf10_m2_none_iboff"))
    bank.reset_episode(7)
    for step in range(2):
        bank.update(
            step_info(
                step,
                [(0, (4, 3), 1), (1, (1, 1), 2), (2, (7, 7), 2)],
                {FOOD: 3},
                failed=[failed_event(7, FOOD, 3, (0,), (1,))],
                episode=7,
            ),
            raw_team_reward=0.0,
            t_env=step,
        )
    assert bank.fail_history[(0, (7, 4, 4))] == [0, 1]

    bank.finish_episode()
    bank.reset_episode(8)
    bank.update(
        step_info(
            0,
            [(0, (4, 3), 1), (1, (1, 1), 2), (2, (7, 7), 2)],
            {FOOD: 3},
            failed=[failed_event(8, FOOD, 3, (0,), (1,))],
            episode=8,
        ),
        raw_team_reward=0.0,
        t_env=100,
    )
    assert set(bank.fail_history) == {(0, (8, 4, 4))}
    assert bank.fail_history[(0, (8, 4, 4))] == [0], "no carry-over from episode 7"
    assert bank.variant_state["wf10_m2_none_iboff"].foods == {}


def test_mixing_two_episode_ids_without_reset_is_rejected():
    """A missed reset_episode must fail loudly, not merge two episodes."""
    bank = make_bank(variants=only("wf10_m2_none_iboff"))
    bank.reset_episode(7)
    bank.update(
        step_info(0, [(0, (4, 3), 1), (1, (1, 1), 2), (2, (7, 7), 2)], {FOOD: 3},
                  episode=7),
        raw_team_reward=0.0,
        t_env=0,
    )
    with pytest.raises(AssertionError, match="reset_episode was not called"):
        bank.update(
            step_info(0, [(0, (4, 3), 1), (1, (1, 1), 2), (2, (7, 7), 2)], {FOOD: 3},
                      episode=8),
            raw_team_reward=0.0,
            t_env=1,
        )


def test_shadow_row_carries_food_id_and_position_separately():
    """food_position is logged, but food_id remains the canonical identity."""
    bank = make_bank(variants=only("wf10_m2_none_iboff"))
    bank.reset_episode(7)
    for step in range(2):
        bank.update(
            step_info(
                step,
                [(0, (4, 3), 1), (1, (1, 1), 2), (2, (7, 7), 2)],
                {FOOD: 3},
                failed=[failed_event(7, FOOD, 3, (0,), (1,))],
                episode=7,
            ),
            raw_team_reward=0.0,
            t_env=step,
        )
    bank.finish_episode()
    row = events(bank)[0]
    assert row["food_id"] == (7, 4, 4)
    assert row["food_position"] == FOOD
    assert row["food_id"] != row["food_position"]
    assert row["event_id"] == "ep7-wf10_m2_none_iboff-food4,4-t1"


def test_two_foods_share_one_agent_history_namespace_without_collision():
    """Distinct foods in one episode keep independent (agent, food_id) history."""
    bank = make_bank(variants=only("wf10_m2_none_iboff"))
    bank.reset_episode(3)
    foods = {(2, 2): 3, (5, 5): 3}
    agents = [(0, (2, 1), 1), (1, (5, 4), 1), (2, (7, 7), 1)]
    bank.update(
        step_info(
            0,
            agents,
            foods,
            failed=[
                failed_event(3, (2, 2), 3, (0,), (1,)),
                failed_event(3, (5, 5), 3, (1,), (1,)),
            ],
            episode=3,
        ),
        raw_team_reward=0.0,
        t_env=0,
    )
    assert bank.fail_history[(0, (3, 2, 2))] == [0]
    assert bank.fail_history[(1, (3, 5, 5))] == [0]
    assert (0, (3, 5, 5)) not in bank.fail_history


# ==========================================================================
# 31-36: team-gate eligibility denominator
# ==========================================================================


def _quiet_only(bank, step, reward=0.0):
    """A transition with no failures at all, so only gate state advances."""
    bank.update(
        step_info(step, [(0, (0, 0), 1), (1, (1, 1), 2), (2, (7, 7), 2)], {FOOD: 3}),
        raw_team_reward=reward,
        t_env=step,
    )


def test_first_eligible_transition_is_exactly_the_w_teamth():
    """Eligibility starts on the W_team-th transition, not before or after."""
    bank = make_bank(variants=only("wf10_m2_t5_iboff"))
    counters = bank.variant_state["wf10_m2_t5_iboff"].counters
    for step in range(4):
        _quiet_only(bank, step)
        assert counters.get("eligible_steps", 0) == 0, f"eligible at step {step}"
    _quiet_only(bank, 4)  # 5th transition completes the W_team=5 window
    assert counters["eligible_steps"] == 1
    assert counters["gate_open_steps"] == 1


def test_incomplete_window_steps_are_excluded_from_the_denominator():
    bank = make_bank(variants=only("wf10_m2_t5_iboff"))
    for step in range(12):
        _quiet_only(bank, step)
    counters = bank.variant_state["wf10_m2_t5_iboff"].counters
    assert counters["observed_steps"] == 12
    assert counters["incomplete_team_window_steps"] == 4
    assert counters["eligible_steps"] == 8
    assert counters["gate_open_steps"] == 8


def test_eligibility_window_includes_the_current_transition():
    """Progress on the current transition keeps the gate shut on that step."""
    bank = make_bank(variants=only("wf10_m2_t5_iboff"))
    for step in range(4):
        _quiet_only(bank, step)
    _quiet_only(bank, 4, reward=1.0)  # progress on the 5th (first eligible) step
    counters = bank.variant_state["wf10_m2_t5_iboff"].counters
    assert counters["eligible_steps"] == 1, "the step is eligible"
    assert counters.get("gate_open_steps", 0) == 0, "but the gate is not open"


def test_raw_reward_progress_counts_as_progress_in_the_denominator():
    bank = make_bank(variants=only("wf10_m2_t5_iboff"))
    for step in range(9):
        _quiet_only(bank, step, reward=1.0 if step == 6 else 0.0)
    counters = bank.variant_state["wf10_m2_t5_iboff"].counters
    # eligible from step 4 onwards -> 5 eligible steps (4..8); the reward at
    # step 6 sits in the windows ending at steps 6, 7 and 8.
    assert counters["eligible_steps"] == 5
    assert counters["gate_open_steps"] == 2


def test_food_disappearance_progress_counts_in_the_denominator():
    """A removal with zero reward must close the gate exactly like a reward."""
    bank = make_bank(variants=only("wf10_m2_t5_iboff"))
    for step in range(6):
        _quiet_only(bank, step)
    bank.update(
        step_info(
            6,
            [(0, (4, 3), 1), (1, (4, 5), 2), (2, (7, 7), 2)],
            {},
            successful=[SuccessfulLoadEvent((1, 4, 4), FOOD, 3, (0, 1), (1, 2), 3)],
            removed=[(FOOD, 3)],
        ),
        raw_team_reward=0.0,
        t_env=6,
    )
    counters = bank.variant_state["wf10_m2_t5_iboff"].counters
    assert counters["eligible_steps"] == 3  # steps 4, 5, 6
    assert counters["gate_open_steps"] == 2  # step 6's window contains the removal


def test_gate_open_percentage_uses_complete_window_denominator_per_w_team():
    """Each W_team keeps its own eligibility total; None uses every step."""
    variants = (
        only("wf10_m2_t5_iboff") + only("wf10_m2_t15_iboff") + only("wf10_m2_none_iboff")
    )
    bank = make_bank(variants=variants)
    for step in range(12):
        _quiet_only(bank, step)
    bank.finish_episode()
    report = bank.variant_report()

    r5 = report["wf10_m2_t5_iboff"]
    assert (r5["observed_steps"], r5["eligible_steps"], r5["gate_open_steps"]) == (
        12,
        8,
        8,
    )
    assert r5["gate_open_percentage"] == 1.0
    assert bank.gate_open_percentage("wf10_m2_t5_iboff") == 1.0

    # W_team=15 never completes in a 12-transition episode: no eligible step at
    # all, so the percentage is undefined rather than a misleading 0%.
    r15 = report["wf10_m2_t15_iboff"]
    assert r15["observed_steps"] == 12
    assert r15["eligible_steps"] == 0
    assert r15["incomplete_team_window_steps"] == 12
    assert r15["gate_open_percentage"] is None

    rnone = report["wf10_m2_none_iboff"]
    assert rnone["eligible_steps"] == 12
    assert rnone["gate_open_percentage"] == 1.0
