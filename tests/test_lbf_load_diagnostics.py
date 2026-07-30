"""Focused tests for the SELMA LBF load-event diagnostics.

Covers the pure per-step computation (src/envs/gymma.py::lbf_load_diagnostics)
directly with hand-crafted field/position/level/action snapshots, plus a couple
of end-to-end checks that drive the real GymmaWrapper + lbforaging env to make
sure the wrapper wiring (snapshot timing, action casting, info merging) is
correct and does not disturb obs/reward/termination.

Run with:  PYTHONPATH=src pytest tests/test_lbf_load_diagnostics.py -v
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import lbforaging  # noqa: E402,F401  (registers the Foraging-* gym ids)
from envs.gymma import GymmaWrapper, lbf_load_diagnostics  # noqa: E402

NONE, NORTH, SOUTH, WEST, EAST, LOAD = 0, 1, 2, 3, 4, 5


def make_field(shape, foods):
    field = np.zeros(shape, dtype=np.int32)
    for (r, c), level in foods.items():
        field[r, c] = level
    return field


# ---------------------------------------------------------------------------
# Pure-function tests (no gym env involved)
# ---------------------------------------------------------------------------


def test_one_solo_load():
    pre_field = make_field((5, 5), {(2, 2): 1})
    post_field = make_field((5, 5), {})  # food removed
    positions = [(2, 3), (0, 0)]  # agent0 east-adjacent to food; agent1 far away
    levels = [3, 1]
    actions = [LOAD, NONE]

    info = lbf_load_diagnostics(pre_field, positions, levels, actions, post_field)

    assert info == {
        "successful_load_events": 1,
        "solo_load_events": 1,
        "multi_agent_load_events": 0,
        "necessary_cooperative_load_events": 0,
        "failed_load_events": 0,
        "failed_load_attempts": 0,
    }


def test_one_two_agent_load_not_necessary():
    # food level 2; agent0 alone (level 2) is already sufficient, agent1 (level 1)
    # joins anyway -> multi-agent participation, but not *necessary* cooperation.
    pre_field = make_field((5, 5), {(2, 2): 2})
    post_field = make_field((5, 5), {})
    positions = [(1, 2), (2, 1)]  # north and west of food
    levels = [2, 1]
    actions = [LOAD, LOAD]

    info = lbf_load_diagnostics(pre_field, positions, levels, actions, post_field)

    assert info["successful_load_events"] == 1
    assert info["solo_load_events"] == 0
    assert info["multi_agent_load_events"] == 1
    assert info["necessary_cooperative_load_events"] == 0
    assert info["failed_load_events"] == 0
    assert info["failed_load_attempts"] == 0


def test_one_necessary_cooperative_load():
    # food level 2; neither agent (level 1 each) can solo it, but together they can.
    pre_field = make_field((5, 5), {(2, 2): 2})
    post_field = make_field((5, 5), {})
    positions = [(1, 2), (2, 1)]
    levels = [1, 1]
    actions = [LOAD, LOAD]

    info = lbf_load_diagnostics(pre_field, positions, levels, actions, post_field)

    assert info["successful_load_events"] == 1
    assert info["solo_load_events"] == 0
    assert info["multi_agent_load_events"] == 1
    assert info["necessary_cooperative_load_events"] == 1
    assert info["failed_load_events"] == 0
    assert info["failed_load_attempts"] == 0


def test_one_failed_load():
    # food level 3; the only adjacent loader has level 1 -> insufficient, food stays.
    pre_field = make_field((5, 5), {(2, 2): 3})
    post_field = make_field((5, 5), {(2, 2): 3})  # unchanged
    positions = [(1, 2), (0, 0)]
    levels = [1, 1]
    actions = [LOAD, NONE]

    info = lbf_load_diagnostics(pre_field, positions, levels, actions, post_field)

    assert info == {
        "successful_load_events": 0,
        "solo_load_events": 0,
        "multi_agent_load_events": 0,
        "necessary_cooperative_load_events": 0,
        "failed_load_events": 1,
        "failed_load_attempts": 1,
    }


def test_two_independent_solo_loads_same_timestep():
    pre_field = make_field((8, 8), {(2, 2): 1, (5, 5): 1})
    post_field = make_field((8, 8), {})  # both removed
    positions = [(2, 3), (5, 6), (0, 0), (0, 1)]
    levels = [2, 2, 1, 1]
    actions = [LOAD, LOAD, NONE, NONE]

    info = lbf_load_diagnostics(pre_field, positions, levels, actions, post_field)

    assert info == {
        "successful_load_events": 2,
        "solo_load_events": 2,
        "multi_agent_load_events": 0,
        "necessary_cooperative_load_events": 0,
        "failed_load_events": 0,
        "failed_load_attempts": 0,
    }


def test_multiple_successful_events_one_timestep_mixed_solo_and_cooperative():
    # food A: solo load by agent0. food B: necessary cooperative load by agent1+agent2.
    pre_field = make_field((8, 8), {(2, 2): 1, (5, 5): 2})
    post_field = make_field((8, 8), {})  # both removed
    positions = [(2, 3), (4, 5), (6, 5)]
    levels = [2, 1, 1]
    actions = [LOAD, LOAD, LOAD]

    info = lbf_load_diagnostics(pre_field, positions, levels, actions, post_field)

    assert info == {
        "successful_load_events": 2,
        "solo_load_events": 1,
        "multi_agent_load_events": 1,
        "necessary_cooperative_load_events": 1,
        "failed_load_events": 0,
        "failed_load_attempts": 0,
    }


def test_failed_and_successful_events_do_not_cross_contaminate():
    # food A succeeds solo; food B (different cell) fails; a bystander LOAD with no
    # adjacent food at all contributes to neither.
    pre_field = make_field((8, 8), {(2, 2): 1, (5, 5): 3})
    post_field = make_field((8, 8), {(5, 5): 3})  # only A removed
    positions = [(2, 3), (4, 5), (0, 0)]
    levels = [2, 1, 1]
    actions = [LOAD, LOAD, LOAD]  # agent2's LOAD has no adjacent food anywhere

    info = lbf_load_diagnostics(pre_field, positions, levels, actions, post_field)

    assert info["successful_load_events"] == 1
    assert info["solo_load_events"] == 1
    assert info["failed_load_events"] == 1
    assert info["failed_load_attempts"] == 1


def test_invariants_hold_by_construction_on_a_larger_random_like_case():
    pre_field = make_field((6, 6), {(1, 1): 1, (3, 3): 2, (4, 4): 5})
    post_field = make_field((6, 6), {(4, 4): 5})  # first two removed, third fails
    positions = [(1, 2), (2, 3), (4, 3), (3, 4), (0, 0)]
    levels = [1, 1, 1, 1, 1]
    actions = [LOAD, LOAD, LOAD, LOAD, NONE]

    info = lbf_load_diagnostics(pre_field, positions, levels, actions, post_field)

    assert info["solo_load_events"] + info["multi_agent_load_events"] == info[
        "successful_load_events"
    ]
    assert info["necessary_cooperative_load_events"] <= info["multi_agent_load_events"]


# ---------------------------------------------------------------------------
# Wiring tests: drive the real GymmaWrapper + lbforaging env end-to-end.
# ---------------------------------------------------------------------------


def _make_lbf_wrapper(key, n_players):
    wrapper = GymmaWrapper(
        key=key,
        time_limit=50,
        pretrained_wrapper=None,
        seed=0,
        common_reward=False,
        reward_scalarisation="sum",
    )
    assert wrapper._is_lbforaging
    wrapper.reset()
    return wrapper


def _force_state(wrapper, foods, players):
    """Overwrite post-reset LBF state to engineer a deterministic scenario.

    `foods`: dict of (row, col) -> level. `players`: list of (position, level).
    Refreshes the env's internal valid-move cache so the new state is honoured
    by env.step's action validation. No env.reset/step calls are made here.
    """
    unwrapped = wrapper._env.unwrapped
    unwrapped.field[:] = 0
    for (r, c), level in foods.items():
        unwrapped.field[r, c] = level
    for player, (pos, level) in zip(unwrapped.players, players):
        player.position = pos
        player.level = level
    unwrapped._gen_valid_moves()


def test_wiring_solo_load_end_to_end():
    wrapper = _make_lbf_wrapper("Foraging-8x8-2p-1f-v3", 2)
    _force_state(
        wrapper,
        foods={(3, 3): 1},
        players=[((3, 4), 2), ((0, 0), 1)],
    )

    obs, reward, done, truncated, info = wrapper.step([LOAD, NONE])

    assert info["successful_load_events"] == 1
    assert info["solo_load_events"] == 1
    assert info["multi_agent_load_events"] == 0
    assert info["necessary_cooperative_load_events"] == 0
    assert info["failed_load_events"] == 0
    assert info["failed_load_attempts"] == 0
    assert wrapper._env.unwrapped.field[3, 3] == 0
    assert reward[0] > 0
    assert reward[1] == 0
    assert len(obs) == 2


def test_wiring_necessary_cooperative_load_end_to_end():
    wrapper = _make_lbf_wrapper("Foraging-8x8-2p-1f-v3", 2)
    _force_state(
        wrapper,
        foods={(3, 3): 2},
        players=[((2, 3), 1), ((3, 2), 1)],
    )

    obs, reward, done, truncated, info = wrapper.step([LOAD, LOAD])

    assert info["successful_load_events"] == 1
    assert info["multi_agent_load_events"] == 1
    assert info["necessary_cooperative_load_events"] == 1
    assert wrapper._env.unwrapped.field[3, 3] == 0
    assert reward[0] > 0
    assert reward[1] > 0
    # HARD invariant (mirrors the one enforced in EpisodeRunner): a multi-agent
    # load implies both participants were rewarded that same step.
    n_rewarded = sum(1 for r in reward if r > 0)
    assert n_rewarded >= 2


def test_lbf_gating_is_module_scoped():
    # GymmaWrapper gates the LBF diagnostics on
    # `type(self._env.unwrapped).__module__.startswith("lbforaging")`. Confirm
    # that check is true for the real lbforaging env class and false for an
    # unrelated class, so other Gymma environments never trigger LBF-only reads
    # of `.field` / `.players` (which they don't have).
    from lbforaging.foraging.environment import ForagingEnv

    assert type(ForagingEnv).__name__ == "type"  # sanity: it's a normal class
    assert ForagingEnv.__module__.startswith("lbforaging")

    class NotAnLbfEnv:
        pass

    assert not NotAnLbfEnv.__module__.startswith("lbforaging")
