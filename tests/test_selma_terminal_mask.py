"""SELMA Stage 0: mandatory terminal-mask probe.

Records, for the two possible episode endings in LBF + GymmaWrapper, what the
episode runner actually stores in ``batch["terminated"]`` and therefore whether
the QMIX target bootstraps the next-state value.

The stored value replicates the exact expression in
``src/runners/episode_runner.py``::

    terminated = terminated or truncated
    "terminated": [(terminated != env_info.get("episode_limit", False),)]

and the bootstrap factor replicates ``src/learners/q_learner.py``::

    targets = rewards + gamma * (1 - terminated) * target_max_qvals

Read-only probe: it does not modify any SELMA/EPyMARL source behaviour.

Run with:  PYTHONPATH=src pytest tests/test_selma_terminal_mask.py -v -s
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import lbforaging  # noqa: E402,F401  (registers the Foraging-* gym ids)
from envs.gymma import GymmaWrapper  # noqa: E402

NONE, NORTH, SOUTH, WEST, EAST, LOAD = 0, 1, 2, 3, 4, 5

KEY = "lbforaging:Foraging-8x8-2p-2f-coop-v3"
TIME_LIMIT = 50


def make_env(seed=91):
    return GymmaWrapper(
        key=KEY,
        time_limit=TIME_LIMIT,
        pretrained_wrapper=None,
        seed=seed,
        common_reward=True,
        reward_scalarisation="sum",
    )


def runner_stored_terminated(raw_terminated, raw_truncated, env_info):
    """Exact reproduction of episode_runner.py's stored terminated flag."""
    terminated = raw_terminated or raw_truncated
    return bool(terminated != env_info.get("episode_limit", False))


def qmix_bootstrap_factor(stored_terminated):
    """Exact reproduction of q_learner.py's (1 - terminated) coefficient."""
    return 1.0 - float(stored_terminated)


def probe(raw_terminated, raw_truncated, env_info, cause):
    stored = runner_stored_terminated(raw_terminated, raw_truncated, env_info)
    factor = qmix_bootstrap_factor(stored)
    record = {
        "cause": cause,
        "raw_terminated": bool(raw_terminated),
        "raw_truncated": bool(raw_truncated),
        "env_info_keys": sorted(env_info.keys()),
        "env_info_has_episode_limit": "episode_limit" in env_info,
        "stored_terminated": stored,
        "qmix_bootstrap_factor": factor,
        "qmix_bootstraps": factor == 1.0,
    }
    print("\nSELMA TERMINAL PROBE:", record)
    return record


def test_true_terminal_all_food_collected():
    """Ending 1: all food collected before the step limit (true task terminal).

    The LBF state is placed deterministically (single food, two adjacent agents
    whose summed level suffices) and then driven through the *real* LBF load and
    termination path, so ``_game_over`` is set by ``field.sum() == 0``.
    """
    env = make_env()
    env.reset()
    u = env._env.unwrapped

    # Deterministic terminal setup: one level-3 food at (4, 4); agents at (4, 3)
    # and (4, 5), levels 2 and 2 -> summed level 4 >= 3, so the LOAD succeeds and
    # clears the last food well before step 50.
    u.field[:] = 0
    u.field[4, 4] = 3
    u.players[0].position = (4, 3)
    u.players[0].level = 2
    u.players[1].position = (4, 5)
    u.players[1].level = 2
    u.current_step = 10
    u._gen_valid_moves()

    _, reward, raw_terminated, raw_truncated, info = env.step([LOAD, LOAD])

    assert int(u.field.sum()) == 0, "probe setup failed: food was not collected"
    assert u.current_step < TIME_LIMIT, "probe setup failed: hit the step limit"
    assert reward > 0.0, "probe setup failed: successful load produced no reward"

    rec = probe(raw_terminated, raw_truncated, info, "true_terminal")

    assert rec["raw_terminated"] is True
    assert rec["raw_truncated"] is False
    assert rec["env_info_has_episode_limit"] is False
    assert rec["stored_terminated"] is True
    assert rec["qmix_bootstraps"] is False


def test_time_limit_with_food_remaining():
    """Ending 2: step limit reached with food still on the field."""
    env = make_env()
    env.reset()
    u = env._env.unwrapped

    raw_terminated = raw_truncated = False
    info = {}
    steps = 0
    # NONE actions never collect food, so the episode can only end at the limit.
    while not (raw_terminated or raw_truncated):
        _, _, raw_terminated, raw_truncated, info = env.step([NONE] * env.n_agents)
        steps += 1
        assert steps <= TIME_LIMIT + 1, "episode did not stop at the time limit"

    assert steps == TIME_LIMIT
    assert int(u.field.sum()) > 0, "probe setup failed: no food remaining"

    rec = probe(raw_terminated, raw_truncated, info, "time_limit")

    assert rec["raw_truncated"] is True
    # lbforaging 2.0.0 also sets its own done at its internal max_episode_steps,
    # so the raw terminated flag is True at the time limit as well.
    assert rec["raw_terminated"] is True
    assert rec["env_info_has_episode_limit"] is False
    assert rec["stored_terminated"] is True
    assert rec["qmix_bootstraps"] is False


def test_stage0_boundary_rule_is_no_bootstrap_in_both_cases():
    """Stage 0 verdict: phi_next_effective must be 0.0 for BOTH endings."""
    env = make_env()
    env.reset()
    u = env._env.unwrapped
    u.field[:] = 0
    u.field[4, 4] = 3
    u.players[0].position = (4, 3)
    u.players[0].level = 2
    u.players[1].position = (4, 5)
    u.players[1].level = 2
    u.current_step = 10
    u._gen_valid_moves()
    _, _, t1, tr1, i1 = env.step([LOAD, LOAD])

    env2 = make_env()
    env2.reset()
    t2 = tr2 = False
    i2 = {}
    while not (t2 or tr2):
        _, _, t2, tr2, i2 = env2.step([NONE] * env2.n_agents)

    for raw_t, raw_tr, inf in ((t1, tr1, i1), (t2, tr2, i2)):
        stored = runner_stored_terminated(raw_t, raw_tr, inf)
        assert stored is True
        assert qmix_bootstrap_factor(stored) == 0.0
