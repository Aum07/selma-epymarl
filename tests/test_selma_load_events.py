"""SELMA Stage A tests: structured LBF load-event extraction.

Covers the pure builder (src/selma/load_events.py) with hand-crafted snapshots,
the GymmaWrapper accessor wiring, and the regression guarantees that Stage A
changes nothing observable about the environment or the existing counters.

Hand-crafted fixtures deliberately keep food away from border rows/columns and
never place an agent adjacent to two foods, matching what lbforaging's spawn
rules can actually produce.

Run with:  PYTHONPATH=src pytest tests/test_selma_load_events.py -v
"""
import os
import sys
import types as pytypes

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import lbforaging  # noqa: E402,F401  (registers the Foraging-* gym ids)
from envs.gymma import GymmaWrapper, lbf_load_diagnostics  # noqa: E402
from selma.load_events import (  # noqa: E402
    assert_no_ambiguous_adjacency,
    build_selma_step_info,
)

NONE, NORTH, SOUTH, WEST, EAST, LOAD = 0, 1, 2, 3, 4, 5

SUPPORTED_ENVS = [
    "lbforaging:Foraging-8x8-2p-2f-coop-v3",
    "lbforaging:Foraging-10x10-3p-3f-v3",
]


def make_field(shape, foods):
    field = np.zeros(shape, dtype=np.int32)
    for (r, c), level in foods.items():
        field[r, c] = level
    return field


def build(pre_foods, post_foods, positions, levels, actions, shape=(8, 8), episode=7):
    """Convenience wrapper: build a step record from food dicts."""
    return build_selma_step_info(
        episode_id=episode,
        episode_step=3,
        pre_field=make_field(shape, pre_foods),
        positions=positions,
        levels=levels,
        actions=actions,
        post_field=make_field(shape, post_foods),
    )


def make_env(key=SUPPORTED_ENVS[0], seed=91, time_limit=50):
    return GymmaWrapper(
        key=key,
        time_limit=time_limit,
        pretrained_wrapper=None,
        seed=seed,
        common_reward=True,
        reward_scalarisation="sum",
    )


# --------------------------------------------------------------------------
# A.4 (1) Invalid LOAD with no adjacent food
# --------------------------------------------------------------------------


def test_invalid_load_with_no_adjacent_food():
    info = build(
        pre_foods={(4, 4): 3},
        post_foods={(4, 4): 3},
        positions=[(1, 1), (6, 6)],
        levels=[1, 2],
        actions=[LOAD, LOAD],
    )
    assert info.invalid_load_agents == (0, 1)
    # No agent was adjacent, so this is not an adjacent_failed_load at all.
    assert info.failed_load_events == ()
    assert info.successful_load_events == ()


def test_adjacent_loader_is_not_flagged_invalid():
    info = build(
        pre_foods={(4, 4): 3},
        post_foods={(4, 4): 3},
        positions=[(4, 3), (6, 6)],
        levels=[1, 2],
        actions=[LOAD, LOAD],
    )
    assert info.invalid_load_agents == (1,)
    assert len(info.failed_load_events) == 1


# --------------------------------------------------------------------------
# A.4 (2) Lone agent adjacent to food with agent_level < food_level
# --------------------------------------------------------------------------


def test_lone_underlevelled_loader_is_coalition_required():
    info = build(
        pre_foods={(4, 4): 3},
        post_foods={(4, 4): 3},
        positions=[(4, 3), (1, 1)],
        levels=[1, 2],
        actions=[LOAD, NONE],
    )
    (event,) = info.failed_load_events
    assert event.food_id == (7, 4, 4)
    assert event.food_position == (4, 4)
    assert event.food_level == 3
    assert event.loader_ids == (0,)
    assert event.loader_levels == (1,)
    assert event.total_loader_level == 1
    assert event.level_deficit == 2
    assert event.coalition_required is True
    assert event.requires_help_by_agent == (True,)
    assert info.coalition_required_failed_loads == (event,)


# --------------------------------------------------------------------------
# A.4 (3) Two simultaneous LOADers whose summed level is still insufficient
# --------------------------------------------------------------------------


def test_two_loaders_with_insufficient_summed_level():
    info = build(
        pre_foods={(4, 4): 5},
        post_foods={(4, 4): 5},
        positions=[(4, 3), (3, 4)],
        levels=[1, 2],
        actions=[LOAD, LOAD],
    )
    (event,) = info.failed_load_events
    assert event.loader_ids == (0, 1)
    assert event.loader_levels == (1, 2)
    assert event.total_loader_level == 3
    assert event.level_deficit == 2
    assert event.coalition_required is True
    assert event.requires_help_by_agent == (True, True)


def test_sum_based_tag_differs_from_necessary_cooperative_semantics():
    """A capable agent plus a weak partner: sum-based and strict rules disagree.

    food_level=3, loaders levels (3, 1): the summed level 4 >= 3, so this is NOT
    coalition_required even though one loader individually could not do it. The
    per-loader requires_help flags still record the asymmetry.
    """
    info = build(
        pre_foods={(4, 4): 3},
        post_foods={(4, 4): 3},
        positions=[(4, 3), (3, 4)],
        levels=[3, 1],
        actions=[LOAD, LOAD],
    )
    (event,) = info.failed_load_events
    assert event.total_loader_level == 4
    assert event.coalition_required is False
    assert event.level_deficit == -1
    assert event.requires_help_by_agent == (False, True)
    assert info.coalition_required_failed_loads == ()


# --------------------------------------------------------------------------
# A.4 (4) A sufficient coalition that successfully removes food
# --------------------------------------------------------------------------


def test_sufficient_coalition_removes_food():
    info = build(
        pre_foods={(4, 4): 3},
        post_foods={},
        positions=[(4, 3), (3, 4)],
        levels=[1, 2],
        actions=[LOAD, LOAD],
    )
    assert info.failed_load_events == ()
    (event,) = info.successful_load_events
    assert event.food_id == (7, 4, 4)
    assert event.food_level == 3
    assert event.loader_ids == (0, 1)
    assert event.loader_levels == (1, 2)
    assert event.total_loader_level == 3


# --------------------------------------------------------------------------
# A.4 (5) An agent individually capable of loading its food
# --------------------------------------------------------------------------


def test_individually_capable_agent_requires_no_help():
    info = build(
        pre_foods={(4, 4): 2},
        post_foods={(4, 4): 2},
        positions=[(4, 3), (1, 1)],
        levels=[3, 1],
        actions=[LOAD, NONE],
    )
    (event,) = info.failed_load_events
    assert event.coalition_required is False
    assert event.requires_help_by_agent == (False,)
    assert event.level_deficit == -1


# --------------------------------------------------------------------------
# A.4 (6) Food identity persists across failed steps and disappears on success
# --------------------------------------------------------------------------


def test_food_identity_persists_then_disappears():
    ids = []
    for _ in range(3):
        info = build(
            pre_foods={(4, 4): 3},
            post_foods={(4, 4): 3},
            positions=[(4, 3), (1, 1)],
            levels=[1, 2],
            actions=[LOAD, NONE],
        )
        (event,) = info.failed_load_events
        ids.append(event.food_id)
    assert len(set(ids)) == 1, "food identity must be stable across failed steps"

    final = build(
        pre_foods={(4, 4): 3},
        post_foods={},
        positions=[(4, 3), (3, 4)],
        levels=[1, 2],
        actions=[LOAD, LOAD],
    )
    assert final.failed_load_events == ()
    assert final.successful_load_events[0].food_id == ids[0]

    # A different episode id yields a different identity for the same cell.
    other = build(
        pre_foods={(4, 4): 3},
        post_foods={(4, 4): 3},
        positions=[(4, 3), (1, 1)],
        levels=[1, 2],
        actions=[LOAD, NONE],
        episode=8,
    )
    assert other.failed_load_events[0].food_id != ids[0]


# --------------------------------------------------------------------------
# Immutability / no live references (constraint 6)
# --------------------------------------------------------------------------


def test_records_are_immutable_and_hold_no_live_reference():
    pre = make_field((8, 8), {(4, 4): 3})
    post = make_field((8, 8), {(4, 4): 3})
    info = build_selma_step_info(
        episode_id=1,
        episode_step=0,
        pre_field=pre,
        positions=[(4, 3), (1, 1)],
        levels=[1, 2],
        actions=[LOAD, NONE],
        post_field=post,
    )
    with pytest.raises(AttributeError):
        info.episode_step = 5
    with pytest.raises(AttributeError):
        info.failed_load_events[0].food_level = 99

    # Mutating the source arrays afterwards must not change the record.
    before = info.post_foods
    post[4, 4] = 0
    pre[4, 4] = 0
    assert info.post_foods == before
    assert info.pre_foods == ((( 4, 4), 3),)


# --------------------------------------------------------------------------
# Constraint 7: ambiguous adjacency is unreachable in supported environments
# --------------------------------------------------------------------------


@pytest.mark.parametrize("key", SUPPORTED_ENVS)
def test_no_ambiguous_adjacency_across_resets(key):
    for seed in range(50):
        env = make_env(key=key, seed=seed)
        env.reset(seed=seed)
        field = np.asarray(env._env.unwrapped.field)
        assert_no_ambiguous_adjacency(field)
        rows, cols = np.nonzero(field)
        assert 0 not in rows and 0 not in cols
        assert field.shape[0] - 1 not in rows and field.shape[1] - 1 not in cols


def test_ambiguity_guard_actually_fires():
    ambiguous = make_field((8, 8), {(3, 4): 2, (5, 4): 2})
    with pytest.raises(AssertionError, match="ambiguous"):
        assert_no_ambiguous_adjacency(ambiguous)


# --------------------------------------------------------------------------
# Wrapper wiring: accessor, staleness, episode identity
# --------------------------------------------------------------------------


def test_accessor_is_none_before_first_step_and_cleared_on_reset():
    env = make_env()
    assert env.get_selma_step_info() is None
    env.reset(seed=5)
    assert env.get_selma_step_info() is None
    env.step([NONE] * env.n_agents)
    first = env.get_selma_step_info()
    assert first is not None
    env.reset(seed=5)
    assert env.get_selma_step_info() is None, "stale record leaked across reset"


def test_episode_id_and_step_advance():
    env = make_env()
    env.reset(seed=5)
    steps = []
    for _ in range(3):
        env.step([NONE] * env.n_agents)
        info = env.get_selma_step_info()
        steps.append((info.episode_id, info.episode_step))
    # The constructor seeds via self._env.reset, not the wrapper's reset, so the
    # first wrapper-level reset yields episode_id 0.
    assert steps == [(0, 0), (0, 1), (0, 2)]

    env.reset(seed=5)
    env.step([NONE] * env.n_agents)
    assert env.get_selma_step_info().episode_step == 0
    assert env.get_selma_step_info().episode_id == 1


def test_selma_step_info_is_not_placed_in_env_info():
    env = make_env()
    env.reset(seed=5)
    _, _, _, _, info = env.step([NONE] * env.n_agents)
    assert "selma_step_info" not in info
    for key, value in info.items():
        if key == "per_agent_reward":
            continue
        assert isinstance(value, (int, float, np.integer, np.floating)), (
            f"env_info[{key!r}] is non-numeric ({type(value)}); EPyMARL sums "
            "env_info values into runner statistics"
        )


def test_invalid_load_agents_match_lbf_own_action_validity():
    """Cross-check invalid_load_agents against lbforaging's own LOAD validity."""
    rng = np.random.default_rng(0)
    env = make_env(key=SUPPORTED_ENVS[1], seed=3)
    env.reset(seed=3)
    u = env._env.unwrapped
    checked = 0
    for _ in range(50):
        from lbforaging.foraging.environment import Action

        load_valid = {
            i: Action.LOAD in u._valid_actions[p] for i, p in enumerate(u.players)
        }
        actions = [int(a) for a in rng.integers(0, 6, size=env.n_agents)]
        _, _, term, trunc, _ = env.step(actions)
        info = env.get_selma_step_info()
        expected = tuple(
            i
            for i, a in enumerate(actions)
            if a == LOAD and not load_valid[i]
        )
        assert info.invalid_load_agents == expected
        checked += 1
        if term or trunc:
            env.reset(seed=3)
            u = env._env.unwrapped
    assert checked == 50


# --------------------------------------------------------------------------
# A.4 (7) Existing aggregate counters remain numerically unchanged
# --------------------------------------------------------------------------


def _head_lbf_load_diagnostics():
    """Load the pre-Stage-A lbf_load_diagnostics straight out of git HEAD~."""
    import subprocess

    repo = os.path.join(os.path.dirname(__file__), "..")
    src = subprocess.run(
        ["git", "show", "8a169f1:src/envs/gymma.py"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    # Strip the gym/wrapper imports and class: only the pure function is needed.
    start = src.index("_LBF_LOAD_ACTION = 5")
    end = src.index("class GymmaWrapper")
    module = pytypes.ModuleType("selma_head_gymma")
    module.__dict__["np"] = np
    exec(compile(src[start:end], "<git:8a169f1 src/envs/gymma.py>", "exec"), module.__dict__)
    return module.lbf_load_diagnostics


def test_aggregate_counters_identical_to_pre_stage_a_implementation():
    """The Stage 0 (HEAD) counter and the current counter must agree exactly."""
    head_fn = _head_lbf_load_diagnostics()
    rng = np.random.default_rng(12345)
    compared = 0
    for key in SUPPORTED_ENVS:
        for seed in range(8):
            env = make_env(key=key, seed=seed)
            env.reset(seed=seed)
            u = env._env.unwrapped
            for _ in range(50):
                pre_field = np.array(u.field, copy=True)
                positions = [tuple(p.position) for p in u.players]
                levels = [p.level for p in u.players]
                actions = [int(a) for a in rng.integers(0, 6, size=env.n_agents)]
                _, _, term, trunc, info = env.step(actions)
                post_field = np.array(u.field, copy=True)

                expected = head_fn(pre_field, positions, levels, actions, post_field)
                actual = {k: info[k] for k in expected}
                assert actual == expected, (key, seed, actual, expected)
                compared += 1
                if term or trunc:
                    break
    assert compared > 200, f"only compared {compared} steps"


def test_structured_events_reproduce_aggregate_counters():
    """len(structured events) must equal the validated aggregate counters."""
    rng = np.random.default_rng(7)
    for key in SUPPORTED_ENVS:
        for seed in range(8):
            env = make_env(key=key, seed=seed)
            env.reset(seed=seed)
            for _ in range(50):
                actions = [int(a) for a in rng.integers(0, 6, size=env.n_agents)]
                _, _, term, trunc, info = env.step(actions)
                sinfo = env.get_selma_step_info()
                assert len(sinfo.failed_load_events) == info["failed_load_events"]
                assert (
                    sum(len(e.loader_ids) for e in sinfo.failed_load_events)
                    == info["failed_load_attempts"]
                )
                assert (
                    len(sinfo.successful_load_events) == info["successful_load_events"]
                )
                # A failed load in a supported env always has a real deficit.
                for e in sinfo.failed_load_events:
                    assert e.coalition_required is True
                    assert e.level_deficit >= 1
                if term or trunc:
                    break


# --------------------------------------------------------------------------
# A.4 (8) sum(raw_individual_rewards) == raw_team_reward
# --------------------------------------------------------------------------


def test_per_agent_rewards_sum_to_team_reward():
    rng = np.random.default_rng(99)
    for key in SUPPORTED_ENVS:
        env = make_env(key=key, seed=11)
        env.reset(seed=11)
        for _ in range(200):
            actions = [int(a) for a in rng.integers(0, 6, size=env.n_agents)]
            _, reward, term, trunc, info = env.step(actions)
            assert np.isclose(
                sum(info["per_agent_reward"]), reward, rtol=0.0, atol=1e-9
            )
            if term or trunc:
                env.reset(seed=11)


# --------------------------------------------------------------------------
# A.4 (9) Stage A changes nothing observable about the environment
# --------------------------------------------------------------------------


def test_stage_a_does_not_perturb_the_environment(monkeypatch):
    """A rollout with the structured builder disabled must be bit-identical."""

    def rollout(disable_builder):
        import envs.gymma as gymma_module

        if disable_builder:
            monkeypatch.setattr(
                gymma_module, "build_selma_step_info", lambda **kwargs: None
            )
        rng = np.random.default_rng(2024)
        env = make_env(key=SUPPORTED_ENVS[1], seed=42)
        obs, _ = env.reset(seed=42)
        trace = [np.array(obs, dtype=np.float64).tobytes()]
        for _ in range(120):
            actions = [int(a) for a in rng.integers(0, 6, size=env.n_agents)]
            obs, reward, term, trunc, info = env.step(actions)
            trace.append(
                (
                    np.array(obs, dtype=np.float64).tobytes(),
                    float(reward),
                    bool(term),
                    bool(trunc),
                    tuple(sorted((k, str(v)) for k, v in info.items())),
                    tuple(actions),
                )
            )
            if term or trunc:
                obs, _ = env.reset(seed=42)
        return trace

    with_selma = rollout(disable_builder=False)
    monkeypatch.undo()
    without_selma = rollout(disable_builder=True)
    assert with_selma == without_selma
