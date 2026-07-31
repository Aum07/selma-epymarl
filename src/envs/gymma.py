from collections.abc import Iterable
import warnings

import gymnasium as gym
from gymnasium.spaces import flatdim
from gymnasium.wrappers import TimeLimit
import numpy as np

from .multiagentenv import MultiAgentEnv
from .wrappers import FlattenObservation
import envs.pretrained as pretrained  # noqa
from selma.load_events import build_selma_step_info

try:
    from .pz_wrapper import PettingZooWrapper  # noqa
except ImportError:
    warnings.warn(
        "PettingZoo is not installed, so these environments will not be available! To install, run `pip install pettingzoo`"
    )

try:
    from .vmas_wrapper import VMASWrapper  # noqa
except ImportError:
    warnings.warn(
        "VMAS is not installed, so these environments will not be available! To install, run `pip install 'vmas[gymnasium]'`"
    )


# SELMA diagnostic: lbforaging (LBF) load-event instrumentation.
# LOAD action index, per lbforaging.foraging.environment.Action.LOAD.value.
# Fixed by lbforaging's Action enum (NONE=0, NORTH=1, SOUTH=2, WEST=3, EAST=4, LOAD=5).
_LBF_LOAD_ACTION = 5


def _lbf_is_orthogonally_adjacent(position, cell):
    row, col = position
    crow, ccol = cell
    return abs(row - crow) + abs(col - ccol) == 1


def lbf_load_diagnostics(pre_field, positions, levels, actions, post_field):
    """SELMA diagnostic: derive LBF per-step load-event stats from a pre/post-step snapshot.

    Pure function (no env access), so it can be exercised directly in tests.
    `pre_field`/`post_field` are 2D arrays of food levels (0 = empty cell).
    `positions`/`levels` are per-agent lists captured immediately before
    env.step; `actions` are the (raw, unvalidated) actions submitted to
    env.step. Read-only: does not consume randomness and does not affect
    observations, rewards, or termination.
    """
    pre_field = np.asarray(pre_field)
    post_field = np.asarray(post_field)
    rows, cols = pre_field.shape

    loaders = [i for i, a in enumerate(actions) if int(a) == _LBF_LOAD_ACTION]

    removed = []  # (row, col, pre_level) for food cells cleared this step
    persisted = []  # (row, col) for food cells that were present both before and after
    for r in range(rows):
        for c in range(cols):
            pre_level = pre_field[r, c]
            if pre_level <= 0:
                continue
            if post_field[r, c] == 0:
                removed.append((r, c, int(pre_level)))
            else:
                persisted.append((r, c))

    successful_load_events = 0
    solo_load_events = 0
    multi_agent_load_events = 0
    necessary_cooperative_load_events = 0
    for row, col, food_level in removed:
        participants = [
            i for i in loaders if _lbf_is_orthogonally_adjacent(positions[i], (row, col))
        ]
        successful_load_events += 1
        if len(participants) <= 1:
            solo_load_events += 1
        else:
            multi_agent_load_events += 1
            if all(food_level > levels[i] for i in participants):
                necessary_cooperative_load_events += 1

    failed_load_events = 0
    failed_load_attempts = 0
    for row, col in persisted:
        participants = [
            i for i in loaders if _lbf_is_orthogonally_adjacent(positions[i], (row, col))
        ]
        if participants:
            failed_load_events += 1
            failed_load_attempts += len(participants)

    if solo_load_events + multi_agent_load_events != successful_load_events:
        raise AssertionError(
            "SELMA LBF invariant violated: solo_load_events "
            f"({solo_load_events}) + multi_agent_load_events "
            f"({multi_agent_load_events}) != successful_load_events "
            f"({successful_load_events})"
        )
    if necessary_cooperative_load_events > multi_agent_load_events:
        raise AssertionError(
            "SELMA LBF invariant violated: necessary_cooperative_load_events "
            f"({necessary_cooperative_load_events}) > multi_agent_load_events "
            f"({multi_agent_load_events})"
        )

    return {
        "successful_load_events": successful_load_events,
        "solo_load_events": solo_load_events,
        "multi_agent_load_events": multi_agent_load_events,
        "necessary_cooperative_load_events": necessary_cooperative_load_events,
        "failed_load_events": failed_load_events,
        "failed_load_attempts": failed_load_attempts,
    }


class GymmaWrapper(MultiAgentEnv):
    def __init__(
        self,
        key,
        time_limit,
        pretrained_wrapper,
        seed,
        common_reward,
        reward_scalarisation,
        **kwargs,
    ):
        self._env = gym.make(f"{key}", **kwargs)
        self._env = TimeLimit(self._env, max_episode_steps=time_limit)
        self._env = FlattenObservation(self._env)

        if pretrained_wrapper:
            self._env = getattr(pretrained, pretrained_wrapper)(self._env)

        # SELMA diagnostic: gate LBF-specific load-event instrumentation to actual
        # lbforaging environments; every other Gymma environment is untouched.
        self._is_lbforaging = type(self._env.unwrapped).__module__.startswith(
            "lbforaging"
        )

        self.n_agents = self._env.unwrapped.n_agents
        self.episode_limit = time_limit
        self._obs = None
        self._info = None

        # SELMA Stage A: structured per-step record, replaced after every
        # env.step and cleared on reset so it can never leak across episodes.
        self._selma_step_info = None
        self._selma_episode_id = -1
        self._selma_step = 0

        self.longest_action_space = max(self._env.action_space, key=lambda x: x.n)
        self.longest_observation_space = max(
            self._env.observation_space, key=lambda x: x.shape
        )

        self._seed = seed
        try:
            self._env.unwrapped.seed(self._seed)
        except:
            self._env.reset(seed=self._seed)

        self.common_reward = common_reward
        if self.common_reward:
            if reward_scalarisation == "sum":
                self.reward_agg_fn = lambda rewards: sum(rewards)
            elif reward_scalarisation == "mean":
                self.reward_agg_fn = lambda rewards: sum(rewards) / len(rewards)
            else:
                raise ValueError(
                    f"Invalid reward_scalarisation: {reward_scalarisation} (only support 'sum' or 'mean')"
                )

    def _pad_observation(self, obs):
        return [
            np.pad(
                o,
                (0, self.longest_observation_space.shape[0] - len(o)),
                "constant",
                constant_values=0,
            )
            for o in obs
        ]

    def step(self, actions):
        """Returns obss, reward, terminated, truncated, info"""
        actions = [int(a) for a in actions]

        # SELMA diagnostic: snapshot LBF state immediately before env.step, so
        # load events can be attributed using pre-step positions/levels/actions.
        # Read-only; does not consume randomness or otherwise touch the env.
        # SELMA Stage A: drop the previous step's record up front, so a failure
        # part-way through this step cannot leave a stale record readable.
        self._selma_step_info = None

        _lbf_pre = None
        if self._is_lbforaging:
            _unwrapped = self._env.unwrapped
            _lbf_pre = {
                "field": np.array(_unwrapped.field, copy=True),
                "positions": [tuple(p.position) for p in _unwrapped.players],
                "levels": [p.level for p in _unwrapped.players],
            }

        obs, reward, done, truncated, self._info = self._env.step(actions)
        self._obs = self._pad_observation(obs)
        # SELMA diagnostic: capture per-agent reward vector before scalarisation.
        # Read-only; does not modify `reward` seen by the learner.
        if isinstance(self._info, dict) and isinstance(reward, Iterable):
            self._info["per_agent_reward"] = [float(r) for r in reward]

        if _lbf_pre is not None:
            # Copy the post-step field once; both the existing aggregate counters
            # and the structured record read from this value snapshot, so neither
            # retains a reference to the live environment array.
            _lbf_post_field = np.array(self._env.unwrapped.field, copy=True)

            # SELMA Stage A: structured, immutable per-step record. Exposed only
            # via get_selma_step_info(); deliberately NOT placed in env_info,
            # which EPyMARL sums into runner statistics as if every value were
            # numeric.
            self._selma_step_info = build_selma_step_info(
                episode_id=self._selma_episode_id,
                episode_step=self._selma_step,
                pre_field=_lbf_pre["field"],
                positions=_lbf_pre["positions"],
                levels=_lbf_pre["levels"],
                actions=actions,
                post_field=_lbf_post_field,
            )

            if isinstance(self._info, dict):
                self._info.update(
                    lbf_load_diagnostics(
                        _lbf_pre["field"],
                        _lbf_pre["positions"],
                        _lbf_pre["levels"],
                        actions,
                        _lbf_post_field,
                    )
                )

        self._selma_step += 1

        if self.common_reward and isinstance(reward, Iterable):
            reward = float(self.reward_agg_fn(reward))
        elif not self.common_reward and not isinstance(reward, Iterable):
            warnings.warn(
                "common_reward is False but received scalar reward from the environment, returning reward as is"
            )

        if isinstance(done, Iterable):
            done = all(done)
        return self._obs, reward, done, truncated, self._info

    def get_selma_step_info(self):
        """SELMA Stage A: immutable record for the most recent env.step.

        Returns None before the first step of an episode, after reset, and for
        every non-lbforaging environment.
        """
        return self._selma_step_info

    def get_selma_grid_shape(self):
        """SELMA Stage B: (rows, cols) of the lbforaging field, else None."""
        if not self._is_lbforaging:
            return None
        return tuple(self._env.unwrapped.field.shape)

    def get_obs(self):
        """Returns all agent observations in a list"""
        return self._obs

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id"""
        raise self._obs[agent_id]

    def get_obs_size(self):
        """Returns the shape of the observation"""
        return flatdim(self.longest_observation_space)

    def get_state(self):
        return np.concatenate(self._obs, axis=0).astype(np.float32)

    def get_state_size(self):
        """Returns the shape of the state"""
        if hasattr(self._env.unwrapped, "state_size"):
            return self._env.unwrapped.state_size
        return self.n_agents * flatdim(self.longest_observation_space)

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id"""
        valid = flatdim(self._env.action_space[agent_id]) * [1]
        invalid = [0] * (self.longest_action_space.n - len(valid))
        return valid + invalid

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take"""
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return flatdim(self.longest_action_space)

    def reset(self, seed=None, options=None):
        """Returns initial observations and info"""
        obs, info = self._env.reset(seed=seed, options=options)
        self._obs = self._pad_observation(obs)
        # SELMA Stage A: no step has happened yet in the new episode, and the
        # previous episode's record must not remain readable.
        self._selma_step_info = None
        self._selma_episode_id += 1
        self._selma_step = 0
        # SELMA diagnostic: record the episode's initial food-cell count, used to
        # bound successful_load_events per episode. Read-only, no extra RNG use.
        if self._is_lbforaging and isinstance(info, dict):
            info["initial_food_count"] = int(
                np.count_nonzero(self._env.unwrapped.field)
            )
        return self._obs, info

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

    def seed(self, seed=None):
        return self._env.unwrapped.seed(seed)

    def save_replay(self):
        pass

    def get_stats(self):
        return {}
