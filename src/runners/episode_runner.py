from functools import partial

import numpy as np

from components.episode_buffer import EpisodeBatch
from envs import REGISTRY as env_REGISTRY
from envs import register_smac, register_smacv2

# SELMA: LBF load-event diagnostic fields accumulated per episode (see envs/gymma.py).
# Zero for non-LBF environments; the underlying computation is gated to lbforaging there.
LBF_LOAD_EVENT_KEYS = (
    "successful_load_events",
    "solo_load_events",
    "multi_agent_load_events",
    "necessary_cooperative_load_events",
    "failed_load_events",
    "failed_load_attempts",
)


class EpisodeRunner:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        # registering both smac and smacv2 causes a pysc2 error
        # --> dynamically register the needed env
        if self.args.env == "sc2":
            register_smac()
        elif self.args.env == "sc2v2":
            register_smacv2()

        self.env = env_REGISTRY[self.args.env](
            **self.args.env_args,
            common_reward=self.args.common_reward,
            reward_scalarisation=self.args.reward_scalarisation,
        )
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(
            EpisodeBatch,
            scheme,
            groups,
            self.batch_size,
            self.episode_limit + 1,
            preprocess=preprocess,
            device=self.args.device,
        )
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        _, reset_info = self.env.reset()
        self.t = 0
        # SELMA: LBF-only episode bound for successful_load_events (None elsewhere)
        self._lbf_initial_food_count = (
            reset_info.get("initial_food_count")
            if isinstance(reset_info, dict)
            else None
        )

    def run(self, test_mode=False):
        self.reset()

        terminated = False
        if self.args.common_reward:
            episode_return = 0
        else:
            episode_return = np.zeros(self.args.n_agents)
        self.mac.init_hidden(batch_size=self.batch_size)
        # SELMA: per-agent episode return accumulator
        per_agent_ep = np.zeros(self.args.n_agents)
        per_agent_steps = 0
        multi_agent_reward_steps = 0   # SELMA: steps where >=2 agents were rewarded together
        single_agent_reward_steps = 0
        # SELMA: per-episode LBF load-event accumulators (see envs/gymma.py)
        load_event_totals = {k: 0 for k in LBF_LOAD_EVENT_KEYS}

        while not terminated:
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()],
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(
                self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode
            )

            _, reward, terminated, truncated, env_info = self.env.step(actions[0])
            terminated = terminated or truncated
            # SELMA: pop per-agent reward (also keeps env_info numeric for stats agg)
            _par = env_info.pop("per_agent_reward", None)
            # SELMA: pop LBF load-event fields (no-op / all-zero for non-LBF envs)
            _step_loads = {k: env_info.pop(k, 0) for k in LBF_LOAD_EVENT_KEYS}
            for _k in LBF_LOAD_EVENT_KEYS:
                load_event_totals[_k] += _step_loads[_k]
            if _par is not None:
                _pa = np.asarray(_par, dtype=float)
                per_agent_ep += _pa
                per_agent_steps += 1
                _nrew = int((_pa > 0).sum())
                if _nrew >= 2:
                    multi_agent_reward_steps += 1
                elif _nrew == 1:
                    single_agent_reward_steps += 1
                # SELMA HARD invariant: a multi-agent load this step must show up as
                # a multi-agent reward step (both derived from the same env.step call)
                if _step_loads["multi_agent_load_events"] >= 1 and _nrew < 2:
                    raise AssertionError(
                        "SELMA LBF invariant violated: multi_agent_load_events="
                        f"{_step_loads['multi_agent_load_events']} at step {self.t} "
                        f"without a matching multi_agent_reward_step (nrew={_nrew})"
                    )
            if test_mode and self.args.render:
                self.env.render()
            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }
            if self.args.common_reward:
                post_transition_data["reward"] = [(reward,)]
            else:
                post_transition_data["reward"] = [tuple(reward)]

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        # SELMA HARD invariant: an episode can't successfully load more food than
        # it started with (each food cell can only be removed once).
        if (
            self._lbf_initial_food_count is not None
            and load_event_totals["successful_load_events"] > self._lbf_initial_food_count
        ):
            raise AssertionError(
                "SELMA LBF invariant violated: successful_load_events "
                f"({load_event_totals['successful_load_events']}) exceeds initial "
                f"food count ({self._lbf_initial_food_count}) for this episode"
            )

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()],
        }
        if test_mode and self.args.render:
            print(f"Episode return: {episode_return}")
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(
            self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode
        )
        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update(
            {
                k: cur_stats.get(k, 0) + env_info.get(k, 0)
                for k in set(cur_stats) | set(env_info)
            }
        )
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)
        cur_stats["multi_agent_reward_steps"] = cur_stats.get("multi_agent_reward_steps", 0) + multi_agent_reward_steps
        cur_stats["single_agent_reward_steps"] = cur_stats.get("single_agent_reward_steps", 0) + single_agent_reward_steps
        for _k in LBF_LOAD_EVENT_KEYS:
            cur_stats[_k] = cur_stats.get(_k, 0) + load_event_totals[_k]
        for _i in range(self.args.n_agents):
            _k = f"agent{_i}_return"
            cur_stats[_k] = cur_stats.get(_k, 0) + float(per_agent_ep[_i])
        if getattr(self.args, 'selma_log_csv', ''):
            import os, csv as _csv
            _f = self.args.selma_log_csv
            write_header = not os.path.exists(_f) or os.path.getsize(_f) == 0
            with open(_f, 'a', newline='') as _fh:
                _w = _csv.writer(_fh)
                if write_header:
                    _w.writerow(['t_env','test_mode','ep_length','team_return',
                                 'multi_agent_reward_steps','single_agent_reward_steps'] +
                                [f'agent{j}_return' for j in range(self.args.n_agents)] +
                                list(LBF_LOAD_EVENT_KEYS))
                if per_agent_steps != self.t:
                    raise RuntimeError(
                        f"SELMA: captured {per_agent_steps} vectors for {self.t} steps")
                _team = float(np.asarray(episode_return).sum())
                if not np.isclose(_team, per_agent_ep.sum(), rtol=0.0, atol=1e-8):
                    raise AssertionError(
                        f"SELMA reward mismatch: team={_team} agents={per_agent_ep.sum()}")
                _logged_t = self.t_env if test_mode else self.t_env + self.t
                _w.writerow([_logged_t, int(test_mode), self.t, _team,
                             multi_agent_reward_steps, single_agent_reward_steps] +
                            [float(x) for x in per_agent_ep] +
                            [load_event_totals[k] for k in LBF_LOAD_EVENT_KEYS])

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat(
                    "epsilon", self.mac.action_selector.epsilon, self.t_env
                )
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        if self.args.common_reward:
            self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
            self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        else:
            for i in range(self.args.n_agents):
                self.logger.log_stat(
                    prefix + f"agent_{i}_return_mean",
                    np.array(returns)[:, i].mean(),
                    self.t_env,
                )
                self.logger.log_stat(
                    prefix + f"agent_{i}_return_std",
                    np.array(returns)[:, i].std(),
                    self.t_env,
                )
            total_returns = np.array(returns).sum(axis=-1)
            self.logger.log_stat(
                prefix + "total_return_mean", total_returns.mean(), self.t_env
            )
            self.logger.log_stat(
                prefix + "total_return_std", total_returns.std(), self.t_env
            )
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(
                    prefix + k + "_mean", v / stats["n_episodes"], self.t_env
                )
        stats.clear()
