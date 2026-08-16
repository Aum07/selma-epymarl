from functools import partial

import numpy as np

from components.episode_buffer import EpisodeBatch
from envs import REGISTRY as env_REGISTRY
from envs import register_smac, register_smacv2
from selma.guidance import PHI_BOUNDARY_TOLERANCE

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

        # SELMA mode contract (src/selma/modes.py). Validated before anything is
        # constructed: off / shadow / active are the only legal modes and every
        # illegal provider, beta, variant or common_reward combination raises
        # here rather than degrading silently to an unshaped baseline.
        from selma.modes import validate_selma_config

        self.selma_config = validate_selma_config(self.args)

        # SELMA Stage B: detector bank. Created for "shadow" (all 48 variants,
        # calibration) and for "active" (the frozen variant only). With the
        # default "off" nothing below ever runs, so plain QMIX behaviour is
        # bit-for-bit unchanged.
        self.selma_bank = None
        self.selma_shadow_writer = None
        if self.selma_config.detector_enabled:
            from selma.detector import ShadowDetectorBank, build_variant_grid
            from selma.shadow_log import ShadowCsvWriter

            # SELMA: optional single-variant restriction. Empty (shadow default) keeps
            # the full 48-variant grid with its original ordering and contents. A
            # non-empty id selects that variant *from* build_variant_grid(), so its
            # parameters are never reconstructed here; an unknown or ambiguous id fails
            # loudly before training rather than silently falling back to all 48.
            # Active mode always names the frozen variant (enforced by the mode
            # contract), so its bank holds exactly one variant.
            selma_variant_id = self.selma_config.variant_id
            if selma_variant_id:
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
                selma_variants = tuple(matches)
            else:
                selma_variants = None

            grid_shape = (
                self.env.get_selma_grid_shape()
                if hasattr(self.env, "get_selma_grid_shape")
                else None
            )
            if grid_shape is not None:
                self.selma_bank = ShadowDetectorBank(
                    variants=selma_variants,
                    # args.n_agents is not populated until run_sequential has
                    # called get_env_info(), so take it from the env directly.
                    n_agents=self.env.get_env_info()["n_agents"],
                    grid_shape=grid_shape,
                    condition=str(self.args.env_args.get("key", self.args.env)),
                    seed=self.args.seed,
                    rearm_steps=getattr(self.args, "selma_rearm_steps", 3),
                    cooldown_steps=getattr(self.args, "selma_cooldown_steps", 5),
                    max_events_per_episode=getattr(
                        self.args, "selma_max_events_per_episode", 2
                    ),
                    w_inbound=getattr(self.args, "selma_w_inbound", 3),
                )
                if getattr(self.args, "selma_shadow_csv", ""):
                    self.selma_shadow_writer = ShadowCsvWriter(
                        self.args.selma_shadow_csv
                    )

        # SELMA Stage C: guidance + PBRS, active mode only. beta defaults to 0.0,
        # which is bit-identical to unshaped QMIX (contract 5A).
        self.selma_guidance = None
        self.selma_guidance_writer = None
        self.selma_accounting = None
        self.selma_cache = None
        self.selma_gateway = None
        if self.selma_config.guidance_enabled:
            from selma.guidance import GuidanceController, GuidanceCsvWriter
            from selma.provider_registry import is_live_provider, make_provider

            if self.selma_bank is None:
                raise ValueError(
                    "selma_mode='active' requires a SELMA detector bank; none was "
                    "constructed (is this an LBF environment?)."
                )
            for accessor in ("get_selma_agent_positions", "get_selma_food_positions"):
                if not hasattr(self.env, accessor):
                    raise ValueError(
                        "selma_mode='active' requires env.{}() for the successor "
                        "potential; this environment does not provide it.".format(accessor)
                    )
            # SELMA Stage D: cache, live-call budget and accounting. The
            # deterministic mock is unaffected -- with the default
            # selma_cache_mode='off' the gateway performs no lookup and the mock
            # makes no network call, so Stage C behaviour is unchanged.
            provider_kwargs = {}
            if is_live_provider(self.selma_config.provider):
                provider_kwargs = {
                    "timeout": self.selma_config.provider_timeout_s,
                    "max_retries": self.selma_config.provider_max_retries,
                    "backoff_s": self.selma_config.provider_backoff_s,
                    "max_output_tokens": self.selma_config.provider_max_output_tokens,
                }
            provider = make_provider(self.selma_config.provider, **provider_kwargs)
            condition = str(self.args.env_args.get("key", self.args.env))
            self.selma_accounting, self.selma_cache, self.selma_gateway = (
                self._build_selma_stage_d(provider, condition)
            )

            self.selma_guidance = GuidanceController(
                provider=self.selma_gateway or provider,
                grid_shape=self.selma_bank.grid_shape,
                beta=self.selma_config.beta,
                gamma=float(self.args.gamma),
                enabled=True,
                variant_id=self.selma_config.variant_id,
                accounting=self.selma_accounting,
            )
            if self.selma_config.guidance_log_csv:
                self.selma_guidance_writer = GuidanceCsvWriter(
                    self.selma_config.guidance_log_csv
                )

    def _build_selma_stage_d(self, provider, condition):
        """Accounting, cache and gateway for one run. Returns the three objects.

        The cache database is identified by condition + seed + provider + model,
        so reopening another run's writable store is refused unless it is
        allowed deliberately.
        """
        from selma.accounting import SelmaAccounting
        from selma.gateway import CACHE_OFF, ProviderGateway
        from selma.provider_registry import is_live_provider

        config = self.selma_config
        accounting = SelmaAccounting(
            cache_mode=config.cache_mode,
            live_call_cap=config.live_call_cap,
            provider=getattr(provider, "provider_id", "unknown"),
            requested_model=getattr(provider, "model", ""),
            prompt_version=getattr(provider, "prompt_version", ""),
            response_schema_version=getattr(provider, "response_schema_version", ""),
            price_table_date=config.price_table_date,
            live_provider=is_live_provider(config.provider),
        )

        cache = None
        if config.cache_mode != CACHE_OFF:
            from selma.cache import SqliteGuidanceCache

            cache = SqliteGuidanceCache(
                config.cache_path,
                run_identity="{}|seed{}|{}|{}".format(
                    condition, self.args.seed,
                    getattr(provider, "provider_id", "unknown"),
                    getattr(provider, "model", ""),
                ),
                allow_shared_write=config.cache_allow_shared_write,
                read_only=config.cache_mode == "read_only",
            )

        gateway = ProviderGateway(
            provider,
            accounting=accounting,
            cache=cache,
            cache_mode=config.cache_mode,
            live_call_cap=config.live_call_cap,
            condition=condition,
            grid_shape=self.selma_bank.grid_shape,
            # A provider that cannot perform network I/O is always allowed; a
            # live one is allowed only while the cap has room.
            allow_live_calls=True,
        )
        if not is_live_provider(config.provider):
            # Offline providers never consume the live-call budget: nothing they
            # do is an HTTP request.
            gateway.live_call_cap = None
            accounting.live_call_cap = None
        return accounting, cache, gateway

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
        # SELMA Stage B: persist the per-variant calibration summary before the
        # process exits. Counters such as eligible_steps exist only in memory --
        # they are not derivable from the shadow CSV, which has rows only where
        # candidates occurred -- so the report generator reads this sidecar.
        self.write_selma_summary()
        self.env.close()

    def write_selma_summary(self):
        """Dump the shadow bank's calibration summary, if one was configured."""
        path = getattr(self.args, "selma_shadow_summary_json", "")
        if self.selma_bank is None or not path:
            return None
        import json

        summary = self.selma_bank.calibration_summary()
        with open(path, "w") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        return path

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

        # SELMA Stage B: training episodes only. Greedy evaluation must never
        # update histories, latches, cooldowns, budgets, pending labels or rows.
        selma_bank = self.selma_bank if not test_mode else None
        if selma_bank is not None:
            selma_bank.reset_episode(None)
        # SELMA Stage C: guidance is training-only. In greedy evaluation the
        # controller is not consulted at all: zero queries, zero activation,
        # zero shaping, and reward stays the raw environment reward.
        selma_guidance = self.selma_guidance if not test_mode else None
        if selma_guidance is not None:
            selma_guidance.reset_episode()
        selma_guidance_writer = self.selma_guidance_writer if not test_mode else None
        # SELMA Stage C: phi_before(t) is phi_after_effective(t-1), carried
        # forward rather than recomputed, so the boundary identity holds by
        # construction. It starts at 0.0 because reset_episode cleared guidance.
        selma_phi_before = 0.0
        selma_shaping_total = 0.0
        selma_last_stored_terminated = False

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

            _, reward, env_terminated, env_truncated, env_info = self.env.step(actions[0])
            terminated = env_terminated or env_truncated
            # SELMA Stage C: per-transition detector output, reset every step so
            # a stale event list can never be reused on a later transition.
            selma_events = ()
            selma_episode_id = -1
            # SELMA: pop per-agent reward (also keeps env_info numeric for stats agg)
            _par = env_info.pop("per_agent_reward", None)
            # SELMA: pop LBF load-event fields (no-op / all-zero for non-LBF envs)
            _step_loads = {k: env_info.pop(k, 0) for k in LBF_LOAD_EVENT_KEYS}
            for _k in LBF_LOAD_EVENT_KEYS:
                load_event_totals[_k] += _step_loads[_k]
            # SELMA Stage A: structured per-step record, available online here.
            # Read-only: no detector, provider, potential, or shaping exists yet,
            # so nothing downstream of this point is affected.
            selma_step_info = (
                self.env.get_selma_step_info()
                if hasattr(self.env, "get_selma_step_info")
                else None
            )
            if selma_step_info is not None:
                # SELMA HARD invariant: the structured events must reproduce the
                # existing validated aggregate counters exactly, and the record
                # must be aligned with the transition index it is stored against.
                if selma_step_info.episode_step != self.t:
                    raise AssertionError(
                        "SELMA invariant violated: step info episode_step="
                        f"{selma_step_info.episode_step} != runner t={self.t}"
                    )
                if len(selma_step_info.failed_load_events) != _step_loads[
                    "failed_load_events"
                ]:
                    raise AssertionError(
                        "SELMA invariant violated: structured failed_load_events "
                        f"({len(selma_step_info.failed_load_events)}) != aggregate "
                        f"counter ({_step_loads['failed_load_events']}) at step {self.t}"
                    )
                if len(selma_step_info.successful_load_events) != _step_loads[
                    "successful_load_events"
                ]:
                    raise AssertionError(
                        "SELMA invariant violated: structured successful_load_events "
                        f"({len(selma_step_info.successful_load_events)}) != aggregate "
                        f"counter ({_step_loads['successful_load_events']}) at step {self.t}"
                    )
                # SELMA Stage B: shadow observation only. Reads the immutable
                # Stage A record and the raw team reward; writes nothing back.
                if selma_bank is not None:
                    if selma_guidance is not None:
                        # SELMA HARD invariant (Stage C): the carried phi_before
                        # must equal Phi(s_t, g_t) recomputed from the Stage A
                        # pre-step record. This is the boundary identity
                        # phi_after_effective(t-1) == phi_before(t) -- the
                        # property that makes the shaping telescope -- and it
                        # also catches any divergence between the Stage A record
                        # and the GymMA post-step accessors.
                        pre_positions = {
                            int(a.agent_id): tuple(int(x) for x in a.position)
                            for a in selma_step_info.agents
                        }
                        pre_foods = tuple(f.position for f in selma_step_info.pre_foods)
                        recomputed = selma_guidance.potential(pre_positions, pre_foods)
                        if abs(recomputed - selma_phi_before) > PHI_BOUNDARY_TOLERANCE:
                            raise AssertionError(
                                "SELMA Stage C invariant violated: carried "
                                f"phi_before={selma_phi_before!r} but Phi(s_t, g_t) "
                                f"recomputes to {recomputed!r} at step {self.t}"
                            )
                        # Rows reach bank.rows only at finish_episode() (resolution
                        # labels resolve there); events are visible at emission time
                        # on bank.pending_rows, which is what guidance needs.
                        selma_n_rows_before = len(selma_bank.pending_rows)

                    selma_bank.update(
                        step_info=selma_step_info,
                        raw_team_reward=float(np.asarray(reward).sum()),
                        t_env=self.t_env + self.t,
                    )

                    if selma_guidance is not None:
                        # Events emitted by the detector on this transition are
                        # candidates for g_{t+1}. The controller does the frozen
                        # variant, feasibility and suppression filtering.
                        selma_events = [
                            entry[0]
                            for entry in selma_bank.pending_rows[selma_n_rows_before:]
                        ]
                        selma_episode_id = selma_step_info.episode_id
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

            stored_terminated = terminated != env_info.get("episode_limit", False)
            post_transition_data = {
                "actions": actions,
                "terminated": [(stored_terminated,)],
            }

            # SELMA Stage C: potential-based shaping. The raw reward is never
            # mutated; the guidance boundary is advanced and the transition
            # priced by the single shared path in GuidanceController, which the
            # ordering tests drive directly.
            if selma_guidance is not None:
                post_positions = self.env.get_selma_agent_positions()
                post_foods = self.env.get_selma_food_positions()
                if post_positions is None or post_foods is None:
                    raise AssertionError(
                        "SELMA Stage C invariant violated: the environment returned "
                        f"no post-step state at step {self.t}"
                    )
                # Terminal cause, for the log only -- it never affects shaping,
                # which follows the stored bootstrap mask alone. This LBF
                # wrapper reports a horizon ending as `terminated`, not
                # `truncated` (docs/selma_stage0_terminal_mask.md), so the env
                # flags alone cannot tell the two endings apart. Clearing the
                # last food is the true terminal; reaching the horizon with food
                # still on the grid is a time limit.
                if not terminated:
                    terminal_cause = ""
                elif not post_foods:
                    terminal_cause = "true_terminal"
                elif env_truncated or (self.t + 1) >= self.episode_limit:
                    terminal_cause = "time_limit"
                else:
                    terminal_cause = "true_terminal"
                selma_record = selma_guidance.apply_transition(
                    step=self.t,
                    raw_team_reward=reward,
                    phi_before=selma_phi_before,
                    post_agent_positions=post_positions,
                    post_food_positions=post_foods,
                    event_rows=selma_events,
                    episode_ended=bool(terminated),
                    stored_terminated=bool(stored_terminated),
                    terminal_cause=terminal_cause,
                )
                reward = selma_record.training_reward
                # The successor potential of this transition IS the predecessor
                # potential of the next one. Carrying it forward is what makes
                # the shaping telescope exactly.
                selma_phi_before = selma_record.phi_after_effective
                selma_shaping_total += selma_record.shaping_contribution
                selma_last_stored_terminated = bool(stored_terminated)
                if selma_guidance_writer is not None:
                    selma_guidance_writer.write_row(
                        record=selma_record,
                        controller=selma_guidance,
                        t_env=self.t_env + self.t,
                        condition=str(
                            self.args.env_args.get("key", self.args.env)
                        ),
                        seed=self.args.seed,
                        episode_id=selma_episode_id,
                        condition_checked=1,
                    )

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

        # SELMA Stage B: resolve this episode's pending natural-resolution
        # labels and append rows to the separate shadow CSV. This is the only
        # additional output shadow mode is permitted to produce.
        if selma_bank is not None:
            selma_bank.finish_episode()
            # SELMA HARD invariant: shadow mode must never query a provider,
            # activate a subgoal, or produce shaping reward.
            if (
                selma_bank.query_count != 0
                or selma_bank.provider_call_count != 0
                or selma_bank.active_subgoal_count != 0
                or selma_bank.total_shaping_reward != 0.0
            ):
                raise AssertionError(
                    "SELMA shadow invariant violated: queries="
                    f"{selma_bank.query_count} providers={selma_bank.provider_call_count} "
                    f"subgoals={selma_bank.active_subgoal_count} "
                    f"shaping={selma_bank.total_shaping_reward}"
                )
            # SELMA HARD invariant (Stage C): guidance must never survive an
            # episode boundary, and shaping must be exactly zero whenever the
            # mode is not active or beta is zero.
            if selma_guidance is not None:
                # A stored terminal removes guidance at its own boundary; a
                # bootstrapped time limit (which this LBF configuration never
                # produces -- see docs/selma_stage0_terminal_mask.md) keeps it
                # for the successor potential and reset_episode clears it.
                if (
                    selma_last_stored_terminated
                    and selma_guidance.active is not None
                ):
                    raise AssertionError(
                        "SELMA Stage C invariant violated: guidance still active at "
                        f"the episode boundary (event {selma_guidance.active.event_id})"
                    )
                if selma_guidance.beta == 0.0 and selma_shaping_total != 0.0:
                    raise AssertionError(
                        "SELMA Stage C invariant violated: beta=0 produced shaping "
                        f"{selma_shaping_total}"
                    )
                # SELMA HARD invariant (Stage D): the eleven accounting counters
                # must reconcile after every episode, not only at run end.
                if self.selma_accounting is not None:
                    self.selma_accounting.assert_invariants(
                        max_retries=self.selma_config.provider_max_retries
                    )
            if self.selma_shadow_writer is not None:
                self.selma_shadow_writer.write(selma_bank.drain_rows())
            else:
                selma_bank.drain_rows()

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
