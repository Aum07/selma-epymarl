"""SELMA Stage B: coordination-stall detector, shadow mode only.

Evaluates 48 detector variants side by side from a single environment
trajectory. Nothing here calls a provider, activates a subgoal, computes a
potential, or touches reward: it only observes the Stage A step records and
appends rows to a separately configured shadow CSV.

Variant grid (3 x 2 x 4 x 2 = 48)::

    W_fail  in {5, 10, 15}
    m       in {2, 3}
    team    in {none, 5, 10, 15}
    inbound in {off, on(W_inbound=3)}

Shared vs per-variant state
---------------------------
Raw failure histories, team-progress history, per-agent BFS distance histories
and food-removal steps are *variant-independent facts about the trajectory*, so
they are stored once per episode and queried with each variant's window. Every
piece of variant-dependent state -- latches, cooldowns, rearm streaks, event
budgets, event counts, pending resolution labels -- is strictly per variant, so
one variant can never suppress, latch, or consume the budget of another.

Food identity
-------------
The canonical identity of a food is the episode-specific Stage A ``food_id``::

    food_id = (episode_id, row, col)

Every raw history, every lifecycle entry, every removal record and the shadow
CSV key on that triple. The bare ``(row, col)`` position is *derived* from it
and is logged in its own ``food_position`` column for readability, but it is
never the key: a position alone is only unique within one episode, so keying on
it would make correctness depend on ``reset_episode`` always being called at
exactly the right moment. ``update`` additionally refuses to mix two episode ids
into one episode's state, so a missed reset fails loudly instead of silently
merging two episodes' histories under a shared coordinate.

State snapshot convention
-------------------------
Distances and coalitions are computed from the *pre-step* agent positions in the
Stage A record together with the *post-step* food set. Pre-step positions are
the state in which the transition's LOAD decision was made; the post-step food
set is exactly the set the "food still exists" condition refers to. The offset
is uniform across every transition, so distance-delta windows remain consistent.

Lifecycle
---------
``ARMED -> CANDIDATE -> FIRED/LATCHED -> RESOLVED -> ARMED`` per (variant, food).
A latched food resolves when the food disappears, when its coalition becomes
infeasible, or when the eligibility condition has been false for ``rearm_steps``
consecutive transitions. Resolution starts a ``cooldown_steps`` cooldown; after
that the food may only fire again on a genuine false-to-true rising edge, so a
continuously true condition never produces a repeat event.
"""

from itertools import product

from .coalition import bfs_distances, min_distance_to_loading_cell, select_helper_coalition

W_FAIL_GRID = (5, 10, 15)
M_GRID = (2, 3)
TEAM_GATE_GRID = (None, 5, 10, 15)
INBOUND_GRID = (False, True)


class DetectorVariant:
    """One point of the shadow grid. Immutable configuration."""

    __slots__ = ("variant_id", "w_fail", "m", "w_team", "use_inbound")

    def __init__(self, variant_id, w_fail, m, w_team, use_inbound):
        self.variant_id = variant_id
        self.w_fail = w_fail
        self.m = m
        self.w_team = w_team
        self.use_inbound = use_inbound

    @property
    def team_gate_variant(self):
        return "none" if self.w_team is None else f"W_team={self.w_team}"

    @property
    def inbound_variant(self):
        return "on" if self.use_inbound else "off"


def build_variant_grid():
    """The 48 shadow variants, in a stable deterministic order."""
    variants = []
    for w_fail, m, w_team, inbound in product(
        W_FAIL_GRID, M_GRID, TEAM_GATE_GRID, INBOUND_GRID
    ):
        team = "none" if w_team is None else f"t{w_team}"
        variants.append(
            DetectorVariant(
                variant_id=f"wf{w_fail}_m{m}_{team}_ib{'on' if inbound else 'off'}",
                w_fail=w_fail,
                m=m,
                w_team=w_team,
                use_inbound=inbound,
            )
        )
    return tuple(variants)


class _FoodLifecycle:
    """Per-(variant, food) latch/cooldown/rearm state for one episode."""

    __slots__ = ("latched", "cooldown_remaining", "false_streak", "fired_count")

    def __init__(self, rearm_steps):
        self.latched = False
        self.cooldown_remaining = 0
        # Start already rearmed so the first genuine rising edge can fire.
        self.false_streak = rearm_steps
        self.fired_count = 0


class _VariantState:
    """Per-variant, per-episode mutable state. Never shared between variants."""

    __slots__ = ("foods", "events_used", "counters")

    def __init__(self):
        self.foods = {}
        self.events_used = 0
        self.counters = {}

    def bump(self, key, amount=1):
        self.counters[key] = self.counters.get(key, 0) + amount


class ShadowDetectorBank:
    """Runs every detector variant over one shared trajectory, in shadow mode.

    The bank is created only when ``selma_mode == "shadow"``. It is updated
    exclusively from training transitions; greedy evaluation never reaches it.
    """

    def __init__(
        self,
        n_agents,
        grid_shape,
        condition,
        seed,
        rearm_steps=3,
        cooldown_steps=5,
        max_events_per_episode=2,
        w_inbound=3,
        variants=None,
    ):
        self.n_agents = n_agents
        self.grid_shape = tuple(grid_shape)
        self.condition = condition
        self.seed = seed
        self.rearm_steps = int(rearm_steps)
        self.cooldown_steps = int(cooldown_steps)
        self.max_events_per_episode = int(max_events_per_episode)
        self.w_inbound = int(w_inbound)
        self.variants = build_variant_grid() if variants is None else tuple(variants)

        self.query_count = 0
        self.provider_call_count = 0
        self.active_subgoal_count = 0
        self.total_shaping_reward = 0.0

        self.rows = []
        self.episode_id = None
        # Cumulative across episodes; deliberately NOT cleared by reset_episode.
        self.cumulative_counters = {v.variant_id: {} for v in self.variants}
        self.events_per_episode_hist = {v.variant_id: {} for v in self.variants}
        self.episodes_seen = 0
        self._reset_episode_state()

    # -- episode boundaries ------------------------------------------------

    def _reset_episode_state(self):
        self.fail_history = {}
        self.progress_history = []
        self.distance_history = {}
        self.removal_step = {}
        self.last_step = -1
        self.variant_state = {v.variant_id: _VariantState() for v in self.variants}
        self.pending_rows = []
        self._event_counter = 0

    def reset_episode(self, episode_id):
        """Drop every per-episode history, latch, budget and pending label."""
        self.episode_id = episode_id
        self._reset_episode_state()

    # -- helpers -----------------------------------------------------------

    def food_id(self, position):
        """Canonical episode-specific food identity for a grid position."""
        return (self.episode_id, int(position[0]), int(position[1]))

    @staticmethod
    def food_position(food_id):
        """The grid position derived from a canonical food id."""
        return (food_id[1], food_id[2])

    def _window_fail_count(self, agent_id, food_id, step, w_fail):
        """Distinct transitions with a coalition-required failed LOAD in the
        inclusive window ``[step - w_fail + 1, step]``, for one
        ``(agent_id, food_id)`` raw history."""
        history = self.fail_history.get((agent_id, food_id))
        if not history:
            return 0
        low = step - w_fail + 1
        return sum(1 for t in history if low <= t <= step)

    def _team_window_complete(self, w_team):
        """True when this episode has produced a full ``w_team`` window.

        ``w_team is None`` is the ungated variant: it has no window, so every
        observed transition is immediately eligible.
        """
        if w_team is None:
            return True
        return len(self.progress_history) >= w_team

    def _team_gate_open(self, step, w_team):
        """True when no environmental progress occurred in a *complete* window.

        The window includes the current transition. Until ``w_team`` real
        transitions have been observed in this episode the gate cannot open: an
        incomplete episode-start window is not a stall.
        """
        if w_team is None:
            return True
        if not self._team_window_complete(w_team):
            return False
        return not any(self.progress_history[-w_team:])

    def _inbound_status(self, coalition, food_id, step):
        """Whether the chosen helper subset is already inbound.

        Inbound requires a *complete* ``w_inbound`` window of distance history
        for every helper and a strict reduction of at least one. A helper
        already adjacent (distance 0) cannot reduce further, so a helper that
        sits next to the food without LOADing is never counted as inbound --
        that is precisely the coordination failure guidance would address.
        """
        if not coalition.feasible or not coalition.helper_ids:
            return False, (), (), (), ()
        starts, ends, deltas = [], [], []
        inbound = True
        for agent_id in coalition.helper_ids:
            history = self.distance_history.get((agent_id, food_id), [])
            window = [d for t, d in history if step - self.w_inbound <= t <= step]
            if len(window) < self.w_inbound + 1 or window[0] is None or window[-1] is None:
                inbound = False
                starts.append(window[0] if window else None)
                ends.append(window[-1] if window else None)
                deltas.append(None)
                continue
            start, end = window[0], window[-1]
            starts.append(start)
            ends.append(end)
            deltas.append(end - start)
            if end > start - 1:
                inbound = False
        return (
            inbound,
            tuple(coalition.helper_ids),
            tuple(starts),
            tuple(ends),
            tuple(deltas),
        )

    # -- main entry point --------------------------------------------------

    def update(self, step_info, raw_team_reward, t_env):
        """Observe one training transition and emit shadow rows.

        Pure observation: returns nothing the caller acts on, mutates no
        environment state, and consumes no RNG of any kind.
        """
        incoming_episode = int(step_info.episode_id)
        if self.episode_id is None:
            self.episode_id = incoming_episode
        elif incoming_episode != self.episode_id:
            # Keys are episode-specific by construction; two episode ids must
            # never share one episode's histories, latches or budgets.
            raise AssertionError(
                "SELMA shadow invariant violated: step record for episode "
                f"{incoming_episode} arrived while episode {self.episode_id} "
                "is open -- reset_episode was not called at the boundary"
            )
        step = int(step_info.episode_step)
        self.last_step = step

        post_positions = {f.position for f in step_info.post_foods}
        pre_positions = {f.position for f in step_info.pre_foods}
        for position in pre_positions - post_positions:
            self.removal_step.setdefault(self.food_id(position), step)

        food_removed = bool(step_info.successful_load_events)
        progress = bool(raw_team_reward > 0) or food_removed
        self.progress_history.append(progress)

        # One BFS per agent per transition, reused for every food and variant.
        distances_by_agent = {
            int(a.agent_id): bfs_distances(a.position, self.grid_shape, post_positions)
            for a in step_info.agents
        }
        food_levels = {f.position: f.level for f in step_info.post_foods}
        loading_cells = {
            position: [
                c
                for c in (
                    (position[0] - 1, position[1]),
                    (position[0] + 1, position[1]),
                    (position[0], position[1] - 1),
                    (position[0], position[1] + 1),
                )
                if 0 <= c[0] < self.grid_shape[0]
                and 0 <= c[1] < self.grid_shape[1]
                and c not in post_positions
            ]
            for position in post_positions
        }
        for agent_id, dist in distances_by_agent.items():
            for position, cells in loading_cells.items():
                self.distance_history.setdefault(
                    (agent_id, self.food_id(position)), []
                ).append((step, min_distance_to_loading_cell(dist, cells)))

        # Record this transition's coalition-required failures, keyed by the
        # canonical episode-specific food id.
        current = {}
        for event in step_info.failed_load_events:
            if not event.coalition_required:
                continue
            fid = self.food_id(event.food_position)
            current[fid] = event
            for agent_id in event.loader_ids:
                self.fail_history.setdefault((agent_id, fid), []).append(step)

        # Coalitions are variant-independent: compute once per food per step.
        coalitions = {
            fid: select_helper_coalition(
                food_position=self.food_position(fid),
                food_level=event.food_level,
                loader_ids=event.loader_ids,
                loader_levels=event.loader_levels,
                agents=step_info.agents,
                food_positions=post_positions,
                grid_shape=self.grid_shape,
                distances_by_agent=distances_by_agent,
            )
            for fid, event in current.items()
        }
        inbound_by_food = {
            fid: self._inbound_status(coalitions[fid], fid, step) for fid in current
        }
        gate_open_by_w = {w: self._team_gate_open(step, w) for w in TEAM_GATE_GRID}
        window_complete_by_w = {
            w: self._team_window_complete(w) for w in TEAM_GATE_GRID
        }

        for variant in self.variants:
            self._update_variant(
                variant,
                step,
                t_env,
                step_info,
                current,
                coalitions,
                inbound_by_food,
                gate_open_by_w,
                window_complete_by_w,
                distances_by_agent,
                raw_team_reward,
                food_removed,
                progress,
                food_levels,
            )

    def _update_variant(
        self,
        variant,
        step,
        t_env,
        step_info,
        current,
        coalitions,
        inbound_by_food,
        gate_open_by_w,
        window_complete_by_w,
        distances_by_agent,
        raw_team_reward,
        food_removed,
        progress,
        food_levels,
    ):
        state = self.variant_state[variant.variant_id]
        gate_open = gate_open_by_w[variant.w_team]
        window_complete = window_complete_by_w[variant.w_team]

        # Gate-statistic denominators. ``observed_steps`` counts every
        # transition; ``eligible_steps`` counts only transitions on which this
        # variant's W_team window is complete, and is therefore the correct --
        # and W_team-dependent -- denominator for gate_open_percentage. The
        # ungated variant has no window, so all its observed steps are eligible.
        state.bump("observed_steps")
        if window_complete:
            state.bump("eligible_steps")
            if gate_open:
                state.bump("gate_open_steps")
        else:
            state.bump("incomplete_team_window_steps")

        # Resolve latched foods that vanished, before any new firing decision.
        for fid, lifecycle in state.foods.items():
            if lifecycle.latched and fid in self.removal_step:
                lifecycle.latched = False
                lifecycle.cooldown_remaining = self.cooldown_steps
                state.bump("resolved_food_gone")

        eligible_foods = {}
        for fid, event in current.items():
            position = self.food_position(fid)
            coalition = coalitions[fid]
            qualifying = []
            for agent_id in event.loader_ids:
                count = self._window_fail_count(agent_id, fid, step, variant.w_fail)
                if count >= variant.m:
                    qualifying.append((agent_id, count))
            if not qualifying:
                continue

            state.bump("raw_candidates", len(qualifying))
            gate_suppressed = variant.w_team is not None and not gate_open
            inbound, inbound_ids, starts, ends, deltas = inbound_by_food[fid]
            inbound_suppressed = variant.use_inbound and inbound

            if gate_suppressed:
                # An episode-start window that has not yet completed is not a
                # stall; it is reported separately so it is never confused with
                # a gate that closed because real progress occurred.
                reason = (
                    "team_window_incomplete"
                    if not window_complete
                    else "team_gate_closed"
                )
            elif not coalition.feasible:
                reason = "no_feasible_coalition"
            elif inbound_suppressed:
                reason = "inbound_help_present"
            else:
                reason = ""

            for agent_id, count in qualifying:
                trigger_distances = self.distance_history.get((agent_id, fid), [])
                recent = [d for _, d in trigger_distances[-(self.w_inbound + 1) :]]
                delta = (
                    recent[-1] - recent[0]
                    if len(recent) >= 2 and None not in (recent[0], recent[-1])
                    else None
                )
                self._add_row(
                    row_type="raw_candidate",
                    variant=variant,
                    step=step,
                    t_env=t_env,
                    event=event,
                    food_id=fid,
                    coalition=coalition,
                    gate_open=gate_open,
                    raw_team_reward=raw_team_reward,
                    food_removed=food_removed,
                    progress=progress,
                    trigger_agent_ids=(agent_id,),
                    failed_count=count,
                    inbound=inbound,
                    inbound_ids=inbound_ids,
                    starts=starts,
                    ends=ends,
                    deltas=deltas,
                    trigger_distance=recent[-1] if recent else None,
                    trigger_distance_delta=delta,
                    raw_would_fire=not gate_suppressed and coalition.feasible,
                    would_fire_after_inbound=(
                        not gate_suppressed
                        and coalition.feasible
                        and not inbound_suppressed
                    ),
                    lifecycle=state.foods.get(fid),
                    events_used=state.events_used,
                    coalesced_candidate_count=len(qualifying),
                    suppression_reason=reason,
                    event_id="",
                )

            if gate_suppressed:
                state.bump("suppressed_by_team_gate", len(qualifying))
                if not window_complete:
                    state.bump("suppressed_by_incomplete_team_window", len(qualifying))
                continue
            if not coalition.feasible:
                state.bump("semantically_invalid", len(qualifying))
                continue
            if inbound_suppressed:
                state.bump("suppressed_by_inbound", len(qualifying))
                continue
            if len(qualifying) > 1:
                state.bump("same_food_coalesced", len(qualifying) - 1)
            eligible_foods[fid] = (
                tuple(sorted(a for a, _ in qualifying)),
                max(c for _, c in qualifying),
                event,
                coalition,
                inbound_by_food[fid],
                len(qualifying),
            )

        # Lifecycle update for every food this variant has ever seen this episode.
        touched = set(eligible_foods) | set(state.foods)
        for fid in sorted(touched):
            position = self.food_position(fid)
            lifecycle = state.foods.get(fid)
            if lifecycle is None:
                lifecycle = _FoodLifecycle(self.rearm_steps)
                state.foods[fid] = lifecycle

            eligible = fid in eligible_foods
            was_rearmed = lifecycle.false_streak >= self.rearm_steps

            just_resolved = False
            if lifecycle.latched and not eligible:
                # Condition stopped holding: resolve after rearm_steps of silence.
                if lifecycle.false_streak + 1 >= self.rearm_steps:
                    lifecycle.latched = False
                    lifecycle.cooldown_remaining = self.cooldown_steps
                    just_resolved = True
                    state.bump("resolved_condition_false")
            if (
                not lifecycle.latched
                and not just_resolved
                and lifecycle.cooldown_remaining > 0
            ):
                lifecycle.cooldown_remaining -= 1

            if not eligible:
                lifecycle.false_streak += 1
                continue
            lifecycle.false_streak = 0

            if lifecycle.latched:
                state.bump("suppressed_by_latch")
                continue
            if lifecycle.cooldown_remaining > 0:
                state.bump("suppressed_by_cooldown")
                continue
            if not was_rearmed:
                state.bump("suppressed_by_rearm")
                continue
            if state.events_used >= self.max_events_per_episode:
                state.bump("suppressed_by_budget")
                continue

            trigger_ids, count, event, coalition, inbound_info, n_cands = eligible_foods[
                fid
            ]
            inbound, inbound_ids, starts, ends, deltas = inbound_info
            lifecycle.latched = True
            lifecycle.fired_count += 1
            state.events_used += 1
            state.bump("final_events")
            self._event_counter += 1
            event_id = (
                f"ep{self.episode_id}-{variant.variant_id}-"
                f"food{position[0]},{position[1]}-t{step}"
            )
            self._add_row(
                row_type="coalesced_event",
                variant=variant,
                step=step,
                t_env=t_env,
                event=event,
                food_id=fid,
                coalition=coalition,
                gate_open=gate_open,
                raw_team_reward=raw_team_reward,
                food_removed=food_removed,
                progress=progress,
                trigger_agent_ids=trigger_ids,
                failed_count=count,
                inbound=inbound,
                inbound_ids=inbound_ids,
                starts=starts,
                ends=ends,
                deltas=deltas,
                trigger_distance=None,
                trigger_distance_delta=None,
                raw_would_fire=True,
                would_fire_after_inbound=True,
                lifecycle=lifecycle,
                events_used=state.events_used,
                coalesced_candidate_count=n_cands,
                suppression_reason="",
                event_id=event_id,
            )

    # -- row construction --------------------------------------------------

    def _add_row(
        self,
        row_type,
        variant,
        step,
        t_env,
        event,
        food_id,
        coalition,
        gate_open,
        raw_team_reward,
        food_removed,
        progress,
        trigger_agent_ids,
        failed_count,
        inbound,
        inbound_ids,
        starts,
        ends,
        deltas,
        trigger_distance,
        trigger_distance_delta,
        raw_would_fire,
        would_fire_after_inbound,
        lifecycle,
        events_used,
        coalesced_candidate_count,
        suppression_reason,
        event_id,
    ):
        position = self.food_position(food_id)
        row = {
            "row_type": row_type,
            "condition": self.condition,
            "seed": self.seed,
            "t_env": t_env,
            "episode_id": self.episode_id,
            "episode_step": step,
            "variant_id": variant.variant_id,
            "w_fail": variant.w_fail,
            "m": variant.m,
            "team_gate_variant": variant.team_gate_variant,
            "team_gate_open": int(bool(gate_open)),
            "inbound_variant": variant.inbound_variant,
            "raw_team_reward": float(raw_team_reward),
            "food_removed": int(bool(food_removed)),
            "progress": int(bool(progress)),
            "trigger_agent_ids": trigger_agent_ids,
            # Canonical identity; the position column is derived from it purely
            # for readability and is never itself a key.
            "food_id": food_id,
            "food_position": position,
            "food_level": event.food_level,
            "failed_load_count_in_window": failed_count,
            "loader_ids": event.loader_ids,
            "loader_levels": event.loader_levels,
            "loader_level_sum": event.total_loader_level,
            "level_deficit": event.level_deficit,
            "required_extra_level": coalition.required_extra_level,
            "helper_ids": coalition.helper_ids,
            "helper_levels": coalition.helper_levels,
            "helper_distances": coalition.helper_distances,
            "total_helper_distance": coalition.total_distance,
            "recipient_ids": coalition.recipient_ids,
            "candidate_loading_cells": coalition.candidate_cells,
            "selected_cell_assignment": coalition.assignment,
            "tiebreak_evidence": coalition.tiebreak_key,
            "n_feasible_subsets": coalition.n_feasible_subsets,
            "coalition_feasible": int(bool(coalition.feasible)),
            "coalition_reason": coalition.reason,
            "trigger_distance": trigger_distance,
            "trigger_distance_delta": trigger_distance_delta,
            "inbound_help": int(bool(inbound)),
            "inbound_helper_ids": inbound_ids,
            "inbound_start_distances": starts,
            "inbound_end_distances": ends,
            "inbound_distance_deltas": deltas,
            "raw_would_fire": int(bool(raw_would_fire)),
            "would_fire_after_inbound": int(bool(would_fire_after_inbound)),
            "latched": int(bool(lifecycle.latched)) if lifecycle else 0,
            "cooldown_remaining": lifecycle.cooldown_remaining if lifecycle else 0,
            "rearm_false_streak": lifecycle.false_streak if lifecycle else 0,
            "events_used": events_used,
            "event_budget": self.max_events_per_episode,
            "coalesced_candidate_count": coalesced_candidate_count,
            "suppression_reason": suppression_reason,
            "resolved_within_5": "",
            "resolved_within_10": "",
            "event_id": event_id,
        }
        self.pending_rows.append((row, step, food_id))

    # -- natural-resolution labels ----------------------------------------

    def _label(self, step, food_id, horizon):
        """Independent per-horizon label: resolved | unresolved | censored."""
        removal = self.removal_step.get(food_id)
        if removal is not None and step < removal <= step + horizon:
            return "resolved"
        if self.last_step >= step + horizon:
            return "unresolved"
        return "censored"

    def finish_episode(self):
        """Resolve pending labels and emit rows in a stable deterministic order.

        Called at the end of every training episode. Censored horizons stay
        censored; they are never silently converted into "unresolved".
        """
        for row, step, food_id in self.pending_rows:
            row["resolved_within_5"] = self._label(step, food_id, 5)
            row["resolved_within_10"] = self._label(step, food_id, 10)
        self.episodes_seen += 1
        for variant_id, state in self.variant_state.items():
            cumulative = self.cumulative_counters[variant_id]
            for key, value in state.counters.items():
                cumulative[key] = cumulative.get(key, 0) + value
            hist = self.events_per_episode_hist[variant_id]
            hist[state.events_used] = hist.get(state.events_used, 0) + 1
        ordered = sorted(
            self.pending_rows,
            key=lambda item: (
                item[1],
                item[0]["variant_id"],
                0 if item[0]["row_type"] == "raw_candidate" else 1,
                item[2],
                item[0]["trigger_agent_ids"],
            ),
        )
        self.rows.extend(row for row, _, _ in ordered)
        self.pending_rows = []

    def drain_rows(self):
        rows, self.rows = self.rows, []
        return rows

    def variant_counters(self):
        """Cumulative per-variant counters across every episode seen so far."""
        return {vid: dict(c) for vid, c in self.cumulative_counters.items()}

    def gate_open_percentage(self, variant_id):
        """Fraction of *eligible* steps on which this variant's gate was open.

        The denominator is ``eligible_steps``: transitions on which this
        variant's ``W_team`` window is complete. Transitions in an incomplete
        episode-start window are excluded, because a window that has not yet
        filled cannot represent a stall. Each ``W_team`` therefore has its own
        eligibility total, and the ungated variant's total is every observed
        step. Returns ``None`` when no eligible step exists, so a variant whose
        window never completes is never silently reported as "0% open".
        """
        counters = self.cumulative_counters[variant_id]
        eligible = counters.get("eligible_steps", 0)
        if eligible == 0:
            return None
        return counters.get("gate_open_steps", 0) / eligible

    def variant_report(self):
        """Per-variant gate statistics with the correct eligibility denominator.

        This is the aggregation surface the Stage B report is built from; it
        exists so the ``gate_open_percentage`` denominator is defined in exactly
        one place rather than recomputed per consumer.
        """
        report = {}
        for variant in self.variants:
            counters = self.cumulative_counters[variant.variant_id]
            raw = counters.get("raw_candidates", 0)
            suppressed = counters.get("suppressed_by_team_gate", 0)
            report[variant.variant_id] = {
                "w_fail": variant.w_fail,
                "m": variant.m,
                "team_gate_variant": variant.team_gate_variant,
                "inbound_variant": variant.inbound_variant,
                "observed_steps": counters.get("observed_steps", 0),
                "eligible_steps": counters.get("eligible_steps", 0),
                "incomplete_team_window_steps": counters.get(
                    "incomplete_team_window_steps", 0
                ),
                "gate_open_steps": counters.get("gate_open_steps", 0),
                "gate_open_percentage": self.gate_open_percentage(variant.variant_id),
                "raw_candidates": raw,
                "suppressed_by_team_gate": suppressed,
                "suppressed_by_incomplete_team_window": counters.get(
                    "suppressed_by_incomplete_team_window", 0
                ),
                "candidates_after_team_gate": raw - suppressed,
                "proportion_raw_candidates_suppressed_by_gate": (
                    suppressed / raw if raw else None
                ),
                "final_events": counters.get("final_events", 0),
            }
        return report
