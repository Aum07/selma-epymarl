"""SELMA Stage C: active-guidance lifecycle, dynamic potential and PBRS shaping.

Implements section 4 of ``docs/project/selma_stage_c_implementation_contract.md``
and the boundary rule fixed by ``docs/selma_stage0_terminal_mask.md``.

Frozen decisions applied here (contract 6.2):

* advice TTL = 15 transitions, ending earlier on target removal, terminal
  transition or replacement;
* ``d_i`` = Manhattan distance from agent ``i`` to the target-food coordinate,
  plus 1 while the food exists (the LOAD action allowance);
* ``d_max_plus_load = (rows - 1) + (cols - 1) + 1``;
* ``Phi_t = mean(phi_i)`` over active recipients, ``0`` when no guidance is active.

``d_i`` is a **bounded progress heuristic**, not an exact shortest-path or
action-count estimate: food blocks movement in LBF, so true navigation cost can
exceed it. PBRS requires a bounded, consistently evaluated potential, not one
that encodes exact navigation cost. With Manhattan distance the bound is exact,
so ``phi_i in [-1, 0]`` always.

The dynamic-potential ordering
------------------------------

:meth:`GuidanceController.apply_transition` is the **single** code path for one
transition, used by the real ``EpisodeRunner`` and by the tests, so the ordering
under test is the ordering that trains. For transition ``t`` it performs, in
order:

1. the guidance in force during ``t`` (``g_t``) records that it served one
   transition -- this is what TTL counts;
2. ``g_{t+1}`` is determined: TTL expiry, target-food removal and terminal
   removal first, then activation or replacement from this step's events;
3. ``phi_after_real`` is computed from ``s_{t+1}`` and **``g_{t+1}``**;
4. a stored terminal forces ``phi_after_effective = 0``.

The value returned as ``phi_after_effective(t)`` is therefore exactly the value
the next transition uses as ``phi_before(t+1)``, which is the property that makes
the transformation telescope. An earlier revision priced the transition with
guidance that was removed immediately afterwards, so every expiry left an
uncancelled ``beta * gamma^(t+1) * Phi`` term in the return -- systematically
negative, because ``Phi <= 0``. That is fixed here and pinned by
``tests/test_selma_stage_c_ordering.py``.

PBRS accounting
---------------

Three distinct quantities, with ``beta`` applied exactly once::

    pbrs_term_unscaled   = gamma * phi_after_effective - phi_before
    shaping_contribution = beta * pbrs_term_unscaled
    training_reward      = raw_team_reward + shaping_contribution

``pbrs_term_unscaled`` is logged beta-free so a run's shaping can be re-derived
at any beta. The name ``shaping_reward`` is deliberately **not** reused: in the
audited revision it held the beta-scaled term, which made the two readings
indistinguishable in a log.
"""

import csv
import os
from typing import NamedTuple, Tuple

#: Advice lifetime: the guidance may be ``guidance_before`` for at most exactly
#: this many environment transitions, and must not influence the (L+1)-th.
TTL_STEPS = 15

#: Tolerance for the phi_after(t) == phi_before(t+1) runtime invariant. The two
#: sides are computed from the same state by the same code, so they agree
#: bit-for-bit in practice; this only absorbs a possible reordering of the mean.
PHI_BOUNDARY_TOLERANCE = 1e-9

EXPIRY_TTL = "ttl"
EXPIRY_TARGET_REMOVED = "target_removed"
EXPIRY_TERMINAL = "terminal"

TRANSITION_ACTIVATION = "activation"
TRANSITION_REPLACEMENT = "replacement"
TRANSITION_EXPIRY = "expiry"
TRANSITION_ONGOING = "ongoing"
TRANSITION_NONE = ""


def d_max_plus_load(grid_shape):
    """(rows - 1) + (cols - 1) + 1 -- 15 for 8x8, 19 for 10x10."""
    rows, cols = int(grid_shape[0]), int(grid_shape[1])
    return (rows - 1) + (cols - 1) + 1


class ActiveGuidance:
    """One validated subgoal in force, with its own age and TTL budget."""

    __slots__ = ("event_id", "subgoal_id", "food_id", "food_position", "recipients",
                 "assignments", "activated_step", "provider_id", "transitions_served",
                 "trigger_agent_ids")

    def __init__(self, event_id, subgoal_id, food_id, food_position, recipients,
                 assignments, activated_step, provider_id, trigger_agent_ids=()):
        self.event_id = event_id
        self.subgoal_id = subgoal_id
        self.food_id = tuple(food_id)
        self.food_position = tuple(food_position)
        self.recipients = tuple(sorted(int(a) for a in recipients))
        self.assignments = dict(assignments)  # agent_id -> (role, loading_cell)
        self.activated_step = int(activated_step)
        self.provider_id = provider_id
        self.trigger_agent_ids = tuple(int(a) for a in trigger_agent_ids)
        #: Transitions for which this guidance has been ``guidance_before``.
        self.transitions_served = 0

    def age(self, step):
        return int(step) - self.activated_step

    def ttl_exhausted(self):
        """True once it has served its full budget of guided transitions."""
        return self.transitions_served >= TTL_STEPS


def phi_components(guidance, agent_positions, food_positions, grid_shape):
    """Per-recipient potential components for an active guidance.

    ``phi_i = -(manhattan(agent_i, food) + 1 if food exists) / d_max_plus_load``
    Returns ``{agent_id: phi_i}``; empty when there is no active guidance.
    """
    if guidance is None:
        return {}
    dmax = d_max_plus_load(grid_shape)
    food_exists = tuple(guidance.food_position) in {tuple(p) for p in (food_positions or ())}
    out = {}
    for a in guidance.recipients:
        pos = (agent_positions or {}).get(a)
        if pos is None:
            continue
        d = abs(int(pos[0]) - guidance.food_position[0]) + abs(int(pos[1]) - guidance.food_position[1])
        if food_exists:
            d += 1  # the terminal LOAD action
        out[a] = -float(d) / float(dmax)
    return out


def phi(guidance, agent_positions, food_positions, grid_shape):
    """Joint potential: mean over active recipients; 0.0 when no guidance."""
    comps = phi_components(guidance, agent_positions, food_positions, grid_shape)
    if not comps:
        return 0.0
    return sum(comps.values()) / float(len(comps))


def pbrs_term_unscaled(gamma, phi_before, phi_after_effective):
    """``gamma * phi_after_effective - phi_before``. Beta-free, never clipped."""
    return float(gamma) * float(phi_after_effective) - float(phi_before)


def shaping_term(beta, gamma, phi_before, phi_after_effective):
    """``beta * (gamma * phi_after_effective - phi_before)``. Never clipped.

    Beta is applied exactly once, here or in :meth:`GuidanceController.shape`,
    never in both.
    """
    return float(beta) * pbrs_term_unscaled(gamma, phi_before, phi_after_effective)


class TransitionRecord(NamedTuple):
    """Everything one transition produced. The log is a projection of this."""

    step: int
    raw_team_reward: object
    training_reward: object
    phi_before: float
    phi_after_real: float
    phi_after_effective: float
    phi_components_after: dict
    pbrs_term_unscaled: float
    shaping_contribution: float
    beta: float
    gamma: float
    guidance_transition: str
    expiry_reason: str
    guidance_active: int
    guidance_age: object
    transitions_served: object
    event_id: str
    subgoal_id: str
    trigger_agent_ids: Tuple[int, ...]
    recipient_agent_ids: Tuple[int, ...]
    target_food_id: object
    query_triggered: int
    queries: int
    provider_valid: int
    fallback_reason: str
    usage: object
    stored_terminated: int
    terminal_cause: str


class GuidanceController:
    """Owns the active guidance, its lifecycle, and the per-transition potential.

    Non-interference by construction: this object consumes no RNG, mutates no
    environment state, and is never constructed in greedy evaluation.
    """

    def __init__(self, provider, grid_shape, beta, gamma, enabled, variant_id,
                 accounting=None):
        self.provider = provider
        #: Optional Stage D accounting (selma.accounting.SelmaAccounting).
        #: ``None`` keeps the Stage C behaviour exactly: nothing is recorded and
        #: no code path below changes.
        self.accounting = accounting
        self.grid_shape = tuple(grid_shape)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.enabled = bool(enabled)
        #: Only events of this frozen detector variant may be offered.
        self.variant_id = str(variant_id)
        self.active = None
        # counters (reconciled against the runner's own tallies)
        self.query_count = 0
        self.provider_call_count = 0
        self.activation_count = 0
        self.replacement_count = 0
        self.expiry_count = 0
        self.rejected_count = 0
        self.terminal_suppressed_events = 0
        self.total_pbrs_unscaled = 0.0
        self.total_shaping_contribution = 0.0
        self.fallback_reasons = {}
        self.expiry_reasons = {}
        self.last_transition = TRANSITION_NONE
        self._subgoal_counter = 0

    #: Cumulative beta-weighted shaping actually added to stored rewards.
    @property
    def total_shaping_reward(self):
        return self.total_shaping_contribution

    # -- episode boundaries -------------------------------------------------

    def reset_episode(self):
        """Clear episode-local guidance. Cumulative counters are NOT cleared."""
        self.active = None
        self.last_transition = TRANSITION_NONE

    # -- potential ----------------------------------------------------------

    def potential(self, agent_positions, food_positions):
        return phi(self.active, agent_positions, food_positions, self.grid_shape)

    def potential_components(self, agent_positions, food_positions):
        return phi_components(self.active, agent_positions, food_positions, self.grid_shape)

    # -- lifecycle ----------------------------------------------------------

    def _eligible(self, event_rows):
        """Frozen-variant, coalition-feasible, unsuppressed coalesced events."""
        return [
            r for r in (event_rows or ())
            if r.get("row_type") == "coalesced_event"
            and str(r.get("variant_id", "")) == self.variant_id
            and r.get("coalition_feasible")
            and not r.get("suppression_reason")
        ]

    def _expire(self, food_positions, stored_terminated):
        """Remove guidance whose life ends at this boundary. Returns the reason."""
        if self.active is None:
            return ""
        if stored_terminated:
            reason = EXPIRY_TERMINAL
        elif tuple(self.active.food_position) not in {
            tuple(p) for p in (food_positions or ())
        }:
            reason = EXPIRY_TARGET_REMOVED
        elif self.active.ttl_exhausted():
            reason = EXPIRY_TTL
        else:
            return ""
        self.active = None
        self.expiry_count += 1
        self.expiry_reasons[reason] = self.expiry_reasons.get(reason, 0) + 1
        return reason

    def _offer(self, event_rows, agent_positions, step):
        """Query the provider for eligible events; activate on a valid response.

        Returns ``(queried, activated, provider_valid, fallback_reason)``. A
        rejected response never partially applies: no active subgoal, Phi = 0.
        """
        from selma.provider import (
            REASON_MISSING_AGENT_POSITION,
            MissingAgentPositionError,
            build_request,
            validate,
        )

        queried = 0
        activated = 0
        provider_valid = 0
        fallback = ""
        for row in self._eligible(event_rows):
            try:
                request = build_request(row, agent_positions=agent_positions)
            except MissingAgentPositionError:
                # Documented rejection: no provider call, no guidance, F_t = 0.
                # No request was built, so this is *not* a considered request
                # and must not be counted as one (Stage D invariant 1's gap).
                self.rejected_count += 1
                fallback = REASON_MISSING_AGENT_POSITION
                self.fallback_reasons[fallback] = self.fallback_reasons.get(fallback, 0) + 1
                if self.accounting is not None:
                    self.accounting.missing_position_count += 1
                continue

            self.query_count += 1
            self.provider_call_count += 1
            queried += 1
            response = self.provider.query(request)
            # The deterministic validator is authoritative for every response,
            # including one served from the Stage D cache.
            verdict = validate(request, response)
            notify = getattr(self.provider, "note_validation", None)
            if notify is not None:
                notify(request, response, verdict.valid)
            if not verdict.valid:
                self.rejected_count += 1
                # A gateway knows *why* there was no usable response (budget
                # exhausted, read-only miss, timeout, refusal...); the bare
                # validator only ever sees "no_response".
                fallback = getattr(self.provider, "last_fallback_reason", "") or verdict.reason
                self.fallback_reasons[fallback] = self.fallback_reasons.get(fallback, 0) + 1
                continue

            provider_valid = 1
            assignments = {a.agent_id: (a.role, tuple(a.loading_cell))
                           for a in verdict.response.assignments}
            if self.active is not None:
                self.replacement_count += 1
                self.last_transition = TRANSITION_REPLACEMENT
            else:
                self.last_transition = TRANSITION_ACTIVATION
            self.activation_count += 1
            self._subgoal_counter += 1
            self.active = ActiveGuidance(
                event_id=verdict.response.event_id,
                subgoal_id="sg{:d}-{}".format(self._subgoal_counter, verdict.response.event_id),
                food_id=verdict.response.target_food_id,
                food_position=row["food_position"],
                recipients=verdict.response.recipients,
                assignments=assignments,
                activated_step=step,
                provider_id=getattr(self.provider, "provider_id", "unknown"),
                trigger_agent_ids=row.get("trigger_agent_ids", ()) or (),
            )
            activated += 1
            if self.accounting is not None:
                self.accounting.guidance_activation_count += 1
        return queried, activated, provider_valid, fallback

    # -- the single per-transition path -------------------------------------

    def apply_transition(self, *, step, raw_team_reward, phi_before,
                         post_agent_positions, post_food_positions, event_rows,
                         episode_ended, stored_terminated, terminal_cause=""):
        """Advance the guidance boundary and price transition ``t``.

        ``phi_before`` is ``Phi(s_t, g_t)`` -- for ``t > 0`` it is the previous
        transition's ``phi_after_effective``, carried forward. Everything after
        the environment step happens here, in the order the contract fixes.

        ``episode_ended`` suppresses every provider query: a subgoal activated on
        the final transition could never influence an action, so querying there
        would only spend a live provider's budget. ``stored_terminated`` is the
        learner's bootstrap mask, and is what zeroes the successor potential.
        """
        phi_before = float(phi_before)

        # Stage D bookkeeping only. Counted before any suppression, so the gap
        # to provider_request_considered_count is exactly the suppressed set.
        if self.accounting is not None:
            self.accounting.detector_event_count += sum(
                1 for r in (event_rows or ())
                if r.get("row_type") == "coalesced_event"
            )

        # 1. TTL accounting: g_t served this transition as guidance_before.
        if self.active is not None:
            self.active.transitions_served += 1

        # 2. Determine g_{t+1}: removal first, then activation/replacement.
        expiry_reason = self._expire(post_food_positions, stored_terminated)
        had_guidance = self.active is not None
        if expiry_reason:
            self.last_transition = TRANSITION_EXPIRY

        queried = activated = provider_valid = 0
        fallback = ""
        if not self.enabled:
            pass
        elif episode_ended:
            suppressed = len(self._eligible(event_rows))
            self.terminal_suppressed_events += suppressed
            if self.accounting is not None:
                self.accounting.terminal_suppressed_event_count += suppressed
        else:
            queried, activated, provider_valid, fallback = self._offer(
                event_rows, post_agent_positions, step
            )

        if activated:
            transition = TRANSITION_REPLACEMENT if had_guidance else TRANSITION_ACTIVATION
        elif expiry_reason:
            transition = TRANSITION_EXPIRY
        elif self.active is not None:
            transition = TRANSITION_ONGOING
        else:
            transition = TRANSITION_NONE
        self.last_transition = transition

        # 3. Successor potential, computed from s_{t+1} and g_{t+1}.
        components = self.potential_components(post_agent_positions, post_food_positions)
        phi_after_real = (
            sum(components.values()) / float(len(components)) if components else 0.0
        )

        # 4. Stage-0 bootstrap verdict: a stored terminal has no successor value,
        #    so it can have no successor potential either.
        phi_after_effective = 0.0 if stored_terminated else phi_after_real

        unscaled = pbrs_term_unscaled(self.gamma, phi_before, phi_after_effective)
        training_reward, contribution = self._apply_beta(raw_team_reward, unscaled)
        self.total_pbrs_unscaled += unscaled
        self.total_shaping_contribution += contribution

        guidance = self.active
        return TransitionRecord(
            step=int(step),
            raw_team_reward=raw_team_reward,
            training_reward=training_reward,
            phi_before=phi_before,
            phi_after_real=phi_after_real,
            phi_after_effective=phi_after_effective,
            phi_components_after=components,
            pbrs_term_unscaled=unscaled,
            shaping_contribution=contribution,
            beta=self.beta,
            gamma=self.gamma,
            guidance_transition=transition,
            expiry_reason=expiry_reason,
            guidance_active=int(guidance is not None),
            guidance_age=guidance.age(step) if guidance else "",
            transitions_served=guidance.transitions_served if guidance else "",
            event_id=guidance.event_id if guidance else "",
            subgoal_id=guidance.subgoal_id if guidance else "",
            trigger_agent_ids=guidance.trigger_agent_ids if guidance else (),
            recipient_agent_ids=guidance.recipients if guidance else (),
            target_food_id=guidance.food_id if guidance else "",
            query_triggered=int(queried > 0),
            queries=queried,
            provider_valid=provider_valid,
            fallback_reason=fallback,
            usage=getattr(self.provider, "last_usage", None) if queried else None,
            stored_terminated=int(bool(stored_terminated)),
            terminal_cause=str(terminal_cause or ""),
        )

    # -- reward -------------------------------------------------------------

    def _apply_beta(self, raw_team_reward, unscaled):
        """``(training_reward, shaping_contribution)``. Beta applied exactly once.

        When ``beta == 0`` the training reward is **assigned directly** from the
        raw reward -- never ``raw + 0.0`` -- so no dtype or floating-point
        representation change can occur (contract 4.2 / 5A).
        """
        if self.beta == 0.0:
            return raw_team_reward, 0.0
        contribution = self.beta * unscaled
        return raw_team_reward + contribution, contribution

    def shape(self, raw_team_reward, phi_before, phi_after_effective):
        """``(training_reward, pbrs_term_unscaled, shaping_contribution)``."""
        unscaled = pbrs_term_unscaled(self.gamma, phi_before, phi_after_effective)
        training_reward, contribution = self._apply_beta(raw_team_reward, unscaled)
        return training_reward, unscaled, contribution


def _fmt(value):
    """Render an optional usage field. ``None`` is 'not applicable', not zero."""
    return "n/a" if value is None else value


class GuidanceCsvWriter:
    """Per-transition PBRS log (spec C.7). Append-only; header on first write.

    Column notes:

    ``condition_checked``
        1 when the frozen detector was consulted on this transition.
    ``query_triggered``
        1 when at least one provider query was issued on this transition. It is
        0 on every final transition by construction: guidance activated there
        could not influence an action.
    ``phi_components``
        per-agent ``phi_i``, ``agent=value`` pairs joined by ``;`` in ascending
        agent order -- one stable column rather than a variable header.
    ``pbrs_term_unscaled`` / ``shaping_contribution``
        the beta-free potential difference and the beta-weighted term actually
        added to ``training_reward``. ``training_reward - raw_team_reward``
        equals ``shaping_contribution`` exactly.
    ``query_tokens_prompt`` / ``query_tokens_completion`` / ``query_latency_s``
    / ``query_cost_usd``
        ``n/a`` for the deterministic mock: it runs no tokenised inference and
        incurs no billable cost, so these are *not applicable* rather than zero
        and must never be summed as zero. A live provider writes measured values.
    """

    COLUMNS = (
        "t_env", "condition", "seed", "episode_id", "episode_step",
        "condition_checked", "query_triggered", "queries",
        "event_id", "subgoal_id", "trigger_agent_ids", "recipient_agent_ids",
        "target_food_id", "provider", "provider_valid", "fallback_reason",
        "guidance_active", "guidance_transition", "guidance_age",
        "transitions_served", "expiry_reason",
        "terminal_cause", "stored_terminated",
        "raw_team_reward", "training_reward",
        "beta", "gamma",
        "phi_before", "potential_after_real", "phi_after_effective",
        "phi_components", "pbrs_term_unscaled", "shaping_contribution",
        "query_tokens_prompt", "query_tokens_completion", "query_latency_s",
        "query_cost_usd",
        "provider_calls", "total_queries", "activations", "replacements",
        "expiries", "rejected", "terminal_suppressed_events",
    )

    def __init__(self, path):
        self.path = path
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)

    def write_row(self, record, controller, t_env, condition, seed, episode_id,
                  condition_checked=1):
        usage = record.usage
        row = {
            "t_env": t_env,
            "condition": condition,
            "seed": seed,
            "episode_id": episode_id,
            "episode_step": record.step,
            "condition_checked": int(bool(condition_checked)),
            "query_triggered": record.query_triggered,
            "queries": record.queries,
            "event_id": record.event_id,
            "subgoal_id": record.subgoal_id,
            "trigger_agent_ids": ";".join(str(a) for a in record.trigger_agent_ids),
            "recipient_agent_ids": ";".join(str(a) for a in record.recipient_agent_ids),
            "target_food_id": ";".join(str(x) for x in record.target_food_id)
            if record.target_food_id else "",
            "provider": getattr(controller.provider, "provider_id", ""),
            "provider_valid": record.provider_valid,
            "fallback_reason": record.fallback_reason,
            "guidance_active": record.guidance_active,
            "guidance_transition": record.guidance_transition,
            "guidance_age": record.guidance_age,
            "transitions_served": record.transitions_served,
            "expiry_reason": record.expiry_reason,
            "terminal_cause": record.terminal_cause,
            "stored_terminated": record.stored_terminated,
            "raw_team_reward": record.raw_team_reward,
            "training_reward": record.training_reward,
            "beta": record.beta,
            "gamma": record.gamma,
            "phi_before": record.phi_before,
            "potential_after_real": record.phi_after_real,
            "phi_after_effective": record.phi_after_effective,
            "phi_components": ";".join(
                "{}={!r}".format(a, record.phi_components_after[a])
                for a in sorted(record.phi_components_after)
            ),
            "pbrs_term_unscaled": record.pbrs_term_unscaled,
            "shaping_contribution": record.shaping_contribution,
            "query_tokens_prompt": _fmt(usage.tokens_prompt if usage else None),
            "query_tokens_completion": _fmt(usage.tokens_completion if usage else None),
            "query_latency_s": _fmt(usage.latency_s if usage else None),
            "query_cost_usd": _fmt(usage.cost_usd if usage else None),
            "provider_calls": controller.provider_call_count,
            "total_queries": controller.query_count,
            "activations": controller.activation_count,
            "replacements": controller.replacement_count,
            "expiries": controller.expiry_count,
            "rejected": controller.rejected_count,
            "terminal_suppressed_events": controller.terminal_suppressed_events,
        }
        need_header = not os.path.exists(self.path) or os.path.getsize(self.path) == 0
        with open(self.path, "a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.COLUMNS)
            if need_header:
                writer.writeheader()
            writer.writerow(row)
