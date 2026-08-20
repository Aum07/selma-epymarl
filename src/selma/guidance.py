"""SELMA Stage C: active-guidance lifecycle, dynamic potential and PBRS shaping.

Implements section 4 of ``docs/project/selma_stage_c_implementation_contract.md``
and the boundary rule fixed by ``docs/selma_stage0_terminal_mask.md``.

Frozen decisions applied here (contract 6.2):

* advice TTL = 15 transitions, ending earlier on target removal, terminal
  transition or replacement;
* V1 ``d_i`` = Manhattan distance from agent ``i`` to the detector target;
  V2 ``d_i`` = distance to that agent's provider-selected target food;
  both add 1 while the relevant food exists (the LOAD action allowance);
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
#:
#: This is the DEFAULT, frozen for 8x8 where ``d_max_plus_load`` is also 15, so
#: an agent could always reach any assigned target before its advice expired.
#: On a larger grid that coincidence breaks: 10x10 has d_max 19 and 15x15 has
#: 29, so a target further away than the TTL loses its guidance mid-journey,
#: every time, and the shaping pulls agents part of the way and then vanishes.
#: ``selma_ttl_steps`` therefore exists so the lifetime can be scaled with the
#: map instead of silently under-serving it.
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
                 "trigger_agent_ids", "selected_plan_id", "guidance_contract",
                 "target_food_ids", "target_positions", "ttl_steps")

    def __init__(self, event_id, subgoal_id, food_id, food_position, recipients,
                 assignments, activated_step, provider_id, trigger_agent_ids=(),
                 selected_plan_id="", guidance_contract="v1_loading_cell",
                 target_food_ids=None, target_positions=None,
                 ttl_steps=TTL_STEPS):
        self.event_id = event_id
        self.subgoal_id = subgoal_id
        self.food_id = tuple(food_id)
        self.food_position = tuple(food_position)
        self.recipients = tuple(sorted(int(a) for a in recipients))
        self.assignments = dict(assignments)  # agent_id -> (role, loading_cell)
        self.activated_step = int(activated_step)
        self.provider_id = provider_id
        self.trigger_agent_ids = tuple(int(a) for a in trigger_agent_ids)
        #: Which code-built joint plan the provider selected.
        self.selected_plan_id = str(selected_plan_id or "")
        self.guidance_contract = str(guidance_contract)
        self.target_food_ids = {
            int(a): tuple(food_id)
            for a, food_id in (target_food_ids or {}).items()
        }
        self.target_positions = {
            int(a): tuple(position)
            for a, position in (target_positions or {}).items()
        }
        #: Advice lifetime for THIS guidance, in transitions.
        self.ttl_steps = int(ttl_steps)
        if self.ttl_steps < 1:
            raise ValueError(
                "ttl_steps={!r} must be >= 1; a guidance that serves no "
                "transition could never influence an action.".format(ttl_steps))
        #: Transitions for which this guidance has been ``guidance_before``.
        self.transitions_served = 0

    def age(self, step):
        return int(step) - self.activated_step

    def ttl_exhausted(self):
        """True once it has served its full budget of guided transitions."""
        return self.transitions_served >= self.ttl_steps


def phi_components(guidance, agent_positions, food_positions, grid_shape):
    """Per-recipient potential components for an active guidance.

    ``phi_i = -(manhattan(agent_i, food) + 1 if food exists) / d_max_plus_load``
    Returns ``{agent_id: phi_i}``; empty when there is no active guidance.
    """
    if guidance is None:
        return {}
    dmax = d_max_plus_load(grid_shape)
    remaining = {tuple(p) for p in (food_positions or ())}
    food_exists = tuple(guidance.food_position) in remaining
    out = {}
    for a in guidance.recipients:
        pos = (agent_positions or {}).get(a)
        if pos is None:
            continue
        if guidance.guidance_contract == "v2_target_allocation":
            target = guidance.target_positions.get(a)
            if target is None:
                # FAIL CLOSED. Skipping this agent would shape over the
                # remaining recipients -- a different quantity, silently. A
                # guidance missing a target for a required recipient yields no
                # components at all, hence Phi = 0 and F_t = 0.
                return {}
            target_exists = tuple(target) in remaining
        else:
            target = guidance.food_position
            target_exists = food_exists
        d = abs(int(pos[0]) - target[0]) + abs(int(pos[1]) - target[1])
        if target_exists:
            d += 1  # the terminal LOAD action
        out[a] = -float(d) / float(dmax)
    return out


def phi(guidance, agent_positions, food_positions, grid_shape):
    """Joint potential: mean over active recipients; 0.0 when no guidance."""
    comps = phi_components(guidance, agent_positions, food_positions, grid_shape)
    if not comps:
        return 0.0
    return sum(comps.values()) / float(len(comps))


def _skip_decision(event_id, decision, plan_count, recipients, outcome):
    """A decision that never reached a cache or a provider.

    Every provider-shaped field is explicitly empty or zero rather than absent,
    so the log can be read without knowing which branch produced a row, and a
    skip can never be mistaken for a cache lookup that happened to miss.
    """
    return DecisionRecord(
        event_id=event_id, decision=decision,
        candidate_plan_count=int(plan_count), recipient_count=int(recipients),
        key_digest="", cache_result="", provider_considered=0,
        live_attempts=0, retries=0, outcome=str(outcome),
        selected_plan_id="", validator_valid=0, validator_reason="",
        physically_valid=0, activated=0,
        tokens_input=None, tokens_cached_input=None, tokens_output=None,
        tokens_reasoning=None, returned_model="", latency_s=None, request_id="",
    )


def pbrs_term_unscaled(gamma, phi_before, phi_after_effective):
    """``gamma * phi_after_effective - phi_before``. Beta-free, never clipped."""
    return float(gamma) * float(phi_after_effective) - float(phi_before)


def shaping_term(beta, gamma, phi_before, phi_after_effective):
    """``beta * (gamma * phi_after_effective - phi_before)``. Never clipped.

    Beta is applied exactly once, here or in :meth:`GuidanceController.shape`,
    never in both.
    """
    return float(beta) * pbrs_term_unscaled(gamma, phi_before, phi_after_effective)


#: What the controller decided for one detector event. These are the four
#: mutually exclusive outcomes of the frozen joint-plan policy plus the
#: documented missing-position rejection; every eligible event lands in exactly
#: one of them.
DECISION_ZERO_PLAN = "zero_plan"
DECISION_SINGLETON = "singleton"
DECISION_MULTIPLE = "multiple"
DECISION_MISSING_POSITION = "missing_position"


class DecisionRecord(NamedTuple):
    """One detector event's full journey, for the measurement log.

    One row per *event*, not per transition: the semantic cache-key digest, the
    candidate-plan count and the provider outcome are all properties of an
    event, and a transition may carry more than one. Aggregating them into the
    per-transition PBRS log would make distinct-key counting impossible.

    ``key_digest`` is the scheme-B semantic key. It is present for every
    multiple-plan decision whose cache is enabled -- **including** one that was
    then refused a live call because the budget was exhausted -- so the number
    of distinct questions asked can be computed independently of whether the
    cache was allowed to fill.
    """

    event_id: str
    decision: str
    candidate_plan_count: int
    recipient_count: int
    key_digest: str
    cache_result: str          # "hit" | "miss" | "off" | "" (no lookup made)
    provider_considered: int
    live_attempts: int
    retries: int
    outcome: str               # "activated" | a stable fallback/skip reason
    selected_plan_id: str
    validator_valid: int
    validator_reason: str
    physically_valid: int
    activated: int
    tokens_input: object
    tokens_cached_input: object
    tokens_output: object
    tokens_reasoning: object
    returned_model: str
    latency_s: object
    request_id: str


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
    selected_plan_id: str
    query_triggered: int
    queries: int
    provider_valid: int
    fallback_reason: str
    usage: object
    stored_terminated: int
    terminal_cause: str
    #: Per-event decisions taken on this transition, in detector order.
    decisions: Tuple[DecisionRecord, ...] = ()
    #: Coalesced detector events seen on this transition, before eligibility.
    detector_events: int = 0
    #: Stable V2 evidence: ``agent=row,col`` pairs from the selected plan.
    target_assignments: str = ""


class GuidanceController:
    """Owns the active guidance, its lifecycle, and the per-transition potential.

    Non-interference by construction: this object consumes no RNG, mutates no
    environment state, and is never constructed in greedy evaluation.
    """

    def __init__(self, provider, grid_shape, beta, gamma, enabled, variant_id,
                 accounting=None, guidance_contract="v1_loading_cell",
                 ttl_steps=TTL_STEPS):
        self.provider = provider
        #: Advice lifetime in transitions. Defaults to the frozen 15; must be
        #: raised on grids where d_max exceeds it or guidance expires before an
        #: agent can reach an assigned target.
        self.ttl_steps = int(ttl_steps)
        if self.ttl_steps < 1:
            raise ValueError("ttl_steps={!r} must be >= 1".format(ttl_steps))
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
        self.guidance_contract = str(guidance_contract)
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
        #: Joint-plan skips, kept apart from ``fallback_reasons`` because no
        #: request was ever considered for them.
        self.skip_reasons = {}
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
        elif self.active.guidance_contract == "v2_target_allocation" and any(
            tuple(position) not in {tuple(p) for p in (food_positions or ())}
            for position in self.active.target_positions.values()
        ):
            reason = EXPIRY_TARGET_REMOVED
        elif (self.active.guidance_contract != "v2_target_allocation"
              and tuple(self.active.food_position) not in {
                  tuple(p) for p in (food_positions or ())
              }):
            reason = EXPIRY_TARGET_REMOVED
        elif self.active.ttl_exhausted():
            reason = EXPIRY_TTL
        else:
            return ""
        self.active = None
        self.expiry_count += 1
        self.expiry_reasons[reason] = self.expiry_reasons.get(reason, 0) + 1
        return reason

    def _offer(self, event_rows, agent_positions, step, *, agent_levels=None,
               remaining_foods=None):
        """Query the provider for eligible events; activate on a valid response.

        Returns ``(queried, activated, provider_valid, fallback_reason)``. A
        rejected response never partially applies: no active subgoal, Phi = 0.

        **Joint-plan policy** (frozen; see
        ``docs/project/selma_stage_d_joint_plan_acceptance_2026-08-16.md``).
        The code enumerates every physically valid joint plan before anything
        else happens, and the plan count decides the path:

        * **zero plans** -- nothing feasible to advise. No provider call, no
          cache lookup, no guidance, ``F_t = 0``, reason
          ``no_valid_joint_plan``;
        * **exactly one plan** -- the "choice" is forced, so consulting a model
          would buy nothing and would spend a live call to be told the only
          answer. No provider call, no cache lookup, no guidance, ``F_t = 0``,
          reason ``singleton_no_choice``. Every activated LLM subgoal is
          therefore one that was selected from at least two valid alternatives;
        * **two or more plans** -- the normal cache -> provider -> validate path.

        The two skips are *not* a heuristic fallback: they activate nothing.
        They are also not fallbacks in the accounting sense -- no request was
        considered, so they must never touch the live-call, cache-hit or
        cache-miss counters (accounting invariants I11-I13).
        """
        from selma.accounting import (
            SKIP_NO_VALID_JOINT_PLAN,
            SKIP_SINGLETON_NO_CHOICE,
        )
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
        decisions = []
        for row in self._eligible(event_rows):
            event_id = str(row.get("event_id", ""))
            try:
                request = build_request(row, agent_positions=agent_positions,
                                        grid_shape=self.grid_shape,
                                        guidance_contract=self.guidance_contract,
                                        agent_levels=agent_levels,
                                        remaining_foods=remaining_foods)
            except MissingAgentPositionError:
                # Documented rejection: no provider call, no guidance, F_t = 0.
                # No request was built, so this is *not* a considered request
                # and must not be counted as one (Stage D invariant 1's gap).
                self.rejected_count += 1
                fallback = REASON_MISSING_AGENT_POSITION
                self.fallback_reasons[fallback] = self.fallback_reasons.get(fallback, 0) + 1
                if self.accounting is not None:
                    self.accounting.missing_position_count += 1
                decisions.append(_skip_decision(
                    event_id, DECISION_MISSING_POSITION, 0, 0, fallback))
                continue

            # -- joint-plan policy, applied before any cache or provider -----
            plan_count = len(request.candidate_joint_plans)
            recipients = len(request.recipient_agents)
            if plan_count < 2:
                skip = (SKIP_NO_VALID_JOINT_PLAN if plan_count == 0
                        else SKIP_SINGLETON_NO_CHOICE)
                fallback = skip
                self.skip_reasons[skip] = self.skip_reasons.get(skip, 0) + 1
                if self.accounting is not None:
                    if plan_count == 0:
                        self.accounting.zero_plan_skip_count += 1
                    else:
                        self.accounting.singleton_skip_count += 1
                decisions.append(_skip_decision(
                    event_id,
                    DECISION_ZERO_PLAN if plan_count == 0 else DECISION_SINGLETON,
                    plan_count, recipients, skip))
                continue
            if self.accounting is not None:
                self.accounting.multiple_plan_decision_count += 1

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
                decisions.append(self._decision(
                    event_id, plan_count, recipients, verdict, response,
                    outcome=fallback, activated=0))
                continue

            provider_valid = 1
            assignments = {a.agent_id: (a.role, tuple(a.loading_cell))
                           for a in verdict.response.assignments}
            target_food_ids = {
                a.agent_id: tuple(a.target_food_id)
                for a in verdict.response.assignments
                if getattr(a, "target_food_id", ())
            }
            target_positions = {
                a.agent_id: tuple(a.target_position)
                for a in verdict.response.assignments
                if getattr(a, "target_position", ())
            }
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
                selected_plan_id=getattr(verdict.response, "selected_plan_id", ""),
                guidance_contract=self.guidance_contract,
                target_food_ids=target_food_ids,
                target_positions=target_positions,
                ttl_steps=self.ttl_steps,
            )
            activated += 1
            if self.accounting is not None:
                self.accounting.guidance_activation_count += 1
            decisions.append(self._decision(
                event_id, plan_count, recipients, verdict, response,
                outcome="activated", activated=1))
        return queried, activated, provider_valid, fallback, tuple(decisions)

    def _decision(self, event_id, plan_count, recipients, verdict, response,
                  *, outcome, activated):
        """Build the durable record for one multiple-plan decision.

        Reads the gateway's instrumentation when present. Never touches a
        prompt, a raw response body or a header: only counts, ids, digests and
        the deterministic verdict.
        """
        from selma.joint_plans import plan_violations

        provider = self.provider
        usage = getattr(provider, "last_usage", None)
        physically_valid = 0
        if verdict.valid and verdict.response is not None:
            physically_valid = 1
        elif response is not None:
            # A response arrived and was rejected; record whether the rejection
            # was a physical-validity one rather than, say, a stale event id.
            physically_valid = 0
        return DecisionRecord(
            event_id=event_id,
            decision=DECISION_MULTIPLE,
            candidate_plan_count=int(plan_count),
            recipient_count=int(recipients),
            key_digest=str(getattr(provider, "last_key_digest", "") or ""),
            cache_result=str(getattr(provider, "last_cache_result", "") or ""),
            provider_considered=1,
            live_attempts=int(getattr(provider, "last_attempts", 0) or 0),
            retries=max(0, int(getattr(provider, "last_attempts", 0) or 0) - 1),
            outcome=str(outcome or ""),
            selected_plan_id=str(
                getattr(verdict.response, "selected_plan_id", "") or ""
                if verdict.response is not None else ""),
            validator_valid=int(bool(verdict.valid)),
            validator_reason=str(verdict.reason or ""),
            physically_valid=physically_valid,
            activated=int(activated),
            tokens_input=getattr(usage, "tokens_prompt", None),
            tokens_cached_input=getattr(usage, "cached_tokens_prompt", None),
            tokens_output=getattr(usage, "tokens_completion", None),
            tokens_reasoning=getattr(usage, "tokens_reasoning", None),
            returned_model=str(getattr(provider, "last_returned_model", "") or ""),
            latency_s=getattr(usage, "latency_s", None),
            request_id=str(getattr(provider, "last_request_id", "") or ""),
        )

    # -- the single per-transition path -------------------------------------

    def apply_transition(self, *, step, raw_team_reward, phi_before,
                         post_agent_positions, post_food_positions, event_rows,
                         episode_ended, stored_terminated, terminal_cause="",
                         post_agent_levels=None, post_food_snapshots=None):
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
        detector_events = sum(
            1 for r in (event_rows or ())
            if r.get("row_type") == "coalesced_event"
        )
        if self.accounting is not None:
            self.accounting.detector_event_count += detector_events

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
        decisions = ()
        if not self.enabled:
            pass
        elif episode_ended:
            suppressed = len(self._eligible(event_rows))
            self.terminal_suppressed_events += suppressed
            if self.accounting is not None:
                self.accounting.terminal_suppressed_event_count += suppressed
        else:
            queried, activated, provider_valid, fallback, decisions = self._offer(
                event_rows, post_agent_positions, step,
                agent_levels=post_agent_levels,
                remaining_foods=post_food_snapshots,
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
        target_assignments = (
            ";".join(
                "{}={},{}".format(agent_id, position[0], position[1])
                for agent_id, position in sorted(guidance.target_positions.items())
            ) if guidance is not None else ""
        )
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
            selected_plan_id=guidance.selected_plan_id if guidance else "",
            query_triggered=int(queried > 0),
            queries=queried,
            provider_valid=provider_valid,
            fallback_reason=fallback,
            usage=getattr(self.provider, "last_usage", None) if queried else None,
            stored_terminated=int(bool(stored_terminated)),
            terminal_cause=str(terminal_cause or ""),
            decisions=decisions,
            detector_events=detector_events,
            target_assignments=target_assignments,
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


class _CsvLogWriter:
    """Append-only CSV with a strictly validated header.

    Every SELMA durable log goes through this. ``DictWriter`` writes values in
    *this* build's column order, so appending to a file whose header has
    different columns shifts every value silently -- a reader then parses one
    column as another and, if the types happen to line up, never notices. That
    is exactly what happened when ``selected_plan_id`` was added and a stale log
    from the previous build was still on disk. Subclasses declare ``COLUMNS``;
    the header is written once and checked on every later append.
    """

    COLUMNS = ()

    def __init__(self, path):
        self.path = path
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)

    def _append(self, row):
        missing = [c for c in self.COLUMNS if c not in row]
        extra = [c for c in row if c not in self.COLUMNS]
        if missing or extra:
            raise ValueError(
                "row does not match {} columns: missing {}, unexpected {}"
                .format(type(self).__name__, missing or "none", extra or "none"))
        need_header = not os.path.exists(self.path) or os.path.getsize(self.path) == 0
        if not need_header:
            self._assert_header_matches()
        with open(self.path, "a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.COLUMNS)
            if need_header:
                writer.writeheader()
            writer.writerow(row)

    def _assert_header_matches(self):
        """Refuse to append rows under a header from a different build."""
        with open(self.path, "r", newline="") as handle:
            existing = next(csv.reader(handle), None)
        if existing is not None and tuple(existing) != tuple(self.COLUMNS):
            missing = [c for c in self.COLUMNS if c not in existing]
            extra = [c for c in existing if c not in self.COLUMNS]
            raise ValueError(
                "refusing to append to {!r}: its header was written by a "
                "different build. Appending would shift every column silently. "
                "Missing here: {}; unexpected: {}. Delete or move the old log."
                .format(self.path, missing or "none", extra or "none")
            )


class SelmaDecisionCsvWriter(_CsvLogWriter):
    """One durable row per detector event the controller considered.

    This is the measurement log. It carries what the per-transition PBRS log
    structurally cannot: the semantic cache-key digest, the candidate joint-plan
    count and the provider outcome are properties of an *event*, and one
    transition may carry several.

    ``key_digest`` is present for every multiple-plan decision whose cache is
    enabled -- including one refused a live call because the budget ran out --
    so the number of distinct questions is measurable independently of whether
    the cache was allowed to fill.

    **Never written here:** prompts, raw response bodies, headers, credentials.
    Only counts, ids, digests, token totals and the deterministic verdict.
    """

    COLUMNS = (
        "t_env", "condition", "seed", "episode_id", "episode_step",
        "greedy_eval", "beta",
        "event_id", "decision", "candidate_plan_count", "recipient_count",
        "key_digest", "cache_result", "provider_considered",
        "live_attempts", "retries", "outcome",
        "selected_plan_id", "validator_valid", "validator_reason",
        "physically_valid", "activated",
        "tokens_input", "tokens_cached_input", "tokens_output",
        "tokens_reasoning", "returned_model", "latency_s", "request_id",
        "detector_events_this_transition",
    )

    def write_rows(self, record, *, t_env, condition, seed, episode_id,
                   greedy_eval=0):
        """One row per decision on this transition. Returns the count written."""
        for decision in record.decisions:
            self._append({
                "t_env": t_env,
                "condition": condition,
                "seed": seed,
                "episode_id": episode_id,
                "episode_step": record.step,
                # Structurally always 0: the runner drops the controller in
                # test_mode, so no decision can originate from greedy
                # evaluation. Logged so the claim is measured, not asserted.
                "greedy_eval": int(bool(greedy_eval)),
                "beta": record.beta,
                "event_id": decision.event_id,
                "decision": decision.decision,
                "candidate_plan_count": decision.candidate_plan_count,
                "recipient_count": decision.recipient_count,
                "key_digest": decision.key_digest,
                "cache_result": decision.cache_result,
                "provider_considered": decision.provider_considered,
                "live_attempts": decision.live_attempts,
                "retries": decision.retries,
                "outcome": decision.outcome,
                "selected_plan_id": decision.selected_plan_id,
                "validator_valid": decision.validator_valid,
                "validator_reason": decision.validator_reason,
                "physically_valid": decision.physically_valid,
                "activated": decision.activated,
                "tokens_input": _fmt(decision.tokens_input),
                "tokens_cached_input": _fmt(decision.tokens_cached_input),
                "tokens_output": _fmt(decision.tokens_output),
                "tokens_reasoning": _fmt(decision.tokens_reasoning),
                "returned_model": decision.returned_model,
                "latency_s": _fmt(decision.latency_s),
                "request_id": decision.request_id,
                "detector_events_this_transition": record.detector_events,
            })
        return len(record.decisions)


class GuidanceCsvWriter(_CsvLogWriter):
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
        "target_food_id", "selected_plan_id", "target_assignments",
        "provider", "provider_valid", "fallback_reason",
        "guidance_active", "guidance_transition", "guidance_age",
        "transitions_served", "expiry_reason",
        "terminal_cause", "stored_terminated",
        "raw_team_reward", "training_reward",
        "beta", "gamma",
        "phi_before", "potential_after_real", "phi_after_effective",
        "phi_components", "pbrs_term_unscaled", "shaping_contribution",
        "query_tokens_prompt", "query_tokens_cached_prompt",
        "query_tokens_completion", "query_latency_s", "query_cost_usd",
        "query_returned_model",
        "detector_events", "decisions", "zero_plan_decisions",
        "singleton_decisions", "multiple_plan_decisions", "greedy_eval",
        # Stage E: per-transition LBF load outcomes, so "was a coordination
        # stall resolved within N transitions" is answerable from this log
        # alone. Read from the Stage A aggregate the runner already computes;
        # nothing here observes or changes the environment.
        "step_successful_loads", "step_cooperative_loads", "step_failed_loads",
        "provider_calls", "total_queries", "activations", "replacements",
        "expiries", "rejected", "terminal_suppressed_events",
    )

    def write_row(self, record, controller, t_env, condition, seed, episode_id,
                  condition_checked=1, greedy_eval=0, step_loads=None):
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
            "selected_plan_id": record.selected_plan_id,
            "target_assignments": record.target_assignments,
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
            "query_tokens_cached_prompt": _fmt(
                getattr(usage, "cached_tokens_prompt", None) if usage else None),
            "query_tokens_completion": _fmt(usage.tokens_completion if usage else None),
            "query_latency_s": _fmt(usage.latency_s if usage else None),
            # Deliberately never computed here: monetary cost is recomputed from
            # token counts and a separately dated price table.
            "query_cost_usd": _fmt(usage.cost_usd if usage else None),
            "query_returned_model": getattr(
                controller.provider, "last_returned_model", "") or "",
            "detector_events": record.detector_events,
            "decisions": len(record.decisions),
            "zero_plan_decisions": sum(
                1 for d in record.decisions if d.decision == DECISION_ZERO_PLAN),
            "singleton_decisions": sum(
                1 for d in record.decisions if d.decision == DECISION_SINGLETON),
            "multiple_plan_decisions": sum(
                1 for d in record.decisions if d.decision == DECISION_MULTIPLE),
            "greedy_eval": int(bool(greedy_eval)),
            "step_successful_loads": int((step_loads or {}).get(
                "successful_load_events", 0)),
            "step_cooperative_loads": int((step_loads or {}).get(
                "necessary_cooperative_load_events", 0)),
            "step_failed_loads": int((step_loads or {}).get(
                "failed_load_events", 0)),
            "provider_calls": controller.provider_call_count,
            "total_queries": controller.query_count,
            "activations": controller.activation_count,
            "replacements": controller.replacement_count,
            "expiries": controller.expiry_count,
            "rejected": controller.rejected_count,
            "terminal_suppressed_events": controller.terminal_suppressed_events,
        }
        self._append(row)
