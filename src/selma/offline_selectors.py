"""SELMA V2: offline joint-plan selectors — the controls for the LLM condition.

Both providers here answer the *same* question the live provider answers ("which
of the code-enumerated joint plans should be adopted?") over the *same* answer
set, and return the identical :class:`~selma.provider.GuidanceResponse` shape.
They differ only in the selection policy. That is what makes the comparison a
controlled one: detector, enumeration, validation, lifecycle, potential, TTL and
logging are byte-for-byte the same across conditions, and the only thing that
varies is who picks the plan.

Neither provider performs any I/O, so both are free to run and carry no monetary
cost. Neither consumes the environment's RNG.

Why these two controls
----------------------

``heuristic_min_travel``
    An upper bound on what a *competent but non-semantic* planner achieves. It
    picks the plan minimising total Manhattan travel from each recipient's real
    position to its assigned loading cell. If the LLM condition does not beat
    this, the LLM's contribution is not planning quality.

``random_valid_plan``
    A lower bound: a uniformly random choice among exactly the plans the LLM was
    offered. It holds query timing, guidance frequency, TTL and every other
    mechanism fixed while destroying the *content* of the decision. If the LLM
    condition does not beat this, the selection content carries no signal.

Determinism
-----------

``random_valid_plan`` is reproducible. Its per-event choice is drawn from a
private :class:`random.Random` seeded by SHA-256 over ``(run_seed, event_id)``,
so it depends on neither call order, nor global RNG state, nor Python's salted
``hash()``. Re-running the same seed reproduces the same selections exactly, and
the draw cannot perturb the environment or the learner.
"""

import hashlib
import random

from selma.provider import (
    USAGE_NOT_APPLICABLE,
    Assignment,
    GuidanceProvider,
    GuidanceResponse,
    ROLE_APPROACH_AND_LOAD,
    ROLE_HOLD_AND_LOAD,
    SCHEMA_VERSION,
)

#: Registry names. Kept here so the registry and the config docs cannot drift.
SELECTOR_HEURISTIC = "heuristic_min_travel"
SELECTOR_RANDOM = "random_valid_plan"


def _positions(request):
    return {int(c.agent_id): (int(c.position[0]), int(c.position[1]))
            for c in request.agent_context}


def _manhattan(a, b):
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def plan_total_travel(plan, positions):
    """Total Manhattan distance from each agent's real position to its cell.

    This is the quantity the heuristic minimises and the quantity the V2
    potential is built from, so "the heuristic plan" and "the plan the potential
    most rewards reaching" are the same object by construction.
    """
    total = 0
    for a in plan.assignments:
        pos = positions.get(int(a.agent_id))
        if pos is None:
            # No real position for a required recipient: treat as unrankable.
            return None
        total += _manhattan(pos, a.loading_cell)
    return total


def _response_for(request, plan, positions, confidence):
    """Build the validated response shape for an already-chosen plan.

    The role is derived from the agent's real position exactly as the Stage C
    mock derives it; nothing here invents a field the validator would reject.
    """
    assignments = tuple(
        Assignment(
            agent_id=int(a.agent_id),
            role=(ROLE_HOLD_AND_LOAD
                  if positions.get(int(a.agent_id)) == (int(a.loading_cell[0]),
                                                        int(a.loading_cell[1]))
                  else ROLE_APPROACH_AND_LOAD),
            loading_cell=(int(a.loading_cell[0]), int(a.loading_cell[1])),
        )
        for a in plan.assignments
    )
    return GuidanceResponse(
        schema_version=SCHEMA_VERSION,
        event_id=request.event_id,
        recipients=tuple(sorted(request.recipient_agents)),
        target_food_id=tuple(request.food_id),
        helper_coalition=tuple(sorted(request.recipient_agents)),
        selected_plan_id=plan.plan_id,
        assignments=assignments,
        confidence=float(confidence),
    )


class HeuristicMinTravelProvider(GuidanceProvider):
    """Selects the offered plan of least total travel. Ties break canonically.

    The tie-break is "lowest ``plan_id`` among the minimisers", which is the
    canonical enumeration order and therefore stable across machines and hash
    seeds. Ties are common in this environment — the audit measured a tie at the
    optimum in 100 % of recorded decisions — so the tie-break is a real part of
    the policy and is stated rather than left to sort stability.
    """

    provider_id = SELECTOR_HEURISTIC

    def __init__(self):
        self.call_count = 0
        self.last_usage = USAGE_NOT_APPLICABLE

    def query(self, request):
        self.call_count += 1
        self.last_usage = USAGE_NOT_APPLICABLE
        plans = request.candidate_joint_plans
        if not plans:
            return None  # no physically valid plan -> no guidance, F_t = 0
        positions = _positions(request)
        best = None
        best_cost = None
        for plan in plans:                      # canonical order: plan_0 first
            cost = plan_total_travel(plan, positions)
            if cost is None:
                return None
            if best_cost is None or cost < best_cost:
                best, best_cost = plan, cost
        return _response_for(request, best, positions, confidence=1.0)


class RandomValidPlanProvider(GuidanceProvider):
    """Selects uniformly at random among the offered plans, reproducibly.

    ``run_seed`` must be supplied explicitly: a default would make two runs of
    the "same" condition silently share a draw sequence, which is precisely the
    confound this control exists to avoid.
    """

    provider_id = SELECTOR_RANDOM

    def __init__(self, run_seed):
        if run_seed is None:
            raise ValueError(
                "provider {!r} requires an explicit run_seed; a defaulted seed "
                "would make separate seeds share a selection sequence."
                .format(self.provider_id)
            )
        self.run_seed = int(run_seed)
        self.call_count = 0
        self.last_usage = USAGE_NOT_APPLICABLE

    def _rng_for(self, event_id):
        """A private RNG determined by (run_seed, event_id) and nothing else."""
        digest = hashlib.sha256(
            "{}|{}".format(self.run_seed, event_id).encode("utf-8")
        ).digest()
        return random.Random(int.from_bytes(digest[:8], "big"))

    def query(self, request):
        self.call_count += 1
        self.last_usage = USAGE_NOT_APPLICABLE
        plans = request.candidate_joint_plans
        if not plans:
            return None
        plan = plans[self._rng_for(request.event_id).randrange(len(plans))]
        return _response_for(request, plan, _positions(request), confidence=1.0)
