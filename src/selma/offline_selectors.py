"""SELMA offline plan selectors -- the free controls for the LLM condition.

Both providers here answer the *same* question the live provider answers over
the *same* code-enumerated answer set, and return the identical response shape.
They differ only in the selection policy, which is what makes the comparison a
controlled one: detector, enumeration, validation, lifetime, potential and
logging are byte-identical across conditions, and the only thing that varies is
who picks the plan.

Both are contract-aware: they work under the V1 loading-cell contract and the
V2 target-allocation contract, because in both cases the offered plans live in
``request.candidate_joint_plans`` and differ only in what an assignment names.

Neither performs any I/O, so both are free to run and cost nothing. Neither
consumes the environment's RNG.

Why these two controls
----------------------

``heuristic_min_travel``
    The ceiling for a *competent but non-semantic* planner: the plan minimising
    total Manhattan travel from each recipient's real position to its assigned
    target. Because the SELMA potential is a monotone decreasing function of
    exactly that quantity, this selector is the potential's argument-maximum by
    construction. If the language model does not beat it, the model's
    contribution is not planning quality.

``random_valid_plan``
    The floor: a uniformly random choice among exactly the plans the model was
    offered. It holds query timing, guidance frequency, lifetime and every other
    mechanism fixed while destroying the *content* of the decision. If the model
    does not beat it, the selection carries no signal.

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
    CONTRACT_V2_TARGET_ALLOCATION,
    SCHEMA_VERSION,
    USAGE_NOT_APPLICABLE,
    V2_SCHEMA_VERSION,
    Assignment,
    GuidanceProvider,
    GuidanceResponse,
    ROLE_APPROACH_AND_LOAD,
    ROLE_HOLD_AND_LOAD,
    ROLE_PURSUE_TARGET,
    TargetGuidanceAssignment,
)

SELECTOR_HEURISTIC = "heuristic_min_travel"
SELECTOR_RANDOM = "random_valid_plan"


def _positions(request):
    return {int(c.agent_id): (int(c.position[0]), int(c.position[1]))
            for c in request.agent_context}


def _manhattan(a, b):
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _target_of(assignment):
    """Where this assignment sends its agent, under either contract."""
    target = getattr(assignment, "target_position", None)
    if target is None:
        target = assignment.loading_cell
    return (int(target[0]), int(target[1]))


def plan_total_travel(plan, positions):
    """Total Manhattan distance from each agent's real position to its target.

    This is the quantity the heuristic minimises and the quantity the potential
    is built from, so "the heuristic plan" and "the plan the potential most
    rewards reaching" are the same object by construction.
    """
    total = 0
    for a in plan.assignments:
        pos = positions.get(int(a.agent_id))
        if pos is None:
            return None            # unrankable: refuse rather than guess
        total += _manhattan(pos, _target_of(a))
    return total


def _response_for(request, plan, positions, confidence=1.0):
    """Build the validated response shape for an already-chosen plan.

    Roles are derived by code exactly as the deterministic mock derives them;
    nothing here invents a field the validator would reject.
    """
    if request.guidance_contract == CONTRACT_V2_TARGET_ALLOCATION:
        assignments = tuple(
            TargetGuidanceAssignment(
                agent_id=int(a.agent_id),
                role=ROLE_PURSUE_TARGET,
                loading_cell=tuple(a.target_position),
                target_food_id=tuple(a.target_food_id),
                target_position=tuple(a.target_position),
                target_level=int(a.target_level),
            )
            for a in plan.assignments
        )
        schema_version = V2_SCHEMA_VERSION
    else:
        assignments = tuple(
            Assignment(
                agent_id=int(a.agent_id),
                role=(ROLE_HOLD_AND_LOAD
                      if positions.get(int(a.agent_id)) == _target_of(a)
                      else ROLE_APPROACH_AND_LOAD),
                loading_cell=(int(a.loading_cell[0]), int(a.loading_cell[1])),
            )
            for a in plan.assignments
        )
        schema_version = SCHEMA_VERSION
    return GuidanceResponse(
        schema_version=schema_version,
        event_id=request.event_id,
        recipients=tuple(sorted(request.recipient_agents)),
        target_food_id=tuple(request.food_id),
        helper_coalition=tuple(sorted(request.recipient_agents)),
        selected_plan_id=plan.plan_id,
        assignments=assignments,
        confidence=float(confidence),
    )


class HeuristicMinTravelProvider(GuidanceProvider):
    """Selects the offered plan of least total travel; ties break canonically.

    The tie-break is "lowest ``plan_id`` among the minimisers", stated
    explicitly rather than left to sort stability, because ties are common:
    under the V1 contract the audit measured a tie at the optimum on 100 % of
    recorded decisions.
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
            return None            # no valid plan -> no guidance, F_t = 0
        positions = _positions(request)
        best = best_cost = None
        for plan in sorted(plans, key=lambda p: p.plan_id):
            cost = plan_total_travel(plan, positions)
            if cost is None:
                return None
            if best_cost is None or cost < best_cost:
                best, best_cost = plan, cost
        return _response_for(request, best, positions)


class RandomValidPlanProvider(GuidanceProvider):
    """Selects uniformly at random among the offered plans, reproducibly.

    ``run_seed`` must be supplied explicitly: a default would make two runs of
    the "same" condition silently share a draw sequence, which is exactly the
    confound this control exists to remove.
    """

    provider_id = SELECTOR_RANDOM

    def __init__(self, run_seed):
        if run_seed is None:
            raise ValueError(
                "provider {!r} requires an explicit run_seed; a defaulted seed "
                "would make separate seeds share a selection sequence."
                .format(self.provider_id))
        self.run_seed = int(run_seed)
        self.call_count = 0
        self.last_usage = USAGE_NOT_APPLICABLE

    def _rng_for(self, event_id):
        """A private RNG determined by (run_seed, event_id) and nothing else."""
        digest = hashlib.sha256(
            "{}|{}".format(self.run_seed, event_id).encode("utf-8")).digest()
        return random.Random(int.from_bytes(digest[:8], "big"))

    def query(self, request):
        self.call_count += 1
        self.last_usage = USAGE_NOT_APPLICABLE
        plans = request.candidate_joint_plans
        if not plans:
            return None
        ordered = sorted(plans, key=lambda p: p.plan_id)
        plan = ordered[self._rng_for(request.event_id).randrange(len(ordered))]
        return _response_for(request, plan, _positions(request))
