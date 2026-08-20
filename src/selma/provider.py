"""SELMA Stage C: guidance provider abstraction, schema and deterministic mock.

Implements sections 3.1-3.4 of ``docs/project/selma_stage_c_implementation_contract.md``.

Three hard rules from the contract are enforced here:

1. **Code, not the provider, fixes the entities.** V1 is built from detector
   output; V2 combines that event with the runner's immutable agent/food
   snapshot. A response may *select among* the complete plans the request
   supplies; it can never introduce a food, agent, coordinate, or role.
2. **Validation is deterministic and total.** Identical input yields an identical
   verdict, independent of wall-clock, RNG or provider. Any failure -> no active
   subgoal -> ``Phi = 0`` -> ``F_t = 0``. A rejected response never partially
   applies.
3. **No network, no credentials, no cache, no live model.** This slice ships the
   deterministic mock only.

The schema is deliberately provider-neutral: a later live provider consumes the
identical :class:`GuidanceRequest` and must return the identical
:class:`GuidanceResponse` shape, so nothing here needs to change to add one.
"""

from typing import NamedTuple, Optional, Tuple

from selma.joint_plans import enumerate_joint_plans, plan_violations
from selma.target_plans import (
    FoodTarget,
    enumerate_target_plans,
    target_plan_violations,
)

#: Response schema. Bumped to v2 by the joint-plan correction: a response now
#: carries ``selected_plan_id`` and no longer a list of independently composable
#: per-agent assignments.
SCHEMA_VERSION = "guidance_schema_v2"

#: Request representation. Bumped to v2 alongside it: a request now carries the
#: enumerated ``candidate_joint_plans`` and the ``grid_shape`` needed to check
#: them. This is what the cache key records as ``request_schema_version``.
REQUEST_VERSION = "selma_guidance_request_v2"

#: Backward-compatible V1 plus the causal V2 contract.  V1 remains the default
#: so historical commands and tests retain their exact meaning.
CONTRACT_V1_LOADING_CELL = "v1_loading_cell"
CONTRACT_V2_TARGET_ALLOCATION = "v2_target_allocation"
ALLOWED_GUIDANCE_CONTRACTS = (
    CONTRACT_V1_LOADING_CELL,
    CONTRACT_V2_TARGET_ALLOCATION,
)
V2_SCHEMA_VERSION = "guidance_schema_v3_target_allocation"
V2_REQUEST_VERSION = "selma_guidance_request_v3_target_allocation"

# Roles the code is willing to offer. The provider may only choose from these.
ROLE_HOLD_AND_LOAD = "hold_and_load"
ROLE_APPROACH_AND_LOAD = "approach_and_load"
ROLE_PURSUE_TARGET = "pursue_target"
ALLOWED_ROLES = (ROLE_HOLD_AND_LOAD, ROLE_APPROACH_AND_LOAD)

#: Documented validation reason for a request that cannot be built because the
#: caller did not supply a position for every recipient.
REASON_MISSING_AGENT_POSITION = "missing_agent_position"


class MissingAgentPositionError(ValueError):
    """A recipient's position was not supplied by the caller.

    The frozen Stage B detector row carries no agent positions (its schema is
    fixed and must not change), so the runner reads them from the Stage A /
    GymMA accessor and passes them in explicitly. An earlier revision silently
    defaulted to ``(-1, -1)``, which sent every provider a fabricated position
    and made every role degenerate. That fallback is gone: a missing position is
    now an error at the call site and a recorded rejection at the controller.
    """


class ProviderUsage(NamedTuple):
    """Per-query cost accounting.

    ``None`` means **not applicable to this provider**, and is rendered in the
    log as ``n/a``. It never means "zero" and must never be summed as zero. The
    deterministic mock performs no tokenised inference, incurs no billable cost
    and has no meaningful call latency to attribute, so it reports ``None`` for
    all four fields. A live provider fills them with measured values.
    """

    tokens_prompt: Optional[int] = None
    tokens_completion: Optional[int] = None
    latency_s: Optional[float] = None
    cost_usd: Optional[float] = None


#: The mock's usage record: not applicable on every field, by construction.
USAGE_NOT_APPLICABLE = ProviderUsage(None, None, None, None)


class AgentContext(NamedTuple):
    """Per-recipient options the provider may choose between."""

    agent_id: int
    level: int
    position: Tuple[int, int]
    allowed_roles: Tuple[str, ...]
    candidate_loading_cells: Tuple[Tuple[int, int], ...]


class GuidanceRequest(NamedTuple):
    """Code-built, provider-neutral context for one struggle event.

    ``candidate_joint_plans`` is the complete, deterministically enumerated set
    of *physically valid* mappings of the required recipients to loading cells.
    A provider selects one of them by id; it never composes a plan out of parts.
    """

    schema_version: str
    event_id: str
    event_type: str
    food_id: Tuple[int, int, int]
    food_position: Tuple[int, int]
    food_level: int
    grid_shape: Tuple[int, int]
    trigger_agent_ids: Tuple[int, ...]
    recipient_agents: Tuple[int, ...]
    level_deficit: int
    required_extra_level: int
    agent_context: Tuple[AgentContext, ...]
    candidate_joint_plans: Tuple[object, ...] = ()
    guidance_contract: str = CONTRACT_V1_LOADING_CELL
    remaining_foods: Tuple[FoodTarget, ...] = ()

    def plan_by_id(self, plan_id):
        """The offered plan with this id, or ``None``. Never raises."""
        if not isinstance(plan_id, str) or not plan_id:
            return None
        for plan in self.candidate_joint_plans:
            if plan.plan_id == plan_id:
                return plan
        return None


class Assignment(NamedTuple):
    agent_id: int
    role: str
    loading_cell: Tuple[int, int]


class TargetGuidanceAssignment(NamedTuple):
    """V2 response assignment, separate so the V1 tuple shape stays exact.

    ``loading_cell`` is a compatibility alias of ``target_position`` for the
    shared controller/cache transport. The V2 validator requires equality and
    the V2 potential reads ``target_position`` explicitly.
    """

    agent_id: int
    role: str
    loading_cell: Tuple[int, int]
    target_food_id: Tuple[int, int, int]
    target_position: Tuple[int, int]
    target_level: int


class GuidanceResponse(NamedTuple):
    """What a provider returns: **one** offered joint plan, selected by id.

    ``assignments`` is derived by code from the selected plan (the role from the
    agent's real position), never supplied by the model. It is carried here so
    the rest of Stage C is unchanged, and the validator checks that it still
    matches the plan the id names -- neither the id nor the mapping is trusted
    on its own.
    """

    schema_version: str
    event_id: str
    recipients: Tuple[int, ...]
    target_food_id: Tuple[int, int, int]
    helper_coalition: Tuple[int, ...]
    selected_plan_id: str
    assignments: Tuple[Assignment, ...]
    confidence: float


class ValidationResult(NamedTuple):
    valid: bool
    reason: str          # "" when valid; a stable machine-readable code otherwise
    response: object     # the GuidanceResponse when valid, else None


def build_request(event_row, agent_positions, grid_shape, *,
                  guidance_contract=CONTRACT_V1_LOADING_CELL,
                  agent_levels=None, remaining_foods=None):
    """Build a :class:`GuidanceRequest` from one detector ``coalesced_event`` row.

    Every field comes from the detector, except ``agent_positions``, which the
    caller must supply explicitly: these are the **post-transition** positions
    (the state ``s_{t+1}`` in which the recipients will act on the advice), read
    from the GymMA/Stage A accessor. The frozen detector row does not carry
    them and its schema must not change.

    ``grid_shape`` is required, not defaulted: physical validity of a loading
    cell cannot be decided without the grid bounds, and a silent default would
    reintroduce exactly the class of defect the ``(-1, -1)`` position fallback
    already caused once.

    Raises :class:`MissingAgentPositionError` if any recipient has no position.
    Nothing is invented here and there is no positional fallback.
    """
    if guidance_contract not in ALLOWED_GUIDANCE_CONTRACTS:
        raise ValueError("unknown SELMA guidance contract {!r}".format(
            guidance_contract))
    if guidance_contract == CONTRACT_V2_TARGET_ALLOCATION:
        recipients = tuple(sorted(int(a) for a in (agent_positions or {})))
    else:
        recipients = tuple(int(a) for a in event_row["recipient_ids"])
    cells = tuple(tuple(int(c) for c in cell) for cell in event_row["candidate_loading_cells"])
    # Recipients are loaders UNION helpers; levels must be sourced from both,
    # otherwise the feasibility re-check sees a level of 0 for every helper and
    # rejects every genuinely feasible coalition.
    levels = {int(a): int(l) for a, l in zip(event_row["loader_ids"], event_row["loader_levels"])}
    levels.update(
        {int(a): int(l) for a, l in zip(event_row.get("helper_ids", ()) or (),
                                        event_row.get("helper_levels", ()) or ())}
    )
    if agent_levels is not None:
        levels.update({int(a): int(level) for a, level in agent_levels.items()})

    positions = {} if agent_positions is None else {int(k): v for k, v in agent_positions.items()}
    missing = [a for a in recipients if positions.get(a) is None]
    if missing:
        raise MissingAgentPositionError(
            "no position supplied for recipient agent(s) {} of event {!r}; "
            "build_request requires a real post-transition position for every "
            "recipient and has no fallback.".format(
                ", ".join(str(a) for a in missing), event_row.get("event_id")
            )
        )

    ctx = []
    for a in recipients:
        ctx.append(
            AgentContext(
                agent_id=a,
                level=levels.get(a, 0),
                position=tuple(int(x) for x in positions[a]),
                allowed_roles=(ALLOWED_ROLES if guidance_contract == CONTRACT_V1_LOADING_CELL
                               else (ROLE_PURSUE_TARGET,)),
                candidate_loading_cells=(cells if guidance_contract == CONTRACT_V1_LOADING_CELL
                                         else ()),
            )
        )
    food_position = tuple(int(x) for x in event_row["food_position"])
    grid = (int(grid_shape[0]), int(grid_shape[1]))
    if guidance_contract == CONTRACT_V2_TARGET_ALLOCATION:
        episode_id = int(tuple(event_row["food_id"])[0])
        foods = []
        for food in remaining_foods or ():
            position = tuple(int(x) for x in (
                food.position if hasattr(food, "position") else food[0]))
            level = int(food.level if hasattr(food, "level") else food[1])
            foods.append(FoodTarget(
                (episode_id, position[0], position[1]), position, level))
        foods = tuple(sorted(foods, key=lambda f: (f.position, f.level)))
        plans = enumerate_target_plans(
            recipients, levels, foods, tuple(event_row["food_id"]),
            agent_positions=positions, grid_shape=grid)
        schema_version = V2_REQUEST_VERSION
    else:
        foods = ()
        plans = enumerate_joint_plans(recipients, cells, food_position, grid)
        schema_version = REQUEST_VERSION

    return GuidanceRequest(
        schema_version=schema_version,
        event_id=str(event_row["event_id"]),
        event_type="coordination_stall",
        food_id=tuple(int(x) for x in event_row["food_id"]),
        food_position=food_position,
        food_level=int(event_row["food_level"]),
        grid_shape=grid,
        trigger_agent_ids=tuple(int(a) for a in event_row["trigger_agent_ids"]),
        recipient_agents=recipients,
        level_deficit=int(event_row["level_deficit"]),
        required_extra_level=int(event_row["required_extra_level"]),
        agent_context=tuple(ctx),
        # The complete answer set, built by code. A provider never widens it.
        candidate_joint_plans=plans,
        guidance_contract=guidance_contract,
        remaining_foods=foods,
    )


def validate(request, response):
    """Deterministic, side-effect-free validation (contract 3.3, joint-plan v2).

    Checks run in a fixed order. The first failure short-circuits with a stable
    reason code; there is no partial application.

    The joint-plan correction changed step 5 fundamentally. The audited version
    checked each assignment *independently* -- "is this cell one of the offered
    cells?" -- which is satisfiable by two agents sharing one cell, and the live
    pilot duly produced exactly that. Validity is now a property of the whole
    mapping: the selected id must name an offered plan, the response's mapping
    must be that plan, and every joint constraint is re-derived from the raw
    request by :func:`~selma.joint_plans.plan_violations` rather than assumed
    from the fact that the plan came out of the enumerator.
    """
    if response is None:
        return ValidationResult(False, "no_response", None)
    if not isinstance(response, GuidanceResponse):
        return ValidationResult(False, "malformed_type", None)

    # 1. schema and version
    expected_schema = (V2_SCHEMA_VERSION
                       if request.guidance_contract == CONTRACT_V2_TARGET_ALLOCATION
                       else SCHEMA_VERSION)
    if response.schema_version != expected_schema:
        return ValidationResult(False, "schema_version_mismatch", None)
    for field in ("event_id", "recipients", "target_food_id", "helper_coalition",
                  "selected_plan_id", "assignments", "confidence"):
        if getattr(response, field, None) is None:
            return ValidationResult(False, "missing_field", None)

    # 2. event identity
    if response.event_id != request.event_id:
        return ValidationResult(False, "event_id_mismatch", None)

    # 3. recipients: exactly the required set, non-empty. A joint plan covers
    #    every required recipient, so a subset is no longer sufficient.
    if not response.recipients:
        return ValidationResult(False, "empty_recipients", None)
    if set(response.recipients) != set(request.recipient_agents):
        return ValidationResult(False, "recipients_not_exact", None)

    # 4. trigger/target food identity remains code-fixed in both contracts.
    if tuple(response.target_food_id) != tuple(request.food_id):
        return ValidationResult(False, "target_food_mismatch", None)

    # 5. the selected plan must be one the code offered
    plan = request.plan_by_id(response.selected_plan_id)
    if plan is None:
        return ValidationResult(False, "unknown_plan_id", None)

    if request.guidance_contract == CONTRACT_V2_TARGET_ALLOCATION:
        if set(response.helper_coalition) != set(request.recipient_agents):
            return ValidationResult(False, "helper_not_recipient", None)
        raw_levels = {c.agent_id: c.level for c in request.agent_context}
        problems = target_plan_violations(
            response.assignments,
            recipient_agents=request.recipient_agents,
            remaining_foods=request.remaining_foods,
            trigger_food_id=request.food_id,
            agent_levels=raw_levels,
            grid_shape=request.grid_shape,
        )
        if problems:
            return ValidationResult(False, problems[0], None)
        claimed = {
            int(a.agent_id): tuple(a.target_food_id)
            for a in response.assignments
        }
        if claimed != plan.mapping():
            return ValidationResult(False, "plan_assignment_mismatch", None)
        offered_problems = target_plan_violations(
            plan.assignments,
            recipient_agents=request.recipient_agents,
            remaining_foods=request.remaining_foods,
            trigger_food_id=request.food_id,
            agent_levels=raw_levels,
            grid_shape=request.grid_shape,
        )
        if offered_problems:
            return ValidationResult(False, "offered_plan_invalid", None)
        by_agent = {c.agent_id: c for c in request.agent_context}
        assigned_levels = {}
        for a in response.assignments:
            ctx = by_agent.get(a.agent_id)
            if ctx is None:
                return ValidationResult(False, "agent_not_in_context", None)
            if a.role != ROLE_PURSUE_TARGET:
                return ValidationResult(False, "role_not_allowed", None)
            fid = tuple(a.target_food_id)
            assigned_levels[fid] = assigned_levels.get(fid, 0) + int(ctx.level)
        food_levels = {tuple(f.food_id): int(f.level)
                       for f in request.remaining_foods}
        if any(total < food_levels[fid]
               for fid, total in assigned_levels.items()):
            return ValidationResult(False, "coalition_infeasible", None)
        try:
            conf = float(response.confidence)
        except (TypeError, ValueError):
            return ValidationResult(False, "confidence_not_numeric", None)
        if not (0.0 <= conf <= 1.0):
            return ValidationResult(False, "confidence_out_of_range", None)
        return ValidationResult(True, "", response)

    # 6. V1 joint physical validity of the mapping actually carried, re-derived
    #    from the request. This is what a per-assignment check cannot express.
    offered_cells = sorted({
        tuple(int(x) for x in cell)
        for ctx in request.agent_context
        for cell in ctx.candidate_loading_cells
    })
    problems = plan_violations(
        response.assignments,
        recipient_agents=request.recipient_agents,
        candidate_cells=offered_cells,
        food_position=request.food_position,
        grid_shape=request.grid_shape,
    )
    if problems:
        return ValidationResult(False, problems[0], None)

    # 7. the mapping must be the plan the id names -- id and content must agree
    claimed = {int(a.agent_id): tuple(int(x) for x in a.loading_cell)
               for a in response.assignments}
    if claimed != plan.mapping():
        return ValidationResult(False, "plan_assignment_mismatch", None)

    # 8. defence in depth: the code-built plan itself is re-checked, so a future
    #    enumerator that widened the answer set could not slip past here.
    if plan_violations(plan.assignments,
                       recipient_agents=request.recipient_agents,
                       candidate_cells=offered_cells,
                       food_position=request.food_position,
                       grid_shape=request.grid_shape):
        return ValidationResult(False, "offered_plan_invalid", None)

    by_agent = {c.agent_id: c for c in request.agent_context}
    for a in response.assignments:
        ctx = by_agent.get(a.agent_id)
        if ctx is None:
            return ValidationResult(False, "agent_not_in_context", None)
        # 9. role must be one the code offered
        if a.role not in set(ctx.allowed_roles):
            return ValidationResult(False, "role_not_allowed", None)

    # 10. coalition feasibility, re-checked against raw state (never trusted)
    if not set(response.helper_coalition).issubset(set(request.recipient_agents)):
        return ValidationResult(False, "helper_not_recipient", None)
    level_sum = sum(by_agent[a].level for a in response.recipients if a in by_agent)
    if level_sum < request.food_level:
        return ValidationResult(False, "coalition_infeasible", None)

    # 11. confidence range (advisory only; never scales beta or the potential)
    try:
        conf = float(response.confidence)
    except (TypeError, ValueError):
        return ValidationResult(False, "confidence_not_numeric", None)
    if not (0.0 <= conf <= 1.0):
        return ValidationResult(False, "confidence_out_of_range", None)

    return ValidationResult(True, "", response)


class GuidanceProvider:
    """Provider interface.

    A live provider implements :meth:`query` and sets :attr:`last_usage` to the
    measured :class:`ProviderUsage` for that call, so token/latency/cost land in
    the guidance log without the controller knowing anything about transports.
    """

    provider_id = "abstract"
    last_usage = USAGE_NOT_APPLICABLE

    def query(self, request):  # pragma: no cover - interface
        raise NotImplementedError


class DeterministicMockProvider(GuidanceProvider):
    """Stage C mock. Pure function of the request; no network, no credentials.

    The policy is intentionally simple and fully determined by the request:
    **select the canonically first offered joint plan**, ``plan_0``, with each
    role decided by whether the agent already stands on its assigned cell.

    This preserves the mock's intended semantics under the joint-plan contract.
    The audited mock assigned agent *i* (in sorted order) to offered cell
    ``i % len(offered)``, i.e. the lexicographically first cells in order --
    which is exactly the mapping ``plan_0`` denotes whenever a plan exists. The
    one behaviour that is deliberately *not* preserved is its modulo wrap: with
    fewer offered cells than recipients the old rule silently reused a cell and
    produced the same physically impossible guidance the live provider did.
    There is now no plan at all in that case, and the mock returns ``None``.
    """

    provider_id = "deterministic_mock"

    def __init__(self):
        self.call_count = 0
        # Constant by construction: no tokens, no billable cost, no attributable
        # latency. See ProviderUsage -- None is "not applicable", never zero.
        self.last_usage = USAGE_NOT_APPLICABLE

    def query(self, request):
        self.call_count += 1
        self.last_usage = USAGE_NOT_APPLICABLE
        plans = request.candidate_joint_plans
        if not plans:
            return None  # no physically valid plan -> no guidance, F_t = 0
        plan = plans[0]
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
            positions = {int(c.agent_id): tuple(int(x) for x in c.position)
                         for c in request.agent_context}
            assignments = tuple(
                Assignment(
                    agent_id=int(a.agent_id),
                    role=(ROLE_HOLD_AND_LOAD
                          if positions.get(int(a.agent_id)) == tuple(a.loading_cell)
                          else ROLE_APPROACH_AND_LOAD),
                    loading_cell=tuple(int(x) for x in a.loading_cell),
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
            confidence=1.0,
        )


PROVIDERS = {DeterministicMockProvider.provider_id: DeterministicMockProvider}


def make_provider(provider_name):
    """Construct a provider by name. Unsupported names fail loudly."""
    if provider_name not in PROVIDERS:
        raise ValueError(
            "selma_provider={!r} is not supported in this build. Available: {}. "
            "No live provider is implemented in the Stage C deterministic slice.".format(
                provider_name, ", ".join(sorted(PROVIDERS))
            )
        )
    return PROVIDERS[provider_name]()
