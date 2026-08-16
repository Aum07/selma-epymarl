"""SELMA Stage C: guidance provider abstraction, schema and deterministic mock.

Implements sections 3.1-3.4 of ``docs/project/selma_stage_c_implementation_contract.md``.

Three hard rules from the contract are enforced here:

1. **Code, not the provider, fixes the entities.** The request is built from
   detector output only. A response may *select among* the options the request
   supplies; it can never introduce a food, an agent, a loading cell or a role
   that was not offered.
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

SCHEMA_VERSION = "guidance_schema_v1"

# Roles the code is willing to offer. The provider may only choose from these.
ROLE_HOLD_AND_LOAD = "hold_and_load"
ROLE_APPROACH_AND_LOAD = "approach_and_load"
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
    """Code-built, provider-neutral context for one struggle event."""

    schema_version: str
    event_id: str
    event_type: str
    food_id: Tuple[int, int, int]
    food_position: Tuple[int, int]
    food_level: int
    trigger_agent_ids: Tuple[int, ...]
    recipient_agents: Tuple[int, ...]
    level_deficit: int
    required_extra_level: int
    agent_context: Tuple[AgentContext, ...]


class Assignment(NamedTuple):
    agent_id: int
    role: str
    loading_cell: Tuple[int, int]


class GuidanceResponse(NamedTuple):
    """What a provider returns. Selection among offered options only."""

    schema_version: str
    event_id: str
    recipients: Tuple[int, ...]
    target_food_id: Tuple[int, int, int]
    helper_coalition: Tuple[int, ...]
    assignments: Tuple[Assignment, ...]
    confidence: float


class ValidationResult(NamedTuple):
    valid: bool
    reason: str          # "" when valid; a stable machine-readable code otherwise
    response: object     # the GuidanceResponse when valid, else None


def build_request(event_row, agent_positions):
    """Build a :class:`GuidanceRequest` from one detector ``coalesced_event`` row.

    Every field comes from the detector, except ``agent_positions``, which the
    caller must supply explicitly: these are the **post-transition** positions
    (the state ``s_{t+1}`` in which the recipients will act on the advice), read
    from the GymMA/Stage A accessor. The frozen detector row does not carry
    them and its schema must not change.

    Raises :class:`MissingAgentPositionError` if any recipient has no position.
    Nothing is invented here and there is no positional fallback.
    """
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
                allowed_roles=ALLOWED_ROLES,
                candidate_loading_cells=cells,
            )
        )
    return GuidanceRequest(
        schema_version=SCHEMA_VERSION,
        event_id=str(event_row["event_id"]),
        event_type="coordination_stall",
        food_id=tuple(int(x) for x in event_row["food_id"]),
        food_position=tuple(int(x) for x in event_row["food_position"]),
        food_level=int(event_row["food_level"]),
        trigger_agent_ids=tuple(int(a) for a in event_row["trigger_agent_ids"]),
        recipient_agents=recipients,
        level_deficit=int(event_row["level_deficit"]),
        required_extra_level=int(event_row["required_extra_level"]),
        agent_context=tuple(ctx),
    )


def validate(request, response):
    """Deterministic, side-effect-free validation (contract 3.3).

    Checks run in the contract's order. The first failure short-circuits with a
    stable reason code; there is no partial application.
    """
    if response is None:
        return ValidationResult(False, "no_response", None)
    if not isinstance(response, GuidanceResponse):
        return ValidationResult(False, "malformed_type", None)

    # 1. schema and version
    if response.schema_version != SCHEMA_VERSION:
        return ValidationResult(False, "schema_version_mismatch", None)
    for field in ("event_id", "recipients", "target_food_id", "helper_coalition",
                  "assignments", "confidence"):
        if getattr(response, field, None) is None:
            return ValidationResult(False, "missing_field", None)

    # 2. event identity
    if response.event_id != request.event_id:
        return ValidationResult(False, "event_id_mismatch", None)

    # 3. recipients subset, non-empty
    if not response.recipients:
        return ValidationResult(False, "empty_recipients", None)
    if not set(response.recipients).issubset(set(request.recipient_agents)):
        return ValidationResult(False, "recipients_not_subset", None)

    # 4. target food
    if tuple(response.target_food_id) != tuple(request.food_id):
        return ValidationResult(False, "target_food_mismatch", None)

    # 5. assignments: one per recipient, no duplicates, agents are recipients
    assigned = [a.agent_id for a in response.assignments]
    if len(assigned) != len(set(assigned)):
        return ValidationResult(False, "duplicate_agent_assignment", None)
    if set(assigned) != set(response.recipients):
        return ValidationResult(False, "assignment_recipient_mismatch", None)

    by_agent = {c.agent_id: c for c in request.agent_context}
    for a in response.assignments:
        ctx = by_agent.get(a.agent_id)
        if ctx is None:
            return ValidationResult(False, "agent_not_in_context", None)
        # 6. loading cell must be one the code offered
        if tuple(a.loading_cell) not in set(ctx.candidate_loading_cells):
            return ValidationResult(False, "loading_cell_not_offered", None)
        # 7. role must be one the code offered
        if a.role not in set(ctx.allowed_roles):
            return ValidationResult(False, "role_not_allowed", None)

    # 8. coalition feasibility, re-checked against raw state (never trusted)
    if not set(response.helper_coalition).issubset(set(request.recipient_agents)):
        return ValidationResult(False, "helper_not_recipient", None)
    level_sum = sum(by_agent[a].level for a in response.recipients if a in by_agent)
    if level_sum < request.food_level:
        return ValidationResult(False, "coalition_infeasible", None)

    # 9. confidence range (advisory only; never scales beta or the potential)
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
    assign every recipient to the candidate loading cell that is smallest in
    lexicographic order among those offered, with the role decided by whether
    the agent already stands on its assigned cell. Ties are broken by sorting,
    so the same request always yields byte-identical output.
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
        assignments = []
        for i, ctx in enumerate(sorted(request.agent_context)):
            offered = sorted(set(ctx.candidate_loading_cells))
            if not offered:
                return None  # nothing admissible to assign -> validation will reject
            cell = offered[i % len(offered)]
            role = ROLE_HOLD_AND_LOAD if tuple(ctx.position) == tuple(cell) else ROLE_APPROACH_AND_LOAD
            assignments.append(Assignment(agent_id=ctx.agent_id, role=role, loading_cell=tuple(cell)))
        return GuidanceResponse(
            schema_version=SCHEMA_VERSION,
            event_id=request.event_id,
            recipients=tuple(sorted(request.recipient_agents)),
            target_food_id=tuple(request.food_id),
            helper_coalition=tuple(sorted(request.recipient_agents)),
            assignments=tuple(assignments),
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
