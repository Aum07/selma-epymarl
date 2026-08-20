"""SELMA Stage D: OpenAI guidance provider adapter (disabled by default).

Frozen offline contract for this slice
--------------------------------------

===================== ==========================================
provider              ``openai``
api                   Responses API (``client.responses.create``)
model                 ``gpt-5.4-nano-2026-03-17`` (exact dated snapshot)
reasoning effort      ``none``
store                 ``false``
structured output     strict JSON Schema
tools                 none
streaming             disabled
===================== ==========================================

The model snapshot is **provisional** until the development-seed provider-quality
pilot, but it is pinned to that exact dated snapshot everywhere -- configuration,
cache key and tests. An unversioned alias (``gpt-5.4-nano``) is rejected, because
a floating alias would silently change the model behind a cache whose keys claim
otherwise.

Safety properties enforced here
-------------------------------

* the OpenAI SDK is imported **lazily**, inside the client factory, so
  ``off`` / ``shadow`` / deterministic-mock runs never import it and never
  construct a client;
* the API key is read **only** from the ``OPENAI_API_KEY`` environment variable.
  It is never accepted from YAML, Sacred, the CLI or any committed file, never
  logged, never stored, and never placed in an exception message;
* a fake client can be injected for tests, so no test needs the network;
* **the model never invents anything.** It selects among deterministic
  ``candidate_assignment_ids`` built by code. Agents, coordinates, foods,
  rewards, ``beta``, TTL and the potential are all fixed by code; the code maps
  the selected ids back to assignments and the existing deterministic validator
  has the final say;
* every failure path -- timeout, rate limit, transport error, refusal,
  incomplete response, schema failure, unknown or duplicate id -- yields **no
  guidance at all**. There is no heuristic fallback, ever.
"""

import json
import os
import re
import time
from typing import NamedTuple, Optional

from selma.accounting import (
    FALLBACK_INCOMPLETE,
    FALLBACK_MALFORMED,
    FALLBACK_RATE_LIMIT,
    FALLBACK_REFUSAL,
    FALLBACK_TIMEOUT,
    FALLBACK_TRANSPORT,
)
from selma.cache import candidate_plan_ids
from selma.joint_plans import describe_plan
from selma.provider import (
    CONTRACT_V2_TARGET_ALLOCATION,
    ROLE_APPROACH_AND_LOAD,
    ROLE_HOLD_AND_LOAD,
    ROLE_PURSUE_TARGET,
    SCHEMA_VERSION,
    V2_SCHEMA_VERSION,
    Assignment,
    GuidanceProvider,
    GuidanceResponse,
    TargetGuidanceAssignment,
)
from selma.target_plans import describe_target_plan

PROVIDER_ID = "openai"
API_SURFACE = "responses"
MODEL_SNAPSHOT = "gpt-5.4-nano-2026-03-17"
REASONING_EFFORT = "none"
PROMPT_VERSION = "selma_guidance_prompt_v3"
RESPONSE_SCHEMA_VERSION = "selma_guidance_response_schema_v2"
API_KEY_ENV = "OPENAI_API_KEY"

DEFAULT_TIMEOUT_S = 30.0
DEFAULT_MAX_RETRIES = 1
DEFAULT_BACKOFF_S = 0.5
DEFAULT_MAX_OUTPUT_TOKENS = 512

SYSTEM_PROMPT = (
    "You are a coordination adviser for Level-Based Foraging (LBF), a "
    "grid-world task. The rules that matter here: every food and every agent "
    "has an integer level. A food is collected only when agents standing in "
    "cells orthogonally adjacent to that food all play the LOAD action on the "
    "same step and the sum of their levels is at least the food's level. An "
    "agent whose own level is below the food's level therefore can never "
    "collect it alone; it needs helpers on the other adjacent cells. Two "
    "agents can never occupy the same cell. "
    "Your decision is exactly this: choose which of several complete, already "
    "valid coordination plans the team should follow. Each candidate joint "
    "plan is a full assignment of every recipient agent to its own distinct "
    "loading cell; the plans were built and checked by code, so any of them is "
    "physically achievable and you are only judging which is best. "
    "You must return exactly one selected_plan_id, taken from the offered "
    "candidate_joint_plans. You do not compose plans, combine parts of "
    "different plans, invent plan ids, or supply coordinates, agents or roles "
    "of your own. You do not decide rewards, discounting, shaping scale or "
    "subgoal lifetime."
)

V2_SYSTEM_PROMPT = (
    "You are a coordination adviser for Level-Based Foraging (LBF), a "
    "grid-world task. Each food and agent has an integer level. A food can be "
    "collected only by adjacent agents whose simultaneous LOAD actions have "
    "enough combined level. Your one decision is to choose among complete, "
    "already feasible agent-to-food allocation plans built by code. Return "
    "exactly one offered selected_plan_id. Do not compose plans or invent "
    "agents, foods, coordinates, actions, rewards, shaping values, or TTL."
)


class MissingApiKeyError(RuntimeError):
    """The OpenAI provider was actively requested without ``OPENAI_API_KEY``."""


class ProviderUsageDetail(NamedTuple):
    """Usage for one considered request.

    ``None`` means *not applicable / not reported*; it is never summed as zero.
    ``cost_usd`` is deliberately always ``None``: monetary cost is recomputed
    from token counts and a separately versioned price-table date, so no price
    is hard-coded here as authoritative evidence.
    """

    tokens_prompt: Optional[int] = None
    tokens_completion: Optional[int] = None
    latency_s: Optional[float] = None
    cost_usd: Optional[float] = None
    cached_tokens_prompt: Optional[int] = None
    tokens_reasoning: Optional[int] = None
    request_id: Optional[str] = None
    returned_model: Optional[str] = None


def response_json_schema(plan_ids):
    """Strict JSON Schema for the structured output.

    A single ``selected_plan_id`` drawn from an enumerated set. There is
    deliberately **no array anywhere**: the pre-correction schema accepted a
    list of independently composable per-agent ids, which is precisely the shape
    that let a model return two agents on one cell. A list of parts is now
    inexpressible, so that failure mode cannot be represented at all.

    The code still re-checks the id and re-derives every joint constraint,
    because the schema is a convenience and never the authority.
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["selected_plan_id", "confidence"],
        "properties": {
            "selected_plan_id": {"type": "string", "enum": list(plan_ids)},
            "confidence": {"type": "number"},
        },
    }


def build_prompt(request):
    """Deterministic user message. Pure function of the code-built request.

    Prompt v3 offers **complete joint plans**. Each entry states the plan's id,
    its full agent-to-cell mapping, and a short rendering of that mapping, so an
    id is never an opaque token the model has to guess the meaning of. The plans
    come straight off the request, so the described set and the selectable
    ``enum`` are the same set by construction and cannot drift apart.

    Nothing episode-scoped enters the payload: no ``event_id`` (which embeds
    the episode number), no ``food_id``, no seed, no step index, no timestamp.
    """
    if request.guidance_contract == CONTRACT_V2_TARGET_ALLOCATION:
        plans = [
            {
                "plan_id": plan.plan_id,
                "assignments": [
                    {
                        "agent_id": int(a.agent_id),
                        # Episode-scoped ids are deliberately omitted. The
                        # position and level give the model all semantics it
                        # needs while preserving cross-episode cache reuse.
                        "target_food": {
                            "position": [int(a.target_position[0]),
                                         int(a.target_position[1])],
                            "level": int(a.target_level),
                        },
                    }
                    for a in plan.assignments
                ],
                "description": describe_target_plan(plan),
            }
            for plan in request.candidate_joint_plans
        ]
        payload = {
            "guidance_contract": request.guidance_contract,
            "event_type": request.event_type,
            "trigger_food": {
                "position": list(request.food_position),
                "level": int(request.food_level),
            },
            "agents": [
                {
                    "agent_id": int(c.agent_id),
                    "level": int(c.level),
                    "position": list(c.position),
                }
                for c in sorted(request.agent_context,
                                key=lambda c: int(c.agent_id))
            ],
            "remaining_foods": [
                {"position": list(f.position), "level": int(f.level)}
                for f in request.remaining_foods
            ],
            "candidate_joint_plans": plans,
            "instructions": (
                "Each candidate is a COMPLETE code-validated allocation of "
                "every agent to one remaining food. For each used food, the "
                "assigned agents' combined level is already sufficient; the "
                "trigger food is assigned in every plan. Choose the allocation "
                "that best reduces travel and coordination conflict from the "
                "current positions. Return exactly one offered selected_plan_id. "
                "Do not invent or modify agents, foods, coordinates, actions, "
                "rewards, shaping values, or plan contents."
            ),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    plans = [
        {
            "plan_id": plan.plan_id,
            "assignments": [
                {"agent_id": int(a.agent_id),
                 "loading_cell": [int(a.loading_cell[0]), int(a.loading_cell[1])]}
                for a in plan.assignments
            ],
            "description": describe_plan(plan),
        }
        for plan in request.candidate_joint_plans
    ]
    payload = {
        "event_type": request.event_type,
        "target_food": {
            "position": list(request.food_position),
            "level": int(request.food_level),
        },
        "level_deficit": int(request.level_deficit),
        "required_extra_level": int(request.required_extra_level),
        "trigger_agent_ids": sorted(int(a) for a in request.trigger_agent_ids),
        "recipients": [
            {
                "agent_id": int(c.agent_id),
                "level": int(c.level),
                "position": list(c.position),
            }
            for c in sorted(request.agent_context, key=lambda c: int(c.agent_id))
        ],
        "candidate_joint_plans": plans,
        "instructions": (
            "Each entry of candidate_joint_plans is a COMPLETE plan: it already "
            "assigns every one of the {n} recipient agent(s) to its own "
            "distinct loading cell adjacent to the target food. Positions and "
            "loading cells are [row, col] on the same grid as "
            "target_food.position. Every listed plan is already physically "
            "valid and already satisfies the level requirement, so you are "
            "choosing between good options, not checking them. "
            "Return exactly one selected_plan_id, copied from the plan_id of "
            "one of the {m} offered plans. Do not combine plans, do not return "
            "more than one, and do not return anything that is not an offered "
            "plan_id. Prefer the plan that reaches its loading configuration in "
            "the fewest total moves from the recipients' current positions."
        ).format(n=len(request.recipient_agents), m=len(plans)),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _classify(error):
    """Map any exception to a fallback reason without importing the SDK.

    Classification is by exception class name, so it works identically for the
    real SDK and for an injected fake, and adds no import-time dependency.
    """
    name = type(error).__name__.lower()
    if "timeout" in name:
        return FALLBACK_TIMEOUT, True
    if "ratelimit" in name or "toomanyrequests" in name:
        return FALLBACK_RATE_LIMIT, True
    return FALLBACK_TRANSPORT, True


def _get(obj, name, default=None):
    """Read an attribute or a mapping key, so fakes may be plain dicts."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _extract_text_and_refusal(response):
    """Return ``(output_text, refusal_text)`` from a Responses API result."""
    refusal = None
    text = _get(response, "output_text")
    chunks = []
    for item in _get(response, "output", []) or []:
        for part in _get(item, "content", []) or []:
            part_type = _get(part, "type")
            if part_type == "refusal":
                refusal = _get(part, "refusal") or "refused"
            elif part_type in ("output_text", "text"):
                value = _get(part, "text")
                if value:
                    chunks.append(value)
    if text is None and chunks:
        text = "".join(chunks)
    return text, refusal


def default_client_factory(api_key_env=API_KEY_ENV, timeout=DEFAULT_TIMEOUT_S):
    """Construct a real OpenAI client. Called lazily, on the first live call.

    The key is read from the environment only, and never appears in any message
    raised or logged from here.
    """
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise MissingApiKeyError(
            "selma_provider='openai' was requested but the {} environment "
            "variable is not set. The API key is read from the environment "
            "only -- never from YAML, Sacred arguments, the CLI or a committed "
            "file.".format(api_key_env)
        )
    from openai import OpenAI  # lazy: never imported in off/shadow/mock runs

    # max_retries=0 disables the SDK's own transport retries so that EVERY retry
    # is one the ledger can see. attempt_hook fires once per adapter attempt,
    # immediately before _create(); an SDK retry would happen inside that call,
    # below the hook, and would therefore consume neither the live-call cap nor
    # the cost ceiling while still costing real requests. The adapter keeps its
    # own single bounded retry (selma_provider_max_retries=1), which IS counted.
    #
    # Historical note: the six accepted Stage E recalibration runs (execution
    # commit 53d9a2e) ran BEFORE this correction and therefore inherited the SDK
    # default of 2. Their log-reconstructed HTTP counts are recorded in
    # docs/project/selma_stage_e_beta_recalibration_runs_2026-08-18.md.
    return OpenAI(api_key=api_key, timeout=timeout, max_retries=0)


class OpenAIGuidanceProvider(GuidanceProvider):
    """Responses-API adapter behind the Stage C provider interface."""

    provider_id = PROVIDER_ID
    api_surface = API_SURFACE
    prompt_version = PROMPT_VERSION
    response_schema_version = RESPONSE_SCHEMA_VERSION

    def __init__(self, *, client=None, client_factory=None, model=MODEL_SNAPSHOT,
                 reasoning_effort=REASONING_EFFORT, timeout=DEFAULT_TIMEOUT_S,
                 max_retries=DEFAULT_MAX_RETRIES, backoff_s=DEFAULT_BACKOFF_S,
                 max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS, sleep=None,
                 api_key_env=API_KEY_ENV):
        if not re.search(r"-\d{4}-\d{2}-\d{2}$", str(model)):
            raise ValueError(
                "selma requires an exact dated model snapshot (…-YYYY-MM-DD), "
                "not an unversioned alias; got {!r}. A floating alias would "
                "change the model behind cache keys that claim otherwise."
                .format(model)
            )
        self.model = str(model)
        self.reasoning_effort = str(reasoning_effort)
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self.backoff_s = float(backoff_s)
        self.max_output_tokens = int(max_output_tokens)
        self.api_key_env = api_key_env
        self._client = client
        self._client_factory = client_factory
        self._sleep = sleep if sleep is not None else time.sleep

        #: Set by the gateway. Called with the zero-based attempt index before
        #: every HTTP attempt; returning False means the live-call budget is
        #: exhausted and the attempt must not be made.
        self.attempt_hook = None

        self.call_count = 0
        self.attempts = 0
        self.last_usage = ProviderUsageDetail()
        self.last_error = ""
        self.last_attempts = 0
        self.last_request_id = None
        self.last_returned_model = None

    # -- client -------------------------------------------------------------

    @property
    def client(self):
        if self._client is None:
            if self._client_factory is not None:
                self._client = self._client_factory()
            else:
                self._client = default_client_factory(self.api_key_env, self.timeout)
        return self._client

    # -- the provider interface --------------------------------------------

    def query(self, request):
        """Return a validated-shape ``GuidanceResponse``, or ``None``.

        ``None`` always means *no guidance*: the caller falls back to zero
        shaping. ``last_error`` carries the machine-readable reason.
        """
        plan_ids = candidate_plan_ids(request)
        self.last_error = ""
        self.last_attempts = 0
        self.last_usage = ProviderUsageDetail()
        if len(plan_ids) < 2:
            # Defence in depth. The controller already skips zero-plan and
            # singleton events before any cache or provider is reached; if a
            # caller bypasses that policy, no request leaves here either.
            self.last_error = FALLBACK_MALFORMED
            return None

        schema = response_json_schema(plan_ids)
        prompt = build_prompt(request)

        for attempt in range(self.max_retries + 1):
            if self.attempt_hook is not None and not self.attempt_hook(attempt):
                # Budget exhausted: the gateway records the reason.
                return None
            self.last_attempts = attempt + 1
            self.attempts += 1
            started = time.time()
            try:
                raw = self._create(prompt, schema)
            except Exception as error:  # noqa: BLE001 - classified, never re-raised
                reason, retryable = _classify(error)
                self.last_error = reason
                if retryable and attempt < self.max_retries:
                    self._sleep(self.backoff_s * (2 ** attempt))
                    continue
                return None
            self.call_count += 1
            latency = time.time() - started
            response, reason = self._interpret(raw, request, latency)
            if response is not None:
                return response
            self.last_error = reason
            if reason in (FALLBACK_MALFORMED,) and attempt < self.max_retries:
                self._sleep(self.backoff_s * (2 ** attempt))
                continue
            return None
        return None

    def _create(self, prompt, schema):
        try:
            is_v2 = (json.loads(prompt).get("guidance_contract")
                     == CONTRACT_V2_TARGET_ALLOCATION)
        except (TypeError, ValueError, AttributeError):
            is_v2 = False
        return self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content":
                 V2_SYSTEM_PROMPT if is_v2 else SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "selma_guidance",
                    "schema": schema,
                    "strict": True,
                }
            },
            reasoning={"effort": self.reasoning_effort},
            tools=[],
            store=False,
            stream=False,
            max_output_tokens=self.max_output_tokens,
            timeout=self.timeout,
        )

    # -- interpretation -----------------------------------------------------

    def _interpret(self, raw, request, latency):
        """``(GuidanceResponse | None, reason)``. Never raises, never invents."""
        self.last_request_id = _get(raw, "id")
        self.last_returned_model = _get(raw, "model")
        usage = _get(raw, "usage")
        self.last_usage = ProviderUsageDetail(
            tokens_prompt=_int_or_none(_get(usage, "input_tokens")),
            tokens_completion=_int_or_none(_get(usage, "output_tokens")),
            latency_s=float(latency),
            cost_usd=None,
            cached_tokens_prompt=_int_or_none(
                _get(_get(usage, "input_tokens_details"), "cached_tokens")),
            tokens_reasoning=_int_or_none(
                _get(_get(usage, "output_tokens_details"), "reasoning_tokens")),
            request_id=self.last_request_id,
            returned_model=self.last_returned_model,
        )

        text, refusal = _extract_text_and_refusal(raw)
        if refusal:
            return None, FALLBACK_REFUSAL
        if _get(raw, "status") == "incomplete":
            return None, FALLBACK_INCOMPLETE
        if not text:
            return None, FALLBACK_MALFORMED
        try:
            payload = json.loads(text)
        except (ValueError, TypeError):
            return None, FALLBACK_MALFORMED
        if not isinstance(payload, dict):
            return None, FALLBACK_MALFORMED

        # The pre-correction body shape is rejected outright rather than
        # reinterpreted: a list of independently composable per-agent ids is
        # exactly what produced physically impossible guidance in the pilot.
        if "assignment_ids" in payload:
            return None, FALLBACK_MALFORMED

        plan = request.plan_by_id(payload.get("selected_plan_id"))
        if plan is None:
            return None, FALLBACK_MALFORMED       # unknown / invented plan id

        recipients = tuple(sorted(int(a) for a in request.recipient_agents))
        mapping = plan.mapping()
        if set(mapping) != set(recipients):
            return None, FALLBACK_MALFORMED
        if request.guidance_contract == CONTRACT_V2_TARGET_ALLOCATION:
            by_agent = {int(a.agent_id): a for a in plan.assignments}
            assignments = tuple(
                TargetGuidanceAssignment(
                    agent_id=agent_id,
                    role=ROLE_PURSUE_TARGET,
                    loading_cell=tuple(by_agent[agent_id].target_position),
                    target_food_id=tuple(by_agent[agent_id].target_food_id),
                    target_position=tuple(by_agent[agent_id].target_position),
                    target_level=int(by_agent[agent_id].target_level),
                )
                for agent_id in recipients
            )
            schema_version = V2_SCHEMA_VERSION
        else:
            positions = {int(c.agent_id): tuple(int(x) for x in c.position)
                         for c in request.agent_context}
            assignments = tuple(
                Assignment(
                    agent_id=agent_id,
                    # The role is decided by code from the real position,
                    # never by the model.
                    role=(ROLE_HOLD_AND_LOAD
                          if positions.get(agent_id) == mapping[agent_id]
                          else ROLE_APPROACH_AND_LOAD),
                    loading_cell=mapping[agent_id],
                )
                for agent_id in recipients
            )
            schema_version = SCHEMA_VERSION
        confidence = payload.get("confidence", 1.0)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            return None, FALLBACK_MALFORMED

        return GuidanceResponse(
            schema_version=schema_version,
            # Identity, target and coalition are code-fixed. The model cannot
            # retarget the subgoal or change who receives it.
            event_id=request.event_id,
            recipients=recipients,
            target_food_id=tuple(int(x) for x in request.food_id),
            helper_coalition=recipients,
            selected_plan_id=plan.plan_id,
            assignments=assignments,
            confidence=confidence,
        ), ""


def _int_or_none(value):
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None
