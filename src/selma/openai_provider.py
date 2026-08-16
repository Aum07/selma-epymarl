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
from selma.cache import candidate_assignment_ids
from selma.provider import (
    ROLE_APPROACH_AND_LOAD,
    ROLE_HOLD_AND_LOAD,
    SCHEMA_VERSION,
    Assignment,
    GuidanceProvider,
    GuidanceResponse,
)

PROVIDER_ID = "openai"
API_SURFACE = "responses"
MODEL_SNAPSHOT = "gpt-5.4-nano-2026-03-17"
REASONING_EFFORT = "none"
PROMPT_VERSION = "selma_guidance_prompt_v2"
RESPONSE_SCHEMA_VERSION = "selma_guidance_response_schema_v1"
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
    "collect it alone; it needs helpers on the other adjacent cells. "
    "Your decision is exactly this: for each recipient agent, which adjacent "
    "loading cell should it move to and load from. You express that decision "
    "by choosing candidate assignment ids that the request supplies. "
    "You may ONLY choose from candidate_assignment_ids: exactly one id per "
    "recipient agent, no duplicates, no invented ids, and no coordinates of "
    "your own. You do not decide rewards, discounting, shaping scale or "
    "subgoal lifetime."
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


def response_json_schema(assignment_ids):
    """Strict JSON Schema for the structured output.

    The selectable ids are enumerated in the schema itself, so a compliant
    structured-output implementation cannot return an unknown id. The code still
    re-checks, because the schema is a convenience and never the authority.
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["assignment_ids", "confidence"],
        "properties": {
            "assignment_ids": {
                "type": "array",
                "items": {"type": "string", "enum": list(assignment_ids)},
            },
            "confidence": {"type": "number"},
        },
    }


def build_prompt(request, assignment_ids):
    """Deterministic user message. Pure function of the code-built request.

    Prompt v2 adds ``candidates``: each offered id spelled out as the (agent,
    loading cell) pair it denotes. v1 sent the ids as bare strings and never
    said what ``a1@3,4`` meant, so the model was choosing between tokens whose
    referents it had to guess. Nothing new is *offered* here -- ``candidates``
    is derived from ``assignment_ids`` by the same parser the response check
    uses, so the described set and the selectable set cannot drift apart.

    Nothing episode-scoped enters the payload: no ``event_id`` (which embeds
    the episode number), no ``food_id``, no seed, no step index, no timestamp.
    """
    candidates = []
    for assignment_id in assignment_ids:
        parsed = parse_assignment_id(assignment_id)
        if parsed is None:                       # unreachable for code-built ids
            continue
        agent_id, cell = parsed
        candidates.append({
            "assignment_id": assignment_id,
            "agent_id": agent_id,
            "loading_cell": [cell[0], cell[1]],
        })
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
        "candidate_assignment_ids": list(assignment_ids),
        "candidates": candidates,
        "instructions": (
            "Each candidate assignment id has the form a<agent_id>@<row>,<col>: "
            "the agent it applies to, and the loading cell that agent should "
            "occupy. The same pairs are listed in `candidates` with the agent "
            "and cell spelled out. Positions are [row, col] on the same grid as "
            "target_food.position. Choose exactly {n} id(s) -- one for each of "
            "the {n} recipient agent(s) -- so that the recipients together "
            "occupy cells adjacent to the target food and their levels sum to "
            "at least target_food.level. Where several assignments are equally "
            "feasible, prefer the one that reaches a loading configuration in "
            "the fewest moves from the recipients' current positions, and "
            "prefer giving different recipients different loading cells. "
            "Return only ids from candidate_assignment_ids."
        ).format(n=len(request.recipient_agents)),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def parse_assignment_id(assignment_id):
    """``a<agent>@<row>,<col>`` -> ``(agent_id, (row, col))``; ``None`` if malformed."""
    try:
        agent_part, cell_part = str(assignment_id).split("@", 1)
        if not agent_part.startswith("a"):
            return None
        agent_id = int(agent_part[1:])
        row_text, col_text = cell_part.split(",", 1)
        return agent_id, (int(row_text), int(col_text))
    except (ValueError, AttributeError):
        return None


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

    return OpenAI(api_key=api_key, timeout=timeout)


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
        ids = candidate_assignment_ids(request)
        self.last_error = ""
        self.last_attempts = 0
        self.last_usage = ProviderUsageDetail()
        if not ids:
            self.last_error = FALLBACK_MALFORMED
            return None

        schema = response_json_schema(ids)
        prompt = build_prompt(request, ids)
        allowed = {i: parse_assignment_id(i) for i in ids}

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
            response, reason = self._interpret(raw, request, allowed, latency)
            if response is not None:
                return response
            self.last_error = reason
            if reason in (FALLBACK_MALFORMED,) and attempt < self.max_retries:
                self._sleep(self.backoff_s * (2 ** attempt))
                continue
            return None
        return None

    def _create(self, prompt, schema):
        return self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
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

    def _interpret(self, raw, request, allowed, latency):
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

        selected = payload.get("assignment_ids")
        if not isinstance(selected, list) or not selected:
            return None, FALLBACK_MALFORMED
        if len(selected) != len(set(selected)):
            return None, FALLBACK_MALFORMED           # duplicate id
        by_agent = {}
        for assignment_id in selected:
            parsed = allowed.get(assignment_id)
            if parsed is None:
                return None, FALLBACK_MALFORMED       # unknown / invented id
            agent_id, cell = parsed
            if agent_id in by_agent:
                return None, FALLBACK_MALFORMED       # two ids for one agent
            by_agent[agent_id] = cell

        recipients = tuple(sorted(int(a) for a in request.recipient_agents))
        if set(by_agent) != set(recipients):
            return None, FALLBACK_MALFORMED

        positions = {int(c.agent_id): tuple(int(x) for x in c.position)
                     for c in request.agent_context}
        assignments = tuple(
            Assignment(
                agent_id=agent_id,
                # The role is decided by code from the real position, never by
                # the model.
                role=(ROLE_HOLD_AND_LOAD if positions.get(agent_id) == by_agent[agent_id]
                      else ROLE_APPROACH_AND_LOAD),
                loading_cell=by_agent[agent_id],
            )
            for agent_id in recipients
        )
        confidence = payload.get("confidence", 1.0)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            return None, FALLBACK_MALFORMED

        return GuidanceResponse(
            schema_version=SCHEMA_VERSION,
            # Identity, target and coalition are code-fixed. The model cannot
            # retarget the subgoal or change who receives it.
            event_id=request.event_id,
            recipients=recipients,
            target_food_id=tuple(int(x) for x in request.food_id),
            helper_coalition=recipients,
            assignments=assignments,
            confidence=confidence,
        ), ""


def _int_or_none(value):
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None
