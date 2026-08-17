"""SELMA Stage D: the eleven permanent counters and their invariants.

Defined by ``docs/project/selma_stage_d_provider_cache_feasibility_2026-08-16.md``
§7. The six quantities the audit insisted on keeping distinct -- detector
events, requests considered, live calls, cache hits, cache misses and guidance
activations -- each have their own counter here and are never substituted for
one another.

Counters are cumulative for a run and are **not** reset at an episode boundary.
``check_invariants`` is called at run end (and by the tests after every
scenario); a violation is an error, not a warning.
"""

from selma.cache import CACHE_KEY_VERSION, CACHE_SCHEMA_VERSION

#: Joint-plan *skips*. These are deliberately **not** fallbacks: no request was
#: ever considered, so they must not enter ``fallback_count`` (which invariant 6
#: balances against ``provider_request_considered_count``). They get their own
#: counters instead, and appear in the per-transition log so a reader can see
#: why ``F_t = 0``. Frozen policy -- see
#: ``docs/project/selma_stage_d_joint_plan_acceptance_2026-08-16.md``:
#:
#: * zero valid joint plans -> no provider call, no cache call, no guidance;
#: * exactly one valid joint plan -> the same, because a forced "choice" is not
#:   a choice. Every activated LLM subgoal was therefore selected from at least
#:   two physically valid alternatives.
SKIP_NO_VALID_JOINT_PLAN = "no_valid_joint_plan"
SKIP_SINGLETON_NO_CHOICE = "singleton_no_choice"

#: Fallback reasons that are *not* a rejected provider response.
FALLBACK_BUDGET_EXHAUSTED = "budget_exhausted"
FALLBACK_MISSING_POSITION = "missing_agent_position"
FALLBACK_TIMEOUT = "provider_timeout"
FALLBACK_RATE_LIMIT = "provider_rate_limit"
FALLBACK_TRANSPORT = "provider_transport_error"
FALLBACK_REFUSAL = "provider_refusal"
FALLBACK_INCOMPLETE = "provider_incomplete"
FALLBACK_MALFORMED = "provider_malformed"
FALLBACK_READ_ONLY_MISS = "cache_read_only_miss"
FALLBACK_PROVIDER_DISABLED = "provider_disabled"

#: Fallbacks where a provider response *did* arrive and was then rejected. The
#: adapter parses and checks the body itself, so the controller may never see a
#: response object -- but one existed, and it must be counted as rejected.
RESPONSE_LEVEL_FALLBACKS = (
    FALLBACK_MALFORMED,
    FALLBACK_REFUSAL,
    FALLBACK_INCOMPLETE,
)

#: Fallbacks whose cause is not "a provider answered badly".
NON_RESPONSE_FALLBACKS = (
    FALLBACK_BUDGET_EXHAUSTED,
    FALLBACK_MISSING_POSITION,
    FALLBACK_TIMEOUT,
    FALLBACK_RATE_LIMIT,
    FALLBACK_TRANSPORT,
    FALLBACK_READ_ONLY_MISS,
    FALLBACK_PROVIDER_DISABLED,
)

COUNTER_NAMES = (
    "detector_event_count",
    "provider_request_considered_count",
    "cache_lookup_count",
    "cache_hit_count",
    "cache_miss_count",
    "live_api_call_count",
    "successful_response_count",
    "rejected_response_count",
    "retry_count",
    "fallback_count",
    "guidance_activation_count",
)


class InvariantViolation(AssertionError):
    """An accounting invariant failed. The run is not trustworthy."""


class SelmaAccounting:
    """The eleven counters, their supporting detail, and the invariant checks."""

    def __init__(self, *, cache_mode="off", live_call_cap=None, provider="none",
                 requested_model="", prompt_version="", response_schema_version="",
                 price_table_date="", live_provider=False):
        for name in COUNTER_NAMES:
            setattr(self, name, 0)
        # supporting detail (not part of the eleven)
        self.terminal_suppressed_event_count = 0
        self.missing_position_count = 0
        #: Joint-plan decision accounting. These three partition every eligible
        #: event that produced a request, and only the third may reach a cache
        #: or a provider.
        self.zero_plan_skip_count = 0
        self.singleton_skip_count = 0
        self.multiple_plan_decision_count = 0
        self.stale_hit_count = 0
        self.budget_exhausted_count = 0
        self.timeout_count = 0
        self.rate_limit_count = 0
        self.transport_error_count = 0
        self.refusal_count = 0
        self.incomplete_count = 0
        self.fallback_reasons = {}
        self.stale_reasons = {}

        # provenance recorded with every run
        self.cache_mode = str(cache_mode)
        self.live_call_cap = live_call_cap
        #: False for any provider that cannot perform network I/O (the
        #: deterministic mock). Such a provider answers a cache miss without an
        #: HTTP request, so invariant 4's lower bound does not apply to it.
        self.live_provider = bool(live_provider)
        self.provider = str(provider)
        self.requested_model = str(requested_model)
        self.returned_models = {}
        self.prompt_version = str(prompt_version)
        self.response_schema_version = str(response_schema_version)
        self.cache_key_version = CACHE_KEY_VERSION
        self.cache_schema_version = CACHE_SCHEMA_VERSION
        self.price_table_date = str(price_table_date)

        # usage, stored as tokens -- never as an authoritative monetary total
        self.input_tokens = 0
        self.cached_input_tokens = 0
        self.output_tokens = 0
        self.reasoning_tokens = 0
        self.total_latency_s = 0.0
        self.request_ids = []

    # -- recording ----------------------------------------------------------

    def note_fallback(self, reason):
        reason = str(reason or "unspecified")
        self.fallback_count += 1
        self.fallback_reasons[reason] = self.fallback_reasons.get(reason, 0) + 1
        if reason == FALLBACK_BUDGET_EXHAUSTED:
            self.budget_exhausted_count += 1
        elif reason == FALLBACK_MISSING_POSITION:
            self.missing_position_count += 1
        elif reason == FALLBACK_TIMEOUT:
            self.timeout_count += 1
        elif reason == FALLBACK_RATE_LIMIT:
            self.rate_limit_count += 1
        elif reason == FALLBACK_TRANSPORT:
            self.transport_error_count += 1
        elif reason == FALLBACK_REFUSAL:
            self.refusal_count += 1
        elif reason == FALLBACK_INCOMPLETE:
            self.incomplete_count += 1

    def note_usage(self, usage, *, returned_model=None, request_id=None):
        """Accumulate token/latency detail. ``None`` fields are skipped, never
        summed as zero (a mock reports ``None`` on every field)."""
        if usage is None:
            return
        self.input_tokens += int(getattr(usage, "tokens_prompt", None) or 0)
        self.output_tokens += int(getattr(usage, "tokens_completion", None) or 0)
        self.cached_input_tokens += int(getattr(usage, "cached_tokens_prompt", 0) or 0)
        self.reasoning_tokens += int(getattr(usage, "tokens_reasoning", 0) or 0)
        latency = getattr(usage, "latency_s", None)
        if latency is not None:
            self.total_latency_s += float(latency)
        if returned_model:
            self.returned_models[str(returned_model)] = (
                self.returned_models.get(str(returned_model), 0) + 1)
        if request_id:
            self.request_ids.append(str(request_id))

    def snapshot(self):
        return {name: getattr(self, name) for name in COUNTER_NAMES}

    def as_dict(self):
        data = self.snapshot()
        data.update({
            "terminal_suppressed_event_count": self.terminal_suppressed_event_count,
            "zero_plan_skip_count": self.zero_plan_skip_count,
            "singleton_skip_count": self.singleton_skip_count,
            "multiple_plan_decision_count": self.multiple_plan_decision_count,
            "missing_position_count": self.missing_position_count,
            "stale_hit_count": self.stale_hit_count,
            "budget_exhausted_count": self.budget_exhausted_count,
            "cache_mode": self.cache_mode,
            "live_call_cap": self.live_call_cap,
            "provider": self.provider,
            "requested_model": self.requested_model,
            "returned_models": dict(self.returned_models),
            "prompt_version": self.prompt_version,
            "response_schema_version": self.response_schema_version,
            "cache_key_version": self.cache_key_version,
            "cache_schema_version": self.cache_schema_version,
            "price_table_date": self.price_table_date,
            "input_tokens": self.input_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_latency_s": self.total_latency_s,
            "fallback_reasons": dict(self.fallback_reasons),
            "stale_reasons": dict(self.stale_reasons),
        })
        return data

    # -- invariants ---------------------------------------------------------

    def check_invariants(self, *, max_retries=0, strict_activation=True):
        """Return the list of violated invariants (empty when all hold).

        Numbering follows §7.1 of the feasibility report. Invariant 2 carries
        the qualification the report's text implies but does not spell out: with
        ``selma_cache_mode='off'`` there is, by definition, no lookup, so the
        equality holds only when the cache is enabled and the counter must be
        zero when it is not.
        """
        problems = []

        def fail(number, text):
            problems.append("I{}: {}".format(number, text))

        if self.detector_event_count < self.provider_request_considered_count:
            fail(1, "detector_event_count {} < provider_request_considered_count {}"
                 .format(self.detector_event_count, self.provider_request_considered_count))

        if self.cache_mode == "off":
            if self.cache_lookup_count != 0:
                fail(2, "cache is off but cache_lookup_count = {}"
                     .format(self.cache_lookup_count))
        elif self.cache_lookup_count != self.provider_request_considered_count:
            fail(2, "cache_lookup_count {} != provider_request_considered_count {}"
                 .format(self.cache_lookup_count, self.provider_request_considered_count))

        if self.cache_hit_count + self.cache_miss_count != self.cache_lookup_count:
            fail(3, "hits {} + misses {} != lookups {}".format(
                self.cache_hit_count, self.cache_miss_count, self.cache_lookup_count))

        # A "call opportunity" is a considered request that reached the point of
        # calling. With a cache that is every miss; with the cache off it is
        # every considered request, because there is nothing to serve it from.
        opportunities = (self.cache_miss_count if self.cache_mode != "off"
                         else self.provider_request_considered_count)
        if self.live_api_call_count != opportunities + self.retry_count:
            # Exact only when every opportunity is permitted to call. Budget
            # exhaustion, read-only mode and a disabled provider each suppress
            # the call, so the identity relaxes to a two-sided bound.
            suppressed = (self.budget_exhausted_count
                          + self.fallback_reasons.get(FALLBACK_READ_ONLY_MISS, 0)
                          + self.fallback_reasons.get(FALLBACK_PROVIDER_DISABLED, 0))
            if self.live_api_call_count > opportunities + self.retry_count:
                fail(4, "live_api_call_count {} > call opportunities {} + retries {}"
                     .format(self.live_api_call_count, opportunities, self.retry_count))
            elif self.live_provider and self.live_api_call_count < (
                    opportunities - suppressed):
                fail(4, "live_api_call_count {} < unsuppressed call opportunities {}"
                     .format(self.live_api_call_count, opportunities - suppressed))

        if (self.cache_hit_count > 0
                and self.live_api_call_count >= self.provider_request_considered_count
                and self.provider_request_considered_count > 0):
            fail(5, "cache hits recorded but live calls did not fall below "
                    "requests considered")

        if (self.successful_response_count + self.fallback_count
                != self.provider_request_considered_count):
            fail(6, "successful {} + fallback {} != considered {}".format(
                self.successful_response_count, self.fallback_count,
                self.provider_request_considered_count))

        if strict_activation:
            if self.guidance_activation_count != self.successful_response_count:
                fail(7, "guidance_activation_count {} != successful_response_count {}"
                     .format(self.guidance_activation_count, self.successful_response_count))
        elif self.guidance_activation_count > self.successful_response_count:
            fail(7, "guidance_activation_count {} > successful_response_count {}"
                 .format(self.guidance_activation_count, self.successful_response_count))

        non_response = sum(self.fallback_reasons.get(r, 0) for r in NON_RESPONSE_FALLBACKS)
        if self.rejected_response_count < self.fallback_count - non_response:
            fail(8, "rejected_response_count {} < response-caused fallbacks {}".format(
                self.rejected_response_count, self.fallback_count - non_response))

        if self.retry_count > max_retries * max(opportunities, 0):
            fail(9, "retry_count {} > max_retries {} x call opportunities {}".format(
                self.retry_count, max_retries, opportunities))

        if self.live_call_cap is not None and self.live_api_call_count > self.live_call_cap:
            fail(10, "live_api_call_count {} exceeds cap {}".format(
                self.live_api_call_count, self.live_call_cap))

        if self.cache_hit_count > self.live_api_call_count + self.cache_hit_count:
            fail(14, "cache hits inflated the live call count")

        # -- joint-plan policy invariants ----------------------------------
        # I11: every event the controller looked at ends in exactly one bucket,
        # and the buckets can never exceed the events the detector produced.
        accounted = (self.zero_plan_skip_count + self.singleton_skip_count
                     + self.multiple_plan_decision_count
                     + self.missing_position_count
                     + self.terminal_suppressed_event_count)
        if accounted > self.detector_event_count:
            fail(11, "plan decisions + skips {} > detector_event_count {}"
                 .format(accounted, self.detector_event_count))

        # I12: only a multiple-plan event may become a considered request. A
        # zero-plan or singleton event must never reach the cache or a provider,
        # so this equality is what proves those skips cost nothing.
        if self.multiple_plan_decision_count != self.provider_request_considered_count:
            fail(12, "multiple_plan_decision_count {} != "
                     "provider_request_considered_count {}".format(
                         self.multiple_plan_decision_count,
                         self.provider_request_considered_count))

        # I13: guidance can only have been activated by a genuine choice.
        if self.guidance_activation_count > self.multiple_plan_decision_count:
            fail(13, "guidance_activation_count {} > multiple_plan_decision_count {}"
                 .format(self.guidance_activation_count,
                         self.multiple_plan_decision_count))

        return problems

    def assert_invariants(self, **kwargs):
        problems = self.check_invariants(**kwargs)
        if problems:
            raise InvariantViolation(
                "SELMA Stage D accounting invariants violated:\n  " + "\n  ".join(problems))
