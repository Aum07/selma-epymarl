"""SELMA Stage D: the provider gateway -- cache, live-call budget, accounting.

One object sits between :class:`~selma.guidance.GuidanceController` and any
provider. It owns the order the feasibility audit fixed:

1. **lookup precedes any call** -- always, in every mode that has a cache;
2. a hit is re-validated before it is offered, so a stale or corrupt row is a
   logged *miss*, never silent reuse;
3. a miss calls the provider only if the mode permits it and the live-call
   budget allows it;
4. the response is written to the cache **only after** the controller's
   deterministic validator has accepted it.

Cache hits never increment ``live_api_call_count``. Budget exhaustion, read-only
mode and a disabled provider all yield **no guidance and zero shaping** -- never
a heuristic substitute, which would confound the LLM condition with the
deterministic ablation it is measured against.
"""

from selma.accounting import (
    FALLBACK_BUDGET_EXHAUSTED,
    FALLBACK_PROVIDER_DISABLED,
    FALLBACK_READ_ONLY_MISS,
    RESPONSE_LEVEL_FALLBACKS,
)
from selma.cache import (
    MISS_STALE_REVALIDATION,
    build_cache_key,
    canonical_json,
    key_digest,
    request_to_dict,
    response_from_dict,
    response_to_dict,
)

CACHE_OFF = "off"
CACHE_READ_WRITE = "read_write"
CACHE_READ_ONLY = "read_only"
CACHE_MODES = (CACHE_OFF, CACHE_READ_WRITE, CACHE_READ_ONLY)


class ProviderGateway:
    """Cache + budget + accounting wrapper around a provider.

    Presents the provider interface (``provider_id``, ``query``, ``last_usage``)
    so :class:`GuidanceController` needs no knowledge of caching, and adds two
    duck-typed hooks the controller calls when present: ``note_validation`` and
    ``last_fallback_reason``.
    """

    def __init__(self, provider, *, accounting, cache=None, cache_mode=CACHE_OFF,
                 live_call_cap=None, condition="", grid_shape=(0, 0),
                 allow_live_calls=True):
        if cache_mode not in CACHE_MODES:
            raise ValueError(
                "selma_cache_mode={!r} is not one of {}".format(cache_mode, CACHE_MODES))
        self.provider = provider
        self.accounting = accounting
        self.cache = cache
        self.cache_mode = cache_mode
        self.live_call_cap = live_call_cap
        self.condition = str(condition)
        self.grid_shape = tuple(grid_shape)
        self.allow_live_calls = bool(allow_live_calls)

        self.last_fallback_reason = ""
        self.last_usage = getattr(provider, "last_usage", None)
        self._pending = None  # state carried from query() to note_validation()

    # -- identity passthrough ------------------------------------------------

    @property
    def provider_id(self):
        return getattr(self.provider, "provider_id", "unknown")

    @property
    def prompt_version(self):
        return getattr(self.provider, "prompt_version", "")

    @property
    def response_schema_version(self):
        return getattr(self.provider, "response_schema_version", "")

    @property
    def model(self):
        return getattr(self.provider, "model", "")

    @property
    def reasoning_effort(self):
        return getattr(self.provider, "reasoning_effort", "")

    # -- budget --------------------------------------------------------------

    def budget_remaining(self):
        if self.live_call_cap is None:
            return None
        return max(0, int(self.live_call_cap) - self.accounting.live_api_call_count)

    def _attempt_hook(self, attempt_index):
        """Called before every HTTP attempt. **Each attempt consumes the cap.**

        The rule is deliberately "every request attempt", not "every logical
        query": a retry is a real request that costs real money and real
        latency, so it must be visible to the cap.
        """
        if self.live_call_cap is not None and \
                self.accounting.live_api_call_count >= int(self.live_call_cap):
            self._pending_budget_exhausted = True
            return False
        self.accounting.live_api_call_count += 1
        if attempt_index > 0:
            self.accounting.retry_count += 1
        return True

    # -- the provider interface ----------------------------------------------

    def query(self, request):
        """Lookup, then (if permitted) call. Returns a response or ``None``."""
        self.last_fallback_reason = ""
        self._pending_budget_exhausted = False
        self.accounting.provider_request_considered_count += 1

        digest = canonical = None
        if self.cache_mode != CACHE_OFF and self.cache is not None:
            payload = build_cache_key(
                request,
                condition=self.condition,
                grid_shape=self.grid_shape,
                provider=self.provider_id,
                model_snapshot=self.model,
                reasoning_effort=self.reasoning_effort,
                prompt_version=self.prompt_version,
                response_schema_version=self.response_schema_version,
            )
            canonical = canonical_json(payload)
            digest = key_digest(canonical)
            self.accounting.cache_lookup_count += 1
            hit = self.cache.lookup(digest, canonical)
            if hit is not None:
                response = self._rebuild(hit, request)
                if response is not None:
                    self.accounting.cache_hit_count += 1
                    self._pending = {"source": "cache", "digest": digest,
                                     "canonical": canonical, "metadata": hit["metadata"],
                                     "returned_model": hit.get("returned_model")}
                    # A hit costs nothing: no call, no tokens, no latency.
                    self.last_usage = None
                    return response
                # Stale or undeserialisable: logged, counted as a miss.
                self.accounting.stale_hit_count += 1
                self._note_stale(MISS_STALE_REVALIDATION)
            self.accounting.cache_miss_count += 1
            self._sync_stale_reasons()

        self._pending = {"source": "live", "digest": digest, "canonical": canonical,
                         "metadata": {}, "returned_model": None}

        if not self.allow_live_calls:
            self.last_fallback_reason = FALLBACK_PROVIDER_DISABLED
            return None
        if self.cache_mode == CACHE_READ_ONLY:
            # A read-only miss must never reach the provider.
            self.last_fallback_reason = FALLBACK_READ_ONLY_MISS
            return None
        if self.budget_remaining() == 0:
            self.last_fallback_reason = FALLBACK_BUDGET_EXHAUSTED
            return None

        previous_hook = getattr(self.provider, "attempt_hook", None)
        has_hook = hasattr(self.provider, "attempt_hook")
        if has_hook:
            self.provider.attempt_hook = self._attempt_hook
        try:
            response = self.provider.query(request)
        finally:
            if has_hook:
                self.provider.attempt_hook = previous_hook
        if not has_hook:
            # A provider without the hook (the deterministic mock) performs no
            # network I/O at all, so it consumes no live-call budget.
            pass

        self.last_usage = getattr(self.provider, "last_usage", None)
        self._pending["metadata"] = self._metadata()
        self._pending["returned_model"] = getattr(self.provider, "last_returned_model", None)
        if response is None:
            if self._pending_budget_exhausted:
                self.last_fallback_reason = FALLBACK_BUDGET_EXHAUSTED
            else:
                self.last_fallback_reason = getattr(self.provider, "last_error", "") or ""
        return response

    # -- post-validation -----------------------------------------------------

    def note_validation(self, request, response, valid):
        """Called by the controller with the authoritative validator verdict.

        A validated response is written to the cache; a rejected one never is.
        """
        pending = self._pending or {}
        self._pending = None
        if valid:
            self.accounting.successful_response_count += 1
            self.accounting.note_usage(
                self.last_usage,
                returned_model=pending.get("returned_model"),
                request_id=getattr(self.provider, "last_request_id", None),
            )
            if (self.cache_mode == CACHE_READ_WRITE and self.cache is not None
                    and pending.get("source") == "live" and pending.get("digest")):
                self.cache.put(
                    digest=pending["digest"],
                    canonical=pending["canonical"],
                    request_payload=request_to_dict(request),
                    response_payload=response_to_dict(response),
                    metadata=pending.get("metadata") or {},
                    provider=self.provider_id,
                    requested_model=self.model,
                    returned_model=pending.get("returned_model"),
                    reasoning_effort=self.reasoning_effort,
                    prompt_version=self.prompt_version,
                    response_schema_version=self.response_schema_version,
                )
            return

        reason = self.last_fallback_reason or (
            getattr(self.provider, "last_error", "") or "invalid_response")
        # A body that arrived and was then rejected -- by the adapter's own
        # parsing or by the deterministic validator -- is a rejected response.
        # A call that never produced a body (budget, read-only, timeout,
        # transport) is not.
        if response is not None or reason in RESPONSE_LEVEL_FALLBACKS:
            self.accounting.rejected_response_count += 1
        self.accounting.note_fallback(reason)

    # -- helpers -------------------------------------------------------------

    def _rebuild(self, hit, request):
        """Rebuild a cached response for the live request, or ``None``."""
        try:
            return response_from_dict(
                hit["response"], event_id=request.event_id,
                target_food_id=request.food_id)
        except (KeyError, TypeError, ValueError):
            return None

    def _metadata(self):
        usage = self.last_usage
        return {
            "request_id": getattr(self.provider, "last_request_id", None),
            "returned_model": getattr(self.provider, "last_returned_model", None),
            "api_surface": getattr(self.provider, "api_surface", ""),
            "input_tokens": getattr(usage, "tokens_prompt", None),
            "cached_input_tokens": getattr(usage, "cached_tokens_prompt", None),
            "output_tokens": getattr(usage, "tokens_completion", None),
            "reasoning_tokens": getattr(usage, "tokens_reasoning", None),
            "latency_s": getattr(usage, "latency_s", None),
            "attempts": getattr(self.provider, "last_attempts", None),
        }

    def _note_stale(self, reason):
        self.accounting.stale_reasons[reason] = (
            self.accounting.stale_reasons.get(reason, 0) + 1)

    def _sync_stale_reasons(self):
        for reason, count in getattr(self.cache, "stale_reasons", {}).items():
            self.accounting.stale_reasons[reason] = count
