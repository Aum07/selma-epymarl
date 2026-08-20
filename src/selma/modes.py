"""SELMA mode contract: the single place every mode/provider/beta rule lives.

Three modes, and nothing else:

``off``
    No detector, no provider, no guidance, no shaping. Bit-for-bit unmodified
    EPyMARL.
``shadow``
    Stage B detector and its diagnostics only. A provider or a non-zero ``beta``
    here is a configuration error, not a silently ignored setting. The variant
    bank keeps its multi-variant calibration behaviour unchanged.
``active``
    Stage C: detector plus an explicitly named provider plus PBRS. ``beta`` may
    be ``0.0`` -- that is the equivalence-testing configuration and it must stay
    bit-identical to ``off``.

**An unknown mode is rejected.** The audited implementation compared
``selma_mode == "shadow"`` and treated every other string -- including typos and
the plausible-looking ``"active"`` -- as ``off``, so a misconfigured experiment
would have run silently as an unshaped baseline. Every rule below fails loudly
at construction time instead.
"""

import math
from typing import NamedTuple

MODE_OFF = "off"
MODE_SHADOW = "shadow"
MODE_ACTIVE = "active"
ALLOWED_MODES = (MODE_OFF, MODE_SHADOW, MODE_ACTIVE)

PROVIDER_NONE = "none"

GUIDANCE_CONTRACT_V1 = "v1_loading_cell"
GUIDANCE_CONTRACT_V2 = "v2_target_allocation"
ALLOWED_GUIDANCE_CONTRACTS = (GUIDANCE_CONTRACT_V1, GUIDANCE_CONTRACT_V2)

#: The Stage B frozen detector variant. Active mode may use no other.
FROZEN_ACTIVE_VARIANT = "wf15_m3_t5_ibon"


class SelmaConfigError(ValueError):
    """A SELMA configuration that must never be allowed to start training."""


class SelmaRuntimeConfig(NamedTuple):
    """The validated, frozen result of reading the SELMA config keys."""

    mode: str
    provider: str
    beta: float
    variant_id: str          # "" means "all variants" (shadow calibration only)
    detector_enabled: bool
    guidance_enabled: bool
    guidance_log_csv: str
    #: Stage D measurement log: one row per detector event. Active mode only.
    decision_log_csv: str = ""
    # -- Stage D: cache and live-call budget. Inert unless mode is active.
    cache_mode: str = "off"
    cache_path: str = ""
    cache_allow_shared_write: bool = False
    live_call_cap: int = 0
    provider_timeout_s: float = 30.0
    provider_max_retries: int = 1
    provider_backoff_s: float = 0.5
    provider_max_output_tokens: int = 512
    price_table_date: str = ""
    #: Dated per-million-token rates. No price is hard-coded anywhere in this
    #: build; a cost ceiling without them is a configuration error.
    price_input_usd_per_mtok: float = 0.0
    price_cached_input_usd_per_mtok: float = 0.0
    price_output_usd_per_mtok: float = 0.0
    #: Conservative spend ceiling in USD. 0.0 means "no monetary ceiling".
    cost_cap_usd: float = 0.0
    #: Advice lifetime in transitions. 15 is the frozen 8x8 value, where
    #: d_max_plus_load is also 15; larger grids need it raised or guidance
    #: expires before an agent can reach its assigned target.
    ttl_steps: int = 15
    #: V1 preserves the audited cell-plan potential; V2 makes the selected
    #: target allocation causal. V1 is the compatibility default.
    guidance_contract: str = GUIDANCE_CONTRACT_V1


def _get(args, name, default):
    return getattr(args, name, default)


def validate_selma_config(args):
    """Validate the SELMA configuration surface. Returns a SelmaRuntimeConfig.

    Raises :class:`SelmaConfigError` on any illegal combination. Pure: reads
    ``args`` and nothing else, so it is testable without an environment.
    """
    mode = _get(args, "selma_mode", MODE_OFF)
    if not isinstance(mode, str) or mode not in ALLOWED_MODES:
        raise SelmaConfigError(
            "selma_mode={!r} is not a known SELMA mode. Legal modes: {}. "
            "Unknown modes are rejected rather than treated as {!r}.".format(
                mode, ", ".join(ALLOWED_MODES), MODE_OFF
            )
        )

    provider = str(_get(args, "selma_provider", PROVIDER_NONE) or PROVIDER_NONE)
    variant_id = str(_get(args, "selma_variant_id", "") or "")
    guidance_log_csv = str(_get(args, "selma_guidance_log_csv", "") or "")
    decision_log_csv = str(_get(args, "selma_decision_log_csv", "") or "")
    guidance_contract = str(
        _get(args, "selma_guidance_contract", GUIDANCE_CONTRACT_V1)
        or GUIDANCE_CONTRACT_V1
    )
    if guidance_contract not in ALLOWED_GUIDANCE_CONTRACTS:
        raise SelmaConfigError(
            "selma_guidance_contract={!r} is not known. Legal contracts: {}."
            .format(guidance_contract, ", ".join(ALLOWED_GUIDANCE_CONTRACTS))
        )
    raw_ttl = _get(args, "selma_ttl_steps", 15)
    try:
        ttl_steps = int(raw_ttl)
    except (TypeError, ValueError):
        raise SelmaConfigError(
            "selma_ttl_steps={!r} is not an integer.".format(raw_ttl))
    if ttl_steps < 1:
        raise SelmaConfigError(
            "selma_ttl_steps={!r} must be >= 1: a guidance that serves no "
            "transition could never influence an action.".format(raw_ttl))
    if ttl_steps != 15 and mode != MODE_ACTIVE:
        raise SelmaConfigError(
            "selma_ttl_steps={!r} requires selma_mode={!r}; got {!r}. Without "
            "guidance there is no advice to give a lifetime to."
            .format(ttl_steps, MODE_ACTIVE, mode))

    if mode != MODE_ACTIVE and guidance_contract != GUIDANCE_CONTRACT_V1:
        raise SelmaConfigError(
            "selma_guidance_contract={!r} requires selma_mode={!r}; got {!r}."
            .format(guidance_contract, MODE_ACTIVE, mode)
        )

    raw_beta = _get(args, "selma_beta", 0.0)
    try:
        beta = float(raw_beta)
    except (TypeError, ValueError):
        raise SelmaConfigError(
            "selma_beta={!r} is not a number.".format(raw_beta)
        )
    if not math.isfinite(beta):
        raise SelmaConfigError(
            "selma_beta={!r} is not finite. A non-finite shaping scale would "
            "poison every stored reward.".format(raw_beta)
        )

    if mode in (MODE_OFF, MODE_SHADOW):
        if provider != PROVIDER_NONE:
            raise SelmaConfigError(
                "selma_provider={!r} requires selma_mode={!r}; got selma_mode={!r}. "
                "In {!r} mode the detector is observation-only and must issue zero "
                "provider calls.".format(provider, MODE_ACTIVE, mode, mode)
            )
        if beta != 0.0:
            raise SelmaConfigError(
                "selma_beta={!r} requires selma_mode={!r}; got selma_mode={!r}. "
                "Shaping cannot exist without guidance.".format(beta, MODE_ACTIVE, mode)
            )
        if guidance_log_csv:
            raise SelmaConfigError(
                "selma_guidance_log_csv is set but selma_mode={!r} produces no "
                "guidance transitions to log.".format(mode)
            )
        if decision_log_csv:
            raise SelmaConfigError(
                "selma_decision_log_csv is set but selma_mode={!r} produces no "
                "guidance decisions to log.".format(mode)
            )

    if mode == MODE_OFF and variant_id:
        raise SelmaConfigError(
            "selma_variant_id={!r} is set but selma_mode={!r} constructs no "
            "detector.".format(variant_id, MODE_OFF)
        )

    if mode == MODE_ACTIVE:
        if provider == PROVIDER_NONE:
            raise SelmaConfigError(
                "selma_mode={!r} requires an explicitly named selma_provider; got "
                "{!r}. Active mode never picks a provider for you.".format(
                    MODE_ACTIVE, PROVIDER_NONE
                )
            )
        # Reject an unsupported provider here, before an environment is built.
        from selma.provider_registry import SUPPORTED_PROVIDERS

        if provider not in SUPPORTED_PROVIDERS:
            raise SelmaConfigError(
                "selma_provider={!r} is not supported in this build. Available: {}."
                .format(provider, ", ".join(sorted(SUPPORTED_PROVIDERS)))
            )
        if not variant_id:
            raise SelmaConfigError(
                "selma_mode={!r} requires an explicit selma_variant_id. Without it "
                "the whole 48-variant bank would be offered to the provider, "
                "producing up to 48 queries and 47 replacements per struggle. "
                "The frozen variant is {!r}.".format(MODE_ACTIVE, FROZEN_ACTIVE_VARIANT)
            )
        if variant_id != FROZEN_ACTIVE_VARIANT:
            raise SelmaConfigError(
                "selma_variant_id={!r} is not the frozen Stage B variant. Active "
                "mode may only use {!r}.".format(variant_id, FROZEN_ACTIVE_VARIANT)
            )
        if not bool(_get(args, "common_reward", True)):
            raise SelmaConfigError(
                "selma_mode={!r} requires common_reward=True. The Stage C potential "
                "is aggregated to a single scalar team term (contract 4.3); there is "
                "no per-agent shaping and none will be invented.".format(MODE_ACTIVE)
            )

    cache_mode, cache_path, allow_shared, live_call_cap = _validate_stage_d(args, mode)

    from selma.pricing import PriceTableError, price_table_from_args

    price_table = price_table_from_args(args, _get)
    try:
        cost_cap = float(_get(args, "selma_cost_cap_usd", 0.0) or 0.0)
    except (TypeError, ValueError):
        raise SelmaConfigError("selma_cost_cap_usd must be a number.")
    if cost_cap < 0.0:
        raise SelmaConfigError(
            "selma_cost_cap_usd={!r} must be >= 0. Use 0 for no monetary "
            "ceiling; there is no unlimited-but-priced setting.".format(cost_cap))
    if cost_cap > 0.0 and not price_table.usable:
        raise SelmaConfigError(
            "selma_cost_cap_usd={} was set without a dated price table. Supply "
            "selma_price_table_date and the per-million-token rates: no price "
            "is hard-coded in this build, so a ceiling without them would "
            "silently permit everything.".format(cost_cap))
    if cost_cap > 0.0 and mode != MODE_ACTIVE:
        raise SelmaConfigError(
            "selma_cost_cap_usd requires selma_mode={!r}; got {!r}."
            .format(MODE_ACTIVE, mode))

    return SelmaRuntimeConfig(
        mode=mode,
        provider=provider,
        beta=beta,
        variant_id=variant_id,
        detector_enabled=mode in (MODE_SHADOW, MODE_ACTIVE),
        guidance_enabled=mode == MODE_ACTIVE,
        guidance_log_csv=guidance_log_csv,
        decision_log_csv=decision_log_csv,
        cache_mode=cache_mode,
        cache_path=cache_path,
        cache_allow_shared_write=allow_shared,
        live_call_cap=live_call_cap,
        provider_timeout_s=float(_get(args, "selma_provider_timeout_s", 30.0)),
        provider_max_retries=int(_get(args, "selma_provider_max_retries", 1)),
        provider_backoff_s=float(_get(args, "selma_provider_backoff_s", 0.5)),
        provider_max_output_tokens=int(
            _get(args, "selma_provider_max_output_tokens", 512)),
        price_table_date=str(_get(args, "selma_price_table_date", "") or ""),
        price_input_usd_per_mtok=price_table.input_usd_per_mtok,
        price_cached_input_usd_per_mtok=price_table.cached_input_usd_per_mtok,
        price_output_usd_per_mtok=price_table.output_usd_per_mtok,
        cost_cap_usd=cost_cap,
        guidance_contract=guidance_contract,
        ttl_steps=ttl_steps,
    )


def _validate_stage_d(args, mode):
    """Validate the Stage D cache / live-call-budget surface.

    Every rule here fails at construction time. A cache that is silently
    disabled, or a budget that is silently unlimited, would be an unrecoverable
    experimental error discovered only in the bill.
    """
    from selma.gateway import CACHE_MODES, CACHE_OFF

    cache_mode = str(_get(args, "selma_cache_mode", CACHE_OFF) or CACHE_OFF)
    if cache_mode not in CACHE_MODES:
        raise SelmaConfigError(
            "selma_cache_mode={!r} is not a known mode. Legal modes: {}."
            .format(cache_mode, ", ".join(CACHE_MODES))
        )
    cache_path = str(_get(args, "selma_cache_path", "") or "")
    allow_shared = bool(_get(args, "selma_cache_allow_shared_write", False))

    if cache_mode != CACHE_OFF:
        if mode != MODE_ACTIVE:
            raise SelmaConfigError(
                "selma_cache_mode={!r} requires selma_mode={!r}; got {!r}. "
                "Only active mode issues guidance requests to cache."
                .format(cache_mode, MODE_ACTIVE, mode)
            )
        if not cache_path:
            raise SelmaConfigError(
                "selma_cache_mode={!r} requires an explicit selma_cache_path. "
                "One database per condition/seed/run; nothing is defaulted, so "
                "two runs can never silently share a store.".format(cache_mode)
            )

    raw_cap = _get(args, "selma_live_call_cap", 0)
    try:
        live_call_cap = int(raw_cap)
    except (TypeError, ValueError):
        raise SelmaConfigError(
            "selma_live_call_cap={!r} is not an integer.".format(raw_cap))
    if live_call_cap < 0:
        raise SelmaConfigError(
            "selma_live_call_cap={!r} must be >= 0. Use 0 to forbid live calls "
            "entirely; there is no unlimited setting.".format(raw_cap))

    return cache_mode, cache_path, allow_shared, live_call_cap
