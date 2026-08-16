"""SELMA Stage D: gateway, cache modes, live-call budget and the eleven counters.

Drives the real :class:`GuidanceController` through the real gateway, so the
ordering under test -- lookup, then call, then validate, then write -- is the
ordering that would run in training. Providers are fakes; nothing here touches
the network or runs an environment.

Run:  PYTHONPATH=src pytest tests/test_selma_stage_d_accounting.py -v
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from selma.accounting import (  # noqa: E402
    COUNTER_NAMES,
    FALLBACK_BUDGET_EXHAUSTED,
    FALLBACK_READ_ONLY_MISS,
    InvariantViolation,
    SelmaAccounting,
)
from selma.gateway import (  # noqa: E402
    CACHE_OFF,
    CACHE_READ_ONLY,
    CACHE_READ_WRITE,
    ProviderGateway,
)
from selma.guidance import GuidanceController  # noqa: E402
from selma.modes import FROZEN_ACTIVE_VARIANT, SelmaConfigError, validate_selma_config  # noqa: E402
from selma.openai_provider import MODEL_SNAPSHOT, OpenAIGuidanceProvider  # noqa: E402
from selma.provider import DeterministicMockProvider  # noqa: E402
from selma.cache import SqliteGuidanceCache  # noqa: E402
from test_selma_stage_d_cache import CONDITION, GRID, event_row, make_request  # noqa: E402
from test_selma_stage_d_provider import FakeClient, api_response, good_payload  # noqa: E402


def build_provider(request_count=8, **kwargs):
    """An OpenAI adapter wired to a fake client that always answers well."""
    request = make_request()
    script = [api_response(good_payload(request)) for _ in range(request_count)]
    kwargs.setdefault("backoff_s", 0.0)
    kwargs.setdefault("sleep", lambda _s: None)
    return OpenAIGuidanceProvider(client=FakeClient(*script), **kwargs)


def build_gateway(provider, *, cache=None, cache_mode=CACHE_OFF, cap=None,
                  live_provider=True):
    accounting = SelmaAccounting(
        cache_mode=cache_mode, live_call_cap=cap,
        provider=getattr(provider, "provider_id", "mock"),
        requested_model=getattr(provider, "model", ""),
        prompt_version=getattr(provider, "prompt_version", ""),
        response_schema_version=getattr(provider, "response_schema_version", ""),
        live_provider=live_provider,
    )
    gateway = ProviderGateway(
        provider, accounting=accounting, cache=cache, cache_mode=cache_mode,
        live_call_cap=cap, condition=CONDITION, grid_shape=GRID)
    return gateway, accounting


def controller(gateway, beta=1.0, gamma=0.99, accounting=None):
    return GuidanceController(
        provider=gateway, grid_shape=GRID, beta=beta, gamma=gamma, enabled=True,
        variant_id=FROZEN_ACTIVE_VARIANT, accounting=accounting)


def fire(ctrl, *, episode_id=1, positions=None, food_pos=(4, 4),
         cells=((4, 3), (3, 4)), stored_terminated=False):
    """Drive one transition that carries exactly one eligible detector event."""
    positions = positions or {0: (4, 3), 1: (6, 6)}
    row = event_row(episode_id=episode_id, food_pos=food_pos, cells=cells)
    return ctrl.apply_transition(
        step=0, raw_team_reward=0.0, phi_before=0.0,
        post_agent_positions=positions, post_food_positions=(food_pos,),
        event_rows=[row], episode_ended=False,
        stored_terminated=stored_terminated, terminal_cause="")


def cache_at(tmp_path, name="c.sqlite3", **kwargs):
    kwargs.setdefault("run_identity", "cond|seed4242|openai|" + MODEL_SNAPSHOT)
    return SqliteGuidanceCache(str(tmp_path / name), **kwargs)


# ==========================================================================
# 1. cache modes
# ==========================================================================


def test_cache_off_performs_no_lookup_and_no_write(tmp_path):
    gateway, acc = build_gateway(build_provider(), cache=cache_at(tmp_path),
                                 cache_mode=CACHE_OFF)
    ctrl = controller(gateway, accounting=acc)
    fire(ctrl)
    assert acc.cache_lookup_count == 0
    assert acc.cache_hit_count == 0 and acc.cache_miss_count == 0
    assert acc.live_api_call_count == 1
    assert gateway.cache.row_count() == 0
    acc.assert_invariants(max_retries=1)


def test_read_write_miss_calls_then_writes(tmp_path):
    cache = cache_at(tmp_path)
    gateway, acc = build_gateway(build_provider(), cache=cache,
                                 cache_mode=CACHE_READ_WRITE)
    ctrl = controller(gateway, accounting=acc)
    fire(ctrl)
    assert (acc.cache_lookup_count, acc.cache_hit_count, acc.cache_miss_count) == (1, 0, 1)
    assert acc.live_api_call_count == 1
    assert acc.successful_response_count == 1
    assert cache.row_count() == 1
    acc.assert_invariants(max_retries=1)


def test_read_write_hit_from_another_episode_makes_no_call(tmp_path):
    """The scheme-B payoff: an identical situation in a different episode hits."""
    cache = cache_at(tmp_path)
    gateway, acc = build_gateway(build_provider(), cache=cache,
                                 cache_mode=CACHE_READ_WRITE)
    ctrl = controller(gateway, accounting=acc)
    fire(ctrl, episode_id=1)
    calls_after_first = acc.live_api_call_count
    ctrl.reset_episode()
    fire(ctrl, episode_id=987)

    assert acc.cache_hit_count == 1
    assert acc.live_api_call_count == calls_after_first     # a hit costs nothing
    assert acc.cache_lookup_count == 2
    assert acc.successful_response_count == 2
    assert acc.guidance_activation_count == 2
    assert cache.row_count() == 1
    acc.assert_invariants(max_retries=1)


def test_changed_absolute_position_is_a_miss(tmp_path):
    cache = cache_at(tmp_path)
    gateway, acc = build_gateway(build_provider(), cache=cache,
                                 cache_mode=CACHE_READ_WRITE)
    ctrl = controller(gateway, accounting=acc)
    fire(ctrl, positions={0: (4, 3), 1: (6, 6)})
    ctrl.reset_episode()
    fire(ctrl, episode_id=2, positions={0: (4, 3), 1: (5, 6)})
    assert acc.cache_hit_count == 0
    assert acc.cache_miss_count == 2
    assert acc.live_api_call_count == 2
    assert cache.row_count() == 2


def test_read_only_hit_serves_without_calling(tmp_path):
    warm = cache_at(tmp_path)
    gateway, acc = build_gateway(build_provider(), cache=warm,
                                 cache_mode=CACHE_READ_WRITE)
    ctrl = controller(gateway, accounting=acc)
    fire(ctrl, episode_id=1)
    warm.close()

    read_only = cache_at(tmp_path, read_only=True)
    gateway2, acc2 = build_gateway(build_provider(), cache=read_only,
                                   cache_mode=CACHE_READ_ONLY)
    ctrl2 = controller(gateway2, accounting=acc2)
    fire(ctrl2, episode_id=55)
    assert acc2.cache_hit_count == 1
    assert acc2.live_api_call_count == 0
    assert ctrl2.active is not None
    acc2.assert_invariants(max_retries=1)


def test_read_only_miss_never_calls_and_yields_no_guidance(tmp_path):
    cache_at(tmp_path).close()
    read_only = cache_at(tmp_path, read_only=True)
    gateway, acc = build_gateway(build_provider(), cache=read_only,
                                 cache_mode=CACHE_READ_ONLY)
    ctrl = controller(gateway, accounting=acc)
    record = fire(ctrl)
    assert acc.cache_miss_count == 1
    assert acc.live_api_call_count == 0
    assert acc.fallback_reasons.get(FALLBACK_READ_ONLY_MISS) == 1
    assert ctrl.active is None
    assert record.shaping_contribution == 0.0
    acc.assert_invariants(max_retries=1)


def test_cache_hits_are_revalidated_against_the_current_request(tmp_path):
    """A stored response is offered to the validator again, never trusted."""
    cache = cache_at(tmp_path)
    gateway, acc = build_gateway(build_provider(), cache=cache,
                                 cache_mode=CACHE_READ_WRITE)
    ctrl = controller(gateway, accounting=acc)
    fire(ctrl, episode_id=1)

    seen = []
    import selma.provider as provider_module
    real_validate = provider_module.validate

    def spy(request, response):
        verdict = real_validate(request, response)
        seen.append(verdict.valid)
        return verdict

    provider_module.validate = spy
    try:
        ctrl.reset_episode()
        fire(ctrl, episode_id=2)
    finally:
        provider_module.validate = real_validate
    assert acc.cache_hit_count == 1
    assert seen == [True]          # the hit went through the validator


def test_rejected_responses_are_never_cached(tmp_path):
    cache = cache_at(tmp_path)
    provider = OpenAIGuidanceProvider(
        client=FakeClient(api_response("not json")), backoff_s=0.0,
        sleep=lambda _s: None, max_retries=0)
    gateway, acc = build_gateway(provider, cache=cache, cache_mode=CACHE_READ_WRITE)
    ctrl = controller(gateway, accounting=acc)
    record = fire(ctrl)
    assert cache.row_count() == 0
    assert acc.successful_response_count == 0
    assert acc.fallback_count == 1
    assert record.shaping_contribution == 0.0
    acc.assert_invariants(max_retries=0)


def test_stale_hit_is_treated_as_a_miss_and_logged(tmp_path):
    cache = cache_at(tmp_path)
    gateway, acc = build_gateway(build_provider(), cache=cache,
                                 cache_mode=CACHE_READ_WRITE)
    ctrl = controller(gateway, accounting=acc)
    fire(ctrl, episode_id=1)

    # Corrupt the stored payload so rebuilding it fails.
    import sqlite3
    cache.close()
    raw = sqlite3.connect(str(tmp_path / "c.sqlite3"))
    with raw:
        raw.execute("UPDATE guidance_cache SET response_json = ?",
                    ('{"schema_version":"x"}',))
    raw.close()

    reopened = cache_at(tmp_path)
    gateway2, acc2 = build_gateway(build_provider(), cache=reopened,
                                   cache_mode=CACHE_READ_WRITE)
    ctrl2 = controller(gateway2, accounting=acc2)
    fire(ctrl2, episode_id=2)
    assert acc2.cache_hit_count == 0
    assert acc2.cache_miss_count == 1
    assert acc2.stale_reasons              # the reason was recorded
    acc2.assert_invariants(max_retries=1)


# ==========================================================================
# 2. the live-call budget
# ==========================================================================


def test_live_call_cap_is_never_exceeded(tmp_path):
    gateway, acc = build_gateway(build_provider(request_count=10), cap=2)
    ctrl = controller(gateway, accounting=acc)
    for episode in range(5):
        ctrl.reset_episode()
        fire(ctrl, episode_id=episode, positions={0: (4, 3), 1: (6, 6 - episode)})
    assert acc.live_api_call_count == 2
    assert acc.budget_exhausted_count == 3
    acc.assert_invariants(max_retries=1)


def test_budget_exhausted_miss_makes_no_call_and_no_guidance():
    gateway, acc = build_gateway(build_provider(), cap=0)
    ctrl = controller(gateway, accounting=acc)
    record = fire(ctrl)
    assert acc.live_api_call_count == 0
    assert acc.fallback_reasons.get(FALLBACK_BUDGET_EXHAUSTED) == 1
    assert ctrl.active is None
    assert record.shaping_contribution == 0.0
    assert record.phi_after_effective == 0.0


def test_budget_exhaustion_still_performs_the_cache_lookup(tmp_path):
    """Exhaustion stops calls, not lookups: a warm entry must still be served."""
    cache = cache_at(tmp_path)
    gateway, acc = build_gateway(build_provider(), cache=cache,
                                 cache_mode=CACHE_READ_WRITE, cap=1)
    ctrl = controller(gateway, accounting=acc)
    fire(ctrl, episode_id=1)                       # miss -> the only live call
    assert acc.live_api_call_count == 1
    ctrl.reset_episode()
    fire(ctrl, episode_id=2)                       # identical situation -> hit
    assert acc.cache_hit_count == 1
    assert acc.live_api_call_count == 1
    assert ctrl.active is not None
    ctrl.reset_episode()
    fire(ctrl, episode_id=3, positions={0: (4, 3), 1: (0, 0)})   # new -> exhausted
    assert acc.cache_lookup_count == 3
    assert acc.live_api_call_count == 1
    assert acc.budget_exhausted_count == 1


def test_every_attempt_including_retries_consumes_the_cap():
    """The declared rule: each HTTP attempt costs one unit of budget."""
    from test_selma_stage_d_provider import FakeTimeout

    request = make_request()
    provider = OpenAIGuidanceProvider(
        client=FakeClient(FakeTimeout(), api_response(good_payload(request))),
        max_retries=1, backoff_s=0.0, sleep=lambda _s: None)
    gateway, acc = build_gateway(provider, cap=5)
    ctrl = controller(gateway, accounting=acc)
    fire(ctrl)
    assert acc.live_api_call_count == 2      # first attempt + one retry
    assert acc.retry_count == 1
    acc.assert_invariants(max_retries=1)


def test_cap_stops_a_retry_mid_flight():
    from test_selma_stage_d_provider import FakeTimeout

    request = make_request()
    provider = OpenAIGuidanceProvider(
        client=FakeClient(FakeTimeout(), api_response(good_payload(request))),
        max_retries=1, backoff_s=0.0, sleep=lambda _s: None)
    gateway, acc = build_gateway(provider, cap=1)
    ctrl = controller(gateway, accounting=acc)
    fire(ctrl)
    assert acc.live_api_call_count == 1      # the retry was denied by the cap
    assert acc.retry_count == 0
    assert ctrl.active is None
    assert acc.fallback_reasons.get(FALLBACK_BUDGET_EXHAUSTED) == 1


def test_cache_hits_do_not_consume_the_cap(tmp_path):
    cache = cache_at(tmp_path)
    gateway, acc = build_gateway(build_provider(), cache=cache,
                                 cache_mode=CACHE_READ_WRITE, cap=1)
    ctrl = controller(gateway, accounting=acc)
    for episode in range(4):
        ctrl.reset_episode()
        fire(ctrl, episode_id=episode)
    assert acc.live_api_call_count == 1
    assert acc.cache_hit_count == 3
    assert acc.guidance_activation_count == 4


def test_fallback_never_substitutes_heuristic_guidance():
    gateway, acc = build_gateway(build_provider(), cap=0)
    ctrl = controller(gateway, beta=2.0, accounting=acc)
    record = fire(ctrl)
    assert ctrl.active is None
    assert record.guidance_active == 0
    assert record.training_reward == record.raw_team_reward
    assert record.shaping_contribution == 0.0


# ==========================================================================
# 3. accounting invariants
# ==========================================================================


def test_counter_set_is_exactly_the_eleven():
    assert len(COUNTER_NAMES) == 11
    acc = SelmaAccounting()
    for name in COUNTER_NAMES:
        assert getattr(acc, name) == 0


def test_hits_plus_misses_equal_lookups(tmp_path):
    cache = cache_at(tmp_path)
    gateway, acc = build_gateway(build_provider(request_count=10), cache=cache,
                                 cache_mode=CACHE_READ_WRITE)
    ctrl = controller(gateway, accounting=acc)
    for episode in range(6):
        ctrl.reset_episode()
        fire(ctrl, episode_id=episode, positions={0: (4, 3), 1: (6, episode)})
    assert acc.cache_hit_count + acc.cache_miss_count == acc.cache_lookup_count
    assert acc.cache_lookup_count == acc.provider_request_considered_count
    acc.assert_invariants(max_retries=1)


def test_activations_never_exceed_successful_responses(tmp_path):
    cache = cache_at(tmp_path)
    gateway, acc = build_gateway(build_provider(request_count=6), cache=cache,
                                 cache_mode=CACHE_READ_WRITE)
    ctrl = controller(gateway, accounting=acc)
    for episode in range(3):
        ctrl.reset_episode()
        fire(ctrl, episode_id=episode)
    assert acc.guidance_activation_count <= acc.successful_response_count
    acc.assert_invariants(max_retries=1)


def test_successful_plus_fallback_equals_considered():
    gateway, acc = build_gateway(build_provider(request_count=2), cap=1)
    ctrl = controller(gateway, accounting=acc)
    for episode in range(3):
        ctrl.reset_episode()
        fire(ctrl, episode_id=episode, positions={0: (4, 3), 1: (6, episode)})
    assert (acc.successful_response_count + acc.fallback_count
            == acc.provider_request_considered_count)


def test_invariant_checker_detects_a_broken_ledger():
    acc = SelmaAccounting(cache_mode=CACHE_READ_WRITE)
    acc.provider_request_considered_count = 3
    acc.cache_lookup_count = 3
    acc.cache_hit_count = 1
    acc.cache_miss_count = 1              # deliberately wrong: 1 + 1 != 3
    problems = acc.check_invariants()
    assert any(p.startswith("I3") for p in problems)
    with pytest.raises(InvariantViolation):
        acc.assert_invariants()


def test_invariant_checker_detects_a_cap_breach():
    acc = SelmaAccounting(cache_mode=CACHE_OFF, live_call_cap=2)
    acc.live_api_call_count = 3
    assert any(p.startswith("I10") for p in acc.check_invariants())


def test_detector_events_are_never_fewer_than_considered_requests():
    gateway, acc = build_gateway(build_provider(request_count=4))
    ctrl = controller(gateway, accounting=acc)
    for episode in range(3):
        ctrl.reset_episode()
        fire(ctrl, episode_id=episode, positions={0: (4, 3), 1: (6, episode)})
    assert acc.detector_event_count >= acc.provider_request_considered_count
    acc.assert_invariants(max_retries=1)


def test_terminal_events_are_counted_but_never_considered():
    gateway, acc = build_gateway(build_provider())
    ctrl = controller(gateway, accounting=acc)
    ctrl.apply_transition(
        step=0, raw_team_reward=0.0, phi_before=0.0,
        post_agent_positions={0: (4, 3), 1: (6, 6)}, post_food_positions=((4, 4),),
        event_rows=[event_row()], episode_ended=True, stored_terminated=True,
        terminal_cause="true_terminal")
    assert acc.detector_event_count == 1
    assert acc.provider_request_considered_count == 0
    assert acc.live_api_call_count == 0
    assert acc.terminal_suppressed_event_count == 1
    acc.assert_invariants(max_retries=1)


def test_usage_and_provenance_are_recorded(tmp_path):
    gateway, acc = build_gateway(build_provider())
    ctrl = controller(gateway, accounting=acc)
    fire(ctrl)
    data = acc.as_dict()
    assert data["provider"] == "openai"
    assert data["requested_model"] == MODEL_SNAPSHOT
    assert data["returned_models"] == {MODEL_SNAPSHOT: 1}
    assert data["input_tokens"] == 120 and data["output_tokens"] == 18
    assert data["cached_input_tokens"] == 64
    assert data["total_latency_s"] >= 0.0
    assert data["cache_key_version"] and data["cache_schema_version"]
    assert acc.request_ids == ["resp_abc123"]
    # no monetary total is asserted anywhere; only tokens and a price-table date
    assert "cost_usd" not in data


# ==========================================================================
# 4. configuration surface
# ==========================================================================


def args_ns(**overrides):
    from types import SimpleNamespace

    base = {
        "selma_mode": "active", "selma_provider": "deterministic_mock",
        "selma_beta": 0.0, "selma_variant_id": FROZEN_ACTIVE_VARIANT,
        "selma_guidance_log_csv": "", "common_reward": True, "gamma": 0.99,
        "selma_cache_mode": "off", "selma_cache_path": "",
        "selma_live_call_cap": 2000,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_openai_provider_is_accepted_by_the_mode_contract():
    config = validate_selma_config(args_ns(selma_provider="openai"))
    assert config.provider == "openai"


def test_cache_requires_active_mode():
    with pytest.raises(SelmaConfigError):
        validate_selma_config(args_ns(
            selma_mode="shadow", selma_provider="none",
            selma_cache_mode="read_write", selma_cache_path="/x.sqlite3"))


def test_cache_requires_an_explicit_path():
    with pytest.raises(SelmaConfigError) as error:
        validate_selma_config(args_ns(selma_cache_mode="read_write"))
    assert "selma_cache_path" in str(error.value)


def test_unknown_cache_mode_is_rejected():
    with pytest.raises(SelmaConfigError):
        validate_selma_config(args_ns(selma_cache_mode="write_only"))


def test_negative_live_call_cap_is_rejected():
    with pytest.raises(SelmaConfigError):
        validate_selma_config(args_ns(selma_live_call_cap=-1))


def test_defaults_keep_stage_d_inert():
    import yaml

    config = yaml.safe_load(open(os.path.join(
        os.path.dirname(__file__), "..", "src", "config", "default.yaml")))
    assert config["selma_cache_mode"] == "off"
    assert config["selma_cache_path"] == ""
    assert config["selma_provider"] == "none"
    assert config["selma_live_call_cap"] == 2000
    assert config["selma_cache_allow_shared_write"] is False


# ==========================================================================
# 5. regressions: the deterministic mock and off/shadow are untouched
# ==========================================================================


def test_deterministic_mock_through_the_gateway_makes_no_live_calls():
    gateway, acc = build_gateway(DeterministicMockProvider(), live_provider=False)
    ctrl = controller(gateway, accounting=acc)
    record = fire(ctrl)
    assert acc.live_api_call_count == 0
    assert acc.successful_response_count == 1
    assert ctrl.active is not None
    assert record.guidance_active == 1
    acc.assert_invariants(max_retries=0)


def test_mock_without_a_gateway_is_unchanged():
    """Stage C's controller/provider pairing still works with no Stage D object."""
    ctrl = GuidanceController(
        provider=DeterministicMockProvider(), grid_shape=GRID, beta=1.0,
        gamma=0.99, enabled=True, variant_id=FROZEN_ACTIVE_VARIANT)
    assert ctrl.accounting is None
    record = fire(ctrl)
    assert ctrl.active is not None and record.guidance_active == 1
