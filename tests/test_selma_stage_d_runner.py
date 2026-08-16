"""SELMA Stage D through the real EpisodeRunner.

Confirms the Stage D objects are constructed only where they should be, that
off/shadow/mock behaviour is unchanged, and that greedy evaluation is
structurally incapable of a lookup, a call or an activation.

No network, no live provider: the runner tests use the deterministic mock.

Run:  PYTHONPATH=src pytest tests/test_selma_stage_d_runner.py -v
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from selma.gateway import CACHE_OFF, CACHE_READ_WRITE, ProviderGateway  # noqa: E402
from selma.modes import FROZEN_ACTIVE_VARIANT  # noqa: E402
from test_selma_shadow_equivalence import build_args, build_stack  # noqa: E402

DEV_SEED = 4242
EPISODES = 40


def active_args(tmp_path, cache_mode=CACHE_OFF, beta=0.0, **extra):
    overrides = {
        "seed": DEV_SEED,
        "selma_mode": "active",
        "selma_provider": "deterministic_mock",
        "selma_variant_id": FROZEN_ACTIVE_VARIANT,
        "selma_beta": beta,
        "selma_cache_mode": cache_mode,
        "selma_cache_path": (str(tmp_path / "cache.sqlite3")
                             if cache_mode != CACHE_OFF else ""),
    }
    overrides.update(extra)
    return build_args(**overrides)


# ==========================================================================
# construction
# ==========================================================================


def test_off_mode_constructs_no_stage_d_objects():
    runner, _, _, _ = build_stack(build_args())
    assert runner.selma_accounting is None
    assert runner.selma_cache is None
    assert runner.selma_gateway is None


def test_shadow_mode_constructs_no_stage_d_objects(tmp_path):
    runner, _, _, _ = build_stack(build_args(
        selma_mode="shadow", selma_shadow_csv=str(tmp_path / "s.csv")))
    assert runner.selma_accounting is None
    assert runner.selma_gateway is None


def test_active_mode_wires_the_gateway_and_accounting(tmp_path):
    runner, _, _, _ = build_stack(active_args(tmp_path))
    assert isinstance(runner.selma_gateway, ProviderGateway)
    assert runner.selma_accounting is not None
    assert runner.selma_guidance.provider is runner.selma_gateway
    assert runner.selma_guidance.accounting is runner.selma_accounting
    # cache off by default: no database was created
    assert runner.selma_cache is None
    assert not os.path.exists(str(tmp_path / "cache.sqlite3"))


def test_offline_provider_never_consumes_the_live_call_budget(tmp_path):
    runner, _, _, _ = build_stack(active_args(tmp_path))
    assert runner.selma_gateway.live_call_cap is None
    assert runner.selma_accounting.live_provider is False


def test_cache_database_is_created_for_this_run(tmp_path):
    runner, _, _, _ = build_stack(active_args(tmp_path, cache_mode=CACHE_READ_WRITE))
    assert runner.selma_cache is not None
    assert os.path.exists(str(tmp_path / "cache.sqlite3"))
    assert "seed{}".format(DEV_SEED) in runner.selma_cache.meta("run_identity")


def test_no_openai_sdk_is_imported_by_a_mock_run(tmp_path):
    sys.modules.pop("openai", None)
    runner, _, _, _ = build_stack(active_args(tmp_path, cache_mode=CACHE_READ_WRITE))
    for _ in range(3):
        runner.run(test_mode=False)
    assert "openai" not in sys.modules


# ==========================================================================
# behaviour through real episodes
# ==========================================================================


@pytest.fixture(scope="module")
def cached_run(tmp_path_factory):
    path = tmp_path_factory.mktemp("stage_d")
    runner, _, _, _ = build_stack(active_args(path, cache_mode=CACHE_READ_WRITE,
                                              beta=0.25))
    for _ in range(EPISODES):
        runner.run(test_mode=False)
    return runner


def test_counters_reconcile_after_a_real_run(cached_run):
    accounting = cached_run.selma_accounting
    assert accounting.provider_request_considered_count > 0, "vacuous otherwise"
    accounting.assert_invariants(max_retries=0)
    assert accounting.cache_lookup_count == accounting.provider_request_considered_count
    assert (accounting.cache_hit_count + accounting.cache_miss_count
            == accounting.cache_lookup_count)
    assert accounting.live_api_call_count == 0          # the mock is offline
    assert accounting.guidance_activation_count == accounting.successful_response_count


def test_detector_events_bound_the_requests_considered(cached_run):
    accounting = cached_run.selma_accounting
    assert accounting.detector_event_count >= accounting.provider_request_considered_count


def test_cache_persisted_only_validated_responses(cached_run):
    assert cached_run.selma_cache.row_count() == cached_run.selma_accounting.cache_miss_count
    assert cached_run.selma_accounting.rejected_response_count == 0


def test_greedy_evaluation_produces_zero_lookups_calls_and_activations(cached_run):
    accounting = cached_run.selma_accounting
    before = accounting.snapshot()
    assert before["guidance_activation_count"] > 0, "vacuous unless training activated"
    rows_before = cached_run.selma_cache.row_count()

    for _ in range(6):
        cached_run.run(test_mode=True)

    after = accounting.snapshot()
    assert after == before, "greedy evaluation touched the Stage D ledger"
    assert after["cache_lookup_count"] == before["cache_lookup_count"]
    assert after["live_api_call_count"] == 0
    assert cached_run.selma_cache.row_count() == rows_before
    assert cached_run.selma_guidance.active is None
