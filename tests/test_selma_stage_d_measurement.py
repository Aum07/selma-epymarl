"""SELMA Stage D: measurement instrumentation, pricing and the cost ceiling.

Every test here was written against the pre-instrumentation tree, where the
durable logs could not answer the questions the beta-zero measurement has to
answer: how many *distinct* questions were asked (no semantic digest was ever
recorded), how many events were skipped as zero-plan or singleton, how many
input tokens the provider reported as cached, or what the run cost.

**No test here touches the network**; ``tests/conftest.py`` enforces that for
the whole suite and every provider below is the deterministic mock or a local
fake.

Run:  PYTHONPATH=src pytest tests/test_selma_stage_d_measurement.py -v
"""

import csv
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from selma.accounting import (  # noqa: E402
    FALLBACK_BUDGET_EXHAUSTED,
    SKIP_NO_VALID_JOINT_PLAN,
    SKIP_SINGLETON_NO_CHOICE,
    SelmaAccounting,
)
from selma.cache import SqliteGuidanceCache  # noqa: E402
from selma.gateway import CACHE_READ_WRITE, ProviderGateway  # noqa: E402
from selma.guidance import (  # noqa: E402
    DECISION_MULTIPLE,
    DECISION_SINGLETON,
    DECISION_ZERO_PLAN,
    GuidanceController,
    GuidanceCsvWriter,
    SelmaDecisionCsvWriter,
)
from selma.modes import SelmaConfigError, validate_selma_config  # noqa: E402
from selma.pricing import (  # noqa: E402
    NO_PRICE_TABLE,
    CostBudget,
    PriceTable,
    PriceTableError,
    accounting_cost_usd,
)
from selma.provider import DeterministicMockProvider  # noqa: E402
from test_selma_stage_d_joint_plans import (  # noqa: E402
    CONDITION,
    GRID,
    POSITIONS,
    event_row,
)

#: The dated table used by the 2026-08-17 beta-zero measurement. Held in the
#: test (and the measurement record), never in src/: no price is hard-coded.
GPT_5_4_NANO_2026_08_17 = PriceTable(
    date="2026-08-17",
    input_usd_per_mtok=0.20,
    cached_input_usd_per_mtok=0.02,
    output_usd_per_mtok=1.25,
)


def _stack(tmp_path, name="m", provider=None, cost_budget=None, cap=None):
    provider = provider or DeterministicMockProvider()
    accounting = SelmaAccounting(cache_mode=CACHE_READ_WRITE, live_call_cap=cap,
                                 provider=getattr(provider, "provider_id", "x"),
                                 live_provider=False)
    cache = SqliteGuidanceCache(str(tmp_path / (name + ".sqlite3")),
                                run_identity=name)
    gateway = ProviderGateway(provider, accounting=accounting, cache=cache,
                              cache_mode=CACHE_READ_WRITE, live_call_cap=cap,
                              condition=CONDITION, grid_shape=GRID,
                              cost_budget=cost_budget)
    controller = GuidanceController(provider=gateway, grid_shape=GRID, beta=0.0,
                                    gamma=0.99, enabled=True,
                                    variant_id=event_row()["variant_id"],
                                    accounting=accounting)
    return provider, accounting, cache, gateway, controller


def _apply(controller, row, positions=None, raw=0.0, step=3):
    return controller.apply_transition(
        step=step, raw_team_reward=raw, phi_before=0.0,
        post_agent_positions=positions or POSITIONS,
        post_food_positions=[row["food_position"]], event_rows=[row],
        episode_ended=False, stored_terminated=False)


#: Genuinely different situations, so each is a distinct semantic key and
#: therefore a real cache miss. Varying only episode_id would produce hits --
#: scheme B is episode-independent by design -- and no call would be made at all.
_DISTINCT_FOODS = ((4, 4), (3, 3), (5, 5), (2, 4), (6, 2), (1, 5))


def _distinct_row(food, episode_id=1):
    row, col = food
    cells = ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1))
    return event_row(food_pos=food, cells=cells, episode_id=episode_id,
                     step=episode_id + 1)


ZERO_PLAN_ROW = dict(cells=((4, 3),))
SINGLETON_ROW = dict(recipients=(0,), cells=((4, 3),), loader_levels=(3,),
                     food_level=3)


# ==========================================================================
# 1. the semantic cache-key digest is durable, and survives budget exhaustion
# ==========================================================================


def test_the_gateway_exposes_the_semantic_key_digest(tmp_path):
    _p, _a, cache, gateway, controller = _stack(tmp_path, "digest")
    record = _apply(controller, event_row())
    (decision,) = record.decisions
    assert decision.decision == DECISION_MULTIPLE
    assert len(decision.key_digest) == 64          # sha256 hex
    assert decision.key_digest == gateway.last_key_digest
    assert decision.cache_result == "miss"
    cache.close()


def test_the_digest_is_episode_independent(tmp_path):
    """The same situation in a different episode is the same question."""
    _p, _a, cache, _g, controller = _stack(tmp_path, "sem")
    first = _apply(controller, event_row(episode_id=1, step=3))
    controller.reset_episode()
    second = _apply(controller, event_row(episode_id=99, step=41))
    assert first.decisions[0].key_digest == second.decisions[0].key_digest
    assert second.decisions[0].cache_result == "hit"
    cache.close()


def test_a_different_situation_has_a_different_digest(tmp_path):
    _p, _a, cache, _g, controller = _stack(tmp_path, "diff")
    first = _apply(controller, event_row())
    controller.reset_episode()
    second = _apply(controller, event_row(food_pos=(2, 2),
                                          cells=((1, 2), (3, 2), (2, 1), (2, 3))),
                    positions={0: (2, 1), 1: (6, 6)})
    assert first.decisions[0].key_digest != second.decisions[0].key_digest
    cache.close()


def test_the_digest_is_recorded_even_when_the_live_budget_is_exhausted(tmp_path):
    """Counting distinct questions must not depend on the cache being allowed
    to fill -- this is the whole point of recording it before the budget check."""
    from selma.openai_provider import OpenAIGuidanceProvider

    class NeverCalled(OpenAIGuidanceProvider):
        def _create(self, prompt, schema):        # pragma: no cover - must not run
            raise AssertionError("a call was made with no budget")

    provider = NeverCalled(client=object(), backoff_s=0.0, sleep=lambda _s: None)
    _p, accounting, cache, _g, controller = _stack(
        tmp_path, "exhausted", provider=provider, cap=0)
    accounting.live_provider = True

    record = _apply(controller, event_row())
    (decision,) = record.decisions

    assert decision.decision == DECISION_MULTIPLE
    assert len(decision.key_digest) == 64, "the question must still be counted"
    assert decision.cache_result == "miss"
    assert decision.provider_considered == 1
    assert decision.activated == 0
    assert decision.outcome == FALLBACK_BUDGET_EXHAUSTED
    assert accounting.live_api_call_count == 0
    assert record.shaping_contribution == 0.0
    assert controller.active is None
    cache.close()


def test_skipped_events_record_no_digest_and_no_lookup(tmp_path):
    """A zero-plan or singleton event never reaches the cache, so it has no
    semantic key -- an empty digest is the evidence, not a missing row."""
    _p, accounting, cache, _g, controller = _stack(tmp_path, "skips")
    for kwargs in (ZERO_PLAN_ROW, SINGLETON_ROW):
        controller.reset_episode()
        record = _apply(controller, event_row(**kwargs))
        (decision,) = record.decisions
        assert decision.key_digest == ""
        assert decision.cache_result == ""
        assert decision.provider_considered == 0
        assert decision.live_attempts == 0
    assert accounting.cache_lookup_count == 0
    cache.close()


# ==========================================================================
# 2. every measurement field required for the beta-zero run
# ==========================================================================


def test_each_decision_lands_in_exactly_one_bucket(tmp_path):
    _p, accounting, cache, _g, controller = _stack(tmp_path, "buckets")
    seen = []
    for kwargs in (ZERO_PLAN_ROW, SINGLETON_ROW, {}):
        controller.reset_episode()
        record = _apply(controller, event_row(**kwargs))
        seen.append(record.decisions[0].decision)
    assert seen == [DECISION_ZERO_PLAN, DECISION_SINGLETON, DECISION_MULTIPLE]
    assert accounting.zero_plan_skip_count == 1
    assert accounting.singleton_skip_count == 1
    assert accounting.multiple_plan_decision_count == 1
    assert accounting.check_invariants() == []
    cache.close()


def test_the_decision_record_carries_the_candidate_plan_count(tmp_path):
    _p, _a, cache, _g, controller = _stack(tmp_path, "count")
    record = _apply(controller, event_row())
    decision = record.decisions[0]
    assert decision.candidate_plan_count == 12      # P(4,2)
    assert decision.recipient_count == 2
    assert record.detector_events == 1
    cache.close()


def test_the_decision_record_carries_the_validator_verdict(tmp_path):
    _p, _a, cache, _g, controller = _stack(tmp_path, "verdict")
    record = _apply(controller, event_row())
    decision = record.decisions[0]
    assert decision.validator_valid == 1
    assert decision.validator_reason == ""
    assert decision.physically_valid == 1
    assert decision.activated == 1
    assert decision.selected_plan_id == "plan_0"
    assert decision.outcome == "activated"
    cache.close()


def test_skips_record_their_frozen_policy_reason(tmp_path):
    _p, _a, cache, _g, controller = _stack(tmp_path, "reasons")
    outcomes = []
    for kwargs in (ZERO_PLAN_ROW, SINGLETON_ROW):
        controller.reset_episode()
        outcomes.append(_apply(controller, event_row(**kwargs)).decisions[0].outcome)
    assert outcomes == [SKIP_NO_VALID_JOINT_PLAN, SKIP_SINGLETON_NO_CHOICE]
    cache.close()


def test_cached_input_tokens_are_recorded_separately(tmp_path):
    """Provider-side cached input is a distinct billed quantity."""
    provider = _ExpensiveProvider(input_tokens=900, cached_tokens=512)
    _p, accounting, cache, _g, controller = _stack(
        tmp_path, "tokens", provider=provider)
    record = _apply(controller, event_row())
    decision = record.decisions[0]
    assert decision.tokens_input == 900
    assert decision.tokens_cached_input == 512
    assert decision.tokens_output == 50
    assert decision.latency_s == 1.0
    assert decision.returned_model == "gpt-5.4-nano-2026-03-17"
    # and they reach the ledger as separate quantities
    assert accounting.input_tokens == 900
    assert accounting.cached_input_tokens == 512
    cache.close()


# ==========================================================================
# 3. the durable decision log
# ==========================================================================


def _read(path):
    with open(path) as handle:
        return list(csv.DictReader(handle))


def test_the_decision_log_writes_one_row_per_event(tmp_path):
    _p, _a, cache, _g, controller = _stack(tmp_path, "log")
    writer = SelmaDecisionCsvWriter(str(tmp_path / "decisions.csv"))
    for i, kwargs in enumerate((ZERO_PLAN_ROW, SINGLETON_ROW, {})):
        controller.reset_episode()
        record = _apply(controller, event_row(**kwargs))
        writer.write_rows(record, t_env=100 * i, condition=CONDITION, seed=91,
                          episode_id=i)
    rows = _read(str(tmp_path / "decisions.csv"))
    assert len(rows) == 3
    assert [r["decision"] for r in rows] == [
        DECISION_ZERO_PLAN, DECISION_SINGLETON, DECISION_MULTIPLE]
    assert rows[2]["key_digest"] and rows[0]["key_digest"] == ""
    assert all(r["greedy_eval"] == "0" for r in rows)
    assert all(r["beta"] == "0.0" for r in rows)
    assert all(r["seed"] == "91" for r in rows)
    cache.close()


def test_the_decision_log_header_covers_every_required_measurement_field():
    required = {
        "t_env", "episode_id", "episode_step",          # 23
        "decision", "candidate_plan_count",             # 2-5
        "key_digest", "cache_result",                   # 6-7
        "provider_considered", "live_attempts", "retries",   # 8-9
        "outcome", "selected_plan_id", "validator_valid",    # 10-12
        "physically_valid", "activated",                # 12-13
        "tokens_input", "tokens_cached_input", "tokens_output",  # 14
        "returned_model", "latency_s",                  # 15-16
        "greedy_eval", "beta",                          # 22
    }
    assert required <= set(SelmaDecisionCsvWriter.COLUMNS)


def test_the_guidance_log_gained_the_missing_measurement_fields():
    for field in ("query_tokens_cached_prompt", "query_returned_model",
                  "detector_events", "decisions", "zero_plan_decisions",
                  "singleton_decisions", "multiple_plan_decisions",
                  "greedy_eval"):
        assert field in GuidanceCsvWriter.COLUMNS, field


def test_a_row_that_does_not_match_the_columns_is_refused(tmp_path):
    writer = SelmaDecisionCsvWriter(str(tmp_path / "d.csv"))
    with pytest.raises(ValueError) as error:
        writer._append({"t_env": 1})
    assert "missing" in str(error.value)


def test_appending_the_decision_log_under_a_stale_header_is_refused(tmp_path):
    """The stale-header defect must remain impossible for every durable log."""
    _p, _a, cache, _g, controller = _stack(tmp_path, "stale")
    record = _apply(controller, event_row())
    path = str(tmp_path / "stale.csv")
    stale = [c for c in SelmaDecisionCsvWriter.COLUMNS if c != "key_digest"]
    with open(path, "w") as handle:
        handle.write(",".join(stale) + "\n")
    writer = SelmaDecisionCsvWriter(path)
    with pytest.raises(ValueError) as error:
        writer.write_rows(record, t_env=1, condition=CONDITION, seed=91,
                          episode_id=1)
    assert "key_digest" in str(error.value)
    cache.close()


def test_the_decision_log_never_contains_a_prompt_or_a_response_body(tmp_path):
    _p, _a, cache, _g, controller = _stack(tmp_path, "clean")
    path = str(tmp_path / "clean.csv")
    writer = SelmaDecisionCsvWriter(path)
    record = _apply(controller, event_row())
    writer.write_rows(record, t_env=1, condition=CONDITION, seed=91, episode_id=1)
    text = open(path).read().lower()
    for forbidden in ("instructions", "candidate_joint_plans", "system",
                      "authorization", "bearer", "sk-", "api_key", "loading_cell"):
        assert forbidden not in text, forbidden
    cache.close()


# ==========================================================================
# 4. pricing: dated, never hard-coded
# ==========================================================================


def test_no_price_is_hard_coded_in_the_source_tree():
    """A rate baked into src/ becomes stale evidence silently."""
    import selma.pricing as pricing

    assert pricing.NO_PRICE_TABLE.usable is False
    source = open(pricing.__file__).read()
    for rate in ("0.20", "0.02", "1.25"):
        assert rate not in source, "a live rate leaked into src/: " + rate


def test_cost_is_none_without_a_dated_table():
    accounting = SelmaAccounting(cache_mode="off")
    accounting.input_tokens = 1000
    assert accounting_cost_usd(accounting, NO_PRICE_TABLE) is None
    assert accounting.cost_usd(NO_PRICE_TABLE) is None


def test_cost_uses_the_non_cached_remainder_of_the_input_tokens():
    """input_tokens INCLUDES the cached ones; billing them twice overstates."""
    table = GPT_5_4_NANO_2026_08_17
    cost = table.cost_usd(input_tokens=1000, cached_input_tokens=400,
                          output_tokens=200)
    expected = (600 * 0.20 + 400 * 0.02 + 200 * 1.25) / 1_000_000.0
    assert cost == pytest.approx(expected)


def test_cached_input_is_not_the_sqlite_response_cache(tmp_path):
    """A SQLite hit makes no request at all, so it bills nothing whatsoever."""
    _p, accounting, cache, _g, controller = _stack(tmp_path, "confuse")
    _apply(controller, event_row())
    controller.reset_episode()
    _apply(controller, event_row(episode_id=42, step=9))
    assert accounting.cache_hit_count == 1
    # the mock reports no tokens at all; a hit contributes none either way
    assert accounting.cached_input_tokens == 0
    assert accounting.input_tokens == 0
    cache.close()


def test_a_cost_ceiling_without_a_dated_table_is_a_configuration_error():
    with pytest.raises(PriceTableError):
        CostBudget(cap_usd=0.25, price_table=NO_PRICE_TABLE)


def test_the_projection_is_conservative():
    budget = CostBudget(cap_usd=0.25, price_table=GPT_5_4_NANO_2026_08_17,
                        max_output_tokens=512, floor_input_tokens=4000)
    worst = budget.worst_case_next_attempt_usd()
    expected = (4000 * 0.20 + 512 * 1.25) / 1_000_000.0
    assert worst == pytest.approx(expected)
    budget.note_usage(9000)
    assert budget.worst_case_next_attempt_usd() > worst, "must track the worst seen"


def test_the_ceiling_refuses_the_attempt_that_could_cross_it():
    budget = CostBudget(cap_usd=0.25, price_table=GPT_5_4_NANO_2026_08_17)
    assert budget.would_exceed(0.0) is False
    assert budget.would_exceed(0.20) is False
    assert budget.would_exceed(0.2499) is True


def test_no_ceiling_never_refuses():
    budget = CostBudget(cap_usd=None, price_table=NO_PRICE_TABLE)
    assert budget.would_exceed(1e9) is False


# ==========================================================================
# 5. the ceiling behaves exactly like the call cap at runtime
# ==========================================================================


class _ExpensiveProvider(DeterministicMockProvider):
    """Answers like the mock but reports live-provider usage and attempts."""

    provider_id = "openai"

    def __init__(self, input_tokens=100000, cached_tokens=0):
        super().__init__()
        from selma.openai_provider import ProviderUsageDetail

        self._detail = ProviderUsageDetail(
            tokens_prompt=input_tokens, tokens_completion=50, latency_s=1.0,
            cached_tokens_prompt=cached_tokens, tokens_reasoning=0,
            request_id="req_x", returned_model="gpt-5.4-nano-2026-03-17")
        self.attempt_hook = None
        self.last_error = ""
        self.last_attempts = 0
        self.last_request_id = "req_x"
        self.last_returned_model = "gpt-5.4-nano-2026-03-17"

    def query(self, request):
        if self.attempt_hook is not None and not self.attempt_hook(0):
            return None
        self.last_attempts = 1
        response = super().query(request)
        # after super(), which resets last_usage to "not applicable"
        self.last_usage = self._detail
        return response


def test_the_cost_ceiling_stops_calls_and_yields_the_safe_path(tmp_path):
    budget = CostBudget(cap_usd=0.02, price_table=GPT_5_4_NANO_2026_08_17,
                        max_output_tokens=512)
    provider = _ExpensiveProvider(input_tokens=50000)
    _p, accounting, cache, _g, controller = _stack(
        tmp_path, "ceiling", provider=provider, cost_budget=budget)
    accounting.live_provider = True

    for i, food in enumerate(_DISTINCT_FOODS):
        controller.reset_episode()
        record = _apply(controller, _distinct_row(food, episode_id=i))
        if record.decisions[0].outcome == FALLBACK_BUDGET_EXHAUSTED:
            assert record.shaping_contribution == 0.0
            assert record.decisions[0].activated == 0
            assert record.decisions[0].key_digest, "the question is still counted"
            break
    else:                                              # pragma: no cover
        pytest.fail("the cost ceiling never bound")

    spent = accounting.cost_usd(GPT_5_4_NANO_2026_08_17)
    assert spent <= 0.02
    assert accounting.cost_exhausted_count >= 1
    assert accounting.check_invariants() == []
    cache.close()


def test_the_call_cap_and_the_cost_ceiling_are_independent(tmp_path):
    """Either ceiling refusing is enough; neither masks the other."""
    provider = _ExpensiveProvider(input_tokens=10)
    _p, accounting, cache, _g, controller = _stack(
        tmp_path, "both", provider=provider, cap=1)
    accounting.live_provider = True
    for i, food in enumerate(_DISTINCT_FOODS[:3]):
        controller.reset_episode()
        _apply(controller, _distinct_row(food, episode_id=i))
    assert accounting.live_api_call_count <= 1
    assert accounting.cost_exhausted_count == 0
    cache.close()


# ==========================================================================
# 6. configuration surface
# ==========================================================================


def _args(**kw):
    from types import SimpleNamespace

    base = dict(selma_mode="active", selma_provider="deterministic_mock",
                selma_variant_id="wf15_m3_t5_ibon", selma_beta=0.0,
                common_reward=True)
    base.update(kw)
    return SimpleNamespace(**base)


def test_the_decision_log_is_active_mode_only():
    config = validate_selma_config(_args(selma_decision_log_csv="/tmp/d.csv"))
    assert config.decision_log_csv == "/tmp/d.csv"
    with pytest.raises(SelmaConfigError) as error:
        validate_selma_config(
            _args(selma_mode="shadow", selma_provider="none",
                  selma_decision_log_csv="/tmp/d.csv"))
    assert "selma_decision_log_csv" in str(error.value)


def test_a_cost_cap_requires_the_dated_rates():
    with pytest.raises(SelmaConfigError) as error:
        validate_selma_config(_args(selma_cost_cap_usd=0.25))
    assert "dated price table" in str(error.value)

    config = validate_selma_config(_args(
        selma_cost_cap_usd=0.25, selma_price_table_date="2026-08-17",
        selma_price_input_usd_per_mtok=0.20,
        selma_price_cached_input_usd_per_mtok=0.02,
        selma_price_output_usd_per_mtok=1.25))
    assert config.cost_cap_usd == 0.25
    assert config.price_table_date == "2026-08-17"


def test_a_negative_cost_cap_is_rejected():
    with pytest.raises(SelmaConfigError):
        validate_selma_config(_args(selma_cost_cap_usd=-1.0))


def test_stage_d_defaults_keep_pricing_inert():
    from types import SimpleNamespace

    config = validate_selma_config(SimpleNamespace(selma_mode="off"))
    assert config.cost_cap_usd == 0.0
    assert config.price_table_date == ""
    assert config.price_input_usd_per_mtok == 0.0
    assert config.decision_log_csv == ""


# ==========================================================================
# 7. the measurement config the beta-zero run will actually use
# ==========================================================================


def test_the_frozen_measurement_configuration_validates():
    config = validate_selma_config(_args(
        selma_provider="openai",
        selma_cache_mode="read_write",
        selma_cache_path="/tmp/measure.sqlite3",
        selma_live_call_cap=200,
        selma_cost_cap_usd=0.25,
        selma_price_table_date="2026-08-17",
        selma_price_input_usd_per_mtok=0.20,
        selma_price_cached_input_usd_per_mtok=0.02,
        selma_price_output_usd_per_mtok=1.25,
        selma_decision_log_csv="/tmp/decisions.csv",
        selma_guidance_log_csv="/tmp/guidance.csv",
        selma_provider_max_output_tokens=512,
        selma_provider_timeout_s=30.0,
        selma_provider_max_retries=1,
    ))
    assert config.mode == "active"
    assert config.provider == "openai"
    assert config.beta == 0.0
    assert config.live_call_cap == 200
    assert config.cost_cap_usd == 0.25
    assert config.cache_mode == "read_write"
    assert config.variant_id == "wf15_m3_t5_ibon"


def test_beta_above_zero_is_not_used_by_this_measurement():
    """Pinned deliberately: this phase must not select or trial a beta."""
    config = validate_selma_config(_args(selma_beta=0.0))
    assert config.beta == 0.0


def test_a_cache_hit_reports_zero_attempts_and_the_stored_model(tmp_path):
    """A hit must never inherit the previous live call's attempts or request id."""
    provider = _ExpensiveProvider(input_tokens=100)
    _p, accounting, cache, gateway, controller = _stack(
        tmp_path, "attrib", provider=provider)

    live = _apply(controller, event_row())
    assert live.decisions[0].cache_result == "miss"
    assert live.decisions[0].live_attempts == 1
    assert live.decisions[0].request_id == "req_x"

    controller.reset_episode()
    hit = _apply(controller, event_row(episode_id=1234, step=7))
    decision = hit.decisions[0]
    assert decision.cache_result == "hit"
    assert decision.live_attempts == 0, "a hit made no request"
    assert decision.retries == 0
    assert decision.request_id == ""
    assert decision.returned_model == "gpt-5.4-nano-2026-03-17"
    assert accounting.live_api_call_count == 1
    cache.close()


def test_retries_are_derived_from_the_attempt_count(tmp_path):
    from selma.openai_provider import FALLBACK_MALFORMED, OpenAIGuidanceProvider

    class Retrying(OpenAIGuidanceProvider):
        def _create(self, prompt, schema):
            raise ValueError("transport")

    provider = Retrying(client=object(), backoff_s=0.0, sleep=lambda _s: None,
                        max_retries=1)
    _p, accounting, cache, _g, controller = _stack(
        tmp_path, "retry", provider=provider, cap=10)
    accounting.live_provider = True
    record = _apply(controller, event_row())
    decision = record.decisions[0]
    assert decision.live_attempts == 2
    assert decision.retries == 1
    assert accounting.retry_count == 1
    assert decision.activated == 0
    assert record.shaping_contribution == 0.0
    assert FALLBACK_MALFORMED  # imported symbol is real
    cache.close()


# ==========================================================================
# 8. the secret cannot reach Sacred
# ==========================================================================


def test_sacred_host_info_captures_no_environment_variables(monkeypatch):
    """The API key lives only in the environment, so Sacred must not read it.

    Sacred *can* record environment variables, but only those named in
    ``SETTINGS.HOST_INFO.CAPTURED_ENV``. This build never adds to that list, so
    the captured ``ENV`` is empty and no credential can enter a Sacred run
    directory. A canary proves it rather than assuming it.
    """
    from sacred import SETTINGS
    from sacred.host_info import get_host_info

    canary = "sk-CANARY-must-never-be-captured-0123456789"
    monkeypatch.setenv("OPENAI_API_KEY", canary)

    assert list(SETTINGS["HOST_INFO"]["CAPTURED_ENV"]) == []
    info = get_host_info()
    assert info.get("ENV") == {}
    assert canary not in json.dumps(info, default=str)


def test_the_api_key_is_never_a_selma_config_key():
    """No SELMA config key may carry a credential; the env var is the only path."""
    from types import SimpleNamespace

    config = validate_selma_config(_args(selma_provider="openai"))
    for name, value in config._asdict().items():
        assert "key" not in name.lower() or "cache" in name.lower(), name
        assert "sk-" not in str(value)
    assert not hasattr(SimpleNamespace(), "openai_api_key")
