"""SELMA Stage D: the live-pilot orchestrator, driven entirely by fake transports.

**No test here touches the network.** ``tests/conftest.py`` enforces that for the
whole suite; every client below is a local fake injected in place of the
production client factory, so the orchestration -- attempt ledger, budget cap,
probe gating, cache replay, read-only behaviour, accounting -- is exercised
without a request ever leaving the process.

Run:  PYTHONPATH=src pytest tests/test_selma_stage_d_live_pilot.py -v
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))

import selma_stage_d_live_pilot as pilot  # noqa: E402
from selma.openai_provider import MODEL_SNAPSHOT  # noqa: E402

SENTINEL_KEY = "sk-live-pilot-sentinel-DO-NOT-LEAK-0123456789"


# ==========================================================================
# fake transport
# ==========================================================================


class ScriptedResponses:
    """Answers every request with one valid id per agent, taken from the enum.

    The ids come out of the request's own JSON Schema, so the fake behaves like a
    compliant structured-output implementation and the pilot's grounding check is
    exercised on real data rather than on a hard-coded answer.
    """

    def __init__(self, *, model=MODEL_SNAPSHOT, error=None, invented=False,
                 fail_after=None):
        self.model = model
        self.error = error
        self.invented = invented
        self.fail_after = fail_after
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.error is not None and (self.fail_after is None
                                       or self.calls > self.fail_after):
            raise self.error
        offered = pilot._offered_ids_from_kwargs(kwargs)
        chosen, seen = [], set()
        for assignment_id in offered:
            agent = assignment_id.split("@", 1)[0]
            if agent in seen:
                continue
            seen.add(agent)
            chosen.append(assignment_id)
        if self.invented:
            chosen = ["a99@0,0"]
        return pilot._fake_response(
            json.dumps({"assignment_ids": chosen, "confidence": 0.8}),
            model=self.model)


class ScriptedClient:
    def __init__(self, responses):
        self.responses = responses


@pytest.fixture
def live_env(monkeypatch, tmp_path):
    """A confirmed-live pilot whose transport is a local fake."""
    monkeypatch.setenv("OPENAI_API_KEY", SENTINEL_KEY)

    def install(responses):
        monkeypatch.setattr(pilot, "default_client_factory",
                            lambda *a, **k: ScriptedClient(responses))
        return responses

    install.report = str(tmp_path / "report.json")
    return install


def run(live_env, responses, *argv):
    live_env(responses)
    code = pilot.main(list(argv) + ["--report", live_env.report])
    with open(live_env.report) as handle:
        return code, json.load(handle)


# ==========================================================================
# 1. the live path is off unless it is explicitly confirmed
# ==========================================================================


def test_live_requires_an_explicit_confirmation_flag(tmp_path):
    path = str(tmp_path / "r.json")
    assert pilot.main(["--report", path]) == 0
    with open(path) as handle:
        report = json.load(handle)
    assert report["live"]["status"] == "skipped"
    assert report["live_attempts"] == 0
    assert report["attempt_ledger"] == []


def test_the_default_attempt_cap_is_one(live_env):
    _code, report = run(live_env, ScriptedResponses(), "--confirm-live")
    assert pilot.DEFAULT_MAX_ATTEMPTS == 1
    assert report["attempt_cap"] == 1
    assert report["live_attempts"] == 1


def test_a_probe_only_run_spends_one_attempt_and_still_passes_every_gate(live_env):
    """The default invocation is a complete, self-contained compatibility probe."""
    code, report = run(live_env, ScriptedResponses(), "--confirm-live")
    failed = [g for g in report["gates"] if not g["passed"]]
    assert failed == [], failed
    assert code == 0
    assert report["live_attempts"] == 1
    assert report["live"]["probe_repeat"]["extra_live_calls"] == 0
    assert report["live"]["cache_rows"] == 1


def test_the_cap_can_be_lowered_but_never_raised_past_the_ceiling(live_env):
    _code, report = run(live_env, ScriptedResponses(), "--confirm-live",
                        "--max-attempts", "9999")
    assert pilot.HARD_ATTEMPT_CEILING == 50
    assert report["attempt_cap"] == 50
    assert report["live_attempts"] <= 50


def test_a_zero_or_negative_cap_is_rejected():
    with pytest.raises(SystemExit):
        pilot.main(["--max-attempts", "0"])


# ==========================================================================
# 2. attempts are counted before the request leaves
# ==========================================================================


def test_an_attempt_is_ledgered_even_when_the_request_raises(live_env):
    class Boom(Exception):
        pass

    _code, report = run(live_env, ScriptedResponses(error=Boom()),
                        "--confirm-live", "--max-attempts", "30")
    ledger = report["attempt_ledger"]
    assert ledger, "a failed request must still consume an attempt"
    assert all(e["outcome"] == "exception" for e in ledger)
    assert all(e["error_class"] == "Boom" for e in ledger)


def test_retries_consume_the_attempt_budget(live_env):
    """A retry is a real request and must be visible in the ledger.

    The probe is capped at one attempt by construction, so the retry has to be
    observed after it: the transport answers the probe and then fails, and the
    adapter's single configured retry shows up as a second ledgered attempt for
    the same fixture.
    """
    class Boom(Exception):
        pass

    _code, report = run(live_env, ScriptedResponses(error=Boom(), fail_after=1),
                        "--confirm-live", "--max-attempts", "30")
    ledger = report["attempt_ledger"]
    assert [e["attempt_index"] for e in ledger] == list(range(1, len(ledger) + 1))
    assert ledger[0]["outcome"] == "response"
    failures = [e for e in ledger if e["outcome"] == "exception"]
    assert len(failures) >= 2, "the configured retry did not consume an attempt"
    assert report["live"]["pilot_accounting"]["retry_count"] >= 1


def test_attempts_never_exceed_the_configured_cap(live_env):
    class Boom(Exception):
        pass

    _code, report = run(live_env, ScriptedResponses(error=Boom()),
                        "--confirm-live", "--max-attempts", "3")
    assert report["live_attempts"] <= 3


# ==========================================================================
# 3. the probe gates the rest of the pilot
# ==========================================================================


def test_a_silently_substituted_model_stops_the_pilot_after_one_attempt(live_env):
    code, report = run(live_env, ScriptedResponses(model="gpt-other-2026-01-01"),
                       "--confirm-live", "--max-attempts", "30")
    assert code != 0
    assert report["live"]["status"] == "stopped_after_probe"
    assert report["live_attempts"] == 1, "no further budget may be spent"
    failed = [g["gate"] for g in report["gates"] if not g["passed"]]
    assert any("model snapshot" in g for g in failed)


def test_an_invented_id_fails_the_probe_and_activates_no_guidance(live_env):
    code, report = run(live_env, ScriptedResponses(invented=True),
                       "--confirm-live", "--max-attempts", "30")
    assert code != 0
    assert report["live"]["probe"]["provider_valid"] == 0
    assert report["live"]["status"] == "stopped_after_probe"


def test_a_transport_failure_stops_the_pilot(live_env):
    class Boom(Exception):
        pass

    code, report = run(live_env, ScriptedResponses(error=Boom()),
                       "--confirm-live", "--max-attempts", "30")
    assert code != 0
    assert report["live"]["status"] == "stopped_after_probe"


# ==========================================================================
# 4. the full happy path
# ==========================================================================


@pytest.fixture
def full_pilot(live_env):
    return run(live_env, ScriptedResponses(), "--confirm-live",
               "--max-attempts", "30")


def test_the_full_pilot_passes_every_gate(full_pilot):
    code, report = full_pilot
    failed = [g for g in report["gates"] if not g["passed"]]
    assert failed == [], failed
    assert code == 0


def test_one_live_attempt_per_distinct_fixture_and_no_more(full_pilot):
    _code, report = full_pilot
    assert report["live_attempts"] == report["fixture_count"]


def test_replaying_the_same_situation_costs_no_further_calls(full_pilot):
    _code, report = full_pilot
    replay = report["live"]["replay"]
    assert replay["count"] == report["fixture_count"]
    assert replay["served_from_cache"] == replay["count"]
    assert replay["extra_live_calls"] == 0


def test_read_only_hits_serve_and_read_only_misses_never_call(full_pilot):
    _code, report = full_pilot
    read_only = report["live"]["read_only"]
    assert read_only["hits"] == read_only["hit_fixtures"] > 0
    assert read_only["attempts_during_read_only"] == 0
    assert read_only["miss_fallback_reason"] == "cache_read_only_miss"
    assert read_only["miss_activated"] == 0


def test_only_validated_live_responses_reach_the_cache(full_pilot):
    _code, report = full_pilot
    assert report["live"]["cache_rows"] == report["live"]["metrics"][
        "cache_writes_expected"]
    assert report["live"]["cache_rows"] == report["fixture_count"]


def test_every_selection_is_grounded_in_the_offered_ids(full_pilot):
    _code, report = full_pilot
    metrics = report["live"]["metrics"]
    assert metrics["invented_or_unoffered_ids"] == 0
    assert metrics["grounded_selections"] == metrics["successful_api_responses"]
    assert metrics["parse_failures"] == 0


def test_accounting_invariants_hold_in_every_phase(full_pilot):
    _code, report = full_pilot
    for gate in report["gates"]:
        if "invariants" in gate["gate"]:
            assert gate["passed"], gate


def test_beta_is_zero_and_no_reward_is_altered(full_pilot):
    _code, report = full_pilot
    assert report["beta"] == 0.0
    assert pilot.PILOT_BETA == 0.0
    assert all(r["beta"] == 0.0 and r["reward_unchanged"]
               for r in report["live"]["records"])


def test_no_training_or_calibration_is_claimed(full_pilot):
    _code, report = full_pilot
    assert report["training_performed"] is False
    assert report["beta_calibration_performed"] is False
    assert report["not_a_performance_result"] is True


def test_cost_is_not_calculated_without_a_dated_price_table(full_pilot):
    _code, report = full_pilot
    assert report["cost_usd"] is None
    assert "NOT calculated" in report["cost_methodology"]
    assert report["live"]["metrics"]["input_tokens"] > 0


def test_versions_are_recorded_for_reproducibility(full_pilot):
    _code, report = full_pilot
    assert report["requested_model"] == MODEL_SNAPSHOT
    assert report["prompt_version"] == "selma_guidance_prompt_v2"
    assert report["response_schema_version"]
    assert report["cache_key_version"]
    assert report["cache_schema_version"]
    assert report["sdk_version"]


# ==========================================================================
# 5. secrets
# ==========================================================================


def test_the_report_never_contains_the_api_key(full_pilot, live_env):
    _code, _report = full_pilot
    with open(live_env.report) as handle:
        raw = handle.read()
    assert SENTINEL_KEY not in raw
    assert "sk-" not in raw
    assert "OPENAI_API_KEY" not in raw


def test_the_report_never_contains_headers_or_authorization(full_pilot, live_env):
    _code, _report = full_pilot
    with open(live_env.report) as handle:
        raw = handle.read().lower()
    for marker in ("authorization", "bearer ", "x-api-key", "cookie", "secret"):
        assert marker not in raw


def test_an_exception_message_is_never_recorded(live_env):
    class LeakyError(Exception):
        pass

    leaky = LeakyError("Authorization: Bearer {}".format(SENTINEL_KEY))
    leaky.status_code = 401
    _code, report = run(live_env, ScriptedResponses(error=leaky),
                        "--confirm-live", "--max-attempts", "30")
    raw = json.dumps(report)
    assert SENTINEL_KEY not in raw
    assert "Bearer" not in raw
    entry = report["attempt_ledger"][0]
    assert entry["error_class"] == "LeakyError"
    assert entry["status_code"] == 401


def test_sanitise_exception_keeps_only_structured_fields():
    class Err(Exception):
        pass

    error = Err("secret body sk-abcdefghijklmnop")
    error.status_code = 429
    error.code = "rate_limit_exceeded"
    info = pilot.sanitise_exception(error)
    assert info == {"error_class": "Err", "status_code": 429,
                    "code": "rate_limit_exceeded"}
    assert "sk-" not in json.dumps(info)


def test_missing_credentials_stop_before_any_attempt(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(pilot, "default_client_factory",
                        lambda *a, **k: ScriptedClient(ScriptedResponses()))
    with pytest.raises(SystemExit):
        pilot.main(["--confirm-live", "--report", str(tmp_path / "r.json")])


# ==========================================================================
# 6. offline gates the pilot always runs
# ==========================================================================


def test_invalid_responses_never_activate_guidance_or_write_the_cache(tmp_path):
    results = pilot.offline_invalid_response_gates(str(tmp_path))
    assert results
    for case in results:
        assert case["guidance_activated"] == 0, case
        assert case["provider_valid"] == 0, case
        assert case["shaping_contribution"] == 0.0, case
        assert case["cache_rows_written"] == 0, case
        assert case["invariant_problems"] == [], case


def test_greedy_evaluation_guard_is_present_in_the_runner():
    assert pilot.greedy_evaluation_gate()["guards_present"]


def test_every_frozen_fixture_offers_a_real_choice():
    from selma.cache import candidate_assignment_ids
    from selma.provider import build_request

    fixtures = pilot.frozen_fixtures()
    assert len(fixtures) >= 10
    names = [f["name"] for f in fixtures]
    assert len(set(names)) == len(names)
    for fixture in fixtures:
        request = build_request(fixture["row"], agent_positions=fixture["positions"])
        ids = candidate_assignment_ids(request)
        per_agent = {}
        for assignment_id in ids:
            per_agent.setdefault(assignment_id.split("@")[0], set()).add(assignment_id)
        assert len(ids) >= 2
        assert all(len(v) >= 2 for v in per_agent.values()), fixture["name"]


def test_translated_and_mirrored_layouts_stay_distinct_under_scheme_b():
    from selma.cache import build_cache_key, canonical_json, key_digest
    from selma.provider import build_request

    context = dict(condition=pilot.CONDITION, grid_shape=pilot.GRID_SHAPE,
                   provider="openai", model_snapshot=MODEL_SNAPSHOT,
                   reasoning_effort="none",
                   prompt_version="selma_guidance_prompt_v2",
                   response_schema_version="selma_guidance_response_schema_v1")
    digests = {}
    for fixture in pilot.frozen_fixtures():
        request = build_request(fixture["row"], agent_positions=fixture["positions"])
        digests[fixture["name"]] = key_digest(
            canonical_json(build_cache_key(request, **context)))
    assert len(set(digests.values())) == len(digests), digests
    assert digests["base_deficit2"] != digests["translated_plus1"]
    assert digests["base_deficit2"] != digests["mirrored_columns"]


def test_replayed_metadata_does_not_change_the_cache_key():
    from selma.cache import build_cache_key, canonical_json, key_digest
    from selma.provider import build_request

    context = dict(condition=pilot.CONDITION, grid_shape=pilot.GRID_SHAPE,
                   provider="openai", model_snapshot=MODEL_SNAPSHOT,
                   reasoning_effort="none",
                   prompt_version="selma_guidance_prompt_v2",
                   response_schema_version="selma_guidance_response_schema_v1")

    def digest_of(fixture):
        request = build_request(fixture["row"], agent_positions=fixture["positions"])
        return key_digest(canonical_json(build_cache_key(request, **context)))

    for fixture in pilot.frozen_fixtures():
        assert digest_of(fixture) == digest_of(pilot.replayed(fixture)), fixture["name"]
