"""SELMA Stage D: the OpenAI adapter, driven entirely by injected fake clients.

**No test here touches the network** -- `tests/conftest.py` enforces that for the
whole suite, and every client used below is a local fake.

Run:  PYTHONPATH=src pytest tests/test_selma_stage_d_provider.py -v
"""

import json
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from selma.accounting import (  # noqa: E402
    FALLBACK_INCOMPLETE,
    FALLBACK_MALFORMED,
    FALLBACK_RATE_LIMIT,
    FALLBACK_REFUSAL,
    FALLBACK_TIMEOUT,
    FALLBACK_TRANSPORT,
)
from selma.cache import candidate_assignment_ids  # noqa: E402
from selma.openai_provider import (  # noqa: E402
    API_KEY_ENV,
    MODEL_SNAPSHOT,
    PROMPT_VERSION,
    RESPONSE_SCHEMA_VERSION,
    MissingApiKeyError,
    OpenAIGuidanceProvider,
    build_prompt,
    parse_assignment_id,
    response_json_schema,
)
from selma.provider import (  # noqa: E402
    ROLE_APPROACH_AND_LOAD,
    ROLE_HOLD_AND_LOAD,
    validate,
)
from selma.provider_registry import make_provider  # noqa: E402
from test_selma_stage_d_cache import make_request  # noqa: E402

SENTINEL_KEY = "sk-test-DO-NOT-LEAK-0123456789"


# ==========================================================================
# fakes
# ==========================================================================


class FakeTimeout(Exception):
    """Named so the adapter's class-name classifier sees a timeout."""


class FakeRateLimitError(Exception):
    pass


class FakeConnectionError(Exception):
    pass


class FakeResponses:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0) if self.script else self.script
        if isinstance(item, Exception):
            raise item
        return item


class FakeClient:
    def __init__(self, *script):
        self.responses = FakeResponses(script)


def usage(input_tokens=120, output_tokens=18, cached=64, reasoning=0):
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_tokens_details=SimpleNamespace(cached_tokens=cached),
        output_tokens_details=SimpleNamespace(reasoning_tokens=reasoning),
    )


def api_response(text, *, status="completed", response_id="resp_abc123",
                 model=MODEL_SNAPSHOT, refusal=None):
    content = ([{"type": "refusal", "refusal": refusal}] if refusal
               else [{"type": "output_text", "text": text}])
    return SimpleNamespace(
        id=response_id, model=model, status=status, usage=usage(),
        output_text=None if refusal else text,
        output=[SimpleNamespace(content=content)],
        incomplete_details=None,
    )


def good_payload(request, confidence=0.9):
    ids = candidate_assignment_ids(request)
    chosen, seen = [], set()
    for assignment_id in ids:
        agent_id, _cell = parse_assignment_id(assignment_id)
        if agent_id in seen:
            continue
        seen.add(agent_id)
        chosen.append(assignment_id)
    return json.dumps({"assignment_ids": chosen, "confidence": confidence})


def provider(*script, **kwargs):
    kwargs.setdefault("backoff_s", 0.0)
    kwargs.setdefault("sleep", lambda _s: None)
    return OpenAIGuidanceProvider(client=FakeClient(*script), **kwargs)


# ==========================================================================
# 1. the frozen offline contract
# ==========================================================================


def test_model_snapshot_is_the_exact_dated_version():
    assert MODEL_SNAPSHOT == "gpt-5.4-nano-2026-03-17"
    assert OpenAIGuidanceProvider(client=FakeClient()).model == MODEL_SNAPSHOT


@pytest.mark.parametrize("alias", ["gpt-5.4-nano", "gpt-5.4", "gpt-5.4-nano-latest"])
def test_unversioned_model_alias_is_rejected(alias):
    with pytest.raises(ValueError) as error:
        OpenAIGuidanceProvider(client=FakeClient(), model=alias)
    assert "snapshot" in str(error.value)


def test_request_uses_the_responses_api_contract():
    request = make_request()
    p = provider(api_response(good_payload(request)))
    p.query(request)
    (kwargs,) = p.client.responses.calls
    assert kwargs["model"] == MODEL_SNAPSHOT
    assert kwargs["store"] is False
    assert kwargs["stream"] is False
    assert kwargs["tools"] == []
    assert kwargs["reasoning"] == {"effort": "none"}
    assert kwargs["text"]["format"]["type"] == "json_schema"
    assert kwargs["text"]["format"]["strict"] is True
    assert kwargs["max_output_tokens"] == 512
    assert kwargs["timeout"] == 30.0


def test_structured_output_schema_enumerates_only_offered_ids():
    request = make_request()
    ids = candidate_assignment_ids(request)
    schema = response_json_schema(ids)
    assert schema["additionalProperties"] is False
    assert schema["properties"]["assignment_ids"]["items"]["enum"] == list(ids)


def test_prompt_is_deterministic_and_carries_no_free_coordinates():
    request = make_request()
    ids = candidate_assignment_ids(request)
    assert build_prompt(request, ids) == build_prompt(request, ids)
    payload = json.loads(build_prompt(request, ids))
    assert payload["candidate_assignment_ids"] == list(ids)


def test_provider_versions_are_exposed_for_the_cache_key():
    p = provider()
    assert p.prompt_version == PROMPT_VERSION
    assert p.response_schema_version == RESPONSE_SCHEMA_VERSION
    assert p.provider_id == "openai"


# ==========================================================================
# 2. credentials
# ==========================================================================


def test_the_network_guard_is_active():
    """The suite-wide guard in conftest.py must actually block connections.

    Without this, "no test touches the network" would be a convention rather
    than a property of the run.
    """
    import socket

    with pytest.raises(Exception) as error:
        socket.create_connection(("127.0.0.1", 9), timeout=0.01)
    assert "forbidden" in str(error.value).lower()
    with pytest.raises(Exception):
        socket.socket().connect(("127.0.0.1", 9))


def test_missing_api_key_fails_clearly_when_the_provider_is_used(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    p = OpenAIGuidanceProvider()          # construction alone must not fail
    with pytest.raises(MissingApiKeyError) as error:
        _ = p.client
    assert API_KEY_ENV in str(error.value)


def test_api_key_is_never_read_from_config_arguments():
    with pytest.raises(TypeError):
        OpenAIGuidanceProvider(api_key=SENTINEL_KEY)   # no such parameter, by design


def test_api_key_never_appears_in_sent_payloads_or_errors(monkeypatch, capsys):
    monkeypatch.setenv(API_KEY_ENV, SENTINEL_KEY)
    request = make_request()
    p = provider(FakeTimeout("connection timed out"),
                 api_response(good_payload(request)))
    p.query(request)
    printed = capsys.readouterr()
    blob = json.dumps(p.client.responses.calls, default=str) + printed.out + printed.err
    blob += repr(p.last_error) + repr(p.last_usage)
    assert SENTINEL_KEY not in blob


def test_no_openai_sdk_import_for_mock_or_off_paths():
    """off / shadow / mock must never import the SDK or build a client."""
    sys.modules.pop("openai", None)
    mock = make_provider("deterministic_mock")
    mock.query(make_request())
    assert "openai" not in sys.modules
    # constructing the adapter with an injected client must not import it either
    OpenAIGuidanceProvider(client=FakeClient())
    assert "openai" not in sys.modules


# ==========================================================================
# 3. a valid response
# ==========================================================================


def test_valid_structured_response_passes_the_deterministic_validator():
    request = make_request()
    p = provider(api_response(good_payload(request)))
    response = p.query(request)
    assert response is not None
    verdict = validate(request, response)
    assert verdict.valid, verdict.reason
    assert p.last_error == ""


def test_identity_target_and_recipients_are_code_fixed_not_model_supplied():
    request = make_request()
    p = provider(api_response(good_payload(request)))
    response = p.query(request)
    assert response.event_id == request.event_id
    assert response.target_food_id == tuple(request.food_id)
    assert set(response.recipients) == set(request.recipient_agents)


def test_role_is_derived_by_code_from_the_real_position():
    request = make_request(positions={0: (4, 3), 1: (6, 6)})
    p = provider(api_response(good_payload(request)))
    response = p.query(request)
    roles = {a.agent_id: a.role for a in response.assignments}
    cells = {a.agent_id: a.loading_cell for a in response.assignments}
    for agent_id, cell in cells.items():
        position = {0: (4, 3), 1: (6, 6)}[agent_id]
        expected = ROLE_HOLD_AND_LOAD if position == cell else ROLE_APPROACH_AND_LOAD
        assert roles[agent_id] == expected


def test_usage_and_latency_are_captured():
    request = make_request()
    p = provider(api_response(good_payload(request)))
    p.query(request)
    assert p.last_usage.tokens_prompt == 120
    assert p.last_usage.tokens_completion == 18
    assert p.last_usage.cached_tokens_prompt == 64
    assert p.last_usage.latency_s is not None and p.last_usage.latency_s >= 0.0
    assert p.last_usage.request_id == "resp_abc123"
    assert p.last_usage.returned_model == MODEL_SNAPSHOT
    # cost is never hard-coded; it is recomputed from tokens and a price table
    assert p.last_usage.cost_usd is None


def test_returned_model_identifier_is_recorded_even_when_it_differs():
    request = make_request()
    p = provider(api_response(good_payload(request), model="gpt-5.4-nano-2026-09-01"))
    p.query(request)
    assert p.last_returned_model == "gpt-5.4-nano-2026-09-01"


# ==========================================================================
# 4. every failure path yields no guidance
# ==========================================================================


def test_refusal_yields_no_guidance():
    request = make_request()
    p = provider(api_response(None, refusal="I cannot help with that"))
    assert p.query(request) is None
    assert p.last_error == FALLBACK_REFUSAL


def test_incomplete_response_yields_no_guidance():
    request = make_request()
    p = provider(api_response(good_payload(request), status="incomplete"))
    assert p.query(request) is None
    assert p.last_error == FALLBACK_INCOMPLETE


def test_timeout_yields_no_guidance_after_bounded_retries():
    request = make_request()
    p = provider(FakeTimeout(), FakeTimeout(), max_retries=1)
    assert p.query(request) is None
    assert p.last_error == FALLBACK_TIMEOUT
    assert p.last_attempts == 2


def test_rate_limit_is_classified_and_retried():
    request = make_request()
    p = provider(FakeRateLimitError(), api_response(good_payload(request)),
                 max_retries=1)
    assert p.query(request) is not None
    assert p.last_attempts == 2


def test_rate_limit_exhausting_retries_yields_no_guidance():
    request = make_request()
    p = provider(FakeRateLimitError(), FakeRateLimitError(), max_retries=1)
    assert p.query(request) is None
    assert p.last_error == FALLBACK_RATE_LIMIT


def test_transport_error_yields_no_guidance():
    request = make_request()
    p = provider(FakeConnectionError(), max_retries=0)
    assert p.query(request) is None
    assert p.last_error == FALLBACK_TRANSPORT


def test_malformed_structured_response_yields_no_guidance():
    request = make_request()
    p = provider(api_response("not json at all"), api_response("still not json"),
                 max_retries=1)
    assert p.query(request) is None
    assert p.last_error == FALLBACK_MALFORMED


def test_malformed_output_gets_exactly_one_bounded_retry():
    request = make_request()
    p = provider(api_response("{"), api_response(good_payload(request)),
                 max_retries=1)
    assert p.query(request) is not None
    assert p.last_attempts == 2


def test_invented_assignment_id_is_rejected():
    request = make_request()
    payload = json.dumps({"assignment_ids": ["a0@0,0", "a1@7,7"], "confidence": 1.0})
    p = provider(api_response(payload), max_retries=0)
    assert p.query(request) is None
    assert p.last_error == FALLBACK_MALFORMED


def test_duplicate_assignment_id_is_rejected():
    request = make_request()
    ids = candidate_assignment_ids(request)
    payload = json.dumps({"assignment_ids": [ids[0], ids[0]], "confidence": 1.0})
    p = provider(api_response(payload), max_retries=0)
    assert p.query(request) is None


def test_two_ids_for_one_agent_are_rejected():
    request = make_request()
    ids = [i for i in candidate_assignment_ids(request) if i.startswith("a0@")]
    assert len(ids) >= 2
    payload = json.dumps({"assignment_ids": ids, "confidence": 1.0})
    p = provider(api_response(payload), max_retries=0)
    assert p.query(request) is None


def test_missing_recipient_selection_is_rejected():
    request = make_request()
    ids = candidate_assignment_ids(request)
    payload = json.dumps({"assignment_ids": [ids[0]], "confidence": 1.0})
    p = provider(api_response(payload), max_retries=0)
    assert p.query(request) is None


def test_non_numeric_confidence_is_rejected():
    request = make_request()
    ids = candidate_assignment_ids(request)
    chosen = json.loads(good_payload(request))["assignment_ids"]
    payload = json.dumps({"assignment_ids": chosen, "confidence": "very"})
    p = provider(api_response(payload), max_retries=0)
    assert p.query(request) is None


def test_no_heuristic_fallback_is_ever_substituted():
    """Every failure path returns None -- never a locally invented subgoal."""
    request = make_request()
    for script in ([FakeTimeout()], [api_response("garbage")],
                   [api_response(None, refusal="no")],
                   [api_response(good_payload(request), status="incomplete")]):
        p = provider(*script, max_retries=0)
        assert p.query(request) is None
