"""SELMA: what the attempt ledger counts, and what it does not.

There are **two independent retry mechanisms** in the live provider path, and
they are easy to confuse because they share a parameter name:

* the **adapter's** retry loop -- ``OpenAIGuidanceProvider.max_retries``, frozen
  at 1 by ``selma_provider_max_retries``. Each of its attempts calls
  ``attempt_hook`` first, so every one is visible to the live-call cap, the cost
  ceiling and ``retry_count``;
* the **SDK's** transport retry -- ``OpenAI(max_retries=...)``, which this build
  never sets, so the SDK default applies. Those retries happen *inside* a single
  ``_create()`` call, below ``attempt_hook``, and are therefore **invisible** to
  the ledger.

The consequence, stated plainly so it is never mistaken for a defect discovered
later: ``live_attempts`` counts **adapter attempts, not HTTP requests**. One
ledgered attempt can issue up to ``1 + sdk_max_retries`` HTTP requests.

Observed in the Stage E recalibration runs, SDK-internal retries were 0.02-0.12 %
of adapter attempts, so the practical inflation is negligible -- but the ceiling
is structural, not empirical, and these tests pin it.

Run:  PYTHONPATH=src pytest tests/test_selma_sdk_retry_accounting.py -v
"""

import ast
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from selma.accounting import SelmaAccounting  # noqa: E402
from selma.gateway import ProviderGateway  # noqa: E402
from selma.openai_provider import (  # noqa: E402
    DEFAULT_MAX_RETRIES,
    OpenAIGuidanceProvider,
    default_client_factory,
)

ADAPTER_SOURCE = os.path.join(os.path.dirname(__file__), "..", "src", "selma",
                              "openai_provider.py")


# ==========================================================================
# 1. what the SDK client is actually configured with
# ==========================================================================


def test_the_client_factory_sets_sdk_max_retries_explicitly():
    """Verified by AST, not by text search.

    A plain substring scan for ``max_retries`` in this module is misleading:
    the adapter has its own attribute of that name. Only the keyword arguments
    of the ``OpenAI(...)`` call inside ``default_client_factory`` decide what
    the transport does.
    """
    with open(ADAPTER_SOURCE) as handle:
        tree = ast.parse(handle.read())
    kwargs = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "default_client_factory":
            for call in ast.walk(node):
                if isinstance(call, ast.Call) and getattr(call.func, "id", "") == "OpenAI":
                    kwargs = [k.arg for k in call.keywords]
    assert kwargs is not None, "no OpenAI(...) call found in the factory"
    assert "max_retries" in kwargs, (
        "the transport retry count must be set explicitly; inheriting the SDK "
        "default hides retries from the attempt ledger")
    assert sorted(kwargs) == ["api_key", "max_retries", "timeout"]


def test_the_sdk_default_that_the_accepted_runs_inherited_is_recorded():
    """Historical fact, deliberately pinned rather than overwritten.

    The six accepted Stage E recalibration conditions (execution commit
    53d9a2e) ran BEFORE this correction and therefore inherited the SDK default.
    Their log-reconstructed HTTP counts in
    docs/project/selma_stage_e_beta_recalibration_runs_2026-08-18.md are only
    interpretable against that number, so it is asserted here rather than left
    to memory.
    """
    import inspect

    from openai import OpenAI

    inherited_default = inspect.signature(
        OpenAI.__init__).parameters["max_retries"].default
    assert inherited_default == 2, (
        "the accepted runs inherited SDK max_retries=2; if the installed SDK "
        "default has changed, the historical record needs re-reading, not this "
        "assertion relaxing")

    record = os.path.join(os.path.dirname(__file__), "..", "docs", "project",
                          "selma_stage_e_beta_recalibration_runs_2026-08-18.md")
    with open(record) as handle:
        text = handle.read()
    assert "inherited = 2" in text or "max_retries = 2" in text
    assert "53d9a2e" in text


def test_the_adapter_retry_setting_is_separate_and_frozen_at_one():
    provider = OpenAIGuidanceProvider(client=object())
    assert provider.max_retries == DEFAULT_MAX_RETRIES == 1


# ==========================================================================
# 2. where attempt_hook sits relative to the SDK call
# ==========================================================================


class _CountingResponses:
    """Simulates an SDK that retries internally before returning."""

    def __init__(self, internal_retries):
        self.internal_retries = internal_retries
        self.http_requests = 0
        self.create_calls = 0

    def create(self, **kwargs):
        self.create_calls += 1
        # the SDK would issue 1 + internal_retries HTTP requests here
        self.http_requests += 1 + self.internal_retries
        raise RuntimeError("transport")     # classified, never re-raised


class _Client:
    def __init__(self, responses):
        self.responses = responses


def _stack(internal_retries, cap=10):
    responses = _CountingResponses(internal_retries)
    provider = OpenAIGuidanceProvider(client=_Client(responses), backoff_s=0.0,
                                      sleep=lambda _s: None)
    accounting = SelmaAccounting(cache_mode="off", live_call_cap=cap,
                                 provider="openai", live_provider=True)
    gateway = ProviderGateway(provider, accounting=accounting, cache=None,
                              cache_mode="off", live_call_cap=cap)
    return responses, provider, accounting, gateway


def _request():
    from test_selma_stage_d_joint_plans import make_request

    return make_request()


def test_the_hook_fires_once_per_adapter_attempt_not_per_http_request():
    responses, _p, accounting, gateway = _stack(internal_retries=2)
    gateway.query(_request())

    # adapter: 1 initial + 1 configured retry = 2 attempts
    assert responses.create_calls == 2
    assert accounting.live_api_call_count == 2
    assert accounting.retry_count == 1
    # transport: each attempt issued 3 HTTP requests
    assert responses.http_requests == 6
    assert accounting.live_api_call_count < responses.http_requests


def test_sdk_internal_retries_are_invisible_to_the_ledger():
    quiet, _p, quiet_acc, quiet_gw = _stack(internal_retries=0)
    noisy, _p2, noisy_acc, noisy_gw = _stack(internal_retries=2)
    quiet_gw.query(_request())
    noisy_gw.query(_request())

    # the ledger cannot tell the two apart ...
    assert quiet_acc.live_api_call_count == noisy_acc.live_api_call_count
    assert quiet_acc.retry_count == noisy_acc.retry_count
    # ... even though one issued three times the HTTP traffic
    assert noisy.http_requests == 3 * quiet.http_requests


def test_the_cap_bounds_adapter_attempts_not_http_requests():
    responses, _p, accounting, gateway = _stack(internal_retries=2, cap=2)
    for _ in range(5):
        gateway.query(_request())
    assert accounting.live_api_call_count == 2, "the cap binds on attempts"
    assert responses.http_requests == 6, "which is not the HTTP total"


def test_the_worst_case_http_multiplier_is_one_plus_sdk_retries():
    import inspect

    from openai import OpenAI

    sdk = inspect.signature(OpenAI.__init__).parameters["max_retries"].default
    responses, _p, accounting, gateway = _stack(internal_retries=sdk)
    gateway.query(_request())
    assert responses.http_requests == accounting.live_api_call_count * (1 + sdk)


# ==========================================================================
# 3. cost semantics
# ==========================================================================


def test_cost_is_derived_only_from_returned_usage_never_from_attempt_counts():
    """A retried-then-failed HTTP request reports no usage, so it adds no cost.

    This is why the cost ceiling cannot be inflated by transport retries: it is
    computed from the ``usage`` block of responses that actually came back, not
    from how many times the wire was touched.
    """
    _r, _p, accounting, gateway = _stack(internal_retries=2)
    gateway.query(_request())
    assert accounting.input_tokens == 0
    assert accounting.output_tokens == 0

    from selma.pricing import PriceTable

    table = PriceTable("2026-08-17", 0.20, 0.02, 1.25)
    assert accounting.cost_usd(table) == 0.0


# ==========================================================================
# 4. FAILS-FIRST: the pre-final-experiment correction
# ==========================================================================
#
# These pin the runtime correction required before any FINAL experiment:
# disable the SDK's hidden transport retries so that every retry that happens
# is a ledgered adapter retry consuming the attempt cap.
#
# They are written against the *uncorrected* tree and fail there, which is the
# point. The correction is deliberately NOT applied while the six calibration
# conditions are running: waves 1-2 executed with the SDK default, so changing
# the transport mid-pilot would give wave 3 different runtime behaviour and
# destroy cross-condition comparability.


def test_the_client_factory_disables_sdk_transport_retries():
    """Every retry must be one the ledger can see."""
    with open(ADAPTER_SOURCE) as handle:
        tree = ast.parse(handle.read())
    kwargs = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "default_client_factory":
            for call in ast.walk(node):
                if isinstance(call, ast.Call) and getattr(call.func, "id", "") == "OpenAI":
                    kwargs = {k.arg: k.value for k in call.keywords}
    assert "max_retries" in kwargs, (
        "default_client_factory must set max_retries explicitly; inheriting the "
        "SDK default hides transport retries from the attempt ledger")
    node = kwargs["max_retries"]
    value = node.value if isinstance(node, ast.Constant) else None
    assert value == 0, "SDK transport retries must be disabled (max_retries=0)"


def test_the_constructed_client_reports_zero_sdk_retries(monkeypatch):
    captured = {}

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import openai

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-DO-NOT-LEAK-0000000000")
    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)
    default_client_factory()
    assert captured.get("max_retries") == 0
    assert "sk-" not in str(captured.get("timeout", ""))


def test_the_single_adapter_retry_is_retained_and_ledgered():
    """Disabling SDK retries must not remove our own bounded retry."""
    provider = OpenAIGuidanceProvider(client=object())
    assert provider.max_retries == 1

    responses, _p, accounting, gateway = _stack(internal_retries=0)
    gateway.query(_request())
    assert responses.create_calls == 2, "1 initial + 1 adapter retry"
    assert accounting.live_api_call_count == 2
    assert accounting.retry_count == 1


def test_with_sdk_retries_disabled_http_equals_ledgered_attempts():
    """The whole point: the ledger becomes an exact HTTP count."""
    responses, _p, accounting, gateway = _stack(internal_retries=0)
    gateway.query(_request())
    assert responses.http_requests == accounting.live_api_call_count


def test_every_permitted_retry_consumes_the_attempt_cap():
    responses, _p, accounting, gateway = _stack(internal_retries=0, cap=3)
    for _ in range(5):
        gateway.query(_request())
    assert accounting.live_api_call_count == 3, "the cap bounds every retry"
    assert responses.http_requests == 3
    assert responses.create_calls == 3
