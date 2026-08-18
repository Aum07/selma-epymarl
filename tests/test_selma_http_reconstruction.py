"""SELMA: reconstructing true HTTP counts and bounding hidden-retry cost.

Offline. Verifies that the reconstruction distinguishes ledgered adapter
attempts from actual HTTP transmissions, keeps the cap check honest, and never
claims hidden retries are free.

Run:  PYTHONPATH=src pytest tests/test_selma_http_reconstruction.py -v
"""

import csv
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))

import selma_http_reconstruction as recon  # noqa: E402
from selma.guidance import SelmaDecisionCsvWriter  # noqa: E402


def _write(tmp_path, name, *, attempts_each=1, n=10, tokens_in=877,
           tokens_out=25, sdk_retry_lines=0):
    directory = os.path.join(str(tmp_path), name)
    os.makedirs(directory, exist_ok=True)
    rows = []
    for i in range(n):
        row = {c: "" for c in SelmaDecisionCsvWriter.COLUMNS}
        row.update({
            "t_env": i, "decision": "multiple", "live_attempts": attempts_each,
            "retries": max(0, attempts_each - 1), "outcome": "activated",
            "tokens_input": tokens_in, "tokens_cached_input": 0,
            "tokens_output": tokens_out, "activated": 1,
            "physically_valid": 1, "selected_plan_id": "plan_0",
            "returned_model": "gpt-5.4-nano-2026-03-17", "greedy_eval": 0,
            "key_digest": "d" * 64, "cache_result": "miss",
        })
        rows.append(row)
    with open(os.path.join(directory, "decisions.csv"), "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=SelmaDecisionCsvWriter.COLUMNS)
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(directory, "run.log"), "w") as h:
        for _ in range(sdk_retry_lines):
            h.write("WARNING Retrying request to /responses in 0.4 seconds\n")
        h.write("done\n")
    return directory


def test_log_reconstructed_http_is_attempts_plus_logged_sdk_retries(tmp_path):
    d = _write(tmp_path, "beta_0p00_seed91", n=10, sdk_retry_lines=3)
    out = recon.reconstruct(d)
    assert out["adapter_attempts"] == 10
    assert out["logged_sdk_internal_retries"] == 3
    assert out["log_reconstructed_observed_http_count"] == 13


def test_adapter_attempts_are_labelled_as_not_http(tmp_path):
    out = recon.reconstruct(_write(tmp_path, "beta_0p00_seed91"))
    assert "NOT HTTP" in out["adapter_attempts_meaning"].upper()


def test_the_structural_ceiling_uses_one_plus_sdk_default(tmp_path):
    out = recon.reconstruct(_write(tmp_path, "beta_0p00_seed91", n=10))
    assert recon.SDK_MAX_RETRIES_DEFAULT == 2
    assert out["structural_http_ceiling"] == 30
    assert out["log_reconstructed_observed_http_count"] <= out["structural_http_ceiling"]


def test_the_cap_check_uses_log_reconstructed_not_ledgered_counts(tmp_path):
    d = _write(tmp_path, "beta_0p10_seed92", n=13499, sdk_retry_lines=5)
    out = recon.reconstruct(d)
    assert out["adapter_attempts"] == 13499
    assert out["log_reconstructed_observed_http_count"] == 13504
    assert out["log_reconstructed_within_cap"] is False, "13504 exceeds 13500"


def test_a_condition_within_the_cap_is_reported_with_headroom(tmp_path):
    out = recon.reconstruct(_write(tmp_path, "beta_0p00_seed91", n=100,
                                   sdk_retry_lines=2))
    assert out["log_reconstructed_within_cap"] is True
    assert out["headroom_under_cap"] == 13500 - 102


def test_hidden_retry_cost_is_bounded_never_assumed_zero(tmp_path):
    out = recon.reconstruct(_write(tmp_path, "beta_0p00_seed91", n=100,
                                   sdk_retry_lines=10))
    assert out["cost_hidden_retry_p95_estimate_usd"] > 0.0
    assert out["cost_upper_bound_usd"] > out["cost_local_successful_responses_usd"]


def test_the_p95_figure_is_labelled_an_estimate_not_a_maximum(tmp_path):
    out = recon.reconstruct(_write(tmp_path, "beta_0p00_seed91", n=100,
                                   sdk_retry_lines=10))
    expected = (10 * 877 * 0.20 + 10 * 25 * 1.25) / 1e6
    assert out["cost_hidden_retry_p95_estimate_usd"] == pytest.approx(expected,
                                                                      abs=1e-6)
    assert "ESTIMATE" in out["cost_note"] and "not a maximum" in out["cost_note"]


def test_the_hard_bound_uses_max_input_and_the_512_output_ceiling(tmp_path):
    """Hard with respect to LOGGED retries -- not unconditional over all HTTP."""
    out = recon.reconstruct(_write(tmp_path, "beta_0p00_seed91", n=100,
                                   tokens_in=900, tokens_out=25,
                                   sdk_retry_lines=10))
    hard = (10 * 900 * 0.20 + 10 * 512 * 1.25) / 1e6
    assert out["cost_hard_upper_bound_for_log_observed_sdk_retries_usd"] == pytest.approx(hard,
                                                                          abs=1e-6)
    assert out["cost_hard_upper_bound_for_log_observed_sdk_retries_usd"] > \
        out["cost_hidden_retry_p95_estimate_usd"]
    assert out["tokens"]["configured_max_output_tokens"] == 512
    scope = out["cost_hard_bound_scope"].lower()
    assert "log-observed" in scope
    assert "not an unconditional bound" in scope
    assert "structural" in scope


def test_the_reconstruction_is_labelled_as_log_derived_not_guaranteed(tmp_path):
    out = recon.reconstruct(_write(tmp_path, "beta_0p00_seed91", sdk_retry_lines=2))
    caveat = out["log_reconstructed_caveat"].lower()
    assert "not a" in caveat and "guaranteed" in caveat
    assert "packet-level" in caveat


def test_zero_sdk_retries_gives_a_zero_hidden_contribution(tmp_path):
    out = recon.reconstruct(_write(tmp_path, "beta_0p00_seed91", n=50))
    assert out["logged_sdk_internal_retries"] == 0
    assert out["cost_hidden_retry_p95_estimate_usd"] == 0.0
    assert out["cost_hard_upper_bound_for_log_observed_sdk_retries_usd"] == 0.0
    assert out["cost_upper_bound_usd"] == out["cost_local_successful_responses_usd"]


def test_local_cost_comes_from_returned_usage(tmp_path):
    out = recon.reconstruct(_write(tmp_path, "beta_0p00_seed91", n=100))
    expected = (100 * 877 * 0.20 + 100 * 25 * 1.25) / 1e6
    assert out["cost_local_successful_responses_usd"] == pytest.approx(expected,
                                                                       abs=1e-6)


def test_reconciliation_states_plainly_when_no_dashboard_figure_exists(tmp_path):
    entries = [recon.reconstruct(_write(tmp_path, "beta_0p00_seed91"))]
    summary = recon.reconcile(entries, dashboard_usd=None)
    assert summary["dashboard_usd"] is None
    assert "NOT PERFORMED" in summary["dashboard_reconciliation"]


def test_reconciliation_compares_against_a_supplied_dashboard_figure(tmp_path):
    entries = [recon.reconstruct(_write(tmp_path, "beta_0p00_seed91", n=100,
                                        sdk_retry_lines=5))]
    summary = recon.reconcile(entries, dashboard_usd=0.02)
    assert summary["dashboard_usd"] == 0.02
    assert "within_upper_bound" in summary
    assert summary["delta_vs_local_usd"] == pytest.approx(
        0.02 - summary["total_local_cost_usd"], abs=1e-6)


def test_the_module_makes_no_network_call():
    source = open(recon.__file__).read()
    for forbidden in ("import requests", "import httpx", "from openai",
                      "urlopen", "OPENAI_API_KEY"):
        assert forbidden not in source, forbidden
