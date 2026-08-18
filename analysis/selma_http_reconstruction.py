#!/usr/bin/env python
"""SELMA: reconstruct true HTTP-request counts and bound hidden-retry cost.

Read-only. Makes no network call and runs no training.

Why this exists
---------------

``live_api_call_count`` is **adapter attempts, not HTTP transmissions**. The
adapter's ``attempt_hook`` fires once per attempt in
``OpenAIGuidanceProvider.query()``, immediately before ``_create()``. The OpenAI
SDK's own transport retries happen *inside* that single call, below the hook, so
they never reach the ledger.

This module reconstructs, **from logs**, what appears to have gone over the wire:

    log_reconstructed_observed_http_count
        =  adapter_attempts  +  logged_sdk_internal_retries

``logged_sdk_internal_retries`` is counted from the SDK's own "Retrying request"
lines. This is a **log-reconstructed observed count, not the guaranteed true
HTTP count.** No packet-level evidence exists here, and the reconstruction would
undercount any transmission the SDK did not log -- a connection reset before the
retry decision, a redirect, or a request retried by an intermediary. The
structural ceiling ``adapter_attempts * (1 + max_retries)`` is reported
alongside it as the bound that does not depend on logging fidelity.

On cost
-------

It is **not** safe to claim hidden retries cost nothing. Our per-condition cost
is computed only from the ``usage`` block of responses that came back, so a
retry that never returned a body contributes nothing *locally*. But the vendor
bills on its own accounting, and a request the server processed before the
connection dropped could be billed without us ever seeing its usage. This module
therefore reports:

* **local successful-response cost** -- what our logs can prove;
* a **p95-based conservative estimate** of the hidden-retry contribution --
  every logged SDK retry charged as a full-price request at the p95 token
  profile. This is an *estimate*, not a maximum: a retry could carry a larger
  prompt than p95;
* a **hard cost upper bound for log-observed SDK retries** -- every *logged*
  retry charged at the largest input-token count actually observed and the
  configured 512-token output ceiling, which no single response can exceed.
  This is hard **with respect to the retries the SDK logged**; it is *not* an
  unconditional bound over every possible HTTP transmission, because a
  transmission the SDK never logged is outside its scope. The
  ``structural_ceiling`` figure is the one that does not depend on logging;
* the sum with that scoped bound, to reconcile against the provider dashboard.
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from selma.pricing import PriceTable  # noqa: E402

PRICE_TABLE = PriceTable("2026-08-17", 0.20, 0.02, 1.25)

#: The SDK logs one line per transport retry it performs.
SDK_RETRY_PATTERN = re.compile(r"Retrying request", re.I)

#: Structural ceiling: this build never sets max_retries, so the SDK default
#: applies. Reported alongside the observed count, never instead of it.
SDK_MAX_RETRIES_DEFAULT = 2

PER_CONDITION_CALL_CAP = 13500

#: Frozen ``selma_provider_max_output_tokens``. No single response can exceed
#: it, which is what makes the hard cost bound hard.
MAX_OUTPUT_TOKENS = 512


def _int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def count_sdk_retries(run_log):
    """Observed SDK transport retries, from the SDK's own log lines."""
    if not run_log or not os.path.exists(run_log):
        return None
    total = 0
    with open(run_log, errors="replace") as handle:
        for line in handle:
            if SDK_RETRY_PATTERN.search(line):
                total += 1
    return total


def reconstruct(condition_dir, *, cap=PER_CONDITION_CALL_CAP,
                price_table=PRICE_TABLE):
    """Reconstruct HTTP counts and cost bounds for one condition."""
    decisions_path = os.path.join(condition_dir, "decisions.csv")
    run_log = os.path.join(condition_dir, "run.log")
    if not os.path.exists(decisions_path):
        raise FileNotFoundError(decisions_path)

    with open(decisions_path, newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["decision"] == "multiple"]

    adapter_attempts = sum(_int(r["live_attempts"]) for r in rows)
    ledger_retries = sum(_int(r["retries"]) for r in rows)
    called = [r for r in rows if _int(r["live_attempts"]) > 0]

    tin = sum(_int(r["tokens_input"]) for r in called)
    tcached = sum(_int(r["tokens_cached_input"]) for r in called)
    tout = sum(_int(r["tokens_output"]) for r in called)

    inputs = sorted(_int(r["tokens_input"]) for r in called
                    if r["tokens_input"] not in ("", "n/a"))
    outputs = sorted(_int(r["tokens_output"]) for r in called
                     if r["tokens_output"] not in ("", "n/a"))

    def p95(values):
        return values[min(len(values) - 1, int(0.95 * len(values)))] if values else 0

    p95_in, p95_out = p95(inputs), p95(outputs)
    max_in = max(inputs) if inputs else 0
    max_out = max(outputs) if outputs else 0

    sdk_retries = count_sdk_retries(run_log)
    observed_http = (adapter_attempts + sdk_retries
                     if sdk_retries is not None else None)
    ceiling_http = adapter_attempts * (1 + SDK_MAX_RETRIES_DEFAULT)

    local_cost = price_table.cost_usd(input_tokens=tin,
                                      cached_input_tokens=tcached,
                                      output_tokens=tout)
    retries = sdk_retries or 0
    # p95-based CONSERVATIVE ESTIMATE -- not a maximum: a retried request could
    # carry a larger prompt than the 95th percentile.
    hidden_p95 = price_table.cost_usd(input_tokens=retries * p95_in,
                                      cached_input_tokens=0,
                                      output_tokens=retries * p95_out)
    # HARD upper bound: the largest input actually observed, and the configured
    # max_output_tokens ceiling that no single response can exceed.
    hidden_hard = price_table.cost_usd(
        input_tokens=retries * max_in, cached_input_tokens=0,
        output_tokens=retries * MAX_OUTPUT_TOKENS)
    ceiling_hidden = price_table.cost_usd(
        input_tokens=(ceiling_http - adapter_attempts) * max_in,
        cached_input_tokens=0,
        output_tokens=(ceiling_http - adapter_attempts) * MAX_OUTPUT_TOKENS)

    return {
        "condition": os.path.basename(condition_dir.rstrip("/")),
        "llm_eligible_decisions": len(rows),
        "adapter_attempts": adapter_attempts,
        "adapter_attempts_meaning": "ledgered attempts, NOT HTTP transmissions",
        "ledger_retries": ledger_retries,
        "logged_sdk_internal_retries": sdk_retries,
        "log_reconstructed_observed_http_count": observed_http,
        "log_reconstructed_caveat": ("observed from SDK log lines; NOT a "
                                     "guaranteed true HTTP count -- no "
                                     "packet-level evidence exists"),
        "structural_http_ceiling": ceiling_http,
        "cap": cap,
        "log_reconstructed_within_cap": (observed_http is not None
                                         and observed_http <= cap),
        "structural_ceiling_within_cap": ceiling_http <= cap,
        "headroom_under_cap": (cap - observed_http
                               if observed_http is not None else None),
        "tokens": {"input": tin, "cached_input": tcached, "output": tout,
                   "p95_input_per_call": p95_in, "p95_output_per_call": p95_out,
                   "max_input_per_call": max_in, "max_output_per_call": max_out,
                   "configured_max_output_tokens": MAX_OUTPUT_TOKENS},
        "cost_local_successful_responses_usd": round(local_cost, 6),
        "cost_hidden_retry_p95_estimate_usd": round(hidden_p95, 6),
        "cost_hard_upper_bound_for_log_observed_sdk_retries_usd":
            round(hidden_hard, 6),
        "cost_hard_bound_scope": ("hard with respect to LOG-OBSERVED SDK retries "
                                  "only; not an unconditional bound over every "
                                  "possible HTTP transmission -- an unlogged "
                                  "transmission lies outside it. See "
                                  "cost_structural_ceiling_usd for the bound "
                                  "that does not depend on logging."),
        "cost_upper_bound_usd": round(local_cost + hidden_hard, 6),
        "cost_structural_ceiling_usd": round(local_cost + ceiling_hidden, 6),
        "cost_note": ("local cost is provable from returned usage. The p95 figure "
                      "is a CONSERVATIVE ESTIMATE, not a maximum. The hard bound "
                      "charges every LOG-OBSERVED retry at the largest observed "
                      "input and the configured {}-token output ceiling; it is "
                      "hard only over those logged retries, not over every "
                      "possible transmission.".format(MAX_OUTPUT_TOKENS)),
    }


def reconcile(entries, dashboard_usd=None):
    local = sum(e["cost_local_successful_responses_usd"] for e in entries)
    p95_est = sum(e["cost_hidden_retry_p95_estimate_usd"] for e in entries)
    upper = sum(e["cost_upper_bound_usd"] for e in entries)
    out = {
        "conditions": len(entries),
        "total_local_cost_usd": round(local, 6),
        "total_hidden_retry_p95_estimate_usd": round(p95_est, 6),
        "total_hard_upper_bound_for_log_observed_sdk_retries_usd":
            round(upper - local, 6),
        "total_upper_bound_usd": round(upper, 6),
        "all_within_cap": all(e["log_reconstructed_within_cap"] for e in entries),
        "all_structural_ceilings_within_cap": all(
            e["structural_ceiling_within_cap"] for e in entries),
        "max_log_reconstructed_http": max(
            (e["log_reconstructed_observed_http_count"] or 0) for e in entries),
    }
    if dashboard_usd is None:
        out["dashboard_reconciliation"] = (
            "NOT PERFORMED: no provider dashboard/project usage figure was "
            "supplied. Reading it requires an authenticated console session, "
            "which this offline audit does not have. The upper bound above is "
            "the figure to compare against it.")
        out["dashboard_usd"] = None
    else:
        out["dashboard_usd"] = float(dashboard_usd)
        out["within_upper_bound"] = float(dashboard_usd) <= upper
        out["delta_vs_local_usd"] = round(float(dashboard_usd) - local, 6)
        out["dashboard_reconciliation"] = (
            "dashboard {:.4f} vs local {:.4f} (upper bound {:.4f})".format(
                float(dashboard_usd), local, upper))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True)
    parser.add_argument("--report", default=None)
    parser.add_argument("--dashboard-usd", type=float, default=None,
                        help="provider dashboard total, if known, to reconcile")
    args = parser.parse_args(argv)

    entries = []
    for name in sorted(os.listdir(args.root)):
        directory = os.path.join(args.root, name)
        if not os.path.isdir(directory) or not name.startswith("beta_"):
            continue
        if not os.path.exists(os.path.join(directory, "decisions.csv")):
            continue
        entries.append(reconstruct(directory))

    summary = reconcile(entries, args.dashboard_usd)
    report = {"kind": "selma_http_reconstruction", "read_only": True,
              "price_table_date": PRICE_TABLE.date,
              "sdk_max_retries_default": SDK_MAX_RETRIES_DEFAULT,
              "conditions": entries, "summary": summary}

    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w") as handle:
            json.dump(report, handle, indent=2, sort_keys=True, default=str)
        print("report: {}".format(args.report))

    print("\n{:<20} {:>10} {:>8} {:>14} {:>7} {:>10} {:>10}".format(
        "condition", "adapterAtt", "sdkRtr", "logReconHTTP", "cap-ok",
        "localCost", "hardUpper"))
    for e in entries:
        print("{:<20} {:>10} {:>8} {:>14} {:>7} {:>10} {:>10}".format(
            e["condition"], e["adapter_attempts"],
            e["logged_sdk_internal_retries"],
            e["log_reconstructed_observed_http_count"],
            "yes" if e["log_reconstructed_within_cap"] else "NO",
            round(e["cost_local_successful_responses_usd"], 4),
            round(e["cost_upper_bound_usd"], 4)))
    print("\nlocal ${:.4f} | p95 ESTIMATE ${:.4f} | HARD bound for LOG-OBSERVED "
          "retries ${:.4f} | total ${:.4f}".format(
              summary["total_local_cost_usd"],
              summary["total_hidden_retry_p95_estimate_usd"],
              summary["total_hard_upper_bound_for_log_observed_sdk_retries_usd"],
              summary["total_upper_bound_usd"]))
    print("  (hard bound is scoped to logged retries; structural ceiling is the "
          "logging-independent figure)")
    print(summary["dashboard_reconciliation"])
    return 0 if summary["all_within_cap"] else 1


if __name__ == "__main__":
    sys.exit(main())
