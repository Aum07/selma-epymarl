# SELMA Stage E — beta recalibration: execution and acceptance record

**Date:** 2026-08-18
**Branch:** `stage-e-beta-recalibration-runs`
**Design commit:** `824f2e98070c2741984070bd977d9a7dfbac114a`
**Execution source commit (all six accepted runs):** **`53d9a2e61c59b59e52aab8e21401f9eaaebe3a94`**

> **Development-seed calibration only.** Two seeds support calibration, **not** a
> statistical performance claim and not a generalisation claim. Nothing here is
> a final result.

---

## 1. Outcome

The preregistered locked rule was applied **once**, after all six conditions had
completed and passed every technical gate.

> ## Selected: **`beta = 0.00`**

`beta = 0` is explicitly permitted to win by rule 6, and it did: it has the
highest cross-seed mean normalised greedy-success AUC over the 18 frozen
post-onset evaluation points. **No positive `beta` outperformed the unshaped
baseline on this calibration.**

### The ranking is not stable across seeds — read §5 before using it

This is the single most important caveat and it is not a hedge:

| | seed 91 | seed 92 |
|---|---|---|
| ordering | `beta=0.05` (0.376) > `beta=0.00` (0.236) > `beta=0.10` (0.209) | `beta=0.00` (0.450) > `beta=0.10` (0.298) > `beta=0.05` (0.196) |

`beta = 0.05` is **best on one seed and worst on the other**. The largest
within-`beta` seed spread (**0.214**) *exceeds* the entire between-`beta` range
of means (**0.090**). The mechanical rule produced a determinate answer, and that
answer stands as preregistered — but the evidence underneath it is
noise-dominated at n = 2, and this record does not claim that shaping is harmful.

---

## 2. The six accepted conditions

All six: exit 0, Sacred `COMPLETED`, valid manifest and `COMPLETED` marker,
**100 % guidance coverage**, caps respected, and zero on every safety counter.

| condition | source | exit | final `t_env` | coverage | attempts / 13 500 | cost / $3.49 | wall |
|---|---|---|---|---|---|---|---|
| `beta_0p00_seed91` | original | 0 | 1 500 002 | 100 % | 6 245 | $1.2903 | 221 min |
| `beta_0p00_seed92` | original | 0 | 1 500 006 | 100 % | 8 676 | $1.7927 | 302 min |
| `beta_0p05_seed91` | original | 0 | 1 500 008 | 100 % | 6 470 | $1.3370 | 235 min |
| `beta_0p05_seed92` | original | 0 | 1 500 046 | 100 % | 7 128 | $1.4730 | 258 min |
| `beta_0p10_seed91` | **recovery** | 0 | 1 500 012 | 100 % | 7 991 | $1.6513 | 282 min |
| `beta_0p10_seed92` | **recovery** | 0 | 1 500 008 | 100 % | 8 125 | $1.6790 | 286 min |

Every condition, on every safety counter: **0** physically invalid activations,
**0** invented plan ids, **0** greedy-evaluation provider calls, **0**
reward-reconciliation errors, **0** non-finite values, telescoping residual
**exactly 0.0**, **0** non-zero terminal φ, returned model exclusively
`gpt-5.4-nano-2026-03-17`, **0** budget-exhausted decisions.

All six re-hashed against their recorded manifests: **INTACT**.

---

## 3. Metrics

### Primary — cross-seed mean normalised greedy-success AUC (18 frozen points, 600k–1 450k)

| `beta` | seed 91 | seed 92 | **primary (mean)** | secondary (final-5 mean) | mean detector events |
|---|---|---|---|---|---|
| **0.00** | 0.23559 | 0.44971 | **0.34265** | 0.531 | 11 718 |
| 0.05 | 0.37618 | 0.19618 | 0.28618 | 0.524 | 9 759 |
| 0.10 | 0.20853 | 0.29765 | 0.25309 | 0.396 | 11 429 |

Every condition matched all **18/18** frozen evaluation points. The secondary
metric was not needed: the primary produced a strict ordering with no tie within
1e-12.

### Guidance funnel

| condition | detector events | activations | distinct keys | cache hits |
|---|---|---|---|---|
| `beta_0p00_seed91` | 8 900 | 8 819 | 6 244 | 2 575 |
| `beta_0p00_seed92` | 14 535 | 14 414 | 8 675 | 5 739 |
| `beta_0p05_seed91` | 9 300 | 9 216 | 6 470 | 2 746 |
| `beta_0p05_seed92` | 10 217 | 10 131 | 7 128 | 3 003 |
| `beta_0p10_seed91` | 11 245 | 11 131 | 7 991 | 3 140 |
| `beta_0p10_seed92` | 11 613 | 11 522 | 8 125 | 3 397 |

Detector-event frequency does **not** vary monotonically with `beta`; the
seed-to-seed variation within a `beta` (8 900 vs 14 535 at `beta = 0`) is larger
than the variation between `beta` values.

---

## 4. Infrastructure incident, exclusion and recovery

### What happened

| time (18 Aug) | evidence |
|---|---|
| 03:47:01 | controller logged wave 3 launch (pids 46227, 46252) |
| **06:08:00 / 06:08:04** | both `beta=0.10` run logs stop mid-stream on a **successful HTTP 200** — no traceback, no error |
| **06:08:08** | `systemd-journald: Under memory pressure, flushing caches` |
| 06:18:17 | kernel OOM killer fires; **one** victim, `pid 16316 (2.1.233)`, 7.8 GB RSS, invoked by a `HeapHelper` thread. **No Python victim was logged** |
| 06:19:57–58 | orderly WSL session shutdown |
| 07:52:11 | new boot |

**Stated as evidence, not as a proven cause:** the runs stalled within four
seconds of the first kernel memory-pressure report, and the session was shut
down orderly eleven minutes later. The kernel logs every OOM kill and logged no
Python victim, so the exact terminating mechanism for our two processes is
**not** established.

**OOM victim identity — suspected, not confirmed.** The victim's process name
`2.1.233` matches the Claude Code version recorded in
`~/.claude/.last-update-result.json` as `version_from` in a 2.1.233 → 2.1.234
update at 2026-08-17T20:22Z; the interactive session predated that update, so it
would still be running the 2.1.233 binary. `HeapHelper` is a V8/Node heap thread
and Claude Code is Node-based; 7.8 GB fits a ~12-hour session. The process is
gone, so no direct process-to-binary link survives. Reported as **suspected**.

### The excluded partial runs

Preserved untouched as infrastructure-failure evidence at
`analysis/out/stage_e_recal/beta_0p10_seed9{1,2}/`. **No `COMPLETED` marker, no
manifest, Sacred status still `RUNNING`.** Their partial learning outcomes are
**never** combined with the accepted runs.

| excluded partial | last `t_env` | attempts spent | cost forfeited |
|---|---|---|---|
| `beta_0p10_seed91` | 849 745 / 1 500 000 (56.6 %) | 4 781 | $0.9880 |
| `beta_0p10_seed92` | 851 035 / 1 500 000 (56.7 %) | 4 714 | $0.9741 |

### Why the rerun was infrastructure-driven, not outcome-driven

* the trigger was a host memory/shutdown event, evidenced above;
* **no performance value of any `beta=0.10` condition had been inspected** — the
  locked analysis had not been run and could not be, since it requires all six;
* the **complete pair** was rerun together, from `t = 0`, never a single seed;
* the frozen configuration, caps, model, prompt, horizon and execution commit are
  unchanged;
* no interrupted artifact was reused, resumed, renamed or overwritten.

Exact continuation was impossible in any case: `save_model=False`, so no
checkpoint, optimizer, replay-buffer, RNG or scheduler state was ever written.
The protocol's bar for resume could not be met.

### Recovery controller provenance

* path: `analysis/out/stage_e_recal_wave3_rerun/recovery_controller.py` (gitignored)
* **SHA-256:** `2d5e99a3e24ca42c352597cb19067245838fdb0e01001e8dbecfc3838c38d34e`
* it **imports the frozen runner** and calls its own `build_condition`/
  `build_command`, so child argv is derived, never retyped
* no tracked runtime source was modified — deliberately **no** `--only-beta`
  flag, which would have made the replacements execute a different tree

**Command-equivalence proof.** Each replacement argv was compared token-by-token
with the original interrupted argv: **35 tokens each, exactly 5 differing**, all
path-valued (`selma_cache_path`, `selma_guidance_log_csv`,
`selma_decision_log_csv`, `selma_log_csv`, `local_results_path`). All **22**
scientific/runtime tokens identical, including `seed=91`/`seed=92`,
`selma_beta=0.1`, `t_max=1500000`, `selma_live_call_cap=13500`,
`selma_cost_cap_usd=3.49`, `test_interval=50000`, `test_nepisode=100`,
`selma_variant_id=wf15_m3_t5_ibon`, `selma_provider_max_retries=1`,
`selma_provider_max_output_tokens=512`.

`git diff --exit-code 53d9a2e -- src/ analysis/selma_stage_e_recal_runner.py`
returned **exit 0** immediately before launch.

---

## 5. Limitations

1. **n = 2 seeds. No significance, no generalisation.** This is calibration.
2. **The ranking is unstable across seeds** (§1). `beta=0.05` is best on seed 91
   and worst on seed 92; the within-`beta` spread (0.214) exceeds the
   between-`beta` range (0.090). The selection is the rule's determinate output,
   not a demonstrated effect.
3. **`beta = 0` winning does not establish that shaping is harmful.** It
   establishes that no positive `beta` beat the baseline *here*.
4. `beta = 0.25` was excluded by the design's exploratory narrowing and is not
   re-examined.
5. One environment, one algorithm, one horizon.
6. The `beta=0.10` pair ran ~9 h later than the others on a rebooted host. The
   configuration and commit are identical; no systematic effect is expected, but
   it is not a same-session comparison.

---

## 6. API accounting

### `live_api_call_count` is **adapter attempts, not HTTP transmissions**

`attempt_hook` fires once per adapter attempt, immediately before `_create()`.
The OpenAI SDK's own transport retries occur *inside* that call, below the hook,
so they never reach the ledger.

* **adapter** retry loop: `selma_provider_max_retries=1` — **counted**
* **SDK** transport retry: `OpenAI(...)` was constructed with only `api_key` and
  `timeout`, so **`max_retries` was inherited = 2** — **not counted**

**All six accepted runs inherited SDK `max_retries = 2`.** The correction to
`max_retries=0` is applied only *after* this record, in a separate commit, and
was deliberately withheld during execution so that all six conditions shared one
runtime.

### Log-reconstructed observed HTTP counts

`log_reconstructed_observed_http_count = adapter_attempts + logged SDK retries`.
This is **observed from SDK log lines, not a guaranteed true HTTP count** — no
packet-level evidence exists, and an unlogged transmission lies outside it.

| condition | adapter attempts | logged SDK retries | log-reconstructed HTTP | ≤ 13 500 |
|---|---|---|---|---|
| `beta_0p00_seed91` | 6 245 | 7 | 6 252 | ✅ |
| `beta_0p00_seed92` | 8 676 | 10 | 8 686 | ✅ |
| `beta_0p05_seed91` | 6 470 | 1 | 6 471 | ✅ |
| `beta_0p05_seed92` | 7 128 | 2 | 7 130 | ✅ |
| `beta_0p10_seed91` (recovery) | 7 991 | 0 | 7 991 | ✅ |
| `beta_0p10_seed92` (recovery) | 8 125 | 0 | 8 125 | ✅ |

### Cost, kept strictly separated

| quantity | accepted six | excluded partials | total operational spend |
|---|---|---|---|
| local successful-response cost (provable from returned usage) | **$9.2234** | $1.9621 | **$11.1855** |
| p95-based hidden-retry **estimate** (not a maximum) | $0.0033 | $0.0000 | $0.0033 |
| **hard bound for log-observed SDK retries** | $0.0130 | $0.0000 | $0.0130 |

The hard bound is hard **with respect to logged retries only** — it charges each
at the largest observed input and the configured 512-token output ceiling. It is
**not** an unconditional bound over every possible transmission.

**Dashboard reconciliation: NOT PERFORMED.** It requires an authenticated console
session this offline audit does not have. The figure to compare against is the
total upper bound, **$11.1985**.

Authorised: $3.49/condition and $20.94 global. Accepted six: **$9.2234**.
Including the forfeited partials: **$11.1855** — within authorisation.

---

## 7. Fails-first evidence for the retry correction

Captured against the uncorrected tree at execution commit `53d9a2e`, before the
correction existed.

* **exact command:**
  `PYTHONPATH=src ~/selma/venv/bin/python -m pytest tests/test_selma_sdk_retry_accounting.py -v --no-header -p no:cacheprovider`
* **result: 2 failed, 11 passed**
* failing tests:
  * `tests/test_selma_sdk_retry_accounting.py::test_the_client_factory_disables_sdk_transport_retries`
  * `tests/test_selma_sdk_retry_accounting.py::test_the_constructed_client_reports_zero_sdk_retries`
* **transcript SHA-256:** `630803dd5e2b212ef8dbc6cc0296a2ad4ac3ac73203d6a9ac3d6cf2b0d123707`
* transcript: `analysis/out/stage_e_recal/fails_first_evidence/pytest_failsfirst.txt` (gitignored)

Those two tests are made to pass by the **next** commit, never by weakening an
assertion.

---

## 8. Credential safety

Scoped scans across both roots found **no real credential**. Two benign classes
were identified and classified without printing any value:

1. the synthetic `DO-NOT-LEAK` sentinel echoed by pytest into the fails-first
   transcript (gitignored, from test source);
2. a 13-character `sk-<epoch>` **substring inside a Cloudflare `set-cookie`
   value** in one *response*-header dump, in the **excluded** interrupted
   `beta_0p10_seed92` run log. Corroborated: **0** `Authorization` occurrences
   and **0** real-key-shaped tokens in that file. Request headers are never
   logged.

The recovery root scanned **0** hits. No credential appears in any command,
manifest, report, Sacred config or tracked file.

---

## 9. Explicit statements

* **Execution source commit for all six accepted runs: `53d9a2e`.** Verified
  byte-identical before wave 3 and again before the recovery launch.
* All six inherited **SDK `max_retries = 2`**; the correction lands only in the
  following commit.
* Waves 1–2 were never rerun, modified or overwritten; re-hashed **INTACT**.
* The interrupted `beta=0.10` pair is preserved untouched and **excluded**.
* The locked analysis was run **exactly once**, after all six completed.
* No condition was inspected, omitted, rerun or modified on the basis of its
  performance.
* No final/evaluation seed was used or inspected; seed 99 remains detector
  held-out.
* **n = 2 supports calibration, not a statistical performance claim.**
