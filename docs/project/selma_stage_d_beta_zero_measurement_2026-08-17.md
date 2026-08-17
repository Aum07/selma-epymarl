# SELMA Stage D — beta-zero online measurement (seed 91, 200k steps)

**Date:** 2026-08-17
**Branch:** `stage-d-beta-zero-measurement`
**Starting commit:** `7b85b2d530d2b9db3ec7115249b89e669408e897` (*selma: enforce valid joint guidance plans*)

> **This is not a performance result and it is not beta selection.**
> `beta` was exactly `0.0` on all 200,050 logged transitions, so the training
> reward was the raw environment reward on every step and the run is
> behaviourally plain QMIX. Nothing here reads episode return, and nothing here
> recommends a `beta`. The purpose is to measure **how often the live provider
> is actually consulted, how much question reuse exists, and what that costs**,
> so a later beta pilot can be budgeted from evidence rather than guesswork.

---

## 1. Why beta must be zero here

Shaping and measurement cannot be mixed in one run. With `beta > 0` the guidance
would alter the stored reward, which alters the policy, which alters which
coordination stalls the detector sees — so the very quantity being measured
(how many distinct questions arise, and how often) would be a function of the
shaping being measured. At `beta = 0` the detector, the joint-plan enumerator,
the cache and the provider all run exactly as they will in a shaped run, while
the policy trajectory stays that of the unshaped baseline. The call and cache
statistics are therefore attributable to the environment and the detector, not
to a shaping strength that has not yet been chosen.

`beta = 0` also makes the run its own control: `training_reward` must equal
`raw_team_reward` bit-for-bit, and any deviation is a defect rather than an
effect.

---

## 2. Exact configuration

| | |
|---|---|
| Algorithm | QMIX (`--config=qmix`) |
| Environment | `lbforaging:Foraging-8x8-2p-2f-coop-v3` (`--env-config=gymma`) |
| Time limit | 50 |
| `t_max` | 200 000 |
| Seed | **91** |
| `common_reward` | True (existing configuration) |
| CUDA | False |
| `selma_mode` | `active` |
| Detector variant | `wf15_m3_t5_ibon` (frozen) |
| Provider | `openai` |
| Model | `gpt-5.4-nano-2026-03-17` (exact dated snapshot) |
| Prompt / response schema | `selma_guidance_prompt_v3` / `selma_guidance_response_schema_v2` |
| Cache key / schema | `selma_cache_key_v2` / `selma_cache_schema_v2` |
| Reasoning effort | `none` |
| `store` / tools / streaming | `false` / none / disabled |
| Strict structured output | enabled |
| Max output tokens | 512 |
| Timeout / retries | 30 s / 1 |
| Cache mode | `read_write`, new empty isolated database |
| `beta` | **0.0** |
| TTL | 15 (frozen) |
| Live-attempt cap | **200**, retries included |
| Cost ceiling | **$0.25**, checked before every attempt |
| Greedy-evaluation provider calls | structurally zero |

### Command

```
python src/main.py --config=qmix --env-config=gymma with \
  env_args.key=lbforaging:Foraging-8x8-2p-2f-coop-v3 env_args.time_limit=50 \
  seed=91 t_max=200000 use_cuda=False common_reward=True \
  selma_mode=active selma_variant_id=wf15_m3_t5_ibon selma_provider=openai \
  selma_beta=0.0 \
  selma_cache_mode=read_write selma_cache_path=<OUT>/guidance_cache.sqlite3 \
  selma_live_call_cap=200 selma_cost_cap_usd=0.25 \
  selma_price_table_date=2026-08-17 \
  selma_price_input_usd_per_mtok=0.20 \
  selma_price_cached_input_usd_per_mtok=0.02 \
  selma_price_output_usd_per_mtok=1.25 \
  selma_provider_timeout_s=30.0 selma_provider_max_retries=1 \
  selma_provider_max_output_tokens=512 \
  selma_guidance_log_csv=<OUT>/guidance.csv \
  selma_decision_log_csv=<OUT>/decisions.csv \
  selma_log_csv=<OUT>/episodes.csv local_results_path=<OUT>/results \
  test_interval=20000 log_interval=20000 \
  learner_log_interval=20000 runner_log_interval=20000
```

`<OUT>` = `analysis/out/beta_zero_measurement_seed91/` (gitignored). The resolved
Sacred configuration is archived there as `sacred_config.json`; the Sacred run
directory is `results/sacred/qmix/lbforaging:Foraging-8x8-2p-2f-coop-v3/52/`
(also gitignored), status `COMPLETED`.

**The secret cannot enter Sacred.** The key is read only from `OPENAI_API_KEY`,
which was sourced into the training subprocess alone. Sacred records environment
variables only when they are named in `SETTINGS.HOST_INFO.CAPTURED_ENV`, which
this build never populates, so the captured `ENV` is `{}`. Pinned with a canary
by `test_sacred_host_info_captures_no_environment_variables`, and confirmed by
scanning the run directory: no `sk-`, `authorization` or `api_key` material.

---

## 3. Instrumentation added

The pre-existing logs could not answer this measurement's central question --
**how many *distinct* questions were asked** -- because no semantic cache-key
digest was ever recorded. Several other required fields were also absent.

**New: `src/selma/pricing.py`.** Dated price tables, cost recomputation, and a
conservative `CostBudget` ceiling. No rate is hard-coded: a ceiling without a
dated table is a configuration error, pinned by
`test_no_price_is_hard_coded_in_the_source_tree`.

**New: `SelmaDecisionCsvWriter` (`selma_decision_log_csv`)** -- one durable row
per detector event. A per-event log rather than more columns on the
per-transition log, because the digest, plan count and provider outcome are
properties of an *event* and one transition may carry several. It records the
decision bucket, candidate plan count, semantic digest, cache result,
attempts/retries, outcome, selected plan id, validator verdict, physical
validity, activation, input/cached-input/output tokens, returned model, latency,
request id, `greedy_eval` and `beta`.

**Critically:** the gateway records the digest *before* the budget and mode
checks, so a multiple-plan event refused a live call because the budget ran out
**still contributes its question**. That is what makes distinct-question counting
independent of whether the cache was allowed to fill -- and this run proves it
mattered: 94 of 304 questions arrived after the cap bound.

**Extended: the per-transition guidance log** gained
`query_tokens_cached_prompt`, `query_returned_model`, `detector_events`,
`decisions`, `zero_plan_decisions`, `singleton_decisions`,
`multiple_plan_decisions` and `greedy_eval`.

**Two real defects found by the new tests, before the run:**

1. the gateway exposed neither `last_returned_model` nor `last_attempts`, so
   those fields would have been empty for every event; and
2. naively passing them through would have made a **cache hit inherit the
   previous live call's attempts, model and request id**. A hit now reports zero
   attempts, no request id, and the model recorded in the stored row.

**Log-schema safety.** Both durable logs share `_CsvLogWriter`, which refuses to
append under a header written by a different build and refuses a row whose keys
do not match its columns. The stale-header defect that previously corrupted a
verifier read remains impossible for every log.

**Not touched:** `src/selma/detector.py`, `joint_plans.py`, `provider.py`,
`cache.py`, `openai_provider.py` are byte-identical to `7b85b2d` -- the detector,
joint-plan enumeration, prompt v3, response schema and physical-validation
semantics are unchanged.

---

## 4. Cost contract

Price table, **checked 2026-08-17**, source
`https://developers.openai.com/api/docs/models/gpt-5.4-nano`:

| GPT-5.4 nano | USD per 1M tokens |
|---|---|
| input | 0.20 |
| cached input | 0.02 |
| output | 1.25 |

```
cost = input_non_cached * 0.20/1e6 + cached_input * 0.02/1e6 + output * 1.25/1e6
```

`input_tokens` as reported by the provider **includes** the cached ones, so the
non-cached remainder is what is billed at the full rate; computing it once in
`PriceTable.cost_usd` avoids double-counting.

**Provider-side cached input is not the application's SQLite response cache.**
They are unrelated: a cached-input token is a prompt prefix the vendor
recognised on a request that really happened; a SQLite hit means no request
happened at all and costs nothing. Cached input was **0** throughout this run.

Both ceilings were enforced before every attempt — `attempts < 200` and a
conservative projection (worst input seen, floor 4 000, plus the full 512-token
output allowance) staying under $0.25. The call cap bound; the cost ceiling never
did.

---

## 5. Results

### A. Run integrity

| | |
|---|---|
| Final `t_env` | **200 049** |
| Episodes | 4 001 |
| Guidance-log rows | 200 050 |
| Decision-log rows | 304 |
| Terminal causes | `time_limit` × 4 001 |
| Process exit code | 0 (Sacred `COMPLETED`) |
| NaN / Inf | none |
| Reward-reconciliation errors | **0** |
| Malformed artifacts | none |
| Wall clock | ~13.5 minutes |

### B. Detector → joint-plan funnel

| Stage | Count | Rate |
|---|---|---|
| Detector events (coalesced) | 306 | — |
| Decisions logged | 304 | — |
| **Zero-plan events** | **0** | 0 % |
| **Singleton events** | **0** | 0 % |
| **Multiple-plan events** | **304** | 100 % |
| Missing-position events | 0 | 0 % |
| Eligible for an LLM decision | 304 | — |
| Guidance activations | **210** | 69.1 % of eligible |
| Not activated (budget exhausted) | 94 | 30.9 % |

Candidate joint-plan count was **6 for every single event** — `P(3,2)`. The
frozen coalition builder excludes a loader's own current cell from
`candidate_loading_cells`, so a 2-recipient event on an 8×8 grid is always
offered exactly three admissible cells. The zero-plan and singleton policies are
therefore **correct but entirely inert in this configuration**; they were
verified offline, not live.

### C. Cache behaviour

| | |
|---|---|
| Questions with a digest | 304 |
| Cache hits | 10 |
| Cache misses | 294 |
| Cache writes | 200 |
| **Distinct semantic keys** | **292** |
| Repeat questions | 12 |
| Keys used exactly once | 280 |
| Mean uses per distinct key | 1.041 |
| Keys reused across episodes | 2 |
| **Observed hit rate** | **3.29 %** |
| **Ideal replay hit rate** (offline, from ordered digests) | **3.95 %** |
| Top-10 key share | 6.6 % |

The most-used key was used **twice**. There is essentially no concentration: the
question distribution in early training is close to all-distinct.

The observed rate (3.29 %) falls just short of the ideal (3.95 %) for one
understood reason: after the 200-attempt cap bound, misses could no longer be
written, so 92 distinct questions never entered the cache and their later repeats
could not hit. Before the cap, observed and ideal were **identical** — scheme B
captured every reuse available to it.

### D. Time blocks

| Block | Transitions | Detector events | Multiple-plan decisions | New keys | Repeats | Live attempts | Cache hits | Activations | Cumulative distinct |
|---|---|---|---|---|---|---|---|---|---|
| 0–50k | 50 000 | 132 | 132 | 127 | 5 | 127 | 5 | 132 | 127 |
| 50–100k | 50 000 | 47 | 47 | 45 | 2 | 45 | 2 | 47 | 172 |
| 100–150k | 50 000 | 41 | 39 | 37 | 2 | 28 | 2 | 30 | 209 |
| 150–200k | 50 000 | 86 | 86 | 83 | 3 | **0** | 1 | 1 | 292 |

| Block | Detector events / 1k steps | Multiple-plan decisions / 1k steps | Repeat rate |
|---|---|---|---|
| 0–50k | 2.640 | 2.640 | 3.79 % |
| 50–100k | 0.940 | 0.940 | 4.26 % |
| 100–150k | 0.820 | 0.780 | 5.13 % |
| 150–200k | 1.720 | 1.720 | 3.49 % |

**The event rate is not monotonic.** It falls 2.64 → 0.94 → 0.82 per 1k steps and
then rises again to 1.72. Coordination stalls do not simply disappear as the
policy improves. This is the single most important caveat on every projection
below: a rate estimated from any one block would have been wrong.

New keys per block declined only from 127 to 83, and the repeat rate stayed
between 3.5 % and 5.1 % — **the distinct-key curve shows no sign of saturating
within 200k steps.**

### E. Provider / API

| | |
|---|---|
| Request attempts (retries included) | **200** (cap 200) |
| Retries | 0 |
| Events that called | 200 |
| Successful responses | 200 |
| HTTP/API failures | none |
| Refusals | 0 |
| Parse + ground rate | **100 %** (200/200) |
| Validator acceptances / rejections | 210 / 0 |
| **Physically valid activations** | **210 / 210 (100 %)** |
| Physically invalid activations | **0** |
| Invented plan ids activated | **0** |
| Returned model | `gpt-5.4-nano-2026-03-17` × 200 (no substitution) |
| Budget-exhausted events | 94 |
| Latency (min / median / p95 / max) | 0.61 / 0.87 / 3.11 / 4.88 s |
| Input tokens | 175 400 |
| Cached input tokens | 0 |
| Output tokens | 5 000 |
| **Calculated cost** | **$0.04133** |

Cost per call: **$0.00020665**. Mean input 877 tokens/call, output 25.

**Budget exhaustion behaved exactly as specified** on 94 events: the cache lookup
still occurred, the semantic digest was still recorded, no attempt was made, no
guidance activated, and `F_t = 0`. The run continued to completion.

### F. Reward / PBRS verification

| Check | Result |
|---|---|
| Distinct `beta` values in the log | `['0.0']` |
| `beta` is 0.0 on every row | ✅ |
| Shaping contribution exactly zero | ✅ (max abs 0.0) |
| `training_reward == raw_team_reward` | ✅ (0 mismatches in 200 050 rows) |
| Non-zero unscaled PBRS terms | 2 916 (guidance really ran) |
| Cumulative unscaled PBRS | 6.9627 |
| Max abs unscaled PBRS | 0.5333 |
| Stored-terminal transitions | 4 001 |
| Terminal `phi_after_effective` all zero | ✅ |
| **Max telescoping residual** | **0.0 (exact)** |
| TTL distribution | 210 subgoals, decaying to 15 transitions served |
| Greedy-evaluation rows, decision log | **0** |
| Greedy-evaluation rows, guidance log | **0** |

The unscaled PBRS term is non-zero on 2 916 transitions, which proves guidance was
genuinely active and being priced — while the beta-weighted contribution stayed
exactly zero. The telescoping residual is exactly 0.0, so
`phi_after_effective(t) == phi_before(t+1)` held bit-for-bit throughout.

---

## 6. Budget projections for a 2M run

From the ordered semantic-key evidence. **Reported as a range, with each method
labelled**, because the two defensible extrapolations disagree.

| | Calls | Cost |
|---|---|---|
| **Measured, 200k** | 200 attempts (capped); **292 distinct questions** | $0.04133 |
| Naive linear ×10 | 2 920 | $0.60 |
| Final-block new-key rate (0.00166 keys/step) | 3 280 | $0.68 |
| **Conservative range, one 2M run** | **2 920 – 3 280** | **$0.60 – $0.68** |
| **Five seeds, isolated per-seed caches** | **14 600 – 16 400** | **$3.02 – $3.39** |

Note that the naive method is here the *lower* end, not the upper one: the final
block's new-key rate was **higher** than the run average, because the event rate
rose again after 150k. That is itself the finding — the usual assumption that
extrapolation is an overestimate does not hold on this evidence.

### Is the provisional 2 000-call cap sufficient?

**No.** A single 2M run needs an estimated 2 920–3 280 calls, so a 2 000-call cap
would exhaust at roughly 60–70 % of training and silently drop guidance for the
rest. Either the cap must rise to ~3 500 per run, or budget exhaustion must be
accepted as a designed condition and reported as such. It must not be left at
2 000 by default and discovered afterwards.

### Limitations of these projections

* **The 200k early-training distribution is not the same as a 2M distribution.**
  The policy state distribution changes as the agents learn, and this run shows
  the coordination-stall rate falling and then rising again within 200k. Extending
  either trend to 2M is extrapolation, not measurement.
* Reuse was measured on **one seed, one environment, one detector variant**.
* The distinct-key curve had not saturated at 200k, so the true 2M count could
  exceed the high end of the range if novelty persists.
* Only 200 of 304 questions were actually asked, so the cost figure is measured
  over the calls made and scaled by the *distinct-question* count — not observed
  directly at scale.
* Per-seed caches share nothing by design, so five seeds multiply calls by five.
  Seeds diverge, so the per-seed distinct-key counts will differ.

---

## 7. Acceptance gates

| Gate | Result |
|---|---|
| Process exits successfully | ✅ exit 0, Sacred `COMPLETED` |
| `t_env` reaches at least 200 000 | ✅ 200 049 |
| Exactly one live measurement condition ran | ✅ one process, one seed, one condition |
| Seed is 91 | ✅ |
| Environment and algorithm match the frozen configuration | ✅ |
| `beta` equals 0.0 everywhere | ✅ |
| Shaping contribution is exactly zero | ✅ |
| Training reward equals raw reward | ✅ 0/200 050 mismatches |
| Live attempts ≤ 200 including retries | ✅ exactly 200, 0 retries |
| Calculated cost ≤ $0.25 | ✅ $0.04133 |
| Exact model snapshot returned for every successful call | ✅ 200/200 |
| Zero invalid physical guidance activated | ✅ |
| Zero invented plan ids activated | ✅ |
| Zero provider calls on cache hits | ✅ |
| Zero provider calls for zero/singleton plans | ✅ (none occurred) |
| Zero provider calls during greedy evaluation | ✅ 0 rows in both logs |
| All accounting invariants reconcile | ✅ |
| All generated joint plans remain physically valid | ✅ 210/210 |
| Artifacts complete and non-empty | ✅ |
| No credential exposure | ✅ |
| Stage C regression remains passing | ✅ 36 checks |

Budget exhaustion occurred (94 events) and is accepted: its safe behaviour was
correct and the run continued to completion.

### Tests and verifiers

| Command | Result |
|---|---|
| `pytest tests/test_selma_stage_d_measurement.py` | 36 passed |
| `pytest tests/test_selma_beta_zero_report.py` | 28 passed |
| `pytest tests/test_selma_stage_d_*.py` | 267 passed |
| `pytest -q` (full suite) | **569 passed** |
| `analysis/selma_stage_d_offline_smoke.py` | PASS — 31 checks |
| `analysis/verify_stage_c.py` | PASS — 36 checks |
| `analysis/selma_stage_d_live_pilot.py` (dry run) | PASS — 0 attempts |
| `analysis/selma_beta_zero_measurement_report.py` | PASS — 19 gates |

The session-wide socket guard in `tests/conftest.py` remained intact; pytest made
no network connection.

---

## 8. Next recommended beta pilot — **design only, not executed**

Nothing below has been run, and no `beta` value is recommended here.

1. **Fix the call budget first.** Raise the per-run cap to ~3 500 for 2M, or
   accept exhaustion explicitly. Leaving it at 2 000 would silently disable
   guidance for the last third of training.
2. **Sweep beta on short runs before committing to 2M.** The measured activation
   pattern (210 activations, TTL 15, mean |unscaled PBRS| well under 1) is what
   determines how large `beta` can be before shaping dominates the raw reward.
   That arithmetic is available from this run's logs at any candidate `beta`
   without new API calls, since `pbrs_term_unscaled` is logged beta-free.
3. **Re-use the cache across the sweep only within a seed.** Per-seed isolation
   must be preserved; sharing across conditions would confound them.
4. **Keep an unshaped control at the same seed**, so any effect is measured
   against the trajectory this run already characterises.
5. **Do not select beta from this run.** It has no shaping and therefore no
   signal about shaping strength.

---

## 9. Explicit statements

* **This is not a performance result.** No episode return was read or reported.
* **This is not beta selection or beta calibration.** No `beta` was trialled,
  swept or recommended; `beta` was `0.0` throughout.
* **Exactly one live measurement run occurred**, seed 91, on the frozen
  environment and algorithm. No second seed, environment or condition was run.
* **`src/selma/detector.py` is unchanged** — zero diff from `7b85b2d`.
* **The joint-plan contract is unchanged** — `joint_plans.py`, `provider.py`,
  `cache.py` and `openai_provider.py` are all byte-identical to `7b85b2d`, so
  prompt v3, the response schema and physical-validation semantics are untouched.
* **`docs/project/selma_seed91_recovery_and_rerun_plan.md` was not touched** —
  unchanged, untracked and unstaged throughout.
* **The API key was never read, printed, copied, hashed or logged.**
  `~/.config/selma/openai.env` was inspected by `stat` metadata only (regular
  file, mode 600, owned by the invoking user) and sourced only into the training
  subprocess. The run log, the Sacred directory and both CSV logs were scanned:
  no `sk-`, `authorization` or `api_key` material. The only `bearer` occurrences
  in the SDK debug log are the boolean scheme flag `{'bearer_auth': True}`.
* Generated CSVs, the SQLite cache, Sacred runs, checkpoints and analysis outputs
  live in gitignored directories and were not staged.
