# SELMA Stage D — offline provider/cache implementation record

**Date:** 2026-08-16
**Branch:** `stage-d-provider-cache`
**Parent commit:** `b9ffb9c0c331f8578616fce9a7ab518388c34f78` (`b9ffb9c`,
`selma: document Stage D provider cache feasibility`), itself a child of the
verified Stage C commit `50b550d`.

**No API key was used. No live API call was made. No training was run.** The
OpenAI adapter is exercised exclusively through injected fake clients, and a
suite-wide network guard makes that a property of the test run rather than a
convention. The Stage B detector is untouched and the verified Stage C PBRS
semantics are unchanged (§8).

---

## 1. What was implemented

| Path | Role |
| --- | --- |
| `src/selma/cache.py` | canonical scheme-B key, SHA-256 digest, SQLite store, corruption detection, credential scrubbing |
| `src/selma/gateway.py` | lookup → call → validate → write ordering, live-call budget, cache modes |
| `src/selma/accounting.py` | the eleven counters, their supporting detail, and the invariant checker |
| `src/selma/openai_provider.py` | Responses-API adapter: lazy SDK import, strict structured output, candidate-id selection, total error handling |
| `src/selma/provider_registry.py` | vendor-aware provider registry, kept out of the frozen `provider.py` |
| `src/selma/modes.py` | Stage D configuration validation (cache mode, path, cap) |
| `src/selma/guidance.py` | four additive, `accounting is None`-gated hooks; no PBRS change |
| `src/runners/episode_runner.py` | constructs accounting/cache/gateway; asserts the ledger every episode |
| `src/config/default.yaml` | ten new keys, all inert by default |
| `analysis/selma_stage_d_offline_smoke.py` | durable offline smoke, fake client only |
| `tests/conftest.py` | suite-wide network guard |
| `tests/test_selma_stage_d_{cache,provider,accounting,runner}.py` | 110 tests |

`src/selma/detector.py` is **unmodified**: `git diff -- src/selma/detector.py`
produces no output.

---

## 2. Provider contract (frozen for this offline slice)

| Setting | Value |
| --- | --- |
| provider | `openai` |
| api | Responses API (`client.responses.create`) |
| model | **`gpt-5.4-nano-2026-03-17`** — exact dated snapshot |
| reasoning effort | `none` |
| store | `false` |
| structured output | strict JSON Schema (`strict: true`, `additionalProperties: false`) |
| tools | none (`tools=[]`) |
| streaming | disabled (`stream=False`) |
| max output tokens | 512 (configurable) |
| timeout | 30 s per attempt (configurable) |
| retries | 1 after the first attempt (configurable) |
| prompt version | `selma_guidance_prompt_v1` |
| response schema version | `selma_guidance_response_schema_v1` |

**The model snapshot is provisional** until the development-seed
provider-quality pilot, but the exact dated snapshot is used everywhere —
configuration, cache key, tests. **An unversioned alias is rejected at
construction** (`gpt-5.4-nano`, `…-latest` and bare `gpt-5.4` all raise): a
floating alias would silently change the model behind cache keys that claim
otherwise.

**Disabled by default.** `selma_provider` remains `none` and `selma_mode`
remains `off` in `default.yaml`. Off, shadow and deterministic-mock behaviour is
byte-for-byte unchanged, and no OpenAI import occurs on any of those paths
(asserted by `test_no_openai_sdk_is_imported_by_a_mock_run`, which runs three
real episodes and then checks `sys.modules`).

### 2.1 The model never invents anything

Code builds a deterministic list of `candidate_assignment_ids` — one
`a<agent>@<row>,<col>` per (recipient, offered loading cell) pair — and the JSON
Schema enumerates exactly those ids. The model returns **only ids**. Code then:

* maps ids back to `(agent_id, cell)` and rejects unknown, duplicate, or
  two-ids-for-one-agent selections;
* derives each **role** itself from the agent's real position versus the
  assigned cell;
* fixes `event_id`, `target_food_id`, `recipients` and `helper_coalition` from
  the request.

So agents, coordinates, foods, rewards, `beta`, TTL and the potential are all
code-fixed. The existing deterministic Stage C validator then has the final say
on every response — **including every cache hit**.

### 2.2 Credentials

The key is read **only** from `OPENAI_API_KEY`, only inside the lazily-invoked
client factory. There is no `api_key` parameter (passing one is a `TypeError`),
no YAML/Sacred/CLI path, no committed value. It is never logged, never included
in an exception message, and never stored: `_scrub` drops any metadata field
whose name contains an authorization/secret/token/header marker before anything
reaches disk, and a test asserts a sentinel key never appears in the SQLite file
or in captured output. Requesting the provider without the variable raises
`MissingApiKeyError` naming the variable.

### 2.3 Error handling — zero heuristic fallback

| Situation | Reason code | Retried? | Result |
| --- | --- | --- | --- |
| timeout | `provider_timeout` | yes | no guidance |
| rate limit | `provider_rate_limit` | yes | no guidance |
| transport / connection error | `provider_transport_error` | yes | no guidance |
| refusal | `provider_refusal` | no | no guidance |
| incomplete response | `provider_incomplete` | no | no guidance |
| malformed / schema failure | `provider_malformed` | one bounded retry | no guidance |
| unknown or duplicate assignment id | `provider_malformed` | no | no guidance |
| budget exhausted | `budget_exhausted` | n/a | no guidance |
| read-only cache miss | `cache_read_only_miss` | n/a | no guidance |

Every path yields **no active subgoal ⇒ Φ = 0 ⇒ F_t = 0**, never a locally
invented subgoal. Substituting a heuristic would confound "LLM guidance helped"
with "the heuristic helped" — precisely the contrast condition C5 exists to
measure. Exceptions are classified by class name, so classification works
identically for the real SDK and for a fake, with no import-time dependency.

---

## 3. Cache: scheme B

Episode-independent, absolute-state, `food_position` never `food_id`, **no
translation, rotation, reflection or symmetry normalisation** (scheme D remains
rejected — its transform proofs do not exist). Canonical JSON, SHA-256 digest,
and a **full canonical-key comparison after every hash match**: the digest is an
index, never an identity.

### 3.1 Included key fields (21)

`cache_schema_version`, `cache_key_version`, `request_schema_version`,
`response_schema_version`, `prompt_version`, `provider`, `model_snapshot`,
`reasoning_effort`, `condition`, `grid_shape`, `event_type`, **`food_position`**,
`food_level`, `trigger_agents` (id + level + position), `helper_agents`
(id + level + position), `recipient_agents` (id + level + position),
`level_deficit`, `required_extra_level`, `candidate_loading_cells`,
`candidate_assignment_ids`, `allowed_roles`.

Agent entries are sorted by id, cells are sorted, roles are sorted — so input
ordering cannot change the key (`test_canonical_ordering_produces_the_same_key`).

### 3.2 Excluded key fields (16)

`episode_id`, `t_env`, `episode_step`, `event_id`, **`food_id`** (it is
`(episode_id, row, col)` — its first element is episode identity), `seed`,
`query_timestamp`, `created_at`, `provider_request_id`, `latency_s`,
`retry_count`, `attempt_index`, `token_usage`, `wall_clock`, `learner_state`,
`raw_team_reward`.

Each is either episode identity, a logging-only artefact, or a measurement of
the call rather than of the question. Including any of them would only prevent
reuse. `INCLUDED_KEY_FIELDS` and `EXCLUDED_KEY_FIELDS` are module constants and
are asserted disjoint and exhaustive, so the key cannot drift silently.

`event_id` and `target_food_id` are episode-scoped and therefore not stored in
the key; a cache hit is **re-bound** to the live request's values, and the
validator then re-checks them, so a wrong binding fails closed.

### 3.3 Storage

Persistent SQLite (`sqlite3`, standard library — no new dependency). One
database per condition/seed/run; `run_identity` is
`condition|seed<N>|provider|model` and reopening another run's writable database
**raises** unless `selma_cache_allow_shared_write=True` is set deliberately.

Each row stores: `key_digest`, full `canonical_key`, cache schema and key
versions, prompt and response-schema versions, provider, requested model,
returned model, reasoning effort, canonical request JSON, validated response
JSON, scrubbed provider metadata, a `row_checksum`, and `created_at`
(**metadata, not part of the key**). Writes are single atomic transactions and
are idempotent. A `cache_meta` table carries the schema version, and opening a
database written under a different schema raises.

**Corruption detection:** `row_checksum` is a SHA-256 over the canonical key,
request JSON, response JSON and metadata. On lookup the checksum is recomputed;
a mismatch is a logged miss (`corrupt_row`), as are a full-key mismatch
(`full_key_mismatch`), a schema mismatch, and unparseable JSON
(`undeserialisable_response`). A hit that cannot be rebuilt into a
`GuidanceResponse` is counted as a **stale hit** and treated as a miss.

### 3.4 Modes

| `selma_cache_mode` | Behaviour |
| --- | --- |
| `off` (default) | no lookup, no write. Live calls still possible if a provider is configured. |
| `read_write` | lookup → on miss call (if permitted) → validate → write. |
| `read_only` | lookup only. **A miss never calls the provider**; it yields no guidance and `cache_read_only_miss`. Opening a non-existent database raises, and writing raises. |

Cache hits never increment `live_api_call_count`, never consume the budget, and
report usage as *not applicable* rather than zero.

---

## 4. Live-call budget

`selma_live_call_cap`, provisional default **2000** (not experimentally frozen).

* The cap applies **only to actual network attempts**. Cache hits do not consume
  it. An offline provider (the deterministic mock) never consumes it — the
  runner sets its cap to `None`, because nothing it does is an HTTP request.
* **Declared rule: every HTTP request attempt consumes the cap, retries
  included.** A retry is a real request that costs real money and latency, so it
  must be visible to the cap. `retry_count` is tracked separately, so the two
  readings are always distinguishable.
* The cap is enforced *inside* the retry loop via an attempt hook, so a retry can
  be denied mid-flight (`test_cap_stops_a_retry_mid_flight`).
* Once exhausted, **the cache lookup still happens** — a warm entry is still
  served (`test_budget_exhaustion_still_performs_the_cache_lookup`).
* A miss after exhaustion produces **no guidance, F_t = 0**, reason
  `budget_exhausted`, and never a heuristic substitute.
* Greedy evaluation is structurally incapable of a call: the controller is not
  consulted in `test_mode`, so no lookup, call or activation can occur.
* `selma_live_call_cap` must be a non-negative integer; `0` forbids live calls
  entirely. There is no unlimited setting.

---

## 5. Accounting

The eleven counters from the feasibility report §7, cumulative per run and never
reset at an episode boundary: `detector_event_count`,
`provider_request_considered_count`, `cache_lookup_count`, `cache_hit_count`,
`cache_miss_count`, `live_api_call_count`, `successful_response_count`,
`rejected_response_count`, `retry_count`, `fallback_count`,
`guidance_activation_count`.

Supporting detail: `terminal_suppressed_event_count`, `missing_position_count`,
`stale_hit_count`, `budget_exhausted_count`, `timeout_count`,
`rate_limit_count`, `transport_error_count`, `refusal_count`,
`incomplete_count`, plus `fallback_reasons` and `stale_reasons` histograms.

Recorded provenance: provider, requested model, **returned model identifier(s)**,
prompt version, cache-key version, cache-schema version, response-schema
version, request ids, input tokens, **cached** input tokens, output tokens,
reasoning tokens, total latency, and `selma_price_table_date`.

**Cost is never hard-coded.** `cost_usd` is always `None`; token counts plus a
separately versioned price-table date are stored so cost can be recomputed from
the official rate card. `as_dict()` contains no `cost_usd` key at all.

### 5.1 Invariants

All fourteen from the feasibility report are implemented in
`SelmaAccounting.check_invariants`, asserted after **every episode** by the
runner, and tested. Two required factual corrections, which have been applied to
the feasibility report as marked in-place corrections (its conclusions are
unchanged):

* **Invariant 2** holds when the cache is enabled; with `cache_mode=off` there is
  by definition no lookup and the counter must be `0`.
* **Invariants 4 and 9** are stated in terms of **call opportunities** —
  `cache_miss_count` when the cache is on, `provider_request_considered_count`
  when it is off — because with the cache off there are live calls and no
  misses. Invariant 4 also relaxes to a two-sided bound when a call is
  suppressed by budget exhaustion, read-only mode or an offline provider.

Both were found by the new tests, not by inspection.

A third refinement: `rejected_response_count` counts every response body that
arrived and was then rejected, including one the *adapter* parsed and rejected
(malformed, refusal, incomplete) before the controller ever saw an object. A
call that produced no body (budget, read-only, timeout, transport) is not a
rejection. This keeps invariant 8 exact.

Missing-position events are counted in `missing_position_count` and are
deliberately **not** counted as considered requests or as fallbacks — no request
was ever built — which is exactly invariant 1's documented gap.

---

## 6. Fails-first evidence

### 6.1 Baseline: against the pre-change tree

The four new test files were copied into a clean `git archive` of the parent
commit `b9ffb9c` and run there. None of `cache.py`, `gateway.py`,
`accounting.py`, `openai_provider.py` or `provider_registry.py` exists in that
tree:

```
exit=2   3 collection errors   "3 errors in 0.29s"
```

Every Stage D test fails against the pre-change behaviour, because the
implementation does not exist.

### 6.2 Mutations: each defect reintroduced

Each defect was reintroduced into a copy of the current tree (outside the
repository) and the focused suites re-run. Baseline of the unmutated copy: 99
passed.

| Reintroduced defect | Caught | Example failing test |
| --- | ---: | --- |
| D1 episode-scoped `food_id` back in the key | 9 | `test_episode_id_does_not_affect_the_cache_key` |
| D2 agent positions dropped from the key | 3 | `test_changed_absolute_position_is_a_miss` |
| D3 digest trusted without the full canonical-key comparison | 1 | `test_hash_match_with_different_canonical_key_is_rejected` |
| D4 row checksum not verified | 1 | `test_corrupted_row_is_rejected` |
| D5 responses cached before validation | 6 | `test_rejected_responses_are_never_cached` |
| D6 cache hits counted as live calls | 5 | `test_cache_hits_do_not_consume_the_cap` |
| D7 read-only miss falls through to a live call | 1 | `test_read_only_miss_never_calls_and_yields_no_guidance` |
| D8 cap checked with `>` instead of `>=` | 1 | `test_cap_stops_a_retry_mid_flight` |
| D9 retries do not consume the cap | 1 | `test_every_attempt_including_retries_consumes_the_cap` |
| D10 unversioned model alias accepted | 1 | `test_unversioned_model_alias_is_rejected` |
| D11 invented assignment ids accepted | 1 | `test_invented_assignment_id_is_rejected` |
| D12 duplicate-id check removed | **0** | — see below |
| D13 metadata stored without scrubbing credentials | 1 | `test_metadata_never_stores_credentials` |
| D14 budget exhaustion substitutes heuristic guidance | 4 | `test_fallback_never_substitutes_heuristic_guidance` |

**D12 is honestly reported as uncaught.** Removing the explicit
`len(selected) != len(set(selected))` check changes nothing observable, because a
repeated id necessarily repeats its agent and is therefore caught one line later
by the per-agent check. The two are redundant by construction and no test can
distinguish them; the explicit check is retained as defence in depth, not
because it is separately covered. (This is the same class of finding as Stage
C's D6.)

Two defects in the delivered code were found by these tests *before* the code was
finished, not by review: the invariant-4/9 `cache_miss_count` error and the
`rejected_response_count` classification error, both described in §5.1.

---

## 7. Test totals

| Suite | Tests |
| --- | ---: |
| `tests/test_selma_stage_d_cache.py` | 35 |
| `tests/test_selma_stage_d_provider.py` | 32 |
| `tests/test_selma_stage_d_accounting.py` | 33 |
| `tests/test_selma_stage_d_runner.py` | 10 |
| **Stage D focused total** | **110** |
| **Full repository suite** | **408 passed** (was 298 at `50b550d`) |

```
/home/aum/selma/venv/bin/python -m pytest -q
408 passed, 112 warnings in 41.68s
```

**No test touches the network.** `tests/conftest.py` installs a session-scoped
autouse guard that replaces `socket.socket.connect`, `socket.socket.connect_ex`
and `socket.create_connection` with functions that raise. The guard is itself
tested (`test_the_network_guard_is_active`), so "no network access" is a property
of the run, not a convention. Every provider test uses an injected fake client.

---

## 8. Stage C regression

```
/home/aum/selma/venv/bin/python analysis/verify_stage_c.py
RESULT: PASS -- 36 checks, every learning-relevant artifact identical.   (exit 0)
```

All 36 β = 0 equivalence checks still pass after the Stage D changes, and the
reported guidance counters are **identical** to the pre-Stage-D run: 6 provider
calls, 6 queries, 6 activations, 0 replacements, 6 expiries (`ttl` 4,
`terminal` 2), 0 rejections, cumulative unscaled PBRS term `0.19766666666666666`,
cumulative shaping contribution `0.0`. The PBRS equations, the lifecycle
ordering, the TTL semantics and the terminal rule are untouched; the four hooks
added to `guidance.py` are gated on `accounting is not None` and record only.

---

## 9. Offline smoke

```
/home/aum/selma/venv/bin/python analysis/selma_stage_d_offline_smoke.py
RESULT: PASS -- 31 checks.   (exit 0)
```

Real gateway, real SQLite cache, real adapter code path, real
`GuidanceController`; injected fake client; no socket, no key, no environment
step.

| Step | Result |
| --- | --- |
| 1. first request | miss, exactly **1** simulated live call, guidance activated, `F_t = -0.057750`, 1 row written |
| 2. identical situation, episode 987 | **hit**, 0 additional calls, 0 additional HTTP requests, guidance activated, still 1 row |
| 3. changed absolute position | miss, 1 further call, 2 rows |
| 4. read-only, unseen situation | miss, **0 calls**, no guidance, `cache_read_only_miss`, `F_t = 0`; a read-only *hit* is still served with 0 calls |
| 5. budget exhausted (cap 1) | lookup still performed, **0 calls**, no guidance, `budget_exhausted`, `F_t = 0`, nothing extra cached |
| 6. reconciliation | invariants hold for all three ledgers |

Final `read_write` ledger: 3 detector events, 3 considered, 3 lookups, 1 hit,
2 misses, **2 live calls**, 3 successful, 3 activations, 0 rejected, 0 retries,
0 fallbacks; 236 input / 34 output / 128 cached tokens; returned model
`gpt-5.4-nano-2026-03-17` ×2.

---

## 10. Limitations

1. **No live API call has ever been made from this code.** Every behaviour above
   is verified against fakes. The real Responses API may differ in field names,
   refusal shape, incomplete-detail structure, error classes or structured-output
   strictness. The adapter reads fields defensively (attribute *or* mapping) and
   classifies errors by class name, but nothing here is evidence about the real
   service.
2. **The model snapshot is provisional.** `gpt-5.4-nano-2026-03-17` has not been
   evaluated for this task. Whether it can select assignment ids competently is
   exactly what the pilot must establish.
3. **Prompt quality is unmeasured.** `selma_guidance_prompt_v1` has never been
   run against a model. The spec's ≥ 95 % parse-and-ground gate on 50–100 saved
   trigger states has not been executed.
4. **The 2000 live-call cap is provisional**, carried over from the feasibility
   audit's position-free bound on a superseded partial run. It is not
   experimentally frozen and is unsupported for 10x10.
5. **The real cache hit rate is still unknown.** The audit's blocker — agent
   positions absent from archived evidence — is unchanged. The infrastructure
   now *measures* it, but the measuring run (feasibility §11 step 1) has not been
   made, because no training was permitted here.
6. **`store=false` means no server-side retention**, so any audit trail must come
   from our own cache rows and counters. If a provider later requires retention
   for some feature, that is a new decision.
7. **Retry/backoff behaviour is untested against real rate limits.** Backoff is
   exponential from a configurable base; no jitter is implemented.
8. **No concurrency.** The gateway assumes the single-threaded episode runner.
   Per-event single-flight coalescing (feasibility §3.5) is not implemented
   because nothing in this build is concurrent.
9. **No cost figure is produced**, deliberately (§5).
10. **Nothing here establishes that guidance helps learning.** No performance
    claim is made or supported.

---

## 11. Decisions still requiring the live pilot

| # | Decision | Current provisional value | Needs |
| --- | --- | --- | --- |
| 1 | model snapshot | `gpt-5.4-nano-2026-03-17` | provider-quality pilot on a development seed |
| 2 | temperature / seed support | not set (reasoning `none`) | verification that the chosen model exposes a reproducible seed at all |
| 3 | prompt version | `selma_guidance_prompt_v1` | ≥ 95 % parse-and-ground on 50–100 saved trigger states |
| 4 | structured-output strictness | strict JSON Schema with an id enum | confirmation the real endpoint honours the enum |
| 5 | timeout | 30 s | measured latency distribution |
| 6 | retry limit / backoff | 1 retry, 0.5 s exponential | measured rate-limit behaviour |
| 7 | live-call cap | 2000 (8x8 @ 2M) | the measured key distribution; unsupported for 10x10 |
| 8 | cache-key version | `selma_cache_key_v1` | frozen once the pilot confirms the field set |
| 9 | price-table date | unset | the rate card in force at run time |
| 10 | β | 0.0 (default) | feasibility D1 — freeze before any β > 0 run |
| 11 | training budget, seed count, C3 scope | open | feasibility D5 / D7 / D6 |

TTL remains frozen at 15. The frozen detector `wf15_m3_t5_ibon` is unchanged.

---

## 12. Explicit confirmations

* **No API key was used or requested.** No `OPENAI_API_KEY` was read, set, or
  needed at any point.
* **No live API, network or external service call was made.** The suite-wide
  guard proves it for the tests; the smoke uses a fake client; the only network
  operations in this work are the authorised `git push`.
* **No training was run.** The runner-level tests step the LBF environment
  in-process for 40 short episodes as unit tests — no training run, no Sacred
  experiment, no seed 1–5, no checkpoint. `analysis/verify_stage_c.py` runs the
  same 40-episode development-seed comparison it always has, as the Stage C
  regression gate.
* **`src/selma/detector.py` is unmodified.**
* **Stage C PBRS semantics are unchanged**, evidenced by 36/36 equivalence with
  bit-identical counters.
* **The recovery-plan document was not touched, staged, moved or deleted.**
