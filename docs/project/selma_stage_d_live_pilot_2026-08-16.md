# SELMA Stage D — live provider contract validation

> ## ⚠ POST-PILOT CORRECTION (added 2026-08-16, after this record was written)
>
> **A defect was found in the validator this pilot ran against. Read this before
> reading anything below.** Nothing below has been edited or removed: the original
> record and the commit that produced it (`10417e6`) stand as immutable historical
> evidence of what was actually observed and claimed at the time.
>
> **What still holds.** API compatibility passed. The exact snapshot
> `gpt-5.4-nano-2026-03-17` answered; `store=false`, `reasoning: none`, strict
> structured output, no tools and no streaming were all accepted; strict
> structured-output parsing succeeded on 10/10 responses; every selected id was
> one of the offered ids (offered-ID grounding, 10/10); the cache, budget and
> accounting behaviour and every ledger reconciled. Those results are unaffected.
>
> **What did not hold.** *Strong physical-plan validity failed on 1 of the 10
> pilot fixtures.* On `two_cells_only` the model returned `a0@5,4` **and**
> `a1@5,4` — both agents assigned to the same loading cell, which LBF cannot
> realise. It is recorded truthfully in §8.2 below as a limitation, but it was
> under-classified: it is a semantic-validity **failure**, not a stylistic
> preference the model declined to follow.
>
> **The "zero invalid guidance activated" gate in §6.2 and §7 must be read
> narrowly.** It meant *zero guidance rejected by the validator as it then
> existed*. That validator checked each assignment independently against the
> offered cells and never examined the selected combination, so it could not
> express "no two agents share a cell" and duly accepted the invalid plan. Under
> the corrected joint-plan validator the same response is rejected
> (`plan_duplicate_cell`). The gate as worded overstated what had been verified.
>
> **No training used the invalid plan.** This pilot ran no training, no learner
> and no environment; `beta` was `0.0` and `training_reward == raw_team_reward`
> on every transition. The invalid plan influenced no policy, no reward and no
> stored artefact.
>
> **What it motivated.** This discovery is the direct cause of the joint-plan
> contract: the code now enumerates only physically valid *complete* plans and the
> model selects exactly one by id, so a duplicate cell is not a bad answer to be
> caught but an answer that cannot be expressed. See
> [`selma_stage_d_joint_plan_acceptance_2026-08-16.md`](selma_stage_d_joint_plan_acceptance_2026-08-16.md).
>
> The cache written by this pilot is **not** reused: `CACHE_SCHEMA_VERSION` and
> `CACHE_KEY_VERSION` were both bumped, so a pilot-era database cannot be opened
> and a pilot-era digest cannot be recomputed.

**Date:** 2026-08-16
**Branch:** `stage-d-live-pilot`
**Starting commit:** `29459e3f1bfa1756e052ca38df4e5741fc16ab0e` (*selma: add Stage D provider cache infrastructure*)
**Scope:** API compatibility, structured output, grounding, cache behaviour and accounting.

> **This is not a performance result.** No training was run, no beta calibration was
> performed, `beta` was `0.0` in every code path, and nothing here says anything about
> whether LLM guidance helps learning. It establishes only that the live provider path
> *works as specified* and that its ledgers reconcile.

---

## 1. What was validated

| | |
|---|---|
| Provider | `openai` (`src/selma/openai_provider.py`) |
| API surface | Responses API, `client.responses.create` |
| Model | `gpt-5.4-nano-2026-03-17` — exact dated snapshot, requested and returned |
| SDK | `openai==3.1.0` (Python 3.10, `~/selma/venv`) |
| Prompt version | `selma_guidance_prompt_v2` (**bumped**, see §3) |
| Response schema version | `selma_guidance_response_schema_v1` (unchanged) |
| Cache key version | `selma_cache_key_v1` (unchanged composition) |
| Cache schema version | `selma_cache_schema_v1` (unchanged) |
| `beta` | `0.0` throughout |
| Live attempts | **11** total, retries included (cap 30, hard ceiling 50) |

---

## 2. Implementation audit (read-only, before any live call)

The whole Stage D path was traced end to end: detector event → code-built candidate
assignments → `build_request` → cache lookup → provider call → strict response parsing →
Stage C `validate` → guidance activation → PBRS.

| # | Property | Verdict | Where |
|---|---|---|---|
| 1 | Prompt defines LBF and the coordination decision | **corrected** — see §3 | `openai_provider.SYSTEM_PROMPT` |
| 2 | Agent identity, roles, target food/level, positions are code-fixed | pass | `_interpret` rebuilds `event_id`, `target_food_id`, `recipients` from the request; role derived from the real position |
| 3 | Model may select only an offered `candidate_assignment_id` | pass | schema `enum` + independent re-check against `allowed` |
| 4 | Model cannot invent coordinates, agents, roles, targets, beta, TTL, rewards, actions | pass | response schema is `{assignment_ids, confidence}` only, `additionalProperties: false` |
| 5 | Candidates carry enough information for a real strategic choice | **corrected** — see §3 | `build_prompt` |
| 6 | Multi-candidate fixtures are genuinely distinct | pass | 10 frozen fixtures, 4–12 candidates each |
| 7 | No episode identity or unstable run metadata in the prompt | pass | no `event_id`, `food_id`, `episode_id`, seed, step or timestamp; asserted by test |
| 8 | Prompt and schema deterministic, canonical, versioned | pass | `json.dumps(sort_keys=True, separators=(",",":"))`; versions enter the cache key |
| 9 | Every response revalidated, cache hits included | pass | `GuidanceController._offer` calls `validate` on every response |
| 10 | Rejected / refused / malformed ⇒ zero guidance, `F_t = 0` | pass | verified live-pilot offline gate, 7 cases |
| 11 | No heuristic fallback | pass | every failure path returns `None` |
| 12 | Greedy evaluation structurally cannot call the provider | pass | `episode_runner.run`: `selma_guidance = self.selma_guidance if not test_mode else None` |
| 13 | Model aliases rejected | pass | `-YYYY-MM-DD` suffix required at construction |
| 14 | `store=false`, no tools, no streaming, reasoning `none`, max output 512, timeout 30 s, one retry | pass | asserted offline **and observed accepted live** |

### Singleton candidate sets — current behaviour, unchanged, decision required

**Current behaviour:** the provider is queried for *every* eligible coalesced event,
including one whose candidate set is a single id. No frozen contract — not the Stage C
implementation contract, not the Stage D feasibility report — mentions singleton sets at
all, so nothing requires the query and nothing forbids it.

**This was deliberately left unchanged.** Short-circuiting singletons is *not*
behaviour-neutral: with a query, a refusal or a transport failure yields zero guidance;
without one, guidance always activates. That changes the activation rate, and therefore
the shaping a training run receives — a scientific-policy change, not a cleanup. It is
recorded here as an **open decision for the training stage**, not resolved by this pilot.

It does not affect any result below: every pilot fixture offers ≥ 4 candidates, so no
singleton request was ever issued.

---

## 3. Prompt corrections (and why they were needed)

The audit examined the *runtime prompt text*, not just its version name, and found two
defects that made the offered choice ungrounded.

**Defect 1 — the candidate ids were unexplained.** v1 sent
`"candidate_assignment_ids": ["a0@4,3", "a1@3,4", …]` and nowhere said what the strings
meant. The model was asked to choose between opaque tokens whose referents — which agent,
which cell — it had to infer.

**Defect 2 — the LBF rule was never stated.** v1's system prompt named "a Level-Based
Foraging grid world" but never said that a food is collected only when adjacent agents
LOAD simultaneously and their levels sum to at least the food's level. Without it, the
prompt never explains why a helper is needed or what makes an assignment good.

**Correction (prompt v2), in `src/selma/openai_provider.py` only:**

* `SYSTEM_PROMPT` states the level-sum/adjacency/simultaneous-LOAD rule and names the
  decision explicitly;
* `build_prompt` adds a `candidates` array spelling each offered id out as
  `{assignment_id, agent_id, loading_cell}`, **derived from the id list by the same
  parser the response check uses**, so the described set and the selectable set cannot
  drift apart;
* `instructions` states the id format, the exact number of ids to return, and the
  objective.

**No new authority was given to the model.** The response schema, the validator, the
cache key composition and the code-fixed entities are all untouched.

**Versioning.** `PROMPT_VERSION` → `selma_guidance_prompt_v2`. `prompt_version` is one of
`INCLUDED_KEY_FIELDS`, so every cache digest changes and pre-correction entries can never
be served. `RESPONSE_SCHEMA_VERSION` and `CACHE_KEY_VERSION` are deliberately *not*
bumped: the response shape and the key composition did not change, and bumping them would
falsely imply they had.

Eight fails-first tests were added in `tests/test_selma_stage_d_provider.py` (§6 of that
file); five failed against the pre-change tree and pass after.

---

## 4. Dependency change

`openai` was absent from `~/selma/venv`. Installed `openai==3.1.0` into that venv only;
the system Python was not touched.

**New manifest: `selma_requirements.txt`.** The SDK is *optional by construction* — it is
imported lazily inside the client factory so `off` / `shadow` / `deterministic_mock` runs
never import it, and `tests/test_selma_stage_d_provider.py` asserts that absence. Adding
it to `requirements.txt` would put it in every base install and weaken that guarantee, so
it follows the repository's existing pattern for optional features
(`pac_requirements.txt`, `env_requirements.txt`). Pinned exactly, because acceptance of
the Responses API fields the adapter sends is a property of a specific SDK version.

SDK signature check (offline): `model`, `input`, `text`, `reasoning`, `tools`, `store`,
`stream`, `max_output_tokens`, `timeout` are all accepted by `Responses.create`.

---

## 5. The live-pilot harness

`analysis/selma_stage_d_live_pilot.py` — durable and separately executable.

* refuses every live call unless `--confirm-live` is passed;
* default attempt cap **1**; `--max-attempts` may lower it, and is clamped to the hard
  ceiling **50** by the transport itself;
* drives the **production** path: `provider_registry` → `ProviderGateway` →
  `SqliteGuidanceCache` → `SelmaAccounting` → `GuidanceController.apply_transition`.
  Validation is never bypassed and no shaping code is reimplemented;
* no training, no environment, no learner; `beta = 0.0`;
* cache database in a temporary directory, removed on exit;
* every attempt is appended to the ledger **before the request leaves**, so retries,
  timeouts and transport failures all consume budget;
* sanitised JSON report; non-zero exit on any failed gate.

**Credential handling.** The key is read only by the production client factory from
`OPENAI_API_KEY`, inside the subprocess that makes the request. The script never reads
`~/.config/selma/openai.env` (verified `mode 600`, regular file, owned by the invoking
user, by `stat` metadata only), never accepts a key as an argument, and checks only that
the variable is non-empty. Exceptions are recorded as **class name plus HTTP status code
only** — never a message, never headers.

---

## 6. Results

### 6.1 One-call compatibility probe (Phase 3)

One attempt, fixture `base_deficit2`.

| Check | Result |
|---|---|
| Attempts | 1 |
| Model requested / returned | `gpt-5.4-nano-2026-03-17` / `gpt-5.4-nano-2026-03-17` |
| Response status | `completed` |
| `store=false` accepted | yes |
| `reasoning: {effort: none}` accepted | yes |
| Strict structured output accepted | yes (`text.format.strict = true`) |
| Tools / streaming | `[]` / `false` |
| Parsed | yes |
| Selected ids ⊆ offered (8 offered) | yes — `a0@4,3`, `a1@3,4` |
| Stage C validator | accepted |
| Token accounting | 774 input / 35 output |
| Latency | 2.51 s |
| Identical request repeated | cache hit, **0** extra live calls |

### 6.2 Capped fixed-fixture pilot (Phase 4)

Ten frozen 8×8 fixtures varying agent positions and levels, food positions and levels,
trigger/helper identity, level deficit (1–4), candidate cell sets (2, 3 and 4 cells),
recipient count (2 and 3), plus a translated and a column-mirrored copy of the base
layout — both of which scheme B correctly treats as **distinct** keys, since it applies no
symmetry normalisation.

| Metric | Value |
|---|---|
| Request attempts (retries included) | **10** |
| Retries | 0 |
| Successful API responses | 10 |
| HTTP/API failures (sanitised) | none |
| Refusals | 0 |
| Parse successes / failures | 10 / 0 |
| Grounded valid-id selections | 10 / 10 (100 %) |
| Invented or unoffered ids | **0** |
| Validator acceptances / rejections | 21 / 0 |
| Cache hits / misses / writes | 11 / 10 / 10 |
| Guidance activations | 21 |
| Candidate-count distribution | 4 → 2, 6 → 2, 8 → 15, 12 → 2 |
| Input / output tokens (this run) | 7 682 / 357 |
| Latency (min / median / p95 / max) | 0.71 / 1.19 / 1.98 / 1.98 s |

**Cache reuse.** All 10 fixtures re-issued with different `episode_id`, `event_id`,
`food_id` and step index: **10/10 served from cache, 0 extra live calls.** Scheme B
carries no episode identity.

**Read-only phase.** 10 read-only hits served with **0** attempts; 1 read-only miss on a
never-seen situation returned `cache_read_only_miss`, made **0** calls and activated no
guidance.

**Offline invalid-response fixtures** (fakes — never requested from the API): invented id,
duplicate id, missing recipient, malformed body, refusal, incomplete, transport failure.
All seven: zero guidance activated, zero shaping, zero cache rows written, invariants hold.

### 6.3 Accounting ledgers

| Counter | Probe phase | Pilot phase | Read-only phase |
|---|---|---|---|
| `detector_event_count` | 2 | 19 | 11 |
| `provider_request_considered_count` | 2 | 19 | 11 |
| `cache_lookup_count` | 2 | 19 | 11 |
| `cache_hit_count` | 1 | 10 | 10 |
| `cache_miss_count` | 1 | 9 | 1 |
| `live_api_call_count` | 1 | 9 | **0** |
| `successful_response_count` | 2 | 19 | 10 |
| `rejected_response_count` | 0 | 0 | 0 |
| `retry_count` | 0 | 0 | 0 |
| `fallback_count` | 0 | 0 | 1 (`cache_read_only_miss`) |
| `guidance_activation_count` | 2 | 19 | 10 |

`returned_models` is `{gpt-5.4-nano-2026-03-17: N}` in every phase — the provider never
silently changed model. Ledger, provider and accounting attempt counts reconcile exactly.

### 6.4 Tokens and cost methodology

| | Input | Output |
|---|---|---|
| Probe run | 774 | 35 |
| Pilot run | 7 682 | 357 |
| **Total** | **8 456** | **392** |

`reasoning_tokens = 0` and `cached_input_tokens = 0` in every phase.

**Cost was not calculated.** No cited, dated price table was supplied
(`--price-table-date` empty, `selma_price_table_date = ""`), and the adapter deliberately
reports `cost_usd = None` rather than embedding a price. Tokens are reported verbatim;
monetary cost must be recomputed from a dated price table.

The spend bound was enforced structurally, not estimated: 11 attempts against a hard
ceiling of 50, each bounded by 512 output tokens and an observed ≈ 850-token request. At
any plausible nano-tier price this is orders of magnitude below the USD $1 authorisation;
no stop condition was approached.

---

## 7. Acceptance gates

All 28 gates reported by the harness passed in both runs (exit 0).

| Gate | Result |
|---|---|
| ≥ 95 % of successful responses parse and ground to an offered id | **PASS** (100 %) |
| Zero invented ids activated | **PASS** |
| Zero invalid guidance activated | **PASS** |
| Zero calls on cache hits | **PASS** |
| Zero calls on read-only misses | **PASS** |
| Zero calls in greedy evaluation | **PASS** (structural guard + offline test) |
| Zero budget overruns | **PASS** (11 attempts, cap 30, ceiling 50) |
| All accounting ledgers reconcile | **PASS** (all three phases) |
| No secret exposure | **PASS** |
| Complete Stage C regression passing | **PASS** |
| Exact model snapshot available | **PASS** |
| SDK accepts implemented Responses API fields | **PASS** |
| `reasoning: none`, `store=false`, strict structured output accepted | **PASS** |
| Cache written only after validation | **PASS** (10 rows = 10 validated live responses) |
| Beta remained zero, rewards unaltered | **PASS** |

### Test and verifier results

| Command | Result |
|---|---|
| `pytest tests/test_selma_stage_d_live_pilot.py` | 32 passed |
| `pytest tests/test_selma_stage_d_*.py` | 150 passed |
| `pytest -q` (full suite) | **448 passed** (baseline 408, +8 prompt-audit, +32 live-pilot) |
| `analysis/selma_stage_d_offline_smoke.py` | PASS — 31 checks |
| `analysis/verify_stage_c.py` | PASS — 36 checks |
| `analysis/selma_stage_d_live_pilot.py` (offline dry run) | PASS — exit 0, 0 attempts |
| Live probe (`--max-attempts 1`) | PASS — exit 0, 1 attempt |
| Live pilot (`--max-attempts 30`) | PASS — exit 0, 10 attempts |

The session-wide socket guard in `tests/conftest.py` was left intact and active; pytest
made no network connection.

---

## 8. Limitations and open decisions

1. **Nothing is claimed about strategic quality.** The pilot measures compatibility,
   grounding, caching and accounting. Ten fixtures, one snapshot, one day.

2. **The model can assign two recipients to the same loading cell, and the frozen
   validator accepts it.** On fixture `two_cells_only` (cells `(5,4)`, `(5,6)`; agent 0
   already on `(5,4)`), the model returned `a0@5,4` *and* `a1@5,4`. Two agents cannot
   occupy one cell in LBF, so that assignment is physically unrealisable — yet it is
   *valid* under the Stage C contract, which checks that each loading cell was offered but
   never that the cells are distinct. Prompt v2 asks for distinct cells as a preference;
   the model did not follow it here. **Open decision:** whether `validate` should reject
   duplicate loading cells. That is a change to the frozen Stage C validator and was
   deliberately *not* made in this pilot.

   > **RESOLVED — and reclassified.** This was a semantic-validity *failure*, not a
   > limitation. The answer was: not a duplicate-cell check bolted onto the existing
   > per-assignment validator, but a change of contract — the code enumerates
   > complete, physically valid joint plans and the model selects one by id. See the
   > correction notice at the top of this document and
   > [`selma_stage_d_joint_plan_acceptance_2026-08-16.md`](selma_stage_d_joint_plan_acceptance_2026-08-16.md).
   > The same fixture now returns `a0@5,4; a1@5,6` — distinct cells.

3. **Assignments are frequently valid but not distance-optimal.** On `far_helper` and
   `deficit3_single_helper` the model moved an agent off a loading cell it already
   occupied. Recorded as an observation only; no quality claim either way.

4. **Singleton candidate sets** — unchanged, and an open decision (§2).

5. `cached_input_tokens` was 0 throughout; prompt caching behaviour at scale is untested.

6. Latency was measured on 10 calls from one location; the distribution is indicative only.

7. The pilot never provoked a live refusal, rate limit or malformed body — those paths are
   covered by offline fakes, by design, rather than by spending attempts to make the API
   misbehave.

---

## 9. Explicit statements

* **No training was run.** No learner, no environment, no episode rollout, no Sacred run.
* **No beta calibration was performed.**
* **`beta` was `0.0` in every phase**, asserted per-record; `training_reward` equalled
  `raw_team_reward` exactly on every transition.
* **This is not a performance result.**
* **`src/selma/detector.py` was not touched** — zero diff.
* **`docs/project/selma_seed91_recovery_and_rerun_plan.md` was not touched** — unchanged,
  untracked and unstaged throughout.
* **The API key was never read, printed, copied, hashed, logged or committed.**
  `~/.config/selma/openai.env` was inspected by `stat` metadata only (regular file, mode
  600, owned by the invoking user) and was sourced only into the subprocess making the
  authorised request.
* Raw and transient pilot output lives in `analysis/out/live_pilot/`, which is gitignored.
  No SQLite database, response dump, Sacred output, cache or temporary file was staged.
