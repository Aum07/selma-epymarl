# SELMA Stage D — joint-plan guidance contract: correction and acceptance

**Date:** 2026-08-16
**Branch:** `stage-d-joint-plans`
**Starting commit:** `10417e6` (*selma: validate Stage D live provider contract*)
**Supersedes the guidance contract validated by:**
[`selma_stage_d_live_pilot_2026-08-16.md`](selma_stage_d_live_pilot_2026-08-16.md)

> **Not a performance result.** No training was run, no beta calibration was
> performed, `beta` was `0.0` throughout. This record establishes semantic
> validity of guidance, and nothing about whether guidance helps learning.

---

## 1. The defect

The Stage D live pilot returned, on fixture `two_cells_only`:

```
a0@5,4   a1@5,4
```

Both agents assigned to the same loading cell. LBF cannot place two agents in one
cell, so the guidance was physically unrealisable — and the validator **accepted
it**, so it activated as a real subgoal.

Reproduced against the pre-correction tree before anything was changed:

```
both agents assigned to cell (5,4)
validator verdict : ACCEPTED
distinct cells    : False
```

### Root cause

`validate()` checked each assignment *independently*:

```python
for a in response.assignments:
    if tuple(a.loading_cell) not in set(ctx.candidate_loading_cells):
        return ValidationResult(False, "loading_cell_not_offered", None)
```

Every individual assignment was legal. `a0 → (5,4)` was offered; `a1 → (5,4)` was
offered. **"No two agents share a cell" is a property of the combination, and a
per-assignment loop cannot express it.** The response contract had the same shape
defect: a *list* of independently composable per-agent ids, from which an invalid
combination could be assembled out of individually valid parts.

### Why prompt-only enforcement was insufficient

Prompt v2 already said *"prefer giving different recipients different loading
cells."* The model did not follow it, and nothing downstream cared. A prompt is a
request, not an enforcement:

* it is advisory — no mechanism makes a violation impossible or even detectable;
* it is unverifiable — a passing gate proves the sentence was in the prompt, not
  that the output obeyed it;
* it is not defence in depth — when it fails, nothing else catches the failure;
* and it is the wrong layer. Physical realisability of a grid-world plan is a
  deterministic property of code-known state. It belongs in code.

The correction is therefore structural, not textual: **make the invalid answer
unrepresentable**, then check it anyway.

---

## 2. The joint-plan contract

### Representation

A **joint plan** is the complete mapping of every required recipient to its own
loading cell (`src/selma/joint_plans.py`):

```python
JointPlan(plan_id="plan_3", assignments=(PlanAssignment(0, (4, 3)),
                                         PlanAssignment(1, (3, 4))))
```

### Enumeration

`enumerate_joint_plans(recipients, candidate_cells, food_position, grid_shape)`:

1. filter the offered cells to those inside the grid, orthogonally adjacent to the
   target food, and not the food cell itself (`admissible_cells`);
2. if there are fewer admissible cells than recipients, return `()` — distinct
   cells are required, so the situation is unsolvable;
3. enumerate every injective assignment of sorted recipients to the sorted
   admissible cells (`permutations`);
4. sort the resulting plans and assign positional ids `plan_0, plan_1, …`
   **after** sorting.

Nothing iterates a `set` or `dict` to produce output, so the plans and their ids
are byte-identical on any machine and under any hash seed. Enumeration is bounded
by construction: a food has at most four orthogonal neighbours, so the answer set
is at most `P(4, k)` — 2 to 24 plans in practice.

### Response contract

The model returns exactly one `selected_plan_id`. The JSON Schema has **no array
anywhere**, so a list of composable parts is not merely discouraged — it cannot
be expressed:

```json
{"type": "object", "additionalProperties": false,
 "required": ["selected_plan_id", "confidence"],
 "properties": {"selected_plan_id": {"type": "string", "enum": ["plan_0", …]},
                "confidence": {"type": "number"}}}
```

The prompt states each plan's id, its full agent→cell mapping and a rendering of
that mapping (`a0->(4,3), a1->(3,4)`), so an id is never an opaque token.

---

## 3. Validator rules

`validate()` runs in a fixed order and short-circuits on the first failure with a
stable reason code. There is no partial application.

| # | Check | Reason on failure |
|---|---|---|
| 1 | response present, right type, schema version | `no_response`, `malformed_type`, `schema_version_mismatch` |
| 2 | all required fields present | `missing_field` |
| 3 | event identity | `event_id_mismatch` |
| 4 | recipients are **exactly** the required set | `recipients_not_exact` |
| 5 | target food | `target_food_mismatch` |
| 6 | the selected id names an **offered** plan | `unknown_plan_id` |
| 7 | joint validity of the mapping actually carried | see below |
| 8 | the mapping **is** the plan the id names | `plan_assignment_mismatch` |
| 9 | the code-built plan re-checked (defence in depth) | `offered_plan_invalid` |
| 10 | role is one the code offered | `role_not_allowed` |
| 11 | coalition feasibility re-derived from raw state | `coalition_infeasible` |
| 12 | confidence in `[0, 1]` | `confidence_out_of_range` |

Step 7 is `plan_violations()`, which re-derives every joint constraint from the
raw request rather than trusting that a plan came out of the enumerator:

| Constraint | Reason |
|---|---|
| every required recipient exactly once | `plan_recipients_not_exact` |
| no agent twice | `plan_duplicate_agent` |
| **no two agents on one cell** | `plan_duplicate_cell` |
| cell inside the grid | `plan_cell_outside_grid` |
| cell is not the target food | `plan_cell_is_target_food` |
| cell orthogonally adjacent to the food | `plan_cell_not_adjacent` |
| cell was offered | `plan_cell_not_offered` |

Reasons are reported in a fixed precedence (`VIOLATION_PRECEDENCE`), not
alphabetically, so the one surfaced is the most specific one.

**Both the id and the mapping are checked.** Neither is authoritative alone: an
offered id carrying a different mapping fails step 8, and a correct mapping under
an unoffered id fails step 6. **Cache hits take the identical path** — a stored
plan is re-validated against the live request, so a stale or tampered row fails
closed instead of activating. JSON-Schema `enum` enforcement is treated as a
convenience and never relied upon.

---

## 4. Zero / singleton / multiple-plan policy — **frozen**

Applied in `GuidanceController._offer`, *before* any cache lookup or provider call.

| Valid plans | Provider call | Cache lookup | Guidance | `F_t` | Reason |
|---|---|---|---|---|---|
| 0 | none | none | none | 0 | `no_valid_joint_plan` |
| 1 | none | none | none | 0 | `singleton_no_choice` |
| ≥ 2 | normal | normal | on validation | PBRS | — |

The singleton rule exists so that **every activated LLM subgoal was selected from
at least two physically valid alternatives**. A forced "choice" would spend a live
call to be told the only answer, and would attribute to the model a decision it
did not make.

Neither skip is a heuristic fallback: they activate nothing. Neither is a
*fallback* in the accounting sense either — no request was ever considered, so
they must never touch the live-call, cache-hit, cache-miss or fallback counters.

### Accounting

New counters: `zero_plan_skip_count`, `singleton_skip_count`,
`multiple_plan_decision_count`. New invariants:

* **I11** — `zero + singleton + multiple + missing_position + terminal_suppressed
  ≤ detector_event_count`;
* **I12** — `multiple_plan_decision_count == provider_request_considered_count`.
  This is what *proves* a skip cost nothing: only a multiple-plan event can become
  a considered request, and only a considered request can reach the cache or a
  provider;
* **I13** — `guidance_activation_count ≤ multiple_plan_decision_count`, so
  guidance can only ever have been activated by a genuine choice.

---

## 5. Version changes and cache invalidation

| Version | Before | After |
|---|---|---|
| `PROMPT_VERSION` | `selma_guidance_prompt_v2` | **`selma_guidance_prompt_v3`** |
| `REQUEST_VERSION` | *(did not exist)* | **`selma_guidance_request_v2`** |
| `SCHEMA_VERSION` (response) | `guidance_schema_v1` | **`guidance_schema_v2`** |
| `RESPONSE_SCHEMA_VERSION` | `selma_guidance_response_schema_v1` | **`selma_guidance_response_schema_v2`** |
| `CACHE_KEY_VERSION` | `selma_cache_key_v1` | **`selma_cache_key_v2`** |
| `CACHE_SCHEMA_VERSION` | `selma_cache_schema_v1` | **`selma_cache_schema_v2`** |

Key composition changed too: `candidate_assignment_ids` **left** the key and the
complete `candidate_joint_plans` definitions **entered** it. Plan ids are
positional — `plan_3` denotes a different mapping in a different layout — so a key
holding only ids would collide across genuinely distinct questions. The key stores
each plan's full agent→cell mapping.

### Proof that pilot-era entries cannot be served

Three independent barriers, any one of which suffices:

1. **The database cannot be opened.** `SqliteGuidanceCache` refuses a store whose
   `cache_schema_version` differs from the current build's, raising `CacheError`
   rather than migrating. Pinned by
   `test_22b_a_pilot_era_cache_database_cannot_be_reopened`.
2. **The digest cannot collide.** `cache_key_version`, `request_schema_version`,
   `prompt_version` and `response_schema_version` are all in the key and all
   changed; the key's field set changed as well. Pinned by
   `test_22_the_new_versions_invalidate_every_prior_cache_key`.
3. **The stored payload would not validate.** A v1 row carries no
   `selected_plan_id`, so rebuilding it raises and the hit is a logged miss.

The pilot cache is **not migrated and not reused**. It lived in a temporary
directory and was discarded; the live regression below ran on an empty store.

---

## 6. Fails-first evidence

1. The defect was reproduced against the pre-correction tree (§1) — the validator
   printed `ACCEPTED` for the exact response the pilot received.
2. `tests/test_selma_stage_d_joint_plans.py` was written before the correction and
   failed to import on the pre-correction tree (`SKIP_NO_VALID_JOINT_PLAN`,
   `selma.joint_plans`, `REQUEST_VERSION` did not exist).
3. During implementation, an intermediate run had **45 passed / 4 failed**; the
   four failures were genuine and were fixed in the implementation and the test
   expectations respectively (violation precedence; isolating plan-level checks
   from the response-level recipients check).

All 22 required scenarios are covered:

| # | Scenario | Test |
|---|---|---|
| 1 | two agents on one cell rejected | `test_1_two_agents_assigned_to_the_same_cell_are_rejected` |
| 2 | duplicate recipient rejected | `test_2_a_duplicate_recipient_agent_is_rejected` |
| 3 | missing required recipient rejected | `test_3_…` |
| 4 | extra recipient rejected | `test_4_…` |
| 5 | cell outside grid rejected | `test_5_…`, `test_5b_…` |
| 6 | cell not adjacent rejected | `test_6_…`, `test_6b_…` (diagonal) |
| 7 | target food cell rejected | `test_7_…`, `test_7b_…` (never enumerated) |
| 8 | assignment outside the selected plan rejected | `test_8_…` |
| 9 | unknown plan id rejected | `test_9_…`, `test_9b_…` (malformed ids) |
| 10 | cache hit with a now-invalid plan rejected | `test_10_…`, `test_10b_…` |
| 11 | enumeration never duplicates cells | `test_11_…`, `test_11b_…` |
| 12 | enumeration canonical across input ordering | `test_12_…`, `test_12b_…` |
| 13 | mirrored/translated stay distinct scheme-B keys | `test_13_…`, `test_13b_…` |
| 14 | prompt describes every plan mapping | `test_14_…`, `test_14b_…` |
| 15 | schema permits only one `selected_plan_id` | `test_15_…` |
| 16 | old response shapes rejected | `test_16_…`, `test_16b_…` |
| 17 | zero plans ⇒ zero calls, `F_t = 0` | `test_17_…` |
| 18 | singleton ⇒ zero calls, `F_t = 0` | `test_18_…` |
| 19 | multiple plans take the normal path | `test_19_…`, `test_19b_…`, `test_19c_…` |
| 20 | greedy evaluation makes zero calls | `test_20_…` |
| 21 | `beta = 0` ⇒ `training_reward == raw_team_reward` | `test_21_…`, `test_21b_…` |
| 22 | version changes invalidate the prior key | `test_22_…`, `test_22b_…`, `test_22c_…` |

---

## 7. Offline verification

| Command | Result |
|---|---|
| `pytest tests/test_selma_stage_d_joint_plans.py` | **52 passed** |
| `pytest tests/test_selma_stage_d_*.py` | 205 passed |
| `pytest -q` (full suite) | **505 passed** |
| `analysis/selma_stage_d_offline_smoke.py` | PASS — 31 checks |
| `analysis/verify_stage_c.py` | PASS — 36 checks |
| `analysis/selma_stage_d_live_pilot.py` (offline dry run) | PASS — 0 attempts |

The session-wide socket guard in `tests/conftest.py` was left intact; pytest made
no network connection.

**Stage C `beta = 0` equivalence still passes** (36 checks): with a live
deterministic-mock provider and `beta = 0`, every learning-relevant artefact is
identical to `off`.

### Deterministic mock — semantics preserved, and why

The mock now **selects the canonically first offered joint plan** (`plan_0`).

The audited mock assigned agent *i* (sorted) to offered cell `i % len(offered)`.
Whenever a plan exists, the lexicographically-first cells in order *are* the
mapping `plan_0` denotes, so the intended semantics — deterministic, pure function
of the request, lexicographic tie-break, byte-identical output — are unchanged.

The one behaviour deliberately **not** preserved is the modulo wrap: with fewer
offered cells than recipients the old rule silently reused a cell and produced
exactly the same physically impossible guidance the live provider did. There is
now no plan in that case and the mock returns `None`.

### Incidental defect found and fixed

Adding a `selected_plan_id` column to the guidance log surfaced a latent defect in
`GuidanceCsvWriter`: it appends without checking that the existing file's header
matches this build's columns. `DictWriter` writes values in the *current* column
order, so appending to a stale log shifts every value silently — `verify_stage_c`
duly read an empty string where `pbrs_term_unscaled` should have been. The writer
now compares the header and raises rather than corrupting the log. Pinned by
`test_appending_under_a_stale_header_is_refused`.

---

## 8. Live regression

Run only after every offline check passed. Isolated empty cache in a temporary
directory; multiple-plan fixtures only.

| | |
|---|---|
| Model requested / returned | `gpt-5.4-nano-2026-03-17` / `gpt-5.4-nano-2026-03-17` |
| API | Responses API, `store=false`, `reasoning: none`, no tools, no streaming, strict structured output |
| Prompt / response schema | `selma_guidance_prompt_v3` / `selma_guidance_response_schema_v2` |
| SDK | `openai==3.1.0` |
| `beta` | `0.0` |
| **Live attempts (retries included)** | **10** (cap 12, hard ceiling 50) |
| Retries | 0 |
| Successful responses | 10 |
| API failures | none |

### Per-fixture selections

| Fixture | Plans offered | Selected | Mapping | Distinct cells |
|---|---|---|---|---|
| `base_deficit2` | 12 | `plan_3` | a0→(4,3), a1→(3,4) | ✅ |
| `translated_plus1` | 12 | `plan_3` | a0→(5,4), a1→(4,5) | ✅ |
| `mirrored_columns` | 12 | `plan_3` | a0→(4,2), a1→(3,3) | ✅ |
| `three_recipients_deficit4` | 24 | `plan_20` | a0→(3,2), a1→(2,1), a2→(1,2) | ✅ |
| `deficit3_single_helper` | 12 | `plan_0` | a0→(5,2), a1→(6,1) | ✅ |
| `trigger_is_agent1` | 12 | `plan_4` | a0→(1,4), a1→(1,6) | ✅ |
| `far_helper` | 12 | `plan_5` | a0→(3,2), a1→(4,3) | ✅ |
| **`two_cells_only`** | **2** | **`plan_0`** | **a0→(5,4), a1→(5,6)** | **✅** |
| `high_levels_deficit3` | 12 | `plan_5` | a0→(2,5), a1→(3,6) | ✅ |
| `edge_row_three_cells` | 6 | `plan_2` | a0→(7,2), a1→(6,3) | ✅ |

`two_cells_only` is the regression fixture: it produced `a0@5,4; a1@5,4` under the
old contract and now returns two distinct cells. Six distinct plan ids were chosen
across ten fixtures, so the model is exercising the choice rather than always
taking the first option.

### Metrics

| Metric | Value |
|---|---|
| Parse successes / failures | 10 / 0 |
| Grounded (selected an offered plan id) | **10 / 10 (100 %)** |
| Invented or unoffered plan ids | **0** |
| Validator acceptances / rejections | 21 / 0 |
| Refusals | 0 |
| **Activated plans physically valid** | **21 / 21 (100 %)** |
| **Activated plans physically invalid** | **0** |
| Cache hits / misses / writes | 11 / 10 / 10 |
| Read-only hits (0 attempts) | 10 |
| Read-only miss | `cache_read_only_miss`, 0 calls, no guidance |
| Input / output tokens | 12 650 / 250 |
| Latency (min/median/p95/max) | 0.68 / 1.20 / 2.19 / 2.19 s |

**Cost was not calculated**: no cited, dated price table was supplied, so tokens
are reported verbatim. The spend bound was structural — 10 attempts, ≤512 output
tokens each — and far below the USD $0.25 authorisation.

### Accounting ledgers

| Counter | Probe | Pilot | Read-only |
|---|---|---|---|
| `detector_event_count` | 2 | 19 | 11 |
| `provider_request_considered_count` | 2 | 19 | 11 |
| `multiple_plan_decision_count` | 2 | 19 | 11 |
| `zero_plan_skip_count` | 0 | 0 | 0 |
| `singleton_skip_count` | 0 | 0 | 0 |
| `cache_lookup_count` | 2 | 19 | 11 |
| `cache_hit_count` | 1 | 10 | 10 |
| `cache_miss_count` | 1 | 9 | 1 |
| `live_api_call_count` | 1 | 9 | **0** |
| `successful_response_count` | 2 | 19 | 10 |
| `rejected_response_count` | 0 | 0 | 0 |
| `retry_count` | 0 | 0 | 0 |
| `guidance_activation_count` | 2 | 19 | 10 |

I12 holds in every phase (`multiple_plan_decision_count ==
provider_request_considered_count`). All invariants pass.

### Offline invalid-response fixtures

Exercised with fakes, never by asking the API to misbehave. All eight cases —
`invented_plan_id`, `legacy_assignment_id_list`, `duplicate_cell_attempt`,
`missing_plan_id`, `malformed_body`, `refusal`, `incomplete`,
`transport_failure` — activated zero guidance, contributed zero shaping, wrote
zero cache rows, and left every invariant intact.

---

## 9. Physical-validity acceptance table

| Gate | Result |
|---|---|
| Exact model snapshot returned | **PASS** |
| 100 % of responses select an offered joint-plan id | **PASS** (10/10) |
| 100 % of activated plans have distinct agents | **PASS** (21/21) |
| 100 % of activated plans have distinct cells | **PASS** (21/21) |
| Zero physically invalid guidance activated | **PASS** |
| Zero invented plan ids | **PASS** |
| Zero calls on cache hits | **PASS** (11 hits, 0 calls) |
| Zero calls on read-only misses | **PASS** |
| Zero calls in greedy evaluation | **PASS** |
| Every offered plan physically valid before any request | **PASS** (10/10 fixtures) |
| All accounting reconciles | **PASS** (3 phases, I1–I13) |
| No secret exposure | **PASS** |
| No training | **PASS** |
| `beta` remained zero | **PASS** |
| Attempts within cap | **PASS** (10 ≤ 12 ≤ 50) |

All 31 harness gates passed; exit 0.

---

## 10. Limitations

1. **No performance claim.** This validates semantic validity, grounding, caching
   and accounting. It says nothing about whether the selected plans are *good*.
2. Ten fixtures, one snapshot, one day, 8×8 grids only.
3. `plan_violations` enforces the seven constraints listed in §3. It does not
   model transient path collisions *en route* to the loading cells, only the
   destination configuration.
4. The frozen detector excludes a loader's own current cell from
   `candidate_loading_cells` (`coalition.py`), so a loader already standing on a
   loading cell is offered only the others. That is pre-existing behaviour of the
   frozen contract, unchanged here, but it means enumeration works from a slightly
   smaller cell set than the raw geometry allows.
5. No zero-plan or singleton event occurred in the live regression (all fixtures
   offered ≥ 2 plans by design), so that policy is verified offline only.
6. `cached_input_tokens` was 0; prompt caching at scale remains untested.

---

## 11. Explicit statements

* **No training was run.** No learner, no environment, no episode rollout beyond
  the two `verify_stage_c` equivalence runs that predate and are unchanged by this
  correction.
* **No beta calibration was performed.**
* **`beta` was `0.0` in every phase**, asserted per record; `training_reward`
  equalled `raw_team_reward` exactly.
* **This is not a performance result.**
* **`src/selma/detector.py` was not touched** — zero diff from `10417e6`.
* **`docs/project/selma_seed91_recovery_and_rerun_plan.md` was not touched** —
  unchanged, untracked and unstaged throughout.
* **The API key was never read, printed, copied, hashed or logged.**
  `~/.config/selma/openai.env` was inspected by `stat` metadata only (regular
  file, mode 600, owned by the invoking user) and sourced only into the subprocess
  making the authorised request.
* **The original live-pilot record and its commit `10417e6` remain immutable
  historical evidence.** The correction was added as a clearly labelled notice;
  nothing was erased.
* Raw and transient output lives in `analysis/out/live_pilot/`, which is
  gitignored. No SQLite database, response dump, Sacred output, cache or temporary
  file was staged.
