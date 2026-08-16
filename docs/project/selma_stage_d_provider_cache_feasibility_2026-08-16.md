# SELMA Stage D — provider / cache feasibility audit

**Date:** 2026-08-16
**Branch:** `stage-d-provider-cache`
**Stage C parent commit:** `50b550de6fb531fdb2dd817d993d2c616854ba69` (`50b550d`,
`selma: verify deterministic Stage C PBRS integration`), itself a child of the
Stage B closure commit `977954ba`.

**Status: AUDIT ONLY. Nothing implemented.** This document is the sole file added
by its commit. No source file, configuration, test or existing document was
modified. `src/selma/detector.py` is untouched.

**No training run was executed for this audit. No live LLM, API or external
service was called.** The only network operations were the two explicitly
authorised `git push` operations. All numbers below come from archived Stage B
summary artefacts and from streaming, read-only passes over archived shadow CSVs.

---

## 1. Verified evidence sources

| Source | What it provides | Status |
| --- | --- | --- |
| `analysis/out/stage_b_paper_metrics/detector_metrics_by_condition_and_seed.csv` | frozen-variant event counts, rates, coverage, cap binding for 8x8 seed 91, 8x8 seed 99, 10x10 seed 91 | **canonical**, published metric table |
| `analysis/out/stage_b_paper_metrics/detector_field_coverage_register.csv` | the 55 raw shadow-CSV fields and their role | canonical |
| `docs/project/selma_stage_b_selection_record.md` | the frozen variant and the locked selection rule | canonical |
| `docs/project/selma_stage_c_implementation_contract.md` §3.5, §6 | the proposed cache key, comparison conditions, open decisions | contract |
| `src/selma/provider.py`, `src/selma/guidance.py` (at `50b550d`) | the implemented request/response/validation/lifecycle/log contracts | implemented |
| `analysis/out/stage_c_beta_zero/B_active_beta0/guidance.csv` | a real Stage C guidance log (41 episodes, mock provider, β = 0) | implemented-path evidence |
| `analysis/out/seed91_primary_20260731T153533Z/8x8/shadow_8x8_seed91_2M_primary.csv` (2.26 GB) | raw per-event fields, 8x8 | **SUPERSEDED, PARTIAL** — truncated by the I/O failure at `t_env` ≈ 909,892 |
| `analysis/out/seed91/10x10/shadow_10x10_seed91_2M.csv` (961 MB) | raw per-event fields, 10x10 | **SUPERSEDED, PARTIAL** — ends at `t_env` ≈ 788,228 |

The canonical raw CSVs for the accepted runs (seed-91 retry `20260802T112458Z`,
seed-99 held-out `20260802T221211Z`) are **not present in the working tree** — the
metric table records them as "external raw store (not committed; ~7.6 GB total)".
Per the standing instruction not to copy or regenerate multi-gigabyte evidence,
they were not retrieved. Aggregate numbers therefore come from the canonical
summary table; per-event key-reuse structure could only be measured on the two
**superseded partial** CSVs that are local, and is labelled as such everywhere
below.

Both partial CSVs were read by streaming row-by-row (`csv.DictReader`, no
whole-file load) and produced no derived file output — only in-memory counters.

---

## 2. Event and query volume for the frozen detector `wf15_m3_t5_ibon`

Exact values from the canonical metric table. All three runs are 2M environment
steps in shadow mode (unshaped policy).

| Metric | 8x8 seed 91 (dev) | 8x8 seed 99 (held-out) | 10x10 seed 91 (validation) |
| --- | ---: | ---: | ---: |
| Environment | `Foraging-8x8-2p-2f-coop-v3` | same | `Foraging-10x10-3p-3f-v3` |
| Agents | 2 | 2 | 3 |
| Observed steps | 2,000,046 | 2,000,004 | 2,000,007 |
| Episodes | 40,003 | 40,020 | 40,019 |
| **Qualifying detector events** | **10,716** | **10,862** | **10,288** |
| **Events per 1,000 steps** | **5.3579** | **5.4310** | **5.1440** |
| Events per episode | 0.26788 | 0.27141 | 0.25708 |
| **Episode coverage** | **23.601 %** | **24.418 %** | **21.570 %** |
| Episodes with 0 events | 30,562 | 30,248 | 31,387 |
| **Episodes with exactly 1 event** | **8,166** | **8,682** | **6,976** |
| **Episodes with 2 events** (the cap) | **1,275** | **1,090** | **1,656** |
| Cap-binding rate | 3.187 % | 2.724 % | 4.138 % |
| Events dropped by the per-episode cap | 172 | 98 | 314 |
| 2-recipient events | 10,716 | 10,862 | 6,551 |
| 3-recipient events | 0 | 0 | 3,737 |
| Natural resolution @10 | 9.313 % | 10.721 % | 1.401 % |

"Episodes with multiple events" equals exactly two in every condition, confirmed
arithmetically: `8,166 + 2×1,275 = 10,716`; `8,682 + 2×1,090 = 10,862`;
`6,976 + 2×1,656 = 10,288`. The per-episode cap `max_events_per_episode = 2`
makes two the maximum.

### 2.1 Projected request volume

| Quantity | 8x8, per 2M-step run | 8x8, five seeds at 2M | 8x8, per 10M-step run | 8x8, five seeds at 10M |
| --- | ---: | ---: | ---: | ---: |
| Detector events (seed-91 rate) | 10,716 | 53,580 | 53,579 | 267,895 |
| Detector events (seed-99 rate) | 10,862 | 54,310 | 54,310 | 271,550 |
| **Planning figure (mean)** | **≈ 10,800** | **≈ 53,900** | **≈ 53,900** | **≈ 269,500** |

These are **detector events**, which is the *upper bound* on provider requests
considered, not a call count. The conversion chain is in §2.2. The 10M column is
a linear extrapolation of the measured per-1,000-step rate and assumes the rate is
stationary, which is **not** established — see §10.

### 2.2 The six quantities, kept distinct

These are **not** interchangeable and the report never substitutes one for another.

1. **Detector events** — coalesced struggle events emitted by the frozen variant.
   Measured: 10,716 / 10,862 / 10,288 per 2M steps.
2. **Guidance activations** — validated subgoals actually put in force. Bounded
   above by (1). Differs from (1) by terminal-transition suppression and by
   rejected responses.
3. **Provider requests considered** — events for which a `GuidanceRequest` is
   built and a cache lookup is performed. Equals (1) minus events on a final
   transition (suppressed by design: guidance activated there can never influence
   an action) minus any event whose recipients lack a position. **Not yet
   quantified from a 2M run** — see §10.
4. **Actual provider calls (live API calls)** — issued only on a cache miss.
   Equals (3) only if the cache never hits.
5. **Cache hits** — lookups served from the store; **zero** live calls, zero
   tokens, zero cost, but a real guidance activation.
6. **Cache misses** — lookups that require a live call.

Evidence from the implemented Stage C path (`stage_c_beta_zero`, 41 episodes,
2,050 transitions, deterministic mock, β = 0): 5 requests considered, 5 provider
calls, 4 activations + 1 replacement, 0 rejections, 0 fallbacks, 4 TTL expiries,
0 events suppressed on a terminal transition, 63 transitions with guidance
active. There is no cache in this build, so (3) = (4) and (5) = 0 by
construction.

---

## 3. Cache-key field availability

The `GuidanceRequest` built by `src/selma/provider.py::build_request` is the only
thing a key may hash. Its fields, and where each comes from:

| Required key field | Available? | Source |
| --- | --- | --- |
| Environment / scenario | **Yes** | `condition` (env key) on every detector row; also `args.env_args["key"]` |
| Target-food identity | **Yes, but episode-scoped** | `food_id = (episode_id, row, col)` — the first element is the episode id |
| Target-food position | **Yes** | `food_position` |
| Target-food level | **Yes** | `food_level` |
| Trigger-agent IDs | **Yes** | `trigger_agent_ids` |
| Trigger-agent levels | **Yes** | `loader_levels` (+ `helper_levels`) |
| **Trigger-agent positions** | **NOT in the detector schema** | injected by the runner from `GymmaWrapper.get_selma_agent_positions()`; the frozen row carries none |
| Selected helper coalition | **Yes** | `helper_ids`, `recipient_ids` (recipients = loaders ∪ helpers) |
| Level deficit | **Yes** | `level_deficit`, `required_extra_level` |
| Valid loading cells | **Yes** | `candidate_loading_cells` |
| Candidate roles / subgoals | **Yes, code-fixed** | `ALLOWED_ROLES = (hold_and_load, approach_and_load)`; `selected_cell_assignment` for the deterministic ablation |
| Episode / time identifiers | **Yes** | `episode_id`, `episode_step`, `t_env`, `event_id` |
| Fields affecting response validity | **Yes** | `schema_version`, `coalition_feasible`, `food_level` vs. recipient level sum, `candidate_loading_cells`, `allowed_roles` |

**Fields that must be EXCLUDED because they only prevent reuse:** `t_env`,
`episode_id`, `episode_step`, `event_id`, wall-clock, any learner/training state,
`raw_team_reward`, `resolved_within_*` (post-hoc labels), and — critically —
**`food_id`, because its first element is the episode id**. A key containing
`food_id` is episode-unique and yields essentially no reuse. The Stage C contract
§3.5 lists `food_id` in the key; this audit finds that to be a defect in the
proposed key and recommends `food_position` in its place.

**Fields that must be INCLUDED to prevent stale reuse across a provider change:**
`schema_version`, `provider_id`, `model_version` (pinned snapshot), `temperature`,
`prompt_version`, and a `cache_key_version` of its own.

### 3.1 The one missing field

**Agent positions are absent from every archived Stage B artefact.** They are not
in the 55-field shadow schema — this is exactly why Stage C injects them
explicitly and raises `MissingAgentPositionError` rather than fabricating them.
Consequently **no cache scheme that includes positions can be evaluated exactly
from existing evidence.** `helper_distances` and `total_helper_distance` give
distances to the target, not coordinates, and do not determine positions.

---

## 4. Measured key-reuse bounds

Streamed, read-only, over the two locally available **superseded partial** shadow
CSVs, restricted to `row_type == coalesced_event` and `variant_id ==
wf15_m3_t5_ibon`. Every key below is a **position-free projection** of the
corresponding scheme. Adding agent positions can only split equivalence classes
further, so **each figure is a strict upper bound** on that scheme's hit rate,
never an estimate of it.

**8x8, seed 91 (superseded partial: 6,318,280 rows streamed, `t_env` 638 →
909,892, 5,724 frozen-variant events in 4,907 distinct episodes)**

| Projection | Unique keys | Repeated | Max multiplicity | Hit-rate **upper bound** |
| --- | ---: | ---: | ---: | ---: |
| K1 — episode-scoped (incl. `food_id`) | 5,452 | 272 | 2 | **4.75 %** |
| K2 — absolute state, position-free | 807 | 4,917 | 45 | **85.90 %** |
| K3 — target-relative geometry, position-free | 16 | 5,708 | 469 | **99.72 %** |
| K4 — symmetry-normalised (D8 dihedral) | 4 | 5,720 | 1,690 | **99.93 %** |

**10x10, seed 91 (superseded partial: 2,606,519 rows streamed, `t_env` 276 →
788,228, 2,861 events in 2,419 episodes)**

| Projection | Unique keys | Repeated | Max multiplicity | Hit-rate **upper bound** |
| --- | ---: | ---: | ---: | ---: |
| K1 — episode-scoped | 2,713 | 148 | 2 | **5.17 %** |
| K2 — absolute state, position-free | 2,335 | 526 | 5 | **18.39 %** |
| K3 — target-relative geometry | 74 | 2,787 | 115 | **97.41 %** |
| K4 — symmetry-normalised | 27 | 2,834 | 412 | **99.06 %** |

Two findings follow immediately.

* **K1 confirms the `food_id` defect empirically.** Including the episode-scoped
  food identity leaves ≤ 5 % reuse: caching would be nearly worthless.
* **The 2-agent and 3-agent conditions behave very differently.** K2 reuse
  collapses from ≤ 85.9 % (8x8, 2 agents) to ≤ 18.4 % (10x10, 3 agents). Any
  budget sized on the 8x8 number will be badly wrong for 10x10.

The event rate in these partial files (6.29 and 3.63 per 1,000 steps) differs
from the canonical 2M rates (5.36 and 5.14), because they cover only the first
~910k / ~788k steps. They are used for **structure** (key multiplicity) only,
never for volume.

---

## 5. Cache-key schemes

### A. Exact validated-request key

**Fields:** the complete `GuidanceRequest` as sent — `schema_version`,
`event_id`, `event_type`, `food_id`, `food_position`, `food_level`,
`trigger_agent_ids`, `recipient_agents`, `level_deficit`,
`required_extra_level`, and the full `agent_context` (id, level, position,
allowed roles, candidate cells) — plus `provider_id`, `model_version`,
`temperature`, `prompt_version`.

* **Correctness conditions:** none beyond byte-stable serialisation. The key is
  the request, so a hit is by definition the same question.
* **Expected reuse:** ≤ 4.75 % (8x8) / ≤ 5.17 % (10x10), measured. `event_id` and
  `food_id` are episode-scoped, so reuse is essentially confined to the two events
  a single episode may fire on the same food.
* **Response transformation:** none.
* **Post-cache validation:** re-run `validate(request, response)` — trivially
  passes.
* **Collision / staleness risk:** effectively zero.
* **Dissertation-safe?** **Yes — but it does not solve the problem.** It leaves
  ~10,200 live calls per 2M-step run.

### B. Episode-independent absolute-state key — **RECOMMENDED**

**Fields:** `cache_key_version`, `schema_version`, `prompt_version`,
`provider_id`, `model_version`, `temperature`, `condition` (env key),
`grid_shape`, **`food_position`** (*not* `food_id`), `food_level`,
`sorted(recipient_ids)`, `sorted((agent_id, level, position))`,
`sorted(candidate_loading_cells)`, `level_deficit`, `required_extra_level`,
`sorted(allowed_roles)`.
**Excluded:** `t_env`, `episode_id`, `episode_step`, `event_id`, `food_id`,
wall-clock, learner state.

* **Correctness conditions:** the response references only agent ids and absolute
  loading cells, both of which are pinned by the key, so a hit is
  interchangeable with a fresh answer *provided* the excluded fields genuinely do
  not affect the correct answer. That holds here because the request contains no
  temporal field the provider is asked to reason about.
* **Expected reuse:** ≤ 85.90 % (8x8) / ≤ 18.39 % (10x10) **before** positions
  are added; the true value is unknown and could be far lower. See §6.
* **Response transformation:** **none** — this is its main safety advantage.
* **Post-cache validation:** mandatory. Re-build the request from live state and
  re-run the full deterministic `validate()` against the cached response. A key
  collision or a drifted grid then fails closed to `F_t = 0`, exactly like any
  other invalid response.
* **Collision / staleness risk:** low. Hash collisions are handled by storing the
  full key alongside the value and comparing it on hit. Staleness across provider
  or prompt changes is prevented by the version fields.
* **Dissertation-safe?** **Yes.** Recommended.

### C. Target-relative geometry key

**Fields:** version/provider fields as in B, plus `grid_shape`, `food_level`,
`level_deficit`, `required_extra_level`, the multiset of recipient levels, and
all cells and agent positions expressed as **offsets from the target-food
coordinate**, with agents canonically ordered (e.g. by `(level, offset)`, ties
broken deterministically).

* **Correctness conditions:** (i) a proved bijection between the absolute and
  offset representations for the specific food coordinate; (ii) a proved inverse
  mapping from canonical agent slots back to real agent ids; (iii) the
  observation that the potential is Manhattan-based and therefore
  translation-invariant. Translation is *not* a symmetry of the LBF grid near the
  boundary — an offset legal near the centre can be off-grid elsewhere — but
  `candidate_loading_cells` are supplied by code per event, so an illegal cell can
  never be selected, and post-cache validation catches any residue.
* **Expected reuse:** ≤ 99.72 % (8x8) / ≤ 97.41 % (10x10) — only **16** and **74**
  distinct keys observed.
* **Response transformation:** **required** — add the food coordinate back to each
  returned offset cell, and map canonical slots back to real agent ids. This
  transformation and its inverse must be proved and tested before use.
* **Post-cache validation:** mandatory, after transformation.
* **Collision / staleness risk:** moderate. The risk is not hash collision but
  *semantic* collision: two situations that differ in something the key discards
  (nearby other food, the other agents' positions) map to one answer.
* **Dissertation-safe?** **Conditionally**, and only with the transformation proof
  and a documented reuse figure. But see the scientific-validity caveat below.

### D. Symmetry-normalised key — **NOT RECOMMENDED**

**Fields:** as C, further reduced by the canonical form over the eight dihedral
transforms of the square grid.

* **Correctness conditions:** would require proving that the coordinate transform
  **and its inverse** are correct for LBF, including that nothing in the request
  or response is orientation-dependent, and that the grid is genuinely square in
  every condition used.
* **Expected reuse:** ≤ 99.93 % (8x8) / ≤ 99.06 % — **4** and **27** distinct keys.
* **Response transformation:** required, and strictly harder than C's.
* **Post-cache validation:** mandatory.
* **Collision / staleness risk:** highest of the four.
* **Dissertation-safe?** **No — not recommended.** The standing instruction is not
  to recommend symmetry normalisation unless the action/coordinate transformations
  and their inverses can be **proved** correct. No such proof exists in this
  repository, none was constructed for this audit, and constructing one is not a
  small task. It is therefore rejected, not deferred.

### 5.1 The scientific-validity caveat on aggressive keys

This is the most important finding of the audit and it cuts against the schemes
with the best reuse. Under K3 the whole 8x8 run collapses to **16 distinct
questions**; under K4, to **4**. A condition in which the language model answers
four questions and every one of ~10,700 guidance activations replays one of those
four answers is no longer meaningfully "LLM-guided per situation" — it is a
four-entry lookup table that the LLM happened to author. That directly undermines
the contrast the dissertation is built on (`C2 − C5`: LLM subgoal selection versus
a deterministic heuristic), because at that level of reuse the LLM condition *is*
a deterministic heuristic.

Scheme B is recommended partly because it is transformation-free and partly
because its coarser reuse keeps a meaningful number of distinct questions. Whatever
scheme is chosen, **the distinct-key count and the hit rate must be reported in
the dissertation as a first-class property of the method**, not buried as an
implementation detail.

---

## 6. What cannot be computed from existing evidence

Per §3.1, **agent positions are missing from every archived artefact.** Therefore:

* the true unique-key count, repeated-key count and hit rate for schemes **A, B, C
  and D as specified** cannot be computed;
* only the position-free upper bounds in §4 are available;
* the projected live-call count per run has a wide, honestly-unresolved range.

**Live-call projection under scheme B, 8x8, per 2M-step run** (using ≈ 10,800
requests considered):

| Assumption | Live calls / run | Live calls / 5 seeds |
| --- | ---: | ---: |
| No reuse at all (worst case, = requests considered) | 10,800 | 54,000 |
| Position-free upper-bound reuse (85.9 %) — **optimistic floor** | ≈ 1,520 | ≈ 7,600 |
| Distinct keys observed in the partial run, saturating (≈ 807 → ~1,000–1,200 at 2M) | ≈ 1,000–1,200 | ≈ 5,000–6,000 |

The true figure lies **between ~1,520 and 10,800 per run**, and nothing in the
existing evidence narrows it further. For 10x10 the same reasoning gives a floor
of ≈ 8,400 live calls per run (≤ 18.4 % reuse) — nearly the no-cache figure.

**Missing fields, stated exactly:** per-recipient agent **positions** at the
transition on which the event fired (`agent_context[i].position`). Everything else
required by every scheme is present. No reuse estimate is invented to fill the
gap, and no new run was started to obtain it. §9 step 1 specifies the cheap,
zero-live-call measurement that would close it.

---

## 7. Permanent accounting counters

All eleven are cumulative per run, written to the run summary and to the guidance
log, and never reset at an episode boundary.

| Counter | Definition |
| --- | --- |
| `detector_event_count` | coalesced events emitted by the frozen variant in training mode |
| `provider_request_considered_count` | events for which a `GuidanceRequest` was built and a cache lookup performed |
| `cache_lookup_count` | cache lookups performed |
| `cache_hit_count` | lookups served from the store |
| `cache_miss_count` | lookups not served from the store |
| `live_api_call_count` | requests actually transmitted to the provider, **including retries** |
| `successful_response_count` | considered requests that ended with a validated response |
| `rejected_response_count` | individual provider responses that failed deterministic validation |
| `retry_count` | additional transmissions beyond the first for one considered request |
| `fallback_count` | considered requests that ended with no guidance (`F_t = 0`) |
| `guidance_activation_count` | subgoals put in force (activations + replacements) |

### 7.1 Invariants

These must be asserted at run end, and violations must fail the run, in the same
style as the Stage C runtime invariants.

1. `detector_event_count ≥ provider_request_considered_count`
   — the gap is exactly `terminal_suppressed_events + missing_position_rejections`.
2. `cache_lookup_count == provider_request_considered_count`
   — **lookup precedes every call**; one lookup per considered request, no exceptions.
   *(Factual correction, 2026-08-16, made during implementation: this holds when
   the cache is enabled. With `selma_cache_mode=off` there is by definition no
   lookup, and the invariant becomes `cache_lookup_count == 0`.)*
3. `cache_hit_count + cache_miss_count == cache_lookup_count`.
4. `live_api_call_count == call_opportunities + retry_count`
   — a hit issues no call; a call opportunity issues exactly one, plus any retries.
   *(Factual correction, 2026-08-16, made during implementation: the original
   text used `cache_miss_count`, which is wrong when the cache is off — there
   are then live calls and no misses. A **call opportunity** is
   `cache_miss_count` when the cache is enabled and
   `provider_request_considered_count` when it is off. The identity also relaxes
   to a two-sided bound when a call is suppressed by budget exhaustion,
   read-only mode or an offline provider.)*
5. `cache_hit_count > 0 ⇒ live_api_call_count < provider_request_considered_count`.
6. `successful_response_count + fallback_count == provider_request_considered_count`
   — every considered request has exactly one terminal outcome.
7. `guidance_activation_count == successful_response_count`
   — a validated response always activates or replaces; nothing is validated and dropped.
8. `rejected_response_count ≥ fallback_count − budget_exhausted_count − timeout_count − transport_error_count`
   — rejections are per-response, fallbacks per-request.
9. `retry_count ≤ max_retries × call_opportunities` *(same correction as
   invariant 4: `cache_miss_count` in the original text)*.
10. `live_api_call_count ≤ live_call_budget` (§8), always.
11. **Greedy evaluation:** every counter's delta across a `test_mode=True` episode
    is exactly `0`. Already true in Stage C and tested; it must remain an assertion.
12. **Mock / deterministic-ablation conditions:** `live_api_call_count == 0`.
13. **β = 0:** counters may be non-zero; `total_shaping_contribution == 0.0` exactly.
14. `cache_hit_count` never increases `live_api_call_count`, tokens, latency or
    cost. Cost is summed over live calls only; a hit records usage as `n/a`
    (the existing `ProviderUsage(None,…)` convention), never as `0`.

---

## 8. Comparison conditions and fairness

Planned conditions (spec D.2/D.3, contract §6): **C0** plain QMIX · **C1**
Stage C wired at β = 0 · **C2** selective SELMA guidance · **C3** non-selective
guidance · **C4** query-budget-matched random guidance · **C5** deterministic
heuristic ablation.

### 8.1 What "non-selective guidance" can operationally mean

| Definition | Calls per 2M-step run | Verdict |
| --- | ---: | --- |
| **N1** one live call every environment step | 2,000,000 | **Rejected outright.** Not affordable, not runnable, and rejected by instruction. |
| **N2** query on every *raw* detector candidate, before latch/cooldown/rearm/cap | 230,268 (8x8 seed 91, measured) | Rejected: ~21× C2, still infeasible live. |
| **N3** "always-on": guidance is always in force, re-queried the moment it expires (TTL = 15) | ≈ 2,000,000 / 15 ≈ **133,000** | Infeasible live; the honest *conceptual* definition of always-on. |
| **N4** fixed-period trigger: query every *K* training steps whenever a coalition-feasible target exists, *K* chosen to give a stated multiple of C2's calls | tunable (e.g. 4 × C2 ≈ 43,000) | **Feasible**, pre-declarable, comparable. |
| **N5** deterministic always-on: N3's coverage with the deterministic heuristic instead of a provider | 0 live calls | **Feasible**, isolates selectivity at zero provider cost. |

**Recommendation:** define non-selective guidance conceptually as **N3**, and run
it as **N4 at a pre-declared multiple of C2's call count**, with **N5** as the
zero-cost companion. Report explicitly that a fully live always-on condition was
not run and why. This corresponds to contract decision **D6** options (b) and (c);
this audit's evidence supports choosing them over (a).

### 8.2 Fairness, dimension by dimension

| Dimension | Requirement |
| --- | --- |
| **Opportunity to provide guidance** | **Deliberately unequal** — that is the independent variable. C2 gets detector-selected moments; C3 gets many more; C4 gets the same *number* at random moments. This must be stated, not equalised. |
| **Live-call budget** | Equal between **C2 and C4** (C4 is defined by matching), and *not* equal for C3, whose whole point is more guidance. Report calls per 1,000 steps for each. |
| **Cache usage** | Identical policy, key scheme, key version and cache mode across all guided conditions. A condition that benefits from a warm cache another condition primed would be contaminated: run each condition with its **own** cache namespace, or a shared read-only prewarmed cache, but never a shared read-write cache across conditions. |
| **Guidance duration** | Identical TTL (15), identical expiry rules, identical replacement semantics. |
| **Valid-response handling** | Identical validator, identical fallback (always `F_t = 0`, never a heuristic substitute — otherwise C2 silently absorbs C5's effect). |
| **Evaluation-time behaviour** | Identical and absolute: **zero** queries, zero shaping, raw reward, in every condition. Already enforced and tested in Stage C. |
| **Cost and latency measurement** | Identical accounting. Only live calls carry tokens/latency/cost; hits are `n/a`. C5 and C1 report exact zeros for live calls. |
| **Matching basis for C4** | Match on `provider_request_considered_count` **and** report `live_api_call_count` separately, since cache behaviour will differ between genuine-struggle states (structurally repetitive) and random states (structurally diverse). Matching only on live calls would let C4 receive more *guidance* than C2 for the same spend. |

C4 must run **after** its C2 counterpart, replaying C2's per-run schedule of
opportunity counts with a separate recorded RNG seed, selecting only valid
training refresh opportunities, never during greedy evaluation.

---

## 9. Recommended query-budget policy

1. **Hard per-run live-call cap** `selma_live_call_budget` (integer, logged in the
   run config, part of the run's provenance). Suggested initial value for 8x8 at
   2M: **2,000** — above the ~1,000–1,200 distinct keys the partial evidence
   suggests, below the ~10,800 no-cache worst case. For 10x10 the evidence
   supports no such cap without measurement; treat it as unset until §9 step 1 is
   done.
2. **Budget exhaustion is a fallback, never a block and never a heuristic.** On
   exhaustion the request ends exactly as a validation failure does:
   `fallback_reason = budget_exhausted`, no guidance, `Φ = 0`, `F_t = 0`, counted
   in `fallback_count`. Training continues at full speed. The run is still valid
   and is reported with the exhaustion step and count.
3. **Lookup strictly precedes any call** (invariant 2). A hit costs nothing and is
   never counted as a call.
4. **Deterministic cache lookup.** The key is a canonical, sorted, versioned
   serialisation hashed with SHA-256; the full key is stored beside the value and
   compared on hit. Identical inputs give an identical key on every machine and
   every run.
5. **Persistent cache with an explicit mode:** `off | read_write | read_only`.
   `read_only` over a prewarmed store gives **exactly zero live calls**, making
   replays, C4 matching and re-analysis free and bit-reproducible.
6. **Post-cache validation is mandatory** on every hit (§5B).
7. **Zero calls during greedy evaluation**, guaranteed structurally: the
   controller is not consulted in `test_mode` (already the Stage C behaviour), plus
   the counter-delta assertion of invariant 11.
8. **Exact cost reporting:** per live call record tokens in/out, latency and cost;
   the run total is the sum over live calls only. Publish calls per 1,000 steps,
   hit rate, distinct-key count and total spend per condition.
9. **Cost is bounded before spending:** `live_call_budget × max_unit_cost` is
   computable before a run starts and must be stated in the pre-registration.

---

## 10. Unresolved decisions requiring approval

None of these may be chosen after seeing outcomes.

| # | Decision | Status | Must be frozen |
| --- | --- | --- | --- |
| 1 | **Provider / model** | Open (contract D3b) | before the first live call |
| 2 | **Model snapshot / version** | Open — must be a pinned dated snapshot, not a floating alias | before the first live call |
| 3 | **Temperature and seed support** | Open — whether the chosen provider offers a reproducible seed at all is unverified | before the first live call |
| 4 | **Structured-output mechanism** | Open — tool/function schema vs. JSON mode vs. parsed text; determines the malformed-response rate | before the offline validation of §11 step 5 |
| 5 | **Prompt version** | Open — no prompt exists yet; must be version-stamped and in the cache key | before the first live call |
| 6 | **Timeout** | Open (contract D9) | before Stage D runs |
| 7 | **Retry limit** | Open — spec D.1 says one bounded retry; not yet fixed in code | before Stage D runs |
| 8 | **Persistent cache format** | Open — recommend append-only JSONL with full key, response, usage and provenance | before implementation |
| 9 | **Cache-key versioning** | Open — recommend an explicit `cache_key_version` bumped on any key change | before implementation |
| 10 | **Invalid-response fallback** | Settled in principle (always `F_t = 0`, never a heuristic), **not** yet enforced for a live provider | before the first live run |
| 11 | **Live-query cap** | Open — §9 proposes 2,000 for 8x8 at 2M; unsupported for 10x10 | before the first live run |
| 12 | **β** | Open (contract D1); `β = 0` remains the default | before any β > 0 run |
| 13 | **TTL** | **Frozen at 15** (contract D8/6.2) — listed for completeness | frozen |
| 14 | **Pilot seed** | Open — must be a development seed, **not** one of the reserved experimental seeds 1–5 (4242 was used for Stage C) | before the pilot |
| 15 | **Final experimental protocol** | Open — training budget (D5), seed count (D7, ≥ 5 treated as a floor), C3 scope (D6) | before C2 |

---

## 11. Ordered Stage D implementation sequence (not implemented)

1. **Measure the real key distribution at zero provider cost.** Add a
   `request_key_hash` column plus the key's constituent fields (including agent
   positions) to the existing guidance log, then run **one mock-provider run** at
   the target length. Deterministic mock ⇒ **zero live calls, zero spend**. This
   closes the single evidence gap in §6 and is the prerequisite for every number
   in §9. Do this before anything else.
2. **Freeze decisions 1–9 of §10** on the basis of that measurement, and record
   them as a frozen-decision table the way Stage B and Stage C did.
3. **Implement the cache as a standalone module** — scheme B key, canonical
   serialisation, SHA-256, full-key comparison on hit, mandatory post-cache
   re-validation, the three modes, JSONL persistence. No provider changes.
4. **Implement the eleven counters and their fourteen invariants** as runtime
   assertions, with fails-first tests for each: cache determinism, replay
   equality, hit issues no call, budget-exhaustion fallback path, zero counter
   delta in greedy evaluation, `live_api_call_count == 0` under the mock.
5. **Offline provider validation before any training spend.** Save 50–100 real
   trigger states from step 1's run, call the live provider on those alone, and
   require ≥ 95 % parse-and-ground success (spec D.1). This is the first live
   spend: bounded, cheap and outside training.
6. **Pilot run:** one development seed, short budget, hard call cap, cache in
   `read_write`. Measure the true hit rate, live-call count, latency, token count
   and cost, and compare against §6's projected range.
7. **Freeze β (D1), training budget (D5), seed count (D7) and C3 scope (D6)** on
   the pilot's cost evidence — before any headline run.
8. **Run the protocol** in the order C1 → C2 → C4 (matched, after its C2) → C5 →
   C3, each with its own cache namespace, then report per-condition volumes,
   costs and the distinct-key count alongside the learning results.

---

## 12. Limitations and conclusions that could NOT be established

1. **No training run and no live LLM/API call occurred in producing this
   document.** The only external operations were the two authorised git pushes.
2. **The true cache hit rate is unknown** for every scheme, because agent
   positions are absent from all archived evidence (§3.1, §6). Only upper bounds
   are reported, and they are labelled as bounds everywhere. No reuse estimate has
   been invented.
3. **The key-structure measurements come from superseded, partial runs** (~910k of
   2M steps for 8x8; ~788k for 10x10), not from the canonical accepted runs whose
   raw CSVs live in an external store. They are used for structure only; all
   volume figures come from the canonical summary table.
4. **All event volumes were measured in shadow mode under an unshaped policy.**
   Under β > 0 the policy changes, so the struggle-event rate will drift in an
   unknown direction. Every projection in §2.1 and §6 inherits that assumption.
5. **The 10M projections are linear extrapolations** of a rate measured at 2M and
   assume stationarity, which is not established.
6. **`provider_request_considered_count` has not been quantified at scale.** The
   gap between detector events and considered requests (terminal-transition
   suppression) is 0 out of 5 in the only Stage C run available — far too small a
   sample to generalise.
7. **Symmetry normalisation is rejected, not merely deferred**, because the
   required transformation and inverse-transformation proofs do not exist (§5D).
8. **No cost figure is given**, because no provider or price is fixed (§10 items
   1–2). Cost is expressed only as a call count multiplied by an unfixed unit price.
9. **Nothing here establishes that guidance helps learning.** No performance claim,
   in any condition, is supported by this document or by anything in Stage C.
10. **The recommended live-call cap of 2,000 is provisional**, derived from a
    position-free bound on a superseded partial run. It must be revisited after
    §11 step 1 and is unsupported for 10x10.
