# SELMA Stage E — revised beta-calibration design

**Date:** 2026-08-17
**Branch:** `stage-e-beta-recalibration-design`
**Starting commit:** `c7a318e018f3c8d867a74f4c0bf8f3d2ec6d785e` (*selma: calibrate Stage E shaping beta*)

> **This task was entirely offline.** No API call, no network access, no
> training run, no credential read. It corrects two factual errors in the Stage E
> record, authenticates the development baseline, and derives a revised pilot
> design mechanically. **Nothing here was executed.**

The document is deliberately partitioned so the reader can always tell which
kind of claim they are reading:

| Part | Contains |
|---|---|
| §1 | **Factual corrections** to the exploratory record |
| §2 | **Observed evidence** — measured, not modelled |
| §3 | **Preregistered rules**, fixed before the relevant values were read |
| §4 | **Mechanical projections** — derived, uncertain, method-labelled |
| §5–6 | **Frozen revised protocol** |
| §7 | **Limitations and open decisions** |

---

## 1. Factual corrections to the Stage E record (§1 = corrections)

Two statements committed in `c7a318e` were wrong. Both are corrected in place in
`selma_stage_e_beta_calibration_2026-08-17.md` under a clearly labelled notice.
**No observed number changed, no evidence was erased, and the UNRESOLVED
conclusion stands.**

### 1.1 "Monotonic" was wrong

The original text claimed detector events and the unguided share rose
*monotonically with `beta`*. Observed decisions were:

| `beta` | 0.00 | 0.05 | 0.10 | 0.25 |
|---|---|---|---|---|
| decisions | 1 294 | 1 660 | 1 996 | **893** |

> **Correct interpretation:** detector-event frequency varied strongly with
> `beta`. It increased from `beta = 0` through `beta = 0.10`, while `beta = 0.25`
> produced substantially fewer events. The fixed call cap created materially
> unequal guidance coverage.

The trend is monotone only over the first three values. Nothing in the confound
argument depended on monotonicity — it depends on coverage being *unequal*
(81.2 % / 63.2 % / 53.5 % / 100 %), which it was.

### 1.2 "19 % of the way through training" conflated two quantities

18.8 % is the share of `beta=0.00`'s **eligible decisions** that received no
guidance. It is **not** a position in training. Measured from the decision logs,
caps were exhausted **late**:

| Condition | 1000th attempt | First `budget_exhausted` | Training elapsed | Steps remaining |
|---|---|---|---|---|
| `beta=0.00` | `t_env` 461 456 | 461 636 | **92.3 %** | ~38 400 |
| `beta=0.05` | 441 088 | 441 106 | **88.2 %** | ~58 900 |
| `beta=0.10` | 378 557 | 378 726 | **75.7 %** | ~121 300 |
| `beta=0.25` | never | never | — | — |

These measured values refine the approximate "~55k remaining / ~89 %"
characterisation, which matches `beta=0.05` closely; `beta=0.00` exhausted with
~38k steps left.

Both corrections are pinned by regression tests
(`test_the_record_never_claims_a_monotonic_relationship_across_all_betas`,
`test_the_record_does_not_say_the_cap_was_exhausted_19_percent_through`), which
scan the record body while permitting the correction notice to quote the errors
it documents.

---

## 2. Authenticated development baseline (§2 = observed)

Located read-only in the local repository. **The external ~7.6 GB shadow archive
on `/mnt/e` was not retrieved**; only the 2 MB episode CSV was needed.

| | |
|---|---|
| Episode CSV | `analysis/out/seed91_retry_20260802T112458Z/8x8/episodes_8x8_seed91_2M_retry.csv` |
| Sacred run | `results/sacred/qmix/lbforaging:Foraging-8x8-2p-2f-coop-v3/**16**` |
| Sacred status | **COMPLETED** |
| Exit marker | `exit_code=0`, `status=COMPLETED`, 2026-08-02T12:47:22Z |
| Seed | **91** |
| Environment | `lbforaging:Foraging-8x8-2p-2f-coop-v3` |
| Algorithm | QMIX |
| `selma_mode` | `shadow` (detector observation only; learning-identical to `off`) |
| Final `t_env` | 2 000 046 (recovery plan §12); last evaluation at 1 950 259 |
| Evaluation interval | **50 000** |
| Test episodes per evaluation | **100** |
| Rows | 44 003 total, **4 000 greedy**, **40 evaluation points** |
| **Episode-level greedy success** | **available** (per-episode `team_return`, `successful_load_events`) |

**Acceptance evidence.** The recovery plan
(`selma_seed91_recovery_and_rerun_plan.md`, untouched and untracked) records
retry `20260802T112458Z` as **COMPLETE** — exit code 0 at 2 000 046 steps, with
post-run verification and 48-variant calibration done (§12), and names this exact
episode CSV. Sacred runs **14 and 15** are the superseded attempts (status
`RUNNING`) and were **rejected**, not used.

Because episode-level success is available, the **primary** branch of the
preregistered non-floor rule applies; the weaker return-threshold fallback was
not needed.

---

## 3. Preregistered horizon rule (§3 = preregistered)

> Recorded in `analysis/selma_stage_e_recalibration_design.py` as executable
> constants **before the baseline learning values were read**. Only artifact
> identity (row counts, column presence, evaluation cadence) was inspected
> beforehand.

**Candidate horizons:** 1 000 000 / 1 500 000 / 2 000 000.

**Sustained non-floor onset** — with episode-level data, the first evaluation
point beginning a run of **five consecutive points at ≥ 5 % greedy success**.
(Fallback, unused here: five consecutive points of strictly positive mean return
also above the `beta=0` 500k floor, with the limitation reported.)

**Selection:** the smallest candidate that is **≥ 500 000 steps after onset** and
provides **≥ 8 scheduled greedy evaluations** after onset. If no candidate
qualifies, or onset cannot be established, **select 2 000 000 conservatively**.

The rule fixes the horizon from the *baseline*, never from whichever point makes
a `beta` look best. No `beta` condition is consulted.

---

## 4. Mechanical results (§4 = derived)

### 4.1 Baseline learning curve and onset

**Sustained non-floor onset: `t_env` 550 050.**
Greedy success at the five qualifying points: **5 %, 5 %, 14 %, 16 %, 23 %**.

The baseline leaves the floor at ~550k — which is *after* the entire 500k
exploratory pilot. **That alone explains why all four exploratory conditions sat
at 0.1 %–1.9 % greedy success: the pilot ended before the baseline had begun to
learn.** No `beta` could have been discriminated there.

### 4.2 Chosen horizon: **1 500 000**

| Candidate | Verdict |
|---|---|
| 1 000 000 | **rejected** — only 449 950 steps after onset; 500 000 required |
| **1 500 000** | **selected** — 949 950 steps after onset, 18 evaluations |
| 2 000 000 | not needed (rule takes the smallest qualifying) |

### 4.3 Per-100k call demand from the completed pilot

Three quantities are kept strictly apart. A repeated semantic key costs **no**
call under a perfect cache, so **calls for full coverage = distinct keys**, not
decisions and certainly not detector events.

| Condition | Detector events | Decisions | Distinct keys | Ideal cache hits | Calls for full coverage | Attempts made | Coverage |
|---|---|---|---|---|---|---|---|
| `beta=0.00` | 1 294 | 1 294 | 1 236 | 58 | **1 236** | 1 000 | 81.2 % |
| `beta=0.05` | 1 660 | 1 660 | 1 588 | 72 | **1 588** | 1 000 | 63.2 % |
| `beta=0.10` | 1 996 | 1 996 | 1 894 | 102 | **1 894** | 1 000 | 53.5 % |
| `beta=0.25` | 893 | 893 | 853 | 40 | **853** | 853 | 100 % |

**New distinct keys per 100k block** — the demand curve is *accelerating*, not
saturating:

| Condition | 0–100k | 100–200k | 200–300k | 300–400k | 400–500k |
|---|---|---|---|---|---|
| `beta=0.00` | 152 | 173 | 148 | 202 | **559** |
| `beta=0.05` | 99 | 90 | 162 | 353 | **884** |
| `beta=0.10` | 162 | 200 | 223 | 579 | **730** |
| `beta=0.25` | 102 | 124 | 118 | 193 | 316 |

The late-training surge is genuine, not a cap artifact: the semantic digest is
recorded for **every** decision including `budget_exhausted` ones, by the Stage D
instrumentation design. In `beta=0.05`'s final block, 884 distinct questions
arose and only 296 were funded.

### 4.4 Projections to 1 500 000 (five methods, all labelled)

| Condition | Distinct @500k | Naive linear | Final-100k rate | Max-100k rate | Cumulative growth | **Worst case** | +25 % |
|---|---|---|---|---|---|---|---|
| `beta=0.00` | 1 236 | 3 708 | 6 826 | 6 826 | 4 266 | **6 826** | 8 533 |
| `beta=0.05` | 1 588 | 4 764 | **10 428** | 10 428 | 6 252 | **10 428** | 13 035 |
| `beta=0.10` | 1 894 | 5 682 | 9 194 | 9 194 | 7 001 | **9 194** | 11 493 |

The final-100k and max-100k methods coincide because the last block *was* the
maximum in every condition. The naive linear method is the **lowest** estimate
here — the opposite of the usual assumption — because demand is accelerating.

### 4.5 Common call cap: **13 500**

Rule: conservative maximum across retained betas → +25 % → round up to the next
500 → never below 3 000.

```
max(6 826, 10 428, 9 194) = 10 428  →  ×1.25 = 13 035  →  →13 500  →  ≥3 000 ✓
```

**One cap for all six conditions.** A per-`beta` cap would reintroduce exactly
the coverage confound that made the exploratory pilot uninterpretable.

### 4.6 Cost ceiling: **$3.49 per condition, $20.94 total**

Frozen 2026-08-17 table: input $0.20/M, cached input $0.02/M, output $1.25/M.

Observed per-call tokens were **degenerate**: every one of the 3 853 pilot calls
used exactly **877 input / 25 output** tokens, because the real detector always
produced 6 candidate plans (3 admissible cells, 2 recipients), giving an
identical prompt shape. So p95 = median = max here.

```
13 500 × (877 × $0.20/M + 25 × $1.25/M) = $2.78978  →  ×1.25  →  $3.49
```

The p95-based rule is used regardless, because it is the rule that would bind if
the distribution widened (a 3-recipient event yields 24 plans and a ~1 500-token
prompt). Cached input is **provider-side prompt caching**, not the application's
SQLite response cache; it was 0 throughout and is not assumed.

---

## 5. Frozen revised protocol (§5 = frozen)

### 5.1 Beta grid: **0.00, 0.05, 0.10** — `beta = 0.25` excluded

**This is an adaptive transition from exploratory to confirmatory calibration,
and is stated as such.** `beta = 0.25` is dropped because:

* it was the **only** positive `beta` with complete guidance coverage (100 %);
* it stayed **within** both shaping-safety limits (max 26.7 % of the smallest
  positive reward against a 50 % limit; P95 11.7 % against 25 %) — so it is not
  excluded on safety grounds;
* it nonetheless produced the **lowest** primary AUC (0.00038) and greedy success
  (0.1 %) of all four conditions, on the one condition that was *not* handicapped
  by missing guidance;
* the exploratory pilot is being used **only to narrow the search**, not to make
  a performance claim.

The honest framing: this narrowing uses exploratory evidence, so the revised
pilot is confirmatory **for the retained grid only**. It cannot rehabilitate
`beta = 0.25`, and it does not claim `0.25` is harmful — only that scarce budget
is better spent elsewhere.

### 5.2 Development seeds: **91 and 92**

Neither is a final evaluation seed. **Seed 99 is detector held-out evidence**
(`selma_stage_b_seed99_heldout_validation_report.md`) and must never be reused
for `beta` selection; the design script raises if it appears in the matrix. **No
final evaluation seed was inspected in this task.**

### 5.3 The six conditions

3 betas × 2 development seeds. Identical in everything except `beta` and `seed`.

| Condition | `beta` | seed | `t_max` | call cap | cost cap |
|---|---|---|---|---|---|
| `beta_0p00_seed91` | 0.00 | 91 | 1 500 000 | 13 500 | $3.49 |
| `beta_0p00_seed92` | 0.00 | 92 | 1 500 000 | 13 500 | $3.49 |
| `beta_0p05_seed91` | 0.05 | 91 | 1 500 000 | 13 500 | $3.49 |
| `beta_0p05_seed92` | 0.05 | 92 | 1 500 000 | 13 500 | $3.49 |
| `beta_0p10_seed91` | 0.10 | 91 | 1 500 000 | 13 500 | $3.49 |
| `beta_0p10_seed92` | 0.10 | 92 | 1 500 000 | 13 500 | $3.49 |

Common to every condition: `lbforaging:Foraging-8x8-2p-2f-coop-v3`,
`time_limit=50`, QMIX, `selma_mode=active`, detector `wf15_m3_t5_ibon`, provider
`openai`, model **`gpt-5.4-nano-2026-03-17`**, prompt **v3**, response schema v2,
strict structured output, reasoning `none`, `store=false`, no tools, no
streaming, max output 512, timeout 30 s, 1 retry, **TTL 15**, common team reward,
`cache_mode=read_write` with a **separate empty cache per condition**,
`use_cuda=False`, and **zero provider/cache/guidance activity in greedy
evaluation**.

**Not executed in this task.**

---

## 6. Revised selection rule (preregistered for the future runs)

### 6.1 Safety exclusions (unchanged, plus a coverage floor)

NaN/Inf · reward-reconciliation failure · telescoping failure · invalid physical
guidance · provider/cache/accounting failure · shaping dominance · learner
instability · materially unequal guidance coverage · API/cost-cap exhaustion.

> **Hard coverage requirement: every condition must reach ≥ 95 % guidance
> coverage.** If any condition falls below it, **`beta` remains UNRESOLVED** and
> **no post-hoc cap increase is permitted** — raising the cap after seeing the
> shortfall would convert a preregistered design into a tuned one.

### 6.2 Metrics

**Primary:** greedy-return AUC over the mechanically selected non-floor
evaluation window (onset → horizon).

**Secondary, in declared order:** final-window greedy return · greedy success ·
cooperative loads · failed-LOAD resolution · stability · shaping magnitude ·
detector/event demand.

### 6.3 Selection

A positive `beta` is selected **only if all five hold**:

1. it passes **every** safety gate on **both** development seeds;
2. its direction of effect versus `beta = 0` is **non-negative on both seeds**;
3. it is **not clearly dominated** by the other positive `beta`;
4. guidance coverage is **comparable** across conditions;
5. where evidence overlaps, the **declared parsimony rule selects the smaller
   `beta`**.

> **If the two seeds disagree materially, or both positive betas remain at the
> floor, report UNRESOLVED rather than forcing a selection.**

Final/evaluation seeds are never inspected.

---

## 7. Limitations and open decisions (§7 = limitations)

1. **Budget and wall-clock authorisation is required and is much larger than
   anything used so far.** $20.94 total is ~21× the Stage E global cap of $1.00.
   At the observed 0.85 s median latency, 13 500 calls ≈ **3.2 h of API latency
   per condition**; with ~100 min of compute for 1.5M steps that is ~5 h per
   condition and **~30 h for all six**. This design should not be executed
   without explicit authorisation of both.
2. **The demand curve is accelerating, so the cap could still bind.** Projections
   extrapolate a rate that tripled in the final block. If acceleration continues
   past 500k, 13 500 may be insufficient; if it plateaus, the cap is generous.
   The ≥95 % coverage gate (§6.1) is what makes this safe: an insufficient cap
   yields UNRESOLVED rather than a confounded answer.
3. **Two seeds is the practical minimum, not a sufficient sample.** It can reveal
   disagreement but cannot estimate between-seed variance.
4. **The token distribution had zero spread** in the pilot, so the p95 rule was
   untested against real variation. A layout with more recipients would widen it.
5. **`beta = 0.25` is excluded on exploratory evidence**, which is an adaptive
   narrowing (§5.1). If the retained grid returns UNRESOLVED, revisiting `0.25`
   at full coverage is a legitimate next step and should be preregistered afresh.
6. **Onset was measured on the `beta = 0` baseline**, which is the right
   reference but assumes shaping does not move onset earlier. If a positive
   `beta` leaves the floor sooner, the shared window is conservative for it.
7. The baseline is `selma_mode=shadow`, learning-identical to `off` but not
   byte-verified as such in this task; the Stage C equivalence verifier covers
   that property separately.

---

## 8. Confirmations

* **No API call, no network access, no training run** occurred in this task.
* **No credential was read**; `~/.config/selma/openai.env` was neither opened nor
  sourced.
* **No final/test evaluation seed was inspected.** Seed 99 was identified as
  detector held-out evidence and excluded by construction.
* **The recovery plan was read only** — never modified, moved, staged or deleted;
  it remains untracked.
* **Frozen Stage D contracts unchanged** — `detector.py`, `joint_plans.py`,
  `provider.py`, `cache.py`, `openai_provider.py` (prompt v3, response schema,
  cache key, physical validator) are byte-identical to `c7a318e`.
* **The exploratory pilot's evidence and its UNRESOLVED conclusion are intact**;
  only two wordings were corrected, under a labelled notice.
* **Nothing in §5 was executed.**
