# SELMA Stage B — 10x10 cross-condition robustness validation report

**Status: CORRECTED-PROTOCOL ANALYSIS COMPLETE (see §9).** The literal-protocol outcome below is
preserved as the original record; PROTOCOL ERRATUM 001 corrected E4's operationalisation before any
ranking was computed. Original status line, retained:

> **ANALYSIS HALTED AT THE SELECTION STEP.** Integrity, reconciliation, invariants and
per-variant metrics are complete and reported. **The frozen selection rule could not be applied**,
because the frozen eligibility rule E4, applied literally, yields an **empty** eligible set on this
condition. Per the standing instruction — *"If the existing analysis tooling cannot apply the frozen
protocol exactly, stop and report the incompatibility rather than improvising a new rule"* — no
substitute rule was invented, no ranking was produced, and no configuration was selected.

**Run:** `seed91_val10x10_20260802T151245Z` · **HEAD** `92ece20` · **Frozen configuration under
test:** `wf15_m3_t5_ibon`

**Nothing was retuned or replaced. `src/` and `default.yaml` are unmodified. Nothing committed or
pushed.**

---

## 1. Integrity verification

| Check | Result |
|---|---|
| `exit_code` | **0** |
| `status` | **COMPLETED** |
| Final `t_env` | **2 000 007** ≥ `t_max` = 2 000 000 ✅ |
| Runtime | 1 h 33 m 49 s (5 629 s) |
| All expected artifacts exist | ✅ 10/10 |
| All artifacts parse | ✅ summary JSON, episodes CSV, raw CSV, manifest, sampler data |
| File sizes stable | ✅ raw CSV and summary unchanged across re-stat; 0 open writers |
| NUL corruption | ✅ **0 NUL bytes** in all 3 708 043 487 bytes scanned |
| Truncation / incomplete final record | ✅ final record has 55/55 fields; file terminates `\r\n` |
| Rows with wrong field count | ✅ **0** of 10 014 507 |
| Header vs `COLUMNS` in `shadow_log.py` | ✅ 55 = 55 |
| Exactly 48 variants | ✅ 48 in summary, 48 in raw CSV |

**Same-run correspondence** — summary, episodes and raw data all match the manifest: condition
`lbforaging:Foraging-10x10-3p-3f-v3`, seed 91, grid `[10,10]`, `n_agents` 3, shadow mode, HEAD
`92ece20`. Train-mode episode rows (40 019) equal `episodes_seen` (40 019). `observed_steps` is
2 000 007 for all 48 variants.

**Run configuration vs manifest** — verified from `run.log`: `seed=91`, `t_max=2000000`,
`use_cuda=False`, `selma_mode=shadow`, `w_inbound=3`, `cooldown_steps=5`, `rearm_steps=3`,
`max_events_per_episode=2`, `qmix`/`q_learner`, `env_args.time_limit=50`, env key as recorded. All
match.

**Zero-guarantees:** `provider_call_count=0`, `query_count=0`, `active_subgoal_count=0`,
`total_shaping_reward=0.0`.

*Two flags raised during checking were my own errors, not data defects, and are recorded for
transparency:* an apparent missing final newline (I wrote `b'\\n'` — a literal backslash-n — in the
check), and an apparent 83-field final record (naive comma-splitting of a line containing quoted
tuple fields). Both cleared under correct parsing. Separately, §1.3 of the Stage C contract states
the shadow schema has "56 fields"; the correct count is **55**, and that document has been corrected.

## 2. Raw-data reconciliation

Performed with the existing frozen tooling (`analysis/selma_calibration_report.py --verify-csv`),
which streams the full raw CSV and independently re-derives counters:

| Quantity | Value |
|---|---|
| `reconciled` | **true** |
| `reconciliation_problems` | **[]** |
| Variants present | **48 / 48** |
| CSV rows | 10 014 507 |
| — `raw_candidate` rows | 9 519 496 |
| — `coalesced_event` rows | 495 011 |
| Row identity | 9 519 496 + 495 011 = 10 014 507 ✅ |
| Episodes with ≥1 row | 10 304 |
| Max `t_env` in CSV | 1 999 856 |
| Final-event count mismatches | **0** — exact agreement for all 48 |
| Recipient-distribution mismatches | **0** — exact agreement for all 48 |
| Semantically-invalid events (re-derived) | **0** for all 48 |

## 3. Invariants — condition-adjusted

15 invariants × 48 variants = 720 checks. **All pass** except the E4 coalescing clause discussed in
§4, which is an eligibility-rule question rather than a data-integrity failure.

Verified for every variant: gate stage identity; inbound stage identity; the full suppression ledger
(`raw_candidates_after_inbound − same_food_coalesced − (latch + cooldown + rearm + budget) =
final_events`); `suppressed_by_incomplete_team_window ⊆ suppressed_by_team_gate`; per-episode
histograms summing to 40 019; phase and recipient distributions summing to `final_events`; the event
cap never exceeded; resolution denominators balancing against censoring;
`semantically_invalid_events = semantically_invalid_candidates = 0`.

**Condition-adjusted recipient rule applied as pre-declared.** Recipient counts must lie in **2–3**;
the 8x8 requirement of *exactly two* recipients was **not** applied. Observed: recipient counts take
values `{2, 3}` only, in all 48 variants — 315 389 two-recipient and 179 622 three-recipient events
across the grid, with three-recipient events being 35.8–36.8 % of events per variant. Recipient
counts reconcile exactly with `final_events` for every variant, both in the summary and in the
independently re-derived raw-CSV counts.

**Solo-load and coalition-relevant populations, reported separately.** Over 40 019 training
episodes: 10 611 successful load events, of which **10 133 (95.5 %) are solo loads** and 478 (4.5 %)
are multi-agent; 363 are necessary-cooperative. 22.6 % of episodes contain at least one solo load;
1.2 % contain at least one multi-agent load. Solo loads being non-zero is expected and legitimate —
this is not a `-coop-` condition. They are a **separate population** from the coalition-required
failed-LOAD population the detector triggers on, and are reported here as a diagnostic only. (In the
8x8 `-coop-` condition solo loads are zero by construction.)

## 4. Eligibility — and the protocol blocker

Frozen rules E1–E4 applied exactly as recorded in `selma_stage_b_selection_record.md` §1.1:

| Rule | As frozen | 10x10 result |
|---|---|---|
| E1 | team gate enabled | **36 / 48** pass |
| E2 | inbound suppression enabled | **24 / 48** pass |
| E3 | all invariants pass | **48 / 48** pass |
| E4 | zero invalid and duplicate events — recorded as `semantically_invalid_events = 0`, `semantically_invalid_candidates = 0`, **and `same_food_coalesced = 0`** | **0 / 48** pass |
| | **Eligible under literal E1–E4** | **0** |

### 4.1 The incompatibility

`same_food_coalesced` is **non-zero in all 48 variants**, ranging 2 473–3 493. In the 8x8
development condition it was **zero in all 48**, so the clause bound on nobody and the discrepancy
could not surface.

Inspection of `src/selma/detector.py:530-531` shows what the counter measures:

```python
if len(qualifying) > 1:
    state.bump("same_food_coalesced", len(qualifying) - 1)
```

It counts candidates **merged into** a single event when several agents independently qualify on the
same food at the same step. The pre-calibration audit describes the same mechanism: *"food-level
coalescing (`same_food_coalesced`, one `coalesced_event` per food per rising edge with sorted trigger
agent ids)"*. The suppression ledger subtracts it from the candidate pool to yield `final_events`.

So a non-zero value means the detector **prevented** duplicate events — it is the de-duplication
mechanism working, not duplicate events surviving into the output. It is not a defect, and with
3 agents and 3 food items multiple agents qualifying on the same food is far more likely than with
2 agents, which is why the counter is zero on 8x8 and non-zero here.

The frozen E4 text therefore contains a clause that does not test what E4 is named for. Read
literally it is a "no coalescing ever occurred" test, which a 3-agent condition will essentially
always fail. Read by intent — zero invalid events and zero duplicate *final* events — all 48
variants pass, since duplicate final events are zero by construction whenever coalescing works.

### 4.2 Why the analysis stops here

Under the literal frozen rule the eligible set is **empty**. The frozen selection rule cannot be
applied to an empty set: there are no normalisation endpoints, no ranking and no Pareto frontier to
compute. This is not a marginal case that could be resolved by a convention.

The standing instruction is explicit: do not change eligibility rules after observing results, and
stop and report an incompatibility rather than improvise. Repairing E4 now — after seeing that the
literal form excludes everything — would be exactly the post-hoc rule change the frozen-selection
discipline exists to prevent, even though the repair looks obviously correct. **The decision is the
user's, not mine.**

**Consequently not computed and not reported anywhere in this document:** eligible ranking,
normalisation endpoints, frozen-rule distances, Pareto frontier, and `wf15_m3_t5_ibon`'s eligibility,
rank, distance or margins.

### 4.3 What is unaffected

**The 8x8 selection stands regardless of how this is resolved.** In the 8x8 condition
`same_food_coalesced` is zero for all 48 variants, so the E4 clause excluded no variant there. The
8x8 eligible set of 18, its normalisation endpoints (coverage `[21.0709, 25.6156]`, res@10
`[8.2449, 10.8175]`), and the selection of `wf15_m3_t5_ibon` at d = 0.607348 are unchanged and remain
frozen.

### 4.4 Options for resolution — for the user to decide

Both are recorded without preference; neither has been applied.

* **Option A — treat E4's `same_food_coalesced = 0` clause as a specification defect and correct it**
  to test duplicate *final* events (which are zero by construction), leaving the invalid-events
  clauses intact. Under this reading E1∩E2∩E3∩E4 would yield a non-empty eligible set and the frozen
  rule would apply unchanged. This must be recorded as an amendment to a frozen rule, made after
  seeing that the literal form was vacuous, with the reasoning and timing stated.
* **Option B — treat the literal rule as binding** and report the 10x10 validation as *not
  evaluable* under the frozen protocol. Cross-condition robustness would remain an open question,
  and the 8x8 selection would stand on development-set evidence alone.

## 5. Per-variant metrics — all 48

Computed with the frozen metric definitions, unchanged. Full table:
`analysis/out/seed91_val10x10_20260802T151245Z/10x10/variant_metrics_10x10.csv`.
Machine-readable results: `…/validation_results_10x10.json`. Report generated by the frozen tooling:
`docs/selma_stage_b_seed91_10x10_calibration.md` and `…/calibration_report.json`.

Ranges across the grid (descriptive only — **no selection applied**):

| Metric | Range |
|---|---|
| Final events | 8 701 – 12 403 |
| Events per 1 000 steps | 4.350 – 6.201 |
| Episode coverage | 18.72 % – 25.75 % |
| Natural resolution @10 | 1.31 % – 1.78 % |
| Cap-binding episodes | 2.72 % – 5.46 % |
| Gate-open | 94.1 % – 100 % |
| Three-recipient share | 35.8 % – 36.8 % |
| `same_food_coalesced` | 2 473 – 3 493 |
| Timing early / mid / late | 44.8–60.5 / 22.2–31.9 / 17.2–23.3 % |

## 6. Cross-condition sensitivity — descriptive only

**Four environmental properties changed jointly.** `Foraging-8x8-2p-2f-coop-v3` →
`Foraging-10x10-3p-3f-v3` alters grid size (8×8 → 10×10), agent count (2 → 3), food count (2 → 3)
and enforced-cooperation status (`-coop-` enforced → absent) **simultaneously**. The environment key
is the only configured field that differs, but it is **not a single experimental variable**.

Accordingly:

* **No difference below is attributed to grid size, agent count, food count or cooperation status
  individually.** Any difference is joint evidence across all four.
* **Firing rates are not directly like-for-like.** Without the `-coop-` constraint, 95.5 % of
  successful loads are solo, so the coalition-required failed-LOAD population the detector triggers
  on is drawn from a materially different distribution.
* **No statistical validation is claimed from one seed.** This is a single seed-91 trajectory per
  condition; nothing here supports a significance claim.

| Quantity | 8x8 (development) | 10x10 (validation) |
|---|---|---|
| Episodes | 40 003 | 40 019 |
| Observed steps | 2 000 046 | 2 000 007 |
| Events/1k range | 4.688 – 6.670 | 4.350 – 6.201 |
| Coverage range | 21.07 % – 29.66 % | 18.72 % – 25.75 % |
| **Natural resolution @10 range** | **8.24 % – 17.09 %** | **1.31 % – 1.78 %** |
| Recipient counts | exactly 2 | 2 and 3 (≈36 % three) |
| `same_food_coalesced` | 0 (all 48) | 2 473 – 3 493 (all 48) |
| Solo loads | 0 by construction | 10 133 (95.5 % of successful loads) |
| Raw CSV | 3.96 GiB, 11 857 607 rows | 3.45 GiB, 10 014 507 rows |

The most conspicuous difference is natural-resolution@10, roughly an order of magnitude lower on
10x10. This is a genuine observation but **not interpretable as detector behaviour alone**: it is
confounded by all four changed properties, and in particular by the absence of enforced cooperation,
which changes what "resolves unaided" means. Firing rates and coverage are broadly comparable in
magnitude; both shift modestly downward. **No conclusion about transferability is drawn here**,
because the selection step that would ground such a conclusion could not be executed (§4).

## 7. Operational note — raw-CSV size projection

| | |
|---|---|
| Projected (central) | ≈ 2.21 GiB |
| Projected (no-decay upper bound) | ≈ 2.39 GiB |
| Provisioned | 3.5 GiB |
| **Actual** | **3.45 GiB** (3 708 043 487 B) |

The projection **underestimated** the footprint. It extrapolated from the aborted 10x10 run's
1 281 B/step at `t_env` 750 218, scaled by the cumulative-rate decay factor (0.927) measured on the
**2-agent cooperative** 8x8 condition. That decay shape did not transfer: with 3 agents, more agents
qualify per food per step, generating more `raw_candidate` rows than the 2-agent shape predicted —
the same mechanism that makes `same_food_coalesced` non-zero here.

The outcome landed inside the 3.5 GiB provision, so no run was at risk and E: retained ~357 GiB free
throughout. **This is an operational estimation error, not an experimental validity issue**: it
concerns capacity planning only and has no bearing on the correctness, completeness or integrity of
the data, all of which are verified in §1–§3. Future cross-condition projections should not borrow a
per-step growth shape across conditions that differ in agent count.

## 8. Status

Integrity ✅ · Reconciliation ✅ (exact, all 48) · Invariants ✅ (720 checks, condition-adjusted) ·
Per-variant metrics ✅ · **Selection ⛔ blocked at E4** · Cross-condition comparison — descriptive only.

`wf15_m3_t5_ibon` remains the frozen 8x8-selected configuration, **not retuned and not replaced**.
`src/` and `default.yaml` unmodified. Stage C not started. No other seed or environment launched. No
shaping enabled, no provider or LLM called. Nothing committed or pushed.

**Awaiting a decision on §4.4 before any ranking or Pareto frontier is computed.**

---

## PROTOCOL ERRATUM 001 — 2026-08-02

**Type:** operationalisation correction to eligibility rule E4. **Not** detector retuning: no
detector parameter, metric formula, selection score or tie-break is changed, and no training was
rerun.

**Timing, stated plainly:** raised **after** the 10x10 run completed and its integrity,
reconciliation, invariants and per-variant metrics were computed, but **before** any 10x10
normalisation endpoint, ranking or Pareto frontier was computed. All analysis downstream of this
erratum is labelled **"corrected-protocol analysis following a documented post-run, pre-ranking
operationalisation erratum"**. It is **not** an untouched confirmatory analysis.

### E.1 Original literal outcome — preserved

E4 as originally frozen, applied literally to the 10x10 condition:

| Rule | 10x10 |
|---|---|
| E1 team gate enabled | 36/48 |
| E2 inbound suppression enabled | 24/48 |
| E3 all invariants pass | 48/48 |
| **E4 literal** (including `same_food_coalesced = 0`) | **0/48** |
| **Eligible under literal E1–E4** | **0** |

**The original literal protocol was non-applicable to the 10x10 condition.** With an empty eligible
set there are no normalisation endpoints, no ranking and no Pareto frontier. This outcome is
retained permanently here and in the machine-readable results as `literal_E4_eligible_count = 0`.

### E.2 Exact old E4 wording

> `| E4 | Zero invalid and zero duplicate events (semantically_invalid_events = 0,`
> `semantically_invalid_candidates = 0, same_food_coalesced = 0) | satisfied by all 48 |`

### E.3 Evidence for the correction

**Intended semantic purpose of E4:** *zero semantically invalid or duplicate FINAL events.*

**`same_food_coalesced` does not measure that.** `src/selma/detector.py:530-531`:

```python
if len(qualifying) > 1:
    state.bump("same_food_coalesced", len(qualifying) - 1)
```

It increments by `len(qualifying) - 1` when several agents independently qualify on the **same food
at the same step**, counting candidate records **deliberately merged into one final event**. The
suppression ledger subtracts it from the candidate pool to yield `final_events`. The pre-calibration
audit describes the same mechanism: *"food-level coalescing (`same_food_coalesced`, one
`coalesced_event` per food per rising edge with sorted trigger agent ids)"*.

A non-zero value is therefore **evidence that deduplication operated**, not that duplicates survived.

| Evidence | 8x8 | 10x10 |
|---|---|---|
| `same_food_coalesced` | 0 across all 48 | **non-zero across all 48**, range 2 473 – 3 493 |
| `semantically_invalid_events` (final) | 0 across all 48 | **0 across all 48** |
| `semantically_invalid_candidates` | 0 across all 48 | 0 across all 48 |
| `coalesced_event` rows | 525 879 | 495 011 |
| Distinct `event_id`s among them | 525 879 | 495 011 |
| **Surviving duplicate final events** | **0** (0/48 variants) | **0** (0/48 variants) |
| Raw-CSV vs summary reconciliation | exact, 48/48 | exact, 48/48 — `reconciled: true`, no problems, 0 final-event and 0 recipient-distribution mismatches |

Duplicate detection uses the **existing** event identity `event_id`
(`ep{episode}-{variant}-food{r},{c}-t{step}`, `detector.py:595-598`) over `coalesced_event` rows. No
new metric was created and event identity was not redefined.

The clause was invisible on 8x8 because with 2 agents in a `-coop-` condition no two agents ever
coalesced on one food; with 3 agents it fires constantly. The literal rule is therefore a *"no
coalescing ever occurred"* test, which a 3-agent condition will essentially always fail.

### E.4 Corrected rule — frozen

**E4-corrected**, frozen as of this erratum and before any ranking:

* `semantically_invalid_final_events == 0`; **and**
* `surviving_duplicate_final_events == 0` after coalescing — measured as duplicate `event_id`
  occurrences among `coalesced_event` rows, using the existing identity definition.

`same_food_coalesced` **remains reported as a descriptive diagnostic** and is **never** an
eligibility failure.

E1, E2 and E3 are unchanged. The selection score
`d = sqrt((1 - coverage_norm)^2 + res@10_norm^2)`, its eligible-only min–max normalisation domain,
and the tie-breaks (lower events per 1 000 steps, then variant name) are **unchanged**.

---

## 9. Corrected-protocol results

**Protocol status: corrected-protocol analysis following a documented post-run, pre-ranking
operationalisation erratum (PROTOCOL ERRATUM 001, 2026-08-02).** This is **not** an untouched
confirmatory analysis. Both outcomes are retained: literal E4 → 0/48 eligible; corrected E4 → the
results below.

### 9.1 8x8 consistency check — nothing changed

E4-corrected applied retrospectively to the preserved 8x8 records, purely as a consistency check
(no reselection, no retuning):

| Quantity | Original | Under E4-corrected |
|---|---|---|
| Eligible set | 18 | **18 — unchanged** |
| Coverage endpoints | [21.0709, 25.6156] | **unchanged** |
| res@10 endpoints | [8.2449, 10.8175] | **unchanged** |
| Rank 1 | `wf15_m3_t5_ibon`, d = 0.607348 | **unchanged** |
| Rank 2 | `wf15_m2_t10_ibon`, d = 0.613259 | **unchanged** |
| Ranking order | — | **unchanged** |

Expected, since `same_food_coalesced` is 0 for all 48 8x8 variants, so the removed clause excluded
nobody there.

### 9.2 10x10 eligibility under corrected E4

**Eligible: 18/48** — this equals the 8x8 count coincidentally, not by assumption; it was computed
independently.

| Exclusion reason | Variants |
|---|---|
| E2 only: inbound suppression off | 18 |
| E1 + E2: ungated **and** inbound off | 6 |
| E1 only: ungated (team gate `none`) | 6 |
| **Total ineligible** | **30** |

E3 excluded none (48/48 pass). E4-corrected excluded none: `semantically_invalid_final_events = 0`
and `surviving_duplicate_final_events = 0` for all 48.

### 9.3 Eligible-only normalisation endpoints (10x10, recomputed — 8x8 endpoints **not** reused)

| Axis | Min | Max | Range |
|---|---|---|---|
| Coverage % | 18.7236 | 23.4514 | 4.7278 |
| res@10 % | 1.3110 | 1.4010 | **0.0900** |

### 9.4 Complete frozen-rule ranking — 18 eligible

Score, normalisation domain and tie-breaks unchanged.

| # | Variant | cov % | res@10 % | n_cov | n_res | d | ev/1k | gate % | Pareto |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `wf15_m2_t10_ibon` | 21.8721 | 1.3493 | 0.665962 | 0.424899 | **0.540482** | 5.2120 | 95.78 | ✅ |
| 2 | `wf10_m2_t10_ibon` | 21.6247 | 1.3583 | 0.613636 | 0.525732 | 0.652435 | 5.1280 | 95.78 | |
| 3 | `wf15_m2_t15_ibon` | 20.4128 | 1.3240 | 0.357294 | 0.144286 | 0.658703 | 4.7765 | 94.08 | ✅ |
| 4 | `wf5_m2_t10_ibon` | 21.2949 | 1.3577 | 0.543869 | 0.518428 | 0.690524 | 5.0475 | 95.78 | |
| 5 | `wf15_m2_t5_ibon` | 23.4514 | 1.3763 | 1.000000 | 0.725847 | 0.725847 | 5.6620 | 97.68 | ✅ |
| 6 | `wf10_m2_t15_ibon` | 20.1654 | 1.3317 | 0.304968 | 0.229414 | 0.731915 | 4.6985 | 94.08 | |
| 7 | `wf5_m2_t15_ibon` | 19.8781 | 1.3266 | 0.244186 | 0.172954 | 0.775350 | 4.6300 | 94.08 | |
| 8 | `wf10_m2_t5_ibon` | 23.2315 | 1.3841 | 0.953488 | 0.812022 | 0.813353 | 5.5810 | 97.68 | |
| 9 | `wf5_m2_t5_ibon` | 22.8941 | 1.3847 | 0.882135 | 0.819177 | 0.827612 | 5.4985 | 97.68 | |
| 10 | `wf10_m3_t5_ibon` | 21.4398 | 1.3825 | 0.574524 | 0.794867 | 0.901578 | 5.0850 | 97.68 | |
| 11 | `wf5_m3_t5_ibon` | 21.2024 | 1.3805 | 0.524313 | 0.771872 | 0.906678 | 5.0555 | 97.68 | |
| 12 | `wf10_m3_t15_ibon` | 18.9385 | 1.3277 | 0.045455 | 0.184888 | 0.972286 | 4.3785 | 94.08 | |
| 13 | `wf5_m3_t15_ibon` | 18.7236 | 1.3110 | 0.000000 | 0.000000 | 1.000000 | 4.3505 | 94.08 | ✅ |
| 14 | `wf10_m3_t10_ibon` | 20.1929 | 1.3770 | 0.310782 | 0.733037 | 1.006163 | 4.7465 | 95.78 | |
| 15 | `wf15_m3_t15_ibon` | 19.0834 | 1.3503 | 0.076110 | 0.436342 | 1.021747 | 4.4370 | 94.08 | |
| 16 | `wf5_m3_t10_ibon` | 19.9430 | 1.3752 | 0.257928 | 0.713248 | 1.029268 | 4.7160 | 95.78 | |
| 17 | `wf15_m3_t10_ibon` | 20.3328 | 1.3846 | 0.340381 | 0.817531 | 1.050455 | 4.8075 | 95.78 | |
| **18** | **`wf15_m3_t5_ibon`** | **21.5698** | **1.4010** | **0.602008** | **1.000000** | **1.076289** | **5.1440** | **97.68** | ❌ |

### 9.5 Pareto frontier — 4 of 18

Same objectives (coverage ↑, res@10 ↓) and same dominance definition as 8x8.

| Variant | cov % | res@10 % | d | rank |
|---|---|---|---|---|
| `wf15_m2_t5_ibon` | 23.4514 | 1.3763 | 0.725847 | 5 |
| `wf15_m2_t10_ibon` | 21.8721 | 1.3493 | 0.540482 | 1 |
| `wf15_m2_t15_ibon` | 20.4128 | 1.3240 | 0.658703 | 3 |
| `wf5_m3_t15_ibon` | 18.7236 | 1.3110 | 1.000000 | 13 |

The 8x8 frontier held 13 of 18; the 10x10 frontier holds 4 of 18.

### 9.6 `wf15_m3_t5_ibon` — full 10x10 result

| Field | Value |
|---|---|
| Eligible | **Yes** (E1 gate `W_team=5`; E2 inbound on; E3 pass; E4-corrected pass) |
| Exclusion reason | none |
| **Pareto-efficient** | **No** |
| **Frozen-rule rank** | **18 of 18 eligible — last** |
| Coverage raw / normalised | 21.5698 % / 0.602008 |
| res@10 raw / normalised | 1.4010 % / **1.000000 — the maximum of the eligible set** |
| **Distance d** | **1.076289** |
| Margin to rank 17 (`wf15_m3_t10_ibon`, d = 1.050455) | **+0.025834** |
| Gap to rank 1 (`wf15_m2_t10_ibon`, d = 0.540482) | **+0.535807** |
| Events per 1 000 steps | 5.1440 (10 288 final events) |
| Cap-binding | 4.1380 % |
| Timing early / mid / late | 55.18 / 25.34 / 19.48 % |
| Recipients | 2 → 6 551; 3 → 3 737 (36.32 % three-recipient) |
| `same_food_coalesced` | 2 953 — diagnostic only, not an exclusion |
| Gate open | **97.68 %** |

**Scale caveat, stated because it materially affects how rank 18 should be read.** The coverage axis
spans 4.7278 pp but the res@10 axis spans only **0.0900 pp**. Min–max normalisation stretches that
0.09 pp onto the full [0, 1] axis, so `wf15_m3_t5_ibon` scoring `n_res = 1.000000` reflects an
absolute res@10 difference of **0.09 percentage points** from the best eligible variant. The 8x8
res@10 range was 2.5726 pp — **28.6× wider**. The ranking is computed exactly as frozen and is
reported as such, but the separation it expresses on this axis is very small in absolute terms.

### 9.7 Gate-open rate — a major cross-condition-sensitivity finding

Under this 10x10 condition the selected `W_team=5` team gate was open **97.68 %** of observed steps.
Across the eligible set the gate-open rate ranges only **94.08 %–97.68 %**, versus **90.8 %–100 %**
on 8x8 with far more separation between gate settings.

**The team-stall gate — the strongest discriminating factor on 8x8 — is nearly always open here, so
it barely discriminates.** That is a substantive finding about how the detector transfers, and it is
recorded as such **under the frozen protocol**. It is **not** a reason to retune `W_team` or any
other parameter, and no such change is made.

### 9.8 Required qualifications

* **Four environmental properties changed jointly** — grid size (8×8 → 10×10), agent count (2 → 3),
  food count (2 → 3) and enforced-cooperation status (`-coop-` → absent). The environment key is the
  only configured field changed, but it is **not a single experimental variable**.
* **No difference may be attributed to grid size alone**, nor to agent count, food count or
  cooperation status individually. All differences are joint evidence.
* **This is one-seed external sensitivity evidence.** A single seed-91 trajectory per condition
  supports no statistical claim.
* **Firing rates are not like-for-like**: 95.5 % of successful loads here are solo, so the
  coalition-required failed-LOAD population differs in kind.
* **The 10x10 result cannot replace or retune the 8x8 selection.** `wf15_m3_t5_ibon` remains the
  frozen development configuration despite ranking last here. Its rank-18 position is cross-condition
  sensitivity evidence, not a selection outcome.
