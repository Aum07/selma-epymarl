# SELMA Stage B — consolidated detector evidence package (publication metrics)

**Purpose.** A single verified source for the dissertation Methods, Results and Limitations
sections. Every number here traces to a machine-readable artifact listed in §H; nothing is
re-derived, re-tuned or re-selected.

**Status.** Stage B is closed. Documentation and verified summarisation only — no training was
rerun, no metric formula changed, no detector parameter altered, and Stage C is not started.

**Validation source.** Base commit `92ece20` **plus the documented selected-variant plumbing patch**
(`selma_variant_id`; two files; `src/selma/` detector logic unchanged). Not a clean checkout.

**Protocol note.** The 10×10 analysis is a *corrected-protocol analysis following a documented
post-run, pre-ranking operationalisation erratum* (PROTOCOL ERRATUM 001, 2026-08-02). See §C.2.

---

## A. Detector method

### A.1 Operational definition of a struggle event

A **coordination stall** is a concrete environment event, not a statistical signal. A raw candidate
exists at step *t* for agent *i* and food *f* when all of the following hold:

* food *f* still exists;
* agent *i* has at least **m** coalition-required failed LOAD attempts on *f* within the last
  **W_fail** steps (a failed LOAD is coalition-required when the loading agents' combined level is
  below the food's level);
* the candidate is not latched, in cooldown, or awaiting re-arm.

A candidate becomes a **final event** only after passing the team gate, inbound-helper suppression,
semantic-feasibility validation, same-food coalescing, the latch/cooldown/re-arm lifecycle and the
per-episode budget.

### A.2 Inputs and outputs

**Inputs** (raw environment information only): per-step agent positions and levels; food positions
and levels; failed and successful LOAD events with loader identities and levels; raw team reward;
food removals; episode and step indices. The detector never reads training reward, Q-values, policy
parameters or any learner state.

**Outputs**: `raw_candidate` and `coalesced_event` rows (55 columns, `src/selma/shadow_log.py`), and
a per-variant aggregate summary (48 fields). Each final event carries an identity
`ep{episode}-{variant}-food{r},{c}-t{step}`, a target food, trigger agents, recipient agents, the
feasible helper coalition, candidate loading cells and a deterministic cell assignment.

### A.3 Suppression components

| Component | Rule |
|---|---|
| **Team gate** (`W_team`) | Suppress when the team has made environmental progress — positive raw team reward or a food disappearing — within the last `W_team` steps. Rationale: a team that is progressing is not stalled. |
| **Inbound-helper suppression** (`w_inbound`) | Suppress when a capable helper has been closing distance on the target food within the last `w_inbound` steps. Rationale: the stall is already being resolved. |
| **Semantic feasibility** | Drop candidates for which no feasible helper coalition exists. |
| **Same-food coalescing** | When several agents qualify on the same food at the same step, emit **one** event with sorted trigger-agent ids. This *prevents* duplicates; the counter records candidates merged. |
| **Latch** | Once an event fires for a `(variant, food)` pair, suppress further candidates on that food until released. |
| **Cooldown** (`cooldown_steps`) | Suppress for a fixed number of steps after an event. |
| **Re-arm** (`rearm_steps`) | Require a streak of consecutive steps with the condition false before the latch releases. |
| **Episode cap** (`max_events_per_episode`) | Hard limit on events per episode. |

### A.4 Processing order (derived from the implementation)

Verified against `src/selma/detector.py`; the order is load-bearing, since each stage's count
feeds the next.

```
for each step, for each variant, for each food with qualifying candidates:
    1. team gate        -> suppressed_by_team_gate          (+ incomplete_team_window subset)
    2. semantic check   -> semantically_invalid_candidates
    3. inbound helper   -> suppressed_by_inbound
    4. same-food coalescing (n_qualifying - 1 merged)  -> same_food_coalesced
    5. latch / cooldown / re-arm / budget -> suppressed_by_{latch,cooldown,rearm,budget}
    6. emit coalesced_event -> final_events
```

**Exact accounting identity, verified for every variant in every run:**

```
raw_candidates
  − suppressed_by_team_gate                        = candidates_after_team_gate
  − semantically_invalid_candidates − suppressed_by_inbound
                                                   = raw_candidates_after_inbound
  − same_food_coalesced
  − (suppressed_by_latch + suppressed_by_cooldown
     + suppressed_by_rearm + suppressed_by_budget) = final_events
```

`suppressed_by_incomplete_team_window` is a **subset** of `suppressed_by_team_gate` and must not be
subtracted separately.

### A.5 Frozen configuration

| Parameter | Value | Origin |
|---|---|---|
| Variant id | **`wf15_m3_t5_ibon`** | selected |
| `W_fail` | **15** | selected |
| `m` | **3** | selected |
| `W_team` | **5** | selected |
| Inbound suppression | **on** | selected |
| `w_inbound` | **3** | fixed across all 48 |
| `cooldown_steps` | **5** | fixed across all 48 |
| `rearm_steps` | **3** | fixed across all 48 |
| `max_events_per_episode` | **2** | fixed across all 48 |

`m` (failed-LOAD threshold) and `max_events_per_episode` (per-episode cap) are distinct quantities
that both happen to be small integers; they are not interchangeable.

### A.6 Pseudocode

```
on each environment transition (training mode only):
    update failed-LOAD history keyed by (agent, food)
    update team-progress window from raw reward and food removals
    for each food f with >= 1 qualifying agent:
        qualifying <- agents with >= m coalition-required failed LOADs on f in last W_fail steps
        if none: continue
        if team progressed within last W_team steps:      suppress(team_gate);      continue
        if no feasible helper coalition for f:             suppress(semantic);       continue
        if capable helper closed distance within w_inbound: suppress(inbound);       continue
        if |qualifying| > 1: same_food_coalesced += |qualifying| - 1     # merge, do not duplicate
        if latched(f) or cooldown(f) > 0 or rearm pending(f) or events_used == cap:
            suppress(corresponding reason);                              continue
        emit one coalesced_event(f, sorted(qualifying), recipients, coalition, cells)
        set latch(f); start cooldown(f); events_used += 1
    # observation only: no action, reward, RNG or learner state is touched
```

---

## B. Experimental protocol

### B.1 Roles of the three runs

| Run | Condition | Seed | Role |
|---|---|---|---|
| `seed91_retry_20260802T112458Z` | 8×8-2p-2f-coop | 91 | **Calibration / selection.** All 48 variants evaluated; the frozen configuration was chosen here. |
| `seed99_heldout_20260802T221211Z` | 8×8-2p-2f-coop | 99 | **Primary held-out validation.** Frozen configuration only; never used for tuning or selection. |
| `seed91_val10x10_20260802T151245Z` | 10×10-3p-3f (non-coop) | 91 | **Secondary cross-condition sensitivity.** Reported separately; not primary validation. |

**Why seed 91 was calibration.** A single seed was used for calibration so that all 48 variants
observe one identical trajectory, making differences attributable solely to detector
parameterisation and removing training noise from the comparison. The cost — that the calibration
reflects one policy trajectory — is carried as a limitation.

**Why seed 99 was held out.** The implementation specification requires one held-out validation
after freezing and before Stage C, on the same condition, to test whether the configuration chosen
on one trajectory behaves consistently on an unseen one.

**Why seed 99 was never used for retuning.** The selection rule, its normalisation domain, the
eligibility criteria and the configuration were all frozen and recorded *before* the seed-99 run
launched. Seed 99 ran the frozen variant **only** — the 48-variant grid was deliberately not run —
so no seed-99 ranking exists and none could be computed. Nothing was changed after seeing it.

### B.2 Shared settings

| Setting | Value |
|---|---|
| Algorithm | QMIX (`q_learner`), scalar team reward |
| Episode time limit | 50 steps |
| Training budget | 2,000,000 environment steps per run |
| Device | CPU (`use_cuda=False`, explicit) |
| Detector mode | shadow |
| Model checkpointing | off |

### B.3 Non-interference guarantees

Shadow mode is a pure observer. Three independent lines of evidence:

1. **Instrumented.** `provider_call_count = 0`, `query_count = 0`, `active_subgoal_count = 0`,
   `total_shaping_reward = 0.0` in all three runs; the runner asserts this at every episode boundary.
2. **Tested.** The suite asserts that shadow mode leaves NumPy and PyTorch RNG states identical to
   `off`, that the off-vs-shadow trajectory and replay buffer are exactly equal, that the ordinary
   episode CSV is byte-identical between modes, and that shaping reward is exactly zero on every
   transition. 134 tests pass.
3. **Reproduced.** The seed-91 8×8 run reproduced two earlier, independently launched and
   prematurely terminated runs byte-for-byte over their full extent (2,260,697,088 B and
   2,337,980,416 B of detector output). A detector perturbing learning could not do this.

**Reward equality.** Stage C is unimplemented, so no shaping write-path exists: reward is inserted
raw into the replay batch (`episode_runner.py:272`). `training_reward == raw_team_reward` exactly.

**Greedy evaluation.** The detector is disabled during greedy evaluation. Verified empirically per
run: `episodes_seen` equals the count of training episodes exactly, and no shadow row originates
from a test episode (seed 99: 40,020 training episodes observed, 4,000 greedy episodes unobserved).

---

## C. Calibration results (seed 91, 8×8-2p-2f-coop)

### C.1 The grid

Fully crossed, fixed before any outcome was observed, no duplicate or missing cells:

`W_fail ∈ {5,10,15}` × `m ∈ {2,3}` × team gate ∈ {`none`,`W_team=5`,`W_team=10`,`W_team=15`}
× inbound ∈ {`off`,`on`} = **48 variants**, all observing one fixed 2,000,046-step trajectory over
40,003 episodes.

### C.2 Eligibility and the E4 erratum

| Rule | Criterion | 8×8 effect |
|---|---|---|
| **E1** | Team gate enabled (non-vacuous) | removes 12 ungated variants |
| **E2** | Inbound suppression enabled | removes 24 variants |
| **E3** | All structural invariants pass | removes 0 (48/48 pass) |
| **E4-corrected** | `semantically_invalid_final_events == 0` **and** `surviving_duplicate_final_events == 0` after coalescing | removes 0 |

**→ 18 eligible of 48.**

**PROTOCOL ERRATUM 001 (2026-08-02).** E4 was originally operationalised as
`semantically_invalid_events == 0 AND semantically_invalid_candidates == 0 AND
same_food_coalesced == 0`. The last clause is wrong: `same_food_coalesced` counts candidates
*merged into* one event (`detector.py:530-531`) — evidence that de-duplication operated, not that
duplicates survived. On 8×8 it is 0 for all 48 variants so the clause bound on nobody; on 10×10 it
is non-zero for all 48, and applying the literal rule yielded **0 eligible variants**, making the
frozen protocol inapplicable. The clause was corrected **after** the 10×10 run completed but
**before** any ranking was computed, and duplicates are now measured directly as duplicate
`event_id`s among coalesced-event rows using the existing identity.

Applying E4-corrected retrospectively to 8×8 leaves the eligible set (18), the normalisation
endpoints, the full ranking and the selection **all unchanged**.

### C.3 Selection rule and outcome

```
coverage_norm = (coverage − 21.0709) / 4.5447        # min–max over the 18 eligible
res@10_norm   = (res@10   −  8.2449) / 2.5726
d             = sqrt((1 − coverage_norm)² + res@10_norm²)     # rank ascending
tie-break 1: lower events per 1,000 steps;  tie-break 2: variant name
```

Normalisation uses the **eligible set only** — not the Pareto subset (which would discard the
interior points the rule must rank against it) and not all 48 (which would let an ineligible variant
set an endpoint).

**Pareto analysis:** 13 of the 18 eligible variants are efficient in (coverage ↑, res@10 ↓).

**Result:**

```
coverage_norm = (23.6007 − 21.0709) / 4.5447 = 0.556656   →  1 − 0.556656 = 0.443344
res@10_norm   = ( 9.3128 −  8.2449) / 2.5726 = 0.415112
d = sqrt(0.443344² + 0.415112²) = sqrt(0.196554 + 0.172318) = 0.607348
```

**`wf15_m3_t5_ibon` ranks 1 of 18** with d = 0.607348. No tie; neither tie-break was exercised.
Runner-up `wf15_m2_t10_ibon` at d = 0.613259 — **margin ≈ 0.0059**.

### C.4 Why this variant was frozen — stated exactly

It minimised the frozen distance score among eligible candidates. That is the whole reason. It is
**the deterministic balanced choice under the frozen rule**, and is explicitly **not** claimed to be
uniquely or statistically superior: the margin is ~1 % of the winning distance, the top four
candidates fall within 0.022, the eligible set spans only 4.5 pp of coverage and 2.6 pp of
resolution, and an all-48 normalisation basis would have selected `wf15_m2_t5_ibon` instead. The
rule's value is that the choice is deterministic, auditable and reproducible rather than
discretionary.

---

## D. Held-out validation (seed 99 vs seed 91, identical 8×8-2p-2f-coop condition)

Frozen detector `wf15_m3_t5_ibon`. Ratios are seed 99 ÷ seed 91; differences are seed 99 − seed 91.
Ratios are omitted where the denominator is zero.

| Metric | seed 91 (calibration) | seed 99 (held-out) | difference | ratio | unit |
|---|---|---|---|---|---|
| Episodes (denominator) | 40,003 | 40,020 | +17 | 1.0004 | episodes |
| Observed steps (denominator) | 2,000,046 | 2,000,004 | -42 | 1.0000 | steps |
| Gate-open rate | 96.1147 | 94.1009 | -2.0138 | 0.9790 | % of observed steps |
| Raw candidates | 230,268 | 251,455 | +21,187 | 1.0920 | candidate records |
| Team-gate suppressed | 1,421 | 2,353 | +932 | 1.6559 | candidate records |
| Inbound-suppressed | 19,871 | 20,652 | +781 | 1.0393 | candidate records |
| Latch-suppressed | 177,660 | 196,004 | +18,344 | 1.1033 | candidate records |
| Cooldown-suppressed | 4,273 | 4,621 | +348 | 1.0814 | candidate records |
| Re-arm-suppressed | 16,155 | 16,865 | +710 | 1.0439 | candidate records |
| Budget-suppressed | 172 | 98 | -74 | 0.5698 | candidate records |
| same_food_coalesced | 0 | 0 | +0 | — | candidate records |
| **Final events** | 10,716 | 10,862 | +146 | 1.0136 | events |
| **Events per 1,000 steps** | 5.3579 | 5.4310 | +0.0731 | 1.0136 | events / 1k steps |
| **Events per episode** | 0.2679 | 0.2714 | +0.0035 | 1.0132 | events / episode |
| **Episode coverage** | 23.6007 | 24.4178 | +0.8171 | 1.0346 | % of episodes |
| Episodes with 0 events | 30,562 | 30,248 | -314 | 0.9897 | episodes |
| Episodes with 1 event | 8,166 | 8,682 | +516 | 1.0632 | episodes |
| Episodes with >1 event | 1,275 | 1,090 | -185 | 0.8549 | episodes |
| **Episode-cap binding** | 3.1873 | 2.7236 | -0.4636 | 0.8545 | % of episodes |
| Events with 2 recipients | 10,716 | 10,862 | +146 | 1.0136 | events |
| Events with 3 recipients | 0 | 0 | +0 | — | events |
| Resolved @5 | 460 | 654 | +194 | 1.4217 | events |
| Unresolved @5 | 9,781 | 9,798 | +17 | 1.0017 | events |
| Valid denominator @5 | 10,241 | 10,452 | +211 | 1.0206 | events |
| Censored @5 | 475 | 410 | -65 | 0.8632 | events |
| **Natural resolution @5** | 4.4917 | 6.2572 | +1.7654 | 1.3930 | % of uncensored |
| Resolved @10 | 908 | 1,076 | +168 | 1.1850 | events |
| Unresolved @10 | 8,842 | 8,960 | +118 | 1.0133 | events |
| Valid denominator @10 | 9,750 | 10,036 | +286 | 1.0293 | events |
| Censored @10 | 966 | 826 | -140 | 0.8551 | events |
| **Natural resolution @10** | 9.3128 | 10.7214 | +1.4086 | 1.1513 | % of uncensored |
| Resolved: food gone | 1,892 | 2,231 | +339 | 1.1792 | events |
| Resolved: condition false | 4,724 | 4,289 | -435 | 0.9079 | events |
| Events early | 7,022 | 7,627 | +605 | 1.0862 | events |
| Events mid | 2,052 | 1,855 | -197 | 0.9040 | events |
| Events late | 1,642 | 1,380 | -262 | 0.8404 | events |
| Early share | 65.5282 | 70.2173 | +4.6891 | 1.0716 | % of that seed's events |
| Mid share | 19.1489 | 17.0779 | -2.0710 | 0.8918 | % of that seed's events |
| Late share | 15.3229 | 12.7048 | -2.6180 | 0.8291 | % of that seed's events |
### D.1 Interpretation — preserved as accepted

* **Event rate differed by only 1.4 %** (5.3579 → 5.4310 per 1,000 steps).
* **Coverage differed by 0.82 percentage points** (23.6007 % → 24.4178 %).
* **Recipient structure was identical** — every event has exactly two recipients on both seeds, as
  the two-agent cooperative condition requires; `same_food_coalesced` is 0 on both.
* **Natural resolution was higher on seed 99** — +1.7654 pp @5 (ratio 1.393) and +1.4086 pp @10
  (ratio 1.151). On the spurious-trigger proxy the held-out seed looks modestly worse.
* **This is held-out variability, not detector failure.** All structural and reconciliation checks
  pass; the differences are a result to report, not a defect to fix, and nothing was retuned.
* **One calibration seed and one held-out seed do not establish statistical robustness.** No
  significance test is possible or reported.

Note also that raw candidates rose 9.2 % while final events rose only 1.4 %: the latch absorbed
+18,344 of the +21,187 additional candidates. The lifecycle behaved as designed.

---

## E. Cross-condition sensitivity (10×10-3p-3f, seed 91) — reported separately

**This is not primary validation.** It is secondary sensitivity evidence and is deliberately
segregated from §D.

| Finding | Value |
|---|---|
| Frozen variant rank among eligible | **18 of 18 (last)**, d = 1.076289 |
| Pareto status | **Not Pareto-efficient** (frontier holds 4 of 18) |
| Rank-1 variant | `wf15_m2_t10_ibon`, d = 0.540482 |
| Gate-open rate of the frozen variant | **97.68 %** of observed steps |
| Eligible gate-open range | 94.08 % – 97.68 % |
| Eligible res@10 range | **0.0900 percentage points** (1.3110 % – 1.4010 %) — 28.6× narrower than 8×8's 2.5726 pp |
| Solo share of successful loads | **≈ 95.5 %** (10,133 of 10,611) |
| Eligible set size | 18 of 48 (computed independently; equals the 8×8 count coincidentally) |

**Four environment properties changed jointly**: grid size (8×8→10×10), agent count (2→3), food
count (2→3) and enforced cooperation (`-coop-` → absent). The environment key is the only configured
field that differs, but it is not a single experimental variable.

Two consequences: because the res@10 axis spans only 0.09 pp, min–max normalisation stretches a
tiny absolute difference across the full [0,1] axis, so rank 18 reflects a 0.09 pp gap on that axis;
and because the team gate — the strongest discriminating factor on 8×8 — was open ~97.7 % of steps,
it barely discriminated under this condition. Both are recorded as findings **under the frozen
protocol**, not as reasons to retune.

**No robust external-generalisation claim is made**, and no difference is attributed to any single
property. **The frozen configuration was not changed.**

---

## F. Integrity evidence

| Check | 8×8 seed 91 | 10×10 seed 91 | 8×8 seed 99 |
|---|---|---|---|
| Exit code / status | 0 / COMPLETED | 0 / COMPLETED | 0 / COMPLETED |
| Final `t_env` (budget 2,000,000) | 2,000,046 | 2,000,007 | 2,000,004 |
| Malformed or truncated records | 0 | 0 | 0 |
| NUL corruption | 0 bytes | 0 bytes | 0 bytes |
| Semantically invalid final events | 0 (all 48) | 0 (all 48) | 0 |
| Surviving duplicate `event_id`s | **0** (525,879 rows, 525,879 distinct) | **0** (495,011 / 495,011) | **0** (10,862 / 10,862) |
| Raw-CSV ↔ summary reconciliation | exact, 48/48 | exact, 48/48 (`reconciled: true`) | exact |
| Recipient reconciliation | exact | exact | exact |
| Lifecycle invariants | 13/13 × 48 pass | 15/15 × 48 pass | 15/15 pass |
| Provider calls / queries / subgoals | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| Total shaping reward | 0.0 | 0.0 | 0.0 |
| `training_reward == raw_team_reward` | yes | yes | yes |
| Detector excluded from greedy evaluation | yes | yes | yes (40,020 train observed; 4,000 greedy unobserved) |
| Full test suite | 134 passed, 0 failed | | |

---

## G. Claims and limitations

Full text: `analysis/out/stage_b_paper_metrics/stage_b_claims_and_limitations.md`. In summary:

* **No ground-truth struggle labels exist**, so **precision, recall, F1 and AUROC cannot be
  reported** and none is reported.
* **Natural resolution is a proxy** for potentially unnecessary triggering, **not** a ground-truth
  false-positive rate; the direction of preference is a stated modelling choice.
* **Firing consistency is not detector accuracy** — consistency across seeds shows the
  implementation is stable, not that the detector fires on the right events.
* **Seed 99 supports implementation consistency and broadly stable firing behaviour, not
  statistical robustness** — one held-out seed, no significance test.
* **10×10 is sensitivity evidence, not external validation** — four properties changed jointly,
  95.5 % solo loads, one seed.
* **The selection rule is transparent development-set calibration, not an unbiased performance
  estimate.**
* **Stage B validates the trigger mechanism and the experimental plumbing. Stage C must evaluate
  whether selective guidance improves MARL performance**, via unmodified QMIX, a call-matched
  random-trigger control, an always-on comparator and a deterministic no-LLM ablation.

---

## H. Source artifacts and verification

| Output | Path |
|---|---|
| Metric data dictionary (62 metrics) | `analysis/out/stage_b_paper_metrics/detector_metric_definitions.csv` |
| Field coverage register (103 fields) | `analysis/out/stage_b_paper_metrics/detector_field_coverage_register.csv` |
| Metrics by condition and seed (65 cols × 3) | `analysis/out/stage_b_paper_metrics/detector_metrics_by_condition_and_seed.csv` |
| Frozen configuration | `analysis/out/stage_b_paper_metrics/frozen_detector_configuration.json` |
| Claims and limitations | `analysis/out/stage_b_paper_metrics/stage_b_claims_and_limitations.md` |
| Figure source | `analysis/out/stage_b_paper_metrics/make_stage_b_figures.py` |
| Figures (PDF + SVG + PNG) | `analysis/out/stage_b_paper_metrics/figures/` |

Upstream verified sources: the three per-run summary JSONs, the two calibration report JSONs, the
10×10 and seed-99 validation results JSONs, the Stage B selection record, the 10×10 validation
report and the seed-99 held-out validation report. SHA-256 hashes of the primary sources are
recorded in `frozen_detector_configuration.json`.

**Verification performed.** Every value in §C, §D, §E and §F was cross-checked against the existing
Stage B reports and machine-readable outputs: **18 of 18 checks agree**. Three initially flagged as
mismatches were traced to comparison precision — the report JSONs store 6-decimal rounded values
while the paper CSV stores full unrounded precision (differences ~1×10⁻⁷). The paper CSV
deliberately retains the exact underlying values. **No unresolved inconsistency exists.**

**Field coverage.** All 48 summary fields are reported in the dictionary. Of the 55 raw-log columns,
6 appear directly and **49 are per-row fields that enter the paper through an aggregate** — each is
listed in the coverage register with its role. **No available detector diagnostic is silently
omitted.**

---

## I. Recommended dissertation placement

### Main Results section

| Item | Why |
|---|---|
| **Figure 1** — calibration trade-off (`fig1_...`) | Shows the selection problem, the eligible set, the frontier and the frozen choice in one view. |
| **Figure 2** — seed-91 vs seed-99 small multiples (`fig2_...`) | The headline held-out result. |
| **Table D** (§D) — reduced to ~10 rows | Keep: events/1k, events per episode, coverage, gate-open, cap binding, res@5, res@10, recipients, final events, denominators. |
| **§C.3–C.4** — selection outcome and the exact reason | Two short paragraphs. |

### Appendix

| Item | Why |
|---|---|
| **Figure 3** — suppression funnel | Valuable mechanism evidence, too detailed for main text. |
| **Figure 4** — episode-phase timing | Supporting diagnostic. |
| **Figure 5** — 10×10 sensitivity | Must stay clearly separated from primary validation; an appendix placement enforces that. |
| Full Table D (all 39 rows) | Complete counts and denominators. |
| Metric data dictionary and field coverage register | Reproducibility. |
| §F integrity table | Verification detail. |
| PROTOCOL ERRATUM 001 in full | Transparency about the post-run correction. |

### Methods

§A in full (definition, inputs/outputs, suppression components, processing order, frozen
configuration, pseudocode) and §B in full (seed roles, shared settings, non-interference
guarantees). §C.1–C.2 (grid and eligibility) belong in Methods rather than Results, since they
describe the protocol rather than its outcome.

### Limitations

§G in full, expanded from `stage_b_claims_and_limitations.md`. The five load-bearing statements are:
no ground-truth labels so no precision/recall/F1/AUROC; natural resolution is a proxy; firing
consistency ≠ accuracy; one held-out seed ≠ statistical robustness; 10×10 is sensitivity, not
external validation. The erratum and the "development-set calibration, not an unbiased performance
estimate" framing also belong here.
