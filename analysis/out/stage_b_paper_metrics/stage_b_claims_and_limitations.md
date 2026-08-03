# SELMA Stage B — claims and limitations

Transferable directly into a dissertation Limitations section. Every claim below is
either verified against a source artifact or explicitly marked as *not* claimed.

## What Stage B establishes

1. **The trigger mechanism is implemented as specified and is internally consistent.**
   The suppression ledger balances exactly for every variant in every run; 15 lifecycle
   invariants pass; independently re-deriving counters by streaming the raw logs reproduces the
   in-process aggregates exactly for all 48 variants on both 8×8 and 10×10.
2. **The experimental plumbing is sound.** Shadow mode is trajectory- and reward-equivalent to
   `off` mode (RNG states, replay contents and the ordinary episode CSV are identical); the
   detector never observes greedy-evaluation episodes; provider calls, queries, active subgoals
   and shaping reward are exactly zero in every run.
3. **Detector firing behaviour is broadly stable across a held-out seed** on the development
   condition: event rate differs by 1.4 %, coverage by 0.82 percentage points, and recipient
   structure is identical.
4. **The selected configuration was chosen by a deterministic, auditable, pre-stated rule**, and
   was not altered by either the 10×10 sensitivity result or the seed-99 held-out result.

## What Stage B does NOT establish

### No ground-truth struggle labels exist

The environment provides no label for "this agent genuinely needed help at this step". There is no
annotated set of true coordination stalls against which detector output can be scored.

**Consequently precision, recall, F1 and AUROC cannot be reported, and none is reported anywhere
in this work.** Any such number would require a ground truth that does not exist. This is a
property of the task, not an omission that further analysis could repair without new labelled data.

### Natural resolution is a proxy, not a false-positive rate

`natural_resolution_rate_5` / `_10` measure the share of uncensored events whose stall cleared
unaided within the horizon. The reasoning is that a stall clearing on its own arguably did not
require assistance.

This is a **plausible correlate of unnecessary triggering, not a measured false-positive rate**. A
stall that clears in nine steps may still have been costly; a stall that persists may have been
unresolvable for reasons unrelated to coordination. The direction of preference (lower is better)
is a **stated modelling choice**, not an established criterion. Every selection result that depends
on this axis inherits the limitation.

### Firing consistency is not detector accuracy

That the detector fires at a similar rate on a held-out seed shows the **implementation is
consistent and the trigger is not seed-fragile**. It does not show the detector fires on the
*right* events. Consistency and correctness are different properties, and only the first is
evidenced here.

### One held-out seed is not statistical robustness

Seed 99 is a **single** held-out run against a **single** calibration run. No significance test is
possible or reported, no confidence interval is computed, and no p-value appears in any figure or
table. Between-seed differences — including the +1.77 pp rise in natural resolution @5 — are
reported as **held-out variability**, which is a result, not a failure and not a licence to retune.

### 10×10 is sensitivity evidence, not external validation

`Foraging-10x10-3p-3f-v3` differs from the 8×8 development condition in **four properties that
change jointly**: grid size, agent count (2→3), food count (2→3) and enforced cooperation
(`-coop-` → absent). The environment key is the only configured field that changes, but it is not a
single experimental variable.

Additionally, 95.5 % of successful loads on 10×10 are solo, so the coalition-required failed-LOAD
population the detector triggers on is drawn from a materially different distribution — **firing
rates are not like-for-like**. The frozen configuration ranked 18/18 among eligible variants there
and was not Pareto-efficient, against an eligible res@10 range spanning only 0.0900 percentage
points, with its team gate open 97.68 % of steps.

**No claim of robust external generalisation is made.** No difference is attributed to grid size,
agent count, food count or cooperation status individually. Isolating any single property would
require additional conditions holding the others fixed.

### The selection rule is development-set calibration

The factorial grid was fixed in advance, but the scalarisation — axes, min–max normalisation,
Euclidean form and tie-break order — was specified with the 48-variant table already visible. **The
rule constitutes transparent development-set calibration rather than an unbiased performance
estimate.** Its purpose is determinism and auditability, not statistical validity.

The margin over the runner-up is ≈ 0.0059 (about 1 % of the winning distance, with the top four
candidates within 0.022), and the outcome depends on the normalisation domain: an all-48 basis would
select a different configuration. `wf15_m3_t5_ibon` is therefore **the deterministic balanced choice
under the frozen rule**, not a uniquely or statistically superior configuration.

### A documented protocol erratum applies

Eligibility rule E4 was corrected after the 10×10 run completed but **before** any ranking was
computed (PROTOCOL ERRATUM 001, 2026-08-02). Its original literal form required
`same_food_coalesced == 0`, which measures candidates *merged into* one event — the de-duplication
mechanism working — rather than surviving duplicates. Applied literally it excluded all 48 10×10
variants. The corrected rule tests `semantically_invalid_final_events == 0` and
`surviving_duplicate_final_events == 0` (duplicate `event_id`s among coalesced-event rows, existing
identity, unchanged).

The 10×10 analysis must therefore be described as a **corrected-protocol analysis following a
documented post-run, pre-ranking operationalisation erratum**, not as an untouched confirmatory
analysis. The 8×8 selection is unaffected: `same_food_coalesced` is 0 for all 48 8×8 variants, and
re-applying E4-corrected retrospectively leaves the eligible set (18), the normalisation endpoints,
the ranking and the selection all unchanged.

### The validation source is not a clean checkout

Seed-99 ran on **base commit `92ece20` plus the documented selected-variant plumbing patch**
(`selma_variant_id`, two files: `src/config/default.yaml` and `src/runners/episode_runner.py`).
Detector logic in `src/selma/` is unchanged. The patch is covered by 22 dedicated tests including a
behavioural-equivalence test proving the full 48-variant bank filtered to the frozen variant equals
the selected-only bank exactly.

## What Stage C must establish

Stage B metrics validate the **trigger mechanism and the experimental plumbing**. They say nothing
about whether intervening at these moments helps.

**Stage C must evaluate whether selective LLM guidance improves MARL performance**, against
unmodified QMIX, a call-matched random-trigger control (isolating the detector's contribution), an
always-on guidance comparator (isolating selectivity) and a deterministic no-LLM ablation sharing
the same detector and shaping (isolating the language model's contribution). Those comparisons —
not the Stage B firing statistics — are what could support a claim that the method works.
