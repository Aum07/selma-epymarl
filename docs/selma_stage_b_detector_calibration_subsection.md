# Struggle Detector Calibration and Configuration Selection

*Draft dissertation subsection. Reports the seed-91 8x8 Stage B shadow calibration
(`analysis/out/seed91_retry_20260802T112458Z/8x8/`).*

*Status at commit: this section was drafted **before** the 10x10 sensitivity run and the seed-99
held-out validation. Its calibration content stands as written and is verified against the preserved
summaries, but for the current position see `docs/project/selma_stage_b_selection_record.md`
(configuration **frozen**), `docs/project/selma_stage_b_10x10_validation_report.md` (10x10
**completed**) and `docs/project/selma_stage_b_seed99_heldout_validation_report.md`. Its unique
contribution to the dissertation is §7, the rationale for the event-based detector, which is not
reproduced in `docs/project/selma_stage_b_paper_metrics_report.md`.*

---

## 1. Scope and status of the calibration environment

The `lbforaging:Foraging-8x8-2p-2f-coop-v3` condition is the **development and calibration
environment** for the struggle detector. It is where the detector's parameterisation is explored and
where the numerical selection rule is exercised. It is explicitly *not* a validation environment:
any configuration chosen here is chosen on data the choice was fitted to, and carries the
corresponding risk of overfitting to a single grid size, a single team composition and a single
seed. The `10x10` condition was reserved for validation; it has since been run and is reported
separately as cross-condition sensitivity evidence in
`docs/project/selma_stage_b_10x10_validation_report.md`.

Because only one environment has been evaluated, the cross-environment consistency criterion
defined for Stage B — a variant fires in both environments, has no semantically invalid event, has a
non-vacuous team gate in both, and its firing rate differs by at most 2× with a rank difference of
at most 12 — **cannot be evaluated**. It is reported as *N/A*. An earlier generated version of the
calibration report contained a consistency table produced by supplying a single environment, which
compared 8x8 against itself and therefore returned a rate ratio of exactly 1.00, a rank difference
of exactly 0, and a `consistent` verdict for all 48 variants. Those were arithmetic identities
rather than measurements, and have been removed from both the Markdown report and the machine
readable `calibration_report.json`. No claim of cross-environment agreement is made anywhere in this
work.

## 2. Shared-trajectory design and the non-interference guarantee

All 48 detector variants were evaluated against **one fixed trajectory**: a single QMIX run at
seed 91 for 2 000 046 environment steps across 40 003 episodes. The variants did not each train a
policy; they observed the same stream of transitions and independently recorded when each *would*
have fired.

This design is what makes the comparison a controlled one. Differences in the table below are
attributable solely to detector parameterisation, because the underlying behaviour being observed is
byte-for-byte identical across variants. It also removes seed and training-noise variance from the
comparison entirely — at the cost that the calibration reflects the behaviour of one particular
policy trajectory, which is a limitation carried forward to §7.

The detector ran in **shadow mode** throughout. Shadow mode is a pure observer: it cannot influence
action selection, reward, or learning. Three independent lines of evidence support this:

1. **Instrumented zero-guarantees.** The run summary records `provider_call_count = 0`,
   `query_count = 0`, `active_subgoal_count = 0` and `total_shaping_reward = 0.0`.
2. **Equivalence tests.** The test suite (112 tests at the time of drafting; 134 at commit, all passing) asserts that shadow mode leaves
   NumPy and PyTorch RNG states identical to `off` mode, that the off-vs-shadow trajectory and
   replay buffer are exactly equal, that the ordinary episode CSV is byte-identical between modes,
   and that shaping reward is exactly zero on every transition.
3. **Empirical reproduction.** The completed run reproduced two earlier, independently launched and
   prematurely terminated runs of the same configuration byte-for-byte over their entire lengths
   (2 260 697 088 B and 2 337 980 416 B of raw detector output respectively). A detector that
   perturbed learning could not reproduce a prior trajectory exactly.

## 3. The factorial grid, fixed in advance

The grid was specified in the implementation contract **before** any calibration outcome was
observed, and was not altered in response to results. It is fully crossed with no duplicate or
missing cells (verified programmatically from `calibration_report.json`):

| Factor | Levels | Count |
|---|---|---|
| `W_fail` — failure-history window (steps) | 5, 10, 15 | 3 |
| `m` — coalition-required failed LOADs within `W_fail` | 2, 3 | 2 |
| Team-stall gate | `none`, `W_team=5`, `W_team=10`, `W_team=15` | 4 |
| Inbound-helper suppression | `off`, `on` | 2 |
| **Total** | 3 × 2 × 4 × 2 | **48** |

The following were held **fixed across all 48 variants**:

| Parameter | Value |
|---|---|
| `w_inbound` (inbound-helper window) | 3 |
| `cooldown_steps` | 5 |
| `rearm_steps` | 3 |
| `max_events_per_episode` | 2 |
| Seed | 91 |
| Condition / grid / agents | `Foraging-8x8-2p-2f-coop-v3`, 8×8, 2 |
| Observed steps (all variants) | 2 000 046 |
| Episodes | 40 003 |

Two distinctions matter for interpretation. First, `w_inbound = 3` is the inbound-helper *window*
and is fixed; the `inbound` factor toggles only whether that suppression is *applied*. Second,
`max_events_per_episode = 2` is a run-level cap held constant, whereas the `m` factor is the
per-variant failed-LOAD threshold under test; they are separate quantities despite both taking small
integer values.

## 4. Verification of the calibration data

Before any variant was compared, the calibration output was verified independently of the
in-process counters. The 4.25 GB raw detector log (11 857 607 rows: 11 331 728 raw-candidate rows
and 525 879 coalesced-event rows) was streamed and the per-variant final-event counts, recipient
count distributions and semantically-invalid counts were re-derived from the raw rows. They agree
**exactly** with the aggregated summary for all 48 variants (`reconciled: true`, zero
reconciliation problems).

Thirteen structural invariants were then checked for each of the 48 variants and all pass. The
central one is the suppression ledger, which balances exactly:

```
raw_candidates
  − suppressed_by_team_gate                → candidates_after_team_gate
  − semantically_invalid − suppressed_by_inbound  → raw_candidates_after_inbound
  − same_food_coalesced
  − (latch + cooldown + rearm + budget)     = final_events
```

Also verified: per-episode histograms sum to 40 003; phase and recipient distributions sum to
`final_events`; resolution denominators balance against censoring; and the per-episode event cap is
never exceeded. Across all 48 variants, `semantically_invalid_events`,
`semantically_invalid_candidates` and `same_food_coalesced` are **all zero**, and every recorded
event has exactly two recipients — both agents — as a two-agent cooperative condition requires.

## 5. Calibration results

Across the full grid, firing rates span only 1.42× (4.688–6.670 events per 1 000 steps; 9 377–13 340
events; 0.234–0.333 per episode) and episode coverage spans 21.07 %–29.66 %. Event timing is
early-weighted in every variant (roughly 70/17/13 early/mid/late at the highest firing rate,
61/22/17 at the lowest); no variant is late-biased.

The suppression profile is stable across the grid: the per-food latch accounts for 89.0–95.5 % of
all suppression, re-arm for 3.4–8.6 %, cooldown for 0.8–2.2 %, and the per-episode budget for only
0.03–0.12 %. The event cap binds in all 48 variants but weakly, affecting 2.09–3.77 % of episodes.

Marginal factor effects (each averaged over the other three factors):

| Factor | Levels | Events / 1k | Coverage % | Natural resolution @10 % |
|---|---|---|---|---|
| `W_fail` | 5 / 10 / 15 | 5.408 / 5.481 / 5.544 | 24.07 / 24.51 / 24.69 | 10.63 / 10.68 / 10.68 |
| `m` | 2 / 3 | 5.741 / 5.215 | 25.59 / 23.26 | 11.82 / 9.50 |
| Team gate | none / 5 / 10 / 15 | 5.870 / 5.737 / 5.361 / 4.942 | 25.97 / 25.45 / 23.93 / 22.34 | 12.36 / 11.76 / 9.87 / 8.65 |
| Inbound | off / on | 5.623 / 5.332 | 25.22 / 23.63 | 11.72 / 9.60 |

The team-stall gate is the strongest factor and `W_fail` by far the weakest — over a 3× change in
window length the firing rate moves by 2.5 % and the natural-resolution rate by 0.05 percentage
points. A notable non-monotonicity: raising `m` from 2 to 3 *reduces* the firing rate, because the
longer per-food latch window displaces more candidates than the additional threshold headroom
admits.

**Natural resolution @10** is the fraction of triggers for which the observed struggle resolved
without intervention within ten steps. It is used here as the closest available proxy for a spurious
trigger, on the reasoning that a stall which clears unaided did not require assistance. It is a
proxy and not a ground-truth false-positive rate: the environment provides no label for "this agent
genuinely needed help", and a stall that resolves in nine steps may still have been costly. This
interpretive limitation is inherited by any selection rule built on the quantity, including the one
in §6.

## 6. Configuration selection

### 6.1 Eligibility

Candidates were restricted by four hard rules: (i) team-stall gate enabled, (ii) inbound-helper
suppression enabled, (iii) all structural invariants pass, and (iv) zero invalid and zero duplicate
events. Rule (i) removes the 12 ungated variants, whose team gate is vacuous by construction — with
no gate there is no team-stall condition to satisfy and `gate_open_percentage` is identically
100 %. Rule (ii) removes the 24 variants that do not suppress a candidate when a helper is already
inbound, which would admit triggers for stalls already being resolved. Rules (iii) and (iv) are
satisfied by all 48 in this run, but are retained as hard gates rather than filters: they are
re-evaluated for every run, and a configuration failing either is excluded regardless of its scores.

**Eighteen variants are eligible**, of which thirteen are Pareto-efficient in
(coverage ↑, natural resolution @10 ↓).

### 6.2 Selection rule

The rule normalises coverage and natural-resolution @10 by min–max **over the 18 eligible
candidates**, computes the Euclidean distance from the ideal point (coverage 1, resolution 0),

  d = √((1 − coverage_norm)² + resolution_norm²)

ranks ascending by d, and breaks ties by lower events per 1 000 steps and then by variant name.
The locked normalisation basis is coverage ∈ [21.0709 %, 25.6156 %] (range 4.5447) and
resolution @10 ∈ [8.2449 %, 10.8175 %] (range 2.5726). The rule selects **`wf15_m3_t5_ibon`**
(`W_fail = 15`, `m = 3`, team gate `W_team=5`, inbound on) with

```
coverage_norm = (23.6007 − 21.0709) / 4.5447 = 0.556656
res@10_norm   = ( 9.3128 −  8.2449) / 2.5726 = 0.415112
d = √(0.443344² + 0.415112²) = 0.607348
```

No tie occurred, so neither tie-break was exercised. The remaining detector parameters are those
held fixed across the grid: `w_inbound = 3`, `cooldown_steps = 5`, `rearm_steps = 3`,
`max_events_per_episode = 2`. The full record, including the frozen validation protocol, is
`docs/project/selma_stage_b_selection_record.md`.

### 6.3 Two caveats stated plainly

**The rule was finalised after the calibration results were inspected.** The factorial grid was
fixed in advance, but the scalarisation — the choice of axes, the min–max normalisation, the
Euclidean form, and the tie-break order — was specified with the 48-variant table already visible.
The rule constitutes transparent development-set calibration rather than an unbiased performance
estimate. The 10x10 data were not used to select or retune the detector. The selection rule and
configuration were frozen before analysing the completed validation run. It is adopted for a
specific and defensible purpose: to make the selection *deterministic, auditable and reproducible*
rather than a matter of judgement. The rule, its normalisation domain, the eligibility criteria and
the selected configuration are **locked as of the selection record, before the 10x10 validation
run**, and will not be altered by that run's outcome.

**The distance advantage is small and the choice is a balanced one, not a superior one.** The margin
over the runner-up `wf15_m2_t10_ibon` is approximately **0.0059** (d = 0.607348 vs 0.613259) — about
1 % of the winning distance, with the top four candidates falling within 0.022 of one another. The
eligible candidates span only 4.5 percentage points of coverage and 2.6 points of resolution, so
they occupy a flat region in which several are close to indistinguishable on the available evidence.
`wf15_m3_t5_ibon` is therefore reported as **the deterministic balanced choice under the frozen
rule**, not as a uniquely or statistically superior configuration; no significance test is implied
and no confidence claim attaches to the ranking.

**The selection depends on the normalisation domain, which is therefore stated explicitly.**
Normalisation is computed over the 18 eligible configurations. The 13 Pareto-efficient
configurations are reported descriptively and do not define the range, since restricting
normalisation to the frontier would discard the interior configurations the rule must rank against
it. The full set of 48 is not used, because it contains configurations removed by the hard
eligibility rules, and an ineligible configuration must not set an endpoint that determines the
scores of eligible ones. This matters materially: under an all-48 basis the rule would select
`wf15_m2_t5_ibon` instead. The eligible-set basis is the locked one.

### 6.4 Pre-registration of the validation protocol

The 10x10 run is a validation of the locked configuration, not a further round of selection. The
protocol was frozen before any 10x10 result was viewed: the same 48-variant factorial grid, the same
hard eligibility rules, and the same normalisation *definition* — min–max over the eligible set
alone, neither the Pareto subset nor all 48 — with endpoints recomputed from 10x10 data. The
validation asks two questions: does `wf15_m3_t5_ibon` remain Pareto-efficient in the 10x10 eligible
set, and what is its 10x10 rank under the unchanged rule? Coverage, natural-resolution@10, event
rate, cap binding, timing distribution and all structural invariants are reported for every variant.

The selected configuration **will not be replaced or retuned from the 10x10 results**. Any
disagreement will be reported as **cross-environment sensitivity** — evidence about the detector's
transferability — rather than absorbed by adjusting parameters until the environments agree. Any
change made after seeing 10x10 data will be reported explicitly as post-hoc, with pre-change and
post-change results both shown.

The validation is a **cross-condition robustness validation**, not a grid-size ablation. The
validation condition `Foraging-10x10-3p-3f-v3` differs from the `Foraging-8x8-2p-2f-coop-v3`
development condition in **four environmental properties that change jointly**: grid size
(8×8 → 10×10), number of agents (2 → 3), number of food items (2 → 3), and enforced-cooperation
status (`-coop-` enforced → absent). The environment key is the only *configured field* that
differs, but it is not a single experimental variable: changing it alters all four properties
simultaneously, and they cannot be separated from one another in the result.

The question the validation answers is therefore whether the detector's calibrated behaviour is
robust when transferred to a materially different cooperative foraging condition — not what any one
environmental property contributes. Two specific consequences follow. With three agents the
recipient-count distribution can no longer be identically 2, so the 8x8 invariant "every event has
exactly two recipients" is a property of the two-agent condition and does not transfer; the
corresponding 10x10 check is that recipient counts sum to `final_events` and lie in the range 2–3.
Without the `-coop-` constraint some food is loadable by a single agent, so the coalition-required
failed-LOAD population is drawn from a different distribution and firing rates are not directly
comparable like-for-like. Any observed disagreement is joint evidence across all four properties and
will not be attributed to grid size, agent count, food count or cooperation status individually.
Attributing an effect to any single property would require additional conditions holding the others
fixed — for example `Foraging-10x10-2p-2f-coop-v3` to isolate grid size — which are separate
experiments and are not performed here.

## 7. Why an event-based LOAD-failure detector replaced the proposed reward-variance / trajectory-entropy detector

The initial proposal characterised struggle through statistical signals over recent experience:
variance in reward and entropy over trajectories or action distributions. The implemented detector
instead defines a coordination stall through a concrete environment event: *at least `m`
coalition-required failed LOAD attempts on the same still-existing food within the last `W_fail`
steps*. This substitution was made in the implementation contract before Stage B, which excludes
contribution deficit, position entropy, action entropy, Q-gap and reward variance from detector v1,
permitting them only as logged diagnostics that do not gate a query. The reasons are as follows.

**Semantic groundedness.** A failed cooperative LOAD is exactly the failure mode SELMA exists to
address: agents attempting to collect food whose level exceeds their individual capacity, requiring
simultaneous cooperation. High reward variance or high action entropy are *correlates* of
difficulty that may arise from ordinary exploration, an unlucky food layout, or a policy in
transition. The event-based condition identifies the specific situation to be helped, which in turn
makes the downstream subgoal well-posed: the target food, the recipient agents and the valid
loading cells are all determined by the triggering event.

**Attribution.** Variance and entropy are properties of a window of behaviour, not of an
identifiable moment. They do not answer *which* food, *which* agents, or *when* — all of which the
provider contract requires as inputs fixed by code rather than by the provider. A stall event
supplies all three by construction.

**Non-stationarity of the statistical signals.** Reward variance and policy entropy vary
systematically over training as ε anneals and the policy sharpens. A fixed threshold on either
would fire at a rate that drifts with training progress, confounding detector behaviour with
learning stage. The failed-LOAD condition is defined on environment events whose meaning does not
change as the policy improves — though its *rate* does, which is why the calibration reports event
timing by episode phase.

**Auditability.** A discrete event condition can be re-derived exactly from a log and checked
against the in-process counters. This is what made the reconciliation in §4 possible: 11 857 607
raw rows were replayed and reproduced the counters for all 48 variants exactly. An estimator built
on running variance or entropy would depend on window state, floating-point accumulation order and
update scheduling, and could not be verified to this standard.

**Falsifiability of the mechanism.** Because each event names a food and a set of agents, an
individual trigger can be inspected and judged. The zero semantically-invalid events reported in §4
is a meaningful check precisely because validity is defined per event — whether the identified
coalition is feasible for the identified food. There is no analogous per-instance validity check for
a variance threshold.

The excluded signals are not claimed to be uninformative. The contract retains them as candidate
diagnostics for later logging. The claim is narrower: they are unsuitable as the *gating* condition
for a query in detector v1, because they lack the attribution and auditability the intervention
mechanism requires.

---

## Artefacts

| Item | Path |
|---|---|
| Calibration report (Markdown) | `docs/selma_stage_b_seed91_calibration.md` |
| Calibration report (JSON) | **External, not committed** — `calibration_report.json` in the run directory. Its committed Markdown counterpart is `docs/selma_stage_b_seed91_calibration.md`; all figures used here derive from the committed summary JSON. |
| Per-variant summary | `analysis/out/seed91_retry_20260802T112458Z/8x8/summary_8x8_seed91_2M_retry.json` |
| Raw detector log (4.25 GB) | **External, not committed** — `seed91_retry_20260802T112458Z/8x8/shadow_8x8_seed91_2M_retry.csv` in the external raw store. Retained outside the repository; all counts derived from it are preserved in the committed summary JSON and calibration report. |
| Episode CSV | **External, not committed** — `episodes_8x8_seed91_2M_retry.csv` in the run directory. Per-episode run output; no reported metric in this section depends on it. |
| Run log / resource samples | **External, not committed** — `run.log` and `rss_samples.csv` in the run directory. Operational logs; not required to reproduce any reported metric. |
| Recovery and verification record | **External, not committed** — `selma_seed91_recovery_and_rerun_plan.md`, an operational planning document. No reported metric depends on it. |

Repository state at calibration: HEAD `92ece20`, clean working tree, no tracked modification.
