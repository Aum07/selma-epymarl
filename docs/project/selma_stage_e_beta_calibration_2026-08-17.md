# SELMA Stage E — shaping-beta calibration pilot (seed 91)

**Date:** 2026-08-17
**Branch:** `stage-e-beta-calibration`
**Starting commit:** `232e11f1081c2cc66fdc272c84b35b5b014a6205` (*selma: measure Stage D beta-zero cache behaviour*)

---

# PART 1 — PREREGISTRATION

> **Written and committed to before any condition was executed.** Everything in
> Part 1 was fixed in advance. It is not edited after results are seen; results
> live in Part 2, which is appended below and clearly separated. If any rule here
> turns out to be awkward once results exist, the rule stands and the awkwardness
> is reported.

## 1.1 What this pilot is, and is not

This is a **one-development-seed hyperparameter pilot** to choose a provisional
PBRS shaping scale `beta` for later work. It is:

* four conditions differing **only** in `beta`;
* one seed (91), one environment, one algorithm;
* 500 000 environment steps per condition;
* evaluated on greedy-evaluation curves from the development seed only.

It is **not**:

* a performance result;
* a claim of statistical significance;
* a final-seed evaluation — final/test seeds are not touched, inspected or run;
* a search over anything except the four preregistered `beta` values.

A single seed cannot establish significance. Training-episode returns within a
run are **not independent samples**: they come from one continuously-updating
policy, so any interval computed from them describes the variability of that one
trajectory, not of the method.

## 1.2 The four conditions (frozen)

Run **sequentially, in this exact order**:

| # | `beta` |
|---|---|
| 1 | **0.00** |
| 2 | **0.05** |
| 3 | **0.10** |
| 4 | **0.25** |

No other `beta` may be added, substituted or re-run. A completed condition is
never re-run.

## 1.3 Configuration (identical except `beta`)

| | |
|---|---|
| Algorithm | QMIX |
| Environment | `lbforaging:Foraging-8x8-2p-2f-coop-v3` |
| `time_limit` | 50 |
| `t_max` | 500 000 |
| Seed | **91** (the development seed; the only seed used) |
| `selma_mode` | `active` |
| Detector variant | `wf15_m3_t5_ibon` (frozen) |
| Provider | `openai` |
| Model | `gpt-5.4-nano-2026-03-17` (exact dated snapshot) |
| Prompt | `selma_guidance_prompt_v3`, unchanged |
| Response schema | `selma_guidance_response_schema_v2`, unchanged |
| Cache key / schema | `selma_cache_key_v2` / `selma_cache_schema_v2`, unchanged |
| Strict structured output | enabled |
| Reasoning effort | `none` |
| `store` / tools / streaming | `false` / none / disabled |
| Max output tokens | 512 |
| Timeout / retries | 30 s / 1 |
| TTL | 15 (frozen) |
| `common_reward` | True |
| Cache mode | `read_write`, **new empty isolated database per condition** |
| Live-attempt cap | **1 000 per condition**, retries included |
| Cost cap | **$0.25 per condition** |
| Global caps | **4 000 attempts, $1.00** |
| CUDA | False |
| `test_interval` | 10 000 (50 greedy evaluations, 20 episodes each) |
| Greedy evaluation | no provider, cache, guidance or shaping activity |

Every condition starts from scratch: empty cache, isolated output directory, no
checkpoint or learner-state reuse, no cache sharing between conditions.

## 1.4 Pricing (already frozen, 2026-08-17)

Source: `https://developers.openai.com/api/docs/models/gpt-5.4-nano`

| GPT-5.4 nano | USD per 1M tokens |
|---|---|
| input | 0.20 |
| cached input | 0.02 |
| output | 1.25 |

Enforced before every live attempt by the existing `CostBudget` mechanism.
Provider-side cached input is **not** the application's SQLite response cache.

## 1.5 Behaviour at a cap

If a condition reaches its call or cost cap: training continues; cache hits may
still be served; cache misses produce `budget_exhausted` with no guidance and
`F_t = 0`. The cap is **not** raised. The exposure difference is recorded.

**If any condition exhausts its cap, `beta` must not be frozen from that
comparison** — unequal guidance exposure makes the conditions non-comparable.
All safe runs still complete, and the report states that a revised pilot is
required.

## 1.6 Safety exclusions (applied before any performance comparison)

A positive `beta` is **excluded** if any of the following occurs in its condition:

1. NaN or Inf in rewards, learner statistics, parameters or logs.
2. Reward-reconciliation failure (`training_reward != raw + shaping`).
3. PBRS telescoping-invariant failure (`phi_after_effective(t) != phi_before(t+1)`).
4. Invalid or physically unreal guidance activated (duplicate agents or cells,
   off-grid, non-adjacent, unoffered, target-food cell, or an unoffered plan id).
5. Provider / cache / accounting invariant failure.
6. **Max |shaping contribution| > 50 % of the smallest positive environmental
   reward observed in the same condition.**
7. **P95 of non-zero |shaping contribution| > 25 % of the median positive
   environmental reward** in the same condition.
8. Clear learner instability relative to `beta = 0` — exploding loss, Q-values or
   gradient norms.
9. API/cost-cap exhaustion producing materially unequal guidance exposure.

Criteria 6 and 7 are the substantive ones: PBRS is invariant-preserving in
theory, but a shaping term comparable to the environmental reward makes the
practical optimisation problem a different one.

## 1.7 Metrics

**Primary calibration metric**

> Area under the **greedy-evaluation team-return curve** from `t_env` 100 000
> through 500 000, computed by the trapezoid rule over evaluation points, then
> normalised by the `t_env` span so it reads as a mean return.

The first 100 000 steps are excluded deliberately: early learning is dominated by
exploration and is the noisiest part of the curve.

**Secondary metrics** (declared tie-breakers only, in this order)

1. mean greedy return over the final 100 000 steps;
2. greedy episode success rate (all food collected within the horizon);
3. successful cooperative loads per greedy episode;
4. failed-LOAD resolution within 10 transitions;
5. detector-event rate;
6. guidance activation rate;
7. learning stability;
8. magnitude of shaping relative to environmental reward.

## 1.8 Selection rule (mechanical, applied only after all four finish)

1. Exclude every `beta` failing any safety exclusion in §1.6.
2. Among the **remaining positive** `beta` values, find the best primary metric.
3. If that value's **bootstrap 95 % interval overlaps** with a smaller safe
   positive `beta`, **select the smaller `beta`** (parsimony under uncertainty).
4. Use secondary metrics only as declared tie-breakers, in the §1.7 order.
5. **If no positive `beta` is both safe and at least directionally competitive
   with `beta = 0`, do not force a positive selection.** Report `beta` as
   unresolved.
6. Final/test seeds are never inspected.

Bootstrap: 10 000 resamples over greedy evaluation points in the 100k–500k
window, resampling evaluation points with replacement. This describes the
variability of **this one trajectory**, not of the method, and is reported as
such.

## 1.9 Prohibition on adaptive changes

Between conditions, only operational information is inspected: process/PID,
`t_env`, disk usage, API attempts, token cost, log growth and correctness
invariants. **Returns, success rates and learning curves are not inspected until
all four conditions have finished.** No configuration is changed mid-pilot, no
condition is stopped because performance looks poor, and no condition is re-run.

The pilot stops entirely — without resuming under modified configuration — on
secret exposure, malformed logging, reward-reconciliation failure, invalid
physical guidance, accounting failure, model substitution, NaN/Inf, an
unexpected greedy-evaluation provider call, or a global cost/attempt overrun.

## 1.10 Expected limitations of one development seed

* **No significance.** One seed gives no estimate of between-seed variance, which
  in sparse-reward LBF is typically the dominant term. Any ranking here is
  provisional.
* **Seed-specific optima.** The `beta` that looks best on seed 91 may not
  generalise; this pilot produces a *candidate*, not a frozen constant.
* **Non-independent episodes.** Returns come from one continuously-updating
  policy; the bootstrap interval reflects trajectory noise only.
* **Guidance exposure may differ** between conditions even at equal caps,
  because a different `beta` changes the policy, which changes which stalls the
  detector sees. §1.5 governs the case where a cap makes this material.
* **Early-training distribution.** 500k steps is a fraction of a full 2M run; the
  coordination-stall rate was already shown to be non-monotonic in the Stage D
  beta-zero measurement.
* **One environment, one algorithm.** Nothing here transfers to other layouts,
  agent counts or learners without re-measurement.

---

# PART 2 — RESULTS

> Appended after all four conditions completed. **Part 1 was not edited.** Where
> a preregistered rule produced an inconvenient outcome, the rule stands and the
> outcome is reported as-is.

> ## ⚠ FACTUAL CORRECTIONS (2026-08-17, commit after `c7a318e`)
>
> Two statements in the originally committed Part 2 were wrong. They are
> corrected in place below. **No observed number, no measurement and no
> conclusion changed** — the pilot outcome remains **UNRESOLVED**, and the
> original evidence and commit `c7a318e` stand as the historical record.
>
> **1. "Monotonic" was wrong.** The original text said detector events and the
> unguided share rose *monotonically with `beta`*. They did not, across all four
> values. Observed decisions were **1 294 / 1 660 / 1 996 / 893** for
> `beta` = 0.00 / 0.05 / 0.10 / 0.25.
>
> > **Correct interpretation:** detector-event frequency varied strongly with
> > `beta`. It increased from `beta = 0` through `beta = 0.10`, while
> > `beta = 0.25` produced substantially fewer events. The fixed call cap created
> > materially unequal guidance coverage.
>
> The monotone trend holds only over the first three values, and is stated that
> way now. Nothing about the confound conclusion depends on monotonicity — it
> depends on coverage being *unequal*, which it was (81.2 % / 63.2 % / 53.5 % /
> 100 %).
>
> **2. "19 % of the way through the run" conflated two different quantities.**
> The 18.8 % figure is the fraction of `beta=0.00`'s *eligible decisions* that
> subsequently received no guidance. It is **not** a position in training.
> Measured from the decision log, the caps were exhausted **late**, not early:
>
> | Condition | 1000th attempt at `t_env` | First `budget_exhausted` | Fraction of training elapsed | Steps remaining |
> |---|---|---|---|---|
> | `beta=0.00` | 461 456 | 461 636 | **92.3 %** | ~38 400 |
> | `beta=0.05` | 441 088 | 441 106 | **88.2 %** | ~58 900 |
> | `beta=0.10` | 378 557 | 378 726 | **75.7 %** | ~121 300 |
> | `beta=0.25` | never | never | — | — |
>
> These are measured values and refine the approximate "~55k remaining, ~89 %"
> characterisation: that figure matches `beta=0.05` closely, while `beta=0.00`
> exhausted with ~38k steps remaining (92.3 % through).
>
> Both corrections are pinned by regression tests in
> `tests/test_selma_stage_e_recalibration_design.py`.

## 2.1 Headline

**`beta` is UNRESOLVED. No value is frozen, and this pilot cannot freeze one.**

Two independent reasons, both preregistered:

1. **Cap exhaustion made the conditions non-comparable (§1.5, exclusion 9).**
   Three of four conditions consumed their entire 1 000-attempt budget. Across
   `beta` = 0.00 / 0.05 / 0.10 the share of guidance decisions left unguided rose
   18.8 % → 36.8 % → 46.5 %; `beta = 0.25` was fully covered (0 %). Guidance
   coverage therefore varied strongly with `beta` and is confounded with the
   variable under test. **The relationship is not monotonic across all four
   values** — see the correction notice at the top of Part 2.
2. **The only positive `beta` with full guidance coverage (`0.25`) was not
   directionally competitive with `beta = 0` (rule 5).**

Additionally, **all four conditions sit at the performance floor** (greedy
success 0.1 %–1.9 %), so the pilot has almost no power to separate them
regardless of exposure.

## 2.2 Execution

| | |
|---|---|
| Conditions run | 4 of 4, sequentially, in the preregistered order |
| Exit status | 0 for all four; orchestrator status `completed` |
| Final `t_env` | 500 049 in every condition |
| Episodes trained | 10 001 in every condition |
| Wall clock | 2 450 / 2 428 / 2 428 / 2 296 s (~2.7 h total) |
| Live attempts | 1 000 / 1 000 / 1 000 / 853 = **3 853** (cap 4 000) |
| Cost | $0.20665 ×3 + $0.17627 = **$0.79622** (cap $1.00) |
| Retries | 0 |
| Model returned | `gpt-5.4-nano-2026-03-17` in every call, all four conditions |
| Performance inspected during execution | **no** |

## 2.3 A. Integrity — all four clean

| Condition | `t_env` | NaN/Inf | Reward reconciliation errors | Telescoping residual |
|---|---|---|---|---|
| `beta=0.00` | 500 049 | none | 0 | **0.0** |
| `beta=0.05` | 500 049 | none | 0 | **0.0** |
| `beta=0.10` | 500 049 | none | 0 | **0.0** |
| `beta=0.25` | 500 049 | none | 0 | **0.0** |

`training_reward == raw_team_reward + shaping_contribution` on all 2 000 200
logged transitions, and the PBRS transformation telescoped exactly.

## 2.4 B. Provider / cache / guidance funnel

| Condition | Decisions | Distinct keys | Cache hits | Attempts | Budget-exhausted | **Unguided** | Activations |
|---|---|---|---|---|---|---|---|
| `beta=0.00` | 1 294 | 1 236 | 51 | 1 000 | 243 | **18.8 %** | 1 051 |
| `beta=0.05` | 1 660 | 1 588 | 49 | 1 000 | 611 | **36.8 %** | 1 049 |
| `beta=0.10` | 1 996 | 1 894 | 68 | 1 000 | 928 | **46.5 %** | 1 068 |
| `beta=0.25` | 893 | 853 | 40 | 853 | **0** | **0.0 %** | 893 |

Provider correctness was perfect in every condition: 100 % parse-and-ground,
**zero** validator rejections, **zero** physically invalid activations, **zero**
greedy-evaluation provider calls, no model substitution. Zero-plan and singleton
decisions: **0** everywhere (the detector always offered 3 admissible cells ⇒ 6
joint plans, as in Stage D).

Tokens: 877 000 input / 25 000 output per capped condition; 748 081 / 21 325 for
`beta=0.25`. Cached input **0** throughout. Median latency 0.80–0.86 s.

**This table is the pilot's main finding.** Detector-event frequency varied
strongly with `beta`. It increased from `beta = 0` through `beta = 0.10`
(1 294 → 1 660 → 1 996) while `beta = 0.25` produced substantially fewer events
(893). Against a fixed 1 000-call budget this created materially unequal
guidance coverage: 81.2 % / 63.2 % / 53.5 % / 100 %. **There is no monotonic
relationship across all four `beta` values**, and none is claimed.

## 2.5 C. Shaping magnitude — all within the preregistered limits

| Condition | max abs shaping | P95 abs (non-zero) | max / smallest positive reward | P95 / median positive reward | Total abs shaping |
|---|---|---|---|---|---|
| `beta=0.00` | 0.000000 | 0.000000 | 0.000 | 0.000 | 0.000 |
| `beta=0.05` | 0.031667 | 0.013333 | 0.063 (limit 0.50) | 0.027 (limit 0.25) | 31.80 |
| `beta=0.10` | 0.070000 | 0.023333 | 0.140 (limit 0.50) | 0.047 (limit 0.25) | 60.39 |
| `beta=0.25` | 0.133333 | 0.058333 | 0.267 (limit 0.50) | 0.117 (limit 0.25) | 129.23 |

Smallest and median positive environmental reward were both 0.5 in every
condition. **Exclusions 6 and 7 did not fire for any `beta`** — even 0.25 stayed
comfortably inside both thresholds. Non-zero unscaled PBRS terms: 14 976 /
14 763 / 15 074 / 12 863. TTL reached its 15-transition budget (max served 14) in
all conditions.

## 2.6 D. Learning — every condition is at the floor

| Condition | **Primary AUC** (100k–500k) | Bootstrap 95 % | Final-100k return | Greedy success | Coop loads/ep |
|---|---|---|---|---|---|
| `beta=0.00` | 0.00109 | [0.00038, 0.00213] | 0.00350 | 0.70 % | 0.007 |
| `beta=0.05` | 0.00256 | [0.00113, 0.00438] | 0.00800 | 1.60 % | 0.016 |
| `beta=0.10` | **0.00353** | [0.00213, 0.00525] | 0.00950 | 1.90 % | 0.019 |
| `beta=0.25` | 0.00038 | [0.00000, 0.00100] | 0.00050 | 0.10 % | 0.001 |

Failed-LOAD resolution within 10 transitions: 1.76 % / 1.50 % / 2.25 % / 1.11 %.

Learner stability (Sacred metrics, matched by `selma_beta`): loss max |101.7|,
|101.9|, |101.9|, |101.9|; grad-norm max ~1 965 in all four; no non-finite
values. **Exclusion 8 did not fire.**

**These are floor-level results.** A 1.9 % greedy success rate over 40
evaluation points × 20 episodes is roughly 15 successful episodes out of 800.
The differences between conditions amount to a handful of episodes and are well
within sampling noise for counts this small. The apparent ordering
(0.10 > 0.05 > 0.00 > 0.25) should not be read as an effect.

## 2.7 E. Comparability

* Identical `t_env` (500 049), episodes (10 001), evaluation schedule (41 points
  in the analysis window) and non-`beta` configuration across all four.
* **Guidance exposure was not equal** — the decisive failure. See §2.4.
* Every condition used a fresh empty cache and its own output directory; no
  cache entry, checkpoint or learner state was shared.
* All artifacts complete and non-empty.

## 2.8 F. Mechanical application of the selection rule

Applied exactly as preregistered in §1.8, without modification:

1. **Exclude unsafe positive `beta`.** `beta=0.05` and `beta=0.10` excluded under
   exclusion 9 (611 and 928 cap-exhausted events ⇒ materially unequal exposure).
   `beta=0.25` survives. *(`beta=0.00` also exhausted its cap, but its coverage
   cannot affect its learning: at `beta=0` guidance contributes exactly zero
   shaping, so the baseline is plain QMIX either way and remains a valid
   reference.)*
2. **Best primary metric among safe positives:** `beta=0.25`, AUC 0.00038.
3. **Rule 5 — directional competitiveness:** 0.00038 < 0.00109 (`beta=0`).
   The only safe positive `beta` is *worse* than no shaping.

> ### Result: **`beta` is UNRESOLVED.** No positive value is selected.

The rule refused to select, which is the correct outcome: the two conditions
that scored highest are exactly the two that were most starved of guidance, and
the only fully-covered positive condition underperformed the baseline.

## 2.9 A revised pilot is required

Per §1.5, because conditions exhausted their caps, **`beta` must not be frozen
from this comparison.** Required changes before re-running:

1. **Raise the per-condition call budget to give full coverage at every `beta`.**
   The highest-demand condition needed ~1 996 calls, and demand grows with
   `beta`; ~2 500–3 000 per 500k-step condition would have covered all four.
   Budget from the *observed* demand curve, not from the `beta=0` rate.
2. **Equalise exposure explicitly, not implicitly.** A fixed call cap is not a
   fixed treatment. Either fund full coverage, or cap *coverage fraction*
   identically across conditions so the confound is removed by design.
3. **Train long enough to leave the floor.** At 500k steps the baseline reaches
   0.7 % greedy success; no `beta` can be discriminated against a floor. Either
   extend to ~2M steps or use a task variant the baseline partially solves.
4. **Multiple seeds.** One seed cannot separate a real effect from trajectory
   noise at these magnitudes.
5. Keep the frozen contracts, the prompt, the price table and the safety
   exclusions unchanged — all of them worked exactly as intended here.

## 2.10 Limitations

* **One development seed (91).** No between-seed variance estimate; no
  significance. The bootstrap intervals describe one trajectory only.
* **Floor-level performance** in all conditions; the pilot has little power.
* **Confounded guidance exposure** (§2.4) — the decisive limitation.
* **Non-independent episodes** from one continuously-updating policy.
* One environment, one algorithm, 500k steps.
* Final/evaluation seeds were **not** touched, run or inspected.

## 2.11 Explicit statements

* **Only seed 91 was used.** No other seed, environment or algorithm was run.
* **No final/test seed was inspected.**
* **No adaptive change occurred.** The four `beta` values, order, caps and
  configuration were fixed before execution; performance was first examined only
  after the fourth condition exited. No condition was stopped early, re-run or
  added.
* **No cap was raised mid-pilot**, including when conditions exhausted their
  budgets late in training (`beta=0.00` at `t_env` 461 636, ~92 % through;
  `beta=0.05` at 441 106, ~88 %; `beta=0.10` at 378 726, ~76 %).
* **This is not a performance result and not a significance claim.**
* **`beta` was not frozen.** The preregistered rule returned UNRESOLVED.
* The frozen Stage D contracts — detector, joint plans, provider, prompt v3,
  response schema, cache key, physical validator, TTL, model — were unchanged.
* The recovery plan was untouched, untracked and unstaged throughout.
* **The API key was never printed, read, hashed, copied or logged.** It was
  sourced only into the orchestrator process and inherited by each training
  child; it never appeared in an argument, a log, a Sacred config or a report.
