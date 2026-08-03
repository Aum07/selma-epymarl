# SELMA Stage B — held-out seed-99 validation of the frozen configuration

**Status: COMPLETE. All structural and reconciliation checks pass.**

**Configuration under test:** `wf15_m3_t5_ibon` — frozen on seed 91 before this run and **not
retuned, re-ranked or replaced** by it.
**Condition:** `lbforaging:Foraging-8x8-2p-2f-coop-v3` (identical to the development condition).
**Run:** `seed99_heldout_20260802T221211Z`.
**Validation source:** **base commit `92ece20` plus the documented selected-variant plumbing patch**
— not a clean `92ece20` checkout.

This is a **one-time held-out validation of a single frozen detector**, not a calibration or
selection exercise. No 48-variant grid was run on seed 99, no ranking or Pareto frontier was
computed, no normalisation endpoints were recomputed, no alternative variants were inspected, and
no post-hoc numerical pass threshold was invented.

---

## 1. Completion

| Check | Result |
|---|---|
| Process exited normally | ✅ `exit_code=0`, `status=COMPLETED` |
| Final `t_env` | ✅ **2 000 004** ≥ `t_max` = 2 000 000 |
| Runtime | 1 h 09 m 13 s (4 157 s) |
| Artifacts present | ✅ 11/11 |
| Artifacts stable | ✅ raw CSV unchanged across re-stat; 0 open writers |
| Artifacts parseable | ✅ summary JSON, episodes CSV, raw CSV, sampler data |
| Config vs post-patch manifest | ✅ **PASS — 15/15 fields** |
| Only `wf15_m3_t5_ibon` in outputs | ✅ single variant id in 262 317 CSV rows; 1 variant in summary |

The last *log* line reads `t_env: 1 950 354` because the log interval is 50 000; the episode CSV and
the run's own completion confirm 2 000 004.

## 2. Integrity and reconciliation

| Check | Result |
|---|---|
| Malformed records | ✅ **0** of 262 317 |
| Truncated final record | ✅ none — 55/55 fields, file ends `\r\n` |
| NUL corruption | ✅ **0** NUL bytes in 93 640 389 scanned |
| Semantically invalid final events | ✅ **0** |
| Semantically invalid candidates | ✅ **0** |
| Surviving duplicate `event_id`s after coalescing | ✅ **0** — 10 862 `coalesced_event` rows, 10 862 distinct ids |
| Final-event reconciliation (raw CSV ↔ summary) | ✅ 10 862 = 10 862 |
| Recipient reconciliation | ✅ `{2: 10 862}` both sides; sums to `final_events` |
| Detector lifecycle invariants | ✅ **15 / 15 pass** |
| Provider calls / queries / active subgoals | ✅ **0 / 0 / 0** |
| Total shaping reward | ✅ **0.0** |
| `training_reward == raw_team_reward` | ✅ see below |
| Detector absent from greedy evaluation | ✅ `episodes_seen` = 40 020 = train episodes; 4 000 greedy episodes unobserved |
| Shadow rows from test episodes | ✅ **none** — 11 812 episodes with rows, all training |
| `run.log` errors / NaN / inf | ✅ 0 matches |
| Monitor log | ✅ start line + expected stop line only |
| `dmesg` | ✅ clean |

Invariants verified: gate stage identity; inbound stage identity; the suppression ledger
(`after_inbound − coalesced − (latch + cooldown + rearm + budget) = final_events`);
`incomplete_team_window ⊆ team_gate`; per-episode histogram summing to 40 020; phase and recipient
sums; recipients exactly 2 (correct for this 2-agent `-coop-` condition); cap `m` respected;
resolution denominators; and denominator + censored = final at both horizons.

**`training_reward == raw_team_reward`.** Shadow mode has no shaping write-path — Stage C is
unimplemented, reward is inserted raw at `episode_runner.py:272`, the runner asserts
`total_shaping_reward == 0.0` at every episode boundary, and the suite tests
`test_shaping_reward_is_exactly_zero_on_every_transition` and
`test_off_vs_shadow_ordinary_csv_is_byte_identical` pass.

## 3. Frozen-detector metrics — seed 99

| Metric | Value |
|---|---|
| Episodes / observed steps | 40 020 / 2 000 004 |
| **Gate-open percentage** | **94.10 %** (1 731 385 steps) |
| **Events per 1 000 steps** | **5.4310** |
| **Events per episode** | **0.2714** |
| Final events | 10 862 |
| **Coverage** | **24.4178 %** (9 772 of 40 020 episodes fire) |
| Episodes 0 / 1 / >1 events | 30 248 / 8 682 / 1 090 |
| **Episode-cap binding** | **2.7236 %** (98 budget-suppressed) |
| **Raw candidates** | **251 455** |
| Team-gate suppressed → after gate | 2 353 → 249 102 |
| **Inbound-suppressed** → after inbound | **20 652** → 228 450 |
| **Latch-suppressed** | **196 004** |
| Cooldown / rearm suppressed | 4 621 / 16 865 |
| **`same_food_coalesced`** | **0** |
| **Recipient distribution** | **2 → 10 862 (100 %)**; 3 → 0 |
| **Natural resolution @5** | **6.2572 %** (654 resolved / 10 452 valid) |
| **Natural resolution @10** | **10.7214 %** (1 076 resolved / 10 036 valid) |
| **Censored @5 / @10** | **410 / 826** |
| Resolved: food gone / condition false | 2 231 / 4 289 |
| **Timing early / mid / late** | **70.22 / 17.08 / 12.70 %** (7 627 / 1 855 / 1 380) |

## 4. Seed 91 versus seed 99 — same detector, same condition

| Metric | seed 91 | seed 99 | abs diff | ratio 99/91 |
|---|---|---|---|---|
| Episodes | 40 003 | 40 020 | +17 | 1.0004 |
| Observed steps | 2 000 046 | 2 000 004 | −42 | 1.0000 |
| **Gate-open %** | 96.11 | 94.10 | **−2.01** | 0.9790 |
| **Final events** | 10 716 | 10 862 | +146 | 1.0136 |
| **Events / 1k steps** | 5.3579 | 5.4310 | +0.0731 | **1.0136** |
| **Events / episode** | 0.2679 | 0.2714 | +0.0035 | 1.0132 |
| **Coverage %** | 23.6007 | 24.4178 | +0.8171 | **1.0346** |
| Episodes 1 event | 8 166 | 8 682 | +516 | 1.0632 |
| Episodes >1 event | 1 275 | 1 090 | −185 | 0.8549 |
| **Cap-binding %** | 3.1873 | 2.7236 | −0.4636 | 0.8545 |
| Budget-suppressed | 172 | 98 | −74 | 0.5698 |
| **Raw candidates** | 230 268 | 251 455 | +21 187 | **1.0920** |
| Team-gate suppressed | 1 421 | 2 353 | +932 | **1.6559** |
| Inbound-suppressed | 19 871 | 20 652 | +781 | 1.0393 |
| Latch-suppressed | 177 660 | 196 004 | +18 344 | 1.1033 |
| Cooldown / rearm | 4 273 / 16 155 | 4 621 / 16 865 | +348 / +710 | 1.0814 / 1.0439 |
| `same_food_coalesced` | 0 | 0 | 0 | — |
| **Recipients = 2** | 10 716 | 10 862 | +146 | 1.0136 |
| Recipients = 3 | 0 | 0 | 0 | — |
| **Nat. resolution @5 %** | 4.4917 | 6.2572 | **+1.7654** | **1.3930** |
| **Nat. resolution @10 %** | 9.3128 | 10.7214 | **+1.4086** | **1.1513** |
| Censored @5 / @10 | 475 / 966 | 410 / 826 | −65 / −140 | 0.8632 / 0.8551 |
| Resolved: food gone | 1 892 | 2 231 | +339 | 1.1792 |
| Resolved: condition false | 4 724 | 4 289 | −435 | 0.9079 |
| **Timing early / mid / late %** | 65.53 / 19.15 / 15.32 | 70.22 / 17.08 / 12.70 | +4.69 / −2.07 / −2.62 | 1.0716 / 0.8918 / 0.8291 |

### 4.1 Reading the comparison

The headline firing behaviour transfers closely. Event rate differs by **1.4 %**
(5.3579 → 5.4310 per 1 000 steps), coverage by **0.82 percentage points**, and events per episode by
1.3 %. The recipient distribution is identical in kind — every event has exactly two recipients, as
the 2-agent cooperative condition requires — and `same_food_coalesced` is 0 on both seeds, so the
E4-corrected duplicate criterion is satisfied identically.

Three quantities differ more than the rest, and are recorded as held-out variability rather than
explained away:

* **Natural resolution @5 is 39 % higher** on seed 99 (4.49 % → 6.26 %) and @10 is 15 % higher
  (9.31 % → 10.72 %). Since natural resolution is the proxy for a trigger on a stall that would have
  cleared unaided, this is the metric on which the held-out seed looks modestly *worse*. The
  absolute change is +1.77 and +1.41 percentage points.
* **Team-gate suppression is 1.66× higher** (1 421 → 2 353) and gate-open is 2.01 points lower
  (96.11 % → 94.10 %). The gate is doing somewhat more work on seed 99.
* **Timing shifts earlier**: 65.5 % → 70.2 % of events in the early phase, with corresponding
  reductions in mid and late.

Raw candidates rose 9.2 % while final events rose only 1.4 %, because latch suppression absorbed
most of the increase (+18 344). That is the lifecycle behaving as designed.

**This variation is a result, not a failure.** It is one held-out seed against one development seed;
it supports no significance claim, and no threshold was defined in advance against which it could
pass or fail. Nothing was retuned in response to it.

## 5. Stage B exit criteria

| Criterion (implementation spec) | Status |
|---|---|
| Shadow mode is trajectory- and reward-equivalent to off mode | ✅ suite tests pass (RNG states, trajectory, replay, byte-identical ordinary CSV, zero shaping); re-confirmed on this run |
| A seed-91 calibration report exists | ✅ `docs/selma_stage_b_seed91_calibration.md` (+ 10x10 cross-condition report) |
| One detector configuration is frozen | ✅ `wf15_m3_t5_ibon`, `docs/project/selma_stage_b_selection_record.md` |
| Seed 99 provides a one-time validation report | ✅ **this document** |
| No live LLM call and no PBRS reward has occurred | ✅ `provider_call_count=0`, `query_count=0`, `active_subgoal_count=0`, `total_shaping_reward=0.0` across every run |

**All five Stage B exit criteria are satisfied. Stage B is closed.** Structural and reconciliation
checks pass on the held-out seed; the between-seed differences in §4 are recorded as held-out
variability evidence and are explicitly *not* grounds for reopening selection.

Two caveats carried forward into Stage C:

* The 10x10 result remains **one-seed cross-condition sensitivity evidence** in which grid size,
  agent count, food count and enforced cooperation changed jointly, with 95.5 % of successful loads
  solo. `wf15_m3_t5_ibon` ranked 18/18 there against an eligible res@10 range of only 0.0900
  percentage points, and its team gate was open 97.68 % of steps. None of this is evidence of robust
  external generalisation.
* Seed 99 is a **single** held-out seed on the development condition. It confirms the detector
  behaves consistently and lawfully; it does not establish statistical robustness.

## 6. Artifacts

| Item | Path |
|---|---|
| Raw shadow CSV | `/mnt/e/selma_shadow/seed99_heldout_20260802T221211Z/8x8/shadow_8x8_seed99_2M_heldout.csv` (93 640 389 B) |
| Summary JSON | `analysis/out/seed99_heldout_20260802T221211Z/8x8/summary_8x8_seed99_2M_heldout.json` |
| Episodes CSV | `…/episodes_8x8_seed99_2M_heldout.csv` |
| Run log / sampler / monitor | `…/run.log`, `…/rss_samples.csv`, `…/monitor_errors.log` |
| Exit marker | `…/exit_status.txt` |
| Machine-readable results | `…/seed99_validation_results.json` |
| Pre-run manifests | `analysis/out/seed99_preflight_20260802/{preflight,postpatch}_provenance_manifest.json` |

The raw CSV is 93.6 MB versus 3.96 GiB for the seed-91 48-variant run — about 1/45th, an independent
confirmation that the single-variant restriction took effect, alongside the direct check that only
`wf15_m3_t5_ibon` appears in the output.
