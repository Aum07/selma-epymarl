# SELMA Stage B — pre-calibration audit

Date: 30 July 2026
Repo: `~/selma/epymarl` @ `8a169f1` + uncommitted Stage 0/A/B work
Scope: audit the three identified pre-calibration issues **against the code
itself**, correct only what is genuinely wrong, and re-prove the Stage B
guarantees. No calibration run was launched; Stage C was not implemented; no
provider, guidance, shaping or frozen configuration was added.

Summary:

| Issue | Verdict | Action |
|---|---|---|
| **A. History identity** | **Confirmed** — histories and lifecycle state were keyed on `(agent_id, food_position)`. Behaviourally equivalent in the current runner, but not the required key and not robust. | Corrected throughout + 6 regression tests |
| **B. Team-gate denominator** | **Confirmed** — gate *behaviour* was already correct; the *eligibility denominator* was not. | Corrected + 6 regression tests |
| **C. Shadow CSV storage** | **Measured — safe.** 0.351 GiB central / 4.99 GiB conservative for 2M steps against 943.85 GiB free. | No format change needed or proposed |

---

## A. History identity

### What the code actually did

Stage A already produced the correct identity. `src/selma/types.py` defines
`food_id = (episode_id, row, col)` on both `FailedLoadEvent` and
`SuccessfulLoadEvent`, and `src/selma/load_events.py:138,155` populates it. The
shadow CSV already had a correct `food_id` column
(`src/selma/shadow_log.py:27`).

The detector, however, did not use it. Before the fix, `src/selma/detector.py`
keyed **every** piece of raw history and lifecycle state on the bare position:

* `self.fail_history[(agent_id, event.food_position)]`
* `self.distance_history[(agent_id, position)]`
* `self.removal_step[position]`
* `state.foods[position]` (latch / cooldown / rearm / fired-count)
* `eligible_foods[position]`, `coalitions[position]`, `inbound_by_food[position]`
* `pending_rows` carried `position`, and `_label()` looked up
  `removal_step[position]`
* the CSV `food_id` column was *reconstructed* as
  `(self.episode_id, position[0], position[1])` at row-build time rather than
  carried through from the key.

So the answer to "does the current implementation already use the correct food
ID?" is **no**: it used `(agent_id, food_position)`, exactly the case the audit
brief says must be corrected.

### Was it producing wrong numbers?

No — not with the present runner. `_reset_episode_state()` clears all four
dictionaries, and `EpisodeRunner.run` calls `selma_bank.reset_episode(None)`
at the top of every training episode. Within one episode `episode_id` is
constant, so position ↔ food_id is a bijection and the results were correct.

The defect is that correctness depended entirely on an external caller
resetting at exactly the right moment, and the CSV's canonical identity was
re-derived from `self.episode_id` at write time rather than from the key. A
missed or mistimed reset would have silently merged two episodes' failure
histories, latches and budgets under a shared coordinate — producing plausible
but wrong events with no error.

### The correction

`src/selma/detector.py`:

* Added `ShadowDetectorBank.food_id(position) -> (episode_id, row, col)` and
  `ShadowDetectorBank.food_position(food_id) -> (row, col)`. The position is
  now *derived* from the id, never the reverse.
* `fail_history`, `distance_history`, `removal_step`, `state.foods`,
  `current`, `coalitions`, `inbound_by_food`, `eligible_foods` and
  `pending_rows` are all keyed on `food_id`.
* `_window_fail_count(agent_id, food_id, step, w_fail)`,
  `_inbound_status(coalition, food_id, step)` and `_label(step, food_id,
  horizon)` take `food_id`.
* `_add_row` now takes `food_id` and writes `row["food_id"] = food_id` plus the
  derived `row["food_position"]`. Both columns are retained: position is logged
  separately, as required, but is no longer the identity.
* `update()` rejects a step record whose `episode_id` differs from the open
  episode, so a missed `reset_episode` now fails loudly instead of merging
  state.
* Module docstring documents the identity rule.

**Preserved unchanged:** food-level coalescing (`same_food_coalesced`,
one `coalesced_event` per food per rising edge with sorted trigger agent ids)
and food-level latching (`_FoodLifecycle` per `(variant, food)` with the same
latch / cooldown / rearm semantics). The only change is the key those
structures are stored under. `event_id` keeps its existing textual form
`ep{episode}-{variant}-food{r},{c}-t{step}`.

### Regression tests (`tests/test_selma_detector.py`)

| Test | Proves |
|---|---|
| `test_raw_history_keys_are_agent_and_episode_specific_food_id` | `fail_history` keys are exactly `{(0,(7,4,4)), (1,(7,4,4))}`; no bare position appears as a key; `distance_history` keys carry the episode id |
| `test_lifecycle_and_removal_state_are_keyed_by_food_id` | `state.foods` and `removal_step` key on the triple |
| `test_same_position_in_a_new_episode_is_a_different_food` | after `reset_episode(8)`, the same coordinate starts a fresh history and inherits no latch |
| `test_mixing_two_episode_ids_without_reset_is_rejected` | a missed reset raises `AssertionError` |
| `test_shadow_row_carries_food_id_and_position_separately` | the row has `food_id == (7,4,4)` **and** `food_position == (4,4)`, and they are not the same value |
| `test_two_foods_share_one_agent_history_namespace_without_collision` | two foods in one episode keep independent `(agent, food_id)` histories |

Six pre-existing tests were updated to the corrected key semantics
(`test_inclusive_w_fail_window_boundary`, `test_distinct_transition_counting`,
`test_food_identity_stable_across_failures`,
`test_food_disappearance_records_removal_and_resolves_latch`,
`test_lifecycle_rising_edge_latch_resolve_cooldown_rearm`, and the gate test
below). No test was deleted or weakened.

---

## B. Team-gate denominator

### What the code actually did

**Gate behaviour was already correct.** `_team_gate_open` (before the fix) read:

```python
if w_team is None:            return True
if len(self.progress_history) < w_team: return False
return not any(self.progress_history[-w_team:])
```

and `progress_history.append(progress)` runs **before** the gate is evaluated on
that transition. So: a complete window of `W_team` actual transitions is
required; the current transition is inside the window (`[-w_team:]` ends at the
current append); and an incomplete episode-start window returns `False`, i.e. is
never treated as a stall. All three requirements were already met, and the
existing tests `test_incomplete_team_window_cannot_open`,
`test_team_window_includes_current_transition` and
`test_progress_closes_the_gate_by_reward_or_food_removal` covered them.

**The denominator was wrong.** `_update_variant` bumped:

```python
state.bump("eligible_steps")                       # every transition
if variant.w_team is not None and gate_open: state.bump("gate_open_steps")
elif variant.w_team is None:                 state.bump("gate_open_steps")
```

`eligible_steps` counted *every* observed transition regardless of whether that
variant's window was complete. Consequences:

1. `gate_open_percentage = gate_open_steps / eligible_steps` was diluted by the
   first `W_team − 1` transitions of every episode, which are structurally
   incapable of contributing to the numerator.
2. The eligibility total was **identical for every `W_team`** (each variant had
   its own counter object, but the value was always "number of transitions"),
   so the four `W_team` settings were not maintaining independent eligibility
   totals as required.

Measured on a 12-transition episode with no progress (all three variants seeing
the same trajectory), before the fix:

| Variant | `eligible_steps` | `gate_open_steps` | reported % | correct % |
|---|---:|---:|---:|---:|
| `W_team=15` | 12 | 0 | 0.0% | undefined (0 eligible steps) |
| `W_team=5` | 12 | 8 | 66.7% | **100%** |
| none | 12 | 12 | 100% | 100% |

LBF episodes are capped at 50 steps, so the bias is not marginal: `W_team=15`
loses 14 of 50 steps (28%) and `W_team=5` loses 4 of 50 (8%) from the
denominator on every episode. Because the spec's "practically vacuous" flag
fires when a gate is **open on ≥ 90% of eligible steps**, an inflated
denominator systematically *understates* openness and would have made a vacuous
gate look useful — precisely the wrong direction of error before a freeze
decision.

### The correction

`src/selma/detector.py`:

* Added `_team_window_complete(w_team)` (`True` for the ungated variant).
* `_team_gate_open` now delegates its completeness test to it; behaviour
  unchanged.
* Counters in `_update_variant` restructured:
  * `observed_steps` — every transition (all variants agree);
  * `eligible_steps` — transitions with a **complete** `W_team` window; this is
    now `W_team`-dependent and is the denominator;
  * `gate_open_steps` — bumped **only** on eligible steps;
  * `incomplete_team_window_steps` — the excluded transitions, reported rather
    than discarded.
* New suppression reason `team_window_incomplete`, distinct from
  `team_gate_closed`, so a start-of-episode window is never mislabelled as a
  detected stall in the shadow rows. New counter
  `suppressed_by_incomplete_team_window` alongside the existing
  `suppressed_by_team_gate`.
* **Report aggregation.** There was no report-aggregation module — the counters
  *are* the aggregation surface, and any consumer would have had to recompute
  the denominator itself. Added two methods so the denominator is defined in
  exactly one place:
  * `gate_open_percentage(variant_id)` — `gate_open_steps / eligible_steps`,
    returning `None` when no eligible step exists (a variant whose window never
    completes is never reported as "0% open");
  * `variant_report()` — per-variant `observed_steps`, `eligible_steps`,
    `incomplete_team_window_steps`, `gate_open_steps`,
    `gate_open_percentage`, `raw_candidates`, `suppressed_by_team_gate`,
    `suppressed_by_incomplete_team_window`, `candidates_after_team_gate`,
    `proportion_raw_candidates_suppressed_by_gate`, `final_events` — i.e. the
    gate statistics B.2 of the spec requires.

Firing behaviour is unchanged: candidates in an incomplete window are still
suppressed (they are not a stall). Only the accounting changed.

### Regression tests (`tests/test_selma_detector.py`)

| Requirement | Test |
|---|---|
| first eligible transition | `test_first_eligible_transition_is_exactly_the_w_teamth` (eligible count is 0 for steps 0–3 and exactly 1 on the 5th transition, `W_team=5`) |
| incomplete-window exclusion | `test_incomplete_window_steps_are_excluded_from_the_denominator` (12 observed = 4 incomplete + 8 eligible) |
| current-transition inclusion | `test_eligibility_window_includes_the_current_transition` (the step is eligible, but progress on it keeps the gate shut) |
| raw-reward progress | `test_raw_reward_progress_counts_as_progress_in_the_denominator` |
| food-disappearance progress | `test_food_disappearance_progress_counts_in_the_denominator` (removal with **zero** reward closes the gate) |
| exact percentage denominator | `test_gate_open_percentage_uses_complete_window_denominator_per_w_team` (`W_team=5` → 8/8 = 100%; `W_team=15` → 0 eligible → `None`; none → 12/12) |
| incomplete ≠ stalled, in the rows | `test_gate_closed_by_progress_is_labelled_separately_from_incomplete_window` |

---

## C. Shadow CSV storage

Measured, not estimated. Full detail in
`docs/selma_shadow_storage_projection.md` and
`analysis/out/shadow_storage_projection.json`.

30,000 steps of `Foraging-8x8-2p-2f-coop-v3`, smoke seed 4242 (reserved seeds
refused by the script), all 48 variants and all 55 columns written losslessly:
15,420 rows, 5,646,714 bytes, 366.1 B/row, **514.0 rows per 1,000 steps**
(68.3% `raw_candidate`, 31.7% `coalesced_event`).

2M-step projection: **1,028,000 rows / 0.351 GiB central**;
**14,643,850 rows / 4.99 GiB conservative** (×14.24, from the worst-seed
end-of-run growth in `failed_load_events` on the same task × 1.5 headroom).
Free space 943.85 GiB → 938.86 GiB remaining after the conservative output,
which is **0.53% of free space**.

Measured shadow overhead **×1.171**; projected 2M runtime 1.31 h shadow vs
1.12 h off.

**No change to the agreed logging format is required, and none is proposed.**
No lossy sampling was introduced, no variant or field was dropped.

---

## Re-validated Stage B guarantees

Command: `PYTHONPATH=src python3 -m pytest tests/ -q` → **88 passed** (was 72
before this audit; +16 new tests, 0 removed). `git diff --check` clean.

`tests/test_selma_shadow_equivalence.py` drives the **real** `EpisodeRunner`,
`basic_mac`, epsilon-greedy selector and QMIX `q_learner` in process, mirroring
`run_sequential`, and compares `selma_mode=off` against `selma_mode=shadow`:

| Proven | Test |
|---|---|
| selected actions, observations, global states, avail actions, raw+stored rewards, termination flags, `filled`, replay/episode batches — all SHA-256 identical per episode; final learner parameters identical | `test_off_vs_shadow_trajectory_and_replay_are_exactly_equal` |
| ordinary CSV byte-identical | `test_off_vs_shadow_ordinary_csv_is_byte_identical` |
| ordinary CSV columns unchanged by Stage B | `test_ordinary_csv_columns_are_unchanged_by_stage_b` |
| Stage A counters identical value-by-value (all 6 load-event columns + both rewarded-step columns), and non-trivially non-zero | `test_stage_a_counters_are_identical_off_vs_shadow` *(new)* |
| shaping reward exactly 0 on every transition (element-wise max |Δ| == 0 on stored rewards) | `test_shaping_reward_is_exactly_zero_on_every_transition` *(new)* |
| provider calls = 0, query count = 0, active subgoals = 0, shaping = 0.0 through the real runner | `test_runner_shadow_mode_has_zero_queries_subgoals_and_shaping` |
| greedy evaluation produces no detector state updates and no shadow rows (episodes seen, all counters, pending rows and shadow-CSV size unchanged across 5 test episodes) | `test_greedy_evaluation_never_updates_detector_state_or_rows` |
| default `selma_mode=off` creates no bank and no writer | `test_default_selma_mode_is_off_and_creates_no_bank` |
| detector consumes no training / action-selection RNG (numpy **and** torch global RNG states identical after off and after shadow) | `test_shadow_mode_leaves_numpy_and_torch_rng_states_identical_to_off` *(new)*, plus `test_detector_consumes_no_numpy_global_rng` |
| the shadow CSV is the only additional output, with the declared 55-column header | `test_off_vs_shadow_trajectory_and_replay_are_exactly_equal`, `test_shadow_mode_writes_rows_and_they_parse` |

## Files changed

| File | Change |
|---|---|
| `src/selma/detector.py` | Issue A (food_id keys throughout, episode-mixing guard, docstring) and Issue B (eligibility counters, `team_window_incomplete` reason, `gate_open_percentage`, `variant_report`) |
| `tests/test_selma_detector.py` | 6 existing tests updated to the corrected key/label semantics; 13 tests added (6 for A, 6 for B, 1 for the suppression-reason split) |
| `tests/test_selma_shadow_equivalence.py` | 3 tests added (RNG isolation, Stage A counter equality, exactly-zero shaping) |
| `analysis/shadow_storage_smoke.py` | new — Issue C measurement |
| `analysis/baseline_manifest.py`, `analysis/analyse_10m_baseline.py` | new — baseline work, no Stage B effect |
| `docs/*.md` | new reports |

`src/config/default.yaml`, `src/envs/gymma.py`, `src/runners/episode_runner.py`,
`src/selma/{__init__,types,load_events,coalition,shadow_log}.py` were **not**
modified by this audit.
