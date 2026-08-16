# SELMA Stage C — deterministic PBRS slice: acceptance record

**Date:** 2026-08-16
**Branch:** `stage-c-mock-pbrs`
**Parent Stage B commit:** `977954ba2cc84180f638fedaa5be46e8e1dfc00e`
(`selma: close Stage B detector validation`, 2026-08-03)

This document records what was accepted, how it was verified, and what the
verification does **not** establish. It covers the deterministic slice only:
guidance is produced by an in-process deterministic mock. **No live LLM was
used anywhere in this work, and no caching layer exists.**

---

## 1. Scope of the slice

Committed source:

| Path | Role |
| --- | --- |
| `src/selma/modes.py` | the mode/provider/beta configuration contract |
| `src/selma/provider.py` | provider schema, deterministic validation, deterministic mock |
| `src/selma/guidance.py` | guidance lifecycle, dynamic potential, PBRS accounting, C.7 log |
| `src/runners/episode_runner.py` | wiring: construction-time validation, per-transition call, runtime invariants |
| `src/envs/gymma.py` | two read-only post-step accessors for the successor potential |
| `src/config/default.yaml` | new keys, all defaulting to inert |
| `tests/test_selma_stage_c_pbrs.py` | provider, validation, potential, PBRS unit tests (49) |
| `tests/test_selma_stage_c_ordering.py` | mode contract, lifecycle ordering, telescoping (93) |
| `tests/test_selma_stage_c_runner.py` | end-to-end through the real runner/MAC/learner (22) |
| `analysis/verify_stage_c.py` | durable out-of-process β = 0 equivalence verifier |

`src/selma/detector.py` is **unmodified**. `git diff -- src/selma/detector.py`
produces no output at the commit this record accompanies.

---

## 2. Mode semantics (exact)

Defined and enforced in `src/selma/modes.py::validate_selma_config`, called from
`EpisodeRunner.__init__` **before any SELMA object is constructed**. Every rule
below raises `SelmaConfigError` at construction time; none degrades silently.

`off` (the default)
: No detector, no provider, no guidance controller, no shaping, no guidance log.
  Bit-for-bit unmodified EPyMARL. `selma_variant_id` must be empty (nothing
  would consume it), `selma_provider` must be `none`, `selma_beta` must be `0.0`,
  `selma_guidance_log_csv` must be empty.

`shadow`
: Stage B detector only, observation-only, writing nothing back into training.
  The variant bank keeps its full 48-variant calibration behaviour when
  `selma_variant_id` is empty. A provider, a non-zero beta, or a guidance log
  path is a **configuration error**, not an ignored setting.

`active`
: Stage C. Detector (frozen variant only) + an explicitly named provider + PBRS.
  Requires: `selma_provider` naming a supported provider (`deterministic_mock`
  is the only one in this build); `selma_variant_id == wf15_m3_t5_ibon`;
  `common_reward=True`. `selma_beta` may be `0.0` — that is the equivalence
  configuration and it must remain bit-identical to `off`.

**Unknown modes are rejected.** The audited implementation compared
`selma_mode == "shadow"` and treated every other string — including the
plausible-looking `"active"` — as `off`, so a misconfigured experiment would
have trained silently as an unshaped baseline. `selma_beta` is additionally
rejected when non-numeric or non-finite.

New config keys, all inert by default:

```yaml
selma_mode: "off"          # off | shadow | active
selma_beta: 0.0            # PBRS scale, applied exactly once
selma_provider: "none"     # none | deterministic_mock
selma_guidance_log_csv: "" # per-transition PBRS log (empty = disabled)
```

---

## 3. Frozen detector

Active mode may use exactly one detector variant:

**`wf15_m3_t5_ibon`** — `W_fail = 15`, `m = 3`, team gate `W_team = 5`, inbound
help `on` — frozen in `docs/project/selma_stage_b_selection_record.md`
(Stage B closed at `977954ba`).

The constant lives in `selma/modes.py::FROZEN_ACTIVE_VARIANT`. In active mode the
runner selects that variant **from** `build_variant_grid()`; its parameters are
never reconstructed. Any other id, or an empty id, is rejected: without the
restriction the whole 48-variant bank would be offered to the provider, which
would produce up to 48 queries and 47 replacements per struggle. Only events
with `row_type == "coalesced_event"`, the frozen `variant_id`,
`coalition_feasible` true and no `suppression_reason` are ever offered
(`GuidanceController._eligible`).

---

## 4. Corrected PBRS equations

Potential (`src/selma/guidance.py`), with `d_max_plus_load = (rows-1)+(cols-1)+1`
(15 for 8×8, 19 for 10×10):

```
d_i   = manhattan(agent_i, target_food) + 1   while the target food exists
        manhattan(agent_i, target_food)       otherwise
phi_i = -d_i / d_max_plus_load                          in [-1, 0]
Phi_t = mean(phi_i) over active recipients;  0.0 when no guidance is active
```

`d_i` is a **bounded progress heuristic**, not an exact shortest-path or
action-count estimate — food blocks movement in LBF, so true navigation cost can
exceed it. PBRS needs a bounded, consistently evaluated potential, not exact
navigation cost; with Manhattan distance the bound is exact, so `phi_i ∈ [-1,0]`
always.

Three **distinct** quantities, with **beta applied exactly once**:

```
pbrs_term_unscaled   = gamma * phi_after_effective - phi_before
shaping_contribution = beta * pbrs_term_unscaled
training_reward      = raw_team_reward + shaping_contribution
```

* `pbrs_term_unscaled` is logged beta-free, so a run's shaping can be re-derived
  at any beta. The audited name `shaping_reward` held the beta-scaled term and
  made the two readings indistinguishable in a log; it is deliberately not reused.
* Nothing is clipped, at any magnitude.
* When `beta == 0.0`, `training_reward` is **assigned directly** from
  `raw_team_reward` — never `raw + 0.0` — so no dtype or floating-point
  representation change can occur.
* The raw environment reward is never mutated; shaping is added on the way into
  the replay buffer only.

Terminal rule (Stage 0 bootstrap verdict,
`docs/selma_stage0_terminal_mask.md`): a **stored** terminal has no successor
value, therefore no successor potential —

```
phi_after_effective = 0.0 if stored_terminated else phi_after_real
```

`phi_after_real` is still computed and logged separately, so the two readings are
distinguishable.

---

## 5. Lifecycle ordering and TTL semantics

`GuidanceController.apply_transition` is the **single** per-transition code path.
The real `EpisodeRunner` and every lifecycle test call it, so the ordering under
test is the ordering that trains; `test_runner_uses_the_shared_transition_path`
pins that the runner does not re-implement the lifecycle.

For transition `t`, in this order:

1. **TTL accounting.** The guidance in force during `t` (`g_t`) records that it
   served one transition. This is what TTL counts.
2. **Determine `g_{t+1}`.** Removal first — terminal, then target-food removal,
   then TTL exhaustion — and only then activation or replacement from this
   step's events.
3. **Successor potential** `phi_after_real` is computed from `s_{t+1}` and
   **`g_{t+1}`** (never from `g_t`).
4. **Stored terminal forces** `phi_after_effective = 0`.

The value returned as `phi_after_effective(t)` is therefore exactly the value the
next transition consumes as `phi_before(t+1)`; the runner **carries it forward**
rather than recomputing it, which is what makes the transformation telescope.
The audited revision priced the transition with guidance that was removed
immediately afterwards, so every expiry left an uncancelled
`beta * gamma^(t+1) * Phi` term in the return — systematically negative, since
`Phi ≤ 0`.

**TTL semantics.** `TTL_STEPS = 15`: the advice may be `guidance_before` for at
most exactly 15 environment transitions and must not influence the 16th. Advice
also ends early on target-food removal, on the terminal transition, or on
replacement; a replacement restarts the budget. Expiry reasons are recorded as
`ttl` / `target_removed` / `terminal`.

**Other lifecycle rules enforced:**

* Guidance is **training-only**. In greedy evaluation (`test_mode=True`) the
  controller is not consulted at all: zero queries, zero activations, zero
  shaping, no log rows, and reward stays the raw environment reward.
* No provider query is issued on a final transition (`episode_ended`): a subgoal
  activated there could never influence an action. Such events are counted as
  `terminal_suppressed_events`.
* `reset_episode()` clears episode-local guidance but keeps cumulative counters,
  so every episode starts at `Phi = 0`.
* A rejected provider response **never partially applies**: no active subgoal,
  `Phi = 0`, `F_t = 0`, with a stable machine-readable `fallback_reason`.
* Agent positions are supplied explicitly by the caller (the frozen detector row
  carries none). A missing recipient position raises
  `MissingAgentPositionError` and is recorded as the documented rejection
  `missing_agent_position`. The audited `(-1, -1)` fallback — which fabricated a
  position and made every role degenerate — is gone.
* Two runtime invariants are asserted in the runner itself, not only in tests:
  the carried `phi_before` must equal `Phi(s_t, g_t)` recomputed from the Stage A
  pre-step record (tolerance `1e-9`), and guidance must not survive a stored
  terminal; `beta == 0` producing non-zero shaping also raises.

---

## 6. Test result

```
/home/aum/selma/venv/bin/python -m pytest -q
298 passed, 95 warnings in 35.48s
```

The full repository suite, including the Stage 0/A/B tests, at the commit this
record accompanies. Stage C contributes 164 of those tests (49 + 93 + 22).

---

## 7. β = 0 equivalence: 36/36

```
/home/aum/selma/venv/bin/python analysis/verify_stage_c.py
RESULT: PASS -- 36 checks, every learning-relevant artifact identical.
```

Two identically seeded 40-episode trainings (development seed **4242**,
deliberately not one of the reserved experimental seeds 1–5), on
`lbforaging:Foraging-8x8-2p-2f-coop-v3`, `time_limit=50`:

* **A** — `selma_mode=off`
* **B** — `selma_mode=active`, `selma_provider=deterministic_mock`,
  `selma_variant_id=wf15_m3_t5_ibon`, `selma_beta=0.0`

In B the guidance machinery is **fully live**: it queries the provider,
activates, ages, replaces and expires subgoals, and writes the C.7 log. Only the
reward term is zero.

All 36 checks passed, covering: both runs exiting 0; every artifact present and
non-empty; the per-episode CSV identical cell-by-cell **and** byte-identical;
learner statistics (loss, grad_norm, q_taken_mean, target_mean, td_error_abs,
returns, episode length) identical series; model parameters and optimizer state
compared as **loaded tensors**, tensor by tensor; the in-process replay tensors
(`obs`, `state`, `actions`, `avail_actions`, `reward`, `terminated`, `filled`)
and stored reward values identical; learner parameters identical after training;
and the β = 0 log invariants — every `shaping_contribution` exactly `0.0`,
`training_reward == raw_team_reward` on every transition, at least one non-zero
`pbrs_term_unscaled` (proving guidance really ran), and no provider query on any
stored-terminal transition.

Scale of what was compared in this run: 241 episode-CSV rows (byte-identical),
8 learner metric series, 20 loaded tensors across `agent.th` / `mixer.th` /
`opt.th`, 40 episodes of replay tensors, and 2050 guidance-log rows of which 67
carried a non-zero `pbrs_term_unscaled`. Guidance counters in B (reported, not
compared): 6 provider calls, 6 queries, 6 activations, 0 replacements, 6
expiries (`ttl` 4, `terminal` 2), 0 rejections, cumulative unscaled PBRS term
`0.1977`, cumulative shaping contribution `0.0`.

The verifier is written to make a *false* pass structurally impossible. An
earlier ad-hoc bash comparison reported `BYTE-IDENTICAL: YES` for two runs that
had both crashed, because `sha256sum` failed on the missing files, both captured
hashes were the empty string, and `[ "" = "" ]` is true. Accordingly: return
codes are checked; every artifact is required present **and** non-empty before
anything is compared; comparisons are of loaded contents, never of filenames or
of two absent values; and **an empty comparison set is itself a failure** — a
check that compared nothing reports FAIL, not PASS. Provider/guidance counters
are reported separately and are expected to differ; they are not learning state.

---

## 8. β > 0 smoke invariants

From `tests/test_selma_stage_c_runner.py`, a 40-episode in-process run at seed
4242 with `beta = 0.25` (`shaped_run`), driving the real runner, MAC, action
selector and QMIX learner. These are **invariants**, not performance results:

* `beta > 0` actually reaches the stored reward: the shaped trajectory differs
  from the `off` trajectory, and `activation_count > 0` (the check is not vacuous).
* The C.7 log header matches `GuidanceCsvWriter.COLUMNS` exactly and contains
  every required field.
* Arithmetic reconciles exactly on **every** row (≤ 1e-12):
  `shaping_contribution == beta * pbrs_term_unscaled`,
  `pbrs_term_unscaled == gamma * phi_after_effective - phi_before`,
  `training_reward == raw_team_reward + shaping_contribution`.
* Boundary identity `phi_after_effective(t) == phi_before(t+1)` holds across the
  whole logged run within each episode (≤ 1e-12).
* Every episode starts at `phi_before = 0.0` and every stored-terminal
  transition has `phi_after_effective = 0.0`.
* No `query_triggered` on any stored-terminal transition; `terminal_cause` is
  `true_terminal` or `time_limit` on terminal rows and empty everywhere else,
  and matches the horizon/remaining-food derivation. (This LBF wrapper reports a
  horizon ending as `terminated`, not `truncated`, so the cause cannot be read
  off the env flags.)
* Mock usage fields (`query_tokens_prompt`, `query_tokens_completion`,
  `query_latency_s`, `query_cost_usd`) are `n/a` on every row — *not applicable*,
  never a fabricated `0` that could be summed.
* Activation, ongoing and expiry transitions all occur; every expiry carries a
  legal reason and `phi_after_effective = 0.0`.
* All stored rewards are finite under shaping.
* Greedy evaluation after training leaves provider calls, queries, activations
  and total shaping unchanged, adds no log rows, and leaves no active guidance.

---

## 9. Fails-first evidence

Every Stage C test in `test_selma_stage_c_ordering.py` was written against the
audited implementation and required a code change to pass. To make that claim
checkable rather than asserted, each audited defect was **reintroduced** into a
scratch copy of the tree (outside the repository) and the suites re-run. Results:

| Reintroduced defect | Caught | Example failing tests |
| --- | --- | --- |
| D1 price transition `t` with `g_t` (pre-removal guidance) | 22 tests | `test_discounted_telescoping_identity_with_ttl_expiry`, `test_activation_boundary_uses_new_guidance_after_and_none_before` |
| D2 beta applied twice | 2 | `test_beta_is_applied_exactly_once`, `test_shape_applies_beta_exactly_once` |
| D3 unknown `selma_mode` silently degrades to `off` | 1 | `test_unknown_selma_mode_is_rejected` |
| D4 missing agent position falls back to `(-1,-1)` | 5 | `test_build_request_requires_agent_positions`, `test_controller_records_missing_positions_as_a_documented_rejection` |
| D5 `beta=0` computes `raw + 0.0` instead of assigning `raw` | 2 | `test_beta_zero_assigns_the_raw_reward_object_directly` |
| D6 stored terminal keeps a non-zero successor potential | **0** | — see below |
| D7 helper levels dropped from the feasibility re-check | 15 | `test_controller_activates_from_a_real_detector_row`, `test_build_request_consumes_a_real_detector_row` |
| D8 TTL off-by-one (guidance serves L+1 transitions) | 3 | `test_ttl_guidance_never_influences_transition_l_plus_one` |
| D9 provider queried on the final transition | 2 | `test_no_provider_query_on_a_terminal_transition` |
| D10 non-frozen detector variants offered to the provider | 2 | `test_only_frozen_variant_events_are_offered` |

**D6 is honestly reported as uncaught.** Removing
`phi_after_effective = 0.0 if stored_terminated else phi_after_real` alone
changes nothing observable — including across the full end-to-end runner suite,
which still passes 22/22 — because terminal expiry has already cleared the
guidance by that point, so `phi_after_real` is `0.0` anyway. The two mechanisms
are redundant by design (defense in depth): the explicit zeroing does not depend
on `_expire` ordering, and `_expire` does not depend on the stored mask being
read correctly downstream. Removing **both** terminal guards *is* caught, by
`test_true_terminal_zeroes_the_successor_potential` and
`test_time_limit_ending_follows_the_stage0_bootstrap_verdict` (and it breaks the
runner suite). No test distinguishes the two guards individually, and none is
claimed to.

The mutation harness lives outside the repository (scratch only) and is not part
of this commit; the results above are the record of that run.

---

## 10. Limitations — what this does **not** establish

1. **No live LLM was used.** The only provider in this build is
   `DeterministicMockProvider`, a pure function of the request: no network, no
   credentials, no tokens, no cost, no cache. `selma_provider` values other than
   `deterministic_mock` are rejected at configuration time. The provider schema
   is deliberately provider-neutral so a later live provider needs no change
   here, but nothing about live-LLM behaviour, latency, cost or response quality
   is tested or implied.
2. **No caching layer exists**, and none was implemented.
3. **No performance conclusion is supported.** Nothing here shows that shaping
   helps, hurts, or is neutral for learning. The β > 0 runs are 40-episode smoke
   runs at a single development seed (4242), executed to check invariants and
   plumbing. They are far too short, too few and too unseeded to say anything
   about returns, sample efficiency or cooperation. Any performance claim
   requires the full seeded experimental protocol, which has not been run.
4. **Single seed, single environment.** All verification is at seed 4242 on
   `Foraging-8x8-2p-2f-coop-v3`. The experimental seeds 1–5 are deliberately
   untouched and reserved.
5. **The β = 0 equivalence is empirical, at this seed and length.** It is a very
   strong check — replay tensors, learner state, optimizer state and stored
   rewards all identical — but it is a test, not a proof over all configurations.
6. **`Phi` uses a bounded progress heuristic**, not true navigation cost
   (section 4). The PBRS policy-invariance argument needs only boundedness and
   consistent evaluation, but the potential is not claimed to be a value-function
   estimate.
7. **The mock's assignment policy is arbitrary** (lexicographically smallest
   offered loading cells, round-robin). It is designed for determinism and
   coverage of the lifecycle, not for good advice.
8. **LBF-only.** The post-step accessors on `GymmaWrapper` return `None` outside
   `lbforaging`, and active mode raises at construction if they are missing.
   Nothing about other environments is claimed.
9. **`beta`, `TTL_STEPS = 15` and the potential shape are frozen choices, not
   tuned ones.** No sweep over them has been run.
10. **The C.7 guidance log is written synchronously per transition** when
    enabled. It is off by default and was not enabled for any timing measurement.

---

## 11. Reproduction

```bash
cd /home/aum/selma/epymarl
git diff --check
git diff -- src/selma/detector.py          # must be empty
/home/aum/selma/venv/bin/python -m pytest -q
/home/aum/selma/venv/bin/python analysis/verify_stage_c.py
```

Verifier outputs land in `analysis/out/stage_c_beta_zero/` (git-ignored);
Sacred run directories land in `results/` (git-ignored). Neither is committed.
