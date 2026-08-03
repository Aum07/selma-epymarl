# SELMA Stage C — implementation contract (mocked subgoals and dynamic PBRS)

**Status: DOCUMENTATION ONLY. Nothing implemented.** Revised 2026-08-02 with a theoretical
correction to §4.6 and §8 (dynamic PBRS admits randomly changing potentials; the earlier
determinism/augmented-state requirement is withdrawn), and with four implementation-shaping
decisions frozen (§6.2: TTL = 15, deterministic mock provider, `guidance_schema_v1`, and D8 —
now **fully frozen**, covering both the Manhattan distance metric and `d_max_plus_load`).
`beta = 0` remains the implementation default.

**Nothing implemented.** No source file, `default.yaml` entry or
detector parameter has been modified. No test, experiment or training run was executed to produce
this document. No provider or LLM was called. Nothing committed or pushed.

**Drafted:** 2026-08-02, from read-only inspection of `src/selma/`, `src/runners/episode_runner.py`,
`src/config/default.yaml` and `docs/project/SELMA_Claude_Code_Implementation_Spec_2026-07-30.md`.

**Concurrent activity:** the single authorised 10x10 shadow-validation run
(`seed91_val10x10_20260802T151245Z`) is operating autonomously and was not disturbed, inspected or
analysed while drafting this. **No partial validation results appear anywhere in this document.**

---

## 1. Frozen Stage B input

### 1.1 The frozen configuration

Stage C consumes **exactly one** detector configuration, selected on the 8x8 development condition
under the locked rule recorded in `docs/project/selma_stage_b_selection_record.md`:

| Parameter | Value | Origin |
|---|---|---|
| Variant id | `wf15_m3_t5_ibon` | selected |
| `W_fail` | 15 | selected |
| `m` | 3 | selected |
| `W_team` | 5 | selected |
| Inbound-helper suppression | on | selected |
| `w_inbound` | 3 | fixed across all 48 |
| `cooldown_steps` | 5 | fixed across all 48 |
| `rearm_steps` | 3 | fixed across all 48 |
| `max_events_per_episode` | 2 | fixed across all 48 |

Selected under: `d = √((1 − coverage_norm)² + res@10_norm²)`, min–max normalised over the 18 eligible
configurations, ascending d, tie-break lower events/1k then variant name. d = 0.607348; margin over
the runner-up ≈ 0.0059. It is **the deterministic balanced choice under the frozen rule**, not a
uniquely or statistically superior configuration.

### 1.2 No detector retuning during Stage C

**Binding.** For the entire duration of Stage C:

* `W_fail`, `m`, `W_team`, inbound suppression, `w_inbound`, `cooldown_steps`, `rearm_steps` and
  `max_events_per_episode` are **frozen at the values above** and are not tuned, swept or adjusted.
* No new detector variant is introduced, and the 48-variant grid is not re-run.
* Detector behaviour is not modified in response to Stage C shaping outcomes. If shaping performs
  poorly, that is a finding about the guidance or the potential, **not** licence to retune the
  trigger.
* Any change to a detector parameter after Stage C begins must be reported explicitly as a post-hoc
  change, with pre-change and post-change results both shown, and invalidates the Stage B selection
  record as a *frozen* input.

The single legitimate reason to revisit the detector is a **correctness defect** — an invariant
failure or a semantic bug — which is a bug fix, must be reported as such, and requires re-running
Stage B calibration.

### 1.3 Exact struggle-event fields Stage C will consume

The shadow row schema is defined by `COLUMNS` in `src/selma/shadow_log.py` (**55** fields). Stage C
consumes the following **subset**. Every field is derived from raw environment information by
existing, tested code; **no field is supplied by, or influenced by, the provider.**

**Event identity and scoping**

| Field | Stage C use |
|---|---|
| `event_id` | canonical identity; cache key component; logged on every shaped transition |
| `condition`, `seed`, `t_env`, `episode_id`, `episode_step` | provenance, replay, cache key component |
| `variant_id` | must equal `wf15_m3_t5_ibon`; a mismatch is a hard error |

**Target and coalition (fix the subgoal; the provider may not alter these)**

| Field | Stage C use |
|---|---|
| `food_id`, `food_position`, `food_level` | target food identity and location |
| `trigger_agent_ids` | which agent(s) stalled |
| `recipient_ids` | the agents a subgoal may be assigned to — **authoritative** |
| `loader_ids`, `loader_levels`, `loader_level_sum` | current coalition capability |
| `level_deficit`, `required_extra_level` | how much additional level is required |
| `helper_ids`, `helper_levels`, `helper_distances` | feasible helper set and cost |
| `candidate_loading_cells` | the **only** admissible cells a role may reference |
| `coalition_feasible`, `coalition_reason` | hard gate: infeasible ⇒ no query, no shaping |
| `n_feasible_subsets` | validator context |

**Lifecycle state (governs activation and repeat suppression)**

| Field | Stage C use |
|---|---|
| `raw_would_fire`, `would_fire_after_inbound` | whether the event is eligible at all |
| `latched`, `cooldown_remaining`, `rearm_false_streak` | repeat-call suppression |
| `events_used`, `event_budget` | per-episode cap enforcement |
| `suppression_reason` | logged; non-empty ⇒ no query |
| `inbound_help`, `inbound_helper_ids` | a helper already inbound ⇒ suppressed |
| `team_gate_open` | gate closed ⇒ no event |

**Diagnostics carried through, never used to gate**

`raw_team_reward`, `food_removed`, `progress`, `trigger_distance`, `trigger_distance_delta`,
`failed_load_count_in_window`, `resolved_within_5`, `resolved_within_10`,
`coalesced_candidate_count`, `selected_cell_assignment`, `tiebreak_evidence`.

**Invariant.** `raw_team_reward` is the *raw* environment reward. The detector must never observe
`training_reward`. This is already asserted by the Stage B test suite and must remain asserted.

---

## 2. Stage C data flow

```
  [1] struggle event            detector, frozen config, raw env info only
        │                       gate: coalition_feasible ∧ ¬suppressed ∧ events_used < budget
        ▼                             ∧ training mode (never during greedy evaluation)
  [2] structured context        code-built, deterministic, provider-neutral
        │                       target food, recipients, levels, deficit,
        ▼                       candidate_loading_cells, allowed roles
  [3] guidance request          cache lookup on canonical context hash FIRST
        │                       miss ⇒ single provider call; hit ⇒ reuse, no call
        ▼
  [4] validated response        deterministic validator; any failure ⇒ fallback
        │                       ⇒ no active subgoal ⇒ Phi = 0 ⇒ F_t = 0
        ▼
  [5] active subgoal            per-recipient validated loading cell + role
        │                       carries activation step, TTL, event_id
        ▼   Phi_t = mean_i(−d_i / d_max_plus_load)
  [6] shaped reward             F_t = beta * (gamma * phi_after_effective − phi_before)
        │                       training_reward = raw_team_reward + F_t   (never clipped)
        ▼
  [7] expiry / reset            TTL elapsed, food gone, episode end, or replacement
                                ⇒ Phi → 0 with the boundary term made explicit and logged
```

**Ordering** follows `C.2` of the implementation spec: shaping is computed after `env.step` and
before `post_transition_data["reward"]` enters the batch; episode-local SELMA state is cleared only
after the final transition has been computed and stored.

**Two hard rules.** Code, not the provider, fixes event identity, target food, recipient agents,
candidate loading cells, allowed roles, TTL, beta and the reward formula. And greedy evaluation
makes **zero** provider calls and uses raw reward only.

---

## 3. Structured guidance schema

### 3.1 Request (code-built, provider-neutral)

Contains only fields already derived and validated by the detector: `event_id`, `event_type`, target
food (`food_id`, `food_position`, `food_level`), `trigger_agent_ids`, `recipient_agents`, and
per-agent `{agent_id, level, position, allowed_roles, candidate_loading_cells}`, plus
`level_deficit` / `required_extra_level`.

The request carries **no reward information, no Q-values, no policy parameters and no training
state**. It is provider-neutral: the identical structure is issued to the deterministic mock and to
any later live provider (spec D.1 requires an identical schema).

### 3.2 Response (what the provider may return)

| Field | Type | Constraint |
|---|---|---|
| `event_id` | string | must equal the request's `event_id` |
| `recipients` | list[int] | **must be a subset of** the request's `recipient_agents` |
| `target_food_id` | tuple | **must equal** the request's `food_id` |
| `helper_coalition` | list[int] | subset of `helper_ids`; combined level must clear `required_extra_level` |
| `assignments` | list[{agent_id, role, loading_cell}] | `role ∈ allowed_roles`; `loading_cell ∈ candidate_loading_cells` for that agent |
| `confidence` | float ∈ [0,1] | advisory only — **never** scales beta or the potential |
| `schema_version` | string | must equal **`guidance_schema_v1`** (frozen, §6.2 D4) |

The provider **selects among code-supplied options**. It cannot introduce a food, an agent, a cell or
a role that the request did not offer. This bounds the blast radius of a bad or adversarial response
to "chose a worse option among valid ones".

### 3.3 Deterministic validation

Validation is pure, side-effect-free and deterministic — identical input yields an identical verdict,
independent of wall-clock, RNG or provider. Checks, in order, all of which must pass:

1. Schema and version match; all required fields present and correctly typed.
2. `event_id` matches; the event is still live (not expired, food still exists).
3. `recipients ⊆ recipient_agents` and non-empty.
4. `target_food_id` equals the request's food.
5. Every `assignment.agent_id ∈ recipients`; no duplicate agent.
6. Every `loading_cell ∈ candidate_loading_cells[agent_id]`.
7. Every `role ∈ allowed_roles[agent_id]`.
8. Coalition level sum ≥ `food_level` (feasibility re-checked against raw state, not trusted).
9. `confidence ∈ [0,1]` if present.

**Any failure ⇒ `provider_valid = false` ⇒ no active subgoal ⇒ Φ = 0 ⇒ F_t = 0.** A rejected
response never partially applies. Validation failures are counted and reported as a fallback rate.

### 3.4 Timeout, malformed response and provider failure

| Situation | Behaviour |
|---|---|
| Timeout (wall-clock budget exceeded) | abandon; `provider_fallback = timeout`; **zero shaping** |
| Malformed / unparseable output | `provider_fallback = malformed`; **zero shaping** |
| Schema-valid but semantically invalid (§3.3) | `provider_fallback = invalid`; **zero shaping** |
| Transport / API error | `provider_fallback = error`; **zero shaping** |
| Repeated failure for one event | event marked failed; **not retried within the episode** |

Every fallback path yields **exactly zero shaping**, not a degraded or default subgoal. The training
signal must never depend on provider availability. A silent fallback to a heuristic subgoal would
confound "LLM guidance helped" with "the heuristic helped" — the deterministic ablation (spec D.3)
exists precisely to measure that heuristic separately, so it must not leak into the LLM condition.

Timeouts must not stall training indefinitely: the budget is bounded and a breach is a fallback, not
a block.

### 3.5 Caching — bounding call volume

**Cache key:** a canonical hash over the *code-built context*, not over free text —
`(condition, schema_version, provider_id, model_version, temperature, food_id, food_level,
sorted(recipients), sorted((agent, level, position)), sorted(candidate_loading_cells),
required_extra_level)`.

Rules:

* **Lookup precedes any call.** A hit reuses the stored response and issues **no** provider call.
* Deliberately excluded from the key: `t_env`, `episode_id`, wall-clock, and any training state — so
  structurally identical situations reuse one response and the cache is replay-stable.
* Deliberately included: `provider_id`, `model_version`, `temperature`, `schema_version` — a
  provider or prompt change must not silently reuse stale guidance.
* Per-event single-flight: one in-flight call per `event_id`; concurrent triggers coalesce.
* Cache is persisted per run and logged (hits, misses, evictions) so call volume is auditable.
* **Uncontrolled repetition is bounded by three independent mechanisms**: the detector's
  latch/cooldown/rearm, the per-episode `max_events_per_episode = 2` cap, and the cache. The cap and
  the latch bound calls even on a total cache miss.

The cache is part of the experimental object. The random-trigger control (spec D.2) must match
SELMA's **call count and recipient-count distribution**, so cache behaviour must be recorded to make
that matching well-defined.

---

## 4. PBRS formulation

### 4.1 Proposed potential

For each agent `i` assigned an active, validated subgoal:

```
d_i = |agent_row_i − food_row| + |agent_col_i − food_col|      # Manhattan distance
      + 1 if the target food still exists                      # the LOAD action allowance

phi_i = − d_i / d_max_plus_load

Phi_t = mean( phi_i for i in active recipients )        if guidance is active
Phi_t = 0                                               otherwise
```

**The distance metric is frozen as Manhattan distance to the target-food coordinate** (§6.2 D8).
Obstacle-aware shortest-path distance is **not** used in Stage C.

This is a **bounded progress heuristic, not an exact shortest-path or action-count estimate.** Food
cells block movement in LBF, so the true navigation cost can exceed `d_i`; `d_i` is deliberately not
claimed to predict it. What the potential must do is vary monotonically with *approach* to the
target and stay bounded, and Manhattan distance does both.

The bound is now exact. Manhattan distance between any two cells of the grid is at most
`(rows − 1) + (cols − 1)`, so with the LOAD allowance `d_i ≤ d_max_plus_load` and therefore
`phi_i ∈ [−1, 0]` and `Phi_t ∈ [−1, 0]` in every configuration — satisfying §4.6 condition 6
(bounded potentials) by construction rather than by assumption.

Note the assigned loading cell remains fully part of the contract: it is validated against
`candidate_loading_cells` (§3.3), it determines role admissibility, and it is logged. It is simply
not the point the potential measures distance *to* — the potential measures approach to the food
itself, which is common to all recipients of an event.

`d_max_plus_load` is a **documented, environment-specific constant**, frozen (§6.2 D8) as
`(rows − 1) + (cols − 1) + 1` — **15** for 8x8, **19** for 10x10. It is recorded per condition, held
fixed for all runs of that condition, and does not depend on any run's data.

Mean (not sum) aggregation keeps the potential scale independent of recipient count, so a 3-recipient
event does not mechanically receive a larger shaping magnitude than a 2-recipient one. This matters
directly for cross-condition comparison, since the validation condition has three agents.

### 4.2 Shaping term

```
F_t            = beta * ( gamma * phi_after_effective − phi_before )
training_reward = raw_team_reward + F_t
```

* `gamma` is **the same** value QMIX uses; it is read from the config, never redefined.
* `F_t` is **never clipped**. Clipping breaks the telescoping identity and voids the potential-based
  form.
* `raw_team_reward` remains available and unmodified for evaluation, detection, reporting and all
  success metrics.
* When `beta == 0`, `training_reward` is **assigned directly** from `raw_team_reward` — not computed
  as `raw + 0.0` — so no dtype or floating-point representation change can occur.

### 4.3 Team-level versus per-agent application

The potential is computed **per agent** (`phi_i`, all logged individually) but aggregated to a
**single scalar team potential** before shaping. QMIX receives **one scalar team reward**; separate
shaped rewards are never fed to separate agents.

This is a deliberate constraint, not an implementation convenience. QMIX (Rashid et al., 2018)
factorises a scalar team value monotonically; injecting per-agent shaped rewards would change the
learning problem into something the factorisation was not defined for, and would place the result
outside the setting of the multi-agent PBRS result in Devlin and Kudenko (2011).

### 4.4 Terminal-state handling

Do not branch on names such as `truncated`. Use the Stage-0 recorded bootstrap verdict:

```
phi_after_effective = 0                     if the replay mask says "no bootstrap"
                    = real next-state Phi   if the replay mask says "bootstrap"
```

* True absorbing terminal ⇒ zero next potential.
* Time limit ⇒ follows the Stage-0 probe: bootstrapped ⇒ retain real next potential;
  non-bootstrapped ⇒ zero, matching the existing learner and the completed baselines.

If EPyMARL's time-limit mask is later corrected, that is a **new experimental protocol** and all
baselines must be rerun.

### 4.5 Activation, expiry and boundary terms

Guidance activation, replacement, expiry and removal all change the time-indexed potential. Their
boundary terms are **part of the formula**, not artefacts to be smoothed away:

| Transition | `phi_before` | `phi_after` |
|---|---|---|
| Activation at `t+1` | `Φ(s_t, no guidance) = 0` | `Φ(s_{t+1}, g_{t+1})` |
| Ongoing | `Φ(s_t, g_t)` | `Φ(s_{t+1}, g_t)` |
| Replacement at `t+1` | `Φ(s_t, g_t)` | `Φ(s_{t+1}, g_{t+1})` — the **new** guidance |
| Expiry / removal at `t+1` | `Φ(s_t, g_t)` | `0` |

**The same guidance object must never be used on both sides of a transition across a change.** Doing
so hides the boundary term and silently breaks the telescoping identity. Each boundary term is
logged with `guidance_transition` and `guidance_age`.

Expiry occurs on the earliest of: **TTL of 15 steps** elapsed (frozen, §6.2 D2), target food
removed, terminal transition, or replacement by newer guidance.

### 4.6 Policy invariance — what is provable and what is not

**The construction does not automatically inherit the *fixed*-potential guarantee of Ng, Harada and
Russell (1999), because the potential here is not fixed. It is a *dynamic* potential, and dynamic
potentials are exactly the case treated by Devlin and Kudenko (2012).**

*What the verified literature establishes.*

* Ng, Harada and Russell (1999) prove that `F(s,a,s') = γΦ(s') − Φ(s)` for a **fixed** potential
  `Φ: S → ℝ` leaves the optimal policy set unchanged in a single-agent MDP.
* Devlin and Kudenko (2012) extend this to a **time-indexed, dynamically changing** potential,
  `F(s,t,s',t') = γΦ(s',t') − Φ(s,t)`. Their treatment explicitly admits potentials that change
  during learning, **including potentials that change randomly**. The guarantee does not rest on the
  potential being a deterministic function of state: it rests on the shaping being the difference of
  the potential evaluated at the two endpoints of the transition, which telescopes over a trajectory.
* Devlin and Kudenko (2011) show that in the multi-agent case PBRS preserves the **Nash-equilibrium
  set** under its assumptions, but **can change which equilibrium learning reaches**.

*Correction to an earlier draft of this contract.* An earlier version of this section asserted that
a stochastic provider breaks the dynamic-PBRS result, and that recovering it would require guidance
to be a deterministic function of an augmented state. **That was wrong and is withdrawn.** Provider
stochasticity affects **reproducibility** — replay, cache behaviour, exact re-derivation of a run —
which is a serious engineering concern and is why §3.5 caches on a canonical context hash. It does
**not**, by itself, invalidate the telescoping reward transformation. Neither the augmented-state
construction nor determinism of guidance is required for the shaping to be potential-based.

*The finite-episode identity.*

For a trajectory of length `T`, the total discounted shaping added to the return is

```
sum_{t=0}^{T-1} gamma^t * beta * ( gamma * Phi_{t+1} - Phi_t )
    = beta * ( -Phi_0 + gamma^T * Phi_T )
```

This holds for **any** sequence of potentials `Phi_0 … Phi_T`, however they were produced —
deterministically, stochastically, by a provider, or arbitrarily. Only the endpoints survive.

**When `Phi_0 = 0` and the terminal `Phi_T = 0`, the cumulative discounted shaping is exactly zero
for each completed trajectory.** SELMA satisfies both by construction: no guidance is active at
episode start, so `Phi_0 = 0`; and terminal potential is forced to zero (§4.4). Shaping therefore
redistributes reward *within* an episode without altering the discounted return of any completed
trajectory.

*The conditions that are actually required.*

What matters is the arithmetic of the transformation, not the provenance of the potential:

1. **Use the learner's exact `gamma`.** Read from the config; never redefined or approximated. A
   mismatched discount breaks the telescoping.
2. **Retain the source potential that existed for the transition.** `Phi_t` must be the potential as
   it stood when the transition occurred — not recomputed later under changed guidance.
3. **Compute the realised successor potential consistently.** `Phi_{t+1}` is the potential actually
   realised at `t+1`, including any change that occurred; both endpoints use the same definition and
   the same state features.
4. **Apply boundary terms at activation, replacement and expiry.** Each contributes its exact
   difference term (§4.5). Reusing one guidance object on both sides of a change hides the boundary
   term and breaks the identity.
5. **Force terminal potential to zero** per the Stage-0 bootstrap verdict (§4.4), so the `gamma^T *
   Phi_T` endpoint vanishes.
6. **Keep potentials bounded.** `Phi` is normalised by a documented environment constant
   (§4.1, §6.2 D8), so no transition can contribute an unbounded shaping term.
7. **Omit no eligible transition.** Every transition in the episode contributes its difference term;
   skipping transitions — including on truncation or early episode-state clearing — leaves a residual
   that does not cancel.

These are properties of the implementation and are verified directly by the tests in §5, including
the discounted telescoping identity on a fixed trajectory.

*Multi-agent qualification — retained.*

Devlin and Kudenko (2011) concerns the **equilibrium set**. Preservation of that set does **not**
imply that exploration is unchanged, nor that the equilibrium learning actually reaches is the same.
Shaping changes the reward signal an agent experiences during learning and can therefore change the
trajectory of learning even where the equilibrium set is preserved.

Accordingly, **no claim is made** that shaping leaves QMIX convergence unchanged, that learned
parameters are identical, that finite-training policies are identical, or that any empirical
improvement follows from the theory. Deep function approximation under finite training is outside
what the cited results cover. The only bit-identity claimed anywhere is the `beta = 0` case, which
is an implementation property verified by test (§5A), not a theoretical result.

*Provable versus empirical — the distinction to maintain in all writing.*

| Claim | Status |
|---|---|
| `F = γΦ(s',t') − Φ(s,t)` implemented in potential-difference form, unclipped | **Verifiable by construction and test** |
| Finite-episode telescoping identity `Σ γ^t β(γΦ_{t+1} − Φ_t) = β(−Φ_0 + γ^T Φ_T)` | **Verifiable by construction and test** |
| With `Φ_0 = 0` and terminal `Φ_T = 0`, cumulative discounted shaping is exactly zero per completed trajectory | **Verifiable by construction and test** |
| `beta = 0` reproduces unshaped QMIX bit-identically | **Verifiable by test** (§5A) |
| Fixed `Φ(s)` preserves the optimal policy set (single-agent MDP) | **Literature-supported** (Ng et al., 1999) — the fixed-potential case |
| Dynamic, time-indexed and randomly changing `Φ(s,t)` preserves the policy result | **Literature-supported** (Devlin and Kudenko, 2012) — **conditional on the construction satisfying the seven conditions above** |
| Multi-agent PBRS preserves the Nash-equilibrium set | **Literature-supported** (Devlin and Kudenko, 2011) — conditional on its assumptions; the equilibrium *reached* may still change |
| QMIX learning benefit under shaping (convergence, learned parameters, finite-training policy) | **Empirical** — measured, never inferred from theory |
| Provider quality — whether LLM-selected subgoals are better than the deterministic heuristic | **Empirical** — the C2−C5 contrast |
| Sample-efficiency or final-return improvement | **Empirical** — the C2−C0, C2−C4, C2−C3 contrasts |

Policy-invariance language is used only for the fixed-potential and dynamic-potential *literature
results*, and only alongside the statement that their application here is conditional on the seven
implementation conditions holding. No policy-invariance claim is made about SELMA's learning
outcomes.

---

## 5. Mandatory acceptance tests before any training

**All must pass before any `beta > 0` training run.** Numbering follows spec C.5–C.6.

**A. `beta = 0` equivalence (non-negotiable).** Compare `selma_mode=off` against
`selma_mode=active, provider=deterministic_mock, beta=0` under: same seed, same initial parameters,
CPU, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, deterministic Torch settings, and a provider/detector
RNG **isolated from all training RNGs**.

*Fixed-trace exact equality:* selected actions; observations and global states; available actions;
raw individual rewards; raw team rewards; stored training-reward tensors; terminated/truncated
causes; stored terminal masks; episode returns; detector-independent CSV fields.

*Short-training exact equality at every learner update:* loss; chosen Q-values; target Q-values;
model parameter tensors or stable hashes; optimizer state; final checkpoint hash.

Any mismatch **blocks** live-provider integration. Diagnose RNG consumption, reward dtype, terminal
masks, batch writes, test-mode leakage and guidance state before proceeding.

**B. No shaping before a valid detector event.** No event, or a suppressed event (latch, cooldown,
rearm, budget, closed gate, inbound helper), or `coalition_feasible = false` ⇒ `F_t == 0` exactly,
and zero provider calls.

**C. No shaping after expiry.** After TTL elapse, food removal, episode end or replacement, the
expiry boundary term is produced exactly once and every subsequent transition has `F_t == 0`.

**D. Terminal potential handling.** True terminal ⇒ `phi_after_effective == 0`. Time-limit
transitions follow the recorded Stage-0 replay mask, tested in both bootstrap and non-bootstrap
configurations.

**E. Invalid or failed provider response ⇒ zero shaping.** Each of timeout, malformed, semantically
invalid and transport error independently yields `F_t == 0`, no active subgoal, and a recorded
fallback reason. Partial application is a test failure.

**F. Deterministic replay and cache behaviour.** Identical context ⇒ identical validated guidance and
identical `F_t`; a cache hit issues no provider call; the cache key changes when `provider_id`,
`model_version`, `temperature` or `schema_version` changes; single-flight holds per `event_id`.

**G. Event cap and recipient enforcement.** `events_used ≤ max_events_per_episode = 2` in every
episode; assigned recipients ⊆ detector `recipient_ids`; assigned cells ⊆ `candidate_loading_cells`;
a response violating either is rejected with zero shaping.

**H. No reward leakage into shadow mode.** In `selma_mode=shadow`: `total_shaping_reward == 0.0`,
`provider_call_count == 0`, `query_count == 0`, `active_subgoal_count == 0`; RNG states identical to
`off`; the ordinary episode CSV byte-identical to `off`. This preserves the guarantee already
verified in Stage B, which is what makes the completed 8x8 calibration valid.

**I. Formula tests** (spec C.5): same gamma as QMIX; ordinary transition equals
`beta*(gamma*phi_after − phi_before)`; activation, replacement and expiry each use the correct
two-sided potentials; no-guidance transition gives zero; no clipping at extreme values; discounted
telescoping identity `Σ γ^t F_t / beta == −Φ_0 + γ^T Φ_T` on an artificial trajectory; mean
aggregation invariant to duplicated recipient scale; detector always reads raw reward; greedy
evaluation uses raw reward and makes zero provider calls.

---

## 6. Proposed experimental controls

Per spec D.2/D.3. Constant across all guided conditions: provider/model, prompt/schema, validator,
potential function, beta, TTL, cooldown/refresh eligibility, fallback behaviour.

| # | Condition | Purpose |
|---|---|---|
| C0 | **Unmodified QMIX** | reference; already have 8x8 baselines |
| C1 | **Stage C wired, `beta = 0`** | proves the machinery is inert; must be bit-identical to C0 |
| C2 | **Selective SELMA shaping** (event-triggered, `beta > 0`) | the proposed method |
| C3 | **Non-selective / always-on guided PBRS** | isolates *selectivity* from *guidance* |
| C4 | **Random-trigger control** | matched to C2's call count and recipient-count distribution |
| C5 | **Deterministic ablation** (frozen detector + deterministic coalition + nearest-feasible cell + same PBRS, **no LLM**) | isolates LLM contribution from the obvious LBF heuristic |

**On C3 (always-on comparator).** Include **only if consistent with the approved methodology** — it
multiplies provider calls and cost, and if the approved protocol caps call volume it may be
infeasible as specified. Two defensible reductions: run it at a reduced training budget as a
cost-bounded probe, or replace it with a *deterministic* always-on variant that removes provider cost
while still isolating selectivity. **Unresolved — see §6.2 D6.**

**Ablations needed to separate detector benefit from shaping benefit.** The pairwise contrasts are
what carry the argument:

| Contrast | Isolates |
|---|---|
| C2 − C0 | total effect of the full method |
| C2 − C4 | benefit of triggering on *genuine struggle* rather than at matched random times — **the detector's contribution** |
| C2 − C5 | benefit of *LLM* subgoal selection over a deterministic heuristic given the same detector and PBRS — **the guidance contribution** |
| C2 − C3 | benefit of *selectivity* over always-on guidance |
| C1 − C0 | that the machinery itself is inert (must be zero) |

C4 must be run **after** the corresponding C2 run so its global step schedule can contain exactly the
same number of calls and a permutation of C2's recipient counts, using a separate recorded RNG seed,
selecting only valid training refresh opportunities, and never triggering during greedy evaluation.

Report separately for each condition: performance/AUC/sample efficiency; environment return and
success; queries per 1 000 steps; tokens, latency and cost; recipient counts; active shaping steps;
fallback rate; natural-resolution rate; deterministic-ablation gap.

### 6.2 Decisions FROZEN now

The following four are frozen as of this document and are not revisited during Stage C.

| id | Decision | **Frozen value** |
|---|---|---|
| **D2** | Advice TTL | **15 steps**, ending earlier on any of: target food removed, terminal transition, or replacement by newer guidance. Whichever occurs first ends the advice and produces the expiry boundary term (§4.5) exactly once. |
| **D3** | Provider for Stage C | **Deterministic mock provider.** Stage C uses the mock exclusively. No live provider or model is frozen here. |
| **D4** | Guidance schema | **`guidance_schema_v1`** — the request/response structure of §3.1–3.2. Version-stamped and part of the cache key (§3.5); any change bumps the version and invalidates the cache. |
| **D8** | Distance metric **and** `d_max_plus_load` | **FULLY FROZEN.** `d_i` = Manhattan distance from agent `i` to the target-food coordinate, `abs(agent_row − food_row) + abs(agent_col − food_col)`, plus the `+1` LOAD allowance while the food exists. `d_max_plus_load = (rows − 1) + (cols − 1) + 1` ⇒ **15** for 8x8, **19** for 10x10. Obstacle-aware shortest-path distance is **not** used. Recorded per condition, held fixed for all runs of that condition, never re-derived from run data. |

**Verification of the reachable-coordinate convention (D8).** Read-only inspection of the installed
LBF source (`lbforaging/foraging/environment.py`) confirms the convention the formula assumes:
`rows`/`cols` are `self.field.shape[0]`/`[1]`; movement validity is bounded by `position[0] > 0`,
`position[0] < self.rows - 1`, `position[1] > 0`, `position[1] < self.cols - 1`. Coordinates are
therefore 0-indexed over the **full** grid with no excluded border ring, so the maximum Manhattan
separation of any two cells is `(rows − 1) + (cols − 1)`, and `+1` accounts for the LOAD action.
With `d_i` frozen as Manhattan distance, `d_i ≤ d_max_plus_load` holds exactly and
`phi_i ∈ [−1, 0]` is guaranteed in every configuration.

**Why the exact navigation cost is not required.** Food cells block movement in LBF, so an
obstacle-aware shortest path can exceed the Manhattan distance. This does not matter and is not a
defect. PBRS requires a potential that is **bounded** and **consistently evaluated at both endpoints
of every transition** (§4.6, conditions 1–7); it does **not** require `Phi` to encode the
environment's exact navigation cost. The potential is a shaping signal, not a value estimate or an
admissible heuristic. Manhattan distance is therefore adopted deliberately: it is exactly bounded,
cheap, deterministic, obstacle-independent — so it cannot change discontinuously when an unrelated
food is collected — and it is trivially reproducible from logged coordinates. `d_i` is a **bounded
progress heuristic, not an exact shortest-path or action-count estimate**, and must be described
that way.

**`beta` remains 0 as the implementation default** (§6.3 D1). Stage C is built, tested and merged
with `beta = 0`, which by §5A must be bit-identical to unshaped QMIX.

### 6.3 Decisions still OPEN

Not frozen here. Choosing any of these after seeing outcomes would convert them into fitted
parameters and destroy the frozen-selection discipline established in Stage B.

| id | Decision | Options | Recommended freeze point |
|---|---|---|---|
| **D1** | **`beta > 0`** (shaping scale) | (a) fixed value chosen so max \|F_t\| is a stated fraction of a raw reward unit; (b) scale-matched so `beta·(1/d_max_plus_load)` is a stated fraction of the per-step reward scale; (c) small pre-declared sweep reported **in full** as sensitivity, never as a best-pick | Freeze **before any `beta>0` run**. `beta = 0` remains the default until then. |
| **D3b** | **Live provider / model** (Stage D only) | one pinned model, pinned version, temperature and structured output | Freeze **before the first Stage D run**; unchanged thereafter |
| **D5** | **Training budget** | (a) 2M to match Stage B; (b) 10M to match existing baselines; (c) 2M for C1/C5, 10M for headline conditions | Freeze **before C2** |
| **D6** | **Always-on comparator scope (C3)** | (a) full; (b) reduced-budget probe; (c) deterministic always-on; (d) omit with stated justification | Decide **before C2** — C4's matching design depends on it |
| **D7** | **Seed count** | (a) 5; (b) 10; (c) power-informed from observed baseline variance | Freeze **before C2**. The 10M baseline analysis indicates seed count must rise well above 5 for power against a variance-scale effect; treat 5 as a floor |
| **D9** | **Provider timeout budget** | fixed wall-clock bound; breach ⇒ fallback | Freeze **before Stage D** |


---

## 7. Post-run 10x10 validation table (empty — to be completed after the run finishes)

**Not populated.** The 10x10 run is in progress and its outputs have not been inspected. Rows are
filled only from the completed run under the rule frozen in
`docs/project/selma_stage_b_selection_record.md` §2.

### 7.1 Frozen-configuration outcome

**Run executed; corrected-protocol analysis complete** (post-run, pre-ranking erratum). Run
`seed91_val10x10_20260802T151245Z`, exit code 0, final `t_env` 2 000 007. Full report:
`docs/project/selma_stage_b_10x10_validation_report.md`.

| Field | Value |
|---|---|
| Selected variant | `wf15_m3_t5_ibon` (frozen from 8x8; **not retuned, not replaced**) |
| Eligible under E1–E4 in 10x10? | Literal E4: **0/48 eligible** (protocol non-applicable). Under **E4-corrected** (PROTOCOL ERRATUM 001): **eligible**, 18/48 eligible set |
| Pareto-efficient in the 10x10 eligible set? | **No** (frontier holds 4 of 18) |
| Rank under the frozen rule (ascending d) | **18 of 18 — last**; rank 1 is `wf15_m2_t10_ibon` at d = 0.540482 |
| d (10x10 eligible-set normalisation) | **1.076289**; endpoints coverage [18.7236, 23.4514], res@10 [1.3110, 1.4010] |
| Episode coverage % | 21.57 (reported descriptively; **not** used for selection) |
| Natural resolution @10 % | 1.40 (descriptive) |
| Events per 1 000 steps | 5.144 (descriptive) |
| Cap-bound episodes % | 4.14 (descriptive) |
| Timing early / mid / late % | 55.2 / 25.3 / 19.5 (descriptive) |
| Invariant status | ✅ all pass (15 × 48 = 720 checks, condition-adjusted) |
| Invalid events | ✅ 0 (`semantically_invalid_events` and `_candidates` both 0, all 48) |
| Duplicate final events | ✅ 0 by construction; `same_food_coalesced` = 2 953 for this variant — merged candidates, **not** surviving duplicates |
| Recipient-count distribution | ✅ values in **{2, 3}** as pre-declared; ≈36 % three-recipient |
| Cross-condition interpretation | Ranks last under the frozen rule, but the res@10 axis spans only 0.09 pp (28.6× narrower than 8x8) and the selected gate was open 97.68 % of steps, so the gate barely discriminates here. Four properties changed jointly; no attribution to any one; one-seed evidence. **`wf15_m3_t5_ibon` is preserved and not retuned.** |

### 7.2 Context required alongside it

| Field | Value |
|---|---|
| Size of the 10x10 eligible set (8x8 was 18) | *(pending)* |
| Size of the 10x10 Pareto front (8x8 was 13) | *(pending)* |
| 10x10 normalisation endpoints — coverage min/max | *(pending)* |
| 10x10 normalisation endpoints — res@10 min/max | *(pending)* |
| Raw-CSV ↔ summary reconciliation | *(pending)* |
| Exit code, terminal `t_env`, runtime | *(pending)* |

### 7.3 Interpretation rules, fixed in advance

* The 8x8 invariant "every event has exactly two recipients" is a property of the **two-agent**
  condition and does **not** transfer. The 10x10 check is: recipient counts sum to `final_events` and
  lie in 2–3.
* Firing rates are **not** directly comparable like-for-like: without the `-coop-` constraint some
  food is single-agent loadable, so the coalition-required failed-LOAD population differs.
* Any disagreement is **cross-condition sensitivity** — joint evidence across grid size, agent count,
  food count and enforced-cooperation status — and is **not** attributed to any single property.
* The selected configuration is **not replaced or retuned** on the basis of these results.

---

## 8. Draft dissertation methodology text

> ### Method overview
>
> SELMA (Selective Language-Model Assistance) addresses a specific failure mode in cooperative
> level-based foraging: agents repeatedly attempt to collect food whose level exceeds their
> individual capacity, requiring simultaneous cooperation that neither agent independently
> coordinates. The method comprises four components developed and validated in sequence — an
> event-based struggle detector, a shadow-mode calibration procedure, a frozen configuration
> selection, and selective language-model guidance delivered as potential-based reward shaping.
>
> **Event-based struggle detection.** A coordination stall is defined as a concrete environment
> event: at least `m` coalition-required failed LOAD attempts by an agent on the same still-existing
> food within the last `W_fail` steps, subject to a team-stall gate, inbound-helper suppression, a
> per-food latch, a cooldown, a re-arm streak and a per-episode event budget. This replaced an
> earlier proposal to characterise struggle statistically through reward variance or trajectory
> entropy. The event formulation was preferred on four grounds: it identifies the specific situation
> to be assisted rather than a correlate of difficulty; it determines the target food, the recipient
> agents and the admissible loading cells, which the downstream guidance contract requires the code
> rather than the provider to fix; it is defined on environment events whose meaning does not drift
> as exploration anneals and the policy sharpens; and it is exactly re-derivable from a log, which
> permits independent verification of the detector's counters. Statistical signals are retained as
> candidate diagnostics but do not gate a query.
>
> **Shadow calibration.** The detector was calibrated in shadow mode, in which it observes
> transitions and records when it would fire without influencing action selection, reward or
> learning. Non-interference is established three ways: instrumented zero-guarantees (zero provider
> calls, queries, active subgoals and zero total shaping reward); equivalence tests asserting
> identical NumPy and PyTorch RNG states, exactly equal trajectories and replay contents, and a
> byte-identical episode log relative to the unmodified learner; and empirical reproduction, in which
> the calibration run reproduced two earlier prematurely terminated runs byte-for-byte over their
> full extent. A 48-variant factorial grid — `W_fail ∈ {5,10,15}` × `m ∈ {2,3}` × team gate
> ∈ {none, 5, 10, 15} × inbound suppression ∈ {off, on} — was specified before any outcome was
> observed and evaluated against a single fixed 2 000 046-step seed-91 QMIX trajectory over 40 003
> episodes, so that differences between variants are attributable solely to detector
> parameterisation. Counters were independently re-derived by streaming the 11 857 607-row raw log
> and agreed exactly with the in-process aggregates for all 48 variants; thirteen structural
> invariants, including an exact suppression ledger, hold for every variant.
>
> **Frozen configuration selection.** Configurations were restricted by four hard eligibility rules —
> team gate enabled, inbound suppression enabled, all invariants passing, and zero invalid and
> duplicate events — leaving eighteen candidates. Selection applied a deterministic rule: min–max
> normalisation of episode coverage and natural-resolution-at-10 over the eligible set, ranking by
> Euclidean distance from the ideal point, with ties broken by lower event rate and then variant
> name. The rule constitutes transparent development-set calibration rather than an unbiased
> performance estimate; the 10x10 data were not used to select or retune the detector, and the
> selection rule and configuration were frozen before analysing the completed validation run. The
> selected configuration is reported as the deterministic balanced choice under the frozen rule, not
> as a uniquely or statistically superior one: its distance advantage over the runner-up is
> approximately 0.006, the eligible candidates span only 4.5 percentage points of coverage, and the
> outcome depends on the stated normalisation domain.
>
> **Cross-condition robustness validation.** The frozen configuration was evaluated on a second
> condition without modification. This is a cross-condition robustness validation rather than a
> grid-size ablation: the environment key is the only configured field that changes, but it jointly
> alters grid size, number of agents, number of food items and enforced-cooperation status. Any
> disagreement between conditions is therefore joint evidence across those four properties and is not
> attributed to any one of them; isolating a single property would require additional conditions
> holding the others fixed.
>
> **Selective language-model guidance.** When an eligible stall event fires during training, a
> code-constructed, provider-neutral context is issued to a guidance provider, which selects among
> options the code supplies — it cannot introduce a food, an agent, a loading cell or a role that the
> request did not offer. Responses pass deterministic validation against the raw environment state;
> any timeout, malformed output, semantically invalid response or provider error yields exactly zero
> shaping rather than a degraded default, so the training signal never depends on provider
> availability. Requests are cached on a canonical context hash, and call volume is bounded
> independently by the detector's latch, cooldown and re-arm logic, by a per-episode event cap, and
> by the cache. Greedy evaluation issues no queries and uses raw reward only.
>
> **Potential-based reward shaping.** Guidance is delivered as a shaping term rather than as a direct
> reward bonus. For each agent holding an active validated subgoal, the potential is the negated Manhattan
> distance from the agent to the target-food coordinate plus a single LOAD-action allowance,
> normalised by a documented per-condition bound of `(rows − 1) + (cols − 1) + 1`, which makes the
> normalised potential lie in `[−1, 0]` by construction. This distance is a bounded progress
> heuristic rather than an exact shortest-path or action-count estimate: potential-based shaping
> requires a bounded, consistently evaluated potential, not one that encodes the environment's exact
> navigation cost; the team potential is the mean over active recipients, and
> zero when no guidance is active. Shaping follows the potential-difference form
> `F_t = β(γΦ_{t+1} − Φ_t)` with the same discount factor as the learner and without clipping, and
> the resulting scalar is added to the raw team reward before it enters the replay buffer, consistent
> with QMIX's monotonic factorisation of a single scalar team reward (Rashid et al., 2018). Raw
> reward is retained unmodified for evaluation, detection and all reported success metrics. Guidance
> activation, replacement and expiry each contribute their exact boundary term.
>
> **Status of the theoretical guarantee.** The potential-difference construction of Ng, Harada and
> Russell (1999) leaves the optimal policy set unchanged for a *fixed* potential. The potential used
> here is not fixed but dynamic, which is the case treated by Devlin and Kudenko (2012); their
> treatment explicitly admits potentials that change during learning, including randomly changing
> potentials, so the fact that guidance is produced by an external and possibly stochastic process
> does not by itself invalidate the transformation. Provider stochasticity is a **reproducibility**
> concern — it affects replay and exact re-derivation of a run, which is why requests are cached on a
> canonical context hash — rather than a theoretical one. What the transformation does require is
> arithmetic discipline: the learner's exact discount factor; retention of the source potential that
> existed for the transition; a realised successor potential computed consistently; exact boundary
> terms at activation, replacement and expiry; terminal potential forced to zero; bounded potentials;
> and no eligible transition omitted. Under these conditions the finite-episode identity
> `Σ_{t<T} γ^t β(γΦ_{t+1} − Φ_t) = β(−Φ_0 + γ^T Φ_T)` holds for any sequence of potentials however
> generated, and since no guidance is active at episode start and terminal potential is forced to
> zero, the cumulative discounted shaping is exactly zero for every completed trajectory: shaping
> redistributes reward within an episode without altering any completed trajectory's discounted
> return. In the multi-agent setting Devlin and Kudenko (2011) establish preservation of the
> Nash-equilibrium set under their assumptions while noting that the equilibrium learning reaches may
> change; preservation of that set does not imply that exploration is unchanged. Accordingly **no
> claim is made** that shaping leaves QMIX convergence unchanged, that learned parameters or
> finite-training policies are identical, or that any empirical improvement follows from the theory —
> deep function approximation under finite training lies outside what the cited results cover. The
> only bit-identity claimed is the `β = 0` case, an implementation property verified by test. Whether
> guidance quality or sample efficiency improves is empirical, measured against unmodified QMIX, a
> call-matched random-trigger control, an always-on guidance comparator, and a deterministic no-LLM
> ablation sharing the same detector and shaping.

**Citations used** — restricted to the project's verified bibliography, with nothing added:
Ng, Harada and Russell (1999); Devlin and Kudenko (2011); Devlin and Kudenko (2012);
Rashid et al. (2018).

---

## 9. Status

Documentation only. Nothing implemented, no test or training run executed, no provider called, no
source or configuration modified, nothing committed or pushed. The Stage B selection remains frozen.
The single authorised 10x10 validation run continues autonomously and was neither disturbed nor
analysed. All Stage C decisions in §6.2 remain open and must be frozen at the stated points before
any `beta > 0` experiment.
