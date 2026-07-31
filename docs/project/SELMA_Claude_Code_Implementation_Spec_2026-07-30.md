# SELMA Event-Based Detector and PBRS Implementation Specification

Date: 30 July 2026

Purpose: implementation contract for Claude Code. Work stage by stage. Do not implement later stages until the current stage's tests and exit criteria pass.

## 1. Frozen project context

- Project: SELMA - Selective LLM-Guided Multi-Agent Reinforcement Learning.
- Algorithm: QMIX in EPyMARL.
- Environment family: Level-Based Foraging (LBF), episode limit 50.
- Primary reported environment: `lbforaging:Foraging-8x8-2p-2f-coop-v3`.
- Secondary reported environment: `lbforaging:Foraging-10x10-3p-3f-v3`.
- Official 2M plain-QMIX baselines are complete.
- Five 10M `10x10-3p-3f` runs using seeds 1-5 may still be running in `~/selma/epymarl-exact`.
- `~/selma/epymarl-exact` is the frozen baseline repository. Do not edit, stop, rename, or run tests in it.
- Implement only in `~/selma/epymarl`.
- Calibration uses seed 91. Seed 99 is a one-time validation seed. Seeds 1-5 are held out for reported experiments and must not tune detector thresholds, beta, TTL, cooldown, prompts, or validators.
- Do not run training, detector queries, or shaping during greedy evaluation.
- Use the term `non-selective LLM-guided baseline`, not `always-on resource`.

## 2. Final design decisions

| Area | Decision |
|---|---|
| Detector unit | A coordination-stall event identified by repeated, coalition-required failed LOADs on the same food |
| Evaluation cadence | Update detector state every environment step |
| Query cadence | Query only on a newly fired, unlatched event |
| Core evidence | Same agent-food failed-LOAD persistence, with insufficient loading level |
| Team-stall gate | Implement and shadow-test; keep only if it measurably filters candidates |
| Trigger-agent distance | Log as target persistence; do not present it as independent navigation progress |
| Coalition | Compute deterministically from levels, distances, and stable tie-breaks |
| LLM authority | Select only allowed roles/loading-cell subgoals for the precomputed recipients and target |
| Inbound help | Compute in shadow mode and compare raw versus suppressed triggers before freezing |
| Combination rule | Strict conjunction; no weighted score and no learned gate |
| Shaping | One scalar, time-indexed joint potential added before replay insertion |
| PBRS boundary | Follow the actual QMIX replay bootstrap mask established by Stage 0 |
| Reward clipping | Never clip shaping reward |
| Main plumbing guard | `beta=0` must be bit-identical to plain QMIX under a deterministic test |
| Architecture | LBF-specific extraction in the wrapper/instrumentation boundary; detector state, provider call, potential, and reward insertion in `episode_runner` |

## 3. Non-negotiable implementation order

1. Stage 0 - local code and terminal-semantics audit.
2. Stage A - counter extension and online event exposure.
3. Stage B - detector in shadow mode only.
4. Freeze detector on seed 91 and validate once on seed 99.
5. Stage C - mocked provider and PBRS plumbing.
6. Pass `beta=0` equivalence and PBRS boundary tests.
7. Only then add a live LLM provider.
8. Only after all unit/smoke tests pass, run experimental conditions.

Stop after each stage and report:

- files changed;
- exact tests run and their results;
- relevant log samples;
- unresolved assumptions;
- `git diff --stat`;
- `git status --short`.

Do not commit, start long experiments, or modify the frozen repository unless the user explicitly requests it.

---

# Stage 0 - Audit before editing

## 0.1 Repository and process safety

From `~/selma/epymarl`, inspect without changing anything:

```bash
pwd
git status --short
git rev-parse --short HEAD
python --version
pip show lbforaging gymnasium
ps aux | grep "[s]rc/main.py"
```

If the working tree is dirty, preserve all user changes. Do not overwrite or revert them.

## 0.2 Inspect existing SELMA instrumentation

Search before designing new code:

```bash
rg -n "failed_load|necessary_cooperative|single_agent_reward|multi_agent_reward|selma_log_csv|field|player.*level|episode_limit|truncated|terminated" src tests
```

Document:

1. Where pre-step field, agent positions, and levels are captured.
2. Where post-step field is captured.
3. Whether the chosen action array is retained before LBF can replace invalid actions with `NONE`.
4. The exact existing definition of `failed_load_events`.
5. Whether `failed_load_events` already requires:
   - at least one LOAD action;
   - an adjacent food before the step;
   - the same food remaining after the step.
6. The existing definition and location of `necessary_cooperative`.
7. The exact point where the scalar common reward is inserted into the episode batch.

Do not rebuild an existing counter under a new name. The expected small extension is a coalition-necessity/level-deficit tag on the already validated adjacent failed-LOAD event.

## 0.3 Mandatory terminal-mask probe

Do not assume that a time-limit episode is bootstrapped.

The public EPyMARL runner collapses `terminated` and `truncated` to stop the loop, then stores:

```python
"terminated": [(terminated != env_info.get("episode_limit", False),)]
```

The public Gymma wrapper does not itself visibly add `env_info["episode_limit"]`. The local fork and installed LBF version must therefore be tested directly.

Create a minimal diagnostic test that records, on the final transition:

- raw environment `terminated`;
- raw environment `truncated`;
- `env_info`;
- the boolean stored in `batch["terminated"]`;
- whether the QMIX target multiplies the next-state value by zero or one.

Run two controlled episode endings:

1. A true task terminal where all food is collected before step 50.
2. A time-limit ending where food remains at step 50.

Record the verdict in a test fixture or short markdown note:

| Ending | Raw terminated | Raw truncated | Stored terminated | QMIX bootstraps? |
|---|---:|---:|---:|---:|
| All food collected |  |  |  |  |
| Step limit with food remaining |  |  |  |  |

### Boundary rule derived from the probe

Use the learner's actual stored bootstrap semantics:

```python
if stored_terminated:
    phi_next_effective = 0.0
else:
    phi_next_effective = phi(next_state, next_guidance)
```

Consequences:

- If the local runner stores `terminated=0` at a time limit, use the real next-state potential and allow QMIX to bootstrap.
- If it stores `terminated=1`, use zero next potential to remain consistent with the current learner and the completed baselines.
- Do not silently "fix" the runner's truncation handling during SELMA integration. Correcting it changes the learning algorithm and requires rerunning every baseline and every comparison condition.
- Log both the semantic cause (`true_terminal` or `time_limit`) and the stored bootstrap mask so the thesis can state exactly what happened.

## Stage 0 exit criteria

- Existing counter semantics are documented from actual code.
- The replay insertion point is identified.
- The two terminal cases have a recorded, passing diagnostic.
- No source files have been modified except an optional isolated diagnostic test/note.

Stop and report before Stage A.

---

# Stage A - Extend the counter, not the reward

Goal: expose coordination-relevant failed LOAD information online, per food and per agent, without changing actions, rewards, termination, observations, or replay contents.

## A.1 Event semantics

Use food coordinate within an episode as the stable food identity:

```text
food_id = (episode_id, row, col)
```

For each LOAD action, preserve the pre-step action and snapshot. Classify events using the exact installed LBF mechanics.

Required classifications:

1. `invalid_load_no_adjacent_food`
   - The original selected action was LOAD.
   - There was no adjacent remaining food in the pre-step snapshot.
   - Diagnostic only; never drives the detector.

2. `adjacent_failed_load`
   - This should reuse the existing validated `failed_load_events` semantics if they already mean that LOADers were adjacent to a food which persisted after the step.

3. `coalition_required_failed_load`
   - It is an `adjacent_failed_load`.
   - The total level of agents simultaneously loading that food was less than the food level.
   - Record the exact deficit:

```text
level_deficit = food_level - sum(level of simultaneous LOADers)
```

4. `successful_load`
   - The food existed before the step and was removed after the step.
   - Record the participating LOADers and their levels.

Also record:

```text
requires_help_for_agent_i = food_level > agent_i_level
```

This per-agent tag distinguishes a lone failed LOAD that genuinely cannot succeed alone from a failure by an individually capable agent.

If the installed LBF implementation can expose ambiguous adjacency to more than one food, reproduce its exact target-selection rule. Do not silently choose a different food. Add a regression test for the observed rule.

## A.2 Online data contract

Expose one immutable step record immediately after `env.step`, through an existing wrapper accessor or a dedicated accessor such as:

```python
get_selma_step_info() -> SelmaStepInfo
```

Do not put a nested object into generic `env_info` if EPyMARL later assumes every `env_info` value is numeric and sums it into runner statistics.

Suggested logical fields:

```python
SelmaStepInfo(
    episode_step: int,
    pre_foods: tuple[FoodSnapshot, ...],
    post_foods: tuple[FoodSnapshot, ...],
    agents: tuple[AgentSnapshot, ...],
    selected_actions: tuple[int, ...],
    invalid_load_agents: tuple[int, ...],
    failed_load_events: tuple[FailedLoadEvent, ...],
    successful_load_events: tuple[SuccessfulLoadEvent, ...],
)

FailedLoadEvent(
    food_id: tuple[int, int, int],
    food_position: tuple[int, int],
    food_level: int,
    loader_ids: tuple[int, ...],
    loader_levels: tuple[int, ...],
    total_loader_level: int,
    level_deficit: int,
    coalition_required: bool,
    requires_help_by_agent: tuple[bool, ...],
)
```

Names can adapt to the existing code, but semantics must not.

## A.3 Architecture boundary

- Wrapper/instrumentation layer:
  - capture LBF-specific field, level, position, and LOAD mechanics;
  - expose immutable structured snapshots/events;
  - preserve raw individual rewards before common-reward scalarisation.
- `episode_runner`:
  - consume structured events online;
  - own detector history, event state, provider calls, potential state, and reward insertion.
- Learner/MAC:
  - remain unchanged for detector v1 and PBRS.
  - do not thread Q-values through the MAC unless a later, separately approved diagnostic requires them.

## A.4 Required tests

Add deterministic tests for:

1. Invalid LOAD with no adjacent food.
2. Lone agent adjacent to food with `agent_level < food_level`.
3. Two simultaneous LOADers whose summed level is still insufficient.
4. A sufficient coalition that successfully removes food.
5. An agent individually capable of loading its food.
6. Food identity persists across failed steps and disappears on success.
7. Existing aggregate counters remain numerically unchanged.
8. `sum(raw_individual_rewards) == raw_team_reward` under the current scalarisation.
9. With SELMA disabled, observations, actions, rewards, termination, and CSV rows are unchanged.

## Stage A exit criteria

- All classifications are mutually auditable from pre-state, actions, and post-state.
- `coalition_required_failed_load` is available online in the runner.
- Existing counter totals still match their previous validated outputs.
- No detector, provider, or shaping code changes reward.

Stop and report before Stage B.

---

# Stage B - Coordination-stall detector in shadow mode

Goal: log when SELMA would query, without calling an LLM, assigning an active subgoal, or changing reward.

## B.1 Core event

Key histories by `(agent_id, food_id)`. A raw coordination-stall candidate exists when:

```text
same food still exists
AND agent i has at least m coalition-required failed LOADs on that food
    within the last W_fail steps
AND the candidate is not latched or in cooldown
```

Initial calibration grid:

```yaml
W_fail: [5, 10, 15]
m: [2, 3]
```

Do not include contribution deficit, position entropy, action entropy, Q-gap, or reward variance in detector v1. They may be logged later as diagnostics, but they do not gate a query.

### Trigger-agent distance correction

Do not use `delta_distance <= 0` as a separate core signal. Repeated adjacent failed LOADs on the same food already supply the meaningful persistence evidence. Log trigger-agent distance only as:

```text
trigger_target_persistence / trigger_distance_delta
```

Do not describe it as independent progress toward food when the agent is already adjacent and attempting LOAD.

## B.2 Team-stall gate as a tested component

Define environmental progress using raw environment information only:

```text
positive raw team reward OR a food cell disappears
```

Candidate gate:

```text
team_stall(W_team) =
    no environmental progress in the previous W_team steps
```

Shadow these variants in the same trajectory:

```yaml
W_team: [5, 10, 15]
no_team_gate: true
```

Required gate statistics:

- fraction of eligible steps where each gate is open;
- raw candidates without the gate;
- candidates remaining with the gate;
- proportion of raw candidates suppressed by the gate;
- natural-resolution rate for suppressed and retained candidates.

Flag the gate as practically vacuous if either:

- it is open on at least 90% of eligible steps; or
- it removes fewer than 5% of raw coordination candidates.

This flag does not silently change configuration. Present the report and freeze one choice after seed 91. If the gate is dropped, document that repeated coalition-required failed LOADs are the true gate. If kept, report its measured filtering contribution.

## B.3 Deterministic minimum coalition

For a failed food, calculate additional helper level required from the current simultaneous LOADers:

```text
required_extra_level =
    food_level - sum(levels of current LOADers)
```

Enumerate feasible subsets of non-loading agents. Choose deterministically by:

1. smallest number of additional agents;
2. smallest sum of distances to valid loading cells;
3. smallest lexicographic tuple of agent IDs.

Distance should be shortest-path distance to an orthogonally adjacent, valid loading cell:

- food cells are blocked;
- grid boundaries are respected;
- other agents are treated as transient, not permanent walls;
- use a stable BFS or an already validated equivalent;
- tie-break cells lexicographically.

The recipient coalition is:

```text
current relevant LOADers + selected helper subset
```

The LLM is not allowed to change this coalition.

## B.4 "Help already inbound" shadow suppression

Implement the signal now, but compare both variants before freezing:

```text
raw_would_fire
would_fire_after_inbound_suppression
```

For each feasible helper subset with enough total level, calculate helper distance history over:

```yaml
W_inbound: 3
```

A sufficient helper subset is `inbound` only when every required moving helper has reduced its distance to a valid loading cell by at least one over the window. A helper already adjacent but repeatedly not choosing LOAD is not "inbound"; that is precisely a coordination failure which guidance may address.

Log:

- whether inbound help exists;
- inbound helper IDs;
- start and end distances;
- which otherwise valid event would be suppressed.

Do not permanently enable suppression until seed-91 shadow results show that it reduces natural-resolution false positives without collapsing trigger coverage. Validate the frozen choice once on seed 99.

## B.5 Event latch and lifecycle

Event identity:

```text
(episode_id, food_id, trigger_agent_id)
```

State machine:

```text
ARMED -> CANDIDATE -> FIRED/LATCHED -> RESOLVED -> ARMED
```

Rules:

- Fire only on the rising edge of a valid event.
- Never fire while the same event is latched.
- Resolve immediately if the target food disappears.
- Resolve if the target/coalition becomes invalid.
- Expired guidance enters cooldown; it must not cause an immediate requery while the condition stays continuously true.
- Rearm only after the condition has been false for `rearm_steps`, then becomes true again.
- Reset all event state at episode reset.
- Add a maximum query/event budget per episode as a safety guard.

Initial shadow-only values:

```yaml
rearm_steps: 3
cooldown_steps: 5
max_events_per_episode: 2
```

These are calibration parameters, not final reported values.

## B.6 Natural-resolution shadow labels

For every raw candidate, create delayed labels:

```text
resolved_within_5 = target food disappears within 5 later steps
resolved_within_10 = target food disappears within 10 later steps
```

No guidance is active in shadow mode, so these estimate would-have-resolved-anyway false positives.

If the episode ends before the full horizon and the food remains, mark the label `censored`; do not count it as resolved or unresolved.

Report natural-resolution rates for:

- raw candidates;
- team-gated candidates;
- inbound-suppressed candidates;
- final latched events.

## B.7 Shadow log schema

Every candidate row must include at least:

```text
condition
seed
t_env
episode_id
episode_step
raw_team_reward
food_removed
team_gate_variant
team_gate_open
trigger_agent_id
food_id
food_position
food_level
failed_load_count_in_window
loader_ids
loader_level_sum
level_deficit
candidate_helper_ids
recipient_agent_ids
trigger_distance_delta
inbound_help
inbound_helper_ids
raw_would_fire
would_fire_after_inbound
latched
cooldown_remaining
suppression_reason
resolved_within_5
resolved_within_10
resolution_censored
```

The ordinary baseline CSV columns must remain unchanged unless a new, separately named shadow CSV is explicitly configured.

## B.8 Shadow invariants

With `selma_mode=shadow`:

- no provider is called;
- no query count is incremented;
- no subgoal becomes active;
- `shaping_reward == 0`;
- `training_reward == raw_team_reward` exactly;
- action sequence and environment trajectory are identical to `selma_mode=off` for the same seed and deterministic setup;
- greedy evaluation does not run the detector or write shadow candidate rows.

## B.9 Calibration protocol

1. Use seed 91 to produce a compact report for all shadowed gate/window/inbound variants from the same trajectories.
2. Select:
   - `W_fail`;
   - `m`;
   - whether the team gate remains;
   - `W_team` if retained;
   - whether inbound suppression remains;
   - latch/cooldown values.
3. Prefer, in order:
   - zero semantically invalid events;
   - non-degenerate event count;
   - lower uncensored natural-resolution rate;
   - fewer final events when validity is comparable.
4. Freeze all values.
5. Run seed 99 once to validate the frozen detector.
6. Do not retune after viewing seed 99 or seeds 1-5.

Report:

- gate-open percentage;
- events per 1,000 environment steps;
- events per episode;
- zero/one/multiple-recipient event frequency;
- raw versus inbound-suppressed events;
- duplicate events suppressed by latch;
- natural-resolution at 5 and 10 steps;
- event count by episode phase;
- number of censored resolution labels.

## Stage B exit criteria

- Shadow mode is trajectory- and reward-equivalent to off mode.
- A seed-91 calibration report exists.
- One detector configuration is frozen.
- Seed 99 provides a one-time validation report.
- No live LLM call and no PBRS reward has occurred.

Stop and report before Stage C.

---

# Stage C - Mocked subgoals and dynamic PBRS

Goal: integrate one deterministic provider and one exact potential-difference reward before any live LLM use.

## C.1 Provider contract

The code, not the provider, fixes:

- event identity;
- target food;
- recipient agents;
- candidate valid loading cells;
- allowed roles;
- TTL;
- beta;
- reward formula.

Input example:

```json
{
  "event_id": "ep17-food4,6-agent0",
  "event_type": "coordination_stall",
  "target_food": {
    "food_id": [17, 4, 6],
    "position": [4, 6],
    "level": 3
  },
  "trigger_agents": [0],
  "recipient_agents": [0, 1],
  "agent_context": [
    {
      "agent_id": 0,
      "level": 1,
      "position": [4, 5],
      "allowed_roles": ["hold_and_load"],
      "candidate_loading_cells": [[4, 5]]
    },
    {
      "agent_id": 1,
      "level": 2,
      "position": [1, 1],
      "allowed_roles": ["approach_and_load"],
      "candidate_loading_cells": [[3, 6], [4, 5], [4, 7], [5, 6]]
    }
  ]
}
```

Output schema:

```json
{
  "event_id": "ep17-food4,6-agent0",
  "assignments": [
    {
      "agent_id": 0,
      "role": "hold_and_load",
      "target_food_id": [17, 4, 6],
      "target_cell": [4, 5]
    },
    {
      "agent_id": 1,
      "role": "approach_and_load",
      "target_food_id": [17, 4, 6],
      "target_cell": [3, 6]
    }
  ]
}
```

Validator requirements:

- exact event ID match;
- every precomputed recipient appears exactly once;
- no unselected agent appears;
- target food is unchanged;
- role is from that agent's allowed enum;
- target cell is from that agent's candidate list;
- loading-cell assignments are collision-safe;
- no numerical reward, potential, beta, TTL, coordinate, agent, or food may be invented.

Start with a pure deterministic heuristic provider. It must use its own RNG only if randomness is explicitly required; it must never consume the environment, Torch, NumPy training, or action-selector RNG state.

## C.2 Runner integration order

The shaping code belongs after `env.step` and before `post_transition_data["reward"]` is inserted into the batch.

Per transition:

1. Keep `raw_team_reward` unchanged.
2. Read pre-step SELMA state and `guidance_t`.
3. Execute the environment action.
4. Read post-step SELMA state and terminal cause.
5. Update detector/event state from raw environment information.
6. If an eligible event fires in training mode, obtain and validate `guidance_t_plus_1`.
7. Compute `phi_before = Phi(state_t, guidance_t, t)`.
8. Compute real `phi_after = Phi(state_t_plus_1, guidance_t_plus_1, t+1)`.
9. Apply the Stage-0 bootstrap verdict to obtain `phi_after_effective`.
10. Compute PBRS without clipping:

```text
F_t = beta * (gamma * phi_after_effective - phi_before)
training_reward = raw_team_reward + F_t
```

11. Store `training_reward` in the replay batch.
12. Log raw reward and shaping components separately.
13. Clear episode-local SELMA state only after the final transition has been calculated and stored.

Guidance activation, replacement, expiry, and removal change the time-indexed potential. Their boundary terms are part of the dynamic formula and must be logged; do not hide them by reusing one goal on both sides of a transition.

## C.3 Joint potential

For each assigned agent, calculate:

```text
d_i =
    shortest path distance to its validated loading cell
    + 1 if the target food still exists
```

The extra one represents the final LOAD action.

Normalize by a documented environment-specific upper bound `d_max_plus_load`:

```text
phi_i = -d_i / d_max_plus_load
```

Aggregate selected agents by mean so potential scale does not grow automatically with recipient count:

```text
Phi_t = mean(phi_i for selected recipient agents)
```

If no guidance is active:

```text
Phi_t = 0
```

Requirements:

- Use the same `gamma` as QMIX.
- Use only validated, deterministic state features.
- Never clip `F_t`.
- Keep raw reward available for evaluation, stall detection, reporting, and success metrics.
- Do not feed separate shaped rewards to separate agents; QMIX receives one scalar team reward.
- Log each `phi_i` as well as the joint potential.

## C.4 Terminal and time-limit handling

Do not branch only on names such as `truncated`.

Use:

```text
phi_after_effective =
    0                        if replay says no bootstrap
    real next-state Phi      if replay says bootstrap
```

For a true absorbing task terminal, this should be zero.

For a time limit, the result depends on the Stage-0 probe:

- bootstrapped time limit -> retain real next-state potential;
- non-bootstrapped time limit -> zero next potential to match the existing learner and completed baselines.

If the project later corrects EPyMARL's time-limit mask, that is a new experimental protocol and all baselines must be rerun.

## C.5 Required PBRS unit tests

1. Same gamma as QMIX config.
2. Ordinary nonterminal transition:

```text
F_t == beta * (gamma * phi_after - phi_before)
```

3. True terminal:

```text
phi_after_effective == 0
```

4. Time-limit transition:
   - expectation follows the recorded Stage-0 replay mask.
5. Guidance activation uses `Phi(state_t, guidance_t)` and `Phi(state_t+1, guidance_t+1)`.
6. Guidance replacement uses the new next-time potential.
7. Guidance expiry/removal produces the exact boundary term.
8. No-guidance transition produces zero shaping.
9. No clipping at large positive or negative potential changes.
10. Discounted telescoping identity for an artificial trajectory:

```text
sum(gamma**t * F_t / beta)
    == -Phi_0 + gamma**T * Phi_T
```

11. Mean aggregation is invariant to duplicated recipient scale in a controlled fixture.
12. Detector always uses raw reward, never `training_reward`.
13. Evaluation always uses raw reward and makes zero provider calls.

## C.6 Non-negotiable beta=0 equivalence test

Compare:

```text
A: selma_mode=off
B: selma_mode=active, provider=deterministic_mock, beta=0
```

Use:

- same seed;
- same initial model parameters;
- CPU;
- `OMP_NUM_THREADS=1`;
- `MKL_NUM_THREADS=1`;
- deterministic Torch settings where supported;
- a provider/detector RNG isolated from all training RNGs.

### Fixed-trace exact comparison

Require exact equality for:

- selected actions;
- observations and global states;
- available actions;
- raw individual rewards;
- raw team rewards;
- stored training reward tensors;
- terminated/truncated causes;
- stored terminal masks;
- episode returns;
- detector-independent CSV fields.

### Short-training exact comparison

For a small deterministic run, require exact equality at every learner update for:

- loss;
- chosen Q-values;
- target Q-values;
- model parameter tensors or stable hashes;
- optimizer state;
- final model checkpoint hash.

`training_reward` must be assigned directly from `raw_team_reward` when `beta == 0`; do not rely on adding a computed floating-point zero if that can change dtype or representation.

Any mismatch blocks live LLM integration and experimental runs. Diagnose RNG consumption, reward dtype, terminal masks, batch writes, test-mode leakage, and guidance state before proceeding.

## C.7 Logging

At minimum:

```text
condition
seed
t_env
episode_id
episode_step
raw_team_reward
training_reward
detector_checked
query_triggered
event_id
trigger_agent_ids
recipient_agent_ids
target_food_id
subgoal_id
provider_type
provider_valid
provider_fallback
guidance_transition
guidance_age
phi_agent_*
potential_before
potential_after_real
potential_after_effective
stored_terminated
terminal_cause
shaping_reward
query_tokens
query_latency
query_cost
```

## Stage C exit criteria

- Mock provider output validates deterministically.
- All PBRS formula, dynamic-boundary, and truncation-mask tests pass.
- `beta=0` fixed-trace and short-training comparisons are bit-identical.
- Greedy evaluation makes zero queries and uses raw reward only.
- A short `beta>0` smoke test shows nonzero shaping without assertions or invalid entities.

Stop and report before adding a live LLM.

---

# Stage D - Live provider and experimental controls

This stage is outside the first implementation pass. Begin only after explicit approval.

## D.1 Live provider requirements

- Same input/output schema as the deterministic mock.
- Temperature and model version frozen.
- Structured output or strict JSON validation.
- Cache by canonical event-context hash.
- No credentials in source, prompt logs, CSVs, or Sacred config.
- One bounded retry for malformed output.
- Deterministic heuristic fallback after retry, with fallback rate reported.
- Validate 50-100 saved trigger states before paying for training.
- At least 95% parse-and-ground success before full experiments.

## D.2 Main comparison conditions

1. Plain QMIX.
2. Non-selective LLM-guided PBRS baseline.
3. SELMA event-triggered LLM-guided PBRS.
4. Random-trigger control matched to SELMA's total call count and recipient-count distribution.

Keep constant across guided conditions:

- provider/model;
- prompt/schema;
- validator;
- potential function;
- beta;
- TTL;
- cooldown/refresh eligibility;
- fallback behavior.

The random-trigger control should be run after the corresponding SELMA run so its global training-step schedule can contain exactly the same number of calls and a permutation of SELMA's recipient counts. Use a separate recorded RNG seed and select only valid training refresh opportunities. Never trigger during greedy evaluation.

## D.3 Required deterministic ablation

Add:

```text
same frozen detector
+ same deterministic coalition
+ deterministic nearest-feasible loading-cell assignments
+ same PBRS
+ no LLM
```

This tests whether gains come from the obvious LBF heuristic rather than from language-model subgoal selection.

Report separately:

- performance/AUC/sample efficiency;
- environment return and success;
- queries per 1,000 steps;
- tokens, latency, and cost;
- recipient counts;
- active shaping steps;
- fallback rate;
- natural-resolution rate;
- deterministic-ablation gap.

---

# Configuration surface

Use explicit, serializable config values. Suggested names:

```yaml
selma_mode: "off"              # off | shadow | heuristic | llm | random
selma_shadow_csv: null
selma_w_fail: 10
selma_failed_load_min: 2
selma_use_team_gate: true
selma_w_team: 10
selma_use_inbound_suppression: true
selma_w_inbound: 3
selma_rearm_steps: 3
selma_cooldown_steps: 5
selma_max_events_per_episode: 2
selma_guidance_ttl: 10
selma_beta: 0.0
selma_provider: "deterministic_mock"
selma_provider_seed: 19091
selma_no_queries_in_test: true
```

Do not accept silent defaults in reported experiments. Sacred output must contain every frozen SELMA setting.

---

# Recommended file structure

Adapt names to the repository after Stage 0; do not duplicate existing modules.

```text
src/
  envs/
    selma_gymma.py or existing instrumented wrapper
  runners/
    episode_runner.py
  selma/
    types.py
    coalition.py
    detector.py
    potential.py
    providers/
      base.py
      deterministic.py
      llm.py
    validation.py
    logging.py
tests/
  test_selma_load_events.py
  test_selma_coalition.py
  test_selma_detector.py
  test_selma_potential.py
  test_selma_terminal_mask.py
  test_selma_beta_zero_equivalence.py
```

Pure logic belongs in small testable modules. Environment introspection and replay insertion remain thin integration layers.

---

# Claude Code execution instruction

Use this document as the implementation contract.

Start with Stage 0 only. Inspect the real repository and installed LBF source before proposing edits. Do not assume public-main behavior matches the local fork. Do not touch `~/selma/epymarl-exact` or the active 10M processes. Work only in `~/selma/epymarl`.

After Stage 0, show the audit and terminal-mask verdict and stop. Continue to Stage A only after approval. At every later stage, implement only that stage, run its tests, show concise evidence, and stop. Do not commit or launch long runs without explicit approval.

When code and this specification differ, preserve existing validated instrumentation and explain the smallest compatible change. Never silently change reward scalarisation, termination handling, action selection, RNG streams, baseline logging, or evaluation behavior.

---

# Technical basis

- Ng, Harada, and Russell (1999), *Policy Invariance Under Reward Transformations: Theory and Application*: potential-based form `F(s,a,s') = gamma Phi(s') - Phi(s)`.
- Devlin and Kudenko (2011), *Theoretical Considerations of Potential-Based Reward Shaping for Multi-Agent Systems*: PBRS preserves the Nash-equilibrium set under its theoretical assumptions, but can affect which equilibrium learning reaches.
- Devlin and Kudenko (2012), *Dynamic Potential-Based Reward Shaping*: time-indexed form `F(s,t,s',t') = gamma Phi(s',t') - Phi(s,t)`.
- Rashid et al. (2018), *QMIX*: centralized monotonic value factorisation with a scalar team reward.
- Public EPyMARL runner: https://github.com/uoe-agents/epymarl/blob/main/src/runners/episode_runner.py
- Public EPyMARL QMIX learner: https://github.com/uoe-agents/epymarl/blob/main/src/learners/q_learner.py
- Public EPyMARL Gymma wrapper: https://github.com/uoe-agents/epymarl/blob/main/src/envs/gymma.py
- Public LBF mechanics: https://github.com/uoe-agents/lb-foraging/blob/master/lbforaging/foraging/environment.py

Theoretical claims must be phrased cautiously: the code implements the potential-difference form, but deep function approximation, finite training, online guidance changes, and any termination-mask mismatch limit claims of empirical policy invariance.
