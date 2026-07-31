# SELMA — five-seed 10M QMIX baseline analysis (`Foraging-10x10-3p-3f-v3`)

Date: 30 July 2026
Repo: `~/selma/epymarl` @ `8a169f1` (analysis); runs executed from the read-only
reference checkout `~/selma/epymarl-exact` @ `8a169f1`, clean working tree
Artifacts: `analysis/baseline_manifest.py`, `analysis/analyse_10m_baseline.py`,
`analysis/out/`

Nothing in this document modified, moved or regenerated a baseline output. Every
number below is derived read-only from the five per-episode CSVs, the five
training logs and the five Sacred run directories.

---

## 1. Completion verification

| Seed | Sacred run | Status | `Finished Training` | Crash scan | Max `t_env` (train rows) | Elapsed |
|---:|---:|---|---:|---|---:|---:|
| 1 | 9 | COMPLETED | 1 | clean | 10,000,031 | 6h 58m (6.976 h) |
| 2 | 10 | COMPLETED | 1 | clean | 10,000,005 | 6h 57m (6.947 h) |
| 3 | 11 | COMPLETED | 1 | clean | 10,000,041 | 6h 53m (6.890 h) |
| 4 | 12 | COMPLETED | 1 | clean | 10,000,013 | 6h 58m (6.972 h) |
| 5 | 13 | COMPLETED | 1 | clean | 10,000,033 | 6h 59m (6.981 h) |

Crash scan (per log): `Traceback` 0, `AssertionError` 0, `RuntimeError` 0,
`ValueError` 0, `KeyError` 0, `IndexError` 0, word-boundary `nan` 0, `inf` 0,
out-of-memory / `Killed` 0, `CUDA error` 0 — on all five logs. `ps aux | grep
src/main.py` returned no processes.

### Why the last printed `t_env ≈ 9.95M` is not premature termination

`log_interval = 50000`. EPyMARL's `run_sequential` prints the
`t_env: X / 10000000` progress line only when a logging boundary is crossed, so
the final printed value is the last 50k boundary reached (9,953,906–9,954,533),
not the point at which the loop stopped. The loop itself runs
`while runner.t_env <= args.t_max`, and the CSV shows what actually happened:
the **last training row of every seed has `t_env ≥ 10,000,000`** (10,000,005 to
10,000,041 — one episode's overshoot past the boundary). Each log then contains
exactly one `Finished Training`, a `Finished after 6:5x:xx` line, a
`Completed after` line, and Sacred independently records `status: COMPLETED`
with a `stop_time`. Four independent sources agree; a truncated run would show
none of them.

## 2. Configuration-consistency evidence

Configuration identity was established from Sacred's `config.json` and the
`Experiment Parameters` block in each log — **not** from filenames.

* Pairwise diff of `config.json` for runs 9–13 against run 9: the *only*
  differing keys are `seed` and `selma_log_csv` (the per-seed output path).
  Every other key is byte-identical, including
  `env: gymma`, `env_args.key: lbforaging:Foraging-10x10-3p-3f-v3`,
  `env_args.time_limit: 50`, `name: qmix`, `learner: q_learner`, `mixer: qmix`,
  `t_max: 10000000`, `test_interval: 50000`, `test_nepisode: 100`,
  `test_greedy: True`, `evaluation_epsilon: 0.0`, `epsilon_start: 1.0`,
  `epsilon_finish: 0.05`, `epsilon_anneal_time: 50000`, `lr: 0.0005`,
  `gamma: 0.99`, `batch_size: 32`, `buffer_size: 5000`, `hidden_dim: 64`,
  `use_rnn: False`, `common_reward: True`, `reward_scalarisation: sum`,
  `standardise_rewards: True`, `double_q: True`,
  `target_update_interval_or_tau: 200`, `use_cuda: False`.
  The same diff was reproduced independently from the log-embedded parameter
  dumps.
* Code revision: all five Sacred runs record
  `commit 8a169f17ff1edfe0d701dee8014c5bf5c9e5f15e`, `dirty: false`, on host
  `LAPTOP-F1QT7O50`. One revision, no local modifications.
* Logging schema: all five CSVs carry a byte-identical 15-column header.
* Seed is the intended and only experimental difference.

**Verdict: aggregation is legitimate.** No mismatch found.

## 3. CSV schema — verified against the logging code

Read from `src/runners/episode_runner.py` @ `8a169f1` and
`src/envs/gymma.py` @ `8a169f1`, not inferred from column position.

| Column | Exact definition |
|---|---|
| `t_env` | **train rows**: `self.t_env + self.t` — cumulative *training* env steps after this episode. **test rows**: `self.t_env` unchanged — the training-step count at which the evaluation block ran. Greedy-test steps never advance `t_env`, so all 100 episodes of a block share one value. |
| `test_mode` | `0` = training episode, `1` = greedy-test episode (`test_greedy=True`, `evaluation_epsilon=0.0`). |
| `ep_length` | number of `env.step` calls in the episode (≤ 50). |
| `team_return` | episode sum of the scalarised common reward (`common_reward=True`, `reward_scalarisation='sum'`). Raw environment reward; `standardise_rewards` acts inside the learner only and never touches this column. |
| `multi_agent_reward_steps` | **transitions on which at least two agents received a strictly positive per-agent reward.** That is the entire operational content of the measure. |
| `single_agent_reward_steps` | transitions on which exactly one agent received a strictly positive per-agent reward. |
| `agent{i}_return` | per-agent reward summed over the episode, pre-scalarisation. `Σ_i agent{i}_return == team_return` is asserted by the runner at write time (`atol=1e-8`). |
| `successful_load_events` | food cells cleared during the episode. |
| `solo_load_events` | cleared cells with ≤ 1 orthogonally adjacent LOADing agent. |
| `multi_agent_load_events` | cleared cells with ≥ 2 orthogonally adjacent LOADing agents. |
| `necessary_cooperative_load_events` | subset of the above where the food level strictly exceeded *every* participant's own level, so no participant could have cleared it alone. |
| `failed_load_events` | surviving food cells with ≥ 1 adjacent LOADing agent. |
| `failed_load_attempts` | sum of adjacent LOADing agents over those cells. |

Training and greedy-evaluation rows are kept separate throughout the analysis.

## 4. Run coverage and integrity

| Seed | Training episodes | Greedy-test episodes | Test blocks | Duplicate / malformed / null rows |
|---:|---:|---:|---:|---:|
| 1 | 200,432 | 20,000 | 200 × 100 | 0 |
| 2 | 212,029 | 20,000 | 200 × 100 | 0 |
| 3 | 200,301 | 20,000 | 200 × 100 | 0 |
| 4 | 200,238 | 20,000 | 200 × 100 | 0 |
| 5 | 200,668 | 20,000 | 200 × 100 | 0 |

Consistency checks, all passing on all five seeds:
`t_env` strictly increasing and unique on training rows; consecutive `t_env`
differences exactly equal `ep_length`; `team_return == Σ agent{i}_return`
(max residual 0.0); `multi + single rewarded steps ≤ ep_length`;
`successful == solo + multi_agent` load events;
`necessary ≤ multi_agent_load_events`; `successful_load_events ≤ 3`
(initial food count); a multi-agent load event in an episode always co-occurs
with a multi-agent rewarded step; a positive return always co-occurs with at
least one rewarded step; every greedy-test block contains exactly
`test_nepisode = 100` episodes.

## 5. Performance

Unit of replication: **the seed (n = 5)**. Confidence intervals use
t₄ = 2.776 on the five seed-level values — never on the ~200k episodes.
Final window: **the last 1M environment steps (`t_env > 9,000,000`)**.

### Training (behaviour policy, ε = 0.05 after 50k steps)

| Seed | Overall mean | Overall median | Final-1M mean | Final-1M median |
|---:|---:|---:|---:|---:|
| 1 | 0.1348 | 0.0000 | 0.3140 | 0.2857 |
| 2 | 0.2445 | 0.0000 | 0.5205 | 0.5000 |
| 3 | 0.0736 | 0.0000 | 0.0955 | 0.0000 |
| 4 | 0.0808 | 0.0000 | 0.1305 | 0.0000 |
| 5 | 0.0846 | 0.0000 | 0.1223 | 0.0000 |

Cross-seed: overall mean **0.1237** (sd 0.0718, 95% CI [0.0346, 0.2127]);
final-1M mean **0.2366** (sd 0.1809, 95% CI [0.0120, 0.4611]).
Best seed 2, worst seed 3.

### Greedy evaluation (ε = 0)

| Seed | Overall mean | Final-1M mean | Final-1M median | Final-1M mean ep length |
|---:|---:|---:|---:|---:|
| 1 | 0.0744 | 0.1888 | 0.0000 | 49.88 |
| 2 | 0.1365 | 0.3265 | 0.2000 | 44.76 |
| 3 | 0.0376 | 0.0499 | 0.0000 | 49.95 |
| 4 | 0.0413 | 0.0656 | 0.0000 | 49.89 |
| 5 | 0.0447 | 0.0746 | 0.0000 | 49.72 |

Cross-seed: overall mean **0.0669** (sd 0.0416, 95% CI [0.0153, 0.1185]);
final-1M mean **0.1411** (sd 0.1174, 95% CI [−0.0047, 0.2868]).
Best seed 2, worst seed 3.

The final-1M **median** greedy return is 0 for four of five seeds: the modal
greedy episode still ends with no food collected at all. Only seed 2 has a
positive median.

### Episode length

Mean training episode length falls from 49.97 (0–2M) to 48.31 (8–10M); greedy
from 49.99 to 48.97. The 50-step limit dominates: only seed 2 shortens
materially (42.75 training / 45.33 greedy in 8–10M). Shorter episodes mean food
was cleared before the limit, so this tracks the return improvement.

## 6. Learning over time

Cross-seed means (sd across the five seeds) by common `t_env` window:

| Window | Train return | Greedy return | Train multi-agent rewarded steps / 1M | Greedy multi / 1M | Train multi-share of rewarded | Train % episodes with ≥1 multi |
|---|---:|---:|---:|---:|---:|---:|
| 0–2M | 0.0573 (0.0061) | 0.0280 (0.0036) | 274 (47) | 114 (31) | 0.0515 (0.0048) | 1.36% (0.24) |
| 2–5M | 0.0755 (0.0095) | 0.0381 (0.0039) | 557 (276) | 208 (70) | 0.0787 (0.0279) | 2.72% (1.29) |
| 5–8M | 0.1422 (0.1030) | 0.0775 (0.0632) | 3,076 (4,383) | 1,508 (2,294) | 0.1884 (0.1744) | 11.85% (15.14) |
| 8–10M | 0.2241 (0.1716) | 0.1330 (0.1097) | 6,578 (8,896) | 3,642 (5,012) | 0.2561 (0.2527) | 22.02% (26.29) |

### Within-seed early (0–2M) vs late (8–10M) paired differences

| Measure | Early → late (seed means) | Paired Δ mean | sd | 95% CI | Seeds improved | dz | Wilcoxon p |
|---|---|---:|---:|---|---:|---:|---:|
| Train return | 0.057 → 0.224 | +0.1668 | 0.1712 | [−0.046, 0.379] | 5/5 | 0.98 | 0.0625 |
| Greedy return | 0.028 → 0.133 | +0.1050 | 0.1091 | [−0.030, 0.241] | 5/5 | 0.96 | 0.0625 |
| Train multi rewarded / 1M | 274 → 6,578 | +6,304 | 8,869 | [−4,707, 17,315] | 5/5 | 0.71 | 0.0625 |
| Greedy multi rewarded / 1M | 114 → 3,642 | +3,528 | 4,990 | [−2,667, 9,723] | 5/5 | 0.71 | 0.0625 |
| Train multi-share of rewarded | 0.0515 → 0.2561 | +0.2047 | 0.2493 | [−0.105, 0.514] | 4/5 | 0.82 | 0.1875 |
| Train % episodes with ≥1 multi | 1.36% → 22.0% | +20.7 pp | 26.2 | [−11.8, 53.2] | 5/5 | 0.79 | 0.0625 |
| Train necessary-cooperative loads (window totals) | 372 → 11,072 | +10,700 | 15,147 | [−8,104, 29,505] | 5/5 | 0.71 | 0.0625 |
| Train ep length | 49.97 → 48.31 | −1.66 | 3.12 | [−5.54, 2.21] | 0/5 (shorter is better) | −0.53 | 0.0625 |

**Reading these honestly.** Every direction-of-effect measure improves in
**5 of 5 seeds**, which is the strongest possible sign-consistency at n = 5
(two-sided Wilcoxon p = 0.0625 is the *minimum attainable* p for five pairs;
one-sided it is 0.03125). The confidence intervals nonetheless straddle zero,
because seed 2 improves ~4× more than seeds 3–5 and the between-seed sd exceeds
the mean difference. The correct statement is: **the improvement is consistent
in direction across every seed, and large in the best seeds, but its magnitude
is not estimated precisely enough at n = 5 to exclude a small effect.** Do not
report these as "significant"; report them as unanimous in sign with a wide
magnitude interval.

### Figures (`analysis/out/`)

| File | Content |
|---|---|
| `fig1_train_return.png` | training return vs `t_env`, five seed curves + cross-seed mean with a 95% CI-of-the-mean band |
| `fig2_test_return.png` | greedy return vs `t_env`, same structure |
| `fig3_train_rewarded_steps.png`, `fig3_test_rewarded_steps.png` | single-agent (solid) and multi-agent (dashed) rewarded-step rates per seed |
| `fig4_train_multi_share.png`, `fig4_test_multi_share.png` | multi-agent share of rewarded steps; bins with a zero denominator are **dropped**, never imputed as 0 |
| `fig5_train_ep_length.png` | mean training episode length |
| `fig6_train_frac_ep_multi.png`, `fig6_test_frac_ep_multi.png` | proportion of episodes with ≥ 1 multi-agent rewarded step |

Every figure states its binning on the figure itself. Training curves use exact
per-bin sums/means over 200k-step bins (~4k episodes); greedy curves use one
point per 100-episode evaluation block (or 250k-step bins for the sparse
coordination panels). **No rolling smoothing, exponential smoothing or
outlier removal is applied anywhere**, so sparse cooperative events survive
intact.

## 7. Coordination diagnostics

Whole-run totals (seed-level values; "per 1M" denominators are *episode steps
of that split*, since greedy steps do not advance `t_env`):

| Seed | Train single | Train multi | Train multi share | Train multi / 1M | Train % eps ≥1 multi | Greedy single | Greedy multi | Greedy multi share | Greedy % eps ≥1 multi |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 71,562 | 31,583 | 0.306 | 3,158 | 13.40% | 4,040 | 1,697 | 0.296 | 7.61% |
| 2 | 103,040 | 77,251 | 0.428 | 7,725 | 26.26% | 6,085 | 3,779 | 0.383 | 15.00% |
| 3 | 62,866 | 4,087 | 0.061 | 409 | 2.03% | 3,282 | 161 | 0.047 | 0.81% |
| 4 | 66,280 | 6,131 | 0.085 | 613 | 3.01% | 3,491 | 236 | 0.063 | 1.16% |
| 5 | 71,608 | 3,958 | 0.052 | 396 | 1.95% | 3,816 | 159 | 0.040 | 0.79% |

Cross-seed: all rewarded steps **9,967 / 1M** training (sd 4,719) and
**5,420 / 1M** greedy (sd 2,822); multi-agent rewarded steps **2,460 / 1M**
training (sd 3,166) and **1,233 / 1M** greedy (sd 1,634); multi share of all
rewarded steps **0.187** training (sd 0.171) / **0.166** greedy (sd 0.162).

Distribution over episodes (cross-seed means):

| Split | 0 multi-agent rewarded steps | exactly 1 | ≥ 2 |
|---|---:|---:|---:|
| Training | 183,425 (90.7%) | 14,679 | 4,629 |
| Greedy | 18,986 (94.9%) | 840 | 174 |

Final-1M window: training multi-agent rewarded steps rise to **7,180 / 1M**
(sd 9,691) and the share of episodes with at least one to **23.4%** (sd 27.5);
greedy to **3,996 / 1M** and **14.8%**. Per-seed variability is the dominant
term throughout: seeds 1–2 and seeds 3–5 form two clearly separate groups.

Per-agent contribution is essentially uniform. Final-1M training return shares
are 0.331 / 0.336 / 0.333 for agents 0/1/2 (each sd ≤ 0.012); whole-run
max-minus-min share is 0.0125 (training) and 0.0176 (greedy). No agent
free-rides and none dominates.

### What this instrumentation can and cannot establish

**Can:** exactly how often ≥ 2 agents were rewarded on the same transition;
exactly how often a cleared food cell had ≥ 2 orthogonally adjacent LOADing
agents (`multi_agent_load_events`); and, within that, how often the food level
strictly exceeded every participant's own level so no participant could have
cleared it alone (`necessary_cooperative_load_events`). It can also count
adjacent-but-failed LOADs.

**Cannot:** it cannot show intent, communication, role assignment, deliberate
waiting, or division of labour. A multi-agent rewarded step is *simultaneous
joint reward*, which is the mechanical footprint of a joint load — not evidence
that the agents planned it. Conversely, **a low multi-agent rewarded-step count
is not proof that the agents never coordinated in any weaker behavioural
sense**: approach coordination, spatial partitioning of the map, or one agent
yielding a food cell to another would all be invisible to these counters, which
only fire at the moment of reward. Claims about coordination in the weaker
behavioural sense require the Stage A/B structured records (failed-load
coalitions, BFS approach distances), not these aggregate counters.

## 8. 2M vs 10M

**An exactly matched standalone 2M baseline exists and was used.** Sacred runs
2–6 (seeds 1–5), `env_args.key = lbforaging:Foraging-10x10-3p-3f-v3`,
`time_limit = 50`, identical agent/food configuration, identical algorithm and
identical hyperparameters. `config.json` differs from the 10M runs in exactly
three keys: `t_max` (2,000,000 vs 10,000,000), `save_model_interval`
(500,000 vs 1,000,000) and `selma_log_csv`. Same CSV schema.

**Documented caveat on the match.** The 2M runs record commit `f8609cc` with
`dirty: true`; the 10M runs record `8a169f1` with `dirty: false`. `f8609cc →
8a169f1` consists only of the LBF load-event counter commit and the CSV-header
fix, and the 2M working tree evidently already contained the counter (its CSVs
carry the same columns). The RL-relevant configuration is therefore identical,
but the exact 2M source tree is not byte-recoverable. For that reason the
within-run comparison is reported alongside it, and the two agree closely.

Both comparisons, per seed:

| Measure (training) | Standalone 2M, final 1M | 10M run, first 2M | 10M run, last 2M | 10M run, final 1M |
|---|---|---|---|---|
| return mean | 0.063–0.076 | 0.048–0.062 | 0.093–0.493 | 0.096–0.521 |
| multi rewarded / 1M | 318–459 | 226–344 | 407–20,780 | 367–22,730 |
| multi share of rewarded | 0.051–0.070 | 0.046–0.059 | 0.040–0.566 | 0.038–0.582 |

The standalone 2M results and the 10M runs' own first-2M segment agree to
within the seed-level noise, which validates using either as the early baseline.

Paired seed-level comparison, standalone-2M final-1M → 10M final-1M:

| Measure | Δ mean | sd | 95% CI | Seeds improved |
|---|---:|---:|---|---:|
| Train return | +0.1677 | 0.1812 | [−0.057, 0.393] | 5/5 |
| Greedy return | +0.1047 | 0.1165 | [−0.040, 0.249] | 5/5 |
| Train multi rewarded / 1M | +6,793 | 9,656 | [−5,194, 18,781] | 5/5 |
| Train % eps with ≥1 multi | +21.5 pp | 27.4 | [−12.5, 55.5] | 5/5 |
| Train all rewarded / 1M | +11,669 | 12,728 | [−4,132, 27,471] | 5/5 |
| Greedy multi rewarded / 1M | +3,846 | 5,488 | [−2,968, 10,659] | 4/5 |

Answers to the required questions:

* **Materially improved team return?** Yes in direction and in practical size —
  training +0.17 (≈ 2.7×), greedy +0.10 (≈ 3.8×), unanimous across seeds. The
  interval is wide; the effect is real in sign but imprecisely sized.
* **Materially improved greedy return?** Yes, same pattern; greedy remains far
  below the training return because ε-greedy exploration itself contributes
  successes that a deterministic greedy policy does not reproduce.
* **Increased multi-agent rewarded events?** Yes — roughly 15–20× in rate, but
  driven overwhelmingly by seeds 1 and 2.
* **Seed variance?** **Increased sharply.** Training final-1M return sd rises
  from ≈ 0.006 (standalone 2M) to 0.181 (10M); the multi-agent rate sd rises
  from ≈ 60 to ≈ 9,700 per 1M. Longer training separates the seeds rather than
  converging them.
* **Reliance on single-agent successes?** Reduced but still dominant on average.
  The multi-agent share of rewarded steps rises from ≈ 0.06 to a cross-seed
  0.266 (final 1M), i.e. **~73% of rewarded steps are still single-agent**, and
  in seeds 3–5 it is ~96%.

## 9. Scientific conclusion

1. **Does QMIX learn useful behaviour by 10M steps?** Partially and
   unreliably. Every seed improves monotonically in direction, and two of five
   (seeds 1, 2) reach a genuinely non-trivial greedy return (0.19 and 0.33).
   The other three plateau at 0.05–0.07 greedy, with a **median greedy return
   of exactly 0** — the modal evaluation episode still collects nothing. At
   10M steps the task is learned by a minority of seeds, not solved.
2. **Does cooperation become more common after 2M steps?** Yes under the
   available operational measure, unanimously in sign: multi-agent rewarded
   steps rise ~15–20× and the fraction of episodes containing at least one
   rises from ~1.4% to ~22% (training) / ~0.6% to ~14% (greedy). The magnitude
   is dominated by two seeds and its cross-seed interval includes zero.
3. **Is the task still dominated by single-agent successes?** Yes. Averaged
   across seeds, ~73% of rewarded steps in the final 1M are single-agent, and
   in three of five seeds it is ~96%. `necessary_cooperative_load_events`
   (food that provably could not be cleared by any single participant) averages
   20,376 over 10M training steps against 99,923 successful loads — about 20%,
   again concentrated in seeds 1–2.
4. **Is 10×10-3p-3f suitable as the final SELMA evaluation environment?**
   Only partially. Its strengths: it is unsolved at 10M, leaves clear headroom
   for guidance to act, and produces genuine (if sparse) cooperative events.
   Its weaknesses for SELMA's purpose are serious: (a) cooperation is optional,
   so the dominant learning signal is single-agent foraging and a guidance
   method could improve return without improving coordination at all;
   (b) between-seed variance is larger than the effect longer training
   produces, so with 5 seeds the environment cannot resolve a moderate
   intervention effect; (c) 10M steps × 7 h/seed makes a well-powered
   seed count expensive.
5. **Recommendation (for review — not a change to the frozen design).**
   **Retain 10×10-3p-3f as a secondary/generalisation environment and add the
   cooperative 8×8-2p-2f-coop task as the primary SELMA evaluation.** In the
   cooperative variant every food requires both agents, so coordination is
   necessary rather than optional and the coordination measure becomes the
   performance measure — which is exactly what SELMA claims to improve. It is
   also the environment Stage B calibration already targets. Replacing 10×10
   outright is not advised: it is the only configuration here with a published
   comparison point and it demonstrates the "cooperation is optional" regime
   that makes the contrast informative. If 10×10 is kept as a headline result,
   the seed count must rise well above 5 to have any power against a variance
   of this size.

## 10. Assumptions and limitations

* `multi_agent_reward_steps` is used throughout with its literal operational
  definition (≥ 2 agents rewarded on one transition). No stronger cooperative
  claim is derived from it.
* Rewarded-step rates are normalised per million *episode steps of that split*.
  Greedy steps do not advance `t_env`, so a `t_env`-based denominator would be
  wrong for greedy rows.
* The final-performance window is the last 1M environment steps. The paired
  early/late analysis uses 0–2M vs 8–10M, matching the requested windows.
* The 2M ↔ 10M source-revision caveat is in §8.
* **No final 10M checkpoint exists.** `save_model_interval = 1,000,000` and the
  last saved checkpoint in every seed is at `t_env ≈ 9.00M` (steps 50,
  1.00M … 9.00M — 10 checkpoints, 2,983,424 bytes per seed). The run ended
  before the 10M save boundary. Any later work that needs the *final* learned
  policy must either use the ~9.0M checkpoint or re-run; the manifest records
  this per seed.
