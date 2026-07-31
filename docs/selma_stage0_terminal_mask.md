# SELMA Stage 0 — terminal-mask probe verdict

Date: 30 July 2026
Repo: `~/selma/epymarl` @ `8a169f1`
Probe: `tests/test_selma_terminal_mask.py` (3 tests, all passing)
Env: `lbforaging:Foraging-8x8-2p-2f-coop-v3`, `time_limit=50`, lbforaging 2.0.0, gymnasium 1.3.0

## Measured result

| Ending | Raw terminated | Raw truncated | Stored terminated | QMIX bootstraps? |
|---|---:|---:|---:|---:|
| All food collected | True | False | 1 | No |
| Step limit with food remaining | True | True | 1 | No |

`env_info` never contains an `episode_limit` key in either case, so the runner's
`terminated != env_info.get("episode_limit", False)` reduces to `terminated`,
which is `True` for both endings.

## Why

1. `lbforaging.foraging.environment.step` sets
   `self._game_over = (self.field.sum() == 0 or self._max_episode_steps <= self.current_step)`
   and always returns `truncated=False`. LBF therefore reports its *own* internal
   50-step limit as `terminated`, conflating the two causes at the source.
2. `gymnasium.wrappers.TimeLimit` additionally sets `truncated=True` at step 50.
3. LBF's `_get_info()` returns `{}`; neither LBF nor `GymmaWrapper` adds
   `episode_limit` to `env_info`.
4. `src/runners/episode_runner.py:118` collapses `terminated = terminated or truncated`,
   and `:148` stores `terminated != env_info.get("episode_limit", False)` -> `True`.
5. `src/learners/q_learner.py:122` computes
   `targets = rewards + gamma * (1 - terminated) * target_max_qvals`,
   so a stored `1` multiplies the next-state value by zero.

## Boundary rule for Stage C

The local learner never bootstraps at a time limit. PBRS must match the learner
and the completed baselines:

```python
# both true_terminal and time_limit end with stored_terminated == 1
phi_after_effective = 0.0 if stored_terminated else phi(next_state, next_guidance)
```

Because the mask is 1 in both cases, `phi_after_effective` is `0.0` on the final
transition of *every* LBF episode under this configuration.

Do **not** correct the truncation handling as part of SELMA integration: that
changes the learning algorithm and invalidates the completed 2M baselines and
the in-flight 10M runs. Log `terminal_cause` (`true_terminal` when
`field.sum() == 0`, else `time_limit`) separately from the stored mask so the
thesis can state the semantics and the mask independently.
