# SELMA: Selective LLM-Guided Potential-Based Reward Shaping for Cooperative MARL

MSc research project, School of Computer Science, University of Galway.
Author: Aum Bhavesh Kanakiya. Supervisor: Dr Karl Mason.

This repository is a fork of [EPyMARL](https://github.com/uoe-agents/epymarl),
itself an extension of [PyMARL](https://github.com/oxwhirl/pymarl). Commits
dated before July 2026 belong to those projects and were inherited with the
fork. The upstream framework's own documentation is preserved in
[`README_EPYMARL.md`](README_EPYMARL.md); everything below concerns the SELMA
addition.

## What SELMA is

SELMA places a language model at a deliberately narrow point in a cooperative
multi-agent reinforcement learning loop. A fixed detector fires when the team is
repeatedly failing to complete a collection. **Code, not the model**, then
enumerates every joint plan feasible under the task's level constraints. The
model returns one plan identifier from that set, and the selection determines
only *which valid potential* a dynamic potential-based shaping term is built
from.

The model cannot generate a plan, cannot act, and cannot alter the form of the
reward, so the policy-invariance guarantee of dynamic PBRS holds whatever it
chooses. The environment is Level-Based Foraging and the learner is QMIX, both
unmodified.

## What the paper reports

**No guided condition improved on unmodified QMIX**, on either layout tested.
The contribution is not a performance result. It is a set of measurements on the
guidance pathway itself:

- A **plan-conditionality assertion**: the fraction of decisions on which at
  least two offered plans induce different shaping terms. The first
  implementation returned **0.00 %** over 20,574 cached decisions, meaning the
  model's answer changed the reward on none of them, while strict output
  validation, physical-realisability checks, an exact telescoping test, a
  reconciled reward ledger and a green test suite all passed. The corrected
  potential returns **93.27 %** on the same decisions.
- **Coverage**: guidance was active on 2.2 % of 15,000,186 training transitions.
- **Magnitude**: over the shaped transitions the median term carried 0.24 % of
  the smallest positive environment reward and the 99th percentile 27 %, with
  the large values arising from guidance switching on and off rather than from
  progress toward the assigned target.

No statistical significance is claimed at these seed counts, and nothing
generalises beyond LBF, QMIX and the tested model snapshot.

## The state the paper reports

Branch **`selma-v3`**, commit **`0c58711`**, tagged **`paper-2026-08`**. The
experiments themselves ran at **`1ac4961`**, from which `0c58711` descends
without any change to training code.

Note that `1ac4961`'s commit message mentions a time-to-live of 20. That was a
launcher default which the launch overrode. Every guided run recorded
`selma_ttl_steps: 13`, and the run configurations are the authority.

## Where things are

| path | contents |
|---|---|
| `src/selma/` | the mechanism: detector, plan enumeration, LLM interface, potential and shaping term |
| `src/selma/detector.py` | struggle detector: per-agent failed-load window, team no-progress gate, inbound suppression, cooldown, rearm, per-episode cap |
| `src/selma/guidance.py` | the potential, the shaping term, guidance lifetime and expiry |
| `src/selma/target_plans.py` | Contract B enumeration and plan identifiers |
| `src/selma/openai_provider.py` | prompt construction and the strict JSON-schema output contract |
| `src/selma/offline_selectors.py` | the random and potential-maximising control selectors |
| `tests/` | test suite for the above |
| `analysis/scripts/` | every script that computes a reported statistic or renders a figure |
| `analysis/reported/` | the exact output of those scripts for the runs in the paper |
| `docs/project/` | stage-by-stage design and acceptance records |
| `run_v3_10x10.sh` | launcher for the primary experiment |

The raw per-seed decision, guidance and evaluation logs are too large for
version control and are available from the author on request.

### Scope of the change to EPyMARL

SELMA is additive. Across the whole fork it modifies five upstream code files
(`src/config/default.yaml`, `src/envs/__init__.py`, `src/envs/gymma.py`,
`src/runners/episode_runner.py`, `src/utils/logging.py`) and deletes eleven
lines of upstream code in total. QMIX, the replay buffer, the network
architectures and the evaluation procedure are untouched. The single
integration point is the addition of the shaping term to the team reward before
it is written to the buffer.

### Known issue

`analysis/scripts/mechanism_v3.py` and `make_report.py` contain a parsing defect
in the multi-target statistic: they split a composite food identifier on a
delimiter that also appears inside its values. The cached output in
`analysis/reported/` reflects it. The figures reported in the paper are
recomputed from the recorded plan assignments and supersede that cached output.

## Branch map

The project was built in stages and the branches are bookmarks left at the end
of each one. **Branch names describe the work that produced the commit, not
necessarily the state of the code at its tip**, and several bookmarks sit on the
same commit. Only `selma-v3` is live.

| branch | tip date | status |
|---|---|---|
| `selma-v3` | 2026-08-27 | **live**. The state reported in the paper is `0c58711`, tagged `paper-2026-08`. Later commits add figure scripts and this README. |
| `selma-v3-target-allocation` | 2026-08-20 | historical. Development of the Contract B enumeration. Its tip commit is titled "V2 plan-conditional potential". |
| `v2-plan-conditioned-potential` | 2026-08-18 | historical. Points at `f7327f8`, the end of the Stage E work, together with the two branches below. |
| `audit/selma-ultracode-20260818` | 2026-08-18 | historical. Independent audit pass over the implementation. Also at `f7327f8`. |
| `stage-e-beta-recalibration-runs` | 2026-08-18 | historical. Also at `f7327f8`. |
| `stage-e-beta-recalibration-design` | 2026-08-17 | historical. Revised shaping-scale calibration design. |
| `stage-e-beta-calibration` | 2026-08-17 | historical. First shaping-scale calibration. |
| `stage-d-beta-zero-measurement` | 2026-08-17 | historical. Unshaped control measurement. |
| `stage-d-joint-plans` | 2026-08-17 | historical. Joint-plan enumeration and validation. |
| `stage-d-live-pilot` | 2026-08-17 | historical. First live provider runs. |
| `stage-d-provider-cache` | 2026-08-16 | historical. Response caching and cost accounting. |
| `stage-c-mock-pbrs` | 2026-08-16 | historical. Deterministic PBRS integration into the runner. |
| `feat/exact-load-counter` | 2026-07-30 | historical. Exact LBF load-event counter. |
| `main` | inherited | unmodified upstream EPyMARL. Contains no SELMA. |

The tags `v.1.0.0` and `v2.0.0` belong to upstream EPyMARL and are **not** SELMA
versions. The only SELMA tag is `paper-2026-08`.

## How the work proceeded

1. **Instrumentation** (July). Per-agent reward vectors, validated episode
   returns and per-step reward participation logged, so that coverage and
   shaping mass could later be measured at all.
2. **Cooperation measurement** (late July). An exact LBF load-event counter, and
   an event-based struggle detector run in **shadow mode**: firing and logging
   events while changing nothing, so the trigger could be characterised before
   it was allowed to affect a reward.
3. **Detector freeze** (early August). Configuration `wf15_m3_t5_ibon` fixed and
   held constant for every subsequent comparison.
4. **Pathway construction** (mid August). PBRS integration with a deterministic
   telescoping test, then the provider cache, the live provider contract, joint
   plan validation, the unshaped control, and shaping-scale calibration.
5. **Contract B and the paper** (late August). Target allocation across foods,
   the primary experiment, then the analysis scripts and reported statistics.

## Use of AI

Claude Code was used throughout this project, under the author's direction: to
build the SELMA module and its test suite to the author's specification, to
build the analysis and figure scripts, to build the automation around the
experimental campaign, and to assist in preparing the written report. Every
experimental number comes from the author's own training runs and is
reproducible from the recorded logs. The research direction, the design
decisions, the analysis rules, the interpretation of the results and the
decision to report the negative finding prominently are the author's, who takes
full responsibility for the content of this work.

## Licence

SELMA inherits EPyMARL's licence; see [`LICENSE`](LICENSE) and
[`NOTICE`](NOTICE). Please cite EPyMARL (Papoudakis et al., NeurIPS Datasets and
Benchmarks Track, 2021) if you use the underlying framework.
