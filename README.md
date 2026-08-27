# SELMA - Selective LLM-Guided Potential-Based Reward Shaping for Cooperative MARL

MSc research project, School of Computer Science, University of Galway.
Author: Aum Bhavesh Kanakiya. Supervisor: Dr Karl Mason.

This repository is a fork of [EPyMARL](https://github.com/uoe-agents/epymarl).
The upstream framework's own documentation is preserved in
[`README_EPYMARL.md`](README_EPYMARL.md); everything below concerns the SELMA
addition.

## What SELMA is

SELMA places a language model at a deliberately narrow point in a cooperative
multi-agent reinforcement learning loop. A fixed detector fires when the team is
repeatedly failing to complete a collection; **code, not the model**, enumerates
every joint plan feasible under the task's level constraints; the model returns
one plan identifier from that set; and the selection determines only *which
valid potential* a dynamic potential-based shaping term is built from.

The model cannot generate a plan, cannot act, and cannot alter the form of the
reward, so the policy-invariance guarantee of dynamic PBRS holds whatever it
chooses. The environment is Level-Based Foraging and the learner is QMIX, both
unmodified.

## What the paper reports

**No guided condition improved on unmodified QMIX**, on either layout tested.
The contribution is not a performance result; it is a set of measurements on the
guidance pathway itself:

- A **plan-conditionality assertion**: the fraction of decisions on which at
  least two offered plans induce different shaping terms. The first
  implementation returned **0.00 %** over 20,574 cached decisions — the model's
  answer changed the reward on none of them — while strict output validation,
  physical-realisability checks, an exact telescoping test, a reconciled reward
  ledger and a green test suite all passed. The corrected potential returns
  **93.27 %** on the same decisions.
- **Coverage**: guidance was active on 2.2 % of 15,000,186 training transitions.
- **Magnitude**: over the shaped transitions, the median term carried 0.24 % of
  the smallest positive environment reward and the 99th percentile 27 %, with
  the large values arising from guidance switching on and off rather than from
  progress toward the assigned target.

No statistical significance is claimed at these seed counts, and nothing
generalises beyond LBF, QMIX and the tested model snapshot.

## The state the paper reports

Branch **`selma-v3`**, commit **`0c58711`**. The experiments themselves ran at
**`1ac4961`**, from which `0c58711` descends without any change to training
code.

## Where things are

| path | contents |
|---|---|
| `src/selma/` | the mechanism: detector, plan enumeration, LLM interface, potential and shaping term |
| `src/selma/detector.py` | struggle detector — per-agent failed-load window, team no-progress gate, inbound suppression, cooldown, rearm, per-episode cap |
| `src/selma/guidance.py` | the potential Φ, the shaping term, guidance lifetime and expiry |
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

### Known issue

`analysis/scripts/mechanism_v3.py` and `make_report.py` contain a parsing defect
in the multi-target statistic: they split a composite food identifier on a
delimiter that also appears inside its values. The cached output in
`analysis/reported/` reflects it. The figures reported in the paper are
recomputed from the recorded plan assignments and supersede that cached output.

## Branch map

The project was built in stages and the branches record them.

| branch | stage |
|---|---|
| `selma-v3` | **current — the state reported in the paper** (Contract B, target allocation) |
| `selma-v3-target-allocation` | development of the Contract B enumeration |
| `v2-plan-conditioned-potential` | the corrected, plan-conditioned potential |
| `stage-c-mock-pbrs` | deterministic PBRS integration into the EPyMARL runner |
| `stage-d-joint-plans` | joint-plan enumeration and validation |
| `stage-d-provider-cache` | response caching and cost accounting |
| `stage-d-live-pilot` | first live provider runs |
| `stage-d-beta-zero-measurement` | β = 0 control measurement |
| `stage-e-beta-calibration` | shaping-scale calibration |
| `stage-e-beta-recalibration-design`, `stage-e-beta-recalibration-runs` | revised calibration |
| `audit/selma-ultracode-20260818` | independent audit pass over the implementation |
| `main` | unmodified upstream EPyMARL |

The tags `v.1.0.0` and `v2.0.0` belong to upstream EPyMARL and are **not**
SELMA versions.

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
[`NOTICE`](NOTICE). Please cite EPyMARL
(Papoudakis et al., NeurIPS Datasets and Benchmarks, 2021) if you use the
underlying framework.
