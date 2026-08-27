# SELMA

**Selective LLM-Guided Potential-Based Reward Shaping for Cooperative Multi-Agent Reinforcement Learning.**

A language model is placed at one narrow point in a cooperative MARL loop. A
fixed detector fires when the team is repeatedly failing to complete a
collection. Code, not the model, enumerates every joint plan feasible under the
task's level constraints. The model returns one plan identifier, and that choice
determines only *which valid potential* a dynamic potential-based shaping term
is built from. The model cannot generate a plan, act, or change the form of the
reward, so the policy-invariance guarantee holds whatever it selects.

Built on [EPyMARL](https://github.com/uoe-agents/epymarl). Environment is
Level-Based Foraging, learner is QMIX, both unmodified.

MSc research project, School of Computer Science, University of Galway.
Author: Aum Bhavesh Kanakiya. Supervisor: Dr Karl Mason.

---

## Result

**No guided condition improved on unmodified QMIX**, on either layout tested.
The contribution is a set of measurements on the guidance pathway rather than a
performance gain:

| measurement | value |
|---|---|
| plan-conditionality of the first implementation | **0.00 %** over 20,574 decisions |
| plan-conditionality after correcting the potential | **93.27 %** on the same decisions |
| transitions with guidance active | **2.2 %** of 15,000,186 |
| median shaping term, against the smallest positive environment reward | **0.24 %** |
| 99th percentile | **27 %**, arising from guidance switching on and off |

The first row is the finding the rest follows from. The original potential was
computed from a quantity the model's answer did not determine, so varying the
answer changed the reward on none of the 20,574 decisions, while strict output
validation, physical-realisability checks, an exact telescoping test, a
reconciled reward ledger and a green test suite all passed.

No statistical significance is claimed at these seed counts, and nothing
generalises beyond LBF, QMIX and the tested model snapshot.

---

## Install

Python 3.10. From a clean virtual environment:

```bash
git clone https://github.com/Aum07/selma-epymarl.git
cd selma-epymarl
git checkout selma-v3
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -r selma_requirements.txt
```

`requirements-lock.txt` pins the exact environment the reported runs used
(PyTorch 2.12.1, NumPy 1.26.4, Gymnasium 1.3.0) if you need to reproduce them
bit for bit.

Guided runs call the OpenAI API:

```bash
export OPENAI_API_KEY=...
```

The baseline condition needs no key.

## Run

The primary experiment, four conditions across six seeds on
`Foraging-10x10-3p-3f-v3`:

```bash
./run_v3_10x10.sh
```

Read the script before launching. It dispatches 24 runs of 2.5M steps each and
the three guided conditions issue roughly 4,700 live API calls per run, about
$1 per run at the prices in `src/selma/pricing.py`.

Each run writes four logs to its output directory: the EPyMARL training log, a
per-step transition log carrying the guidance state and shaping term, a decision
log with one row per detector event, and a guidance log with one row per
activation. Every number in the paper comes from these after the fact, with no
additional training.

## Reproduce the reported numbers

```bash
python3 analysis/scripts/analyse_v3.py     # learning curves, AUC, per-seed values
python3 analysis/scripts/mechanism_v3.py   # detector, selection and shaping diagnostics
```

Both default to `~/selma/v3/analysis/out/v3_10x10`; pass a different output
directory as the first argument. Their exact output for the reported runs is
committed in `analysis/reported/`, so you can diff against it.

`analysis/scripts/make_report.py OUTDIR TMAX DMAX REPORTDIR` runs everything and
assembles the figures, tables and a summary in one folder.

The raw per-seed logs are too large for version control and are available from
the author on request.

---

## Layout

| path | contents |
|---|---|
| `src/selma/detector.py` | struggle detector: failed-load window, team no-progress gate, inbound suppression, cooldown, rearm, per-episode cap |
| `src/selma/target_plans.py` | feasible joint-plan enumeration and plan identifiers |
| `src/selma/openai_provider.py` | prompt construction and the strict JSON-schema output contract |
| `src/selma/offline_selectors.py` | random and potential-maximising control selectors |
| `src/selma/guidance.py` | the potential, the shaping term, guidance lifetime and expiry |
| `src/selma/` (rest) | caching, cost accounting, logging, types |
| `tests/` | test suite |
| `analysis/scripts/` | every script that computes a reported statistic or renders a figure |
| `analysis/reported/` | their exact output for the runs in the paper |
| `docs/project/` | stage-by-stage design and acceptance records |

**Scope of the change to EPyMARL.** SELMA is additive. It modifies five upstream
code files (`src/config/default.yaml`, `src/envs/__init__.py`,
`src/envs/gymma.py`, `src/runners/episode_runner.py`, `src/utils/logging.py`)
and deletes eleven lines of upstream code in total. QMIX, the replay buffer, the
network architectures and the evaluation procedure are untouched. The single
integration point is the addition of the shaping term to the team reward before
it is written to the buffer.

## Versions

The state reported in the paper is tag **`paper-2026-08`** (commit `0c58711`).
The experiments ran at `1ac4961`, from which it descends without any change to
training code.

Two notes for anyone comparing the repository against the paper. `1ac4961`'s
commit message mentions a time-to-live of 20; that was a launcher default the
launch overrode, and every guided run recorded `selma_ttl_steps: 13`. And the
paper's Availability section describes a defective multi-target parse in the
analysis scripts; it was present at `0c58711` and has since been corrected, with
`analysis/reported/` regenerated.

Branch names describe the work that produced a commit rather than the state of
the code at its tip, and several are bookmarks left on the same commit. **Only
`selma-v3` is live.** `main` is unmodified upstream EPyMARL. The remaining
branches are historical: `selma-v3-target-allocation`,
`v2-plan-conditioned-potential`, `stage-c-mock-pbrs`, `stage-d-*`, `stage-e-*`,
`audit/selma-ultracode-20260818`, `feat/exact-load-counter`.

The tags `v.1.0.0` and `v2.0.0` belong to upstream EPyMARL and are not SELMA
versions.

---

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

## Licence and attribution

This repository is a fork of [EPyMARL](https://github.com/uoe-agents/epymarl),
itself an extension of [PyMARL](https://github.com/oxwhirl/pymarl). Commits
dated before July 2026 belong to those projects and were inherited with the
fork. Upstream's documentation is preserved in
[`README_EPYMARL.md`](README_EPYMARL.md).

SELMA inherits EPyMARL's licence; see [`LICENSE`](LICENSE) and
[`NOTICE`](NOTICE). If you use the underlying framework, please cite Papoudakis
et al., *Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in
Cooperative Tasks*, NeurIPS Datasets and Benchmarks Track, 2021.
