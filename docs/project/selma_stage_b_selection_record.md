# SELMA Stage B — selection record and frozen 10x10 validation protocol

**Status: STAGE B CLOSED.** Selection rule **LOCKED**; `wf15_m3_t5_ibon` **frozen**; 10x10
cross-condition sensitivity **complete** (§7–§8, corrected protocol per ERRATUM 001); held-out
seed-99 validation **complete** (§9). All five Stage B exit criteria satisfied.

Working tree carries the authorised `selma_variant_id` plumbing patch (two files;
`src/selma/` detector logic unchanged). Validation source is **base commit `92ece20` plus the
documented selected-variant plumbing patch**. Nothing committed or pushed.

**Date:** 2026-08-02.
**Evidence base:** `analysis/out/seed91_retry_20260802T112458Z/8x8/` — seed-91 QMIX,
`lbforaging:Foraging-8x8-2p-2f-coop-v3`, 2 000 046 steps, 40 003 episodes, exit code 0.
**Repository state:** HEAD `92ece20`, clean tracked tree.

---

## 1. Selection domain

### 1.1 Hard eligibility rules

A configuration is **eligible** if and only if all four hold:

| # | Rule | Effect on the 48 |
|---|---|---|
| E1 | Team-stall gate enabled (`team_gate_variant ≠ none`) | removes 12 ungated configurations |
| E2 | Inbound-helper suppression enabled (`inbound_variant = on`) | removes 24 configurations |
| E3 | All structural invariants pass | satisfied by all 48 |
| E4 | Zero invalid and zero duplicate events (`semantically_invalid_events = 0`, `semantically_invalid_candidates = 0`, `same_food_coalesced = 0`) | satisfied by all 48 |

E1 excludes configurations whose team gate is **vacuous by construction**: with no gate there is no
team-stall condition to satisfy, so the gate cannot discriminate and `gate_open_percentage` is
identically 100 %. E2 excludes configurations that do not suppress a candidate when a helper is
already inbound, which admits triggers for stalls that were already being resolved.

E3 and E4 bind on nobody in this run. They are retained as **hard gates, not filters**: they must be
re-evaluated for every future run, and a configuration failing either is excluded regardless of its
scores.

**Result: 18 eligible configurations.**

### 1.2 Normalisation domain

Min–max normalisation of coverage and natural-resolution@10 is computed **over the 18 eligible
configurations**, and over those alone.

* The **13 Pareto-efficient** configurations (efficient in coverage ↑ / res@10 ↓ within the eligible
  set) are reported **descriptively**. They characterise the trade-off frontier. They **do not**
  define the normalisation range, because restricting normalisation to the frontier would discard
  the interior configurations that the rule is meant to rank against it.
* The **complete set of 48** is **not** used for normalisation, because it contains configurations
  removed by the hard eligibility rules E1 and E2. Allowing an ineligible configuration to set an
  endpoint of the normalisation range would let a configuration that can never be selected determine
  the scores of those that can.

This choice is consequential and is recorded as such: under an all-48 normalisation basis the rule
selects `wf15_m2_t5_ibon` instead of `wf15_m3_t5_ibon`. The eligible-set basis defined here is the
locked one.

### 1.3 Normalisation constants (locked)

Derived from the 18 eligible configurations of the seed-91 8x8 run:

| Quantity | Minimum | Maximum | Range |
|---|---|---|---|
| Episode coverage (%) | 21.0709 | 25.6156 | 4.5447 |
| Natural resolution @10 (%) | 8.2449 | 10.8175 | 2.5726 |

```
coverage_norm = (coverage − 21.0709) / 4.5447
res@10_norm   = (res@10   −  8.2449) / 2.5726
```

---

## 2. Locked selection rule

```
d = sqrt( (1 − coverage_norm)^2 + res@10_norm^2 )
```

* Domain: the 18 eligible configurations (§1.1).
* Normalisation: min–max over those same 18 (§1.2–1.3).
* Rank: **ascending d** (distance from the ideal point: coverage 1, resolution 0).
* Tie-break 1: **lower events per 1 000 steps**.
* Tie-break 2: **variant name**, ascending lexicographic.

The rule is deterministic and fully reproducible from the summary JSON. **It is now frozen** and
will not be altered before or after the 10x10 validation run.

Interpretive note carried with the rule: natural-resolution@10 is a *proxy* for a spurious trigger —
the fraction of triggers whose stall cleared unaided within ten steps. The environment provides no
ground-truth label for "this agent genuinely needed help", so the second axis measures a plausible
correlate of unnecessary intervention, not a measured false-positive rate.

---

## 3. Selected development configuration

**`wf15_m3_t5_ibon`**

| Parameter | Value | Status |
|---|---|---|
| `W_fail` | 15 | selected |
| `m` | 3 | selected |
| `W_team` | 5 | selected |
| inbound suppression | on | selected |
| `w_inbound` | 3 | fixed across all 48 |
| `cooldown_steps` | 5 | fixed across all 48 |
| `rearm_steps` | 3 | fixed across all 48 |
| `max_events_per_episode` | 2 | fixed across all 48 |

Measured on the seed-91 8x8 development trajectory:

| Metric | Value |
|---|---|
| Episode coverage | 23.6007 % |
| Natural resolution @10 | 9.3128 % |
| Events per 1 000 steps | 5.3579 |
| Final events | 10 716 |
| Cap-bound episodes | 3.19 % |
| Timing early/mid/late | 65.5 / 19.1 / 15.3 % |
| Invalid / duplicate events | 0 / 0 |

Rule arithmetic:

```
coverage_norm = (23.6007 − 21.0709) / 4.5447 = 0.556656   → (1 − 0.556656) = 0.443344
res@10_norm   = ( 9.3128 −  8.2449) / 2.5726 = 0.415112
d = sqrt(0.443344² + 0.415112²) = sqrt(0.196554 + 0.172318) = 0.607348
```

No tie occurred at the winning distance; neither tie-break was exercised.

**This is a record, not an application.** `src/config/default.yaml` retains its calibration values
(`selma_rearm_steps: 3`, `selma_cooldown_steps: 5`, `selma_max_events_per_episode: 2`,
`selma_w_inbound: 3`) and is unmodified. No detector code has been changed.

---

## 4. Strength of the selection — stated honestly

The distance advantage over the runner-up `wf15_m2_t10_ibon` (d = 0.613259) is **approximately
0.0059**.

`wf15_m3_t5_ibon` is therefore described as **the deterministic balanced choice under the frozen
rule**. It is **not** claimed to be uniquely or statistically superior. Three facts support that
framing:

1. The margin (≈ 0.0059) is roughly 1 % of the winning distance, and the top four candidates fall
   within 0.022 of one another.
2. The eligible set spans only 4.5 percentage points of coverage and 2.6 points of
   natural-resolution@10 — the candidates occupy a flat region.
3. The outcome depends on the normalisation domain: an all-48 basis would select a different
   configuration (§1.2).

The rule's purpose is to make the choice **deterministic, auditable and reproducible**, replacing
discretionary judgement with a stated procedure. It is not a significance test and no confidence
claim attaches to the ranking.

---

## 5. Frozen 10x10 validation protocol

**Frozen before any 10x10 result is viewed.**

### 5.1 Procedure

1. Run the **same 48-variant factorial grid** — `W_fail ∈ {5,10,15}` × `m ∈ {2,3}` ×
   team gate ∈ {`none`, `W_team=5`, `W_team=10`, `W_team=15`} × inbound ∈ {`off`,`on`} — with the
   same fixed lifecycle parameters (`w_inbound=3`, `cooldown=5`, `rearm=3`, `max_events=2`), in
   shadow mode, seed 91, 2 000 000 steps.
2. Apply the **same hard eligibility rules** E1–E4 (§1.1) to the 10x10 results.
3. Recompute min–max normalisation over the **10x10 eligible set**, using the **same 18-candidate
   normalisation definition** — that is, the same rule for *which* configurations form the basis
   (eligible only; not the Pareto subset; not all 48). The numeric endpoints will differ because
   they are derived from 10x10 data; the *definition* is unchanged. If the 10x10 eligible set does
   not number exactly 18, that is itself a reportable finding and the count will be stated.
4. Evaluate whether `wf15_m3_t5_ibon` **remains Pareto-efficient** in the 10x10 eligible set, and
   report **its 10x10 rank under the unchanged rule**.
5. Report for every variant: coverage, natural-resolution@10, event rate, cap binding, timing
   distribution (early/mid/late) and **all structural invariants**, plus the raw-CSV ↔ summary
   reconciliation.

### 5.2 Binding constraints

* **The selected configuration will not be replaced or retuned from the 10x10 results.**
* **Any disagreement is reported as cross-environment sensitivity** — evidence about the detector's
  transferability — and never absorbed by adjusting parameters until the environments agree.
* The selection rule, its normalisation definition, the eligibility rules and the selected
  configuration are all frozen as of this document. Any post-10x10 change to any of them will be
  reported explicitly as a post-hoc change, with pre-change and post-change results both shown.

### 5.3 Cross-condition robustness validation — four properties change jointly

The 10x10 condition is **`lbforaging:Foraging-10x10-3p-3f-v3`**. Relative to the 8x8 development
condition it differs in **four** respects simultaneously, not one:

| Property | 8x8 (development) | 10x10 (validation) |
|---|---|---|
| Key | `Foraging-8x8-2p-2f-coop-v3` | `Foraging-10x10-3p-3f-v3` |
| Grid | 8 × 8 | 10 × 10 |
| Agents | 2 | 3 |
| Food | 2 | 3 |
| Cooperative requirement | `-coop-` (enforced) | **not** a `-coop-` variant |

This is a **cross-condition robustness validation, not a grid-size ablation**. The environment key
is the only *configured field* that differs, but it is **not a single experimental variable**:
changing it alters grid size, number of agents, number of food items and enforced-cooperation status
**jointly**, and those four properties cannot be separated in the result. The validation therefore
asks whether the detector's calibrated behaviour is robust when transferred to a materially
different cooperative foraging condition — not what any individual property contributes. Two
consequences follow and must be stated in any report of the validation:

* With 3 agents, the recipient-count distribution can no longer be identically 2. The 8x8 invariant
  "every event has exactly two recipients" is a property of the 2-agent condition and does **not**
  transfer; the corresponding 10x10 check is that recipient counts sum to `final_events` and lie in
  the range 2–3.
* Without the `-coop-` constraint, some food is loadable by a single agent, so `solo_load_events`
  may be non-zero and the "coalition-required failed LOAD" population is drawn from a different
  distribution. Detector firing rates are therefore **not** directly comparable to 8x8 on a
  like-for-like basis.

Consequently, any 8x8↔10x10 disagreement is **joint evidence across all four properties**. It
cannot be attributed to grid size, agent count, food count or cooperation status individually, and
no such attribution will be made. Isolating any single property would require additional conditions
holding the others fixed — for example `Foraging-10x10-2p-2f-coop-v3` to isolate grid size — which
are separate experiments and are **not** authorised here.

---

## 6. Proposed 10x10 run — specification only, NOT launched

### 6.1 Environment identity

| Field | Value |
|---|---|
| `env` | `gymma` |
| `env_args.key` | `lbforaging:Foraging-10x10-3p-3f-v3` |
| `env_args.time_limit` | 50 |
| `seed` | 91 (also `env_args.seed`, derived at `src/main.py:41`) |
| Agents / food / grid | 3 / 3 / 10 × 10 |
| Algorithm | `qmix`, `q_learner`, `use_cuda=False` |
| `t_max` | 2 000 000 |
| `selma_mode` | `shadow` |

Verified against the preserved aborted 10x10 run log (`seed91/10x10/run.log`), whose
configuration this reproduces exactly apart from output paths. That log is **external and not
committed** — it is an operational run log, retained on disk; no reported metric depends on it,
and the configuration it verifies is restated in full in the table above.

### 6.2 Exact proposed command

```bash
TS=$(date -u +%Y%m%dT%H%M%SZ)
RETRY=/home/aum/selma/epymarl/analysis/out/seed91_val10x10_${TS}/10x10
EDIR=/mnt/e/selma_shadow/seed91_val10x10_${TS}/10x10

/home/aum/selma/venv/bin/python src/main.py \
    --config=qmix --env-config=gymma \
    with \
    env_args.key="lbforaging:Foraging-10x10-3p-3f-v3" \
    env_args.time_limit=50 \
    seed=91 \
    t_max=2000000 \
    use_cuda=False \
    local_results_path="$RETRY/results" \
    selma_mode=shadow \
    selma_log_csv="$RETRY/episodes_10x10_seed91_2M_val.csv" \
    selma_shadow_csv="$EDIR/shadow_10x10_seed91_2M_val.csv" \
    selma_shadow_summary_json="$RETRY/summary_10x10_seed91_2M_val.json" \
    >> "$RETRY/run.log" 2>&1
```

Run under `tmux`, with the same passive sampler and error monitor, the same
`find_training.sh` single-instance assert, and the same exit-status marker used for the 8x8 retry.

`use_cuda=False` must be passed explicitly: `torch.cuda.is_available()` is `True` on this machine,
and `src/run.py:273` only forces the flag off when CUDA is *absent*. Omitting it would move the run
to GPU and break comparability with the 8x8 calibration.

### 6.3 Output paths

| Artefact | Path |
|---|---|
| Raw shadow CSV | `/mnt/e/selma_shadow/seed91_val10x10_<TS>/10x10/shadow_10x10_seed91_2M_val.csv` |
| — Windows path | `E:\selma_shadow\seed91_val10x10_<TS>\10x10\shadow_10x10_seed91_2M_val.csv` |
| Summary JSON | `analysis/out/seed91_val10x10_<TS>/10x10/summary_10x10_seed91_2M_val.json` |
| Episode CSV | `analysis/out/seed91_val10x10_<TS>/10x10/episodes_10x10_seed91_2M_val.csv` |
| Run log | `analysis/out/seed91_val10x10_<TS>/10x10/run.log` |
| Sampler / monitor | `analysis/out/seed91_val10x10_<TS>/10x10/{rss_samples.csv,monitor_errors.log}` |
| Exit marker / manifest | `analysis/out/seed91_val10x10_<TS>/10x10/{exit_status.txt,path_manifest.json}` |

The raw CSV goes to **E:** — the ext4 VHDX is physically on C:, and routing the large sequential
write off C: is what distinguished the successful 8x8 retry from the two aborted attempts.

### 6.4 Expected raw-CSV size

Measured basis, not a guess:

| Source | Value |
|---|---|
| Aborted 10x10 run at `t_env` = 750 218 | 961 093 632 B, **no NUL tail** → **1 281 B/step** cumulative |
| 8x8 cumulative rate at `t_env` ≈ 750 050 | 2 294 B/step |
| 8x8 cumulative rate at `t_env` = 2 000 046 | 2 126 B/step (final 4 251 670 608 B) |
| 8x8 decay factor, 750 k → 2 M | **0.927** |

Applying the 8x8 decay shape to the 10x10 measurement:

```
1 281 B/step × 0.927 × 2 000 000 ≈ 2.375 GB ≈ 2.21 GiB   (central estimate)
1 281 B/step × 1.000 × 2 000 000 ≈ 2.562 GB ≈ 2.39 GiB   (no-decay upper bound)
```

**Plan for ≈ 2.2–2.4 GiB; provision 3.5 GiB to absorb the different agent/food count.** The decay
shape is borrowed from a 2-agent cooperative condition and may not transfer to a 3-agent non-coop
one, so the provision is deliberately generous. Roughly half the 8x8 footprint.

### 6.5 Storage preflight (current, read-only)

| Volume | Free | Assessment |
|---|---|---|
| E: (raw CSV target) | **357 GiB** | ample — 3.5 GiB provision is 1 % of free space |
| C: (backs the ext4 VHDX) | **38 GiB** | adequate; no large write is directed here |
| WSL `/` | 939 GiB (sparse, backed by C:) | small artefacts only |
| RAM / swap | 6.5 GiB available, `pswpin`/`pswpout` = 0 | 8x8 used a flat ~786 MiB |

`/mnt/e/selma_shadow` exists and is writable, currently holding 4.0 GiB (the 8x8 raw CSV). E: was
verified read-write earlier with an 8 MiB write + `fsync` + SHA-256 read-back + append + delete, at
98.4 MiB/s sustained append against a requirement near 1.2 MiB/s.

To be re-checked immediately before any launch: OneDrive paused, no disk-heavy task active, single
training instance, repository clean, `dmesg` free of I/O/EXT4/OOM entries, and E:/C: free space
re-confirmed.

### 6.6 Configuration comparison — 8x8 development vs 10x10 validation

| Setting | 8x8 (completed) | 10x10 (proposed) | Same? |
|---|---|---|---|
| `env` | `gymma` | `gymma` | ✅ |
| `env_args.key` | `Foraging-8x8-2p-2f-coop-v3` | `Foraging-10x10-3p-3f-v3` | ❌ **intended — jointly alters 4 properties (§5.3)** |
| `env_args.time_limit` | 50 | 50 | ✅ |
| `seed` | 91 | 91 | ✅ |
| `t_max` | 2 000 000 | 2 000 000 | ✅ |
| `use_cuda` | False | False | ✅ |
| algorithm / learner | `qmix` / `q_learner` | `qmix` / `q_learner` | ✅ |
| `selma_mode` | `shadow` | `shadow` | ✅ |
| `selma_w_inbound` | 3 | 3 | ✅ |
| `selma_cooldown_steps` | 5 | 5 | ✅ |
| `selma_rearm_steps` | 3 | 3 | ✅ |
| `selma_max_events_per_episode` | 2 | 2 | ✅ |
| `save_model` | False | False | ✅ |
| `log_interval` / `test_interval` | 50 000 / 50 000 | 50 000 / 50 000 | ✅ |
| Detector variant grid | 48 | 48 | ✅ |
| Output paths | `seed91_retry_…` | `seed91_val10x10_…` | ❌ intended |

Every configured setting is identical except the environment key and the output paths. The
environment key is the only configured field changed, but it is **not a single experimental
variable**: it jointly alters four environmental properties (grid size, agent count, food count,
enforced-cooperation status), as set out in §5.3. All remaining epymarl parameters take their repository defaults at commit `92ece20`,
exactly as in the 8x8 run, which was verified identical to its predecessor on all 68 non-path keys.

---

## 7. Status

Selection domain defined; rule locked; development configuration recorded; 10x10 protocol frozen.
**Nothing launched.** No detector code or configuration modified. No commit, no push. The 10x10
validation run awaits explicit authorisation.


---

## 7. 10x10 validation result — ANALYSIS HALTED AT THE SELECTION STEP

**Run:** `seed91_val10x10_20260802T151245Z` · exit code 0 · final `t_env` 2 000 007 · HEAD `92ece20`.
Full report: `docs/project/selma_stage_b_10x10_validation_report.md`.

### 7.1 What completed

| Stage | Result |
|---|---|
| Integrity | ✅ 0 NUL bytes in 3 708 043 487 B; 0 malformed rows of 10 014 507; complete final record; artifacts stable and mutually consistent |
| Raw-CSV ↔ summary reconciliation | ✅ `reconciled: true`, no problems, **48/48** variants, **0** final-event and **0** recipient-distribution mismatches |
| Invariants (condition-adjusted) | ✅ 15 × 48 = 720 checks pass; recipient counts in **{2, 3}** as pre-declared, reconciling exactly with `final_events` |
| Per-variant metrics | ✅ all 48, frozen definitions unchanged |
| **Selection** | ⛔ **BLOCKED** |

### 7.2 The blocker

E4 as frozen in §1.1 includes the clause `same_food_coalesced = 0`. In 10x10 that counter is
**non-zero in all 48 variants** (range 2 473–3 493), so:

* **E1** team gate enabled: 36/48 · **E2** inbound on: 24/48 · **E3** invariants: 48/48 ·
  **E4 literal: 0/48**
* **Eligible under literal frozen E1–E4: 0** ⇒ the frozen selection rule is **inapplicable** (no
  normalisation endpoints, no ranking, no Pareto frontier).

`same_food_coalesced` counts candidates *merged into* one event when several agents qualify on the
same food at the same step (`detector.py:530-531`); it is the mechanism that **prevents** duplicate
events, not evidence of duplicates surviving. It is zero on 8x8 (2 agents) and non-zero here
(3 agents) for that reason. The clause therefore does not test what E4 is named for.

**No substitute rule was invented and no ranking was produced.** Amending E4 after observing that
the literal form excludes everything would be precisely the post-hoc change this record exists to
prevent — that decision belongs to the user. Two options are set out in the validation report §4.4.

### 7.3 The 8x8 selection is unaffected

`same_food_coalesced` is **0 for all 48 8x8 variants**, so the E4 clause excluded nobody there. The
8x8 eligible set of **18**, the locked normalisation endpoints (coverage `[21.0709, 25.6156]`,
res@10 `[8.2449, 10.8175]`) and the selection of **`wf15_m3_t5_ibon`** at d = 0.607348 are
**preserved unchanged**. The configuration has **not** been retuned or replaced.

---

## PROTOCOL ERRATUM 001 — 2026-08-02

**Type:** operationalisation correction to eligibility rule E4. **Not** detector retuning: no
detector parameter, metric formula, selection score or tie-break is changed, and no training was
rerun.

**Timing, stated plainly:** raised **after** the 10x10 run completed and its integrity,
reconciliation, invariants and per-variant metrics were computed, but **before** any 10x10
normalisation endpoint, ranking or Pareto frontier was computed. All analysis downstream of this
erratum is labelled **"corrected-protocol analysis following a documented post-run, pre-ranking
operationalisation erratum"**. It is **not** an untouched confirmatory analysis.

### E.1 Original literal outcome — preserved

E4 as originally frozen, applied literally to the 10x10 condition:

| Rule | 10x10 |
|---|---|
| E1 team gate enabled | 36/48 |
| E2 inbound suppression enabled | 24/48 |
| E3 all invariants pass | 48/48 |
| **E4 literal** (including `same_food_coalesced = 0`) | **0/48** |
| **Eligible under literal E1–E4** | **0** |

**The original literal protocol was non-applicable to the 10x10 condition.** With an empty eligible
set there are no normalisation endpoints, no ranking and no Pareto frontier. This outcome is
retained permanently here and in the machine-readable results as `literal_E4_eligible_count = 0`.

### E.2 Exact old E4 wording

> `| E4 | Zero invalid and zero duplicate events (semantically_invalid_events = 0,`
> `semantically_invalid_candidates = 0, same_food_coalesced = 0) | satisfied by all 48 |`

### E.3 Evidence for the correction

**Intended semantic purpose of E4:** *zero semantically invalid or duplicate FINAL events.*

**`same_food_coalesced` does not measure that.** `src/selma/detector.py:530-531`:

```python
if len(qualifying) > 1:
    state.bump("same_food_coalesced", len(qualifying) - 1)
```

It increments by `len(qualifying) - 1` when several agents independently qualify on the **same food
at the same step**, counting candidate records **deliberately merged into one final event**. The
suppression ledger subtracts it from the candidate pool to yield `final_events`. The pre-calibration
audit describes the same mechanism: *"food-level coalescing (`same_food_coalesced`, one
`coalesced_event` per food per rising edge with sorted trigger agent ids)"*.

A non-zero value is therefore **evidence that deduplication operated**, not that duplicates survived.

| Evidence | 8x8 | 10x10 |
|---|---|---|
| `same_food_coalesced` | 0 across all 48 | **non-zero across all 48**, range 2 473 – 3 493 |
| `semantically_invalid_events` (final) | 0 across all 48 | **0 across all 48** |
| `semantically_invalid_candidates` | 0 across all 48 | 0 across all 48 |
| `coalesced_event` rows | 525 879 | 495 011 |
| Distinct `event_id`s among them | 525 879 | 495 011 |
| **Surviving duplicate final events** | **0** (0/48 variants) | **0** (0/48 variants) |
| Raw-CSV vs summary reconciliation | exact, 48/48 | exact, 48/48 — `reconciled: true`, no problems, 0 final-event and 0 recipient-distribution mismatches |

Duplicate detection uses the **existing** event identity `event_id`
(`ep{episode}-{variant}-food{r},{c}-t{step}`, `detector.py:595-598`) over `coalesced_event` rows. No
new metric was created and event identity was not redefined.

The clause was invisible on 8x8 because with 2 agents in a `-coop-` condition no two agents ever
coalesced on one food; with 3 agents it fires constantly. The literal rule is therefore a *"no
coalescing ever occurred"* test, which a 3-agent condition will essentially always fail.

### E.4 Corrected rule — frozen

**E4-corrected**, frozen as of this erratum and before any ranking:

* `semantically_invalid_final_events == 0`; **and**
* `surviving_duplicate_final_events == 0` after coalescing — measured as duplicate `event_id`
  occurrences among `coalesced_event` rows, using the existing identity definition.

`same_food_coalesced` **remains reported as a descriptive diagnostic** and is **never** an
eligibility failure.

E1, E2 and E3 are unchanged. The selection score
`d = sqrt((1 - coverage_norm)^2 + res@10_norm^2)`, its eligible-only min–max normalisation domain,
and the tie-breaks (lower events per 1 000 steps, then variant name) are **unchanged**.

---

## 8. 10x10 corrected-protocol result — the 8x8 selection is PRESERVED

**Protocol status:** corrected-protocol analysis following PROTOCOL ERRATUM 001 (post-run,
pre-ranking operationalisation correction). Full detail:
`docs/project/selma_stage_b_10x10_validation_report.md` §9.

### 8.1 8x8 consistency check under E4-corrected — nothing changed

Eligible set **18** (unchanged) · coverage endpoints **[21.0709, 25.6156]** (unchanged) · res@10
endpoints **[8.2449, 10.8175]** (unchanged) · rank 1 **`wf15_m3_t5_ibon` at d = 0.607348**
(unchanged) · ranking order unchanged. `same_food_coalesced` is 0 for all 48 8x8 variants, so the
corrected clause excluded nobody there.

### 8.2 10x10 outcome

18/48 eligible (computed, not assumed). `wf15_m3_t5_ibon` is **eligible**, **not Pareto-efficient**,
and ranks **18 of 18 — last** at d = 1.076289 (rank 1: `wf15_m2_t10_ibon`, d = 0.540482).

Two facts qualify how that rank should be read, both recorded in the validation report:

* The res@10 axis spans only **0.09 percentage points** on 10x10 (versus 2.5726 pp on 8x8, 28.6×
  wider). Min–max normalisation stretches that onto [0, 1], so rank 18 reflects a 0.09 pp absolute
  difference on that axis.
* The selected `W_team=5` gate was open **97.68 %** of steps, with the whole eligible set spanning
  only 94.08–97.68 %. The team gate — the strongest discriminating factor on 8x8 — barely
  discriminates under this condition.

### 8.3 The selection stands

**`wf15_m3_t5_ibon` is preserved as the frozen development configuration and is NOT retuned or
replaced**, regardless of its 10x10 rank. Four environmental properties changed jointly (grid size,
agent count, food count, enforced cooperation); this is one-seed external sensitivity evidence; no
difference is attributable to any single property; and a validation condition cannot reselect a
development-set configuration. The rank-18 result is recorded as **cross-condition sensitivity
evidence**, not as a selection outcome.

---

## 9. HELD-OUT SEED-99 VALIDATION — Stage B CLOSED

**Labelled clearly: this is the one-time held-out validation required by the implementation
specification, performed after freezing on seed 91 and before Stage C.** It is not a calibration or
selection exercise. No 48-variant grid was run on seed 99, no ranking or Pareto frontier computed,
no normalisation endpoints recomputed, no alternatives inspected, no retuning, and no post-hoc pass
threshold invented.

Full report: `docs/project/selma_stage_b_seed99_heldout_validation_report.md`.

**Run:** `seed99_heldout_20260802T221211Z` · `Foraging-8x8-2p-2f-coop-v3` · seed 99 ·
`t_max` 2 000 000 · exit code 0 · final `t_env` **2 000 004** · 1 h 09 m 13 s.
**Validation source:** base commit `92ece20` **plus the documented selected-variant plumbing patch**
(`selma_variant_id`, two files, detector logic unchanged) — not a clean `92ece20` checkout.

### 9.1 Integrity and reconciliation — all pass

0 malformed rows of 262 317 · 0 NUL bytes · complete final record · only `wf15_m3_t5_ibon` in all
outputs · 0 semantically invalid final events · **0 surviving duplicate `event_id`s** (10 862
`coalesced_event` rows, 10 862 distinct) · exact final-event and recipient reconciliation ·
**15/15 lifecycle invariants pass** · provider calls / queries / active subgoals all 0 ·
`total_shaping_reward = 0.0` · `training_reward == raw_team_reward` · detector absent from greedy
evaluation (`episodes_seen` 40 020 = training episodes; 4 000 greedy episodes unobserved; no shadow
rows from test episodes).

### 9.2 Frozen-detector metrics, seed 91 → seed 99

| Metric | seed 91 | seed 99 | Δ | ratio |
|---|---|---|---|---|
| Gate-open % | 96.11 | 94.10 | −2.01 | 0.979 |
| Events / 1k steps | 5.3579 | 5.4310 | +0.0731 | 1.014 |
| Events / episode | 0.2679 | 0.2714 | +0.0035 | 1.013 |
| Coverage % | 23.6007 | 24.4178 | +0.8171 | 1.035 |
| Cap-binding % | 3.1873 | 2.7236 | −0.4636 | 0.855 |
| Raw candidates | 230 268 | 251 455 | +21 187 | 1.092 |
| Inbound-suppressed | 19 871 | 20 652 | +781 | 1.039 |
| Latch-suppressed | 177 660 | 196 004 | +18 344 | 1.103 |
| `same_food_coalesced` | 0 | 0 | 0 | — |
| Recipients (all events) | 2 | 2 | — | — |
| Nat. resolution @5 % | 4.4917 | 6.2572 | **+1.7654** | **1.393** |
| Nat. resolution @10 % | 9.3128 | 10.7214 | **+1.4086** | **1.151** |
| Censored @5 / @10 | 475 / 966 | 410 / 826 | −65 / −140 | 0.863 / 0.855 |
| Timing early/mid/late % | 65.5/19.2/15.3 | 70.2/17.1/12.7 | +4.7/−2.1/−2.6 | — |

Firing behaviour transfers closely: event rate differs by 1.4 %, coverage by 0.82 pp, recipient
structure identically 2. Natural resolution is the metric that moved most (+1.77 pp @5, +1.41 pp
@10) — the held-out seed looks modestly worse on the spurious-trigger proxy. Raw candidates rose
9.2 % while final events rose 1.4 %, the latch absorbing the difference as designed.

**This is technically valid between-seed variation, recorded as a result, not a failure.** One
held-out seed against one development seed supports no significance claim. Nothing was retuned.

### 9.3 Stage B exit criteria — all satisfied

1. Shadow/off trajectory and reward equivalence ✅ 2. Seed-91 calibration report ✅
3. One configuration frozen (`wf15_m3_t5_ibon`) ✅ 4. Seed-99 one-time validation report ✅
5. No live LLM call, no PBRS reward ✅

**STAGE B IS CLOSED.** `wf15_m3_t5_ibon` remains the frozen configuration, unchanged by both the
10x10 cross-condition result and the seed-99 held-out result.
