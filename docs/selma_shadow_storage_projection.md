# SELMA Stage B — shadow CSV size and disk-space projection

Date: 30 July 2026
Repo: `~/selma/epymarl` @ `8a169f1` + uncommitted Stage B work
Script: `analysis/shadow_storage_smoke.py`
Machine-readable result: `analysis/out/shadow_storage_projection.json`
Raw measurement artifacts: `analysis/out/smoke/`

## Method

Two deliberately short runs launched from `~/selma/epymarl` on the **intended
8×8 calibration environment**, with a **non-reserved smoke seed**:

```
python3 src/main.py --config=qmix --env-config=gymma with \
  env_args.key=lbforaging:Foraging-8x8-2p-2f-coop-v3 env_args.time_limit=50 \
  seed=4242 t_max=30000 use_cuda=False save_model=False \
  test_interval=100000000 log_interval=100000000 \
  selma_mode=shadow selma_shadow_csv=analysis/out/smoke/shadow_8x8_seed4242_30000.csv

python3 src/main.py ... seed=4242 t_max=30000 ... selma_mode=off     # timing reference
```

30,000 environment steps = 1.5% of the 2M calibration target: unambiguously an
engineering measurement, not a calibration run. The script refuses seeds
{1,2,3,4,5,91,99} and refuses `--steps > 100000`. Seeds 1–5, 91 and 99 were not
run. No sampling, no dropped variants, no dropped fields: all 48 variants and
all 55 columns were written exactly as the agreed format specifies.

## Measured

| Quantity | Value |
|---|---:|
| Environment steps | 30,000 (`max_t_env` in the rows: 30,016) |
| Episodes | 600 (time_limit 50); 166 produced at least one shadow row |
| CSV rows (excl. header) | 15,420 |
| Bytes written | 5,646,714 (5.39 MiB), of which 861 B header |
| Columns | 55 |
| Bytes per row | 366.1 |
| Rows per 1,000 env steps | **514.0** |
| Rows per episode | 25.7 |
| Bytes per env step | 188.2 |
| Variants present | 48 / 48 (rows per variant 86 … 602) |

Rows by type:

| `row_type` | Rows | Share |
|---|---:|---:|
| `raw_candidate` | 10,528 | 68.3% |
| `coalesced_event` | 4,892 | 31.7% |

## Projection to 2,000,000 steps

**Central projection** (measured rate extrapolated directly):

| Quantity | Value |
|---|---:|
| Rows | 1,028,000 |
| Bytes | 376,391,061 (**0.351 GiB**) |

**Conservative upper estimate.** The smoke sample is taken at the very start of
training, where ε ≈ 1 and the policy rarely attempts LOAD. Shadow rows are
driven by *coalition-required failed LOADs*, which become far more frequent as
the policy learns to approach and LOAD food. The multiplier is therefore
grounded in measured behaviour on the same task: the growth of
`failed_load_events` per environment step between the first 30k steps and the
last 100k steps of the completed 8×8 cooperative 2M runs.

| Reference seed | early failed/step | late failed/step | ratio |
|---:|---:|---:|---:|
| 1 | 0.0282 | 0.2675 | **9.50** |
| 2 | 0.0355 | 0.2199 | 6.20 |
| 3 | 0.0296 | 0.2426 | 8.20 |
| 4 | 0.0273 | 0.1452 | 5.33 |
| 5 | 0.0244 | 0.1601 | 6.57 |

Worst seed 9.50 × 1.5 extra headroom (for variant-grid and lifecycle
interactions a 30k random-policy sample cannot exercise) = **14.24×**.

| Quantity | Conservative value |
|---|---:|
| Rows | 14,643,850 |
| Bytes | 5,361,686,960 (**4.99 GiB**) |

This is deliberately pessimistic: it applies end-of-training failure density to
the entire 2M run, whereas the real run spends most of its steps below that
density.

## Disk space

| Quantity | Value |
|---|---:|
| Filesystem | `/dev/sdd` mounted at `/` (contains `~/selma`) |
| Total | 1,007 GiB |
| Free now | **943.85 GiB** |
| Free after central projection | 943.50 GiB |
| Free after conservative projection | 938.86 GiB |
| Conservative output as a share of free space | **0.53%** |

**Verdict: storage is safe with an enormous margin, for a single seed and for
many.** Even 20 seeds at the conservative rate would consume ≈ 100 GiB, about
10% of free space. No lossy sampling, streaming gzip, partitioned output or any
other change to the agreed logging format is needed or proposed.

## Runtime with measured shadow overhead

| Quantity | `off` | `shadow` |
|---|---:|---:|
| 30,000 steps wall clock | 60.51 s | 70.87 s |
| Steps / second | 495.8 | 423.3 |
| Projected 2M steps | 1.12 h | **1.31 h** |

Measured shadow overhead: **×1.171 (+17.1%)**, i.e. about +12 minutes on a 2M
calibration run. Both figures exclude periodic greedy evaluation, which was
disabled in the smoke (`test_interval=100000000`); a real calibration run adds
its usual evaluation cost, which is identical in both modes because greedy
evaluation never reaches the detector.

## Caveats

* One smoke seed on one machine, at the start of training. The central
  projection is a lower-variance estimate of the *early* rate; the conservative
  estimate is the one to plan against.
* The 8×8 cooperative task requires both agents for every food, so
  coalition-required failures — and hence shadow rows — are structurally more
  common there than on 10×10-3p-3f. Applying these numbers to 10×10 would
  over-estimate.
* `runs.*.last_logged_t_env` in the JSON reads 50 only because `log_interval`
  was set beyond `t_max` to keep the smoke quiet; the actual step count is
  confirmed by `max_t_env_in_rows = 30016`.
