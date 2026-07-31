#!/usr/bin/env python3
"""SELMA Stage B: measure shadow-CSV storage cost and project it to 2M steps.

This is an engineering measurement, not a calibration run. It launches two
deliberately short runs on the intended 8x8 calibration environment with a
non-reserved smoke seed:

  * ``selma_mode=shadow`` -- produces the shadow CSV that is measured;
  * ``selma_mode=off``    -- the timing reference for shadow overhead.

Seeds 1-5 (10M baselines), 91 (calibration) and 99 (reserved) are refused.

Nothing is sampled away and no field or variant is dropped: the projection is
of the agreed lossless logging format exactly as it stands.

Usage:
    python3 analysis/shadow_storage_smoke.py [--steps 30000] [--seed 4242]
"""
import argparse
import collections
import csv
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(HERE, "out")
SMOKE = os.path.join(OUT, "smoke")

ENV_KEY = "lbforaging:Foraging-8x8-2p-2f-coop-v3"
TIME_LIMIT = 50
TARGET_STEPS = 2_000_000
RESERVED_SEEDS = {1, 2, 3, 4, 5, 91, 99}

# Long-run growth reference: the shadow row rate is driven by
# coalition-required failed LOADs, and a random early-training policy produces
# fewer of them than a trained one. The conservative projection scales the
# measured rate by the observed within-run growth of failed_load_events on the
# same 8x8 cooperative task over a full 2M run.
REF_8X8_CSV = os.path.expanduser("~/selma/final_8x8coop_seed{seed}_2M.csv")
REF_SEEDS = [1, 2, 3, 4, 5]


def run_one(mode, steps, seed, shadow_csv, tag):
    cmd = [
        sys.executable, "src/main.py", "--config=qmix", "--env-config=gymma",
        "with",
        f"env_args.key={ENV_KEY}",
        f"env_args.time_limit={TIME_LIMIT}",
        f"seed={seed}",
        f"t_max={steps}",
        "use_cuda=False",
        "save_model=False",
        "test_interval=100000000",
        "log_interval=100000000",
        f"selma_mode={mode}",
    ]
    if shadow_csv:
        cmd.append(f"selma_shadow_csv={shadow_csv}")
    log_path = os.path.join(SMOKE, f"{tag}.log")
    env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "src"))
    t0 = time.time()
    with open(log_path, "w") as fh:
        rc = subprocess.call(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    if rc != 0:
        raise SystemExit(f"{tag} failed (rc={rc}); see {log_path}")
    env_steps = None
    with open(log_path, errors="replace") as fh:
        for line in fh:
            if "t_env:" in line and "/" in line:
                try:
                    env_steps = int(line.split("t_env:")[1].split("/")[0].strip())
                except (IndexError, ValueError):
                    pass
    return {"mode": mode, "elapsed_seconds": elapsed, "log": log_path,
            "last_logged_t_env": env_steps, "command": " ".join(cmd)}


def measure_shadow_csv(path):
    size = os.path.getsize(path)
    by_type = collections.Counter()
    by_variant = collections.Counter()
    episodes = set()
    max_t_env = 0
    n = 0
    with open(path, newline="") as fh:
        rdr = csv.DictReader(fh)
        header_cols = rdr.fieldnames
        for row in rdr:
            n += 1
            by_type[row["row_type"]] += 1
            by_variant[row["variant_id"]] += 1
            episodes.add(row["episode_id"])
            try:
                max_t_env = max(max_t_env, int(float(row["t_env"])))
            except ValueError:
                pass
    header_bytes = len(",".join(header_cols)) + 1
    return {
        "path": os.path.abspath(path),
        "bytes": size,
        "header_bytes": header_bytes,
        "n_rows": n,
        "n_columns": len(header_cols),
        "rows_by_type": dict(by_type),
        "n_variants_present": len(by_variant),
        "rows_per_variant_min": min(by_variant.values()) if by_variant else 0,
        "rows_per_variant_max": max(by_variant.values()) if by_variant else 0,
        "n_episodes_with_rows": len(episodes),
        "max_t_env_in_rows": max_t_env,
        "bytes_per_row": (size - header_bytes) / n if n else 0.0,
    }


def growth_factor_from_reference(early_cut=30_000, late_cut=1_900_000):
    """Ratio of late-run to early-run failed_load_events on the same task.

    Used only to build the *conservative upper* projection; the central
    projection uses the measured smoke rate directly.

    The 8x8 reference CSVs predate the header fix (8a169f1) and are headerless,
    so columns are read positionally against the documented schema:
    ``t_env, test_mode, ep_length, team_return, multi_agent_reward_steps,
    single_agent_reward_steps, agent0..agent{n-1}_return, successful,
    solo, multi_load, necessary, failed_load_events, failed_load_attempts``.
    ``failed_load_events`` is therefore the second-to-last field.
    """
    ratios = []
    detail = {}
    for seed in REF_SEEDS:
        path = REF_8X8_CSV.format(seed=seed)
        if not os.path.exists(path):
            continue
        early_f = early_s = late_f = late_s = 0.0
        with open(path, newline="") as fh:
            for fields in csv.reader(fh):
                if not fields:
                    continue
                try:
                    t = float(fields[0])
                    test_mode = int(float(fields[1]))
                    el = float(fields[2])
                    fl = float(fields[-2])
                except ValueError:
                    continue  # header row, if present
                if test_mode:
                    continue
                if t <= early_cut:
                    early_f += fl
                    early_s += el
                if t > late_cut:
                    late_f += fl
                    late_s += el
        if early_f > 0 and early_s > 0 and late_s > 0:
            r = (late_f / late_s) / (early_f / early_s)
            ratios.append(r)
            detail[str(seed)] = {
                "early_failed_per_step": early_f / early_s,
                "late_failed_per_step": late_f / late_s,
                "ratio": r,
            }
    if not ratios:
        return None, {}
    return max(ratios), detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--skip-off", action="store_true")
    args = ap.parse_args()

    if args.seed in RESERVED_SEEDS:
        raise SystemExit(f"seed {args.seed} is reserved; pick a smoke seed")
    if args.steps > 100_000:
        raise SystemExit("this is a smoke measurement, not a calibration run")

    os.makedirs(SMOKE, exist_ok=True)
    shadow_csv = os.path.join(SMOKE, f"shadow_8x8_seed{args.seed}_{args.steps}.csv")
    if os.path.exists(shadow_csv):
        os.remove(shadow_csv)

    runs = {"shadow": run_one("shadow", args.steps, args.seed, shadow_csv, "shadow")}
    if not args.skip_off:
        runs["off"] = run_one("off", args.steps, args.seed, "", "off")

    csv_stats = measure_shadow_csv(shadow_csv)
    steps = args.steps  # t_max is the number of training env steps actually run
    rows = csv_stats["n_rows"]
    body_bytes = csv_stats["bytes"] - csv_stats["header_bytes"]

    rows_per_1k = rows / steps * 1000
    bytes_per_step = body_bytes / steps
    proj_rows = rows / steps * TARGET_STEPS
    proj_bytes = body_bytes / steps * TARGET_STEPS + csv_stats["header_bytes"]

    growth, growth_detail = growth_factor_from_reference()
    # Conservative: worst-seed growth in coalition-required failure opportunity,
    # with a further 1.5x headroom for variant-grid / lifecycle interactions
    # that a 30k-step random-policy sample cannot exercise.
    cons_mult = (growth or 3.0) * 1.5
    cons_rows = proj_rows * cons_mult
    cons_bytes = proj_bytes * cons_mult

    du = shutil.disk_usage(REPO)
    off_s = runs.get("off", {}).get("elapsed_seconds")
    sh_s = runs["shadow"]["elapsed_seconds"]
    overhead = (sh_s / off_s) if off_s else None

    result = {
        "environment": ENV_KEY,
        "time_limit": TIME_LIMIT,
        "smoke_seed": args.seed,
        "measured_env_steps": steps,
        "runs": runs,
        "shadow_csv": csv_stats,
        "rates": {
            "rows_per_1000_steps": rows_per_1k,
            "bytes_per_row": csv_stats["bytes_per_row"],
            "bytes_per_step": bytes_per_step,
            "rows_per_episode": rows / (steps / TIME_LIMIT),
        },
        "projection_2M": {
            "central_rows": proj_rows,
            "central_bytes": proj_bytes,
            "central_gib": proj_bytes / 2**30,
            "conservative_multiplier": cons_mult,
            "conservative_multiplier_basis": {
                "worst_seed_failed_load_growth_8x8_2M": growth,
                "extra_headroom": 1.5,
                "per_seed_reference": growth_detail,
            },
            "conservative_rows": cons_rows,
            "conservative_bytes": cons_bytes,
            "conservative_gib": cons_bytes / 2**30,
        },
        "disk": {
            "mount_probe_path": REPO,
            "total_bytes": du.total,
            "used_bytes": du.used,
            "free_bytes": du.free,
            "free_gib": du.free / 2**30,
            "free_after_central_gib": (du.free - proj_bytes) / 2**30,
            "free_after_conservative_gib": (du.free - cons_bytes) / 2**30,
            "conservative_output_as_fraction_of_free": cons_bytes / du.free,
        },
        "timing": {
            "shadow_seconds": sh_s,
            "off_seconds": off_s,
            "shadow_overhead_ratio": overhead,
            "shadow_steps_per_second": steps / sh_s,
            "off_steps_per_second": (steps / off_s) if off_s else None,
            "projected_2M_shadow_hours": TARGET_STEPS / (steps / sh_s) / 3600,
            "projected_2M_off_hours": (
                TARGET_STEPS / (steps / off_s) / 3600 if off_s else None
            ),
        },
    }
    with open(os.path.join(OUT, "shadow_storage_projection.json"), "w") as fh:
        json.dump(result, fh, indent=2, default=float)
    print(json.dumps(result, indent=2, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
