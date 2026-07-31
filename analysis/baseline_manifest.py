#!/usr/bin/env python3
"""SELMA: build a reproducible manifest of the five 10M QMIX baseline runs.

Read-only. Never copies, moves or rewrites the baseline outputs; it only
stats/hashes them and records where they live.

Outputs (under analysis/out/):
  baseline_manifest_10M.json   full manifest
  baseline_manifest_10M.csv    flat one-row-per-seed view

Small metadata files (sacred config.json / run.json) are hashed with sha256.
The multi-GB-class artefacts (per-episode CSVs, ~11-14 MB each here, and the
training logs) are *not* hashed by default -- pass --hash-large to force it.

Usage:
    python3 analysis/baseline_manifest.py [--hash-large]
"""
import argparse
import csv
import datetime as _dt
import glob
import hashlib
import json
import os
import re
import sys

HOME = os.path.expanduser("~")
SELMA = os.path.join(HOME, "selma")
# The 10M baselines were launched from the read-only reference checkout.
RUN_CWD = os.path.join(SELMA, "epymarl-exact")
SACRED_ROOT = os.path.join(
    RUN_CWD, "results", "sacred", "qmix", "lbforaging:Foraging-10x10-3p-3f-v3"
)
MODELS_ROOT = os.path.join(RUN_CWD, "results", "models")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

SEEDS = [1, 2, 3, 4, 5]
LOG_TMPL = os.path.join(SELMA, "long_10x10_seed{seed}_10M.log")
CSV_TMPL = os.path.join(SELMA, "long_10x10_seed{seed}_10M.csv")

# Config keys that are *allowed* to differ between the five runs.
EXPECTED_DIFFERING_KEYS = {"seed", "selma_log_csv"}


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def stat_entry(path, do_hash=False):
    if not os.path.exists(path):
        return {"path": path, "exists": False}
    st = os.stat(path)
    e = {
        "path": os.path.abspath(path),
        "exists": True,
        "size_bytes": st.st_size,
        "mtime_utc": _dt.datetime.utcfromtimestamp(st.st_mtime).isoformat() + "Z",
    }
    if do_hash:
        e["sha256"] = sha256(path)
    return e


def scan_log(path):
    """Completion / failure evidence straight out of the training log."""
    finished = 0
    completed_after = None
    finished_after = None
    last_t_env = None
    t_max = None
    bad = {
        "traceback": 0,
        "assertion_error": 0,
        "runtime_error": 0,
        "value_error": 0,
        "key_error": 0,
        "index_error": 0,
        "nan": 0,
        "inf": 0,
        "oom": 0,
        "cuda_error": 0,
    }
    re_tenv = re.compile(r"t_env:\s*(\d+)\s*/\s*(\d+)")
    with open(path, errors="replace") as fh:
        for line in fh:
            if "Finished Training" in line:
                finished += 1
            if "Completed after" in line:
                completed_after = line.split("Completed after")[-1].strip()
            if "Finished after" in line:
                finished_after = line.split("Finished after")[-1].strip()
            m = re_tenv.search(line)
            if m:
                last_t_env = int(m.group(1))
                t_max = int(m.group(2))
            low = line.lower()
            if "traceback" in low:
                bad["traceback"] += 1
            if "assertionerror" in low:
                bad["assertion_error"] += 1
            if "runtimeerror" in low:
                bad["runtime_error"] += 1
            if "valueerror" in low:
                bad["value_error"] += 1
            if "keyerror" in low:
                bad["key_error"] += 1
            if "indexerror" in low:
                bad["index_error"] += 1
            if re.search(r"\bnan\b", low):
                bad["nan"] += 1
            if re.search(r"\binf\b", low):
                bad["inf"] += 1
            if "out of memory" in low or "memoryerror" in low or "killed" in low:
                bad["oom"] += 1
            if "cuda error" in low:
                bad["cuda_error"] += 1
    return {
        "completion_marker_count": finished,
        "sacred_completed_after": completed_after,
        "main_finished_after": finished_after,
        "last_periodic_t_env": last_t_env,
        "log_reported_t_max": t_max,
        "failure_scan": bad,
        "crash_free": all(v == 0 for v in bad.values()),
    }


def scan_csv(path):
    """Row counts + max t_env, without loading the whole file into memory."""
    n_rows = n_train = n_test = 0
    max_t_train = max_t_test = 0
    header = None
    malformed = 0
    with open(path, newline="") as fh:
        rdr = csv.reader(fh)
        header = next(rdr)
        ncol = len(header)
        for row in rdr:
            if len(row) != ncol:
                malformed += 1
                continue
            n_rows += 1
            try:
                t = int(float(row[0]))
                tm = int(float(row[1]))
            except ValueError:
                malformed += 1
                continue
            if tm:
                n_test += 1
                max_t_test = max(max_t_test, t)
            else:
                n_train += 1
                max_t_train = max(max_t_train, t)
    return {
        "header": header,
        "n_data_rows": n_rows,
        "n_train_rows": n_train,
        "n_test_rows": n_test,
        "max_t_env_train_rows": max_t_train,
        "max_t_env_test_rows": max_t_test,
        "malformed_rows": malformed,
    }


def find_sacred_run(seed, csv_path):
    """Match a sacred run dir by (seed, t_max, selma_log_csv) -- not by name."""
    for d in sorted(glob.glob(os.path.join(SACRED_ROOT, "*")), key=lambda p: p):
        cfg_p = os.path.join(d, "config.json")
        if not os.path.exists(cfg_p):
            continue
        try:
            cfg = json.load(open(cfg_p))
        except Exception:
            continue
        if (
            cfg.get("seed") == seed
            and cfg.get("t_max") == 10_000_000
            and os.path.abspath(cfg.get("selma_log_csv", "")) == os.path.abspath(csv_path)
        ):
            return d, cfg
    return None, None


def find_checkpoint_dir(seed, start_time_iso):
    """Model dir names embed the local wall-clock start time; match on the date."""
    pat = os.path.join(
        MODELS_ROOT, f"qmix_seed{seed}_lbforaging:Foraging-10x10-3p-3f-v3_*"
    )
    cands = []
    for d in glob.glob(pat):
        steps = sorted(
            (int(x) for x in os.listdir(d) if x.isdigit()),
        )
        if not steps:
            continue
        cands.append((max(steps), d, steps))
    if not cands:
        return None
    # the 10M run is the one that reached the largest t_env
    best = max(cands, key=lambda c: c[0])
    total = 0
    for root, _, files in os.walk(best[1]):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return {
        "path": os.path.abspath(best[1]),
        "checkpoint_steps": best[2],
        "max_checkpoint_t_env": best[0],
        "total_bytes": total,
        "n_checkpoints": len(best[2]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hash-large", action="store_true",
                    help="also sha256 the per-episode CSVs and training logs")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    manifest = {
        "generated_utc": _dt.datetime.utcnow().isoformat() + "Z",
        "description": "SELMA five-seed 10M QMIX baseline on "
                       "lbforaging:Foraging-10x10-3p-3f-v3",
        "run_cwd": RUN_CWD,
        "seeds": {},
    }

    configs = {}
    for seed in SEEDS:
        log_p = LOG_TMPL.format(seed=seed)
        csv_p = CSV_TMPL.format(seed=seed)
        sacred_dir, cfg = find_sacred_run(seed, csv_p)
        run_json = {}
        if sacred_dir:
            try:
                run_json = json.load(open(os.path.join(sacred_dir, "run.json")))
            except Exception:
                run_json = {}
        configs[seed] = cfg or {}

        repos = run_json.get("experiment", {}).get("repositories", [])
        started = run_json.get("start_time")
        stopped = run_json.get("stop_time")
        elapsed_s = None
        if started and stopped:
            fmt = "%Y-%m-%dT%H:%M:%S.%f"
            try:
                elapsed_s = (
                    _dt.datetime.strptime(stopped, fmt)
                    - _dt.datetime.strptime(started, fmt)
                ).total_seconds()
            except ValueError:
                pass

        entry = {
            "seed": seed,
            "env": (cfg or {}).get("env_args", {}).get("key"),
            "algorithm": (cfg or {}).get("name"),
            "t_max_requested": (cfg or {}).get("t_max"),
            "log": dict(stat_entry(log_p, do_hash=args.hash_large), **scan_log(log_p)),
            "episode_csv": dict(
                stat_entry(csv_p, do_hash=args.hash_large), **scan_csv(csv_p)
            ),
            "sacred": {
                "run_dir": os.path.abspath(sacred_dir) if sacred_dir else None,
                "run_id": os.path.basename(sacred_dir) if sacred_dir else None,
                "status": run_json.get("status"),
                "start_time": started,
                "stop_time": stopped,
                "elapsed_seconds": elapsed_s,
                "hostname": run_json.get("host", {}).get("hostname"),
                "source_revisions": [
                    {"url": r.get("url"), "commit": r.get("commit"),
                     "dirty": r.get("dirty")}
                    for r in repos
                ],
                "config_json": stat_entry(
                    os.path.join(sacred_dir, "config.json"), do_hash=True
                ) if sacred_dir else None,
                "run_json": stat_entry(
                    os.path.join(sacred_dir, "run.json"), do_hash=True
                ) if sacred_dir else None,
                "metrics_json": stat_entry(
                    os.path.join(sacred_dir, "metrics.json")
                ) if sacred_dir else None,
                "cout_txt": stat_entry(
                    os.path.join(sacred_dir, "cout.txt")
                ) if sacred_dir else None,
            },
            "checkpoints": find_checkpoint_dir(seed, started),
        }
        manifest["seeds"][str(seed)] = entry

    # ---- configuration-identity check across the five runs -----------------
    ref = configs[SEEDS[0]]
    diffs = {}
    for seed in SEEDS[1:]:
        c = configs[seed]
        d = {
            k: [ref.get(k), c.get(k)]
            for k in set(ref) | set(c)
            if ref.get(k) != c.get(k)
        }
        if d:
            diffs[str(seed)] = d
    unexpected = {
        s: {k: v for k, v in d.items() if k not in EXPECTED_DIFFERING_KEYS}
        for s, d in diffs.items()
    }
    unexpected = {s: d for s, d in unexpected.items() if d}
    manifest["config_identity"] = {
        "reference_seed": SEEDS[0],
        "differences_vs_reference": diffs,
        "unexpected_differences": unexpected,
        "identical_apart_from_seed_and_csv_path": not unexpected,
        "reference_config": ref,
    }
    revs = {
        tuple(sorted(
            (r["commit"], r["dirty"])
            for r in manifest["seeds"][str(s)]["sacred"]["source_revisions"]
        ))
        for s in SEEDS
    }
    manifest["config_identity"]["single_source_revision"] = len(revs) == 1
    manifest["config_identity"]["source_revisions_observed"] = [
        [list(x) for x in r] for r in revs
    ]

    with open(os.path.join(OUT, "baseline_manifest_10M.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)

    flat_cols = [
        "seed", "env", "algorithm", "t_max_requested",
        "log_path", "log_size_bytes", "log_mtime_utc",
        "csv_path", "csv_size_bytes", "csv_mtime_utc",
        "sacred_run_dir", "sacred_run_id", "sacred_status",
        "sacred_start_time", "sacred_stop_time", "elapsed_seconds",
        "source_commit", "source_dirty",
        "checkpoint_dir", "n_checkpoints", "max_checkpoint_t_env",
        "checkpoint_bytes",
        "completion_marker_count", "crash_free", "last_periodic_t_env",
        "csv_rows_total", "csv_rows_train", "csv_rows_test",
        "csv_max_t_env_train", "csv_malformed_rows",
        "config_json_sha256",
    ]
    with open(os.path.join(OUT, "baseline_manifest_10M.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(flat_cols)
        for seed in SEEDS:
            e = manifest["seeds"][str(seed)]
            sc = e["sacred"]
            ck = e["checkpoints"] or {}
            rev = (sc["source_revisions"] or [{}])[0]
            w.writerow([
                seed, e["env"], e["algorithm"], e["t_max_requested"],
                e["log"]["path"], e["log"]["size_bytes"], e["log"]["mtime_utc"],
                e["episode_csv"]["path"], e["episode_csv"]["size_bytes"],
                e["episode_csv"]["mtime_utc"],
                sc["run_dir"], sc["run_id"], sc["status"],
                sc["start_time"], sc["stop_time"], sc["elapsed_seconds"],
                rev.get("commit"), rev.get("dirty"),
                ck.get("path"), ck.get("n_checkpoints"),
                ck.get("max_checkpoint_t_env"), ck.get("total_bytes"),
                e["log"]["completion_marker_count"], e["log"]["crash_free"],
                e["log"]["last_periodic_t_env"],
                e["episode_csv"]["n_data_rows"], e["episode_csv"]["n_train_rows"],
                e["episode_csv"]["n_test_rows"],
                e["episode_csv"]["max_t_env_train_rows"],
                e["episode_csv"]["malformed_rows"],
                (sc["config_json"] or {}).get("sha256"),
            ])

    print(f"wrote {OUT}/baseline_manifest_10M.json")
    print(f"wrote {OUT}/baseline_manifest_10M.csv")
    ok = manifest["config_identity"]["identical_apart_from_seed_and_csv_path"]
    print(f"config identity across five seeds: {'OK' if ok else 'MISMATCH'}")
    if not ok:
        print(json.dumps(unexpected, indent=2))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
