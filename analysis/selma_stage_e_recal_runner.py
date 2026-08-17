#!/usr/bin/env python
"""SELMA Stage E: execute the frozen beta-recalibration, two jobs at a time.

Runs the six conditions frozen by
``docs/project/selma_stage_e_beta_recalibration_design_2026-08-17.md`` in three
waves of two. Concurrency is **never** allowed above two, a wave only starts if
the previous one fully succeeded, and nothing is ever overwritten or resumed.

    wave 1   beta=0.00 seed 91 | beta=0.00 seed 92
    wave 2   beta=0.05 seed 91 | beta=0.05 seed 92
    wave 3   beta=0.10 seed 91 | beta=0.10 seed 92

**It never reads a performance value.** Between waves it inspects only exit
status, artifact presence, coverage, accounting, safety invariants and cost --
never a return, a success rate or a learning curve. A controller that can see
performance is a controller that can be tempted to adapt, which is exactly what
preregistration exists to prevent.

**Credentials.** ``OPENAI_API_KEY`` is inherited from this process's environment
by each training child and appears in no argument, manifest, log or report.

Run (inside tmux, with the secret sourced into this process only)::

    set -a; . ~/.config/selma/openai.env; set +a
    PYTHONPATH=src python analysis/selma_stage_e_recal_runner.py --confirm-live
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "src")

# ===========================================================================
# the frozen design (commit 824f2e9). None of this may be overridden.
# ===========================================================================
DESIGN_COMMIT = "824f2e98070c2741984070bd977d9a7dfbac114a"

FROZEN_BETAS = (0.00, 0.05, 0.10)
FROZEN_SEEDS = (91, 92)
FROZEN_T_MAX = 1500000
FROZEN_ENV_KEY = "lbforaging:Foraging-8x8-2p-2f-coop-v3"
FROZEN_TIME_LIMIT = 50
FROZEN_ALGORITHM = "qmix"
FROZEN_VARIANT = "wf15_m3_t5_ibon"
FROZEN_MODEL = "gpt-5.4-nano-2026-03-17"
FROZEN_PROMPT_VERSION = "selma_guidance_prompt_v3"
FROZEN_TTL = 15

#: Evaluation schedule. 50 000 is what the design's own horizon arithmetic
#: assumed (it divided the post-onset span by the baseline's interval to get
#: 18 points), so it is fixed here rather than chosen. 100 episodes matches the
#: authenticated baseline; 20 would quantise greedy success to 5 % steps, too
#: coarse for a metric whose observed range is 5-25 %.
FROZEN_TEST_INTERVAL = 50000
FROZEN_TEST_NEPISODE = 100

PER_CONDITION_CALL_CAP = 13500
PER_CONDITION_COST_CAP_USD = 3.49
GLOBAL_COST_CAP_USD = 20.94
GLOBAL_CALL_CAP = PER_CONDITION_CALL_CAP * 6

PRICE_TABLE_DATE = "2026-08-17"
PRICE_INPUT = 0.20
PRICE_CACHED_INPUT = 0.02
PRICE_OUTPUT = 1.25

MAX_CONCURRENCY = 2
MIN_FREE_GIB_BETWEEN_WAVES = 50

#: Forbidden everywhere. 0.25 was dropped by the design; 99 is detector
#: held-out evidence and is never a calibration seed.
FORBIDDEN_BETAS = (0.25,)
FORBIDDEN_SEEDS = (99,)

MIN_GUIDANCE_COVERAGE = 0.95

DEFAULT_ROOT = os.path.join(REPO, "analysis", "out", "stage_e_recal")


class RunnerError(RuntimeError):
    """A design, isolation or safety violation. Nothing proceeds after one."""


# ===========================================================================
# matrix
# ===========================================================================


def condition_name(beta, seed):
    return "beta_{:.2f}_seed{}".format(beta, seed).replace(".", "p")


def frozen_matrix(root=DEFAULT_ROOT):
    """The six conditions, in wave order. Beta-major, seed-minor."""
    conditions = []
    for beta in FROZEN_BETAS:
        for seed in FROZEN_SEEDS:
            conditions.append(build_condition(beta, seed, root))
    return conditions


def waves(root=DEFAULT_ROOT):
    matrix = frozen_matrix(root)
    return [matrix[i:i + MAX_CONCURRENCY]
            for i in range(0, len(matrix), MAX_CONCURRENCY)]


def build_condition(beta, seed, root=DEFAULT_ROOT):
    if beta in FORBIDDEN_BETAS:
        raise RunnerError(
            "beta={} was excluded by the frozen design and may not be run"
            .format(beta))
    if seed in FORBIDDEN_SEEDS:
        raise RunnerError(
            "seed={} is detector held-out evidence and may never be used for "
            "beta calibration".format(seed))
    if beta not in FROZEN_BETAS:
        raise RunnerError("beta={} is not in the frozen list {}".format(
            beta, list(FROZEN_BETAS)))
    if seed not in FROZEN_SEEDS:
        raise RunnerError("seed={} is not in the frozen list {}".format(
            seed, list(FROZEN_SEEDS)))

    name = condition_name(beta, seed)
    directory = os.path.join(root, name)
    return {
        "name": name,
        "beta": beta,
        "seed": seed,
        "t_max": FROZEN_T_MAX,
        "env_key": FROZEN_ENV_KEY,
        "time_limit": FROZEN_TIME_LIMIT,
        "algorithm": FROZEN_ALGORITHM,
        "variant": FROZEN_VARIANT,
        "model": FROZEN_MODEL,
        "ttl": FROZEN_TTL,
        "dir": directory,
        # every writable path is condition-unique
        "cache_path": os.path.join(directory, "guidance_cache.sqlite3"),
        "decision_log": os.path.join(directory, "decisions.csv"),
        "guidance_log": os.path.join(directory, "guidance.csv"),
        "episode_log": os.path.join(directory, "episodes.csv"),
        "results_path": os.path.join(directory, "results"),
        "run_log": os.path.join(directory, "run.log"),
        "manifest": os.path.join(directory, "manifest.json"),
        "done_marker": os.path.join(directory, "COMPLETED"),
        "live_call_cap": PER_CONDITION_CALL_CAP,
        "cost_cap_usd": PER_CONDITION_COST_CAP_USD,
        "test_interval": FROZEN_TEST_INTERVAL,
        "test_nepisode": FROZEN_TEST_NEPISODE,
    }


def writable_paths(condition):
    return [condition[k] for k in ("cache_path", "decision_log", "guidance_log",
                                   "episode_log", "results_path", "run_log",
                                   "manifest", "done_marker")]


def build_command(condition, python=None):
    """Exact argv. The API key is never here -- only in the environment."""
    return [
        python or sys.executable, os.path.join(SRC, "main.py"),
        "--config={}".format(condition["algorithm"]),
        "--env-config=gymma", "with",
        "env_args.key={}".format(condition["env_key"]),
        "env_args.time_limit={}".format(condition["time_limit"]),
        "seed={}".format(condition["seed"]),
        "t_max={}".format(condition["t_max"]),
        "use_cuda=False",
        "common_reward=True",
        "selma_mode=active",
        "selma_variant_id={}".format(condition["variant"]),
        "selma_provider=openai",
        "selma_beta={}".format(condition["beta"]),
        "selma_cache_mode=read_write",
        "selma_cache_path={}".format(condition["cache_path"]),
        "selma_live_call_cap={}".format(condition["live_call_cap"]),
        "selma_cost_cap_usd={}".format(condition["cost_cap_usd"]),
        "selma_price_table_date={}".format(PRICE_TABLE_DATE),
        "selma_price_input_usd_per_mtok={}".format(PRICE_INPUT),
        "selma_price_cached_input_usd_per_mtok={}".format(PRICE_CACHED_INPUT),
        "selma_price_output_usd_per_mtok={}".format(PRICE_OUTPUT),
        "selma_provider_timeout_s=30.0",
        "selma_provider_max_retries=1",
        "selma_provider_max_output_tokens=512",
        "selma_guidance_log_csv={}".format(condition["guidance_log"]),
        "selma_decision_log_csv={}".format(condition["decision_log"]),
        "selma_log_csv={}".format(condition["episode_log"]),
        "local_results_path={}".format(condition["results_path"]),
        "test_interval={}".format(condition["test_interval"]),
        "test_nepisode={}".format(condition["test_nepisode"]),
        "log_interval={}".format(condition["test_interval"]),
        "learner_log_interval={}".format(condition["test_interval"]),
        "runner_log_interval={}".format(condition["test_interval"]),
    ]


# ===========================================================================
# isolation and safety
# ===========================================================================


def assert_isolated(conditions):
    """No two conditions may share any writable path."""
    seen = {}
    for condition in conditions:
        for path in writable_paths(condition):
            resolved = os.path.abspath(path)
            if resolved in seen:
                raise RunnerError(
                    "conditions {!r} and {!r} would share the writable path "
                    "{!r}".format(seen[resolved], condition["name"], resolved))
            seen[resolved] = condition["name"]
    return len(seen)


def assert_empty_output(condition):
    """Refuse to start into an existing condition directory. Never overwrite."""
    directory = condition["dir"]
    if os.path.exists(directory) and os.listdir(directory):
        raise RunnerError(
            "condition directory {!r} already contains data. Existing evidence "
            "is never overwritten, deleted or resumed; move it aside "
            "deliberately if a fresh run is really intended.".format(directory))
    os.makedirs(directory, exist_ok=True)


def assert_no_conflicting_process():
    try:
        output = subprocess.run(["ps", "-eo", "cmd"], capture_output=True,
                                text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return 0
    hits = [l for l in output.splitlines()
            if "src/main.py" in l and "--config=" in l]
    if hits:
        raise RunnerError(
            "{} training process(es) already running; refusing to launch."
            .format(len(hits)))
    return 0


def free_gib(path=REPO):
    return shutil.disk_usage(path).free / (1024.0 ** 3)


def sha256_file(path, limit_bytes=None):
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
            if limit_bytes and size >= limit_bytes:
                break
    return digest.hexdigest()


def artifact_hashes(condition):
    out = {}
    for key in ("decision_log", "guidance_log", "episode_log", "cache_path"):
        path = condition[key]
        if os.path.exists(path):
            out[key] = {"path": path, "bytes": os.path.getsize(path),
                        "sha256": sha256_file(path)}
    return out


# ===========================================================================
# operational verification -- never performance
# ===========================================================================


def _int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def verify_condition(condition, exit_code):
    """Operational gates only. Reads no return, success rate or curve."""
    problems = []
    checks = {}

    checks["exit_code"] = exit_code
    if exit_code != 0:
        problems.append("non-zero exit code {}".format(exit_code))

    for key in ("decision_log", "guidance_log", "episode_log"):
        if not os.path.exists(condition[key]):
            problems.append("missing artifact: {}".format(key))
    if problems:
        return {"problems": problems, "checks": checks}

    with open(condition["guidance_log"], newline="") as handle:
        guidance = list(csv.DictReader(handle))
    with open(condition["decision_log"], newline="") as handle:
        decisions = list(csv.DictReader(handle))

    final_t = max((_int(r["t_env"]) for r in guidance), default=0)
    checks["final_t_env"] = final_t
    if final_t < condition["t_max"] - condition["time_limit"]:
        problems.append("horizon not reached: t_env {} < {}".format(
            final_t, condition["t_max"]))

    considered = [r for r in decisions if r["decision"] == "multiple"]
    exhausted = sum(1 for r in considered if r["outcome"] == "budget_exhausted")
    coverage = ((len(considered) - exhausted) / float(len(considered))
                if considered else 1.0)
    checks["decisions"] = len(considered)
    checks["budget_exhausted"] = exhausted
    checks["guidance_coverage"] = coverage
    if coverage < MIN_GUIDANCE_COVERAGE:
        problems.append("guidance coverage {:.3f} < {:.2f}".format(
            coverage, MIN_GUIDANCE_COVERAGE))

    attempts = sum(_int(r["live_attempts"]) for r in considered)
    checks["live_attempts"] = attempts
    if attempts > condition["live_call_cap"]:
        problems.append("attempts {} exceed cap {}".format(
            attempts, condition["live_call_cap"]))

    called = [r for r in considered if _int(r["live_attempts"]) > 0]
    tin = sum(_int(r["tokens_input"]) for r in called)
    tcached = sum(_int(r["tokens_cached_input"]) for r in called)
    tout = sum(_int(r["tokens_output"]) for r in called)
    cost = ((tin - tcached) * PRICE_INPUT + tcached * PRICE_CACHED_INPUT
            + tout * PRICE_OUTPUT) / 1e6
    checks["input_tokens"] = tin
    checks["cached_input_tokens"] = tcached
    checks["output_tokens"] = tout
    checks["cost_usd"] = cost
    if cost > condition["cost_cap_usd"]:
        problems.append("cost ${:.4f} exceeds cap ${}".format(
            cost, condition["cost_cap_usd"]))

    models = {r["returned_model"] for r in called if r["returned_model"]}
    checks["returned_models"] = sorted(models)
    if models - {FROZEN_MODEL}:
        problems.append("model substitution: {}".format(sorted(models)))

    activated = [r for r in decisions if _int(r["activated"])]
    invalid = sum(1 for r in activated if not _int(r["physically_valid"]))
    invented = sum(1 for r in activated if not r["selected_plan_id"])
    checks["activations"] = len(activated)
    checks["physically_invalid_activations"] = invalid
    checks["invented_plan_ids"] = invented
    if invalid:
        problems.append("{} physically invalid activations".format(invalid))
    if invented:
        problems.append("{} invented plan ids".format(invented))

    greedy_calls = sum(1 for r in decisions if _int(r["greedy_eval"]))
    checks["greedy_eval_decisions"] = greedy_calls
    if greedy_calls:
        problems.append("{} decisions during greedy evaluation".format(greedy_calls))

    reward_errors = 0
    non_finite = 0
    residual = 0.0
    terminal_phi_nonzero = 0
    by_episode = {}
    for row in guidance:
        raw = _float(row["raw_team_reward"])
        train = _float(row["training_reward"])
        shaping = _float(row["shaping_contribution"], 0.0)
        for value in (raw, train, shaping, _float(row["pbrs_term_unscaled"], 0.0),
                      _float(row["phi_before"], 0.0),
                      _float(row["phi_after_effective"], 0.0)):
            if value is None or value != value or value in (float("inf"),
                                                            float("-inf")):
                non_finite += 1
        if None not in (raw, train) and abs(train - (raw + shaping)) > 1e-9:
            reward_errors += 1
        if _int(row["stored_terminated"]) and \
                _float(row["phi_after_effective"], 0.0) != 0.0:
            terminal_phi_nonzero += 1
        by_episode.setdefault(row["episode_id"], []).append(row)
    for rows in by_episode.values():
        rows.sort(key=lambda r: _int(r["episode_step"]))
        for previous, current in zip(rows, rows[1:]):
            residual = max(residual, abs(
                _float(previous["phi_after_effective"], 0.0)
                - _float(current["phi_before"], 0.0)))

    checks["reward_reconciliation_errors"] = reward_errors
    checks["non_finite_values"] = non_finite
    checks["max_telescoping_residual"] = residual
    checks["terminal_phi_nonzero"] = terminal_phi_nonzero
    if reward_errors:
        problems.append("{} reward reconciliation errors".format(reward_errors))
    if non_finite:
        problems.append("{} non-finite values".format(non_finite))
    if residual > 1e-9:
        problems.append("telescoping residual {:.3g}".format(residual))
    if terminal_phi_nonzero:
        problems.append("{} terminal transitions with non-zero phi".format(
            terminal_phi_nonzero))

    if os.path.exists(condition["run_log"]):
        with open(condition["run_log"], errors="replace") as handle:
            text = handle.read()
        for marker in ("sk-", "OPENAI_API_KEY", "Authorization:"):
            if marker in text:
                problems.append("possible credential material in run log")
                break
        checks["run_log_bytes"] = os.path.getsize(condition["run_log"])

    return {"problems": problems, "checks": checks}


def _float(value, default=None):
    if value in (None, "", "n/a"):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ===========================================================================
# execution
# ===========================================================================


def child_env():
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC + os.pathsep + env.get("PYTHONPATH", "")
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    return env


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                              capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def write_manifest(condition, record):
    """Durable per-condition record. Contains no credential, by construction."""
    payload = dict(record)
    payload["condition"] = {k: v for k, v in condition.items()}
    payload["design_commit"] = DESIGN_COMMIT
    payload["execution_commit"] = git_commit()
    payload["frozen"] = {
        "betas": list(FROZEN_BETAS), "seeds": list(FROZEN_SEEDS),
        "t_max": FROZEN_T_MAX, "model": FROZEN_MODEL,
        "prompt_version": FROZEN_PROMPT_VERSION, "ttl": FROZEN_TTL,
        "call_cap": PER_CONDITION_CALL_CAP,
        "cost_cap_usd": PER_CONDITION_COST_CAP_USD,
        "test_interval": FROZEN_TEST_INTERVAL,
        "test_nepisode": FROZEN_TEST_NEPISODE,
    }
    text = json.dumps(payload, indent=2, sort_keys=True, default=str)
    for marker in ("sk-", "OPENAI_API_KEY"):
        if marker in text:
            raise RunnerError("refusing to write a manifest containing {!r}"
                              .format(marker))
    with open(condition["manifest"], "w") as handle:
        handle.write(text)
    return payload


def run_wave(wave, *, env, python=None, stagger_s=30):
    """Launch a wave concurrently (max two) and wait for both.

    The launches are staggered so Sacred allocates its run ids sequentially.
    Sacred itself catches the directory race, and every condition is uniquely
    identifiable by its own log paths, so this is belt and braces rather than
    the only protection.
    """
    if len(wave) > MAX_CONCURRENCY:
        raise RunnerError("wave of {} exceeds max concurrency {}".format(
            len(wave), MAX_CONCURRENCY))

    processes = []
    for index, condition in enumerate(wave):
        assert_empty_output(condition)
        command = build_command(condition, python=python)
        handle = open(condition["run_log"], "w")
        if index and stagger_s:
            time.sleep(stagger_s)
        started = time.time()
        process = subprocess.Popen(command, cwd=REPO, env=env, stdout=handle,
                                   stderr=subprocess.STDOUT)
        processes.append({
            "condition": condition, "process": process, "handle": handle,
            "command": command, "pid": process.pid, "started": started,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                         time.gmtime(started)),
        })
        print("  launched {} beta={} seed={} pid={}".format(
            condition["name"], condition["beta"], condition["seed"],
            process.pid), flush=True)

    results = []
    for entry in processes:
        exit_code = entry["process"].wait()
        entry["handle"].close()
        condition = entry["condition"]
        verification = verify_condition(condition, exit_code)
        record = {
            "name": condition["name"],
            "beta": condition["beta"],
            "seed": condition["seed"],
            "command": entry["command"],
            "pid": entry["pid"],
            "started_utc": entry["started_utc"],
            "ended_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "wall_clock_s": round(time.time() - entry["started"], 1),
            "exit_code": exit_code,
            "checks": verification["checks"],
            "problems": verification["problems"],
            "artifact_hashes": artifact_hashes(condition),
        }
        write_manifest(condition, record)
        if not verification["problems"]:
            with open(condition["done_marker"], "w") as marker:
                marker.write("exit_code=0\nverified_utc={}\n".format(
                    record["ended_utc"]))
        results.append(record)
        print("  finished {} exit={} wall={}s coverage={} cost=${}".format(
            condition["name"], exit_code, record["wall_clock_s"],
            verification["checks"].get("guidance_coverage"),
            round(verification["checks"].get("cost_usd", 0.0), 4)), flush=True)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--confirm-live", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan; launch nothing")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--report", default=None)
    parser.add_argument("--python", default=None)
    parser.add_argument("--stagger-s", type=int, default=30)
    args = parser.parse_args(argv)

    plan = waves(args.root)
    matrix = [c for wave in plan for c in wave]
    assert_isolated(matrix)

    report = {
        "kind": "selma_stage_e_beta_recalibration_execution",
        "design_commit": DESIGN_COMMIT,
        "execution_commit": git_commit(),
        "frozen_betas": list(FROZEN_BETAS),
        "frozen_seeds": list(FROZEN_SEEDS),
        "t_max": FROZEN_T_MAX,
        "model": FROZEN_MODEL,
        "per_condition_call_cap": PER_CONDITION_CALL_CAP,
        "per_condition_cost_cap_usd": PER_CONDITION_COST_CAP_USD,
        "global_cost_cap_usd": GLOBAL_COST_CAP_USD,
        "max_concurrency": MAX_CONCURRENCY,
        "performance_inspected_during_execution": False,
        "waves": [[c["name"] for c in wave] for wave in plan],
        "conditions": [],
    }

    print("SELMA Stage E beta recalibration -- {} conditions in {} waves of {}"
          .format(len(matrix), len(plan), MAX_CONCURRENCY))
    for index, wave in enumerate(plan, 1):
        print("  wave {}: {}".format(index, ", ".join(
            "{} (beta={}, seed={})".format(c["name"], c["beta"], c["seed"])
            for c in wave)))
    print("  t_max={} model={} cap={} calls / ${} per condition; ${} global"
          .format(FROZEN_T_MAX, FROZEN_MODEL, PER_CONDITION_CALL_CAP,
                  PER_CONDITION_COST_CAP_USD, GLOBAL_COST_CAP_USD))

    if args.dry_run or not args.confirm_live:
        report["status"] = "dry_run" if args.dry_run else "not_confirmed"
        print("\n{}: nothing launched.".format(report["status"]))
        _write(report, args.report)
        return 0

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not set in this process. Source the credential "
            "into the shell that starts this controller; it is never read from "
            "a file or an argument here.")
    assert_no_conflicting_process()
    if free_gib() < MIN_FREE_GIB_BETWEEN_WAVES:
        raise RunnerError("only {:.1f} GiB free; refusing to start."
                          .format(free_gib()))

    env = child_env()
    spent = 0.0
    status = "completed"

    for index, wave in enumerate(plan, 1):
        print("\n=== wave {} of {} ===".format(index, len(plan)), flush=True)
        results = run_wave(wave, env=env, python=args.python,
                           stagger_s=args.stagger_s)
        report["conditions"].extend(results)
        spent += sum(r["checks"].get("cost_usd", 0.0) for r in results)
        report["cumulative_cost_usd"] = round(spent, 6)
        _write(report, args.report)

        failed = [r for r in results if r["problems"]]
        if failed:
            status = "stopped_failed_condition"
            print("\nSTOPPING before the next wave. Failures:", flush=True)
            for record in failed:
                print("  {}: {}".format(record["name"], record["problems"]))
            break
        if spent > GLOBAL_COST_CAP_USD:
            status = "stopped_global_cost"
            print("\nSTOPPING: global cost ${:.4f} exceeds ${}".format(
                spent, GLOBAL_COST_CAP_USD))
            break
        if index < len(plan):
            os.sync() if hasattr(os, "sync") else None
            free = free_gib()
            print("  free disk before next wave: {:.1f} GiB".format(free))
            if free < MIN_FREE_GIB_BETWEEN_WAVES:
                status = "stopped_low_disk"
                print("\nSTOPPING: only {:.1f} GiB free".format(free))
                break

    report["status"] = status
    report["cumulative_cost_usd"] = round(spent, 6)
    report["conditions_completed"] = sum(
        1 for r in report["conditions"] if not r["problems"])
    _write(report, args.report)
    print("\nstatus: {}  completed: {}/{}  cost: ${:.4f}".format(
        status, report["conditions_completed"], len(matrix), spent))
    return 0 if status == "completed" else 1


def _write(report, path):
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    text = json.dumps(report, indent=2, sort_keys=True, default=str)
    if "sk-" in text or "OPENAI_API_KEY" in text:
        raise RunnerError("refusing to write a report containing credentials")
    with open(path, "w") as handle:
        handle.write(text)


if __name__ == "__main__":
    sys.exit(main())
