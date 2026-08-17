#!/usr/bin/env python
"""SELMA Stage E: run the preregistered four-condition beta-calibration pilot.

Executes exactly four 500 000-step QMIX conditions, sequentially, differing only
in the PBRS shaping scale ``beta``:

    beta = 0.00, 0.05, 0.10, 0.25

The design is frozen in
``docs/project/selma_stage_e_beta_calibration_2026-08-17.md`` and this script
refuses to depart from it: a different beta list, a different seed or a different
``t_max`` is rejected before anything runs. That is deliberate. An orchestrator
that will happily run a slightly different pilot is an invitation to adapt the
design after seeing a result, which is exactly what preregistration exists to
prevent.

**It never inspects performance.** Nothing here reads a return, a success rate or
a learning curve, so no later condition can be influenced by an earlier one.
Only operational state -- process, ``t_env``, disk, attempts, cost, invariants --
is examined while running.

Safety
------

* live runs are refused unless ``--confirm-live`` is passed;
* per-condition caps: 1 000 live attempts (retries included) and $0.25;
* global caps: 4 000 attempts and $1.00, checked before launching each condition;
* every condition gets an **empty** cache and a fresh output directory, and the
  script refuses to reuse or overwrite an existing one;
* ``OPENAI_API_KEY`` is inherited from the environment by the child process and
  is never placed in an argument, a log line, a report or a Sacred config.

Run (dry run -- prints the four resolved configurations, touches nothing)::

    PYTHONPATH=src python analysis/selma_stage_e_beta_pilot.py --dry-run

Run (live; the caller sources the secret into this process only)::

    set -a; . ~/.config/selma/openai.env; set +a
    PYTHONPATH=src python analysis/selma_stage_e_beta_pilot.py --confirm-live
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from selma.pricing import PriceTable  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "src")

# -- the preregistered design. None of these may be overridden. --------------
FROZEN_BETAS = (0.0, 0.05, 0.10, 0.25)
FROZEN_SEED = 91
FROZEN_T_MAX = 500000
FROZEN_ENV_KEY = "lbforaging:Foraging-8x8-2p-2f-coop-v3"
FROZEN_TIME_LIMIT = 50
FROZEN_ALGORITHM = "qmix"
FROZEN_VARIANT = "wf15_m3_t5_ibon"
FROZEN_PROVIDER = "openai"
FROZEN_MODEL = "gpt-5.4-nano-2026-03-17"
FROZEN_TEST_INTERVAL = 10000

PER_CONDITION_ATTEMPT_CAP = 1000
PER_CONDITION_COST_CAP_USD = 0.25
GLOBAL_ATTEMPT_CAP = 4000
GLOBAL_COST_CAP_USD = 1.00

#: Frozen 2026-08-17 table; see the preregistration record for the source.
PRICE_TABLE = PriceTable("2026-08-17", 0.20, 0.02, 1.25)

DEFAULT_ROOT = os.path.join(REPO, "analysis", "out", "stage_e_beta_pilot")


class PilotError(RuntimeError):
    """A preregistration or safety violation. Nothing runs after one."""


def condition_name(beta):
    return "beta_{:.2f}".format(beta).replace(".", "p")


def build_config(beta, root):
    """The resolved configuration for one condition. No secret is present."""
    out = os.path.join(root, condition_name(beta))
    return {
        "beta": beta,
        "name": condition_name(beta),
        "dir": out,
        "algorithm": FROZEN_ALGORITHM,
        "env_key": FROZEN_ENV_KEY,
        "time_limit": FROZEN_TIME_LIMIT,
        "seed": FROZEN_SEED,
        "t_max": FROZEN_T_MAX,
        "variant": FROZEN_VARIANT,
        "provider": FROZEN_PROVIDER,
        "model": FROZEN_MODEL,
        "cache_mode": "read_write",
        "cache_path": os.path.join(out, "guidance_cache.sqlite3"),
        "guidance_log": os.path.join(out, "guidance.csv"),
        "decision_log": os.path.join(out, "decisions.csv"),
        "episode_log": os.path.join(out, "episodes.csv"),
        "results_path": os.path.join(out, "results"),
        "run_log": os.path.join(out, "run.log"),
        "live_call_cap": PER_CONDITION_ATTEMPT_CAP,
        "cost_cap_usd": PER_CONDITION_COST_CAP_USD,
        "test_interval": FROZEN_TEST_INTERVAL,
        "use_cuda": False,
        "common_reward": True,
    }


def build_command(config, python=None):
    """The exact argv. The API key is never an argument -- only an env var."""
    return [
        python or sys.executable, os.path.join(SRC, "main.py"),
        "--config={}".format(config["algorithm"]),
        "--env-config=gymma", "with",
        "env_args.key={}".format(config["env_key"]),
        "env_args.time_limit={}".format(config["time_limit"]),
        "seed={}".format(config["seed"]),
        "t_max={}".format(config["t_max"]),
        "use_cuda={}".format(config["use_cuda"]),
        "common_reward={}".format(config["common_reward"]),
        "selma_mode=active",
        "selma_variant_id={}".format(config["variant"]),
        "selma_provider={}".format(config["provider"]),
        "selma_beta={}".format(config["beta"]),
        "selma_cache_mode={}".format(config["cache_mode"]),
        "selma_cache_path={}".format(config["cache_path"]),
        "selma_live_call_cap={}".format(config["live_call_cap"]),
        "selma_cost_cap_usd={}".format(config["cost_cap_usd"]),
        "selma_price_table_date={}".format(PRICE_TABLE.date),
        "selma_price_input_usd_per_mtok={}".format(PRICE_TABLE.input_usd_per_mtok),
        "selma_price_cached_input_usd_per_mtok={}".format(
            PRICE_TABLE.cached_input_usd_per_mtok),
        "selma_price_output_usd_per_mtok={}".format(PRICE_TABLE.output_usd_per_mtok),
        "selma_provider_timeout_s=30.0",
        "selma_provider_max_retries=1",
        "selma_provider_max_output_tokens=512",
        "selma_guidance_log_csv={}".format(config["guidance_log"]),
        "selma_decision_log_csv={}".format(config["decision_log"]),
        "selma_log_csv={}".format(config["episode_log"]),
        "local_results_path={}".format(config["results_path"]),
        "test_interval={}".format(config["test_interval"]),
        "log_interval={}".format(config["test_interval"]),
        "learner_log_interval={}".format(config["test_interval"]),
        "runner_log_interval={}".format(config["test_interval"]),
    ]


# ---------------------------------------------------------------------------
# preregistration guards
# ---------------------------------------------------------------------------


def assert_frozen_design(betas, seed, t_max):
    """Refuse anything but the preregistered design."""
    if tuple(betas) != FROZEN_BETAS:
        raise PilotError(
            "the beta list is preregistered as {} and may not be changed; got {}. "
            "Adding, removing or reordering a beta after the fact would make this "
            "an unregistered search.".format(list(FROZEN_BETAS), list(betas)))
    if int(seed) != FROZEN_SEED:
        raise PilotError(
            "seed is preregistered as {} (the development seed) and may not be "
            "changed; got {}. Final/evaluation seeds are never used here."
            .format(FROZEN_SEED, seed))
    if int(t_max) != FROZEN_T_MAX:
        raise PilotError(
            "t_max is preregistered as {} and may not be changed; got {}."
            .format(FROZEN_T_MAX, t_max))


def assert_clean_directory(path):
    """A condition may never reuse or overwrite an existing directory."""
    if os.path.exists(path) and os.listdir(path):
        raise PilotError(
            "condition directory {!r} already exists and is not empty. A completed "
            "condition is never re-run and a partial one is never resumed: move it "
            "aside deliberately if you really mean to start over.".format(path))
    os.makedirs(path, exist_ok=True)


def assert_no_other_training_process():
    try:
        output = subprocess.run(
            ["ps", "-eo", "cmd"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return 0
    running = [line for line in output.splitlines()
               if line.strip().endswith("main.py") or "src/main.py --config=" in line]
    if running:
        raise PilotError(
            "another training process is already running; refusing to launch a "
            "second. Exactly one process runs at a time in this pilot.")
    return 0


def free_disk_gb(path):
    usage = shutil.disk_usage(path)
    return usage.free / (1024.0 ** 3)


# ---------------------------------------------------------------------------
# per-condition accounting, read back from the durable decision log
# ---------------------------------------------------------------------------


def read_condition_usage(config):
    """Attempts, tokens and cost for one finished condition.

    Read from the decision log rather than tracked in this process, so the
    numbers are the ones actually durably recorded.
    """
    import csv

    path = config["decision_log"]
    if not os.path.exists(path):
        return {"attempts": 0, "retries": 0, "input_tokens": 0,
                "cached_input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
                "decisions": 0, "budget_exhausted": 0, "activations": 0}

    def num(value):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return 0

    attempts = retries = tin = tcached = tout = 0
    decisions = exhausted = activations = 0
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            decisions += 1
            attempts += num(row.get("live_attempts"))
            retries += num(row.get("retries"))
            activations += num(row.get("activated"))
            if row.get("outcome") == "budget_exhausted":
                exhausted += 1
            if num(row.get("live_attempts")) > 0:
                tin += num(row.get("tokens_input"))
                tcached += num(row.get("tokens_cached_input"))
                tout += num(row.get("tokens_output"))
    return {
        "attempts": attempts, "retries": retries, "input_tokens": tin,
        "cached_input_tokens": tcached, "output_tokens": tout,
        "cost_usd": PRICE_TABLE.cost_usd(input_tokens=tin,
                                         cached_input_tokens=tcached,
                                         output_tokens=tout),
        "decisions": decisions, "budget_exhausted": exhausted,
        "activations": activations,
    }


def correctness_scan(config):
    """Operational correctness only -- never a performance quantity.

    Looks for the failures that must stop the pilot: a crash, an accounting or
    reward-reconciliation violation, non-finite values, a model substitution, or
    credential material in the run log.
    """
    problems = []
    log = config["run_log"]
    if os.path.exists(log):
        with open(log, errors="replace") as handle:
            text = handle.read()
        for marker in ("Traceback", "InvariantViolation", "AssertionError"):
            if marker in text:
                problems.append("run log contains {}".format(marker))
        for secret in ("sk-", "OPENAI_API_KEY", "Authorization:"):
            if secret in text:
                problems.append("possible credential material in run log")
                break

    import csv

    path = config["decision_log"]
    if os.path.exists(path):
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                model = row.get("returned_model") or ""
                if model and model != FROZEN_MODEL:
                    problems.append("model substitution: {!r}".format(model))
                    break
                if row.get("greedy_eval") not in ("0", "", None):
                    problems.append("a decision was made during greedy evaluation")
                    break
                if row.get("activated") == "1" and row.get("physically_valid") != "1":
                    problems.append("physically invalid guidance activated")
                    break
    return problems


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------


def run_condition(config, *, env, python=None, dry_run=False):
    """Launch one condition and wait. Returns a record; never inspects returns."""
    command = build_command(config, python=python)
    record = {
        "name": config["name"], "beta": config["beta"], "dir": config["dir"],
        "command": command, "dry_run": bool(dry_run),
    }
    if dry_run:
        record.update({"skipped": True, "exit_code": None})
        return record

    assert_clean_directory(config["dir"])
    record["started_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    started = time.time()
    with open(config["run_log"], "w") as handle:
        process = subprocess.Popen(command, cwd=REPO, env=env, stdout=handle,
                                   stderr=subprocess.STDOUT)
        record["pid"] = process.pid
        exit_code = process.wait()
    record["exit_code"] = exit_code
    record["ended_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    record["wall_clock_s"] = round(time.time() - started, 1)
    record["usage"] = read_condition_usage(config)
    record["correctness_problems"] = correctness_scan(config)
    return record


def child_env():
    """The child's environment. The key is inherited, never logged or passed."""
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC + os.pathsep + env.get("PYTHONPATH", "")
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    return env


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--confirm-live", action="store_true",
                        help="required to actually execute the four conditions")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the resolved configurations and exit; "
                             "makes no network call and writes no run data")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--report", default=None)
    parser.add_argument("--python", default=None,
                        help="interpreter for the child process")
    parser.add_argument("--betas", default=",".join(str(b) for b in FROZEN_BETAS),
                        help="preregistered; any other value is refused")
    parser.add_argument("--seed", type=int, default=FROZEN_SEED)
    parser.add_argument("--t-max", type=int, default=FROZEN_T_MAX)
    args = parser.parse_args(argv)

    betas = tuple(float(b) for b in args.betas.split(","))
    assert_frozen_design(betas, args.seed, args.t_max)

    configs = [build_config(beta, args.root) for beta in betas]
    report = {
        "kind": "selma_stage_e_beta_calibration_pilot",
        "preregistered_betas": list(FROZEN_BETAS),
        "seed": FROZEN_SEED,
        "t_max": FROZEN_T_MAX,
        "model": FROZEN_MODEL,
        "price_table_date": PRICE_TABLE.date,
        "per_condition_attempt_cap": PER_CONDITION_ATTEMPT_CAP,
        "per_condition_cost_cap_usd": PER_CONDITION_COST_CAP_USD,
        "global_attempt_cap": GLOBAL_ATTEMPT_CAP,
        "global_cost_cap_usd": GLOBAL_COST_CAP_USD,
        "not_a_performance_result": True,
        "performance_inspected_during_execution": False,
        "configs": [{k: v for k, v in c.items()} for c in configs],
        "conditions": [],
    }

    print("SELMA Stage E beta-calibration pilot")
    print("  betas       : {}".format(list(FROZEN_BETAS)))
    print("  seed        : {}   t_max: {}".format(FROZEN_SEED, FROZEN_T_MAX))
    print("  model       : {}".format(FROZEN_MODEL))
    print("  caps        : {} attempts / ${} per condition; {} / ${} global".format(
        PER_CONDITION_ATTEMPT_CAP, PER_CONDITION_COST_CAP_USD,
        GLOBAL_ATTEMPT_CAP, GLOBAL_COST_CAP_USD))
    for config in configs:
        print("  condition {:<10} beta={:<5} dir={}".format(
            config["name"], config["beta"], config["dir"]))

    if args.dry_run or not args.confirm_live:
        report["status"] = "dry_run" if args.dry_run else "not_confirmed"
        print("\n{}: no condition was executed and no network call was made."
              .format(report["status"]))
        _write(report, args.report)
        return 0

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not set in this process. Source the credential "
            "into the subprocess that runs this script; it is never read from a "
            "file, a config or an argument here.")

    assert_no_other_training_process()
    free_gb = free_disk_gb(REPO)
    report["free_disk_gb_at_start"] = round(free_gb, 1)
    if free_gb < 5.0:
        raise PilotError("only {:.1f} GB free; refusing to start.".format(free_gb))

    env = child_env()
    spent = 0.0
    attempts = 0
    status = "completed"

    for config in configs:
        remaining_cost = GLOBAL_COST_CAP_USD - spent
        remaining_attempts = GLOBAL_ATTEMPT_CAP - attempts
        if remaining_cost < PER_CONDITION_COST_CAP_USD or \
                remaining_attempts < PER_CONDITION_ATTEMPT_CAP:
            report["conditions"].append({
                "name": config["name"], "beta": config["beta"],
                "skipped": True,
                "reason": "global budget would be exceeded: ${:.4f} and {} "
                          "attempts remain".format(remaining_cost, remaining_attempts),
            })
            status = "stopped_global_budget"
            break

        print("\n launching {} (beta={}) ...".format(config["name"], config["beta"]),
              flush=True)
        record = run_condition(config, env=env, python=args.python)
        report["conditions"].append(record)
        spent += record["usage"]["cost_usd"]
        attempts += record["usage"]["attempts"]
        report["cumulative_cost_usd"] = round(spent, 6)
        report["cumulative_attempts"] = attempts
        _write(report, args.report)

        print("   exit={} wall={}s attempts={} cost=${:.5f} (cumulative ${:.5f})"
              .format(record["exit_code"], record["wall_clock_s"],
                      record["usage"]["attempts"], record["usage"]["cost_usd"], spent))

        if record["exit_code"] != 0:
            status = "stopped_nonzero_exit"
            print("   STOPPING: non-zero exit; later conditions are not launched.")
            break
        if record["correctness_problems"]:
            status = "stopped_correctness"
            print("   STOPPING: {}".format(record["correctness_problems"]))
            break
        if record["usage"]["attempts"] > PER_CONDITION_ATTEMPT_CAP:
            status = "stopped_attempt_overrun"
            print("   STOPPING: per-condition attempt cap exceeded.")
            break
        if spent > GLOBAL_COST_CAP_USD:
            status = "stopped_global_cost"
            print("   STOPPING: global cost cap exceeded.")
            break

    report["status"] = status
    report["cumulative_cost_usd"] = round(spent, 6)
    report["cumulative_attempts"] = attempts
    report["any_condition_hit_its_cap"] = any(
        c.get("usage", {}).get("budget_exhausted", 0) > 0
        for c in report["conditions"])
    _write(report, args.report)

    print("\n status: {}  conditions run: {}  attempts: {}  cost: ${:.5f}".format(
        status, sum(1 for c in report["conditions"] if not c.get("skipped")),
        attempts, spent))
    return 0 if status == "completed" else 1


def _write(report, path):
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, default=str)


if __name__ == "__main__":
    sys.exit(main())
