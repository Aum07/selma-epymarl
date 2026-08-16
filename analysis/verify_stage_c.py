#!/usr/bin/env python
"""SELMA Stage C: durable beta = 0 equivalence verification.

Runs two short, identically seeded trainings --

  A.  ``selma_mode=off``                                    (unshaped QMIX)
  B.  ``selma_mode=active`` + ``deterministic_mock`` + ``beta=0``

-- and requires that **every learning-relevant artifact** is exactly equal. The
guidance machinery in B is fully live: it queries the provider, activates
subgoals, ages them and expires them. Only the reward term is zero. If any of
that leaks into learning, this script fails.

Why it is written this way
--------------------------

An earlier ad-hoc bash comparison reported ``BYTE-IDENTICAL: YES`` for two runs
that had both crashed, because ``sha256sum`` failed on the missing files, both
captured hashes were the empty string, and ``[ "" = "" ]`` is true. Every
structural decision here exists to make that class of false pass impossible:

* the output directories are created by this script, not assumed;
* each training is launched with :func:`subprocess.run` and its **return code is
  checked**; a non-zero exit fails the verification immediately;
* every expected artifact is checked for existence **and** non-emptiness before
  any comparison runs;
* comparisons are of loaded contents -- tensors, optimizer state, parsed CSV
  values -- never of filenames, and never of two absent values;
* an empty comparison set is itself a failure: a check that compared nothing
  reports FAIL, not PASS.

Provider and guidance counters are reported **separately** and are expected to
differ between A and B. They are not learning state.

Usage::

    /home/aum/selma/venv/bin/python analysis/verify_stage_c.py [--out DIR] [--episodes N]

Exit code 0 only if every learning-state check passes.
"""

import argparse
import csv
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "src")
PYTHON = sys.executable

#: Development seed. Deliberately NOT one of the final experimental seeds 1-5,
#: which are reserved for the reported experiments.
DEV_SEED = 4242

FROZEN_VARIANT = "wf15_m3_t5_ibon"
ENV_KEY = "lbforaging:Foraging-8x8-2p-2f-coop-v3"

#: Learner statistics that must match exactly. These are the QMIX training
#: signals: if shaping leaked anywhere, at least one of them moves.
LEARNER_STATS = (
    "loss", "grad_norm", "q_taken_mean", "target_mean", "td_error_abs",
    "return_mean", "test_return_mean", "ep_length_mean",
)


class Failure(Exception):
    pass


class Report:
    """Collects check results. Nothing is a pass by default."""

    def __init__(self):
        self.rows = []

    def check(self, name, passed, detail=""):
        self.rows.append((name, bool(passed), detail))
        print("  [{}] {}{}".format(
            "PASS" if passed else "FAIL", name, "   " + detail if detail else ""))
        return bool(passed)

    def require_nonempty(self, name, count, detail=""):
        """A comparison that compared nothing is a failure, never a pass."""
        return self.check(
            name, count > 0,
            detail or "compared {} item(s)".format(count),
        )

    @property
    def failed(self):
        return [name for name, passed, _ in self.rows if not passed]


# ---------------------------------------------------------------------------
# running the two conditions
# ---------------------------------------------------------------------------


def run_condition(name, out_dir, episodes, extra):
    """Launch one training run. Returns its output paths. Raises on failure."""
    os.makedirs(out_dir, exist_ok=True)
    t_max = episodes * 50  # time_limit=50, so this is an episode budget
    log_path = os.path.join(out_dir, "run.log")
    common = [
        PYTHON, os.path.join(SRC, "main.py"),
        "--config=qmix", "--env-config=gymma", "with",
        "env_args.key={}".format(ENV_KEY),
        "env_args.time_limit=50",
        "seed={}".format(DEV_SEED),
        "t_max={}".format(t_max),
        "use_cuda=False",
        "test_interval={}".format(t_max),
        "log_interval={}".format(max(1, t_max // 4)),
        "learner_log_interval={}".format(max(1, t_max // 4)),
        "runner_log_interval={}".format(max(1, t_max // 4)),
        "save_model=True",
        "save_model_interval={}".format(max(1, t_max // 2)),
        "local_results_path={}".format(os.path.join(out_dir, "results")),
        "selma_log_csv={}".format(os.path.join(out_dir, "episodes.csv")),
    ] + extra

    env = dict(os.environ)
    env["PYTHONPATH"] = SRC + os.pathsep + env.get("PYTHONPATH", "")
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"

    print("  launching {} ...".format(name), flush=True)
    with open(log_path, "w") as handle:
        completed = subprocess.run(
            common, cwd=REPO, env=env, stdout=handle,
            stderr=subprocess.STDOUT,
        )
    if completed.returncode != 0:
        raise Failure(
            "condition {} exited with code {} (see {}). Nothing is compared when "
            "a run fails.".format(name, completed.returncode, log_path)
        )
    print("    {} exit=0".format(name))
    return {
        "dir": out_dir,
        "log": log_path,
        "episodes_csv": os.path.join(out_dir, "episodes.csv"),
        "results": os.path.join(out_dir, "results"),
    }


# ---------------------------------------------------------------------------
# artifact discovery and preconditions
# ---------------------------------------------------------------------------


def require_file(report, label, path):
    exists = os.path.exists(path)
    size = os.path.getsize(path) if exists else 0
    return report.check(
        "artifact present and non-empty: {}".format(label),
        exists and size > 0,
        "{} ({} bytes)".format(path, size),
    )


def find_model_dir(results_root):
    """The deepest directory holding the saved .th checkpoints."""
    models = os.path.join(results_root, "models")
    if not os.path.isdir(models):
        return None
    candidates = []
    for root, _dirs, files in os.walk(models):
        if any(f.endswith(".th") for f in files):
            candidates.append(root)
    return sorted(candidates)[-1] if candidates else None


#: Sacred's FileStorageObserver is wired to <repo>/results/sacred in src/main.py
#: and ignores local_results_path, so the run directory is identified by
#: snapshotting that tree before and after each launch rather than guessed.
SACRED_ROOT = os.path.join(REPO, "results", "sacred", "qmix", ENV_KEY)


def sacred_snapshot():
    if not os.path.isdir(SACRED_ROOT):
        return set()
    return set(os.listdir(SACRED_ROOT))


def find_new_sacred_metrics(before):
    """metrics.json of the run directory that appeared since ``before``."""
    new = sorted(sacred_snapshot() - before,
                 key=lambda name: int(name) if name.isdigit() else -1)
    for name in reversed(new):
        candidate = os.path.join(SACRED_ROOT, name, "metrics.json")
        if os.path.exists(candidate):
            return candidate
    return None


# ---------------------------------------------------------------------------
# comparisons
# ---------------------------------------------------------------------------


def compare_episode_csv(report, a_path, b_path):
    """Actions/observations are not in this file, but every stored per-episode
    aggregate is: returns, lengths, per-agent rewards and the Stage A counters."""
    with open(a_path) as handle:
        a_rows = list(csv.DictReader(handle))
    with open(b_path) as handle:
        b_rows = list(csv.DictReader(handle))

    ok = report.require_nonempty("episode CSV has rows", len(a_rows))
    ok &= report.check("episode CSV row counts match", len(a_rows) == len(b_rows),
                       "off={} active={}".format(len(a_rows), len(b_rows)))
    if not ok:
        return False
    ok &= report.check(
        "episode CSV columns match",
        list(a_rows[0].keys()) == list(b_rows[0].keys()),
    )
    mismatches = []
    for index, (a_row, b_row) in enumerate(zip(a_rows, b_rows)):
        for column in a_row:
            if a_row[column] != b_row.get(column):
                mismatches.append((index, column, a_row[column], b_row.get(column)))
    ok &= report.check(
        "episode CSV values identical (raw rewards, returns, lengths, counters)",
        not mismatches,
        "{} mismatching cells{}".format(
            len(mismatches), "" if not mismatches else "; first " + str(mismatches[0])),
    )
    with open(a_path, "rb") as handle:
        a_bytes = handle.read()
    with open(b_path, "rb") as handle:
        b_bytes = handle.read()
    ok &= report.check("episode CSV byte-identical", a_bytes == b_bytes)
    return ok


def compare_learner_stats(report, a_metrics, b_metrics):
    with open(a_metrics) as handle:
        a = json.load(handle)
    with open(b_metrics) as handle:
        b = json.load(handle)

    compared = 0
    mismatches = []
    for key in sorted(set(a) | set(b)):
        if not any(key.startswith(stat) for stat in LEARNER_STATS):
            continue
        a_values = a.get(key, {}).get("values")
        b_values = b.get(key, {}).get("values")
        if a_values is None or b_values is None:
            mismatches.append((key, "missing in one run"))
            continue
        compared += 1
        if a_values != b_values:
            mismatches.append((key, "values differ"))
    ok = report.require_nonempty(
        "learner statistics were actually compared", compared,
        "compared {} metric series".format(compared))
    ok &= report.check(
        "learner statistics identical (loss, grad_norm, Q-values, td_error)",
        not mismatches, str(mismatches[:3]) if mismatches else "")
    return ok


def compare_checkpoints(report, a_dir, b_dir):
    """Load and compare model parameters and optimizer state tensor by tensor."""
    import torch as th

    a_files = sorted(f for f in os.listdir(a_dir) if f.endswith(".th"))
    b_files = sorted(f for f in os.listdir(b_dir) if f.endswith(".th"))
    ok = report.require_nonempty("checkpoint files were saved", len(a_files),
                                 "off={} active={}".format(a_files, b_files))
    ok &= report.check("checkpoint file sets match", a_files == b_files)
    if not ok:
        return False

    tensors_compared = 0
    mismatches = []
    for name in a_files:
        a_state = th.load(os.path.join(a_dir, name), map_location="cpu")
        b_state = th.load(os.path.join(b_dir, name), map_location="cpu")
        for key, a_value, b_value in _walk_state(a_state, b_state, name):
            if isinstance(a_value, th.Tensor) and isinstance(b_value, th.Tensor):
                tensors_compared += 1
                if a_value.shape != b_value.shape or not th.equal(a_value, b_value):
                    mismatches.append(key)
            elif a_value != b_value:
                mismatches.append(key)
    ok &= report.require_nonempty(
        "checkpoint tensors were actually compared", tensors_compared,
        "compared {} tensors across {}".format(tensors_compared, a_files))
    ok &= report.check(
        "model parameters and optimizer state identical",
        not mismatches, str(mismatches[:3]) if mismatches else "")
    return ok


def _walk_state(a, b, prefix):
    """Yield (path, a_value, b_value) over nested state dicts."""
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            path = "{}:{}".format(prefix, key)
            if key not in a or key not in b:
                yield path, a.get(key), b.get(key)
            else:
                for item in _walk_state(a[key], b[key], path):
                    yield item
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            yield prefix, a, b
        else:
            for index, (a_item, b_item) in enumerate(zip(a, b)):
                for item in _walk_state(a_item, b_item, "{}[{}]".format(prefix, index)):
                    yield item
    else:
        yield prefix, a, b


def compare_trajectories_in_process(report, episodes):
    """Compare the replay tensors themselves: obs, state, actions, avail_actions,
    stored training rewards, termination masks and the learner's parameters.

    These live in the EpisodeBatch and never reach disk, so this runs the same
    two conditions in process, using the real runner/MAC/learner stack.
    """
    sys.path.insert(0, SRC)
    sys.path.insert(0, os.path.join(REPO, "tests"))
    from test_selma_shadow_equivalence import run_trace  # noqa: E402

    off_trace, off_params, _ = run_trace(
        n_episodes=episodes, seed=DEV_SEED, selma_mode="off")
    active_trace, active_params, active_runner = run_trace(
        n_episodes=episodes, seed=DEV_SEED, selma_mode="active",
        selma_provider="deterministic_mock", selma_variant_id=FROZEN_VARIANT,
        selma_beta=0.0,
    )

    ok = report.require_nonempty("replay tensors were actually compared",
                                 len(off_trace), "{} episodes".format(len(off_trace)))
    keys = ("actions", "obs", "state", "avail", "reward", "terminated", "filled")
    for key in keys:
        differing = [
            index for index, (a, b) in enumerate(zip(off_trace, active_trace))
            if a[key] != b[key]
        ]
        ok &= report.check(
            "replay tensor identical: {}".format(key), not differing,
            "" if not differing else "episodes {}".format(differing[:3]))
    ok &= report.check(
        "stored training rewards identical (value by value)",
        all(a["reward_values"] == b["reward_values"]
            for a, b in zip(off_trace, active_trace)))
    ok &= report.check("learner parameters identical after training",
                       off_params == active_params)
    return ok, active_runner


def report_counters(active_runner):
    """Provider/guidance counters: expected to differ from off. Not learning state."""
    guidance = active_runner.selma_guidance
    print("\nProvider / guidance counters (expected to differ from `off`;")
    print("these are not learning state and are reported, not compared):")
    for label, value in [
        ("provider calls", guidance.provider_call_count),
        ("queries", guidance.query_count),
        ("activations", guidance.activation_count),
        ("replacements", guidance.replacement_count),
        ("expiries", guidance.expiry_count),
        ("expiry reasons", guidance.expiry_reasons),
        ("rejected responses", guidance.rejected_count),
        ("fallback reasons", guidance.fallback_reasons),
        ("events suppressed on terminal transitions", guidance.terminal_suppressed_events),
        ("cumulative unscaled PBRS term", guidance.total_pbrs_unscaled),
        ("cumulative shaping contribution", guidance.total_shaping_contribution),
    ]:
        print("    {:<45} {}".format(label, value))
    return guidance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None,
                        help="output directory (created if absent)")
    parser.add_argument("--episodes", type=int, default=40,
                        help="episodes per condition (default 40)")
    args = parser.parse_args()

    out = args.out or os.path.join(REPO, "analysis", "out", "stage_c_beta_zero")
    os.makedirs(out, exist_ok=True)

    report = Report()
    print("SELMA Stage C -- beta = 0 equivalence verification")
    print("  repo      : {}".format(REPO))
    print("  python    : {}".format(PYTHON))
    print("  seed      : {} (development seed; NOT an experimental seed 1-5)".format(DEV_SEED))
    print("  episodes  : {} per condition".format(args.episodes))
    print("  output    : {}\n".format(out))

    print("1. subprocess training runs (return codes are checked)")
    try:
        before_a = sacred_snapshot()
        a = run_condition("A off", os.path.join(out, "A_off"), args.episodes, [])
        a_metrics = find_new_sacred_metrics(before_a)
        before_b = sacred_snapshot()
        b = run_condition(
            "B active+mock+beta=0", os.path.join(out, "B_active_beta0"), args.episodes,
            [
                "selma_mode=active",
                "selma_provider=deterministic_mock",
                "selma_variant_id={}".format(FROZEN_VARIANT),
                "selma_beta=0.0",
                "selma_guidance_log_csv={}".format(
                    os.path.join(out, "B_active_beta0", "guidance.csv")),
            ],
        )
        b_metrics = find_new_sacred_metrics(before_b)
    except Failure as error:
        print("\nFAILED: {}".format(error))
        return 1
    report.check("both training runs exited 0", True)

    print("\n2. artifact preconditions (absence is never equality)")
    present = True
    present &= require_file(report, "A episodes.csv", a["episodes_csv"])
    present &= require_file(report, "B episodes.csv", b["episodes_csv"])
    present &= require_file(report, "A run.log", a["log"])
    present &= require_file(report, "B run.log", b["log"])
    guidance_csv = os.path.join(b["dir"], "guidance.csv")
    present &= require_file(report, "B guidance.csv", guidance_csv)

    present &= report.check("A sacred metrics.json found", bool(a_metrics), str(a_metrics))
    present &= report.check("B sacred metrics.json found", bool(b_metrics), str(b_metrics))
    a_models = find_model_dir(a["results"])
    b_models = find_model_dir(b["results"])
    present &= report.check("A checkpoint directory found", bool(a_models), str(a_models))
    present &= report.check("B checkpoint directory found", bool(b_models), str(b_models))

    if not present:
        print("\nFAILED: a required artifact is missing or empty; nothing was compared.")
        return 1

    print("\n3. per-episode stored outputs")
    compare_episode_csv(report, a["episodes_csv"], b["episodes_csv"])

    print("\n4. learner statistics")
    compare_learner_stats(report, a_metrics, b_metrics)

    print("\n5. model parameters and optimizer state (loaded tensors, not filenames)")
    compare_checkpoints(report, a_models, b_models)

    print("\n6. replay tensors: obs, state, actions, avail_actions, rewards, masks")
    _ok, active_runner = compare_trajectories_in_process(report, args.episodes)

    print("\n7. beta = 0 guidance log invariants")
    with open(guidance_csv) as handle:
        rows = list(csv.DictReader(handle))
    report.require_nonempty("guidance log has rows", len(rows))
    report.check("every shaping contribution is exactly 0.0",
                 all(float(r["shaping_contribution"]) == 0.0 for r in rows))
    report.check("training reward equals raw reward on every transition",
                 all(r["training_reward"] == r["raw_team_reward"] for r in rows))
    unscaled_nonzero = sum(1 for r in rows if float(r["pbrs_term_unscaled"]) != 0.0)
    report.check(
        "the unscaled PBRS term is still measured (guidance really ran)",
        unscaled_nonzero > 0, "{} non-zero unscaled terms".format(unscaled_nonzero))
    report.check("no provider query on any stored-terminal transition",
                 all(r["query_triggered"] == "0"
                     for r in rows if r["stored_terminated"] == "1"))

    guidance = report_counters(active_runner)
    if guidance.provider_call_count == 0:
        report.check("the guidance machinery was exercised at all", False,
                     "zero provider calls: this run proves nothing about beta=0")

    print("\n" + "=" * 72)
    if report.failed:
        print("RESULT: FAIL -- {} check(s) failed:".format(len(report.failed)))
        for name in report.failed:
            print("  - {}".format(name))
        return 1
    print("RESULT: PASS -- {} checks, every learning-relevant artifact identical."
          .format(len(report.rows)))
    print("beta = 0 with a live deterministic-mock provider is exactly unshaped QMIX.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
