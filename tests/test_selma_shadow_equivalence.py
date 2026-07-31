"""SELMA Stage B: off-vs-shadow equivalence and shadow-mode isolation.

Drives the *real* EpisodeRunner, MAC, action selector and QMIX learner in
process (mirroring src/run.py::run_sequential) and requires exact equality
between ``selma_mode=off`` and ``selma_mode=shadow`` for every trajectory and
replay tensor. The separately configured shadow CSV is the only permitted
difference.

Run with:  PYTHONPATH=src pytest tests/test_selma_shadow_equivalence.py -v
"""
import hashlib
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch as th
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

SRC = os.path.join(os.path.dirname(__file__), "..", "src")

import lbforaging  # noqa: E402,F401
from components.episode_buffer import ReplayBuffer  # noqa: E402
from components.transforms import OneHot  # noqa: E402
from controllers import REGISTRY as mac_REGISTRY  # noqa: E402
from learners import REGISTRY as le_REGISTRY  # noqa: E402
from runners import REGISTRY as r_REGISTRY  # noqa: E402
from selma.shadow_log import SHADOW_CSV_COLUMNS  # noqa: E402

KEY = "lbforaging:Foraging-8x8-2p-2f-coop-v3"


class _NullLogger:
    """Logger stub: run_sequential's logger is only used for stat plumbing."""

    def __init__(self):
        self.console_logger = SimpleNamespace(
            info=lambda *a, **k: None, warning=lambda *a, **k: None
        )
        self.stats = {}
        self.use_tb = False
        self.use_wandb = False
        self.use_sacred = False
        self.use_hdf = False

    def log_stat(self, *args, **kwargs):
        pass

    def print_recent_stats(self):
        pass


def _load_yaml(*parts):
    with open(os.path.join(SRC, "config", *parts)) as handle:
        return yaml.safe_load(handle)


def build_args(**overrides):
    config = _load_yaml("default.yaml")
    config.update(_load_yaml("envs", "gymma.yaml"))
    config.update(_load_yaml("algs", "qmix.yaml"))
    config["env_args"] = dict(config.get("env_args") or {})
    config["env_args"].update({"key": KEY, "time_limit": 50})
    config.update(
        {
            "use_cuda": False,
            "device": "cpu",
            "buffer_cpu_only": True,
            "seed": 91,
            "batch_size": 2,
            "batch_size_run": 1,
            "buffer_size": 8,
            "t_max": 400,
            "target_update_interval_or_tau": 20,
            "use_tensorboard": False,
            "save_model": False,
            "checkpoint_path": "",
            "evaluate": False,
            "render": False,
            "runner": "episode",
        }
    )
    config.update(overrides)
    config["env_args"]["seed"] = config["seed"]
    return SimpleNamespace(**config)


def build_stack(args):
    """Replicate run_sequential's construction, using the real components."""
    np.random.seed(args.seed)
    th.manual_seed(args.seed)

    runner = r_REGISTRY[args.runner](args=args, logger=_NullLogger())
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]

    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {
            "vshape": (env_info["n_actions"],),
            "group": "agents",
            "dtype": th.int,
        },
        "terminated": {"vshape": (1,), "dtype": th.uint8},
        "reward": {"vshape": (1,)},
    }
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}
    buffer = ReplayBuffer(
        scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
        preprocess=preprocess, device="cpu",
    )
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, _NullLogger(), args)
    return runner, buffer, mac, learner


def digest(tensor):
    return hashlib.sha256(
        np.ascontiguousarray(tensor.detach().cpu().numpy()).tobytes()
    ).hexdigest()


def run_trace(n_episodes=8, test_every=4, **overrides):
    """Run a short deterministic training loop and fingerprint everything."""
    args = build_args(**overrides)
    runner, buffer, mac, learner = build_stack(args)

    trace = []
    for episode in range(n_episodes):
        batch = runner.run(test_mode=False)
        trace.append(
            {
                "episode": episode,
                "actions": digest(batch["actions"]),
                "obs": digest(batch["obs"]),
                "state": digest(batch["state"]),
                "avail": digest(batch["avail_actions"]),
                "reward": digest(batch["reward"]),
                "terminated": digest(batch["terminated"]),
                "filled": digest(batch["filled"]),
                "reward_values": batch["reward"].detach().cpu().numpy().tolist(),
                "t_env": runner.t_env,
            }
        )
        buffer.insert_episode_batch(batch)
        if buffer.can_sample(args.batch_size):
            sample = buffer.sample(args.batch_size)
            max_t = sample.max_t_filled()
            sample = sample[:, :max_t]
            learner.train(sample, runner.t_env, episode)
        if test_every and (episode + 1) % test_every == 0:
            runner.run(test_mode=True)

    params = hashlib.sha256(
        b"".join(
            np.ascontiguousarray(p.detach().cpu().numpy()).tobytes()
            for p in learner.mac.parameters()
        )
    ).hexdigest()
    return trace, params, runner


# ==========================================================================
# 28: exact deterministic off-vs-shadow equivalence
# ==========================================================================


def test_off_vs_shadow_trajectory_and_replay_are_exactly_equal(tmp_path):
    shadow_csv = tmp_path / "shadow.csv"
    off_trace, off_params, off_runner = run_trace(selma_mode="off")
    shadow_trace, shadow_params, shadow_runner = run_trace(
        selma_mode="shadow", selma_shadow_csv=str(shadow_csv)
    )

    assert off_runner.selma_bank is None
    assert shadow_runner.selma_bank is not None

    assert off_trace == shadow_trace, "trajectory/replay tensors must be identical"
    assert off_params == shadow_params, "learner parameters must be identical"

    # The shadow CSV is the only permitted extra output, and it must exist.
    assert shadow_csv.exists()
    with open(shadow_csv) as handle:
        header = handle.readline().strip().split(",")
    assert header == list(SHADOW_CSV_COLUMNS)


def test_off_vs_shadow_ordinary_csv_is_byte_identical(tmp_path):
    off_csv = tmp_path / "off_ordinary.csv"
    shadow_csv = tmp_path / "shadow_ordinary.csv"
    shadow_rows = tmp_path / "shadow_rows.csv"

    run_trace(selma_mode="off", selma_log_csv=str(off_csv))
    run_trace(
        selma_mode="shadow",
        selma_log_csv=str(shadow_csv),
        selma_shadow_csv=str(shadow_rows),
    )

    assert off_csv.read_bytes() == shadow_csv.read_bytes()
    assert off_csv.stat().st_size > 0


def test_default_selma_mode_is_off_and_creates_no_bank(tmp_path):
    args = build_args()
    assert args.selma_mode == "off"
    assert args.selma_shadow_csv == ""
    runner, _, _, _ = build_stack(args)
    assert runner.selma_bank is None
    assert runner.selma_shadow_writer is None


def test_ordinary_csv_columns_are_unchanged_by_stage_b(tmp_path):
    csv_path = tmp_path / "ordinary.csv"
    run_trace(
        selma_mode="shadow",
        selma_log_csv=str(csv_path),
        selma_shadow_csv=str(tmp_path / "s.csv"),
    )
    header = csv_path.read_text().splitlines()[0]
    assert header == (
        "t_env,test_mode,ep_length,team_return,multi_agent_reward_steps,"
        "single_agent_reward_steps,agent0_return,agent1_return,"
        "successful_load_events,solo_load_events,multi_agent_load_events,"
        "necessary_cooperative_load_events,failed_load_events,failed_load_attempts"
    )


# ==========================================================================
# 23: greedy evaluation isolation
# ==========================================================================


def test_greedy_evaluation_never_updates_detector_state_or_rows(tmp_path):
    shadow_csv = tmp_path / "shadow.csv"
    args = build_args(selma_mode="shadow", selma_shadow_csv=str(shadow_csv))
    runner, _, _, _ = build_stack(args)
    bank = runner.selma_bank

    runner.run(test_mode=False)
    baseline = {
        "episodes": bank.episodes_seen,
        "counters": bank.variant_counters(),
        "rows": len(bank.rows),
    }
    size_after_train = shadow_csv.stat().st_size if shadow_csv.exists() else 0

    for _ in range(5):
        runner.run(test_mode=True)

    assert bank.episodes_seen == baseline["episodes"], "test episodes counted"
    assert bank.variant_counters() == baseline["counters"], "test episodes updated state"
    assert bank.pending_rows == [], "test episodes produced pending rows"
    size_after_test = shadow_csv.stat().st_size if shadow_csv.exists() else 0
    assert size_after_test == size_after_train, "test episodes wrote shadow rows"


# ==========================================================================
# 24: zero provider / query / subgoal / shaping through the real runner
# ==========================================================================


def test_runner_shadow_mode_has_zero_queries_subgoals_and_shaping(tmp_path):
    args = build_args(
        selma_mode="shadow", selma_shadow_csv=str(tmp_path / "shadow.csv")
    )
    runner, _, _, _ = build_stack(args)
    for _ in range(6):
        batch = runner.run(test_mode=False)
    bank = runner.selma_bank
    assert bank.query_count == 0
    assert bank.provider_call_count == 0
    assert bank.active_subgoal_count == 0
    assert bank.total_shaping_reward == 0.0
    # Stored reward must be the raw team reward: an all-integer LBF-normalised
    # reward tensor with no shaping term added anywhere.
    assert th.isfinite(batch["reward"]).all()


def test_shadow_mode_writes_rows_and_they_parse(tmp_path):
    shadow_csv = tmp_path / "shadow.csv"
    args = build_args(selma_mode="shadow", selma_shadow_csv=str(shadow_csv))
    runner, _, _, _ = build_stack(args)
    for _ in range(20):
        runner.run(test_mode=False)
    assert shadow_csv.exists()
    import csv

    with open(shadow_csv) as handle:
        rows = list(csv.DictReader(handle))
    assert rows, "expected at least one shadow row from 20 episodes"
    assert {r["row_type"] for r in rows} <= {"raw_candidate", "coalesced_event"}
    for row in rows:
        assert row["resolved_within_5"] in {"resolved", "unresolved", "censored"}
        assert row["resolved_within_10"] in {"resolved", "unresolved", "censored"}


# ==========================================================================
# 29-31: RNG isolation, Stage A counters, and exactly-zero shaping
# ==========================================================================


def test_shadow_mode_leaves_numpy_and_torch_rng_states_identical_to_off(tmp_path):
    """Detector work must not consume training / action-selection randomness."""
    run_trace(selma_mode="off")
    off_np = np.random.get_state()[1].tobytes()
    off_th = th.random.get_rng_state().numpy().tobytes()

    run_trace(
        selma_mode="shadow", selma_shadow_csv=str(tmp_path / "shadow.csv")
    )
    assert np.random.get_state()[1].tobytes() == off_np, "numpy RNG advanced"
    assert th.random.get_rng_state().numpy().tobytes() == off_th, "torch RNG advanced"


def test_stage_a_counters_are_identical_off_vs_shadow(tmp_path):
    """Every Stage A load-event counter column is unchanged, value by value."""
    off_csv = tmp_path / "off.csv"
    shadow_csv = tmp_path / "shadow.csv"
    run_trace(selma_mode="off", selma_log_csv=str(off_csv))
    run_trace(
        selma_mode="shadow",
        selma_log_csv=str(shadow_csv),
        selma_shadow_csv=str(tmp_path / "rows.csv"),
    )
    import csv as _csv

    stage_a = [
        "successful_load_events",
        "solo_load_events",
        "multi_agent_load_events",
        "necessary_cooperative_load_events",
        "failed_load_events",
        "failed_load_attempts",
        "multi_agent_reward_steps",
        "single_agent_reward_steps",
    ]
    off_rows = list(_csv.DictReader(off_csv.open()))
    shadow_rows = list(_csv.DictReader(shadow_csv.open()))
    assert off_rows and len(off_rows) == len(shadow_rows)
    for column in stage_a:
        assert [r[column] for r in off_rows] == [r[column] for r in shadow_rows], column
    # and the counters are not trivially all-zero, so the check has content
    assert any(float(r["failed_load_attempts"]) > 0 for r in off_rows)


def test_shaping_reward_is_exactly_zero_on_every_transition(tmp_path):
    """Stored rewards match ``off`` bit-for-bit; no shaping term exists."""
    off_trace, _, _ = run_trace(selma_mode="off")
    shadow_trace, _, shadow_runner = run_trace(
        selma_mode="shadow", selma_shadow_csv=str(tmp_path / "shadow.csv")
    )
    for off_ep, shadow_ep in zip(off_trace, shadow_trace):
        assert off_ep["reward_values"] == shadow_ep["reward_values"]
        diffs = [
            abs(a - b)
            for row_a, row_b in zip(off_ep["reward_values"], shadow_ep["reward_values"])
            for a, b in zip(np.ravel(row_a).tolist(), np.ravel(row_b).tolist())
        ]
        assert diffs and max(diffs) == 0.0, "per-transition shaping must be exactly 0"
    assert shadow_runner.selma_bank.total_shaping_reward == 0.0
