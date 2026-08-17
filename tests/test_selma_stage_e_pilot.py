"""SELMA Stage E: the beta-calibration orchestrator, with no real training.

Every test replaces the child process with a local fake, so the orchestration --
preregistration guards, isolation, sequencing, caps, secret handling, stop
conditions -- is exercised without a network call or a training run.

Run:  PYTHONPATH=src pytest tests/test_selma_stage_e_pilot.py -v
"""

import csv
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))

import selma_stage_e_beta_pilot as pilot  # noqa: E402
from selma.guidance import SelmaDecisionCsvWriter  # noqa: E402

SENTINEL_KEY = "sk-stage-e-sentinel-DO-NOT-LEAK-0123456789"
MODEL = pilot.FROZEN_MODEL


# ==========================================================================
# 1. the preregistered design cannot be altered
# ==========================================================================


def test_the_four_betas_are_frozen():
    assert pilot.FROZEN_BETAS == (0.0, 0.05, 0.10, 0.25)
    pilot.assert_frozen_design(pilot.FROZEN_BETAS, 91, 500000)


@pytest.mark.parametrize("betas", [
    (0.0, 0.05, 0.10),                 # one removed
    (0.0, 0.05, 0.10, 0.25, 0.5),      # one added
    (0.0, 0.05, 0.15, 0.25),           # one changed
    (0.25, 0.10, 0.05, 0.0),           # reordered
    (0.05, 0.10, 0.25),                # baseline dropped
])
def test_any_other_beta_list_is_refused(betas):
    with pytest.raises(pilot.PilotError) as error:
        pilot.assert_frozen_design(betas, 91, 500000)
    assert "preregistered" in str(error.value)


@pytest.mark.parametrize("seed", [0, 1, 42, 92, 99, 12345])
def test_any_other_seed_is_refused(seed):
    with pytest.raises(pilot.PilotError) as error:
        pilot.assert_frozen_design(pilot.FROZEN_BETAS, seed, 500000)
    assert "seed" in str(error.value)


@pytest.mark.parametrize("t_max", [200000, 499999, 1000000, 2000000])
def test_any_other_t_max_is_refused(t_max):
    with pytest.raises(pilot.PilotError):
        pilot.assert_frozen_design(pilot.FROZEN_BETAS, 91, t_max)


def test_the_cli_refuses_a_changed_design(tmp_path):
    with pytest.raises(pilot.PilotError):
        pilot.main(["--dry-run", "--betas", "0.0,0.5", "--root", str(tmp_path)])
    with pytest.raises(pilot.PilotError):
        pilot.main(["--dry-run", "--seed", "7", "--root", str(tmp_path)])


# ==========================================================================
# 2. the resolved configuration matches the preregistration
# ==========================================================================


def test_every_condition_shares_the_frozen_configuration(tmp_path):
    configs = [pilot.build_config(b, str(tmp_path)) for b in pilot.FROZEN_BETAS]
    assert [c["beta"] for c in configs] == list(pilot.FROZEN_BETAS)
    for config in configs:
        assert config["seed"] == 91
        assert config["t_max"] == 500000
        assert config["env_key"] == "lbforaging:Foraging-8x8-2p-2f-coop-v3"
        assert config["time_limit"] == 50
        assert config["algorithm"] == "qmix"
        assert config["variant"] == "wf15_m3_t5_ibon"
        assert config["provider"] == "openai"
        assert config["model"] == MODEL
        assert config["cache_mode"] == "read_write"
        assert config["live_call_cap"] == 1000
        assert config["cost_cap_usd"] == 0.25
        assert config["use_cuda"] is False
        assert config["common_reward"] is True


def test_conditions_differ_only_in_beta(tmp_path):
    configs = [pilot.build_config(b, str(tmp_path)) for b in pilot.FROZEN_BETAS]
    varying = set()
    for key in configs[0]:
        values = {str(c[key]) for c in configs}
        if len(values) > 1:
            varying.add(key)
    # beta varies, and the per-condition paths derived from it
    assert varying == {"beta", "name", "dir", "cache_path", "guidance_log",
                       "decision_log", "episode_log", "results_path", "run_log"}


def test_each_condition_gets_an_isolated_cache_and_directory(tmp_path):
    configs = [pilot.build_config(b, str(tmp_path)) for b in pilot.FROZEN_BETAS]
    caches = [c["cache_path"] for c in configs]
    dirs = [c["dir"] for c in configs]
    assert len(set(caches)) == 4, "cache entries must never be shared across betas"
    assert len(set(dirs)) == 4


def test_the_command_carries_the_frozen_flags(tmp_path):
    config = pilot.build_config(0.05, str(tmp_path))
    command = pilot.build_command(config, python="/usr/bin/python3")
    joined = " ".join(command)
    for expected in ("--config=qmix", "--env-config=gymma",
                     "seed=91", "t_max=500000", "use_cuda=False",
                     "common_reward=True", "selma_mode=active",
                     "selma_variant_id=wf15_m3_t5_ibon", "selma_provider=openai",
                     "selma_beta=0.05", "selma_cache_mode=read_write",
                     "selma_live_call_cap=1000", "selma_cost_cap_usd=0.25",
                     "selma_provider_timeout_s=30.0",
                     "selma_provider_max_retries=1",
                     "selma_provider_max_output_tokens=512",
                     "env_args.time_limit=50",
                     "selma_price_table_date=2026-08-17"):
        assert expected in joined, expected


def test_the_price_table_is_the_frozen_one():
    assert pilot.PRICE_TABLE.date == "2026-08-17"
    assert pilot.PRICE_TABLE.input_usd_per_mtok == 0.20
    assert pilot.PRICE_TABLE.cached_input_usd_per_mtok == 0.02
    assert pilot.PRICE_TABLE.output_usd_per_mtok == 1.25


# ==========================================================================
# 3. the secret never appears anywhere
# ==========================================================================


def test_the_api_key_is_never_a_command_argument(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", SENTINEL_KEY)
    config = pilot.build_config(0.10, str(tmp_path))
    command = pilot.build_command(config)
    joined = " ".join(command)
    assert SENTINEL_KEY not in joined
    assert "OPENAI_API_KEY" not in joined
    assert "sk-" not in joined


def test_the_key_is_inherited_by_the_child_but_not_recorded(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", SENTINEL_KEY)
    env = pilot.child_env()
    assert env["OPENAI_API_KEY"] == SENTINEL_KEY, "the child needs it"
    config = pilot.build_config(0.0, str(tmp_path))
    assert SENTINEL_KEY not in json.dumps(config)


def test_the_dry_run_report_contains_no_credential(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", SENTINEL_KEY)
    report_path = str(tmp_path / "r.json")
    assert pilot.main(["--dry-run", "--root", str(tmp_path / "runs"),
                       "--report", report_path]) == 0
    text = open(report_path).read()
    assert SENTINEL_KEY not in text
    assert "sk-" not in text
    assert "OPENAI_API_KEY" not in text


# ==========================================================================
# 4. dry run touches nothing
# ==========================================================================


def test_a_dry_run_makes_no_network_call_and_writes_no_run_data(tmp_path):
    """The session-wide socket guard would raise on any connection attempt."""
    root = tmp_path / "runs"
    report = str(tmp_path / "dry.json")
    assert pilot.main(["--dry-run", "--root", str(root), "--report", report]) == 0
    assert not root.exists(), "a dry run must create no condition directory"
    data = json.load(open(report))
    assert data["status"] == "dry_run"
    assert data["conditions"] == []


def test_without_confirm_live_nothing_executes(tmp_path):
    root = tmp_path / "runs"
    report = str(tmp_path / "nc.json")
    assert pilot.main(["--root", str(root), "--report", report]) == 0
    assert json.load(open(report))["status"] == "not_confirmed"
    assert not root.exists()


def test_the_dry_run_lists_all_four_resolved_configurations(tmp_path, capsys):
    pilot.main(["--dry-run", "--root", str(tmp_path)])
    out = capsys.readouterr().out
    for name in ("beta_0p00", "beta_0p05", "beta_0p10", "beta_0p25"):
        assert name in out
    assert "seed        : 91" in out
    assert MODEL in out


# ==========================================================================
# 5. isolation: a condition directory is never reused
# ==========================================================================


def test_an_existing_non_empty_condition_directory_is_refused(tmp_path):
    path = tmp_path / "beta_0p05"
    path.mkdir()
    (path / "guidance.csv").write_text("stale\n")
    with pytest.raises(pilot.PilotError) as error:
        pilot.assert_clean_directory(str(path))
    assert "never re-run" in str(error.value)


def test_an_empty_directory_is_acceptable(tmp_path):
    path = tmp_path / "beta_0p05"
    path.mkdir()
    pilot.assert_clean_directory(str(path))
    assert path.exists()


# ==========================================================================
# 6. execution, sequencing and caps -- with a fake child process
# ==========================================================================


def _decision_row(**over):
    row = {c: "" for c in SelmaDecisionCsvWriter.COLUMNS}
    row.update({
        "t_env": 1, "condition": "c", "seed": 91, "episode_id": 1,
        "episode_step": 1, "greedy_eval": 0, "beta": 0.0, "event_id": "e",
        "decision": "multiple", "candidate_plan_count": 6, "recipient_count": 2,
        "key_digest": "d" * 64, "cache_result": "miss", "provider_considered": 1,
        "live_attempts": 1, "retries": 0, "outcome": "activated",
        "selected_plan_id": "plan_0", "validator_valid": 1,
        "validator_reason": "", "physically_valid": 1, "activated": 1,
        "tokens_input": 800, "tokens_cached_input": 0, "tokens_output": 25,
        "tokens_reasoning": 0, "returned_model": MODEL, "latency_s": 1.0,
        "request_id": "req", "detector_events_this_transition": 1,
    })
    row.update(over)
    return row


class FakeChild:
    """Stands in for the training process; writes a plausible decision log."""

    def __init__(self, *, exit_code=0, rows=5, row_over=None, log_text="ok\n"):
        self.exit_code = exit_code
        self.rows = rows
        self.row_over = row_over or {}
        self.log_text = log_text
        self.launched = []

    def install(self, monkeypatch, configs):
        by_dir = {c["dir"]: c for c in configs}

        def fake_popen(command, cwd=None, env=None, stdout=None, stderr=None):
            target = next(a.split("=", 1)[1] for a in command
                          if a.startswith("selma_decision_log_csv="))
            config = by_dir[os.path.dirname(target)]
            self.launched.append(config["name"])
            with open(config["decision_log"], "w", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=SelmaDecisionCsvWriter.COLUMNS)
                writer.writeheader()
                for _ in range(self.rows):
                    writer.writerow(_decision_row(**self.row_over))
            if stdout is not None:
                stdout.write(self.log_text)
                stdout.flush()
            return _FakeProcess(self.exit_code)

        monkeypatch.setattr(pilot.subprocess, "Popen", fake_popen)
        monkeypatch.setattr(pilot, "assert_no_other_training_process", lambda: 0)


class _FakeProcess:
    def __init__(self, exit_code):
        self.pid = 4242
        self._exit = exit_code

    def wait(self):
        return self._exit


@pytest.fixture
def live(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", SENTINEL_KEY)

    def go(child, extra=()):
        root = str(tmp_path / "runs")
        configs = [pilot.build_config(b, root) for b in pilot.FROZEN_BETAS]
        for config in configs:
            os.makedirs(config["dir"], exist_ok=True)
        child.install(monkeypatch, configs)
        report = str(tmp_path / "report.json")
        code = pilot.main(["--confirm-live", "--root", root, "--report", report,
                           *extra])
        return code, json.load(open(report))

    return go


def test_all_four_conditions_run_sequentially_in_order(live):
    child = FakeChild()
    code, report = live(child)
    assert code == 0
    assert report["status"] == "completed"
    assert child.launched == ["beta_0p00", "beta_0p05", "beta_0p10", "beta_0p25"]
    assert [c["beta"] for c in report["conditions"]] == [0.0, 0.05, 0.10, 0.25]


def test_each_condition_records_pid_times_and_exit_status(live):
    _code, report = live(FakeChild())
    for record in report["conditions"]:
        assert record["exit_code"] == 0
        assert record["pid"] == 4242
        assert record["started_utc"] and record["ended_utc"]
        assert record["wall_clock_s"] >= 0


def test_a_nonzero_exit_stops_later_conditions(live):
    child = FakeChild(exit_code=1)
    code, report = live(child)
    assert code != 0
    assert report["status"] == "stopped_nonzero_exit"
    assert child.launched == ["beta_0p00"], "later conditions must not launch"


def test_a_correctness_failure_stops_later_conditions(live):
    child = FakeChild(log_text="boom\nTraceback (most recent call last):\n")
    code, report = live(child)
    assert code != 0
    assert report["status"] == "stopped_correctness"
    assert child.launched == ["beta_0p00"]


def test_a_model_substitution_stops_the_pilot(live):
    child = FakeChild(row_over={"returned_model": "gpt-other-2020-01-01"})
    code, report = live(child)
    assert code != 0
    assert report["status"] == "stopped_correctness"
    assert any("substitution" in p
               for p in report["conditions"][0]["correctness_problems"])


def test_physically_invalid_guidance_stops_the_pilot(live):
    child = FakeChild(row_over={"activated": 1, "physically_valid": 0})
    code, report = live(child)
    assert code != 0
    assert any("physically invalid" in p
               for p in report["conditions"][0]["correctness_problems"])


def test_a_greedy_evaluation_decision_stops_the_pilot(live):
    child = FakeChild(row_over={"greedy_eval": 1})
    code, report = live(child)
    assert code != 0
    assert any("greedy evaluation" in p
               for p in report["conditions"][0]["correctness_problems"])


def test_ordinary_api_failure_through_the_safe_path_is_not_a_correctness_failure(live):
    """budget_exhausted / refusal / timeout are recorded, not treated as faults."""
    child = FakeChild(row_over={
        "outcome": "budget_exhausted", "activated": 0, "validator_valid": 0,
        "physically_valid": 0, "live_attempts": 0, "selected_plan_id": "",
        "returned_model": "", "tokens_input": "n/a", "tokens_output": "n/a"})
    code, report = live(child)
    assert code == 0
    assert report["status"] == "completed"
    assert child.launched == ["beta_0p00", "beta_0p05", "beta_0p10", "beta_0p25"]
    assert report["any_condition_hit_its_cap"] is True


def test_the_global_budget_stops_further_conditions(live, monkeypatch):
    """A condition that spends the whole global budget stops the pilot."""
    # just above one condition cap, so the first run consumes the headroom
    monkeypatch.setattr(pilot, "GLOBAL_COST_CAP_USD", 0.2505)
    child = FakeChild(rows=5)
    code, report = live(child)
    assert report["status"] == "stopped_global_budget"
    assert len(child.launched) == 1
    skipped = [c for c in report["conditions"] if c.get("skipped")]
    assert skipped and "global budget" in skipped[0]["reason"]


def test_usage_and_cost_are_read_from_the_durable_log(live):
    _code, report = live(FakeChild(rows=5))
    usage = report["conditions"][0]["usage"]
    assert usage["attempts"] == 5
    assert usage["input_tokens"] == 5 * 800
    assert usage["output_tokens"] == 5 * 25
    expected = (5 * 800 * 0.20 + 5 * 25 * 1.25) / 1_000_000.0
    assert usage["cost_usd"] == pytest.approx(expected)
    assert report["cumulative_attempts"] == 20


def test_the_report_declares_that_performance_was_not_inspected(live):
    _code, report = live(FakeChild())
    assert report["not_a_performance_result"] is True
    assert report["performance_inspected_during_execution"] is False


def test_the_orchestrator_never_reads_a_return_or_success_rate():
    """Structural: it configures the episode-log path but never opens it.

    This is what keeps later conditions uninfluenced by earlier ones: if the
    orchestrator cannot see a return, it cannot adapt to one.
    """
    source = open(pilot.__file__).read()
    for forbidden in ("team_return", "test_return", "success_rate",
                      "return_mean", "test_return_mean"):
        assert forbidden not in source, forbidden
    # the episode log is configured for the child, and never read back here
    assert "episode_log" in source
    assert 'open(config["episode_log"]' not in source
    assert "read_condition_usage" in source
    usage_body = source.split("def read_condition_usage")[1].split("\ndef ")[0]
    assert "episode_log" not in usage_body, "usage must come from the decision log"


def test_missing_credentials_stop_before_any_launch(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(pilot, "assert_no_other_training_process", lambda: 0)
    with pytest.raises(SystemExit):
        pilot.main(["--confirm-live", "--root", str(tmp_path / "r"),
                    "--report", str(tmp_path / "r.json")])


def test_a_second_training_process_is_refused(monkeypatch, tmp_path):
    class Result:
        stdout = "/usr/bin/python src/main.py --config=qmix --env-config=gymma\n"

    monkeypatch.setattr(pilot.subprocess, "run", lambda *a, **k: Result())
    with pytest.raises(pilot.PilotError) as error:
        pilot.assert_no_other_training_process()
    assert "already running" in str(error.value)
