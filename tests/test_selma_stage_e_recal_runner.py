"""SELMA Stage E: the parallel recalibration controller, with fake children.

Nothing here launches a real training run or touches the network. Every test
replaces the child process, so isolation, wave sequencing, concurrency, caps,
failure handling and credential hygiene are exercised offline.

Run:  PYTHONPATH=src pytest tests/test_selma_stage_e_recal_runner.py -v
"""

import csv
import json
import os
import sys
import threading

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))

import selma_stage_e_recal_runner as runner  # noqa: E402
from selma.guidance import GuidanceCsvWriter, SelmaDecisionCsvWriter  # noqa: E402

SENTINEL_KEY = "sk-recal-sentinel-DO-NOT-LEAK-0123456789"
MODEL = runner.FROZEN_MODEL


# ==========================================================================
# 1. the frozen matrix
# ==========================================================================


def test_the_matrix_is_exactly_six_frozen_conditions(tmp_path):
    matrix = runner.frozen_matrix(str(tmp_path))
    assert len(matrix) == 6
    assert sorted({c["beta"] for c in matrix}) == [0.00, 0.05, 0.10]
    assert sorted({c["seed"] for c in matrix}) == [91, 92]
    assert {(c["beta"], c["seed"]) for c in matrix} == {
        (b, s) for b in (0.00, 0.05, 0.10) for s in (91, 92)}


def test_every_condition_carries_the_frozen_configuration(tmp_path):
    for condition in runner.frozen_matrix(str(tmp_path)):
        assert condition["t_max"] == 1500000
        assert condition["env_key"] == "lbforaging:Foraging-8x8-2p-2f-coop-v3"
        assert condition["time_limit"] == 50
        assert condition["algorithm"] == "qmix"
        assert condition["variant"] == "wf15_m3_t5_ibon"
        assert condition["model"] == MODEL
        assert condition["ttl"] == 15
        assert condition["live_call_cap"] == 13500
        assert condition["cost_cap_usd"] == 3.49
        assert condition["test_interval"] == 50000
        assert condition["test_nepisode"] == 100


@pytest.mark.parametrize("beta", [0.25, 0.01, 0.5, 1.0])
def test_a_beta_outside_the_frozen_list_is_refused(beta, tmp_path):
    with pytest.raises(runner.RunnerError):
        runner.build_condition(beta, 91, str(tmp_path))


def test_beta_0p25_is_named_as_excluded_by_the_design(tmp_path):
    with pytest.raises(runner.RunnerError) as error:
        runner.build_condition(0.25, 91, str(tmp_path))
    assert "excluded by the frozen design" in str(error.value)


@pytest.mark.parametrize("seed", [99, 0, 1, 100, 12345])
def test_a_seed_outside_the_frozen_list_is_refused(seed, tmp_path):
    with pytest.raises(runner.RunnerError):
        runner.build_condition(0.05, seed, str(tmp_path))


def test_seed_99_is_refused_as_detector_heldout(tmp_path):
    with pytest.raises(runner.RunnerError) as error:
        runner.build_condition(0.05, 99, str(tmp_path))
    assert "held-out" in str(error.value)


def test_the_command_forwards_the_caps_and_frozen_flags_exactly(tmp_path):
    condition = runner.build_condition(0.05, 92, str(tmp_path))
    joined = " ".join(runner.build_command(condition, python="/py"))
    for expected in ("--config=qmix", "seed=92", "t_max=1500000",
                     "selma_beta=0.05", "selma_live_call_cap=13500",
                     "selma_cost_cap_usd=3.49", "selma_mode=active",
                     "selma_variant_id=wf15_m3_t5_ibon",
                     "selma_provider=openai", "use_cuda=False",
                     "common_reward=True", "env_args.time_limit=50",
                     "selma_provider_max_output_tokens=512",
                     "selma_provider_timeout_s=30.0",
                     "selma_provider_max_retries=1",
                     "selma_cache_mode=read_write",
                     "test_interval=50000", "test_nepisode=100",
                     "selma_price_table_date=2026-08-17"):
        assert expected in joined, expected


def test_the_planned_global_cost_cannot_exceed_the_authorised_maximum():
    planned = runner.PER_CONDITION_COST_CAP_USD * 6
    assert planned <= runner.GLOBAL_COST_CAP_USD
    assert runner.GLOBAL_COST_CAP_USD == 20.94
    assert runner.PER_CONDITION_CALL_CAP * 6 == runner.GLOBAL_CALL_CAP


# ==========================================================================
# 2. waves and concurrency
# ==========================================================================


def test_there_are_three_waves_of_two_in_the_frozen_order(tmp_path):
    plan = runner.waves(str(tmp_path))
    assert [[c["name"] for c in wave] for wave in plan] == [
        ["beta_0p00_seed91", "beta_0p00_seed92"],
        ["beta_0p05_seed91", "beta_0p05_seed92"],
        ["beta_0p10_seed91", "beta_0p10_seed92"],
    ]


def test_max_concurrency_is_two():
    assert runner.MAX_CONCURRENCY == 2


def test_every_wave_holds_at_most_two_conditions(tmp_path):
    for wave in runner.waves(str(tmp_path)):
        assert len(wave) <= runner.MAX_CONCURRENCY


def test_a_wave_larger_than_two_is_refused(tmp_path):
    matrix = runner.frozen_matrix(str(tmp_path))
    with pytest.raises(runner.RunnerError) as error:
        runner.run_wave(matrix[:3], env={})
    assert "max concurrency" in str(error.value)


# ==========================================================================
# 3. isolation
# ==========================================================================


def test_no_two_conditions_share_any_writable_path(tmp_path):
    matrix = runner.frozen_matrix(str(tmp_path))
    count = runner.assert_isolated(matrix)
    assert count == 6 * len(runner.writable_paths(matrix[0]))


def test_shared_writable_paths_are_detected(tmp_path):
    matrix = runner.frozen_matrix(str(tmp_path))
    matrix[1]["cache_path"] = matrix[0]["cache_path"]
    with pytest.raises(runner.RunnerError) as error:
        runner.assert_isolated(matrix)
    assert "share the writable path" in str(error.value)


def test_each_condition_has_its_own_cache_logs_and_results(tmp_path):
    matrix = runner.frozen_matrix(str(tmp_path))
    for key in ("cache_path", "decision_log", "guidance_log", "episode_log",
                "results_path", "run_log", "manifest", "done_marker"):
        assert len({c[key] for c in matrix}) == 6, key


def test_an_existing_non_empty_output_directory_is_refused(tmp_path):
    condition = runner.build_condition(0.10, 91, str(tmp_path))
    os.makedirs(condition["dir"], exist_ok=True)
    with open(os.path.join(condition["dir"], "old.csv"), "w") as handle:
        handle.write("previous evidence\n")
    with pytest.raises(runner.RunnerError) as error:
        runner.assert_empty_output(condition)
    assert "never overwritten" in str(error.value)


def test_an_empty_output_directory_is_acceptable(tmp_path):
    condition = runner.build_condition(0.10, 92, str(tmp_path))
    os.makedirs(condition["dir"], exist_ok=True)
    runner.assert_empty_output(condition)


# ==========================================================================
# 4. fake children: sequencing, failure handling, concurrency ceiling
# ==========================================================================


def _decision_rows(n=100, *, exhausted=0, model=MODEL, activated=1,
                   physically_valid=1, plan_id="plan_0", greedy=0):
    rows = []
    for i in range(n):
        is_exhausted = i >= n - exhausted
        row = {c: "" for c in SelmaDecisionCsvWriter.COLUMNS}
        row.update({
            "t_env": (i + 1) * 10, "condition": "c", "seed": 91,
            "episode_id": i, "episode_step": 1, "greedy_eval": greedy,
            "beta": 0.05, "event_id": "e%d" % i, "decision": "multiple",
            "candidate_plan_count": 6, "recipient_count": 2,
            "key_digest": "d%060d" % i,
            "cache_result": "miss",
            "provider_considered": 1,
            "live_attempts": 0 if is_exhausted else 1, "retries": 0,
            "outcome": "budget_exhausted" if is_exhausted else "activated",
            "selected_plan_id": "" if is_exhausted else plan_id,
            "validator_valid": 0 if is_exhausted else 1,
            "validator_reason": "", "physically_valid": 0 if is_exhausted else physically_valid,
            "activated": 0 if is_exhausted else activated,
            "tokens_input": "n/a" if is_exhausted else 877,
            "tokens_cached_input": "n/a" if is_exhausted else 0,
            "tokens_output": "n/a" if is_exhausted else 25,
            "tokens_reasoning": 0,
            "returned_model": "" if is_exhausted else model,
            "latency_s": "n/a" if is_exhausted else 0.8,
            "request_id": "r", "detector_events_this_transition": 1,
        })
        rows.append(row)
    return rows


def _guidance_rows(t_max, *, step=50000, terminal_phi=0.0, reward_error=False,
                   greedy=0):
    rows = []
    for i in range(0, t_max + 1, step):
        row = {c: "" for c in GuidanceCsvWriter.COLUMNS}
        row.update({
            "t_env": i, "condition": "c", "seed": 91, "episode_id": i // step,
            "episode_step": 0, "beta": 0.05, "gamma": 0.99,
            "raw_team_reward": 0.0,
            "training_reward": 1.0 if reward_error else 0.0,
            "shaping_contribution": 0.0, "pbrs_term_unscaled": 0.0,
            "phi_before": 0.0, "phi_after_effective": terminal_phi,
            "stored_terminated": 1, "terminal_cause": "time_limit",
            "detector_events": 1, "greedy_eval": greedy,
        })
        rows.append(row)
    return rows


def _write_condition_artifacts(condition, *, decisions=None, guidance=None,
                               t_max=None, log="ok\n"):
    os.makedirs(condition["dir"], exist_ok=True)
    with open(condition["decision_log"], "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SelmaDecisionCsvWriter.COLUMNS)
        writer.writeheader()
        writer.writerows(decisions if decisions is not None else _decision_rows())
    with open(condition["guidance_log"], "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=GuidanceCsvWriter.COLUMNS)
        writer.writeheader()
        writer.writerows(guidance if guidance is not None
                         else _guidance_rows(t_max or condition["t_max"]))
    with open(condition["episode_log"], "w") as handle:
        handle.write("t_env,test_mode,team_return\n1,1,0.0\n")
    with open(condition["run_log"], "w") as handle:
        handle.write(log)


class FakeChildren:
    """Replaces Popen; records concurrency and writes plausible artifacts."""

    def __init__(self, *, exit_codes=None, artifacts=None):
        self.exit_codes = exit_codes or {}
        self.artifacts = artifacts or {}
        self.launched = []
        self.live = 0
        self.max_live = 0
        self._lock = threading.Lock()

    def install(self, monkeypatch, root):
        by_log = {}
        for condition in runner.frozen_matrix(root):
            by_log[condition["decision_log"]] = condition
        real_popen = runner.subprocess.Popen

        def fake_popen(command, *args, **kwargs):
            target = next((a.split("=", 1)[1] for a in command
                           if isinstance(a, str)
                           and a.startswith("selma_decision_log_csv=")), None)
            if target is None:
                # not a training launch (e.g. the controller's own git call)
                return real_popen(command, *args, **kwargs)
            condition = by_log[target]
            self.launched.append(condition["name"])
            with self._lock:
                self.live += 1
                self.max_live = max(self.max_live, self.live)
            kwargs = self.artifacts.get(condition["name"], {})
            _write_condition_artifacts(condition, **kwargs)
            return _FakeProcess(self.exit_codes.get(condition["name"], 0), self)

        monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)
        monkeypatch.setattr(runner, "assert_no_conflicting_process", lambda: 0)
        monkeypatch.setattr(runner, "free_gib", lambda *a, **k: 500.0)


class _FakeProcess:
    def __init__(self, exit_code, parent):
        self.pid = 10000 + abs(hash(exit_code)) % 1000
        self._exit = exit_code
        self._parent = parent

    def wait(self):
        with self._parent._lock:
            self._parent.live -= 1
        return self._exit


@pytest.fixture
def live(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", SENTINEL_KEY)

    def go(children, extra=()):
        root = str(tmp_path / "runs")
        children.install(monkeypatch, root)
        report_path = str(tmp_path / "report.json")
        code = runner.main(["--confirm-live", "--root", root,
                            "--report", report_path, "--stagger-s", "0",
                            *extra])
        with open(report_path) as handle:
            return code, json.load(handle)

    return go


def test_all_six_conditions_run_in_three_waves_of_two(live):
    children = FakeChildren()
    code, report = live(children)
    assert code == 0
    assert report["status"] == "completed"
    assert children.launched == [
        "beta_0p00_seed91", "beta_0p00_seed92",
        "beta_0p05_seed91", "beta_0p05_seed92",
        "beta_0p10_seed91", "beta_0p10_seed92"]
    assert report["conditions_completed"] == 6


def test_concurrency_never_exceeds_two(live):
    children = FakeChildren()
    live(children)
    assert children.max_live <= runner.MAX_CONCURRENCY


def test_a_failing_condition_stops_the_next_wave(live):
    children = FakeChildren(exit_codes={"beta_0p00_seed91": 1})
    code, report = live(children)
    assert code != 0
    assert report["status"] == "stopped_failed_condition"
    assert children.launched == ["beta_0p00_seed91", "beta_0p00_seed92"]


def test_a_failing_condition_lets_its_companion_finish(live):
    children = FakeChildren(exit_codes={"beta_0p00_seed91": 1})
    _code, report = live(children)
    names = [r["name"] for r in report["conditions"]]
    assert "beta_0p00_seed92" in names
    companion = next(r for r in report["conditions"]
                     if r["name"] == "beta_0p00_seed92")
    assert companion["exit_code"] == 0
    assert companion["problems"] == []


def test_nothing_is_rerun_automatically(live):
    children = FakeChildren(exit_codes={"beta_0p05_seed92": 1})
    _code, _report = live(children)
    assert children.launched.count("beta_0p05_seed92") == 1


def test_a_partial_run_is_not_marked_complete(live, tmp_path):
    short = _guidance_rows(500000)          # far short of 1.5M
    children = FakeChildren(artifacts={"beta_0p00_seed91": {"guidance": short}})
    code, report = live(children)
    assert code != 0
    record = next(r for r in report["conditions"] if r["name"] == "beta_0p00_seed91")
    assert any("horizon not reached" in p for p in record["problems"])
    assert not os.path.exists(
        runner.build_condition(0.0, 91, str(tmp_path / "runs"))["done_marker"])


def test_a_completed_condition_writes_a_marker_and_manifest(live, tmp_path):
    live(FakeChildren())
    for beta, seed in ((0.0, 91), (0.10, 92)):
        condition = runner.build_condition(beta, seed, str(tmp_path / "runs"))
        assert os.path.exists(condition["done_marker"])
        assert os.path.exists(condition["manifest"])


# ==========================================================================
# 5. operational gates -- and never performance
# ==========================================================================


def test_low_coverage_fails_the_condition(live):
    rows = _decision_rows(100, exhausted=10)          # 90 % coverage
    children = FakeChildren(artifacts={"beta_0p00_seed91": {"decisions": rows}})
    code, report = live(children)
    assert code != 0
    record = next(r for r in report["conditions"] if r["name"] == "beta_0p00_seed91")
    assert any("guidance coverage" in p for p in record["problems"])
    assert record["checks"]["guidance_coverage"] == pytest.approx(0.90)


def test_coverage_at_the_threshold_passes(live):
    rows = _decision_rows(100, exhausted=5)           # exactly 95 %
    children = FakeChildren(artifacts={"beta_0p00_seed91": {"decisions": rows}})
    code, _report = live(children)
    assert code == 0


def test_model_substitution_fails_the_condition(live):
    rows = _decision_rows(50, model="gpt-other-2020-01-01")
    children = FakeChildren(artifacts={"beta_0p00_seed92": {"decisions": rows}})
    code, report = live(children)
    assert code != 0
    record = next(r for r in report["conditions"] if r["name"] == "beta_0p00_seed92")
    assert any("substitution" in p for p in record["problems"])


def test_physically_invalid_activation_fails_the_condition(live):
    rows = _decision_rows(50, physically_valid=0)
    children = FakeChildren(artifacts={"beta_0p00_seed91": {"decisions": rows}})
    code, report = live(children)
    assert code != 0
    record = next(r for r in report["conditions"] if r["name"] == "beta_0p00_seed91")
    assert any("physically invalid" in p for p in record["problems"])


def test_an_invented_plan_id_fails_the_condition(live):
    rows = _decision_rows(50, plan_id="")
    children = FakeChildren(artifacts={"beta_0p00_seed91": {"decisions": rows}})
    code, report = live(children)
    assert code != 0
    record = next(r for r in report["conditions"] if r["name"] == "beta_0p00_seed91")
    assert any("invented plan" in p for p in record["problems"])


def test_a_greedy_evaluation_call_fails_the_condition(live):
    rows = _decision_rows(50, greedy=1)
    children = FakeChildren(artifacts={"beta_0p00_seed91": {"decisions": rows}})
    code, report = live(children)
    assert code != 0
    record = next(r for r in report["conditions"] if r["name"] == "beta_0p00_seed91")
    assert any("greedy evaluation" in p for p in record["problems"])


def test_a_reward_reconciliation_error_fails_the_condition(live):
    bad = _guidance_rows(1500000, reward_error=True)
    children = FakeChildren(artifacts={"beta_0p00_seed91": {"guidance": bad}})
    code, report = live(children)
    assert code != 0
    record = next(r for r in report["conditions"] if r["name"] == "beta_0p00_seed91")
    assert any("reward reconciliation" in p for p in record["problems"])


def test_a_non_zero_terminal_phi_fails_the_condition(live):
    bad = _guidance_rows(1500000, terminal_phi=-0.5)
    children = FakeChildren(artifacts={"beta_0p00_seed91": {"guidance": bad}})
    code, report = live(children)
    assert code != 0
    record = next(r for r in report["conditions"] if r["name"] == "beta_0p00_seed91")
    assert any("terminal" in p for p in record["problems"])


def test_the_attempt_cap_is_enforced_in_verification(tmp_path):
    condition = runner.build_condition(0.05, 91, str(tmp_path))
    rows = _decision_rows(20)
    for row in rows:
        row["live_attempts"] = 1000            # 20 000 total, over the cap
    _write_condition_artifacts(condition, decisions=rows)
    result = runner.verify_condition(condition, 0)
    assert any("exceed cap" in p for p in result["problems"])


def test_the_cost_cap_is_enforced_in_verification(tmp_path):
    condition = runner.build_condition(0.05, 91, str(tmp_path))
    rows = _decision_rows(50)
    for row in rows:
        row["tokens_input"] = 100000000
    _write_condition_artifacts(condition, decisions=rows)
    result = runner.verify_condition(condition, 0)
    assert any("cost" in p for p in result["problems"])


def test_wave_verification_reads_no_performance_value():
    """Structural: the controller must not touch a return or success rate."""
    source = open(runner.__file__).read()
    for forbidden in ("team_return", "test_return", "success_rate",
                      "return_mean", "mean_return", "auc", "AUC"):
        assert forbidden not in source, forbidden


def test_the_report_declares_performance_was_not_inspected(live):
    _code, report = live(FakeChildren())
    assert report["performance_inspected_during_execution"] is False


# ==========================================================================
# 6. credentials
# ==========================================================================


def test_the_api_key_is_never_in_a_command(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", SENTINEL_KEY)
    condition = runner.build_condition(0.10, 91, str(tmp_path))
    joined = " ".join(runner.build_command(condition))
    assert SENTINEL_KEY not in joined
    assert "OPENAI_API_KEY" not in joined
    assert "sk-" not in joined


def test_the_child_inherits_the_key_but_it_is_never_recorded(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", SENTINEL_KEY)
    env = runner.child_env()
    assert env["OPENAI_API_KEY"] == SENTINEL_KEY
    condition = runner.build_condition(0.0, 91, str(tmp_path))
    assert SENTINEL_KEY not in json.dumps(condition)


def test_no_manifest_or_report_may_contain_a_credential(live, tmp_path):
    _code, report = live(FakeChildren())
    blob = json.dumps(report)
    assert SENTINEL_KEY not in blob and "sk-" not in blob
    for beta, seed in ((0.0, 91), (0.05, 92), (0.10, 91)):
        condition = runner.build_condition(beta, seed, str(tmp_path / "runs"))
        text = open(condition["manifest"]).read()
        assert SENTINEL_KEY not in text
        assert "sk-" not in text
        assert "OPENAI_API_KEY" not in text


def test_a_manifest_containing_a_credential_is_refused(tmp_path):
    condition = runner.build_condition(0.0, 91, str(tmp_path))
    os.makedirs(condition["dir"], exist_ok=True)
    with pytest.raises(runner.RunnerError):
        runner.write_manifest(condition, {"leak": SENTINEL_KEY})


def test_a_credential_in_the_run_log_fails_the_condition(tmp_path):
    condition = runner.build_condition(0.0, 91, str(tmp_path))
    _write_condition_artifacts(condition, log="oops sk-abcdefghijklmnop\n")
    result = runner.verify_condition(condition, 0)
    assert any("credential" in p for p in result["problems"])


def test_missing_credentials_stop_before_any_launch(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(runner, "assert_no_conflicting_process", lambda: 0)
    with pytest.raises(SystemExit):
        runner.main(["--confirm-live", "--root", str(tmp_path / "r"),
                     "--report", str(tmp_path / "r.json")])


def test_a_dry_run_launches_nothing(tmp_path):
    root = str(tmp_path / "runs")
    report_path = str(tmp_path / "dry.json")
    assert runner.main(["--dry-run", "--root", root,
                        "--report", report_path]) == 0
    assert json.load(open(report_path))["status"] == "dry_run"
    assert not os.path.exists(root)


def test_a_second_training_process_is_refused(monkeypatch):
    class Result:
        stdout = "/py src/main.py --config=qmix --env-config=gymma\n"

    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: Result())
    with pytest.raises(runner.RunnerError) as error:
        runner.assert_no_conflicting_process()
    assert "already running" in str(error.value)
