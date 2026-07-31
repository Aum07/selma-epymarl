"""SELMA Stage B tests: calibration summary and report generator.

Covers ``ShadowDetectorBank.calibration_summary()`` (the aggregation surface
built on ``variant_report()``) and ``analysis/selma_calibration_report.py``.

Run with:  PYTHONPATH=src pytest tests/test_selma_calibration_report.py -v
"""
import importlib.util
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from selma.detector import (  # noqa: E402
    EPISODE_PHASES,
    RESOLUTION_HORIZONS,
    build_variant_grid,
    episode_phase,
)
from selma.shadow_log import SHADOW_CSV_COLUMNS, ShadowCsvWriter  # noqa: E402

from test_selma_detector import (  # noqa: E402
    FOOD,
    failed_event,
    make_bank,
    only,
    step_info,
)


def _load_generator():
    path = os.path.join(
        os.path.dirname(__file__), "..", "analysis", "selma_calibration_report.py"
    )
    spec = importlib.util.spec_from_file_location("selma_calibration_report", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REPORT = _load_generator()


def drive_stall(bank, steps=8, episode=1, food=FOOD, food_level=4):
    """Drive a persistent two-agent coordination stall on one food."""
    agents = [(0, (4, 3), 1), (1, (0, 0), 2), (2, (7, 7), 2)]
    for step in range(steps):
        bank.update(
            step_info=step_info(
                step,
                agents,
                {food: food_level},
                failed=[failed_event(episode, food, food_level, (0,), (1,))],
                episode=episode,
            ),
            raw_team_reward=0.0,
            t_env=step,
        )
    bank.finish_episode()


# ==========================================================================
# episode_phase
# ==========================================================================


def test_episode_phase_splits_into_thirds_of_the_actual_episode():
    assert episode_phase(0, 30) == "early"
    assert episode_phase(9, 30) == "early"
    assert episode_phase(10, 30) == "mid"
    assert episode_phase(19, 30) == "mid"
    assert episode_phase(20, 30) == "late"
    assert episode_phase(29, 30) == "late"


def test_episode_phase_uses_the_real_length_not_the_time_limit():
    """An episode that ends early has its own thirds, not the time limit's."""
    assert episode_phase(5, 6) == "late"  # 5/6 of a 6-step episode
    assert episode_phase(5, 60) == "early"  # same step, long episode
    assert episode_phase(0, 0) == "early"  # degenerate, never raises


# ==========================================================================
# calibration_summary structure
# ==========================================================================


def test_summary_covers_all_48_variants_and_keeps_variant_report_fields():
    bank = make_bank()
    drive_stall(bank)
    summary = bank.calibration_summary()
    report = bank.variant_report()

    assert len(summary["variants"]) == 48
    assert set(summary["variants"]) == {v.variant_id for v in build_variant_grid()}
    for vid, entry in report.items():
        for key, value in entry.items():
            assert summary["variants"][vid][key] == value, key


def test_summary_reports_zero_guarantees_and_lifecycle_config():
    bank = make_bank(rearm_steps=3, cooldown_steps=5, max_events_per_episode=2)
    drive_stall(bank)
    summary = bank.calibration_summary()

    assert summary["zero_guarantees"] == {
        "query_count": 0,
        "provider_call_count": 0,
        "active_subgoal_count": 0,
        "total_shaping_reward": 0.0,
    }
    assert summary["lifecycle_config"]["rearm_steps"] == 3
    assert summary["lifecycle_config"]["cooldown_steps"] == 5
    assert summary["lifecycle_config"]["max_events_per_episode"] == 2
    assert summary["episodes_seen"] == 1


def test_summary_is_json_serialisable():
    bank = make_bank()
    drive_stall(bank)
    json.dumps(bank.calibration_summary())


# ==========================================================================
# rates and per-episode distribution
# ==========================================================================


def test_events_per_1000_steps_and_per_episode_match_the_counters():
    bank = make_bank(variants=only("wf5_m2_none_iboff"))
    drive_stall(bank, steps=10)
    entry = bank.calibration_summary()["variants"]["wf5_m2_none_iboff"]

    assert entry["observed_steps"] == 10
    assert entry["final_events"] > 0
    assert entry["events_per_1000_steps"] == pytest.approx(
        1000.0 * entry["final_events"] / 10
    )
    assert entry["events_per_episode"] == pytest.approx(entry["final_events"] / 1.0)


def test_episode_event_histogram_partitions_every_episode():
    bank = make_bank(variants=only("wf5_m2_none_iboff"))
    drive_stall(bank, steps=10, episode=1)
    bank.reset_episode(2)
    # A quiet episode: no failed loads at all, so no event can fire.
    for step in range(4):
        bank.update(
            step_info=step_info(step, [(0, (0, 0), 1)], {FOOD: 4}, episode=2),
            raw_team_reward=0.0,
            t_env=100 + step,
        )
    bank.finish_episode()

    summary = bank.calibration_summary()
    entry = summary["variants"]["wf5_m2_none_iboff"]
    total = (
        entry["episodes_with_0_events"]
        + entry["episodes_with_1_event"]
        + entry["episodes_with_multiple_events"]
    )
    assert summary["episodes_seen"] == 2
    assert total == 2
    assert entry["episodes_with_0_events"] >= 1


def test_rate_denominators_are_none_rather_than_zero_when_undefined():
    bank = make_bank(variants=only("wf5_m2_none_iboff"))
    entry = bank.calibration_summary()["variants"]["wf5_m2_none_iboff"]
    assert entry["events_per_1000_steps"] is None
    assert entry["events_per_episode"] is None
    assert entry["natural_resolution_rate_5"] is None


# ==========================================================================
# semantic validity
# ==========================================================================


def test_semantically_invalid_events_are_zero_while_candidates_may_not_be():
    """No feasible coalition means suppressed, so no event can be invalid."""
    bank = make_bank(variants=only("wf5_m2_none_iboff"), n_agents=1)
    # One agent, level 1, food level 9: no coalition can ever clear it.
    agents = [(0, (4, 3), 1)]
    for step in range(8):
        bank.update(
            step_info=step_info(
                step,
                agents,
                {FOOD: 9},
                failed=[failed_event(1, FOOD, 9, (0,), (1,))],
            ),
            raw_team_reward=0.0,
            t_env=step,
        )
    bank.finish_episode()

    entry = bank.calibration_summary()["variants"]["wf5_m2_none_iboff"]
    assert entry["semantically_invalid_candidates"] > 0
    assert entry["semantically_invalid_events"] == 0
    assert entry["final_events"] == 0


def test_every_variant_reports_zero_semantically_invalid_events():
    bank = make_bank()
    drive_stall(bank)
    summary = bank.calibration_summary()
    assert all(
        e["semantically_invalid_events"] == 0 for e in summary["variants"].values()
    )


# ==========================================================================
# resolution, censoring, recipients, phase
# ==========================================================================


def test_resolution_denominator_excludes_censored_horizons():
    bank = make_bank(variants=only("wf5_m2_none_iboff"))
    drive_stall(bank, steps=6)  # short episode: 10-step horizon must censor
    entry = bank.calibration_summary()["variants"]["wf5_m2_none_iboff"]

    for horizon in RESOLUTION_HORIZONS:
        labelled = (
            entry[f"resolved_within_{horizon}"]
            + entry[f"unresolved_within_{horizon}"]
            + entry[f"censored_within_{horizon}"]
        )
        assert labelled == entry["final_events"]
        assert entry[f"valid_denominator_{horizon}"] == (
            entry[f"resolved_within_{horizon}"] + entry[f"unresolved_within_{horizon}"]
        )
    assert entry["censored_within_10"] >= entry["censored_within_5"]


def test_recipient_distribution_totals_the_final_events():
    bank = make_bank()
    drive_stall(bank)
    for entry in bank.calibration_summary()["variants"].values():
        assert sum(entry["recipient_count_distribution"].values()) == entry[
            "final_events"
        ]


def test_phase_distribution_totals_the_final_events():
    bank = make_bank()
    drive_stall(bank, steps=12)
    for entry in bank.calibration_summary()["variants"].values():
        assert sum(entry["events_by_episode_phase"][p] for p in EPISODE_PHASES) == entry[
            "final_events"
        ]


# ==========================================================================
# team-gate vacuity
# ==========================================================================


def test_ungated_variants_are_labelled_ungated_not_vacuous():
    bank = make_bank()
    drive_stall(bank)
    summary = bank.calibration_summary()
    for vid, entry in summary["variants"].items():
        if entry["team_gate_variant"] == "none":
            assert entry["team_gate_vacuity"] == "ungated"
            assert entry["gate_open_percentage"] == 1.0


def test_gate_that_never_completes_a_window_is_flagged_not_reported_as_zero():
    bank = make_bank(variants=only("wf5_m2_t15_iboff"))
    drive_stall(bank, steps=6)  # never reaches a complete W_team=15 window
    entry = bank.calibration_summary()["variants"]["wf5_m2_t15_iboff"]
    assert entry["eligible_steps"] == 0
    assert entry["gate_open_percentage"] is None
    assert entry["team_gate_vacuity"] == "vacuous_no_eligible_steps"


def test_gate_open_throughout_a_stall_is_flagged_as_always_open():
    bank = make_bank(variants=only("wf5_m2_t5_iboff"))
    drive_stall(bank, steps=12)  # no progress at all, so the gate never closes
    entry = bank.calibration_summary()["variants"]["wf5_m2_t5_iboff"]
    assert entry["gate_open_percentage"] == 1.0
    assert entry["team_gate_vacuity"] == "vacuous_always_open"


def test_team_gate_filtering_contribution_matches_retained_and_suppressed():
    bank = make_bank()
    drive_stall(bank, steps=12)
    for entry in bank.calibration_summary()["variants"].values():
        assert (
            entry["team_gate_retained"] + entry["team_gate_suppressed"]
            == entry["raw_candidates"]
        )
        if entry["raw_candidates"]:
            assert entry["team_gate_filtering_contribution"] == pytest.approx(
                entry["team_gate_suppressed"] / entry["raw_candidates"]
            )


# ==========================================================================
# report generator
# ==========================================================================


def _write_summary(tmp_path, name, condition, steps):
    bank = make_bank()
    bank.condition = condition
    drive_stall(bank, steps=steps)
    summary = bank.calibration_summary()
    path = tmp_path / name
    path.write_text(json.dumps(summary))
    return bank, str(path)


def test_generator_ranks_all_48_variants_per_environment(tmp_path):
    _, path_a = _write_summary(tmp_path, "a.json", "envA", 10)
    _, path_b = _write_summary(tmp_path, "b.json", "envB", 12)

    out_md = tmp_path / "report.md"
    out_json = tmp_path / "report.json"
    REPORT.main(
        [
            "--run", f"A={path_a}",
            "--run", f"B={path_b}",
            "--out", str(out_md),
            "--json", str(out_json),
        ]
    )

    payload = json.loads(out_json.read_text())
    assert set(payload["ranked"]) == {"A", "B"}
    for entries in payload["ranked"].values():
        assert len(entries) == 48
        assert [e["rank"] for e in entries] == list(range(1, 49))
    text = out_md.read_text()
    assert "Ranked 48-variant table" in text
    assert "Cross-environment consistency" in text
    assert "Zero-guarantee confirmation" in text


def test_generator_ranks_by_firing_rate_descending(tmp_path):
    _, path_a = _write_summary(tmp_path, "a.json", "envA", 10)
    out_json = tmp_path / "r.json"
    REPORT.main(["--run", f"A={path_a}", "--out", str(tmp_path / "r.md"),
                 "--json", str(out_json)])
    entries = json.loads(out_json.read_text())["ranked"]["A"]
    rates = [e["events_per_1000_steps"] or -1.0 for e in entries]
    assert rates == sorted(rates, reverse=True)


def test_cross_environment_flags_a_variant_silent_in_one_environment(tmp_path):
    bank_a, path_a = _write_summary(tmp_path, "a.json", "envA", 12)
    # envB: a quiet run, so nothing fires anywhere.
    bank_b = make_bank()
    bank_b.condition = "envB"
    for step in range(6):
        bank_b.update(
            step_info=step_info(step, [(0, (0, 0), 1)], {FOOD: 4}),
            raw_team_reward=0.0,
            t_env=step,
        )
    bank_b.finish_episode()
    path_b = tmp_path / "b.json"
    path_b.write_text(json.dumps(bank_b.calibration_summary()))

    out_json = tmp_path / "r.json"
    REPORT.main(["--run", f"A={path_a}", "--run", f"B={path_b}",
                 "--out", str(tmp_path / "r.md"), "--json", str(out_json)])
    rows = json.loads(out_json.read_text())["cross_environment"]
    assert rows
    assert all("silent in at least one environment" in r["verdict"] for r in rows)


def test_generator_reconciles_against_the_shadow_csv(tmp_path):
    bank = make_bank()
    bank.condition = "envA"
    drive_stall(bank, steps=12)
    csv_path = tmp_path / "shadow.csv"
    ShadowCsvWriter(str(csv_path)).write(bank.drain_rows())
    summary_path = tmp_path / "a.json"
    summary_path.write_text(json.dumps(bank.calibration_summary()))

    out_json = tmp_path / "r.json"
    REPORT.main(
        [
            "--run", f"A={summary_path}",
            "--csv", f"A={csv_path}",
            "--verify-csv",
            "--out", str(tmp_path / "r.md"),
            "--json", str(out_json),
        ]
    )
    payload = json.loads(out_json.read_text())
    assert payload["verification"]["A"]["reconciled"] is True
    assert payload["verification"]["A"]["coalesced_event_rows"] > 0
    assert payload["storage"]["A"]["shadow_csv"]["bytes"] > 0


def test_reconciliation_fails_loudly_on_a_tampered_csv(tmp_path):
    bank = make_bank(variants=only("wf5_m2_none_iboff"))
    bank.condition = "envA"
    drive_stall(bank, steps=12)
    rows = bank.drain_rows()
    csv_path = tmp_path / "shadow.csv"
    # Drop one final event from the CSV; the counters must disagree.
    kept = [r for r in rows if r["row_type"] != "coalesced_event"][:]
    events = [r for r in rows if r["row_type"] == "coalesced_event"]
    assert events
    ShadowCsvWriter(str(csv_path)).write(kept + events[1:])
    summary_path = tmp_path / "a.json"
    summary_path.write_text(json.dumps(bank.calibration_summary()))

    with pytest.raises(SystemExit):
        REPORT.main(
            [
                "--run", f"A={summary_path}",
                "--csv", f"A={csv_path}",
                "--verify-csv",
                "--out", str(tmp_path / "r.md"),
                "--json", str(tmp_path / "r.json"),
            ]
        )


def test_generator_records_runtime_storage_and_commands(tmp_path):
    _, path_a = _write_summary(tmp_path, "a.json", "envA", 10)
    out_md = tmp_path / "r.md"
    REPORT.main(
        [
            "--run", f"A={path_a}",
            "--runtime", "A=4713.5",
            "--command", "A=python3 src/main.py --config=qmix",
            "--out", str(out_md),
        ]
    )
    text = out_md.read_text()
    assert "1h 18m" in text
    assert "python3 src/main.py --config=qmix" in text


def test_csv_columns_are_unchanged_by_the_report_work():
    """The report reads the agreed 55-column format; it must not alter it."""
    assert len(SHADOW_CSV_COLUMNS) == 55
    assert SHADOW_CSV_COLUMNS[0] == "row_type"
    assert SHADOW_CSV_COLUMNS[-1] == "event_id"
