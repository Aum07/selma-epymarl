"""SELMA Stage B: separately configured shadow CSV writer.

This file is the *only* additional output permitted in shadow mode. It never
touches the ordinary training CSV written by the episode runner.
"""

import csv
import os

SHADOW_CSV_COLUMNS = (
    "row_type",
    "condition",
    "seed",
    "t_env",
    "episode_id",
    "episode_step",
    "variant_id",
    "w_fail",
    "m",
    "team_gate_variant",
    "team_gate_open",
    "inbound_variant",
    "raw_team_reward",
    "food_removed",
    "progress",
    "trigger_agent_ids",
    "food_id",
    "food_position",
    "food_level",
    "failed_load_count_in_window",
    "loader_ids",
    "loader_levels",
    "loader_level_sum",
    "level_deficit",
    "required_extra_level",
    "helper_ids",
    "helper_levels",
    "helper_distances",
    "total_helper_distance",
    "recipient_ids",
    "candidate_loading_cells",
    "selected_cell_assignment",
    "tiebreak_evidence",
    "n_feasible_subsets",
    "coalition_feasible",
    "coalition_reason",
    "trigger_distance",
    "trigger_distance_delta",
    "inbound_help",
    "inbound_helper_ids",
    "inbound_start_distances",
    "inbound_end_distances",
    "inbound_distance_deltas",
    "raw_would_fire",
    "would_fire_after_inbound",
    "latched",
    "cooldown_remaining",
    "rearm_false_streak",
    "events_used",
    "event_budget",
    "coalesced_candidate_count",
    "suppression_reason",
    "resolved_within_5",
    "resolved_within_10",
    "event_id",
)


class ShadowCsvWriter:
    """Appends shadow rows, writing the header for a missing or empty file."""

    def __init__(self, path):
        self.path = path

    def write(self, rows):
        if not rows:
            return 0
        write_header = not os.path.exists(self.path) or os.path.getsize(self.path) == 0
        with open(self.path, "a", newline="") as handle:
            writer = csv.writer(handle)
            if write_header:
                writer.writerow(SHADOW_CSV_COLUMNS)
            for row in rows:
                writer.writerow([_render(row[c]) for c in SHADOW_CSV_COLUMNS])
        return len(rows)


def _render(value):
    if value is None:
        return ""
    if isinstance(value, (tuple, list)):
        return repr(tuple(value))
    return value
