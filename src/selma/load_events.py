"""SELMA Stage A: structured LBF load-event extraction.

Pure functions over a pre/post-step snapshot, so they can be exercised directly
in tests with no environment. Read-only: no randomness is consumed and nothing
here can affect observations, actions, rewards, termination, or replay contents.

Event classification deliberately reuses the semantics of the already validated
``envs/gymma.py::lbf_load_diagnostics`` counters:

* a *participant* in a food's load is an agent whose originally selected action
  was LOAD and which was orthogonally adjacent to that food's cell pre-step;
* a *failed* load is a food present both before and after the step with at least
  one participant -- identical to ``failed_load_events``;
* a *successful* load is a food present before and absent after the step.

The genuinely new, failure-side information added here is the sum-based
``coalition_required`` tag and its ``level_deficit``. This is not the existing
``necessary_cooperative_load_events`` counter, which is a stricter,
success-side, per-participant measure.

Adjacency ambiguity (one agent orthogonally adjacent to two remaining foods)
is unreachable under lbforaging's spawn rules for the supported environments --
food spawns are confined to rows/cols ``[1, n-2]`` and rejected within the 3x3
neighbourhood or within distance 2 on the row/col cross of existing food. This
module therefore does not implement any food-selection tie-break; it asserts the
condition instead, so a future LBF version that breaks the assumption fails
loudly rather than silently attributing events to the wrong food.
"""

import numpy as np

from .types import (
    AgentSnapshot,
    FailedLoadEvent,
    FoodSnapshot,
    SelmaStepInfo,
    SuccessfulLoadEvent,
)

# LOAD action index, per lbforaging.foraging.environment.Action.LOAD.value.
# Fixed by lbforaging's Action enum (NONE=0, NORTH=1, SOUTH=2, WEST=3, EAST=4, LOAD=5).
LOAD_ACTION = 5


def is_orthogonally_adjacent(position, cell):
    row, col = position
    crow, ccol = cell
    return abs(row - crow) + abs(col - ccol) == 1


def _food_cells(field):
    """Remaining food cells as ((row, col), level), in row-major order."""
    field = np.asarray(field)
    rows, cols = field.shape
    return [
        ((r, c), int(field[r, c]))
        for r in range(rows)
        for c in range(cols)
        if field[r, c] > 0
    ]


def assert_no_ambiguous_adjacency(field):
    """Raise if any cell is orthogonally adjacent to more than one food.

    Guards the assumption that a LOADing agent's target food is unambiguous, so
    no N/S/W/E selection rule is needed to attribute events.
    """
    field = np.asarray(field)
    rows, cols = field.shape
    for r in range(rows):
        for c in range(cols):
            neighbours = [
                (r + dr, c + dc)
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                if 0 <= r + dr < rows and 0 <= c + dc < cols and field[r + dr, c + dc] > 0
            ]
            if len(neighbours) > 1:
                raise AssertionError(
                    "SELMA LBF invariant violated: cell "
                    f"{(r, c)} is orthogonally adjacent to {len(neighbours)} foods "
                    f"{neighbours}; food-target attribution is ambiguous"
                )


def build_selma_step_info(
    episode_id,
    episode_step,
    pre_field,
    positions,
    levels,
    actions,
    post_field,
):
    """Build the immutable per-step SELMA record from a pre/post-step snapshot.

    ``pre_field``/``post_field`` are 2D arrays of food levels (0 = empty cell).
    ``positions``/``levels`` are per-agent, captured immediately before
    ``env.step``. ``actions`` are the originally selected actions, before
    lbforaging replaces invalid ones with ``Action.NONE``.

    Both fields are copied into value records; no live environment array is
    retained by the returned object.
    """
    pre_field = np.asarray(pre_field)
    post_field = np.asarray(post_field)

    positions = [tuple(int(x) for x in p) for p in positions]
    levels = [int(v) for v in levels]
    actions = tuple(int(a) for a in actions)

    loaders = [i for i, a in enumerate(actions) if a == LOAD_ACTION]

    pre_cells = _food_cells(pre_field)
    post_cells = _food_cells(post_field)
    post_positions = {pos for pos, _ in post_cells}

    def participants(cell):
        return [i for i in loaders if is_orthogonally_adjacent(positions[i], cell)]

    invalid_load_agents = tuple(
        i
        for i in loaders
        if not any(is_orthogonally_adjacent(positions[i], pos) for pos, _ in pre_cells)
    )

    failed_events = []
    successful_events = []
    for (row, col), food_level in pre_cells:
        loader_ids = tuple(participants((row, col)))
        if (row, col) in post_positions:
            if not loader_ids:
                continue
            loader_levels = tuple(levels[i] for i in loader_ids)
            total = int(sum(loader_levels))
            failed_events.append(
                FailedLoadEvent(
                    food_id=(int(episode_id), row, col),
                    food_position=(row, col),
                    food_level=food_level,
                    loader_ids=loader_ids,
                    loader_levels=loader_levels,
                    total_loader_level=total,
                    level_deficit=food_level - total,
                    coalition_required=total < food_level,
                    requires_help_by_agent=tuple(
                        food_level > lvl for lvl in loader_levels
                    ),
                )
            )
        else:
            loader_levels = tuple(levels[i] for i in loader_ids)
            successful_events.append(
                SuccessfulLoadEvent(
                    food_id=(int(episode_id), row, col),
                    food_position=(row, col),
                    food_level=food_level,
                    loader_ids=loader_ids,
                    loader_levels=loader_levels,
                    total_loader_level=int(sum(loader_levels)),
                )
            )

    return SelmaStepInfo(
        episode_id=int(episode_id),
        episode_step=int(episode_step),
        pre_foods=tuple(FoodSnapshot(pos, lvl) for pos, lvl in pre_cells),
        post_foods=tuple(FoodSnapshot(pos, lvl) for pos, lvl in post_cells),
        agents=tuple(
            AgentSnapshot(i, positions[i], levels[i]) for i in range(len(positions))
        ),
        selected_actions=actions,
        invalid_load_agents=invalid_load_agents,
        failed_load_events=tuple(failed_events),
        successful_load_events=tuple(successful_events),
    )
