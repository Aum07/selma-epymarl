"""Code-built target-allocation plans for SELMA V2.

The model is never allowed to invent agents, foods, coordinates, actions, or
rewards.  Code enumerates complete mappings from every agent to one of the
remaining foods and retains only mappings whose assigned coalition has enough
level to collect every used food.  A provider contributes one bounded choice:
the id of an offered plan.
"""

import hashlib
import itertools
import json
from typing import NamedTuple, Tuple


class FoodTarget(NamedTuple):
    """One remaining food, identified by its episode-stable coordinate id."""

    food_id: Tuple[int, int, int]
    position: Tuple[int, int]
    level: int


class TargetAssignment(NamedTuple):
    """One agent's target in a complete allocation plan."""

    agent_id: int
    target_food_id: Tuple[int, int, int]
    target_position: Tuple[int, int]
    target_level: int


class TargetAllocationPlan(NamedTuple):
    """A complete, code-validated agent-to-food allocation."""

    plan_id: str
    assignments: Tuple[TargetAssignment, ...]

    def mapping(self):
        return {int(a.agent_id): tuple(a.target_food_id) for a in self.assignments}

    def position_mapping(self):
        return {int(a.agent_id): tuple(a.target_position) for a in self.assignments}


def _normalise_foods(remaining_foods, episode_id):
    foods = []
    for item in remaining_foods or ():
        if hasattr(item, "position"):
            position = item.position
            level = item.level
            food_id = getattr(item, "food_id", None)
        elif isinstance(item, dict):
            position = item["position"]
            level = item["level"]
            food_id = item.get("food_id")
        else:
            # Compact test/caller form: ((row, col), level), or
            # ((episode, row, col), (row, col), level).
            if len(item) == 2:
                position, level = item
                food_id = None
            else:
                food_id, position, level = item
        pos = (int(position[0]), int(position[1]))
        fid = ((int(episode_id), pos[0], pos[1]) if food_id is None
               else tuple(int(x) for x in food_id))
        foods.append(FoodTarget(fid, pos, int(level)))
    return tuple(sorted(foods, key=lambda f: (f.position, f.level, f.food_id)))


def _plan_signature(assignments):
    # Episode ids are deliberately excluded: absolute state repeats across
    # episodes must enumerate the same semantic plan_id -> mapping relation.
    return tuple((int(a.agent_id), tuple(a.target_position), int(a.target_level))
                 for a in assignments)


def enumerate_target_plans(agent_ids, agent_levels, remaining_foods,
                           trigger_food_id, *, agent_positions=None,
                           grid_shape=None, max_plans=256):
    """Enumerate feasible complete target allocations deterministically.

    Every agent receives exactly one target.  The detector's trigger food must
    remain represented, and the combined level assigned to every represented
    food must meet that food's level.  Plan order is a deterministic hash of
    the state plus mapping instead of a semantically privileged lexicographic
    order, preventing ``plan_0`` from always meaning the same allocation.

    More than ``max_plans`` feasible mappings fails closed with an empty plan
    set.  This keeps prompt size and provider cost bounded by code.
    """
    agents = tuple(sorted(int(a) for a in agent_ids))
    levels = {int(a): int(v) for a, v in (agent_levels or {}).items()}
    trigger = tuple(int(x) for x in trigger_food_id)
    episode_id = trigger[0]
    foods = _normalise_foods(remaining_foods, episode_id)
    by_id = {f.food_id: f for f in foods}
    if (not agents or not foods or trigger not in by_id
            or any(a not in levels for a in agents)):
        return ()

    feasible = []
    for chosen in itertools.product(foods, repeat=len(agents)):
        used = {f.food_id for f in chosen}
        if trigger not in used:
            continue
        assigned_level = {food_id: 0 for food_id in used}
        assigned_count = {food_id: 0 for food_id in used}
        for agent_id, food in zip(agents, chosen):
            assigned_level[food.food_id] += levels[agent_id]
            assigned_count[food.food_id] += 1
        if any(assigned_level[fid] < by_id[fid].level for fid in used):
            continue
        if grid_shape is not None:
            rows, cols = int(grid_shape[0]), int(grid_shape[1])
            loading_cells = {
                fid: sum((food.position[0] > 0,
                          food.position[0] + 1 < rows,
                          food.position[1] > 0,
                          food.position[1] + 1 < cols))
                for fid, food in by_id.items()
            }
            if any(assigned_count[fid] > loading_cells[fid] for fid in used):
                continue
        assignments = tuple(
            TargetAssignment(agent_id, food.food_id, food.position, food.level)
            for agent_id, food in zip(agents, chosen)
        )
        feasible.append(assignments)
        if len(feasible) > int(max_plans):
            return ()

    state = {
        "agents": [
            [a, levels[a], list((agent_positions or {}).get(a, ()))]
            for a in agents
        ],
        "foods": [[list(f.position), f.level] for f in foods],
        "trigger": [trigger[1], trigger[2]],
        "grid_shape": list(grid_shape) if grid_shape is not None else None,
    }
    prefix = json.dumps(state, sort_keys=True, separators=(",", ":"))

    def order_key(assignments):
        body = json.dumps(_plan_signature(assignments), separators=(",", ":"))
        return hashlib.sha256((prefix + "|" + body).encode("utf-8")).hexdigest()

    feasible.sort(key=lambda assignments: (order_key(assignments),
                                            _plan_signature(assignments)))
    return tuple(TargetAllocationPlan("target_plan_{}".format(i), assignments)
                 for i, assignments in enumerate(feasible))


def target_plan_violations(assignments, *, recipient_agents, remaining_foods,
                           trigger_food_id, agent_levels=None, grid_shape=None):
    """Re-derive V2 allocation validity from raw request state."""
    recipients = tuple(sorted(int(a) for a in recipient_agents))
    trigger = tuple(int(x) for x in trigger_food_id)
    foods = _normalise_foods(remaining_foods, trigger[0])
    by_id = {f.food_id: f for f in foods}
    items = tuple(assignments or ())
    ids = [int(a.agent_id) for a in items]
    if len(ids) != len(set(ids)):
        return ("duplicate_agent_assignment",)
    if set(ids) != set(recipients):
        return ("recipients_not_exact",)
    targets = [tuple(a.target_food_id) for a in items]
    if any(target not in by_id for target in targets):
        return ("target_not_remaining",)
    if trigger not in set(targets):
        return ("trigger_food_unassigned",)
    for a in items:
        food = by_id[tuple(a.target_food_id)]
        if tuple(a.target_position) != food.position or int(a.target_level) != food.level:
            return ("target_metadata_mismatch",)
        if (hasattr(a, "loading_cell")
                and tuple(a.loading_cell) != tuple(a.target_position)):
            return ("target_metadata_mismatch",)
    if agent_levels is not None:
        levels = {int(a): int(level) for a, level in agent_levels.items()}
        if any(agent_id not in levels for agent_id in recipients):
            return ("missing_agent_level",)
        totals = {}
        for a in items:
            fid = tuple(a.target_food_id)
            totals[fid] = totals.get(fid, 0) + levels[int(a.agent_id)]
        if any(total < by_id[fid].level for fid, total in totals.items()):
            return ("coalition_infeasible",)
    if grid_shape is not None:
        rows, cols = int(grid_shape[0]), int(grid_shape[1])
        counts = {}
        for target in targets:
            counts[target] = counts.get(target, 0) + 1
        for fid, count in counts.items():
            row, col = by_id[fid].position
            capacity = sum((row > 0, row + 1 < rows,
                            col > 0, col + 1 < cols))
            if count > capacity:
                return ("insufficient_loading_cells",)
    return ()


def target_plan_to_dict(plan):
    return {
        "plan_id": str(plan.plan_id),
        "assignments": [
            {
                "agent_id": int(a.agent_id),
                "target_position": [int(x) for x in a.target_position],
                "target_level": int(a.target_level),
            }
            for a in plan.assignments
        ],
    }


def describe_target_plan(plan):
    return "; ".join(
        "agent {} -> food at {} (level {})".format(
            a.agent_id, tuple(a.target_position), a.target_level)
        for a in plan.assignments
    )
