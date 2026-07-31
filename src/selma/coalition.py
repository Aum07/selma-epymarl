"""SELMA Stage B: deterministic BFS distances and minimum helper coalitions.

Pure functions with no environment access and no randomness. Every tie is
broken by an explicit, reproducible rule so the same state always yields the
same coalition.

Grid model
----------
* Food cells are impassable.
* Grid boundaries are respected.
* Other agents are transient and never block a path.
* A *valid loading cell* for a food is an in-bounds, orthogonally adjacent cell
  that does not itself contain food.
* Cells currently occupied by the food's own LOADers are excluded from the
  helper candidate cells: those agents are staying put, so routing a helper onto
  them would not be collision-safe.
"""

from collections import deque
from itertools import combinations, permutations
from typing import NamedTuple, Tuple

NEIGHBOUR_OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def bfs_distances(start, grid_shape, blocked):
    """Shortest-path distances from ``start`` to every reachable cell.

    ``blocked`` is the set of impassable (food) cells. Deterministic: BFS layer
    order is fixed and distances are unique regardless of expansion order.
    """
    rows, cols = grid_shape
    start = (int(start[0]), int(start[1]))
    dist = {start: 0}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        base = dist[cur]
        for dr, dc in NEIGHBOUR_OFFSETS:
            nxt = (cur[0] + dr, cur[1] + dc)
            if not (0 <= nxt[0] < rows and 0 <= nxt[1] < cols):
                continue
            if nxt in blocked or nxt in dist:
                continue
            dist[nxt] = base + 1
            queue.append(nxt)
    return dist


def valid_loading_cells(food_position, grid_shape, food_positions):
    """In-bounds, orthogonally adjacent, non-food cells, sorted lexicographically."""
    rows, cols = grid_shape
    r, c = food_position
    cells = [
        (r + dr, c + dc)
        for dr, dc in NEIGHBOUR_OFFSETS
        if 0 <= r + dr < rows
        and 0 <= c + dc < cols
        and (r + dr, c + dc) not in food_positions
    ]
    return tuple(sorted(cells))


def min_distance_to_loading_cell(agent_distances, cells):
    """Smallest BFS distance from one agent to any of ``cells``; None if unreachable."""
    reachable = [agent_distances[cell] for cell in cells if cell in agent_distances]
    return min(reachable) if reachable else None


class CoalitionResult(NamedTuple):
    """Deterministic helper-coalition choice plus all tie-break evidence."""

    feasible: bool
    reason: str
    required_extra_level: int
    helper_ids: Tuple[int, ...]
    helper_levels: Tuple[int, ...]
    helper_distances: Tuple[int, ...]
    total_distance: int
    assignment: Tuple[Tuple[int, Tuple[int, int]], ...]
    candidate_cells: Tuple[Tuple[int, int], ...]
    recipient_ids: Tuple[int, ...]
    n_feasible_subsets: int
    tiebreak_key: Tuple


def _best_assignment(helper_ids, distances_by_agent, cells):
    """Cheapest collision-safe assignment of helpers to distinct loading cells.

    Ties on total distance are broken by the lexicographically smallest tuple of
    assigned cells, with helpers taken in ascending ID order.
    """
    if len(cells) < len(helper_ids):
        return None
    best = None
    for permutation in permutations(cells, len(helper_ids)):
        total = 0
        for agent_id, cell in zip(helper_ids, permutation):
            d = distances_by_agent[agent_id].get(cell)
            if d is None:
                total = None
                break
            total += d
        if total is None:
            continue
        key = (total, permutation)
        if best is None or key < best[0]:
            best = (key, permutation, total)
    if best is None:
        return None
    _, permutation, total = best
    return tuple(zip(helper_ids, permutation)), total


def select_helper_coalition(
    food_position,
    food_level,
    loader_ids,
    loader_levels,
    agents,
    food_positions,
    grid_shape,
    distances_by_agent,
):
    """Choose the deterministic minimum helper coalition for a failed food.

    ``agents`` is the tuple of ``AgentSnapshot``; ``distances_by_agent`` maps
    agent_id -> BFS distance dict (one BFS per agent per transition, reused).

    Selection order: fewest additional agents, then lowest total BFS distance to
    distinct valid loading cells, then lexicographically smallest helper-ID
    tuple, then lexicographically smallest loading-cell assignment.
    """
    required_extra = int(food_level) - int(sum(loader_levels))
    loader_id_set = set(int(i) for i in loader_ids)
    loader_cells = {
        tuple(a.position) for a in agents if int(a.agent_id) in loader_id_set
    }

    all_cells = valid_loading_cells(food_position, grid_shape, food_positions)
    candidate_cells = tuple(c for c in all_cells if c not in loader_cells)

    if required_extra <= 0:
        # Defensive: a coalition-required failed load always has a real deficit.
        return CoalitionResult(
            feasible=True,
            reason="no_helper_needed",
            required_extra_level=required_extra,
            helper_ids=(),
            helper_levels=(),
            helper_distances=(),
            total_distance=0,
            assignment=(),
            candidate_cells=all_cells,
            recipient_ids=tuple(sorted(loader_id_set)),
            n_feasible_subsets=0,
            tiebreak_key=(),
        )

    pool = sorted(
        (int(a.agent_id) for a in agents if int(a.agent_id) not in loader_id_set)
    )
    levels = {int(a.agent_id): int(a.level) for a in agents}

    feasible = []
    for size in range(1, len(pool) + 1):
        for subset in combinations(pool, size):
            if sum(levels[i] for i in subset) < required_extra:
                continue
            best = _best_assignment(subset, distances_by_agent, candidate_cells)
            if best is None:
                continue
            assignment, total = best
            feasible.append((len(subset), total, subset, assignment))
        if feasible:
            # Smallest coalition wins outright; no need to look at larger sizes.
            break

    if not feasible:
        return CoalitionResult(
            feasible=False,
            reason="no_feasible_coalition",
            required_extra_level=required_extra,
            helper_ids=(),
            helper_levels=(),
            helper_distances=(),
            total_distance=-1,
            assignment=(),
            candidate_cells=candidate_cells,
            recipient_ids=(),
            n_feasible_subsets=0,
            tiebreak_key=(),
        )

    feasible.sort(key=lambda x: (x[0], x[1], x[2], tuple(c for _, c in x[3])))
    size, total, subset, assignment = feasible[0]
    helper_distances = tuple(
        distances_by_agent[agent_id][cell] for agent_id, cell in assignment
    )
    return CoalitionResult(
        feasible=True,
        reason="ok",
        required_extra_level=required_extra,
        helper_ids=tuple(subset),
        helper_levels=tuple(levels[i] for i in subset),
        helper_distances=helper_distances,
        total_distance=total,
        assignment=tuple(assignment),
        candidate_cells=candidate_cells,
        recipient_ids=tuple(sorted(loader_id_set | set(subset))),
        n_feasible_subsets=len(feasible),
        tiebreak_key=(size, total, subset, tuple(c for _, c in assignment)),
    )
