"""SELMA Stage D: code-built candidate joint plans and their physical validity.

Why this module exists
----------------------

The Stage D live pilot (``docs/project/selma_stage_d_live_pilot_2026-08-16.md``)
returned, on fixture ``two_cells_only``::

    a0@5,4   a1@5,4

Two agents assigned to one cell. LBF cannot place two agents in the same cell, so
the guidance was physically unrealisable -- and the Stage C validator accepted it,
because it checked each assignment *independently* against the offered cells and
never looked at the selected combination. Independent per-assignment checks cannot
express a joint constraint, and prompt wording ("prefer distinct cells") is a
request, not an enforcement.

The correction is structural: the code enumerates the **complete set of
physically valid joint plans** and the model selects exactly one of them by id.
A joint plan is the whole mapping of required recipients to loading cells, so a
duplicate cell is not a bad answer that must be caught -- it is not in the answer
set at all. The deterministic validator then re-derives every joint constraint
from scratch (defence in depth), because a JSON Schema ``enum`` is a convenience
and is never the authority.

Determinism
-----------

Nothing here iterates a ``set`` or a ``dict`` to produce output. Agents are
sorted, cells are sorted, the permutation order of a sorted sequence is fixed,
and the resulting plan list is sorted again before ids are assigned. The same
request therefore yields byte-identical plans, with identical ids, on any machine
and under any hash seed.
"""

from itertools import permutations
from typing import NamedTuple, Tuple

#: Plan ids are positional and assigned only after canonical sorting, so
#: ``plan_0`` always denotes the same mapping for the same request.
PLAN_ID_PREFIX = "plan_"

# -- joint-validity reason codes (stable, machine-readable) -----------------
PLAN_DUPLICATE_AGENT = "plan_duplicate_agent"
PLAN_DUPLICATE_CELL = "plan_duplicate_cell"
PLAN_RECIPIENTS_NOT_EXACT = "plan_recipients_not_exact"
PLAN_CELL_NOT_OFFERED = "plan_cell_not_offered"
PLAN_CELL_NOT_ADJACENT = "plan_cell_not_adjacent"
PLAN_CELL_OUTSIDE_GRID = "plan_cell_outside_grid"
PLAN_CELL_IS_TARGET_FOOD = "plan_cell_is_target_food"
PLAN_EMPTY = "plan_empty"

#: Report order: structural problems first, then cell problems from the most
#: specific to the least. It is a fixed precedence rather than an alphabetical
#: sort so that the single reason a caller surfaces is the most informative one
#: -- an out-of-grid cell reports ``plan_cell_outside_grid``, not the
#: alphabetically earlier ``plan_cell_not_adjacent`` that is also technically
#: true of it. Deterministic either way; this one is legible.
VIOLATION_PRECEDENCE = (
    PLAN_EMPTY,
    PLAN_DUPLICATE_AGENT,
    PLAN_DUPLICATE_CELL,
    PLAN_RECIPIENTS_NOT_EXACT,
    PLAN_CELL_OUTSIDE_GRID,
    PLAN_CELL_IS_TARGET_FOOD,
    PLAN_CELL_NOT_ADJACENT,
    PLAN_CELL_NOT_OFFERED,
)

#: Every constraint an enumerated plan satisfies by construction and that the
#: validator re-checks anyway. Documented here so the set cannot drift silently.
JOINT_CONSTRAINTS = (
    "every required recipient appears exactly once",
    "no agent appears more than once",
    "every loading cell was offered by the detector",
    "every loading cell is orthogonally adjacent to the target food",
    "every loading cell lies inside the grid",
    "no two agents share a loading cell",
    "the target food cell is never an agent destination",
)


class PlanAssignment(NamedTuple):
    """One agent sent to one loading cell. Never meaningful on its own."""

    agent_id: int
    loading_cell: Tuple[int, int]


class JointPlan(NamedTuple):
    """A complete, physically valid mapping of the required recipients to cells.

    ``assignments`` is always ordered by ``agent_id``, so two plans compare and
    serialise identically iff they are the same plan.
    """

    plan_id: str
    assignments: Tuple[PlanAssignment, ...]

    @property
    def agent_ids(self):
        return tuple(int(a.agent_id) for a in self.assignments)

    @property
    def cells(self):
        return tuple(tuple(int(x) for x in a.loading_cell) for a in self.assignments)

    def mapping(self):
        """``{agent_id: (row, col)}`` -- the plan's content, id excluded."""
        return {int(a.agent_id): tuple(int(x) for x in a.loading_cell)
                for a in self.assignments}


def _cell(value):
    return (int(value[0]), int(value[1]))


def in_grid(cell, grid_shape):
    rows, cols = int(grid_shape[0]), int(grid_shape[1])
    return 0 <= int(cell[0]) < rows and 0 <= int(cell[1]) < cols


def is_adjacent(cell, food_position):
    """Orthogonal adjacency: exactly the cells an LBF agent can LOAD from."""
    return (abs(int(cell[0]) - int(food_position[0]))
            + abs(int(cell[1]) - int(food_position[1]))) == 1


def admissible_cells(candidate_cells, food_position, grid_shape):
    """The offered cells a plan may legally use, sorted and de-duplicated.

    Filters, independently of whatever the detector offered: inside the grid,
    orthogonally adjacent to the target food, and not the food cell itself. The
    detector already applies its own bounds check, but this module never trusts
    an upstream filter it does not perform itself.
    """
    keep = set()
    for raw in candidate_cells or ():
        cell = _cell(raw)
        if not in_grid(cell, grid_shape):
            continue
        if cell == _cell(food_position):
            continue
        if not is_adjacent(cell, food_position):
            continue
        keep.add(cell)
    return tuple(sorted(keep))          # sorted: never set-iteration order


def enumerate_joint_plans(recipient_agents, candidate_cells, food_position,
                          grid_shape):
    """Every physically valid joint plan, canonically ordered and identified.

    Returns ``()`` when no valid plan exists -- fewer admissible cells than
    recipients, no recipients, or a malformed recipient list. An empty result is
    a first-class outcome: the caller must then issue no provider call and no
    cache lookup at all (see the zero-plan policy in
    ``docs/project/selma_stage_d_joint_plan_acceptance_2026-08-16.md``).
    """
    recipients = tuple(sorted(int(a) for a in recipient_agents or ()))
    if not recipients or len(set(recipients)) != len(recipients):
        return ()
    cells = admissible_cells(candidate_cells, food_position, grid_shape)
    if len(cells) < len(recipients):
        # Distinct cells are required, so fewer cells than agents is unsolvable.
        return ()

    # permutations of a sorted sequence are generated in a fixed order; the
    # explicit sort afterwards makes canonicity a property of the code rather
    # than of an itertools implementation detail.
    assignments = sorted(
        tuple(PlanAssignment(agent_id=agent, loading_cell=cell)
              for agent, cell in zip(recipients, chosen))
        for chosen in permutations(cells, len(recipients))
    )
    return tuple(
        JointPlan(plan_id="{}{:d}".format(PLAN_ID_PREFIX, index),
                  assignments=plan)
        for index, plan in enumerate(assignments)
    )


def plan_violations(assignments, *, recipient_agents, candidate_cells,
                    food_position, grid_shape):
    """Every joint constraint an assignment set breaks. Empty means valid.

    Deliberately independent of :func:`enumerate_joint_plans`: it re-derives each
    constraint from the raw request rather than trusting that a plan came out of
    the enumerator. That is what makes it useful against a tampered cache row, a
    hand-built response or a future refactor that widens the answer set.

    Returns the broken codes in :data:`VIOLATION_PRECEDENCE` order, so a caller
    may report ``problems[0]`` as *the* reason deterministically.
    """
    problems = set()
    pairs = tuple(assignments or ())
    if not pairs:
        return [PLAN_EMPTY]

    agents = [int(a.agent_id) for a in pairs]
    cells = [_cell(a.loading_cell) for a in pairs]

    if len(agents) != len(set(agents)):
        problems.add(PLAN_DUPLICATE_AGENT)
    if len(cells) != len(set(cells)):
        problems.add(PLAN_DUPLICATE_CELL)
    if sorted(agents) != sorted(int(a) for a in recipient_agents or ()):
        # Covers a missing required recipient and an extra one alike.
        problems.add(PLAN_RECIPIENTS_NOT_EXACT)

    offered = {_cell(c) for c in candidate_cells or ()}
    food = _cell(food_position)
    for cell in cells:
        if not in_grid(cell, grid_shape):
            problems.add(PLAN_CELL_OUTSIDE_GRID)
        if cell == food:
            problems.add(PLAN_CELL_IS_TARGET_FOOD)
        elif not is_adjacent(cell, food):
            problems.add(PLAN_CELL_NOT_ADJACENT)
        if cell not in offered:
            problems.add(PLAN_CELL_NOT_OFFERED)
    return [reason for reason in VIOLATION_PRECEDENCE if reason in problems]


def plan_to_dict(plan):
    """Canonical, JSON-safe form of a complete plan. Used by the cache key."""
    return {
        "plan_id": str(plan.plan_id),
        "assignments": [
            {"agent_id": int(a.agent_id), "loading_cell": list(_cell(a.loading_cell))}
            for a in plan.assignments
        ],
    }


def plan_from_dict(payload):
    return JointPlan(
        plan_id=str(payload["plan_id"]),
        assignments=tuple(
            PlanAssignment(agent_id=int(a["agent_id"]),
                           loading_cell=_cell(a["loading_cell"]))
            for a in payload["assignments"]
        ),
    )


def describe_plan(plan):
    """Human-readable mapping for the prompt: ``a0->(4,3), a1->(3,4)``.

    The prompt must state what an id *means*; an opaque ``plan_3`` would put the
    model back in the position of guessing what it is selecting.
    """
    return ", ".join(
        "a{}->({},{})".format(int(a.agent_id), _cell(a.loading_cell)[0],
                              _cell(a.loading_cell)[1])
        for a in plan.assignments
    )
