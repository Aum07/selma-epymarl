"""SELMA Stage A: immutable per-step record types.

These are plain `NamedTuple`s so that every record is hashable, immutable, and
trivially serialisable. Nothing here touches the environment, consumes
randomness, or influences observations, actions, rewards, or termination.

Food identity within an episode is the food's grid coordinate:

    food_id = (episode_id, row, col)

which is stable for as long as the food exists (LBF never moves food, and a
cleared cell is never refilled during an episode).
"""

from typing import NamedTuple, Tuple


class FoodSnapshot(NamedTuple):
    """One remaining food cell at a single point in time."""

    position: Tuple[int, int]
    level: int


class AgentSnapshot(NamedTuple):
    """One agent's pre-step position and level."""

    agent_id: int
    position: Tuple[int, int]
    level: int


class FailedLoadEvent(NamedTuple):
    """A food that was adjacent-LOADed but still present after the step.

    This is the structured form of the existing, validated
    ``failed_load_events`` counter (see ``envs/gymma.py::lbf_load_diagnostics``):
    the food existed before the step, still exists after the step, and at least
    one agent whose *originally selected* action was LOAD was orthogonally
    adjacent to it.

    ``requires_help_by_agent`` is aligned element-wise with ``loader_ids``: entry
    ``k`` is ``food_level > loader_levels[k]``, i.e. whether that particular
    loader could never have cleared this food alone. It distinguishes a lone
    failed LOAD that genuinely cannot succeed from a failure by an individually
    capable agent.
    """

    food_id: Tuple[int, int, int]
    food_position: Tuple[int, int]
    food_level: int
    loader_ids: Tuple[int, ...]
    loader_levels: Tuple[int, ...]
    total_loader_level: int
    level_deficit: int
    coalition_required: bool
    requires_help_by_agent: Tuple[bool, ...]


class SuccessfulLoadEvent(NamedTuple):
    """A food that existed before the step and was removed after the step."""

    food_id: Tuple[int, int, int]
    food_position: Tuple[int, int]
    food_level: int
    loader_ids: Tuple[int, ...]
    loader_levels: Tuple[int, ...]
    total_loader_level: int


class SelmaStepInfo(NamedTuple):
    """Immutable snapshot of one ``env.step`` for SELMA's online consumers.

    ``episode_step`` is the pre-step transition index, matching the episode
    runner's ``self.t`` for the transition that produced this record.

    ``selected_actions`` are the *originally selected* actions, captured before
    lbforaging substitutes ``Action.NONE`` for invalid actions.

    ``invalid_load_agents`` holds agents whose selected action was LOAD while no
    remaining food was orthogonally adjacent in the pre-step snapshot. It is
    diagnostic only and never drives the detector.
    """

    episode_id: int
    episode_step: int
    pre_foods: Tuple[FoodSnapshot, ...]
    post_foods: Tuple[FoodSnapshot, ...]
    agents: Tuple[AgentSnapshot, ...]
    selected_actions: Tuple[int, ...]
    invalid_load_agents: Tuple[int, ...]
    failed_load_events: Tuple[FailedLoadEvent, ...]
    successful_load_events: Tuple[SuccessfulLoadEvent, ...]

    @property
    def coalition_required_failed_loads(self) -> Tuple[FailedLoadEvent, ...]:
        """Failed loads whose simultaneous loaders were jointly under-levelled."""
        return tuple(e for e in self.failed_load_events if e.coalition_required)
