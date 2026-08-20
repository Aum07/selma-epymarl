"""SELMA V2 tests: the plan-conditioned potential and the two offline controls.

What these tests are for
------------------------

The Stage E experiments measured a potential that reads the detector-chosen food
coordinate and the code-pinned recipient set, and nothing else. The provider's
selected joint plan was stored on the active guidance and never read, so every
offered plan produced a bit-identical potential and the provider had no causal
path to the reward at all. ``test_v1_potential_is_identical_across_all_plans``
below pins that property in place as a *regression test of the documented V1
behaviour*, so the defect can never be reintroduced silently once V2 exists.

Everything else here asserts the V2 property that was missing: the content of
the selected plan changes the shaping reward, in the direction it should.

Run:  PYTHONPATH=src pytest tests/test_selma_v2_plan_potential.py -v
"""

import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from selma.guidance import (  # noqa: E402
    POTENTIAL_FOOD,
    POTENTIAL_MODES,
    POTENTIAL_PLAN,
    ActiveGuidance,
    GuidanceController,
    d_max_plus_load,
    pbrs_term_unscaled,
    phi,
    phi_components,
)
from selma.modes import (  # noqa: E402
    FROZEN_ACTIVE_VARIANT,
    SelmaConfigError,
    validate_selma_config,
)
from selma.offline_selectors import (  # noqa: E402
    HeuristicMinTravelProvider,
    RandomValidPlanProvider,
    plan_total_travel,
)
from selma.provider import (  # noqa: E402
    DeterministicMockProvider,
    build_request,
    validate,
)
from selma.provider_registry import make_provider, needs_run_seed  # noqa: E402

GRID = (8, 8)
FOOD = (4, 4)
CELLS = ((4, 3), (3, 4))
#: Agent 0 already stands on one loading cell; agent 1 is far away. The two
#: offered plans therefore differ in total travel, which is what makes the
#: plan-conditioned potential distinguishable at all.
POSITIONS = {0: (4, 3), 1: (6, 6)}


def event_row(recipients=(0, 1), cells=CELLS, levels=(1, 2), food_level=3):
    return {
        "row_type": "coalesced_event",
        "event_id": "ep1-{}-food4,4-t3".format(FROZEN_ACTIVE_VARIANT),
        "variant_id": FROZEN_ACTIVE_VARIANT,
        "food_id": (1, FOOD[0], FOOD[1]),
        "food_position": FOOD,
        "food_level": food_level,
        "trigger_agent_ids": (0,),
        "recipient_ids": recipients,
        "loader_ids": recipients,
        "loader_levels": levels,
        "level_deficit": food_level - sum(levels),
        "required_extra_level": max(0, food_level - sum(levels)),
        "candidate_loading_cells": cells,
        "coalition_feasible": True,
        "suppression_reason": "",
    }


def request_for(positions=None):
    return build_request(event_row(), agent_positions=positions or POSITIONS,
                         grid_shape=GRID)


def guidance_for(plan, food_position=FOOD):
    """An ActiveGuidance carrying one plan's assignments, as the controller builds it."""
    return ActiveGuidance(
        event_id="e", subgoal_id="sg1-e",
        food_id=(1, food_position[0], food_position[1]),
        food_position=food_position,
        recipients=tuple(int(a.agent_id) for a in plan.assignments),
        assignments={int(a.agent_id): ("approach_and_load",
                                       tuple(int(x) for x in a.loading_cell))
                     for a in plan.assignments},
        activated_step=0, provider_id="test", selected_plan_id=plan.plan_id,
    )


def moved(agent_id, position, base=None):
    """``POSITIONS`` with one agent relocated. Integer keys, so not ``dict(**)``."""
    out = dict(base if base is not None else POSITIONS)
    out[int(agent_id)] = position
    return out


def all_plans():
    plans = request_for().candidate_joint_plans
    assert len(plans) >= 2, "fixture must offer at least two distinct plans"
    return plans


# -- 1. the V1 defect, pinned so it cannot come back silently -----------------


def test_v1_potential_is_identical_across_all_plans():
    """Documents the Stage E behaviour: plan content cannot move the potential."""
    values = {
        plan.plan_id: phi(guidance_for(plan), POSITIONS, (FOOD,), GRID,
                          potential_mode=POTENTIAL_FOOD)
        for plan in all_plans()
    }
    assert len(set(values.values())) == 1, (
        "V1 potential must be plan-independent; got {!r}".format(values)
    )


def test_v1_is_the_default_everywhere():
    """An unmodified config and an unmodified controller both stay on V1."""
    plan = all_plans()[0]
    assert phi(guidance_for(plan), POSITIONS, (FOOD,), GRID) == phi(
        guidance_for(plan), POSITIONS, (FOOD,), GRID, potential_mode=POTENTIAL_FOOD)
    ctrl = GuidanceController(
        provider=DeterministicMockProvider(), grid_shape=GRID, beta=0.0,
        gamma=0.99, enabled=True, variant_id=FROZEN_ACTIVE_VARIANT)
    assert ctrl.potential_mode == POTENTIAL_FOOD


# -- 2. the V2 property that was missing --------------------------------------


def test_changing_plan_id_changes_the_potential():
    values = {
        plan.plan_id: phi(guidance_for(plan), POSITIONS, (FOOD,), GRID,
                          potential_mode=POTENTIAL_PLAN)
        for plan in all_plans()
    }
    assert len(set(values.values())) > 1, (
        "V2 potential must depend on which plan was selected; got {!r}".format(values)
    )


def test_swapping_two_agents_assignments_changes_the_potential():
    """The same cells, permuted between agents, must not be equivalent."""
    plan_a, plan_b = all_plans()[0], all_plans()[1]
    assert set(plan_a.cells) == set(plan_b.cells), "fixture plans must be permutations"
    assert (phi(guidance_for(plan_a), POSITIONS, (FOOD,), GRID,
                potential_mode=POTENTIAL_PLAN)
            != phi(guidance_for(plan_b), POSITIONS, (FOOD,), GRID,
                   potential_mode=POTENTIAL_PLAN))


def test_moving_toward_the_assigned_cell_raises_the_potential():
    plan = all_plans()[0]
    g = guidance_for(plan)
    target = g.assignments[1][1]
    here = POSITIONS[1]
    step = (here[0] + (1 if target[0] > here[0] else -1), here[1])
    assert abs(step[0] - target[0]) < abs(here[0] - target[0]), "fixture must move closer"
    before = phi(g, POSITIONS, (FOOD,), GRID, potential_mode=POTENTIAL_PLAN)
    after = phi(g, moved(1, step), (FOOD,), GRID,
                potential_mode=POTENTIAL_PLAN)
    assert after > before


def test_moving_away_from_the_assigned_cell_lowers_the_potential():
    plan = all_plans()[0]
    g = guidance_for(plan)
    away = (POSITIONS[1][0] + 1, POSITIONS[1][1])
    before = phi(g, POSITIONS, (FOOD,), GRID, potential_mode=POTENTIAL_PLAN)
    after = phi(g, moved(1, away), (FOOD,), GRID,
                potential_mode=POTENTIAL_PLAN)
    assert after < before


def test_agent_on_its_assigned_cell_is_the_per_agent_maximum():
    plan = all_plans()[0]
    g = guidance_for(plan)
    target = g.assignments[0][1]
    on_cell = phi_components(g, moved(0, target), (FOOD,), GRID,
                             potential_mode=POTENTIAL_PLAN)[0]
    dmax = d_max_plus_load(GRID)
    assert on_cell == pytest.approx(-1.0 / dmax)  # only the terminal LOAD remains
    for offset in ((1, 0), (0, 1), (-1, 0), (0, -1)):
        neighbour = (target[0] + offset[0], target[1] + offset[1])
        if not (0 <= neighbour[0] < GRID[0] and 0 <= neighbour[1] < GRID[1]):
            continue
        assert phi_components(g, moved(0, neighbour), (FOOD,), GRID,
                              potential_mode=POTENTIAL_PLAN)[0] < on_cell


# -- 3. failing closed --------------------------------------------------------


def test_missing_assignment_yields_no_shaping_rather_than_a_food_fallback():
    """A recipient with no plan entry must give phi = 0, never a silent V1 fallback."""
    plan = all_plans()[0]
    g = guidance_for(plan)
    g.assignments.pop(1)
    assert phi_components(g, POSITIONS, (FOOD,), GRID,
                          potential_mode=POTENTIAL_PLAN) == {}
    assert phi(g, POSITIONS, (FOOD,), GRID, potential_mode=POTENTIAL_PLAN) == 0.0


def test_no_active_guidance_is_zero_in_both_modes():
    for mode in POTENTIAL_MODES:
        assert phi(None, POSITIONS, (FOOD,), GRID, potential_mode=mode) == 0.0


def test_unknown_potential_mode_is_rejected_loudly():
    plan = all_plans()[0]
    with pytest.raises(ValueError):
        phi_components(guidance_for(plan), POSITIONS, (FOOD,), GRID,
                       potential_mode="distance")
    with pytest.raises(ValueError):
        GuidanceController(provider=DeterministicMockProvider(), grid_shape=GRID,
                           beta=0.0, gamma=0.99, enabled=True,
                           variant_id=FROZEN_ACTIVE_VARIANT,
                           potential_mode="distance")


# -- 4. terminal handling and telescoping ------------------------------------


def test_terminal_effective_potential_is_zero_and_telescoping_holds():
    """Sum of F_t over an episode with phi_terminal = 0 collapses to -phi_0."""
    gamma = 0.99
    plan = all_plans()[0]
    g = guidance_for(plan)
    walk = [POSITIONS,
            moved(1, (5, 6)),
            moved(1, (4, 6)),
            moved(1, (4, 5))]
    phis = [phi(g, p, (FOOD,), GRID, potential_mode=POTENTIAL_PLAN) for p in walk]
    successors = phis[1:] + [0.0]        # terminal transition: phi_effective = 0

    # The policy-invariance statement itself: the *discounted* sum of the
    # shaping terms telescopes to -phi_0 exactly, so shaping adds a constant
    # that depends only on the start state and can never change which policy
    # is optimal. This is the property the whole design rests on.
    discounted = sum(
        (gamma ** t) * pbrs_term_unscaled(gamma, before, after)
        for t, (before, after) in enumerate(zip(phis, successors))
    )
    assert discounted == pytest.approx(-phis[0], abs=1e-12)

    # The undiscounted sum the reward ledger reconciles against.
    undiscounted = sum(
        pbrs_term_unscaled(gamma, before, after)
        for before, after in zip(phis, successors)
    )
    assert undiscounted == pytest.approx(
        -phis[0] + (gamma - 1.0) * sum(phis[1:]), abs=1e-12)


def test_shaping_is_bounded_by_the_potential_range():
    """|F_t| can never exceed (1 + gamma) * max|phi|, so a reward cannot blow up."""
    gamma, dmax = 0.99, d_max_plus_load(GRID)
    bound = (1.0 + gamma) * (float(dmax) / float(dmax))
    plan = all_plans()[0]
    g = guidance_for(plan)
    for r0 in range(GRID[0]):
        for c0 in range(GRID[1]):
            for r1 in range(GRID[0]):
                positions = {0: (r0, c0), 1: (r1, 0)}
                value = phi(g, positions, (FOOD,), GRID, potential_mode=POTENTIAL_PLAN)
                assert -1.0 <= value <= 0.0
                assert abs(pbrs_term_unscaled(gamma, value, 0.0)) <= bound


# -- 5. the offline controls --------------------------------------------------


def test_heuristic_selects_the_minimum_travel_plan():
    req = request_for()
    positions = {int(c.agent_id): tuple(int(x) for x in c.position)
                 for c in req.agent_context}
    costs = {p.plan_id: plan_total_travel(p, positions) for p in req.candidate_joint_plans}
    chosen = HeuristicMinTravelProvider().query(req)
    assert costs[chosen.selected_plan_id] == min(costs.values())


def test_heuristic_breaks_ties_at_the_lowest_plan_id():
    """With both agents equidistant, the canonical first minimiser must win."""
    positions = {0: FOOD, 1: FOOD}  # symmetric: every plan costs the same
    req = request_for(positions)
    costs = {p.plan_id: plan_total_travel(p, positions) for p in req.candidate_joint_plans}
    assert len(set(costs.values())) == 1, "fixture must be a genuine tie"
    chosen = HeuristicMinTravelProvider().query(req)
    assert chosen.selected_plan_id == min(costs)


def test_heuristic_and_mock_can_disagree():
    """Otherwise the 'heuristic' control would be the mock under another name."""
    req = request_for()
    assert (HeuristicMinTravelProvider().query(req).selected_plan_id
            != DeterministicMockProvider().query(req).selected_plan_id)


def test_random_selector_is_reproducible_for_a_seed():
    req = request_for()
    a = RandomValidPlanProvider(run_seed=1).query(req).selected_plan_id
    b = RandomValidPlanProvider(run_seed=1).query(req).selected_plan_id
    assert a == b


def test_random_selector_does_not_consume_global_rng():
    """The control must not perturb any other stochastic component of the run."""
    random.seed(12345)
    expected = [random.random() for _ in range(3)]
    random.seed(12345)
    RandomValidPlanProvider(run_seed=7).query(request_for())
    assert [random.random() for _ in range(3)] == expected


def test_random_selector_requires_an_explicit_seed():
    with pytest.raises(ValueError):
        RandomValidPlanProvider(run_seed=None)
    assert needs_run_seed("random_valid_plan")
    assert not needs_run_seed("heuristic_min_travel")


def test_random_selector_spans_the_offered_plans_across_events():
    """A degenerate control that always returns plan_0 would be no control at all."""
    provider = RandomValidPlanProvider(run_seed=3)
    seen = set()
    base = event_row()
    for step in range(200):
        row = dict(base, event_id="ep{}-t{}".format(step, step))
        req = build_request(row, agent_positions=POSITIONS, grid_shape=GRID)
        seen.add(provider.query(req).selected_plan_id)
    assert len(seen) == len(all_plans())


@pytest.mark.parametrize("name", ["heuristic_min_travel", "random_valid_plan"])
def test_offline_selectors_produce_responses_the_validator_accepts(name):
    kwargs = {"run_seed": 11} if needs_run_seed(name) else {}
    provider = make_provider(name, **kwargs)
    req = request_for()
    verdict = validate(req, provider.query(req))
    assert verdict.valid, verdict.reason


@pytest.mark.parametrize("name", ["heuristic_min_travel", "random_valid_plan"])
def test_offline_selectors_report_no_usage_and_no_cost(name):
    kwargs = {"run_seed": 11} if needs_run_seed(name) else {}
    provider = make_provider(name, **kwargs)
    provider.query(request_for())
    assert provider.last_usage.cost_usd is None
    assert provider.last_usage.tokens_prompt is None


@pytest.mark.parametrize("name", ["heuristic_min_travel", "random_valid_plan"])
def test_offline_selectors_return_none_when_no_plan_exists(name):
    """Fewer admissible cells than recipients: no plan, no guidance, F_t = 0."""
    kwargs = {"run_seed": 11} if needs_run_seed(name) else {}
    provider = make_provider(name, **kwargs)
    row = event_row(cells=((4, 3),))
    req = build_request(row, agent_positions=POSITIONS, grid_shape=GRID)
    assert req.candidate_joint_plans == ()
    assert provider.query(req) is None


# -- 6. configuration surface -------------------------------------------------


class _Args(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


def _active_args(**overrides):
    base = {
        "selma_mode": "active",
        "selma_provider": "heuristic_min_travel",
        "selma_variant_id": FROZEN_ACTIVE_VARIANT,
        "selma_beta": 0.05,
        "common_reward": True,
    }
    base.update(overrides)
    return _Args(base)


def test_config_defaults_to_the_v1_potential():
    assert validate_selma_config(_active_args()).potential_mode == POTENTIAL_FOOD


def test_config_accepts_the_plan_potential_in_active_mode():
    cfg = validate_selma_config(_active_args(selma_potential_mode="plan"))
    assert cfg.potential_mode == POTENTIAL_PLAN


def test_config_rejects_an_unknown_potential():
    with pytest.raises(SelmaConfigError):
        validate_selma_config(_active_args(selma_potential_mode="distance"))


def test_config_rejects_the_plan_potential_outside_active_mode():
    with pytest.raises(SelmaConfigError):
        validate_selma_config(_Args({"selma_mode": "off",
                                     "selma_potential_mode": "plan"}))


def test_config_accepts_both_offline_selectors():
    for name in ("heuristic_min_travel", "random_valid_plan"):
        cfg = validate_selma_config(_active_args(selma_provider=name))
        assert cfg.provider == name


# -- 7. an invalid response can never reach the reward ------------------------


def test_rejected_response_leaves_no_guidance_and_no_shaping():
    """No active guidance means phi = 0 in V2 exactly as in V1."""
    ctrl = GuidanceController(
        provider=DeterministicMockProvider(), grid_shape=GRID, beta=1.0,
        gamma=0.99, enabled=True, variant_id=FROZEN_ACTIVE_VARIANT,
        potential_mode=POTENTIAL_PLAN)
    assert ctrl.active is None
    assert ctrl.potential(POSITIONS, (FOOD,)) == 0.0
    assert ctrl.potential_components(POSITIONS, (FOOD,)) == {}
