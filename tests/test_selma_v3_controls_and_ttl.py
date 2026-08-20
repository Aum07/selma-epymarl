"""SELMA V3: configurable advice lifetime and the contract-aware controls.

Two additions are covered.

**Advice lifetime.** ``TTL_STEPS`` was frozen at 15 for 8x8, where
``d_max_plus_load`` is also 15, so an agent could always reach any assigned
target before its advice expired. That coincidence does not hold on larger
grids -- 19 on 10x10, 29 on 15x15 -- where a distant target loses its guidance
mid-journey every time. ``selma_ttl_steps`` makes the lifetime scale with the
map; 15 remains the default so every historical command is unchanged.

**Offline controls.** ``heuristic_min_travel`` and ``random_valid_plan`` answer
the same question over the same answer set as the live provider, under either
guidance contract, and differ only in policy.

Run:  PYTHONPATH=src pytest tests/test_selma_v3_controls_and_ttl.py -v
"""

import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from selma.guidance import (  # noqa: E402
    TTL_STEPS,
    ActiveGuidance,
    GuidanceController,
    phi,
    phi_components,
)
from selma.modes import FROZEN_ACTIVE_VARIANT, SelmaConfigError, validate_selma_config  # noqa: E402
from selma.offline_selectors import (  # noqa: E402
    HeuristicMinTravelProvider,
    RandomValidPlanProvider,
    plan_total_travel,
)
from selma.provider import (  # noqa: E402
    CONTRACT_V2_TARGET_ALLOCATION,
    DeterministicMockProvider,
    build_request,
    validate,
)
from selma.provider_registry import make_provider, needs_run_seed  # noqa: E402

GRID = (10, 10)
FOOD = (1, 6)
POSITIONS = {0: (1, 1), 1: (5, 5), 2: (7, 3)}
LEVELS = {0: 1, 1: 2, 2: 3}
# Plain 10x10-3p-3f: mixed food levels, so subsets can clear some foods alone
# and allocation is a genuine choice. The -coop variant would collapse this to
# a single feasible plan, which is why it is not used.
REMAINING = [((1, 6), 3), ((4, 2), 4), ((7, 7), 2)]   # (position, level) pairs


def event_row(recipients=(0, 1, 2)):
    return {
        "row_type": "coalesced_event",
        "event_id": "ep1-{}-food1,6-t3".format(FROZEN_ACTIVE_VARIANT),
        "variant_id": FROZEN_ACTIVE_VARIANT,
        "food_id": (1, FOOD[0], FOOD[1]),
        "food_position": FOOD,
        "food_level": 3,
        "trigger_agent_ids": (0,),
        "recipient_ids": recipients,
        "loader_ids": recipients,
        "loader_levels": tuple(LEVELS[a] for a in recipients),
        "level_deficit": 0,
        "required_extra_level": 0,
        "candidate_loading_cells": ((1, 5), (1, 7), (0, 6), (2, 6)),
        "coalition_feasible": True,
        "suppression_reason": "",
    }


def v2_request(positions=None):
    return build_request(
        event_row(), agent_positions=positions or POSITIONS, grid_shape=GRID,
        guidance_contract=CONTRACT_V2_TARGET_ALLOCATION,
        agent_levels=LEVELS, remaining_foods=REMAINING)


def v1_request(positions=None):
    return build_request(event_row(), agent_positions=positions or POSITIONS,
                         grid_shape=GRID)


# -- 1. the environment actually offers a decision ---------------------------


def test_plain_10x10_offers_a_real_choice():
    """The premise of the whole redesign: more than one feasible allocation."""
    plans = v2_request().candidate_joint_plans
    assert len(plans) >= 2, (
        "plain 10x10-3p-3f must offer multiple allocations; got {}".format(len(plans)))
    targets = {tuple(sorted(tuple(a.target_position) for a in p.assignments))
               for p in plans}
    assert len(targets) >= 2, "offered plans must differ in where agents are sent"


def test_coop_style_levels_collapse_to_one_plan():
    """Pins WHY the -coop variant is unusable, so it cannot be chosen by mistake."""
    from selma.target_plans import enumerate_target_plans
    total = sum(LEVELS.values())
    coop = [(pos, total) for pos, _ in REMAINING]   # every food needs everyone
    plans = enumerate_target_plans([0, 1, 2], LEVELS, coop, (1, 1, 6),
                                   agent_positions=POSITIONS, grid_shape=GRID)
    assert len(plans) == 1, (
        "a coop-style level assignment must leave exactly one feasible plan; "
        "got {}".format(len(plans)))


# -- 2. configurable advice lifetime ----------------------------------------


def test_ttl_defaults_to_the_frozen_value():
    assert TTL_STEPS == 15
    ctrl = GuidanceController(provider=DeterministicMockProvider(), grid_shape=GRID,
                              beta=0.0, gamma=0.99, enabled=True,
                              variant_id=FROZEN_ACTIVE_VARIANT)
    assert ctrl.ttl_steps == 15


def test_ttl_can_be_raised_and_is_carried_onto_the_guidance():
    ctrl = GuidanceController(provider=DeterministicMockProvider(), grid_shape=GRID,
                              beta=0.0, gamma=0.99, enabled=True,
                              variant_id=FROZEN_ACTIVE_VARIANT, ttl_steps=25)
    assert ctrl.ttl_steps == 25


def test_guidance_expires_exactly_at_its_own_ttl():
    for ttl in (3, 15, 25):
        g = ActiveGuidance(event_id="e", subgoal_id="sg", food_id=(1,) + FOOD,
                           food_position=FOOD, recipients=(0, 1), assignments={},
                           activated_step=0, provider_id="test", ttl_steps=ttl)
        for served in range(ttl):
            g.transitions_served = served
            assert not g.ttl_exhausted(), "expired early at {}/{}".format(served, ttl)
        g.transitions_served = ttl
        assert g.ttl_exhausted()


def test_a_ttl_below_one_is_rejected():
    with pytest.raises(ValueError):
        ActiveGuidance(event_id="e", subgoal_id="sg", food_id=(1,) + FOOD,
                       food_position=FOOD, recipients=(0,), assignments={},
                       activated_step=0, provider_id="t", ttl_steps=0)
    with pytest.raises(ValueError):
        GuidanceController(provider=DeterministicMockProvider(), grid_shape=GRID,
                           beta=0.0, gamma=0.99, enabled=True,
                           variant_id=FROZEN_ACTIVE_VARIANT, ttl_steps=0)


# -- 3. the potential fails closed ------------------------------------------


def test_missing_target_gives_no_shaping_rather_than_partial_shaping():
    """A recipient without a target must give phi = 0, not phi over the rest."""
    g = ActiveGuidance(
        event_id="e", subgoal_id="sg", food_id=(1,) + FOOD, food_position=FOOD,
        recipients=(0, 1), assignments={}, activated_step=0, provider_id="t",
        guidance_contract="v2_target_allocation",
        target_positions={0: (1, 6)})           # agent 1 deliberately missing
    assert phi_components(g, POSITIONS, (FOOD,), GRID) == {}
    assert phi(g, POSITIONS, (FOOD,), GRID) == 0.0


# -- 4. the controls, under both contracts ----------------------------------


@pytest.mark.parametrize("contract", ["v1", "v2"])
def test_heuristic_selects_the_minimum_travel_plan(contract):
    req = v1_request() if contract == "v1" else v2_request()
    positions = {int(c.agent_id): tuple(int(x) for x in c.position)
                 for c in req.agent_context}
    costs = {p.plan_id: plan_total_travel(p, positions)
             for p in req.candidate_joint_plans}
    chosen = HeuristicMinTravelProvider().query(req)
    assert costs[chosen.selected_plan_id] == min(costs.values())


@pytest.mark.parametrize("contract", ["v1", "v2"])
def test_controls_produce_responses_the_validator_accepts(contract):
    req = v1_request() if contract == "v1" else v2_request()
    for provider in (HeuristicMinTravelProvider(), RandomValidPlanProvider(run_seed=7)):
        verdict = validate(req, provider.query(req))
        assert verdict.valid, "{}: {}".format(provider.provider_id, verdict.reason)


def test_random_control_is_reproducible_and_does_not_touch_global_rng():
    req = v2_request()
    a = RandomValidPlanProvider(run_seed=1).query(req).selected_plan_id
    b = RandomValidPlanProvider(run_seed=1).query(req).selected_plan_id
    assert a == b
    random.seed(4242)
    expected = [random.random() for _ in range(3)]
    random.seed(4242)
    RandomValidPlanProvider(run_seed=9).query(req)
    assert [random.random() for _ in range(3)] == expected


def test_random_control_spans_the_offered_plans():
    provider = RandomValidPlanProvider(run_seed=3)
    seen = set()
    base = event_row()
    for i in range(400):
        row = dict(base, event_id="ep{}-t{}".format(i, i))
        req = build_request(row, agent_positions=POSITIONS, grid_shape=GRID,
                            guidance_contract=CONTRACT_V2_TARGET_ALLOCATION,
                            agent_levels=LEVELS, remaining_foods=REMAINING)
        seen.add(provider.query(req).selected_plan_id)
    assert len(seen) == len(v2_request().candidate_joint_plans)


@pytest.mark.parametrize("name", ["heuristic_min_travel", "random_valid_plan"])
def test_controls_are_free_and_report_no_usage(name):
    kwargs = {"run_seed": 5} if needs_run_seed(name) else {}
    provider = make_provider(name, **kwargs)
    provider.query(v2_request())
    assert provider.last_usage.cost_usd is None
    assert provider.last_usage.tokens_prompt is None


# -- 5. configuration surface -----------------------------------------------


class _Args(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


def _active(**over):
    base = {"selma_mode": "active", "selma_provider": "heuristic_min_travel",
            "selma_variant_id": FROZEN_ACTIVE_VARIANT, "selma_beta": 0.05,
            "common_reward": True}
    base.update(over)
    return _Args(base)


def test_config_defaults_ttl_to_fifteen():
    assert validate_selma_config(_active()).ttl_steps == 15


def test_config_accepts_a_scaled_ttl():
    assert validate_selma_config(_active(selma_ttl_steps=25)).ttl_steps == 25


def test_config_rejects_a_nonsense_ttl():
    for bad in (0, -1, "abc"):
        with pytest.raises(SelmaConfigError):
            validate_selma_config(_active(selma_ttl_steps=bad))


def test_config_rejects_a_raised_ttl_outside_active_mode():
    with pytest.raises(SelmaConfigError):
        validate_selma_config(_Args({"selma_mode": "off", "selma_ttl_steps": 25}))


@pytest.mark.parametrize("name", ["heuristic_min_travel", "random_valid_plan"])
def test_config_accepts_both_controls(name):
    assert validate_selma_config(_active(selma_provider=name)).provider == name
