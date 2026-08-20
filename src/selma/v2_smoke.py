"""No-network SELMA V2 causal-wiring smoke check.

Run from the repository root::

    PYTHONPATH=src python -m selma.v2_smoke

This does not train an agent and does not call an LLM.  It proves the contract
that must hold before either is worth doing: a provider-selected plan reaches
active guidance, different selected plans produce different per-agent
potentials in the same state, and the chosen plan contributes to PBRS.
"""

import json

from selma.guidance import ActiveGuidance, GuidanceController, phi, phi_components
from selma.modes import FROZEN_ACTIVE_VARIANT
from selma.provider import (
    CONTRACT_V2_TARGET_ALLOCATION,
    DeterministicMockProvider,
    build_request,
)
from selma.types import FoodSnapshot


def _event():
    return {
        "row_type": "coalesced_event",
        "event_id": "ep7-v2-smoke-food4,4-t3",
        "variant_id": FROZEN_ACTIVE_VARIANT,
        "food_id": (7, 4, 4),
        "food_position": (4, 4),
        "food_level": 2,
        "trigger_agent_ids": (0,),
        "recipient_ids": (0, 1),
        "loader_ids": (0,),
        "loader_levels": (1,),
        "helper_ids": (1,),
        "helper_levels": (1,),
        "level_deficit": 1,
        "required_extra_level": 1,
        "candidate_loading_cells": ((3, 4), (4, 3), (4, 5), (5, 4)),
        "coalition_feasible": 1,
        "suppression_reason": "",
    }


def _active(plan):
    return ActiveGuidance(
        event_id="smoke", subgoal_id="smoke", food_id=(7, 4, 4),
        food_position=(4, 4), recipients=(0, 1, 2), assignments={},
        activated_step=3, provider_id="deterministic_mock",
        selected_plan_id=plan.plan_id,
        guidance_contract=CONTRACT_V2_TARGET_ALLOCATION,
        target_food_ids={a.agent_id: a.target_food_id for a in plan.assignments},
        target_positions={a.agent_id: a.target_position for a in plan.assignments},
    )


def run_smoke():
    grid = (10, 10)
    positions = {0: (4, 3), 1: (1, 2), 2: (8, 7)}
    levels = {0: 1, 1: 1, 2: 1}
    foods = (
        FoodSnapshot((4, 4), 2),
        FoodSnapshot((1, 1), 1),
        FoodSnapshot((8, 8), 1),
    )
    food_positions = tuple(f.position for f in foods)
    request = build_request(
        _event(), positions, grid,
        guidance_contract=CONTRACT_V2_TARGET_ALLOCATION,
        agent_levels=levels, remaining_foods=foods,
    )
    if len(request.candidate_joint_plans) < 2:
        raise AssertionError("V2 smoke state did not produce a real plan choice")

    component_sets = [
        phi_components(_active(plan), positions, food_positions, grid)
        for plan in request.candidate_joint_plans
    ]
    if all(components == component_sets[0] for components in component_sets[1:]):
        raise AssertionError("different V2 plans are still potential-invariant")
    joint_potentials = [
        phi(_active(plan), positions, food_positions, grid)
        for plan in request.candidate_joint_plans
    ]
    if len(set(joint_potentials)) < 2:
        raise AssertionError("different V2 plans are still team-reward invariant")

    provider = DeterministicMockProvider()
    controller = GuidanceController(
        provider=provider, grid_shape=grid, beta=0.1, gamma=0.99,
        enabled=True, variant_id=FROZEN_ACTIVE_VARIANT,
        guidance_contract=CONTRACT_V2_TARGET_ALLOCATION,
    )
    record = controller.apply_transition(
        step=3, raw_team_reward=0.0, phi_before=0.0,
        post_agent_positions=positions,
        post_food_positions=food_positions,
        post_agent_levels=levels,
        post_food_snapshots=foods,
        event_rows=(_event(),),
        episode_ended=False,
        stored_terminated=False,
    )
    if not record.provider_valid or controller.active is None:
        raise AssertionError("provider selection did not activate V2 guidance")
    if not controller.active.target_positions:
        raise AssertionError("selected target mapping was lost before potential")
    if record.phi_after_effective == 0.0:
        raise AssertionError("selected V2 plan did not reach PBRS")

    return {
        "status": "PASS",
        "guidance_contract": CONTRACT_V2_TARGET_ALLOCATION,
        "candidate_plan_count": len(request.candidate_joint_plans),
        "selected_plan_id": controller.active.selected_plan_id,
        "target_assignments": controller.active.target_positions,
        "phi_components": record.phi_components_after,
        "distinct_joint_potentials": len(set(joint_potentials)),
        "phi_after_effective": record.phi_after_effective,
        "shaping_contribution": record.shaping_contribution,
        "provider_calls": provider.call_count,
        "network_calls": 0,
    }


def main():
    print(json.dumps(run_smoke(), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
