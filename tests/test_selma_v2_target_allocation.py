"""Offline proof that SELMA V2 makes provider content causally effective."""

import json
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from selma.guidance import (
    EXPIRY_TARGET_REMOVED,
    ActiveGuidance,
    GuidanceController,
    phi,
    phi_components,
)
from selma.cache import (
    build_cache_key,
    canonical_json,
    response_from_dict,
    response_to_dict,
)
from selma.modes import FROZEN_ACTIVE_VARIANT, SelmaConfigError, validate_selma_config
from selma.openai_provider import (
    OpenAIGuidanceProvider,
    V2_SYSTEM_PROMPT,
    build_prompt,
)
from selma.provider import (
    CONTRACT_V2_TARGET_ALLOCATION,
    ROLE_PURSUE_TARGET,
    DeterministicMockProvider,
    TargetGuidanceAssignment,
    build_request,
    validate,
)
from selma.types import FoodSnapshot


GRID = (10, 10)
POSITIONS = {0: (4, 3), 1: (1, 2), 2: (8, 7)}
LEVELS = {0: 1, 1: 1, 2: 1}
FOODS = (
    FoodSnapshot((4, 4), 2),
    FoodSnapshot((1, 1), 1),
    FoodSnapshot((8, 8), 1),
)


def event_row():
    return {
        "row_type": "coalesced_event",
        "event_id": "ep7-v2-food4,4-t3",
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


def request():
    return build_request(
        event_row(), POSITIONS, GRID,
        guidance_contract=CONTRACT_V2_TARGET_ALLOCATION,
        agent_levels=LEVELS,
        remaining_foods=FOODS,
    )


def active_from_plan(plan):
    return ActiveGuidance(
        event_id="e", subgoal_id="sg", food_id=(7, 4, 4),
        food_position=(4, 4), recipients=(0, 1, 2), assignments={},
        activated_step=3, provider_id="test", selected_plan_id=plan.plan_id,
        guidance_contract=CONTRACT_V2_TARGET_ALLOCATION,
        target_food_ids={a.agent_id: a.target_food_id for a in plan.assignments},
        target_positions={a.agent_id: a.target_position for a in plan.assignments},
    )


def test_v2_offers_multiple_complete_feasible_target_allocations():
    req = request()
    assert req.guidance_contract == CONTRACT_V2_TARGET_ALLOCATION
    assert req.recipient_agents == (0, 1, 2)
    assert len(req.candidate_joint_plans) >= 2
    trigger = req.food_id
    for plan in req.candidate_joint_plans:
        assert set(plan.mapping()) == {0, 1, 2}
        assert trigger in set(plan.mapping().values())


def test_different_selected_plans_change_phi_components_in_same_state():
    req = request()
    first = phi_components(active_from_plan(req.candidate_joint_plans[0]),
                           POSITIONS, [f.position for f in FOODS], GRID)
    assert any(
        phi_components(active_from_plan(plan), POSITIONS,
                       [f.position for f in FOODS], GRID) != first
        for plan in req.candidate_joint_plans[1:]
    )
    values = {
        phi(active_from_plan(plan), POSITIONS,
            [f.position for f in FOODS], GRID)
        for plan in req.candidate_joint_plans
    }
    assert len(values) >= 2  # the scalar team reward is causally different


def test_moving_toward_own_selected_target_strictly_improves_phi():
    plan = request().candidate_joint_plans[0]
    guidance = active_from_plan(plan)
    agent = plan.assignments[0].agent_id
    target = plan.assignments[0].target_position
    start = POSITIONS[agent]
    if start[0] != target[0]:
        moved = (start[0] + (1 if target[0] > start[0] else -1), start[1])
    else:
        moved = (start[0], start[1] + (1 if target[1] > start[1] else -1))
    before = phi_components(guidance, POSITIONS,
                            [f.position for f in FOODS], GRID)[agent]
    after_positions = dict(POSITIONS)
    after_positions[agent] = moved
    after = phi_components(guidance, after_positions,
                           [f.position for f in FOODS], GRID)[agent]
    assert after > before


def test_provider_selection_reaches_active_guidance_and_potential():
    provider = DeterministicMockProvider()
    controller = GuidanceController(
        provider=provider, grid_shape=GRID, beta=0.1, gamma=0.99,
        enabled=True, variant_id=FROZEN_ACTIVE_VARIANT,
        guidance_contract=CONTRACT_V2_TARGET_ALLOCATION,
    )
    record = controller.apply_transition(
        step=3, raw_team_reward=0.0, phi_before=0.0,
        post_agent_positions=POSITIONS,
        post_food_positions=[f.position for f in FOODS],
        post_agent_levels=LEVELS, post_food_snapshots=FOODS,
        event_rows=[event_row()], episode_ended=False,
        stored_terminated=False,
    )
    assert record.provider_valid == 1
    assert controller.active.selected_plan_id.startswith("target_plan_")
    assert controller.active.target_positions
    assert record.phi_components_after == phi_components(
        controller.active, POSITIONS, [f.position for f in FOODS], GRID)
    assert record.phi_after_effective != 0.0


def test_v2_activation_and_target_expiry_telescope_exactly():
    controller = GuidanceController(
        provider=DeterministicMockProvider(), grid_shape=GRID,
        beta=0.1, gamma=0.99, enabled=True,
        variant_id=FROZEN_ACTIVE_VARIANT,
        guidance_contract=CONTRACT_V2_TARGET_ALLOCATION,
    )
    first = controller.apply_transition(
        step=3, raw_team_reward=0.0, phi_before=0.0,
        post_agent_positions=POSITIONS,
        post_food_positions=[f.position for f in FOODS],
        post_agent_levels=LEVELS, post_food_snapshots=FOODS,
        event_rows=[event_row()], episode_ended=False,
        stored_terminated=False,
    )
    assigned_target = next(iter(controller.active.target_positions.values()))
    remaining = tuple(f for f in FOODS if f.position != assigned_target)
    second = controller.apply_transition(
        step=4, raw_team_reward=0.0,
        phi_before=first.phi_after_effective,
        post_agent_positions=POSITIONS,
        post_food_positions=[f.position for f in remaining],
        post_agent_levels=LEVELS, post_food_snapshots=remaining,
        event_rows=(), episode_ended=False, stored_terminated=False,
    )
    assert second.expiry_reason == EXPIRY_TARGET_REMOVED
    assert second.phi_after_effective == 0.0
    assert abs(first.shaping_contribution
               + controller.gamma * second.shaping_contribution) < 1e-12


def test_validator_rejects_tampered_target_even_with_offered_plan_id():
    req = request()
    good = DeterministicMockProvider().query(req)
    assert validate(req, good).valid
    first = good.assignments[0]
    bad_assignment = TargetGuidanceAssignment(
        agent_id=first.agent_id, role=ROLE_PURSUE_TARGET,
        loading_cell=(9, 9), target_food_id=(7, 9, 9),
        target_position=(9, 9), target_level=1,
    )
    bad = good._replace(assignments=(bad_assignment,) + good.assignments[1:])
    assert validate(req, bad).reason == "target_not_remaining"


def test_v2_prompt_exposes_only_complete_offered_plan_selection():
    req = request()
    payload = json.loads(build_prompt(req))
    assert payload["guidance_contract"] == CONTRACT_V2_TARGET_ALLOCATION
    assert len(payload["candidate_joint_plans"]) == len(req.candidate_joint_plans)
    assert "reward" not in payload
    text = build_prompt(req)
    assert req.event_id not in text
    assert '"food_id"' not in text


def test_live_adapter_shape_selects_v2_plan_without_network():
    class FakeResponses:
        def __init__(self):
            self.kwargs = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            ids = kwargs["text"]["format"]["schema"]["properties"][
                "selected_plan_id"]["enum"]
            return {
                "id": "fake-request",
                "model": "gpt-5.4-nano-2026-03-17",
                "status": "completed",
                "output_text": json.dumps({
                    "selected_plan_id": ids[1], "confidence": 0.8}),
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }

    class FakeClient:
        def __init__(self):
            self.responses = FakeResponses()

    req = request()
    client = FakeClient()
    provider = OpenAIGuidanceProvider(client=client, max_retries=0)
    response = provider.query(req)
    assert response.selected_plan_id == req.candidate_joint_plans[1].plan_id
    assert validate(req, response).valid
    assert client.responses.kwargs["input"][0]["content"] == V2_SYSTEM_PROMPT


def test_v2_config_is_explicit_and_unknown_contract_fails_closed():
    base = dict(
        selma_mode="active", selma_provider="deterministic_mock",
        selma_beta=0.1, selma_variant_id=FROZEN_ACTIVE_VARIANT,
        common_reward=True,
    )
    cfg = validate_selma_config(SimpleNamespace(
        **base, selma_guidance_contract=CONTRACT_V2_TARGET_ALLOCATION))
    assert cfg.guidance_contract == CONTRACT_V2_TARGET_ALLOCATION
    with pytest.raises(SelmaConfigError):
        validate_selma_config(SimpleNamespace(
            **base, selma_guidance_contract="v3_magic"))


def test_v2_plan_ids_and_cache_question_are_episode_independent():
    first = request()
    row = event_row()
    row["event_id"] = "ep8-v2-food4,4-t3"
    row["food_id"] = (8, 4, 4)
    second = build_request(
        row, POSITIONS, GRID,
        guidance_contract=CONTRACT_V2_TARGET_ALLOCATION,
        agent_levels=LEVELS, remaining_foods=FOODS,
    )
    context = dict(
        condition="lbforaging:Foraging-10x10-3p-3f-v3",
        grid_shape=GRID, provider="openai", model_snapshot="dated-test-model",
        reasoning_effort="none", prompt_version="v3",
        response_schema_version="v2",
    )
    assert canonical_json(build_cache_key(first, **context)) == canonical_json(
        build_cache_key(second, **context))
    cached = response_to_dict(DeterministicMockProvider().query(first))
    rebound = response_from_dict(
        cached, event_id=second.event_id, target_food_id=second.food_id)
    assert validate(second, rebound).valid


def test_v1_remains_the_default_contract():
    req = build_request(event_row(), {0: POSITIONS[0], 1: POSITIONS[1]}, GRID)
    assert req.guidance_contract == "v1_loading_cell"
    assert all(hasattr(a, "loading_cell")
               for plan in req.candidate_joint_plans for a in plan.assignments)
