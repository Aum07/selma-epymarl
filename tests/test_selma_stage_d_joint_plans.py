"""SELMA Stage D: the joint-plan guidance contract.

Every test here was written against the pre-correction tree, where the provider
offered independently composable per-agent assignment ids and the validator
checked each assignment on its own. The live pilot returned ``a0@5,4`` *and*
``a1@5,4`` for one fixture -- two agents in one cell, which LBF cannot realise --
and the validator accepted it. These tests fail on that tree and pass on this one.

**No test here touches the network**; ``tests/conftest.py`` enforces that for the
whole suite and every provider below is a local fake or the deterministic mock.

Run:  PYTHONPATH=src pytest tests/test_selma_stage_d_joint_plans.py -v
"""

import inspect
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from selma.accounting import (  # noqa: E402
    FALLBACK_MALFORMED,
    SKIP_NO_VALID_JOINT_PLAN,
    SKIP_SINGLETON_NO_CHOICE,
    SelmaAccounting,
)
from selma.cache import (  # noqa: E402
    CACHE_KEY_VERSION,
    CACHE_SCHEMA_VERSION,
    INCLUDED_KEY_FIELDS,
    SqliteGuidanceCache,
    build_cache_key,
    canonical_json,
    key_digest,
)
from selma.gateway import CACHE_READ_WRITE, ProviderGateway  # noqa: E402
from selma.guidance import GuidanceController  # noqa: E402
from selma.joint_plans import (  # noqa: E402
    PLAN_CELL_IS_TARGET_FOOD,
    PLAN_CELL_NOT_ADJACENT,
    PLAN_CELL_NOT_OFFERED,
    PLAN_CELL_OUTSIDE_GRID,
    PLAN_DUPLICATE_AGENT,
    PLAN_DUPLICATE_CELL,
    PLAN_RECIPIENTS_NOT_EXACT,
    JointPlan,
    PlanAssignment,
    describe_plan,
    enumerate_joint_plans,
    plan_violations,
)
from selma.modes import FROZEN_ACTIVE_VARIANT  # noqa: E402
from selma.openai_provider import (  # noqa: E402
    MODEL_SNAPSHOT,
    PROMPT_VERSION,
    RESPONSE_SCHEMA_VERSION,
    OpenAIGuidanceProvider,
    build_prompt,
    response_json_schema,
)
from selma.provider import (  # noqa: E402
    REQUEST_VERSION,
    SCHEMA_VERSION,
    Assignment,
    DeterministicMockProvider,
    GuidanceResponse,
    build_request,
    validate,
)

GRID = (8, 8)
FOOD = (4, 4)
CELLS = ((3, 4), (5, 4), (4, 3), (4, 5))
POSITIONS = {0: (4, 3), 1: (6, 6)}
CONDITION = "lbforaging:Foraging-8x8-2p-2f-coop-v3"


def event_row(episode_id=7, food_pos=FOOD, recipients=(0, 1), cells=CELLS,
              loader_levels=(1,), helper_levels=(2,), food_level=3, step=3):
    return {
        "row_type": "coalesced_event",
        "event_id": "ep{}-{}-food{},{}-t{}".format(
            episode_id, FROZEN_ACTIVE_VARIANT, food_pos[0], food_pos[1], step),
        "variant_id": FROZEN_ACTIVE_VARIANT,
        "food_id": (episode_id, food_pos[0], food_pos[1]),
        "food_position": food_pos,
        "food_level": food_level,
        "trigger_agent_ids": (0,),
        "recipient_ids": recipients,
        "loader_ids": (0,),
        "loader_levels": loader_levels,
        "helper_ids": (1,),
        "helper_levels": helper_levels,
        "level_deficit": food_level - sum(loader_levels),
        "required_extra_level": max(0, food_level - sum(loader_levels)),
        "candidate_loading_cells": cells,
        "coalition_feasible": 1,
        "suppression_reason": "",
    }


def make_request(positions=None, grid_shape=GRID, **row_kwargs):
    return build_request(event_row(**row_kwargs),
                         agent_positions=positions or POSITIONS,
                         grid_shape=grid_shape)


def good_response(request, plan=None):
    """A well-formed response selecting one offered plan."""
    plan = plan or request.candidate_joint_plans[0]
    return GuidanceResponse(
        schema_version=SCHEMA_VERSION,
        event_id=request.event_id,
        recipients=tuple(sorted(request.recipient_agents)),
        target_food_id=tuple(request.food_id),
        helper_coalition=tuple(sorted(request.recipient_agents)),
        selected_plan_id=plan.plan_id,
        assignments=tuple(
            Assignment(agent_id=a.agent_id, role="approach_and_load",
                       loading_cell=a.loading_cell)
            for a in plan.assignments),
        confidence=0.9,
    )


def with_assignments(request, pairs, plan_id=None):
    """A response whose assignments are stated directly (for tampering tests).

    ``recipients`` deliberately keeps the full required set, so these fixtures
    exercise the *joint-plan* constraints rather than tripping the earlier
    response-level recipients check.
    """
    base = good_response(request)
    return base._replace(
        selected_plan_id=plan_id or base.selected_plan_id,
        assignments=tuple(
            Assignment(agent_id=int(a), role="approach_and_load",
                       loading_cell=tuple(c)) for a, c in pairs),
    )


KEY_CONTEXT = dict(
    condition=CONDITION, grid_shape=GRID, provider="openai",
    model_snapshot=MODEL_SNAPSHOT, reasoning_effort="none",
    prompt_version=PROMPT_VERSION,
    response_schema_version=RESPONSE_SCHEMA_VERSION,
)


# ==========================================================================
# 1-8. joint physical validity is enforced by the deterministic validator
# ==========================================================================


def test_1_two_agents_assigned_to_the_same_cell_are_rejected():
    """The exact live-pilot defect: `two_cells_only` returned a0@5,4 and a1@5,4."""
    request = make_request()
    bad = with_assignments(request, [(0, (4, 3)), (1, (4, 3))])
    verdict = validate(request, bad)
    assert not verdict.valid
    assert verdict.reason == PLAN_DUPLICATE_CELL
    assert verdict.response is None


def test_2_a_duplicate_recipient_agent_is_rejected():
    request = make_request()
    bad = with_assignments(request, [(0, (4, 3)), (0, (3, 4))])
    verdict = validate(request, bad)
    assert not verdict.valid
    assert verdict.reason == PLAN_DUPLICATE_AGENT


def test_2b_a_response_recipient_set_that_is_not_the_required_set_is_rejected():
    """The response-level recipients field is checked before the plan itself."""
    request = make_request()
    bad = good_response(request)._replace(recipients=(0,))
    assert validate(request, bad).reason == "recipients_not_exact"


def test_3_a_missing_required_recipient_is_rejected():
    request = make_request()
    bad = with_assignments(request, [(0, (4, 3))])
    verdict = validate(request, bad)
    assert not verdict.valid
    assert verdict.reason == PLAN_RECIPIENTS_NOT_EXACT


def test_4_an_extra_recipient_is_rejected():
    request = make_request()
    bad = with_assignments(request, [(0, (4, 3)), (1, (3, 4)), (2, (5, 4))])
    verdict = validate(request, bad)
    assert not verdict.valid
    assert verdict.reason == PLAN_RECIPIENTS_NOT_EXACT


def test_5_a_cell_outside_the_grid_is_rejected():
    request = make_request()
    bad = with_assignments(request, [(0, (4, 3)), (1, (-1, 4))])
    verdict = validate(request, bad)
    assert not verdict.valid
    assert verdict.reason == PLAN_CELL_OUTSIDE_GRID


def test_5b_a_cell_beyond_the_far_edge_is_rejected():
    request = make_request()
    bad = with_assignments(request, [(0, (4, 3)), (1, (8, 4))])
    assert not validate(request, bad).valid


def test_6_a_cell_not_adjacent_to_the_target_food_is_rejected():
    request = make_request()
    bad = with_assignments(request, [(0, (4, 3)), (1, (0, 0))])
    verdict = validate(request, bad)
    assert not verdict.valid
    assert verdict.reason in (PLAN_CELL_NOT_ADJACENT, PLAN_CELL_NOT_OFFERED)


def test_6b_a_diagonal_neighbour_is_not_adjacent():
    request = make_request()
    bad = with_assignments(request, [(0, (4, 3)), (1, (3, 3))])
    assert not validate(request, bad).valid


def test_7_the_target_food_cell_is_never_a_valid_destination():
    request = make_request()
    bad = with_assignments(request, [(0, (4, 3)), (1, FOOD)])
    verdict = validate(request, bad)
    assert not verdict.valid
    assert verdict.reason in (PLAN_CELL_IS_TARGET_FOOD, PLAN_CELL_NOT_OFFERED)


def test_7b_the_food_cell_is_never_enumerated_even_if_the_detector_offers_it():
    plans = enumerate_joint_plans((0, 1), CELLS + (FOOD,), FOOD, GRID)
    assert plans
    for plan in plans:
        assert FOOD not in plan.cells


def test_8_assignments_outside_the_selected_joint_plan_are_rejected():
    """The id and the mapping must agree; neither alone is authoritative."""
    request = make_request()
    plans = {p.plan_id: p for p in request.candidate_joint_plans}
    first = request.candidate_joint_plans[0]
    other = next(p for p in request.candidate_joint_plans
                 if p.mapping() != first.mapping())
    # a valid plan id, but the assignments of a different valid plan
    bad = good_response(request, other)._replace(selected_plan_id=first.plan_id)
    verdict = validate(request, bad)
    assert not verdict.valid
    assert verdict.reason == "plan_assignment_mismatch"
    assert plans[first.plan_id].mapping() != other.mapping()


def test_9_an_unknown_plan_id_is_rejected():
    request = make_request()
    bad = good_response(request)._replace(selected_plan_id="plan_9999")
    verdict = validate(request, bad)
    assert not verdict.valid
    assert verdict.reason == "unknown_plan_id"


@pytest.mark.parametrize("plan_id", ["", "plan_", "PLAN_0", "a0@4,3", None])
def test_9b_malformed_plan_ids_are_rejected(plan_id):
    request = make_request()
    bad = good_response(request)._replace(selected_plan_id=plan_id)
    assert not validate(request, bad).valid


def test_10_a_cache_hit_carrying_a_now_invalid_joint_plan_is_rejected(tmp_path):
    """A stored plan is re-validated against the live request, never trusted."""
    from selma.cache import response_from_dict, response_to_dict

    request = make_request()
    stored = response_to_dict(good_response(request))
    # the stored plan is tampered with: both agents onto one cell
    stored["assignments"] = [
        {"agent_id": 0, "role": "approach_and_load", "loading_cell": [4, 3]},
        {"agent_id": 1, "role": "approach_and_load", "loading_cell": [4, 3]},
    ]
    rebuilt = response_from_dict(stored, event_id=request.event_id,
                                 target_food_id=request.food_id)
    verdict = validate(request, rebuilt)
    assert not verdict.valid
    assert verdict.reason == PLAN_DUPLICATE_CELL


def test_10b_a_cached_plan_from_a_different_layout_is_rejected():
    """Plan ids are positional: the same id means different things elsewhere."""
    here = make_request()
    elsewhere = make_request(food_pos=(2, 2), cells=((1, 2), (3, 2), (2, 1), (2, 3)),
                             positions={0: (2, 1), 1: (6, 6)})
    stale = good_response(elsewhere)._replace(
        event_id=here.event_id, target_food_id=here.food_id)
    assert not validate(here, stale).valid


# ==========================================================================
# 11-13. enumeration is total, canonical and symmetry-free
# ==========================================================================


def test_11_enumeration_never_produces_a_plan_with_duplicate_cells():
    for cells in (CELLS, CELLS[:2], CELLS[:3]):
        for recipients in ((0, 1), (0, 1, 2)):
            for plan in enumerate_joint_plans(recipients, cells, FOOD, GRID):
                assert len(set(plan.cells)) == len(plan.cells)
                assert len(set(plan.agent_ids)) == len(plan.agent_ids)
                assert sorted(plan.agent_ids) == sorted(recipients)


def test_11b_every_enumerated_plan_passes_the_independent_violation_check():
    request = make_request()
    for plan in request.candidate_joint_plans:
        assert plan_violations(
            plan.assignments, recipient_agents=request.recipient_agents,
            candidate_cells=CELLS, food_position=FOOD, grid_shape=GRID) == []


def test_12_enumeration_is_canonical_across_input_ordering():
    """Neither agent order nor cell order may change the plans or their ids."""
    baseline = enumerate_joint_plans((0, 1), CELLS, FOOD, GRID)
    for cells in (tuple(reversed(CELLS)), (CELLS[2], CELLS[0], CELLS[3], CELLS[1])):
        for recipients in ((0, 1), (1, 0)):
            assert enumerate_joint_plans(recipients, cells, FOOD, GRID) == baseline
    # duplicated input cells must not create duplicated plans
    assert enumerate_joint_plans((0, 1), CELLS + CELLS, FOOD, GRID) == baseline


def test_12b_plan_ids_are_positional_and_stable():
    plans = enumerate_joint_plans((0, 1), CELLS, FOOD, GRID)
    assert [p.plan_id for p in plans] == ["plan_{}".format(i)
                                          for i in range(len(plans))]
    assert len(plans) == 12          # P(4,2): four admissible cells, two agents
    assert len({p.mapping() and json.dumps(p.mapping(), sort_keys=True)
                for p in plans}) == len(plans)


def test_13_mirrored_and_translated_layouts_stay_distinct_scheme_b_keys():
    base = make_request()
    translated = make_request(food_pos=(5, 5),
                              cells=((4, 5), (6, 5), (5, 4), (5, 6)),
                              positions={0: (5, 4), 1: (7, 7)})
    mirrored = make_request(food_pos=(4, 3),
                            cells=((3, 3), (5, 3), (4, 2), (4, 4)),
                            positions={0: (4, 4), 1: (6, 1)})
    digests = {
        name: key_digest(canonical_json(build_cache_key(req, **KEY_CONTEXT)))
        for name, req in (("base", base), ("translated", translated),
                          ("mirrored", mirrored))
    }
    assert len(set(digests.values())) == 3, digests


def test_13b_the_cache_key_carries_complete_plan_definitions_not_just_ids():
    """Two layouts sharing plan ids must still key differently."""
    request = make_request()
    payload = build_cache_key(request, **KEY_CONTEXT)
    assert "candidate_joint_plans" in INCLUDED_KEY_FIELDS
    assert "candidate_assignment_ids" not in INCLUDED_KEY_FIELDS
    plans = payload["candidate_joint_plans"]
    assert len(plans) == len(request.candidate_joint_plans)
    for entry in plans:
        assert set(entry) == {"plan_id", "assignments"}
        assert entry["assignments"], "the full mapping must be in the key"
        for pair in entry["assignments"]:
            assert set(pair) == {"agent_id", "loading_cell"}
    serialised = canonical_json(payload)
    assert '"loading_cell"' in serialised


# ==========================================================================
# 14-16. prompt and response schema
# ==========================================================================


def test_14_the_prompt_describes_every_joint_plan_mapping():
    request = make_request()
    payload = json.loads(build_prompt(request))
    described = payload["candidate_joint_plans"]
    assert len(described) == len(request.candidate_joint_plans)
    for entry, plan in zip(described, request.candidate_joint_plans):
        assert entry["plan_id"] == plan.plan_id
        assert entry["assignments"] == [
            {"agent_id": int(a.agent_id),
             "loading_cell": [int(a.loading_cell[0]), int(a.loading_cell[1])]}
            for a in plan.assignments]
        assert entry["description"] == describe_plan(plan)


def test_14b_the_prompt_states_the_one_plan_rule_and_leaks_no_episode_identity():
    request = make_request()
    text = build_prompt(request)
    payload = json.loads(text)
    assert "selected_plan_id" in payload["instructions"]
    assert "exactly one" in payload["instructions"].lower()
    for leaked in ("event_id", "food_id", "episode_id", "seed", "t_env"):
        assert leaked not in payload
    assert request.event_id not in text
    assert "ep7" not in text
    assert text == build_prompt(request)          # deterministic
    assert text == json.dumps(json.loads(text), sort_keys=True,
                              separators=(",", ":"))


def test_15_the_response_schema_permits_only_one_selected_plan_id():
    request = make_request()
    plan_ids = [p.plan_id for p in request.candidate_joint_plans]
    schema = response_json_schema(plan_ids)
    assert set(schema["properties"]) == {"selected_plan_id", "confidence"}
    assert schema["additionalProperties"] is False
    assert sorted(schema["required"]) == ["confidence", "selected_plan_id"]
    selected = schema["properties"]["selected_plan_id"]
    assert selected["type"] == "string"
    assert selected["enum"] == plan_ids
    # no array anywhere: a list of composable assignments must be unexpressible
    assert "array" not in json.dumps(schema)


def test_16_old_response_shapes_are_rejected():
    """The pre-correction body must not be silently reinterpreted."""
    request = make_request()
    legacy = json.dumps({"assignment_ids": ["a0@4,3", "a1@4,3"], "confidence": 0.9})
    provider = _provider(_api_response(legacy), max_retries=0)
    assert provider.query(request) is None
    assert provider.last_error == FALLBACK_MALFORMED


def test_16b_a_body_carrying_both_shapes_is_still_rejected():
    request = make_request()
    both = json.dumps({"selected_plan_id": "plan_0",
                       "assignment_ids": ["a0@4,3"], "confidence": 0.9})
    provider = _provider(_api_response(both), max_retries=0)
    assert provider.query(request) is None
    assert provider.last_error == FALLBACK_MALFORMED


# ==========================================================================
# 17-19. the zero / singleton / multiple-plan policy
# ==========================================================================


class CountingProvider(DeterministicMockProvider):
    """The deterministic mock, with the calls it received made observable."""


def _controller(provider, accounting, beta=0.0):
    return GuidanceController(provider=provider, grid_shape=GRID, beta=beta,
                              gamma=0.99, enabled=True,
                              variant_id=FROZEN_ACTIVE_VARIANT,
                              accounting=accounting)


def _apply(controller, row, positions=None, raw=0.0):
    return controller.apply_transition(
        step=3, raw_team_reward=raw, phi_before=0.0,
        post_agent_positions=positions or POSITIONS,
        post_food_positions=[row["food_position"]], event_rows=[row],
        episode_ended=False, stored_terminated=False)


def _gateway_stack(tmp_path, provider, name="c"):
    accounting = SelmaAccounting(cache_mode=CACHE_READ_WRITE, live_call_cap=None,
                                 provider=getattr(provider, "provider_id", "x"),
                                 live_provider=False)
    cache = SqliteGuidanceCache(str(tmp_path / (name + ".sqlite3")),
                                run_identity=name)
    gateway = ProviderGateway(provider, accounting=accounting, cache=cache,
                              cache_mode=CACHE_READ_WRITE, condition=CONDITION,
                              grid_shape=GRID)
    return accounting, cache, gateway


def test_17_zero_valid_plans_cause_no_call_no_lookup_and_zero_shaping(tmp_path):
    """One admissible cell for two agents: no plan exists, so nothing is asked."""
    provider = CountingProvider()
    accounting, cache, gateway = _gateway_stack(tmp_path, provider, "zero")
    controller = _controller(gateway, accounting)

    row = event_row(cells=((4, 3),))          # one cell, two recipients
    record = _apply(controller, row)

    assert accounting.zero_plan_skip_count == 1
    assert accounting.singleton_skip_count == 0
    assert accounting.multiple_plan_decision_count == 0
    assert provider.call_count == 0
    assert accounting.live_api_call_count == 0
    assert accounting.cache_lookup_count == 0
    assert accounting.cache_hit_count == accounting.cache_miss_count == 0
    assert accounting.provider_request_considered_count == 0
    assert accounting.guidance_activation_count == 0
    assert controller.active is None
    assert record.fallback_reason == SKIP_NO_VALID_JOINT_PLAN
    assert record.phi_after_effective == 0.0
    assert record.shaping_contribution == 0.0
    assert accounting.check_invariants() == []
    cache.close()


def test_18_a_singleton_plan_causes_no_call_no_lookup_and_zero_shaping(tmp_path):
    """Two admissible cells and two agents still gives 2 plans; force exactly 1."""
    provider = CountingProvider()
    accounting, cache, gateway = _gateway_stack(tmp_path, provider, "single")
    controller = _controller(gateway, accounting)

    # one recipient, one admissible cell -> exactly one possible plan
    row = event_row(recipients=(0,), cells=((4, 3),), loader_levels=(3,),
                    food_level=3)
    request = build_request(row, agent_positions=POSITIONS, grid_shape=GRID)
    assert len(request.candidate_joint_plans) == 1

    record = _apply(controller, row)

    assert accounting.singleton_skip_count == 1
    assert accounting.zero_plan_skip_count == 0
    assert accounting.multiple_plan_decision_count == 0
    assert provider.call_count == 0
    assert accounting.cache_lookup_count == 0
    assert accounting.provider_request_considered_count == 0
    assert accounting.guidance_activation_count == 0
    assert controller.active is None
    assert record.fallback_reason == SKIP_SINGLETON_NO_CHOICE
    assert record.shaping_contribution == 0.0
    assert accounting.check_invariants() == []
    cache.close()


def test_19_multiple_plans_take_the_normal_cache_and_provider_path(tmp_path):
    provider = CountingProvider()
    accounting, cache, gateway = _gateway_stack(tmp_path, provider, "multi")
    controller = _controller(gateway, accounting)

    row = event_row()
    request = build_request(row, agent_positions=POSITIONS, grid_shape=GRID)
    assert len(request.candidate_joint_plans) > 1

    record = _apply(controller, row)

    assert accounting.multiple_plan_decision_count == 1
    assert accounting.zero_plan_skip_count == accounting.singleton_skip_count == 0
    assert provider.call_count == 1
    assert accounting.cache_lookup_count == 1
    assert accounting.cache_miss_count == 1
    assert accounting.provider_request_considered_count == 1
    assert accounting.guidance_activation_count == 1
    assert record.provider_valid == 1
    assert controller.active is not None
    assert cache.row_count() == 1
    assert accounting.check_invariants() == []

    # the identical situation in a later episode is served from the cache
    controller.reset_episode()
    _apply(controller, event_row(episode_id=99, step=12))
    assert provider.call_count == 1, "a cache hit must not reach the provider"
    assert accounting.cache_hit_count == 1
    assert accounting.multiple_plan_decision_count == 2
    assert accounting.check_invariants() == []
    cache.close()


def test_19b_skips_never_touch_the_live_cache_or_activation_counters(tmp_path):
    provider = CountingProvider()
    accounting, cache, gateway = _gateway_stack(tmp_path, provider, "mixed")
    controller = _controller(gateway, accounting)

    for row in (event_row(cells=((4, 3),)),                       # zero plans
                event_row(recipients=(0,), cells=((4, 3),),
                          loader_levels=(3,), food_level=3),      # singleton
                event_row()):                                     # multiple
        controller.reset_episode()
        _apply(controller, row)

    assert accounting.zero_plan_skip_count == 1
    assert accounting.singleton_skip_count == 1
    assert accounting.multiple_plan_decision_count == 1
    assert accounting.provider_request_considered_count == 1
    assert accounting.cache_lookup_count == 1
    assert accounting.detector_event_count == 3
    assert accounting.check_invariants() == []
    cache.close()


def test_19c_the_activation_invariant_ties_decisions_to_considered_requests():
    accounting = SelmaAccounting(cache_mode="read_write")
    accounting.detector_event_count = 5
    accounting.multiple_plan_decision_count = 3
    accounting.provider_request_considered_count = 2      # deliberately wrong
    accounting.cache_lookup_count = 2
    accounting.cache_miss_count = 2
    accounting.successful_response_count = 2
    accounting.guidance_activation_count = 2
    problems = accounting.check_invariants()
    assert any(p.startswith("I12") for p in problems), problems


# ==========================================================================
# 20-22. properties the correction must not break
# ==========================================================================


def test_20_greedy_evaluation_still_makes_zero_provider_calls():
    from runners.episode_runner import EpisodeRunner

    source = inspect.getsource(EpisodeRunner.run)
    assert "selma_guidance = self.selma_guidance if not test_mode else None" in source
    assert "selma_bank = self.selma_bank if not test_mode else None" in source


def test_21_beta_zero_leaves_the_training_reward_exactly_unchanged(tmp_path):
    import numpy as np

    provider = CountingProvider()
    accounting, cache, gateway = _gateway_stack(tmp_path, provider, "beta0")
    controller = _controller(gateway, accounting, beta=0.0)

    for raw in (0.0, 1.0, -0.5, np.float32(0.25)):
        controller.reset_episode()
        record = _apply(controller, event_row(), raw=raw)
        assert record.training_reward is raw, "beta=0 must assign, never add"
        assert record.shaping_contribution == 0.0
        assert record.beta == 0.0
    cache.close()


def test_21b_a_valid_activation_still_produces_a_bounded_potential(tmp_path):
    provider = CountingProvider()
    accounting, cache, gateway = _gateway_stack(tmp_path, provider, "phi")
    controller = _controller(gateway, accounting)
    record = _apply(controller, event_row())
    assert controller.active is not None
    assert -1.0 <= record.phi_after_effective <= 0.0
    cache.close()


def test_22_the_new_versions_invalidate_every_prior_cache_key():
    request = make_request()
    new = key_digest(canonical_json(build_cache_key(request, **KEY_CONTEXT)))
    old_context = dict(KEY_CONTEXT,
                       prompt_version="selma_guidance_prompt_v2",
                       response_schema_version="selma_guidance_response_schema_v1")
    old = key_digest(canonical_json(build_cache_key(request, **old_context)))
    assert new != old

    assert PROMPT_VERSION == "selma_guidance_prompt_v3"
    assert RESPONSE_SCHEMA_VERSION == "selma_guidance_response_schema_v2"
    assert REQUEST_VERSION == "selma_guidance_request_v2"
    assert SCHEMA_VERSION == "guidance_schema_v2"
    assert CACHE_KEY_VERSION == "selma_cache_key_v2"
    assert CACHE_SCHEMA_VERSION == "selma_cache_schema_v2"


def test_22b_a_pilot_era_cache_database_cannot_be_reopened(tmp_path):
    """The v1 store is refused outright, so no pilot row can ever be served."""
    import sqlite3

    path = str(tmp_path / "pilot_v1.sqlite3")
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE cache_meta (name TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT INTO cache_meta VALUES('cache_schema_version', "
                 "'selma_cache_schema_v1')")
    conn.commit()
    conn.close()

    from selma.cache import CacheError

    with pytest.raises(CacheError) as error:
        SqliteGuidanceCache(path, run_identity="new")
    assert "selma_cache_schema_v1" in str(error.value)


def test_22c_the_request_version_is_what_the_key_records():
    request = make_request()
    assert request.schema_version == REQUEST_VERSION
    assert build_cache_key(request, **KEY_CONTEXT)["request_schema_version"] == \
        REQUEST_VERSION


# ==========================================================================
# adapter helpers (local fakes only -- never a real client)
# ==========================================================================


class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _FakeResponses:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0) if self.script else None
        if isinstance(item, Exception):
            raise item
        return item


class _FakeClient:
    def __init__(self, *script):
        self.responses = _FakeResponses(script)


def _api_response(text, *, status="completed", model=MODEL_SNAPSHOT):
    return _Obj(id="resp_test", model=model, status=status,
                usage=_Obj(input_tokens=100, output_tokens=10,
                           input_tokens_details=_Obj(cached_tokens=0),
                           output_tokens_details=_Obj(reasoning_tokens=0)),
                output_text=text,
                output=[_Obj(content=[_Obj(type="output_text", text=text)])])


def _provider(*script, **kwargs):
    kwargs.setdefault("backoff_s", 0.0)
    kwargs.setdefault("sleep", lambda _s: None)
    return OpenAIGuidanceProvider(client=_FakeClient(*script), **kwargs)


def test_adapter_selects_a_plan_and_derives_assignments_by_code():
    request = make_request()
    plan = request.candidate_joint_plans[2]
    body = json.dumps({"selected_plan_id": plan.plan_id, "confidence": 0.7})
    provider = _provider(_api_response(body))
    response = provider.query(request)

    assert response is not None
    assert response.selected_plan_id == plan.plan_id
    assert {a.agent_id: tuple(a.loading_cell) for a in response.assignments} == \
        plan.mapping()
    assert validate(request, response).valid
    # identity, target and coalition remain code-fixed
    assert response.event_id == request.event_id
    assert response.target_food_id == tuple(request.food_id)
    assert response.recipients == tuple(sorted(request.recipient_agents))


def test_adapter_offers_only_plan_ids_in_the_schema_it_sends():
    request = make_request()
    plan = request.candidate_joint_plans[0]
    body = json.dumps({"selected_plan_id": plan.plan_id, "confidence": 1.0})
    provider = _provider(_api_response(body))
    provider.query(request)
    (kwargs,) = provider.client.responses.calls
    enum = kwargs["text"]["format"]["schema"]["properties"]["selected_plan_id"]["enum"]
    assert enum == [p.plan_id for p in request.candidate_joint_plans]
    assert kwargs["store"] is False and kwargs["stream"] is False
    assert kwargs["tools"] == [] and kwargs["reasoning"] == {"effort": "none"}


def test_adapter_rejects_an_invented_plan_id():
    request = make_request()
    body = json.dumps({"selected_plan_id": "plan_424242", "confidence": 1.0})
    provider = _provider(_api_response(body), max_retries=0)
    assert provider.query(request) is None
    assert provider.last_error == FALLBACK_MALFORMED


def test_adapter_never_queries_when_no_plan_exists():
    """Defence in depth: the adapter refuses even if a caller skips the policy."""
    request = make_request(cells=((4, 3),))
    assert request.candidate_joint_plans == ()
    provider = _provider(_api_response("{}"), max_retries=0)
    assert provider.query(request) is None
    assert provider.client.responses.calls == []
    assert provider.last_error == FALLBACK_MALFORMED


def test_the_deterministic_mock_selects_the_canonically_first_plan():
    request = make_request()
    response = DeterministicMockProvider().query(request)
    assert response.selected_plan_id == "plan_0"
    assert validate(request, response).valid
    cells = [tuple(a.loading_cell) for a in response.assignments]
    assert len(set(cells)) == len(cells)
    assert response == DeterministicMockProvider().query(request)


def test_the_deterministic_mock_returns_nothing_when_no_plan_exists():
    request = make_request(cells=((4, 3),))
    assert DeterministicMockProvider().query(request) is None


def test_plan_violations_reports_every_broken_constraint():
    problems = plan_violations(
        (PlanAssignment(0, (4, 3)), PlanAssignment(0, (9, 9))),
        recipient_agents=(0, 1), candidate_cells=CELLS,
        food_position=FOOD, grid_shape=GRID)
    assert PLAN_DUPLICATE_AGENT in problems
    assert PLAN_RECIPIENTS_NOT_EXACT in problems
    assert PLAN_CELL_OUTSIDE_GRID in problems
    assert PLAN_CELL_NOT_OFFERED in problems
    # reported in fixed precedence order, most specific first
    from selma.joint_plans import VIOLATION_PRECEDENCE

    assert problems == [r for r in VIOLATION_PRECEDENCE if r in problems]
    assert problems[0] == PLAN_DUPLICATE_AGENT


def test_a_joint_plan_namedtuple_exposes_its_mapping():
    plan = JointPlan("plan_0", (PlanAssignment(1, (3, 4)), PlanAssignment(0, (4, 3))))
    assert plan.mapping() == {1: (3, 4), 0: (4, 3)}
    assert describe_plan(plan) == "a1->(3,4), a0->(4,3)"


# ==========================================================================
# guidance log: the joint plan is recorded, and a stale header is refused
# ==========================================================================


def test_the_guidance_log_records_the_selected_plan_id(tmp_path):
    from selma.guidance import GuidanceCsvWriter

    provider = CountingProvider()
    accounting = SelmaAccounting(cache_mode="off")
    controller = _controller(provider, accounting)
    record = _apply(controller, event_row())
    assert record.selected_plan_id == "plan_0"

    path = str(tmp_path / "guidance.csv")
    writer = GuidanceCsvWriter(path)
    writer.write_row(record, controller, t_env=1, condition="c", seed=1,
                     episode_id=1)
    with open(path) as handle:
        header, row = handle.read().splitlines()[:2]
    assert "selected_plan_id" in header.split(",")
    assert "plan_0" in row.split(",")


def test_appending_under_a_stale_header_is_refused(tmp_path):
    """A shifted column is silent data corruption; it must fail loudly."""
    from selma.guidance import GuidanceCsvWriter

    provider = CountingProvider()
    accounting = SelmaAccounting(cache_mode="off")
    controller = _controller(provider, accounting)
    record = _apply(controller, event_row())

    path = str(tmp_path / "old.csv")
    stale = [c for c in GuidanceCsvWriter.COLUMNS if c != "selected_plan_id"]
    with open(path, "w") as handle:
        handle.write(",".join(stale) + "\n")

    writer = GuidanceCsvWriter(path)
    with pytest.raises(ValueError) as error:
        writer.write_row(record, controller, t_env=1, condition="c", seed=1,
                         episode_id=1)
    assert "selected_plan_id" in str(error.value)
