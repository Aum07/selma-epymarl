"""SELMA Stage C tests: deterministic mock provider, validation, potential, PBRS.

Covers every requirement in section D of the Stage C task and the corresponding
contract clauses. No live provider, no network, no credentials, no cache.

Run:  PYTHONPATH=src pytest tests/test_selma_stage_c_pbrs.py -v
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from selma.guidance import (  # noqa: E402
    ActiveGuidance,
    GuidanceController,
    d_max_plus_load,
    pbrs_term_unscaled,
    phi,
    phi_components,
    shaping_term,
)
from selma.modes import FROZEN_ACTIVE_VARIANT  # noqa: E402
from selma.provider import (  # noqa: E402
    ALLOWED_ROLES,
    SCHEMA_VERSION,
    USAGE_NOT_APPLICABLE,
    Assignment,
    DeterministicMockProvider,
    GuidanceResponse,
    build_request,
    make_provider,
    validate,
)

GRID = (8, 8)
FOOD = (4, 4)
#: Post-transition agent positions, supplied explicitly by the caller. The frozen
#: detector row carries none, so there is no fallback and no default.
POSITIONS = {0: (4, 3), 1: (6, 6)}


def event_row(event_id="ep1-wf15_m3_t5_ibon-food4,4-t3", recipients=(0, 1),
              cells=((4, 3), (3, 4)), levels=(1, 2), food_level=3):
    return {
        "row_type": "coalesced_event",
        "event_id": event_id,
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


def request_for(row=None, positions=None):
    return build_request(row or event_row(), agent_positions=positions or POSITIONS)


def controller(beta=0.0, gamma=0.99, provider=None):
    return GuidanceController(
        provider=provider or DeterministicMockProvider(),
        grid_shape=GRID, beta=beta, gamma=gamma, enabled=True,
        variant_id=FROZEN_ACTIVE_VARIANT,
    )


def offer(ctrl, rows, step=3, positions=None, episode_ended=False):
    """Offer events through the shared transition path used by the runner."""
    return ctrl.apply_transition(
        step=step, raw_team_reward=0.0, phi_before=0.0,
        post_agent_positions=positions or POSITIONS, post_food_positions=(FOOD,),
        event_rows=rows, episode_ended=episode_ended, stored_terminated=False,
        terminal_cause="",
    )


def guidance_obj(recipients, food_position=FOOD):
    """An ActiveGuidance built the way the controller builds one."""
    return ActiveGuidance(
        event_id="e", subgoal_id="sg1-e", food_id=(1, food_position[0], food_position[1]),
        food_position=food_position, recipients=recipients, assignments={},
        activated_step=0, provider_id="mock",
    )


# -- 1. deterministic mock provider -------------------------------------------


def test_mock_provider_is_deterministic():
    p = DeterministicMockProvider()
    req = request_for()
    a, b, c = p.query(req), p.query(req), p.query(req)
    assert a == b == c


def test_two_provider_instances_agree():
    req = request_for()
    assert DeterministicMockProvider().query(req) == DeterministicMockProvider().query(req)


def test_mock_provider_output_validates():
    req = request_for()
    assert validate(req, DeterministicMockProvider().query(req)).valid


def test_mock_provider_selects_only_offered_options():
    row = event_row()
    req = request_for(row)
    resp = DeterministicMockProvider().query(req)
    offered = set(tuple(c) for c in row["candidate_loading_cells"])
    for a in resp.assignments:
        assert tuple(a.loading_cell) in offered
        assert a.role in ALLOWED_ROLES
    assert set(resp.recipients) <= set(row["recipient_ids"])
    assert tuple(resp.target_food_id) == tuple(row["food_id"])


def test_unsupported_provider_fails_loudly():
    with pytest.raises(ValueError) as e:
        make_provider("gpt-please")
    assert "not supported" in str(e.value)


def test_no_network_or_credentials_in_provider_module():
    import selma.provider as mod
    src = open(mod.__file__).read()
    for banned in ("requests", "urllib", "http", "socket", "api_key", "API_KEY", "openai", "anthropic"):
        assert banned not in src, f"provider module references {banned!r}"


# -- 2. malformed provider output rejected ------------------------------------

def _good(req):
    return DeterministicMockProvider().query(req)


@pytest.mark.parametrize("mutate,reason", [
    (lambda r: None, "no_response"),
    (lambda r: "not a response", "malformed_type"),
    (lambda r: r._replace(schema_version="v0"), "schema_version_mismatch"),
    (lambda r: r._replace(event_id="other"), "event_id_mismatch"),
    (lambda r: r._replace(recipients=()), "empty_recipients"),
    (lambda r: r._replace(recipients=(0, 1, 7)), "recipients_not_subset"),
    (lambda r: r._replace(target_food_id=(1, 9, 9)), "target_food_mismatch"),
    (lambda r: r._replace(assignments=tuple(list(r.assignments) + [r.assignments[0]])),
     "duplicate_agent_assignment"),
    (lambda r: r._replace(assignments=(Assignment(0, "hold_and_load", (7, 7)),)),
     "assignment_recipient_mismatch"),
    (lambda r: r._replace(confidence=1.5), "confidence_out_of_range"),
])
def test_malformed_response_is_rejected(mutate, reason):
    req = request_for()
    v = validate(req, mutate(_good(req)))
    assert not v.valid
    assert v.reason == reason
    assert v.response is None


def test_cell_not_offered_is_rejected():
    req = request_for()
    good = _good(req)
    bad = good._replace(assignments=tuple(
        Assignment(a.agent_id, a.role, (0, 0)) for a in good.assignments))
    v = validate(req, bad)
    assert not v.valid and v.reason == "loading_cell_not_offered"


def test_role_not_allowed_is_rejected():
    req = request_for()
    good = _good(req)
    bad = good._replace(assignments=tuple(
        Assignment(a.agent_id, "teleport", a.loading_cell) for a in good.assignments))
    v = validate(req, bad)
    assert not v.valid and v.reason == "role_not_allowed"


def test_infeasible_coalition_is_rejected():
    req = request_for(event_row(levels=(1, 1), food_level=5))
    v = validate(req, _good(req))
    assert not v.valid and v.reason == "coalition_infeasible"


def test_rejected_response_yields_zero_shaping_and_no_guidance():
    class BadProvider(DeterministicMockProvider):
        def query(self, request):
            self.call_count += 1
            return GuidanceResponse(SCHEMA_VERSION, "wrong-id", (0,), request.food_id,
                                    (0,), (Assignment(0, "hold_and_load", (4, 3)),), 1.0)
    c = controller(beta=1.0, provider=BadProvider())
    record = offer(c, [event_row()])
    assert c.active is None
    assert c.rejected_count == 1
    assert record.fallback_reason == "event_id_mismatch"
    assert record.provider_valid == 0
    # No guidance means Phi = 0 on both sides, so the term is exactly zero.
    assert record.pbrs_term_unscaled == 0.0
    assert record.shaping_contribution == 0.0
    assert c.total_shaping_reward == 0.0


# -- 3. potential calculation --------------------------------------------------


def test_d_max_plus_load_values():
    assert d_max_plus_load((8, 8)) == 15
    assert d_max_plus_load((10, 10)) == 19


def test_potential_is_zero_without_guidance():
    assert phi(None, {0: (0, 0)}, (FOOD,), GRID) == 0.0
    assert controller().potential({0: (0, 0)}, (FOOD,)) == 0.0


def test_potential_manhattan_plus_load_allowance():
    g = guidance_obj((0,))
    # agent at (4,3): manhattan 1, +1 LOAD allowance -> 2/15
    assert phi(g, {0: (4, 3)}, (FOOD,), GRID) == pytest.approx(-2.0 / 15.0)


def test_potential_drops_load_allowance_when_food_gone():
    g = guidance_obj((0,))
    assert phi(g, {0: (4, 3)}, (), GRID) == pytest.approx(-1.0 / 15.0)


def test_potential_is_mean_over_recipients_not_sum():
    g = guidance_obj((0, 1))
    comps = phi_components(g, {0: (4, 3), 1: (4, 5)}, (FOOD,), GRID)
    assert len(comps) == 2
    assert phi(g, {0: (4, 3), 1: (4, 5)}, (FOOD,), GRID) == pytest.approx(sum(comps.values()) / 2)


def test_potential_is_bounded_in_minus_one_zero():
    g = guidance_obj((0,))
    for pos in [(0, 0), (7, 7), (0, 7), (7, 0), (4, 4)]:
        assert -1.0 <= phi(g, {0: pos}, (FOOD,), GRID) <= 0.0


def test_only_selected_recipients_contribute():
    g = guidance_obj((0,))
    only0 = phi(g, {0: (4, 3), 1: (0, 0)}, (FOOD,), GRID)
    without1 = phi(g, {0: (4, 3)}, (FOOD,), GRID)
    assert only0 == without1  # agent 1 is not a recipient -> no contribution


# -- 4. exact PBRS formula -----------------------------------------------------


@pytest.mark.parametrize("beta,gamma,pb,pa", [
    (1.0, 0.99, -0.5, -0.2), (0.5, 0.9, -1.0, 0.0), (2.0, 1.0, 0.0, -1.0),
    (1.0, 0.99, -1e6, 1e6),  # no clipping
])
def test_shaping_term_matches_formula(beta, gamma, pb, pa):
    assert shaping_term(beta, gamma, pb, pa) == pytest.approx(beta * (gamma * pa - pb))


def test_no_clipping_at_extreme_values():
    assert shaping_term(1.0, 0.99, -1e9, 1e9) > 1e8


def test_discounted_telescoping_identity():
    """sum gamma^t * F_t / beta == -Phi_0 + gamma^T * Phi_T for any Phi sequence."""
    import random
    rng = random.Random(0)
    gamma, beta, T = 0.99, 0.7, 40
    phis = [rng.uniform(-1, 0) for _ in range(T + 1)]
    total = sum(gamma ** t * shaping_term(beta, gamma, phis[t], phis[t + 1]) for t in range(T))
    assert total / beta == pytest.approx(-phis[0] + gamma ** T * phis[T])


def test_zero_cumulative_shaping_when_endpoints_zero():
    """Phi_0 = 0 and terminal Phi_T = 0 -> cumulative discounted shaping is 0."""
    import random
    rng = random.Random(1)
    gamma, beta, T = 0.99, 1.3, 25
    phis = [0.0] + [rng.uniform(-1, 0) for _ in range(T - 1)] + [0.0]
    total = sum(gamma ** t * shaping_term(beta, gamma, phis[t], phis[t + 1]) for t in range(T))
    assert total == pytest.approx(0.0, abs=1e-12)


def test_randomly_changing_potential_does_not_break_identity():
    """Devlin & Kudenko (2012) admits randomly changing potentials."""
    import random
    rng = random.Random(7)
    gamma, beta, T = 0.95, 1.0, 60
    phis = [rng.choice([-1.0, -0.5, 0.0, rng.uniform(-1, 0)]) for _ in range(T + 1)]
    total = sum(gamma ** t * shaping_term(beta, gamma, phis[t], phis[t + 1]) for t in range(T))
    assert total / beta == pytest.approx(-phis[0] + gamma ** T * phis[T])


# -- 5. beta = 0 --------------------------------------------------------------


def test_beta_zero_training_reward_is_raw_object_identity():
    c = controller(beta=0.0)
    raw = 1.2345
    tr, unscaled, contribution = c.shape(raw, -0.4, -0.1)
    assert tr is raw          # assigned directly, not raw + 0.0
    assert contribution == 0.0
    assert unscaled == pytest.approx(0.99 * -0.1 - -0.4)  # measured, beta-free
    assert c.total_shaping_reward == 0.0


def test_beta_zero_never_accumulates_shaping():
    c = controller(beta=0.0)
    for pb, pa in [(-0.9, -0.1), (0.0, -1.0), (-0.5, -0.5)]:
        tr, _unscaled, contribution = c.shape(3.0, pb, pa)
        assert contribution == 0.0 and tr == 3.0
    assert c.total_shaping_reward == 0.0


def test_shape_applies_beta_exactly_once():
    c = controller(beta=0.25)
    _tr, unscaled, contribution = c.shape(0.0, -0.4, -0.1)
    assert contribution == pytest.approx(0.25 * unscaled)


# -- 6. guidance lifecycle ------------------------------------------------------
#
# The lifecycle (activation, persistence, replacement, TTL, removal, terminal,
# reset) is driven through the runner's shared transition path and is covered in
# tests/test_selma_stage_c_ordering.py. What remains here is the provider-facing
# behaviour of that path.


def test_activation_sets_guidance():
    c = controller(beta=1.0)
    record = offer(c, [event_row()])
    assert c.active is not None
    assert c.activation_count == 1
    assert record.guidance_transition == "activation"
    assert record.subgoal_id.startswith("sg1-")


def test_replacement_uses_the_new_guidance():
    c = controller(beta=1.0)
    offer(c, [event_row(event_id="e1")], step=1)
    first = c.active.event_id
    record = offer(c, [event_row(event_id="e2")], step=4)
    assert c.active.event_id == "e2" != first
    assert c.replacement_count == 1 and record.guidance_transition == "replacement"


def test_episode_reset_clears_guidance_but_keeps_counters():
    c = controller(beta=1.0)
    offer(c, [event_row()], step=0)
    c.reset_episode()
    assert c.active is None
    assert c.activation_count == 1  # cumulative, deliberately not cleared
    assert c.potential({0: (0, 0)}, (FOOD,)) == 0.0


# -- 7. terminal / truncation semantics ---------------------------------------


def test_terminal_rule_matches_stage0_verdict():
    """Stage 0: stored_terminated == 1 on every LBF episode end -> phi_after = 0."""
    c = controller(beta=1.0, gamma=0.99)
    offer(c, [event_row()], step=0)
    phi_before = c.potential({0: (4, 3), 1: (4, 5)}, (FOOD,))
    phi_after_real = c.potential({0: (4, 4), 1: (4, 5)}, (FOOD,))
    for stored_terminated, expected_after in [(True, 0.0), (False, phi_after_real)]:
        eff = 0.0 if stored_terminated else phi_after_real
        assert eff == expected_after
        _, _unscaled, contribution = c.shape(0.0, phi_before, eff)
        assert contribution == pytest.approx(1.0 * (0.99 * expected_after - phi_before))


def test_true_terminal_zeroes_successor_potential():
    c = controller(beta=1.0, gamma=0.99)
    offer(c, [event_row()], step=0)
    pb = c.potential({0: (4, 3)}, (FOOD,))
    _, _unscaled, contribution = c.shape(0.0, pb, 0.0)   # stored terminal -> 0
    assert contribution == pytest.approx(-pb)            # beta * (gamma*0 - pb)


# -- 8. counters ---------------------------------------------------------------


def test_provider_and_query_counters_reconcile():
    p = DeterministicMockProvider()
    c = controller(beta=1.0, provider=p)
    for i in range(4):
        offer(c, [event_row(event_id=f"e{i}")], step=i)
    assert c.query_count == 4
    assert c.provider_call_count == 4
    assert p.call_count == 4
    assert c.activation_count == 4
    assert c.replacement_count == 3


def test_mock_reports_usage_as_not_applicable_never_as_zero():
    p = DeterministicMockProvider()
    p.query(request_for())
    assert p.last_usage is USAGE_NOT_APPLICABLE
    assert p.last_usage.tokens_prompt is None
    assert p.last_usage.cost_usd is None


def test_disabled_controller_never_queries():
    p = DeterministicMockProvider()
    c = GuidanceController(provider=p, grid_shape=GRID, beta=1.0, gamma=0.99,
                           enabled=False, variant_id=FROZEN_ACTIVE_VARIANT)
    offer(c, [event_row()], step=0)
    assert c.query_count == 0 and p.call_count == 0 and c.active is None


# -- 9. frozen Stage B detector unchanged -------------------------------------


def test_frozen_detector_grid_and_config_unchanged():
    from selma.detector import build_variant_grid
    grid = build_variant_grid()
    assert len(grid) == 48
    (v,) = [x for x in grid if x.variant_id == "wf15_m3_t5_ibon"]
    assert (v.w_fail, v.m, v.w_team, v.use_inbound) == (15, 3, 5, True)


def test_stage_c_imports_do_not_alter_detector_defaults():
    import selma.detector as det
    assert det.W_FAIL_GRID == (5, 10, 15)
    assert det.M_GRID == (2, 3)
    assert det.TEAM_GATE_GRID == (None, 5, 10, 15)
    assert det.INBOUND_GRID == (False, True)


def test_default_config_is_shaping_off():
    import yaml
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..",
                                           "src", "config", "default.yaml")))
    assert cfg["selma_beta"] == 0.0
    assert cfg["selma_provider"] == "none"
    assert cfg["selma_mode"] == "off"
    assert cfg["selma_guidance_log_csv"] == ""
