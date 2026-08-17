#!/usr/bin/env python
"""SELMA Stage D: completely offline provider/cache smoke.

Exercises the real gateway, the real SQLite cache, the real OpenAI adapter code
path and the real :class:`GuidanceController` -- with an **injected fake client**.
No network socket is opened, no API key is read, no environment is stepped and
no training is run.

Scenario (each step asserted, not merely printed):

1. first request                      -> cache miss, exactly one simulated live call
2. identical situation, new episode   -> cache hit, zero further calls
3. changed absolute agent position    -> cache miss, one more call
4. read-only mode, unseen situation   -> miss, zero calls, no guidance
5. exhausted live-call budget         -> miss, zero calls, no guidance, F_t = 0
6. every counter reconciles against the Stage D invariants

Usage::

    /home/aum/selma/venv/bin/python analysis/selma_stage_d_offline_smoke.py

Exit code 0 only if every step and every invariant holds.
"""

import json
import os
import sys
import tempfile
from types import SimpleNamespace

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from selma.accounting import (  # noqa: E402
    FALLBACK_BUDGET_EXHAUSTED,
    FALLBACK_READ_ONLY_MISS,
    SelmaAccounting,
)
from selma.cache import SqliteGuidanceCache, candidate_plan_ids  # noqa: E402
from selma.gateway import (  # noqa: E402
    CACHE_READ_ONLY,
    CACHE_READ_WRITE,
    ProviderGateway,
)
from selma.guidance import GuidanceController  # noqa: E402
from selma.modes import FROZEN_ACTIVE_VARIANT  # noqa: E402
from selma.openai_provider import (  # noqa: E402
    MODEL_SNAPSHOT,
    OpenAIGuidanceProvider,
)
from selma.provider import build_request  # noqa: E402

CONDITION = "lbforaging:Foraging-8x8-2p-2f-coop-v3"
GRID = (8, 8)
FOOD = (4, 4)
BETA = 0.25
GAMMA = 0.99


# ---------------------------------------------------------------------------
# the fake provider client -- the only thing standing in for a network
# ---------------------------------------------------------------------------


class FakeResponses:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        # Answer well: select the first offered joint plan.
        schema = kwargs["text"]["format"]["schema"]
        offered = schema["properties"]["selected_plan_id"]["enum"]
        body = json.dumps({"selected_plan_id": offered[0], "confidence": 0.9})
        return SimpleNamespace(
            id="resp_fake_{}".format(len(self.calls)),
            model=MODEL_SNAPSHOT,
            status="completed",
            output_text=body,
            output=[SimpleNamespace(content=[{"type": "output_text", "text": body}])],
            usage=SimpleNamespace(
                input_tokens=118, output_tokens=17,
                input_tokens_details=SimpleNamespace(cached_tokens=64),
                output_tokens_details=SimpleNamespace(reasoning_tokens=0),
            ),
        )


class FakeClient:
    def __init__(self):
        self.responses = FakeResponses()


# ---------------------------------------------------------------------------
# scenario helpers
# ---------------------------------------------------------------------------


def event_row(episode_id, food_pos=FOOD, cells=((4, 3), (3, 4))):
    return {
        "row_type": "coalesced_event",
        "event_id": "ep{}-{}-food{},{}-t3".format(
            episode_id, FROZEN_ACTIVE_VARIANT, food_pos[0], food_pos[1]),
        "variant_id": FROZEN_ACTIVE_VARIANT,
        "food_id": (episode_id, food_pos[0], food_pos[1]),
        "food_position": food_pos,
        "food_level": 3,
        "trigger_agent_ids": (0,),
        "recipient_ids": (0, 1),
        "loader_ids": (0,),
        "loader_levels": (1,),
        "helper_ids": (1,),
        "helper_levels": (2,),
        "level_deficit": 2,
        "required_extra_level": 2,
        "candidate_loading_cells": cells,
        "coalition_feasible": 1,
        "suppression_reason": "",
    }


def make_provider(client, max_retries=1):
    return OpenAIGuidanceProvider(client=client, max_retries=max_retries,
                                  backoff_s=0.0, sleep=lambda _s: None)


def make_stack(provider, cache, cache_mode, cap, live_provider=True):
    accounting = SelmaAccounting(
        cache_mode=cache_mode, live_call_cap=cap, provider=provider.provider_id,
        requested_model=provider.model, prompt_version=provider.prompt_version,
        response_schema_version=provider.response_schema_version,
        price_table_date="", live_provider=live_provider)
    gateway = ProviderGateway(
        provider, accounting=accounting, cache=cache, cache_mode=cache_mode,
        live_call_cap=cap, condition=CONDITION, grid_shape=GRID)
    controller = GuidanceController(
        provider=gateway, grid_shape=GRID, beta=BETA, gamma=GAMMA, enabled=True,
        variant_id=FROZEN_ACTIVE_VARIANT, accounting=accounting)
    return accounting, gateway, controller


def fire(controller, episode_id, positions):
    controller.reset_episode()
    return controller.apply_transition(
        step=0, raw_team_reward=0.0, phi_before=0.0,
        post_agent_positions=positions, post_food_positions=(FOOD,),
        event_rows=[event_row(episode_id)], episode_ended=False,
        stored_terminated=False, terminal_cause="")


class Report:
    def __init__(self):
        self.rows = []

    def check(self, name, passed, detail=""):
        self.rows.append((name, bool(passed)))
        print("    [{}] {}{}".format("PASS" if passed else "FAIL", name,
                                     "   " + detail if detail else ""))
        return bool(passed)

    @property
    def failed(self):
        return [name for name, passed in self.rows if not passed]


def main():
    report = Report()
    print("SELMA Stage D -- offline provider/cache smoke")
    print("  repo     : {}".format(REPO))
    print("  provider : openai adapter with an INJECTED FAKE CLIENT")
    print("  model    : {}".format(MODEL_SNAPSHOT))
    print("  network  : none. no socket, no API key, no training.\n")

    workdir = tempfile.mkdtemp(prefix="selma_stage_d_smoke_")
    cache_path = os.path.join(workdir, "guidance_cache.sqlite3")
    client = FakeClient()
    provider = make_provider(client)
    cache = SqliteGuidanceCache(
        cache_path, run_identity="{}|seed4242|openai|{}".format(CONDITION, MODEL_SNAPSHOT))
    accounting, gateway, controller = make_stack(
        provider, cache, CACHE_READ_WRITE, cap=10)

    print("1. first request -> miss + exactly one simulated live call")
    record = fire(controller, episode_id=1, positions={0: (4, 3), 1: (6, 6)})
    report.check("cache miss recorded", accounting.cache_miss_count == 1)
    report.check("exactly one live call", accounting.live_api_call_count == 1)
    report.check("one fake HTTP request issued", len(client.responses.calls) == 1)
    report.check("guidance activated", controller.active is not None)
    report.check("shaping applied", record.shaping_contribution != 0.0,
                 "F_t = {:.6f}".format(record.shaping_contribution))
    report.check("row written to the cache", cache.row_count() == 1)

    print("\n2. identical situation, different episode -> cache hit, no call")
    calls_before = accounting.live_api_call_count
    record = fire(controller, episode_id=987, positions={0: (4, 3), 1: (6, 6)})
    report.check("cache hit recorded", accounting.cache_hit_count == 1)
    report.check("no additional live call",
                 accounting.live_api_call_count == calls_before)
    report.check("no additional fake HTTP request", len(client.responses.calls) == 1)
    report.check("guidance activated from the cached response",
                 controller.active is not None)
    report.check("episode id did not change the key", cache.row_count() == 1)

    print("\n3. changed absolute agent position -> miss, one further call")
    record = fire(controller, episode_id=988, positions={0: (4, 3), 1: (0, 0)})
    report.check("second cache miss", accounting.cache_miss_count == 2)
    report.check("second live call", accounting.live_api_call_count == 2)
    report.check("second row written", cache.row_count() == 2)
    cache.close()

    print("\n4. read-only mode, unseen situation -> miss, zero calls, no guidance")
    ro_client = FakeClient()
    ro_provider = make_provider(ro_client)
    ro_cache = SqliteGuidanceCache(
        cache_path, run_identity="ro", read_only=True)
    ro_accounting, _gw, ro_controller = make_stack(
        ro_provider, ro_cache, CACHE_READ_ONLY, cap=10)
    record = fire(ro_controller, episode_id=5, positions={0: (1, 1), 1: (7, 7)})
    report.check("read-only miss recorded", ro_accounting.cache_miss_count == 1)
    report.check("zero live calls in read-only mode",
                 ro_accounting.live_api_call_count == 0)
    report.check("no fake HTTP request issued", len(ro_client.responses.calls) == 0)
    report.check("no guidance activated", ro_controller.active is None)
    report.check("fallback reason is the read-only miss",
                 ro_accounting.fallback_reasons.get(FALLBACK_READ_ONLY_MISS) == 1)
    report.check("F_t is exactly zero", record.shaping_contribution == 0.0)
    # a read-only hit still works
    record = fire(ro_controller, episode_id=6, positions={0: (4, 3), 1: (6, 6)})
    report.check("read-only hit served without a call",
                 ro_accounting.cache_hit_count == 1
                 and ro_accounting.live_api_call_count == 0)
    ro_cache.close()

    print("\n5. exhausted live-call budget -> no call, no guidance, F_t = 0")
    ex_client = FakeClient()
    ex_provider = make_provider(ex_client)
    ex_cache = SqliteGuidanceCache(
        os.path.join(workdir, "budget.sqlite3"), run_identity="budget")
    ex_accounting, _gw2, ex_controller = make_stack(
        ex_provider, ex_cache, CACHE_READ_WRITE, cap=1)
    fire(ex_controller, episode_id=1, positions={0: (4, 3), 1: (6, 6)})
    record = fire(ex_controller, episode_id=2, positions={0: (2, 2), 1: (5, 5)})
    report.check("budget consumed by the first request only",
                 ex_accounting.live_api_call_count == 1)
    report.check("second request made no call", len(ex_client.responses.calls) == 1)
    report.check("budget_exhausted recorded",
                 ex_accounting.fallback_reasons.get(FALLBACK_BUDGET_EXHAUSTED) == 1)
    report.check("no guidance after exhaustion", ex_controller.active is None)
    report.check("F_t is exactly zero on the exhausted request",
                 record.shaping_contribution == 0.0)
    report.check("lookup still happened after exhaustion",
                 ex_accounting.cache_lookup_count == 2)
    report.check("nothing extra was cached", ex_cache.row_count() == 1)
    ex_cache.close()

    print("\n6. counter reconciliation")
    for label, ledger, retries in (("read_write", accounting, 1),
                                   ("read_only", ro_accounting, 1),
                                   ("budget", ex_accounting, 1)):
        problems = ledger.check_invariants(max_retries=retries)
        report.check("invariants hold ({})".format(label), not problems,
                     "; ".join(problems))

    print("\n  read_write ledger:")
    for name, value in sorted(accounting.snapshot().items()):
        print("    {:<38} {}".format(name, value))
    print("    {:<38} {}".format("input/output/cached tokens",
                                 (accounting.input_tokens, accounting.output_tokens,
                                  accounting.cached_input_tokens)))
    print("    {:<38} {}".format("returned models", accounting.returned_models))

    print("\n" + "=" * 68)
    if report.failed:
        print("RESULT: FAIL -- {} check(s) failed:".format(len(report.failed)))
        for name in report.failed:
            print("  - {}".format(name))
        return 1
    print("RESULT: PASS -- {} checks. Cache, budget and accounting behave "
          "offline exactly as specified.".format(len(report.rows)))
    print("No network call, no API key, no training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
