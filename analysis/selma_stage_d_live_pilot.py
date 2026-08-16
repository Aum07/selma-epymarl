#!/usr/bin/env python
"""SELMA Stage D: bounded live-provider compatibility pilot.

**This is not an experiment.** It validates exactly five things about the live
provider path and claims nothing else:

1. the exact dated model snapshot is reachable and the installed SDK accepts the
   Responses API fields the adapter sends;
2. strict structured output comes back and parses;
3. every selected id is one the code offered (grounding), and the frozen Stage C
   validator has the final say;
4. the cache is written only after validation, and an identical question is then
   served without a further live call;
5. the eleven accounting counters reconcile and no invariant is violated.

It performs **no training**, constructs no learner and no environment, keeps
``beta = 0`` throughout, and never touches ``src/selma/detector.py``.

Safety
------

* live calls are refused unless ``--confirm-live`` is passed;
* the default live-attempt cap is **1** (the compatibility probe alone);
* ``--max-attempts`` may lower the cap but can never raise it above
  :data:`HARD_ATTEMPT_CEILING`;
* every attempt is recorded in the ledger **before** the request leaves, so a
  retry, a timeout and a transport failure all consume the budget;
* the API key is read by the production client factory from ``OPENAI_API_KEY``
  and is never read, printed, stored or passed as an argument by this script.
  Exceptions are recorded as a class name and, when present, an HTTP status
  code -- never as a message, never with headers.

Run (offline dry run -- no network, no key needed)::

    PYTHONPATH=src python analysis/selma_stage_d_live_pilot.py

Run (live; the caller sources the secret into this process only)::

    set -a; . ~/.config/selma/openai.env; set +a
    PYTHONPATH=src python analysis/selma_stage_d_live_pilot.py \
        --confirm-live --max-attempts 30

Exit status is 0 only when every gate passed.
"""

import argparse
import json
import os
import shutil
import statistics
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from selma.accounting import SelmaAccounting  # noqa: E402
from selma.cache import (  # noqa: E402
    CACHE_KEY_VERSION,
    CACHE_SCHEMA_VERSION,
    SqliteGuidanceCache,
    candidate_assignment_ids,
)
from selma.gateway import (  # noqa: E402
    CACHE_READ_ONLY,
    CACHE_READ_WRITE,
    ProviderGateway,
)
from selma.guidance import GuidanceController  # noqa: E402
from selma.modes import FROZEN_ACTIVE_VARIANT  # noqa: E402
from selma.openai_provider import (  # noqa: E402
    API_KEY_ENV,
    MODEL_SNAPSHOT,
    PROMPT_VERSION,
    RESPONSE_SCHEMA_VERSION,
    default_client_factory,
)
from selma.provider import build_request  # noqa: E402
from selma.provider_registry import PROVIDER_OPENAI, make_provider  # noqa: E402

#: Infrastructure ceiling. ``--max-attempts`` is clamped to this and the
#: ledgering transport refuses to issue attempt number ``HARD + 1`` whatever the
#: rest of the stack believes.
HARD_ATTEMPT_CEILING = 50

#: Default: the one-call compatibility probe and nothing more.
DEFAULT_MAX_ATTEMPTS = 1

GRID_SHAPE = (8, 8)
CONDITION = "lbforaging:Foraging-8x8-3p-2f-coop-v3"
PILOT_GAMMA = 0.99
PILOT_BETA = 0.0            # never anything else, in any code path below

#: One writable cache database for the whole live pilot. The store refuses to be
#: reopened for writing under a different identity, so this is shared
#: deliberately between the probe and the pilot phases and by nothing else.
LIVE_RUN_IDENTITY = "selma_stage_d_live_pilot"

REPORT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "out", "live_pilot")


# ==========================================================================
# frozen fixtures
# ==========================================================================


def scenario(name, *, episode_id, step, food_pos, food_level, loaders, helpers,
             cells, positions):
    """One frozen coordination fixture: a detector-shaped row plus positions.

    ``loaders`` / ``helpers`` are ``{agent_id: level}``. The triggering agents
    are the loaders; the recipients are loaders and helpers together. Nothing
    here is sampled, learned or time-dependent.
    """
    loader_ids = tuple(sorted(loaders))
    helper_ids = tuple(sorted(helpers))
    recipients = tuple(sorted(set(loader_ids) | set(helper_ids)))
    deficit = int(food_level) - sum(loaders[a] for a in loader_ids)
    row = {
        "row_type": "coalesced_event",
        "event_id": "ep{}-{}-food{},{}-t{}".format(
            episode_id, FROZEN_ACTIVE_VARIANT, food_pos[0], food_pos[1], step),
        "variant_id": FROZEN_ACTIVE_VARIANT,
        "food_id": (episode_id, food_pos[0], food_pos[1]),
        "food_position": tuple(food_pos),
        "food_level": int(food_level),
        "trigger_agent_ids": loader_ids,
        "recipient_ids": recipients,
        "loader_ids": loader_ids,
        "loader_levels": tuple(loaders[a] for a in loader_ids),
        "helper_ids": helper_ids,
        "helper_levels": tuple(helpers[a] for a in helper_ids),
        "level_deficit": deficit,
        "required_extra_level": max(0, deficit),
        "candidate_loading_cells": tuple(tuple(c) for c in cells),
        "coalition_feasible": 1,
        "suppression_reason": "",
    }
    return {"name": name, "row": row, "positions": dict(positions),
            "food_positions": [tuple(food_pos)]}


def frozen_fixtures():
    """Ten distinct 8x8 coordination situations. Order is part of the record.

    Deliberately varied along every axis the pilot claims to cover: agent and
    food positions, agent and food levels, which agent triggers and which helps,
    the level deficit, the offered loading cells (four, three and two of them),
    and both a translated and a column-mirrored copy of the base layout -- which
    scheme B must treat as *distinct* questions, since it applies no symmetry
    normalisation.
    """
    return [
        scenario("base_deficit2", episode_id=1, step=4,
                 food_pos=(4, 4), food_level=3,
                 loaders={0: 1}, helpers={1: 2},
                 cells=[(3, 4), (5, 4), (4, 3), (4, 5)],
                 positions={0: (4, 3), 1: (6, 6)}),
        scenario("translated_plus1", episode_id=2, step=6,
                 food_pos=(5, 5), food_level=3,
                 loaders={0: 1}, helpers={1: 2},
                 cells=[(4, 5), (6, 5), (5, 4), (5, 6)],
                 positions={0: (5, 4), 1: (7, 7)}),
        scenario("mirrored_columns", episode_id=3, step=5,
                 food_pos=(4, 3), food_level=3,
                 loaders={0: 1}, helpers={1: 2},
                 cells=[(3, 3), (5, 3), (4, 2), (4, 4)],
                 positions={0: (4, 4), 1: (6, 1)}),
        scenario("three_recipients_deficit4", episode_id=4, step=9,
                 food_pos=(2, 2), food_level=5,
                 loaders={0: 1}, helpers={1: 2, 2: 2},
                 cells=[(1, 2), (3, 2), (2, 1), (2, 3)],
                 positions={0: (2, 1), 1: (0, 2), 2: (4, 4)}),
        scenario("deficit3_single_helper", episode_id=5, step=3,
                 food_pos=(6, 2), food_level=4,
                 loaders={0: 1}, helpers={1: 3},
                 cells=[(5, 2), (7, 2), (6, 1), (6, 3)],
                 positions={0: (6, 1), 1: (3, 2)}),
        scenario("trigger_is_agent1", episode_id=6, step=7,
                 food_pos=(1, 5), food_level=3,
                 loaders={1: 1}, helpers={0: 2},
                 cells=[(0, 5), (2, 5), (1, 4), (1, 6)],
                 positions={0: (3, 7), 1: (1, 4)}),
        scenario("far_helper", episode_id=7, step=2,
                 food_pos=(3, 3), food_level=3,
                 loaders={0: 1}, helpers={1: 2},
                 cells=[(2, 3), (4, 3), (3, 2), (3, 4)],
                 positions={0: (3, 2), 1: (7, 0)}),
        scenario("two_cells_only", episode_id=8, step=11,
                 food_pos=(5, 5), food_level=4,
                 loaders={0: 2}, helpers={1: 2},
                 cells=[(5, 4), (5, 6)],
                 positions={0: (5, 4), 1: (2, 5)}),
        scenario("high_levels_deficit3", episode_id=9, step=8,
                 food_pos=(2, 6), food_level=6,
                 loaders={0: 3}, helpers={1: 3},
                 cells=[(1, 6), (3, 6), (2, 5), (2, 7)],
                 positions={0: (2, 5), 1: (6, 6)}),
        scenario("edge_row_three_cells", episode_id=10, step=1,
                 food_pos=(7, 3), food_level=3,
                 loaders={0: 2}, helpers={1: 1},
                 cells=[(6, 3), (7, 2), (7, 4)],
                 positions={0: (7, 2), 1: (4, 3)}),
    ]


def replayed(fixture, offset=100):
    """The same situation, relabelled with different episode/run metadata.

    ``episode_id``, ``event_id``, ``food_id`` and the step index all change; the
    physical situation does not. Scheme B excludes every one of those, so this
    must hit the cache. If it ever misses, the key has picked up episode
    identity again -- the exact defect the feasibility audit found in the Stage C
    proposal.
    """
    row = dict(fixture["row"])
    episode_id = int(row["food_id"][0]) + offset
    step = int(row["event_id"].rsplit("-t", 1)[1]) + 37
    row["food_id"] = (episode_id, row["food_position"][0], row["food_position"][1])
    row["event_id"] = "ep{}-{}-food{},{}-t{}".format(
        episode_id, FROZEN_ACTIVE_VARIANT,
        row["food_position"][0], row["food_position"][1], step)
    return {"name": fixture["name"] + "@replay", "row": row,
            "positions": dict(fixture["positions"]),
            "food_positions": list(fixture["food_positions"])}


def unseen_fixture():
    """A situation deliberately never sent live: the read-only-miss probe."""
    return scenario("unseen_readonly_miss", episode_id=90, step=2,
                    food_pos=(0, 0), food_level=3,
                    loaders={0: 1}, helpers={1: 2},
                    cells=[(0, 1), (1, 0)],
                    positions={0: (0, 1), 1: (5, 5)})


# ==========================================================================
# ledgering transport (counts an attempt before every request)
# ==========================================================================


class PilotCeilingExceeded(RuntimeError):
    """The hard infrastructure ceiling was reached. Named so the adapter's
    class-name classifier sees a transport error, not a retryable timeout."""


def sanitise_exception(error):
    """Class name and, if present, an HTTP status code. **Never the message.**

    A provider exception message can echo request content and, in principle,
    header material. Only the two structured fields below are recorded, so
    nothing a server said can reach the report verbatim.
    """
    info = {"error_class": type(error).__name__}
    status = getattr(error, "status_code", None)
    if isinstance(status, int):
        info["status_code"] = status
    code = getattr(error, "code", None)
    if isinstance(code, str) and len(code) <= 64:
        info["code"] = code
    return info


def _offered_ids_from_kwargs(kwargs):
    try:
        schema = kwargs["text"]["format"]["schema"]
        return list(schema["properties"]["assignment_ids"]["items"]["enum"])
    except (KeyError, TypeError):
        return []


def _selected_ids_from_response(response):
    """Parsed ids only. The raw body is never stored."""
    text = getattr(response, "output_text", None)
    if not text:
        for item in getattr(response, "output", None) or []:
            for part in getattr(item, "content", None) or []:
                if getattr(part, "type", None) in ("output_text", "text"):
                    text = (text or "") + (getattr(part, "text", "") or "")
    if not text:
        return None
    try:
        payload = json.loads(text)
    except (ValueError, TypeError):
        return None
    ids = payload.get("assignment_ids") if isinstance(payload, dict) else None
    return ids if isinstance(ids, list) else None


class LedgeringResponses:
    """Wraps ``client.responses`` so every attempt is counted before it leaves.

    This sits at the transport boundary and changes nothing about the request:
    the adapter builds it, this object forwards it verbatim. It exists so the
    attempt count in the report is measured where the money is actually spent,
    independently of what the gateway's budget believes.
    """

    def __init__(self, inner, ledger, ceiling):
        self._inner = inner
        self._ledger = ledger
        self._ceiling = int(ceiling)

    def create(self, **kwargs):
        if len(self._ledger) >= self._ceiling:
            raise PilotCeilingExceeded(
                "refusing attempt {}: the pilot's hard ceiling is {}".format(
                    len(self._ledger) + 1, self._ceiling))
        offered = _offered_ids_from_kwargs(kwargs)
        entry = {
            "attempt_index": len(self._ledger) + 1,
            "requested_model": kwargs.get("model"),
            "store": kwargs.get("store"),
            "stream": kwargs.get("stream"),
            "tools": kwargs.get("tools"),
            "reasoning": kwargs.get("reasoning"),
            "max_output_tokens": kwargs.get("max_output_tokens"),
            "timeout": kwargs.get("timeout"),
            "strict_structured_output": bool(
                kwargs.get("text", {}).get("format", {}).get("strict")),
            "offered_id_count": len(offered),
            "outcome": "in_flight",
        }
        self._ledger.append(entry)          # counted BEFORE the request leaves
        started = time.time()
        try:
            response = self._inner.create(**kwargs)
        except Exception as error:          # noqa: BLE001 - sanitised, re-raised
            entry["outcome"] = "exception"
            entry["latency_s"] = time.time() - started
            entry.update(sanitise_exception(error))
            raise
        entry["latency_s"] = time.time() - started
        entry["outcome"] = "response"
        entry["returned_model"] = getattr(response, "model", None)
        entry["response_id"] = getattr(response, "id", None)
        entry["status"] = getattr(response, "status", None)
        usage = getattr(response, "usage", None)
        entry["input_tokens"] = getattr(usage, "input_tokens", None)
        entry["output_tokens"] = getattr(usage, "output_tokens", None)
        selected = _selected_ids_from_response(response)
        entry["parsed"] = selected is not None
        if selected is not None:
            entry["selected_ids"] = list(selected)
            offered_set = set(offered)
            entry["invented_ids"] = [i for i in selected if i not in offered_set]
            entry["grounded"] = not entry["invented_ids"]
        return response


class LedgeringClient:
    def __init__(self, inner, ledger, ceiling):
        self.responses = LedgeringResponses(inner.responses, ledger, ceiling)


# ==========================================================================
# the pilot harness
# ==========================================================================


class Phase:
    """One accounting ledger plus the gateway and controller that feed it.

    Every phase is the production path: registry provider -> gateway -> cache ->
    :class:`GuidanceController`, which runs the frozen Stage C validator on every
    response, cached or live. Nothing here reimplements validation or shaping.
    """

    def __init__(self, provider, *, cache_path, cache_mode, live_call_cap,
                 run_identity, read_only=False):
        self.accounting = SelmaAccounting(
            cache_mode=cache_mode,
            live_call_cap=live_call_cap,
            provider=getattr(provider, "provider_id", "unknown"),
            requested_model=getattr(provider, "model", ""),
            prompt_version=getattr(provider, "prompt_version", ""),
            response_schema_version=getattr(provider, "response_schema_version", ""),
            price_table_date="",
            live_provider=True,
        )
        self.cache = SqliteGuidanceCache(
            cache_path, run_identity=run_identity, read_only=read_only)
        self.gateway = ProviderGateway(
            provider,
            accounting=self.accounting,
            cache=self.cache,
            cache_mode=cache_mode,
            live_call_cap=live_call_cap,
            condition=CONDITION,
            grid_shape=GRID_SHAPE,
            allow_live_calls=True,
        )
        self.controller = GuidanceController(
            provider=self.gateway,
            grid_shape=GRID_SHAPE,
            beta=PILOT_BETA,
            gamma=PILOT_GAMMA,
            enabled=True,
            variant_id=FROZEN_ACTIVE_VARIANT,
            accounting=self.accounting,
        )
        self.records = []

    def run_fixture(self, fixture):
        """Drive one fixture through the real per-transition path."""
        before_hits = self.accounting.cache_hit_count
        before_calls = self.accounting.live_api_call_count
        before_activations = self.accounting.guidance_activation_count

        request = build_request(fixture["row"], agent_positions=fixture["positions"])
        offered = candidate_assignment_ids(request)

        self.controller.reset_episode()
        record = self.controller.apply_transition(
            step=int(fixture["row"]["event_id"].rsplit("-t", 1)[1]),
            raw_team_reward=0.0,
            phi_before=0.0,
            post_agent_positions=fixture["positions"],
            post_food_positions=fixture["food_positions"],
            event_rows=[fixture["row"]],
            episode_ended=False,
            stored_terminated=False,
        )

        active = self.controller.active
        usage = record.usage
        result = {
            "fixture": fixture["name"],
            "candidate_count": len(offered),
            "recipient_count": len(request.recipient_agents),
            "food_level": int(request.food_level),
            "level_deficit": int(request.level_deficit),
            "served_from": ("cache" if self.accounting.cache_hit_count > before_hits
                            else "live"),
            "live_calls_for_this_fixture":
                self.accounting.live_api_call_count - before_calls,
            "provider_valid": int(record.provider_valid),
            "activated": int(
                self.accounting.guidance_activation_count > before_activations),
            "fallback_reason": record.fallback_reason,
            "beta": record.beta,
            "shaping_contribution": record.shaping_contribution,
            "reward_unchanged": record.training_reward == record.raw_team_reward,
            "selected_assignments": (
                sorted("a{}@{},{}".format(a, c[1][0], c[1][1])
                       for a, c in active.assignments.items())
                if active is not None else []),
            "tokens_prompt": getattr(usage, "tokens_prompt", None),
            "tokens_completion": getattr(usage, "tokens_completion", None),
            "latency_s": getattr(usage, "latency_s", None),
        }
        result["all_selected_ids_offered"] = all(
            i in set(offered) for i in result["selected_assignments"])
        self.records.append(result)
        return result

    def close(self):
        self.cache.close()


# ==========================================================================
# offline invalid-response fixtures (never requested from the API)
# ==========================================================================


class _FakeResponses:
    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        item = self.script.pop(0) if self.script else None
        if isinstance(item, Exception):
            raise item
        return item


class _FakeClient:
    def __init__(self, *script):
        self.responses = _FakeResponses(script)


class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _fake_response(text, *, status="completed", refusal=None,
                   model=MODEL_SNAPSHOT):
    content = ([_Obj(type="refusal", refusal=refusal)] if refusal
               else [_Obj(type="output_text", text=text)])
    return _Obj(id="resp_offline_fake", model=model, status=status,
                usage=_Obj(input_tokens=100, output_tokens=10,
                           input_tokens_details=_Obj(cached_tokens=0),
                           output_tokens_details=_Obj(reasoning_tokens=0)),
                output_text=None if refusal else text,
                output=[_Obj(content=content)])


class _FakeTimeoutError(Exception):
    pass


def offline_invalid_response_gates(tmpdir):
    """Bad provider answers must yield zero guidance, zero shaping, zero writes.

    Exercised with fakes, never by asking the live API to misbehave: deliberately
    provoking a refusal or a malformed body would spend real attempts to learn
    something the offline fixture establishes exactly.
    """
    fixture = frozen_fixtures()[0]
    cases = [
        ("invented_id", _fake_response(json.dumps(
            {"assignment_ids": ["a9@1,1"], "confidence": 0.9}))),
        ("duplicate_id", _fake_response(json.dumps(
            {"assignment_ids": ["a0@4,3", "a0@4,3"], "confidence": 0.9}))),
        ("missing_recipient", _fake_response(json.dumps(
            {"assignment_ids": ["a0@4,3"], "confidence": 0.9}))),
        ("malformed_body", _fake_response("not json at all")),
        ("refusal", _fake_response(None, refusal="I will not answer")),
        ("incomplete", _fake_response(json.dumps(
            {"assignment_ids": ["a0@4,3", "a1@3,4"], "confidence": 0.9}),
            status="incomplete")),
        ("transport_failure", _FakeTimeoutError()),
    ]
    results = []
    for name, scripted in cases:
        provider = make_provider(
            PROVIDER_OPENAI, client=_FakeClient(scripted, scripted),
            backoff_s=0.0, sleep=lambda _s: None)
        phase = Phase(provider,
                      cache_path=os.path.join(tmpdir, "offline_{}.sqlite3".format(name)),
                      cache_mode=CACHE_READ_WRITE, live_call_cap=4,
                      run_identity="offline|{}".format(name))
        record = phase.run_fixture(fixture)
        rows = phase.cache.row_count()
        problems = phase.accounting.check_invariants(max_retries=1)
        phase.close()
        results.append({
            "case": name,
            "guidance_activated": record["activated"],
            "provider_valid": record["provider_valid"],
            "fallback_reason": record["fallback_reason"],
            "shaping_contribution": record["shaping_contribution"],
            "cache_rows_written": rows,
            "invariant_problems": problems,
            "passed": (record["activated"] == 0 and record["provider_valid"] == 0
                       and record["shaping_contribution"] == 0.0 and rows == 0
                       and not problems),
        })
    return results


def greedy_evaluation_gate():
    """Greedy evaluation must be *structurally* unable to reach a provider.

    The runner drops the controller for ``test_mode``, so no query can be issued
    from an evaluation episode however the provider is configured. Asserted here
    on the real source, and covered behaviourally by
    ``tests/test_selma_stage_d_runner.py::
    test_greedy_evaluation_produces_zero_lookups_calls_and_activations``.
    """
    import inspect

    from runners.episode_runner import EpisodeRunner

    source = inspect.getsource(EpisodeRunner.run)
    guards = [
        "selma_guidance = self.selma_guidance if not test_mode else None",
        "selma_bank = self.selma_bank if not test_mode else None",
    ]
    missing = [g for g in guards if g not in source]
    return {"guards_present": not missing, "missing_guards": missing}


# ==========================================================================
# reporting
# ==========================================================================


def latency_summary(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    return {
        "n": len(values),
        "min_s": round(values[0], 4),
        "median_s": round(statistics.median(values), 4),
        "p95_s": round(values[min(len(values) - 1, int(0.95 * len(values)))], 4),
        "max_s": round(values[-1], 4),
    }


def distribution(items):
    out = {}
    for item in items:
        out[str(item)] = out.get(str(item), 0) + 1
    return dict(sorted(out.items()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--confirm-live", action="store_true",
                        help="required for any live API request; without it the "
                             "pilot runs its offline gates only")
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS,
                        help="cap on live request attempts, retries included "
                             "(default {}, hard ceiling {})".format(
                                 DEFAULT_MAX_ATTEMPTS, HARD_ATTEMPT_CEILING))
    parser.add_argument("--report", default=None,
                        help="path for the sanitised JSON report")
    parser.add_argument("--price-table-date", default="",
                        help="dated price table; without one, cost is NOT "
                             "calculated and only tokens are reported")
    parser.add_argument("--keep-workdir", action="store_true",
                        help="keep the temporary cache directory for inspection")
    args = parser.parse_args(argv)

    if args.max_attempts < 1:
        parser.error("--max-attempts must be at least 1")
    attempt_cap = min(int(args.max_attempts), HARD_ATTEMPT_CEILING)

    workdir = tempfile.mkdtemp(prefix="selma_stage_d_live_pilot_")
    ledger = []
    gates = []
    report = {
        "kind": "selma_stage_d_live_pilot",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "not_a_performance_result": True,
        "training_performed": False,
        "beta_calibration_performed": False,
        "beta": PILOT_BETA,
        "provider": PROVIDER_OPENAI,
        "requested_model": MODEL_SNAPSHOT,
        "prompt_version": PROMPT_VERSION,
        "response_schema_version": RESPONSE_SCHEMA_VERSION,
        "cache_key_version": CACHE_KEY_VERSION,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "attempt_cap": attempt_cap,
        "hard_attempt_ceiling": HARD_ATTEMPT_CEILING,
        "live_confirmed": bool(args.confirm_live),
    }

    def gate(name, passed, detail=None):
        gates.append({"gate": name, "passed": bool(passed), "detail": detail})
        return bool(passed)

    try:
        # -- offline gates, always ------------------------------------------
        report["sdk_version"] = _sdk_version()
        report["offline_invalid_response_gates"] = offline_invalid_response_gates(workdir)
        gate("invalid responses activate no guidance and write no cache row",
             all(c["passed"] for c in report["offline_invalid_response_gates"]))

        report["greedy_evaluation"] = greedy_evaluation_gate()
        gate("greedy evaluation cannot reach the provider",
             report["greedy_evaluation"]["guards_present"],
             report["greedy_evaluation"]["missing_guards"])

        fixtures = frozen_fixtures()
        report["fixture_count"] = len(fixtures)
        report["candidate_count_distribution"] = distribution(
            len(candidate_assignment_ids(
                build_request(f["row"], agent_positions=f["positions"])))
            for f in fixtures)
        gate("every fixture offers multiple candidate assignments",
             all(int(k) >= 2 for k in report["candidate_count_distribution"]))

        if not args.confirm_live:
            report["live"] = {"status": "skipped",
                              "reason": "--confirm-live was not supplied"}
            report["live_attempts"] = 0
        else:
            report["live"] = _run_live(fixtures, workdir, ledger, attempt_cap, gate)
            report["live_attempts"] = len(ledger)

        report["attempt_ledger"] = ledger
        report["latency_distribution"] = latency_summary(
            e.get("latency_s") for e in ledger)
        report["price_table_date"] = args.price_table_date
        report["cost_usd"] = None
        report["cost_methodology"] = (
            "cost was NOT calculated: no cited, dated price table was supplied "
            "(--price-table-date is empty). Token counts are reported verbatim; "
            "monetary cost must be recomputed from a dated price table."
            if not args.price_table_date else
            "tokens reported verbatim; recompute cost from the price table dated "
            + args.price_table_date)

        gate("live attempts never exceeded the cap", len(ledger) <= attempt_cap,
             {"attempts": len(ledger), "cap": attempt_cap})

    finally:
        if args.keep_workdir:
            report["workdir"] = workdir
        else:
            shutil.rmtree(workdir, ignore_errors=True)

    # Last, over everything the report will actually contain -- gates included.
    gate("no credential material anywhere in the report",
         _report_is_clean(dict(report, gates=list(gates))))

    report["gates"] = gates
    report["all_gates_passed"] = all(g["passed"] for g in gates)

    path = args.report or os.path.join(
        REPORT_DIR, "live_pilot_{}.json".format(time.strftime("%Y%m%dT%H%M%SZ",
                                                              time.gmtime())))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)

    print("report: {}".format(path))
    print("live attempts: {} (cap {}, ceiling {})".format(
        report["live_attempts"], attempt_cap, HARD_ATTEMPT_CEILING))
    for entry in gates:
        print("  [{}] {}".format("PASS" if entry["passed"] else "FAIL", entry["gate"]))
    return 0 if report["all_gates_passed"] else 1


def _sdk_version():
    try:
        import openai

        return getattr(openai, "__version__", "unknown")
    except ImportError:
        return None


def _report_is_clean(report):
    """No credential-shaped material anywhere in the serialised report."""
    text = json.dumps(report).lower()
    markers = ("sk-", "openai_api_key", "authorization", "bearer ", "api_key",
               "x-api-key", "secret")
    return not any(marker in text for marker in markers)


def _run_live(fixtures, workdir, ledger, attempt_cap, gate):
    """Probe, then the capped pilot, then the cache-behaviour phases."""
    if not os.environ.get(API_KEY_ENV):
        # Presence only. The value is never read, printed, measured or hashed.
        raise SystemExit(
            "{} is not set in this process. Source the credential into the "
            "subprocess that runs this script; it is never read from a file, "
            "a config or an argument here.".format(API_KEY_ENV))

    def client_factory():
        return LedgeringClient(default_client_factory(API_KEY_ENV, 30.0),
                               ledger, HARD_ATTEMPT_CEILING)

    provider = make_provider(PROVIDER_OPENAI, client_factory=client_factory)
    cache_path = os.path.join(workdir, "live_pilot.sqlite3")
    live = {"status": "ran"}

    # -- Phase 3: exactly one live attempt ---------------------------------
    probe_phase = Phase(provider, cache_path=cache_path,
                        cache_mode=CACHE_READ_WRITE, live_call_cap=1,
                        run_identity=LIVE_RUN_IDENTITY)
    probe = probe_phase.run_fixture(fixtures[0])
    probe_entry = ledger[0] if ledger else {}
    live["probe"] = {
        "fixture": probe["fixture"],
        "attempts": len(ledger),
        "served_from": probe["served_from"],
        "provider_valid": probe["provider_valid"],
        "fallback_reason": probe["fallback_reason"],
        "returned_model": probe_entry.get("returned_model"),
        "response_status": probe_entry.get("status"),
        "parsed": probe_entry.get("parsed"),
        "grounded": probe_entry.get("grounded"),
        "input_tokens": probe_entry.get("input_tokens"),
        "output_tokens": probe_entry.get("output_tokens"),
        "request_fields": {k: probe_entry.get(k) for k in
                           ("requested_model", "store", "stream", "tools",
                            "reasoning", "max_output_tokens", "timeout",
                            "strict_structured_output")},
    }

    probe_ok = gate("probe: exactly one live attempt", len(ledger) == 1,
                    {"attempts": len(ledger)})
    probe_ok &= gate("probe: the exact model snapshot answered",
                     probe_entry.get("returned_model") == MODEL_SNAPSHOT,
                     {"returned_model": probe_entry.get("returned_model")})
    probe_ok &= gate("probe: strict structured output parsed",
                     bool(probe_entry.get("parsed")))
    probe_ok &= gate("probe: selected id was one of the offered ids",
                     bool(probe_entry.get("grounded")))
    probe_ok &= gate("probe: the Stage C validator accepted the response",
                     probe["provider_valid"] == 1,
                     {"fallback_reason": probe["fallback_reason"]})
    probe_ok &= gate("probe: token accounting is available",
                     isinstance(probe_entry.get("input_tokens"), int)
                     and isinstance(probe_entry.get("output_tokens"), int))
    probe_ok &= gate("probe: no tools and no streaming were used",
                     probe_entry.get("tools") == [] and probe_entry.get("stream") is False)
    probe_ok &= gate("probe: store=false and reasoning effort none were accepted",
                     probe_entry.get("store") is False
                     and probe_entry.get("reasoning") == {"effort": "none"})

    if not probe_ok:
        # Stop the live pilot here. No substitute model, no further spend.
        live["status"] = "stopped_after_probe"
        live["accounting"] = probe_phase.accounting.as_dict()
        probe_phase.close()
        return live

    # the identical question again: must be a hit, must cost nothing
    before = probe_phase.accounting.live_api_call_count
    repeat = probe_phase.run_fixture(fixtures[0])
    live["probe_repeat"] = {
        "served_from": repeat["served_from"],
        "extra_live_calls": probe_phase.accounting.live_api_call_count - before,
        "provider_valid": repeat["provider_valid"],
    }
    gate("probe: the identical request is a cache hit with zero extra calls",
         repeat["served_from"] == "cache"
         and probe_phase.accounting.live_api_call_count == before)
    probe_records = list(probe_phase.records)
    probe_problems = probe_phase.accounting.check_invariants(max_retries=1)
    live["probe_accounting"] = probe_phase.accounting.as_dict()
    gate("probe accounting invariants hold", not probe_problems, probe_problems)
    probe_phase.close()

    # -- Phase 4: the capped fixed-fixture pilot ---------------------------
    remaining = attempt_cap - len(ledger)
    pilot_phase = Phase(provider, cache_path=cache_path,
                        cache_mode=CACHE_READ_WRITE, live_call_cap=remaining,
                        run_identity=LIVE_RUN_IDENTITY)
    for fixture in fixtures[1:]:
        if pilot_phase.accounting.live_api_call_count >= remaining:
            live["stopped_early_at"] = fixture["name"]
            break
        pilot_phase.run_fixture(fixture)

    live_records = list(pilot_phase.records)
    live["live_calls_in_pilot_phase"] = pilot_phase.accounting.live_api_call_count

    # every fixture that actually went live, relabelled: all must hit the cache
    sent_live = {r["fixture"] for r in probe_records + live_records
                 if r["served_from"] == "live" and r["provider_valid"] == 1}
    before = pilot_phase.accounting.live_api_call_count
    replay_records = [pilot_phase.run_fixture(replayed(f))
                      for f in fixtures if f["name"] in sent_live]
    live["replay"] = {
        "count": len(replay_records),
        "served_from_cache": sum(1 for r in replay_records if r["served_from"] == "cache"),
        "extra_live_calls": pilot_phase.accounting.live_api_call_count - before,
    }
    gate("cache hits cause zero live calls",
         live["replay"]["extra_live_calls"] == 0
         and live["replay"]["served_from_cache"] == len(replay_records),
         live["replay"])

    all_records = probe_records + live_records + replay_records
    successful = [r for r in all_records if r["provider_valid"] == 1]
    #: Exactly the responses a cache write is expected for: validated *and*
    #: obtained live. A validated cache hit is never re-written.
    written = [r for r in all_records
               if r["provider_valid"] == 1 and r["served_from"] == "live"]
    attempted = [e for e in ledger if e["outcome"] == "response"]
    parsed = [e for e in attempted if e.get("parsed")]
    grounded = [e for e in parsed if e.get("grounded")]
    invented = [e for e in parsed if e.get("invented_ids")]

    live["metrics"] = {
        "request_attempts_including_retries": len(ledger),
        "successful_api_responses": len(attempted),
        "api_failures_by_category": distribution(
            "{}{}".format(e.get("error_class", "unknown"),
                          "/" + str(e["status_code"]) if "status_code" in e else "")
            for e in ledger if e["outcome"] == "exception"),
        "refusals": sum(1 for r in all_records
                        if r["fallback_reason"] == "provider_refusal"),
        "parse_successes": len(parsed),
        "parse_failures": len(attempted) - len(parsed),
        "grounded_selections": len(grounded),
        "invented_or_unoffered_ids": sum(len(e["invented_ids"]) for e in invented),
        "validator_acceptances": len(successful),
        "validator_rejections": len(all_records) - len(successful),
        "cache_writes_expected": len(written),
        "input_tokens": sum(e.get("input_tokens") or 0 for e in attempted),
        "output_tokens": sum(e.get("output_tokens") or 0 for e in attempted),
        "candidate_count_distribution": distribution(
            r["candidate_count"] for r in all_records),
        "selected_assignment_distribution": distribution(
            ";".join(r["selected_assignments"])
            for r in all_records if r["selected_assignments"]),
        "guidance_activations": sum(r["activated"] for r in all_records),
        "cache_hits": sum(1 for r in all_records if r["served_from"] == "cache"),
        "cache_misses": sum(1 for r in all_records if r["served_from"] == "live"),
    }

    gate("at least 95% of successful responses parse and ground to an offered id",
         not attempted or len(grounded) / float(len(attempted)) >= 0.95,
         {"grounded": len(grounded), "responses": len(attempted)})
    gate("zero invented ids were activated",
         live["metrics"]["invented_or_unoffered_ids"] == 0)
    gate("zero invalid guidance activated",
         all(r["activated"] == 0 for r in all_records if r["provider_valid"] == 0))
    gate("every activated assignment came from the offered id set",
         all(r["all_selected_ids_offered"] for r in all_records if r["activated"]))
    gate("beta stayed zero and no reward was altered",
         all(r["beta"] == 0.0 and r["reward_unchanged"] for r in all_records))

    live["pilot_accounting"] = pilot_phase.accounting.as_dict()
    problems = pilot_phase.accounting.check_invariants(max_retries=1)
    gate("pilot accounting invariants hold", not problems, problems)
    cache_rows = pilot_phase.cache.row_count()
    pilot_phase.close()
    live["cache_rows"] = cache_rows
    gate("only validated live responses were written to the cache",
         cache_rows == len(written), {"rows": cache_rows,
                                      "validated_live": len(written)})

    # -- read-only phase: hits serve, misses never call --------------------
    ro_phase = Phase(provider, cache_path=cache_path, cache_mode=CACHE_READ_ONLY,
                     live_call_cap=0, run_identity=LIVE_RUN_IDENTITY + "|readonly",
                     read_only=True)
    before = len(ledger)
    ro_hits = [ro_phase.run_fixture(replayed(f, offset=200))
               for f in fixtures if f["name"] in sent_live]
    ro_miss = ro_phase.run_fixture(unseen_fixture())
    live["read_only"] = {
        "hits": sum(1 for r in ro_hits if r["served_from"] == "cache"),
        "hit_fixtures": len(ro_hits),
        "miss_served_from": ro_miss["served_from"],
        "miss_fallback_reason": ro_miss["fallback_reason"],
        "miss_activated": ro_miss["activated"],
        "attempts_during_read_only": len(ledger) - before,
        "accounting": ro_phase.accounting.as_dict(),
    }
    ro_problems = ro_phase.accounting.check_invariants(max_retries=1)
    ro_phase.close()

    gate("read-only hits serve without a live call",
         live["read_only"]["hits"] == len(ro_hits)
         and live["read_only"]["attempts_during_read_only"] == 0)
    gate("read-only misses never call and activate no guidance",
         ro_miss["served_from"] == "live" and ro_miss["activated"] == 0
         and ro_miss["fallback_reason"] == "cache_read_only_miss")
    gate("read-only accounting invariants hold", not ro_problems, ro_problems)

    gate("the provider never silently changed model",
         set(pilot_phase.accounting.returned_models) <= {MODEL_SNAPSHOT},
         dict(pilot_phase.accounting.returned_models))
    gate("attempt counts reconcile across ledger, provider and accounting",
         len(ledger) == provider.attempts,
         {"ledger": len(ledger), "provider": provider.attempts})

    live["records"] = all_records + ro_hits + [ro_miss]
    return live


if __name__ == "__main__":
    sys.exit(main())
