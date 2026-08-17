"""SELMA Stage D: canonical cache key (scheme B) and the SQLite store.

Every test here was written against the pre-change tree, where none of
``selma.cache`` existed, and each one fails without the corresponding
implementation. See the fails-first section of
``docs/project/selma_stage_d_provider_cache_implementation_2026-08-16.md``.

Run:  PYTHONPATH=src pytest tests/test_selma_stage_d_cache.py -v
"""

import json
import os
import sqlite3
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from selma.cache import (  # noqa: E402
    CACHE_KEY_VERSION,
    CACHE_SCHEMA_VERSION,
    EXCLUDED_KEY_FIELDS,
    INCLUDED_KEY_FIELDS,
    MISS_CORRUPT_ROW,
    MISS_FULL_KEY_MISMATCH,
    MISS_UNDESERIALISABLE,
    CacheError,
    SqliteGuidanceCache,
    _row_checksum,
    build_cache_key,
    canonical_json,
    key_digest,
    request_to_dict,
    response_from_dict,
    response_to_dict,
)
from selma.modes import FROZEN_ACTIVE_VARIANT  # noqa: E402
from selma.provider import DeterministicMockProvider, build_request  # noqa: E402

GRID = (8, 8)
FOOD = (4, 4)
CONDITION = "lbforaging:Foraging-8x8-2p-2f-coop-v3"
MODEL = "gpt-5.4-nano-2026-03-17"

KEY_CONTEXT = dict(
    condition=CONDITION, grid_shape=GRID, provider="openai", model_snapshot=MODEL,
    reasoning_effort="none", prompt_version="selma_guidance_prompt_v3",
    response_schema_version="selma_guidance_response_schema_v2",
)


def event_row(episode_id=7, food_pos=FOOD, recipients=(0, 1),
              cells=((4, 3), (3, 4)), loader_levels=(1,), helper_levels=(2,),
              food_level=3, step=3):
    return {
        "row_type": "coalesced_event",
        "event_id": "ep{}-{}-food{},{}-t{}".format(
            episode_id, FROZEN_ACTIVE_VARIANT, food_pos[0], food_pos[1], step),
        "variant_id": FROZEN_ACTIVE_VARIANT,
        # (episode_id, row, col): the first element is episode identity.
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
    positions = positions or {0: (4, 3), 1: (6, 6)}
    return build_request(event_row(**row_kwargs), agent_positions=positions,
                         grid_shape=grid_shape)


def key_of(request, **overrides):
    context = dict(KEY_CONTEXT)
    context.update(overrides)
    return canonical_json(build_cache_key(request, **context))


# ==========================================================================
# 1. key composition: what is in, what is out
# ==========================================================================


def test_included_and_excluded_field_sets_are_disjoint_and_documented():
    assert set(INCLUDED_KEY_FIELDS) & set(EXCLUDED_KEY_FIELDS) == set()
    payload = build_cache_key(make_request(), **KEY_CONTEXT)
    assert set(payload) == set(INCLUDED_KEY_FIELDS)


def test_episode_id_does_not_affect_the_cache_key():
    """The whole point of scheme B: the same situation in another episode hits."""
    assert key_of(make_request(episode_id=7)) == key_of(make_request(episode_id=999))


def test_episode_scoped_food_id_is_excluded_from_the_key():
    payload = build_cache_key(make_request(episode_id=7), **KEY_CONTEXT)
    assert "food_id" not in payload
    serialised = canonical_json(payload)
    # the episode component of food_id must not survive anywhere in the key
    assert "\"food_id\"" not in serialised


def test_event_id_and_time_fields_are_excluded():
    payload = build_cache_key(make_request(step=3), **KEY_CONTEXT)
    for banned in ("event_id", "episode_id", "episode_step", "t_env", "seed",
                   "created_at", "latency_s", "retry_count", "provider_request_id"):
        assert banned not in payload


def test_food_position_changes_the_key():
    assert key_of(make_request(food_pos=(4, 4), cells=((4, 3), (3, 4)))) != \
        key_of(make_request(food_pos=(5, 4), cells=((5, 3), (4, 4))))


def test_changed_agent_position_changes_the_key():
    a = key_of(make_request(positions={0: (4, 3), 1: (6, 6)}))
    b = key_of(make_request(positions={0: (4, 3), 1: (6, 5)}))
    assert a != b


def test_changed_helper_coalition_changes_the_key():
    a = key_of(make_request())
    b = key_of(make_request(helper_levels=(3,)))
    assert a != b


def test_changed_recipient_set_changes_the_key():
    base = make_request()
    other = build_request(
        event_row(recipients=(0,)), agent_positions={0: (4, 3), 1: (6, 6)},
        grid_shape=GRID)
    assert key_of(base) != key_of(other)


def test_food_level_and_deficit_change_the_key():
    assert key_of(make_request(food_level=3)) != key_of(make_request(food_level=4))


def test_candidate_loading_cells_change_the_key():
    assert key_of(make_request(cells=((4, 3), (3, 4)))) != \
        key_of(make_request(cells=((4, 3), (3, 4), (5, 4))))


@pytest.mark.parametrize("field,value", [
    ("prompt_version", "selma_guidance_prompt_v2"),
    ("model_snapshot", "gpt-5.4-nano-2026-09-01"),
    ("response_schema_version", "selma_guidance_response_schema_v9"),
    ("provider", "some_other_provider"),
    ("reasoning_effort", "low"),
    ("condition", "lbforaging:Foraging-10x10-3p-3f-v3"),
])
def test_version_and_provider_changes_cause_a_miss(field, value):
    request = make_request()
    assert key_of(request) != key_of(request, **{field: value})


def test_canonical_ordering_produces_the_same_key():
    """Input ordering must not matter: only content does."""
    a = build_request(event_row(recipients=(0, 1), cells=((4, 3), (3, 4))),
                      agent_positions={0: (4, 3), 1: (6, 6)}, grid_shape=GRID)
    b = build_request(event_row(recipients=(1, 0), cells=((3, 4), (4, 3))),
                      agent_positions={1: (6, 6), 0: (4, 3)}, grid_shape=GRID)
    assert key_of(a) == key_of(b)
    assert key_digest(key_of(a)) == key_digest(key_of(b))


def test_canonical_json_is_byte_stable_and_sorted():
    text = key_of(make_request())
    assert text == canonical_json(json.loads(text))
    assert ", " not in text and '": ' not in text  # no insignificant whitespace


def test_key_carries_its_own_versions():
    payload = build_cache_key(make_request(), **KEY_CONTEXT)
    assert payload["cache_schema_version"] == CACHE_SCHEMA_VERSION
    assert payload["cache_key_version"] == CACHE_KEY_VERSION


def test_no_symmetry_normalisation_is_applied():
    """Scheme D was rejected; a reflected layout must NOT collide with its mirror."""
    original = make_request(food_pos=(2, 3), cells=((2, 2), (1, 3)),
                            positions={0: (2, 2), 1: (6, 6)})
    mirrored = make_request(food_pos=(3, 2), cells=((2, 2), (3, 1)),
                            positions={0: (2, 2), 1: (6, 6)})
    assert key_of(original) != key_of(mirrored)


def test_translated_layout_is_not_normalised_away():
    a = make_request(food_pos=(2, 2), cells=((2, 1), (1, 2)),
                     positions={0: (2, 1), 1: (6, 6)})
    b = make_request(food_pos=(5, 5), cells=((5, 4), (4, 5)),
                     positions={0: (5, 4), 1: (6, 6)})
    assert key_of(a) != key_of(b)


# ==========================================================================
# 2. the SQLite store
# ==========================================================================


def store(tmp_path, name="cache.sqlite3", **kwargs):
    kwargs.setdefault("run_identity", "cond|seed4242|openai|" + MODEL)
    return SqliteGuidanceCache(str(tmp_path / name), **kwargs)


def put(cache, request, response, **overrides):
    canonical = key_of(request, **overrides)
    digest = key_digest(canonical)
    cache.put(
        digest=digest, canonical=canonical,
        request_payload=request_to_dict(request),
        response_payload=response_to_dict(response),
        metadata={"request_id": "resp_123", "input_tokens": 11},
        provider="openai", requested_model=MODEL, returned_model=MODEL,
        reasoning_effort="none", prompt_version="selma_guidance_prompt_v3",
        response_schema_version="selma_guidance_response_schema_v2",
    )
    return digest, canonical


def test_roundtrip_hit_returns_an_equivalent_response(tmp_path):
    cache = store(tmp_path)
    request = make_request(episode_id=1)
    response = DeterministicMockProvider().query(request)
    digest, canonical = put(cache, request, response)

    later = make_request(episode_id=42)          # different episode, same situation
    hit = cache.lookup(key_digest(key_of(later)), key_of(later))
    assert hit is not None
    rebuilt = response_from_dict(hit["response"], event_id=later.event_id,
                                 target_food_id=later.food_id)
    assert rebuilt.event_id == later.event_id            # rebound to this event
    assert rebuilt.target_food_id == tuple(later.food_id)
    assert [a.loading_cell for a in rebuilt.assignments] == \
        [a.loading_cell for a in response.assignments]


def test_miss_returns_none(tmp_path):
    cache = store(tmp_path)
    request = make_request()
    assert cache.lookup(key_digest(key_of(request)), key_of(request)) is None


def test_hash_match_with_different_canonical_key_is_rejected(tmp_path):
    """A digest is an index, never an identity: the full key decides."""
    cache = store(tmp_path)
    request = make_request()
    response = DeterministicMockProvider().query(request)
    digest, canonical = put(cache, request, response)

    forged = canonical.replace("\"food_level\":3", "\"food_level\":9")
    assert forged != canonical
    assert cache.lookup(digest, forged) is None
    assert cache.stale_reasons.get(MISS_FULL_KEY_MISMATCH) == 1


def test_corrupted_row_is_rejected(tmp_path):
    cache = store(tmp_path)
    request = make_request()
    response = DeterministicMockProvider().query(request)
    digest, canonical = put(cache, request, response)
    cache.close()

    raw = sqlite3.connect(str(tmp_path / "cache.sqlite3"))
    with raw:
        raw.execute("UPDATE guidance_cache SET response_json = ? WHERE key_digest = ?",
                    ('{"schema_version":"tampered"}', digest))
    raw.close()

    reopened = store(tmp_path)
    assert reopened.lookup(digest, canonical) is None
    assert reopened.stale_reasons.get(MISS_CORRUPT_ROW) == 1


def test_undeserialisable_response_is_rejected(tmp_path):
    """Checksum-consistent but unparseable JSON must still be a logged miss."""
    cache = store(tmp_path)
    request = make_request()
    digest, canonical = put(cache, request, DeterministicMockProvider().query(request))
    cache.close()

    broken = "{not json"
    raw = sqlite3.connect(str(tmp_path / "cache.sqlite3"))
    raw.row_factory = sqlite3.Row
    row = raw.execute("SELECT * FROM guidance_cache WHERE key_digest = ?",
                      (digest,)).fetchone()
    repaired = _row_checksum(row["canonical_key"], row["request_json"], broken,
                             row["provider_metadata_json"])
    with raw:
        raw.execute(
            "UPDATE guidance_cache SET response_json = ?, row_checksum = ? "
            "WHERE key_digest = ?", (broken, repaired, digest))
    raw.close()

    reopened = store(tmp_path)
    assert reopened.lookup(digest, canonical) is None
    assert reopened.stale_reasons.get(MISS_UNDESERIALISABLE) == 1


def test_metadata_never_stores_credentials(tmp_path):
    cache = store(tmp_path)
    request = make_request()
    response = DeterministicMockProvider().query(request)
    canonical = key_of(request)
    cache.put(
        digest=key_digest(canonical), canonical=canonical,
        request_payload=request_to_dict(request),
        response_payload=response_to_dict(response),
        metadata={
            "request_id": "resp_1",
            "authorization": "Bearer sk-should-never-be-stored",
            "headers": {"Authorization": "Bearer sk-nope"},
            "api_key": "sk-nope",
        },
        provider="openai", requested_model=MODEL, returned_model=MODEL,
        reasoning_effort="none", prompt_version="p", response_schema_version="s",
    )
    blob = open(str(tmp_path / "cache.sqlite3"), "rb").read()
    assert b"sk-should-never-be-stored" not in blob
    assert b"sk-nope" not in blob
    assert b"Authorization" not in blob
    assert b"resp_1" in blob   # auditable metadata is kept


def test_schema_version_and_metadata_are_recorded(tmp_path):
    cache = store(tmp_path)
    assert cache.meta("cache_schema_version") == CACHE_SCHEMA_VERSION
    assert cache.meta("cache_key_version") == CACHE_KEY_VERSION
    assert cache.meta("run_identity").startswith("cond|seed4242|openai|")
    assert cache.meta("created_at")


def test_created_at_is_metadata_not_part_of_the_key(tmp_path):
    cache = store(tmp_path)
    request = make_request()
    digest, canonical = put(cache, request, DeterministicMockProvider().query(request))
    assert "created_at" not in canonical
    assert cache.lookup(digest, canonical)["created_at"]


def test_a_second_run_may_not_silently_share_a_writable_cache(tmp_path):
    store(tmp_path).close()
    with pytest.raises(CacheError) as error:
        store(tmp_path, run_identity="cond|seed99|openai|" + MODEL)
    assert "run identity" in str(error.value)


def test_sharing_a_writable_cache_is_possible_but_never_silent(tmp_path):
    store(tmp_path).close()
    shared = store(tmp_path, run_identity="cond|seed99|openai|" + MODEL,
                   allow_shared_write=True)
    assert shared.row_count() == 0


def test_read_only_open_requires_an_existing_database(tmp_path):
    with pytest.raises(CacheError):
        store(tmp_path, name="absent.sqlite3", read_only=True)


def test_read_only_cache_refuses_to_write(tmp_path):
    store(tmp_path).close()
    read_only = store(tmp_path, read_only=True)
    request = make_request()
    with pytest.raises(CacheError):
        read_only.put(
            digest="d", canonical="c", request_payload={}, response_payload={},
            metadata={}, provider="openai", requested_model=MODEL,
            returned_model=MODEL, reasoning_effort="none", prompt_version="p",
            response_schema_version="s")


def test_writes_are_idempotent_and_atomic(tmp_path):
    cache = store(tmp_path)
    request = make_request()
    response = DeterministicMockProvider().query(request)
    put(cache, request, response)
    put(cache, request, response)
    assert cache.row_count() == 1


def test_mismatched_schema_version_refuses_to_open(tmp_path):
    cache = store(tmp_path)
    cache.set_meta("cache_schema_version", "selma_cache_schema_v0")
    cache.close()
    with pytest.raises(CacheError) as error:
        store(tmp_path)
    assert "schema" in str(error.value)
