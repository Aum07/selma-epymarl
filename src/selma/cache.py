"""SELMA Stage D: canonical cache key (scheme B) and the persistent SQLite store.

Implements the scheme recommended by
``docs/project/selma_stage_d_provider_cache_feasibility_2026-08-16.md`` §5B:

* **episode-independent** -- nothing episode-scoped may enter the key;
* **absolute state** -- no translation, rotation, reflection or any other
  symmetry normalisation. Scheme D was rejected in the feasibility audit because
  the coordinate transform and its inverse are unproven, and nothing here
  reintroduces it;
* keyed on ``food_position``, **never** on ``food_id``. The detector's
  ``food_id`` is ``(episode_id, row, col)``: its first element is the episode
  identity, so a key containing it is episode-unique and yields ~5 % reuse
  (measured). This is the defect the audit found in the Stage C contract's
  proposed key;
* canonical JSON serialisation, SHA-256 digest, and a **full canonical-key
  comparison after every hash match** -- the digest is an index, never the
  identity.

Nothing in this module imports a provider SDK, performs I/O to a network, or
touches credentials.
"""

import hashlib
import json
import os
import sqlite3
import time

#: Storage schema. Bumping this invalidates every stored row.
CACHE_SCHEMA_VERSION = "selma_cache_schema_v1"

#: Key composition. Bumping this changes every digest and so invalidates reuse.
CACHE_KEY_VERSION = "selma_cache_key_v1"

#: Reasons a lookup that matched a digest is nevertheless treated as a miss.
MISS_FULL_KEY_MISMATCH = "full_key_mismatch"
MISS_CORRUPT_ROW = "corrupt_row"
MISS_SCHEMA_MISMATCH = "cache_schema_mismatch"
MISS_UNDESERIALISABLE = "undeserialisable_response"
MISS_STALE_REVALIDATION = "stale_revalidation_failed"

#: Every semantic field that enters the key. Documented here (and asserted in
#: the Stage D tests) so the key composition can never drift silently.
INCLUDED_KEY_FIELDS = (
    "cache_schema_version",
    "cache_key_version",
    "request_schema_version",
    "response_schema_version",
    "prompt_version",
    "provider",
    "model_snapshot",
    "reasoning_effort",
    "condition",
    "grid_shape",
    "event_type",
    "food_position",
    "food_level",
    "trigger_agents",
    "helper_agents",
    "recipient_agents",
    "level_deficit",
    "required_extra_level",
    "candidate_loading_cells",
    "candidate_assignment_ids",
    "allowed_roles",
)

#: Everything deliberately kept out. Each of these either carries episode
#: identity, is a logging-only artefact, or is a measurement of the call rather
#: than of the question -- including any of them would only prevent reuse.
EXCLUDED_KEY_FIELDS = (
    "episode_id",
    "t_env",
    "episode_step",
    "event_id",
    "food_id",            # episode-scoped: (episode_id, row, col)
    "seed",
    "query_timestamp",
    "created_at",
    "provider_request_id",
    "latency_s",
    "retry_count",
    "attempt_index",
    "token_usage",
    "wall_clock",
    "learner_state",
    "raw_team_reward",
)


def _cell(value):
    return [int(value[0]), int(value[1])]


def candidate_assignment_ids(request):
    """Deterministic subgoal identifiers offered to a provider.

    One id per (recipient, offered loading cell) pair, in a stable sorted order.
    A provider selects *among these ids*; it never supplies a coordinate. The
    ids are part of the cache key because they define the legal answer set.
    """
    ids = []
    for ctx in sorted(request.agent_context, key=lambda c: int(c.agent_id)):
        for cell in sorted({tuple(int(x) for x in c) for c in ctx.candidate_loading_cells}):
            ids.append("a{}@{},{}".format(int(ctx.agent_id), cell[0], cell[1]))
    return tuple(sorted(ids))


def build_cache_key(request, *, condition, grid_shape, provider, model_snapshot,
                    reasoning_effort, prompt_version, response_schema_version):
    """The canonical key payload for one guidance request (scheme B).

    Pure: identical inputs give an identical structure on any machine. Agent
    entries are sorted by id, cells are sorted, and every value is a plain JSON
    type, so ``canonical_json`` is byte-stable.
    """
    by_id = {int(c.agent_id): c for c in request.agent_context}
    triggers = sorted(int(a) for a in request.trigger_agent_ids)
    recipients = sorted(int(a) for a in request.recipient_agents)
    helpers = [a for a in recipients if a not in set(triggers)]

    def agent_entries(ids):
        out = []
        for agent_id in sorted(ids):
            ctx = by_id.get(agent_id)
            if ctx is None:
                # An agent named by the event but absent from the context would
                # change the legal answer; record it explicitly rather than
                # dropping it silently.
                out.append({"agent_id": agent_id, "level": None, "position": None})
                continue
            out.append({
                "agent_id": agent_id,
                "level": int(ctx.level),
                "position": _cell(ctx.position),
            })
        return out

    cells = sorted({tuple(int(x) for x in cell)
                    for ctx in request.agent_context
                    for cell in ctx.candidate_loading_cells})
    roles = sorted({str(r) for ctx in request.agent_context for r in ctx.allowed_roles})

    return {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "cache_key_version": CACHE_KEY_VERSION,
        "request_schema_version": str(request.schema_version),
        "response_schema_version": str(response_schema_version),
        "prompt_version": str(prompt_version),
        "provider": str(provider),
        "model_snapshot": str(model_snapshot),
        "reasoning_effort": str(reasoning_effort),
        "condition": str(condition),
        "grid_shape": [int(grid_shape[0]), int(grid_shape[1])],
        "event_type": str(request.event_type),
        "food_position": _cell(request.food_position),
        "food_level": int(request.food_level),
        "trigger_agents": agent_entries(triggers),
        "helper_agents": agent_entries(helpers),
        "recipient_agents": agent_entries(recipients),
        "level_deficit": int(request.level_deficit),
        "required_extra_level": int(request.required_extra_level),
        "candidate_loading_cells": [_cell(c) for c in cells],
        "candidate_assignment_ids": list(candidate_assignment_ids(request)),
        "allowed_roles": roles,
    }


def canonical_json(payload):
    """Byte-stable JSON: sorted keys, no insignificant whitespace, ASCII-safe."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True)


def key_digest(canonical):
    """SHA-256 of the canonical key. An index into the store, not an identity."""
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# request / response serialisation
# ---------------------------------------------------------------------------


def request_to_dict(request):
    return {
        "schema_version": request.schema_version,
        "event_type": request.event_type,
        "food_position": _cell(request.food_position),
        "food_level": int(request.food_level),
        "trigger_agent_ids": [int(a) for a in request.trigger_agent_ids],
        "recipient_agents": [int(a) for a in request.recipient_agents],
        "level_deficit": int(request.level_deficit),
        "required_extra_level": int(request.required_extra_level),
        "agent_context": [
            {
                "agent_id": int(c.agent_id),
                "level": int(c.level),
                "position": _cell(c.position),
                "allowed_roles": [str(r) for r in c.allowed_roles],
                "candidate_loading_cells": [_cell(x) for x in c.candidate_loading_cells],
            }
            for c in sorted(request.agent_context, key=lambda c: int(c.agent_id))
        ],
    }


def response_to_dict(response):
    return {
        "schema_version": response.schema_version,
        "event_id": response.event_id,
        "recipients": [int(a) for a in response.recipients],
        "target_food_id": [int(x) for x in response.target_food_id],
        "helper_coalition": [int(a) for a in response.helper_coalition],
        "assignments": [
            {"agent_id": int(a.agent_id), "role": str(a.role),
             "loading_cell": _cell(a.loading_cell)}
            for a in response.assignments
        ],
        "confidence": float(response.confidence),
    }


def response_from_dict(payload, *, event_id, target_food_id):
    """Rebuild a GuidanceResponse for the *current* request.

    ``event_id`` and ``target_food_id`` are episode-scoped and therefore are not
    stored in the key; they are re-bound from the live request. The
    deterministic validator then re-checks them, so a wrong binding fails closed
    rather than activating stale guidance.
    """
    from selma.provider import Assignment, GuidanceResponse

    return GuidanceResponse(
        schema_version=payload["schema_version"],
        event_id=event_id,
        recipients=tuple(int(a) for a in payload["recipients"]),
        target_food_id=tuple(int(x) for x in target_food_id),
        helper_coalition=tuple(int(a) for a in payload["helper_coalition"]),
        assignments=tuple(
            Assignment(agent_id=int(a["agent_id"]), role=str(a["role"]),
                       loading_cell=tuple(int(x) for x in a["loading_cell"]))
            for a in payload["assignments"]
        ),
        confidence=float(payload["confidence"]),
    )


def _row_checksum(canonical_key, request_json, response_json, metadata_json):
    digest = hashlib.sha256()
    for part in (CACHE_SCHEMA_VERSION, canonical_key, request_json,
                 response_json, metadata_json):
        digest.update(part.encode("utf-8"))
        digest.update(b"\x1f")
    return digest.hexdigest()


class CacheError(RuntimeError):
    """A cache configuration or integrity problem that must stop the run."""


class SqliteGuidanceCache:
    """Persistent guidance cache. One database per condition/seed/run.

    A writable database records the identity of the run that created it and
    refuses to be reopened for writing by a different run unless sharing is
    explicitly allowed, so two final runs can never silently share a store.
    """

    def __init__(self, path, *, run_identity, allow_shared_write=False,
                 read_only=False):
        self.path = str(path)
        self.run_identity = str(run_identity)
        self.read_only = bool(read_only)
        self.stale_reasons = {}
        directory = os.path.dirname(os.path.abspath(self.path))
        if directory and not read_only:
            os.makedirs(directory, exist_ok=True)
        if read_only and not os.path.exists(self.path):
            raise CacheError(
                "selma_cache_mode='read_only' but no cache database exists at "
                "{!r}. A read-only run never creates or fills a cache.".format(self.path)
            )
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema(allow_shared_write)

    # -- schema -------------------------------------------------------------

    def _init_schema(self, allow_shared_write):
        with self._conn:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS cache_meta ("
                " name TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS guidance_cache ("
                " key_digest TEXT PRIMARY KEY,"
                " canonical_key TEXT NOT NULL,"
                " cache_schema_version TEXT NOT NULL,"
                " cache_key_version TEXT NOT NULL,"
                " prompt_version TEXT NOT NULL,"
                " response_schema_version TEXT NOT NULL,"
                " provider TEXT NOT NULL,"
                " requested_model TEXT NOT NULL,"
                " returned_model TEXT,"
                " reasoning_effort TEXT,"
                " request_json TEXT NOT NULL,"
                " response_json TEXT NOT NULL,"
                " provider_metadata_json TEXT NOT NULL,"
                " row_checksum TEXT NOT NULL,"
                " created_at TEXT NOT NULL)"
            )
        stored_schema = self.meta("cache_schema_version")
        if stored_schema is None:
            if not self.read_only:
                self.set_meta("cache_schema_version", CACHE_SCHEMA_VERSION)
                self.set_meta("cache_key_version", CACHE_KEY_VERSION)
                self.set_meta("run_identity", self.run_identity)
                self.set_meta("created_at", _now())
        elif stored_schema != CACHE_SCHEMA_VERSION:
            raise CacheError(
                "cache at {!r} was written with schema {!r}; this build uses "
                "{!r}. Refusing to mix schemas.".format(
                    self.path, stored_schema, CACHE_SCHEMA_VERSION)
            )
        owner = self.meta("run_identity")
        if (not self.read_only and owner is not None
                and owner != self.run_identity and not allow_shared_write):
            raise CacheError(
                "cache at {!r} was created by run identity {!r} but this run is "
                "{!r}. Two final runs must not share a writable cache; use one "
                "database per condition/seed/run, or set "
                "selma_cache_allow_shared_write=True deliberately.".format(
                    self.path, owner, self.run_identity)
            )

    def meta(self, name):
        row = self._conn.execute(
            "SELECT value FROM cache_meta WHERE name = ?", (name,)).fetchone()
        return None if row is None else row["value"]

    def set_meta(self, name, value):
        with self._conn:
            self._conn.execute(
                "INSERT INTO cache_meta(name, value) VALUES(?, ?) "
                "ON CONFLICT(name) DO UPDATE SET value = excluded.value",
                (name, str(value)))

    # -- lookup / store -----------------------------------------------------

    def _note_stale(self, reason):
        self.stale_reasons[reason] = self.stale_reasons.get(reason, 0) + 1

    def lookup(self, digest, canonical):
        """Return the stored response payload, or ``None`` for a miss.

        A digest match is **not** a hit on its own: the full canonical key is
        compared, the row checksum is recomputed, and the stored JSON must
        deserialise. Any failure is a logged miss, never a silent reuse.
        """
        row = self._conn.execute(
            "SELECT * FROM guidance_cache WHERE key_digest = ?", (digest,)).fetchone()
        if row is None:
            return None
        if row["canonical_key"] != canonical:
            self._note_stale(MISS_FULL_KEY_MISMATCH)
            return None
        if row["cache_schema_version"] != CACHE_SCHEMA_VERSION:
            self._note_stale(MISS_SCHEMA_MISMATCH)
            return None
        expected = _row_checksum(row["canonical_key"], row["request_json"],
                                 row["response_json"], row["provider_metadata_json"])
        if expected != row["row_checksum"]:
            self._note_stale(MISS_CORRUPT_ROW)
            return None
        try:
            payload = json.loads(row["response_json"])
            metadata = json.loads(row["provider_metadata_json"])
        except (ValueError, TypeError):
            self._note_stale(MISS_UNDESERIALISABLE)
            return None
        return {"response": payload, "metadata": metadata,
                "returned_model": row["returned_model"],
                "created_at": row["created_at"]}

    def put(self, *, digest, canonical, request_payload, response_payload,
            metadata, provider, requested_model, returned_model, reasoning_effort,
            prompt_version, response_schema_version):
        """Store one validated response. Atomic; never called before validation."""
        if self.read_only:
            raise CacheError("refusing to write to a read-only cache")
        request_json = canonical_json(request_payload)
        response_json = canonical_json(response_payload)
        metadata_json = canonical_json(_scrub(metadata))
        checksum = _row_checksum(canonical, request_json, response_json, metadata_json)
        with self._conn:  # a single atomic transaction
            self._conn.execute(
                "INSERT INTO guidance_cache("
                " key_digest, canonical_key, cache_schema_version, cache_key_version,"
                " prompt_version, response_schema_version, provider, requested_model,"
                " returned_model, reasoning_effort, request_json, response_json,"
                " provider_metadata_json, row_checksum, created_at)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
                " ON CONFLICT(key_digest) DO NOTHING",
                (digest, canonical, CACHE_SCHEMA_VERSION, CACHE_KEY_VERSION,
                 str(prompt_version), str(response_schema_version), str(provider),
                 str(requested_model), None if returned_model is None else str(returned_model),
                 str(reasoning_effort), request_json, response_json, metadata_json,
                 checksum, _now()),
            )

    def row_count(self):
        return self._conn.execute("SELECT COUNT(*) AS n FROM guidance_cache").fetchone()["n"]

    def close(self):
        self._conn.close()


#: Metadata keys that must never be persisted, whatever a provider returns.
_FORBIDDEN_METADATA = ("authorization", "api_key", "api-key", "openai_api_key",
                       "secret", "token", "bearer", "cookie", "headers",
                       "set-cookie", "x-api-key", "password", "credential")


def _scrub(metadata):
    """Drop anything that could carry a credential before it reaches disk.

    Auditing needs the request id, the returned model and token usage. It never
    needs a header, so no header is stored: there is nothing to leak.
    """
    clean = {}
    for name, value in (metadata or {}).items():
        lowered = str(name).lower()
        if any(bad in lowered for bad in _FORBIDDEN_METADATA):
            continue
        if isinstance(value, dict):
            clean[name] = _scrub(value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            clean[name] = value
        else:
            clean[name] = str(value)
    return clean


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
