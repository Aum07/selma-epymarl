"""SELMA Stage B: calibration-report generator.

Consumes the per-variant calibration summaries produced by
``ShadowDetectorBank.calibration_summary()`` (itself built on
``variant_report()``) and renders the Stage B calibration report: one ranked
48-variant table per environment plus a cross-environment consistency
comparison.

This script only reads and formats. It selects nothing, freezes nothing, and
writes no detector configuration.

Usage::

    python3 analysis/selma_calibration_report.py \
        --run 8x8=analysis/out/seed91/summary_8x8.json \
        --run 10x10=analysis/out/seed91/summary_10x10.json \
        --csv 8x8=analysis/out/seed91/shadow_8x8.csv \
        --csv 10x10=analysis/out/seed91/shadow_10x10.csv \
        --runtime 8x8=4713.2 --runtime 10x10=5011.8 \
        --verify-csv \
        --out docs/selma_stage_b_seed91_calibration.md \
        --json analysis/out/seed91/calibration_report.json

``--verify-csv`` streams each shadow CSV and independently re-derives the final
event count, the semantically-invalid event count and the recipient-count
distribution, then cross-checks them against the in-process counters. It is the
only part that touches the (large) CSVs.
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from selma.detector import (  # noqa: E402
    EPISODE_PHASES,
    RESOLUTION_HORIZONS,
    build_variant_grid,
)

# Presentational ordering only. Sorting by firing rate is an objective property
# of the trajectory, not a preference between variants: the report ranks so the
# table is readable, and deliberately expresses no selection.
RANK_KEY = "events_per_1000_steps"


def parse_labelled(values, name):
    out = {}
    for item in values or ():
        if "=" not in item:
            raise SystemExit(f"--{name} expects LABEL=VALUE, got {item!r}")
        label, value = item.split("=", 1)
        out[label] = value
    return out


def fmt(value, spec=".3f", none="-"):
    if value is None:
        return none
    if isinstance(value, float):
        return format(value, spec)
    return str(value)


def fmt_pct(value):
    return "-" if value is None else f"{100.0 * value:.2f}%"


def fmt_dist(mapping):
    if not mapping:
        return "-"
    return " ".join(f"{k}:{v}" for k, v in sorted(mapping.items(), key=_dist_key))


def _dist_key(item):
    try:
        return (0, int(item[0]))
    except (TypeError, ValueError):
        return (1, item[0])


def human_bytes(n):
    if n is None:
        return "-"
    step = 1024.0
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < step or unit == "TiB":
            return f"{n:,.1f} {unit}" if unit != "B" else f"{n:,.0f} B"
        n /= step


def human_time(seconds):
    if seconds is None:
        return "-"
    seconds = float(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{int(hours)}h {int(minutes)}m {secs:.1f}s"


# -- CSV cross-check -------------------------------------------------------


def verify_csv(path):
    """Independently re-derive event statistics by streaming the shadow CSV."""
    stats = {
        "rows": 0,
        "raw_candidate_rows": 0,
        "coalesced_event_rows": 0,
        "final_events_by_variant": {},
        "semantically_invalid_events_by_variant": {},
        "recipients_by_variant": {},
        "variants_present": set(),
        "max_t_env": 0,
        "episodes": set(),
    }
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            stats["rows"] += 1
            vid = row["variant_id"]
            stats["variants_present"].add(vid)
            stats["episodes"].add(row["episode_id"])
            try:
                stats["max_t_env"] = max(stats["max_t_env"], int(row["t_env"]))
            except (TypeError, ValueError):
                pass
            if row["row_type"] == "raw_candidate":
                stats["raw_candidate_rows"] += 1
                continue
            stats["coalesced_event_rows"] += 1
            stats["final_events_by_variant"][vid] = (
                stats["final_events_by_variant"].get(vid, 0) + 1
            )
            if row["coalition_feasible"].strip() not in ("1", "True", "true"):
                stats["semantically_invalid_events_by_variant"][vid] = (
                    stats["semantically_invalid_events_by_variant"].get(vid, 0) + 1
                )
            n_recipients = len(_parse_tuple(row["recipient_ids"]))
            bucket = stats["recipients_by_variant"].setdefault(vid, {})
            bucket[str(n_recipients)] = bucket.get(str(n_recipients), 0) + 1
    stats["variants_present"] = sorted(stats["variants_present"])
    stats["n_episodes_with_rows"] = len(stats.pop("episodes"))
    return stats


def _parse_tuple(text):
    text = (text or "").strip()
    if not text or text == "()":
        return ()
    try:
        value = eval(text, {"__builtins__": {}}, {})  # noqa: S307 - fixed literal format
    except Exception:
        return ()
    return tuple(value) if isinstance(value, (tuple, list)) else (value,)


def reconcile(summary, csv_stats):
    """Compare in-process counters against the CSV-derived counts."""
    problems = []
    for vid, entry in summary["variants"].items():
        counted = csv_stats["final_events_by_variant"].get(vid, 0)
        if counted != entry["final_events"]:
            problems.append(
                f"{vid}: final_events counter={entry['final_events']} csv={counted}"
            )
        invalid = csv_stats["semantically_invalid_events_by_variant"].get(vid, 0)
        if invalid != entry["semantically_invalid_events"]:
            problems.append(
                f"{vid}: semantically_invalid_events counter="
                f"{entry['semantically_invalid_events']} csv={invalid}"
            )
        csv_recipients = csv_stats["recipients_by_variant"].get(vid, {})
        if csv_recipients != entry["recipient_count_distribution"]:
            problems.append(
                f"{vid}: recipient distribution counter="
                f"{entry['recipient_count_distribution']} csv={csv_recipients}"
            )
    return problems


# -- table rendering -------------------------------------------------------

VARIANT_COLUMNS = (
    ("rank", "Rank", lambda e: e["rank"]),
    ("variant_id", "Variant", lambda e: f"`{e['variant_id']}`"),
    ("w_fail", "W_fail", lambda e: e["w_fail"]),
    ("m", "m", lambda e: e["m"]),
    ("team_gate_variant", "Team gate", lambda e: e["team_gate_variant"]),
    ("inbound_variant", "Inbound", lambda e: e["inbound_variant"]),
    ("observed_steps", "Observed steps", lambda e: f"{e['observed_steps']:,}"),
    ("eligible_steps", "Eligible steps", lambda e: f"{e['eligible_steps']:,}"),
    (
        "incomplete_team_window_steps",
        "Incomplete-window steps",
        lambda e: f"{e['incomplete_team_window_steps']:,}",
    ),
    ("raw_candidates", "Raw candidates", lambda e: f"{e['raw_candidates']:,}"),
    (
        "semantically_invalid_events",
        "Invalid events",
        lambda e: e["semantically_invalid_events"],
    ),
    (
        "semantically_invalid_candidates",
        "Invalid candidates (dropped)",
        lambda e: f"{e['semantically_invalid_candidates']:,}",
    ),
    ("final_events", "Final events", lambda e: f"{e['final_events']:,}"),
    (
        "events_per_1000_steps",
        "Events / 1k steps",
        lambda e: fmt(e["events_per_1000_steps"], ".4f"),
    ),
    (
        "events_per_episode",
        "Events / episode",
        lambda e: fmt(e["events_per_episode"], ".4f"),
    ),
    ("episodes_with_0_events", "Ep. 0 events", lambda e: f"{e['episodes_with_0_events']:,}"),
    ("episodes_with_1_event", "Ep. 1 event", lambda e: f"{e['episodes_with_1_event']:,}"),
    (
        "episodes_with_multiple_events",
        "Ep. >1 events",
        lambda e: f"{e['episodes_with_multiple_events']:,}",
    ),
    (
        "recipient_count_distribution",
        "Recipient counts",
        lambda e: fmt_dist(e["recipient_count_distribution"]),
    ),
    (
        "natural_resolution_rate_5",
        "Nat. res. @5",
        lambda e: fmt_pct(e["natural_resolution_rate_5"]),
    ),
    (
        "natural_resolution_rate_10",
        "Nat. res. @10",
        lambda e: fmt_pct(e["natural_resolution_rate_10"]),
    ),
    ("valid_denominator_5", "Valid denom. @5", lambda e: f"{e['valid_denominator_5']:,}"),
    ("valid_denominator_10", "Valid denom. @10", lambda e: f"{e['valid_denominator_10']:,}"),
    ("censored_within_5", "Censored @5", lambda e: f"{e['censored_within_5']:,}"),
    ("censored_within_10", "Censored @10", lambda e: f"{e['censored_within_10']:,}"),
    (
        "gate_open_percentage",
        "Gate open % (corrected)",
        lambda e: fmt_pct(e["gate_open_percentage"]),
    ),
    ("team_gate_retained", "Gate retained", lambda e: f"{e['team_gate_retained']:,}"),
    ("team_gate_suppressed", "Gate suppressed", lambda e: f"{e['team_gate_suppressed']:,}"),
    (
        "team_gate_filtering_contribution",
        "Gate filtering contribution",
        lambda e: fmt_pct(e["team_gate_filtering_contribution"]),
    ),
    ("team_gate_vacuity", "Gate vacuity", lambda e: e["team_gate_vacuity"]),
    (
        "raw_candidates_after_inbound",
        "After inbound suppression",
        lambda e: f"{e['raw_candidates_after_inbound']:,}",
    ),
    (
        "suppressed_by_inbound",
        "Inbound-suppressed",
        lambda e: f"{e['suppressed_by_inbound']:,}",
    ),
    (
        "same_food_coalesced",
        "Same-food coalesced",
        lambda e: f"{e['same_food_coalesced']:,}",
    ),
    ("suppressed_by_latch", "Latch-suppressed", lambda e: f"{e['suppressed_by_latch']:,}"),
    (
        "suppressed_by_cooldown",
        "Cooldown-suppressed",
        lambda e: f"{e['suppressed_by_cooldown']:,}",
    ),
    ("suppressed_by_rearm", "Rearm-suppressed", lambda e: f"{e['suppressed_by_rearm']:,}"),
    (
        "suppressed_by_budget",
        "Budget-suppressed",
        lambda e: f"{e['suppressed_by_budget']:,}",
    ),
    (
        "events_by_episode_phase",
        "Events by phase (early/mid/late)",
        lambda e: "/".join(str(e["events_by_episode_phase"][p]) for p in EPISODE_PHASES),
    ),
)


def rank_entries(summary):
    """Attach a presentational rank to every variant entry."""
    entries = []
    for vid, entry in summary["variants"].items():
        item = dict(entry)
        item["variant_id"] = vid
        entries.append(item)
    entries.sort(
        key=lambda e: (
            -(e[RANK_KEY] if e[RANK_KEY] is not None else -1.0),
            e["variant_id"],
        )
    )
    for index, entry in enumerate(entries, start=1):
        entry["rank"] = index
    return entries


def render_table(entries):
    header = "| " + " | ".join(c[1] for c in VARIANT_COLUMNS) + " |"
    sep = "|" + "|".join("---" for _ in VARIANT_COLUMNS) + "|"
    lines = [header, sep]
    for entry in entries:
        lines.append("| " + " | ".join(str(c[2](entry)) for c in VARIANT_COLUMNS) + " |")
    return "\n".join(lines)


def cross_environment(ranked_by_label):
    """Per-variant behaviour across environments, with a consistency verdict."""
    labels = list(ranked_by_label)
    by_variant = {
        label: {e["variant_id"]: e for e in entries}
        for label, entries in ranked_by_label.items()
    }
    rows = []
    for variant in build_variant_grid():
        vid = variant.variant_id
        entries = [by_variant[label].get(vid) for label in labels]
        if any(e is None for e in entries):
            continue
        rates = [e["events_per_1000_steps"] for e in entries]
        ranks = [e["rank"] for e in entries]
        vacuity = [e["team_gate_vacuity"] for e in entries]
        invalid = [e["semantically_invalid_events"] for e in entries]
        fires_everywhere = all(r is not None and r > 0 for r in rates)
        positive = [r for r in rates if r]
        rate_ratio = (max(positive) / min(positive)) if len(positive) == len(rates) and positive else None
        rank_delta = max(ranks) - min(ranks)
        non_vacuous = all(v in ("discriminating", "ungated") for v in vacuity)

        if not fires_everywhere:
            verdict = "inconsistent: silent in at least one environment"
        elif any(i > 0 for i in invalid):
            verdict = "invalid: semantically invalid events present"
        elif not non_vacuous:
            verdict = "inconsistent: team gate vacuous in at least one environment"
        elif rate_ratio is not None and rate_ratio <= 2.0 and rank_delta <= 12:
            verdict = "consistent"
        elif rate_ratio is not None and rate_ratio <= 4.0 and rank_delta <= 24:
            verdict = "partially consistent"
        else:
            verdict = "inconsistent: rate or rank differs strongly"

        rows.append(
            {
                "variant_id": vid,
                "w_fail": variant.w_fail,
                "m": variant.m,
                "team_gate_variant": variant.team_gate_variant,
                "inbound_variant": variant.inbound_variant,
                "rates": dict(zip(labels, rates)),
                "ranks": dict(zip(labels, ranks)),
                "events_per_episode": dict(
                    zip(labels, [e["events_per_episode"] for e in entries])
                ),
                "natural_resolution_10": dict(
                    zip(labels, [e["natural_resolution_rate_10"] for e in entries])
                ),
                "vacuity": dict(zip(labels, vacuity)),
                "rate_ratio": rate_ratio,
                "rank_delta": rank_delta,
                "verdict": verdict,
            }
        )
    rows.sort(key=lambda r: (r["verdict"] != "consistent", r["rate_ratio"] or 1e9))
    return labels, rows


def render_cross_table(labels, rows):
    head = ["Variant", "W_fail", "m", "Team gate", "Inbound"]
    for label in labels:
        head += [f"{label} ev/1k", f"{label} rank", f"{label} res@10", f"{label} gate"]
    head += ["Rate ratio", "Rank Δ", "Verdict"]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    for row in rows:
        cells = [
            f"`{row['variant_id']}`",
            str(row["w_fail"]),
            str(row["m"]),
            row["team_gate_variant"],
            row["inbound_variant"],
        ]
        for label in labels:
            cells += [
                fmt(row["rates"][label], ".4f"),
                str(row["ranks"][label]),
                fmt_pct(row["natural_resolution_10"][label]),
                row["vacuity"][label],
            ]
        cells += [fmt(row["rate_ratio"], ".2f"), str(row["rank_delta"]), row["verdict"]]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# -- main ------------------------------------------------------------------


def build_report(args):
    summaries = {}
    for label, path in parse_labelled(args.run, "run").items():
        with open(path) as handle:
            summaries[label] = json.load(handle)
        summaries[label]["_summary_path"] = path

    csv_paths = parse_labelled(args.csv, "csv")
    runtimes = {k: float(v) for k, v in parse_labelled(args.runtime, "runtime").items()}
    commands = parse_labelled(args.command, "command")

    ranked = {label: rank_entries(summary) for label, summary in summaries.items()}

    verification = {}
    if args.verify_csv:
        for label, path in csv_paths.items():
            stats = verify_csv(path)
            problems = reconcile(summaries[label], stats)
            stats["reconciliation_problems"] = problems
            stats["reconciled"] = not problems
            verification[label] = stats

    storage = {}
    for label in summaries:
        entry = {}
        for kind, path in (
            ("shadow_csv", csv_paths.get(label)),
            ("summary_json", summaries[label].get("_summary_path")),
        ):
            if path and os.path.exists(path):
                entry[kind] = {"path": path, "bytes": os.path.getsize(path)}
        entry["runtime_seconds"] = runtimes.get(label)
        entry["command"] = commands.get(label)
        storage[label] = entry

    labels, cross_rows = cross_environment(ranked)

    return {
        "summaries": summaries,
        "ranked": ranked,
        "verification": verification,
        "storage": storage,
        "cross_labels": labels,
        "cross_rows": cross_rows,
    }


def render_markdown(report, title):
    out = [f"# {title}", ""]
    out.append(
        "Ranking is presentational: variants are ordered by "
        f"`{RANK_KEY}` (descending). No detector configuration is selected, "
        "frozen or applied by this report."
    )
    out.append("")

    out.append("## Run inventory")
    out.append("")
    out.append(
        "| Environment | Seed | Agents | Grid | Episodes | Observed steps | Runtime | Shadow CSV | CSV size | Summary size |"
    )
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for label, summary in report["summaries"].items():
        any_entry = next(iter(summary["variants"].values()))
        store = report["storage"].get(label, {})
        csv_info = store.get("shadow_csv", {})
        json_info = store.get("summary_json", {})
        out.append(
            f"| `{summary['condition']}` ({label}) | {summary['seed']} | "
            f"{summary['n_agents']} | {summary['grid_shape'][0]}x{summary['grid_shape'][1]} | "
            f"{summary['episodes_seen']:,} | {any_entry['observed_steps']:,} | "
            f"{human_time(store.get('runtime_seconds'))} | "
            f"`{csv_info.get('path', '-')}` | {human_bytes(csv_info.get('bytes'))} | "
            f"{human_bytes(json_info.get('bytes'))} |"
        )
    out.append("")

    out.append("## Zero-guarantee confirmation")
    out.append("")
    out.append(
        "| Environment | Provider calls | Queries | Active subgoals | Shaping reward |"
    )
    out.append("|---|---|---|---|---|")
    for label, summary in report["summaries"].items():
        z = summary["zero_guarantees"]
        out.append(
            f"| {label} | {z['provider_call_count']} | {z['query_count']} | "
            f"{z['active_subgoal_count']} | {z['total_shaping_reward']} |"
        )
    out.append("")

    if report["verification"]:
        out.append("## Independent shadow-CSV reconciliation")
        out.append("")
        out.append(
            "| Environment | CSV rows | raw_candidate | coalesced_event | Variants present | Reconciled |"
        )
        out.append("|---|---|---|---|---|---|")
        for label, stats in report["verification"].items():
            out.append(
                f"| {label} | {stats['rows']:,} | {stats['raw_candidate_rows']:,} | "
                f"{stats['coalesced_event_rows']:,} | {len(stats['variants_present'])}/48 | "
                f"{'yes' if stats['reconciled'] else 'NO'} |"
            )
        out.append("")
        for label, stats in report["verification"].items():
            for problem in stats["reconciliation_problems"]:
                out.append(f"* **{label} mismatch**: {problem}")
        out.append("")

    for label, entries in report["ranked"].items():
        summary = report["summaries"][label]
        out.append(f"## Ranked 48-variant table — `{summary['condition']}`")
        out.append("")
        cfg = summary["lifecycle_config"]
        out.append(
            f"Lifecycle: rearm={cfg['rearm_steps']}, cooldown={cfg['cooldown_steps']}, "
            f"event budget={cfg['max_events_per_episode']}/episode, "
            f"W_inbound={cfg['w_inbound']}. Episodes: {summary['episodes_seen']:,}."
        )
        out.append("")
        out.append(render_table(entries))
        out.append("")

    out.append("## Cross-environment consistency")
    out.append("")
    out.append(
        "A variant is `consistent` when it fires in both environments, has no "
        "semantically invalid event, has a non-vacuous team gate in both, and "
        "its firing rate differs by at most 2x with a rank difference of at "
        "most 12."
    )
    out.append("")
    out.append(render_cross_table(report["cross_labels"], report["cross_rows"]))
    out.append("")

    out.append("## Commands")
    out.append("")
    for label, store in report["storage"].items():
        if store.get("command"):
            out.append(f"**{label}**")
            out.append("")
            out.append("```")
            out.append(store["command"])
            out.append("```")
            out.append("")

    return "\n".join(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", metavar="LABEL=SUMMARY_JSON")
    parser.add_argument("--csv", action="append", metavar="LABEL=SHADOW_CSV")
    parser.add_argument("--runtime", action="append", metavar="LABEL=SECONDS")
    parser.add_argument("--command", action="append", metavar="LABEL=COMMAND")
    parser.add_argument("--verify-csv", action="store_true")
    parser.add_argument("--title", default="SELMA Stage B — seed-91 shadow calibration")
    parser.add_argument("--out", help="Markdown output path")
    parser.add_argument("--json", dest="json_out", help="Machine-readable output path")
    args = parser.parse_args(argv)

    if not args.run:
        parser.error("at least one --run LABEL=SUMMARY_JSON is required")

    report = build_report(args)
    markdown = render_markdown(report, args.title)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as handle:
            handle.write(markdown + "\n")
    else:
        print(markdown)

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        payload = {
            "rank_key": RANK_KEY,
            "resolution_horizons": list(RESOLUTION_HORIZONS),
            "storage": report["storage"],
            "verification": report["verification"],
            "summaries": report["summaries"],
            "ranked": report["ranked"],
            "cross_environment": report["cross_rows"],
        }
        with open(args.json_out, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)

    failed = [
        label
        for label, stats in report["verification"].items()
        if not stats["reconciled"]
    ]
    if failed:
        raise SystemExit(f"CSV reconciliation failed for: {', '.join(failed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
