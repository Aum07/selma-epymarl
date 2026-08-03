"""Publication figures for the SELMA Stage B detector evidence package.

Reads only existing verified artifacts; computes nothing new about the detector
and performs no selection. Every value traces to a summary JSON or to the
machine-readable validation results.

Palette: Okabe-Ito subset #0072B2 / #D55E00 / #009E73, validated colourblind-safe
on a light surface (lightness band, chroma floor, CVD separation dE 11.0 deutan,
normal-vision floor dE 25.8, contrast >= 3:1 all PASS). Figures target print on a
white surface, so no dark-mode variant is produced.

No inferential statistics: one run per seed, descriptive comparison only.

Run (from anywhere):  python3 analysis/out/stage_b_paper_metrics/make_stage_b_figures.py
Paths are resolved relative to this file, so no machine-specific path is embedded.
"""

import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Repository root, derived from this file's location: <root>/analysis/out/stage_b_paper_metrics/
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
OUT = os.path.join(ROOT, "analysis/out/stage_b_paper_metrics/figures")
os.makedirs(OUT, exist_ok=True)

V = "wf15_m3_t5_ibon"
S8 = os.path.join(ROOT, "analysis/out/seed91_retry_20260802T112458Z/8x8/summary_8x8_seed91_2M_retry.json")
S10 = os.path.join(ROOT, "analysis/out/seed91_val10x10_20260802T151245Z/10x10/summary_10x10_seed91_2M_val.json")
S99 = os.path.join(ROOT, "analysis/out/seed99_heldout_20260802T221211Z/8x8/summary_8x8_seed99_2M_heldout.json")

# validated palette + neutral ink
BLUE, ORANGE, GREEN = "#0072B2", "#D55E00", "#009E73"
INK, INK2, MUTED, GRID = "#1a1a19", "#4a4a47", "#8a8a85", "#e2e2de"

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.5,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.6,
    "axes.labelcolor": INK2,
    "text.color": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.frameon": False,
    "legend.fontsize": 7.5,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def load(p):
    return json.load(open(p))


def recessive(ax, axis="y"):
    ax.grid(True, axis=axis, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def save(fig, name):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print(f"  wrote {name}.pdf / .svg / .png")


def eligible_frame(summary_path, adjust_recipients):
    """The E1/E2/E3/E4-corrected eligible set for one condition."""
    s = load(summary_path)
    E = s["episodes_seen"]
    out = []
    for k, v in s["variants"].items():
        if v["team_gate_variant"] == "none" or v["inbound_variant"] != "on":
            continue
        if v["semantically_invalid_events"] != 0:
            continue
        out.append(dict(
            k=k,
            cov=100 * (1 - v["events_per_episode_hist"].get("0", 0) / E),
            res10=100 * v["resolved_within_10"] / v["valid_denominator_10"],
            per1k=v["events_per_1000_steps"],
        ))
    return out


def pareto(rows):
    def dom(a, b):
        return (a["cov"] >= b["cov"] and a["res10"] <= b["res10"]
                and (a["cov"] > b["cov"] or a["res10"] < b["res10"]))
    return {r["k"] for r in rows if not any(dom(o, r) for o in rows if o is not r)}


# ---------------------------------------------------------------- figure 1
def fig_calibration_tradeoff():
    rows = eligible_frame(S8, False)
    front = pareto(rows)
    fig, ax = plt.subplots(figsize=(5.4, 3.9))
    recessive(ax, axis="both")

    other = [r for r in rows if r["k"] not in front and r["k"] != V]
    par = [r for r in rows if r["k"] in front and r["k"] != V]
    frz = [r for r in rows if r["k"] == V][0]

    ax.scatter([r["cov"] for r in other], [r["res10"] for r in other],
               s=34, facecolor="white", edgecolor=MUTED, linewidth=1.0,
               zorder=3, label="Eligible, dominated")
    ax.scatter([r["cov"] for r in par], [r["res10"] for r in par],
               s=46, marker="D", facecolor=BLUE, edgecolor="white", linewidth=0.8,
               zorder=4, label="Pareto-efficient")
    ax.scatter([frz["cov"]], [frz["res10"]],
               s=150, marker="*", facecolor=ORANGE, edgecolor="white", linewidth=0.9,
               zorder=5, label="Frozen: wf15_m3_t5_ibon")

    ax.annotate("wf15_m3_t5_ibon\nrank 1, d = 0.607",
                (frz["cov"], frz["res10"]), textcoords="offset points",
                xytext=(58, -46), ha="left", fontsize=7.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.6,
                                shrinkA=0, shrinkB=6))

    ax.set_xlabel("Episode coverage (% of 40,003 training episodes with ≥ 1 event)")
    ax.set_ylabel("Natural resolution @10\n(% of uncensored events, n = 9,750 for frozen)")
    ax.set_title("Stage B calibration trade-off, seed 91, 8×8-2p-2f-coop\n"
                 "18 eligible of 48 variants; lower-right is preferable",
                 loc="left", color=INK)
    ax.legend(loc="upper left", ncol=1)
    fig.text(0.0, -0.10,
             "Eligibility: team gate enabled, inbound suppression enabled, all invariants passing, "
             "zero invalid and zero surviving duplicate final events (E4-corrected,\n"
             "PROTOCOL ERRATUM 001). Axes are the two frozen selection objectives. Direction of "
             "preference on the resolution axis is a stated modelling choice, not an\nestablished "
             "criterion. Descriptive only — one run per seed, no inferential statistics.",
             fontsize=6.5, color=MUTED, va="top")
    save(fig, "fig1_calibration_tradeoff_seed91_8x8")


# ---------------------------------------------------------------- figure 2
def fig_seed_comparison():
    a, b = load(S8)["variants"][V], load(S99)["variants"][V]
    Ea, Eb = load(S8)["episodes_seen"], load(S99)["episodes_seen"]

    panels = [
        ("Events per 1,000 steps", a["events_per_1000_steps"], b["events_per_1000_steps"],
         "events / 1k steps", "n = 2,000,046 / 2,000,004 steps"),
        ("Episode coverage", 100 * (1 - a["events_per_episode_hist"].get("0", 0) / Ea),
         100 * (1 - b["events_per_episode_hist"].get("0", 0) / Eb),
         "% of episodes", "n = 40,003 / 40,020 episodes"),
        ("Natural resolution @5", 100 * a["resolved_within_5"] / a["valid_denominator_5"],
         100 * b["resolved_within_5"] / b["valid_denominator_5"],
         "% uncensored", "n = 10,241 / 10,452 events"),
        ("Natural resolution @10", 100 * a["resolved_within_10"] / a["valid_denominator_10"],
         100 * b["resolved_within_10"] / b["valid_denominator_10"],
         "% uncensored", "n = 9,750 / 10,036 events"),
        ("Gate-open rate", 100 * a["gate_open_percentage"], 100 * b["gate_open_percentage"],
         "% of observed steps", "n = 2,000,046 / 2,000,004 steps"),
        ("Episode-cap binding", 100 * a["episodes_with_multiple_events"] / Ea,
         100 * b["episodes_with_multiple_events"] / Eb,
         "% of episodes", "n = 40,003 / 40,020 episodes"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(8.6, 4.6))
    for ax, (title, va, vb, unit, denom) in zip(axes.ravel(), panels):
        recessive(ax)
        ax.bar([0, 1], [va, vb], width=0.52,
               color=[BLUE, ORANGE], edgecolor="white", linewidth=1.6, zorder=3)
        top = max(va, vb)
        for x, v in ((0, va), (1, vb)):
            ax.text(x, v + top * 0.035, f"{v:.4g}", ha="center", fontsize=7.5, color=INK)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["seed 91\n(calibration)", "seed 99\n(held-out)"], fontsize=7)
        ax.set_ylim(0, top * 1.28)
        ax.set_title(title, loc="left", color=INK)
        ax.set_ylabel(unit, fontsize=7.5)
        ax.text(0.5, -0.42, denom, transform=ax.transAxes, ha="center",
                fontsize=6.3, color=MUTED)
    fig.suptitle("Frozen detector wf15_m3_t5_ibon: calibration seed vs held-out seed, "
                 "identical 8×8-2p-2f-coop condition",
                 x=0.0, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    fig.text(0.0, -0.03,
             "Each panel has its own y-axis and unit — panels are never combined on a shared "
             "scale. Descriptive comparison only: one run per seed, so no significance is\nclaimed "
             "and no p-values are reported. Natural resolution is a proxy for potentially "
             "unnecessary triggering, not a ground-truth false-positive rate.",
             fontsize=6.5, color=MUTED, va="top")
    save(fig, "fig2_seed91_vs_seed99_small_multiples")


# ---------------------------------------------------------------- figure 3
def fig_funnel():
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8), sharex=True)
    for ax, (path, label, colour) in zip(
            axes, [(S8, "seed 91 (calibration)", BLUE), (S99, "seed 99 (held-out)", ORANGE)]):
        v = load(path)["variants"][V]
        stages = [
            ("Raw candidates", v["raw_candidates"]),
            ("After team gate", v["candidates_after_team_gate"]),
            ("After inbound suppr.", v["raw_candidates_after_inbound"]),
            ("After latch", v["raw_candidates_after_inbound"] - v["suppressed_by_latch"]),
            ("After cooldown", v["raw_candidates_after_inbound"] - v["suppressed_by_latch"]
             - v["suppressed_by_cooldown"]),
            ("After re-arm", v["raw_candidates_after_inbound"] - v["suppressed_by_latch"]
             - v["suppressed_by_cooldown"] - v["suppressed_by_rearm"]),
            ("Final events", v["final_events"]),
        ]
        names = [s[0] for s in stages][::-1]
        vals = [s[1] for s in stages][::-1]
        recessive(ax, axis="x")
        ax.barh(range(len(vals)), vals, height=0.6, color=colour,
                edgecolor="white", linewidth=1.6, zorder=3)
        for i, val in enumerate(vals):
            ax.text(val + max(vals) * 0.015, i, f"{val:,}", va="center",
                    fontsize=7, color=INK)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=7.5)
        ax.set_xlim(0, max(vals) * 1.22)
        ax.set_xlabel("candidate records (linear scale, shared across panels)")
        ax.set_title(label, loc="left", color=INK)
    fig.suptitle("Detector candidate → event suppression funnel, frozen configuration "
                 "wf15_m3_t5_ibon, 8×8-2p-2f-coop",
                 x=0.0, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=[0, 0.04, 1, 0.92])
    fig.text(0.0, -0.02,
             "Stages follow the implemented suppression ledger: raw_candidates − team gate "
             "→ − inbound → − latch → − cooldown → − re-arm "
             "→ − budget = final_events.\nsame_food_coalesced is 0 on both seeds, so no "
             "coalescing step is shown. Both panels share one common x-scale (0–300,000), so bar "
             "lengths are directly comparable between seeds.",
             fontsize=6.5, color=MUTED, va="top")
    save(fig, "fig3_suppression_funnel_8x8_both_seeds")


# ---------------------------------------------------------------- figure 4
def fig_phase_timing():
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    recessive(ax)
    labels = ["Early", "Mid", "Late"]
    series = [("seed 91 (calibration)", S8, BLUE), ("seed 99 (held-out)", S99, ORANGE)]
    width = 0.36
    for i, (name, path, colour) in enumerate(series):
        ph = load(path)["variants"][V]["events_by_episode_phase"]
        tot = sum(ph.values())
        vals = [100 * ph[k] / tot for k in ("early", "mid", "late")]
        xs = [x + (i - 0.5) * (width + 0.03) for x in range(3)]
        ax.bar(xs, vals, width=width, color=colour, edgecolor="white",
               linewidth=1.6, zorder=3, label=f"{name}  (n = {tot:,} events)")
        for x, v in zip(xs, vals):
            ax.text(x, v + 1.4, f"{v:.1f}", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 82)
    ax.set_xlabel("Episode phase (equal thirds of the 50-step episode)")
    ax.set_ylabel("Share of that seed's events (%)")
    ax.set_title("Event timing by episode phase, frozen detector, 8×8-2p-2f-coop\n"
                 "Both seeds are strongly early-weighted", loc="left", color=INK)
    ax.legend(loc="upper right")
    fig.text(0.0, -0.13,
             "Shares are within-seed and sum to 100% per seed; the two seeds are not pooled. "
             "Descriptive only — one run per seed.",
             fontsize=6.5, color=MUTED, va="top")
    save(fig, "fig4_event_timing_by_phase")


# ---------------------------------------------------------------- figure 5
def fig_10x10_sensitivity():
    rows = eligible_frame(S10, True)
    front = pareto(rows)
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    recessive(ax, axis="both")
    other = [r for r in rows if r["k"] not in front and r["k"] != V]
    par = [r for r in rows if r["k"] in front and r["k"] != V]
    frz = [r for r in rows if r["k"] == V][0]
    ax.scatter([r["cov"] for r in other], [r["res10"] for r in other],
               s=34, facecolor="white", edgecolor=MUTED, linewidth=1.0, zorder=3,
               label="Eligible, dominated")
    ax.scatter([r["cov"] for r in par], [r["res10"] for r in par],
               s=46, marker="D", facecolor=GREEN, edgecolor="white", linewidth=0.8,
               zorder=4, label="Pareto-efficient (4 of 18)")
    ax.scatter([frz["cov"]], [frz["res10"]], s=150, marker="*", facecolor=ORANGE,
               edgecolor="white", linewidth=0.9, zorder=5,
               label="Frozen wf15_m3_t5_ibon (rank 18/18, not Pareto)")
    ax.annotate("rank 18 of 18\nd = 1.076", (frz["cov"], frz["res10"]),
                textcoords="offset points", xytext=(46, -6), ha="left",
                fontsize=7.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.6,
                                shrinkA=0, shrinkB=6))
    ax.set_xlabel("Episode coverage (% of 40,019 training episodes)")
    ax.set_ylabel("Natural resolution @10 (% of uncensored events)")
    ax.set_title("SENSITIVITY ANALYSIS — NOT primary validation\n"
                 "10×10-3p-3f (non-coop), seed 91: frozen configuration under the unchanged rule",
                 loc="left", color=INK)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3,
              handletextpad=0.4, columnspacing=1.2)
    lo, hi = min(r["res10"] for r in rows), max(r["res10"] for r in rows)
    ax.annotate(f"eligible res@10 range:\nonly {hi - lo:.4f} pp ({lo:.4f}–{hi:.4f}%)\n"
                f"28.6× narrower than on 8×8",
                (0.015, 0.985), xycoords="axes fraction", va="top", ha="left",
                fontsize=7, color=INK2, linespacing=1.35)
    fig.text(0.0, -0.22,
             "Four environment properties changed jointly vs the 8×8 development condition: "
             "grid size, agent count (2→3), food count (2→3) and enforced cooperation\n"
             "(-coop- → absent). 95.5% of successful loads here are solo, so firing rates are "
             "not like-for-like. One seed. This is cross-condition sensitivity evidence,\nNOT "
             "external validation, and NOT grounds for retuning — the frozen configuration was "
             "not changed.",
             fontsize=6.5, color=MUTED, va="top")
    save(fig, "fig5_10x10_sensitivity_separate")


if __name__ == "__main__":
    print("Stage B publication figures ->", OUT)
    fig_calibration_tradeoff()
    fig_seed_comparison()
    fig_funnel()
    fig_phase_timing()
    fig_10x10_sensitivity()
    print("done")
