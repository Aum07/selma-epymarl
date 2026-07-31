#!/usr/bin/env python3
"""SELMA: analysis of the five-seed 10M QMIX baseline on Foraging-10x10-3p-3f-v3.

Read-only with respect to the baseline outputs. Everything it writes goes to
analysis/out/.

CSV schema (verified against src/runners/episode_runner.py @ 8a169f1 and
src/envs/gymma.py @ 8a169f1 -- do not infer these from column position):

  t_env                 train rows: self.t_env + self.t, i.e. the cumulative
                        number of *training* environment steps AFTER this
                        episode. test rows: self.t_env unchanged, i.e. the
                        training-step count at which the greedy evaluation
                        block was run (shared by all 100 episodes of a block;
                        greedy-test steps never advance t_env).
  test_mode             0 = training episode, 1 = greedy-test episode
                        (test_greedy=True, evaluation_epsilon=0.0).
  ep_length             number of env.step calls in the episode (<= 50).
  team_return           sum over the episode of the scalarised common reward
                        (reward_scalarisation='sum', common_reward=True).
                        Raw env reward; reward standardisation is applied
                        inside the learner only and does not touch this.
  multi_agent_reward_steps   number of transitions in the episode on which
                        AT LEAST TWO agents received a strictly positive
                        per-agent reward. That is the whole operational
                        meaning -- it is not by itself proof of intent.
  single_agent_reward_steps  transitions on which EXACTLY ONE agent received
                        a strictly positive per-agent reward.
  agent{i}_return       per-agent reward summed over the episode (pre
                        scalarisation). sum_i agent{i}_return == team_return
                        is asserted by the runner at write time.
  successful_load_events        food cells cleared during the episode.
  solo_load_events              cleared cells with <=1 adjacent LOADing agent.
  multi_agent_load_events       cleared cells with >=2 adjacent LOADing agents.
  necessary_cooperative_load_events  subset of the above where the food level
                        strictly exceeded EVERY participant's own level, so no
                        participant could have cleared it alone.
  failed_load_events    surviving food cells with >=1 adjacent LOADing agent.
  failed_load_attempts  sum of adjacent LOADing agents over those cells.

Usage:
    python3 analysis/analyse_10m_baseline.py [--no-plots]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
SELMA = os.path.expanduser("~/selma")

SEEDS = [1, 2, 3, 4, 5]
CSV_10M = os.path.join(SELMA, "long_10x10_seed{seed}_10M.csv")
# Exactly-matched standalone 2M baseline (same env / algorithm / hyperparameters;
# see the audit report for the revision caveat).
CSV_2M = os.path.join(SELMA, "final_10x10_seed{seed}_2M.csv")

T_MAX = 10_000_000
FINAL_WINDOW = 1_000_000          # final-performance window: last 1M env steps
WINDOWS = [(0, 2_000_000), (2_000_000, 5_000_000),
           (5_000_000, 8_000_000), (8_000_000, 10_000_000)]
EARLY_WIN = (0, 2_000_000)
LATE_WIN = (8_000_000, 10_000_000)
T_CRIT_95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}

AGENT_COLS = ["agent0_return", "agent1_return", "agent2_return"]
LOAD_COLS = ["successful_load_events", "solo_load_events",
             "multi_agent_load_events", "necessary_cooperative_load_events",
             "failed_load_events", "failed_load_attempts"]

# dataviz: fixed categorical order, assigned by seed identity and never cycled.
SEED_COLORS = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a",
               4: "#eda100", 5: "#e87ba4"}
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#d8d7d2"
MEAN_C = "#4a3aa7"


# ----------------------------------------------------------------- statistics
def seed_level_stats(values):
    """Aggregate across seeds. The seed is the unit of replication -- never the
    individual episode."""
    a = np.asarray(values, dtype=float)
    n = len(a)
    mean = float(a.mean())
    sd = float(a.std(ddof=1)) if n > 1 else float("nan")
    sem = sd / np.sqrt(n) if n > 1 else float("nan")
    t = T_CRIT_95.get(n, 1.96)
    return {
        "n_seeds": n,
        "mean": mean,
        "sd_across_seeds": sd,
        "sem": sem,
        "ci95_lo": mean - t * sem if n > 1 else float("nan"),
        "ci95_hi": mean + t * sem if n > 1 else float("nan"),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def paired_diff_stats(early, late):
    """Within-seed paired differences, aggregated across seeds."""
    d = np.asarray(late, dtype=float) - np.asarray(early, dtype=float)
    s = seed_level_stats(d)
    s["per_seed_diff"] = [float(x) for x in d]
    s["n_seeds_improved"] = int((d > 0).sum())
    # Cohen's d_z for the paired design (practical effect size, 5 seeds).
    sd = s["sd_across_seeds"]
    s["cohens_dz"] = float(s["mean"] / sd) if sd and sd > 0 else float("nan")
    try:
        from scipy import stats as _st
        t, p = _st.ttest_rel(np.asarray(late, dtype=float),
                             np.asarray(early, dtype=float))
        s["paired_t"] = float(t)
        s["paired_p"] = float(p)
        w = _st.wilcoxon(np.asarray(late, dtype=float),
                         np.asarray(early, dtype=float))
        s["wilcoxon_p"] = float(w.pvalue)
    except Exception:
        pass
    return s


# ----------------------------------------------------------------------- load
def load_csv(path):
    df = pd.read_csv(path)
    df["test_mode"] = df["test_mode"].astype(int)
    return df


def integrity_checks(df, seed, t_max):
    """Row-level and counter-level consistency. Nothing here is cosmetic: each
    check corresponds to an invariant asserted or implied by the logging code."""
    tr = df[df.test_mode == 0]
    te = df[df.test_mode == 1]
    chk = {}
    chk["n_rows"] = int(len(df))
    chk["n_train_rows"] = int(len(tr))
    chk["n_test_rows"] = int(len(te))
    chk["nulls"] = int(df.isna().sum().sum())
    chk["exact_duplicate_rows"] = int(df.duplicated().sum())
    chk["train_t_env_strictly_increasing"] = bool(tr.t_env.is_monotonic_increasing
                                                  and tr.t_env.is_unique)
    chk["train_max_t_env"] = int(tr.t_env.max())
    chk["reached_t_max"] = bool(tr.t_env.max() >= t_max)
    chk["train_t_env_step_equals_ep_length"] = bool(
        np.array_equal(np.diff(tr.t_env.to_numpy()), tr.ep_length.to_numpy()[1:])
    )
    chk["ep_length_min"] = int(df.ep_length.min())
    chk["ep_length_max"] = int(df.ep_length.max())
    # team_return == sum of per-agent returns (runner asserts atol=1e-8)
    resid = (df.team_return - df[AGENT_COLS].sum(axis=1)).abs()
    chk["max_team_vs_agent_sum_residual"] = float(resid.max())
    chk["team_equals_agent_sum"] = bool(resid.max() < 1e-8)
    # rewarded steps cannot exceed the number of transitions
    chk["rewarded_steps_le_ep_length"] = bool(
        ((df.multi_agent_reward_steps + df.single_agent_reward_steps)
         <= df.ep_length).all()
    )
    # load-event partition
    chk["successful_eq_solo_plus_multi"] = bool(
        (df.successful_load_events
         == df.solo_load_events + df.multi_agent_load_events).all()
    )
    chk["necessary_le_multi_load"] = bool(
        (df.necessary_cooperative_load_events <= df.multi_agent_load_events).all()
    )
    chk["successful_le_initial_food_3"] = bool((df.successful_load_events <= 3).all())
    # a multi-agent load in an episode implies a multi-agent rewarded step
    chk["multi_load_implies_multi_reward_step"] = bool(
        (df.loc[df.multi_agent_load_events > 0, "multi_agent_reward_steps"] > 0).all()
    )
    # a positive team return implies at least one rewarded step
    chk["positive_return_implies_rewarded_step"] = bool(
        (df.loc[df.team_return > 0,
                ["multi_agent_reward_steps", "single_agent_reward_steps"]]
         .sum(axis=1) > 0).all()
    )
    # greedy-test blocks are exactly test_nepisode episodes at one t_env
    blk = te.groupby("t_env").size()
    chk["n_test_blocks"] = int(len(blk))
    chk["test_block_sizes_unique"] = sorted(set(int(x) for x in blk.unique()))
    chk["all_test_blocks_are_100"] = bool((blk == 100).all())
    chk["all_checks_pass"] = all(
        v is True for k, v in chk.items()
        if isinstance(v, bool) and k != "reached_t_max"
    ) and chk["reached_t_max"]
    return chk


# --------------------------------------------------------------- aggregations
def block_summary(sub):
    """Summarise a set of episode rows (one split, one window, one seed)."""
    n = len(sub)
    if n == 0:
        return {k: float("nan") for k in
                ["n_episodes", "env_steps", "return_mean", "return_median",
                 "return_std_episodes", "ep_length_mean",
                 "single_rew_steps_total", "multi_rew_steps_total",
                 "rewarded_steps_total", "multi_frac_of_rewarded",
                 "rew_steps_per_M", "multi_rew_steps_per_M",
                 "frac_episodes_with_multi", "n_ep_multi_0", "n_ep_multi_1",
                 "n_ep_multi_ge2", "successful_load_events_total",
                 "solo_load_events_total", "multi_load_events_total",
                 "necessary_coop_load_events_total",
                 "failed_load_events_total", "failed_load_attempts_total"]}
    steps = float(sub.ep_length.sum())
    single = float(sub.single_agent_reward_steps.sum())
    multi = float(sub.multi_agent_reward_steps.sum())
    rew = single + multi
    mac = sub.multi_agent_reward_steps.to_numpy()
    d = {
        "n_episodes": int(n),
        "env_steps": steps,
        "return_mean": float(sub.team_return.mean()),
        "return_median": float(sub.team_return.median()),
        "return_std_episodes": float(sub.team_return.std(ddof=1)) if n > 1 else float("nan"),
        "ep_length_mean": float(sub.ep_length.mean()),
        "single_rew_steps_total": single,
        "multi_rew_steps_total": multi,
        "rewarded_steps_total": rew,
        # ratio is undefined when nothing was rewarded at all
        "multi_frac_of_rewarded": float(multi / rew) if rew > 0 else float("nan"),
        "rew_steps_per_M": float(rew / steps * 1e6),
        "multi_rew_steps_per_M": float(multi / steps * 1e6),
        "frac_episodes_with_multi": float((mac > 0).mean()),
        "n_ep_multi_0": int((mac == 0).sum()),
        "n_ep_multi_1": int((mac == 1).sum()),
        "n_ep_multi_ge2": int((mac >= 2).sum()),
    }
    for c in LOAD_COLS:
        key = {"successful_load_events": "successful_load_events_total",
               "solo_load_events": "solo_load_events_total",
               "multi_agent_load_events": "multi_load_events_total",
               "necessary_cooperative_load_events": "necessary_coop_load_events_total",
               "failed_load_events": "failed_load_events_total",
               "failed_load_attempts": "failed_load_attempts_total"}[c]
        d[key] = float(sub[c].sum())
    tot_agent = sub[AGENT_COLS].sum()
    grand = float(tot_agent.sum())
    for i, c in enumerate(AGENT_COLS):
        d[f"agent{i}_return_total"] = float(tot_agent[c])
        d[f"agent{i}_return_share"] = float(tot_agent[c] / grand) if grand > 0 else float("nan")
    # per-agent share dispersion: 0 == perfectly even credit across agents
    if grand > 0:
        shares = (tot_agent / grand).to_numpy()
        d["agent_share_max_minus_min"] = float(shares.max() - shares.min())
        d["agent_share_gini"] = float(
            np.abs(shares[:, None] - shares[None, :]).sum()
            / (2 * len(shares) ** 2 * shares.mean())
        )
    else:
        d["agent_share_max_minus_min"] = float("nan")
        d["agent_share_gini"] = float("nan")
    return d


def window_rows(df, lo, hi):
    return df[(df.t_env > lo) & (df.t_env <= hi)] if lo > 0 else \
        df[(df.t_env >= lo) & (df.t_env <= hi)]


def curve(df, bin_width, value_fn, t_max=T_MAX):
    """Bin episodes by t_env and reduce -- used for all learning curves.
    Returns (bin_centre, value, n_episodes_in_bin)."""
    edges = np.arange(0, t_max + bin_width, bin_width)
    idx = np.clip(np.digitize(df.t_env.to_numpy(), edges) - 1, 0, len(edges) - 2)
    out_x, out_y, out_n = [], [], []
    for b in range(len(edges) - 1):
        sel = df[idx == b]
        if len(sel) == 0:
            continue
        out_x.append((edges[b] + edges[b + 1]) / 2.0)
        out_y.append(value_fn(sel))
        out_n.append(len(sel))
    return np.array(out_x), np.array(out_y), np.array(out_n)


# ---------------------------------------------------------------------- plots
def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 140, "savefig.dpi": 140,
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": GRID, "grid.linewidth": 0.6,
        "legend.frameon": False, "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
    })
    return plt


def _finish(ax, title, subtitle, xlabel, ylabel):
    import textwrap
    sub = "\n".join(textwrap.wrap(" ".join(subtitle.split()), 96))
    pad = 12 + 9 * sub.count("\n")
    ax.set_title(title, loc="left", fontweight="bold", pad=pad)
    ax.text(0, 1.012, sub, transform=ax.transAxes, fontsize=7.2,
            color=INK2, va="bottom", linespacing=1.4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.15, 10.75)


def plot_per_seed(plt, curves, title, subtitle, ylabel, path, mean_band=True):
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    xs_all = []
    for seed in SEEDS:
        x, y = curves[seed]
        xs_all.append(x)
        ax.plot(x / 1e6, y, lw=1.4, color=SEED_COLORS[seed], alpha=0.9,
                label=f"seed {seed}")
        # direct label at the right edge (relief rule: several slots sit below
        # 3:1 on the light surface, so identity is never colour-alone)
        ax.annotate(f"s{seed}", (x[-1] / 1e6, y[-1]), xytext=(4, 0),
                    textcoords="offset points", fontsize=7, color=INK2,
                    va="center")
    if mean_band:
        grid = curves[SEEDS[0]][0]
        stack = np.vstack([np.interp(grid, curves[s][0], curves[s][1])
                           for s in SEEDS])
        m = stack.mean(axis=0)
        sd = stack.std(axis=0, ddof=1)
        sem = sd / np.sqrt(len(SEEDS))
        ax.plot(grid / 1e6, m, lw=2.2, color=MEAN_C, label="cross-seed mean")
        ax.fill_between(grid / 1e6, m - T_CRIT_95[5] * sem, m + T_CRIT_95[5] * sem,
                        color=MEAN_C, alpha=0.16, lw=0,
                        label="95% CI of the mean (5 seeds, t$_4$)")
    _finish(ax, title, subtitle, "environment steps (millions)", ylabel)
    ax.legend(loc="upper left", fontsize=7.5, ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    data = {}
    integrity = {}
    for s in SEEDS:
        df = load_csv(CSV_10M.format(seed=s))
        data[s] = df
        integrity[s] = integrity_checks(df, s, T_MAX)
        print(f"seed {s}: {len(df)} rows, integrity "
              f"{'OK' if integrity[s]['all_checks_pass'] else 'FAIL'}")

    # manifest of run coverage / elapsed time
    try:
        man = json.load(open(os.path.join(OUT, "baseline_manifest_10M.json")))
    except Exception:
        man = {"seeds": {}}

    # ------------------------------------------------ per-seed summary table
    rows = []
    for s in SEEDS:
        df = data[s]
        tr, te = df[df.test_mode == 0], df[df.test_mode == 1]
        rec = {"seed": s}
        me = man.get("seeds", {}).get(str(s), {})
        rec["elapsed_seconds"] = (me.get("sacred") or {}).get("elapsed_seconds")
        rec["elapsed_hours"] = (rec["elapsed_seconds"] / 3600.0
                                if rec["elapsed_seconds"] else None)
        rec["max_t_env"] = int(tr.t_env.max())
        rec["completion_markers"] = (me.get("log") or {}).get(
            "completion_marker_count")
        rec["crash_free"] = (me.get("log") or {}).get("crash_free")
        rec["integrity_pass"] = integrity[s]["all_checks_pass"]
        rec["malformed_or_duplicate_rows"] = (
            integrity[s]["exact_duplicate_rows"] + integrity[s]["nulls"])
        for split, sub in (("train", tr), ("test", te)):
            for k, v in block_summary(sub).items():
                rec[f"{split}_{k}"] = v
            fin = sub[sub.t_env > T_MAX - FINAL_WINDOW]
            for k, v in block_summary(fin).items():
                rec[f"{split}_final1M_{k}"] = v
            for lo, hi in WINDOWS:
                w = window_rows(sub, lo, hi)
                tag = f"{split}_w{lo//1000000}_{hi//1000000}M"
                bs = block_summary(w)
                for k in ("return_mean", "return_median", "ep_length_mean",
                          "multi_rew_steps_per_M", "rew_steps_per_M",
                          "multi_frac_of_rewarded", "frac_episodes_with_multi",
                          "n_episodes", "multi_rew_steps_total",
                          "single_rew_steps_total",
                          "necessary_coop_load_events_total",
                          "successful_load_events_total"):
                    rec[f"{tag}_{k}"] = bs[k]
        rows.append(rec)
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(os.path.join(OUT, "per_seed_summary_10M.csv"), index=False)

    # ------------------------------------------------- cross-seed aggregate
    agg_rows = []
    skip = {"seed", "crash_free", "integrity_pass", "completion_markers"}
    for col in per_seed.columns:
        if col in skip:
            continue
        v = pd.to_numeric(per_seed[col], errors="coerce")
        if v.isna().all():
            continue
        st = seed_level_stats(v.to_numpy())
        st["metric"] = col
        st["per_seed"] = [float(x) for x in v.to_numpy()]
        agg_rows.append(st)
    agg = pd.DataFrame(agg_rows)[
        ["metric", "n_seeds", "mean", "sd_across_seeds", "sem",
         "ci95_lo", "ci95_hi", "min", "max", "per_seed"]]
    agg.to_csv(os.path.join(OUT, "cross_seed_summary_10M.csv"), index=False)

    # ------------------------------------------ early vs late paired analysis
    paired = {}
    for split in ("train", "test"):
        for metric in ("return_mean", "return_median", "ep_length_mean",
                       "multi_rew_steps_per_M", "rew_steps_per_M",
                       "multi_frac_of_rewarded", "frac_episodes_with_multi",
                       "necessary_coop_load_events_total",
                       "successful_load_events_total"):
            e = per_seed[f"{split}_w0_2M_{metric}"].to_numpy(dtype=float)
            l = per_seed[f"{split}_w8_10M_{metric}"].to_numpy(dtype=float)
            paired[f"{split}:{metric}"] = paired_diff_stats(e, l)
            paired[f"{split}:{metric}"]["early_per_seed"] = [float(x) for x in e]
            paired[f"{split}:{metric}"]["late_per_seed"] = [float(x) for x in l]

    # ------------------------------------------- standalone 2M vs 10M compare
    twoM = {}
    have_2m = all(os.path.exists(CSV_2M.format(seed=s)) for s in SEEDS)
    if have_2m:
        for s in SEEDS:
            d2 = load_csv(CSV_2M.format(seed=s))
            twoM[s] = {
                "train": block_summary(d2[d2.test_mode == 0]),
                "test": block_summary(d2[d2.test_mode == 1]),
                "train_final1M": block_summary(
                    d2[(d2.test_mode == 0) & (d2.t_env > 1_000_000)]),
                "test_final1M": block_summary(
                    d2[(d2.test_mode == 1) & (d2.t_env > 1_000_000)]),
            }

    comparison = {}
    for split in ("train", "test"):
        for metric in ("return_mean", "multi_rew_steps_per_M",
                       "multi_frac_of_rewarded", "frac_episodes_with_multi",
                       "rew_steps_per_M"):
            within_e = per_seed[f"{split}_w0_2M_{metric}"].to_numpy(dtype=float)
            within_l = per_seed[f"{split}_w8_10M_{metric}"].to_numpy(dtype=float)
            entry = {
                "within_run_first2M": [float(x) for x in within_e],
                "within_run_last2M": [float(x) for x in within_l],
                "within_run_paired": paired_diff_stats(within_e, within_l),
            }
            if have_2m:
                st2 = np.array([twoM[s][split][metric] for s in SEEDS], float)
                fin10 = per_seed[f"{split}_final1M_{metric}"].to_numpy(dtype=float)
                entry["standalone_2M_whole_run"] = [float(x) for x in st2]
                entry["standalone_2M_final1M"] = [
                    float(twoM[s][f"{split}_final1M"][metric]) for s in SEEDS]
                entry["run10M_final1M"] = [float(x) for x in fin10]
                entry["standalone2M_vs_10Mfinal_paired"] = paired_diff_stats(
                    np.array(entry["standalone_2M_final1M"], float), fin10)
            comparison[f"{split}:{metric}"] = entry

    report = {
        "integrity": integrity,
        "paired_early_vs_late": paired,
        "two_million_comparison": comparison,
        "standalone_2M_available": have_2m,
        "definitions": {
            "final_window": f"episodes with t_env > {T_MAX - FINAL_WINDOW} "
                            f"(last {FINAL_WINDOW/1e6:.0f}M environment steps)",
            "early_window": "0-2M t_env", "late_window": "8-10M t_env",
            "windows": [[a, b] for a, b in WINDOWS],
            "unit_of_replication": "seed (n=5); 95% CI uses t_4 = 2.776",
            "multi_agent_reward_steps": "transitions on which at least two "
                                        "agents received a strictly positive "
                                        "per-agent reward",
        },
    }
    with open(os.path.join(OUT, "analysis_10M.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=float)

    # ------------------------------------------------------------ plot block
    if not args.no_plots:
        plt = setup_mpl()
        BIN_TRAIN = 200_000     # 200k-step bins -> ~4000 training episodes/bin
        note_tr = (f"Each point = mean over all training episodes in a "
                   f"{BIN_TRAIN//1000}k-step bin (~4k episodes); no additional "
                   f"smoothing.")
        note_te = ("Each point = mean over one 100-episode greedy evaluation "
                   "block (test_greedy, epsilon=0); no additional smoothing.")

        # training return
        c = {s: curve(data[s][data[s].test_mode == 0], BIN_TRAIN,
                      lambda d: d.team_return.mean())[:2] for s in SEEDS}
        plot_per_seed(plt, c, "QMIX training return, Foraging-10x10-3p-3f-v3",
                      note_tr + "  Behaviour policy: epsilon-greedy (eps=0.05 "
                      "after 50k steps).",
                      "team return per episode",
                      os.path.join(OUT, "fig1_train_return.png"))

        # greedy test return
        ct = {s: curve(data[s][data[s].test_mode == 1], 50_000,
                       lambda d: d.team_return.mean())[:2] for s in SEEDS}
        plot_per_seed(plt, ct, "QMIX greedy-evaluation return, "
                      "Foraging-10x10-3p-3f-v3", note_te,
                      "team return per greedy episode",
                      os.path.join(OUT, "fig2_test_return.png"))

        # rewarded-step trends: single vs multi, training
        for tag, split, binw, note in (("train", 0, BIN_TRAIN, note_tr),
                                       ("test", 1, 250_000, note_te)):
            fig, ax = plt.subplots(figsize=(7.2, 4.0))
            for s in SEEDS:
                d = data[s][data[s].test_mode == split]
                x, y, _ = curve(d, binw,
                                lambda z: z.single_agent_reward_steps.sum()
                                / z.ep_length.sum() * 1e6)
                ax.plot(x / 1e6, y, lw=1.2, color=SEED_COLORS[s], alpha=0.75)
                x2, y2, _ = curve(d, binw,
                                  lambda z: z.multi_agent_reward_steps.sum()
                                  / z.ep_length.sum() * 1e6)
                ax.plot(x2 / 1e6, y2, lw=1.2, ls="--", color=SEED_COLORS[s],
                        alpha=0.75)
            ax.plot([], [], color=INK2, lw=1.2, label="single-agent rewarded steps")
            ax.plot([], [], color=INK2, lw=1.2, ls="--",
                    label="multi-agent rewarded steps")
            for s in SEEDS:
                ax.plot([], [], color=SEED_COLORS[s], lw=1.2, label=f"seed {s}")
            _finish(ax, f"Rewarded-step rate ({tag}), Foraging-10x10-3p-3f-v3",
                    note + "  Counts are exact per-bin sums, not smoothed; "
                    "rare events are preserved.",
                    "environment steps (millions)",
                    "rewarded steps per million episode steps")
            ax.legend(fontsize=7.5, ncol=3, loc="upper left")
            fig.tight_layout()
            fig.savefig(os.path.join(OUT, f"fig3_{tag}_rewarded_steps.png"))
            plt.close(fig)

        # proportion of rewarded steps that are multi-agent (denominator > 0)
        for tag, split, binw, note in (("train", 0, BIN_TRAIN, note_tr),
                                       ("test", 1, 250_000, note_te)):
            fig, ax = plt.subplots(figsize=(7.2, 4.0))
            for s in SEEDS:
                d = data[s][data[s].test_mode == split]

                def f(z):
                    den = (z.single_agent_reward_steps.sum()
                           + z.multi_agent_reward_steps.sum())
                    return (z.multi_agent_reward_steps.sum() / den
                            if den > 0 else np.nan)
                x, y, _ = curve(d, binw, f)
                ax.plot(x / 1e6, y, lw=1.4, color=SEED_COLORS[s],
                        label=f"seed {s}")
                ax.annotate(f"s{s}", (x[-1] / 1e6, y[-1]), xytext=(4, 0),
                            textcoords="offset points", fontsize=7, color=INK2,
                            va="center")
            _finish(ax, f"Share of rewarded steps that are multi-agent ({tag})",
                    note + "  Bins with zero rewarded steps are dropped "
                    "(ratio undefined), not imputed as 0.",
                    "environment steps (millions)",
                    "multi-agent / all rewarded steps")
            ax.legend(fontsize=7.5, ncol=3, loc="upper left")
            fig.tight_layout()
            fig.savefig(os.path.join(OUT, f"fig4_{tag}_multi_share.png"))
            plt.close(fig)

        # episode length
        c = {s: curve(data[s][data[s].test_mode == 0], BIN_TRAIN,
                      lambda d: d.ep_length.mean())[:2] for s in SEEDS}
        plot_per_seed(plt, c, "Mean training episode length",
                      note_tr + "  Time limit = 50 steps; shorter means food "
                      "was cleared earlier.", "steps per episode",
                      os.path.join(OUT, "fig5_train_ep_length.png"))

        # fraction of episodes with >=1 multi-agent rewarded step
        for tag, split, binw, note in (("train", 0, BIN_TRAIN, note_tr),
                                       ("test", 1, 250_000, note_te)):
            fig, ax = plt.subplots(figsize=(7.2, 4.0))
            for s in SEEDS:
                d = data[s][data[s].test_mode == split]
                x, y, _ = curve(d, binw,
                                lambda z: float(
                                    (z.multi_agent_reward_steps > 0).mean()))
                ax.plot(x / 1e6, y, lw=1.4, color=SEED_COLORS[s],
                        label=f"seed {s}")
                ax.annotate(f"s{s}", (x[-1] / 1e6, y[-1]), xytext=(4, 0),
                            textcoords="offset points", fontsize=7, color=INK2,
                            va="center")
            _finish(ax, f"Episodes with >=1 multi-agent rewarded step ({tag})",
                    note, "environment steps (millions)",
                    "proportion of episodes")
            ax.legend(fontsize=7.5, ncol=3, loc="upper left")
            fig.tight_layout()
            fig.savefig(os.path.join(OUT, f"fig6_{tag}_frac_ep_multi.png"))
            plt.close(fig)

        print("figures written to", OUT)

    print("wrote per_seed_summary_10M.csv, cross_seed_summary_10M.csv, "
          "analysis_10M.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
