#!/usr/bin/env python3
"""
compare_horizons.py  —  HorizonBreak multi-H comparison plotter
================================================================
Generates comparison graphs across all H= variations for a given
task, optionally filtered to one architecture.

Usage:
    python compare_horizons.py <root_dir> --task <task> [options]

    <root_dir> is scanned for subdirectories whose names contain
    --task and match *_H<N>* (e.g. tdmpc2_dmc_walker_walk_H3_seed0).
    Runs are grouped by (arch, H) and plotted together.

Options:
    --task   walker_walk      Task substring to match (required)
    --arch   dreamer|tdmpc2   Restrict to one architecture (default: all)
    --out    <dir>            Output root (default: <root_dir>/compare_plots/)
                              Plots land in <out>/<task>/<arch>/
    --show                    Open plots interactively after saving
    --smooth <int>            Rolling-window size for train/gap curves (default: 5)

Examples:
    python compare_horizons.py ../logdir/ --task walker_walk
    python compare_horizons.py ../logdir/ --task walker_walk --arch tdmpc2
    python compare_horizons.py ../logdir/ --task cheetah_run --out ../plots/compare
"""

import argparse
import re
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── shared helpers from plot_metrics.py ──────────────────────────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
from plot_metrics import (
    detect_format, load_dreamer, load_tdmpc2, steps_values, k_fmt, save_fig
)

# ── colour palette ────────────────────────────────────────────────────────────
# Up to 8 distinct (arch, H) combinations; colourblind-safe
LINE_COLORS = [
    "#E24A33",  # red
    "#348ABD",  # blue
    "#988ED5",  # purple
    "#8EBA42",  # green
    "#FBC15E",  # amber
    "#777777",  # grey
    "#56B4E9",  # sky blue
    "#D55E00",  # vermillion
]

ARCH_DASH = {"dreamer": "-", "tdmpc2": "--"}   # solid=dreamer, dashed=tdmpc2

# ── helpers ───────────────────────────────────────────────────────────────────

def rolling_mean(arr, window):
    if window <= 1 or len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    padded = np.pad(arr, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def make_label(arch, h_val, mean_final=None, multi_arch=True):
    """
    When multiple arches are plotted together include arch name.
    Otherwise just H= value.
    """
    base = f"{arch} H={h_val}" if multi_arch else f"H={h_val}"
    if mean_final is not None:
        base += f"  ({mean_final:.0f})"
    return base


def scan_runs(root: Path, task: str, arch_filter: str | None):
    """
    Scan root for run dirs containing task substring and _H<N>.
    Returns { (arch, h_val): [(run_name, series), ...] }
    """
    pattern = re.compile(r"_H(\d+)", re.IGNORECASE)
    grouped = defaultdict(list)

    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        name = d.name

        if task.lower() not in name.lower():
            continue

        match = pattern.search(name)
        if not match:
            continue
        h_val = int(match.group(1))

        try:
            fmt = detect_format(d)
        except SystemExit:
            continue

        if arch_filter and fmt != arch_filter.lower():
            continue

        try:
            series = load_dreamer(d) if fmt == "dreamer" else load_tdmpc2(d)
        except Exception as e:
            print(f"  [warn] Could not load {d.name}: {e}")
            continue

        grouped[(fmt, h_val)].append((name, series))
        print(f"  loaded  arch={fmt:8s}  H={h_val:3d}  {name}")

    return dict(grouped)


def _interp_mean(grouped_key, series_key, smooth=1):
    """
    Given a list of (run_name, series) for one (arch, H) group,
    interpolate all runs onto a common step grid and return
    (common_steps, mean, std).  Returns (None, None, None) if no data.
    """
    all_steps, all_vals = [], []
    for _, series in grouped_key:
        ss, vv = steps_values(series, series_key)
        if ss is None:
            continue
        all_steps.append(ss)
        all_vals.append(rolling_mean(vv, smooth))

    if not all_steps:
        return None, None, None

    common = np.union1d(*all_steps) if len(all_steps) > 1 else all_steps[0]
    interped = np.array([np.interp(common, s, v) for s, v in zip(all_steps, all_vals)])
    return common, interped.mean(axis=0), interped.std(axis=0)


def _bin_train(grouped_key, smooth=1, n_bins=200):
    """
    Bin-then-smooth train episode data for a single (arch, H) group.
    Each run is binned independently into n_bins equal-width bins,
    then runs are averaged across bins.  Returns (bin_centers, mean, std)
    where std is across bins (within-run smoothed variance), not across
    zero-width per-point spikes.
    """
    per_run_means, per_run_stds, bin_centers_ref = [], [], None

    for _, series in grouped_key:
        ss, vv = steps_values(series, "episode/score")
        if ss is None:
            continue

        max_step = int(ss.max())
        bin_edges = np.linspace(0, max_step, n_bins + 1)
        centers   = (bin_edges[:-1] + bin_edges[1:]) / 2

        b_means, b_stds, valid_c = [], [], []
        for lo, hi, c in zip(bin_edges[:-1], bin_edges[1:], centers):
            mask = (ss >= lo) & (ss < hi)
            if mask.sum() == 0:
                continue
            b_means.append(np.mean(vv[mask]))
            b_stds.append(np.std(vv[mask]))
            valid_c.append(c)

        if not valid_c:
            continue

        valid_c  = np.array(valid_c)
        b_means  = rolling_mean(np.array(b_means), smooth)
        b_stds   = rolling_mean(np.array(b_stds),  smooth)

        per_run_means.append((valid_c, b_means))
        per_run_stds.append((valid_c, b_stds))
        if bin_centers_ref is None:
            bin_centers_ref = valid_c

    if not per_run_means:
        return None, None, None

    # interpolate all runs onto the first run's grid
    ref = bin_centers_ref
    mean_mat = np.array([np.interp(ref, c, m) for c, m in per_run_means])
    std_mat  = np.array([np.interp(ref, c, s) for c, s in per_run_stds])

    # combined: mean-of-means, mean-of-stds (within-run variance)
    return ref, mean_mat.mean(axis=0), std_mat.mean(axis=0)


def _gap_series(runs, smooth=1):
    """Derive imagination gap for a list of (run_name, series)."""
    all_steps, all_vals = [], []
    for _, series in runs:
        gs, gv = steps_values(series, "train/imagination_gap")
        if gs is None:
            ir, iv = steps_values(series, "train/tar")
            er, ev = steps_values(series, "train/ret_replay_mean")
            if ir is not None and er is not None:
                ev_i = np.interp(ir, er, ev)
                gs, gv = ir, iv - ev_i
        if gs is None:
            continue
        all_steps.append(gs)
        all_vals.append(rolling_mean(gv, smooth))

    if not all_steps:
        return None, None, None
    common = np.union1d(*all_steps) if len(all_steps) > 1 else all_steps[0]
    interped = np.array([np.interp(common, s, v) for s, v in zip(all_steps, all_vals)])
    return common, interped.mean(axis=0), interped.std(axis=0)


# ── shared plotter core ───────────────────────────────────────────────────────

def _plot_series(ax, grouped, series_key, smooth, multi_arch, label_suffix_fn=None):
    """
    Plot one (arch, H) group per line.  Returns True if anything was plotted.
    """
    arches = sorted({a for a, _ in grouped})
    keys   = sorted(grouped.keys(), key=lambda x: (x[0], x[1]))
    plotted = False

    for i, (arch, h_val) in enumerate(keys):
        color = LINE_COLORS[i % len(LINE_COLORS)]
        ls    = ARCH_DASH.get(arch, "-")
        cs, mv, sv = _interp_mean(grouped[(arch, h_val)], series_key, smooth)
        if cs is None:
            continue
        lbl = make_label(arch, h_val, mv[-1], multi_arch=(len(arches) > 1))
        ax.fill_between(cs, mv - sv, mv + sv, color=color, alpha=0.15)
        ax.plot(cs, mv, color=color, lw=2, linestyle=ls, label=lbl)
        plotted = True

    return plotted


# ── plot A: eval curves ───────────────────────────────────────────────────────

def plot_eval_comparison(grouped, out: Path, task: str, arch_label: str, show: bool):
    fig, ax = plt.subplots(figsize=(9, 5))
    arches = sorted({a for a, _ in grouped})
    keys   = sorted(grouped.keys(), key=lambda x: (x[0], x[1]))
    plotted = False

    for i, (arch, h_val) in enumerate(keys):
        color = LINE_COLORS[i % len(LINE_COLORS)]
        all_steps, all_vals = [], []
        for _, series in grouped[(arch, h_val)]:
            ss, vv = steps_values(series, "episode/eval_score")
            if ss is None:
                continue
            ax.plot(ss, vv, color=color, lw=1.0, alpha=0.35)
            all_steps.append(ss)
            all_vals.append(vv)
        if not all_steps:
            continue
        common = np.union1d(*all_steps) if len(all_steps) > 1 else all_steps[0]
        interped = np.array([np.interp(common, s, v) for s, v in zip(all_steps, all_vals)])
        mv = interped.mean(axis=0)
        lbl = make_label(arch, h_val, mv[-1], multi_arch=(len(arches) > 1))
        ls  = ARCH_DASH.get(arch, "-")
        ax.plot(common, mv, color=color, lw=2.5, linestyle=ls, label=lbl)
        plotted = True

    if not plotted:
        print("  [skip] no eval data to compare")
        plt.close(fig); return

    ax.set_xlabel("Env steps", fontsize=12)
    ax.set_ylabel("Episode return", fontsize=12)
    ax.set_title(f"{task}  |  {arch_label}\nEval return", fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(k_fmt))
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=10, framealpha=0.9)
    fig.tight_layout()
    save_fig(fig, out / "eval_comparison.png", show)


# ── plot B: train curves ──────────────────────────────────────────────────────

def plot_train_comparison(grouped, out: Path, task: str, arch_label: str,
                          show: bool, smooth: int = 5):
    fig, ax = plt.subplots(figsize=(9, 5))
    arches = sorted({a for a, _ in grouped})
    keys   = sorted(grouped.keys(), key=lambda x: (x[0], x[1]))
    plotted = False

    for i, (arch, h_val) in enumerate(keys):
        color = LINE_COLORS[i % len(LINE_COLORS)]
        ls    = ARCH_DASH.get(arch, "-")
        cs, mv, sv = _bin_train(grouped[(arch, h_val)], smooth)
        if cs is None:
            continue
        lbl = make_label(arch, h_val, mv[-1], multi_arch=(len(arches) > 1))
        ax.fill_between(cs, mv - sv, mv + sv, color=color, alpha=0.20)
        ax.plot(cs, mv, color=color, lw=2, linestyle=ls, label=lbl)
        plotted = True

    if not plotted:
        print("  [skip] no train data to compare")
        plt.close(fig); return

    ax.set_xlabel("Env steps", fontsize=12)
    ax.set_ylabel("Episode return", fontsize=12)
    ax.set_title(f"{task}  |  {arch_label}\nTrain return", fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(k_fmt))
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=10, framealpha=0.9)
    fig.tight_layout()
    save_fig(fig, out / "train_comparison.png", show)


# ── plot C: imagination gap ───────────────────────────────────────────────────

def plot_gap_comparison(grouped, out: Path, task: str, arch_label: str,
                        show: bool, smooth: int = 5):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axhline(0, color="black", lw=0.9, linestyle="--", zorder=1)
    arches = sorted({a for a, _ in grouped})
    keys   = sorted(grouped.keys(), key=lambda x: (x[0], x[1]))
    plotted = False

    for i, (arch, h_val) in enumerate(keys):
        color = LINE_COLORS[i % len(LINE_COLORS)]
        ls    = ARCH_DASH.get(arch, "-")
        cs, mv, sv = _gap_series(grouped[(arch, h_val)], smooth)
        if cs is None:
            continue
        lbl = make_label(arch, h_val, None, multi_arch=(len(arches) > 1))
        ax.fill_between(cs, mv - sv, mv + sv, color=color, alpha=0.15)
        ax.plot(cs, mv, color=color, lw=2, linestyle=ls, label=lbl)
        plotted = True

    if not plotted:
        print("  [skip] no imagination-gap data to compare")
        plt.close(fig); return

    ax.set_xlabel("Env steps", fontsize=12)
    ax.set_ylabel("Imagined − replay return", fontsize=12)
    ax.set_title(f"{task}  |  {arch_label}\nImagination-reality gap",
                 fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(k_fmt))
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=10, framealpha=0.9)
    fig.tight_layout()
    save_fig(fig, out / "gap_comparison.png", show)


# ── plot D: value prediction error ───────────────────────────────────────────

def plot_vpe_comparison(grouped, out: Path, task: str, arch_label: str,
                        show: bool, smooth: int = 5):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axhline(0, color="black", lw=0.9, linestyle="--", zorder=1)
    arches = sorted({a for a, _ in grouped})
    plotted = _plot_series(ax, grouped, "value_pred_error", smooth,
                           multi_arch=(len(arches) > 1))
    if not plotted:
        print("  [skip] no value_pred_error data to compare")
        plt.close(fig); return

    ax.set_xlabel("Env steps", fontsize=12)
    ax.set_ylabel("Q(s₀,a₀) − actual return", fontsize=12)
    ax.set_title(f"{task}  |  {arch_label}\nValue prediction error",
                 fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(k_fmt))
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=10, framealpha=0.9)
    fig.tight_layout()
    save_fig(fig, out / "vpe_comparison.png", show)


# ── plot E: 2×2 summary panel ────────────────────────────────────────────────

def plot_summary_panel(grouped, out: Path, task: str, arch_label: str,
                       show: bool, smooth: int = 5):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax_eval, ax_train, ax_gap, ax_vpe = axes.flat
    ax_gap.axhline(0, color="black", lw=0.9, linestyle="--", zorder=1)
    ax_vpe.axhline(0, color="black", lw=0.9, linestyle="--", zorder=1)

    arches = sorted({a for a, _ in grouped})
    keys   = sorted(grouped.keys(), key=lambda x: (x[0], x[1]))
    multi  = len(arches) > 1

    for i, (arch, h_val) in enumerate(keys):
        color = LINE_COLORS[i % len(LINE_COLORS)]
        ls    = ARCH_DASH.get(arch, "-")
        lbl   = make_label(arch, h_val, None, multi_arch=multi)

        # eval
        cs, mv, _ = _interp_mean(grouped[(arch, h_val)], "episode/eval_score", 1)
        if cs is not None:
            ax_eval.plot(cs, mv, color=color, lw=2, linestyle=ls,
                         label=make_label(arch, h_val, mv[-1], multi))

        # train (binned to avoid vertical-bar artefacts)
        cs, mv, sv = _bin_train(grouped[(arch, h_val)], smooth)
        if cs is not None:
            ax_train.fill_between(cs, mv - sv, mv + sv, color=color, alpha=0.20)
            ax_train.plot(cs, mv, color=color, lw=2, linestyle=ls, label=lbl)

        # gap
        cs, mv, sv = _gap_series(grouped[(arch, h_val)], smooth)
        if cs is not None:
            ax_gap.fill_between(cs, mv - sv, mv + sv, color=color, alpha=0.15)
            ax_gap.plot(cs, mv, color=color, lw=2, linestyle=ls, label=lbl)

        # vpe
        cs, mv, sv = _interp_mean(grouped[(arch, h_val)], "value_pred_error", smooth)
        if cs is not None:
            ax_vpe.fill_between(cs, mv - sv, mv + sv, color=color, alpha=0.15)
            ax_vpe.plot(cs, mv, color=color, lw=2, linestyle=ls, label=lbl)

    fmt = mticker.FuncFormatter(k_fmt)
    for ax, title, ylabel in [
        (ax_eval,  "Eval return",             "Episode return"),
        (ax_train, "Train return",            "Episode return"),
        (ax_gap,   "Imagination-reality gap", "Imagined − replay"),
        (ax_vpe,   "Value prediction error",  "Q(s₀,a₀) − return"),
    ]:
        ax.set_xlabel("Env steps", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.xaxis.set_major_formatter(fmt)
        ax.grid(True, alpha=0.3, linestyle="--")
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=9, framealpha=0.9)

    for ax in (ax_eval, ax_train):
        ax.set_ylim(bottom=0)

    fig.suptitle(f"{task}  |  {arch_label}", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    save_fig(fig, out / "summary_panel.png", show)


# ── summary table ─────────────────────────────────────────────────────────────

def write_comparison_summary(grouped, out: Path, task: str, arch_label: str):
    lines = [f"Task: {task}   Arch: {arch_label}", "=" * 75,
             f"{'arch':>10}  {'H':>5}  {'Runs':>5}  {'EvalFinal':>10}  {'EvalMax':>10}  {'TrainFinal':>11}"]
    lines.append("-" * 75)

    for arch, h_val in sorted(grouped.keys(), key=lambda x: (x[0], x[1])):
        eval_finals, eval_maxes, train_finals = [], [], []
        for _, series in grouped[(arch, h_val)]:
            _, ev = steps_values(series, "episode/eval_score")
            if ev is not None:
                eval_finals.append(ev[-1]); eval_maxes.append(ev.max())
            _, tv = steps_values(series, "episode/score")
            if tv is not None:
                train_finals.append(tv[-1])

        def fmt_v(lst):
            return f"{np.mean(lst):8.1f}" if lst else "       —"

        lines.append(
            f"{arch:>10}  H={h_val:>3}  {len(grouped[(arch,h_val)]):>5}  "
            f"{fmt_v(eval_finals)}  {fmt_v(eval_maxes)}  {fmt_v(train_finals)}"
        )

    path = out / "comparison_summary.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"  saved → {path}")
    print("\n".join(lines))


# ── output path helper ────────────────────────────────────────────────────────

def make_outdir(base_out: Path, task: str, arch_filter: str | None) -> Path:
    arch_slug = arch_filter if arch_filter else "all"
    return base_out / task / arch_slug


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare TD-MPC2 / DreamerV3 runs across H= values for a task"
    )
    parser.add_argument("root_dir", type=Path,
                        help="Root logdir containing run subdirectories")
    parser.add_argument("--task", required=True,
                        help="Task substring to match (e.g. walker_walk) — required")
    parser.add_argument("--arch", default=None,
                        help="Architecture filter: dreamer | tdmpc2 (default: all)")
    parser.add_argument("--out",  type=Path, default=None,
                        help="Output root (default: <root_dir>/compare_plots/)")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--smooth", type=int, default=5,
                        help="Rolling-window size for smoothed curves (default: 5)")
    args = parser.parse_args()

    root      = args.root_dir.resolve()
    base_out  = (args.out or root / "compare_plots").resolve()
    out       = make_outdir(base_out, args.task, args.arch)
    arch_label = args.arch if args.arch else "all archs"

    print(f"\nScanning : {root}")
    print(f"Task     : {args.task}")
    print(f"Arch     : {arch_label}")
    grouped = scan_runs(root, args.task, args.arch)
    if not grouped:
        sys.exit("[ERROR] No matching run directories found.")

    print(f"\nFound (arch, H) keys: {sorted(grouped.keys())}")
    print(f"Output dir           : {out}\n")

    print("Generating comparison plots...")
    plot_eval_comparison(grouped, out, args.task, arch_label, args.show)
    plot_train_comparison(grouped, out, args.task, arch_label, args.show, args.smooth)
    plot_gap_comparison(grouped, out, args.task, arch_label, args.show, args.smooth)
    plot_vpe_comparison(grouped, out, args.task, arch_label, args.show, args.smooth)
    plot_summary_panel(grouped, out, args.task, arch_label, args.show, args.smooth)

    print("\nSummary table:")
    write_comparison_summary(grouped, out, args.task, arch_label)
    print(f"\nDone. All outputs in: {out}/")


if __name__ == "__main__":
    main()
