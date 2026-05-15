#!/usr/bin/env python3
"""
plot_metrics.py  —  HorizonBreak experiment visualizer
Usage (from scripts/ directory):
    python plot_metrics.py <logdir> [--out <output_dir>] [--show]

Examples:
    python plot_metrics.py ../logdir/dreamer_dmc_walker_walk_H15_seed0
    python plot_metrics.py ../logdir/dreamer_dmc_walker_walk_H15_seed0 --out ../plots --show

Reads:  <logdir>/metrics.jsonl
Writes: <output_dir>/eval_curve.png
        <output_dir>/train_curve.png
        <output_dir>/losses.png
        <output_dir>/summary.txt
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ── colour palette (colourblind-friendly) ────────────────────────────────────
C = {
    "eval":   "#E24A33",
    "train":  "#348ABD",
    "band":   "#348ABD",
    "loss":   ["#348ABD", "#E24A33", "#988ED5", "#777777",
               "#FBC15E", "#8EBA42", "#FFB5B8"],
}

# ── helpers ───────────────────────────────────────────────────────────────────

def load_metrics(logdir: Path) -> dict[str, list]:
    """Parse metrics.jsonl → {key: [(step, value), ...]}"""
    path = logdir / "metrics.jsonl"
    if not path.exists():
        sys.exit(f"[ERROR] metrics.jsonl not found in {logdir}")

    series: dict[str, list] = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            step = row.get("step")
            if step is None:
                continue
            for k, v in row.items():
                if k == "step" or v is None:
                    continue
                try:
                    series[k].append((int(step), float(v)))
                except (TypeError, ValueError):
                    pass

    # sort each series by step
    for k in series:
        series[k].sort(key=lambda x: x[0])
    return dict(series)


def steps_values(series, key):
    """Return (steps_array, values_array) or (None, None) if key missing."""
    if key not in series:
        return None, None
    pairs = series[key]
    return np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])


def k_fmt(x, _):
    return f"{int(x/1000)}k" if x >= 1000 else str(int(x))


def save_fig(fig, path: Path, show: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  saved → {path}")
    if show:
        plt.show()
    plt.close(fig)


# ── plot 1: eval learning curve ───────────────────────────────────────────────

def plot_eval(series, out: Path, tag: str, show: bool):
    es, ev = steps_values(series, "episode/eval_score")
    if es is None:
        print("  [skip] no episode/eval_score found")
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(es, ev, color=C["eval"], lw=2.5, marker="o", ms=5, label="Eval score")

    # light fill under eval curve
    ax.fill_between(es, 0, ev, color=C["eval"], alpha=0.08)

    # annotate final value
    ax.annotate(
        f"{ev[-1]:.1f}",
        xy=(es[-1], ev[-1]),
        xytext=(10, 6), textcoords="offset points",
        fontsize=10, color=C["eval"],
    )

    ax.set_xlabel("Env steps", fontsize=12)
    ax.set_ylabel("Episode return", fontsize=12)
    ax.set_title(f"{tag}\nEval learning curve", fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(k_fmt))
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=11)
    fig.tight_layout()
    save_fig(fig, out / "eval_curve.png", show)


# ── plot 2: train episodes ────────────────────────────────────────────────────

def plot_train(series, out: Path, tag: str, show: bool):
    ts, tv = steps_values(series, "episode/score")
    if ts is None:
        print("  [skip] no episode/score found")
        return

    # ── bin episodes into windows so each bin has all 16 envs ──────────────
    # auto-detect bin size as the smallest gap between distinct step clusters
    unique_steps = np.unique(ts)
    if len(unique_steps) > 1:
        gaps = np.diff(unique_steps)
        bin_size = int(np.min(gaps[gaps > 0])) * 16  # one full env-batch
    else:
        bin_size = 16000  # fallback

    max_step = int(ts.max())
    bin_edges = np.arange(0, max_step + bin_size, bin_size)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    grp_means, grp_stds = [], []
    valid_centers = []
    for lo, hi, center in zip(bin_edges[:-1], bin_edges[1:], bin_centers):
        mask = (ts >= lo) & (ts < hi)
        if mask.sum() == 0:
            continue
        vals = tv[mask]
        grp_means.append(np.mean(vals))
        grp_stds.append(np.std(vals))
        valid_centers.append(center)

    grp_steps = np.array(valid_centers)
    grp_means = np.array(grp_means)
    grp_stds  = np.array(grp_stds)
    # ── end binning ────────────────────────────────────────────────────────

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.fill_between(grp_steps, grp_means - grp_stds, grp_means + grp_stds,
                    color=C["band"], alpha=0.20, label="Train ±1σ")
    ax.plot(grp_steps, grp_means, color=C["train"], lw=2, linestyle="--",
            label="Train mean (all envs)")

    # overlay eval if present
    es, ev = steps_values(series, "episode/eval_score")
    if es is not None:
        ax.plot(es, ev, color=C["eval"], lw=2.5, marker="o", ms=5,
                label="Eval score")

    ax.set_xlabel("Env steps", fontsize=12)
    ax.set_ylabel("Episode return", fontsize=12)
    ax.set_title(f"{tag}\nTrain vs eval return", fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(k_fmt))
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=11)
    fig.tight_layout()
    save_fig(fig, out / "train_curve.png", show)


# ── plot 3: loss curves ───────────────────────────────────────────────────────

LOSS_KEYS = [
    "train/loss/dyn",
    "train/loss/rep",
    "train/loss/rew",
    "train/loss/value",
    "train/loss/policy",
    "train/loss/repval",
    "train/opt/loss",
]

def plot_losses(series, out: Path, tag: str, show: bool):
    available = [(k, *steps_values(series, k)) for k in LOSS_KEYS
                 if steps_values(series, k)[0] is not None]
    if not available:
        print("  [skip] no loss keys found")
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for i, (key, ss, vv) in enumerate(available):
        label = key.replace("train/loss/", "").replace("train/opt/", "opt/")
        ax.plot(ss, vv, lw=1.8, color=C["loss"][i % len(C["loss"])], label=label)

    ax.set_xlabel("Env steps", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(f"{tag}\nTraining losses", fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(k_fmt))
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    save_fig(fig, out / "losses.png", show)


# ── plot 4: imagination-reality gap (metric 3) ────────────────────────────────

def plot_imagination_gap(series, out: Path, tag: str, show: bool):
    ir, iv = steps_values(series, "train/tar")
    er, ev = steps_values(series, "train/ret_replay_mean")
    if ir is None:
        print("  [skip] train/tar not found — skipping gap plot")
        return

    # interpolate eval scores to imagined_return steps
    ev_interp = np.interp(ir, er, ev)
    gap = iv - ev_interp

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ir, gap, color=C["eval"], lw=2)
    ax.axhline(0, color="black", lw=0.8, linestyle="--", label="No gap")
    ax.fill_between(ir, 0, gap, where=(gap > 0), color=C["eval"], alpha=0.15,
                    label="Over-optimism")
    ax.fill_between(ir, 0, gap, where=(gap < 0), color=C["train"], alpha=0.15,
                    label="Under-optimism")
    ax.set_xlabel("Env steps", fontsize=12)
    ax.set_ylabel("Imagined − replay return", fontsize=12)
    ax.set_title(f"{tag}\nImagination-reality gap", fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(k_fmt))
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=11)
    fig.tight_layout()
    save_fig(fig, out / "imagination_gap.png", show)


# ── plot 5: world-model MSE per rollout step (metric 2) ───────────────────────
# Keys expected:  train/wm_mse/step_0, train/wm_mse/step_1, ...

def plot_wm_mse(series, out: Path, tag: str, show: bool):
    mse_keys = sorted([k for k in series if k.startswith("train/wm_mse/step_")])
    if not mse_keys:
        print("  [skip] train/wm_mse/* not yet logged — skipping MSE plot")
        return

    # use last logged value for each rollout step
    rollout_steps, mse_vals = [], []
    for key in mse_keys:
        idx = int(key.split("_")[-1])
        rollout_steps.append(idx)
        _, vv = steps_values(series, key)
        mse_vals.append(float(vv[-1]) if vv is not None else np.nan)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(rollout_steps, mse_vals, color=C["train"], alpha=0.85)
    ax.set_xlabel("Imagination rollout step", fontsize=12)
    ax.set_ylabel("Prediction MSE", fontsize=12)
    ax.set_title(f"{tag}\nWM prediction error vs rollout depth", fontsize=13,
                 fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    fig.tight_layout()
    save_fig(fig, out / "wm_mse.png", show)


# ── summary text ─────────────────────────────────────────────────────────────

def write_summary(series, out: Path, tag: str):
    lines = [f"Run: {tag}", "=" * 60]

    for key in ("episode/eval_score", "episode/score"):
        ss, vv = steps_values(series, key)
        if ss is None:
            continue
        label = "Eval" if "eval" in key else "Train"
        lines += [
            f"{label} score:",
            f"  first={vv[0]:.1f} @ step {ss[0]:,}",
            f"  final={vv[-1]:.1f} @ step {ss[-1]:,}",
            f"  max  ={vv.max():.1f} @ step {ss[vv.argmax()]:,}",
        ]

    path = out / "summary.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"  saved → {path}")
    print("\n".join(lines))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot HorizonBreak metrics.jsonl")
    parser.add_argument("logdir", type=Path,
                        help="Path to the run logdir containing metrics.jsonl")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output directory for plots (default: <logdir>/plots/)")
    parser.add_argument("--show", action="store_true",
                        help="Open each plot interactively after saving")
    args = parser.parse_args()

    logdir = args.logdir.resolve()
    out    = (args.out or logdir / "plots").resolve()
    tag    = logdir.name  # e.g. dreamer_dmc_walker_walk_H15_seed0

    print(f"\nLoading metrics from: {logdir}/metrics.jsonl")
    series = load_metrics(logdir)
    print(f"  {len(series)} metric keys found\n")

    print("Generating plots...")
    plot_eval(series, out, tag, args.show)
    plot_train(series, out, tag, args.show)
    plot_losses(series, out, tag, args.show)
    plot_imagination_gap(series, out, tag, args.show)  # stub until metric 3 logged
    plot_wm_mse(series, out, tag, args.show)           # stub until metric 2 logged

    print("\nSummary:")
    write_summary(series, out, tag)
    print(f"\nDone. All outputs in: {out}/")


if __name__ == "__main__":
    main()
