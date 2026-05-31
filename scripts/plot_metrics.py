#!/usr/bin/env python3
"""
plot_metrics.py  —  HorizonBreak experiment visualizer
Supports both DreamerV3 (metrics.jsonl) and TD-MPC2 (eval.csv + train.csv) log formats.

Usage (from scripts/ directory):
    # DreamerV3 run:
    python plot_metrics.py ../logdir/dreamer_dmc_walker_walk_H15

    # TD-MPC2 run:
    python plot_metrics.py ../logdir/tdmpc2_dmc_walker_walk_H3

    # With options:
    python plot_metrics.py ../logdir/... --out ../plots --show

Reads (DreamerV3):  <logdir>/metrics.jsonl
Reads (TD-MPC2):    <logdir>/eval.csv  +  <logdir>/train.csv

Writes: <output_dir>/eval_curve.png
        <output_dir>/train_curve.png
        <output_dir>/losses.png          (DreamerV3 only)
        <output_dir>/imagination_gap.png
        <output_dir>/value_pred_error.png (TD-MPC2 only)
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


# ── format helpers ────────────────────────────────────────────────────────────

def k_fmt(x, _):
    return f"{int(x/1000)}k" if x >= 1000 else str(int(x))


def save_fig(fig, path: Path, show: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  saved → {path}")
    if show:
        plt.show()
    plt.close(fig)


# ── source detection ──────────────────────────────────────────────────────────

def detect_format(logdir: Path) -> str:
    """Return 'dreamer' or 'tdmpc2' based on which files exist."""
    if (logdir / "eval.csv").exists() or (logdir / "train.csv").exists():
        return "tdmpc2"
    if (logdir / "metrics.jsonl").exists():
        return "dreamer"
    sys.exit(f"[ERROR] No metrics.jsonl, eval.csv, or train.csv found in {logdir}")


# ── DreamerV3 loader ──────────────────────────────────────────────────────────

def load_dreamer(logdir: Path) -> dict:
    path = logdir / "metrics.jsonl"
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
    for k in series:
        series[k].sort(key=lambda x: x[0])
    return dict(series)


def steps_values(series, key):
    if key not in series:
        return None, None
    pairs = series[key]
    return np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])


# ── TD-MPC2 loader ────────────────────────────────────────────────────────────

def load_tdmpc2(logdir: Path) -> dict:
    """
    Load TD-MPC2 eval.csv and train.csv into a unified series dict
    using key names that match the shared plot functions below.
    """
    import csv
    series = {}

    def read_csv(path):
        rows = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                parsed = {}
                for k, v in row.items():
                    try:
                        parsed[k] = float(v)
                    except (ValueError, TypeError):
                        parsed[k] = None
                rows.append(parsed)
        return rows

    # eval.csv → episode/eval_score, value_pred_error
    eval_path = logdir / "eval.csv"
    if eval_path.exists():
        rows = read_csv(eval_path)
        for key, col in [
            ("episode/eval_score",    "episode_reward"),
            ("value_pred_error",      "value_pred_error"),
            ("value_pred_error_abs",  "value_pred_error_abs"),
            ("episode/eval_length",   "episode_length"),
        ]:
            pairs = [(int(r["step"]), r[col]) for r in rows
                     if r.get("step") is not None and r.get(col) is not None]
            if pairs:
                series[key] = sorted(pairs, key=lambda x: x[0])

    # train.csv → episode/score, imagination_gap, imagined_return
    train_path = logdir / "train.csv"
    if train_path.exists():
        rows = read_csv(train_path)
        for key, col in [
            ("episode/score",       "episode_reward"),
            ("train/imagination_gap",    "imagination_gap"),
            ("train/imagined_return",    "imagined_return"),
            ("train/replay_return",      "replay_return"),
        ]:
            pairs = [(int(r["step"]), r[col]) for r in rows
                     if r.get("step") is not None and r.get(col) is not None]
            if pairs:
                series[key] = sorted(pairs, key=lambda x: x[0])

    return series


# ── shared plot 1: eval learning curve ───────────────────────────────────────

def plot_eval(series, out: Path, tag: str, show: bool):
    es, ev = steps_values(series, "episode/eval_score")
    if es is None:
        print("  [skip] no episode/eval_score found")
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(es, ev, color=C["eval"], lw=2.5, marker="o", ms=5, label="Eval score")
    ax.fill_between(es, 0, ev, color=C["eval"], alpha=0.08)
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


# ── shared plot 2: train curve ────────────────────────────────────────────────

def plot_train(series, out: Path, tag: str, show: bool):
    ts, tv = steps_values(series, "episode/score")
    if ts is None:
        print("  [skip] no episode/score found")
        return

    unique_steps = np.unique(ts)
    if len(unique_steps) > 1:
        gaps = np.diff(unique_steps)
        bin_size = int(np.min(gaps[gaps > 0])) * 16
    else:
        bin_size = 16000

    max_step = int(ts.max())
    bin_edges = np.arange(0, max_step + bin_size, bin_size)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    grp_means, grp_stds, valid_centers = [], [], []
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

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.fill_between(grp_steps, grp_means - grp_stds, grp_means + grp_stds,
                    color=C["band"], alpha=0.20, label="Train ±1σ")
    ax.plot(grp_steps, grp_means, color=C["train"], lw=2, linestyle="--",
            label="Train mean")

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


# ── DreamerV3-only plot: losses ───────────────────────────────────────────────

LOSS_KEYS = [
    "train/loss/dyn", "train/loss/rep", "train/loss/rew",
    "train/loss/value", "train/loss/policy", "train/loss/repval",
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


# ── shared plot 3: imagination gap ───────────────────────────────────────────

def plot_imagination_gap(series, out: Path, tag: str, show: bool):
    # DreamerV3 keys
    ir, iv = steps_values(series, "train/tar")
    er, ev = steps_values(series, "train/ret_replay_mean")

    # TD-MPC2 keys
    if ir is None:
        ir, iv = steps_values(series, "train/imagined_return")
        er, ev = steps_values(series, "train/replay_return")
        gap_s, gap_v = steps_values(series, "train/imagination_gap")
        if gap_s is not None:
            ir, iv = gap_s, gap_v  # use precomputed gap directly
            er = None

    if ir is None:
        print("  [skip] no imagination gap data found")
        return

    if er is not None:
        # DreamerV3 path: compute gap by interpolation
        ev_interp = np.interp(ir, er, ev)
        gap = iv - ev_interp
        steps = ir
    else:
        # TD-MPC2 path: gap already computed
        gap = iv
        steps = ir

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, gap, color=C["eval"], lw=2)
    ax.axhline(0, color="black", lw=0.8, linestyle="--", label="No gap")
    ax.fill_between(steps, 0, gap, where=(gap > 0), color=C["eval"], alpha=0.15,
                    label="Over-optimism")
    ax.fill_between(steps, 0, gap, where=(gap < 0), color=C["train"], alpha=0.15,
                    label="Under-optimism")
    ax.set_xlabel("Env steps", fontsize=12)
    ax.set_ylabel("Imagined − replay return", fontsize=12)
    ax.set_title(f"{tag}\nImagination-reality gap", fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(k_fmt))
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=11)
    fig.tight_layout()
    save_fig(fig, out / "imagination_gap.png", show)


# ── DreamerV3-only plot: WM MSE ───────────────────────────────────────────────

def plot_wm_mse(series, out: Path, tag: str, show: bool):
    mse_keys = sorted([k for k in series if k.startswith("train/wm_mse/step_")])
    if not mse_keys:
        print("  [skip] train/wm_mse/* not found — skipping MSE plot")
        return

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


# ── TD-MPC2-only plot: value prediction error ─────────────────────────────────

def plot_value_pred_error(series, out: Path, tag: str, show: bool):
    vs, vv = steps_values(series, "value_pred_error")
    if vs is None:
        print("  [skip] no value_pred_error found")
        return

    abs_s, abs_v = steps_values(series, "value_pred_error_abs")

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(vs, vv, color=C["train"], lw=2, label="Value pred error (signed)")
    ax.fill_between(vs, 0, vv, where=(np.array(vv) > 0), color=C["eval"],
                    alpha=0.15, label="Overestimate")
    ax.fill_between(vs, 0, vv, where=(np.array(vv) < 0), color=C["train"],
                    alpha=0.15, label="Underestimate")
    if abs_s is not None:
        ax.plot(abs_s, abs_v, color=C["eval"], lw=1.5, linestyle="--",
                label="Abs error")

    ax.axhline(0, color="black", lw=0.8, linestyle="--")
    ax.annotate(
        f"{vv[-1]:.1f}",
        xy=(vs[-1], vv[-1]),
        xytext=(10, 6), textcoords="offset points",
        fontsize=10, color=C["train"],
    )
    ax.set_xlabel("Env steps", fontsize=12)
    ax.set_ylabel("Q(s₀,a₀) − actual return", fontsize=12)
    ax.set_title(f"{tag}\nValue prediction error", fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(k_fmt))
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=11)
    fig.tight_layout()
    save_fig(fig, out / "value_pred_error.png", show)


# ── summary text ──────────────────────────────────────────────────────────────

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
    for key, label in [("value_pred_error", "Value pred error (final)"),
                        ("train/imagination_gap", "Imagination gap (final)")]:
        ss, vv = steps_values(series, key)
        if ss is not None:
            lines.append(f"{label}: {vv[-1]:.2f}")

    path = out / "summary.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"  saved → {path}")
    print("\n".join(lines))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot HorizonBreak metrics")
    parser.add_argument("logdir", type=Path,
                        help="Path to run logdir (metrics.jsonl or eval.csv/train.csv)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output directory for plots (default: <logdir>/plots/)")
    parser.add_argument("--show", action="store_true",
                        help="Open each plot interactively after saving")
    args = parser.parse_args()

    logdir = args.logdir.resolve()
    out    = (args.out or logdir / "plots").resolve()
    tag    = logdir.name

    fmt = detect_format(logdir)
    print(f"\nDetected format: {fmt.upper()}")
    print(f"Loading metrics from: {logdir}")

    if fmt == "dreamer":
        series = load_dreamer(logdir)
    else:
        series = load_tdmpc2(logdir)

    print(f"  {len(series)} metric keys found\n")
    print("Generating plots...")

    # Shared plots (both algorithms)
    plot_eval(series, out, tag, args.show)
    plot_train(series, out, tag, args.show)
    plot_imagination_gap(series, out, tag, args.show)

    # Algorithm-specific plots
    if fmt == "dreamer":
        plot_losses(series, out, tag, args.show)
        plot_wm_mse(series, out, tag, args.show)
    else:
        plot_value_pred_error(series, out, tag, args.show)

    print("\nSummary:")
    write_summary(series, out, tag)
    print(f"\nDone. All outputs in: {out}/")


if __name__ == "__main__":
    main()
