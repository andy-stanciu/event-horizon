#!/usr/bin/env python3
"""
poc_eval_logger.py
------------------
Proof-of-concept instrumentation for EventHorizon Phase 1:
Episode return + success rate logging for r2dreamer on DMC.

Two modes:
  1. LIVE: monkey-patch OnlineTrainer to emit W&B + structured JSON
           alongside the existing TensorBoard logs.
  2. POST: parse an existing metrics.jsonl logfile and print/export
           a clean summary CSV.

Usage:
  # Live (called from train.py via Hydra override):
  python train.py +horizon_break.wandb=true +horizon_break.run_name=dreamer_dmc_H15

  # Post-hoc analysis:
  python poc_eval_logger.py --logdir ./logdir/test
"""

import argparse
import json
import pathlib
import sys

import numpy as np


# ---------------------------------------------------------------------------
# 1.  Post-hoc parser  (no torch/env deps -- safe to run anywhere)
# ---------------------------------------------------------------------------

def parse_metrics_jsonl(logdir: pathlib.Path) -> list:
    """Read every line from metrics.jsonl and return as a list of dicts."""
    path = logdir / "metrics.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No metrics.jsonl found at {path}")
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def summarise(records: list) -> dict:
    """
    Extract episode return and success rate from parsed records.

    Trainer writes:
      Training : episode/score, episode/length          (per-episode, logged at env step)
      Eval     : episode/eval_score, episode/eval_length,
                 episode/eval_success  (if env emits log_success)
    """
    def _series(key):
        return [(r["step"], r[key]) for r in records if key in r]

    train_returns = _series("episode/score")
    eval_returns  = _series("episode/eval_score")
    eval_success  = _series("episode/eval_success")   # empty for DMC (no sparse success signal)
    eval_lengths  = _series("episode/eval_length")

    def _stats(series, label):
        if not series:
            return {}
        steps, vals = zip(*series)
        vals = np.array(vals, dtype=np.float32)
        return {
            "label":           label,
            "n":               len(vals),
            "mean":            float(vals.mean()),
            "std":             float(vals.std()),
            "min":             float(vals.min()),
            "max":             float(vals.max()),
            "final":           float(vals[-1]),
            "steps_at_final":  int(steps[-1]),
        }

    return {
        "train_return": _stats(train_returns, "train/episode_return"),
        "eval_return":  _stats(eval_returns,  "eval/episode_return"),
        "eval_success": _stats(eval_success,  "eval/success_rate"),
        "eval_length":  _stats(eval_lengths,  "eval/episode_length"),
    }


def export_csv(records: list, out_path: pathlib.Path):
    """Write a tidy CSV with one row per episode-level log step."""
    all_keys = {k for r in records for k in r}
    episode_keys = sorted(k for k in all_keys if k.startswith("episode/") or k == "step")
    rows = [r for r in records if any(k in r for k in episode_keys if k != "step")]
    if not rows:
        print("  No episode-level rows to export.")
        return
    import csv
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=episode_keys, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in episode_keys})
    print(f"  Exported {len(rows)} rows -> {out_path}")


# ---------------------------------------------------------------------------
# 2.  Live W&B patch  (call before policy_trainer.begin(agent))
# ---------------------------------------------------------------------------

def patch_trainer_with_wandb(trainer, config_dict: dict, run_name: str = "dreamer_dmc"):
    """
    Monkey-patch an OnlineTrainer instance to additionally log to W&B.

    Example usage in train.py, after constructing OnlineTrainer:

        from poc_eval_logger import patch_trainer_with_wandb
        policy_trainer = OnlineTrainer(config.trainer, ...)
        patch_trainer_with_wandb(policy_trainer, OmegaConf.to_container(config), run_name="dreamer_dmc_H15")
        policy_trainer.begin(agent)

    W&B logs only episode/* scalars to keep the dashboard clean.
    All other scalars (train/*) remain TensorBoard-only.
    """
    try:
        import wandb
    except ImportError:
        print("[poc_eval_logger] wandb not installed -- skipping W&B patch.")
        return None

    run = wandb.init(
        project="horizonbreak",
        name=run_name,
        config=config_dict,
        resume="allow",
    )

    original_write = trainer.logger.write

    def patched_write(step, fps=False):
        # Snapshot before the original write clears self._scalars
        scalars_snapshot = dict(trainer.logger._scalars)
        original_write(step, fps=fps)
        wb_payload = {k: v for k, v in scalars_snapshot.items() if k.startswith("episode/")}
        if wb_payload:
            wb_payload["env_step"] = step
            run.log(wb_payload, step=step)

    trainer.logger.write = patched_write
    print(f"[poc_eval_logger] W&B live patch active -> project=horizonbreak, run={run_name}")
    return run


# ---------------------------------------------------------------------------
# 3.  EpisodeSink -- lightweight structured JSON per episode
#     (use this if you prefer not to depend on W&B during training)
# ---------------------------------------------------------------------------

class EpisodeSink:
    """
    Writes one JSON line per completed episode to episodes.jsonl.

    Example usage:

        from poc_eval_logger import EpisodeSink
        sink = EpisodeSink(logdir)
        sink.attach(policy_trainer)
        policy_trainer.begin(agent)

    episodes.jsonl format (one JSON object per line):
        {"step": 12000, "episode/score": 312.4, "episode/length": 120}
        {"step": 24000, "episode/score": 418.7, "episode/length": 115, "episode/eval_score": 390.1, ...}
    """

    def __init__(self, logdir: pathlib.Path):
        self._path = logdir / "episodes.jsonl"
        self._f = self._path.open("a", buffering=1)   # line-buffered for safety
        print(f"[EpisodeSink] Writing per-episode records -> {self._path}")

    def attach(self, trainer):
        """Patch trainer.logger.write in-place."""
        original_write = trainer.logger.write

        def patched_write(step, fps=False):
            scalars = dict(trainer.logger._scalars)
            original_write(step, fps=fps)
            episode_keys = {k for k in scalars if k.startswith("episode/")}
            if episode_keys:
                record = {"step": step, **{k: scalars[k] for k in episode_keys}}
                self._f.write(json.dumps(record) + "\n")

        trainer.logger.write = patched_write

    def close(self):
        self._f.close()


# ---------------------------------------------------------------------------
# 4.  CLI  (post-hoc summary mode)
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="HorizonBreak PoC -- parse r2dreamer logdir and report episode metrics"
    )
    parser.add_argument("--logdir", type=pathlib.Path, required=True,
                        help="Path to r2dreamer logdir containing metrics.jsonl")
    parser.add_argument("--csv", type=pathlib.Path, default=None,
                        help="Optional output path for episode CSV (default: logdir/episode_summary.csv)")
    args = parser.parse_args()

    print("\n=== HorizonBreak PoC -- Episode Return & Success Rate ===")
    print(f"Logdir : {args.logdir}\n")

    records = parse_metrics_jsonl(args.logdir)
    print(f"Loaded {len(records)} log records.\n")

    summary = summarise(records)
    any_data = False
    for section, stats in summary.items():
        if not stats:
            continue
        any_data = True
        print(f"--- {stats['label']} ---")
        print(f"  Episodes logged : {stats['n']}")
        print(f"  Mean +/- Std    : {stats['mean']:.2f} +/- {stats['std']:.2f}")
        print(f"  Min / Max       : {stats['min']:.2f} / {stats['max']:.2f}")
        print(f"  Final value     : {stats['final']:.2f}  (at step {stats['steps_at_final']:,})")
        print()

    if not any_data:
        print("  No episode/* keys found. Has training started writing metrics?")
        sys.exit(1)

    csv_out = args.csv or (args.logdir / "episode_summary.csv")
    export_csv(records, csv_out)


if __name__ == "__main__":
    main()
