# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib"]
# ///
"""Compare real training runs saved by the laptop page (runs/*.json).

Prints each run's time to reach a target accuracy, the median per worker count with
the speedup over one worker, and plots accuracy over time. Times start after round 1,
which includes one-off data downloads and GPU warm-up on freshly opened worker pages.

Run: uv run ml/analyze.py [--target 0.78] [--task fashion-mnist-v1] [files...]
"""
import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Write a PNG; no window needed.
import matplotlib.pyplot as plt


def seconds_after_warmup(run):
    """(seconds since the end of round 1, accuracy) for each round from round 1 on."""
    history = run["history"]
    return [((p["elapsedMs"] - history[1]["elapsedMs"]) / 1000, p.get("accuracy")) for p in history[1:]]


def seconds_to(run, target):
    """Seconds after warm-up until test accuracy first reached the target, or None."""
    return next((s for s, accuracy in seconds_after_warmup(run) if (accuracy or 0) >= target), None)


def describe(run):
    devices = ", ".join(f"{w['device']} {w['backend']}" for w in run["workers"])
    return f"{len(run['workers'])} worker{'s' if len(run['workers']) > 1 else ''} ({devices})"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", type=Path, help="run files (default: runs/*.json)")
    parser.add_argument("--target", type=float, default=0.78, help="target test accuracy (default 0.78)")
    parser.add_argument("--task", default="fashion-mnist-v1")
    parser.add_argument("--out", type=Path, default=Path("runs/accuracy.png"))
    args = parser.parse_args()

    runs = [{**json.loads(path.read_text()), "file": path.name} for path in args.files or sorted(Path("runs").glob("*.json"))]
    runs = [run for run in runs if run["taskId"] == args.task and run["outcome"] == "completed"]
    if not runs:
        raise SystemExit(f"No completed {args.task} runs found.")

    print(f"\nTime to {args.target:.0%} test accuracy after warm-up ({args.task}, completed runs only)\n")
    for run in runs:
        seconds = seconds_to(run, args.target)
        print(f"  {run['file'][:19]}  {describe(run):<40} K={run['localSteps']:<3} {run['optimizer']:<5}"
              f"  final {run['history'][-1]['accuracy']:.1%}  "
              f"{'not reached' if seconds is None else f'{seconds:5.1f} s'}")

    by_workers = {}
    for run in runs:
        by_workers.setdefault(len(run["workers"]), []).append(seconds_to(run, args.target))
    one = statistics.median(by_workers[1]) if None not in by_workers.get(1, [None]) else None
    print("\nMedian per worker count:")
    for workers, times in sorted(by_workers.items()):
        if None in times:
            print(f"  {workers} worker(s): some runs never reached the target")
            continue
        median = statistics.median(times)
        speedup = f"  ({one / median:.2f}x vs 1 worker)" if one else ""
        print(f"  {workers} worker(s): {median:5.1f} s over {len(times)} run(s){speedup}")

    for run in runs:
        points = [(s, a) for s, a in seconds_after_warmup(run) if a is not None]
        plt.plot([s for s, _ in points], [100 * a for _, a in points], label=describe(run))
    plt.axhline(100 * args.target, color="gray", linestyle="--", linewidth=1, label=f"target {args.target:.0%}")
    plt.xlabel("seconds after warm-up (round 1)")
    plt.ylabel("test accuracy (%)")
    plt.title(f"{args.task}: accuracy over time")
    plt.legend(fontsize=7)
    plt.grid(alpha=0.3)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nPlot: {args.out}")


if __name__ == "__main__":
    main()
