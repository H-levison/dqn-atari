"""Run my ten epsilon-schedule experiments, one after another.

Everything except the epsilon schedule is held at the group baseline, so any
difference in the results is caused by exploration and nothing else.

    uv run experiments/HonourGod/run_all.py

Safe to stop and restart: a run whose model.pth already exists is skipped.
"""

import csv
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT_ROOT = Path(__file__).resolve().parent
PYTHON = REPO / ".venv" / "Scripts" / "python.exe"

# (experiment, eps_start, eps_end, eps_decay, what it tests)
EXPERIMENTS = [
    ("exp01", 1.0, 0.01, 200_000, "baseline reference"),
    ("exp02", 1.0, 0.01, 50_000, "fast decay"),
    ("exp03", 1.0, 0.01, 20_000, "very fast decay, little exploration"),
    ("exp04", 1.0, 0.01, 350_000, "slow decay, barely finishes in budget"),
    ("exp05", 1.0, 0.10, 200_000, "high exploration floor"),
    ("exp06", 1.0, 0.20, 200_000, "very high floor, never stops exploring"),
    ("exp07", 1.0, 0.001, 200_000, "near-zero floor"),
    ("exp08", 0.5, 0.01, 200_000, "low start"),
    ("exp09", 0.1, 0.01, 200_000, "almost greedy from the first step"),
]

# Held constant across all runs so the comparison is fair.
BASELINE = ["--lr", "1e-4", "--gamma", "0.99", "--batch-size", "32",
            "--steps", "400000", "--buffer-size", "50000", "--seed", "0"]


def run_one(name, eps_start, eps_end, eps_decay):
    """Run one experiment and return how long it took, in minutes."""
    out_dir = OUT_ROOT / name
    command = [str(PYTHON), str(REPO / "train.py"), *BASELINE,
               "--eps-start", str(eps_start),
               "--eps-end", str(eps_end),
               "--eps-decay", str(eps_decay),
               "--out", str(out_dir)]

    started = time.time()
    subprocess.run(command, cwd=REPO, check=True)
    return (time.time() - started) / 60


def final_reward(name):
    """Mean reward over the last 10 episodes of a finished run."""
    rows = list(csv.DictReader(open(OUT_ROOT / name / "log.csv")))
    last = [float(r["reward"]) for r in rows[-10:]]
    return sum(last) / len(last) if last else float("nan")


def main():
    summary_path = OUT_ROOT / "summary.csv"
    done = []

    for name, eps_start, eps_end, eps_decay, note in EXPERIMENTS:
        if (OUT_ROOT / name / "model.pth").exists():
            print(f"{name}: already finished, skipping", flush=True)
            done.append((name, eps_start, eps_end, eps_decay, note, 0.0))
            continue

        print(f"\n=== {name}: eps {eps_start} -> {eps_end} over {eps_decay:,} "
              f"({note}) ===", flush=True)
        minutes = run_one(name, eps_start, eps_end, eps_decay)
        print(f"{name}: finished in {minutes:.1f} min, "
              f"last-10 mean reward {final_reward(name):.1f}", flush=True)
        done.append((name, eps_start, eps_end, eps_decay, note, minutes))

    # One table I can read straight into the results markdown.
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "eps_start", "eps_end", "eps_decay",
                         "tests", "minutes", "final_reward"])
        for name, eps_start, eps_end, eps_decay, note, minutes in done:
            writer.writerow([name, eps_start, eps_end, eps_decay, note,
                             f"{minutes:.1f}", f"{final_reward(name):.1f}"])

    print(f"\nAll done. Summary written to {summary_path}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
