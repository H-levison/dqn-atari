"""Plot training curves from the CSV log written by train.py.

Expected CSV columns (one row per episode):
    episode, step, reward, length, epsilon, loss

Usage:
    uv run plot_results.py Alliane/exp03.csv
    uv run plot_results.py Alliane/exp03.csv --out plots/best_experiment.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Plot DQN training curves from a CSV log.")
    parser.add_argument("csv", help="path to the training log CSV")
    parser.add_argument("--out", help="where to save the PNG (default: next to the CSV)")
    return parser.parse_args()


def plot_training(log, title):
    """Draw the four training plots on one figure and return it."""
    # Average reward over the last 100 episodes smooths out the noise.
    mean_100_reward = log["reward"].rolling(window=100, min_periods=1).mean()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title)

    # 1. Reward per episode
    axes[0, 0].plot(log["episode"], log["reward"], alpha=0.4, label="episode reward")
    axes[0, 0].plot(log["episode"], mean_100_reward, label="mean of last 100")
    axes[0, 0].set(title="Reward per episode", xlabel="Episode", ylabel="Reward")
    axes[0, 0].legend()

    # 2. Mean-100 reward against environment steps
    axes[0, 1].plot(log["step"], mean_100_reward)
    axes[0, 1].set(title="Mean reward (last 100) vs steps", xlabel="Step", ylabel="Mean reward")

    # 3. Episode length
    axes[1, 0].plot(log["episode"], log["length"])
    axes[1, 0].set(title="Episode length", xlabel="Episode", ylabel="Steps in episode")

    # 4. Epsilon decay
    axes[1, 1].plot(log["step"], log["epsilon"])
    axes[1, 1].set(title="Epsilon decay", xlabel="Step", ylabel="Epsilon")

    for ax in axes.flat:
        ax.grid(alpha=0.3)

    fig.tight_layout()
    return fig


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    out_path = Path(args.out) if args.out else csv_path.with_suffix(".png")

    log = pd.read_csv(csv_path)
    fig = plot_training(log, title=csv_path.stem)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
