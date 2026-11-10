# DQN on Atari

A Deep Q-Network implemented from scratch in PyTorch and trained to play
`ALE/Pong-v5`. No Stable-Baselines3, RLlib, CleanRL or TorchRL.

> Skeleton. Each section is filled in by its owner — see the task allocation.

## Group members

- Alliane
- Anthony
- HonourGod

## The environment

<!-- TODO (Alliane): the game, its objective, the action space, the reward,
     and the preprocessing the agent sees. -->

## DQN architecture

<!-- TODO (Anthony): diagram of the network, and the online/target pair. -->

## How training works

<!-- TODO (Anthony): replay buffer, epsilon-greedy, the Bellman target,
     the loss, and when the target network is synced. -->

## Results

<!-- TODO (HonourGod): training plots for the best run, plus episode length,
     number of steps and rewards. -->

Per-member experiment tables are in the `*Hyperparameter_experiment_results.md`
files at the root of this repository.

## Gameplay demonstration

<!-- TODO (Alliane): video of the best trained agent playing. -->

## Running it

This project uses [uv](https://docs.astral.sh/uv/). Do not use pip.

```bash
uv sync            # create the environment from pyproject.toml / uv.lock
uv run train.py    # train (see --help for the hyperparameters)
uv run play.py     # watch the trained agent play
```

## Repository layout

```
train.py                                   training script
play.py                                    load dqn_model.pth and play greedily
plot_results.py                            training curves from a run's CSV
experiments/<Name>/experiments.ipynb       that member's 10 runs
experiments/<Name>/                        their models, logs and plots
*Hyperparameter_experiment_results.md      one table per member
dqn_model.pth                              the final trained model
Task allocation sheet.pdf                  who did what
```
