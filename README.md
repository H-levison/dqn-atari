# DQN on Atari

A Deep Q-Network implemented from scratch in PyTorch and trained to play
`ALE/Pong-v5`. No Stable-Baselines3, RLlib, CleanRL or TorchRL.

> Skeleton. Each section is filled in by its owner — see the task allocation.

## Group members

- Alliane
- Anthony
- HonourGod

## The environment

We use `ALE/Pong-v5` from Gymnasium, which runs the Atari 2600 game through the
Arcade Learning Environment (ale-py). The setup is in `environment.py`.

### Game objective

Our agent controls the green paddle on the right and plays against the
computer's paddle on the left. A point is scored when the other side misses
the ball. The first player to reach 21 points wins and the episode ends.

The score of an episode is our points minus the opponent's points, so it goes
from -21 (we lost every point) to +21 (we won every point). A random agent
scores about -21. The agent has learned to play when its average score goes
above 0, which means it wins more points than it loses.

### Actions

Pong has 6 discrete actions:

| Index | Name | What it does |
| --- | --- | --- |
| 0 | NOOP | do nothing |
| 1 | FIRE | serve the ball |
| 2 | RIGHT | move the paddle up |
| 3 | LEFT | move the paddle down |
| 4 | RIGHTFIRE | move up and fire |
| 5 | LEFTFIRE | move down and fire |

In practice the agent only needs up, down and stay still, but we keep all 6
and let it learn which ones are useful.

### Rewards

The agent gets +1 when the opponent misses the ball, -1 when it misses the
ball, and 0 on every other frame. Most frames give 0, so the agent has to learn
which earlier moves led to a point. This is why the discount factor gamma
matters.

We also clip rewards to their sign (-1, 0 or +1). Pong rewards are already in
that range, so this changes nothing here, but it is part of the standard DQN
setup for Atari.

### What the agent sees

The game gives a 210 x 160 colour image. Before it reaches the network, we
process it the same way as the original DQN paper (Mnih et al., 2015):

1. At the start of each episode the agent does a random number of no-op
   actions (up to 30), so every game starts from a slightly different state.
2. Frame skip of 4: the chosen action is repeated for 4 frames and the agent
   only sees the last one. Neighbouring frames are almost identical, so this
   makes training about 4 times faster.
3. Each frame is converted to grayscale and resized to 84 x 84. Colour does not
   matter in Pong, and a smaller image means less computation.
4. The last 4 frames are stacked together. One frame shows where the ball is
   but not where it is going. With 4 frames the network can see the ball's
   direction and speed.

The observation is an array of shape `(4, 84, 84)` with pixel values 0 to 255.
We store it as `uint8` in the replay buffer to save memory, and the network
divides by 255 when it receives it.

Pong-v5 also has sticky actions: 25% of the time the game repeats the previous
action instead of the one the agent chose. This makes the game less predictable,
so the agent cannot memorise one fixed sequence of moves.

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
