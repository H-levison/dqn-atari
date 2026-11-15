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

Our best agent is experiment 9 from Alliane's sweep, saved as `dqn_model.pth`.
It was trained with:

```bash
uv run train.py --lr 1e-4 --gamma 0.99 --batch-size 128 \
                --eps-start 1.0 --eps-end 0.01 --eps-decay 200000 \
                --steps 400000 --buffer-size 50000 --seed 0 \
                --out experiments/Alliane/exp09
```

![Training curves for the best run](experiments/Alliane/exp09/log.png)

### Training performance

| | Start of training | End of training |
| --- | --- | --- |
| **Reward** (100-episode average) | -19.97 | **-15.74** |
| **Episode length** (50-episode average) | 918 steps | **2,805 steps** |
| **Number of steps** | 400,000 environment steps over 230 episodes | |

A random agent scores about -21, so the reward shows real learning rather than
noise. The 100-episode average first rose above -20 at step 101,471 and kept
climbing to the end of the budget, which means training was still improving
when it stopped.

Episode length is the clearer signal. A Pong episode ends when one side reaches
21 points, so a longer game means more of the ball being returned instead of
missed. Games grew three times longer, from 918 to 2,805 steps.

### Evaluation

Training averages include some exploration, so we also measured the saved model
playing greedily through `play.py` over ten fixed seeds:

| Model | Mean score | Median | Best | Worst |
| --- | --- | --- | --- | --- |
| **Alliane exp09** (saved as `dqn_model.pth`) | **-10.10** | -10.5 | -6 | **-13** |
| HonourGod exp10 | -11.20 | -14.0 | -1 | -20 |
| HonourGod exp03 | -14.70 | -14.5 | -10 | -18 |

We picked exp09 over exp10 for consistency rather than for its average. The two
means are within each other's error, and exp10 reached the better training
average (-14.39) and the only winning game of the project (+8). But across ten
greedy episodes exp10 ranged from -1 to -20 while exp09 stayed between -6 and
-13. A model that trains to a better average does not necessarily play more
reliably, and for a single demonstration the steadier agent is the better
choice.

### What limits the result

400,000 steps is a small budget for Pong — the original DQN paper trains for
tens of millions of frames, and agents usually need 1 to 2 million steps before
they score positively on average. We chose 400,000 so that all three of us
could run ten experiments each on the hardware we had, which made the
hyperparameter comparison the point of the project rather than the final score.
The reward curve had not flattened at 400,000 steps, so a longer run with the
same configuration would be expected to keep improving.

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
