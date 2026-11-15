# Hyperparameter experiment results

**Member name:** HonourGod
**Environment:** ALE/Pong-v5
**Steps per experiment:** 400,000, the same for every run
**Primary axis:** the epsilon schedule — start, end and decay length

Rewards below are 100-episode averages unless stated otherwise. "Crossed -20"
is the step where the 100-episode average first rose above -20, which is the
clearest single marker of when a run started playing better than random.

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay | Observed behaviour |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 200000 | Baseline. Learned, but late: crossed -20 only at 329k of 400k steps and finished at -19.36, up from -20.32 in the first 100 episodes. Games got longer (918 to 1780 steps), so the agent was returning the ball by the end. Best game -14. Half the step budget went on a decaying epsilon. |
| 2 | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 50000 | The only failed run. Never crossed -20 and ended at -20.87, worse than it started. Games got shorter (851 to 784 steps). The loss gives it away: the median is 0.0291, 0.0303, 0.0298, 0.0304 across the four quarters, flat and sitting on 1/32 = 0.03125, and 93% of logged losses after the first quarter land on a multiple of 1/32. That is the constant-Q collapse — the network returns the same Q-value for every state and misses each +1 or -1 reward by exactly 1. Best game -18. |
| 3 | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 20000 | Best run. Crossed -20 at 169k steps, the earliest of all ten, and finished at -16.30 (last 10 episodes -14.70). Games grew from 833 to 2172 steps. Best game -8. Loss stayed healthy and varied (median 0.0035 to 0.0056). |
| 4 | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 350000 | Slowest schedule, decay barely finishing inside the budget. Crossed -20 at 347k steps, the latest of any successful run, and finished at -19.41 — statistically the same as the baseline. Games 928 to 1802 steps, best game -14. Spending 87% of the run on exploration bought nothing. |
| 5 | 1e-4 | 0.99 | 32 | 1.0 | 0.10 | 200000 | A 10% floor did not hurt as much as expected: crossed -20 at 273k and finished at -18.00, ahead of the baseline. Games 902 to 1642 steps, best game -13. In Pong a random action is often harmless, because the paddle has time to recover. |
| 6 | 1e-4 | 0.99 | 32 | 1.0 | 0.20 | 200000 | A 20% floor was clearly too much. Crossed -20 at 282k, finished at -19.02, and the final games were the shortest of any successful run (893 to 1553 steps). Best game -14. One action in five being random keeps costing points the learned policy would have saved. |
| 7 | 1e-4 | 0.99 | 32 | 1.0 | 0.001 | 200000 | A near-zero floor was the best of the floor group: crossed -20 at 236k, finished at -17.41, games 882 to 2078 steps, best game -9. No sign of the agent getting stuck for lack of exploration. |
| 8 | 1e-4 | 0.99 | 32 | 0.5 | 0.01 | 200000 | Starting half-random instead of fully random helped. Crossed -20 at 202k, finished at -17.46, and the final games were the longest in the whole sweep (866 to 2361 steps). Best game -8. The 10,000-step random warmup already fills the buffer, so a fully random start adds little. |
| 9 | 1e-4 | 0.99 | 32 | 0.1 | 0.01 | 200000 | Almost greedy from step one, and it still worked: crossed -20 at 252k, finished at -18.32, games 846 to 1937 steps, best game -10. With an untrained network, greedy actions are close to arbitrary anyway, so this behaves much like a short decay. |
| 10 | 1e-4 | 0.99 | 128 | 1.0 | 0.01 | 20000 | Best model in the sweep, and the only run to win a game. My best schedule from experiment 3 combined with Alliane's best batch size of 128. Crossed -20 at 136k steps, earlier than any other run, and finished at -14.39 (last 10 episodes -9.70). Best game **+8**, meaning it took more points than the opponent. Games grew from 888 to 3951 steps, more than four times longer. Loss fell steadily across the quarters (0.0084, 0.0054, 0.0058, 0.0042) with no sign of collapse. Note that this run changes two things at once against the baseline, so it is not part of the controlled epsilon comparison. |

## Best configuration

**Best epsilon schedule: experiment 3** — epsilon 1.0 to 0.01 over 20,000
steps. Of the nine runs that change only the epsilon schedule, it crossed -20
first (169k steps), finished with the best 100-episode average (-16.30) and the
best single game (-8), and its games more than doubled in length. This is the
answer to the question my part of the sweep was testing, because it is the only
variable that differs.

**Best training result: experiment 10** — the same schedule with the batch size
raised to 128, which was Alliane's best setting. It finished at -14.39, the best
100-episode average of any run in the group, took +8 in its best game (the only
winning game anyone recorded) and ran games of nearly 4,000 steps. I keep it
separate from the claim above because it changes two hyperparameters at once, so
on its own it cannot show which of the two produced the gain. What it does show
is that the two findings compound rather than cancel: the schedule gain from
experiment 3 survives the larger batch.

Experiment 10 is **not** the model we submitted as `dqn_model.pth`. Played
greedily through `play.py` over ten fixed seeds it averaged -11.20 but ranged
from -1 to -20, while Alliane's experiment 9 averaged -10.10 within a much
narrower -6 to -13 — the two averages are within each other's error, so the
deciding factor was consistency rather than the mean.

That comparison was then overtaken. We ran the group's agreed configuration
(batch 128 from Alliane, gamma 0.99 from Anthony, my epsilon floor of 0.01) for
2,000,000 steps instead of 400,000, and that run is what we submitted: it
averages +10.78 in training and +12.30 across the same ten greedy seeds, winning
all ten. My experiment 10 remains the best of my own 400,000-step sweep.

Two things I take from this. The first is that a better training average does
not guarantee more reliable greedy play — experiment 10 beat Alliane's
experiment 9 on the training average and lost to it on consistency, because the
training average still carries the 1% exploration floor and a single run's final
100 episodes is a noisy estimate of a policy. The second is larger: at 400,000
steps every run in my sweep was still improving when the budget ran out, so the
ordering I measured is an ordering of *early* learning speed. It told us which
configuration to spend the long run on, which is what it was for. It does not
tell us where any of those runs would have finished.

The pattern across the whole sweep is that **the step where a run crossed -20
predicts where it finished**, almost in order:

| Crossed -20 at | 136k | 169k | 202k | 236k | 252k | 273k | 282k | 329k | 347k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Experiment | 10 | 3 | 8 | 7 | 9 | 5 | 6 | 1 | 4 |
| Final average | -14.39 | -16.30 | -17.46 | -17.41 | -18.32 | -18.00 | -19.02 | -19.36 | -19.41 |

The only pair out of order is 7 and 8, which differ by 0.05 of a point and are
a tie.

## Final configuration, and why

`--lr 1e-4 --gamma 0.99 --batch-size 128 --eps-start 1.0 --eps-end 0.01
--eps-decay 20000`

The epsilon decay of 20,000 steps is 10 times shorter than the group baseline.
I chose it because at a 400,000-step budget the cost of exploring is losing
steps that could be spent learning, and the ordering above shows that cost
directly. The batch size of 128 comes from Alliane's sweep; gamma stays at 0.99
from Anthony's. Keeping the floor at 0.01 rather than 0.001 costs nothing
measurable (experiments 1, 7) and leaves a little exploration in place, which
matters more over a longer run than the one I could afford here.

With only 400,000 steps the binding constraint is not exploration, it is how
much of the budget is left for learning once exploration is done. Every change
that shortened exploration — a faster decay, a lower floor, a lower start —
finished ahead of the baseline, and the slowest schedule (experiment 4) was the
worst of the successful runs. The group baseline of 200,000 decay steps spends
half the run exploring, which is why it finishes near the bottom.

## Worst configuration

Experiment 2: epsilon decayed over 50,000 steps. It ended at -20.87 having got
worse than its own starting point, with games 8% shorter at the end than at
the start, and it never crossed -20.

This was not slow learning, it was the network collapsing to a constant
output. The evidence is in the loss column rather than the reward: a healthy
run in this sweep has a median loss near 0.005 that moves between quarters,
while experiment 2 sits at 0.030 in all four quarters and 93% of its losses
after the first quarter fall on a multiple of 1/32. With a batch of 32, that
value is the number of reward transitions in the batch divided by the batch
size, which is exactly what the mean squared error becomes when the network
predicts one value everywhere and misses each reward by 1. Alliane found the
same signature in her runs with learning rates of 2.5e-4 and above, so it is a
failure mode of this implementation rather than of this hyperparameter.

**What I cannot claim from this data.** Experiment 3 decays even faster than
experiment 2 — 20,000 steps against 50,000 — and it was the best run in the
sweep, not a collapse. So a 50,000-step decay cannot be the cause of the
collapse on its own. Each configuration was run once, and `train.py` seeds only
the environment, not the network initialisation or the replay sampling, so
experiment 2 may simply have been an unlucky initialisation. Separating the two
would need several seeds per configuration, which did not fit the compute
budget. The collapse mechanism is certain because the loss shows it directly;
the trigger is not.

The same caution applies to small gaps elsewhere: experiments 7 and 8 differ by
0.05 of a point and should be treated as tied. The trends that are large enough
to trust are the ordering above — roughly 3 points between the fastest and
slowest schedules — and the harm from a 0.20 exploration floor.
