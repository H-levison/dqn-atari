# Hyperparameter experiment results

**Member name:** Alliane
**Environment:** ALE/Pong-v5
**Steps per experiment:** 400,000, the same for every run
**Primary axis:** learning rate, with batch size as the secondary

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay | Observed behaviour |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 200000 | Baseline. Learned slowly: the 100-episode average stayed around -20.4 until the last 5k steps and ended at -19.9. Games got a little longer (901 to 1015 steps). Best game -16. |
| 2 | 1e-5 | 0.99 | 32 | 1.0 | 0.01 | 200000 | Too small to learn in 400k steps. The loss kept going down but the reward never improved: final average -20.3 and games got shorter (914 to 875 steps). Best game -17. |
| 3 | 5e-5 | 0.99 | 32 | 1.0 | 0.01 | 200000 | Best learning rate. The average passed -20 at about 260k steps and ended at -18.5. Games got much longer (904 to 1549 steps), so the agent was returning the ball. Best game -12. |
| 4 | 2.5e-4 | 0.99 | 32 | 1.0 | 0.01 | 200000 | Collapsed in the first 100k steps. After that almost every logged loss is close to 0, 1/32 or 2/32: the network gives the same Q-value for every state and only misses the rewards. Reward fell to -20.9 and games got shorter (783 steps). |
| 5 | 5e-4 | 0.99 | 32 | 1.0 | 0.01 | 200000 | Learned normally for the first 100k steps (median loss 0.001), then collapsed the same way as experiment 4 (median loss 0.03, about 1/32). Final average -20.9, games 793 steps. |
| 6 | 1e-3 | 0.99 | 32 | 1.0 | 0.01 | 200000 | Same pattern as experiment 5: fine for 100k steps, then collapsed. Final average -20.9, worse than where it started (-20.6 in the first quarter). Games 787 steps. |
| 7 | 1e-4 | 0.99 | 16 | 1.0 | 0.01 | 200000 | Small batch, noisier updates, but it still learned a bit. The average passed -20 at about 367k steps and ended at -19.4. Games 897 to 1315 steps. Best game -12. Close to experiment 1, so the difference may be chance. |
| 8 | 1e-4 | 0.99 | 64 | 1.0 | 0.01 | 200000 | Clear improvement. The average passed -20 at about 132k steps and ended at -16.8. Games got 2.5 times longer (946 to 2412 steps). Best game -8. |
| 9 | 1e-4 | 0.99 | 128 | 1.0 | 0.01 | 200000 | Best run. The average passed -20 at about 99k steps, the earliest of all, and ended at -15.7. Games 1015 to 2530 steps. Best game -2. Most stable loss (0.002 to 0.016 after 100k steps). In play.py with no exploration it scored -11, -2 and -9. |
| 10 | 2.5e-4 | 0.99 | 64 | 1.0 | 0.01 | 200000 | Collapsed in the first 100k steps like experiment 4, with losses close to 0, 1/64 and 2/64. A bigger batch did not make this learning rate stable. Final average -20.9. |

## Best configuration

Experiment 9: learning rate 1e-4 with batch size 128. It started improving
first (about 99k steps), finished with the best 100-episode average (-15.7),
had the best single game (-2) and the steadiest loss.

Batch size had the biggest positive effect in my runs. Every run does one
update every 4 steps, so the number of updates is the same, but a batch of 128
learns from 4 times more transitions per update than a batch of 32 and each
update is less noisy. The results go in that order: batch 64 and 128 were
clearly better than 16 and 32.

## Worst configuration

Experiments 4, 5, 6 and 10, every run with a learning rate of 2.5e-4 or more.
All four ended at -20.9 with shorter games than at the start, so the agent got
worse than when it was playing almost randomly.

The loss shows what happened. In a healthy run the logged loss takes many
different small values. In these runs it is almost always close to 0, 1/32 or
2/32 (1/64 and 2/64 with batch 64). That is the number of reward transitions in
the batch divided by the batch size, which is what you get when the network
predicts the same Q-value for every state and misses each +1 or -1 reward by
exactly 1. The updates were too big, the network collapsed to a constant
output and stopped learning. With 2.5e-4 this happened in the first 100k
steps, and with 5e-4 and 1e-3 between 100k and 200k steps.

Experiment 10 shows the learning rate was the cause: doubling the batch to 64
did not stop the collapse.

One limitation: train.py only seeds the environment, not the network weights
or the replay sampling, so small differences between runs (for example
experiment 1 against experiment 7) can be partly chance. The large effects,
the collapse above 1e-4 and the gain from bigger batches, are much bigger than
that noise.
