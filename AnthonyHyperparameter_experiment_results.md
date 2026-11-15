# Hyperparameter experiment results

**Member name:** Anthony
**Environment:** ALE/Pong-v5
**Steps per experiment:** 400,000, the same for every run
**Primary axis:** gamma (0.90, 0.95, 0.99, 0.999), with batch size (16, 32, 64) as the secondary. Learning rate, epsilon schedule and target-sync stay at the group's defaults for every run.

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay | Observed behaviour |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1e-4 | 0.90 | 16 | 1.0 | 0.01 | 200000 | Mean-of-100 reward ended at -18.3 after 340 episodes. Epsilon was already down to 0.01 well before the run finished. Nothing broke, it just didn't learn much with a batch this small. |
| 2 | 1e-4 | 0.90 | 32 | 1.0 | 0.01 | 200000 | A bit better than exp01, -18.1 over 375 episodes. The curve was choppier than exp03's but never fell apart. |
| 3 | 1e-4 | 0.90 | 64 | 1.0 | 0.01 | 200000 | Best of the three gamma=0.90 runs: -16.0 over 284 episodes. Fewer episodes than exp01/exp02, so the agent was probably surviving longer in each game. |
| 4 | 1e-4 | 0.95 | 16 | 1.0 | 0.01 | 200000 | Worst of the gamma=0.95 group, -19.0 over 367 episodes. Same story as exp01, small batch, not much progress. |
| 5 | 1e-4 | 0.95 | 32 | 1.0 | 0.01 | 200000 | -16.7 over 308 episodes. Better than exp04, in line with what bigger batches were doing everywhere else in this sweep. |
| 6 | 1e-4 | 0.95 | 64 | 1.0 | 0.01 | 200000 | Second best run overall, -13.3 over 291 episodes. Beats every gamma=0.90 run, so batch size 64 looks like it's doing most of the work here, not gamma. |
| 7 | 1e-4 | 0.99 | 16 | 1.0 | 0.01 | 200000 | Worst run of the whole sweep, -19.7 over 399 episodes, the shortest average rallies I saw. Nothing crashed, Q-values and loss stayed normal the whole time, it just never really learned with such a small batch. |
| 8 | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 200000 | -19.1 over 321 episodes, close to exp07. Batch 16 and 32 both struggled at gamma=0.99. |
| 9 | 1e-4 | 0.99 | 64 | 1.0 | 0.01 | 200000 | Best run by far, -11.4 over only 282 episodes, the longest rallies of any experiment. Stable the entire way through, no divergence at all. Gamma=0.99 plus the big batch was clearly the right combo. |
| 10 | 1e-4 | 0.999 | 32 | 1.0 | 0.01 | 200000 | Pushed gamma to 0.999 and got -15.2 over 276 episodes. Better than the gamma=0.99 runs with batch 16 or 32, worse than the batch=64 ones. The higher gamma didn't cause any instability, loss stayed normal. |

Fill in each row as the run finishes, not at the end. Write what actually
happened in the last column — "diverged, Q-values past 100 by 80k steps,
reward stuck at -21" rather than "performed poorly". That column is what you
will be questioned on.

## Best configuration

Experiment 9 (gamma=0.99, batch_size=64) won, and it wasn't close: -11.4 mean reward, well ahead of everything else. The same pattern shows up at every gamma I tested, batch 64 came out on top each time (exp03: -16.0, exp06: -13.3, exp09: -11.4). So batch size looks like the real driver here, a bigger batch means less noisy gradient estimates. Gamma=0.99 on top of that just pushed it a bit further. Nothing unstable happened during this run, epsilon decayed the way it should and the loss stayed flat the whole time.

## Worst configuration

Experiment 7 (gamma=0.99, batch_size=16) was the worst, -19.7, barely better than just playing randomly (-21). It didn't crash or blow up, the loss and Q-values looked fine throughout, it just never learned much with such a small batch feeding it noisy updates. Same thing happened in exp01 (-18.3) and exp04 (-19.0), small batch, weak result, no matter what gamma was. Matches what Alliane found too, bigger batches helped her runs as well.
