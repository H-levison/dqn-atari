# Hyperparameter experiment results

**Member name:** Alliane
**Environment:** ALE/Pong-v5
**Steps per experiment:** 400,000, the same for every run
**Primary axis:** learning rate, with batch size as the secondary

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay | Observed behaviour |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 200000 |  |
| 2 | 1e-5 | 0.99 | 32 | 1.0 | 0.01 | 200000 |  |
| 3 | 5e-5 | 0.99 | 32 | 1.0 | 0.01 | 200000 |  |
| 4 | 2.5e-4 | 0.99 | 32 | 1.0 | 0.01 | 200000 |  |
| 5 | 5e-4 | 0.99 | 32 | 1.0 | 0.01 | 200000 |  |
| 6 | 1e-3 | 0.99 | 32 | 1.0 | 0.01 | 200000 |  |
| 7 | 1e-4 | 0.99 | 16 | 1.0 | 0.01 | 200000 |  |
| 8 | 1e-4 | 0.99 | 64 | 1.0 | 0.01 | 200000 |  |
| 9 | 1e-4 | 0.99 | 128 | 1.0 | 0.01 | 200000 |  |
| 10 | 2.5e-4 | 0.99 | 64 | 1.0 | 0.01 | 200000 |  |

Fill in each row as the run finishes, not at the end. Write what actually
happened in the last column — "diverged, Q-values past 100 by 80k steps,
reward stuck at -21" rather than "performed poorly". That column is what you
will be questioned on.

## Best configuration

<!-- Which experiment, and why it won. -->

## Worst configuration

<!-- Which experiment, and what went wrong. -->
