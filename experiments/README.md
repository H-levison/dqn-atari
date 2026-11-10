# Experiment outputs

One folder per member, each with its own notebook so all three of us can run
experiments at the same time without conflicting over a shared file.

```
experiments/
├── Alliane/
├── Anthony/
└── HonourGod/
    ├── experiments.ipynb   that member's 10 runs
    ├── models/             exp01.pth ... exp10.pth
    ├── logs/               exp01.csv ... exp10.csv
    └── plots/              exp01_reward.png, ...
```

Rules:

- The notebook calls `train.py`. No training code is copied into it, so the
  algorithm stays in one place.
- Every run uses the same environment and the same step budget, or the results
  across members cannot be compared.
- One model per experiment, ten per member. Never overwrite a previous run.
- `dqn_model.pth` at the repository root is the long final run only, not an
  experiment checkpoint.
- Your results table lives at the repository root as
  `[YourName]Hyperparameter_experiment_results.md`, not in here.
