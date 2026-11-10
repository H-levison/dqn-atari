# Experiment outputs

One folder per member, each with its own notebook so all three of us can run
experiments at the same time without conflicting over a shared file.

Each run gets its own folder, passed to `train.py` as `--out`:

```
uv run train.py --lr 1e-4 --gamma 0.99 --out experiments/HonourGod/exp01
```

which produces:

```
experiments/
├── Alliane/
├── Anthony/
└── HonourGod/
    ├── experiments.ipynb   that member's 10 runs
    ├── exp01/
    │   ├── log.csv         one row per episode
    │   ├── model.pth       the trained weights
    │   └── log.png         plot_results.py writes this next to the CSV
    └── exp02/ ...
```

Rules:

- The notebook calls `train.py`. No training code is copied into it, so the
  algorithm stays in one place.
- Every run uses the same environment and the same step budget, or the results
  across members cannot be compared.
- One folder per experiment, ten per member. Never overwrite a previous run.
- `dqn_model.pth` at the repository root is the long final run only, not an
  experiment checkpoint.
- Your results table lives at the repository root as
  `[YourName]Hyperparameter_experiment_results.md`, not in here.
