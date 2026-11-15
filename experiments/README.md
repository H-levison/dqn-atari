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
├── HonourGod/
│   ├── experiments.ipynb   that member's 10 runs
│   ├── exp01/
│   │   ├── log.csv         one row per episode
│   │   ├── model.pth       the trained weights
│   │   └── log.png         plot_results.py writes this next to the CSV
│   └── exp02/ ...
└── final_run/              the submitted agent, see below
```

Rules:

- The notebook calls `train.py`. No training code is copied into it, so the
  algorithm stays in one place.
- Every run uses the same environment and the same step budget, or the results
  across members cannot be compared.
- One folder per experiment, ten per member. Never overwrite a previous run.
- `dqn_model.pth` at the repository root is the long final run only, not an
  experiment checkpoint. It is a copy of `final_run/model.pth`.
- Your results table lives at the repository root as
  `[YourName]Hyperparameter_experiment_results.md`, not in here.

## final_run

`final_run/` is not part of anyone's sweep. It is the single long run we
submitted, trained on the configuration the three sweeps agreed on:

```bash
uv run train.py --lr 1e-4 --gamma 0.99 --batch-size 128 \
                --eps-start 1.0 --eps-end 0.01 --eps-decay 200000 \
                --steps 2000000 --buffer-size 50000 --seed 0 \
                --out experiments/final_run
```

It holds the same three files as any experiment folder - `log.csv`, `model.pth`
and `log.png` - over 2,000,000 steps instead of 400,000. Its `model.pth` is what
`dqn_model.pth` at the repository root is a copy of, so `play.py` plays this
agent. The results are written up in the main `README.md`.

The per-member folders are deliberately left at 400,000 steps. That shared
budget is what makes the three sweeps comparable to each other, and changing it
afterwards would have thrown that away.
