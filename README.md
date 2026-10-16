# intertemporal-bf-1

Standalone BayesFlow parameter-recovery project for 8 intertemporal-choice DFT/RUDT models.

The repository is designed to be cloned and run directly on Ohio Supercomputer Center (OSC)
without requiring a pre-existing Conda environment or a pre-installed BayesFlow installation.

## Repository structure

```text
intertemporal-bf-1/
├── simulators_8models.py
├── s1_train_parameter_posterior.py
├── s2_parameter_recovery.py
├── setup_osc.sh
├── verify_gpu.sbatch
├── train.sbatch
├── recovery.sbatch
├── requirements.txt
├── .gitignore
└── output/                       # created automatically; not tracked by Git
```

## Eight models

```text
dft_pow_basic_none
dft_pow_basic_fixed
dft_pow_basic_driftsign
dft_pow_modified_none
rudt_sym_fixed
rudt_sym_decreasing
rudt_asym_fixed
rudt_asym_decreasing
```

| model | parameters |
|---|---|
| `dft_pow_basic_none` | `w, theta_star, alpha, beta, Ter` |
| `dft_pow_basic_fixed` | `w, theta_star, alpha, beta, Ter, z_star` |
| `dft_pow_basic_driftsign` | `w, theta_star, alpha, beta, Ter, z_star` |
| `dft_pow_modified_none` | `w, theta_star, alpha, beta, Ter` |
| `rudt_sym_fixed` | `alpha, beta, s, c, theta, Ter, delta` |
| `rudt_sym_decreasing` | `alpha, beta, s, c, theta, Ter, delta, n` |
| `rudt_asym_fixed` | `alpha, beta, s, c, theta, Ter, delta_S, delta_L` |
| `rudt_asym_decreasing` | `alpha, beta, s, c, theta, Ter, delta_S, delta_L, n` |

## Self-contained simulator

All model formulas, priors, screening rules, the 96-trial Demurie design, and trial encoding are
implemented locally in `simulators_8models.py`.

The project does **not** import code or data from
`individual-difference-Intertemporal-choice` or any other repository.

The Demurie design is reconstructed locally:

- SS amount: EUR 0, 5, 10, 20, 30 today
- LL option: EUR 30 after 0, 1, 2, 7, or 14 days
- 4 repetitions per problem
- EUR30 today vs EUR30 today removed
- 24 problems x 4 repetitions = 96 trials
- fixed shuffle with `random_state=42`

The original simulator notebook derived the Ter prior from `Demurie_2013.csv`.
Because this standalone project does not carry that participant-level dataset, the values already
reported by the source notebook are frozen locally:

```python
TER_POP = (0.694, 0.252, 0.2, 1.31)
```

## Neural posterior architecture

Training intentionally follows the previous BayesFlow pipeline:

```text
BayesFlow 2.0.13
Keras backend = torch
SetTransformer(summary_dim=32)
CouplingFlow(transform="spline")
Adam(lr=5e-4, clipnorm=1.5)
ReduceLROnPlateau(factor=.5, patience=8, min_lr=1e-6)
EarlyStopping(patience=20, restore_best_weights=True)
50 batches / epoch
batch size = 64
validation_data = 2000
```

Formal mode has a 5000-epoch hard ceiling and normally stops by EarlyStopping.
Test mode runs 5 epochs.

# OSC setup

## 1. Clone

Example on Pitzer:

```bash
cd /fs/scratch/PDE0114/projects
git clone https://github.com/bs-zhao/intertemporal-bf-1.git
cd intertemporal-bf-1
```

No Conda environment needs to be activated.

## 2. Install BayesFlow and project dependencies once

Run:

```bash
bash setup_osc.sh
```

The setup script:

1. resets loaded OSC modules;
2. loads OSC's `pytorch/2.8.0` module;
3. creates `.python_packages/` inside this repository;
4. installs `bayesflow==2.0.13`, SSMS, NumPy, SciPy, pandas, and matplotlib there;
5. verifies the imports and the Keras torch backend.

PyTorch itself is **not** installed with pip. The project uses OSC's centrally installed
GPU-enabled PyTorch module.

The project-local package directory is ignored by Git.

If you ever want a clean reinstall:

```bash
rm -rf .python_packages
bash setup_osc.sh
```

## 3. Verify GPU access

After setup:

```bash
sbatch verify_gpu.sbatch
```

Check the resulting:

```text
slurm-verify-<jobid>.out
```

A successful job should report:

```text
BayesFlow: 2.0.13
Keras backend: torch
CUDA available: True
CUDA device count: 1
GPU: ...
Environment verification passed.
```

This check is performed inside an actual GPU allocation. CUDA can legitimately be unavailable on
a login node, so the GPU check should be done through Slurm.

# Training

Do **not** submit the Python file itself with `sbatch`.
Use `train.sbatch`, which loads the OSC PyTorch module and activates the repository-local
Python packages before launching the Python training script.

Five-epoch test:

```bash
sbatch train.sbatch --model rudt_sym_fixed --mode test
```

Formal training:

```bash
sbatch train.sbatch --model rudt_sym_fixed --mode formal
```

Another example:

```bash
sbatch train.sbatch --model rudt_asym_decreasing --mode formal
```

To intentionally retrain an existing model/mode:

```bash
sbatch train.sbatch --model rudt_sym_fixed --mode formal --overwrite
```

The training wrapper currently requests:

```text
account = PDE0114
GPU = 1
CPU = 4
memory = 32 GB
walltime = 7 days
```

## Training outputs

Example:

```text
output/
└── rudt_sym_fixed/
    └── formal/
        ├── checkpoints/
        ├── posterior_final.keras
        ├── training_history.csv
        ├── training_metadata.json
        └── training_summary.json
```

Each model and mode has its own directory.

# Parameter recovery and diagnostics

After a network has been trained:

```bash
sbatch recovery.sbatch --model rudt_sym_fixed --mode formal
```

For the test-mode network:

```bash
sbatch recovery.sbatch --model rudt_sym_fixed --mode test
```

Optional diagnostic-size overrides:

```bash
sbatch recovery.sbatch --model rudt_sym_fixed --mode formal --n-test 500 --posterior-draws 500 --batch-size 20
```

Formal defaults:

```text
held-out participants = 1000
posterior draws per participant = 1000
inference batch size = 25
```

Outputs:

```text
output/<model>/<mode>/diagnostics/
├── parameter_recovery_long.csv
├── parameter_recovery_metrics.csv
├── posterior_correlation_mean_signed.csv
├── posterior_correlation_mean_abs.csv
├── diagnostics_summary.json
└── figures/
    ├── training_loss.png
    ├── learning_rate.png
    ├── parameter_recovery_scatter.png
    ├── coverage_90.png
    ├── posterior_contraction.png
    ├── posterior_correlation_mean_signed.png
    ├── posterior_correlation_mean_abs.png
    └── n_recovery_confusion.png
```

Metrics include Pearson correlation, Spearman correlation, RMSE, normalized RMSE, MAE, bias,
90% posterior interval coverage, interval width, posterior contraction, and posterior-rank
calibration diagnostics.

## Decreasing-threshold RUDT parameter `n`

`n` is an integer simulator parameter in:

```text
rudt_sym_decreasing
rudt_asym_decreasing
```

The neural posterior architecture is intentionally kept identical to the earlier continuous
CouplingFlow pipeline, so `n` is learned as a continuous target. Diagnostics additionally round
the posterior median to the allowed integer range and report discrete recovery accuracy and a
confusion matrix.

# Minimal OSC workflow

```bash
cd /fs/scratch/PDE0114/projects
git clone https://github.com/bs-zhao/intertemporal-bf-1.git
cd intertemporal-bf-1
bash setup_osc.sh
sbatch verify_gpu.sbatch
sbatch train.sbatch --model rudt_sym_fixed --mode test
```

After the 5-epoch smoke test succeeds:

```bash
sbatch train.sbatch --model rudt_sym_fixed --mode formal
```

After formal training finishes:

```bash
sbatch recovery.sbatch --model rudt_sym_fixed --mode formal
```
