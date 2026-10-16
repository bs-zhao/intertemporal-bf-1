# intertemporal-bf-1

Self-contained BayesFlow parameter-recovery project for 8 DFT/RUDT models

This repository is **standalone**. It does not import model code or data from any other repository.
After cloning it on OSC, activate the `bf312` environment, `cd` into the repository root,
and run the scripts directly.

## Files

```text
intertemporal-bf-1/
├── simulators_8models.py          # all 8 models, priors, 96-trial design, screening
├── s1_train_parameter_posterior.py # train one model's BayesFlow posterior
├── s2_parameter_recovery.py        # parameter recovery + diagnostics
├── requirements.txt
├── README.md
└── output/                         # created automatically
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

Parameter sets:

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

## Self-contained Demurie design

`simulators_8models.py` reconstructs the Demurie choice design directly:

- SS amount: EUR 0, 5, 10, 20, 30 today
- LL option: EUR 30 after 0, 1, 2, 7, or 14 days
- 4 repetitions per problem
- EUR30-today vs EUR30-today is removed because both models are undefined there
- 24 problems x 4 repetitions = **96 trials**
- one fixed shuffle with `random_state=42`, matching the supplied simulator notebook

No `Demurie_2013.csv` is required.

### Frozen Ter prior

The supplied main-branch ZIP does not include `Demurie_2013.csv`, while the original
`demurie2013_sim.ipynb` computed the Ter prior from that file. To make this repository truly
standalone, the values printed by that notebook are frozen locally:

```python
TER_POP = (0.694, 0.252, 0.2, 1.31)
```

That means truncated Normal(mean=0.694, SD=0.252, lower=0.2, upper=1.31).
Everything else in the prior/simulator is implemented locally from the supplied source.

## Neural posterior training

The neural-network/training layer is intentionally kept the same as the previous two-model
BayesFlow pipeline:

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

Formal mode uses a 5000-epoch hard ceiling and normally stops by EarlyStopping.
Test mode runs exactly 5 epochs.

## OSC: clone and run

Clone the project and enter the repository root:

```bash
cd /fs/scratch/PDE0114/projects
git clone https://github.com/bs-zhao/intertemporal-bf-1.git
cd intertemporal-bf-1
conda activate bf312
```

Verify the environment if desired:

```bash
python -c "import bayesflow, keras, torch, ssms; print('BayesFlow', bayesflow.__version__); print('Keras backend', keras.backend.backend()); print('PyTorch', torch.__version__); print('CUDA', torch.cuda.is_available())"
```

If the existing `bf312` environment is missing the non-PyTorch dependencies:

```bash
python -m pip install -r requirements.txt
```

The scripts use paths relative to the repository root, so run `sbatch` from this directory.

Five-epoch smoke test:

```bash
sbatch s1_train_parameter_posterior.py --model rudt_sym_fixed --mode test
```

Formal training:

```bash
sbatch s1_train_parameter_posterior.py --model rudt_sym_fixed --mode formal
```

Another example:

```bash
sbatch s1_train_parameter_posterior.py --model rudt_asym_decreasing --mode formal
```

The `.py` file itself contains the Slurm header:

```text
account = PDE0114
gpu = 1
cpus = 4
memory = 32 GB
walltime = 7 days
```

## Training outputs

For example:

```text
output/
└── rudt_sym_fixed/
    ├── test/
    │   ├── checkpoints/
    │   ├── posterior_final.keras
    │   ├── training_history.csv
    │   ├── training_metadata.json
    │   └── training_summary.json
    └── formal/
        └── ...
```

Different models and modes never overwrite one another.

## Parameter recovery / diagnostics

After training:

```bash
sbatch s2_parameter_recovery.py --model rudt_sym_fixed --mode formal
```

For a lightweight test:

```bash
python s2_parameter_recovery.py --model rudt_sym_fixed --mode test
```

Optional overrides:

```bash
python s2_parameter_recovery.py --model rudt_sym_fixed --mode formal --n-test 500 --posterior-draws 500 --batch-size 20
```

Formal defaults:

```text
held-out participants = 1000
posterior draws / participant = 1000
inference batch size = 25
```

Diagnostics are written under the same model/mode directory:

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
    ├── parameter_recovery_scatter.png
    ├── coverage_90.png
    ├── posterior_contraction.png
    ├── posterior_correlation_mean_signed.png
    ├── posterior_correlation_mean_abs.png
    ├── training_loss.png
    ├── learning_rate.png
    └── n_recovery_confusion.png   # only for decreasing RUDT models
```

Metrics include Pearson `r`, Spearman rho, RMSE, NRMSE, MAE, bias, 90% interval coverage,
mean interval width, posterior contraction, and posterior-rank calibration statistics.

## Note about `n`

`n` is an integer in the two decreasing-threshold RUDT simulators. Because the explicit
requirement is to keep the same BayesFlow neural-posterior architecture used previously,
`n` is represented as a continuous target in the same `CouplingFlow`. Diagnostics therefore
report both the raw posterior median and rounded-`n` recovery, including exact rounded
accuracy and an `n` confusion matrix.

## Quick start

Five-epoch smoke test:

```bash
sbatch s1_train_parameter_posterior.py --model rudt_sym_fixed --mode test
```

Formal training:

```bash
sbatch s1_train_parameter_posterior.py --model rudt_sym_fixed --mode formal
```

Parameter recovery after training:

```bash
sbatch s2_parameter_recovery.py --model rudt_sym_fixed --mode formal
```

All generated results are written under `output/`, which is ignored by Git.
