#!/usr/bin/env python
#SBATCH --account=PDE0114
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --job-name=bf8-param
#SBATCH --output=slurm-%j.out

"""
Train one BayesFlow neural parameter posterior for one of the 8 Demurie DFT/RUDT models.

Run from this directory (model_fitting_8models/), e.g.

    sbatch s1_train_parameter_posterior.py --model dft_pow_basic_none --mode test
    sbatch s1_train_parameter_posterior.py --model rudt_asym_decreasing --mode formal

The neural-network/training configuration intentionally follows the previous
BayesFlow DFT/RUDT pipeline:

- BayesFlow 2.0.13, Keras torch backend
- SetTransformer(summary_dim=32)
- CouplingFlow(transform="spline")
- standardize inference + summary variables
- Adam(lr=5e-4, clipnorm=1.5)
- ReduceLROnPlateau(patience=8, factor=.5, min_lr=1e-6)
- EarlyStopping(patience=20, restore_best_weights=True)
- 50 batches/epoch, batch size 64, validation_data=2000
- formal mode: 5000-epoch hard ceiling + early stopping
- test mode: exactly 5 epochs (all other training settings unchanged)

Only the model/simulator/prior layer is replaced. The 8 model simulators are imported
DIRECTLY from ../simulator/models/, and the Demurie design/prior/screening rules reproduce
simulator/demurie2013_sim.ipynb.
"""

import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import argparse
import importlib
import json
import sys
from pathlib import Path
from importlib.metadata import version as package_version

import numpy as np
import pandas as pd
from scipy.stats import norm, truncnorm

import keras
import bayesflow as bf


# =============================================================================
# Fixed project/model specification
# =============================================================================

MODEL_NAMES = (
    "dft_pow_basic_none",
    "dft_pow_basic_fixed",
    "dft_pow_basic_driftsign",
    "dft_pow_modified_none",
    "rudt_sym_fixed",
    "rudt_sym_decreasing",
    "rudt_asym_fixed",
    "rudt_asym_decreasing",
)

BASE_SEED = 20261015
BF_REQUIRED_VERSION = "2.0.13"

# Paths are deliberately relative to this independent folder.
# Run the scripts from model_fitting_8models/.
SIMULATOR_DIR = Path("../simulator")
MODEL_DIR = Path("../simulator/models")
DATA_PATH = Path("../data/Demurie_2013.csv")
OUTPUT_ROOT = Path("../output/model_fitting_8models")

# Demurie simulator settings copied from simulator/demurie2013_sim.ipynb.
ATTRS = ["ss", "delay_ss", "ll", "delay_ll"]
MAX_REDRAW = 200
SCREEN_MAX_MEAN_RT = 10.0
# IMPORTANT: this follows the current executable code in demurie2013_sim.ipynb.
SCREEN_P_LL = (0.025, 0.975)

# Priors copied from the current Demurie 8-model simulator notebook.
DFT_POP_BASE = {
    "w": (0.54, 0.29, 0.05, 0.95),
    "theta_star": (1.76, 0.33, 0.01, 100.0),
    "alpha": (0.57, 0.35, 0.01, 1.0),
    "beta": (0.45, 0.38, 0.01, 1.0),
}
DFT_Z_STAR = (0.34, 0.27)  # |z*|, truncated to [0, 0.9 * theta_star]

RUDT_POP_NORMAL_BASE = {
    "alpha": (0.64, 0.35, 0.01, 1.0),
    "beta": (0.63, 0.37, 0.01, 1.0),
    "theta": (1.10, 0.47, 0.01, np.inf),
}
RUDT_POP_LOGNORMAL = {
    "s": (0.62, 8.14, 0.0, 100.0),
    "c": (0.08, 0.28, 0.0, 1.0),
    "delta": (0.09, 0.40, 0.0, None),
}
RUDT_N_RANGE = (2, 12)

TRIAL_FEATURE_NAMES = [
    "ss_scaled",
    "delay_ss_scaled",
    "ll_scaled",
    "delay_ll_scaled",
    "choice_signed",
    "log1p_rt",
]
DESIGN_SCALE = 80.0  # intentionally unchanged from the previous BayesFlow pipeline

INITIAL_LR = 5e-4
MIN_LR = 1e-6

FORMAL_TRAIN_CFG = dict(
    epochs=5000,                 # hard ceiling only; EarlyStopping normally ends sooner
    num_batches_per_epoch=50,
    batch_size=64,
    validation_data=2000,
)
TEST_TRAIN_CFG = dict(
    epochs=5,                    # requested smoke/test mode
    num_batches_per_epoch=50,
    batch_size=64,
    validation_data=2000,
)


# =============================================================================
# Environment / model registry
# =============================================================================

def check_environment():
    bf_version = package_version("bayesflow")
    if bf_version != BF_REQUIRED_VERSION:
        raise RuntimeError(
            f"Expected BayesFlow {BF_REQUIRED_VERSION}, found {bf_version}."
        )
    if keras.backend.backend() != "torch":
        raise RuntimeError(
            f"Expected Keras torch backend, found {keras.backend.backend()!r}."
        )

    try:
        import torch
        print("PyTorch:", torch.__version__)
        print("CUDA available:", torch.cuda.is_available())
        print("CUDA device count:", torch.cuda.device_count())
        if torch.cuda.device_count() > 0:
            print("GPU:", torch.cuda.get_device_name(0))
    except Exception as exc:
        print("Torch device inspection warning:", repr(exc))

    print("BayesFlow:", bf_version)
    print("Keras backend:", keras.backend.backend())


def load_model_registry():
    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Missing {MODEL_DIR}. Run this script from model_fitting_8models/."
        )

    for p in (SIMULATOR_DIR, MODEL_DIR):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)

    registry = {}
    for name in MODEL_NAMES:
        mod = importlib.import_module(name)
        simulate = getattr(mod, f"simulate_{name}")
        registry[name] = (simulate, tuple(mod.PARAM_NAMES))
    return registry


# =============================================================================
# Demurie 96-trial design + Ter prior
# =============================================================================

def load_demurie_design_and_ter():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing {DATA_PATH}. The current 8-model simulator derives both the "
            "96-trial design and the Ter prior from this file."
        )

    demurie = pd.read_csv(DATA_PATH).rename(
        columns={
            "ss_value": "ss",
            "ss_time": "delay_ss",
            "ll_value": "ll",
            "ll_time": "delay_ll",
        }
    )

    counts = demurie.groupby(["subj_idx", *ATTRS]).size().unstack("subj_idx")
    if not (counts.notna().all().all() and (counts.nunique(axis=1) == 1).all()):
        raise RuntimeError("Demurie design consistency check failed.")

    problems = counts.iloc[:, 0].rename("n_rep").reset_index()
    identical = (problems["ss"] == problems["ll"]) & (
        problems["delay_ss"] == problems["delay_ll"]
    )
    problems = problems[~identical]

    design = (
        problems.loc[problems.index.repeat(problems["n_rep"]), ATTRS]
        .assign(rep=lambda df: df.groupby(ATTRS).cumcount() + 1)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )
    if len(design) != 96:
        raise RuntimeError(f"Expected 96 Demurie trials after dropping identical options; got {len(design)}.")

    fastest_rt = demurie[demurie["rt"] >= 0.2].groupby("subj_idx")["rt"].min()
    ter_pop = (
        float(fastest_rt.median()),
        float(fastest_rt.std()),
        0.2,
        float(fastest_rt.max()),
    )
    return design, ter_pop


# =============================================================================
# Priors: copied from simulator/demurie2013_sim.ipynb
# =============================================================================

def _truncnorm(mean, sd, lo, hi, size, rng):
    a = (np.asarray(lo) - mean) / sd
    b = (np.asarray(hi) - mean) / sd
    return truncnorm.rvs(a, b, loc=mean, scale=sd, size=size, random_state=rng)


def _trunclognorm(median, mean, lo, hi, size, rng):
    mu = np.log(median)
    sigma = np.sqrt(2 * np.log(mean / median))
    with np.errstate(divide="ignore"):
        p_lo = norm.cdf((np.log(lo) - mu) / sigma)
        p_hi = norm.cdf((np.log(hi) - mu) / sigma)
    return np.exp(mu + sigma * norm.ppf(rng.uniform(p_lo, p_hi, size)))


def rudt_delta_max(design_df, alpha, beta, s):
    x_max = design_df[["ss", "ll"]].to_numpy().max()
    t_max = design_df[["delay_ss", "delay_ll"]].to_numpy().max()
    return np.maximum(x_max**alpha, s * t_max**beta)


def sample_dft_params(model, size, rng, param_names, ter_pop):
    dft_pop = dict(DFT_POP_BASE)
    dft_pop["Ter"] = ter_pop

    draws = {k: _truncnorm(*dft_pop[k], size, rng) for k in dft_pop}
    bias_mode = model.split("_")[-1]

    if bias_mode != "none":
        z_mag = _truncnorm(
            DFT_Z_STAR[0],
            DFT_Z_STAR[1],
            0.0,
            0.9 * draws["theta_star"],
            size,
            rng,
        )
        if bias_mode == "fixed":
            z_mag = z_mag * rng.choice([-1.0, 1.0], size=size)
        draws["z_star"] = z_mag

    return pd.DataFrame(draws)[list(param_names)]


def sample_rudt_params(model, size, design_df, rng, param_names, ter_pop):
    rudt_normal = dict(RUDT_POP_NORMAL_BASE)
    rudt_normal["Ter"] = ter_pop

    draws = {k: _truncnorm(*rudt_normal[k], size, rng) for k in rudt_normal}
    for k in ("s", "c"):
        draws[k] = _trunclognorm(*RUDT_POP_LOGNORMAL[k], size, rng)

    delta_max = rudt_delta_max(design_df, draws["alpha"], draws["beta"], draws["s"])
    median, mean, lo, _ = RUDT_POP_LOGNORMAL["delta"]
    delta_names = ("delta",) if "delta" in param_names else ("delta_S", "delta_L")
    for k in delta_names:
        draws[k] = _trunclognorm(median, mean, lo, delta_max, size, rng)

    if "n" in param_names:
        draws["n"] = rng.integers(RUDT_N_RANGE[0], RUDT_N_RANGE[1] + 1, size=size)

    return pd.DataFrame(draws)[list(param_names)]


def sample_population_params(model, size, design_df, rng, param_names, ter_pop):
    if model.startswith("dft_"):
        return sample_dft_params(model, size, rng, param_names, ter_pop)
    return sample_rudt_params(model, size, design_df, rng, param_names, ter_pop)


# =============================================================================
# Screening + BayesFlow simulator
# =============================================================================

def screen(sim_df):
    if sim_df["rt"].mean() > SCREEN_MAX_MEAN_RT:
        return "slow"
    p_ll = sim_df["choice"].mean()
    if not SCREEN_P_LL[0] <= p_ll <= SCREEN_P_LL[1]:
        return "one-sided"
    return None


def encode_trials(sim_df):
    choice = np.asarray(sim_df["choice"], dtype=float)
    return np.column_stack(
        [
            np.asarray(sim_df["ss"], dtype=float) / DESIGN_SCALE,
            np.asarray(sim_df["delay_ss"], dtype=float) / DESIGN_SCALE,
            np.asarray(sim_df["ll"], dtype=float) / DESIGN_SCALE,
            np.asarray(sim_df["delay_ll"], dtype=float) / DESIGN_SCALE,
            2.0 * choice - 1.0,
            np.log1p(np.asarray(sim_df["rt"], dtype=float)),
        ]
    ).astype("float32")


def draw_screened_subject(model, registry, design_df, ter_pop, rng, redraw_stats=None):
    simulate, param_names = registry[model]
    if redraw_stats is None:
        redraw_stats = {"p_stop": 0, "slow": 0, "one-sided": 0}

    for _ in range(MAX_REDRAW + 1):
        params = sample_population_params(
            model, 1, design_df, rng, param_names, ter_pop
        ).iloc[0]
        param_dict = params.to_dict()
        # Ensure discrete n is a Python integer for the actual simulator.
        if "n" in param_dict:
            param_dict["n"] = int(param_dict["n"])

        try:
            sim_df = simulate(design_df, param_dict, rng=rng)
            reason = screen(sim_df)
        except ValueError:
            reason = "p_stop"

        if reason is None:
            return params, sim_df
        redraw_stats[reason] += 1

    raise RuntimeError(
        f"{model}: no accepted parameter/data draw after {MAX_REDRAW} redraws."
    )


def make_parameter_simulator(model, registry, design_df, ter_pop, seed):
    rng = np.random.default_rng(seed)
    param_names = registry[model][1]
    redraw_stats = {"p_stop": 0, "slow": 0, "one-sided": 0, "accepted": 0}

    def simulate_one():
        params, sim_df = draw_screened_subject(
            model=model,
            registry=registry,
            design_df=design_df,
            ter_pop=ter_pop,
            rng=rng,
            redraw_stats=redraw_stats,
        )
        redraw_stats["accepted"] += 1

        out = {}
        for name in param_names:
            # BayesFlow's continuous flow receives scalar float targets.
            # For decreasing RUDT, n remains integer in the simulator itself,
            # but is represented as float for the continuous posterior network.
            out[name] = float(params[name])
        out["trials"] = encode_trials(sim_df)
        return out

    return bf.make_simulator(simulate_one), redraw_stats


# =============================================================================
# Neural posterior: training code intentionally matches the previous pipeline
# =============================================================================

def make_optimizer():
    return keras.optimizers.Adam(
        learning_rate=INITIAL_LR,
        clipnorm=1.5,
    )


def make_train_callbacks():
    return [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=8,
            min_delta=1e-4,
            cooldown=0,
            min_lr=MIN_LR,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=20,
            min_delta=1e-4,
            restore_best_weights=True,
            verbose=1,
        ),
    ]


def make_parameter_adapter(model, param_names):
    adapter = bf.Adapter()

    for name in param_names:
        if name == "w":
            adapter = adapter.constrain(name, lower=0, upper=1)
        elif name == "z_star" and model == "dft_pow_basic_fixed":
            # Signed z_star is the one parameter among these 8 models whose
            # support crosses zero. Leave it unconstrained and let standardization
            # handle its scale, while the simulator prior enforces |z*| < theta*.
            pass
        else:
            adapter = adapter.constrain(name, lower=0)

    return (
        adapter
        .convert_dtype("float64", "float32")
        .concatenate(list(param_names), into="inference_variables")
        .rename("trials", "summary_variables")
    )


def build_parameter_workflow(model, param_names, simulator, checkpoint_dir):
    return bf.BasicWorkflow(
        simulator=simulator,
        adapter=make_parameter_adapter(model, param_names),
        summary_network=bf.networks.SetTransformer(summary_dim=32),
        inference_network=bf.networks.CouplingFlow(transform="spline"),
        standardize=["inference_variables", "summary_variables"],
        optimizer=make_optimizer(),
        checkpoint_filepath=str(checkpoint_dir),
        checkpoint_name="posterior_best",
        save_best_only=True,
    )


def get_run_dir(model, mode, output_root=OUTPUT_ROOT):
    return Path(output_root) / model / mode


# =============================================================================
# CLI + training
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train one of the eight Demurie BayesFlow parameter posteriors."
    )
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--mode", required=True, choices=("test", "formal"))
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing posterior_final.keras in this run folder.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    check_environment()
    registry = load_model_registry()
    design, ter_pop = load_demurie_design_and_ter()

    model_idx = MODEL_NAMES.index(args.model)
    param_names = registry[args.model][1]
    train_cfg = TEST_TRAIN_CFG.copy() if args.mode == "test" else FORMAL_TRAIN_CFG.copy()

    run_dir = get_run_dir(args.model, args.mode)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    final_path = run_dir / "posterior_final.keras"
    if final_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{final_path} already exists. Use --overwrite only if you intentionally "
            "want to train this model/mode again."
        )

    seed = BASE_SEED + 1000 * model_idx
    keras.utils.set_random_seed(seed)

    simulator, redraw_stats = make_parameter_simulator(
        model=args.model,
        registry=registry,
        design_df=design,
        ter_pop=ter_pop,
        seed=seed,
    )

    # Simulator smoke check before allocating/training the network.
    smoke = simulator.sample(3)
    print("\nSimulator smoke check:")
    print({k: np.asarray(v).shape for k, v in smoke.items()})
    print("parameter names:", param_names)
    print("design trials:", len(design))
    print("Ter prior (mean, sd, lower, upper):", tuple(round(x, 6) for x in ter_pop))

    workflow = build_parameter_workflow(
        model=args.model,
        param_names=param_names,
        simulator=simulator,
        checkpoint_dir=checkpoint_dir,
    )

    adapted = workflow.adapter(simulator.sample(2))
    print("adapted shapes:", {k: np.asarray(v).shape for k, v in adapted.items()})

    lr = workflow.optimizer.learning_rate
    print("optimizer:", type(workflow.optimizer).__name__)
    print("learning_rate:", lr)
    if isinstance(lr, keras.optimizers.schedules.LearningRateSchedule):
        raise RuntimeError(
            "Learning rate is unexpectedly a schedule; ReduceLROnPlateau requires a scalar LR."
        )

    metadata = {
        "model": args.model,
        "mode": args.mode,
        "parameter_names": list(param_names),
        "n_trials": int(len(design)),
        "trial_feature_names": TRIAL_FEATURE_NAMES,
        "design_scale": DESIGN_SCALE,
        "seed": seed,
        "bayesflow_version": package_version("bayesflow"),
        "keras_backend": keras.backend.backend(),
        "training_config": train_cfg,
        "initial_lr": INITIAL_LR,
        "min_lr": MIN_LR,
        "screen_max_mean_rt": SCREEN_MAX_MEAN_RT,
        "screen_p_ll": list(SCREEN_P_LL),
        "max_redraw": MAX_REDRAW,
        "ter_pop": list(ter_pop),
        "output_dir": str(run_dir),
    }
    (run_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 80)
    print(f"TRAINING: model={args.model} | mode={args.mode} | N={len(design)}")
    print("=" * 80)
    print("config:", train_cfg)
    print("output:", run_dir)

    history = workflow.fit_online(
        **train_cfg,
        callbacks=make_train_callbacks(),
    )

    history_df = pd.DataFrame(history.history)
    history_path = run_dir / "training_history.csv"
    history_df.to_csv(history_path, index=False)

    n_epochs = len(history.history["loss"])
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    best_val_loss = float(np.min(history.history["val_loss"]))

    workflow.approximator.save(final_path)

    summary = {
        **metadata,
        "epochs_completed": n_epochs,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "final_model": str(final_path),
        "training_history": str(history_path),
        "redraw_stats_during_training_process": dict(redraw_stats),
    }
    (run_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\nTraining complete.")
    print("epochs completed:", n_epochs)
    print("best epoch:", best_epoch)
    print("best val_loss:", f"{best_val_loss:.6f}")
    print("saved history:", history_path)
    print("saved final model:", final_path)
    print("redraw stats:", redraw_stats)


if __name__ == "__main__":
    main()