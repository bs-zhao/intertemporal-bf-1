#!/usr/bin/env python

"""
Train one BayesFlow neural parameter posterior for one of the 8 self-contained models.

IMPORTANT
---------
This repository is standalone. This script imports ONLY the local
`simulators_8models.py`; it does not import model code or data from any other repository.

Run from the repository root:

    sbatch train.sbatch --model dft_pow_basic_none --mode test
    sbatch train.sbatch --model rudt_asym_decreasing --mode formal

Neural-network/training settings intentionally follow the previous DFT/RUDT BayesFlow
pipeline. Only the model/simulator layer has changed.
"""

import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import argparse
import json
from pathlib import Path
from importlib.metadata import version as package_version

import numpy as np
import pandas as pd
import keras
import bayesflow as bf

import simulators_8models as sim


# =============================================================================
# Configuration copied from the previous parameter-posterior pipeline
# =============================================================================

BF_REQUIRED_VERSION = "2.0.13"
BASE_SEED = 20261015

INITIAL_LR = 5e-4
MIN_LR = 1e-6

FORMAL_TRAIN_CFG = dict(
    epochs=5000,                 # hard ceiling; EarlyStopping normally ends sooner
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

# Standalone output tree inside this repository. Run from the repository root.
OUTPUT_ROOT = Path("output")


# =============================================================================
# Environment
# =============================================================================

def check_environment():
    if not Path("simulators_8models.py").exists():
        raise RuntimeError(
            "Run this script from the repository root so the local "
            "simulators_8models.py and output/ paths are resolved correctly."
        )

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
    print("Simulator source: local simulators_8models.py")
    print("Demurie trials:", len(sim.DEMURIE_DESIGN))
    print("Frozen Ter prior:", sim.TER_POP)


# =============================================================================
# BayesFlow workflow -- intentionally same architecture/training policy as before
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
            # Fixed-sign z_star can be negative or positive.
            pass
        else:
            # Same policy as the previous pipeline: positive-support parameters are
            # lower-bounded but otherwise learned by the same continuous flow.
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


def get_run_dir(model, mode):
    return OUTPUT_ROOT / model / mode


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train one self-contained eight-model BayesFlow parameter posterior."
    )
    parser.add_argument("--model", required=True, choices=sim.MODEL_NAMES)
    parser.add_argument("--mode", required=True, choices=("test", "formal"))
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing posterior_final.keras for this model/mode.",
    )
    return parser.parse_args()


# =============================================================================
# Train
# =============================================================================

def main():
    args = parse_args()
    check_environment()

    param_names = sim.PARAM_NAMES[args.model]
    model_idx = sim.MODEL_NAMES.index(args.model)
    train_cfg = TEST_TRAIN_CFG.copy() if args.mode == "test" else FORMAL_TRAIN_CFG.copy()

    run_dir = get_run_dir(args.model, args.mode)
    checkpoint_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    final_path = run_dir / "posterior_final.keras"
    if final_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{final_path} already exists. Use --overwrite only if you intentionally "
            "want to retrain this model/mode."
        )

    seed = BASE_SEED + 1000 * model_idx
    keras.utils.set_random_seed(seed)

    simulator, redraw_stats = sim.make_bayesflow_parameter_simulator(
        bf=bf,
        model=args.model,
        seed=seed,
    )

    # Smoke-test the exact simulator which will feed BayesFlow.
    smoke = simulator.sample(3)
    print("\nSimulator smoke check:")
    print({k: np.asarray(v).shape for k, v in smoke.items()})
    print("model:", args.model)
    print("parameters:", param_names)
    print("n_trials:", len(sim.DEMURIE_DESIGN))

    workflow = build_parameter_workflow(
        model=args.model,
        param_names=param_names,
        simulator=simulator,
        checkpoint_dir=checkpoint_dir,
    )

    adapted = workflow.adapter(simulator.sample(2))
    adapted_shapes = {k: np.asarray(v).shape for k, v in adapted.items()}
    print("adapted shapes:", adapted_shapes)

    expected_inference_shape = (2, len(param_names))
    if adapted_shapes.get("inference_variables") != expected_inference_shape:
        raise RuntimeError(
            f"Adapter shape error: expected inference_variables "
            f"{expected_inference_shape}, got {adapted_shapes.get('inference_variables')}."
        )

    lr = workflow.optimizer.learning_rate
    print("optimizer:", type(workflow.optimizer).__name__)
    print("learning_rate:", lr)
    if isinstance(lr, keras.optimizers.schedules.LearningRateSchedule):
        raise RuntimeError(
            "Learning rate is unexpectedly a schedule; ReduceLROnPlateau requires scalar LR."
        )

    metadata = {
        "model": args.model,
        "mode": args.mode,
        "parameter_names": list(param_names),
        "n_trials": len(sim.DEMURIE_DESIGN),
        "trial_feature_names": list(sim.TRIAL_FEATURE_NAMES),
        "design_scale": sim.DESIGN_SCALE,
        "seed": seed,
        "bayesflow_version": package_version("bayesflow"),
        "keras_backend": keras.backend.backend(),
        "training_config": train_cfg,
        "initial_lr": INITIAL_LR,
        "min_lr": MIN_LR,
        "screen_max_mean_rt": sim.SCREEN_MAX_MEAN_RT,
        "screen_p_ll": list(sim.SCREEN_P_LL),
        "max_redraw": sim.MAX_REDRAW,
        "ter_pop_frozen": list(sim.TER_POP),
        "simulator_source": "local simulators_8models.py",
        "output_dir": str(run_dir),
    }
    (run_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 80)
    print(f"TRAINING: model={args.model} | mode={args.mode} | N=96")
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
